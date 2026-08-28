"""Recurrent whole-episode Double-Q/TD(lambda) learner for Stage 1."""

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F

from open_score.stage1.curriculum import Scale
from open_score.stage1.entity_qmix import VariableScaleQMIX
from open_score.stage1.replay import PaddedEpisodeBatch


@dataclass(frozen=True)
class LearnerMetrics:
    loss: float
    mean_absolute_td: float
    grad_norm: float
    q_total_mean: float
    target_mean: float
    learner_step: int
    td_by_scale: Mapping[Scale, float]


class SequenceQMIXLearner:
    """Train one cooperative team policy across padded variable-scale episodes."""

    def __init__(
        self,
        online: VariableScaleQMIX,
        learning_rate: float = 5e-4,
        gamma: float = 0.99,
        td_lambda: float = 0.6,
        target_update_interval: int = 200,
        max_grad_norm: float = 10.0,
    ):
        if not 0.0 <= td_lambda <= 1.0:
            raise ValueError("td_lambda must be in [0, 1]")
        self.online = online
        self.target = copy.deepcopy(online).eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.td_lambda = td_lambda
        self.target_update_interval = target_update_interval
        self.max_grad_norm = max_grad_norm
        self.learner_step = 0

    @property
    def device(self) -> torch.device:
        return next(self.online.parameters()).device

    @staticmethod
    def _unroll_agent(model: VariableScaleQMIX, batch: PaddedEpisodeBatch) -> Tuple[Tensor, ...]:
        hidden: Optional[Tensor] = None
        sequence = []
        for time in range(batch.max_steps + 1):
            q_values, hidden = model.agent_q(batch.observation_at(time), hidden)
            sequence.append(q_values)
        return tuple(sequence)

    @staticmethod
    def _mix_sequence(
        model: VariableScaleQMIX,
        batch: PaddedEpisodeBatch,
        chosen_q: Tuple[Tensor, ...],
        start_time: int = 0,
    ) -> Tensor:
        totals = []
        for offset, values in enumerate(chosen_q):
            time = start_time + offset
            totals.append(model.mix(values, batch.observation_at(time), batch.state_at(time)))
        return torch.stack(totals, dim=1)

    def train_batch(self, batch: PaddedEpisodeBatch) -> LearnerMetrics:
        self.online.train()
        online_q = self._unroll_agent(self.online, batch)
        chosen = tuple(
            online_q[time]
            .gather(-1, batch.actions[:, time].unsqueeze(-1))
            .squeeze(-1)
            for time in range(batch.max_steps)
        )
        q_total = self._mix_sequence(self.online, batch, chosen)

        with torch.no_grad():
            target_q = self._unroll_agent(self.target, batch)
            next_chosen = []
            for time in range(1, batch.max_steps + 1):
                greedy = online_q[time].argmax(dim=-1)
                next_chosen.append(target_q[time].gather(-1, greedy.unsqueeze(-1)).squeeze(-1))
            next_total = self._mix_sequence(self.target, batch, tuple(next_chosen), start_time=1)
            targets = torch.zeros_like(q_total)
            next_return = next_total[:, -1]
            for time in reversed(range(batch.max_steps)):
                bootstrap = (1.0 - self.td_lambda) * next_total[:, time] + self.td_lambda * next_return
                next_return = batch.rewards[:, time] + self.gamma * (
                    1.0 - batch.done[:, time]
                ) * bootstrap
                targets[:, time] = next_return

        td_error = q_total - targets
        element_loss = F.smooth_l1_loss(q_total, targets, reduction="none")
        normalizer = batch.filled.sum().clamp_min(1.0)
        loss = (element_loss * batch.filled).sum() / normalizer
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.learner_step += 1
        if self.learner_step % self.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())

        absolute = td_error.detach().abs()
        per_episode = (absolute * batch.filled).sum(dim=1) / batch.filled.sum(dim=1).clamp_min(1.0)
        td_by_scale: Dict[Scale, list] = {}
        for scale_tensor, value in zip(batch.scales.detach().cpu(), per_episode.detach().cpu()):
            scale = (int(scale_tensor[0]), int(scale_tensor[1]))
            td_by_scale.setdefault(scale, []).append(float(value))
        reduced = {scale: sum(values) / len(values) for scale, values in td_by_scale.items()}
        return LearnerMetrics(
            loss=float(loss.detach().cpu()),
            mean_absolute_td=float((absolute * batch.filled).sum().cpu() / normalizer.cpu()),
            grad_norm=float(torch.as_tensor(grad_norm).detach().cpu()),
            q_total_mean=float((q_total.detach() * batch.filled).sum().cpu() / normalizer.cpu()),
            target_mean=float((targets.detach() * batch.filled).sum().cpu() / normalizer.cpu()),
            learner_step=self.learner_step,
            td_by_scale=reduced,
        )

    def save(self, path: Path, extra: Optional[Mapping[str, object]] = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "learner_step": self.learner_step,
                "extra": dict(extra or {}),
            },
            path,
        )

    def load(self, path: Path, map_location: Optional[torch.device] = None) -> Mapping[str, object]:
        checkpoint = torch.load(path, map_location=map_location or self.device)
        self.online.load_state_dict(checkpoint["online"])
        self.target.load_state_dict(checkpoint["target"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.learner_step = int(checkpoint["learner_step"])
        return checkpoint.get("extra", {})


def linear_epsilon(
    environment_steps: int,
    start: float = 1.0,
    finish: float = 0.05,
    anneal_steps: int = 100_000,
) -> float:
    fraction = min(1.0, max(0.0, environment_steps / max(1, anneal_steps)))
    return start + fraction * (finish - start)
