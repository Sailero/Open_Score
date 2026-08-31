"""Auditable VDN and MAPPO baselines for variable-roster HAD Stage 1.

All baselines consume the same entity, task, agent and action masks as the
Stage-1 QMIX implementation.  The shared contract prevents an accidental
observation or action-space advantage from being attributed to an algorithm.
"""

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from open_score.contracts import GlobalState, TeamObservation
from open_score.nn import MaskedSetEncoder
from open_score.stage1.entity_qmix import (
    PermutationEquivariantActionHead,
    SingleAgentQueryAttentionEncoder,
    VariableEntityAgent,
)
from open_score.stage1.replay import PaddedEpisodeBatch


class VariableScaleVDN(nn.Module):
    """Parameter-shared recurrent VDN with explicit inactive-agent masking.

    This is the standard additive value-decomposition baseline.  Unlike the
    QMIX model, it has no state-conditioned mixer.  The sum is intentionally
    not divided by roster size so that the implementation retains VDN's
    original additive semantics; scale is already present in each agent's
    observation.
    """

    def __init__(
        self,
        entity_dim: int,
        self_dim: int,
        task_dim: int,
        state_entity_dim: int,
        action_dim: int,
        agent_hidden_dim: int = 128,
        encoder_kind: str = "deepset",
        attention_heads: int = 4,
        **_: object,
    ):
        super().__init__()
        del state_entity_dim
        self.encoder_kind = encoder_kind
        self.attention_heads = int(attention_heads)
        self.agent = VariableEntityAgent(
            entity_dim,
            self_dim,
            task_dim,
            action_dim,
            hidden_dim=agent_hidden_dim,
            encoder_kind=encoder_kind,
            attention_heads=attention_heads,
        )

    def agent_q(
        self, observation: TeamObservation, hidden: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        return self.agent(observation, hidden)

    def mix(
        self,
        chosen_q: Tensor,
        observation: TeamObservation,
        state: GlobalState,
    ) -> Tensor:
        del state
        return (chosen_q * observation.agent_mask.to(chosen_q.dtype)).sum(dim=1)

    @torch.no_grad()
    def act(
        self,
        observation: TeamObservation,
        hidden: Optional[Tensor] = None,
        epsilon: float = 0.0,
    ) -> Tuple[Tensor, Tensor]:
        q_values, next_hidden = self.agent_q(observation, hidden)
        greedy = q_values.argmax(dim=-1)
        if epsilon <= 0.0:
            return greedy, next_hidden
        random_scores = torch.rand_like(q_values).masked_fill(
            ~observation.avail_actions.bool(), -1.0
        )
        random_actions = random_scores.argmax(dim=-1)
        explore = torch.rand_like(greedy, dtype=torch.float32) < epsilon
        actions = torch.where(explore, random_actions, greedy)
        actions = torch.where(
            observation.agent_mask.bool(), actions, torch.zeros_like(actions)
        )
        return actions, next_hidden


class VariableEntityActor(nn.Module):
    """One recurrent categorical actor shared by every roster slot and scale."""

    def __init__(
        self,
        entity_dim: int,
        self_dim: int,
        task_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        encoder_kind: str = "deepset",
        attention_heads: int = 4,
    ):
        super().__init__()
        if encoder_kind not in {"deepset", "saqa"}:
            raise ValueError("encoder_kind must be 'deepset' or 'saqa'")
        self.hidden_dim = hidden_dim
        self.encoder_kind = encoder_kind
        self.attention_heads = int(attention_heads)
        if encoder_kind == "deepset":
            self.entity_encoder: nn.Module = MaskedSetEncoder(
                entity_dim, hidden_dim, hidden_dim
            )
        else:
            self.entity_encoder = SingleAgentQueryAttentionEncoder(
                entity_dim,
                self_dim + task_dim,
                hidden_dim,
                self.attention_heads,
            )
        self.input_layer = nn.Sequential(
            nn.Linear(hidden_dim + self_dim + task_dim, hidden_dim),
            nn.Tanh(),
        )
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.policy_head = PermutationEquivariantActionHead(hidden_dim, action_dim)

    def initial_hidden(self, batch: int, agents: int, device: torch.device) -> Tensor:
        return torch.zeros(batch, agents, self.hidden_dim, device=device)

    def forward(
        self, observation: TeamObservation, hidden: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        observation.validate()
        batch, agents, entities, entity_dim = observation.entity_obs.shape
        flat_entities = observation.entity_obs.reshape(
            batch * agents, entities, entity_dim
        )
        flat_mask = observation.entity_mask.reshape(batch * agents, entities)
        if self.encoder_kind == "deepset":
            entity_embeddings = self.entity_encoder.element(flat_entities).reshape(
                batch, agents, entities, -1
            )
            context = self.entity_encoder(flat_entities, flat_mask).reshape(
                batch, agents, -1
            )
        else:
            flat_query = torch.cat(
                [observation.self_obs, observation.task_obs], dim=-1
            ).reshape(batch * agents, -1)
            flat_context, flat_embeddings = self.entity_encoder(
                flat_entities, flat_mask, flat_query
            )
            context = flat_context.reshape(batch, agents, -1)
            entity_embeddings = flat_embeddings.reshape(
                batch, agents, entities, -1
            )
        actor_input = self.input_layer(
            torch.cat(
                [context, observation.self_obs, observation.task_obs], dim=-1
            )
        )
        if hidden is None:
            hidden = self.initial_hidden(batch, agents, actor_input.device)
        next_hidden = self.rnn(
            actor_input.reshape(batch * agents, -1),
            hidden.reshape(batch * agents, -1),
        ).reshape(batch, agents, -1)
        logits = self.policy_head(
            next_hidden,
            entity_embeddings,
            observation.action_entity_index,
            observation.action_target_type,
        ).masked_fill(
            ~observation.avail_actions.bool(), -1e9
        )
        active = observation.agent_mask.unsqueeze(-1).to(next_hidden.dtype)
        return logits, next_hidden * active


class CentralStateValue(nn.Module):
    """Permutation-invariant centralized critic used only while training."""

    def __init__(self, state_entity_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = MaskedSetEncoder(
            state_entity_dim, hidden_dim, hidden_dim
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: GlobalState) -> Tensor:
        state.validate()
        return self.value_head(
            self.encoder(state.entities, state.entity_mask)
        ).squeeze(-1)


class VariableScaleMAPPO(nn.Module):
    """Parameter-shared recurrent actor and centralized entity-state critic."""

    def __init__(
        self,
        entity_dim: int,
        self_dim: int,
        task_dim: int,
        state_entity_dim: int,
        action_dim: int,
        actor_hidden_dim: int = 128,
        critic_hidden_dim: int = 128,
        encoder_kind: str = "deepset",
        attention_heads: int = 4,
    ):
        super().__init__()
        self.encoder_kind = encoder_kind
        self.attention_heads = int(attention_heads)
        self.actor = VariableEntityActor(
            entity_dim,
            self_dim,
            task_dim,
            action_dim,
            hidden_dim=actor_hidden_dim,
            encoder_kind=encoder_kind,
            attention_heads=attention_heads,
        )
        self.critic = CentralStateValue(state_entity_dim, critic_hidden_dim)

    def actor_logits(
        self, observation: TeamObservation, hidden: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        return self.actor(observation, hidden)

    def value(self, state: GlobalState) -> Tensor:
        return self.critic(state)


@dataclass(frozen=True)
class MAPPOMetrics:
    loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float
    grad_norm: float
    actor_grad_norm: float
    critic_grad_norm: float
    learner_step: int
    learning_signal_by_scale: Mapping[Tuple[int, int], float]


class RunningValueNormalizer:
    """Running scalar return normalizer used by the MAPPO value objective.

    Statistics are updated once per on-policy batch (not once per PPO epoch),
    and padded timesteps are excluded.  Keeping this state outside the model
    makes the checkpoint contract explicit and prevents resumed critics from
    silently interpreting normalized values under fresh statistics.
    """

    def __init__(self, device: torch.device, epsilon: float = 1e-5):
        self.mean = torch.zeros((), dtype=torch.float64, device=device)
        self.count = torch.as_tensor(epsilon, dtype=torch.float64, device=device)
        self.second_moment = self.count.clone()
        self.epsilon = float(epsilon)

    @property
    def variance(self) -> Tensor:
        return (self.second_moment / self.count).clamp_min(self.epsilon)

    def update(self, values: Tensor, mask: Tensor) -> None:
        selected = values.detach()[mask.bool()].to(dtype=torch.float64)
        if selected.numel() == 0:
            return
        batch_count = torch.as_tensor(
            selected.numel(), dtype=torch.float64, device=selected.device
        )
        batch_mean = selected.mean()
        batch_second_moment = ((selected - batch_mean).square()).sum()
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        self.second_moment = (
            self.second_moment
            + batch_second_moment
            + delta.square() * self.count * batch_count / total_count
        )
        self.mean = self.mean + delta * batch_count / total_count
        self.count = total_count

    def normalize(self, values: Tensor) -> Tensor:
        mean = self.mean.to(device=values.device, dtype=values.dtype)
        std = self.variance.sqrt().to(device=values.device, dtype=values.dtype)
        return (values - mean) / std

    def denormalize(self, values: Tensor) -> Tensor:
        mean = self.mean.to(device=values.device, dtype=values.dtype)
        std = self.variance.sqrt().to(device=values.device, dtype=values.dtype)
        return values * std + mean

    def state_dict(self) -> Dict[str, object]:
        return {
            "mean": self.mean.detach().cpu(),
            "second_moment": self.second_moment.detach().cpu(),
            "count": self.count.detach().cpu(),
            "epsilon": self.epsilon,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        required = {"mean", "second_moment", "count", "epsilon"}
        if set(state) != required:
            raise ValueError(
                "invalid MAPPO value-normalizer state: expected "
                f"{sorted(required)}, found {sorted(state)}"
            )
        device = self.mean.device
        self.mean = torch.as_tensor(state["mean"], dtype=torch.float64, device=device)
        self.second_moment = torch.as_tensor(
            state["second_moment"], dtype=torch.float64, device=device
        )
        self.count = torch.as_tensor(
            state["count"], dtype=torch.float64, device=device
        )
        self.epsilon = float(state["epsilon"])
        if self.count <= 0 or self.epsilon <= 0:
            raise ValueError("invalid MAPPO value-normalizer count/epsilon")


class SequenceMAPPOLearner:
    """Whole-episode MAPPO with GAE, clipped ratios and padded-sequence masks.

    The actor objective is evaluated per active agent while the centralized
    critic and GAE target are team-level.  Episodes are on-policy: callers must
    clear their rollout batch after every :meth:`train_batch` call.
    """

    def __init__(
        self,
        model: VariableScaleMAPPO,
        learning_rate: float = 5e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_coefficient: float = 0.5,
        entropy_coefficient: float = 0.01,
        max_grad_norm: float = 10.0,
        epochs: int = 4,
    ):
        if not 0.0 <= gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")
        if clip_ratio <= 0.0 or epochs < 1:
            raise ValueError("clip_ratio and epochs must be positive")
        self.model = model
        self.actor_optimizer = torch.optim.Adam(
            model.actor.parameters(), lr=learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            model.critic.parameters(), lr=learning_rate
        )
        self.value_normalizer = RunningValueNormalizer(next(model.parameters()).device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coefficient = value_coefficient
        self.entropy_coefficient = entropy_coefficient
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.learner_step = 0

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @staticmethod
    def _unroll_actor(
        model: VariableScaleMAPPO, batch: PaddedEpisodeBatch
    ) -> Tensor:
        hidden: Optional[Tensor] = None
        sequence = []
        for time in range(batch.max_steps + 1):
            logits, hidden = model.actor_logits(batch.observation_at(time), hidden)
            sequence.append(logits)
        return torch.stack(sequence, dim=1)

    @staticmethod
    def _unroll_value(
        model: VariableScaleMAPPO, batch: PaddedEpisodeBatch
    ) -> Tensor:
        return torch.stack(
            [model.value(batch.state_at(time)) for time in range(batch.max_steps + 1)],
            dim=1,
        )

    def _gae(self, values: Tensor, batch: PaddedEpisodeBatch) -> Tuple[Tensor, Tensor]:
        advantages = torch.zeros_like(batch.rewards)
        running = torch.zeros(batch.batch_size, device=values.device)
        for time in reversed(range(batch.max_steps)):
            nonterminal = 1.0 - batch.done[:, time]
            delta = (
                batch.rewards[:, time]
                + self.gamma * nonterminal * values[:, time + 1]
                - values[:, time]
            ) * batch.filled[:, time]
            running = (
                delta
                + self.gamma * self.gae_lambda * nonterminal * running
            ) * batch.filled[:, time]
            advantages[:, time] = running
        returns = advantages + values[:, :-1]
        return advantages, returns

    def _update_value_scale(
        self, old_values_raw: Tensor, returns: Tensor, mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Put old predictions and targets on the same newly updated scale."""

        self.value_normalizer.update(returns, mask)
        return (
            self.value_normalizer.normalize(old_values_raw),
            self.value_normalizer.normalize(returns),
        )

    def train_batch(self, batch: PaddedEpisodeBatch) -> MAPPOMetrics:
        self.model.train()
        with torch.no_grad():
            old_logits = self._unroll_actor(self.model, batch)[:, :-1]
            old_log_probabilities = F.log_softmax(old_logits, dim=-1)
            old_action_log_prob = old_log_probabilities.gather(
                -1, batch.actions.unsqueeze(-1)
            ).squeeze(-1)
            old_values = self._unroll_value(self.model, batch)
            old_values_raw = self.value_normalizer.denormalize(old_values)
            advantages, returns = self._gae(old_values_raw, batch)
            valid_time = batch.filled.bool()
            valid_advantages = advantages[valid_time]
            advantage_mean = valid_advantages.mean()
            advantage_std = valid_advantages.std(unbiased=False).clamp_min(1e-6)
            normalized_advantage = (advantages - advantage_mean) / advantage_std
            per_episode_signal = (
                advantages.abs() * batch.filled
            ).sum(dim=1) / batch.filled.sum(dim=1).clamp_min(1.0)
            grouped_signal: Dict[Tuple[int, int], list] = {}
            for scale_tensor, value in zip(
                batch.scales.detach().cpu(), per_episode_signal.detach().cpu()
            ):
                scale = (int(scale_tensor[0]), int(scale_tensor[1]))
                grouped_signal.setdefault(scale, []).append(float(value))
            learning_signal_by_scale = {
                scale: sum(values) / len(values)
                for scale, values in grouped_signal.items()
            }
            old_values_on_updated_scale, normalized_returns = (
                self._update_value_scale(old_values_raw, returns, batch.filled)
            )

        actor_mask = (
            batch.agent_mask[:, :-1].to(batch.rewards.dtype)
            * batch.filled.unsqueeze(-1)
        )
        actor_normalizer = actor_mask.sum().clamp_min(1.0)
        value_loss_normalizer = batch.filled.sum().clamp_min(1.0)
        last = None
        for _ in range(self.epochs):
            logits = self._unroll_actor(self.model, batch)[:, :-1]
            log_probabilities = F.log_softmax(logits, dim=-1)
            action_log_prob = log_probabilities.gather(
                -1, batch.actions.unsqueeze(-1)
            ).squeeze(-1)
            ratio = torch.exp(action_log_prob - old_action_log_prob)
            team_advantage = normalized_advantage.unsqueeze(-1)
            unclipped = ratio * team_advantage
            clipped = ratio.clamp(
                1.0 - self.clip_ratio, 1.0 + self.clip_ratio
            ) * team_advantage
            policy_loss = -(
                torch.minimum(unclipped, clipped) * actor_mask
            ).sum() / actor_normalizer

            probabilities = log_probabilities.exp()
            entropy_per_agent = -(probabilities * log_probabilities).sum(dim=-1)
            entropy = (entropy_per_agent * actor_mask).sum() / actor_normalizer

            values = self._unroll_value(self.model, batch)[:, :-1]
            value_delta = values - old_values_on_updated_scale[:, :-1]
            clipped_values = old_values_on_updated_scale[:, :-1] + value_delta.clamp(
                -self.clip_ratio, self.clip_ratio
            )
            value_error = (values - normalized_returns).square()
            clipped_value_error = (clipped_values - normalized_returns).square()
            value_loss = 0.5 * (
                torch.maximum(value_error, clipped_value_error) * batch.filled
            ).sum() / value_loss_normalizer

            actor_loss = policy_loss - self.entropy_coefficient * entropy
            critic_loss = self.value_coefficient * value_loss
            loss = actor_loss + critic_loss
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.actor.parameters(), self.max_grad_norm
            )
            self.actor_optimizer.step()
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.critic.parameters(), self.max_grad_norm
            )
            self.critic_optimizer.step()

            with torch.no_grad():
                approximate_kl = (
                    (old_action_log_prob - action_log_prob) * actor_mask
                ).sum() / actor_normalizer
                clip_fraction = (
                    ((ratio - 1.0).abs() > self.clip_ratio).to(ratio.dtype)
                    * actor_mask
                ).sum() / actor_normalizer
            last = (
                loss,
                policy_loss,
                value_loss,
                entropy,
                approximate_kl,
                clip_fraction,
                actor_grad_norm,
                critic_grad_norm,
            )

        self.learner_step += 1
        assert last is not None
        return MAPPOMetrics(
            loss=float(last[0].detach().cpu()),
            policy_loss=float(last[1].detach().cpu()),
            value_loss=float(last[2].detach().cpu()),
            entropy=float(last[3].detach().cpu()),
            approximate_kl=float(last[4].detach().cpu()),
            clip_fraction=float(last[5].detach().cpu()),
            grad_norm=float(
                torch.maximum(torch.as_tensor(last[6]), torch.as_tensor(last[7]))
                .detach()
                .cpu()
            ),
            actor_grad_norm=float(torch.as_tensor(last[6]).detach().cpu()),
            critic_grad_norm=float(torch.as_tensor(last[7]).detach().cpu()),
            learner_step=self.learner_step,
            learning_signal_by_scale=learning_signal_by_scale,
        )

    def save(
        self, path: Path, extra: Optional[Mapping[str, object]] = None
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "checkpoint_contract": (
                    "sequence-mappo-v2-separate-optimizers-running-valuenorm"
                ),
                "model": self.model.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "value_normalizer": self.value_normalizer.state_dict(),
                "learner_step": self.learner_step,
                "extra": dict(extra or {}),
            },
            path,
        )

    def load(
        self, path: Path, map_location: Optional[torch.device] = None
    ) -> Mapping[str, object]:
        checkpoint = torch.load(path, map_location=map_location or self.device)
        expected_contract = (
            "sequence-mappo-v2-separate-optimizers-running-valuenorm"
        )
        if checkpoint.get("checkpoint_contract") != expected_contract:
            raise ValueError(
                "incompatible MAPPO checkpoint contract; legacy combined-optimizer "
                "checkpoints cannot safely restore independent optimizer and "
                "value-normalizer state"
            )
        self.model.load_state_dict(checkpoint["model"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.value_normalizer.load_state_dict(checkpoint["value_normalizer"])
        self.learner_step = int(checkpoint["learner_step"])
        return checkpoint.get("extra", {})


def frozen_mappo(model: VariableScaleMAPPO) -> VariableScaleMAPPO:
    frozen = copy.deepcopy(model).eval()
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)
    return frozen
