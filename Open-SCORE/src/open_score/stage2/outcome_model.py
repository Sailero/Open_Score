"""Simple supervised outcome-and-time model for one canonical small subgame."""

from typing import Dict

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class OutcomeTimeMLP(nn.Module):
    """Predict the joint terminal side and terminal-time bin.

    Classes 0..K-1 mean a defender win in time bin k, K..2K-1 mean
    an attacker breach, and class 2K means timeout.  A single categorical
    distribution keeps win probability and remaining-time estimates coherent.
    """

    def __init__(
        self,
        input_dim: int,
        horizon_bins: int = 20,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.horizon_bins = horizon_bins
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * horizon_bins + 1),
        )

    def forward(self, canonical_state: Tensor) -> Tensor:
        return self.network(canonical_state)

    def summarize(
        self,
        canonical_state: Tensor,
        steps_per_bin: int = 10,
        command_bins: int = 1,
    ) -> Dict[str, Tensor]:
        probabilities = torch.softmax(self(canonical_state), dim=-1)
        k = self.horizon_bins
        defender = probabilities[..., :k]
        breach = probabilities[..., k : 2 * k]
        timeout = probabilities[..., -1]
        bin_steps = torch.arange(1, k + 1, device=probabilities.device, dtype=probabilities.dtype)
        terminal_mass = defender + breach
        expected_steps = (terminal_mass * bin_steps).sum(-1) * steps_per_bin
        expected_steps = expected_steps + timeout * (k * steps_per_bin)
        command_bins = max(1, min(command_bins, k))
        return {
            "defender_win_probability": defender.sum(-1) + timeout,
            "breach_probability": breach.sum(-1),
            "breach_within_command": breach[..., :command_bins].sum(-1),
            "expected_remaining_steps": expected_steps,
            "timeout_probability": timeout,
        }


class BootstrapOutcomeEnsemble(nn.Module):
    """A small MLP ensemble that exposes empirical 95% model intervals."""

    def __init__(
        self,
        members: int,
        input_dim: int,
        horizon_bins: int = 20,
        hidden_dim: int = 64,
    ):
        super().__init__()
        if members < 2:
            raise ValueError("an uncertainty ensemble requires at least two members")
        self.members = nn.ModuleList(
            [OutcomeTimeMLP(input_dim, horizon_bins, hidden_dim) for _ in range(members)]
        )

    @torch.no_grad()
    def predict_interval(
        self,
        canonical_state: Tensor,
        steps_per_bin: int = 10,
        command_bins: int = 1,
    ) -> Dict[str, Tensor]:
        summaries = [
            member.summarize(canonical_state, steps_per_bin, command_bins)
            for member in self.members
        ]
        result: Dict[str, Tensor] = {}
        for key in summaries[0]:
            samples = torch.stack([summary[key] for summary in summaries], dim=0)
            result[key] = samples.mean(dim=0)
            result[f"{key}_lower95"] = torch.quantile(samples, 0.025, dim=0)
            result[f"{key}_upper95"] = torch.quantile(samples, 0.975, dim=0)
        return result


def outcome_time_loss(logits: Tensor, terminal_class: Tensor) -> Tensor:
    """Supervised negative log-likelihood for the joint terminal label."""

    return F.cross_entropy(logits, terminal_class.long())
