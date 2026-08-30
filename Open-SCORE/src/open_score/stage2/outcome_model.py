"""Supervised joint outcome-and-time models for one canonical small subgame."""

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
        if input_dim < 1 or horizon_bins < 1 or hidden_dim < 1:
            raise ValueError("all model dimensions must be positive")
        self.input_dim = input_dim
        self.horizon_bins = horizon_bins
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * horizon_bins + 1),
        )

    def forward(self, canonical_state: Tensor) -> Tensor:
        return self.network(canonical_state)

    def probabilities(self, canonical_state: Tensor, temperature: float = 1.0) -> Tensor:
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        return torch.softmax(self(canonical_state) / temperature, dim=-1)

    def summarize_probabilities(
        self,
        probabilities: Tensor,
        steps_per_bin: int = 10,
        command_bins: int = 1,
    ) -> Dict[str, Tensor]:
        """Derive coherent and directly interpretable quantities."""

        if probabilities.shape[-1] != 2 * self.horizon_bins + 1:
            raise ValueError("probability class dimension does not match horizon bins")
        if steps_per_bin < 1:
            raise ValueError("steps_per_bin must be positive")
        k = self.horizon_bins
        defender = probabilities[..., :k]
        breach = probabilities[..., k : 2 * k]
        timeout = probabilities[..., -1]
        bin_steps = (
            torch.arange(k, device=probabilities.device, dtype=probabilities.dtype) + 0.5
        ) * steps_per_bin
        expected_steps = ((defender + breach) * bin_steps).sum(-1)
        expected_steps = expected_steps + timeout * (k * steps_per_bin)
        command_bins = max(1, min(command_bins, k))
        defender_success = defender.sum(-1) + timeout
        return {
            "defender_success_probability": defender_success,
            # Backward-compatible alias used by the original scaffold.
            "defender_win_probability": defender_success,
            "breach_probability": breach.sum(-1),
            "breach_within_command": breach[..., :command_bins].sum(-1),
            "expected_remaining_steps": expected_steps,
            "timeout_probability": timeout,
        }

    def summarize(
        self,
        canonical_state: Tensor,
        steps_per_bin: int = 10,
        command_bins: int = 1,
    ) -> Dict[str, Tensor]:
        return self.summarize_probabilities(
            self.probabilities(canonical_state), steps_per_bin, command_bins
        )


class BootstrapOutcomeEnsemble(nn.Module):
    """Bootstrap MLP ensemble exposing disagreement, not coverage guarantees."""

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
        self.input_dim = input_dim
        self.horizon_bins = horizon_bins
        self.hidden_dim = hidden_dim
        self.members = nn.ModuleList(
            [OutcomeTimeMLP(input_dim, horizon_bins, hidden_dim) for _ in range(members)]
        )

    def forward(self, canonical_state: Tensor) -> Tensor:
        """Return member logits with shape ``[members, batch, classes]``."""

        return torch.stack([member(canonical_state) for member in self.members], dim=0)

    def member_probabilities(self, canonical_state: Tensor) -> Tensor:
        return torch.softmax(self(canonical_state), dim=-1)

    def mean_probabilities(self, canonical_state: Tensor) -> Tensor:
        return self.member_probabilities(canonical_state).mean(dim=0)

    @torch.no_grad()
    def predict_interval(
        self,
        canonical_state: Tensor,
        steps_per_bin: int = 10,
        command_bins: int = 1,
    ) -> Dict[str, Tensor]:
        """Return quantiles explicitly labelled as ensemble disagreement."""

        summaries = [
            member.summarize(canonical_state, steps_per_bin, command_bins)
            for member in self.members
        ]
        result: Dict[str, Tensor] = {}
        for key in summaries[0]:
            samples = torch.stack([summary[key] for summary in summaries], dim=0)
            result[key] = samples.mean(dim=0)
            result[f"{key}_ensemble_q025"] = torch.quantile(samples, 0.025, dim=0)
            result[f"{key}_ensemble_q975"] = torch.quantile(samples, 0.975, dim=0)
            result[f"{key}_ensemble_std"] = samples.std(dim=0, unbiased=False)
        return result


def outcome_time_loss(logits: Tensor, terminal_class: Tensor) -> Tensor:
    """Supervised negative log-likelihood for the joint terminal label."""

    return F.cross_entropy(logits, terminal_class.long())
