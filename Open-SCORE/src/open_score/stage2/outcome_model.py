"""DeepHit-style discrete competing-risk models for Stage 2."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class OutcomeTimeMLP(nn.Module):
    """Tabular MLP over ``2*K`` event masses plus one survival-tail mass."""

    def __init__(self, input_dim: int, horizon_bins: int = 20, hidden_dim: int = 64):
        super().__init__()
        if input_dim < 1 or horizon_bins < 1 or hidden_dim < 1:
            raise ValueError("all model dimensions must be positive")
        self.input_dim = input_dim
        self.horizon_bins = horizon_bins
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * horizon_bins + 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)

    def probabilities(self, features: Tensor, temperature: float = 1.0) -> Tensor:
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        return torch.softmax(self(features) / temperature, dim=-1)

    def summarize_probabilities(
        self,
        probabilities: Tensor,
        steps_per_bin: int = 10,
        command_bins: int = 1,
    ) -> Dict[str, Tensor]:
        if probabilities.shape[-1] != 2 * self.horizon_bins + 1:
            raise ValueError("probability class dimension does not match horizon bins")
        k = self.horizon_bins
        defender = probabilities[..., :k]
        breach = probabilities[..., k : 2 * k]
        survival_tail = probabilities[..., -1]
        bin_steps = (
            torch.arange(k, device=probabilities.device, dtype=probabilities.dtype) + 0.5
        ) * steps_per_bin
        restricted_mean = ((defender + breach) * bin_steps).sum(-1)
        restricted_mean += survival_tail * (k * steps_per_bin)
        command_bins = max(1, min(command_bins, k))
        return {
            "defender_success_probability": defender.sum(-1) + survival_tail,
            "defender_win_probability": defender.sum(-1),
            "breach_probability": breach.sum(-1),
            "breach_within_command": breach[..., :command_bins].sum(-1),
            "restricted_mean_remaining_steps": restricted_mean,
            # Compatibility alias; under censoring this is a restricted mean,
            # not an uncensored expected event time.
            "expected_remaining_steps": restricted_mean,
            "survival_through_horizon_probability": survival_tail,
        }

    def summarize(
        self, features: Tensor, steps_per_bin: int = 10, command_bins: int = 1
    ) -> Dict[str, Tensor]:
        return self.summarize_probabilities(
            self.probabilities(features), steps_per_bin, command_bins
        )


class HADDeepSetOutcomeNet(OutcomeTimeMLP):
    """Permutation-invariant entity encoder followed by a competing-risk head.

    The canonical HAD prefix is
    ``target(8), Red 4x9, Blue 4x9, counts(4), remaining_horizon(1)``.
    Red and Blue entity slots are pooled with masked sum and mean.  Consequently
    slot order cannot affect predictions, matching the Deep Sets construction.
    """

    HAD_STATE_DIM = 85
    SLOTS_PER_SIDE = 4
    ENTITY_DIM = 9

    def __init__(
        self,
        input_dim: int,
        horizon_bins: int = 20,
        hidden_dim: int = 64,
        state_dim: int = HAD_STATE_DIM,
    ):
        if state_dim != self.HAD_STATE_DIM or input_dim < state_dim:
            raise ValueError("had_deepset requires the 85-value HAD canonical prefix")
        nn.Module.__init__(self)
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.horizon_bins = horizon_bins
        self.hidden_dim = hidden_dim
        entity_hidden = max(16, hidden_dim // 2)
        self.entity_phi = nn.Sequential(
            nn.Linear(self.ENTITY_DIM, entity_hidden), nn.ReLU(),
            nn.Linear(entity_hidden, entity_hidden), nn.ReLU(),
        )
        pooled_dim = entity_hidden * 4
        context_dim = input_dim - state_dim
        head_input = 8 + pooled_dim + 5 + context_dim
        self.network = nn.Sequential(
            nn.Linear(head_input, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * horizon_bins + 1),
        )

    def _pool(self, entities: Tensor) -> Tensor:
        # Presence bits remain unnormalised by the formal feature normaliser.
        mask = entities[..., 8:9].clamp(0.0, 1.0)
        embedded = self.entity_phi(entities) * mask
        summed = embedded.sum(dim=-2)
        mean = summed / mask.sum(dim=-2).clamp_min(1.0)
        return torch.cat((summed, mean), dim=-1)

    def forward(self, features: Tensor) -> Tensor:
        if features.shape[-1] != self.input_dim:
            raise ValueError("feature dimension differs from HAD DeepSet model")
        target = features[..., :8]
        red_start = 8
        blue_start = red_start + self.SLOTS_PER_SIDE * self.ENTITY_DIM
        counts_start = blue_start + self.SLOTS_PER_SIDE * self.ENTITY_DIM
        red = features[..., red_start:blue_start].reshape(
            *features.shape[:-1], self.SLOTS_PER_SIDE, self.ENTITY_DIM
        )
        blue = features[..., blue_start:counts_start].reshape(
            *features.shape[:-1], self.SLOTS_PER_SIDE, self.ENTITY_DIM
        )
        counts = features[..., counts_start:self.state_dim]
        context = features[..., self.state_dim:]
        representation = torch.cat(
            (target, self._pool(red), self._pool(blue), counts, context), dim=-1
        )
        return self.network(representation)


def make_outcome_model(
    model_kind: str,
    input_dim: int,
    horizon_bins: int,
    hidden_dim: int,
    state_dim: Optional[int] = None,
) -> OutcomeTimeMLP:
    if model_kind == "mlp":
        return OutcomeTimeMLP(input_dim, horizon_bins, hidden_dim)
    if model_kind == "had_deepset":
        return HADDeepSetOutcomeNet(
            input_dim, horizon_bins, hidden_dim, state_dim or HADDeepSetOutcomeNet.HAD_STATE_DIM
        )
    raise ValueError("model_kind must be mlp or had_deepset")


def make_physical_head(
    model_kind: str,
    input_dim: int,
    target_count: int,
    hidden_dim: int,
    state_dim: Optional[int] = None,
) -> OutcomeTimeMLP:
    """Build a regression head with the same symmetry contract as the event head."""

    if target_count < 1:
        raise ValueError("target_count must be positive")
    head = make_outcome_model(
        model_kind, input_dim, horizon_bins=1, hidden_dim=hidden_dim, state_dim=state_dim
    )
    head.network[-1] = nn.Linear(hidden_dim, target_count)
    return head


class BootstrapOutcomeEnsemble(nn.Module):
    """Lineage-bootstrap ensemble exposing disagreement, not a CI guarantee."""

    def __init__(
        self,
        members: int,
        input_dim: int,
        horizon_bins: int = 20,
        hidden_dim: int = 64,
        model_kind: str = "mlp",
        state_dim: Optional[int] = None,
        physical_target_count: int = 0,
    ):
        super().__init__()
        if members < 2:
            raise ValueError("an uncertainty ensemble requires at least two members")
        self.input_dim = input_dim
        self.horizon_bins = horizon_bins
        self.hidden_dim = hidden_dim
        self.model_kind = model_kind
        self.state_dim = state_dim
        self.physical_target_count = int(physical_target_count)
        if self.physical_target_count < 0:
            raise ValueError("physical_target_count must be non-negative")
        self.members = nn.ModuleList(
            [
                make_outcome_model(
                    model_kind, input_dim, horizon_bins, hidden_dim, state_dim
                )
                for _ in range(members)
            ]
        )
        # A separate normalised regression head per bootstrap member makes
        # ensemble dispersion available for every physical indicator.  The
        # event-time head remains the censor-aware competing-risk model above.
        self.physical_heads = nn.ModuleList(
            [
                make_physical_head(
                    model_kind,
                    input_dim,
                    self.physical_target_count,
                    hidden_dim,
                    state_dim,
                )
                for _ in range(members)
            ]
            if self.physical_target_count
            else []
        )

    def forward(self, features: Tensor) -> Tensor:
        return torch.stack([member(features) for member in self.members], dim=0)

    def member_probabilities(self, features: Tensor) -> Tensor:
        return torch.softmax(self(features), dim=-1)

    def mean_probabilities(self, features: Tensor) -> Tensor:
        return self.member_probabilities(features).mean(dim=0)

    def physical_forward(self, features: Tensor) -> Tensor:
        """Return member x batch x target normalised physical predictions."""

        if not self.physical_heads:
            return features.new_empty((len(self.members), len(features), 0))
        return torch.stack([head(features) for head in self.physical_heads], dim=0)


def competing_risk_nll(
    logits: Tensor,
    event_class: Tensor,
    event_observed: Tensor,
    censor_bin: Tensor,
    horizon_bins: int,
    reduction: str = "mean",
) -> Tensor:
    """Negative log likelihood with proper right-censoring.

    For an observed event, use its cause-time probability mass.  For a
    censored rollout at bin ``c``, use the probability of either event after
    ``c`` plus the survival-tail mass.  This is the discrete competing-risk
    likelihood used by DeepHit-style models, without its optional ranking term.
    """

    probabilities = torch.softmax(logits, dim=-1)
    if probabilities.shape[-1] != 2 * horizon_bins + 1:
        raise ValueError("logit class dimension differs from horizon_bins")
    indices = torch.arange(len(probabilities), device=probabilities.device)
    observed_probability = probabilities[indices, event_class.long()]
    bins = torch.arange(horizon_bins, device=probabilities.device).unsqueeze(0)
    after = bins >= censor_bin.long().unsqueeze(1)
    survivor = (
        (probabilities[:, :horizon_bins] * after).sum(dim=1)
        + (probabilities[:, horizon_bins : 2 * horizon_bins] * after).sum(dim=1)
        + probabilities[:, -1]
    )
    likelihood = torch.where(event_observed.bool(), observed_probability, survivor)
    loss = -torch.log(likelihood.clamp_min(1e-12))
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError("reduction must be none, sum or mean")


def outcome_time_loss(logits: Tensor, terminal_class: Tensor) -> Tensor:
    """Backward-compatible observed-event categorical loss."""

    return F.cross_entropy(logits, terminal_class.long())


__all__ = [
    "BootstrapOutcomeEnsemble",
    "HADDeepSetOutcomeNet",
    "OutcomeTimeMLP",
    "competing_risk_nll",
    "make_outcome_model",
    "outcome_time_loss",
]
