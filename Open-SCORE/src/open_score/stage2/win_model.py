"""Minimal dynamic-scale Stage-2 Red win-probability evaluator."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DynamicHADWinNet(nn.Module):
    """One Deep Sets classifier shared by arbitrary HAD roster sizes.

    ``forward`` retains the frozen 85-value round-01 serialization for
    checkpoint compatibility.  New deployment code should call
    ``forward_entities`` with dynamically padded Red and Blue sets; the padding
    length is chosen per batch and is not a model or environment limit.
    """

    STATE_DIM = 85  # legacy round-01 serialized input only
    SLOTS_PER_SIDE = 4  # legacy round-01 serialized input only
    ENTITY_DIM = 9

    def __init__(self, entity_hidden_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        if entity_hidden_dim < 1 or hidden_dim < 1:
            raise ValueError("hidden dimensions must be positive")
        self.entity_hidden_dim = int(entity_hidden_dim)
        self.hidden_dim = int(hidden_dim)
        self.entity_phi = nn.Sequential(
            nn.Linear(self.ENTITY_DIM, entity_hidden_dim),
            nn.ReLU(),
            nn.Linear(entity_hidden_dim, entity_hidden_dim),
            nn.ReLU(),
        )
        # target(8) + four pooled entity summaries + counts/horizon(5)
        head_input = 8 + 4 * entity_hidden_dim + 5
        self.head = nn.Sequential(
            nn.Linear(head_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _pool(self, entities: Tensor, mask: Tensor | None = None) -> Tensor:
        if entities.shape[-1] != self.ENTITY_DIM:
            raise ValueError("each dynamic HAD entity needs 9 state values")
        presence = entities[..., 8:9].clamp(0.0, 1.0)
        if mask is not None:
            if mask.shape != entities.shape[:-1]:
                raise ValueError("entity mask must match the entity-set shape")
            presence = presence * mask.unsqueeze(-1).to(presence.dtype)
        embedded = self.entity_phi(entities) * presence
        summed = embedded.sum(dim=-2)
        mean = summed / presence.sum(dim=-2).clamp_min(1.0)
        return torch.cat((summed, mean), dim=-1)

    def forward_entities(
        self,
        target: Tensor,
        red_entities: Tensor,
        blue_entities: Tensor,
        context: Tensor,
        red_mask: Tensor | None = None,
        blue_mask: Tensor | None = None,
    ) -> Tensor:
        """Predict logits from two entity sets with no fixed cardinality."""

        if target.shape[-1] != 8 or context.shape[-1] != 5:
            raise ValueError("dynamic HAD target/context dimensions must be 8 and 5")
        if target.shape[:-1] != context.shape[:-1]:
            raise ValueError("target and context batch dimensions must match")
        if red_entities.shape[:-2] != target.shape[:-1]:
            raise ValueError("Red entity-set batch dimensions must match target")
        if blue_entities.shape[:-2] != target.shape[:-1]:
            raise ValueError("Blue entity-set batch dimensions must match target")
        features = torch.cat(
            (
                target,
                self._pool(red_entities, red_mask),
                self._pool(blue_entities, blue_mask),
                context,
            ),
            dim=-1,
        )
        return self.head(features).squeeze(-1)

    def forward(self, state: Tensor) -> Tensor:
        """Compatibility path for the frozen round-01 85-value dataset."""

        if state.shape[-1] != self.STATE_DIM:
            raise ValueError(
                "legacy HAD input expects 85 values; use forward_entities for arbitrary rosters"
            )
        target = state[..., :8]
        red_start = 8
        blue_start = red_start + self.SLOTS_PER_SIDE * self.ENTITY_DIM
        counts_start = blue_start + self.SLOTS_PER_SIDE * self.ENTITY_DIM
        red = state[..., red_start:blue_start].reshape(
            *state.shape[:-1], self.SLOTS_PER_SIDE, self.ENTITY_DIM
        )
        blue = state[..., blue_start:counts_start].reshape(
            *state.shape[:-1], self.SLOTS_PER_SIDE, self.ENTITY_DIM
        )
        counts_and_horizon = state[..., counts_start:]
        return self.forward_entities(
            target,
            red,
            blue,
            counts_and_horizon,
            red[..., 8] > 0.0,
            blue[..., 8] > 0.0,
        )

    def predict_probability(self, state: Tensor) -> Tensor:
        return torch.sigmoid(self(state))

    def predict_entity_probability(
        self,
        target: Tensor,
        red_entities: Tensor,
        blue_entities: Tensor,
        context: Tensor,
        red_mask: Tensor | None = None,
        blue_mask: Tensor | None = None,
    ) -> Tensor:
        return torch.sigmoid(
            self.forward_entities(
                target,
                red_entities,
                blue_entities,
                context,
                red_mask,
                blue_mask,
            )
        )


__all__ = ["DynamicHADWinNet"]
