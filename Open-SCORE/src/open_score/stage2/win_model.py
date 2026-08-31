"""Minimal dynamic-scale Stage-2 Red win-probability evaluator."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DynamicHADWinNet(nn.Module):
    """One Deep Sets classifier shared by every registered HAD scale."""

    STATE_DIM = 85
    SLOTS_PER_SIDE = 4
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

    def _pool(self, entities: Tensor) -> Tensor:
        presence = entities[..., 8:9].clamp(0.0, 1.0)
        embedded = self.entity_phi(entities) * presence
        summed = embedded.sum(dim=-2)
        mean = summed / presence.sum(dim=-2).clamp_min(1.0)
        return torch.cat((summed, mean), dim=-1)

    def forward(self, state: Tensor) -> Tensor:
        if state.shape[-1] != self.STATE_DIM:
            raise ValueError("dynamic HAD win model expects 85 state values")
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
        features = torch.cat(
            (
                target,
                self._pool(red),
                self._pool(blue),
                counts_and_horizon,
            ),
            dim=-1,
        )
        return self.head(features).squeeze(-1)

    def predict_probability(self, state: Tensor) -> Tensor:
        return torch.sigmoid(self(state))


__all__ = ["DynamicHADWinNet"]
