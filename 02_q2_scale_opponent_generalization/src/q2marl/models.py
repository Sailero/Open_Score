"""Architecture-level HPN-style baseline and opponent-conditioned extension.

This is a clean-room, minimal implementation of the architectural contract used
in the Q2 study.  It is not copied from PyMARL3 and does not claim numerical
reproduction of the ICLR 2023 paper.  The contract is:

1. entity order is irrelevant for invariant actions;
2. reordering target entities reorders target-action logits in the same way;
3. parameter count is independent of the number of entities;
4. the proposed model adds only a fixed-dimensional temporal opponent context.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn


def _masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(x.dtype).unsqueeze(-1)
    return (x * weights).sum(dim=-2) / weights.sum(dim=-2).clamp_min(1.0)


def _masked_max(x: Tensor, mask: Tensor) -> Tensor:
    masked = x.masked_fill(~mask.unsqueeze(-1), torch.finfo(x.dtype).min)
    values = masked.max(dim=-2).values
    has_item = mask.any(dim=-1, keepdim=True)
    return torch.where(has_item, values, torch.zeros_like(values))


@dataclass
class PolicyOutput:
    fixed_logits: Tensor
    target_logits: Tensor
    value: Tensor
    recurrent_state: Tensor
    opponent_state: Optional[Tensor] = None
    auxiliary_prediction: Optional[Tensor] = None


class HyperEntityEncoder(nn.Module):
    """Entity-wise hypernetwork followed by a permutation-invariant reduction.

    For each entity x_e, a shared hypernetwork generates a small affine map.  The
    same operation is applied independently to all entities and reduced with a
    symmetric sum/sqrt(n), preserving invariance while avoiding scale explosion.
    """

    def __init__(self, entity_dim: int, hidden_dim: int):
        super().__init__()
        self.entity_dim = entity_dim
        self.hidden_dim = hidden_dim
        self.hyper = nn.Sequential(
            nn.Linear(entity_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, entity_dim * hidden_dim + hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def entity_embeddings(self, entities: Tensor) -> Tensor:
        generated = self.hyper(entities)
        weight_size = self.entity_dim * self.hidden_dim
        weights = generated[..., :weight_size].reshape(
            *entities.shape[:-1], self.entity_dim, self.hidden_dim
        )
        bias = generated[..., weight_size:]
        embedded = torch.einsum("...d,...dh->...h", entities, weights) + bias
        return torch.relu(embedded)

    def forward(self, entities: Tensor, mask: Tensor) -> Tensor:
        embeddings = self.entity_embeddings(entities)
        weights = mask.to(embeddings.dtype).unsqueeze(-1)
        count = weights.sum(dim=-2).clamp_min(1.0)
        pooled = (embeddings * weights).sum(dim=-2) / count.sqrt()
        return self.norm(pooled)


class HPNPolicy(nn.Module):
    """Minimal Hyper Policy Network-style PI/PE actor-critic.

    ``fixed_logits`` correspond to invariant actions such as movement/no-op.
    ``target_logits`` correspond one-to-one to enemy entities and therefore are
    permutation equivariant.  All masks use True for valid entities.
    """

    def __init__(
        self,
        entity_dim: int = 16,
        self_dim: int = 8,
        hidden_dim: int = 64,
        n_fixed_actions: int = 10,
    ):
        super().__init__()
        self.entity_dim = entity_dim
        self.self_encoder = nn.Sequential(
            nn.Linear(self_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.entity_encoder = HyperEntityEncoder(entity_dim, hidden_dim)
        self.pre_core = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim)
        )
        self.core = nn.GRUCell(hidden_dim, hidden_dim)
        self.fixed_head = nn.Linear(hidden_dim, n_fixed_actions)
        self.target_key = nn.Sequential(
            nn.Linear(entity_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.target_query = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        self.hidden_dim = hidden_dim

    def initial_state(self, batch_size: int, *, device=None, dtype=None) -> Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)

    def encode_agent(
        self,
        self_features: Tensor,
        entities: Tensor,
        entity_mask: Tensor,
        recurrent_state: Optional[Tensor],
    ) -> Tensor:
        if recurrent_state is None:
            recurrent_state = self.initial_state(
                self_features.shape[0], device=self_features.device, dtype=self_features.dtype
            )
        own = self.self_encoder(self_features)
        set_embedding = self.entity_encoder(entities, entity_mask)
        core_input = self.pre_core(torch.cat([own, set_embedding], dim=-1))
        return self.core(core_input, recurrent_state)

    def action_heads(
        self, hidden: Tensor, enemies: Tensor, enemy_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        fixed_logits = self.fixed_head(hidden)
        query = self.target_query(hidden).unsqueeze(-2)
        keys = self.target_key(enemies)
        target_logits = (query * keys).sum(dim=-1) / self.hidden_dim**0.5
        target_logits = target_logits.masked_fill(~enemy_mask, -1e9)
        return fixed_logits, target_logits

    def forward(
        self,
        self_features: Tensor,
        entities: Tensor,
        entity_mask: Tensor,
        enemies: Tensor,
        enemy_mask: Tensor,
        recurrent_state: Optional[Tensor] = None,
    ) -> PolicyOutput:
        hidden = self.encode_agent(self_features, entities, entity_mask, recurrent_state)
        fixed_logits, target_logits = self.action_heads(hidden, enemies, enemy_mask)
        return PolicyOutput(
            fixed_logits=fixed_logits,
            target_logits=target_logits,
            value=self.value_head(hidden).squeeze(-1),
            recurrent_state=hidden,
        )


class OpponentContextEncoder(nn.Module):
    """Fixed-dimensional online summary of a variable-size opponent set."""

    def __init__(self, entity_dim: int, context_dim: int):
        super().__init__()
        self.step_encoder = nn.Sequential(
            nn.Linear(entity_dim, context_dim), nn.ReLU(), nn.Linear(context_dim, context_dim)
        )
        self.count_encoder = nn.Sequential(nn.Linear(1, context_dim), nn.Tanh())
        self.step_fusion = nn.Sequential(
            nn.Linear(3 * context_dim, context_dim), nn.ReLU(), nn.LayerNorm(context_dim)
        )
        self.temporal_core = nn.GRUCell(context_dim, context_dim)
        self.context_dim = context_dim

    def initial_state(self, batch_size: int, *, device=None, dtype=None) -> Tensor:
        return torch.zeros(batch_size, self.context_dim, device=device, dtype=dtype)

    def forward(
        self, enemies: Tensor, enemy_mask: Tensor, state: Optional[Tensor]
    ) -> tuple[Tensor, Tensor]:
        encoded = self.step_encoder(enemies)
        mean = _masked_mean(encoded, enemy_mask)
        maximum = _masked_max(encoded, enemy_mask)
        # log1p prevents the raw population size from dominating the context.
        count = torch.log1p(enemy_mask.sum(dim=-1, keepdim=True).to(enemies.dtype))
        step = self.step_fusion(torch.cat([mean, maximum, self.count_encoder(count)], dim=-1))
        if state is None:
            state = self.initial_state(
                enemies.shape[0], device=enemies.device, dtype=enemies.dtype
            )
        new_state = self.temporal_core(step, state)
        return new_state, step


class OpponentConditionedHPN(HPNPolicy):
    """Recommended OC-HPN: HPN backbone + temporal opponent context + auxiliary loss."""

    def __init__(
        self,
        entity_dim: int = 16,
        self_dim: int = 8,
        hidden_dim: int = 64,
        context_dim: int = 32,
        n_fixed_actions: int = 10,
    ):
        super().__init__(entity_dim, self_dim, hidden_dim, n_fixed_actions)
        self.opponent_context = OpponentContextEncoder(entity_dim, context_dim)
        self.film = nn.Linear(context_dim, 2 * hidden_dim)
        self.context_gate = nn.Sequential(
            nn.Linear(hidden_dim + context_dim, hidden_dim), nn.Sigmoid()
        )
        self.auxiliary_head = nn.Sequential(
            nn.Linear(hidden_dim + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, context_dim),
        )

    def forward(
        self,
        self_features: Tensor,
        entities: Tensor,
        entity_mask: Tensor,
        enemies: Tensor,
        enemy_mask: Tensor,
        recurrent_state: Optional[Tensor] = None,
        opponent_state: Optional[Tensor] = None,
    ) -> PolicyOutput:
        hidden = self.encode_agent(self_features, entities, entity_mask, recurrent_state)
        opponent_state, opponent_step = self.opponent_context(
            enemies, enemy_mask, opponent_state
        )
        scale, shift = self.film(opponent_state).chunk(2, dim=-1)
        conditioned = hidden * (1.0 + 0.1 * torch.tanh(scale)) + shift
        gate = self.context_gate(torch.cat([hidden, opponent_state], dim=-1))
        conditioned = gate * conditioned + (1.0 - gate) * hidden
        fixed_logits, target_logits = self.action_heads(conditioned, enemies, enemy_mask)
        auxiliary = self.auxiliary_head(torch.cat([conditioned, opponent_state], dim=-1))
        return PolicyOutput(
            fixed_logits=fixed_logits,
            target_logits=target_logits,
            value=self.value_head(conditioned).squeeze(-1),
            recurrent_state=hidden,
            opponent_state=opponent_state,
            auxiliary_prediction=auxiliary,
        )

