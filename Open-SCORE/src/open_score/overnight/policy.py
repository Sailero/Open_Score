"""Batched entity attention and identity-free candidate coalition scoring."""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn

from open_score.grouping.domain import DecisionState, Grouping


class CandidateNetwork(nn.Module):
    """One scalar per complete proposal plus a global PPO state value.

    All learned operations process padded batches. Python preparation maps
    identities to gather indices; IDs themselves are never neural features.
    The same score head can represent categorical logits or candidate Q.
    """
    FEATURE_DIM = 102
    MEMORY_DIM = 64
    LOWER_ACTION_DIM = 27

    def __init__(self, hidden_dim=256, heads=8, layers=3):
        super().__init__()
        if hidden_dim <= 0 or heads <= 0 or hidden_dim % heads or layers <= 0:
            raise ValueError('positive dimensions, layers and divisible attention heads required')
        self.hidden_dim, self.heads, self.layers = hidden_dim, heads, layers
        self.entity_input = nn.Sequential(nn.Linear(self.FEATURE_DIM, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.context_input = nn.Linear(4, hidden_dim)
        self.previous_group = nn.Sequential(nn.Linear(2 * hidden_dim + 1, hidden_dim), nn.GELU())
        self.reserve_relation = nn.Parameter(torch.zeros(hidden_dim))
        layer = nn.TransformerEncoderLayer(hidden_dim, heads, dim_feedforward=2 * hidden_dim,
                                            dropout=0.0, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.group_head = nn.Sequential(nn.Linear(2 * hidden_dim + 5, hidden_dim), nn.GELU(),
                                        nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.score_head = nn.Sequential(nn.Linear(4 * hidden_dim + 4, hidden_dim), nn.GELU(),
                                        nn.Linear(hidden_dim, 1))
        self.value_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    @property
    def config(self):
        return {'hidden_dim': self.hidden_dim, 'heads': self.heads, 'layers': self.layers}

    def forward(self, states: list[DecisionState], candidate_batches: list[list[Grouping]]):
        if not states or len(states) != len(candidate_batches) or any(not pool for pool in candidate_batches):
            raise ValueError('nonempty states and matching nonempty candidate pools required')
        device, dtype = next(self.parameters()).device, next(self.parameters()).dtype
        batches = len(states)
        entities = [[*state.alive('red'), *state.alive('blue'), *state.alive('targets')] for state in states]
        width = max(len(row) for row in entities) + 1
        kmax = max(map(len, candidate_batches))
        raw = np.zeros((batches, width, self.FEATURE_DIM), dtype=np.float32)
        padding = np.ones((batches, width), dtype=bool)
        context, maps = [], []
        prev_members, prev_targets, prev_owner, prev_red = [], [], [], []
        candidate_members, candidate_targets, candidate_owner, geometry = [], [], [], []
        reserve_indices, reserve_owners, candidate_features = [], [], np.zeros((batches * kmax, 4), dtype=np.float32)
        candidate_mask = np.ones((batches, kmax), dtype=bool)
        for b, (state, pool) in enumerate(zip(states, candidate_batches)):
            if state.max_steps < 1:
                raise ValueError('state max_steps must be positive')
            sides = [state.alive(side) for side in ('red', 'blue', 'targets')]
            mapping, offset = {}, 1
            for kind, side in enumerate(sides):
                for entity in side:
                    row = raw[b, offset]
                    row[kind] = 1.0
                    row[3:6] = np.asarray(entity.position) / 2500.0
                    row[6:9] = np.asarray(entity.velocity) / 500.0
                    row[9] = entity.health / (1.2 if kind == 2 else 1.0)
                    row[10] = max(0.0, 1.0 - state.step / state.max_steps)
                    if kind == 0:
                        memory = state.memory.get(entity.id, (0.0,) * self.MEMORY_DIM)
                        if len(memory) != self.MEMORY_DIM:
                            raise ValueError('lower memory must have 64 entries')
                        row[11:75] = memory
                        action = int(state.last_actions.get(entity.id, -1))
                        if not -1 <= action < self.LOWER_ACTION_DIM:
                            raise ValueError('last lower action must be -1 or 0..26')
                        if action >= 0:
                            row[75 + action] = 1.0
                    mapping[(kind, entity.id)] = b * width + offset
                    offset += 1
            padding[b, :offset] = False
            context.append([state.step / state.max_steps, *(math.log1p(len(side)) for side in sides)])
            maps.append(mapping)
            previous = state.previous.prune(state.ids('red'))
            for group in previous.groups:
                if (2, group.target) not in mapping:
                    continue
                member_indices = [mapping[(0, i)] for i in group.members]
                prev_members.append(member_indices + [b * width] * (4 - len(member_indices)))
                prev_targets.append(mapping[(2, group.target)])
                group_index = len(prev_members) - 1
                prev_owner.extend([group_index] * len(member_indices))
                prev_red.extend(member_indices)
            positions = {entity.id: np.asarray(entity.position) for entity in sides[0]}
            target_positions = {entity.id: np.asarray(entity.position) for entity in sides[2]}
            old_assignments = previous.assignment()
            for k, candidate in enumerate(pool):
                candidate.validate(state.ids('red'), state.ids('targets'))
                owner = b * kmax + k
                candidate_mask[b, k] = False
                assignments = candidate.assignment()
                changed = sum(assignments.get(i) != old_assignments.get(i) for i in state.ids('red'))
                candidate_features[owner] = [len(candidate.groups) / max(1, len(sides[0])),
                                             len(candidate.reserve) / max(1, len(sides[0])),
                                             changed / max(1, len(sides[0])),
                                             sum(len(g.members) == 1 for g in candidate.groups) / max(1, len(candidate.groups))]
                for group in candidate.groups:
                    indices = [mapping[(0, i)] for i in group.members]
                    candidate_members.append(indices + [b * width] * (4 - len(indices)))
                    candidate_targets.append(mapping[(2, group.target)])
                    candidate_owner.append(owner)
                    relative = np.asarray([positions[i] - target_positions[group.target] for i in group.members]) / 2500.0
                    center = relative.mean(0)
                    spread = float(np.linalg.norm(relative - center, axis=1).mean())
                    geometry.append([len(indices) / 4.0, *center, spread])
                reserve_indices.extend(mapping[(0, i)] for i in candidate.reserve)
                reserve_owners.extend([owner] * len(candidate.reserve))
        tensor = lambda values, kind=dtype: torch.as_tensor(values, dtype=kind, device=device)
        base = self.entity_input(tensor(raw))
        # Context/padding positions cannot act as synthetic zero-valued members.
        context_mask = torch.zeros((batches, width, 1), device=device, dtype=dtype)
        context_mask[:, 1:] = 1.0
        base = base * context_mask
        flat = base.reshape(-1, self.hidden_dim)
        relation = flat.new_zeros(flat.shape)
        if prev_members:
            member_idx = tensor(prev_members, torch.long)
            member_mask = (member_idx.remainder(width) != 0).to(dtype)
            pooled = (flat[member_idx] * member_mask.unsqueeze(-1)).sum(1) / member_mask.sum(1, keepdim=True).clamp_min(1)
            previous_tokens = self.previous_group(torch.cat((pooled, flat[tensor(prev_targets, torch.long)],
                                                              member_mask.sum(1, keepdim=True) / 4.0), -1))
            relation = relation.index_add(0, tensor(prev_red, torch.long), previous_tokens[tensor(prev_owner, torch.long)])
        reserve_entity_indices = [mapping[(0, entity.id)] for state, mapping in zip(states, maps)
                                  for entity in state.alive('red') if entity.id in state.previous.reserve]
        if reserve_entity_indices:
            relation = relation.index_add(0, tensor(reserve_entity_indices, torch.long),
                                          self.reserve_relation.expand(len(reserve_entity_indices), -1))
        tokens = (flat + relation).reshape(batches, width, self.hidden_dim)
        # Functional concatenation avoids in-place autograd version hazards.
        tokens = torch.cat((self.context_input(tensor(context)).unsqueeze(1), tokens[:, 1:]), dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=tensor(padding, torch.bool))
        flat = encoded.reshape(-1, self.hidden_dim)
        total_candidates = batches * kmax
        mean_groups = flat.new_zeros((total_candidates, self.hidden_dim))
        max_groups = flat.new_full((total_candidates, self.hidden_dim), -1e9)
        counts = flat.new_zeros((total_candidates, 1))
        if candidate_members:
            member_idx = tensor(candidate_members, torch.long)
            member_mask = (member_idx.remainder(width) != 0).to(dtype)
            pooled = (flat[member_idx] * member_mask.unsqueeze(-1)).sum(1) / member_mask.sum(1, keepdim=True).clamp_min(1)
            group_vectors = self.group_head(torch.cat((pooled, flat[tensor(candidate_targets, torch.long)], tensor(geometry)), -1))
            owners = tensor(candidate_owner, torch.long)
            mean_groups = mean_groups.index_add(0, owners, group_vectors)
            counts = counts.index_add(0, owners, flat.new_ones((len(owners), 1)))
            max_groups = max_groups.scatter_reduce(0, owners[:, None].expand(-1, self.hidden_dim), group_vectors,
                                                   reduce='amax', include_self=True)
        mean_groups = mean_groups / counts.clamp_min(1)
        max_groups = torch.where(counts > 0, max_groups, torch.zeros_like(max_groups))
        reserve = flat.new_zeros((total_candidates, self.hidden_dim))
        reserve_counts = flat.new_zeros((total_candidates, 1))
        if reserve_indices:
            owners = tensor(reserve_owners, torch.long)
            reserve = reserve.index_add(0, owners, flat[tensor(reserve_indices, torch.long)])
            reserve_counts = reserve_counts.index_add(0, owners, flat.new_ones((len(owners), 1)))
        reserve = reserve / reserve_counts.clamp_min(1)
        global_context = encoded[:, 0]
        features = torch.cat((global_context[:, None].expand(-1, kmax, -1).reshape(total_candidates, -1),
                              mean_groups, max_groups, reserve, tensor(candidate_features)), -1)
        scores = self.score_head(features).reshape(batches, kmax)
        scores = scores.masked_fill(tensor(candidate_mask, torch.bool), -1e9)
        values = self.value_head(global_context).squeeze(-1)
        return scores, values
