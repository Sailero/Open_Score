"""Network definitions retained for v4 checkpoint inference."""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
from open_score.grouping.domain import Grouping


class CandidateNetwork(nn.Module):
    """Permutation invariant scoring of complete identity partitions.

    Entity IDs are used only to gather membership; numerical IDs never enter
    features. Group cardinality is unrestricted and encoded logarithmically.
    Separate instances MUST be used for an actor and its critic.
    """
    def __init__(self, hidden_dim=256, heads=8, layers=3):
        super().__init__()
        if hidden_dim < 1 or heads < 1 or hidden_dim % heads or layers < 1:
            raise ValueError('Invalid attention model dimensions')
        self.hidden_dim, self.heads, self.layers = hidden_dim, heads, layers
        self.entity_input = nn.Sequential(nn.Linear(11, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.context_input = nn.Linear(4, hidden_dim)
        self.previous_group = nn.Sequential(nn.Linear(2*hidden_dim+1, hidden_dim), nn.GELU())
        layer = nn.TransformerEncoderLayer(hidden_dim, heads, 2*hidden_dim,
                                          dropout=0., activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.group_head = nn.Sequential(nn.Linear(2*hidden_dim+1, hidden_dim), nn.GELU(),
                                        nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.score_head = nn.Sequential(nn.Linear(4*hidden_dim+2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.value_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    @property
    def config(self):
        return dict(hidden_dim=self.hidden_dim, heads=self.heads, layers=self.layers)

    def forward(self, states, candidate_batches):
        if not states or len(states) != len(candidate_batches) or any(not x for x in candidate_batches):
            raise ValueError('Nonempty states and matching candidate pools required')
        device = next(self.parameters()).device
        width = 1 + max(sum(len(s.alive(k)) for k in ('red', 'blue', 'targets')) for s in states)
        raw = np.zeros((len(states), width, 11), np.float32)
        padding = np.ones((len(states), width), bool)
        maps, contexts = [], []
        for b, state in enumerate(states):
            mapping, offset = {}, 1
            sides = [state.alive(k) for k in ('red', 'blue', 'targets')]
            for kind, entities in enumerate(sides):
                for entity in entities:
                    row = raw[b, offset]
                    row[kind] = 1.
                    row[3:6] = np.asarray(entity.position)/2500.
                    row[6:9] = np.asarray(entity.velocity)/500.
                    row[9] = entity.health/(1.2 if kind == 2 else 1.)
                    row[10] = state.step/max(1, state.max_steps)
                    mapping[kind, entity.id] = offset
                    offset += 1
            padding[b, :offset] = False
            maps.append(mapping)
            contexts.append([state.step/max(1, state.max_steps), *[math.log1p(len(x)) for x in sides]])
        base = self.entity_input(torch.as_tensor(raw, device=device))
        previous_members, previous_owners, previous_targets, previous_sizes = [], [], [], []
        for b, state in enumerate(states):
            for group in state.previous.prune(state.ids('red')).groups:
                if (2, group.target) not in maps[b]:
                    continue
                previous_members.extend(b*width+maps[b][0, i] for i in group.members)
                previous_owners.extend([len(previous_targets)]*len(group.members))
                previous_targets.append(b*width+maps[b][2, group.target])
                previous_sizes.append(len(group.members))
        flat = base.reshape(-1, self.hidden_dim)
        relation = torch.zeros_like(flat)
        if previous_targets:
            owner = torch.tensor(previous_owners, device=device)
            members = torch.tensor(previous_members, device=device)
            pooled = flat.new_zeros(len(previous_targets), self.hidden_dim).index_add(0, owner, flat[members])
            size = flat.new_tensor(previous_sizes)[:, None]
            representation = self.previous_group(torch.cat((pooled/size,
                flat[torch.tensor(previous_targets, device=device)], size.log1p()), -1))
            relation = relation.index_add(0, members, representation[owner])
        tokens = (flat+relation).reshape(len(states), width, self.hidden_dim)
        tokens = torch.cat((self.context_input(base.new_tensor(contexts))[:, None], tokens[:, 1:]), 1)
        encoded = self.encoder(tokens, src_key_padding_mask=torch.as_tensor(padding, device=device))
        kmax = max(map(len, candidate_batches))
        members, member_owners, targets, sizes, owners = [], [], [], [], []
        reserve_members, reserve_owners, features = [], [], []
        mask = np.ones((len(states), kmax), bool)
        for b, (state, pool) in enumerate(zip(states, candidate_batches)):
            for k, grouping in enumerate(pool):
                grouping.validate(state.ids('red'), state.ids('targets'), max_members=None)
                candidate_owner = b*kmax+k
                mask[b, k] = False
                for group in grouping.groups:
                    members.extend(b*width+maps[b][0, i] for i in group.members)
                    member_owners.extend([len(targets)]*len(group.members))
                    targets.append(b*width+maps[b][2, group.target])
                    sizes.append(len(group.members))
                    owners.append(candidate_owner)
                reserve_members.extend(b*width+maps[b][0, i] for i in grouping.reserve)
                reserve_owners.extend([candidate_owner]*len(grouping.reserve))
                features.append([math.log1p(len(grouping.groups)), math.log1p(len(grouping.reserve))])
            features.extend([[0., 0.]]*(kmax-len(pool)))
        flat = encoded.reshape(-1, self.hidden_dim)
        total = len(states)*kmax
        mean_group = flat.new_zeros(total, self.hidden_dim)
        max_group = flat.new_full((total, self.hidden_dim), -1e9)
        counts = flat.new_zeros(total, 1)
        if targets:
            pooled = flat.new_zeros(len(targets), self.hidden_dim).index_add(0,
                torch.tensor(member_owners, device=device), flat[torch.tensor(members, device=device)])
            size = flat.new_tensor(sizes)[:, None]
            group_values = self.group_head(torch.cat((pooled/size,
                flat[torch.tensor(targets, device=device)], size.log1p()), -1))
            owner = torch.tensor(owners, device=device)
            mean_group = mean_group.index_add(0, owner, group_values)
            counts = counts.index_add(0, owner, flat.new_ones(len(owners), 1))
            max_group = max_group.scatter_reduce(0, owner[:, None].expand(-1, self.hidden_dim),
                                                 group_values, reduce='amax', include_self=True)
        mean_group = mean_group/counts.clamp_min(1)
        max_group = torch.where(counts > 0, max_group, torch.zeros_like(max_group))
        reserve = flat.new_zeros(total, self.hidden_dim)
        reserve_counts = flat.new_zeros(total, 1)
        if reserve_members:
            owner = torch.tensor(reserve_owners, device=device)
            reserve = reserve.index_add(0, owner, flat[torch.tensor(reserve_members, device=device)])
            reserve_counts = reserve_counts.index_add(0, owner, flat.new_ones(len(reserve_members), 1))
        reserve = reserve/reserve_counts.clamp_min(1)
        complete = torch.cat((encoded[:, 0, None].expand(-1, kmax, -1).reshape(total, -1),
                              mean_group, max_group, reserve, flat.new_tensor(features)), -1)
        scores = self.score_head(complete).reshape(len(states), kmax)
        scores = scores.masked_fill(torch.as_tensor(mask, device=device), -1e9)
        return scores, self.value_head(encoded[:, 0]).squeeze(-1)


class StateValueNetwork(CandidateNetwork):
    """Independent state critic; do not score every actor candidate to get V."""
    def forward(self, states, candidate_batches):
        return super().forward(states, [[Grouping((), s.ids('red'))] for s in states])
