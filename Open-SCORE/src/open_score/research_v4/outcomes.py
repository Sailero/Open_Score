"""Local/global S2 network definitions and inference for retained models."""
from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from open_score.grouping.domain import DecisionState, Group, Grouping
from open_score.grouping.storage import fingerprint


def _features(entity, kind, state):
    return [float(kind == i) for i in range(3)] + [v/2500. for v in entity.position] + [
        v/500. for v in entity.velocity] + [entity.health/(1.2 if kind == 2 else 1.),
                                         max(0., 1.-state.step/state.max_steps)]


class GlobalOutcomeNetwork(nn.Module):
    """Permutation invariant, unbounded group sets with global interactions.

    First encode physical entities plus old group relations. Then attend over
    complete proposed group tokens, Blue entities and targets. Group tokens are
    *not* independently scored and summed. No S1 memory or action is an input.
    Forward follows ``(scores[B,K], values[B])`` for policy/search integration.
    """
    def __init__(self, hidden_dim=128, heads=4, layers=2):
        super().__init__()
        if min(hidden_dim, heads, layers) < 1 or hidden_dim % heads:
            raise ValueError('positive dimensions, divisible attention heads required')
        self.hidden_dim, self.heads, self.layers = hidden_dim, heads, layers
        self.entity = nn.Sequential(nn.Linear(11, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.context = nn.Linear(4, hidden_dim)
        self.previous_group = nn.Sequential(nn.Linear(2*hidden_dim+1, hidden_dim), nn.GELU())
        self.group = nn.Sequential(nn.Linear(3*hidden_dim+1, hidden_dim), nn.GELU(),
                                   nn.Linear(hidden_dim, hidden_dim))
        self.reserve = nn.Parameter(torch.zeros(hidden_dim))
        self.entity_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            hidden_dim, heads, 2*hidden_dim, dropout=0., batch_first=True,
            activation='gelu'), layers, enable_nested_tensor=False)
        self.partition_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            hidden_dim, heads, 2*hidden_dim, dropout=0., batch_first=True,
            activation='gelu'), layers, enable_nested_tensor=False)
        self.score_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.value_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    @property
    def config(self):
        return {'hidden_dim': self.hidden_dim, 'heads': self.heads, 'layers': self.layers}

    def forward(self, states, candidate_batches):
        if not states or len(states) != len(candidate_batches) or any(not p for p in candidate_batches):
            raise ValueError('nonempty matching states and candidate pools required')
        device = next(self.parameters()).device
        tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
        mappings, inputs = [], []
        for state in states:
            records = [(kind, e) for kind, side in enumerate(('red', 'blue', 'targets')) for e in state.alive(side)]
            mapping = {(kind, e.id): i+1 for i, (kind, e) in enumerate(records)}
            rows = self.entity(tensor([_features(e, kind, state) for kind, e in records]).reshape(-1, 11))
            old = state.previous.prune(state.ids('red'))
            relation = rows.new_zeros(rows.shape)
            for group in old.groups:
                if (2, group.target) not in mapping:
                    continue
                indices = torch.tensor([mapping[(0, i)]-1 for i in group.members], device=device)
                pooled = rows[indices].mean(0)
                token = self.previous_group(torch.cat((pooled, rows[mapping[(2, group.target)]-1],
                                                       tensor([math.log1p(len(group.members))]))))
                relation = relation.index_add(0, indices, token.expand(len(indices), -1))
            reserve_indices = [mapping[(0, i)]-1 for i in old.reserve]
            if reserve_indices:
                indices = torch.tensor(reserve_indices, device=device)
                relation = relation.index_add(0, indices, self.reserve.expand(len(indices), -1))
            context = self.context(tensor([state.step/state.max_steps, *[
                math.log1p(len(state.ids(side))) for side in ('red', 'blue', 'targets')]]))
            inputs.append(torch.cat((context[None], rows+relation), 0))
            mappings.append(mapping)
        padded = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
        mask = torch.arange(padded.shape[1], device=device)[None] >= torch.tensor([len(x) for x in inputs], device=device)[:, None]
        encoded = self.entity_encoder(padded, src_key_padding_mask=mask)
        proposals, owners, group_members, group_targets, group_owners = [], [], [], [], []
        proposal_groups = []
        width = encoded.shape[1]
        for b, (state, pool, mapping) in enumerate(zip(states, candidate_batches, mappings)):
            for k, candidate in enumerate(pool):
                candidate.validate(state.ids('red'), state.ids('targets'), max_members=None)
                groups_for_proposal = []
                for group in candidate.groups:
                    groups_for_proposal.append(len(group_members))
                    group_members.append([b*width+mapping[(0, i)] for i in group.members])
                    group_targets.append(b*width+mapping[(2, group.target)])
                proposal_groups.append(groups_for_proposal)
                owners.append((b, k))
        vectors = None
        if group_members:
            max_members = max(map(len, group_members))
            indices = torch.tensor([ids+[0]*(max_members-len(ids)) for ids in group_members], device=device)
            presence = torch.arange(max_members, device=device)[None] < torch.tensor([len(ids) for ids in group_members], device=device)[:, None]
            members = encoded.reshape(-1, self.hidden_dim)[indices] * presence[..., None]
            summed = members.sum(1)
            sizes = presence.sum(1, keepdim=True).to(summed.dtype)
            vectors = self.group(torch.cat((summed/sizes, summed,
                encoded.reshape(-1, self.hidden_dim)[torch.tensor(group_targets, device=device)], torch.log1p(sizes)), -1))
        for (b, k), group_indices in zip(owners, proposal_groups):
                state, candidate, mapping = states[b], candidate_batches[b][k], mappings[b]
                tokens = [encoded[b, 0:1]]
                if group_indices:
                    tokens.append(vectors[torch.tensor(group_indices, device=device)])
                if candidate.reserve:
                    tokens.append((encoded[b, [mapping[(0, i)] for i in candidate.reserve]].mean(0)+self.reserve)[None])
                context_indices = [index for (kind, _), index in mapping.items() if kind != 0]
                if context_indices:
                    tokens.append(encoded[b, context_indices])
                proposals.append(torch.cat(tokens))
        joined = nn.utils.rnn.pad_sequence(proposals, batch_first=True)
        mask = torch.arange(joined.shape[1], device=device)[None] >= torch.tensor([len(x) for x in proposals], device=device)[:, None]
        represented = self.partition_encoder(joined, src_key_padding_mask=mask)[:, 0]
        flat_scores = self.score_head(represented).squeeze(-1)
        width = max(map(len, candidate_batches))
        scores = flat_scores.new_full((len(states), width), -1e9)
        bi, ki = zip(*owners)
        scores = scores.index_put((torch.tensor(bi, device=device), torch.tensor(ki, device=device)), flat_scores)
        return scores, self.value_head(encoded[:, 0]).squeeze(-1)


class LocalOutcomeNetwork(nn.Module):
    """Target-relative isolated outcome model; groups may exceed four.

    The physical world itself is not translation invariant at its boundaries.
    Training includes the actual target layouts; features do not explicitly
    encode distance to the world boundary, a documented representation limit.
    """
    def __init__(self, hidden_dim=128, heads=4, layers=2):
        super().__init__()
        self.hidden_dim, self.heads, self.layers = hidden_dim, heads, layers
        self.entity = nn.Sequential(nn.Linear(11, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.head = nn.Sequential(nn.Linear(4*hidden_dim+14, hidden_dim), nn.GELU(),
                                  nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    @property
    def config(self):
        return {'hidden_dim': self.hidden_dim, 'heads': self.heads, 'layers': self.layers}

    def forward(self, states, candidate_batches):
        if not states or len(states) != len(candidate_batches) or any(not p for p in candidate_batches):
            raise ValueError('nonempty states and pools required')
        device = next(self.parameters()).device
        tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
        red_rows, blue_rows, contexts, owners = [], [], [], []
        for b, (state, pool) in enumerate(zip(states, candidate_batches)):
            reds, targets = {x.id: x for x in state.alive('red')}, {x.id: x for x in state.alive('targets')}
            for k, action in enumerate(pool):
                if len(action.groups) != 1 or action.reserve:
                    raise ValueError('local model requires a single deployed coalition')
                group = action.groups[0]
                target = targets[group.target]
                def centered(entity, kind):
                    values = _features(entity, kind, state)
                    values[3:6] = [(a-b)/2500. for a, b in zip(entity.position, target.position)]
                    return values
                red_rows.append([centered(reds[i], 0) for i in group.members])
                blue_rows.append([centered(e, 1) for e in state.alive('blue')])
                contexts.append(centered(target, 2)+[
                    math.log1p(len(group.members)), math.log1p(len(state.ids('blue'))), state.step/state.max_steps])
                owners.append((b, k))
        features = []
        for rows in (red_rows, blue_rows):
            width = max(1, max(map(len, rows)))
            values = np.zeros((len(rows), width, 11), dtype=np.float32)
            presence = np.zeros((len(rows), width, 1), dtype=np.float32)
            for i, row in enumerate(rows):
                if row:
                    values[i, :len(row)] = row
                    presence[i, :len(row)] = 1.
            mask = tensor(presence)
            summed = (self.entity(tensor(values))*mask).sum(1)
            features.extend((summed/mask.sum(1).clamp_min(1.), summed))
        flat = self.head(torch.cat([*features, tensor(contexts)], -1)).squeeze(-1)
        result = flat.new_full((len(states), max(map(len, candidate_batches))), -1e9)
        bi, ki = zip(*owners)
        result = result.index_put((torch.tensor(bi, device=device), torch.tensor(ki, device=device)), flat)
        return result, result[:, 0]*0.


def local_state(state, group, blue_ids):
    """Create only an inference view; never mutate shared simulator contacts."""
    red_set, blue_set = set(group.members), set(blue_ids)
    action = Grouping((group,))
    return replace(state, red=tuple(e for e in state.red if e.id in red_set),
                   blue=tuple(e for e in state.blue if e.id in blue_set),
                   targets=tuple(e for e in state.targets if e.id == group.target),
                   previous=action, memory={}, last_actions={})


class OutcomeScorer:
    def __init__(self, model=None, device='cpu', kind='global', metadata=None, counts=None):
        self.kind, self.metadata, self.counts = kind, metadata or {}, counts
        self.aggregation = 'product'
        self.coverage = {'local_queries': 0, 'unseen_roster_queries': 0,
                         'unseen_rosters': {}, 'undefended_reachability_proxy_queries': 0,
                         'neural_rows_evaluated': 0, 'cached_unique_queries': 0}
        self._cache_identity, self._local_cache = None, {}
        self.device = torch.device('cuda' if device == 'auto' and torch.cuda.is_available() else 'cpu' if device == 'auto' else device)
        self.model = model.to(self.device).eval() if model is not None else None

    def predict_local(self, state, group, blue_ids):
        if not blue_ids:
            return 1.0
        if not group.members:
            raise ValueError('no-defender cases use an explicit reachability proxy, not local network inference')
        self._record_local_query(group, blue_ids)
        if self.kind == 'count':
            cells = self.counts['cells']
            key = f'{len(group.members)}:{len(blue_ids)}'
            if key not in cells:
                key = min(cells, key=lambda k: (abs(int(k.split(':')[0])-len(group.members)) +
                                               abs(int(k.split(':')[1])-len(blue_ids)), k))
            return float(cells[key]['probability'])
        view = local_state(state, group, blue_ids)
        with torch.inference_mode():
            logits, _ = self.model([view], [[Grouping((group,))]])
        self.coverage['neural_rows_evaluated'] += 1
        return float(logits[0, 0].sigmoid().item())

    def _record_local_query(self, group, blue_ids):
        query_key = f'{len(group.members)}:{len(blue_ids)}'
        observed = (set(self.counts['cells']) if self.kind == 'count'
                    else set(self.metadata.get('training_rosters', [])))
        self.coverage['local_queries'] += 1
        if query_key not in observed:
            self.coverage['unseen_roster_queries'] += 1
            values = self.coverage['unseen_rosters']
            values[query_key] = values.get(query_key, 0)+1

    def _batched_local_probabilities(self, state, pairs):
        identity = (fingerprint(state.to_dict()),
                    next(self.model.parameters())._version if self.model is not None else None)
        if identity != self._cache_identity:
            self._local_cache, self._cache_identity = {}, identity
        requests = {(g.target, g.members, tuple(ids)): (g, tuple(ids))
                    for g, ids in pairs if ids}
        missing = [key for key in requests if key not in self._local_cache]
        self.coverage['cached_unique_queries'] += len(requests)-len(missing)
        # Historical/count wrappers may override predict_local and have no
        # neural model. Keep those definitions, including exact support checks.
        if self.model is None:
            for key in missing:
                group, ids = requests[key]
                self._local_cache[key] = self.predict_local(state, group, ids)
        elif missing:
            views = [local_state(state, *requests[key]) for key in missing]
            pools = [[Grouping((requests[key][0],))] for key in missing]
            with torch.inference_mode():
                logits, _ = self.model(views, pools)
            values = logits[:, 0].sigmoid().cpu().tolist()
            self.coverage['neural_rows_evaluated'] += len(missing)
            for key, value in zip(missing, values):
                self._record_local_query(*requests[key])
                self._local_cache[key] = value
        return self._local_cache

    def __call__(self, state, pool):
        if self.kind == 'global':
            with torch.inference_mode():
                logits, _ = self.model([state], [pool])
            return logits[0, :len(pool)].sigmoid().cpu().tolist()
        from .environment import responsibilities, threat_targets
        blue_lookup, targets = {e.id: e for e in state.alive('blue')}, {e.id: e for e in state.alive('targets')}
        scores = []
        pair_pools = [responsibilities(state, action) for action in pool]
        probabilities = self._batched_local_probabilities(state, [pair for pairs in pair_pools for pair in pairs])
        for pairs in pair_pools:
            covered = set()
            target_logs = {identity: 0. for identity in targets}
            for group, blue_ids in pairs:
                covered.update(blue_ids)
                probability = probabilities[(group.target, group.members, tuple(blue_ids))] if blue_ids else 1.
                target_logs[group.target] += math.log(max(1e-6, probability))
            # Uncovered attackers are handled by a documented conservative
            # reachability proxy. They are never silently deleted or labeled win.
            for blue_id in set(blue_lookup)-covered:
                blue = blue_lookup[blue_id]
                target_id = threat_targets(state).get(blue_id)
                target = targets.get(target_id)
                reachable = target is not None and np.linalg.norm(np.asarray(blue.position)-np.asarray(target.position)) <= 500.+300.*(state.max_steps-state.step)
                self.coverage['undefended_reachability_proxy_queries'] += 1
                if reachable:
                    target_logs[target_id] += math.log(1e-6)
            if self.aggregation not in ('product', 'min'):
                raise ValueError('local aggregation must be product or min over target products')
            scores.append((min(target_logs.values()) if target_logs else 0.)
                          if self.aggregation == 'min' else sum(target_logs.values()))
        return scores


def load_evaluator(path, device='cpu'):
    path = Path(path)
    if path.suffix == '.json':
        payload = json.loads(path.read_text(encoding='utf-8'))
        if payload.get('schema') != 'rule-s2-count-v1':
            raise ValueError('unknown count evaluator schema')
        return OutcomeScorer(kind='count', counts=payload, metadata=payload['metadata'])
    selected_device = 'cuda' if device == 'auto' and torch.cuda.is_available() else 'cpu' if device == 'auto' else device
    payload = torch.load(path, map_location=selected_device, weights_only=False)
    if payload.get('schema') != 'rule-s2-checkpoint-v1':
        raise ValueError('checkpoint is not a rule-grounded S2 evaluator')
    kind = payload['kind']
    model = (GlobalOutcomeNetwork if kind == 'global' else LocalOutcomeNetwork)(**payload['config']['model'])
    model.load_state_dict(payload['model'])
    return OutcomeScorer(model, selected_device, kind, payload['metadata'])
