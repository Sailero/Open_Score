"""Rule-grounded local/global S2 models and decision-relevant held-out audits.

The global model scores a *complete* partition, including cross-group context.
The local model learns genuine isolated one-target outcomes; its log-product
use in the shared world remains an explicitly named proxy, never a calibrated
global probability. Neither model assumes diminishing coalition returns.
"""
from __future__ import annotations

from collections import defaultdict
import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from open_score.grouping.domain import DecisionState, Group, Grouping
from open_score.grouping.storage import (atomic_checkpoint, atomic_json, append_jsonl,
    fingerprint, random_state, restore_random_state, seed_everything, sha256)


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


def _row_prediction(scorer, row):
    state, action = DecisionState.from_dict(row['state']), Grouping.from_dict(row['action'])
    if scorer.kind in ('local', 'count') and row.get('kind') == 'local':
        return scorer.predict_local(state, action.groups[0], state.ids('blue'))
    if scorer.kind != 'global':
        raise ValueError('local log-product is not a calibrated global success probability')
    return scorer(state, [action])[0]


def evaluate_evaluator(scorer, rows):
    """Held-out probability and same-state action ranking, with explicit ties."""
    if not rows:
        return {'rows': 0, 'status': 'no_data'}
    predictions = [_row_prediction(scorer, row) for row in rows]
    grouped, families = defaultdict(list), defaultdict(list)
    for index, (row, probability) in enumerate(zip(rows, predictions)):
        grouped[row['state_id']].append((row, probability))
        families[row['family_id']].append(index)
    weights = np.zeros(len(rows), dtype=float)
    for indices in families.values():
        weights[indices] = 1./(len(families)*len(indices))
    p = np.clip(np.asarray(predictions, float), 1e-6, 1.-1e-6)
    # Score every Bernoulli terminal, not only the sample-mean label: the latter
    # would incorrectly remove irreducible outcome variability from Brier.
    brier = np.asarray([np.mean([(v-y)**2 for y in r['outcomes']]) for r, v in zip(rows, p)])
    nll = np.asarray([-np.mean([y*np.log(v)+(1-y)*np.log(1-v) for y in r['outcomes']]) for r, v in zip(rows, p)])
    labels = np.asarray([r['y'] for r in rows], float)
    bins, ece = [], 0.
    bin_ids = np.minimum((p*10).astype(int), 9)
    for bin_id in range(10):
        left, selected = bin_id/10., bin_ids == bin_id
        mass = weights[selected].sum()
        if mass <= 0:
            continue
        predicted = float(np.average(p[selected], weights=weights[selected]))
        observed = float(np.average(labels[selected], weights=weights[selected]))
        bins.append({'left': float(left), 'rows': int(selected.sum()), 'mass': float(mass),
                     'predicted': predicted, 'observed': observed})
        ece += mass*abs(predicted-observed)
    comparisons = correct = score_ties = outcome_ties = 0
    regret, all_equal, difference_error = [], 0, []
    for values in grouped.values():
        truth, estimated = np.asarray([v[0]['y'] for v in values]), np.asarray([v[1] for v in values])
        if len(values) < 2:
            continue
        all_equal += int(np.all(truth == truth[0]))
        # Validation oracle is a sample maximum, not true optimal value.
        regret.append(float(truth.max()-truth[estimated.argmax()]))
        for i in range(len(values)):
            for j in range(i):
                d, q = truth[i]-truth[j], estimated[i]-estimated[j]
                difference_error.append(abs(d-q))
                if d == 0:
                    outcome_ties += 1
                else:
                    comparisons += 1
                    score_ties += int(abs(q) < 1e-10)
                    correct += int(d*q > 0)
    return {'rows': len(rows), 'families': len(families), 'status': 'complete',
            'brier': float(weights@brier), 'nll': float(weights@nll), 'ece_10': float(ece),
            'calibration': bins, 'mean_prediction': float(weights@p),
            'observed_success': float(weights@labels), 'paired_states': len(regret),
            'all_equal_states': all_equal, 'non_tied_pairs': comparisons,
            'outcome_tied_pairs': outcome_ties, 'prediction_tied_pairs': score_ties,
            'pairwise_accuracy': correct/comparisons if comparisons else None,
            'sample_oracle_regret': float(np.mean(regret)) if regret else None,
            'pair_difference_mae': float(np.mean(difference_error)) if difference_error else None,
            'coverage': copy.deepcopy(getattr(scorer, 'coverage', {})),
            'uncertainty_note': 'paired Monte Carlo labels; empirical best is not true oracle; family-weighted probability metrics'}


def _load_rows(data_dir):
    from .data import read_dataset
    manifest = json.loads((Path(data_dir)/'manifest.json').read_text(encoding='utf-8'))
    rows = read_dataset(data_dir)
    # Re-check durable content, not just a previously written status string.
    actual = {p.name: sha256(p) for p in sorted((Path(data_dir)/'families').glob('*.json'))}
    if fingerprint(actual) != manifest['dataset_hash']:
        raise ValueError('dataset families changed since manifest')
    families = defaultdict(set)
    for row in rows:
        families[row['family_id']].add(row['split'])
    if any(len(splits) != 1 for splits in families.values()):
        raise ValueError('family leakage across dataset splits')
    return manifest, rows


def train_evaluator(data_dir, output, *, kind='global', epochs=40, seconds=None,
                    device='cpu', resume=False, hidden_dim=128, heads=4, layers=2,
                    seed=20260906, batch_size=64, learning_rate=2e-4):
    """Proper Bernoulli fit, validation-only selection, atomic recoverable state."""
    manifest, rows = _load_rows(data_dir)
    from .data import source_identity
    training_source = source_identity()
    if manifest['config']['source']['hash'] != training_source['hash']:
        raise ValueError('dataset physics/executor/continuation source differs from current training source')
    if manifest['config']['kind'] != kind or kind not in ('global', 'local'):
        raise ValueError('model and dataset semantics differ')
    train = [r for r in rows if r['split'] == 'train']
    validation = [r for r in rows if r['split'] == 'validation']
    test = [r for r in rows if r['split'] == 'test']
    if not train or not validation or not test:
        raise ValueError('independent train, validation and test families required')
    output = Path(output)
    config = {'kind': kind, 'model': {'hidden_dim': hidden_dim, 'heads': heads, 'layers': layers},
              'dataset_hash': manifest['dataset_hash'], 'dataset_identity': manifest['identity'],
              'training_source_hash': training_source['hash'],
              'seed': int(seed), 'learning_rate': learning_rate, 'batch_size': batch_size}
    identity = fingerprint(config)
    path = output/'latest.pt'
    seed_everything(seed)
    model = (GlobalOutcomeNetwork if kind == 'global' else LocalOutcomeNetwork)(hidden_dim, heads, layers)
    metadata = {**manifest['config'], 'training_rosters': sorted({
        f'{len(DecisionState.from_dict(r["state"]).ids("red"))}:{len(DecisionState.from_dict(r["state"]).ids("blue"))}'
        for r in train})}
    if kind == 'local':
        metadata['coordinate_system'] = 'positions relative to defended target; no explicit world-edge distances'
        metadata['representation_limit'] = 'target layouts are sampled; boundary effects are not assumed translation invariant'
    scorer = OutcomeScorer(model, device, kind, metadata)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    start_epoch, best, history, elapsed = 0, float('inf'), [], 0.
    if path.exists():
        if not resume:
            raise ValueError('evaluator exists; use resume or a new output')
        payload = torch.load(path, map_location=scorer.device, weights_only=False)
        if payload['identity'] != identity:
            raise ValueError('evaluator data, source, policy or model configuration changed')
        model.load_state_dict(payload['model'])
        optimizer.load_state_dict(payload['optimizer'])
        start_epoch, best, history, elapsed = payload['epoch'], payload['best_validation_brier'], payload['history'], payload['elapsed_seconds']
        restore_random_state(payload['rng'])
    elif output.exists() and any(output.iterdir()):
        raise ValueError('nonempty evaluator directory has no recoverable checkpoint')
    output.mkdir(parents=True, exist_ok=True)
    family_counts = defaultdict(int)
    for row in train:
        family_counts[row['family_id']] += 1
    sample_weights = np.asarray([1./family_counts[r['family_id']] for r in train])
    sample_weights /= sample_weights.mean()
    started = time.monotonic()
    if not path.exists():
        # A valid initialized checkpoint exists even when the first optimizer
        # step fails or the time allowance expires before training starts.
        model.eval()
        best = evaluate_evaluator(scorer, validation)['brier']
        initial = {'schema': 'rule-s2-checkpoint-v1', 'identity': identity, 'config': config,
                   'metadata': metadata, 'kind': kind, 'model': model.state_dict(),
                   'optimizer': optimizer.state_dict(), 'epoch': 0,
                   'best_validation_brier': best, 'history': [],
                   'elapsed_seconds': time.monotonic()-started, 'rng': random_state()}
        atomic_checkpoint(path, initial)
        atomic_checkpoint(output/'best.pt', initial)
    elif start_epoch == 0 and not (output/'best.pt').exists():
        atomic_checkpoint(output/'best.pt', payload)
    for epoch in range(start_epoch, int(epochs)):
        if seconds is not None and elapsed+time.monotonic()-started >= float(seconds):
            break
        model.train()
        order, losses = np.random.permutation(len(train)), []
        for offset in range(0, len(order), batch_size):
            indices = order[offset:offset+batch_size]
            batch = [train[i] for i in indices]
            states = [DecisionState.from_dict(r['state']) for r in batch]
            pools = [[Grouping.from_dict(r['action'])] for r in batch]
            logits, _ = model(states, pools)
            targets = torch.tensor([r['y'] for r in batch], device=scorer.device)
            weights = torch.as_tensor(sample_weights[indices], dtype=torch.float32, device=scorer.device)
            loss = (F.binary_cross_entropy_with_logits(logits[:, 0], targets, reduction='none')*weights).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite S2 loss; previous complete checkpoint preserved')
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        valid_report = evaluate_evaluator(scorer, validation)
        improved = valid_report['brier'] < best
        best = min(best, valid_report['brier'])
        history.append({'epoch': epoch+1, 'train_nll': float(np.mean(losses)),
                        'validation_brier': valid_report['brier'],
                        'validation_pairwise_accuracy': valid_report['pairwise_accuracy']})
        payload = {'schema': 'rule-s2-checkpoint-v1', 'identity': identity, 'config': config,
                   'metadata': metadata, 'kind': kind, 'model': model.state_dict(),
                   'optimizer': optimizer.state_dict(), 'epoch': epoch+1,
                   'best_validation_brier': best, 'history': history,
                   'elapsed_seconds': elapsed+time.monotonic()-started, 'rng': random_state()}
        if improved:
            atomic_checkpoint(output/'best.pt', payload)
        atomic_checkpoint(path, payload)
        atomic_json(output/'training_history.json', history)
    selected = load_evaluator(output/'best.pt', device)
    report = {'kind': kind, 'identity': identity, 'epochs_completed': len(history),
              'trained': bool(history),
              'parameters': sum(p.numel() for p in model.parameters()),
              'selection': 'minimum family-weighted validation Bernoulli Brier',
              'validation': evaluate_evaluator(selected, validation),
              'test': evaluate_evaluator(selected, test),
              'checkpoint': str((output/'best.pt').resolve()),
              'test_by_roster': {f'{r}:{b}': evaluate_evaluator(selected, [x for x in test
                   if (x['initial_red'], x['initial_blue']) == (r, b)])
                   for r, b in sorted({(x['initial_red'], x['initial_blue']) for x in test})}}
    atomic_json(output/'report.json', report)
    _plot_history(output, history)
    return report


def _plot_history(output, history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, key, title in zip(axes, ('train_nll', 'validation_brier'), ('Training terminal NLL', 'Independent validation Brier')):
        ax.plot([r['epoch'] for r in history], [r[key] for r in history])
        ax.set(xlabel='Epoch', title=title)
        ax.grid(alpha=.2)
    figure.tight_layout()
    figure.savefig(Path(output)/'training_curves.png', dpi=150)
    plt.close(figure)


def train_count_model(data_dir, output):
    manifest, rows = _load_rows(data_dir)
    if manifest['config']['kind'] != 'local':
        raise ValueError('count baseline requires isolated rule labels')
    cells = defaultdict(lambda: {'success': 0., 'mass': 0.})
    counts = defaultdict(int)
    for row in rows:
        if row['split'] == 'train':
            counts[row['family_id']] += 1
    for row in rows:
        if row['split'] != 'train':
            continue
        state = DecisionState.from_dict(row['state'])
        key = f'{len(state.ids("red"))}:{len(state.ids("blue"))}'
        weight = 1./counts[row['family_id']]
        cells[key]['success'] += weight*row['y']
        cells[key]['mass'] += weight
    for cell in cells.values():
        cell['probability'] = (cell['success']+.5)/(cell['mass']+1.)
    payload = {'schema': 'rule-s2-count-v1', 'kind': 'count', 'cells': dict(cells),
               'metadata': manifest['config'], 'dataset_hash': manifest['dataset_hash'],
               'unseen_roster_rule': 'nearest Manhattan count pair; lexicographic deterministic tie',
               'scope': 'isolated rule outcome; no geometry; family-weighted Jeffreys smoothing'}
    atomic_json(output, payload)
    scorer = OutcomeScorer(kind='count', counts=payload, metadata=manifest['config'])
    test = [r for r in rows if r['split'] == 'test']
    report = evaluate_evaluator(scorer, test)
    report['trained_rosters'] = sorted(cells)
    atomic_json(Path(output).with_suffix('.report.json'), report)
    return report


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


def historical_transfer_audit(data_dir, checkpoint=None):
    """Zero-shot old-S2 transfer only within its native 1..4 roster domain."""
    from open_score.grouping.baselines import FrozenDLOM
    root = Path(__file__).resolve().parents[3]
    checkpoint = Path(checkpoint or root/'assets/frozen/dlom.pt')
    if not checkpoint.exists():
        return {'status': 'historical_checkpoint_unavailable', 'metrics': None,
                'checkpoint': str(checkpoint),
                'claim': 'new rule S2 training does not require historical frozen weights'}
    manifest, rows = _load_rows(data_dir)
    if manifest['config']['kind'] != 'local':
        raise ValueError('historical local transfer must use isolated local terminal labels')
    selected = [r for r in rows if r['split'] == 'test' and
                1 <= len(DecisionState.from_dict(r['state']).ids('red')) <= 4 and
                1 <= len(DecisionState.from_dict(r['state']).ids('blue')) <= 4]
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    report = {'schema': 'old-s2-rule-transfer-v1', 'checkpoint_sha256': sha256(checkpoint),
              'original_execution_semantics': payload['execution_semantics'],
              'new_execution_semantics': manifest['config'], 'eligible_rows': len(selected),
              'excluded_test_rows': sum(r['split'] == 'test' for r in rows)-len(selected),
              'claim': 'zero-shot executor transfer within original roster domain; not a global grouping value',
              'shape_priors_in_old_checkpoint': payload.get('shape_regularization', {}),
              'style': 'rush (the fixed physical Blue movement rule)'}
    if not selected:
        report.update(status='no_in_domain_test_data', metrics=None)
        return report
    frozen = FrozenDLOM(checkpoint)
    class Historical:
        kind = 'local'
        def predict_local(self, state, group, blue_ids):
            return frozen.predict(state, [(group.target, group.members, tuple(blue_ids))], 'rush')[0]
    report.update(status='complete', metrics=evaluate_evaluator(Historical(), selected))
    return report


def cross_evaluator_comparison(data_dir, local_scorer, global_scorer, checkpoint=None):
    """Compare evaluators on identical global states and complete candidate sets.

    For the three-way comparison an entire state/pool is excluded if *any*
    historical local query exceeds its trained 1..4 support. The larger new
    local/global comparison is separately reported; candidates are never dropped
    selectively to flatter the old model. Local products remain uncalibrated
    proxy probabilities; Brier against real global outcomes tests that proxy.
    """
    from .environment import responsibilities
    from open_score.grouping.baselines import FrozenDLOM

    manifest, rows = _load_rows(data_dir)
    if manifest['config']['kind'] != 'global':
        raise ValueError('cross-evaluator ranking requires complete global proposal outcomes')
    rows = [r for r in rows if r['split'] == 'test']
    pools = defaultdict(list)
    for row in rows:
        pools[row['state_id']].append(row)
    eligible_states, excluded = [], {}
    for state_id, values in pools.items():
        violations = set()
        for row in values:
            state, action = DecisionState.from_dict(row['state']), Grouping.from_dict(row['action'])
            for group, blue_ids in responsibilities(state, action):
                if blue_ids and not (1 <= len(group.members) <= 4 and 1 <= len(blue_ids) <= 4):
                    violations.add(f'{len(group.members)}:{len(blue_ids)}')
        if violations:
            excluded[state_id] = sorted(violations)
        else:
            eligible_states.append(state_id)
    selected = [r for r in rows if r['state_id'] in set(eligible_states)]
    class ProxyProbability:
        kind = 'global'
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.coverage = wrapped.coverage
        def __call__(self, state, pool):
            if self.wrapped.aggregation != 'product':
                raise ValueError('probability audit uses the declared product proxy')
            return [math.exp(max(-700., float(v))) for v in self.wrapped(state, pool)]
    report = {'schema': 'rule-s2-global-cross-evaluator-v1',
              'dataset_hash': manifest['dataset_hash'], 'test_rows': len(rows),
              'test_states': len(pools), 'three_way_eligible_states': len(eligible_states),
              'three_way_eligible_rows': len(selected), 'excluded_old_rosters_by_state': excluded,
              'label_semantics': 'same proposal-one-event then frozen-rule continuation to native global terminal',
              'proxy_warning': 'local probability products assume a decomposition absent from shared physics; uncalibrated global proxy, tested rather than assumed',
              'all_states': {'new_local_product_proxy': evaluate_evaluator(ProxyProbability(local_scorer), rows),
                             'new_global': evaluate_evaluator(global_scorer, rows)},
              'common_old_supported_states': {'new_local_product_proxy': evaluate_evaluator(ProxyProbability(local_scorer), selected),
                                             'new_global': evaluate_evaluator(global_scorer, selected)}}
    root = Path(__file__).resolve().parents[3]
    checkpoint = Path(checkpoint or root/'assets/frozen/dlom.pt')
    if not checkpoint.exists():
        report['historical_status'] = 'checkpoint_unavailable'
        return report
    report['historical_checkpoint_sha256'] = sha256(checkpoint)
    if not selected:
        report['historical_status'] = 'no_complete_supported_candidate_pool'
        return report
    frozen = FrozenDLOM(checkpoint)
    class Historical(OutcomeScorer):
        def __init__(self):
            super().__init__(kind='local')
        def predict_local(self, state, group, blue_ids):
            if not blue_ids:
                return 1.
            self.coverage['local_queries'] += 1
            return frozen.predict(state, [(group.target, group.members, tuple(blue_ids))], 'rush')[0]
    report['common_old_supported_states']['old_local_product_proxy'] = evaluate_evaluator(ProxyProbability(Historical()), selected)
    report['historical_status'] = 'complete'
    return report
