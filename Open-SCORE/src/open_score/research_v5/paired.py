"""State-balanced, paired terminal supervision used only by v5 T6.

The BCE and advantage models have identical encoders and initial parameters.
Differences are their supervised target and their output link; no outcome ties
or unsuccessful states are discarded. Search counts its rule anchor query.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
import json
import time

import numpy as np
import torch
from torch.nn import functional as F

from open_score.grouping.domain import DecisionState, Grouping
from open_score.grouping.storage import (append_jsonl, atomic_checkpoint, atomic_json,
    fingerprint, random_state, restore_random_state, seed_everything)
from open_score.research_v4.actions import partition_key, rule_grouping, search
from open_score.research_v4.outcomes import GlobalOutcomeNetwork


def group_rows(rows):
    """Validate CRN data and preserve each physical state's entire candidate set."""
    groups = {}
    for row in rows:
        state = DecisionState.from_dict(row['state'])
        family = str(row.get('family_id', row.get('family_seed', row.get('family', ''))))
        key = (family, fingerprint(state.to_dict()))
        groups.setdefault(key, []).append(row)
    result = []
    for (family, state_hash), items in groups.items():
        state = DecisionState.from_dict(items[0]['state'])
        actions = [Grouping.from_dict(row['action']) for row in items]
        if len({partition_key(a) for a in actions}) != len(actions):
            raise ValueError('duplicate candidate in a state group')
        rule_key = partition_key(rule_grouping(state))
        try:
            anchor = [partition_key(a) for a in actions].index(rule_key)
        except ValueError as error:
            raise ValueError('every paired state must include the common rule') from error
        seeds = items[0]['branch_seeds']
        if not seeds or len(seeds) != len(set(seeds)):
            raise ValueError('unique, nonempty independent branch seeds required')
        if any(row['branch_seeds'] != seeds for row in items):
            raise ValueError('candidate branches must share paired seeds')
        z = np.asarray([row['outcomes'] for row in items], dtype=np.float32)
        if z.shape != (len(items), len(seeds)) or not np.isin(z, [0., 1.]).all():
            raise ValueError('terminal branches must be Bernoulli outcomes')
        y = z.mean(1)
        if any(abs(float(row['y']) - float(target)) > 1e-7 for row, target in zip(items, y)):
            raise ValueError('absolute labels disagree with recorded terminal branches')
        d = z-z[anchor]
        result.append({'family_id': family, 'state_id': state_hash, 'state': state,
                       'actions': actions, 'anchor': anchor, 'y': y,
                       'advantage': d.mean(1), 'branch_differences': d,
                       'branch_seeds': seeds, 'episode_spec': items[0].get('episode_spec', {}),
                       'source_state_id': items[0].get('state_id', state_hash)})
    return result


def label_statistics(groups):
    branches = np.concatenate([g['branch_differences'].ravel() for g in groups]) if groups else np.array([])
    return {'families': len({g['family_id'] for g in groups}), 'states': len(groups),
            'candidates': sum(len(g['actions']) for g in groups),
            'all_zero_states': sum(bool(np.all(g['y'] == 0)) for g in groups),
            'all_tie_states': sum(bool(np.ptp(g['y']) == 0) for g in groups),
            'non_tie_states': sum(bool(np.ptp(g['y']) > 0) for g in groups),
            'positive_difference_states': sum(bool(np.any(g['advantage'] > 0)) for g in groups),
            'negative_difference_states': sum(bool(np.any(g['advantage'] < 0)) for g in groups),
            'zero_difference_states': sum(bool(np.all(g['advantage'] == 0)) for g in groups),
            'positive_difference_branches': int((branches > 0).sum()),
            'zero_difference_branches': int((branches == 0).sum()),
            'negative_difference_branches': int((branches < 0).sum())}


def state_balanced_loss(model, groups, kind):
    if kind not in ('bce', 'adv') or not groups:
        raise ValueError('kind must be bce/adv and batch must contain states')
    scores, _ = model([g['state'] for g in groups], [g['actions'] for g in groups])
    losses = []
    for i, group in enumerate(groups):
        prediction = scores[i, :len(group['actions'])]
        target = prediction.new_tensor(group['y'] if kind == 'bce' else group['advantage'])
        if kind == 'bce':
            losses.append(F.binary_cross_entropy_with_logits(prediction, target))
        else:
            mask = torch.ones_like(prediction)
            mask[group['anchor']] = 0.
            losses.append(F.mse_loss(torch.tanh(prediction)*mask, target))
    return torch.stack(losses).mean()


@torch.no_grad()
def audit_model(model, groups, kind):
    model.eval()
    state_brier, errors, predictions, targets, regrets = [], [], [], [], []
    correct = pairs = 0
    per_cell = defaultdict(list)
    for group in groups:
        scores, _ = model([group['state']], [group['actions']])
        raw = scores[0, :len(group['actions'])]
        probability = torch.sigmoid(raw).cpu().numpy() if kind == 'bce' else None
        score = (probability-probability[group['anchor']]) if kind == 'bce' else torch.tanh(raw).cpu().numpy()
        score[group['anchor']] = 0.
        y = group['y']
        if probability is not None:
            state_brier.append(float(np.mean((probability-y)**2)))
            predictions.extend(map(float, probability)); targets.extend(map(float, y))
        errors.append(float(np.mean((score-group['advantage'])**2)))
        selected = int(np.argmax(score)) if float(np.max(score)) > 0 else group['anchor']
        regrets.append(float(np.max(y)-y[selected]))
        state_correct = state_pairs = 0
        for first in range(len(y)):
            for second in range(first):
                if y[first] != y[second]:
                    pairs += 1
                    difference = score[first]-score[second]
                    contribution = .5 if difference == 0 else float(difference*(y[first]-y[second]) > 0)
                    correct += contribution; state_correct += contribution; state_pairs += 1
        spec = group.get('episode_spec', {})
        cell = f"{spec.get('red_count', len(group['state'].red))}v{spec.get('blue_count', len(group['state'].blue))}"
        per_cell[cell].append({'advantage_mse': errors[-1], 'empirical_candidate_regret': regrets[-1],
            'correct': state_correct, 'pairs': state_pairs,
            'brier': state_brier[-1] if probability is not None else None,
            'predictions': probability.tolist() if probability is not None else [],
            'labels': y.tolist(),
            'all_zero': bool(np.all(y == 0)), 'non_tie': bool(np.ptp(y) > 0)})
    result = {'kind': kind, **label_statistics(groups),
              'advantage_mse': float(np.mean(errors)) if errors else None,
              'non_tie_pairs': pairs, 'ranking_accuracy': correct/pairs if pairs else None,
              'empirical_candidate_regret': float(np.mean(regrets)) if regrets else None,
              'brier': float(np.mean(state_brier)) if state_brier else None,
              'calibration_bins': [], 'ece': None, 'by_cell': {},
              'weighting': 'state_equal_losses_and_Brier; candidate_weighted_ECE; non_tie_pair_weighted_ranking'}
    for cell, items in per_cell.items():
        cell_pairs = sum(r['pairs'] for r in items)
        result['by_cell'][cell] = {'states': len(items),
            'non_tie_states': sum(r['non_tie'] for r in items),
            'all_zero_states': sum(r['all_zero'] for r in items),
            'non_tie_pairs': cell_pairs,
            'ranking_accuracy': sum(r['correct'] for r in items)/cell_pairs if cell_pairs else None,
            'advantage_mse': float(np.mean([r['advantage_mse'] for r in items])),
            'empirical_candidate_regret': float(np.mean([r['empirical_candidate_regret'] for r in items])),
            'brier': float(np.mean([r['brier'] for r in items])) if kind == 'bce' else None}
        if kind == 'bce':
            cp = np.concatenate([r['predictions'] for r in items])
            cy = np.concatenate([r['labels'] for r in items]); ce = 0.
            for index in range(10):
                mask = (cp >= index/10) & ((cp < (index+1)/10) if index < 9 else cp <= 1.)
                if mask.any():
                    ce += float(mask.mean()*abs(cp[mask].mean()-cy[mask].mean()))
            result['by_cell'][cell]['ece'] = ce
    if predictions:
        p, y = np.asarray(predictions), np.asarray(targets)
        ece = 0.
        for index in range(10):
            mask = (p >= index/10) & ((p < (index+1)/10) if index < 9 else (p <= 1.))
            if mask.any():
                confidence, observed = float(p[mask].mean()), float(y[mask].mean())
                count = int(mask.sum()); ece += count/len(p)*abs(confidence-observed)
                result['calibration_bins'].append({'lower': index/10, 'upper': (index+1)/10,
                    'count': count, 'prediction': confidence, 'observed': observed})
        result['ece'] = ece
    return result


def train_model(train, validation, output, *, kind, seed, epochs=40, batch_states=16,
                device='cpu', learning_rate=3e-4, model_config=None, gpu=None,
                progress=None, metadata=None):
    """Resume only complete epochs, including optimizer, all RNG and counters."""
    if not train:
        raise ValueError('training requires nonempty complete state groups')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    metadata = metadata or {}
    identity = fingerprint({'kind': kind, 'seed': int(seed), 'epochs': int(epochs),
        'batch_states': int(batch_states), 'learning_rate': learning_rate,
        'model_config': model_config or {}, 'metadata': metadata,
        'train': [(g['family_id'], g['state_id'], g['y'].tolist(),
                   [a.to_dict() for a in g['actions']], g['branch_seeds'],
                   g['branch_differences'].tolist()) for g in train],
        'validation': [(g['family_id'], g['state_id'], g['y'].tolist(),
                        g['branch_seeds']) for g in validation]})
    seed_everything(seed)
    model = GlobalOutcomeNetwork(**(model_config or {})).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epoch = updates = 0; elapsed_before = 0.
    checkpoint = output/'latest.pt'
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved['identity'] != identity:
            raise ValueError('paired training protocol/data changed; use a new output directory')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        epoch, updates, elapsed_before = saved['epoch'], saved['updates'], saved['training_seconds']
        restore_random_state(saved['rng'])
    started = time.monotonic()
    for current_epoch in range(epoch+1, int(epochs)+1):
        indices = np.random.permutation(len(train)); weighted_loss = 0.; states_seen = 0
        model.train()
        for start in range(0, len(indices), int(batch_states)):
            batch = [train[int(i)] for i in indices[start:start+int(batch_states)]]
            with gpu() if gpu else nullcontext():
                optimizer.zero_grad(set_to_none=True)
                loss = state_balanced_loss(model, batch, kind)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
                optimizer.step()
            weighted_loss += float(loss.detach())*len(batch); states_seen += len(batch); updates += 1
        with gpu() if gpu else nullcontext():
            metrics = audit_model(model, validation, kind)
        elapsed = elapsed_before+time.monotonic()-started
        row = {'epoch': current_epoch, 'epochs': int(epochs), 'updates': updates,
               'kind': kind, 'loss': weighted_loss/states_seen, 'gradient_norm': float(norm),
               'training_seconds': elapsed, 'validation': metrics}
        atomic_checkpoint(checkpoint, {'schema': 'v5-paired-value-v1', 'kind': kind,
            'model': model.state_dict(), 'model_config': model.config,
            'optimizer': optimizer.state_dict(), 'rng': random_state(), 'epoch': current_epoch,
            'updates': updates, 'training_seconds': elapsed, 'identity': identity,
            'metadata': metadata})
        append_jsonl(output/'training.jsonl', row); atomic_json(output/'progress.json', row)
        if progress:
            progress(row)
    result = {'kind': kind, 'epochs': int(epochs), 'updates': updates,
              'training_seconds': elapsed_before+time.monotonic()-started,
              'checkpoint': str(checkpoint), 'selection': 'final_epoch',
              'training_labels': label_statistics(train)}
    atomic_json(output/'training_result.json', result)
    return model, result


class PairedPolicy:
    """Both supervision arms use the same search and strict rule-relative gate."""
    def __init__(self, model, kind, *, budget=64, threshold=0., device='cpu'):
        if kind not in ('bce', 'adv') or budget < 3 or threshold < 0:
            raise ValueError('paired policy requires bce/adv, budget >=3, threshold >=0')
        self.model = model.to(device).eval(); self.kind = kind
        self.budget, self.threshold = int(budget), float(threshold)
        self.last_trace = {}

    @torch.no_grad()
    def score_candidates(self, state, candidates):
        """Independent held-out diagnostics; never invoked as extra online search work."""
        rule = rule_grouping(state); keys = [partition_key(a) for a in candidates]
        rule_key = partition_key(rule)
        if rule_key not in keys:
            raise ValueError('diagnostic candidate set must contain rule anchor')
        scores, _ = self.model([state], [candidates])
        scores = scores[0, :len(candidates)]
        values = torch.sigmoid(scores) if self.kind == 'bce' else torch.tanh(scores)
        if self.kind == 'bce':
            values = values-values[keys.index(rule_key)]
        result = values.cpu().numpy().copy(); result[keys.index(rule_key)] = 0.
        return result

    @torch.no_grad()
    def predict_candidate_probabilities(self, state, candidates):
        """ADV is a signed difference and deliberately has no probability link."""
        if self.kind != 'bce':
            return None
        scores, _ = self.model([state], [candidates])
        return torch.sigmoid(scores[0, :len(candidates)]).cpu().numpy()

    @torch.no_grad()
    def act(self, state):
        rule = rule_grouping(state); rule_key = partition_key(rule)
        anchor_probability = None; evaluated = []; query_count = 0
        def scorer(view, proposals):
            nonlocal anchor_probability, query_count
            scores, _ = self.model([view], [proposals])
            raw = scores[0, :len(proposals)]
            query_count += len(proposals)
            keys = [partition_key(p) for p in proposals]
            if self.kind == 'bce':
                probabilities = torch.sigmoid(raw).cpu().numpy()
                if rule_key in keys:
                    anchor_probability = float(probabilities[keys.index(rule_key)])
                if anchor_probability is None:
                    raise RuntimeError('the search must score the rule in its initial budgeted roots')
                values = probabilities-anchor_probability
            else:
                values = torch.tanh(raw).cpu().numpy()
            values = np.asarray(values, dtype=float)
            if rule_key in keys:
                values[keys.index(rule_key)] = 0.
            evaluated.extend({'candidate': p.to_dict(), 'advantage': float(value)}
                             for p, value in zip(proposals, values))
            return values
        proposed, trace = search(state, scorer, budget=self.budget, return_trace=True)
        accepted = trace['best_value'] > self.threshold
        selected = proposed if accepted else rule
        self.last_trace = {**trace, 'kind': self.kind, 'threshold': self.threshold,
            'scored_candidates': query_count, 'accepted_replacement': accepted,
            'selected_advantage': trace['best_value'] if accepted else 0.,
            'rule_included': any(partition_key(Grouping.from_dict(r['candidate'])) == rule_key for r in evaluated),
            'candidates': evaluated, 'rule': rule.to_dict()}
        return selected


def load_policy(path, device='cpu', **opts):
    saved = torch.load(Path(path), map_location=device, weights_only=False)
    model = GlobalOutcomeNetwork(**saved['model_config'])
    model.load_state_dict(saved['model'])
    return PairedPolicy(model, saved['kind'], device=device,
                        budget=opts.get('budget', 64), threshold=opts.get('threshold', 0.))


def benchmark_records(ctx, namespace):
    """Three independent calibration-only physical states, including late play."""
    records = []; collection_steps = 0
    indices = [0, len(ctx.config['cells'])//2, len(ctx.config['cells'])-1]
    for index, wanted_step in zip(indices, (5, 12, 20)):
        spec = ctx.spec(index, split='calibration', namespace=namespace)
        env = ctx.make_env(spec)
        try:
            state, snapshot = env.state(), env.snapshot()
            while not env.done and env.state().step < wanted_step:
                state, snapshot = env.state(), env.snapshot()
                _, _, _, info = env.step(rule_grouping(state))
                collection_steps += int(info['delta'])
                if not env.done:
                    state, snapshot = env.state(), env.snapshot()
            records.append({'state': state, 'snapshot': snapshot, 'family_id': spec.family_id,
                'state_id': f'{spec.family_id}:{state.step}', 'episode_spec': spec.to_dict()})
        finally:
            env.close()
    return records, collection_steps
