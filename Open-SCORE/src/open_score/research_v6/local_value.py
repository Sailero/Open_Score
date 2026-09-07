"""Shared single-target counterfactual data, frozen S2 and Blotto policies.

The simulator is used only by offline label collection. Deployed G1/G2 and
R4's scorer use the frozen network and public observations exclusively.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import heapq
import itertools
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from open_score.grouping.domain import Group, Grouping
from open_score.research_v4.actions import count_allocations, neighbors
from open_score.research_v5.protocol import stable_seed
from open_score.research_v5.storage import Store
from .environment import (V6Env, public_projection, local_env_from_state, native_terminal,
                          decode_counts, grand_grouping, rule_grouping, responsibilities,
                          threat_targets)


def _grand(assignment):
    grouped = {}
    for identity, target in sorted(assignment.items()):
        if target is not None:
            grouped.setdefault(target, []).append(identity)
    return Grouping(tuple(Group(target, tuple(ids)) for target, ids in grouped.items()),
                    tuple(i for i, target in assignment.items() if target is None))


def _spatial(state, grouping):
    positions = {e.id: np.asarray(e.position) for e in state.alive('red')}
    result = []
    for group in grouping.groups:
        ids = list(group.members)
        if len(ids) > 1:
            axis = int(np.argmax(np.ptp([positions[i] for i in ids], axis=0)))
            ids.sort(key=lambda i: (positions[i][axis], i))
            middle = len(ids)//2
            result.extend((Group(group.target, tuple(ids[:middle])),
                           Group(group.target, tuple(ids[middle:]))))
        else:
            result.append(group)
    return Grouping(tuple(result), grouping.reserve)


def _repair_partition(old, assignment):
    groups, used = [], set()
    for group in old.groups:
        keep = tuple(i for i in group.members if assignment.get(i) == group.target)
        if keep:
            groups.append(Group(group.target, keep))
            used.update(keep)
    groups.extend(Group(target, (i,)) for i, target in sorted(assignment.items())
                  if target is not None and i not in used)
    return Grouping(tuple(groups), tuple(i for i, t in assignment.items() if t is None))


def partition_neighbors(state, grouping):
    """Fixed-z edits in a reproducible order, including genuine merge/split."""
    groups = grouping.groups
    for target in sorted(state.ids('targets')):
        indices = [j for j, group in enumerate(groups) if group.target == target]
        for first, second in itertools.combinations(indices, 2):
            combined = Group(target, groups[first].members + groups[second].members)
            yield Grouping(tuple(g for j, g in enumerate(groups)
                                 if j not in (first, second)) + (combined,), grouping.reserve)
        for index in indices:
            if len(groups[index].members) > 1:
                split = _spatial(state, Grouping((groups[index],)))
                yield Grouping(tuple(g for j, g in enumerate(groups) if j != index)
                               + split.groups, grouping.reserve)
    yield from neighbors(state, grouping, mode='group_only')


def partition_candidates(state, assignment, limit, rng):
    """Data candidates vary P while holding target, roster and physics fixed."""
    grand = _grand(assignment)
    singleton = Grouping(tuple(Group(t, (i,)) for i, t in sorted(assignment.items())
                               if t is not None), grand.reserve)
    result, seen = [], set()
    def add(action):
        if action not in seen and len(result) < int(limit):
            action.validate(state.ids('red'), state.ids('targets'), max_members=None)
            result.append(action)
            seen.add(action)
    for action in (grand, singleton, _spatial(state, grand)):
        add(action)
    for _ in range(max(8, int(limit)*2)):
        groups = []
        for group in grand.groups:
            ordered = list(map(int, rng.permutation(group.members)))
            pieces = []
            for identity in ordered:
                destination = int(rng.integers(len(pieces)+1))
                if destination == len(pieces):
                    pieces.append([])
                pieces[destination].append(identity)
            groups.extend(Group(group.target, tuple(piece)) for piece in pieces)
        add(Grouping(tuple(groups), grand.reserve))
        if len(result) >= limit:
            break
    if len(result) < limit:
        for neighbor in partition_neighbors(state, grand):
            add(neighbor)
            if len(result) >= limit:
                break
    return result


def _random_grouping(state, rng):
    targets = list(state.ids('targets')) + [None]
    assignment = {i: targets[int(rng.integers(len(targets)))] for i in state.ids('red')}
    options = partition_candidates(state, assignment, 8, rng)
    return options[int(rng.integers(len(options)))]


def _public_threat(state):
    targets = state.alive('targets')
    if not targets:
        return 0.
    return sum(1./(1.+min(np.linalg.norm(np.asarray(b.position)-t.position)
                         for t in targets)) for b in state.alive('blue'))


def _mother_states(env, controller, rng, max_states):
    state, states = env.reset(), []
    while not env.done:
        states.append(state)
        action = (_random_grouping(state, rng) if controller == 'legal_random' else
                  grand_grouping(state) if controller == 'grand' else rule_grouping(state))
        state, _, _, _ = env.step(action)
    if not states:
        return [], int(state.step)
    choices = [0]
    if len(states) > 1:
        choices.extend((int(rng.integers(1, len(states))),
                        max(range(1, len(states)), key=lambda i: (_public_threat(states[i]), -i))))
    return [states[i] for i in dict.fromkeys(choices)][:max_states], int(state.step)


def _allocation_projection(state, cfg, rng):
    targets = list(state.ids('targets'))
    threatened = sorted(set(threat_targets(state).values()))
    choices = (threatened if threatened and rng.random() < cfg['threatened_target_probability']
               else targets)
    target = int(rng.choice(choices))
    n = len(state.ids('red'))
    anchors = sorted(set([0, min(1, n), n] + [int(np.rint(f*n)) for f in (.25, .5, .75)]))
    amount = (int(rng.choice(anchors)) if rng.random() < cfg['allocation']['anchor_probability']
              else int(rng.integers(n+1)))
    # Remaining K buckets are the K-1 other targets and reserve.
    tails = list(count_allocations(n-amount, max(1, len(targets)-1), reserve=True))
    tail = tails[int(rng.integers(len(tails)))]
    counts, cursor = [], 0
    for t in targets:
        if t == target:
            counts.append(amount)
        else:
            counts.append(tail[cursor])
            cursor += 1
    counts.append(tail[-1])
    if rng.random() < cfg['allocation']['distance_identity_probability']:
        grouping = decode_counts(state, counts)
    else:
        ids = list(map(int, rng.permutation(state.ids('red'))))
        mapping, start = {}, 0
        for t, count in zip(targets+[None], counts):
            mapping.update({i: t for i in ids[start:start+count]})
            start += count
        grouping = _grand(mapping)
    selected = tuple(i for i, t in grouping.assignment().items() if t == target)
    return public_projection(state, target, selected, grouping)


def _local_outcomes(local_state, candidates, branch_seeds):
    outcomes, physical_steps = [], 0
    for action in candidates:
        wins = []
        for seed in branch_seeds:
            env = local_env_from_state(replace(local_state, previous=action), int(seed))
            try:
                start = env.state().step
                if not env.done:
                    env.step(action)
                while not env.done:
                    env.step(grand_grouping(env.state()))
                physical_steps += env.state().step-start
                wins.append(int(env.native_success))
            finally:
                env.close()
        outcomes.append(wins)
    return outcomes, physical_steps


def collect_data(config, run_dir):
    """Resume by completed mother family; a single authoritative SQLite copy."""
    settings, data = config['environment'], config['s2']['data']
    specs = []
    single = data['single_target']
    split_sequence = [split for split, n in single['split'].items() for _ in range(n)]
    for pair_index, (red, blue) in enumerate(single['count_pairs']):
        for index, split in enumerate(split_sequence):
            specs.append((f'single/{pair_index}/{index}', red, blue, 1, split,
                          data['controllers'][(pair_index+index) % len(data['controllers'])], 'single'))
    multi = data['multi_target']
    for scenario in config['scenarios']:
        for controller in multi['controllers']:
            index = 0
            for split, count in controller['split'].items():
                for _ in range(count):
                    specs.append((f'multi/{scenario["id"]}/{controller["name"]}/{index}',
                                  scenario['red'], scenario['blue'], scenario['targets'],
                                  split, controller['name'], scenario['id']))
                    index += 1
    started = time.monotonic()
    with Store(Path(run_dir)/'shared') as store:
        completed = {x['family_id'] for x in store.rows('s2/families')}
        initial_completed = len(completed)
        for family, red, blue, targets, split, controller, scenario in specs:
            if family in completed:
                continue
            family_seed = stable_seed(config['version'], 's2', family, split)
            rng = np.random.default_rng(stable_seed(family_seed, 'policy'))
            env = V6Env(red, blue, targets, seed=stable_seed(family_seed, 'opening'),
                        opponent_seed=stable_seed(family_seed, 'opponent'),
                        max_steps=settings['max_physical_steps'],
                        command_interval=settings['command_interval'],
                        reward_coefficient=config['reward']['shaping']['coefficient'])
            try:
                states, mother_steps = _mother_states(env, controller, rng, data['states_per_family'])
            finally:
                env.close()
            blocks, simulated = [], 0
            for state_index, state in enumerate(states):
                if targets == 1:
                    base = Grouping((Group(state.targets[0].id, tuple(state.ids('red'))),)) if state.ids('red') else Grouping(())
                    local = public_projection(state, state.targets[0].id, state.ids('red'), base)
                else:
                    local = _allocation_projection(state, multi, rng)
                assignment = {i: local.targets[0].id for i in local.ids('red')}
                candidates = partition_candidates(local, assignment, data['candidates_per_state'], rng)
                branch_seeds = [stable_seed(family_seed, state_index, 'branch', j)
                                for j in range(data['branches'][split])]
                outcomes, steps = _local_outcomes(local, candidates, branch_seeds)
                simulated += steps
                blocks.append(dict(state=local, candidates=candidates, outcomes=outcomes,
                                   family_id=family, state_index=state_index,
                                   branch_seeds=branch_seeds, scenario=scenario,
                                   source_state=state, controller=controller))
            with store.transaction():
                store.save_torch(f's2/{split}/{family}', blocks)
                store.append('s2/families', dict(family_id=family, split=split, states=len(blocks),
                             mother_physical_steps=mother_steps, simulation_physical_steps=simulated,
                             controller=controller, scenario=scenario), source=f's2/{family}', source_line=0)
            completed.add(family)
            elapsed = time.monotonic()-started
            rate = (len(completed)-initial_completed)/max(elapsed, 1e-9)
            store.put('progress/S2_data', dict(phase='s2_data', completed=len(completed), total=len(specs),
                      families_per_second=rate, remaining_seconds=(len(specs)-len(completed))/max(rate, 1e-9),
                      last_family_simulation_steps=simulated))
            print(f'[S2 data] {len(completed)}/{len(specs)} families; {family}; {simulated} branch steps', flush=True)
        rows = store.rows('s2/families')
        result = dict(complete=True, families=len(rows), states=sum(x['states'] for x in rows),
                      mother_physical_steps=sum(x['mother_physical_steps'] for x in rows),
                      simulation_physical_steps=sum(x['simulation_physical_steps'] for x in rows),
                      wall_time_s=time.monotonic()-started)
        store.put('s2_data_result', result)
    return result


class BlottoPolicy:
    def __init__(self, method, config, scorer):
        self.method, self.config, self.scorer = method, config, scorer

    def act(self, state, physical_steps=0, explore=False, rng=None):
        started = time.monotonic()
        targets = tuple(state.ids('targets'))
        if not targets:
            return Grouping((), state.ids('red')), dict(candidates_scored=0)
        rule = rule_grouping(state)
        count_map = Counter(t for t in rule.assignment().values())
        rule_counts = tuple(count_map[t] for t in targets)+(count_map[None],)
        vectors = list(count_allocations(len(state.ids('red')), len(targets)))
        if len(vectors) > self.config['blotto']['count_enumeration']['max_configurations']:
            raise ValueError('Complete count space exceeds configured enumeration budget')
        vectors = [rule_counts]+[v for v in vectors if v != rule_counts]
        actions = [decode_counts(state, counts) for counts in vectors]
        rows_before = self.scorer.rows_scored
        values = self.scorer.log_probabilities(state, actions).sum(axis=1)
        if np.isnan(values).any():
            raise ValueError('S2 returned a NaN count score')
        index = int(np.argmax(values))
        chosen, value = actions[index], float(values[index])
        metrics = dict(count_configurations=len(actions), count_score=value)
        if self.method == 'BLOTTO_Group':
            chosen, search_metrics = self._group_search(state, chosen, value)
            metrics.update(search_metrics)
        # Grand's already computed count score is reused; it occupies the
        # partition budget but is not an additional scoring call.
        metrics.update(candidates_scored=len(actions)+max(0, metrics.get('partition_evaluations', 0)-1),
                       s2_scoring_rows=self.scorer.rows_scored-rows_before,
                       decision_latency_s=time.monotonic()-started)
        return chosen, metrics

    def _group_search(self, state, grand, grand_score):
        budget = int(self.config['blotto']['group_search']['evaluations'])
        if budget < 1:
            raise ValueError('Partition budget must include grand start')
        assignment = grand.assignment()
        roots = [grand, _repair_partition(state.previous, assignment),
                 Grouping(tuple(Group(t, (i,)) for i, t in sorted(assignment.items()) if t is not None), grand.reserve),
                 _spatial(state, grand)]
        seen, evaluated, frontier = {grand}, [(grand, grand_score)], [(-grand_score, 0, grand)]
        generated = 0
        def evaluate(proposals):
            nonlocal generated
            fresh = []
            for action in proposals:
                generated += 1
                if action in seen:
                    continue
                if len(evaluated)+len(fresh) >= budget:
                    break
                if action.assignment() != assignment:
                    raise ValueError('Fixed-assignment search changed a target or reserve')
                seen.add(action)
                fresh.append(action)
            if fresh:
                scores = self.scorer.log_probabilities(state, fresh).sum(axis=1)
                for action, score in zip(fresh, scores):
                    if np.isnan(score):
                        raise ValueError('S2 returned a NaN partition score')
                    order = len(evaluated)
                    evaluated.append((action, float(score)))
                    heapq.heappush(frontier, (-float(score), order, action))
        evaluate(roots[1:])
        while frontier and len(evaluated) < budget:
            _, _, source = heapq.heappop(frontier)
            evaluate(partition_neighbors(state, source))
        best = max(range(len(evaluated)), key=lambda i: (evaluated[i][1], -i))
        return evaluated[best][0], dict(partition_evaluations=len(evaluated),
              partition_score=evaluated[best][1], partition_proposals_generated=generated,
              grand_start_score_reused=True)


def make_policy(method, config, scorer):
    if method not in ('BLOTTO_Count', 'BLOTTO_Group'):
        raise ValueError(f'Unsupported Blotto method {method}')
    return BlottoPolicy(method, config, scorer)


class GroupLocalS2(nn.Module):
    """One target's entire partition, batched across physical local inputs."""
    def __init__(self, config):
        super().__init__()
        from .learning import EntityEncoder, _transformer
        model = config['model']
        d = int(model['hidden'])
        self.encoder = EntityEncoder(model)
        self.group_fusion = nn.Sequential(nn.Linear(3*d+5, d), nn.GELU())
        self.kind = nn.Embedding(3, d)
        self.relations = _transformer(model, model['group_layers'])
        self.output = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, states, candidates):
        if len(states) != len(candidates):
            raise ValueError('S2 requires one complete local partition per local state')
        if not states:
            return self.kind.weight.new_empty((0,))
        encoded, contact_rows = [], []
        for state, action in zip(states, candidates):
            # The executor is stateless: old group features are normalized to
            # grand. Only the explicit candidate determines local execution.
            target_id = state.targets[0].id
            base = replace(state, previous=_grand({i: target_id for i in state.ids('red')}))
            encoded.append(self.encoder.features(base))
            contact_rows.append(dict(responsibilities(base, action)) if not native_terminal(base)[0] else {})
        batch = len(states)
        max_entities = max(len(row[0]) for row in encoded)
        max_groups = max(1, max(len(action.groups) for action in candidates))
        raw = np.zeros((batch, max_entities, self.encoder.feature_size), np.float32)
        entity_mask = np.ones((batch, max_entities), bool)
        red_pool = np.zeros((batch, max_groups, max_entities), np.float32)
        blue_pool = np.zeros_like(red_pool)
        statistics = np.zeros((batch, max_groups, 5), np.float32)
        group_mask = np.ones((batch, max_groups), bool)
        target_indices = []
        for b, ((rows, reds, blues, targets), action, contacts) in enumerate(zip(encoded, candidates, contact_rows)):
            raw[b, :len(rows)], entity_mask[b, :len(rows)] = rows, False
            rmap = {e.id: 1+j for j, e in enumerate(reds)}
            bmap = {e.id: 1+len(reds)+j for j, e in enumerate(blues)}
            rpos = {e.id: np.asarray(e.position) for e in reds}
            target_indices.append(1+len(reds)+len(blues) if targets else 0)
            for g, group in enumerate(action.groups):
                red = [rmap[i] for i in group.members if i in rmap]
                blue = [bmap[i] for i in contacts.get(group, ()) if i in bmap]
                if red:
                    red_pool[b, g, red] = 1./len(red)
                if blue:
                    blue_pool[b, g, blue] = 1./len(blue)
                center = np.mean([rpos[i] for i in group.members if i in rpos], 0) if red else np.zeros(3)
                statistics[b, g] = [len(red)/20., len(blue)/20., *(center/5000.)]
                group_mask[b, g] = False
        device = self.kind.weight.device
        h = self.encoder.attention(F.gelu(self.encoder.input(torch.as_tensor(raw, device=device))),
                                   src_key_padding_mask=torch.as_tensor(entity_mask, device=device))
        target = h[torch.arange(batch, device=device), torch.tensor(target_indices, device=device)]
        red = torch.bmm(torch.as_tensor(red_pool, device=device), h)
        blue = torch.bmm(torch.as_tensor(blue_pool, device=device), h)
        groups = self.group_fusion(torch.cat([red, blue, target[:, None].expand(-1, max_groups, -1),
                                             torch.as_tensor(statistics, device=device)], dim=-1))
        tokens = torch.cat([(h[:, 0]+self.kind.weight[0])[:, None],
                            (target+self.kind.weight[1])[:, None],
                            groups+self.kind.weight[2]], dim=1)
        mask = torch.as_tensor(np.concatenate([np.zeros((batch, 2), bool), group_mask], axis=1), device=device)
        result = self.relations(tokens, src_key_padding_mask=mask)
        result = (result*~mask[:, :, None]).sum(1)/(~mask).sum(1)[:, None]
        return self.output(result).squeeze(-1)


def _local_key(state, action):
    def entities(rows):
        return tuple((e.id, e.position, e.velocity, e.health) for e in rows)
    return (state.step, state.max_steps, state.command_interval, state.opponent,
            state.initial_red_count, state.initial_blue_count, state.initial_blue_health,
            state.initial_target_count, entities(state.red), entities(state.blue),
            entities(state.targets), action)


class FrozenLocalScorer:
    def __init__(self, config, payload, device='cpu'):
        self.config, self.device = config, torch.device(device)
        self.model = GroupLocalS2(config).to(self.device)
        self.model.load_state_dict(payload['model'])
        self.model.eval().requires_grad_(False)
        self.seed = int(payload['seed'])
        self.rows_scored = 0
        self.last_score_rows = 0
        self.last_requested_rows = 0

    def export(self):
        return dict(seed=self.seed, model={k: v.detach().cpu() for k, v in self.model.state_dict().items()})

    def log_probabilities(self, state, plans):
        targets = tuple(sorted(state.alive('targets'), key=lambda t: t.id))
        initial_rows = self.rows_scored
        self.last_requested_rows = len(plans)*len(targets)
        result = np.empty((len(plans), len(targets)), np.float64)
        pending, keys, destinations = [], {}, []
        for i, action in enumerate(plans):
            for j, target in enumerate(targets):
                ids = tuple(k for k, t in action.assignment().items() if t == target.id)
                local = public_projection(state, target.id, ids, action)
                terminal, success, _ = native_terminal(local)
                if terminal:
                    result[i, j] = 0. if success else -np.inf
                    continue
                key = _local_key(local, local.previous)
                if key not in keys:
                    keys[key] = len(pending)
                    pending.append((local, local.previous))
                destinations.append((i, j, keys[key]))
        values = []
        size = int(self.config['resources']['s2_inference_row_microbatch'])
        with torch.inference_mode():
            for start in range(0, len(pending), size):
                rows = pending[start:start+size]
                logits = self.model([x[0] for x in rows], [x[1] for x in rows])
                values.extend(F.logsigmoid(logits).cpu().double().tolist())
                self.rows_scored += len(rows)
        for i, j, index in destinations:
            result[i, j] = values[index]
        self.last_score_rows = self.rows_scored-initial_rows
        return result

    def probabilities(self, state, plans):
        return np.exp(self.log_probabilities(state, plans))


def _checkpoint_path(run_dir, seed, kind):
    return Path(run_dir)/'shared'/'models'/f's2_{int(seed)}_{kind}.pt'


def _save_model(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix('.pending')
    torch.save(payload, pending)
    pending.replace(path)


def load_scorer(config, run_dir, seed, device='cpu'):
    path = _checkpoint_path(run_dir, seed, 'best')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    return FrozenLocalScorer(config, payload, device)


def _dataset(store, split):
    return [block for key in store.blob_keys(f's2/{split}/') for block in store.load_torch(key)]


def _metrics(model, blocks, size):
    """Family-weighted probability metrics and within-state ranking/regret."""
    family_counts = Counter(block['family_id'] for block in blocks)
    if not blocks:
        return dict(states=0, brier=None, ece=None)
    results, calibration = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(blocks), size):
            chunk = blocks[start:start+size]
            states = [b['state'] for b in chunk for _ in b['candidates']]
            candidates = [p for b in chunk for p in b['candidates']]
            probabilities = model(states, candidates).sigmoid().cpu().numpy()
            offset = 0
            for block in chunk:
                n = len(block['candidates'])
                predicted = probabilities[offset:offset+n]
                offset += n
                terminal, success, _ = native_terminal(block['state'])
                if terminal:
                    predicted = np.full(n, float(success))
                truth = np.mean(block['outcomes'], axis=1)
                weight = 1./(len(family_counts)*family_counts[block['family_id']])
                pairs = list(itertools.combinations(range(n), 2))
                non_ties = [(i, j) for i, j in pairs if truth[i] != truth[j]]
                accuracy = np.mean([float((predicted[i]-predicted[j])*(truth[i]-truth[j]) > 0)
                                    + .5*float(predicted[i] == predicted[j]) for i, j in non_ties]) if non_ties else None
                results.append(dict(weight=weight, brier=float(np.mean((predicted-truth)**2)),
                    ranking=accuracy, tie_fraction=1.-len(non_ties)/max(1, len(pairs)),
                    selection_loss=float(truth.max()-truth[int(np.argmax(predicted))]),
                    all_zero=bool(np.all(truth == 0)), non_tie=bool(non_ties),
                    scale=f'{len(block["state"].ids("red"))}v{len(block["state"].ids("blue"))}'))
                calibration.extend((float(p), float(y), weight/n) for p, y in zip(predicted, truth))
    def aggregate(rows):
        mass = sum(row['weight'] for row in rows)
        ranks = [row for row in rows if row['ranking'] is not None]
        return dict(states=len(rows), brier=sum(x['weight']*x['brier'] for x in rows)/mass,
                    nontie_ranking_accuracy=(sum(x['weight']*x['ranking'] for x in ranks)/sum(x['weight'] for x in ranks) if ranks else None),
                    tie_fraction=sum(x['weight']*x['tie_fraction'] for x in rows)/mass,
                    selection_loss=sum(x['weight']*x['selection_loss'] for x in rows)/mass,
                    all_zero_states=sum(x['all_zero'] for x in rows), non_tie_states=sum(x['non_tie'] for x in rows))
    result = aggregate(results)
    ece = 0.
    for index in range(10):
        rows = [(p, y, w) for p, y, w in calibration if min(int(p*10), 9) == index]
        if rows:
            ece += abs(sum(w*(p-y) for p, y, w in rows))
    result['ece'] = ece
    result['by_scale'] = {scale: aggregate([r for r in results if r['scale'] == scale])
                          for scale in sorted({r['scale'] for r in results})}
    return result


def fit(config, run_dir, seed, device):
    """Forty fixed epochs, selecting Brier on held-out mother families."""
    started = time.monotonic()
    settings = config['s2']
    seed = int(seed)
    with Store(Path(run_dir)/'shared') as store:
        prior = store.get(f's2_result/{seed}')
        if prior and prior.get('complete'):
            return prior
        train, validation, test = (_dataset(store, split) for split in ('train', 'validation', 'test'))
    if not train or not validation or not test:
        raise ValueError('Collect all shared S2 family splits before fitting')
    torch.manual_seed(seed)
    model = GroupLocalS2(config).to(device)
    opt = settings['optimizer']
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt['learning_rate'],
                                 weight_decay=opt['weight_decay'], eps=opt['epsilon'], betas=tuple(opt['betas']))
    rng = np.random.default_rng(stable_seed(seed, 's2_fit'))
    start_epoch, best, best_epoch = 0, math.inf, 0
    resume = _checkpoint_path(run_dir, seed, 'resume')
    if resume.exists():
        payload = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(payload['model'])
        optimizer.load_state_dict(payload['optimizer'])
        rng.bit_generator.state = payload['numpy_rng']
        start_epoch, best, best_epoch = payload['epoch'], payload['best'], payload['best_epoch']
    counts = Counter(block['family_id'] for block in train)
    batch = int(settings['batch_state_blocks'])
    for epoch in range(start_epoch+1, int(settings['epochs'])+1):
        model.train()
        order = rng.permutation(len(train))
        total_loss, total_blocks = 0., 0
        for start in range(0, len(order), batch):
            blocks = [train[i] for i in order[start:start+batch]]
            states = [b['state'] for b in blocks for _ in b['candidates']]
            actions = [p for b in blocks for p in b['candidates']]
            targets = np.concatenate([np.mean(b['outcomes'], axis=1) for b in blocks])
            logits = model(states, actions)
            losses = F.binary_cross_entropy_with_logits(logits, logits.new_tensor(targets), reduction='none')
            terms, offset = [], 0
            for block in blocks:
                n = len(block['candidates'])
                # Shuffling state blocks with this weight gives equal family,
                # then equal state, then equal candidate contribution.
                scale = len(train)/(len(counts)*counts[block['family_id']])
                terms.append(losses[offset:offset+n].mean()*scale)
                offset += n
            loss = torch.stack(terms).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), opt['gradient_clip'])
            optimizer.step()
            total_loss += float(loss.detach())*len(blocks)
            total_blocks += len(blocks)
        held = _metrics(model, validation, batch)
        payload = dict(seed=seed, epoch=epoch, model=model.state_dict(), config=config['model'])
        if held['brier'] < best:
            best, best_epoch = held['brier'], epoch
            _save_model(_checkpoint_path(run_dir, seed, 'best'), payload)
        if epoch == settings['epochs']:
            _save_model(_checkpoint_path(run_dir, seed, 'final'), payload)
        _save_model(resume, dict(**payload, optimizer=optimizer.state_dict(), numpy_rng=rng.bit_generator.state,
                                 best=best, best_epoch=best_epoch))
        with Store(Path(run_dir)/'shared') as store:
            store.append('s2_fit', dict(seed=seed, epoch=epoch, train_loss=total_loss/max(total_blocks, 1),
                         validation_brier=held['brier'], validation_nontie_ranking_accuracy=held['nontie_ranking_accuracy'],
                         validation_selection_loss=held['selection_loss']), source=f's2_fit/{seed}', source_line=epoch)
            store.put(f'progress/{seed}', dict(phase='s2_fit', completed=epoch, total=settings['epochs'],
                      validation_brier=held['brier'],
                      remaining_seconds=(time.monotonic()-started)/max(epoch-start_epoch, 1)*(settings['epochs']-epoch)))
        print(f'[S2 {seed}] epoch {epoch}/{settings["epochs"]}; validation Brier={held["brier"]:.5f}', flush=True)
    payload = torch.load(_checkpoint_path(run_dir, seed, 'best'), map_location=device, weights_only=False)
    model.load_state_dict(payload['model'])
    metrics = _metrics(model, test, batch)
    result = dict(complete=True, seed=seed, best_epoch=best_epoch, best_validation_brier=best,
                  **{f'test_{k}': v for k, v in metrics.items()}, wall_time_s=time.monotonic()-started)
    with Store(Path(run_dir)/'shared') as store:
        store.put(f's2_result/{seed}', result)
    if resume.exists():
        resume.unlink()
    return result
