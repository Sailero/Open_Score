"""Rule-executor counterfactual data with complete terminal labels and family splits.

A label executes the proposal for one command event and then uses one *frozen*
continuation policy. It is Q^mu, not the value of arbitrary future replanning.
Every candidate sees the same independent branch seeds; real future RNG is never
queried. Whole original episodes (and every descendant) belong to one split.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from open_score.grouping.domain import DecisionState, Grouping
from open_score.grouping.storage import (atomic_json, fingerprint, random_state,
                                        restore_random_state, sha256)


SCHEMA = 'rule-s2-counterfactual-v1'
CONTINUATION_VERSION = 'rule_grouping_v1'


def source_identity():
    root = Path(__file__).resolve().parents[3]
    def source_hash(path):
        return hashlib.sha256(path.read_text(encoding='utf-8-sig').replace('\r\n', '\n').encode('utf-8')).hexdigest()
    files = {p.relative_to(root).as_posix(): source_hash(p)
             for folder in ('research_v4', 'grouping')
             for p in sorted((root / 'src/open_score' / folder).glob('*.py'))}
    # Physical simulator source is part of the label-generating protocol too.
    files.update({p.relative_to(root).as_posix(): source_hash(p)
                  for folder in ('src/HAD_Env', 'src/open_score/envs')
                  for p in sorted((root / folder).rglob('*.py'))})
    return {'hash': fingerprint(files), 'files': files}


def family_split(index):
    """A fixed five-family block has 3 train, 1 validation and 1 test family."""
    return ('train', 'train', 'train', 'validation', 'test')[int(index) % 5]


def collect_counterfactuals(env, state, pool, branch_seeds, continuation=None,
                           continuation_version=CONTINUATION_VERSION):
    """Return one row per complete proposal; restore simulator and ambient RNG.

    Rows contain paired Bernoulli outcomes, *remaining* terminal times and actual
    physical work. No short rollout, potential, local target success inference,
    or eventual outcome at another continuation policy is used as a label.
    """
    from .actions import rule_grouping
    from .environment import RULE_VERSION

    if (continuation is not None and continuation is not rule_grouping
            and continuation_version == CONTINUATION_VERSION):
        raise ValueError('custom continuation requires an explicit nondefault policy version')
    continuation = continuation or rule_grouping
    seeds = [int(seed) for seed in branch_seeds]
    if not pool or not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('nonempty candidates and unique independent branch seeds required')
    if not continuation_version:
        raise ValueError('a frozen continuation policy version is required')
    if env.done or state.to_dict() != env.state().to_dict():
        raise ValueError('counterfactual state must be the current nonterminal environment state')
    snapshot, ambient = env.snapshot(), random_state()
    rows = []
    try:
        for proposal in pool:
            proposal.validate(state.ids('red'), state.ids('targets'), max_members=None)
            outcomes, times, work, terminal_causes = [], [], [], []
            for branch_seed in seeds:
                env.restore(snapshot)
                env.set_rng(branch_seed)
                # A stochastic user-supplied continuation also gets CRN without
                # advancing the real execution's random stream.
                from open_score.grouping.storage import seed_everything
                seed_everything(branch_seed)
                following, _, done, info = env.step(proposal)
                steps = int(info['delta'])
                while not done:
                    action = continuation(following)
                    if hasattr(action, 'action'):
                        action = action.action
                    following, _, done, info = env.step(action)
                    steps += int(info['delta'])
                outcomes.append(int(info['success']))
                times.append(int(following.step - state.step))
                work.append(steps)
                terminal_causes.append(str(info.get('event_reason', 'terminal')))
            rows.append({'schema': SCHEMA, 'state': state.to_dict(),
                         'action': proposal.to_dict(), 'y': float(np.mean(outcomes)),
                         'outcomes': outcomes, 'terminal_steps': times,
                         'physical_steps': work, 'terminal_causes': terminal_causes,
                         'branch_seeds': seeds, 'executor_version': RULE_VERSION,
                         'opponent': state.opponent,
                         'continuation_version': str(continuation_version),
                         'label_semantics': 'proposal_one_event_then_frozen_continuation_to_native_terminal'})
    finally:
        env.restore(snapshot)
        restore_random_state(ambient)
    return rows


def paired_terminal(env, state, candidates, *, branches=4, seed=0,
                    continuation=None, continuation_version=CONTINUATION_VERSION):
    """Small online teacher API: ``(scores, physical_simulation_steps)``."""
    rng = np.random.default_rng(int(seed))
    seeds = rng.choice(np.iinfo(np.int32).max, size=int(branches), replace=False)
    rows = collect_counterfactuals(env, state, candidates, seeds, continuation,
                                  continuation_version)
    return [row['y'] for row in rows], sum(sum(row['physical_steps']) for row in rows)


def read_dataset(output, split=None):
    output = Path(output)
    manifest_path = output/'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else None
    paths = sorted((output/'families').glob('*.json'))
    if manifest and manifest.get('status') == 'complete':
        actual = {p.name: sha256(p) for p in paths}
        if fingerprint(actual) != manifest['dataset_hash']:
            raise ValueError('dataset families changed since completed manifest')
    rows = []
    for path in paths:
        family = json.loads(path.read_text(encoding='utf-8'))
        if family.get('complete') is not True:
            raise ValueError(f'incomplete committed family {path}')
        if manifest and family['identity'] != manifest['identity']:
            raise ValueError('family identity differs from dataset protocol')
        if split is None or family['split'] == split:
            rows.extend(family['rows'])
    return rows


def _counts(kind, index, scales, rng):
    if kind == 'global':
        n = scales[(index // 5) % len(scales)]
        return n, n
    maximum = max(scales)
    # Repeated blocks cover identical count pairs in independent split families.
    small = [(r, b) for r in range(1, min(4, maximum) + 1)
             for b in range(1, min(4, maximum) + 1)]
    large = sorted(set((r, b) for r in scales for b in scales))
    coverage = small + [pair for pair in large if pair not in small]
    # Interleave full supported counts early, retaining 1..4 transfer samples.
    anchors = [(min(2, maximum), min(2, maximum)), (maximum, maximum),
               (1, maximum), (maximum, 1)]
    coverage = anchors + coverage
    return coverage[(index // 5) % len(coverage)]


def _collect_family_worker(config, index, identity, continuation=None):
    """Spawn-safe complete family computation; only the parent writes files."""
    import torch
    torch.set_num_threads(1)
    from .actions import candidate_pool, grand_grouping, rule_grouping
    from .environment import make_env
    kind, seed, scales = config['kind'], config['seed'], config['scales']
    opponent, candidates, branches = config['opponent'], config['candidates'], config['branches']
    states_per_family, version = config['states_per_family'], config['continuation_version']
    policy = continuation or (grand_grouping if kind == 'local' else rule_grouping)
    family_seed = int(seed) + 1_000_003 * index
    rng = np.random.default_rng(family_seed)
    red, blue = _counts(kind, index, scales, rng)
    env = make_env(red, blue, opponent=opponent, seed=family_seed,
                   targets=config['targets'], target_positions=(
                       [[-2100., config['local_target_y'][index % 3], 100.]]
                       if kind == 'local' else None))
    try:
        state, snapshots = env.reset(seed=family_seed), []
        done = False
        behavior_steps, incoming_event = 0, 'initial'
        while not done:
            snapshots.append((state, env.snapshot(), incoming_event))
            state, _, done, info = env.step(policy(state))
            behavior_steps += int(info['delta'])
            incoming_event = info['event_reason']
        # Prespecified state categories. A category can be absent or share
        # a physical snapshot with another; never invent a casualty state.
        selected = {0: ['initial']}
        casualty = next((i for i, (_, _, event) in enumerate(snapshots)
                         if event == 'casualty'), None)
        def threat_distance(index):
            s = snapshots[index][0]
            return min((np.linalg.norm(np.asarray(b.position)-np.asarray(t.position))
                        for b in s.alive('blue') for t in s.alive('targets')),
                       default=float('inf'))
        threat = min(range(len(snapshots)), key=threat_distance)
        requested = ([('casualty', casualty)] if casualty is not None else []) + [('threat', threat)]
        for category, selected_index in requested:
            if selected_index in selected:
                selected[selected_index].append(category)
            elif len(selected) < int(states_per_family):
                selected[selected_index] = [category]
        # Values >3 request additional uniformly spread trajectory states.
        for extra_index in np.linspace(0, len(snapshots)-1, min(int(states_per_family), len(snapshots)), dtype=int):
            if len(selected) >= int(states_per_family):
                break
            selected.setdefault(int(extra_index), ['trajectory'])
        rows = []
        for event, snapshot_index in enumerate(sorted(selected)):
            state, snapshot, _ = snapshots[snapshot_index]
            env.restore(snapshot)
            pool = ([grand_grouping(state)] if kind == 'local'
                    else candidate_pool(state, budget=int(candidates), rng=rng))
            branch_seeds = rng.choice(np.iinfo(np.int32).max, size=int(branches), replace=False)
            labeled = collect_counterfactuals(env, state, pool, branch_seeds, policy, version)
            for candidate_id, row in enumerate(labeled):
                row.update(kind=kind, family_id=f'{kind}:{index}', family_index=index,
                           split=family_split(index), event_id=event,
                           state_id=f'{kind}:{index}:{event}', candidate_id=candidate_id,
                           state_kind='+'.join(selected[snapshot_index]),
                           state_categories=selected[snapshot_index],
                           initial_red=red, initial_blue=blue, family_seed=family_seed,
                           dataset_identity=identity)
            rows.extend(labeled)
        return {'identity': identity, 'complete': True,
                                 'family_id': f'{kind}:{index}',
                                 'split': family_split(index), 'rows': rows,
                                 'behavior_physical_steps': behavior_steps,
                                 'state_categories_available': {'initial': True,
                                     'casualty': casualty is not None,
                                     'threat': math.isfinite(threat_distance(threat))}}
    finally:
        env.close()


def collect_dataset(output, *, families=120, scales=(8, 12, 16, 24, 32), seed=20260906,
                    candidates=8, branches=4, states_per_family=3, kind='global',
                    seconds=None, resume=False, continuation=None,
                    continuation_version=CONTINUATION_VERSION, opponent='reactive', workers=1):
    """Collect durable complete episode families; interrupted families restart.

    ``local`` means a genuine isolated one-target environment with all Red in
    one group. It does not infer local counterfactual outcomes from an early
    terminal in a multi-target world. Local coverage is explicitly reported.
    """
    from .actions import candidate_pool, grand_grouping, rule_grouping
    from .environment import make_env, RULE_VERSION

    workers = max(1, min(6, int(workers)))
    if continuation is not None and workers != 1:
        raise ValueError('custom continuation collection requires workers=1')
    output = Path(output)
    scales = tuple(sorted(set(int(n) for n in scales)))
    if (kind not in ('global', 'local') or not scales or min(scales) < 1
            or min(families, candidates, branches, states_per_family) < 1):
        raise ValueError('valid kind, positive counts and nonempty positive scales required')
    if kind == 'local' and continuation is not None:
        raise ValueError('isolated local data always uses frozen grand-group continuation')
    if (continuation is not None and continuation is not rule_grouping
            and continuation_version == CONTINUATION_VERSION):
        raise ValueError('custom continuation requires an explicit nondefault policy version')
    policy = continuation or (grand_grouping if kind == 'local' else rule_grouping)
    version = 'grand_grouping_v1' if kind == 'local' else continuation_version
    config = {'schema': SCHEMA, 'kind': kind, 'seed': int(seed), 'scales': list(scales),
              'candidates': int(candidates), 'branches': int(branches),
              'states_per_family': int(states_per_family), 'opponent': opponent,
              'executor_version': RULE_VERSION, 'continuation_version': version,
              'max_steps': 50, 'command_interval': 5,
              'targets': 1 if kind == 'local' else 2,
              'local_target_y': [-650., 0., 650.] if kind == 'local' else None,
              'source': source_identity()}
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        if not resume:
            raise ValueError('dataset exists; use resume or a new output')
        if previous['config'] != config:
            raise ValueError('dataset source/executor/opponent/continuation settings changed')
        if int(families) < previous['requested_families']:
            raise ValueError('cannot silently shrink a resumed dataset')
    elif output.exists() and any(output.iterdir()):
        raise ValueError('nonempty dataset directory has no identity manifest')
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    identity = fingerprint(config)
    manifest = {'config': config, 'identity': identity,
                'requested_families': int(families), 'status': 'collecting', 'collection_workers': workers}
    atomic_json(manifest_path, manifest)
    pending_indices = []
    for index in range(int(families)):
        family_path = output/'families'/f'{index:06d}.json'
        if family_path.exists():
            committed = json.loads(family_path.read_text(encoding='utf-8'))
            if committed['identity'] != identity or not committed['complete']:
                raise ValueError('committed family identity differs')
        else:
            pending_indices.append(index)
    def expired():
        return seconds is not None and time.monotonic()-started >= float(seconds)
    def commit_family(index, value):
        atomic_json(output/'families'/f'{index:06d}.json', value)
    if workers == 1:
        for index in pending_indices:
            if expired():
                break
            commit_family(index, _collect_family_worker(config, index, identity, continuation))
    elif pending_indices and not expired():
        from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
        import multiprocessing
        import os
        # Children import NumPy/Torch after these limits are inherited. Only
        # owned worker processes are involved; environment is restored after.
        names = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                 'CUDA_VISIBLE_DEVICES')
        before_environment = {name: os.environ.get(name) for name in names}
        for name in names:
            os.environ[name] = '' if name == 'CUDA_VISIBLE_DEVICES' else '1'
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as executor:
                iterator, active = iter(pending_indices), {}
                def fill():
                    while len(active) < workers and not expired():
                        index = next(iterator, None)
                        if index is None:
                            break
                        active[executor.submit(_collect_family_worker, config, index, identity)] = index
                fill()
                while active:
                    finished, _ = wait(active, return_when=FIRST_COMPLETED)
                    for future in finished:
                        index = active.pop(future)
                        commit_family(index, future.result())
                    # At deadline no further jobs are queued. Already running
                    # complete families finish and commit, bounded by workers.
                    fill()
        finally:
            for name, value in before_environment.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
    rows = read_dataset(output)
    completed = len(list((output / 'families').glob('*.json')))
    observed = sorted({(len(DecisionState.from_dict(r['state']).ids('red')),
                        len(DecisionState.from_dict(r['state']).ids('blue'))) for r in rows})
    committed = [json.loads(p.read_text(encoding='utf-8'))
                 for p in sorted((output/'families').glob('*.json'))]
    behavior_steps = sum(f['behavior_physical_steps'] for f in committed)
    simulation_steps = sum(sum(r['physical_steps']) for r in rows)
    manifest.update(status='complete' if completed == int(families) else 'partial',
                    completed_families=completed, rows=len(rows),
                    split_rows={s: sum(r['split'] == s for r in rows)
                                for s in ('train', 'validation', 'test')},
                    observed_rosters=[list(x) for x in observed],
                    physical_simulation_steps=simulation_steps,
                    behavior_physical_steps=behavior_steps,
                    total_physical_steps=simulation_steps+behavior_steps,
                    state_category_family_availability={category: sum(
                        f['state_categories_available'][category] for f in committed)
                        for category in ('initial', 'casualty', 'threat')},
                    state_category_rows={category: sum(category in r['state_categories'] for r in rows)
                        for category in ('initial', 'casualty', 'threat')},
                    elapsed_seconds=time.monotonic()-started,
                    full_cartesian_roster_coverage=False,
                    family_hashes={p.name: sha256(p) for p in sorted((output/'families').glob('*.json'))})
    manifest['dataset_hash'] = fingerprint(manifest['family_hashes'])
    atomic_json(manifest_path, manifest)
    return manifest


def prepare(config, output, resume=False):
    """Shared dependency: collect local/global S2 once, fit and audit once."""
    from .outcomes import (train_evaluator, train_count_model, historical_transfer_audit,
                           cross_evaluator_comparison, load_evaluator)
    import torch

    output = Path(output)
    torch.set_num_threads(max(1, int(config.get('threads', 1))))
    smoke = bool(config.get('smoke', False))
    seed = int(config.get('seed', config.get('seeds', [20260906])[0]))
    scales = tuple(config.get('scales', (8, 12, 16, 24, 32)))
    common = dict(scales=scales, seed=seed + 41_000_000,
                  branches=int(config.get('s2_branches', 1 if smoke else 4)),
                  states_per_family=int(config.get('s2_states', 1 if smoke else 3)),
                  opponent=config.get('opponent', 'reactive'), resume=resume,
                  seconds=config.get('s2_collect_seconds'),
                  workers=int(config.get('data_workers', 1 if smoke else 6)))
    global_data = collect_dataset(output/'global_data', kind='global',
                                  families=int(config.get('s2_families', 10 if smoke else 120)),
                                  candidates=int(config.get('s2_candidates', 2 if smoke else 8)), **common)
    local_data = collect_dataset(output/'local_data', kind='local',
                                 families=int(config.get('s2_local_families', 10 if smoke else 160)),
                                 candidates=1, **common)
    fitting = dict(epochs=int(config.get('s2_epochs', 2 if smoke else 40)),
                   seconds=config.get('s2_seconds'), device=config.get('device', 'cpu'),
                   resume=resume, hidden_dim=32 if smoke else 128,
                   heads=4, layers=1 if smoke else 2, seed=seed)
    global_fit = train_evaluator(output/'global_data', output/'global_model', kind='global', **fitting)
    local_fit = train_evaluator(output/'local_data', output/'local_model', kind='local', **fitting)
    count_path = output/'count_model.json'
    count_report = train_count_model(output/'local_data', count_path)
    transfer = historical_transfer_audit(output/'local_data')
    atomic_json(output/'historical_transfer.json', transfer)
    comparison = cross_evaluator_comparison(output/'global_data',
        load_evaluator(output/'local_model/best.pt', fitting['device']),
        load_evaluator(output/'global_model/best.pt', fitting['device']))
    atomic_json(output/'cross_evaluator_comparison.json', comparison)
    complete = (global_data['status'] == local_data['status'] == 'complete'
                and global_fit['trained'] and local_fit['trained'])
    result = {'status': 'complete' if complete else 'partial',
              'global_checkpoint': str((output/'global_model/best.pt').resolve()),
              'local_checkpoint': str((output/'local_model/best.pt').resolve()),
              'count_checkpoint': str(count_path.resolve()),
              'global_data': global_data, 'local_data': local_data,
              'global_fit': global_fit, 'local_fit': local_fit,
              'count_report': count_report, 'historical_transfer': transfer,
              'cross_evaluator_comparison': comparison,
              'protocol': 'fixed-rule lower; complete native terminal; family split; frozen continuation'}
    atomic_json(output/'preparation_report.json', result)
    return result
