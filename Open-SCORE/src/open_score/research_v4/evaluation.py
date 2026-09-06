"""Paired physical-world evaluation, with resumable raw episode records."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import numpy as np

from .runner import atomic_json, file_hash, read_json


def read_records(path, *, repair_tail=False):
    path = Path(path)
    if not path.exists():
        return []
    text = path.read_text(encoding='utf-8')
    lines = text.splitlines(keepends=True)
    result = []
    good = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            result.append(json.loads(line))
            good.append(line if line.endswith('\n') else line + '\n')
        except json.JSONDecodeError:
            if index != len(lines) - 1 or not repair_tail:
                raise ValueError(f'Invalid evaluation record at {path}:{index + 1}')
            backup = path.with_name(path.name + f'.incomplete-{time.time_ns()}')
            backup.write_text(text, encoding='utf-8')
            path.write_text(''.join(good), encoding='utf-8')
    return result


def wilson(wins, count):
    if count == 0:
        return [None, None]
    p = wins / count
    z = 1.959963984540054
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    width = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [center - width, center + width]


def seed_for(training_seed, scale, episode):
    return int(training_seed) + 300_000_000 + int(scale) * 100_000 + int(episode)


def _membership(grouping):
    result = {member: (None, ()) for member in grouping.reserve}
    for group in grouping.groups:
        signature = (group.target, tuple(sorted(group.members)))
        result.update({member: signature for member in group.members})
    return result


def rollout(policy, *, scale, seed, config, trace=False):
    from .environment import make_env
    env = make_env(scale, opponent=config.get('opponent', 'reactive'), seed=seed,
                   max_steps=config.get('max_steps', 50),
                   command_interval=config.get('command_interval', 5))
    state = env.reset(seed=seed)
    timings, changes, target_changes, group_sizes, records = [], [], [], [], []
    done, steps, info = False, 0, {}
    try:
        if hasattr(policy, 'reset'):
            policy.reset()
        while not done:
            before = _membership(state.previous.prune(state.ids('red')))
            start = time.perf_counter()
            action = policy.act(state) if hasattr(policy, 'act') else policy(state)
            if hasattr(action, 'action'):
                action = action.action
            timings.append(time.perf_counter() - start)
            action.validate(state.ids('red'), state.ids('targets'), max_members=None)
            after = _membership(action)
            if state.step > 0:
                changes.append(sum(before.get(i) != after.get(i) for i in state.ids('red')) / max(1, len(state.ids('red'))))
                target_changes.append(sum(before.get(i, (None,))[0] != after.get(i, (None,))[0] for i in state.ids('red')) / max(1, len(state.ids('red'))))
            group_sizes.extend(len(group.members) for group in action.groups)
            if trace:
                records.append({'step': state.step, 'grouping': action.to_dict(),
                    'entities': {side: [{'id': entity.id, 'position': list(entity.position),
                        'velocity': list(entity.velocity), 'health': float(entity.health)}
                        for entity in state.alive(side)] for side in ('red', 'blue', 'targets')}})
            previous_step = state.step
            state, reward, done, info = env.step(action)
            delta = int(info.get('delta', state.step - previous_step))
            if delta <= 0:
                raise ValueError('Evaluation did not advance physical time')
            steps += delta
            if steps > config.get('max_steps', 50):
                raise ValueError('Evaluation exceeded the physical horizon')
        result = {'scale': int(scale), 'seed': int(seed), 'success': bool(info.get('success', False)),
                  'physical_steps': steps, 'upper_events': len(timings),
                  'decision_seconds': timings,
                  'decision_p95_seconds': float(np.quantile(timings, .95)) if timings else 0.,
                  'member_change': float(np.mean(changes)) if changes else 0.,
                  'target_change': float(np.mean(target_changes)) if target_changes else 0.,
                  'mean_group_size': float(np.mean(group_sizes)) if group_sizes else 0.,
                  'red_survivors': len(state.ids('red')), 'blue_survivors': len(state.ids('blue')),
                  'constraint_violations': 0}
        return result, records
    finally:
        if hasattr(env, 'close'):
            env.close()


def summary(rows, *, route, variants, scales, episodes):
    groups = []
    for variant in variants:
        for scale in scales:
            selected = [row for row in rows if row['variant'] == variant and row['scale'] == scale]
            wins = sum(row['success'] for row in selected)
            groups.append({'route': route, 'variant': variant, 'scale': scale,
                'wins': wins, 'episodes': len(selected),
                'success_rate': wins / len(selected) if selected else None,
                'wilson95': wilson(wins, len(selected)),
                **{key: float(np.mean([row[key] for row in selected])) if selected else None
                   for key in ('physical_steps', 'member_change', 'target_change', 'mean_group_size', 'decision_p95_seconds')}})
    return {'schema': 'research-v4-evaluation-v1', 'route': route,
            'episodes_per_scale': episodes, 'variants': list(variants), 'scales': list(scales),
            'complete': all(group['episodes'] == episodes for group in groups),
            'groups': groups, 'raw_episode_count': len(rows)}


def evaluation_identity(config, checkpoints, assets):
    from .config import protocol, provenance
    fixed = dict(protocol(config))
    fixed.pop('assets', None)
    fixed.pop('shared_dir', None)
    fixed.pop('data_workers', None)
    route = config['route']
    kind = ('count' if route == 'b1_counts' else 'local' if route in
            ('b2_local', 'b2_count', 'b2_min') else 'global') if route.startswith('b') or route in ('group_only', 'target_only') else None
    selected = {key: value for key, value in assets.items() if key == f'{kind}_checkpoint'} if kind else {}
    return {'protocol': fixed, 'source_hash': provenance()['source_hash'],
            'checkpoint_hashes': {name: file_hash(path) for name, path in checkpoints.items()},
            'assets': {key: file_hash(value) for key, value in selected.items()}}


def evaluate(config, output, *, assets, resume=False):
    import torch
    from .policies import make_policy
    torch.set_num_threads(int(config.get('threads', 1)))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    route = config['route']
    episodes = int(config.get('eval_episodes', 100))
    scales = [int(scale) for scale in config.get('scales', (8, 12, 16, 24, 32))]
    checkpoints = {}
    if route.startswith('r') and route in ('r1_ppo', 'r2_teacher_ppo', 'r3_ddqn'):
        from .training import load_policy
        for variant in ('best', 'latest', 'initialized'):
            checkpoint = output / f'{variant}.pt'
            if checkpoint.exists():
                checkpoints[variant] = checkpoint
        if not checkpoints:
            raise FileNotFoundError(f'No learned checkpoint found in {output}; train this route first')
        factories = {name: (lambda p=path: load_policy(p, config['device'])) for name, path in checkpoints.items()}
    else:
        factories = {'policy': lambda: make_policy(route, assets,
            device=config.get('device', 'cpu'), search_budget=config.get('search_budget', 64),
            seed=config['seed'])}
    identity = evaluation_identity(config, checkpoints, assets)
    existing = read_json(output / 'evaluation_manifest.json')
    if existing and existing != identity:
        if existing.get('source_hash') != identity['source_hash'] or existing.get('protocol') != identity['protocol']:
            raise ValueError('Evaluation source or protocol changed; use a new output directory')
        # A continued learner produces new checkpoint versions. Preserve prior tests separately.
        if not resume:
            raise ValueError('Evaluation identity changed; use --resume to archive previous checkpoint evaluations')
        archive = output / f'evaluation_archive_{time.time_ns()}'
        archive.mkdir()
        for name in ('evaluation_manifest.json', 'evaluation.jsonl', 'evaluation_summary.json'):
            source = output / name
            if source.exists():
                source.replace(archive / name)
    elif existing and not resume:
        raise ValueError('Evaluation output already exists; use --resume')
    atomic_json(output / 'evaluation_manifest.json', identity)
    rows = read_records(output / 'evaluation.jsonl', repair_tail=resume)
    seen = {(row['variant'], row['scale'], row['episode']) for row in rows}
    if len(seen) != len(rows):
        raise ValueError('Duplicate evaluation episodes; raw records require inspection')
    requested_rows = [row for row in rows if row['variant'] in factories and row['scale'] in scales and row['episode'] < episodes]
    with (output / 'evaluation.jsonl').open('a', encoding='utf-8', buffering=1) as stream:
        coverage = {}
        for variant, factory in factories.items():
            policy = factory()
            for scale in scales:
                for episode in range(episodes):
                    if (variant, scale, episode) in seen:
                        continue
                    seed = seed_for(config['seed'], scale, episode)
                    row, trace = rollout(policy, scale=scale, seed=seed, config=config, trace=episode == 0)
                    row.update(route=route, variant=variant, episode=episode, training_seed=config['seed'])
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                    requested_rows.append(row)
                    if trace:
                        atomic_json(output / 'traces' / f'{variant}_{scale}.json', trace)
                    atomic_json(output / 'evaluation_summary.json', summary(requested_rows, route=route,
                        variants=factories, scales=scales, episodes=episodes))
            if hasattr(policy, 'scorer') and hasattr(policy.scorer, 'coverage'):
                coverage[variant] = dict(policy.scorer.coverage)
            del policy
    result = summary(requested_rows, route=route, variants=factories, scales=scales, episodes=episodes)
    result['evaluator_coverage_this_invocation'] = coverage
    atomic_json(output / 'evaluation_summary.json', result)
    return result
