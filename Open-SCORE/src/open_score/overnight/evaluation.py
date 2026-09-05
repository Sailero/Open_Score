"""Resumable paired held-out evaluation of learned and pure-rule groupings."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import re
import time

import numpy as np
import torch

from open_score.grouping.baselines import StaticPolicy
from open_score.grouping.domain import Grouping
from open_score.grouping.evaluation import grouping_changes
from open_score.grouping.storage import (append_jsonl, atomic_json, fingerprint,
    random_state, read_jsonl, restore_random_state, seed_everything, sha256)
from .actions import candidate_actions, rule_action
from .environment import make_env
from .policy import CandidateNetwork


BASELINES = ('static', 'dynamic_rule', 'compact_rule', 'all_reserve')
BEHAVIOR_KEYS = ('release_ratio', 'task_change', 'team_change', 'mean_reserve_fraction',
                 'mean_group_size', 'singleton_fraction', 'singleton_group_fraction',
                 'red_survivors', 'blue_survivors', 'physical_steps', 'upper_events')
METRIC_DEFINITIONS = {
    'release_ratio': 'Fraction of surviving members whose target or teammates change; excludes initial deployment.',
    'task_change': 'Fraction whose target/reserve assignment changes; excludes initial deployment and casualties.',
    'team_change': 'Changed co-membership pairs divided by all live pairs; excludes initial deployment.',
    'mean_reserve_fraction': 'Mean over upper decisions of reserve members / live red members, including deployment.',
    'mean_group_size': 'Mean over upper decisions of the mean non-reserve group size; zero if no groups.',
    'singleton_fraction': 'Mean over upper decisions of singleton members / deployed red members; zero if none deployed.',
    'singleton_group_fraction': 'Mean over upper decisions of singleton groups / active groups; zero if no groups.',
}


def evaluation_seed(config, scale, episode, *, validation=False):
    """Training uses +100,000; validation and final test use disjoint domains."""
    return int(config['seed']) + (20_000_000 if validation else 30_000_000) + int(scale) * 10_000 + int(episode)


def changed_members(previous, action, live_ids):
    """A label-free membership comparison, including regrouping at one target."""
    def membership(grouping):
        result = {i: (None, ()) for i in grouping.reserve}
        for group in grouping.groups:
            result.update({i: (group.target, group.members) for i in group.members})
        return result
    before = membership(previous.prune(live_ids))
    after = membership(action)
    return tuple(i for i in live_ids if before.get(i) != after.get(i))


def behavior(action, live_count):
    sizes = [len(group.members) for group in action.groups]
    singletons = sizes.count(1)
    return {'mean_reserve_fraction': len(action.reserve) / max(1, live_count),
            'mean_group_size': float(np.mean(sizes)) if sizes else 0.,
            'singleton_fraction': singletons / max(1, sum(sizes)),
            'singleton_group_fraction': singletons / max(1, len(sizes))}


def _action(method, model, state, config):
    if model is not None:
        pool = candidate_actions(state, limit=int(config['candidate_limit']))
        scores, _ = model([state], [pool])
        if not torch.isfinite(scores[0, :len(pool)]).all():
            raise FloatingPointError(f'Nonfinite evaluation scores for {method}')
        return pool[int(scores[0, :len(pool)].argmax().item())]
    if method == 'static':
        return StaticPolicy().act(state).action
    if method == 'dynamic_rule':
        return rule_action(state)
    if method == 'compact_rule':
        from .actions import compact_action
        return compact_action(state)
    if method == 'all_reserve':
        return Grouping((), state.ids('red'))
    raise ValueError(f'Unknown untrained baseline: {method}')


def _sync(model):
    if model is not None and next(model.parameters()).is_cuda:
        torch.cuda.synchronize(next(model.parameters()).device)


def rollout_episode(env, method, model, config, *, seed, save_trace=False):
    seed_everything(seed)
    state = env.reset(seed=seed)
    done, physical_steps, timings, traces = False, 0, [], []
    release, task, team, behaviors = [], [], [], []
    while not done:
        _sync(model)
        started = time.perf_counter()
        with torch.no_grad():
            action = _action(method, model, state, config)
        action.validate(state.ids('red'), state.ids('targets'))
        _sync(model)
        timings.append(time.perf_counter() - started)
        released = changed_members(state.previous, action, state.ids('red'))
        changes = grouping_changes(state.previous, action, state.ids('red'))
        current_behavior = behavior(action, len(state.ids('red')))
        behaviors.append(current_behavior)
        if state.step > 0:
            release.append(len(released) / max(1, len(state.ids('red'))))
            task.append(changes['task_change'])
            team.append(changes['team_change'])
        if save_trace:
            traces.append({'step': state.step, 'red_alive': len(state.ids('red')),
                'blue_alive': len(state.ids('blue')), 'released': list(released),
                'action': action.to_dict(), **changes, **current_behavior,
                'public_state': {side: [{'id': entity.id, 'position': list(entity.position),
                    'velocity': list(entity.velocity), 'health': entity.health}
                    for entity in state.alive(side)] for side in ('red', 'blue', 'targets')}})
        state, _reward, done, info = env.step(action)
        delta = int(info['delta'])
        if delta < 1:
            raise ValueError('Evaluation transition must advance physical time')
        physical_steps += delta
    row = {'seed': int(seed), 'method': method, 'success': bool(info['success']),
           'physical_steps': physical_steps, 'upper_events': len(timings),
           'decision_seconds': timings, 'decision_median_seconds': float(np.median(timings)),
           'decision_p95_seconds': float(np.quantile(timings, .95)),
           'release_ratio': float(np.mean(release)) if release else 0.,
           'task_change': float(np.mean(task)) if task else 0.,
           'team_change': float(np.mean(team)) if team else 0.,
           **{key: float(np.mean([value[key] for value in behaviors])) for key in behaviors[0]},
           'constraint_violations': 0, 'red_survivors': len(state.ids('red')),
           'blue_survivors': len(state.ids('blue'))}
    return row, traces


def summarize(rows, methods, *, planned_units=None):
    units = {}
    for row in rows:
        unit_key = (int(row['scale']), int(row['seed']))
        if planned_units is not None and unit_key not in planned_units:
            continue
        if row['method'] not in methods:
            raise ValueError(f"Unexpected evaluation method in raw rows: {row['method']}")
        unit = units.setdefault(unit_key, {})
        if row['method'] in unit:
            raise ValueError(f'Duplicate raw evaluation result: {unit_key}, {row["method"]}')
        unit[row['method']] = row
    complete = [unit for unit in units.values() if set(unit) == set(methods)]
    table = []
    for scale in sorted({key[0] for key in units}):
        for method in methods:
            selected = [unit[method] for unit in complete if unit[method]['scale'] == scale]
            if not selected:
                continue
            count, successes = len(selected), sum(bool(row['success']) for row in selected)
            rate, z = successes / count, 1.96
            center = (rate + z*z/(2*count)) / (1+z*z/count)
            half = z * math.sqrt(rate*(1-rate)/count + z*z/(4*count*count)) / (1+z*z/count)
            times = [value for row in selected for value in row['decision_seconds']]
            table.append({'scale': int(scale), 'method': method, 'episodes': count,
                'successes': successes, 'success_rate': rate,
                'wilson_low': max(0., center-half), 'wilson_high': min(1., center+half),
                'decision_median_ms': float(np.median(times)) * 1000,
                'decision_p95_ms': float(np.quantile(times, .95)) * 1000,
                **{key: float(np.mean([row[key] for row in selected])) for key in BEHAVIOR_KEYS},
                'constraint_violations': sum(row['constraint_violations'] for row in selected)})
    return {'complete_pairs': len(complete), 'incomplete_pairs': len(units)-len(complete),
            'raw_episodes': sum(len(unit) for unit in units.values()),
            'complete_episodes': len(complete)*len(methods), 'table': table}


def _load_models(config, checkpoints):
    models, hashes, parameters = {}, {}, {}
    device = config.get('device', 'cpu')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for method, checkpoint in checkpoints.items():
        if not re.fullmatch(r'[A-Za-z0-9_-]+', method):
            raise ValueError('Method names must be safe file names using letters, numbers, _ or -')
        if checkpoint is None:
            if method not in BASELINES:
                raise ValueError(f'Unknown untrained baseline: {method}')
            models[method], hashes[method], parameters[method] = None, f'pure_rule:{method}', 0
        else:
            checkpoint = Path(checkpoint)
            hashes[method] = sha256(checkpoint)
            payload = torch.load(checkpoint, map_location=device, weights_only=False)
            model = CandidateNetwork(**payload['config']['model']).to(device)
            model.load_state_dict(payload['model'], strict=True)
            model.requires_grad_(False).eval()
            models[method] = model
            parameters[method] = sum(parameter.numel() for parameter in model.parameters())
    return models, hashes, parameters


def _read_raw(path):
    """Recover an interrupted final JSONL append without joining bad bytes.

    The incomplete bytes are retained separately; malformed interior records
    remain an error. Completed records are never rewritten on resume.
    """
    path = Path(path)
    if path.exists():
        blocks = path.read_bytes().splitlines(keepends=True)
        valid_bytes = 0
        for index, block in enumerate(blocks):
            if block.strip():
                try:
                    json.loads(block)
                except (ValueError, UnicodeDecodeError):
                    if index != len(blocks)-1:
                        raise ValueError('Malformed interior evaluation JSONL record') from None
                    backup = path.with_name(f'evaluation_truncated_tail_{time.time_ns()}.bin')
                    backup.write_bytes(block)
                    with path.open('r+b') as stream:
                        stream.truncate(valid_bytes)
                    break
            valid_bytes += len(block)
    return read_jsonl(path)


def _identity(config, checkpoints, scales, validation):
    keys = ('seed', 'opponent', 'max_steps', 'command_interval', 'executor_scope',
            'executor_device', 'device', 'candidate_limit', 'evaluation_decoding')
    root = Path(__file__).resolve().parents[3]
    source_paths = [Path(__file__).with_name(name)
                    for name in ('evaluation.py', 'actions.py', 'policy.py', 'environment.py')]
    source_paths += [Path(__file__).parents[1] / 'grouping' / name
                     for name in ('domain.py', 'environment.py', 'frozen.py', 'opponents.py', 'baselines.py')]
    return {'schema': 'overnight-paired-evaluation-v1', 'checkpoints': checkpoints,
            'config': {key: config.get(key) for key in keys}, 'scales': list(scales),
            'split': 'validation' if validation else 'held_out_test',
            'evaluation_seed_base': int(config['seed']) + (20_000_000 if validation else 30_000_000),
            'metric_definitions': METRIC_DEFINITIONS,
            'code_hashes': {path.relative_to(root).as_posix(): sha256(path) for path in source_paths},
            'frozen_assets': {path.relative_to(root).as_posix(): sha256(path)
                              for path in sorted((root / 'assets/frozen').rglob('*')) if path.is_file()}}


def write_report(output, summary):
    output = Path(output)
    table = summary['table']
    fields = list(table[0]) if table else ['scale', 'method', 'episodes', 'successes', 'success_rate']
    with (output / 'results.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)
    lines = ['# 已知对手动态分组配对评估', '',
        f"评估集合：{summary['split']}；完整配对 {summary['complete_pairs']}/{summary['planned_pairs']}；"
        f"完整比较回合 {summary['complete_episodes']}；未完成配对 {summary['incomplete_pairs']}。", '',
        f"环境：已知 {summary['environment']['opponent']} 对手；"
        f"每局最多 {summary['environment']['max_steps']} 步；"
        f"冻结下层观测适配 {summary['environment']['executor_scope']}，所有方法保持一致。", '',
        '每个配对使用相同初始状态和环境随机种子；只有所有方法完成的配对计入下表。',
        'static、dynamic_rule、compact_rule、all_reserve 都不训练；学习策略均对同一候选生成器作确定性贪心选择。',
        '这是单训练种子评估；Wilson 区间描述测试回合不确定性，不代表跨训练种子的可靠性。', '',
        '| 规模 | 方法 | 成功/回合 | 成功率 | 后备比例 | 平均组大小 | 单人成员比例 | 决策 P95(ms) |',
        '|---:|---|---:|---:|---:|---:|---:|---:|']
    for row in table:
        lines.append(f"| {row['scale']} | {row['method']} | {row['successes']}/{row['episodes']} | "
            f"{row['success_rate']:.1%} | {row['mean_reserve_fraction']:.3f} | "
            f"{row['mean_group_size']:.2f} | {row['singleton_fraction']:.3f} | {row['decision_p95_ms']:.2f} |")
    lines += ['', '行为指标包含初始部署；变化指标排除初始部署及自然死亡。',
        '单人成员比例以已部署成员为分母；全后备时为 0，必须同时查看后备比例。',
        '未完成配对仍保留在 evaluation.jsonl，恢复时只补缺失方法；它们不进入成功率统计。', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    if not table:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods = list(dict.fromkeys(row['method'] for row in table))
    scales = sorted({row['scale'] for row in table})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    width = .8 / len(methods)
    for method_index, method in enumerate(methods):
        lookup = {row['scale']: row for row in table if row['method'] == method}
        x = np.arange(len(scales)) + (method_index-(len(methods)-1)/2)*width
        success_label = 'Validation success rate' if summary['split'] == 'validation' else 'Held-out success rate'
        for axis, key, label in zip(axes, ('success_rate', 'mean_reserve_fraction', 'mean_group_size'),
                                   (success_label, 'Reserve fraction', 'Mean group size')):
            axis.bar(x, [lookup.get(scale, {}).get(key, 0.) for scale in scales], width=width, label=method)
            axis.set_xticks(np.arange(len(scales)), [f'{scale}v{scale}' for scale in scales])
            axis.set_ylabel(label)
    axes[0].set_ylim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[2].set_ylim(0, 4.2)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / 'comparison.png', dpi=160)
    plt.close(fig)


def evaluate_models(config, checkpoints, output, *, episodes, wall_seconds, validation=False):
    """Finish paired units, retain partial raw rows, and restore training RNG.

    Budget checks occur between units. One in-flight paired unit may finish
    after the nominal budget; the outer portfolio enforces its hard deadline.
    Increasing ``episodes`` can resume the same model/task output directory.
    """
    if not checkpoints or int(episodes) != episodes or not 1 <= episodes < 10_000:
        raise ValueError('Provide models and between 1 and 9,999 episodes per scale')
    if wall_seconds is not None and (not math.isfinite(wall_seconds) or wall_seconds < 0):
        raise ValueError('Evaluation wall_seconds must be finite and nonnegative')
    scales = [int(value) for value in config['validation_scales' if validation else 'eval_scales']]
    if not scales or len(set(scales)) != len(scales) or any(scale <= 0 or scale >= 900 for scale in scales):
        raise ValueError('Provide distinct positive evaluation scales below 900 for disjoint seed domains')
    if config.get('evaluation_decoding', 'greedy') != 'greedy':
        raise ValueError('The overnight comparison uses greedy decoding for every learned route')
    saved_rng, started = random_state(), time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    try:
        models, checkpoint_hashes, parameters = _load_models(config, checkpoints)
        identity = _identity(config, checkpoint_hashes, scales, validation)
        manifest = {'identity_hash': fingerprint(identity), **identity}
        manifest_path = output / 'evaluation_manifest.json'
        if manifest_path.exists():
            if json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
                raise ValueError('Evaluation model/task differs; use a new evaluation output directory')
        elif (output / 'evaluation.jsonl').exists():
            raise ValueError('Raw evaluation exists without an identity manifest; choose a new output directory')
        else:
            atomic_json(manifest_path, manifest)
        raw_path = output / 'evaluation.jsonl'
        rows = _read_raw(raw_path)
        planned = {(scale, evaluation_seed(config, scale, episode, validation=validation))
                   for scale in scales for episode in range(episodes)}
        methods = list(models)
        # Validate duplicate records before using them as resume evidence.
        summarize(rows, methods, planned_units=planned)
        completed = {(int(row['scale']), int(row['seed']), row['method']) for row in rows}

        def summary_now():
            result = summarize(rows, methods, planned_units=planned)
            result.update(planned_pairs=len(planned), complete=result['complete_pairs'] == len(planned),
                          elapsed_seconds=time.perf_counter()-started, requested_seconds=wall_seconds,
                          split=identity['split'], checkpoint_hashes=checkpoint_hashes,
                          model_parameters=parameters, metric_definitions=METRIC_DEFINITIONS,
                          environment={key: config[key] for key in ('opponent', 'max_steps', 'command_interval', 'executor_scope')},
                          interpretation='One training seed; episode uncertainty is not across-training-seed uncertainty.')
            atomic_json(output / 'summary.json', result)
            return result

        summary_now()
        for scale in scales:
            missing = [episode for episode in range(episodes) if not all(
                (scale, evaluation_seed(config, scale, episode, validation=validation), method) in completed
                for method in methods)]
            if not missing:
                continue
            if wall_seconds is not None and time.perf_counter()-started >= wall_seconds:
                break
            env = make_env(red=scale, blue=scale, opponent=config['opponent'],
                executor_scope=config['executor_scope'], max_steps=config['max_steps'],
                command_interval=config['command_interval'], device=config.get('executor_device', 'cpu'))
            try:
                for episode in missing:
                    if wall_seconds is not None and time.perf_counter()-started >= wall_seconds:
                        break
                    seed = evaluation_seed(config, scale, episode, validation=validation)
                    for method, model in models.items():
                        if (scale, seed, method) in completed:
                            continue
                        row, trace = rollout_episode(env, method, model, config, seed=seed, save_trace=episode == 0)
                        row['scale'] = scale
                        # Trace first: a crash before row append only reruns this method.
                        if trace:
                            atomic_json(output / f'trace_{method}_{scale}.json', trace)
                        append_jsonl(raw_path, row)
                        rows.append(row)
                        completed.add((scale, seed, method))
                    result = summary_now()
                    print(f"[evaluation] {identity['split']} scale={scale} pair={episode+1}/{episodes}; "
                          f"total={result['complete_pairs']}/{len(planned)}", flush=True)
            finally:
                env.close()
        result = summary_now()
        write_report(output, result)
        result['elapsed_seconds'] = time.perf_counter()-started
        atomic_json(output / 'summary.json', result)
        return result
    except BaseException:
        # An interrupted paired unit remains visible but cannot affect rates.
        if 'summary_now' in locals():
            summary_now()
        raise
    finally:
        restore_random_state(saved_rng)
