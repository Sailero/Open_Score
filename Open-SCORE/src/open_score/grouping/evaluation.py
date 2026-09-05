"""Paired complete-episode evaluation, structural metrics and briefing artifacts."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .environment import KnownOpponentEnv
from .storage import (append_jsonl, atomic_json, fingerprint, read_jsonl,
                      seed_everything, sha256)
from .training import load_policy, make_policy


def grouping_changes(previous, action, live_ids):
    ids = tuple(live_ids)
    old = previous.prune(ids)
    before, after = old.assignment(), action.assignment()
    task = sum(before[i] != after[i] for i in ids) / max(1, len(ids))
    def pairs(grouping):
        return {frozenset((i, j)) for group in grouping.groups
                for index, i in enumerate(group.members) for j in group.members[index + 1:]}
    denominator = len(ids) * (len(ids) - 1) / 2
    team = len(pairs(old).symmetric_difference(pairs(action))) / denominator if denominator else 0.0
    return {'task_change': float(task), 'team_change': float(team)}


def _sync(policy):
    if isinstance(policy, torch.nn.Module) and next(policy.parameters()).is_cuda:
        torch.cuda.synchronize()


def rollout_episode(env, policy, *, seed, first_action=None, static_after_first=False, save_trace=False):
    seed_everything(seed + 7000000)
    state = env.reset(seed=seed)
    done, steps, release, task_changes, team_changes = False, 0, [], [], []
    timings, trace, actual_first = [], [], None
    while not done:
        _sync(policy)
        started = time.perf_counter()
        with torch.no_grad():
            if state.step == 0 and first_action is not None:
                action, released = first_action, tuple(state.ids('red'))
            elif static_after_first and state.step > 0:
                action, released = state.previous.prune(state.ids('red')), ()
            else:
                decision = policy.act(state)
                action, released = decision.action, decision.released_ids
        action.validate(state.ids('red'), state.ids('targets'))
        _sync(policy)
        timings.append(time.perf_counter() - started)
        change = grouping_changes(state.previous, action, state.ids('red'))
        if state.step:
            release.append(len(released) / max(1, len(state.ids('red'))))
            task_changes.append(change['task_change'])
            team_changes.append(change['team_change'])
        else:
            actual_first = action
        if save_trace:
            trace.append({'step': state.step, 'red_alive': len(state.ids('red')),
                          'blue_alive': len(state.ids('blue')), 'released': list(released),
                          'action': action.to_dict(), **change,
                          'positions': {str(e.id): list(e.position) for e in state.alive('red')},
                          'targets': {str(e.id): list(e.position) for e in state.targets}})
        state, reward, done, info = env.step(action)
        steps += info['delta']
    row = {'seed': seed, 'success': bool(info['success']), 'physical_steps': steps,
           'upper_events': len(timings), 'decision_seconds': timings,
           'decision_median_seconds': float(np.median(timings)),
           'decision_p95_seconds': float(np.quantile(timings, .95)),
           'release_ratio': float(np.mean(release)) if release else 0.,
           'task_change': float(np.mean(task_changes)) if task_changes else 0.,
           'team_change': float(np.mean(team_changes)) if team_changes else 0.,
           'constraint_violations': 0,
           'red_survivors': len(state.ids('red')), 'blue_survivors': len(state.ids('blue'))}
    return row, actual_first, trace


def evaluate(config, checkpoints, output, *, scales=(8,), episodes=20, wall_seconds=300., matched_static=True):
    """Checkpoints maps display names to paths, or to None for a rule baseline.

    The time budget is checked between complete paired units. A unit is one
    initial state evaluated by every requested policy; partial units never
    contribute to the comparison table and resume finishes missing members.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if episodes < 1:
        raise ValueError('episodes must be positive')
    policies, checkpoint_hashes = {}, {}
    for name, path in checkpoints.items():
        if path is None:
            baseline_config = dict(config, method=name)
            policies[name] = make_policy(baseline_config)
            checkpoint_hashes[name] = fingerprint(baseline_config)
        else:
            policies[name], payload = load_policy(path, config.get('device', 'cpu'))
            checkpoint_hashes[name] = sha256(path)
    reference = 'trained' if 'trained' in policies else next(iter(policies))
    if matched_static:
        policies['static_matched'] = policies[reference]
        checkpoint_hashes['static_matched'] = checkpoint_hashes[reference]
    identity = {'checkpoints': checkpoint_hashes, 'opponent': config['opponent'],
                'max_steps': config['max_steps'], 'command_interval': config['command_interval'],
                'scales': list(scales), 'evaluation_seed_base': int(config['seed']) + 3000000}
    identity_hash = fingerprint(identity)
    meta_path = output / 'evaluation_manifest.json'
    if meta_path.exists() and json.loads(meta_path.read_text(encoding='utf-8'))['identity_hash'] != identity_hash:
        raise ValueError('Evaluation model/task differs; use a new evaluation output directory')
    atomic_json(meta_path, {'identity_hash': identity_hash, **identity})
    path = output / 'evaluation.jsonl'
    rows = read_jsonl(path)
    completed = {(row['scale'], row['seed'], row['method']) for row in rows}
    started = time.perf_counter()
    for scale in scales:
        env = KnownOpponentEnv(red=scale, blue=scale, opponent=config['opponent'],
                               max_steps=config['max_steps'], command_interval=config['command_interval'],
                               device=config.get('executor_device', 'cpu'))
        try:
            for episode in range(episodes):
                seed = int(config['seed']) + 3000000 + int(scale) * 10000 + episode
                if all((scale, seed, method) in completed for method in policies):
                    continue
                if wall_seconds is not None and time.perf_counter() - started >= wall_seconds:
                    break
                matched_action = None
                if matched_static:
                    seed_everything(seed + 7000000)
                    first_state = env.reset(seed=seed)
                    with torch.no_grad():
                        matched_action = policies[reference].act(first_state).action
                for method, policy in policies.items():
                    if (scale, seed, method) in completed:
                        continue
                    row, _, trace = rollout_episode(env, policy, seed=seed,
                        first_action=matched_action if method == 'static_matched' else None,
                        static_after_first=method == 'static_matched', save_trace=episode == 0)
                    row.update(scale=int(scale), method=method)
                    append_jsonl(path, row)
                    rows.append(row)
                    completed.add((scale, seed, method))
                    if trace:
                        atomic_json(output / f'trace_{method}_{scale}.json', trace)
                summary = summarize(rows, list(policies))
                atomic_json(output / 'evaluation_summary.json', summary)
                print(f'[evaluation] scale={scale} paired={episode + 1}/{episodes}', flush=True)
        finally:
            env.close()
        if wall_seconds is not None and time.perf_counter() - started >= wall_seconds:
            break
    summary = summarize(rows, list(policies))
    summary.update(planned_pairs=len(scales) * episodes, elapsed_seconds=time.perf_counter() - started,
                   requested_seconds=wall_seconds, interpretation='Single training seed; episode uncertainty is not across-training-seed uncertainty.')
    atomic_json(output / 'evaluation_summary.json', summary)
    write_report(output, summary)
    return summary


def summarize(rows, methods):
    units = {}
    for row in rows:
        units.setdefault((row['scale'], row['seed']), {})[row['method']] = row
    complete = [unit for unit in units.values() if set(methods).issubset(unit)]
    table = []
    for scale in sorted({row['scale'] for row in rows}):
        for method in methods:
            selected = [unit[method] for unit in complete if unit[method]['scale'] == scale]
            if not selected:
                continue
            times = [value for row in selected for value in row['decision_seconds']]
            n, successes = len(selected), sum(row['success'] for row in selected)
            p, z = successes / n, 1.96
            center = (p + z*z/(2*n)) / (1+z*z/n)
            half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
            table.append({'scale': scale, 'method': method, 'episodes': n, 'successes': successes,
                          'success_rate': p, 'wilson_low': max(0., center-half), 'wilson_high': min(1., center+half),
                          'decision_median_ms': float(np.median(times))*1000,
                          'decision_p95_ms': float(np.quantile(times, .95))*1000,
                          **{key: float(np.mean([row[key] for row in selected]))
                             for key in ('release_ratio', 'task_change', 'team_change', 'red_survivors', 'physical_steps')},
                          'constraint_violations': sum(row['constraint_violations'] for row in selected)})
    return {'complete_pairs': len(complete), 'incomplete_pairs': len(units)-len(complete), 'table': table}


def write_report(output, summary=None):
    output = Path(output)
    summary = summary or json.loads((output / 'evaluation_summary.json').read_text(encoding='utf-8'))
    table = summary['table']
    if table:
        with (output / 'results.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    train_root = output.parent
    training = read_jsonl(train_root / 'training.jsonl')
    status_file = train_root / 'status.json'
    status = json.loads(status_file.read_text(encoding='utf-8')) if status_file.exists() else {}
    lines = ['# 已知对手动态分组：开发验证', '',
             '设定：公开冻结完整蓝方策略，固定红方下层，每局最多50步，两个任务点，组容量4。', '',
             '这是单训练种子的短训练结果。区间仅描述评估回合不确定性，不能证明跨训练种子的算法优势。', '',
             f"已完成训练交互：{status.get('physical_steps', '见训练日志')}；更新：{status.get('updates', '见训练日志')}。",
             f"完整配对测试：{summary['complete_pairs']}；未完成配对：{summary['incomplete_pairs']}。", '',
             '| 规模 | 方法 | 成功/回合 | 成功率 | 决策p95(ms) | 释放比例 | 任务变化 | 队友变化 |',
             '|---:|---|---:|---:|---:|---:|---:|---:|']
    for row in table:
        lines.append(f"| {row['scale']} | {row['method']} | {row['successes']}/{row['episodes']} | "
                     f"{row['success_rate']:.1%} | {row['decision_p95_ms']:.2f} | {row['release_ratio']:.3f} | "
                     f"{row['task_change']:.3f} | {row['team_change']:.3f} |")
    lines += ['', '主方法先选择需要调整的成员，再合法重建其分组；完整重构每次重新考虑全部存活成员。',
              '实际换组量不等于释放人数，死亡删除不计主动调整。static_matched沿用训练后策略的同一初始编组。', '',
              '训练总步数由实测吞吐、诊断与剩余时间决定；本报告没有预设已达到收敛。']
    (output / 'briefing.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(max(10, len(table) * .8), 3.6))
        valid = [row for row in training if row.get('recent_success_rate') is not None]
        axes[0].plot([r['physical_steps'] for r in valid], [r['recent_success_rate'] for r in valid], marker='.')
        axes[0].set(xlabel='Physical training steps', ylabel='Last 20 training episodes: success', ylim=(-.05, 1.05))
        labels = [f"{r['method']}\n{r['scale']}v{r['scale']}" for r in table]
        axes[1].bar(labels, [r['success_rate'] for r in table])
        axes[1].set(ylabel='Paired evaluation success', ylim=(0, 1))
        axes[1].tick_params(axis='x', labelsize=7)
        fig.tight_layout()
        fig.savefig(output / 'training_and_evaluation.png', dpi=160)
        plt.close(fig)
        traces = sorted(output.glob('trace_trained_*.json'))
        if traces:
            trace = json.loads(traces[0].read_text(encoding='utf-8'))
            ids = sorted({int(i) for row in trace for i in row['positions']})
            fig, axis = plt.subplots(figsize=(7, 5))
            for i in ids:
                xy = [row['positions'][str(i)] for row in trace if str(i) in row['positions']]
                axis.plot([p[0] for p in xy], [p[1] for p in xy], marker='.', label=f'Red {i}')
            for i, xyz in trace[0]['targets'].items():
                axis.scatter([xyz[0]], [xyz[1]], marker='*', s=140, color='black')
                axis.annotate(f'Target {i}', (xyz[0], xyz[1]))
            axis.set(xlabel='x', ylabel='y', title='One real evaluation episode (upper event snapshots)')
            axis.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(output / 'episode_trajectory.png', dpi=160)
            plt.close(fig)
    except ImportError:
        pass  # CSV/Markdown remain complete even without optional plotting.
