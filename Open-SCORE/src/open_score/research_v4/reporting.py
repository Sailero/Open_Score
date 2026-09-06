"""Auditable per-scale curves and six-route comparisons."""
from __future__ import annotations

from collections import defaultdict, deque
import csv
import json
import math
from pathlib import Path

import numpy as np

from .runner import atomic_json, read_json


def _rows(path):
    path = Path(path)
    if not path.exists():
        return []
    result = []
    lines = path.read_text(encoding='utf-8-sig').splitlines()
    for index, line in enumerate(lines):
        if line.strip():
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                if index != len(lines) - 1:
                    raise
    return result


def _finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def curve_data(output):
    output = Path(output)
    episodes = _rows(output / 'episodes.jsonl')
    updates = _rows(output / 'training.jsonl')
    validations = _rows(output / 'validation_history.jsonl')
    success, metrics, validation = defaultdict(list), defaultdict(list), []
    windows = defaultdict(lambda: deque(maxlen=100))
    for row in episodes:
        if 'success' not in row:
            continue
        scale = row.get('scale', row.get('red_count', 'unspecified'))
        phase = row.get('phase', 'train')
        key = f'{phase}, scale {scale}'
        windows[key].append(float(bool(row['success'])))
        x = row.get('physical_steps', row.get('total_steps', row.get('steps')))
        if _finite(x):
            success[key].append([x, float(np.mean(windows[key])), len(windows[key])])
    for index, row in enumerate(updates):
        x = row.get('physical_steps', row.get('total_steps', row.get('steps', index + 1)))
        if not _finite(x):
            continue
        for key, value in row.items():
            if _finite(value) and any(token in key.lower() for token in ('loss', 'entropy', 'kl', 'td_error', 'q_mean')):
                metrics[key].append([x, value])
    for row in validations:
        if row.get('complete') is False:
            continue
        x = row.get('physical_steps', row.get('total_steps', row.get('steps')))
        if not _finite(x):
            continue
        rate = row.get('success_rate', row.get('mean_success', row.get('mean_win_rate')))
        if _finite(rate):
            validation.append([x, rate])
        elif row.get('episodes', 0) > 0 and _finite(row.get('wins')):
            validation.append([x, row['wins'] / row['episodes']])
        elif isinstance(row.get('results'), list):
            items = row['results']
            wins = sum(item.get('wins', 0) for item in items)
            count = sum(item.get('episodes', 0) for item in items)
            if count:
                validation.append([x, wins / count])
    return {'schema': 'research-v4-curves-v1', 'native_success': dict(success),
            'validation': validation, 'metrics': dict(metrics),
            'notes': ['Rolling success separates scales and teacher/training phases.',
                      'Early windows use available episodes (up to 100).',
                      'Held-out test results never enter training or checkpoint-selection curves.']}


def plot_training(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    data = curve_data(output)
    atomic_json(output / 'training_curves.json', data)
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for name, points in data['native_success'].items():
        values = np.asarray(points)
        axes[0, 0].plot(values[:, 0], values[:, 1], label=name)
    axes[0, 0].set(title='Native success, rolling 100 per scale/phase', ylabel='Success rate', ylim=(0, 1))
    if data['native_success']:
        axes[0, 0].legend(fontsize=7)
    else:
        axes[0, 0].text(.5, .5, 'No learner episodes recorded', ha='center', transform=axes[0, 0].transAxes)
    if data['validation']:
        values = np.asarray(data['validation'])
        axes[0, 1].plot(values[:, 0], values[:, 1], marker='o', markersize=3)
    else:
        axes[0, 1].text(.5, .5, 'No validation points recorded', ha='center', transform=axes[0, 1].transAxes)
    axes[0, 1].set(title='Independent validation (selection only)', ylabel='Success rate', ylim=(0, 1))
    plotted = [0, 0]
    for name, points in data['metrics'].items():
        index = 0 if 'loss' in name else 1
        values = np.asarray(points)
        axes[1, index].plot(values[:, 0], values[:, 1], label=name, alpha=.8)
        plotted[index] += 1
    for index in range(2):
        axes[1, index].set_title('Optimization losses' if index == 0 else 'Policy/value diagnostics')
        if plotted[index]:
            axes[1, index].legend(fontsize=7)
        else:
            axes[1, index].text(.5, .5, 'No applicable recorded metrics', ha='center', transform=axes[1, index].transAxes)
    for ax in axes.flat:
        ax.set_xlabel('Recorded physical interaction steps (updates if unavailable)')
        ax.grid(alpha=.2)
    figure.suptitle(output.parent.name + ' / ' + output.name)
    figure.savefig(output / 'training_curves.png', dpi=140)
    plt.close(figure)
    return data


def paired_difference(rows, reference):
    lookup = {(row['scale'], row['seed']): row for row in reference}
    differences = [float(row['success']) - float(lookup[row['scale'], row['seed']]['success'])
                   for row in rows if (row['scale'], row['seed']) in lookup]
    if not differences:
        return {'pairs': 0, 'difference': None, 'bootstrap95': [None, None]}
    values = np.asarray(differences)
    rng = np.random.default_rng(40719)
    samples = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
    return {'pairs': len(values), 'difference': float(values.mean()),
            'bootstrap95': [float(value) for value in np.quantile(samples, [.025, .975])]}


def report(output):
    output = Path(output)
    groups, evaluations = [], []
    for path in sorted(output.glob('*/seed_*/evaluation_summary.json')):
        data = read_json(path)
        seed = int(path.parent.name.removeprefix('seed_'))
        groups.extend([{**group, 'training_seed': seed} for group in data['groups']])
        evaluations.append({'path': str(path), 'complete': data['complete'], 'route': data['route'], 'seed': seed})
    destination = output / 'comparison_results.csv'
    if groups:
        with destination.open('w', encoding='utf-8-sig', newline='') as stream:
            fields = ('route', 'training_seed', 'variant', 'scale', 'wins', 'episodes', 'success_rate',
                      'member_change', 'target_change', 'mean_group_size', 'decision_p95_seconds')
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(groups)
    pairs = []
    for entry in evaluations:
        if entry['route'] == 'rule':
            continue
        folder = Path(entry['path']).parent
        rule_rows = _rows(output / 'rule' / f'seed_{entry["seed"]}' / 'evaluation.jsonl')
        rows = _rows(folder / 'evaluation.jsonl')
        for variant in sorted({row['variant'] for row in rows}):
            pairs.append({'route': entry['route'], 'training_seed': entry['seed'], 'variant': variant,
                          **paired_difference([row for row in rows if row['variant'] == variant], rule_rows)})
    manifest = read_json(output / 'run_manifest.json', {})
    planned = set()
    for request in manifest.get('requests', []):
        if request.get('command') in ('run', 'evaluate'):
            planned.update((route, seed) for route in request.get('routes', []) + request.get('controls', [])
                           for seed in request.get('seeds', []))
    completed = {(entry['route'], entry['seed']) for entry in evaluations if entry['complete']}
    missing = [{'route': route, 'seed': seed} for route, seed in sorted(planned - completed)]
    result = {'schema': 'research-v4-comparison-v1', 'evaluations': evaluations, 'groups': groups,
              'missing_requested_evaluations': missing, 'complete': bool(planned) and not missing,
              'paired_vs_rule': pairs, 'all_available_evaluations_complete': bool(evaluations) and all(x['complete'] for x in evaluations)}
    atomic_json(output / 'comparison_summary.json', result)
    lines = ['# 共同规则底层：六条上层路线比较', '',
        '所有路线使用同一个规则执行器、已知对手及原生终局奖励。B1/B2/B3 使用共享准备阶段产出的支付模型；B1 没有伪造的策略训练曲线。', '',
        'best 仅由独立验证选出；latest 与 initialized 单列。单个训练种子只能用于方向筛选。尚未完成的评估不能当作零胜率。', '',
        '| 路线 | 种子 | 模型 | 规模 | 成功/局数 | 成功率 |', '|---|---:|---|---:|---:|---:|']
    for group in groups:
        rate = f'{100 * group["success_rate"]:.1f}%' if group['success_rate'] is not None else '未评估'
        lines.append(f'| {group["route"]} | {group["training_seed"]} | {group["variant"]} | {group["scale"]} | {group["wins"]}/{group["episodes"]} | {rate} |')
    if not groups:
        lines.append('| 尚无完成的逐局结果 | | | | | |')
    if missing:
        lines += ['', '尚未完成的请求：' + '、'.join(f'{entry["route"]} / {entry["seed"]}' for entry in missing)]
    lines += ['', '## 与规则上层的配对比较', '',
              '以相同训练种子对应的相同测试开局配对；下表区间为局级配对 bootstrap，不能替代跨训练种子不确定性。', '',
              '| 路线 | 种子 | 模型 | 配对局数 | 胜率差（百分点） | 95% 区间 |', '|---|---:|---|---:|---:|---|']
    for pair in pairs:
        delta = f'{100 * pair["difference"]:+.1f}' if pair['difference'] is not None else '未完成'
        interval = ', '.join(f'{100*x:+.1f}' for x in pair['bootstrap95']) if pair['pairs'] else '无配对数据'
        lines.append(f'| {pair["route"]} | {pair["training_seed"]} | {pair["variant"]} | {pair["pairs"]} | {delta} | {interval} |')
    lines += ['', '逐局数据位于各路线 seed 目录的 evaluation.jsonl；训练曲线位于 training_curves.png。',
              'S2 的训练与留出审计位于 shared；模型训练成本与实际执行交互成本须分别报告。']
    (output / 'comparison_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return result
