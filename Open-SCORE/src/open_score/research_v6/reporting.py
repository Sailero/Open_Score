"""Update the one V6 report and its figures directly from recorded outcomes."""
from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime
import json
from pathlib import Path
import sqlite3

import numpy as np

from . import METHODS, RL_METHODS, load_config


_READ_CACHE = {}


def _read(directory):
    path = Path(directory) / 'data.sqlite'
    if not path.exists():
        return {}, {}, []
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True) as connection:
        connection.execute('BEGIN')
        metadata = {k: json.loads(v) for k, v in connection.execute('SELECT key,value FROM metadata')}
        maximum = connection.execute('SELECT COALESCE(MAX(id),0) FROM records').fetchone()[0]
        key = str(path.resolve())
        identity = path.stat().st_ino
        revision = metadata.get('s2_revision', metadata.get('config', {}).get('s2', {}).get('revision'))
        cache = _READ_CACHE.get(key)
        if (cache is None or maximum < cache['last_id'] or identity != cache['identity']
                or revision != cache.get('revision')):
            cache = dict(last_id=0, identity=identity, revision=revision,
                         bins={}, recent=defaultdict(lambda: deque(maxlen=200)),
                         streams=defaultdict(list), memory=None, vram=None, episodes=0, gap=None)
            _READ_CACHE[key] = cache
        # Read each committed record once. Retain coarse history plus every one
        # of the newest 200 collection blocks per seed so live progress is visible.
        for record_id, stream, value in connection.execute(
                'SELECT id,stream,value FROM records WHERE id>? AND id<=? ORDER BY id',
                (cache['last_id'], maximum)):
            if stream not in ('training', 'train_episodes', 's2_fit', 'diagnostics'):
                continue
            row = json.loads(value)
            if stream == 'training':
                seed = int(row['seed'])
                cache['bins'][(seed, int(row['physical_steps'])//50000)] = (record_id, row)
                cache['recent'][seed].append((record_id, row))
                for field, target in (('resident_memory_bytes', 'memory'), ('peak_vram_bytes', 'vram')):
                    if row.get(field) is not None:
                        cache[target] = max(cache[target] or 0, float(row[field]))
            elif stream == 'train_episodes':
                if row.get('shaped_return') is not None and row.get('native_return') is not None:
                    cache['episodes'] += 1
                    gap = abs(row['shaped_return']-row['native_return'])
                    cache['gap'] = max(cache['gap'] or 0., gap)
            else:
                cache['streams'][stream].append(row)
        cache['last_id'] = maximum
        streams = defaultdict(list, cache['streams'])
        selected = dict(cache['bins'].values())
        for recent in cache['recent'].values():
            selected.update(recent)
        streams['training'] = [row for _, row in sorted(selected.items())]
        metadata['report_resource_peaks'] = dict(memory=cache['memory'], vram=cache['vram'])
        metadata['report_return_check'] = dict(episodes=cache['episodes'], maximum_gap=cache['gap'])
        episodes = []
        for method, split, checkpoint, family, value in connection.execute(
                'SELECT method,split,checkpoint,family,value FROM episodes'):
            row = json.loads(value)
            episodes.append(dict(row, method=method, split=split,
                                 family_id=row.get('family_id', family),
                                 checkpoint_step=row.get('checkpoint_step', checkpoint.split('_')[-1])))
    return metadata, streams, episodes


def _step(row):
    value = str(row.get('checkpoint_step', 'final'))
    return int(value) if value.isdigit() else value


def _final(method, rows, final_steps):
    target = final_steps if method in RL_METHODS else 'final'
    return [r for r in rows if r['method'] == method and r['split'] == 'test' and _step(r) == target]


def _cells(rows):
    result = defaultdict(dict)
    for row in rows:
        result[(int(row['seed']), row['scenario_id'])][row['family_id']] = int(row['success_native'])
    return {key: (sum(outcomes.values()), len(outcomes)) for key, outcomes in result.items()}


def _rate(cells, seed, scenarios, quota):
    values = [cells.get((int(seed), scenario)) for scenario in scenarios]
    if any(value is None or value[1] != quota for value in values):
        return None
    return float(np.mean([wins/count for wins, count in values]))


def _paired(first, second, final_rows, seeds, scenarios, quota):
    def slot(method, seed, scenario):
        # One rule outcome per opening; reuse it as the reference, never count
        # it as three independent executions or three fitted rule models.
        return (None if method == 'FrozenRule' else int(seed), scenario)
    indexed = {}
    for method in (first, second):
        values = defaultdict(dict)
        for row in final_rows[method]:
            values[slot(method, row['seed'], row['scenario_id'])][row['family_id']] = float(row['success_native'])
        indexed[method] = values
    strata = []
    for scenario in scenarios:
        families = [set(indexed[method][slot(method, seed, scenario)]) for method in (first, second) for seed in seeds]
        if any(len(family) != quota or family != families[0] for family in families):
            return dict(available=False, reason='相同开局族的完整配对结果尚未齐备（规则每个开局只评一次）')
        strata.append(np.asarray([np.mean([indexed[first][slot(first, seed, scenario)][family]
                                         -indexed[second][slot(second, seed, scenario)][family] for seed in seeds])
                                  for family in sorted(families[0])]))
    rng = np.random.default_rng(20260914)
    samples = np.mean([v[rng.integers(len(v), size=(2000, len(v)))].mean(axis=1) for v in strata], axis=0)
    return dict(available=True, difference=float(np.mean([v.mean() for v in strata])),
                interval=np.quantile(samples, [.025, .975]).tolist(),
                opening_families=sum(map(len, strata)), seeds=list(seeds))


def _number(value, digits=4):
    return '未记录' if value is None else f'{float(value):.{digits}f}'


def summarize(run_dir):
    """Overwrite one report/three figure paths; never invent unfinished results."""
    run = Path(run_dir)
    shared, common, _ = _read(run/'shared')
    config = shared.get('config') or load_config()
    seeds = [int(s) for s in config['seeds']['initializations']]
    scenes = config['scenarios']
    main = config['evaluation']['main_scenarios']
    auxiliary = config['evaluation']['auxiliary_scenarios']
    final_steps = int(config['training']['physical_steps_per_method_seed'])
    quota = int(config['evaluation']['test_per_scenario'])
    validation_quota = int(config['evaluation']['validation_per_scenario'])
    methods = ['FrozenRule', *METHODS.values()]
    data = {name: _read(run/name) for name in methods}
    rows = [r for _, _, records in data.values() for r in records]
    final = {method: _final(method, rows, final_steps) for method in methods}
    cells = {method: _cells(final[method]) for method in methods}
    main_rates, auxiliary_rates = {}, {}
    for method in methods:
        method_seeds = seeds if method != 'FrozenRule' else sorted({int(r['seed']) for r in final[method]})
        main_rates[method] = [_rate(cells[method], seed, main, quota) for seed in method_seeds]
        auxiliary_rates[method] = [_rate(cells[method], seed, auxiliary, quota) for seed in method_seeds]
    s2_revision = config['s2'].get('revision', 'binary_v1')
    def current_s2(row):
        return row.get('revision', 'binary_v1') == s2_revision
    fits = [r for r in common.get('s2_fit', []) if current_s2(r)]
    s2_results = {seed: result if current_s2(result) else {}
                  for seed in seeds for result in [shared.get(f's2_result/{seed}', {})]}
    training = {method: data[method][1].get('training', []) for method in RL_METHODS}
    # Before real measurements exist there is no result figure and no empty report.
    if not rows and not fits and not any(training.values()):
        return dict(status='waiting_for_recorded_results', report=None, evaluation_episodes=0)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = run/'figures'
    figures.mkdir(parents=True, exist_ok=True)
    image_paths = []
    if any(training.values()) or any(r['split'] == 'validation' for r in rows):
        fig, axes = plt.subplots(4, 2, figsize=(12, 13), constrained_layout=True)
        for axis, method in zip(axes[:2].flat, RL_METHODS):
            for index, seed in enumerate(seeds):
                color = f'C{index}'
                sparse = sorted((r for r in training[method] if int(r['seed']) == seed),
                                key=lambda r: r['physical_steps'])
                points, previous_wins, previous_episodes = [], 0, 0
                for row in sparse:
                    wins, episodes = row.get('native_success_count'), row.get('episodes')
                    if wins is not None and episodes is not None and episodes > previous_episodes:
                        points.append((row['physical_steps'], (wins-previous_wins)/(episodes-previous_episodes)))
                        previous_wins, previous_episodes = wins, episodes
                if points:
                    x, y = np.asarray(points).T
                    axis.plot(x/1e6, y*100,
                              '--', marker='.', alpha=.45, color=color, label=f'{seed} train mix')
                validation = [r for r in rows if r['method'] == method and r['split'] == 'validation'
                              and int(r['seed']) == seed]
                points = []
                for step in sorted({_step(r) for r in validation if isinstance(_step(r), int)}):
                    value = _rate(_cells([r for r in validation if _step(r) == step]), seed, main, validation_quota)
                    if value is not None:
                        points.append((step, value))
                if points:
                    axis.plot([p[0]/1e6 for p in points], [100*p[1] for p in points],
                              'o-', color=color, label=f'{seed} validation A-D')
            axis.set(title=method, xlabel='Physical steps (M)', ylabel='Native success (%)', ylim=(0, 100))
            axis.grid(alpha=.2)
            if axis.lines:
                axis.legend(fontsize=7)
        for axis, method in zip(axes[2:].flat, RL_METHODS):
            fields = ('actor_loss', 'value_loss') if method == 'MAPPO_Intent' else ('q_loss', 'proposal_loss')
            for index, seed in enumerate(seeds):
                entries = sorted((r for r in training[method] if int(r['seed']) == seed),
                                 key=lambda r: r['physical_steps'])
                for field, style in zip(fields, ('-', '--')):
                    valid = [r for r in entries if r.get(field) is not None]
                    if valid:
                        axis.plot([r['physical_steps']/1e6 for r in valid], [r[field] for r in valid],
                                  style, marker='.', color=f'C{index}', label=f'{seed} {field}')
            axis.set(title=method+' optimization', xlabel='Physical steps (M)', ylabel='Loss')
            axis.grid(alpha=.2)
            if axis.lines:
                axis.legend(fontsize=7)
        has_points = any(axis.lines for axis in axes.flat)
        if has_points:
            fig.savefig(figures/'learning.png', dpi=160)
        plt.close(fig)
        if has_points:
            image_paths.append(('learning.png', '上半部分：训练混合胜率（虚线，相邻显示点之间的成功数/回合数差；每种子最近200个采集块逐块显示，更早历史约50k步一个点）与主场景验证胜率（实线），两者分布不同，仅完整验证点计入实线。下半部分：实际Q/proposal或actor/critic训练损失。数据库保留全部原始记录，损失下降不等于任务有效。'))
    chart_methods = [m for m in methods if any(v is not None for v in main_rates[m]+auxiliary_rates[m])]
    if chart_methods:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        for axis, category, label in zip(axes, (main_rates, auxiliary_rates), ('Main A-D', 'Auxiliary E')):
            for position, method in enumerate(chart_methods):
                values = [v for v in category[method] if v is not None]
                if values:
                    axis.bar(position, 100*np.mean(values), color='C0', alpha=.65)
                    axis.scatter([position]*len(values), np.asarray(values)*100, color='black', s=15)
                if method in RL_METHODS:
                    earlier = _cells([r for r in rows if r['method'] == method and r['split'] == 'test' and _step(r) == 2_000_000])
                    previous = [_rate(earlier, seed, main if category is main_rates else auxiliary, quota) for seed in seeds]
                    previous = [v for v in previous if v is not None]
                    if previous:
                        axis.scatter([position], [100*np.mean(previous)], marker='D', color='C3', s=38)
            axis.set_xticks(range(len(chart_methods)), chart_methods, rotation=35, ha='right')
            axis.set(title=label+' (bar: final; diamond: 2M)', ylabel='Native success (%)', ylim=(0, 100))
            if category['FrozenRule'] and category['FrozenRule'][0] is not None:
                axis.axhline(100*category['FrozenRule'][0], color='black', linestyle=':', label='FrozenRule')
                axis.legend(fontsize=8)
            axis.grid(axis='y', alpha=.2)
        fig.savefig(figures/'results.png', dpi=160)
        plt.close(fig)
        image_paths.append(('results.png', '最终模型及同次训练2M模型。黑点为已完成种子；未完成种子缺席，不能据部分结果宣称三种子优势。'))
    if fits:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for seed in seeds:
            entries = sorted([r for r in fits if int(r['seed']) == seed], key=lambda r: r['epoch'])
            for axis, field, label in zip(axes.flat,
                    ('train_loss', 'validation_brier', 'validation_time_mae', 'validation_nontie_ranking_accuracy'),
                    ('Joint outcome-time NLL' if s2_revision == 'joint_v2' else 'Training loss',
                     'Validation success Brier', 'Validation remaining-time MAE (steps)',
                     'Validation non-tie ranking accuracy')):
                entries_field = [r for r in entries if r.get(field) is not None]
                if entries_field:
                    axis.plot([r['epoch'] for r in entries_field], [r[field] for r in entries_field], label=str(seed))
                axis.set(xlabel='Epoch', ylabel=label)
                axis.grid(alpha=.2)
                if axis.lines:
                    axis.legend(fontsize=8)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axhline(.5, color='grey', linestyle=':', linewidth=1)
        fig.savefig(figures/'s2.png', dpi=160)
        plt.close(fig)
        image_paths.append(('s2.png', f'当前S2版本{s2_revision}：共享数据上三个初始化的训练损失、胜率Brier、剩余时间MAE和非平局候选排序；只显示当前版本实际记录，旧二分类曲线不与重训拼接。损失或时间误差下降不等于在线方法有效。'))
    elif s2_revision == 'joint_v2':
        # Replace the old revision's image immediately; fitting has not begun.
        sampled = shared.get('s2_data_result', {}) or shared.get('progress/S2_data', {})
        if current_s2(sampled):
            dataset = config['s2']['data']
            split_names = ('train', 'validation', 'test')
            quotas = [len(dataset['single_target']['count_pairs'])*dataset['single_target']['split'][split]
                      + len(scenes)*sum(controller['split'][split] for controller in dataset['multi_target']['controllers'])
                      for split in split_names]
            counts = sampled.get('completed_by_split', sampled.get('families_by_split', {}))
            values = [counts.get(split, 0) for split in split_names]
            fig, axis = plt.subplots(figsize=(9, 4), constrained_layout=True)
            axis.barh(split_names, quotas, color='lightgrey', label='Fixed quota')
            bars = axis.barh(split_names, values, label='Collected families')
            axis.bar_label(bars, labels=[f'{n:,} / {q:,}' for n, q in zip(values, quotas)], padding=4)
            axis.set(xlabel='Independent mother episode families', title='S2 joint_v2: collecting data; fitting not started')
            axis.legend()
            fig.savefig(figures/'s2.png', dpi=160)
            plt.close(fig)
            image_paths.append(('s2.png', '新S2独立母回合采集进度；采集完成后自动拟合，届时此图原位更新为联合损失、Brier、时间MAE和排序曲线。'))

    if not image_paths:
        return dict(status='waiting_for_plottable_results', report=None, evaluation_episodes=len(rows))
    complete = all(len(main_rates[m]) == 3 and all(v is not None for v in main_rates[m]+auxiliary_rates[m])
                   for m in METHODS.values()) and bool(main_rates['FrozenRule']) and all(
                       v is not None for v in main_rates['FrozenRule']+auxiliary_rates['FrozenRule'])
    core = shared.get('native_core', {})
    text = ['# v6 动态分组实验报告', '', f'更新时间：{datetime.now().astimezone().isoformat(timespec="seconds")}。', '',
            f'当前状态：{"六方法最终正式评价齐备" if complete else "实验尚未完成；以下只汇总已落盘记录"}。正式评价记录累计 {sum(r["split"] == "test" for r in rows):,} 局执行（含2M/最终及共享开局重复执行，不等于独立开局数）。', '',
            '## 研究问题与固定设计', '',
            '在已知 reactive 对手、共享 rule_group_v1 底层和原生物理世界中，比较目标归属与同目标分组。每局最多50物理步，5步及非终止伤亡重新决策；不设组大小上限、后备配额或强制部署。', '',
            f'本轮使用独立新版had_env库。运行记录CORE_VERSION：{core.get("core_version", "尚未记录")}；PHYSICS_PROTOCOL：{core.get("physics_protocol", "尚未记录")}；来源：{core.get("package_path", "尚未记录")}。', '',
            '四个RL方法各训练三种子，每种子五场景合计5M物理步；2M和5M是同次训练的检查点。所有RL采用λ=0.5、γ=1蓝方生命损耗潜势，真实终局潜势归零；S2始终学习原生终局标签。', '',
            '| 场景 | 红/蓝 | 目标数 | 汇总范围 |', '|---|---:|---:|---|']
    text += [f'| {s["id"]} | {s["red"]}/{s["blue"]} | {s["targets"]} | {s["role"]} |' for s in scenes]
    text += ['', '## 总体效果', '', '| 方法 | 主场景均值±种子标准差 | 辅助E | 完整种子 |', '|---|---:|---:|---:|']
    for method in methods:
        rates, extra = [v for v in main_rates[method] if v is not None], [v for v in auxiliary_rates[method] if v is not None]
        result = f'{100*np.mean(rates):.2f}% ± {100*np.std(rates, ddof=1):.2f}' if len(rates)>1 else (f'{100*rates[0]:.2f}%' if rates else '待完成')
        text.append(f'| {method} | {result} | {100*np.mean(extra):.2f}% | {len(rates)}/{1 if method == "FrozenRule" else 3} |' if extra else f'| {method} | {result} | 待完成 | {len(rates)}/{1 if method == "FrozenRule" else 3} |')
    for path, caption in image_paths:
        text += ['', f'![{caption}](figures/{path})', '', caption]
    comparisons = {}
    for title, key in (('首先比较：六方法相对规则是否有效', 'primary_comparisons'),
                       ('其次比较：三项机制归因', 'mechanism_comparisons')):
        text += ['', f'## {title}', '', '仅使用A–D主场景；先在每个开局族内对三个模型求差值均值，再按场景分层重采样开局族2000次。规则每个开局只运行一次，复用同一结果作参照。区间条件于已训练模型，不将模型重复执行当作独立开局；各区间为逐比较区间。', '', '| 比较 | 差值（百分点） | 配对95%区间 |', '|---|---:|---:|']
        for first, second in config['evaluation'].get(key, []):
            value = _paired(first, second, final, seeds, main, quota)
            comparisons[f'{first}-{second}'] = value
            text.append(f'| {first}−{second} | {100*value["difference"]:.2f} | [{100*value["interval"][0]:.2f}, {100*value["interval"][1]:.2f}] |' if value['available'] else f'| {first}−{second} | 待完成 | {value["reason"]} |')
    for method in methods:
        text += ['', f'## {method}', '']
        if method != 'FrozenRule':
            text += ['本方法首先与FrozenRule比较（A–D，单个已完成种子的条件区间）：', '',
                     '| 模型种子 | 相对规则差值（百分点） | 配对95%区间 |', '|---|---:|---:|']
            for seed in seeds:
                gain = _paired(method, 'FrozenRule', final, [seed], main, quota)
                text.append(f'| {seed} | {100*gain["difference"]:.2f} | [{100*gain["interval"][0]:.2f}, {100*gain["interval"][1]:.2f}] |' if gain['available'] else f'| {seed} | 待完成 | 相同开局结果尚未齐备 |')
            text.append('')
        if method.startswith('ALMA_'):
            text += [f'配置：AQL，{config["aql"]["candidates"]["current"]}候选、每批{config["aql"]["batch_events"]}事件、每{config["aql"]["events_per_update"]}新增事件一次Q/proposal更新；RMSprop学习率{config["aql"]["optimizer"]["learning_rate"]}，回放{config["aql"]["replay"]["capacity_complete_episodes"]}完整回合。', '']
        elif method == 'MAPPO_Intent':
            text += [f'配置：每批{config["mappo"]["episodes_per_batch"]}同版本完整回合、{config["mappo"]["epochs"]}epoch、{config["mappo"]["n_step_events"]}事件n-step、逐成员clip={config["mappo"]["clip"]}；Adam学习率{config["mappo"]["optimizer"]["learning_rate"]}。', '']
        elif method.startswith('BLOTTO_'):
            text += ['配置：完整枚举人数和后备配置，按距离确定身份，冻结局部S2的对数概率求和；'+('固定目标归属后评分64个不同完整分区。' if method == 'BLOTTO_Group' else '每个目标形成一个大组。'), '']
        else:
            text += ['冻结规则上层参照，仅在共同正式开局上评价一次，不进行学习。', '']
        text += ['| 种子 | 检查点 | 场景 | 成功/局数 | 胜率 |', '|---|---|---|---:|---:|']
        evaluated = [r for r in rows if r['method'] == method and r['split'] == 'test']
        for step in sorted({_step(r) for r in evaluated}, key=str):
            for (seed, scenario), (wins, count) in sorted(_cells([r for r in evaluated if _step(r)==step]).items()):
                text.append(f'| {seed} | {step} | {scenario} | {wins}/{count} | {100*wins/count:.2f}% |')
        if not evaluated:
            text.append('| — | — | — | 尚无正式评价记录 | — |')
        last = {int(r['seed']): r for r in data[method][1].get('training', [])}
        if last:
            text += ['', '实际训练进度：'+ '；'.join(f'{s}: {r["physical_steps"]:,}物理步，{r.get("events", "未记录")}事件，{r.get("episodes", "未记录")}回合' for s,r in sorted(last.items()))+'。']
            text += ['', '| 种子 | 成功数/首次成功步 | 当前replay正例 | 连续无原生奖励回合 | 优化器step | TD RMSE | 最新速度(物理步/s) |', '|---|---|---:|---:|---:|---:|---:|']
            for seed, row in sorted(last.items()):
                text.append(f'| {seed} | {row.get("native_success_count", "—")}/{row.get("first_success_physical_step", "尚无成功")} | {row.get("replay_positive_episodes", "不适用或未记录")} | {row.get("no_native_reward_window", "—")} | {row.get("optimizer_steps", "—")} | {_number(row.get("td_rmse"))} | {_number(row.get("physical_steps_per_second"), 2)} |')
            text += ['', '最后采集块学习指标：'+ '；'.join(f'{s}: '+', '.join(f'{k}={_number(r[k])}' for k in ('q_loss','proposal_loss','actor_loss','critic_loss','s2_rows','internal_tokens','candidate_generated','candidate_scored') if r.get(k) is not None) for s,r in sorted(last.items()))+'。']
            peaks = data[method][0]['report_resource_peaks']
            text += ['', f'完整训练记录中的进程内存峰值：{_number(peaks["memory"]/2**30 if peaks["memory"] is not None else None, 2)} GiB；显存峰值：{_number(peaks["vram"]/2**30 if peaks["vram"] is not None else None, 2)} GiB（不是全机各并发进程同步总峰值）。']
        if final[method]:
            latencies = [r['decision_latency_ms_mean'] for r in final[method] if r.get('decision_latency_ms_mean') is not None]
            cost = sum(r.get('physical_steps', 0) for r in final[method])
            wall = sum(r.get('wall_time_s', 0) for r in final[method])
            text += ['', f'最终评价成本：{cost:,}物理步、累计回合耗时{wall:.1f}秒；每回合平均决策时延再取均值：{_number(np.mean(latencies) if latencies else None, 2)}毫秒。并发累计耗时不等于墙钟完成时间。']
            details = []
            for field in ('sustained_all_reserve','reserve_exposure','reassignment_count','partition_changes_at_fixed_assignment','nonterminal_positive_shaping','terminal_compensation','s2_scoring_rows','candidates_scored'):
                values = [r[field] for r in final[method] if r.get(field) is not None]
                if values:
                    details.append(f'{field}={np.mean(values):.4g}')
            failures = [r['blue_health_loss'] for r in final[method] if not r['success_native'] and r.get('blue_health_loss') is not None]
            text += ['', '最终评价行为（逐局均值）：'+'；'.join(details)+f'；失败局蓝方生命损耗={_number(np.mean(failures) if failures else None)}。']
            reasons = defaultdict(int)
            for row in final[method]:
                reasons[row.get('termination_reason', '未记录')] += 1
            text += ['', '终止原因局数：'+'；'.join(f'{key}={value}' for key,value in sorted(reasons.items()))+'。']
        check = data[method][0].get('report_return_check', {})
        if check.get('episodes'):
            text += ['', f'已记录完整训练回合原生/塑形总回报最大绝对差：{check["maximum_gap"]:.3g}（{check["episodes"]:,}局）；正塑形反馈不作为原生成功计数。']
    text += ['', '## S2能力评价与额外成本', '', f'当前评价器版本：`{s2_revision}`。']
    if s2_revision == 'joint_v2':
        text += ['', '保留局部物理状态和候选分组关系，联合预测原生胜负与剩余结束物理步数；训练使用结局与时间联合负对数似然。G1/G2仍以成功概率边缘分布评分，搜索、人数枚举和目标归属逻辑不变，时间不作为额外惩罚或奖励。每个ALMA_S2种子待其新评价器完成并冻结后重新训练，旧训练记录不与本次曲线拼接。', '',
                 '旧二分类评价器与受其影响的训练记录、权重保留为binary_v1归档；以下训练曲线、留出指标及在线结果只计当前版本。增加独立局面和恢复时间监督是待验证的改进，不能提前承诺排序或在线胜率提升。']
    dataset = config['s2']['data']
    split_quotas = {split: len(dataset['single_target']['count_pairs'])*int(count)
                    + len(scenes)*sum(int(controller['split'][split])
                                      for controller in dataset['multi_target']['controllers'])
                    for split, count in dataset['single_target']['split'].items()}
    progress = shared.get('progress/S2_data', {})
    if not current_s2(progress):
        progress = {}
    collection = shared.get('s2_data_result', {})
    if not current_s2(collection):
        collection = {}
    collected = collection or progress
    text += ['', f'固定采样配额：{sum(split_quotas.values()):,}个独立母回合族（训练{split_quotas.get("train", 0):,}、验证{split_quotas.get("validation", 0):,}、测试{split_quotas.get("test", 0):,}）；每族最多{dataset["states_per_family"]}个状态，每状态最多{dataset["candidates_per_state"]}个不同候选。训练/验证/测试的每候选配对分支数为'+ '/'.join(str(dataset['branches'][split]) for split in ('train', 'validation', 'test'))+'。同一母回合族不跨划分，全部失败和全部成功状态均保留。']
    completed_families = collected.get('families', collected.get('completed', 0))
    text += ['', f'已落盘采样：{completed_families:,}/{sum(split_quotas.values()):,}母回合族；状态{collected.get("states", "未记录")}，候选{collected.get("candidates", "未记录")}，非平局状态{collected.get("non_tie_states", "未记录")}；母回合{collected.get("mother_physical_steps", "未记录")}物理步、标签续行{collected.get("simulation_physical_steps", "未记录")}模拟物理步。']
    if progress and not collection.get('complete'):
        remaining = progress.get('remaining_seconds')
        text += ['', f'最近记录采样速度：{_number(progress.get("families_per_second"), 3)}族/秒；采样剩余时间估计：{_number(remaining/3600 if remaining is not None else None, 2)}小时，仅含数据采集，不含后续三个评价器及ALMA_S2训练。该估计随并发和局面长度变化。']
    completed_by_split = collected.get('completed_by_split', collected.get('families_by_split', {}))
    if completed_by_split:
        text += ['', '| 数据划分 | 已完成母回合族 | 固定配额 |', '|---|---:|---:|']
        text += [f'| {split} | {completed_by_split.get(split, 0)} | {quota} |'
                 for split, quota in split_quotas.items()]
    text += ['', f'三个S2初始化共用同一份采样数据，不能重复计为三份独立数据。每个模型保留{config["s2"]["epochs"]}轮配额；下表阶段来自最近一次落盘记录，不代替进程存活检查。', '',
             '| 初始化 | 最近记录阶段 | 已完成epoch/配额 | 验证Brier | 验证时间MAE(步) | 验证非平局排序 |', '|---|---|---:|---:|---:|---:|']
    for seed in seeds:
        entries = sorted([r for r in fits if int(r['seed']) == seed], key=lambda r: r['epoch'])
        latest = entries[-1] if entries else {}
        stage = '已完成' if s2_results[seed].get('complete') else ('拟合中或等待恢复' if latest else '等待采样或调度')
        text.append(f'| {seed} | {stage} | {latest.get("epoch", 0)}/{config["s2"]["epochs"]} | '
                    + ' | '.join(_number(latest.get(key)) for key in
                                 ('validation_brier', 'validation_time_mae', 'validation_nontie_ranking_accuracy'))+' |')
    text += ['', '| 初始化 | 选中epoch | 测试Brier | ECE | 时间MAE(步) | 非平局排序 | 平局率 | 选择损失 | 全零/非平局/总状态 |', '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for seed in seeds:
        result = s2_results[seed]
        text.append(f'| {seed} | {result.get("best_epoch", "待完成")} | '+ ' | '.join(_number(result.get(key)) for key in ('test_brier','test_ece','test_time_mae','test_nontie_ranking_accuracy','test_tie_fraction','test_selection_loss'))+f' | {result.get("test_all_zero_states", "—")}/{result.get("test_non_tie_states", "—")}/{result.get("test_states", "—")} |')
    if any(result.get('test_time_mae') is not None for result in s2_results.values()):
        text += ['', '| 初始化 | 成功条件时间MAE(步) | 失败条件时间MAE(步) | 测试联合NLL |', '|---|---:|---:|---:|']
        for seed, result in s2_results.items():
            text.append(f'| {seed} | '+' | '.join(_number(result.get(key)) for key in
                        ('test_success_time_mae', 'test_failure_time_mae', 'test_joint_nll'))+' |')
        text += ['', '时间误差按实际终局标签评价；成功和失败分列，守满50步成功与快速失败不会被当作同一种结果。模型选择仍依据验证胜率Brier，测试集不用于选模型。']
    scale_rows = [(seed, scale, value) for seed in seeds for scale, value in s2_results[seed].get('test_by_scale', {}).items()]
    if scale_rows:
        text += ['', '| S2初始化 | 局部规模 | 实际留出指标 |', '|---|---|---|']
        text += [f'| {seed} | {scale} | '+json.dumps(value, ensure_ascii=False, separators=(',', ':'))+' |' for seed,scale,value in scale_rows]
    diagnostics = common.get('diagnostics', [])
    text += ['', '## 有限原因分析', '']
    if diagnostics:
        better, partition, rankings = [], [], []
        invalid_assignment = 0
        for row in diagnostics:
            values = np.mean(np.asarray(row['outcomes'], dtype=float), axis=1)
            selected = row['methods']
            if 'FrozenRule' in selected:
                better.append(float(values.max()-values[selected['FrozenRule']]))
            if all(m in selected for m in ('BLOTTO_Count','BLOTTO_Group')):
                actions = [row['candidates'][selected[m]] for m in ('BLOTTO_Count','BLOTTO_Group')]
                allocations = [{**{int(i): g['target'] for g in a['groups'] for i in g['members']},
                                **{int(i): None for i in a.get('reserve', [])}} for a in actions]
                if allocations[0] == allocations[1]:
                    partition.append(float(values[selected['BLOTTO_Group']]-values[selected['BLOTTO_Count']]))
                else:
                    invalid_assignment += 1
            scores = row.get('s2_scores')
            if scores is not None and len(scores) == len(values):
                pair_scores = []
                for i in range(len(values)):
                    for j in range(i):
                        if values[i] != values[j] and scores[i] is not None and scores[j] is not None:
                            product = (values[i]-values[j])*(scores[i]-scores[j])
                            pair_scores.append(1. if product>0 else .5 if product==0 else 0.)
                if pair_scores:
                    rankings.append(float(np.mean(pair_scores)))
        text += [f'已记录{len(diagnostics)}个固定状态，使用第一初始化模型；每候选执行一个事件后按共同规则至真正终局，同状态使用配对分支。成本{sum(r.get("simulation_physical_steps", 0) for r in diagnostics):,}模拟物理步，仅用于事后诊断。', '',
                 f'候选中经验最优相对规则平均增益：{_number(np.mean(better) if better else None)}；显示改善的状态{sum(v>0 for v in better)}/{len(better)}。经验最优仍受有限分支的选优偏差影响，不是已证明可实现增益。', '',
                 f'同目标归属G2−G1平均续行增益：{_number(np.mean(partition) if partition else None)}；正/负/平局状态={sum(v>0 for v in partition)}/{sum(v<0 for v in partition)}/{sum(v==0 for v in partition)}；发现归属不一致而排除{invalid_assignment}状态。', '',
                 f'S2代理对共享世界非平局候选的排序准确率（先状态内平均）：{_number(np.mean(rankings) if rankings else None)}，有可比非平局的状态{len(rankings)}；预测平局记0.5。该分析只针对这批状态与第一初始化，不扩大为三种子机制证据。']
    else:
        text += ['尚无已完成的固定状态诊断；候选是否存在改善、固定归属分组收益及S2向共享世界迁移尚不能下结论。']
    text += ['', '## 复现边界与结论范围', '',
             'BLOTTO_Count是人数枚举与距离身份分配；BLOTTO_Group固定相同目标归属后只搜索分区。ALMA_Alloc是固定底层上的AQL分配适配；MAPPO_Intent使用同时成员选择与旧意图。ALMA_Group、ALMA_S2是项目扩展，后者仅向全局Q提供冻结局部S2能力。', '',
             '六法在线选动作均不调用真实模拟器反复试选。真实终局模拟仅用于共享S2离线标签和事先限定的原因分析；S2删除跨目标物理耦合，输出局部续行概率，不能当作完整世界无误差模型。', '',
             '局部模型校准、训练损失或正奖励改善不能单独证明方法有效；主结论依据主场景原生胜率、配对差值与三个训练种子的一致性。辅助E单列，不抬高主胜率。尚未记录的评价与原因分析均保持未完成；不能把仍存活目标在全局提前失败时标为局部成功。', '',
             '网络使用128维、4头、两层实体编码；完整参数、种子及预算见[config.json](config.json)，实际运行资源记录于共享数据库runtime。2M/5M按完整采样块边界保存，逐局checkpoint_physical_steps及权重记录实际训练步数。', '',
             '数据与模型按方法保存在本目录；共享数据为[shared/data.sqlite](shared/data.sqlite)。本文件为本轮唯一正式报告，图表更新既有路径。', '']
    path = run/'实验报告.md'
    path.write_text('\n'.join(text), encoding='utf-8')
    return dict(status='final_evaluations_complete' if complete else 'in_progress', report=str(path),
                evaluation_episodes=len(rows), final_main_rates=main_rates, comparisons=comparisons)
