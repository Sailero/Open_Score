"""One version report and two figures per method, read directly from SQLite."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import io
import json
import math
import os
from pathlib import Path
import re

from .storage import Store, method_dir


TASKS = {
    'T1': ('真实终局 rollout', ['T1_rollout'], '真实终局模拟能否发现并选择更好的编组？'),
    'T2': ('MCTS-DPW', ['T2_mcts_dpw'], '多事件树搜索能否改善决策？'),
    'T3': ('候选 PPO 与 AR-PPO', ['t3_candidate', 't3_autoregressive'], '直接策略能否学到有效的候选选择或完整编组？'),
    'T4': ('ExIt 风格策略迭代', ['t4_exit'], '模拟教师的收益能否迁移到学生？'),
    'T5': ('BRIDGE 分组 DQN', ['t5_bridge_grouping'], '固定目标与身份归属后，学习合并是否有效？'),
    'T6': ('配对价值学习', ['t6_bce', 't6_adv', 't6_adv_gated'], '配对优势监督和规则门控是否改善方案选择？'),
}
CONTROLS = ['rule', 'grand', 'singleton', 'frozen_b1', 'frozen_b3']
MAIN = dict.fromkeys(['T1_rollout', 'T2_mcts_dpw', 't3_candidate', 't3_autoregressive', 't4_exit'], 'latest')
MAIN.update(t5_bridge_grouping='final_construction', t6_bce='final_epoch', t6_adv='final_epoch', t6_adv_gated='final_epoch')
MAIN.update(dict.fromkeys(CONTROLS, 'frozen'))
NAMES = dict(T1_rollout='真实 rollout', T2_mcts_dpw='MCTS-DPW', t3_candidate='候选 PPO',
    t3_autoregressive='AR-PPO', t4_exit='ExIt 学生', t5_bridge_grouping='BRIDGE 分组 DQN',
    t6_bce='BCE 价值', t6_adv='ADV 价值', t6_adv_gated='ADV-gated', rule='规则上层',
    grand='每目标大组', singleton='同归属单体组', frozen_b1='冻结 v4 B1', frozen_b3='冻结 v4 B3')
METHOD_TASK = {m: t for t, (_, arms, _) in TASKS.items() for m in arms} | dict.fromkeys(CONTROLS, 'T1')
RATIOS = {.5: '较易', .75: '中等', 1.: '等人数'}


def atomic_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_bytes(content if isinstance(content, bytes) else content.encode('utf-8'))
    os.replace(temporary, path)


def finite(value):
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def mean(values):
    values = [float(v) for v in values if finite(v)]
    return sum(values)/len(values) if values else None


def pct(value):
    return '尚无数据' if not finite(value) else f'{float(value):.2%}'


def number(value):
    return '未记录' if not finite(value) else f'{float(value):.4f}'


def success(row):
    return int(bool(row.get('success_native', row.get('success'))))


def valid_rows(rows):
    unique = {}
    for row in rows:
        if (row.get('family_id') and row.get('success_native', row.get('success')) in (False, True, 0, 1)
                and row.get('execution_status', 'ok') not in ('error', 'failed_execution')):
            unique[row['family_id']] = row
    return list(unique.values())


def wilson(k, n):
    if not n:
        return [None, None]
    z, p = 1.959963984540054, k/n
    mid = (p+z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [min(p, max(0., mid-half)), max(p, min(1., mid+half))]


def paired_interval(rows, baseline):
    from scipy.stats import beta
    reference = {r['family_id']: r for r in baseline}
    pairs = [(success(r), success(reference[r['family_id']])) for r in rows if r['family_id'] in reference]
    n = len(pairs)
    counts = {f'n{a}{b}': sum(x == a and y == b for x, y in pairs) for a in (0, 1) for b in (0, 1)}
    if not n:
        return dict(pairs=0, difference=None, interval95=[None, None], **counts)
    def bounds(k):
        return [0. if k == 0 else float(beta.ppf(.0125, k, n-k+1)),
                1. if k == n else float(beta.ppf(.9875, k+1, n-k))]
    positive, negative = bounds(counts['n10']), bounds(counts['n01'])
    return dict(pairs=n, difference=(counts['n10']-counts['n01'])/n,
        interval95=[positive[0]-negative[1], positive[1]-negative[0]], **counts)


def aggregate(rows, cells, per_cell, baseline=()):
    groups = []
    for red, blue in cells:
        subset = [r for r in rows if (r.get('red_count'), r.get('blue_count')) == (red, blue)]
        ref = [r for r in baseline if (r.get('red_count'), r.get('blue_count')) == (red, blue)]
        k, n = sum(map(success, subset)), len(subset)
        groups.append(dict(red_count=red, blue_count=blue, wins=k, episodes=n,
            success_rate=k/n if n else None, wilson95=wilson(k, n), complete=n == per_cell,
            paired_vs_rule=paired_interval(subset, ref) if ref else None))
    difficulties = []
    for ratio, name in RATIOS.items():
        selected = [g for g in groups if g['blue_count']/g['red_count'] == ratio]
        k, n = sum(g['wins'] for g in selected), sum(g['episodes'] for g in selected)
        difficulties.append(dict(name=name, ratio=ratio, wins=k, episodes=n,
            success_rate=k/n if n else None, complete=bool(selected) and all(g['complete'] for g in selected)))
    k, n = sum(map(success, rows)), len(rows)
    return dict(wins=k, episodes=n, expected_episodes=len(cells)*per_cell,
        success_rate=k/n if n else None, wilson95=wilson(k, n),
        macro_success_rate=mean(g['success_rate'] for g in groups), cells=groups, difficulties=difficulties,
        complete=bool(cells) and n == len(cells)*per_cell and all(g['complete'] for g in groups))


def checkpoint_axis(checkpoint):
    for prefix, axis in [('step_', 'Physical training steps'), ('construction_', 'Construction episodes'), ('round_', 'ExIt rounds')]:
        if re.fullmatch(prefix+r'\d+', checkpoint):
            return axis, int(checkpoint[len(prefix):])
    return 'Checkpoint', 0


def collect(run_dir):
    run = Path(run_dir).resolve()
    config = json.loads((run/'config.json').read_text(encoding='utf-8-sig'))
    cells = config['cells']
    stores = {p: Store(p) for p in {method_dir(run, m) for m in MAIN}}
    shared = Store(run/'shared')
    evaluations = {}
    for store in stores.values():
        for method, split, checkpoint in store.evaluation_keys():
            evaluations[method, split, checkpoint] = valid_rows(store.episodes(method, split, checkpoint))
    baseline = evaluations.get(('rule', 'test', 'frozen'), [])
    baseline_complete = aggregate(baseline, cells, config['eval_per_cell'])['complete']
    gate = stores[method_dir(run, 't6_adv')].get('gate_selection', {})
    chosen = next((r for r in gate.get('thresholds', []) if r['threshold'] == gate.get('chosen_threshold')), {})
    gate_method = chosen.get('summary', {}).get('method_id')
    if gate_method:
        evaluations['t6_adv_gated', 'validation', 'selected_gate'] = evaluations.get((gate_method, 'validation', 'final_epoch'), [])
    methods = []
    for method, checkpoint in MAIN.items():
        store = stores[method_dir(run, method)]
        rows = evaluations.get((method, 'test', checkpoint), [])
        stats = aggregate(rows, cells, config['eval_per_cell'], baseline)
        paired = paired_interval(rows, baseline)
        effect = '等待完整配对结果'
        if method == 'rule':
            effect = '公共参照'
        elif stats['complete'] and baseline_complete and paired['pairs'] == stats['expected_episodes']:
            effect = '本次单种子有正收益证据' if paired['interval95'][0] > 0 else (
                '未观察到正胜率差' if paired['difference'] <= 0 else '正差值尚不确定')
        task = METHOD_TASK[method]
        training = [r for r in store.rows('training') if (r.get('method_id') or
            ('t6_'+r['kind'] if task == 'T6' and r.get('kind') else 't5_bridge_grouping' if task == 'T5' else task)) == method]
        validation = []
        for (m, split, ck), values in evaluations.items():
            if m == method and split == 'validation':
                axis, x = checkpoint_axis(ck)
                validation.append(dict(checkpoint_id=ck, x_axis=axis, x=x,
                    **aggregate(values, cells, config['validation_per_cell'])))
        validation.sort(key=lambda v: (v['x'], v['checkpoint_id']))
        offline = [dict(epoch=r['epoch'], **r['validation']) for r in training if r.get('validation')]
        progress = store.get('progress', {})
        result = store.get('result', store.get('task_result', {}))
        methods.append(dict(method_id=method, checkpoint_id=checkpoint, task_id=task,
            **stats, paired_vs_rule=paired, evidence_status=effect, training=training, validation=validation,
            offline=offline, progress=progress, result=result,
            never_deployed_rate=mean(r.get('never_deployed') for r in rows),
            reserve_exposure_fraction=mean(r.get('reserve_exposure_fraction') for r in rows),
            real_environment_steps=sum(r.get('real_environment_steps', r.get('physical_steps', 0)) for r in rows),
            planner_sim_steps=sum(r.get('planner_sim_steps', 0) for r in rows),
            episode_median_latency_mean_ms=mean(r.get('decision_time_p50_ms') for r in rows)))
    tasks = []
    for task, (title, arms, question) in TASKS.items():
        selected = [m for m in methods if m['method_id'] in arms]
        complete = all(m['complete'] for m in selected)
        active = next((m for m in selected if not m['complete']), selected[-1])
        tasks.append(dict(task_id=task, title=title, question=question, arms=arms,
            execution_status='complete' if complete else 'running' if active['progress'] else 'not_started',
            phase='complete' if complete else active['progress'].get('phase', 'pending'), progress=active['progress']))
    teacher_pairs = []
    for rnd in range(1, config.get('t4', {}).get('rounds', 4)+1):
        teacher = evaluations.get((f't4_teacher_round_{rnd}', 'validation', f'round_{rnd}'), [])
        student = evaluations.get((f't4_student_subset_round_{rnd}', 'validation', f'round_{rnd}'), [])
        if not student:
            student = evaluations.get(('t4_exit', 'validation', f'round_{rnd}'), [])
        lookup = {r['family_id']: r for r in student}
        pairs = [(r, lookup[r['family_id']]) for r in teacher if r['family_id'] in lookup]
        if pairs:
            teacher_pairs.append(dict(round=rnd, episodes=len(pairs), teacher_wins=sum(success(a) for a, _ in pairs),
                student_wins=sum(success(b) for _, b in pairs)))
    result = dict(updated=datetime.now().astimezone().isoformat(timespec='seconds'), run_dir=str(run),
        config=config, methods=methods, tasks=tasks, complete=all(m['complete'] for m in methods),
        evaluation_complete=all(m['complete'] for m in methods), expected_main_and_control_arms=len(MAIN),
        missing_methods=[m['method_id'] for m in methods if not m['episodes']], teacher_pairs=teacher_pairs,
        diagnostics=shared.rows('diagnostics'), gate=gate,
        partition=stores[method_dir(run, 't5_bridge_grouping')].get('pure_partition_diagnostic', {}))
    for store in stores.values():
        store.close()
    shared.close()
    return result


def make_figures(run, methods):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    grouped = defaultdict(list)
    for method in methods:
        grouped[method_dir(run, method['method_id'])].append(method)
    paths = {}
    for directory, arms in grouped.items():
        series = defaultdict(dict)
        def add(title, axis, label, x, y):
            if finite(x) and finite(y):
                series[title, axis].setdefault(label, {})[float(x)] = float(y)
        for method in arms:
            name = method['method_id']
            for row in method['training']:
                if 'physical_steps' in row:
                    axis, x = 'Physical training steps', row['physical_steps']
                elif 'construction_episode' in row:
                    axis, x = 'Construction episodes', row['construction_episode']
                else:
                    axis, x = 'Epoch', row.get('epoch')
                phase = row.get('phase', 'train')
                if phase == 'distill':
                    phase += f" round {row.get('round')}"
                for key in ('loss', 'actor_loss', 'value_loss', 'td_loss'):
                    add('Loss (not task success)', axis, f'{name}: {phase} {key}', x, row.get(key))
                add('Training native success (window)', axis, name, x, row.get('success_window'))
                add('Construction potential difference', axis, name, x, row.get('return'))
            for value in method['validation']:
                if not value['complete'] or value['checkpoint_id'] == 'selected_gate':
                    continue
                add('Validation native success', value['x_axis'], name, value['x'], value['success_rate'])
                for difficulty in value['difficulties']:
                    add('Validation native success by difficulty', value['x_axis'],
                        f"{name}: B/R={difficulty['ratio']:g}", value['x'], difficulty['success_rate'])
            for value in method['offline']:
                for key in ('brier', 'ece', 'advantage_mse', 'ranking_accuracy', 'empirical_candidate_regret'):
                    add('Held-out '+key+' (not native win rate)', 'Epoch', name, value['epoch'], value.get(key))
        if series:
            fig, axes = plt.subplots(math.ceil(len(series)/2), 2, squeeze=False,
                figsize=(13, 3.5*math.ceil(len(series)/2)))
            for ax, ((title, axis), curves) in zip(axes.flat, sorted(series.items())):
                for label, values in curves.items():
                    points = sorted(values.items())
                    ax.plot([x for x, _ in points], [y for _, y in points], linewidth=1.2,
                        marker='o' if len(points) < 20 else None, markersize=3, label=label)
                ax.set(title=title, xlabel=axis)
                if 'native success' in title or 'ranking_accuracy' in title:
                    ax.set_ylim(0, 1)
                ax.grid(alpha=.2); ax.legend(fontsize=7)
            for ax in list(axes.flat)[len(series):]:
                ax.set_visible(False)
            fig.tight_layout()
            buffer = io.BytesIO(); fig.savefig(buffer, format='png', dpi=120); plt.close(fig)
            path = directory/'figures/learning.png'; atomic_text(path, buffer.getvalue())
            paths[path.relative_to(run).as_posix()] = '训练与固定验证'
        available = [m for m in arms if m['episodes']]
        if available:
            fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
            ax = axes[0]
            for i, method in enumerate(arms):
                if not method['episodes']:
                    ax.text(.02, i, 'Pending', va='center'); continue
                value = method['success_rate']; lo, hi = method['wilson95']
                ax.barh(i, value, alpha=1. if method['complete'] else .4)
                ax.errorbar(value, i, xerr=[[max(0., value-lo)], [max(0., hi-value)]], fmt='none', color='black', capsize=3)
                ax.text(min(.94, hi+.02), i, f"{method['wins']}/{method['episodes']}", va='center', fontsize=8)
            ax.set(yticks=range(len(arms)), yticklabels=[m['method_id'] for m in arms], xlim=(0, 1),
                xlabel='Native success rate; Wilson 95% interval', title='Fixed final-policy test')
            ax.invert_yaxis(); ax.grid(axis='x', alpha=.2)
            ax = axes[1]
            colors = ['#237786', '#b8792e', '#8753a6']
            for method in available:
                for (ratio, _), color in zip(RATIOS.items(), colors):
                    cells = [c for c in method['cells'] if c['blue_count']/c['red_count'] == ratio and c['episodes']]
                    ax.plot([c['red_count'] for c in cells], [c['success_rate'] for c in cells], '-o',
                        color=color if len(available) == 1 else None, label=f"{method['method_id']}, B/R={ratio:g}")
            ax.set(xlabel='Red team size', ylabel='Native success rate', ylim=(0, 1), title='Test scale and difficulty')
            ax.grid(alpha=.2); ax.legend(fontsize=6)
            fig.tight_layout()
            buffer = io.BytesIO(); fig.savefig(buffer, format='png', dpi=120); plt.close(fig)
            path = directory/'figures/results.png'; atomic_text(path, buffer.getvalue())
            paths[path.relative_to(run).as_posix()] = '末期测试与规模曲线'
    return paths


def table(headers, rows):
    clean = lambda value: str(value).replace('|', '\\|').replace('\n', ' ')
    return ['| '+' | '.join(map(clean, headers))+' |', '| '+' | '.join('---' for _ in headers)+' |',
        *['| '+' | '.join(map(clean, row))+' |' for row in rows], '']


def markdown(summary, figures):
    config, methods = summary['config'], summary['methods']
    lines = ['# v5 实验报告', '', f"更新时间：{summary['updated']}。",
        f"固定末期测试完成 {sum(m['complete'] for m in methods)}/14 臂；{'训练与正式评估已完成' if summary['complete'] else '本报告仍为中间结果'}。", '',
        '## 研究问题与共同协议', '',
        '在已知 reactive 对手、共享规则底层下，比较真实规划、策略学习和价值学习如何动态分组与分配兵力。',
        f"训练种子 {config.get('seed')}；两个目标；最多 {config.get('max_steps', 50)} 个物理步；每 {config.get('command_interval', 5)} 步及非终止伤亡事件重新决策；原生终局奖励、gamma=1，合法后备，无四人上限。",
        '15 个规模配比等权混合：红方 8、12、16、24、32 人，蓝/红比例 1/2、3/4、1。'
        f"固定验证每配比 {config.get('validation_per_cell')} 局，正式测试每配比 {config.get('eval_per_cell')} 局。",
        '只描述本次训练种子与测试开局；不宣称跨训练种子稳健。主表使用预先指定的末期模型；验证最优权重不替代末期测试。', '',
        '## 整理与剩余工作调整', '',
        '按用户要求，停止后迁移已有进度并继续剩余训练、固定验证和正式评估。未完成的额外候选诊断、搜索预算对比及独立串行部署延迟测量已取消；已经产生的数据保存在 SQLite。'
        '原方案的这些附加项目不能标记为全部完成。',
        '各方法数据只保存在各自 data.sqlite；共享开局和已有诊断存 shared/data.sqlite。网页服务、重复报告和工程测试输出已取消。', '',
        '## 正式测试总表', '',
        '未完成的臂保留实际分子分母，未运行不记为零；混合难度总胜率不能直接与 v4 等人数结果比较。', '']
    rows = []
    for m in methods:
        p = m['paired_vs_rule']; lo, hi = p['interval95']
        rows.append([NAMES[m['method_id']], m['checkpoint_id'], f"{m['wins']}/{m['episodes']}", pct(m['success_rate']),
            '完整' if m['complete'] else '未完整', f"{p['difference']*100:+.2f}" if p['pairs'] else '待配对',
            f'[{lo*100:+.2f}, {hi*100:+.2f}]' if p['pairs'] else '待配对', m['evidence_status']])
    lines += table(['方法', '固定检查点', '成功/局数', '胜率', '覆盖', '比规则差（百分点）', '保守95%配对区间', '判定'], rows)
    lines += ['配对按相同回合族；区间由两个 97.5% Clopper–Pearson 区间经 Bonferroni 组合。该区间不包含训练种子之间的变异。', '', '## 分难度与规模', '']
    lines += table(['方法', '较易', '中等', '等人数'], [[NAMES[m['method_id']], *[
        f"{d['wins']}/{d['episodes']}（{pct(d['success_rate'])}）" for d in m['difficulties']]] for m in methods])
    lines += table(['方法', *[f'{r}v{b}' for r, b in config['cells']]], [[NAMES[m['method_id']], *[
        f"{c['wins']}/{c['episodes']}" if c['episodes'] else '未运行' for c in m['cells']]] for m in methods])
    lines += ['## 六路线方法与结果', '', '本轮是已有机制在当前环境的迁移与重实现；没有完整运行原论文环境，不声称原作者实验的逐项复现。', '']
    for task in summary['tasks']:
        task_id = task['task_id']; arms = [m for m in methods if m['method_id'] in task['arms']]
        lines += [f"### {task_id}：{task['title']}", '', task['question'],
            f"执行状态：{task['execution_status']}；阶段：`{task['phase']}`。", '',
            '实际参数：`'+json.dumps(config.get(task_id.lower(), {}), ensure_ascii=False)+'`。', '']
        if task_id == 'T1':
            lines += ['候选先执行一个事件，再由规则续行到原生终局；平局优先规则。属于 rollout 策略改进的场景适配。', '']
        elif task_id == 'T2':
            lines += ['完整编组为动作，双渐进扩展处理动作和随机后继；叶节点用规则终局续行。规划收益须结合模拟成本解释。', '']
        elif task_id == 'T3':
            lines += [f"独立 actor/critic；编码器 {json.dumps(config.get('model', {}))}；规则初始化 {config.get('bc_episodes')} 回合、{config.get('bc_epochs')} epoch。"
                '候选 PPO 学习选择；AR 逐成员构造规范编组，按整个事件的联合概率裁剪。AR 熵正则为平均条件熵，候选臂为候选分布熵。', '']
        elif task_id == 'T4':
            lines += ['每轮用当前学生访问状态产生 rollout 教师，再蒸馏 10 epoch；属于 ExIt 风格适配。教师经验最优没有超过学生或全部平局时采用基准模仿。'
                '下表使用学生完整固定验证集；教师子集不能替代完整学生验证。', '']
        elif task_id == 'T5':
            lines += ['规则固定目标、身份归属和后备；DQN 学习同目标内 merge＋STOP，奖励为固定配对终局势差。保留 BRIDGE 上层机制，学习率采用当前场景的 3e-4。', '']
        else:
            lines += ['BCE 学习绝对成功概率；ADV 学习相对规则的配对优势，两者共享数据、搜索与规则锚点。ADV-gated 共用 ADV 权重，由验证集选择阈值，'
                '不宣称 SPIBB 算法复现或安全保证。', f"已选阈值：{summary['gate'].get('chosen_threshold', '尚无记录')}；其验证成绩参与选择，不是独立测试。", '']
        for m in arms:
            lines += [f"**{NAMES[m['method_id']]}**：正式成功 {m['wins']}/{m['episodes']}（{pct(m['success_rate'])}）；{m['evidence_status']}。", '']
            if m['training']:
                last = m['training'][-1]
                counters = [f'{label} {last[key]}' for key, label in
                    [('physical_steps', '累计物理步'), ('construction_episode', '构造回合'), ('round', '轮次'), ('epoch', 'epoch')]
                    if key in last]
                lines += ['最近训练记录：'+'；'.join(counters)+f"；窗口原生胜率 {pct(last.get('success_window'))}。", '']
            if m['validation']:
                lines += table(['检查点', '验证成功/局数', '较易', '中等', '等人数', '完整'], [[v['checkpoint_id'],
                    f"{v['wins']}/{v['episodes']}（{pct(v['success_rate'])}）", *[pct(d['success_rate']) for d in v['difficulties']],
                    '是' if v['complete'] else '否'] for v in m['validation']])
            if m['offline']:
                lines += table(['epoch', 'Brier', 'ECE', '配对MSE', '排序准确率', '候选损失', '全零/状态', '非平局/状态'], [
                    [v['epoch'], *[number(v.get(k)) for k in ('brier', 'ece', 'advantage_mse', 'ranking_accuracy', 'empirical_candidate_regret')],
                    f"{v.get('all_zero_states', '?')}/{v.get('states', '?')}", f"{v.get('non_tie_states', '?')}/{v.get('states', '?')}"] for v in m['offline']])
                lines += ['以上为留出状态的预测和排序指标；不是环境胜率。ADV 没有绝对概率时不填 Brier/ECE。', '']
        if task_id == 'T4' and summary['teacher_pairs']:
            lines += table(['轮次', '相同开局数', '教师成功', '学生成功'], [[r['round'], r['episodes'], r['teacher_wins'], r['student_wins']] for r in summary['teacher_pairs']])
        if task_id == 'T5' and summary['partition'].get('rows'):
            rows = summary['partition']['rows']
            lines += ['同归属划分的已完成配对终局诊断（不是正式在线胜率）：', '']
            lines += table(['划分', '记录状态', '平均续行成功率', '相对大组均值'], [[name, len(rows), number(mean(r.get(name) for r in rows)),
                number(mean(r.get(name+'_minus_grand') for r in rows))] for name in ('rule', 'grand', 'singletons', 'learned')])
        diagnostic = [r for r in summary['diagnostics'] if r.get('task_id') == task_id]
        if diagnostic:
            groups = defaultdict(list)
            for r in diagnostic:
                groups[r.get('method_id'), r.get('state_set', r.get('split', '未标记'))].append(r)
            lines += ['已经完成的独立诊断，按状态集分开报告；未完成部分已取消：', '']
            lines += table(['方法/状态集', '状态数', '方案－规则复核均值', '教师－方法复核均值', '状态排序准确率均值'], [[f'{m}/{s}', len(rows),
                *[number(mean(r.get(k) for r in rows)) for k in ('selected_vs_rule_verification_gain', 'teacher_minus_method_verification_gap', 'ranking_accuracy')]]
                for (m, s), rows in groups.items()])
        used = {method_dir(Path(summary['run_dir']), m['method_id']).name for m in arms}
        for path, label in figures.items():
            if path.split('/')[0] in used:
                lines += [f'![{task_id} {label}]({path})', '']
    lines += ['## 已记录成本与解释边界', '',
        '下列时延是并行实验期间每回合决策中位数的平均，不是逐事件中位数，也不是独占部署基准。独立串行测量已取消，不据此作严格速度优劣结论。', '']
    lines += table(['方法', '正式测试真实步', '规划模拟步', '回合中位延迟均值/ms', '全程未部署比例', '后备暴露'], [[
        NAMES[m['method_id']], m['real_environment_steps'], m['planner_sim_steps'], number(m['episode_median_latency_mean_ms']),
        pct(m['never_deployed_rate']), number(m['reserve_exposure_fraction'])] for m in methods])
    lines += ['损失下降、部署增加、预测校准更好都不能替代原生成功率证据。真实规划比学习路线好，说明模拟中存在可利用决策空间；'
        '不能仅凭这点区分表示、候选覆盖、标签噪声、优化或访问分布的影响。负结果和已完成的诊断同样保留。', '']
    return '\n'.join(lines)


def summarize(run_dir):
    """Update the existing report and figures; return the summary without extra files."""
    run = Path(run_dir).resolve()
    summary = collect(run)
    figures = make_figures(run, summary['methods'])
    atomic_text(run/'实验报告.md', markdown(summary, figures))
    summary['curves'] = figures
    return summary
