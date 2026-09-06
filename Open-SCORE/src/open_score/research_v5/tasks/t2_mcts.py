"""T2: online MCTS-DPW and explicitly accounted depth/budget diagnostics."""
from __future__ import annotations

import json
import time

from open_score.grouping.storage import atomic_json
from open_score.research_v4.actions import rule_grouping
from open_score.research_v5.planning import MCTSPolicy, stable_seed, terminal_rollouts
from .t1_rollout import load_historical_controls, own_diagnostics, timed_phase


def budget_diagnostics(ctx, b1=None):
    started = time.perf_counter()
    records = ctx.collect_states(ctx.config['diagnostic_states'], 'diagnostic', namespace='shared')
    collection_wall = time.perf_counter()-started
    ctx.log('phase_costs', dict(phase='common_state_collection', wall_seconds=collection_wall,
            states=len(records), real_environment_steps=sum(r.get('collection_physical_steps', 0) for r in records),
            simulation_physical_steps=sum(r.get('collection_planning_steps', 0) for r in records)))
    directory = ctx.output/'budget_diagnostics'
    directory.mkdir(parents=True, exist_ok=True)
    main_iterations = ctx.config['t2']['iterations']
    budgets = sorted(set([max(1, main_iterations//2), main_iterations, main_iterations*2]))
    variants = [(depth, iterations) for depth in (1, ctx.config['t2']['depth']) for iterations in budgets]
    variants = list(dict.fromkeys(variants))
    rows = []
    for index, record in enumerate(records):
        path = directory/f'{index:05d}.json'
        if path.exists():
            rows.extend(json.loads(path.read_text(encoding='utf-8')))
            continue
        state_started = time.perf_counter()
        env = ctx.make_env(record['episode_spec'])
        env.restore(record['snapshot'])
        state = record['state']
        selected, traces = [rule_grouping(state)], []
        for depth, iterations in variants:
            parameters = dict(ctx.config['t2'], depth=depth, iterations=iterations)
            planner = MCTSPolicy(seed=ctx.seed, b1=b1, **parameters)
            plan = planner.act_env(env)
            selected.append(plan)
            traces.append(planner.last_trace)
        # A separate depth-one search matches the main depth-three search's
        # actual physical simulation work. The final complete simulation may
        # overshoot; both target and realized work are retained in the report.
        reference = next(t for (d, n), t in zip(variants, traces)
                         if d == ctx.config['t2']['depth'] and n == main_iterations)
        target_work = max(1, reference['simulation_physical_steps'])
        matched_params = dict(ctx.config['t2'], depth=1, iterations=max(1024, target_work*4), physical_budget=target_work)
        matched = MCTSPolicy(seed=ctx.seed, b1=b1, **matched_params)
        selected.append(matched.act_env(env))
        traces.append(matched.last_trace)
        row_variants = variants + [(1, 'matched_physical_work')]
        unique = list(dict.fromkeys(selected))
        seeds = [stable_seed('T2-budget-verification', ctx.seed, record['state_id'], b)
                 for b in range(ctx.config['verification_branches'])]
        verified = terminal_rollouts(env, unique, seeds)
        by_plan = {p: r for p, r in zip(unique, verified)}
        rule_value = by_plan[selected[0]]['value']
        state_rows = []
        for (depth, iterations), plan, trace in zip(row_variants, selected[1:], traces):
            result = dict(state_id=record['state_id'], family_id=record['family_id'],
                episode_spec=record['episode_spec'], depth=depth, iterations=iterations,
                success_value=by_plan[plan]['value'], paired_gain_vs_rule=by_plan[plan]['value']-rule_value,
                simulation_physical_steps=trace['simulation_physical_steps'],
                decision_seconds=trace['decision_seconds'], actual_max_depth=trace['actual_max_depth'],
                actual_mean_depth=trace['actual_mean_depth'], root_action_count=len(trace['root_actions']),
                iterations_executed=trace['iterations'], physical_budget=trace.get('physical_budget'),
                physical_budget_excess=trace.get('physical_budget_excess', 0),
                root_actions=trace['root_actions'], selected=plan.to_dict())
            state_rows.append(result)
            ctx.log('budget_curve', result)
            for branch in trace['simulation_draws']:
                ctx.log('branches', dict(state_id=record['state_id'], depth_limit=depth,
                                         iterations=iterations, stage='planning', **branch))
        # Verification is shared within the state, charged exactly once.
        state_rows[0]['verification_physical_steps'] = sum(b['physical_steps'] for r in verified for b in r['branches'])
        for candidate in verified:
            for branch in candidate['branches']:
                ctx.log('branches', dict(state_id=record['state_id'], stage='independent_verification',
                                         action=candidate['action'], **branch))
        atomic_json(path, state_rows)
        ctx.log('diagnostic_costs', dict(phase='budget_diagnostic', state_id=record['state_id'],
            red_count=record['episode_spec']['red_count'], blue_count=record['episode_spec']['blue_count'],
            remaining_steps=state.max_steps-state.step, wall_seconds=time.perf_counter()-state_started,
            searches=len(state_rows), iterations=sum(r['iterations_executed'] for r in state_rows),
            simulation_physical_steps=sum(r['simulation_physical_steps']+r.get('verification_physical_steps', 0)
                                          for r in state_rows)))
        rows.extend(state_rows)
        ctx.progress('budget_diagnostics', completed=index+1, total=len(records))
        env.close()
    aggregate = []
    for depth, iterations in variants + [(1, 'matched_physical_work')]:
        subset = [r for r in rows if r['depth'] == depth and r['iterations'] == iterations]
        n = len(subset)
        aggregate.append(dict(depth=depth, iterations=iterations, states=n,
            mean_gain_vs_rule=sum(r['paired_gain_vs_rule'] for r in subset)/max(1, n),
            total_simulation_physical_steps=sum(r['simulation_physical_steps'] for r in subset),
            mean_decision_seconds=sum(r['decision_seconds'] for r in subset)/max(1, n),
            mean_actual_depth=sum(r['actual_mean_depth'] for r in subset)/max(1, n)))
    result = dict(states=len(records), collection_wall_seconds=collection_wall, curves=aggregate, records=rows,
        comparison_note='Read gain against ACTUAL simulation physical work; equal iterations are not equal cost.',
        simulation_physical_steps=sum(r['simulation_physical_steps']+r.get('verification_physical_steps', 0) for r in rows))
    atomic_json(ctx.output/'budget_diagnostics.json', result)
    return result


def run(ctx):
    started = time.perf_counter()
    frozen, metadata = load_historical_controls(ctx.config)
    planner = MCTSPolicy(seed=ctx.seed, b1=frozen.get('frozen_b1'), **ctx.config['t2'])
    ctx.progress('online_evaluation', method='T2_mcts_dpw')
    main = timed_phase(ctx, 'T2_online_test', lambda: ctx.evaluate(planner, 'T2_mcts_dpw'))
    curves = timed_phase(ctx, 'budget_diagnostics', lambda: budget_diagnostics(ctx, frozen.get('frozen_b1')))
    own = timed_phase(ctx, 'own_diagnostics', lambda: own_diagnostics(ctx, planner, 'T2_mcts_dpw'))
    summary = dict(task='T2', execution_status='completed', adaptation='algorithm_reimplementation_MCTS_DPW',
                   reference='https://github.com/JuliaPOMDP/MCTS.jl', main=main, budget_diagnostics=curves,
                   own_diagnostics=own, historical_artifacts=metadata, wall_seconds=time.perf_counter()-started)
    atomic_json(ctx.output/'summary.json', summary)
    text = ['# T2 随机 MCTS-DPW 结果', '',
            f"正式测试完成 {main['episodes']} 局，成功 {main['wins']} 局。树深 {ctx.config['t2']['depth']}，每事件 {ctx.config['t2']['iterations']} 次模拟迭代。", '',
            '该实现按 MCTS.jl 的双渐进扩展机制重实现，叶节点使用真实规则终局续行，不是原作者代码直接运行。', '',
            '| 显式深度 | 迭代预算 | 独立复核平均差 | 实际规划模拟物理步 |',
            '|---:|---|---:|---:|']
    for row in curves['curves']:
        text.append(f"| {row['depth']} | {row['iterations']} | {row['mean_gain_vs_rule']:.6f} | {row['total_simulation_physical_steps']} |")
    text.extend(['', '差值相对同状态规则续行。matched_physical_work 对照匹配主深度的实际模拟物理步，最后一个完整终局分支允许少量超出并单独记录。', '',
                 '同迭代数不等于同计算成本；独立复核分支不参与选择。小预算没有改善不等于多事件规划无效。'])
    (ctx.output/'analysis.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    ctx.progress('completed', completed=1, total=1)
    return summary
