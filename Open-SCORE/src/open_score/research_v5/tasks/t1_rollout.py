"""T1: full-terminal rollout, shared historical controls and grouping audits."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from open_score.grouping.storage import atomic_json, sha256
from open_score.research_v4.actions import grand_grouping, rule_grouping
from open_score.research_v4.policies import RulePolicy, make_policy
from open_score.research_v5.planning import (
    RolloutPolicy, choose_with_rule_ties, paired_grouping, propose_plans,
    singleton_grouping, stable_seed, terminal_rollouts,
)
from open_score.research_v5.protocol import ROOT


class SingletonPolicy:
    last_trace = {}

    def act(self, state):
        return singleton_grouping(state)

    def reset(self):
        self.last_trace = {}


class FirstEventPolicy:
    def __init__(self, planner):
        self.planner = planner
        self.reset()

    def reset(self):
        self.used = False
        self.last_trace = {}
        self.planner.reset()

    def act_env(self, env):
        if self.used:
            self.last_trace = dict(algorithm='rule_continuation', online_planner_steps=0)
            return rule_grouping(env.state())
        self.used = True
        action = self.planner.act_env(env)
        self.last_trace = self.planner.last_trace
        return action


def historical_assets(config):
    root = Path(config.get('frozen_v4_assets', ROOT/'outputs/v4_comparison/shared/seed_20260906'))
    return {'frozen_b1': root/'count_model.json', 'frozen_b3': root/'global_model/best.pt'}


def load_historical_controls(config):
    policies, metadata = {}, {}
    for method, path in historical_assets(config).items():
        if not path.is_file():
            metadata[method] = dict(status='missing_artifact', artifact=str(path))
            continue
        route = 'b1_counts' if method == 'frozen_b1' else 'b3_global'
        kind = 'count_checkpoint' if method == 'frozen_b1' else 'global_checkpoint'
        policies[method] = make_policy(route, {kind: path}, device='cpu', search_budget=64)
        metadata[method] = dict(status='available', artifact=str(path), sha256=sha256(path),
                                original_protocol='v4', evaluation_protocol=config['version'])
    return policies, metadata


def timed_phase(ctx, phase, function):
    """Explicit units for concurrent throughput extrapolation, not just wall time."""
    started = time.perf_counter()
    result = function()
    row = dict(phase=phase, wall_seconds=max(0., time.perf_counter()-started-result.get('collection_wall_seconds', 0.)),
               states=int(result.get('states', 0)), episodes=int(result.get('episodes', 0)),
               simulation_physical_steps=int(result.get('simulation_physical_steps', 0)),
               real_environment_steps=0, excludes_separately_logged_collection=True)
    if result.get('path'):
        path = Path(result['path'])/'episodes.jsonl'
        if path.is_file():
            episodes = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
            row['real_environment_steps'] = sum(r['real_environment_steps'] for r in episodes)
            row['simulation_physical_steps'] = sum(r['planner_sim_steps'] for r in episodes)
            row['command_events'] = sum(r['command_events'] for r in episodes)
    ctx.log('phase_costs', row)
    return result


def _write_candidate_rows(ctx, state_row, stage, rows):
    for candidate in rows:
        base = dict(state_id=state_row['state_id'], family_id=state_row['family_id'],
                    stage=stage, candidate_index=candidate['candidate_index'], action=candidate['action'])
        ctx.log('candidates', dict(base, value=candidate['value'], outcomes=candidate['outcomes']))
        for branch in candidate['branches']:
            ctx.log('branches', dict(base, **branch))


def common_diagnostics(ctx, planner, frozen_b3=None):
    """Shared state-level selection and independently repeated verification."""
    started = time.perf_counter()
    records = ctx.collect_states(ctx.config['diagnostic_states'], 'diagnostic', namespace='shared')
    collection_wall = time.perf_counter()-started
    ctx.log('phase_costs', dict(phase='common_state_collection', wall_seconds=collection_wall,
            states=len(records), real_environment_steps=sum(r.get('collection_physical_steps', 0) for r in records),
            simulation_physical_steps=sum(r.get('collection_planning_steps', 0) for r in records)))
    directory = ctx.output/'common_diagnostics'
    directory.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, record in enumerate(records):
        path = directory/f'{index:05d}.json'
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding='utf-8')))
            continue
        state_started = time.perf_counter()
        env = ctx.make_env(record['episode_spec'])
        env.restore(record['snapshot'])
        state = record['state']
        candidates = propose_plans(state, planner.candidates, planner.seed, planner.b1)
        variants = {'grand': grand_grouping(state), 'singletons': singleton_grouping(state),
                    'pairs': paired_grouping(state), 'rule': rule_grouping(state)}
        selection = list(dict.fromkeys(candidates + list(variants.values())))
        selection_seeds = [stable_seed('T1-common-selection', ctx.seed, record['state_id'], b)
                           for b in range(ctx.config['selection_branches'])]
        selected_rows = terminal_rollouts(env, selection, selection_seeds)
        scores = {plan: row['value'] for plan, row in zip(selection, selected_rows)}
        selection_by_plan = dict(zip(selection, selected_rows))
        winner_index = choose_with_rule_ties(state, candidates, [scores[p] for p in candidates])
        chosen = dict(variants, selection_teacher=candidates[winner_index])
        # Prespecified nested branch budgets reuse exact generated draws. Their
        # independent verification tests the selections, without counting these
        # cached prefixes as additional physically simulated episodes.
        branch_budgets = sorted(set(min(ctx.config['selection_branches'], b) for b in
                                   (max(1, planner.branches//2), planner.branches,
                                    ctx.config['selection_branches'])))
        budget_cost = {}
        for branch_budget in branch_budgets:
            means = [float(np.mean(selection_by_plan[p]['outcomes'][:branch_budget])) for p in candidates]
            chosen[f'rollout_budget_{branch_budget}'] = candidates[choose_with_rule_ties(state, candidates, means)]
            budget_cost[branch_budget] = sum(b['physical_steps'] for p in candidates
                                             for b in selection_by_plan[p]['branches'][:branch_budget])
        # The deployed planner uses its own frozen online budget; do not call
        # the 16-branch diagnostic teacher the deployed 8-branch policy.
        chosen['rollout'] = planner.act_env(env)
        online_work = planner.last_trace['online_planner_steps']
        if frozen_b3 is not None:
            chosen['frozen_b3'] = frozen_b3.act(state)
        verification = list(dict.fromkeys(chosen.values()))
        seeds = [stable_seed('T1-common-verification', ctx.seed, record['state_id'], b)
                 for b in range(ctx.config['verification_branches'])]
        verified_rows = terminal_rollouts(env, verification, seeds)
        verified = {plan: row for plan, row in zip(verification, verified_rows)}
        rule_value = verified[variants['rule']]['value']
        grand = verified[variants['grand']]
        summary = dict(state_id=record['state_id'], family_id=record['family_id'],
            episode_spec=record['episode_spec'], state_kind=record['state_kind'],
            candidate_count=len(candidates), selection_all_zero=all(scores[p] == 0 for p in candidates),
            selection_all_tie=len({scores[p] for p in candidates}) == 1,
            verification_values={name: verified[plan]['value'] for name, plan in chosen.items()},
            paired_gain_vs_rule={name: verified[plan]['value']-rule_value for name, plan in chosen.items()},
            group_only={name: dict(mean_difference=verified[variants[name]]['value']-grand['value'],
                                  branch_vector_changed=verified[variants[name]]['outcomes'] != grand['outcomes'])
                        for name in ('singletons', 'pairs', 'rule')},
            simulation_physical_steps=online_work+sum(b['physical_steps'] for rows in (selected_rows, verified_rows)
                                                       for r in rows for b in r['branches']))
        summary['budget_curve'] = [dict(branches=b, candidate_count=len(candidates),
            selection_simulation_physical_steps=budget_cost[b],
            verification_gain_vs_rule=verified[chosen[f'rollout_budget_{b}']]['value']-rule_value)
            for b in branch_budgets]
        for budget_row in summary['budget_curve']:
            ctx.log('budget_curve', dict(state_id=record['state_id'], family_id=record['family_id'], **budget_row))
        _write_candidate_rows(ctx, record, 'selection', selected_rows)
        _write_candidate_rows(ctx, record, 'verification', verified_rows)
        ctx.log('states', dict(state_id=record['state_id'], family_id=record['family_id'], state=state.to_dict()))
        atomic_json(path, summary)
        ctx.log('diagnostic_costs', dict(phase='common_diagnostic', state_id=record['state_id'],
            red_count=record['episode_spec']['red_count'], blue_count=record['episode_spec']['blue_count'],
            remaining_steps=state.max_steps-state.step, wall_seconds=time.perf_counter()-state_started,
            terminal_branches=sum(len(r['branches']) for rows in (selected_rows, verified_rows) for r in rows),
            simulation_physical_steps=summary['simulation_physical_steps']))
        summaries.append(summary)
        ctx.progress('common_diagnostics', completed=index+1, total=len(records))
        env.close()
    totals = dict(states=len(summaries), collection_wall_seconds=collection_wall,
                  family_count=len({r['family_id'] for r in summaries}),
                  all_zero_states=sum(r['selection_all_zero'] for r in summaries),
                  all_tie_states=sum(r['selection_all_tie'] for r in summaries),
                  simulation_physical_steps=sum(r['simulation_physical_steps'] for r in summaries),
                  grouping={}, records=summaries)
    for name in ('singletons', 'pairs', 'rule'):
        differences = [r['group_only'][name]['mean_difference'] for r in summaries]
        totals['grouping'][name] = dict(better=sum(x > 0 for x in differences),
            worse=sum(x < 0 for x in differences), equal=sum(x == 0 for x in differences),
            branch_sensitive=sum(r['group_only'][name]['branch_vector_changed'] for r in summaries),
            mean_difference=float(np.mean(differences)) if differences else None)
    atomic_json(ctx.output/'common_diagnostics.json', totals)
    return totals


def own_diagnostics(ctx, policy, method):
    started = time.perf_counter()
    records = ctx.collect_states(ctx.config['own_diagnostic_states'], 'onpolicy_diagnostic',
                                 policy=policy, namespace=method)
    collection_wall = time.perf_counter()-started
    collection_planning = sum(r.get('collection_planning_steps', 0) for r in records)
    ctx.log('phase_costs', dict(phase='own_state_collection', wall_seconds=collection_wall,
            states=len(records), real_environment_steps=sum(r.get('collection_physical_steps', 0) for r in records),
            simulation_physical_steps=collection_planning))
    directory = ctx.output/'own_diagnostics'
    directory.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, record in enumerate(records):
        path = directory/f'{index:05d}.json'
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding='utf-8')))
            continue
        env = ctx.make_env(record['episode_spec'])
        env.restore(record['snapshot'])
        state = record['state']
        selected = policy.act_env(env) if hasattr(policy, 'act_env') else policy.act(state)
        planning_work = policy.last_trace.get('online_planner_steps', 0)
        rule = rule_grouping(state)
        plans = list(dict.fromkeys([rule, selected]))
        seeds = [stable_seed('own-verification', method, ctx.seed, record['state_id'], b)
                 for b in range(ctx.config['verification_branches'])]
        verified = terminal_rollouts(env, plans, seeds)
        values = {plan: row['value'] for plan, row in zip(plans, verified)}
        summary = dict(method=method, state_id=record['state_id'], family_id=record['family_id'],
            episode_spec=record['episode_spec'], rule_value=values[rule], selected_value=values[selected],
            paired_gain_vs_rule=values[selected]-values[rule], same_action=selected == rule,
            simulation_physical_steps=planning_work+sum(b['physical_steps'] for r in verified for b in r['branches']))
        _write_candidate_rows(ctx, record, 'onpolicy_verification', verified)
        atomic_json(path, summary)
        summaries.append(summary)
        ctx.progress('own_diagnostics', completed=index+1, total=len(records))
        env.close()
    result = dict(states=len(summaries), collection_wall_seconds=collection_wall,
                  collection_simulation_physical_steps=collection_planning,
                  family_count=len({r['family_id'] for r in summaries}),
                  mean_gain_vs_rule=float(np.mean([r['paired_gain_vs_rule'] for r in summaries])) if summaries else None,
                  simulation_physical_steps=sum(r['simulation_physical_steps'] for r in summaries), records=summaries)
    atomic_json(ctx.output/'own_diagnostics.json', result)
    return result


def run(ctx):
    started = time.perf_counter()
    frozen, metadata = load_historical_controls(ctx.config)
    atomic_json(ctx.output/'historical_artifacts.json', metadata)
    planner = RolloutPolicy(seed=ctx.seed, candidates=ctx.config['t1']['candidates'],
                            branches=ctx.config['t1']['branches'], b1=frozen.get('frozen_b1'))
    ctx.progress('online_evaluation', method='T1_rollout')
    main = timed_phase(ctx, 'T1_online_test', lambda: ctx.evaluate(planner, 'T1_rollout'))
    controls = dict(rule=RulePolicy('rule'), grand=RulePolicy('grand'), singleton=SingletonPolicy(), **frozen)
    baseline_summaries = {}
    for method, policy in controls.items():
        ctx.progress('shared_baselines', method=method)
        baseline_summaries[method] = timed_phase(ctx, f'baseline_{method}',
                                                lambda: ctx.evaluate(policy, method, checkpoint='frozen'))
    diagnostics = timed_phase(ctx, 'common_diagnostics', lambda: common_diagnostics(ctx, planner, frozen.get('frozen_b3')))
    continuous = {}
    policies = {'T1_once_then_rule': FirstEventPolicy(planner), 'T1_continuous': planner}
    if 'frozen_b3' in frozen:
        policies['B3_continuous'] = frozen['frozen_b3']
    for method, policy in policies.items():
        ctx.progress('control_diagnostic', method=method)
        continuous[method] = timed_phase(ctx, method, lambda: ctx.evaluate(policy, method, split='control_diagnostic',
                                          limit=ctx.config['t1']['control_episodes']))
    own = timed_phase(ctx, 'own_diagnostics', lambda: own_diagnostics(ctx, planner, 'T1_rollout'))
    summary = dict(task='T1', execution_status='completed', adaptation='rollout_policy_improvement_fixed_rule',
                   main=main, baselines=baseline_summaries, historical_artifacts=metadata,
                   common_diagnostics=diagnostics, own_diagnostics=own, control_diagnostic=continuous,
                   wall_seconds=time.perf_counter()-started)
    atomic_json(ctx.output/'summary.json', summary)
    text = ['# T1 真实终局 rollout 结果', '',
            f"正式测试完成 {main['episodes']} 局，成功 {main['wins']} 局。全部结果使用固定规则底层和原生终局。", '',
            '此任务是固定规则续行的 rollout 适配；有限分支估计不继承精确策略改进保证。', '',
            f"共同诊断 {diagnostics['states']} 个状态，其中全零 {diagnostics['all_zero_states']}，全平局 {diagnostics['all_tie_states']}。", '',
            '| 同目标分区对照 | 优于大组 | 劣于大组 | 平均结果相同 |',
            '|---|---:|---:|---:|']
    for name, row in diagnostics['grouping'].items():
        text.append(f"| {name} | {row['better']} | {row['worse']} | {row['equal']} |")
    text.extend(['', '方向按独立 verification 分支均值计算；同回合族相关性和有限分支不确定性必须保留。', '',
                 '完整逐候选、逐分支、预算曲线、单次与持续控制、自访问诊断及共享基线见本目录 JSON/JSONL。'])
    (ctx.output/'analysis.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    ctx.progress('completed', completed=1, total=1)
    return summary
