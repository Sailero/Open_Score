"""Serial deployment latency after all owned training/evaluation workers exit."""
from __future__ import annotations

import time
import numpy as np
import torch

from open_score.research_v4.policies import RulePolicy
from open_score.research_v4.runner import atomic_json, read_json
from .protocol import TASKS
from .simulator import choose, env_from_snapshot


def deployment_policies(run_dir, config):
    from .neural import load_policy as neural_policy
    from .paired import load_policy as paired_policy
    from .planning import RolloutPolicy, MCTSPolicy
    from .tasks.t1_rollout import SingletonPolicy, load_historical_controls
    from .tasks.t5_bridge_grouping import load_policy as merge_policy
    frozen, _ = load_historical_controls(config)
    if len(frozen) != 2:
        raise FileNotFoundError('Both frozen transfer controls are required for the 14-arm latency table')
    policies = dict(rule=RulePolicy('rule'),grand=RulePolicy('grand'),singleton=SingletonPolicy(),**frozen)
    policies['T1_rollout'] = RolloutPolicy(seed=config['seed'],candidates=config['t1']['candidates'],
        branches=config['t1']['branches'],b1=frozen['frozen_b1'])
    policies['T2_mcts_dpw'] = MCTSPolicy(seed=config['seed'],b1=frozen['frozen_b1'],**config['t2'])
    for kind in ('candidate','autoregressive'):
        policies['t3_'+kind] = neural_policy(run_dir/'T3'/kind/'latest.pt')
    policies['t4_exit'] = neural_policy(run_dir/'T4/latest.pt')
    policies['t5_bridge_grouping'] = merge_policy(run_dir/'T5/latest.pt')
    gate = read_json(run_dir/'T6/gate_selection.json')['chosen_threshold']
    for name,kind,threshold in [('t6_bce','bce',0.),('t6_adv','adv',0.),('t6_adv_gated','adv',gate)]:
        policies[name] = paired_policy(run_dir/'T6'/kind/'latest.pt',budget=config['t6']['search_budget'],threshold=threshold)
    return policies


def measure_latency(run_dir, config):
    from .orchestrate import resource_snapshot
    from .runtime import TaskContext
    ctx = TaskContext(run_dir,'serial_latency',config)
    directory = run_dir/'reports/serial_latency'
    directory.mkdir(parents=True,exist_ok=True)
    samples = ctx.collect_states(config['diagnostic_states'],'diagnostic',namespace='shared')[:config['latency_states']]
    # These are the same predeclared diagnostic families for all 14 arms, never test openings.
    torch.set_num_threads(1)
    resources_before = resource_snapshot({})
    summaries = []
    for method,policy in deployment_policies(run_dir,config).items():
        path = directory/(method+'.json')
        if path.exists():
            summaries.append(read_json(path)); continue
        rows = []
        for sample in samples:
            env = env_from_snapshot(sample['snapshot'])
            try:
                if hasattr(policy,'reset'):
                    policy.reset()
                started = time.perf_counter()
                action = choose(policy,env)
                seconds = time.perf_counter()-started
                trace = getattr(policy,'last_trace',{})
                rows.append(dict(family_id=sample['family_id'],state_id=sample['state_id'],
                    red_count=sample['episode_spec']['red_count'],blue_count=sample['episode_spec']['blue_count'],
                    seconds=seconds,executed_grouping=action.to_dict(),
                    simulated_physical_steps=trace.get('online_planner_steps',trace.get('simulation_physical_steps',0))))
            finally:
                env.close()
        values = [r['seconds'] for r in rows]
        result = dict(method_id=method,states=len(rows),complete=len(rows)==config['latency_states'],
            device='cpu',numeric_threads=1,owned_training_workers=0,
            includes_first_call=True,excludes_environment_restore=True,
            mean_seconds=float(np.mean(values)),p50_seconds=float(np.median(values)),
            p95_seconds=float(np.percentile(values,95)),rows=rows)
        atomic_json(path,result);summaries.append(result)
        print(f'[serial latency] {method}: {len(rows)} states, p50={result["p50_seconds"]:.4f}s p95={result["p95_seconds"]:.4f}s',flush=True)
    result = dict(complete=len(summaries)==14 and all(r['complete'] for r in summaries),
        protocol='serial CPU deployment after all six owned workers exit; external OS load recorded, not controlled',
        resources_before=resources_before,resources_after=resource_snapshot({}),methods=summaries)
    atomic_json(run_dir/'reports/serial_latency.json',result)
    return result
