"""Concurrent production-shape throughput probes and auditable fixed quotas.

The time limit belongs exclusively to calibration. Formal jobs always finish
their selected count quotas. Learning, simulation, validation and online tests
are extrapolated separately; no smoke-run/total-step multiplication is used.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import importlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

from open_score.research_v4.runner import atomic_json, read_json
from .protocol import ROOT, TASKS, defaults, digest, episode_spec, source_identity


def _configure_threads():
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
    import torch
    torch.set_num_threads(1)


def _decision_probe(job):
    """A full native state trajectory followed by one production-budget search."""
    _configure_threads()
    from .simulator import make_env
    from .planning import RolloutPolicy, MCTSPolicy
    from .tasks.t1_rollout import load_historical_controls
    from open_score.research_v4.actions import rule_grouping
    task, config, spec, stage = job
    env = make_env(spec)
    snapshots = []
    collection_started = time.perf_counter()
    while not env.done:
        snapshots.append(env.snapshot())
        env.step(rule_grouping(env.state()))
    collection_steps = env.state().step
    collection_seconds = time.perf_counter()-collection_started
    index = (0, len(snapshots)//2, len(snapshots)-1)[stage]
    env.restore(snapshots[index])
    frozen, _ = load_historical_controls(config)
    if task == 'T1':
        policy = RolloutPolicy(seed=config['seed'], b1=frozen.get('frozen_b1'),
                              candidates=config['t1']['candidates'], branches=config['t1']['branches'])
    else:
        policy = MCTSPolicy(seed=config['seed'], b1=frozen.get('frozen_b1'), **config['t2'])
    started = time.perf_counter()
    policy.act_env(env)
    trace = policy.last_trace
    result = dict(wall_seconds=time.perf_counter()-started,
                  simulation_physical_steps=trace['simulation_physical_steps'],
                  collection_seconds=collection_seconds, collection_steps=collection_steps,
                  decisions=1, red_count=spec['red_count'], blue_count=spec['blue_count'],
                  physical_step=env.state().step, remaining_steps=50-env.state().step,
                  terminal_branches=(sum(len(r['branches']) for r in trace.get('candidates', []))
                                     if task == 'T1' else trace.get('leaf_rollouts', 0)),
                  iterations=trace.get('iterations', 0), actual_depth=trace.get('actual_max_depth'))
    env.close()
    return result


def _evaluation_probe(ctx, policy, label, cycle):
    """Use the actual compressed logging and CPU episode pool, not a proxy."""
    probe = copy.copy(ctx)
    probe.config = copy.deepcopy(ctx.config)
    # Rotate difficulty within each representative size across cycles.
    probe.config['cells'] = [ctx.config['cells'][i+(cycle+j)%3] for j, i in enumerate((0, 6, 12))]
    # Keep the absolute gzip temporary path safely below legacy Windows
    # MAX_PATH. The phase record retains the descriptive label; the disposable
    # calibration directory uses short, collision-resistant identifiers.
    probe.output = ctx.output/'p'/f'{cycle:x}'
    probe.output.mkdir(parents=True, exist_ok=True)
    method_key = digest(label)[:8]
    # Queue several complete waves so the final large episode does not turn a
    # one-wave straggler into the assumed steady-state cost of 1,500 episodes.
    count = max(3, 3*ctx.cpu_quota)
    started = time.perf_counter()
    result = probe.evaluate(policy, method_key, split='cal', checkpoint='c', limit=count)
    rows = [json.loads(line) for line in (Path(result['path'])/'episodes.jsonl').read_text(encoding='utf-8').splitlines() if line]
    return dict(wall_seconds=time.perf_counter()-started, units=len(rows), unit='episodes',
                episodes=len(rows), real_physical_steps=sum(r['real_environment_steps'] for r in rows),
                simulation_physical_steps=sum(r['planner_sim_steps'] for r in rows),
                command_events=sum(r['command_events'] for r in rows),
                worker_decision_seconds=sum(r['decision_time_total_s'] for r in rows),
                episode_counts_by_scale={str(n):sum(r['red_count'] == n for r in rows)
                                         for n in sorted({r['red_count'] for r in rows})},
                worker_quota=ctx.cpu_quota, includes_logging=True,
                parallel_episode_pool=True)


def _save_partial(ctx, cycle, phases):
    """Diagnostic recovery record; never counted as a completed benchmark cycle."""
    atomic_json(ctx.output/f'partial_{cycle:04d}.json',
                dict(task_id=ctx.task_id, cycle=cycle, complete=False, phases=phases,
                     updated_unix=time.time()))


def planner_benchmark(ctx, cycle):
    from .planning import RolloutPolicy, MCTSPolicy
    from .tasks.t1_rollout import load_historical_controls, SingletonPolicy
    from open_score.research_v4.policies import RulePolicy
    task = ctx.task_id
    jobs = []
    for size_index in (0, 6, 12):
        index = size_index+cycle%3
        spec = episode_spec(ctx.seed, index+15*cycle, 'calibration', task, ctx.config['cells']).to_dict()
        jobs.extend((task, ctx.config, spec, stage) for stage in range(3))
    started = time.perf_counter()
    if ctx.cpu_quota > 1:
        with ProcessPoolExecutor(ctx.cpu_quota, mp_context=mp.get_context('spawn')) as pool:
            records = list(pool.map(_decision_probe, jobs))
    else:
        records = [_decision_probe(job) for job in jobs]
    phases = {'plan_decision': dict(wall_seconds=time.perf_counter()-started,
        worker_seconds=sum(r['wall_seconds'] for r in records), units=len(records), unit='decisions',
        decisions=len(records), simulation_physical_steps=sum(r['simulation_physical_steps'] for r in records),
        terminal_branches=sum(r['terminal_branches'] for r in records), records=records),
        'rule_collection': dict(wall_seconds=sum(r['collection_seconds'] for r in records),
            units=len(records), unit='episodes', episodes=len(records),
            real_physical_steps=sum(r['collection_steps'] for r in records), parallel_episode_pool=False)}
    _save_partial(ctx, cycle, phases)
    frozen, _ = load_historical_controls(ctx.config)
    if task == 'T1':
        policy = RolloutPolicy(ctx.seed, ctx.config['t1']['candidates'], ctx.config['t1']['branches'], frozen.get('frozen_b1'))
    else:
        policy = MCTSPolicy(ctx.seed, b1=frozen.get('frozen_b1'), **ctx.config['t2'])
    phases['evaluation_main'] = _evaluation_probe(ctx, policy, task, cycle)
    _save_partial(ctx, cycle, phases)
    if task == 'T1':
        controls = dict(rule=RulePolicy('rule'), grand=RulePolicy('grand'), singleton=SingletonPolicy(), **frozen)
        for name, control in controls.items():
            phases['evaluation_'+name] = _evaluation_probe(ctx, control, name, cycle)
            _save_partial(ctx, cycle, phases)
    return dict(task_id=task, phases=phases)


def _normalize(task, raw):
    if 'phases' in raw:
        phases = copy.deepcopy(raw['phases'])
        for name, phase in phases.items():
            phase['wall_seconds'] = float(phase.get('wall_seconds', phase.get('wall_s', 0.)))
            phase.setdefault('units', phase.get('optimizer_steps', phase.get('states', phase.get('episodes', 0))))
        return phases
    if task == 'T3':
        phases = {}
        for arm, row in raw.items():
            phases[arm+'_sampling'] = dict(wall_seconds=row['sample_seconds'], units=row['physical_steps'],
                unit='real_physical_steps', real_physical_steps=row['physical_steps'], events=row['events'])
            phases[arm+'_update'] = dict(wall_seconds=row['update_seconds'], units=row['update_examples'],
                unit='state_examples', optimizer_steps=row.get('optimizer_steps', 0), examples=row['update_examples'])
        return phases
    if task == 'T4':
        return {'teacher_label': dict(wall_seconds=raw['sample_seconds'], units=raw['states'], unit='states',
                    states=raw['states'], simulation_physical_steps=raw['simulated_physical_steps'], branches=raw['branches']),
                'distill_update': dict(wall_seconds=raw['update_seconds'], units=raw['update_examples'],
                    unit='state_examples', examples=raw['update_examples'], optimizer_steps=raw.get('optimizer_steps', 1))}
    raise ValueError(f'Unsupported benchmark result for {task}')


def worker(run_dir, task, deadline):
    _configure_threads()
    from open_score.grouping.storage import seed_everything
    from .runtime import TaskContext, append
    from .orchestrate import MODULES
    ctx = TaskContext(run_dir, task)
    from .protocol import stable_seed
    seed_everything(stable_seed(ctx.seed, 'production_calibration', task))
    cycle = 0
    try:
        while cycle == 0 or time.time() < deadline:
            ctx.progress('calibration', cycle=cycle, deadline_utc=deadline)
            started, gpu_before = time.perf_counter(), ctx._gpu_wait_s
            if task in ('T1', 'T2'):
                raw = planner_benchmark(ctx, cycle)
            else:
                module = importlib.import_module(f'open_score.research_v5.tasks.{MODULES[task]}')
                raw = module.benchmark(ctx, events=128) if task == 'T3' else module.benchmark(ctx)
            phases = _normalize(task, raw)
            _save_partial(ctx, cycle, phases)
            if task in ('T3', 'T4'):
                from .neural import NeuralPolicy, make_actor
                for kind in (('candidate', 'autoregressive') if task == 'T3' else ('candidate',)):
                    budget = ctx.config['t3']['candidates'] if task == 'T3' else ctx.config['t4']['candidates']
                    policy = NeuralPolicy(make_actor(kind, ctx.config['model']), kind, budget)
                    phases['evaluation_'+kind] = _evaluation_probe(ctx, policy, f'{task}_{kind}', cycle)
                    _save_partial(ctx, cycle, phases)
            row = dict(task_id=task, cycle=cycle, started_unix=time.time()-(time.perf_counter()-started),
                wall_seconds=time.perf_counter()-started, gpu_wait_seconds=ctx._gpu_wait_s-gpu_before,
                phases=phases, production_model=ctx.config['model'], complete=True)
            append(Path(run_dir)/'samples'/f'{task}.jsonl', row)
            cycle += 1
        atomic_json(ctx.output/'benchmark_result.json', dict(complete=True, cycles=cycle))
    except BaseException as error:
        atomic_json(ctx.output/'benchmark_result.json', dict(complete=False, cycles=cycle,
                    error=str(error), traceback=traceback.format_exc()))
        ctx.progress('calibration_error', error=str(error), cycles=cycle, complete=False)
        raise


def aggregate_samples(samples):
    """Sum comparable work/wall measures, retaining observed worker concurrency."""
    totals = {}
    for task, records in samples.items():
        totals[task] = {}
        for record in records:
            for name, row in record['phases'].items():
                merged = totals[task].setdefault(name, dict(samples=0))
                merged['samples'] += 1
                for key, value in row.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                        merged[key] = merged.get(key, 0.)+value
                    elif key == 'parallel_episode_pool':
                        merged[key] = value
    return totals


def estimate_workload(stats, config, multiplier, *, prepared_seconds=0., calibration_seconds=0., bc_records=None):
    """Count-based stage model. Missing evidence remains unknown, never zero."""
    q = float(multiplier)
    cfg = copy.deepcopy(config)
    scale = q/float(config.get('multiplier', 1.))
    cfg['multiplier'] = q
    for task, fields in {'t3':['steps'], 't4':['states_per_round'], 't5':['states','episodes'], 't6':['train_states']}.items():
        for field in fields:
            cfg[task][field] = max(1, int(round(config[task][field]*scale)))
    n = len(cfg['cells'])*cfg['eval_per_cell']
    v = len(cfg['cells'])*cfg['validation_per_cell']
    d, own = cfg['diagnostic_states'], cfg['own_diagnostic_states']
    missing, components = [], {task: {} for task in TASKS}

    def rate(task, phase, units='units', seconds='wall_seconds'):
        row = stats.get(task, {}).get(phase, {})
        if row.get(units, 0) <= 0 or row.get(seconds, 0) <= 0:
            missing.append(f'{task}.{phase}.{units}/{seconds}')
            return None
        return row[seconds]/row[units]

    def add(task, name, count, per_unit):
        components[task][name] = dict(units=float(count), seconds_per_unit=per_unit,
                                     estimated_seconds=None if per_unit is None else count*per_unit)

    def evaluation_rate(task, phase):
        value = rate(task, phase, 'episodes')
        row = stats.get(task, {}).get(phase, {})
        # T5/T6 benchmark serial episodes; formal evaluator uses its measured
        # CPU quota. This parallel speedup is an explicit model assumption.
        if value is not None and not row.get('parallel_episode_pool', False):
            value /= max(1, cfg['cpu_quotas'][task])
        return value

    rule_episode = rate('T1', 'rule_collection', 'episodes')
    rule_branch = rate('T1', 'plan_decision', 'terminal_branches', 'worker_seconds')
    plan1 = rate('T1', 'plan_decision', 'decisions', 'worker_seconds')
    plan2 = rate('T2', 'plan_decision', 'decisions', 'worker_seconds')
    eval1 = evaluation_rate('T1', 'evaluation_main')
    eval2 = evaluation_rate('T2', 'evaluation_main')
    add('T1', 'online_test', n, eval1)
    for name in ('rule', 'grand', 'singleton', 'frozen_b1', 'frozen_b3'):
        add('T1', 'baseline_'+name, n, evaluation_rate('T1', 'evaluation_'+name))
    # Upper bounds on distinct diagnostic plans, not unobserved wins/labels.
    add('T1', 'common_selection_verification', d*(12*cfg['selection_branches']+12*cfg['verification_branches']), rule_branch)
    add('T1', 'common_online_choices', d, plan1)
    add('T1', 'own_rollout_collection', own, eval1*cfg['cpu_quotas']['T1'] if eval1 is not None else None)
    add('T1', 'own_verification', own*2*cfg['verification_branches'], rule_branch)
    add('T1', 'continuous_control', 2*cfg['t1']['control_episodes'], eval1)
    add('T1', 'frozen_control', cfg['t1']['control_episodes'], evaluation_rate('T1', 'evaluation_frozen_b3'))
    add('T2', 'online_test', n, eval2)
    # 0.5/1/2 x two depths plus one matched-work depth-one search.
    add('T2', 'budget_curve_searches', 8*d, plan2)
    add('T2', 'budget_verification', d*8*cfg['verification_branches'], rule_branch)
    add('T2', 'own_collection', own, eval2*cfg['cpu_quotas']['T2'] if eval2 is not None else None)
    add('T2', 'own_verification', own*2*cfg['verification_branches'], rule_branch)
    bc_count = float(bc_records if bc_records is not None else cfg['bc_episodes']*20)
    for kind in ('candidate', 'autoregressive'):
        sample = stats.get('T3', {}).get(kind+'_sampling', {})
        events_per_step = sample.get('events', 0)/max(1., sample.get('real_physical_steps', 0))
        add('T3', kind+'_real_sampling', cfg['t3']['steps'], rate('T3', kind+'_sampling', 'real_physical_steps'))
        update_rate = rate('T3', kind+'_update', 'examples')
        add('T3', kind+'_ppo_updates', cfg['t3']['steps']*events_per_step*cfg['t3']['epochs'], update_rate)
        add('T3', kind+'_bc', bc_count*cfg['bc_epochs'], update_rate)
        add('T3', kind+'_test_validation', n+len(cfg['t3']['validation_fractions'])*v,
            evaluation_rate('T3', 'evaluation_'+kind))
        candidate_count = 42  # 8 including B1 + frozen B3 + 32 provider samples + selected
        add('T3', kind+'_diagnostics', (d+own)*candidate_count*(cfg['selection_branches']+cfg['verification_branches']), rule_branch)
        add('T3', kind+'_own_collection', own, evaluation_rate('T3', 'evaluation_'+kind))
    teacher_rate = rate('T4', 'teacher_label', 'states')
    student_eval = evaluation_rate('T4', 'evaluation_candidate')
    states, rounds = cfg['t4']['states_per_round'], cfg['t4']['rounds']
    add('T4', 'rule_first_teacher', states*cfg['t4']['candidates']*cfg['t4']['branches'], rule_branch)
    add('T4', 'neural_teacher', states*(rounds-1), teacher_rate)
    add('T4', 'state_collection', states*rounds, student_eval*cfg['cpu_quotas']['T4'] if student_eval is not None else None)
    add('T4', 'distillation', 2*states*rounds*cfg['t4']['epochs']+bc_count*cfg['bc_epochs'], rate('T4', 'distill_update', 'examples'))
    add('T4', 'student_test_validation', n+rounds*(v+cfg['teacher_validation_episodes']), student_eval)
    # Teacher executes at each event: measured neural terminal-label seconds per
    # state times observed event count, divided by formal episode workers.
    evalrow = stats.get('T3', {}).get('evaluation_candidate', {})
    events_episode = evalrow.get('command_events', 0)/max(1., evalrow.get('episodes', 0))
    add('T4', 'teacher_online_validation', rounds*cfg['teacher_validation_episodes']*events_episode,
        teacher_rate/max(1, cfg['cpu_quotas']['T4']) if teacher_rate is not None else None)
    add('T4', 'teacher_verification', rounds*own*2*cfg['verification_branches']/max(1,cfg['t4']['candidates']*cfg['t4']['branches']), teacher_rate)
    add('T4', 'final_diagnostics', (d+own)*10*(cfg['selection_branches']+cfg['verification_branches']), rule_branch)
    t5 = stats.get('T5', {}).get('potential_simulation', {})
    transitions = t5.get('internal_transitions', 0)/max(1., t5.get('construction_episodes', 0))
    potentials = t5.get('candidate_rows', 0)/max(1., t5.get('construction_episodes', 0))
    # 1.5 protects against learned longer merge paths; bounded by legal N-1
    # merges and one initial value. No cache hit savings assumed across episodes.
    mean_red = float(np.mean([x[0] for x in cfg['cells']]))
    transitions = min(mean_red, 1.5*transitions)
    potentials = min(mean_red+1, 1.5*potentials)
    add('T5', 'state_collection', cfg['t5']['states'], rate('T5', 'state_collection', 'states'))
    add('T5', 'potential_labels', cfg['t5']['episodes']*potentials, rate('T5', 'potential_simulation', 'candidate_rows'))
    add('T5', 'td_updates', max(0,cfg['t5']['episodes']*transitions-cfg['t5']['batch_size']+1), rate('T5','optimization','optimizer_steps'))
    add('T5', 'online_test', n, evaluation_rate('T5', 'evaluation'))
    add('T5', 'quarterly_validation', 4*v, evaluation_rate('T5', 'evaluation'))
    add('T5', 'diagnostics', (d+own)*13*(cfg['selection_branches']+cfg['verification_branches']), rule_branch)
    add('T5', 'own_collection', own, evaluation_rate('T5', 'evaluation'))
    add('T6', 'state_collection', cfg['t6']['train_states']+cfg['t6']['validation_states'], rate('T6','state_collection','states'))
    add('T6', 'label_collection', cfg['t6']['train_states']+cfg['t6']['validation_states'], rate('T6','label_collection','states'))
    for kind in ('bce','adv'):
        # Treat no-gradient calibration pass as costly as an update: explicit
        # conservative bound where a separate validation microbenchmark is absent.
        batches = cfg['t6']['epochs']*(math.ceil(cfg['t6']['train_states']/cfg['t6']['batch_size'])+
                                     math.ceil(cfg['t6']['validation_states']/cfg['t6']['batch_size']))
        add('T6', kind+'_train_validation', batches, rate('T6','train_'+kind,'optimizer_steps'))
    add('T6', 'online_tests_and_gate', 3*n+len(cfg['t6']['thresholds'])*v, evaluation_rate('T6','evaluation'))
    add('T6', 'diagnostics', (3*d+own)*10*(cfg['selection_branches']+cfg['verification_branches']), rule_branch)
    add('T6', 'own_collection', own, evaluation_rate('T6','evaluation'))
    totals = {task: (None if any(x['estimated_seconds'] is None for x in rows.values()) else
                     sum(x['estimated_seconds'] for x in rows.values())) for task, rows in components.items()}
    complete = not missing and all(x is not None for x in totals.values())
    shared_estimate = float(prepared_seconds) if prepared_seconds else (cfg['bc_episodes']+d)*(rule_episode or 0.)
    # All fourteen deployment arms are measured one at a time after the six
    # tasks finish. This serial phase is outside the concurrent-task maximum.
    latency_count = int(cfg.get('latency_states', 30))
    latency = {}
    def latency_cost(name, task, phase, direct=None):
        if direct is None:
            direct = rate(task, phase, 'command_events')
            if direct is not None and stats.get(task, {}).get(phase, {}).get('parallel_episode_pool', False):
                direct *= max(1, cfg['cpu_quotas'][task])
        latency[name] = dict(states=latency_count, seconds_per_state=direct,
                             estimated_seconds=None if direct is None else latency_count*direct)
    latency_cost('T1_rollout', 'T1', 'plan_decision', direct=plan1)
    latency_cost('T2_mcts', 'T2', 'plan_decision', direct=plan2)
    for baseline_name in ('rule', 'grand', 'singleton', 'frozen_b1', 'frozen_b3'):
        latency_cost(baseline_name, 'T1', 'evaluation_'+baseline_name)
    for kind in ('candidate', 'autoregressive'):
        latency_cost('T3_'+kind, 'T3', 'evaluation_'+kind)
    latency_cost('T4_exit', 'T4', 'evaluation_candidate')
    latency_cost('T5_bridge', 'T5', 'evaluation')
    for kind in ('bce', 'adv', 'adv_gated'):
        latency_cost('T6_'+kind, 'T6', 'evaluation')
    complete = complete and not missing and all(x['estimated_seconds'] is not None for x in latency.values())
    latency_seconds = sum(x['estimated_seconds'] for x in latency.values()) if complete else None
    raw_seconds = max(totals.values())+shared_estimate+latency_seconds if complete else None
    return dict(multiplier=q, complete=complete, missing_measurements=sorted(set(missing)),
        task_seconds=totals, components=components, prepared_seconds=shared_estimate,
        exclusive_latency_components=latency, exclusive_latency_seconds=latency_seconds,
        exclusive_latency_decisions=len(latency)*latency_count,
        raw_makespan_seconds=raw_seconds,
        estimated_total_seconds=raw_seconds*1.2+calibration_seconds if raw_seconds is not None else None,
        safety_factor=1.2, expected_formal_test_executions=14*n,
        config=cfg, assumptions=[
            'All six workers were benchmarked concurrently; phase wall times include observed contention and GPU waiting.',
            'Formal phases are summed within each task; tasks overlap, so makespan uses the slowest task.',
            'Four T5 quarterly validation passes are included; all fourteen exclusive deployment latency probes run afterward and are added serially.',
            'T5/T6 serial evaluation benchmarks scale by CPU quota; measured speedup is not guaranteed.',
            'Diagnostic distinct-candidate upper bounds and no cross-episode T5 cache savings are conservative.',
            'PPO update cost scales by consumed state examples; BC uses the same update cost as a conservative proxy.',
            'T6 validation forward cost is bounded using measured full update cost.',
            'A global 20% margin covers phase imbalance, checkpoint/logging overhead and episode-length variation.',
            'This is a throughput estimate, not a convergence or performance guarantee.'])


def choose_budget(estimates, target_seconds=20*3600):
    valid = [x for x in estimates if x['complete'] and x['estimated_total_seconds'] <= target_seconds]
    if valid:
        return max(valid, key=lambda x:x['multiplier']), 'highest_measured_budget_within_target'
    smallest = min(estimates, key=lambda x:x['multiplier'])
    reason = 'insufficient_measurements_smallest_budget_only' if not all(x['complete'] for x in estimates) else 'smallest_budget_exceeds_target_will_still_complete'
    return smallest, reason


def calibrate(run_dir, seconds=900):
    """Launch six isolated benchmark subprocesses; choose, but never freeze."""
    _configure_threads()
    from .runtime import append
    from .orchestrate import resource_snapshot
    run_dir = Path(run_dir).resolve()
    if float(seconds) <= 0:
        raise ValueError('calibration duration must be positive')
    if read_json(run_dir/'shared/budget_manifest.json', {}).get('frozen'):
        raise ValueError('Cannot recalibrate a frozen formal run')
    original = read_json(run_dir/'shared/config_resolved.json')
    if not original:
        raise ValueError('prepare must create the formal configuration before calibration')
    config = copy.deepcopy(original)
    if config.get('smoke'):
        raise ValueError('Production calibration requires --smoke=false and production model shapes')
    initial_source = source_identity()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    directory = run_dir/'calibration'/stamp
    shared = directory/'shared'
    shared.mkdir(parents=True)
    atomic_json(shared/'config_resolved.json', config)
    atomic_json(shared/'protocol_resolved.json', dict(protocol_hash=digest(config), **initial_source))
    logs = directory/'logs'; logs.mkdir()
    started, deadline = time.time(), time.time()+float(seconds)
    active, streams, exits = {}, {}, {}
    for task in TASKS:
        streams[task] = (logs/f'{task}.log').open('w', encoding='utf-8')
        command = [sys.executable,'-u','-m','open_score.research_v5.calibration','worker',
                   '--run-dir',str(directory),'--task',task,'--deadline',str(deadline)]
        active[task] = subprocess.Popen(command,cwd=ROOT,stdout=streams[task],stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    last_print = last_resource = 0.
    while active:
        current = time.time()
        if current-last_print >= 10:
            status = []
            for task in TASKS:
                progress = read_json(directory/task/'progress.json', {})
                phase = ('failed' if exits.get(task, 0) != 0 else 'completed' if task in exits else progress.get('phase','starting'))
                sample_path = directory/'samples'/f'{task}.jsonl'
                completed_cycles = (sum(bool(line.strip()) for line in sample_path.read_text(encoding='utf-8').splitlines())
                                    if sample_path.exists() else 0)
                status.append(f'{task}:{phase}/complete{completed_cycles}')
            print(f'[calibration {current-started:.0f}/{seconds:.0f}s] '+' | '.join(status), flush=True)
            last_print = current
        if current-last_resource >= 60:
            append(directory/'resources.jsonl', resource_snapshot(active)); last_resource = current
        for task, process in list(active.items()):
            code = process.poll()
            if code is not None:
                exits[task] = code
                if code:
                    print(f'[calibration] {task} failed ({code}); see {logs/task}.log', flush=True)
                streams.pop(task).close()
                del active[task]
        if active:
            time.sleep(.5)
    elapsed = time.time()-started
    samples = {task: [json.loads(x) for x in (directory/'samples'/f'{task}.jsonl').read_text(encoding='utf-8').splitlines() if x]
               if (directory/'samples'/f'{task}.jsonl').exists() else [] for task in TASKS}
    stats = aggregate_samples(samples)
    bc = read_json(run_dir/'shared/bc_manifest.json', {})
    prep = read_json(run_dir/'shared/progress.json', {}).get('session_elapsed_s', 0.)
    estimates = [estimate_workload(stats, config, q, prepared_seconds=prep, calibration_seconds=elapsed,
                                  bc_records=bc.get('records')) for q in (.5, 1., 2.)]
    selected, reason = choose_budget(estimates)
    final_source = source_identity()
    stable = final_source['source_hash'] == initial_source['source_hash']
    # Calibration is development work, not frozen formal evidence. Record source
    # changes (e.g. concurrent report edits) without laundering formal artifacts
    # or rejecting measured throughput solely for an unrelated reporting edit.
    usable = all(exits.get(task) == 0 and samples[task] for task in TASKS)
    if not usable:
        selected = estimates[0]
        reason = 'benchmark_execution_failed_smallest_budget_provisional'
    report = dict(schema='v5-production-phase-calibration-v1', directory=str(directory),
        requested_seconds=float(seconds), elapsed_seconds=elapsed, benchmark_exits=exits,
        cycles={task:len(rows) for task, rows in samples.items()}, source_stable=stable,
        source_before=initial_source, source_after=final_source, phase_totals=stats,
        gpu_wait_seconds_by_task={task:sum(r.get('gpu_wait_seconds', 0.) for r in rows) for task, rows in samples.items()},
        estimates=estimates, selected_multiplier=selected['multiplier'], selection_reason=reason,
        usable=usable and selected['complete'], frozen=False,
        fixed_training_seed=config['seed'], expected_formal_test_executions=selected['expected_formal_test_executions'])
    atomic_json(run_dir/'calibration_report.json', report)
    atomic_json(directory/'calibration_report.json', report)
    resolved = copy.deepcopy(config)
    resolved['multiplier'] = selected['multiplier']
    for task, fields in {'t3':['steps'], 't4':['states_per_round'], 't5':['states','episodes'], 't6':['train_states']}.items():
        for field in fields:
            resolved[task][field] = selected['config'][task][field]
    resolved['calibration'] = dict(report=str(run_dir/'calibration_report.json'), usable=report['usable'],
        selected_multiplier=selected['multiplier'], estimated_total_seconds=selected['estimated_total_seconds'],
        source_hash=final_source['source_hash'])
    atomic_json(run_dir/'shared/config_resolved.json', resolved)
    atomic_json(run_dir/'shared/budget_estimate.json', dict(selected_multiplier=selected['multiplier'],
        reason=reason, estimated_total_seconds=selected['estimated_total_seconds'], frozen=False, usable=report['usable']))
    lines = ['# v5.1 联合吞吐校准', '',
             f'六任务同时运行，实际校准 {elapsed/60:.1f} 分钟；一个训练种子 {config["seed"]}。', '',
             '| 档位倍率 | 含 20% 余量及校准的预计小时 | 测量完整 |', '|---:|---:|---|']
    for estimate in estimates:
        duration = estimate['estimated_total_seconds']
        lines.append(f'| {estimate["multiplier"]} | {duration/3600:.2f} | {estimate["complete"]} |' if duration is not None else
                     f'| {estimate["multiplier"]} | 未能完整估计 | {estimate["complete"]} |')
    lines.extend(['', f'选择倍率：{selected["multiplier"]}；原因：{reason}。', '',
                  '正式评估包含 14 个方法/对照 × 15 个规模难度单元 × 100 局，共 21,000 局；验证和诊断另计。', '',
                  '预算尚未冻结。正式运行不会因达到 20 小时而停止。阶段分解、GPU 等待、工作量分母与原始短测见 JSON。', '',
                  '若测量缺失，最小档只是暂定配置，报告 usable=false，不能冒充完成校准。源码变动另列审计，正式运行仍需最后冻结。'])
    (run_dir/'calibration_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'[calibration] selected q={selected["multiplier"]}, usable={report["usable"]}; {reason}', flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['worker'])
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args(argv)
    worker(args.run_dir, args.task, args.deadline)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
