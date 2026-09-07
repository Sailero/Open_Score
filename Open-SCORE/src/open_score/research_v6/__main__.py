"""One invocation: shared S2, six methods, paired evaluation and one report."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from . import ROOT, METHODS, RL_METHODS, EpisodeSpec, episode_spec, load_config, make_episode_env
from open_score.research_v5.protocol import stable_seed
from open_score.research_v5.storage import Store


def now():
    return datetime.now(timezone.utc).isoformat()


def configure_threads():
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
    import torch
    torch.set_num_threads(1)


def cpu_payload(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_payload(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_payload(v) for v in value)
    return value


def policy(config, run, method, seed, device='cpu'):
    from .learning import make_learner
    from .local_value import load_scorer, make_policy
    if method == 'FrozenRule':
        return None
    scorer = load_scorer(config, run, seed, device) if config['methods'][method]['s2'] else None
    return (make_policy(method, config, scorer) if method.startswith('BLOTTO_') else
            make_learner(method, config, seed, device, scorer))


def initialize_sampler(run, method, seed):
    global _CONFIG, _METHOD, _POLICY
    configure_threads()
    _CONFIG = json.loads((Path(run)/'config.json').read_text(encoding='utf-8'))
    _METHOD = method
    _POLICY = policy(_CONFIG, run, method, seed, 'cpu')


def sample_episode(spec_dict, weights, physical_steps, explore, keep_transitions):
    import numpy as np
    import torch
    from .environment import rule_grouping, partition_key, responsibilities
    spec = EpisodeSpec.from_dict(spec_dict)
    torch.manual_seed(spec.policy_seed)
    rng = np.random.default_rng(spec.policy_seed)
    if weights is not None:
        _POLICY.load_policy_state_dict(weights)
    env = make_episode_env(_CONFIG, spec)
    state = env.reset()
    transitions, delays, decisions = [], [], []
    native_total = shaped_total = positive_shaping = compensation = 0.
    reserve_exposure = positive_events = reassignments = fixed_z_changes = 0
    first_deployment = None
    counters = dict(internal_tokens=0, candidates_scored=0, s2_scoring_rows=0)
    events = 0
    started = time.perf_counter()
    last_info = {}
    try:
        while not env.done:
            tick = time.perf_counter()
            if _POLICY is None:
                action, metrics = rule_grouping(state), {}
            else:
                with torch.inference_mode():
                    action, metrics = _POLICY.act(state, physical_steps=physical_steps+state.step,
                                                   explore=explore, rng=rng)
            delays.append((time.perf_counter()-tick)*1000.)
            old_z, new_z = state.previous.assignment(), action.assignment()
            reassignments += sum(old_z.get(i) != new_z.get(i) for i in state.ids('red'))
            fixed_z_changes += int(old_z == new_z and partition_key(state.previous) != partition_key(action))
            if first_deployment is None and action.groups:
                first_deployment = state.step
            reserve_exposure += len(action.reserve)/max(1, len(state.ids('red')))
            for key in counters:
                alias = {'candidates_scored': 'candidate_scored', 's2_scoring_rows': 's2_rows'}.get(key, key)
                counters[key] += int(metrics.get(key, metrics.get(alias, 0)))
            following, native, done, info = env.step(action)
            events += 1
            positive_events += int(info['nonterminal_positive_shaping'] > 0)
            native_total += native
            shaped_total += info['shaped_reward']
            positive_shaping += info['nonterminal_positive_shaping']
            compensation += info['terminal_compensation']
            if keep_transitions:
                transitions.append(dict(state=state, action=action, next_state=following,
                    reward=info['shaped_reward'], native_reward=native, done=done,
                    delta=info['delta'], old_logp=metrics.get('old_logp'),
                    policy_version=metrics.get('policy_version', 0)))
            if events <= 2:
                decisions.append(dict(step=state.step, grouping=action.to_dict(),
                    target_counts={str(t): sum(len(g.members) for g in action.groups if g.target == t)
                                   for t in state.ids('targets')}, reserve_count=len(action.reserve),
                    group_sizes=[len(g.members) for g in action.groups],
                    blue_contacts_per_group=[len(ids) for _, ids in responsibilities(state, action)]))
            state, last_info = following, info
        success = bool(env.native_success)
        # From a normal reset the native and shaped trajectory returns coincide.
        if abs(native_total-shaped_total) > 1e-6:
            raise RuntimeError('Potential terminal compensation did not telescope')
        row = dict(**spec.to_dict(), success_native=success, physical_steps=state.step,
            real_events=events, native_return=native_total, shaped_return=shaped_total,
            shaped_native_gap=shaped_total-native_total, nonterminal_positive_shaping=positive_shaping,
            positive_shaping_event_fraction=positive_events/max(events, 1), terminal_compensation=compensation,
            blue_health_loss=last_info.get('blue_health_loss', 0.),
            termination_reason=last_info.get('termination_reason', ''), first_deployment=first_deployment,
            sustained_all_reserve=first_deployment is None, reserve_exposure=reserve_exposure/max(events, 1),
            reassignment_count=reassignments, partition_changes_at_fixed_assignment=fixed_z_changes,
            wall_time_s=time.perf_counter()-started,
            decision_latency_ms_mean=float(np.mean(delays)) if delays else 0.,
            decision_sample=decisions, **counters)
        return transitions, row
    finally:
        env.close()


def evaluate(config, run, method, seed, step, weights, pool, store, split='test', actual_steps=None):
    per_scene = config['evaluation'][f'{split}_per_scenario']
    checkpoint = f'{seed}_{step}'
    completed = {row['family_id'] for row in store.episodes(method, split, checkpoint)}
    specs = [episode_spec(config, i, split, seed, method) for i in range(per_scene*len(config['scenarios']))]
    remaining = [s for s in specs if s.family_id not in completed]
    workers = config['resources']['environment_processes_per_job']
    started = time.monotonic()
    wins = sum(int(row['success_native']) for row in store.episodes(method, split, checkpoint))
    for start in range(0, len(remaining), workers):
        pending = [pool.submit(sample_episode, spec.to_dict(), weights, int(step) if isinstance(step, int) else 0,
                               False, False) for spec in remaining[start:start+workers]]
        with store.transaction():
            for future in pending:
                _, row = future.result()
                row.update(seed=seed, checkpoint_step=step, checkpoint_physical_steps=actual_steps)
                store.save_episode(method, split, checkpoint, row)
                wins += int(row['success_native'])
        newly_completed = min(start+workers, len(remaining))
        completed_count = len(completed)+newly_completed
        store.put(f'progress/{seed}', dict(phase=split, checkpoint=step,
            completed=completed_count, total=len(specs), wins=wins,
            provisional_win_rate=wins/max(completed_count, 1),
            remaining_seconds=(time.monotonic()-started)/max(newly_completed, 1)*(len(specs)-completed_count),
            updated=now()))
        if start % 20 == 0:
            print(f'[{method}/{seed}] {split} {step}: {len(completed)+min(start+workers,len(remaining))}/{len(specs)}', flush=True)


def save_checkpoint(path, learner, progress):
    from open_score.grouping.storage import atomic_checkpoint, random_state
    atomic_checkpoint(path, dict(learner=learner.state_dict(), progress=progress, rng=random_state()))


def train(config, run, method, seed, device):
    import numpy as np
    import torch
    from open_score.grouping.storage import atomic_checkpoint, restore_random_state
    learner = policy(config, run, method, seed, device)
    folder = run/method
    model_dir = folder/'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    store = Store(folder)
    completed_run = store.get(f'finished/{seed}', {})
    if completed_run.get('complete'):
        store.close()
        return completed_run
    resume = model_dir/f'{seed}_resume.pt'
    progress = dict(physical_steps=0, episodes=0, events=0, validations=[], tests=[], best_validation=-1.,
                    first_success_physical_step=None, native_success_count=0, no_native_reward_window=0)
    if resume.exists():
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        learner.load_state_dict(saved['learner'])
        progress.update(saved['progress'])
        restore_random_state(saved['rng'])
        # Remove only the uncheckpointed metric tail for this run/seed on resume.
        with store.transaction():
            for stream in ('training', 'train_episodes'):
                store.connection.execute("DELETE FROM records WHERE stream=? AND json_extract(value,'$.seed')=? AND json_extract(value,'$.physical_steps')>?",
                                         (stream, seed, progress['physical_steps']))
    goal = config['training']['physical_steps_per_method_seed']
    count = config['mappo']['episodes_per_batch'] if method == 'MAPPO_Intent' else config['resources']['environment_processes_per_job']
    last_saved = progress['physical_steps']
    started, initial_steps = time.monotonic(), progress['physical_steps']
    context = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=config['resources']['environment_processes_per_job'],
            mp_context=context, initializer=initialize_sampler, initargs=(str(run), method, seed)) as pool:
        while True:
            weights = cpu_payload(learner.policy_state_dict())
            for mark in config['training']['validation_steps']:
                if mark <= progress['physical_steps'] and mark not in progress['validations']:
                    save_checkpoint(resume, learner, progress)
                    evaluate(config, run, method, seed, mark, weights, pool, store, 'validation', progress['physical_steps'])
                    progress['validations'].append(mark)
                    rows = store.episodes(method, 'validation', f'{seed}_{mark}')
                    main = [r for r in rows if r['scenario_id'] in config['evaluation']['main_scenarios']]
                    rate = np.mean([r['success_native'] for r in main])
                    progress['latest_validation'] = dict(step=mark, main_success_rate=float(rate),
                                                        wins=sum(r['success_native'] for r in main), episodes=len(main))
                    if rate > progress['best_validation']:
                        progress['best_validation'] = float(rate)
                        atomic_checkpoint(model_dir/f'{seed}_best.pt', dict(policy=weights, physical_steps=progress['physical_steps']))
                    if mark in config['training']['checkpoint_steps']:
                        name = 'final' if mark == goal else f'{mark//1_000_000}m'
                        atomic_checkpoint(model_dir/f'{seed}_{name}.pt', dict(policy=weights, physical_steps=progress['physical_steps']))
                    save_checkpoint(resume, learner, progress)
            # Intermediate and final test use exactly their retained policy snapshot.
            for mark in config['training']['checkpoint_steps']:
                if mark <= progress['physical_steps'] and mark not in progress['tests']:
                    name = 'final' if mark == goal else f'{mark//1_000_000}m'
                    retained = torch.load(model_dir/f'{seed}_{name}.pt', map_location='cpu', weights_only=False)
                    evaluate(config, run, method, seed, mark, retained['policy'], pool, store,
                             actual_steps=retained['physical_steps'])
                    progress['tests'].append(mark)
                    save_checkpoint(resume, learner, progress)
            if progress['physical_steps'] >= goal:
                break
            specs = [episode_spec(config, progress['episodes']+i, 'train', seed, method) for i in range(count)]
            futures = [pool.submit(sample_episode, spec.to_dict(), weights, progress['physical_steps'], True, True) for spec in specs]
            collected = [future.result() for future in futures]
            episodes = [episode for episode, row in collected]
            update_metrics = learner.learn(episodes)
            with store.transaction():
                for episode, row in collected:
                    progress['physical_steps'] += row['physical_steps']
                    progress['episodes'] += 1
                    progress['events'] += row['real_events']
                    progress['native_success_count'] += int(row['success_native'])
                    progress['no_native_reward_window'] = 0 if row['success_native'] else progress['no_native_reward_window']+1
                    if row['success_native'] and progress['first_success_physical_step'] is None:
                        progress['first_success_physical_step'] = progress['physical_steps']
                    row['episode_physical_steps'] = row['physical_steps']
                    row.update(seed=seed, episode=progress['episodes'], physical_steps=progress['physical_steps'])
                    if progress['episodes'] % 100:
                        row.pop('decision_sample', None)
                    store.append('train_episodes', row)
                elapsed = time.monotonic()-started
                speed = (progress['physical_steps']-initial_steps)/max(elapsed, 1e-6)
                metric = dict(**update_metrics, seed=seed, physical_steps=progress['physical_steps'],
                    episodes=progress['episodes'], events=progress['events'], phase='training',
                    native_success_rate=float(np.mean([r['success_native'] for _, r in collected])),
                    shaped_native_gap=max(abs(r['shaped_native_gap']) for _, r in collected),
                    native_success_count=progress['native_success_count'],
                    first_success_physical_step=progress['first_success_physical_step'],
                    no_native_reward_window=progress['no_native_reward_window'],
                    latest_validation=progress.get('latest_validation'),
                    physical_steps_per_second=speed, remaining_training_seconds=(goal-progress['physical_steps'])/max(speed, 1e-9),
                    updated=now())
                if torch.cuda.is_available() and device.startswith('cuda'):
                    metric['peak_vram_bytes'] = torch.cuda.max_memory_allocated()
                import psutil
                metric['resident_memory_bytes'] = psutil.Process().memory_info().rss
                store.append('training', metric)
                store.put(f'progress/{seed}', metric)
            if progress['episodes'] % 20 == 0:
                print(f'[{method}/{seed}] {progress["physical_steps"]:,}/{goal:,} steps; {speed:.1f} steps/s including learning; block win={metric["native_success_rate"]:.1%}', flush=True)
            if progress['physical_steps']-last_saved >= 25_000:
                save_checkpoint(resume, learner, progress)
                last_saved = progress['physical_steps']
        store.put(f'finished/{seed}', dict(complete=True, **progress, finished=now()))
        # Resume contains a large replay; final/best/2M weights remain authoritative.
        if resume.exists():
            resume.unlink()
    store.close()
    return progress


def nonlearning(config, run, method, seed):
    folder = run/method
    store = Store(folder)
    with ProcessPoolExecutor(max_workers=config['resources']['environment_processes_per_job'],
            mp_context=mp.get_context('spawn'), initializer=initialize_sampler,
            initargs=(str(run), method, seed)) as pool:
        evaluate(config, run, method, seed, 'final', None, pool, store)
    store.put(f'finished/{seed}', dict(complete=True, finished=now()))
    store.close()
    return dict(complete=True)


def jobs(config):
    seeds = config['seeds']['initializations']
    result = [('S2_data', 'S2_data', 0, [])]
    for seed in seeds:
        result.append((f'S2_{seed}', 'S2_fit', seed, ['S2_data']))
        result.append((f'ALMA_S2_{seed}', 'ALMA_S2', seed, [f'S2_{seed}']))
        for method in ('ALMA_Alloc', 'MAPPO_Intent', 'ALMA_Group'):
            result.append((f'{method}_{seed}', method, seed, []))
        for method in ('BLOTTO_Count', 'BLOTTO_Group'):
            result.append((f'{method}_{seed}', method, seed, [f'S2_{seed}']))
    result.append(('FrozenRule', 'FrozenRule', seeds[0], []))
    if config['diagnostics']['enabled']:
        result.append(('diagnostics', 'diagnostics', seeds[0], [item[0] for item in result]))
    return result


def worker(run, job, device):
    config = json.loads((run/'config.json').read_text(encoding='utf-8'))
    selected = next(item for item in jobs(config) if item[0] == job)
    _, method, seed, _ = selected
    shared = Store(run/'shared')
    shared.put(f'progress/{job}', dict(phase='running', pid=os.getpid(), started=now()))
    try:
        if method == 'S2_data':
            from .local_value import collect_data
            result = collect_data(config, run)
        elif method == 'S2_fit':
            from .local_value import fit
            result = fit(config, run, seed, device)
        elif method == 'diagnostics':
            result = diagnostics(config, run)
        elif method in RL_METHODS:
            result = train(config, run, method, seed, device)
        else:
            result = nonlearning(config, run, method, seed)
        shared.put(f'finished/{job}', dict(complete=True, finished=now(), result=result))
        shared.put(f'progress/{job}', dict(phase='complete', finished=now()))
    except BaseException as error:
        shared.put(f'progress/{job}', dict(phase='error', error=str(error), traceback=traceback.format_exc(), updated=now()))
        raise
    finally:
        shared.close()


def diagnostics(config, run):
    """Small fixed paired branches, never used by an online action selector."""
    import numpy as np
    import torch
    from .environment import rule_grouping, partition_key
    from .local_value import load_scorer
    from open_score.grouping.domain import Group, Grouping
    seed = config['seeds']['initializations'][0]
    shared = Store(run/'shared')
    scorer = load_scorer(config, run, seed, 'cpu')
    policies = {m: policy(config, run, m, seed) for m in METHODS.values()}
    for name, actor in policies.items():
        if name in RL_METHODS:
            saved = torch.load(run/name/'models'/f'{seed}_final.pt', map_location='cpu', weights_only=False)
            actor.load_policy_state_dict(saved['policy'])
    total = config['diagnostics']['states_per_scenario']*len(config['scenarios'])
    for index in range(total):
        key = f'diagnostic/{index}'
        if shared.get(key) is not None:
            continue
        spec = episode_spec(config, index, 'diagnostics', seed)
        env = make_episode_env(config, spec)
        state = env.reset()
        # Half initial states; half maximum public threat events of a rule episode.
        if index//len(config['scenarios']) % 2:
            selected, maximum = None, -float('inf')
            while not env.done:
                env.step(rule_grouping(env.state()))
                if not env.done:
                    current_state = env.state()
                    targets = current_state.alive('targets')
                    threat = sum(float(b.health)/max(1., min(float(np.linalg.norm(np.asarray(b.position)-t.position)) for t in targets))
                                 for b in current_state.alive('blue')) if targets else 0.
                    if threat > maximum:
                        maximum, selected = threat, env.snapshot()
            if selected is not None:
                env.restore(selected)
            state = env.state()
        if env.done:
            shared.put(key, dict(terminal_before_selection=True, family_id=spec.family_id))
            env.close()
            continue
        snapshot = env.snapshot()
        choices = {'FrozenRule': rule_grouping(state)}
        for method, actor in policies.items():
            torch.manual_seed(stable_seed(spec.family_id, method, 'candidate_torch'))
            with torch.inference_mode():
                choices[method] = actor.act(state, explore=False,
                    rng=np.random.default_rng(stable_seed(spec.family_id, method, 'candidate')))[0]
        count = choices['BLOTTO_Count']
        choices['fixed_assignment_singleton'] = Grouping(tuple(Group(g.target, (i,)) for g in count.groups for i in g.members), count.reserve)
        unique = []
        for action in choices.values():
            if partition_key(action) not in [partition_key(a) for a in unique]:
                unique.append(action)
        unique = unique[:config['diagnostics']['candidates_per_state']]
        s2_scores = scorer.log_probabilities(state, unique).sum(axis=1).tolist()
        outcomes = []
        simulated = 0
        for candidate in unique:
            branch_values = []
            for branch in range(config['diagnostics']['paired_terminal_branches']):
                env.restore(snapshot)
                env.set_rng(stable_seed(spec.family_id, 'diagnostic_branch', branch))
                _, _, done, info = env.step(candidate)
                simulated += info['delta']
                while not done:
                    _, _, done, info = env.step(rule_grouping(env.state()))
                    simulated += info['delta']
                branch_values.append(int(info['success']))
            outcomes.append(branch_values)
        row = dict(family_id=spec.family_id, scenario_id=spec.scenario_id, seed=seed,
            state=state.to_dict(), methods={m: unique.index(a) for m, a in choices.items() if a in unique},
            candidates=[a.to_dict() for a in unique], outcomes=outcomes, simulation_physical_steps=simulated,
            s2_scores=s2_scores,
            model_scope='first_initialization_only', continuation='one_event_then_frozen_rule')
        shared.put(key, row)
        shared.append('diagnostics', row)
        env.close()
    shared.close()
    return dict(completed=True, fixed_states=total, model_seed=seed)


def run_all(args):
    from .reporting import refresh, summarize
    config = load_config(args.config)
    run = Path(args.output or ROOT/config['storage']['output_root']).resolve()
    run.mkdir(parents=True, exist_ok=True)
    path = run/'config.json'
    if path.exists():
        retained = json.loads(path.read_text(encoding='utf-8'))
        for key in ('seeds', 'scenarios', 'environment', 'reward', 'model', 's2', 'aql', 'mappo', 'training'):
            if retained[key] != config[key]:
                raise ValueError(f'Existing V6 {key} differs; keep the original configuration when resuming')
        config = retained
    else:
        from open_score.grouping.storage import atomic_json
        atomic_json(path, config)
    shared = Store(run/'shared')
    shared.put('config', config)
    import had_env
    from had_env.core.version import CORE_VERSION, PHYSICS_PROTOCOL
    native_core = dict(core_version=CORE_VERSION, physics_protocol=PHYSICS_PROTOCOL,
                       package_path=str(Path(had_env.__file__).resolve()))
    expected_core = config['environment']['engine_source']
    if (Path(expected_core['path']).resolve() not in Path(native_core['package_path']).parents
            or CORE_VERSION != expected_core['core_version']
            or PHYSICS_PROTOCOL != expected_core['physics_protocol']):
        raise ValueError('V6 requires the configured updated HAD Workbench installation')
    previous_core = shared.get('native_core')
    if previous_core and previous_core != native_core:
        raise ValueError('The native HAD installation/protocol changed; resume with the recorded environment')
    shared.put('native_core', native_core)
    import psutil
    previous = shared.get('scheduler', {})
    try:
        if previous.get('pid'):
            active_process = psutil.Process(previous['pid'])
            if 'open_score.research_v6' in ' '.join(active_process.cmdline()):
                raise RuntimeError('This V6 experiment is already running; read its existing terminal/log')
    except (psutil.Error, TypeError):
        pass
    device = args.device or config['resources']['device']
    import torch
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Configured CUDA is unavailable; pass --device cpu explicitly to use CPU')
    cap = args.workers or config['resources']['max_heavy_jobs']
    if not 1 <= cap <= config['resources']['max_heavy_jobs']:
        raise ValueError('At most three heavy jobs are permitted')
    pending = [j for j in jobs(config) if not shared.get(f'finished/{j[0]}', {}).get('complete')]
    active, logs = {}, {}
    shared.put('runtime', dict(device=device, max_heavy_jobs=cap,
               environment_processes_per_job=config['resources']['environment_processes_per_job'],
               numerical_threads=1, version=config['version']))
    shared.put('scheduler', dict(pid=os.getpid(), started=now(), workers=cap, device=device))
    last_display = last_report = 0.
    reporter = ThreadPoolExecutor(max_workers=1, thread_name_prefix='v6-report')
    report_future = None
    try:
        while pending or active:
            # Memory pressure changes concurrency, never an algorithm or its quota.
            allowed = min(cap, config['resources']['fallback_heavy_jobs']) if psutil.virtual_memory().available < 4*1024**3 else cap
            for item in list(pending):
                job, method, seed, dependencies = item
                if len(active) >= allowed:
                    break
                if any(not shared.get(f'finished/{d}', {}).get('complete') for d in dependencies):
                    continue
                # A method owns a single log/database, so its seeds run serially.
                owner = 'shared' if method.startswith('S2') or method == 'diagnostics' else method
                if any(a[1] == owner for a in active.values()):
                    continue
                folder = run/owner
                folder.mkdir(parents=True, exist_ok=True)
                stream = (folder/'run.log').open('a', encoding='utf-8')
                process = subprocess.Popen([sys.executable, '-X', 'utf8', '-u', '-m',
                    'open_score.research_v6', '_job', '--output', str(run), '--job', job, '--device', device],
                    cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                logs[job], active[job] = stream, (process, owner)
                pending.remove(item)
            for job, (process, _) in list(active.items()):
                if process.poll() is not None:
                    logs.pop(job).close()
                    del active[job]
                    if process.returncode:
                        raise RuntimeError(f'{job} failed; see its method run.log. Completed work is retained.')
            current = time.monotonic()
            if current-last_display >= 10:
                done = sum(bool(shared.get(f'finished/{j[0]}', {}).get('complete')) for j in jobs(config))
                print(f'[{datetime.now():%H:%M:%S}] v6 jobs {done}/{len(jobs(config))}; running: {", ".join(active) or "none"}; queued {len(pending)}', flush=True)
                for job, (_, owner) in active.items():
                    seed = next(j[2] for j in jobs(config) if j[0] == job)
                    with Store(run/owner) as values:
                        progress = values.get(f'progress/{seed}', values.get(f'progress/{job}', {}))
                    details = f'  {job}: {progress.get("phase","preparing")} steps={progress.get("physical_steps","-")} completed={progress.get("completed","-")}/{progress.get("total","-")}'
                    if 'optimizer_steps' in progress:
                        details += f' updates={progress["optimizer_steps"]} train_win={progress["native_success_rate"]:.1%} speed={progress["physical_steps_per_second"]:.2f} steps/s'
                    validation = progress.get('latest_validation')
                    if validation:
                        details += f' validation_A-D={validation["wins"]}/{validation["episodes"]}'
                    if 'provisional_win_rate' in progress:
                        details += f' current_{progress["phase"]}={progress["wins"]}/{progress["completed"]}'
                    eta = progress.get('remaining_seconds', progress.get('remaining_training_seconds'))
                    if eta is not None:
                        details += f' phase_remaining~{max(0, eta)/3600:.2f}h'
                    if 'validation_brier' in progress:
                        details += f' validation_Brier={progress["validation_brier"]:.5f}'
                    print(details, flush=True)
                last_display = current
            if current-last_report >= 10 and (report_future is None or report_future.done()):
                if report_future is not None:
                    try:
                        report_future.result()
                    except Exception as error:
                        print(f'[report] Refresh failed; retrying: {error}', flush=True)
                report_future = reporter.submit(refresh, run)
                last_report = current
            if active:
                time.sleep(.5)
            elif pending:
                raise RuntimeError('Unresolved job dependency; no work was silently skipped')
        shared.put('run_result', dict(complete=True, finished=now()))
        reporter.submit(summarize, run).result()
    finally:
        for process, _ in active.values():
            try:
                children = psutil.Process(process.pid).children(recursive=True)
                for child in children:
                    child.terminate()
                process.terminate()
            except psutil.Error:
                pass
        for stream in logs.values():
            stream.close()
        reporter.shutdown(wait=True)
        shared.put('scheduler', dict(pid=None, stopped=now()))
        shared.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run-all', '_job'])
    parser.add_argument('--config')
    parser.add_argument('--output')
    parser.add_argument('--workers', type=int)
    parser.add_argument('--device')
    parser.add_argument('--job', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    configure_threads()
    if args.command == '_job':
        worker(Path(args.output), args.job, args.device)
    else:
        run_all(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
