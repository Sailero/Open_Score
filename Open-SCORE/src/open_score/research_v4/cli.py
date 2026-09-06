"""Commands for preparing, training, comparing, and inspecting v4 research."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from .runner import (ROUTES, atomic_json, directory_lock, file_hash, now, read_json, resolve_device,
                     route_output, run_pool, validate_manifest)


CONTROLS = ('rule', 'grand', 'static_rule', 'random', 'b2_count', 'b2_min',
            'b3_rebuild', 'group_only', 'target_only')


def parser():
    result = argparse.ArgumentParser(description='Common-rule-executor Blotto/S2/RL comparison. No frozen S1 is used.')
    result.add_argument('command', choices=('prepare', 'train', 'evaluate', 'run', 'diagnose', 'benchmark', 'report'))
    result.add_argument('--output', type=Path, default=Path('outputs/v4_comparison'))
    result.add_argument('--route', choices=(*ROUTES, *CONTROLS), default='b3_global')
    result.add_argument('--routes', nargs='+', choices=ROUTES, default=list(ROUTES))
    result.add_argument('--seeds', nargs='+', type=int, default=[20260906])
    result.add_argument('--workers', type=int, default=6)
    result.add_argument('--data-workers', type=int, default=None, help='Shared S2 collection workers, at most 6; defaults to min(workers, 6), or 2 in smoke.')
    result.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    result.add_argument('--steps', type=int, default=None, help='RL physical-step target; 0 uses the time budget. May increase on resume.')
    result.add_argument('--train-hours', type=float, default=None, help='Per learner budget, including teacher and validation; default 6.5 hours.')
    result.add_argument('--total-hours', type=float, default=0, help='Optional task deadline measured from preparation start; preparation itself uses --collect-seconds/--s2-seconds. 0 has no parent cutoff.')
    result.add_argument('--eval-episodes', type=int, default=None, help='Held-out episodes PER SCALE (100 normally; 2 in smoke).')
    result.add_argument('--scales', nargs='+', type=int, default=None)
    result.add_argument('--search-budget', type=int, default=None)
    result.add_argument('--opponent', default='reactive')
    result.add_argument('--smoke', action='store_true', help='Small data and short learner validation; does not establish efficacy.')
    result.add_argument('--resume', action='store_true')
    result.add_argument('--dry-run', action='store_true', help='Print the resolved request without creating an output directory.')
    result.add_argument('--shared-dir', type=Path, default=None)
    result.add_argument('--controls', nargs='*', choices=CONTROLS, default=['rule', 'grand', 'static_rule', 'random'])
    result.add_argument('--s2-families', type=int, default=None)
    result.add_argument('--s2-local-families', type=int, default=None)
    result.add_argument('--s2-candidates', type=int, default=None)
    result.add_argument('--s2-branches', type=int, default=None)
    result.add_argument('--s2-states', type=int, default=None)
    result.add_argument('--s2-epochs', type=int, default=None)
    result.add_argument('--s2-seconds', type=float, default=None)
    result.add_argument('--collect-seconds', type=float, default=None)
    result.add_argument('--benchmark-workers', nargs='+', type=int, default=[3, 6])
    result.add_argument('--benchmark-steps', type=int, default=1024)
    result.add_argument('--benchmark-repeats', type=int, default=1)
    result.add_argument('--skip-prepare', action='store_true', help=argparse.SUPPRESS)
    result.add_argument('--evaluate-after', action='store_true', help=argparse.SUPPRESS)
    return result


def request_config(args, route=None, seed=None):
    from .config import configuration
    device = resolve_device(args.device)
    overrides = {'opponent': args.opponent, 'data_workers': args.data_workers if args.data_workers is not None else min(args.workers, 2 if args.smoke else 6),
                 'eval_episodes': args.eval_episodes if args.eval_episodes is not None else (2 if args.smoke else 100),
                 'steps': args.steps if args.steps is not None else (256 if args.smoke else 100000),
                 'seconds': (args.train_hours * 3600) if args.train_hours is not None else (60 if args.smoke else 6.5 * 3600),
                 'scales': args.scales if args.scales is not None else ([4, 8] if args.smoke else [8, 12, 16, 24, 32]),
                 'search_budget': args.search_budget if args.search_budget is not None else (8 if args.smoke else 64)}
    for name in ('s2_families', 's2_local_families', 's2_candidates', 's2_branches', 's2_states',
                 's2_epochs', 's2_seconds', 'collect_seconds'):
        value = getattr(args, name)
        if value is not None:
            overrides[name] = value
    if args.collect_seconds is not None:
        overrides['s2_collect_seconds'] = args.collect_seconds
        overrides.pop('collect_seconds', None)
    for name in ('train_scales', 'eval_scales', 'validation_scales'):
        overrides[name] = overrides['scales']
    selected_route = route or args.route
    config = configuration(route=selected_route if selected_route in ROUTES else 'b3_global',
                           seed=seed if seed is not None else args.seeds[0],
                           smoke=args.smoke, device=device, **overrides)
    config['route'] = selected_route
    return config


def _identity(config):
    from .config import protocol, provenance
    shared = dict(protocol(config))
    for key in ('route', 'seed', 'seeds', 'shared_dir', 'assets', 's2_seconds', 's2_collect_seconds', 'data_workers'):
        shared.pop(key, None)
    source = provenance()
    return {'protocol': shared, 'source_hash': source['source_hash'],
            'executor_version': source['executor_version']}


def _manifest(args, config):
    from .config import provenance
    result = validate_manifest(args.output, _identity(config), resume=args.resume,
        request={'command': args.command, 'routes': args.routes if args.command == 'run' else [args.route],
                 'controls': args.controls if args.command == 'run' else [],
                 'seeds': args.seeds, 'workers': args.workers, 'steps': config['steps'],
                 'seconds': config['seconds'], 'eval_episodes': config['eval_episodes']})
    atomic_json(args.output / 'source_provenance.json', provenance())
    return result


def shared_folder(args, seed):
    return (args.shared_dir or args.output / 'shared' / f'seed_{seed}').resolve()


def shared_assets(args, config, *, prepare=True):
    folder = shared_folder(args, config['seed'])
    prepared_identity = {'training_seed': int(config['seed']), 'identity': _identity(config)}
    previous = read_json(folder / 'preparation_request.json')
    if previous is not None and previous != prepared_identity:
        raise ValueError('Shared evaluator training seed or source/protocol differs from this route')
    if prepare:
        from .data import prepare as prepare_data
        result = prepare_data(config, folder, resume=args.resume or args.shared_dir is not None)
        atomic_json(folder / 'preparation_request.json', prepared_identity)
        atomic_json(folder / 'prepared_assets.json', result)
        if result.get('status') != 'complete':
            raise RuntimeError('Shared preparation is partial; increase its collection budget and resume before training')
        return {key: value for key, value in result.items() if key.endswith('_checkpoint')}
    result = read_json(folder / 'prepared_assets.json')
    if result is None:
        if config['route'] == 'r1_ppo' or config['route'] == 'r3_ddqn':
            return {}
        raise FileNotFoundError(f'Shared preparation is missing: {folder}; run prepare first')
    if previous is None:
        raise ValueError('Shared evaluator preparation identity is missing; cannot establish its training seed')
    if result.get('status') != 'complete':
        raise RuntimeError('Shared preparation is partial; complete prepare before running dependent routes')
    return {key: value for key, value in result.items() if key.endswith('_checkpoint')}


def _train_one(args, config, assets):
    from .reporting import plot_training
    folder = route_output(args.output, config['route'], config['seed'])
    with directory_lock(folder, '.route.lock'):
        config = {**config, 'shared_dir': str(shared_folder(args, config['seed'])), 'assets': assets}
        route = config['route']
        if route == 'r2_teacher_ppo':
            shared = Path(config['shared_dir']) / 'global_data' / 'families'
            dependencies = {path.name: file_hash(path) for path in sorted(shared.glob('*.json'))}
        elif route in ('r1_ppo', 'r3_ddqn'):
            dependencies = {}
        else:
            dependencies = {key: file_hash(value) for key, value in assets.items() if key.endswith('_checkpoint')}
        identity = {'run': _identity(config), 'route': config['route'], 'seed': config['seed'],
                    'assets': dependencies}
        previous_identity = read_json(folder / 'runner_identity.json')
        if previous_identity is not None:
            if not args.resume:
                raise ValueError(f'Route output exists; use --resume: {folder}')
            changing_solver_model = (route.startswith('b') and previous_identity.get('run') == identity['run']
                                     and previous_identity.get('seed') == identity['seed']
                                     and previous_identity.get('route') == identity['route'])
            if previous_identity != identity and not changing_solver_model:
                raise ValueError('Route source/protocol/assets changed; use a new output directory')
        atomic_json(folder / 'runner_identity.json', identity)
        atomic_json(folder / 'request.json', config)
        atomic_json(folder / 'status.json', {'state': 'training', 'started': now(), 'route': config['route']})
        try:
            if config['route'].startswith('r') and config['route'] in ROUTES:
                from .training import train
                previous = read_json(folder / 'training_result.json', {}) if args.resume else {}
                counts = previous.get('counters', {})
                reached_steps = config['steps'] > 0 and counts.get('physical_steps', 0) >= config['steps']
                reached_time = previous.get('training_seconds', 0) >= config['seconds']
                if previous and (folder / 'latest.pt').exists() and (reached_steps or reached_time):
                    result = previous
                else:
                    learner_config = {key: value for key, value in config.items()
                                      if not key.startswith('s2_') and key not in ('assets', 'collect_seconds', 'data_workers')}
                    result = train(learner_config, folder, resume=args.resume)
                plot_training(folder)
            else:
                from .policies import make_policy
                make_policy(config['route'], assets, device=config['device'], search_budget=config['search_budget'], seed=config['seed'])
                result = {'route': config['route'], 'seed': config['seed'], 'state': 'ready',
                          'policy_training_required': False, 'shared_preparation': config['shared_dir'],
                          'note': 'This solver consumes a shared outcome/payoff model; no separate policy training is performed.'}
                atomic_json(folder / 'training_result.json', result)
            if args.evaluate_after:
                from .evaluation import evaluate
                atomic_json(folder / 'status.json', {'state': 'evaluating', 'updated': now(), 'route': config['route']})
                evaluate(config, folder, assets=assets, resume=args.resume)
            atomic_json(folder / 'status.json', {'state': 'complete', 'finished': now(), 'route': config['route']})
            return result
        except BaseException as error:
            atomic_json(folder / 'status.json', {'state': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                                               'updated': now(), 'error': repr(error), 'route': config['route']})
            raise


def _child_command(args, config, route, seed, *, command='train'):
    script = Path(__file__).resolve().parents[3] / 'scripts' / 'run_research_comparison.py'
    argv = [sys.executable, str(script), command, '--output', str(args.output.resolve()),
            '--route', route, '--seeds', str(seed), '--device', config['device'],
            '--steps', str(config['steps']), '--train-hours', str(config['seconds'] / 3600),
            '--eval-episodes', str(config['eval_episodes']), '--search-budget', str(config['search_budget']),
            '--data-workers', str(config.get('data_workers', 1)),
            '--opponent', config.get('opponent', 'reactive'), '--scales', *map(str, config['scales']),
            '--shared-dir', str(shared_folder(args, seed)), '--skip-prepare']
    if command == 'train':
        argv.append('--evaluate-after')
    if args.smoke:
        argv.append('--smoke')
    if args.resume:
        argv.append('--resume')
    for name in ('s2_families', 's2_local_families', 's2_candidates', 's2_branches', 's2_states', 's2_epochs', 's2_seconds', 'collect_seconds'):
        if getattr(args, name) is not None:
            argv += ['--' + name.replace('_', '-'), str(getattr(args, name))]
    return argv


def benchmark(args):
    """Measure actual short CPU PPO training, including updates and checkpoint I/O."""
    from .reporting import report
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    trials = []
    for repeat in range(args.benchmark_repeats):
        for workers in args.benchmark_workers:
            folder = root / f'workers_{workers}_repeat_{repeat}'
            jobs = []
            for index in range(workers):
                target = folder / f'job_{index}'
                script = Path(__file__).resolve().parents[3] / 'scripts' / 'run_research_comparison.py'
                command = [sys.executable, str(script), 'train', '--route', 'r1_ppo', '--device', 'cpu',
                           '--output', str(target), '--seeds', str(70_000_000 + repeat * 100 + index),
                           '--steps', str(args.benchmark_steps), '--train-hours', str(120 / 3600),
                           '--scales', '8', '--skip-prepare', '--smoke']
                if args.resume:
                    command.append('--resume')
                jobs.append({'id': f'cpu_{index}', 'command': command})
            started = time.monotonic()
            result = run_pool(jobs, workers=workers, output=folder)
            duration = time.monotonic() - started
            records = [read_json(path) for path in folder.glob('job_*/r1_ppo/seed_*/training_result.json')]
            physical = sum(record.get('counters', record).get('physical_steps', 0) for record in records)
            trials.append({'workers': workers, 'repeat': repeat, 'seconds': duration,
                           'physical_steps': physical, 'aggregate_steps_per_second': physical / duration,
                           'pool': result, 'training_results': records})
            atomic_json(root / 'benchmark.json', {'trials': trials,
                'protocol': 'CPU PPO, smoke network, real simulation and optimizer updates; not a seven-hour production prediction.'})
    return {'trials': trials, 'note': 'Compare aggregate throughput; this short smoke-network measurement does not predict all six heterogeneous routes.'}


def main(argv=None):
    args = parser().parse_args(argv)
    if args.workers < 1 or any(seed < 0 for seed in args.seeds):
        raise ValueError('workers must be positive and seeds nonnegative')
    if args.data_workers is not None and not 1 <= args.data_workers <= 6:
        raise ValueError('data-workers must be between 1 and 6')
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Training seeds must be unique')
    if args.shared_dir is not None and len(args.seeds) != 1:
        raise ValueError('An explicit shared directory belongs to exactly one training seed')
    if args.steps is not None and args.steps < 0:
        raise ValueError('steps must be nonnegative')
    if args.eval_episodes is not None and args.eval_episodes < 1:
        raise ValueError('eval-episodes must be positive')
    if args.scales is not None and any(scale < 1 for scale in args.scales):
        raise ValueError('scales must be positive')
    if args.search_budget is not None and args.search_budget < 1:
        raise ValueError('search-budget must be positive')
    if args.train_hours is not None and args.train_hours <= 0:
        raise ValueError('train-hours must be positive')
    if args.total_hours < 0 or args.benchmark_steps < 1:
        raise ValueError('total-hours must be nonnegative and benchmark-steps positive')
    for key in ('s2_families', 's2_local_families', 's2_candidates', 's2_branches', 's2_states', 's2_epochs', 's2_seconds', 'collect_seconds'):
        value = getattr(args, key)
        if value is not None and value <= 0:
            raise ValueError(f'{key.replace("_", "-")} must be positive')
    if args.dry_run and args.command in ('benchmark', 'report'):
        print(json.dumps({'command': args.command, 'output': str(args.output.resolve()),
            'benchmark_workers': args.benchmark_workers, 'benchmark_steps': args.benchmark_steps,
            'benchmark_repeats': args.benchmark_repeats}, indent=2))
        return 0
    if args.command == 'report':
        from .reporting import report
        print(json.dumps(report(args.output), ensure_ascii=False, indent=2))
        return 0
    if args.command == 'benchmark':
        if args.benchmark_repeats < 1 or any(value < 1 for value in args.benchmark_workers):
            raise ValueError('benchmark repeats and worker counts must be positive')
        result = benchmark(args)
        print(json.dumps({'trials': [{key: value for key, value in trial.items()
                         if key not in ('pool', 'training_results')} for trial in result['trials']],
                          'note': result['note']}, ensure_ascii=False, indent=2))
        return 0
    config = request_config(args)
    if args.dry_run:
        print(json.dumps({'command': args.command, 'config': config, 'output': str(args.output.resolve()),
                          'routes': args.routes, 'seeds': args.seeds, 'workers': args.workers,
                          'identity': _identity(config)}, ensure_ascii=False, indent=2))
        return 0
    if args.command == 'diagnose':
        from .diagnostics import run_diagnostics
        print(json.dumps(run_diagnostics(args.output, episodes=config['eval_episodes'],
                         scales=tuple(config['scales']), seed=config['seed'], opponent=args.opponent), ensure_ascii=False, indent=2))
        return 0
    if args.skip_prepare:
        # Parent owns the root manifest and preparation. Children own only their route directory.
        assets = shared_assets(args, config, prepare=False)
        if args.command == 'train':
            for seed in args.seeds:
                _train_one(args, request_config(args, seed=seed), assets)
        elif args.command == 'evaluate':
            from .evaluation import evaluate
            for seed in args.seeds:
                current = request_config(args, seed=seed)
                folder = route_output(args.output, args.route, seed)
                with directory_lock(folder, '.route.lock'):
                    evaluate(current, folder, assets=assets, resume=args.resume)
        else:
            raise ValueError('--skip-prepare is for train/evaluate workers only')
        return 0
    with directory_lock(args.output):
        _manifest(args, config)
        started = time.monotonic()
        assets_by_seed = {}
        for seed in args.seeds:
            current = request_config(args, seed=seed)
            atomic_json(args.output / 'preparation_status.json', {'state': 'preparing', 'training_seed': seed,
                'completed_seeds': list(assets_by_seed), 'updated': now()})
            assets_by_seed[seed] = shared_assets(args, current, prepare=args.command != 'evaluate')
        atomic_json(args.output / 'preparation_status.json', {'state': 'complete',
            'completed_seeds': list(assets_by_seed), 'updated': now()})
        if args.command == 'prepare':
            print(json.dumps(assets_by_seed, ensure_ascii=False, indent=2))
            return 0
        if args.command == 'train':
            for seed in args.seeds:
                _train_one(args, request_config(args, seed=seed), assets_by_seed[seed])
        elif args.command == 'evaluate':
            from .evaluation import evaluate
            for seed in args.seeds:
                current = request_config(args, seed=seed)
                folder = route_output(args.output, args.route, seed)
                with directory_lock(folder, '.route.lock'):
                    evaluate(current, folder, assets=assets_by_seed[seed], resume=args.resume)
        else:
            jobs = [{'id': f'{route}_{seed}', 'command': _child_command(args, config, route, seed)}
                    for seed in args.seeds for route in args.routes]
            jobs.extend({'id': f'{route}_{seed}', 'command': _child_command(args, config, route, seed, command='evaluate')}
                        for seed in args.seeds for route in args.controls)
            deadline = started + args.total_hours * 3600 if args.total_hours > 0 else None
            result = run_pool(jobs, workers=args.workers, output=args.output, deadline=deadline)
            atomic_json(args.output / 'run_result.json', result)
            from .reporting import report
            report(args.output)
            return 1 if result['failed'] or result['interrupted'] else 0
    from .reporting import report
    report(args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
