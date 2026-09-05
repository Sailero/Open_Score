"""One entry point for independently trainable known-opponent grouping."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import torch
import yaml

from .evaluation import evaluate, write_report
from .storage import ROOT, atomic_json, fingerprint, read_jsonl, sha256, provenance
from .training import METHODS, contract, train


def parser():
    result = argparse.ArgumentParser(description='Known-opponent dynamic grouping v2.0 (50-step episodes).')
    result.add_argument('command', choices=('run', 'train', 'evaluate', 'diagnose', 'compare', 'report'))
    result.add_argument('--config', type=Path, default=ROOT / 'configs/known_opponent_v2.yaml')
    result.add_argument('--profile', choices=('smoke', 'briefing', 'minimal', 'formal'), default='briefing')
    result.add_argument('--method', choices=METHODS)
    result.add_argument('--opponent', choices=('reactive', 'concentrated', 'balanced'))
    result.add_argument('--seed', type=int)
    result.add_argument('--device', choices=('cpu', 'cuda'))
    result.add_argument('--steps', type=int, help='Additional physical interactions; 0 means time budget only.')
    result.add_argument('--train-seconds', type=float, help='Additional training wall-time budget; includes sampling and updates.')
    result.add_argument('--eval-seconds', type=float)
    result.add_argument('--eval-episodes', type=int)
    result.add_argument('--train-scales', type=int, nargs='+')
    result.add_argument('--eval-scales', type=int, nargs='+')
    result.add_argument('--output', type=Path, default=ROOT / 'outputs/v2_main')
    result.add_argument('--checkpoint', type=Path)
    result.add_argument('--resume', action='store_true', help='Resume compatible completed updates; start new episodes.')
    result.add_argument('--methods', nargs='+', choices=METHODS, help='Methods for compare; defaults to B0-B6.')
    result.add_argument('--diagnostic-states', type=int, default=3)
    result.add_argument('--diagnostic-rollouts', type=int, default=2)
    result.add_argument('--diagnostic-seconds', type=float, default=120)
    result.add_argument('--all-seeds', action='store_true', help='Compare all configured formal training seeds.')
    result.add_argument('--all-opponents', action='store_true', help='Also separately train configured review opponents.')
    return result


def get_config(args):
    raw = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    config = {key: value for key, value in raw.items() if key != 'profiles'}
    config.update(raw['profiles'][args.profile])
    for key in ('method', 'opponent', 'seed', 'device', 'steps', 'train_seconds', 'eval_seconds',
                'eval_episodes', 'train_scales', 'eval_scales'):
        if getattr(args, key, None) is not None:
            config[key] = getattr(args, key)
    if config['steps'] == 0:
        config['steps'] = None
    if config['max_steps'] != 50:
        raise ValueError('Registered v2 task uses max_steps=50; do not change the task to shorten training.')
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable. Use --device cpu.')
    torch.set_num_threads(int(config.get('torch_threads', 1)))
    return config


def eval_training(config, output, checkpoint=None):
    output = Path(output)
    checkpoint = checkpoint or output / 'latest.pt'
    if not checkpoint.exists():
        raise FileNotFoundError(f'Train a model first: {checkpoint}')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    # The task attached to the checkpoint is authoritative, including its opponent.
    evaluation_config = dict(payload['config'], device=config['device'])
    checkpoints = {'trained': checkpoint}
    initial = checkpoint.parent / 'initialized.pt'
    if initial.exists():
        checkpoints = {'initialized': initial, **checkpoints}
    result_dir = output / f"eval_step_{payload['counters']['physical_steps']}"
    result = evaluate(evaluation_config, checkpoints, result_dir, scales=config['eval_scales'],
                      episodes=config['eval_episodes'], wall_seconds=config.get('eval_seconds'), matched_static=True)
    atomic_json(output / 'latest_evaluation.json', {'directory': str(result_dir.resolve()), **result})
    return result_dir, result


def comparison(config, args):
    from .diagnostics import run_diagnostics
    methods = args.methods or list(METHODS)
    # B3's count distribution is estimated only from B4 training, never testing.
    train_order = ['selective', 'full', 'random', 'rule', 'alma']
    seeds = config.get('training_seeds', [config['seed']]) if args.all_seeds else [config['seed']]
    opponents = [config['opponent']] + (config.get('review_opponents', []) if args.all_opponents else [])
    results = []
    for opponent in dict.fromkeys(opponents):
        selected_methods = methods if opponent == config['opponent'] else config.get('review_methods', ['dlom', 'full', 'selective'])
        for seed in seeds:
            base = args.output / opponent / f'seed_{seed}'
            run_config = dict(config, opponent=opponent, seed=seed)
            checkpoints, release_distribution = {}, None
            for method in train_order:
                if method not in selected_methods:
                    continue
                current = copy.deepcopy(dict(run_config, method=method))
                if method == 'random':
                    if release_distribution is None:
                        raise ValueError('B3 comparison requires B4 training in the same invocation to freeze release counts')
                    current['release_distribution'] = release_distribution
                target_dir = base / method
                prior = target_dir / 'latest.pt'
                remaining_steps, remaining_seconds = config['steps'], config.get('train_seconds')
                status = None
                if args.resume and prior.exists():
                    payload = torch.load(prior, map_location='cpu', weights_only=False)
                    evidence = provenance()
                    if (payload['config_hash'] != fingerprint(contract(current))
                            or payload['provenance']['source_hash'] != evidence['source_hash']
                            or payload['provenance']['assets'] != evidence['assets']):
                        raise ValueError('Comparison resume task, source or frozen assets differ')
                    counts = payload['counters']
                    remaining_steps = None if config['steps'] is None else max(0, config['steps'] - counts['physical_steps'])
                    remaining_seconds = None if config.get('train_seconds') is None else max(0., config['train_seconds'] - counts['training_seconds'])
                    if remaining_steps == 0 or remaining_seconds == 0:
                        status = {**counts, 'release_counts': payload['release_counts']}
                        print(f'[compare] retain completed {opponent}/{seed}/{method}', flush=True)
                if status is None:
                    status = train(current, target_dir, steps=remaining_steps,
                                   wall_seconds=remaining_seconds, resume=args.resume)
                checkpoints[method] = base / method / 'latest.pt'
                if method == 'selective':
                    release_distribution = status['release_counts'] or {0: 1}
            for method in selected_methods:
                if method in ('static', 'dlom'):
                    checkpoints[method] = None
            evaluation_id = fingerprint({name: sha256(path) if path else name for name, path in checkpoints.items()})[:12]
            evaluation_dir = base / 'comparison' / evaluation_id
            summary = evaluate(run_config, checkpoints, evaluation_dir, scales=config['eval_scales'],
                               episodes=config['eval_episodes'], wall_seconds=config.get('eval_seconds'), matched_static=False)
            results.append({'seed': seed, 'opponent': opponent, 'evaluation_directory': str(evaluation_dir), **summary})
            diagnostic_dir = base / 'diagnostics' / evaluation_id
            previous_diagnostic = diagnostic_dir / 'diagnostics.json'
            reuse_diagnostic = False
            if args.resume and previous_diagnostic.exists():
                stored = json.loads(previous_diagnostic.read_text(encoding='utf-8'))
                reuse_diagnostic = (stored['status'] == 'complete' and stored['requested_states'] == args.diagnostic_states
                                    and stored['requested_rollouts'] == args.diagnostic_rollouts)
            if not reuse_diagnostic:
                run_diagnostics(run_config, diagnostic_dir, checkpoint=checkpoints.get('selective'),
                                states=args.diagnostic_states, rollouts=args.diagnostic_rollouts,
                                wall_seconds=args.diagnostic_seconds)
            atomic_json(args.output / 'comparison_runs.json', results)
    aggregate_comparison(args.output, results)
    return results


def aggregate_comparison(output, runs):
    """Resample training-seed means, not pooled episodes, when multiple seeds exist."""
    import numpy as np
    grouped = {}
    for run in runs:
        for row in run['table']:
            key = (run['opponent'], row['scale'], row['method'])
            grouped.setdefault(key, []).append({'seed': run['seed'], 'success_rate': row['success_rate']})
    result = []
    rng = np.random.default_rng(20260905)
    for (opponent, scale, method), values in grouped.items():
        means = np.asarray([row['success_rate'] for row in values])
        interval = None
        if len(means) > 1:
            samples = rng.choice(means, size=(2000, len(means)), replace=True).mean(axis=1)
            interval = np.quantile(samples, [.025, .975]).tolist()
        result.append({'opponent': opponent, 'scale': scale, 'method': method,
                       'mean_success_rate': float(means.mean()), 'training_seeds': values,
                       'across_training_seed_interval': interval})
    atomic_json(Path(output) / 'comparison_summary.json', result)
    lines = ['# 已知对手核心比较', '', '单种子结果不提供跨训练种子区间；不同对手分别训练。', '',
             '| 对手 | 人数/方 | 方法 | 训练种子数 | 平均成功率 |', '|---|---:|---|---:|---:|']
    for row in result:
        lines.append(f"| {row['opponent']} | {row['scale']} | {row['method']} | {len(row['training_seeds'])} | {row['mean_success_rate']:.1%} |")
    (Path(output) / 'comparison_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main(argv=None):
    args = parser().parse_args(argv)
    config = get_config(args)
    args.output = args.output.resolve()
    started = time.perf_counter()
    if args.command in ('train', 'run'):
        result = train(config, args.output, steps=config['steps'], wall_seconds=config.get('train_seconds'), resume=args.resume)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.command == 'run':
            directory, _ = eval_training(config, args.output)
            print(f'Report: {directory / "briefing.md"}')
    elif args.command == 'evaluate':
        directory, result = eval_training(config, args.output, args.checkpoint)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f'Report: {directory / "briefing.md"}')
    elif args.command == 'diagnose':
        from .diagnostics import run_diagnostics
        result = run_diagnostics(config, args.output, checkpoint=args.checkpoint,
                                 states=args.diagnostic_states, rollouts=args.diagnostic_rollouts,
                                 wall_seconds=args.diagnostic_seconds)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == 'compare':
        comparison(config, args)
        print(f'Report: {args.output / "comparison_report.md"}')
    elif args.command == 'report':
        write_report(args.output)
        print(f'Report: {args.output / "briefing.md"}')
    print(f'Total wall time: {time.perf_counter() - started:.1f}s', flush=True)


if __name__ == '__main__':
    main()
