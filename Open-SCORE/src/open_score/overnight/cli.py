"""Worker entry point used by the eight-hour, three-route portfolio runner."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time


@contextmanager
def route_lock(output):
    stream = (output/'.worker.lock').open('a+b')
    if stream.seek(0, os.SEEK_END) == 0:
        stream.write(b'0')
        stream.flush()
    stream.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        stream.close()
        raise RuntimeError(f'Another worker is using {output}') from None
    try:
        yield
    finally:
        stream.close()


def main():
    import torch
    from open_score.grouping.storage import atomic_json, fingerprint, sha256
    from .config import ROUTES, configuration
    from .training import Trainer
    from .evaluation import evaluate_models

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['worker'])
    parser.add_argument('--route', required=True, choices=ROUTES)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--train-seconds', type=float, default=25200)
    parser.add_argument('--eval-seconds', type=float, default=2700)
    parser.add_argument('--seed', type=int, default=20260906)
    parser.add_argument('--steps', type=int, default=0)
    parser.add_argument('--checkpoint-every', type=int, default=25000)
    parser.add_argument('--eval-episodes', type=int, default=100)
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if min(args.train_seconds,args.eval_seconds,args.checkpoint_every,args.eval_episodes) <= 0 or args.steps < 0:
        parser.error('Positive wall budgets and counts, steps>=0 required')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if device=='cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable')
    config = configuration(args.route, seed=args.seed, smoke=args.smoke, device=device)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with route_lock(output):
        request = dict(train_seconds=args.train_seconds, eval_seconds=args.eval_seconds,
                       steps=args.steps, eval_episodes=args.eval_episodes,
                       checkpoint_every=args.checkpoint_every, config_hash=fingerprint(config))
        request_path = output/'worker_request.json'
        if request_path.exists():
            if not args.resume:
                raise FileExistsError('Existing worker output; use --resume or a new directory')
            if json.loads(request_path.read_text(encoding='utf-8')) != request:
                raise ValueError('Worker budgets or configuration changed; use a new directory')
        elif (output/'latest.pt').exists():
            raise ValueError('Checkpoint exists without worker budget identity; use its original entry point')
        else:
            atomic_json(request_path, request)
        trainer = Trainer(config, output, seconds=args.train_seconds, steps=args.steps,
                          resume=args.resume, checkpoint_every=args.checkpoint_every)
        training_path = output/'training_result.json'
        if args.resume and training_path.exists():
            training = json.loads(training_path.read_text(encoding='utf-8'))
        else:
            training = trainer.run()
            if training['stopped']:
                atomic_json(output/'route_result.json', dict(status='interrupted', route=args.route,
                    actual_steps=training['physical_steps'], training_seconds=training['training_seconds']))
                return 130
            atomic_json(training_path, training)
        from .reporting import plot_training
        plot_training(output)
        checkpoints = {'latest':output/'latest.pt', 'static':None, 'dynamic_rule':None,
                       'compact_rule':None, 'all_reserve':None}
        if (output/'best.pt').exists():
            checkpoints = {'best':output/'best.pt', **checkpoints}
        identity = fingerprint({k:sha256(v) if v is not None else k for k,v in checkpoints.items()})[:12]
        destination = output/'evaluation'/identity
        atomic_json(output/'progress.json', dict(phase='evaluate', actual_steps=training['physical_steps'],
                   evaluation_directory=str(destination), device=device))
        summary = evaluate_models(config, checkpoints, destination, episodes=args.eval_episodes,
                                  wall_seconds=args.eval_seconds)
        result = dict(status='complete' if summary['complete'] else 'partial_evaluation',
            route=args.route, actual_steps=training['physical_steps'],
            teacher_simulation_steps=training['teacher_simulation_steps'],
            training_seconds=training['training_seconds'], total_invocation_seconds=time.monotonic()-started,
            parameters=training['parameters'], config=config, evaluation=summary,
            evaluation_directory=str(destination), training_seed=args.seed,
            note='Three alternative routes, one training seed each; no multi-seed robustness claim. Native success is the primary outcome.')
        atomic_json(output/'route_result.json', result)
        atomic_json(output/'progress.json', dict(phase='complete' if summary['complete'] else 'partial_evaluation', **result))
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0 if summary['complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
