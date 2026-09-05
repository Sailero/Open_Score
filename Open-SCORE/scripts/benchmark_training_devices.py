"""Short, isolated end-to-end PPO device/batch benchmarks, not learning evaluations."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    import torch
    import yaml
    from open_score.grouping.storage import atomic_json
    from open_score.grouping.training import train

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--seconds', type=float, default=45)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory; benchmarks never resume training runs.')
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    raw = yaml.safe_load((ROOT / 'configs/known_opponent_v2.yaml').read_text(encoding='utf-8'))
    config = {k: v for k, v in raw.items() if k != 'profiles'}
    config.update(raw['profiles']['minimal'])
    rows = []
    for batch_size in (16, 64):
        for device, executor in (('cpu', 'cpu'), ('cuda', 'cpu'), ('cuda', 'cuda')):
            if device == 'cuda' and not torch.cuda.is_available():
                continue
            current = copy.deepcopy(config)
            current.update(device=device, executor_device=executor)
            current['ppo']['batch_size'] = batch_size
            name = f'{device}_{executor}_batch{batch_size}'
            if device == 'cuda':
                torch.ones((128, 128), device='cuda').square().sum().item()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            result = train(current, args.output / name, steps=args.steps, wall_seconds=args.seconds)
            if device == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            row = dict(name=name, device=device, executor_device=executor, batch_size=batch_size,
                       physical_steps=result['invocation_steps'], updates=result['updates'],
                       total_seconds=elapsed, total_steps_per_second=result['invocation_steps']/elapsed,
                       training_steps_per_second=result['steps_per_second'],
                       peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20 if device == 'cuda' else 0)
            rows.append(row)
            atomic_json(args.output / 'benchmark.json', dict(torch=torch.__version__,
                        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                        requested_steps=args.steps, seconds_cap=args.seconds, results=rows,
                        note='Short sequential throughput probe; initializations and numerical trajectories differ. Not a convergence or parallel scaling test.'))
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
