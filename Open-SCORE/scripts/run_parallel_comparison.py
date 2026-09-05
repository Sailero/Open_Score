"""Run selective/rule comparisons in isolated processes with a shared deadline."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
METHODS = ('selective', 'rule')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    from open_score.grouping.storage import atomic_json
    atomic_json(path, value)


def milestones(steps, interval):
    return sorted(set(range(interval, steps + 1, interval)) | {steps})


def checkpoint_path(folder, step):
    return Path(folder) / 'budgets' / f'step_{step:09d}.pt'


def checked_checkpoint(path, config):
    import torch
    from open_score.grouping.storage import fingerprint, provenance
    from open_score.grouping.training import contract
    payload = torch.load(path, map_location='cpu', weights_only=False)
    evidence = provenance()
    if (payload['config_hash'] != fingerprint(contract(config))
            or payload['provenance']['source_hash'] != evidence['source_hash']
            or payload['provenance']['assets'] != evidence['assets']):
        raise ValueError(f'Incompatible checkpoint: {path}')
    return payload


def train_worker(job):
    from open_score.grouping.training import train
    config, folder = job['config'], Path(job['directory'])
    latest = folder / 'latest.pt'
    started = time.perf_counter()
    for target in job['milestones']:
        saved = checkpoint_path(folder, target)
        if saved.exists():
            payload = checked_checkpoint(saved, config)
            if not target <= payload['counters']['physical_steps'] <= target + 4:
                raise ValueError(f'Incorrect physical-step milestone: {saved}')
            continue
        counts = {'physical_steps': 0, 'training_seconds': 0.0}
        if latest.exists():
            counts = checked_checkpoint(latest, config)['counters']
        remaining = min(job['train_seconds'] - counts['training_seconds'],
                        job['available_seconds'] - (time.perf_counter() - started))
        if counts['physical_steps'] < target:
            if remaining <= 0:
                break
            train(config, folder, steps=target - counts['physical_steps'],
                  wall_seconds=remaining, resume=latest.exists())
            counts = checked_checkpoint(latest, config)['counters']
        if counts['physical_steps'] < target:
            break
        if counts['physical_steps'] > target + 4:
            raise ValueError('Cannot reconstruct a missing historical milestone from a later model')
        saved.parent.mkdir(parents=True, exist_ok=True)
        temporary = saved.with_suffix('.tmp')
        shutil.copyfile(latest, temporary)
        os.replace(temporary, saved)
    completed = [step for step in job['milestones'] if checkpoint_path(folder, step).exists()]
    write(job['result'], {'seed': config['seed'], 'method': config['method'],
          'completed_milestones': completed, 'target_complete': job['milestones'][-1] in completed})


def evaluate_worker(job):
    from open_score.grouping.evaluation import evaluate
    from open_score.grouping.storage import fingerprint, sha256
    config, base = job['config'], Path(job['directory'])
    checkpoints = {method: checkpoint_path(base / method, job['common_steps']) for method in METHODS}
    actual_steps = {method: checked_checkpoint(checkpoint, dict(config, method=method))['counters']['physical_steps']
                    for method, checkpoint in checkpoints.items()}
    identity = fingerprint({method: sha256(path) for method, path in checkpoints.items()})[:12]
    directory = base / 'comparison' / identity
    write(directory / 'training_budget.json', {'requested_steps': job['common_steps'],
                                             'actual_steps': actual_steps})
    manifest = directory / 'evaluation_manifest.json'
    summary_file = directory / 'evaluation_summary.json'
    reusable = False
    if manifest.exists() and summary_file.exists():
        prior, summary = read(manifest), read(summary_file)
        reusable = (prior['checkpoints'] == {m: sha256(p) for m, p in checkpoints.items()}
                    and prior['scales'] == config['eval_scales']
                    and summary['complete_pairs'] == job['eval_episodes'] * len(config['eval_scales'])
                    and summary.get('planned_pairs') == summary['complete_pairs'])
    if not reusable:
        summary = evaluate(config, checkpoints, directory, scales=config['eval_scales'],
                           episodes=job['eval_episodes'], wall_seconds=job['available_seconds'],
                           matched_static=False)
    from plot_core_results import plot
    plot(directory / 'results.csv')
    write(job['result'], {'seed': config['seed'], 'opponent': config['opponent'],
          'evaluation_directory': str(directory), 'common_training_steps': job['common_steps'], **summary})


def worker(phase, path):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    job = read(path)
    folder = Path(job['directory'])
    folder.mkdir(parents=True, exist_ok=True)
    # A surviving worker also protects its own files if its parent was closed.
    with output_lock(folder):
        (train_worker if phase == 'train' else evaluate_worker)(job)


@contextmanager
def output_lock(output):
    """OS releases this lock even if the parent is killed; no stale PID files."""
    stream = (output / '.runner.lock').open('a+b')
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
        raise RuntimeError(f'Another runner is using {output}') from None
    try:
        yield
    finally:
        stream.close()


def run_pool(jobs, phase, output, workers, deadline, scheduler):
    from open_score.grouping.storage import append_jsonl
    pending, active, completed = list(jobs), [], []
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', PYGAME_HIDE_SUPPORT_PROMPT='1')
    last_display = -math.inf

    def record(event, item):
        append_jsonl(output / 'scheduler_events.jsonl', {'time': time.time(), 'phase': phase,
                     'event': event, 'job': item['id'], 'pid': item['process'].pid})

    try:
        while pending or active:
            if time.monotonic() >= deadline:
                scheduler['deadline_reached'] = True
                break
            while pending and len(active) < workers:
                job = pending.pop(0)
                job['available_seconds'] = max(.01, min(job['phase_seconds'], deadline - time.monotonic() - 2))
                path = output / 'jobs' / f"{job['id']}.json"
                write(path, job)
                log = (output / 'logs' / f"{job['id']}.log").open('a', encoding='utf-8')
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                    '_worker', phase, str(path)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                item = {'id': job['id'], 'process': process, 'log': log, 'job': job}
                active.append(item)
                scheduler['peak_workers'] = max(scheduler['peak_workers'], len(active))
                record('started', item)
                print(f"[{phase}] started {job['id']} (PID {process.pid})", flush=True)
            for item in list(active):
                code = item['process'].poll()
                if code is None:
                    continue
                record('finished', item)
                item['log'].close()
                active.remove(item)
                if code:
                    raise RuntimeError(f"{item['id']} failed ({code}); see logs/{item['id']}.log")
                completed.append(item['id'])
                print(f"[{phase}] finished {item['id']}", flush=True)
            if time.monotonic() - last_display >= 30:
                progress = {'phase': phase, 'active': [i['id'] for i in active],
                            'queued': len(pending), 'completed_in_phase': completed,
                            'remaining_phase_seconds': max(0, deadline-time.monotonic()), **scheduler}
                write(output / 'progress.json', progress)
                print(f"[{phase}] {len(active)} running, {len(pending)} queued; "
                      f"phase time left {progress['remaining_phase_seconds']/60:.1f} min", flush=True)
                last_display = time.monotonic()
            if active:
                time.sleep(.2)
    finally:
        # Only this runner's children are stopped. Completed model updates are on disk.
        for item in active:
            if item['process'].poll() is None:
                item['process'].terminate()
        for item in active:
            try:
                item['process'].wait(timeout=3)
            except subprocess.TimeoutExpired:
                item['process'].kill()
                item['process'].wait(timeout=3)
            record('stopped', item)
            item['log'].close()
    return completed


def make_config(args, seed, method):
    from open_score.grouping.cli import get_config, parser as cli_parser
    options = cli_parser().parse_args(['train', '--profile', args.profile,
               '--method', method, '--seed', str(seed), '--opponent', args.opponent, '--device', 'cpu'])
    config = get_config(options)
    config.update(steps=args.steps, train_seconds=args.train_minutes*60,
                  eval_episodes=args.eval_episodes, eval_seconds=args.eval_minutes*60)
    return config


def append_paired_report(output, runs, common, target, seeds):
    lines = ['', '## 主算法与规则释放的配对比较', '',
             f'统一评估训练预算：约 {common:,} 个物理步；请求上限：{target:,} 步。',
             '每个模型从头训练，并在相同物理步数节点保存；恢复节点从新回合继续。',
             'latest.pt 可能比统一评估节点训练得更多，不混入本表。', '',
             '| 种子 | 规模 | B4 成功率 | B6 成功率 | B4−B6（百分点） | 配对回合数 |',
             '|---:|---:|---:|---:|---:|---:|']
    differences = []
    for run in runs:
        for scale in sorted({row['scale'] for row in run['table']}):
            rows = {row['method']: row for row in run['table'] if row['scale'] == scale}
            if set(METHODS).issubset(rows):
                a, b = rows['selective'], rows['rule']
                delta = a['success_rate'] - b['success_rate']
                differences.append({'seed': run['seed'], 'scale': scale, 'success_rate_difference': delta,
                                    'paired_episodes': a['episodes']})
                lines.append(f"| {run['seed']} | {scale} | {a['success_rate']:.1%} | {b['success_rate']:.1%} | {delta*100:+.1f} | {a['episodes']} |")
    lines += ['', f'计划 {len(seeds)} 个训练种子；实际生成评估结果 {len(runs)} 个。',
              '三个训练种子的区间仅供探索，不能自动判定方法有效。若所有方法都接近零成功率，先检查任务和训练信号。',
              '规则释放 B6 仍训练重建器；本实验检验学习选择成员的价值。']
    with (output / 'comparison_report.md').open('a', encoding='utf-8') as stream:
        stream.write('\n'.join(lines) + '\n')
    write(output / 'paired_differences.json', differences)


def run(args):
    from open_score.grouping.cli import aggregate_comparison
    from open_score.grouping.storage import provenance, sha256
    started = time.monotonic()
    output = args.output.resolve()
    settings = {'seeds': args.seeds, 'methods': list(METHODS), 'workers': args.workers,
                'profile': args.profile, 'opponent': args.opponent, 'steps': args.steps,
                'checkpoint_every': args.checkpoint_every, 'train_seconds': args.train_minutes*60,
                'eval_episodes': args.eval_episodes, 'eval_seconds': args.eval_minutes*60,
                'total_seconds': args.total_minutes*60, 'device': 'cpu'}
    if args.dry_run:
        print(json.dumps(settings, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=True)
    for folder in ('jobs', 'logs'):
        (output / folder).mkdir(exist_ok=True)
    with output_lock(output):
        evidence = provenance()
        manifest = {'settings': settings, 'source_hash': evidence['source_hash'],
                    'assets': evidence['assets'], 'runner_sha256': sha256(__file__),
                    'plotter_sha256': sha256(ROOT / 'scripts/plot_core_results.py')}
        path = output / 'runner_manifest.json'
        if path.exists():
            if not args.resume:
                raise FileExistsError(f'{output} exists; pass --resume')
            if read(path) != manifest:
                raise ValueError('Runner settings, source or assets changed; use a new output directory')
        write(path, manifest)
        points = milestones(args.steps, args.checkpoint_every)
        deadline = started + args.total_minutes*60
        reserve = min(120., args.total_minutes*60*.03)
        train_deadline = deadline - args.eval_minutes*60 - reserve
        scheduler = {'peak_workers': 0, 'deadline_reached': False}
        jobs, folders = [], []
        for seed in args.seeds:
            for method in METHODS:
                folder = output / args.opponent / f'seed_{seed}' / method
                folders.append(folder)
                job_id = f'train_{seed}_{method}'
                jobs.append({'id': job_id, 'config': make_config(args, seed, method),
                     'directory': str(folder), 'milestones': points, 'train_seconds': args.train_minutes*60,
                     'phase_seconds': args.train_minutes*60,
                     'result': str(output / 'jobs' / f'{job_id}.result.json')})
        try:
            run_pool(jobs, 'train', output, args.workers, train_deadline, scheduler)
            shared = set(points)
            for folder in folders:
                shared &= {step for step in points if checkpoint_path(folder, step).exists()}
            if not shared:
                write(output / 'progress.json', {'status': 'no_common_checkpoint', **scheduler})
                print('No common completed physical-step milestone; resume to complete one.', flush=True)
                return 2
            common = max(shared)
            print(f'[evaluate] all six models share the {common:,}-step checkpoint', flush=True)
            jobs = []
            for seed in args.seeds:
                job_id = f'evaluate_{seed}'
                jobs.append({'id': job_id, 'config': make_config(args, seed, 'selective'),
                    'directory': str(output / args.opponent / f'seed_{seed}'), 'common_steps': common,
                    'eval_episodes': args.eval_episodes, 'phase_seconds': args.eval_minutes*60,
                    'result': str(output / 'jobs' / f'{job_id}.result.json')})
            completed = run_pool(jobs, 'evaluate', output, args.workers, deadline-reserve,
                                 scheduler)
            runs = [read(job['result']) for job in jobs if job['id'] in completed]
            aggregate_comparison(output, runs)
            append_paired_report(output, runs, common, args.steps, args.seeds)
            write(output / 'comparison_runs.json', runs)
            complete = (len(runs) == len(args.seeds) and all(
                r['complete_pairs'] == len(make_config(args, r['seed'], 'selective')['eval_scales'])*args.eval_episodes
                for r in runs))
            status = {'status': 'complete' if complete else 'partial_evaluation',
                      'common_training_steps': common, 'target_training_steps': args.steps,
                      'target_training_complete': common == args.steps,
                      'evaluated_seeds': [r['seed'] for r in runs],
                      'invocation_seconds': time.monotonic()-started, **scheduler}
            write(output / 'progress.json', status)
            print(json.dumps(status, indent=2), flush=True)
            print(f"Report: {output / 'comparison_report.md'}", flush=True)
            return 0 if complete else 2
        except KeyboardInterrupt:
            write(output / 'progress.json', {'status': 'interrupted', 'invocation_seconds': time.monotonic()-started,
                                           **scheduler})
            print('Stopped owned workers. Completed checkpoints are retained; use --resume.', flush=True)
            return 130
        except Exception as error:
            write(output / 'progress.json', {'status': 'failed', 'error': str(error), **scheduler})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs=3, type=int, default=[20260905, 20260906, 20260907])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--steps', type=int, default=150000)
    parser.add_argument('--checkpoint-every', type=int, default=25000)
    parser.add_argument('--train-minutes', type=float, default=80)
    parser.add_argument('--total-minutes', type=float, default=180)
    parser.add_argument('--eval-minutes', type=float, default=10)
    parser.add_argument('--eval-episodes', type=int, default=100)
    parser.add_argument('--profile', choices=('smoke', 'minimal'), default='minimal')
    parser.add_argument('--opponent', choices=('reactive', 'balanced', 'concentrated'), default='reactive')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/v2_parallel_3seeds')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if len(set(args.seeds)) != 3 or any(seed < 0 for seed in args.seeds):
        parser.error('Provide exactly three distinct nonnegative seeds')
    if any(getattr(args, name) <= 0 for name in ('workers', 'steps', 'checkpoint_every',
           'train_minutes', 'total_minutes', 'eval_minutes', 'eval_episodes')):
        parser.error('Counts and budgets must be positive')
    if math.ceil(6/args.workers)*args.train_minutes + args.eval_minutes >= args.total_minutes:
        parser.error('Training waves plus evaluation must leave time inside --total-minutes')
    return run(args)


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '_worker':
        worker(sys.argv[2], Path(sys.argv[3]))
    else:
        raise SystemExit(main())
