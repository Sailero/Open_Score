"""Real isolated workers, resumable paired results, and owned-process cleanup."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from open_score.grouping.storage import sha256

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/run_parallel_comparison.py'


def load_runner():
    spec = importlib.util.spec_from_file_location('parallel_runner_test', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_output_lock_prevents_duplicate_writers_and_releases(tmp_path):
    runner = load_runner()
    with runner.output_lock(tmp_path):
        with pytest.raises(RuntimeError, match='Another runner'):
            with runner.output_lock(tmp_path):
                pytest.fail('A concurrent writer acquired the same output')
    with runner.output_lock(tmp_path):
        pass


def test_six_real_training_workers_pair_at_shared_budget_and_resume(tmp_path, monkeypatch):
    command = [sys.executable, str(SCRIPT), '--profile', 'smoke', '--steps', '64',
               '--checkpoint-every', '32', '--workers', '4', '--train-minutes', '1',
               '--eval-minutes', '1', '--total-minutes', '5', '--eval-episodes', '2',
               '--output', str(tmp_path)]
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    first = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True,
                           encoding='utf-8', timeout=150)
    assert first.returncode == 0, first.stdout + first.stderr
    progress = json.loads((tmp_path / 'progress.json').read_text())
    assert progress['status'] == 'complete'
    assert progress['peak_workers'] == 4
    assert progress['common_training_steps'] == 64
    assert len(progress['evaluated_seeds']) == 3
    models = list(tmp_path.rglob('latest.pt'))
    evaluations = list(tmp_path.rglob('evaluation.jsonl'))
    assert len(models) == 6 and len(evaluations) == 3
    assert len(list(tmp_path.rglob('budgets/*.pt'))) == 12
    for path in evaluations:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == 4
        assert {r['method'] for r in rows} == {'selective', 'rule'}
    events = [json.loads(line) for line in (tmp_path / 'scheduler_events.jsonl').read_text().splitlines()]
    active, peak = set(), 0
    for event in events:
        if event['event'] == 'started':
            active.add(event['pid'])
        else:
            active.remove(event['pid'])
        peak = max(peak, len(active))
    assert peak == 4 and not active
    before = {path: sha256(path) for path in models + evaluations}
    resumed = subprocess.run(command + ['--resume'], cwd=ROOT, env=env, capture_output=True,
                             text=True, encoding='utf-8', timeout=120)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert {path: sha256(path) for path in before} == before

    # One run lacks the final milestone: all six must be evaluated at 32 steps,
    # not a mixture of final checkpoints with different interaction budgets.
    next(tmp_path.rglob('budgets/step_000000064.pt')).unlink()
    runner = load_runner()
    original_pool = runner.run_pool

    def capped_training(jobs, phase, output, workers, deadline, scheduler):
        if phase == 'train':
            scheduler['deadline_reached'] = True
            return []
        return original_pool(jobs, phase, output, workers, deadline, scheduler)

    monkeypatch.setattr(runner, 'run_pool', capped_training)
    from types import SimpleNamespace
    args = SimpleNamespace(output=tmp_path, seeds=[20260905, 20260906, 20260907],
           workers=4, steps=64, checkpoint_every=32, train_minutes=1, eval_minutes=1,
           total_minutes=5, eval_episodes=2, profile='smoke', opponent='reactive',
           resume=True, dry_run=False)
    assert runner.run(args) == 0
    progress = json.loads((tmp_path / 'progress.json').read_text())
    assert progress['common_training_steps'] == 32
    assert not progress['target_training_complete']
    assert len(progress['evaluated_seeds']) == 3
    for row in json.loads((tmp_path / 'comparison_runs.json').read_text()):
        budget = json.loads((Path(row['evaluation_directory']) / 'training_budget.json').read_text())
        assert budget['requested_steps'] == 32
        assert all(32 <= value <= 36 for value in budget['actual_steps'].values())


@pytest.mark.parametrize('fail', [False, True])
def test_pool_stops_its_owned_children_on_deadline_or_failure(tmp_path, monkeypatch, fail):
    runner = load_runner()
    original = subprocess.Popen
    processes = []

    def launch(*args, **kwargs):
        program = 'raise SystemExit(3)' if fail and not processes else 'import time; time.sleep(30)'
        process = original([sys.executable, '-c', program], **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(runner.subprocess, 'Popen', launch)
    (tmp_path / 'logs').mkdir()
    jobs = [{'id': f'job_{i}', 'phase_seconds': 5} for i in range(3)]
    scheduler = {'peak_workers': 0, 'deadline_reached': False}
    deadline = time.monotonic() + (5 if fail else .5)
    if fail:
        with pytest.raises(RuntimeError, match='failed'):
            runner.run_pool(jobs, 'train', tmp_path, 2, deadline, scheduler)
    else:
        runner.run_pool(jobs, 'train', tmp_path, 2, deadline, scheduler)
        assert scheduler['deadline_reached']
    assert len(processes) == 2
    assert all(process.poll() is not None for process in processes)
