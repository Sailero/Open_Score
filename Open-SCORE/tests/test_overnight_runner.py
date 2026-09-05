"""Exercise portfolio contracts and real OS-process failure/resume behavior."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/run_overnight_portfolio.py'


def load_runner():
    spec = importlib.util.spec_from_file_location('overnight_runner_test', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_jobs(runner, output):
    (output / 'logs').mkdir(exist_ok=True)
    (output / 'jobs').mkdir(exist_ok=True)
    settings = runner.make_settings(runner.parser().parse_args(['--smoke']))
    return runner.make_jobs(output, settings)


def fake_command(job, resume, *, exit_code=0, delay=.1, write_result=True):
    # These workers exercise the real scheduler without depending on an ML run.
    program = ('import json, pathlib, sys, time; '
               'folder=pathlib.Path(sys.argv[1]); folder.mkdir(parents=True, exist_ok=True); '
               'time.sleep(float(sys.argv[2])); ')
    if write_result:
        program += ("(folder/'route_result.json').write_text(json.dumps({"
                    "'status':'complete', 'actual_steps':int(sys.argv[3]), 'training_seconds':.1}),"
                    "encoding='utf-8'); ")
    program += f'raise SystemExit({exit_code})'
    steps = {'ppo_structured': 101, 'ppo_teacher': 203, 'candidate_q': 307}[job['route']]
    return [sys.executable, '-c', program, job['output'], str(delay), str(steps)]


def test_default_plan_is_three_routes_time_based_and_dry_run_does_not_create_output(tmp_path, capsys):
    runner = load_runner()
    args = runner.parser().parse_args(['--output', str(tmp_path / 'new'), '--dry-run'])
    assert runner.run(args) == 0
    plan = json.loads(capsys.readouterr().out)
    settings = plan['settings']
    assert settings['workers'] == 3 and settings['steps'] == 0
    assert settings['device'] == 'auto'
    assert settings['train_seconds'] == 7 * 3600
    assert settings['eval_seconds'] == 45 * 60
    assert settings['total_seconds'] == 8 * 3600
    assert settings['reserve_seconds'] == 15 * 60
    assert len(plan['plans']) == 3 and len({p['route'] for p in plan['plans']}) == 3
    for item in plan['plans']:
        command = item['command']
        assert command[1:4] == ['-m', 'open_score.overnight.cli', 'worker']
        assert command[command.index('--seed') + 1] == '20260906'
        assert command[command.index('--steps') + 1] == '0'
    assert not args.output.exists()


def test_smoke_budget_and_explicit_overrides_are_consistent():
    runner = load_runner()
    settings = runner.make_settings(runner.parser().parse_args(['--smoke']))
    runner.validate_settings(settings)
    assert settings['steps'] == 256
    assert settings['eval_episodes'] == 2
    assert settings['total_seconds'] == 180
    assert settings['train_seconds'] == settings['eval_seconds'] == 60
    args = runner.parser().parse_args(['--smoke', '--steps', '512', '--eval-episodes', '3',
                                      '--device', 'cuda', '--resume'])
    settings = runner.make_settings(args)
    assert settings['steps'] == 512 and settings['eval_episodes'] == 3
    command = runner.worker_command(runner.make_jobs(Path('out'), settings)[0], True)
    assert '--smoke' in command and '--resume' in command
    assert command[command.index('--device') + 1] == 'cuda'
    settings['workers'] = 1
    with pytest.raises(ValueError, match='exceed'):
        runner.validate_settings(settings)


def test_output_lock_rejects_duplicate_writer_and_releases(tmp_path):
    runner = load_runner()
    with runner.output_lock(tmp_path):
        with pytest.raises(RuntimeError, match='Another portfolio'):
            with runner.output_lock(tmp_path):
                pytest.fail('Duplicate writer acquired the root lock')
    with runner.output_lock(tmp_path):
        pass


def test_real_workers_have_individual_budgets_and_completed_resume_is_read_only(tmp_path, monkeypatch):
    runner = load_runner()
    original_popen = subprocess.Popen
    processes = []

    def launch(*args, **kwargs):
        if os.name == 'nt':
            assert kwargs['creationflags'] & subprocess.CREATE_NO_WINDOW
        assert kwargs['env']['OMP_NUM_THREADS'] == '1'
        process = original_popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(runner.subprocess, 'Popen', launch)
    monkeypatch.setattr(runner, 'worker_command', fake_command)
    monkeypatch.setattr(runner, 'source_identity', lambda: {'source': 'fixed-test-source'})
    args = runner.parser().parse_args(['--smoke', '--output', str(tmp_path)])
    assert runner.run(args) == 0
    summary = runner.read_json(tmp_path / 'portfolio_summary.json')
    assert summary['status'] == 'complete' and summary['peak_workers'] == 3
    assert {r['result']['actual_steps'] for r in summary['routes'].values()} == {101, 203, 307}
    assert all(process.poll() == 0 for process in processes)
    before = {path: path.read_bytes() for path in tmp_path.glob('*/route_result.json')}
    args.resume = True
    assert runner.run(args) == 0
    assert len(processes) == 3
    assert {path: path.read_bytes() for path in before} == before
    assert all(r['reused'] for r in runner.read_json(tmp_path / 'progress.json')['routes'].values())
    args.steps = 999
    with pytest.raises(ValueError, match='settings, source or frozen assets changed'):
        runner.run(args)
    args.steps = None
    monkeypatch.setattr(runner, 'source_identity', lambda: {'source': 'changed-test-source'})
    with pytest.raises(ValueError, match='settings, source or frozen assets changed'):
        runner.run(args)


def test_failed_route_does_not_cancel_successful_routes_and_resume_only_retries_failure(tmp_path, monkeypatch):
    runner = load_runner()
    jobs = prepare_jobs(runner, tmp_path)
    launched = []

    def command(job, resume):
        launched.append((job['route'], resume))
        if job['route'] == 'ppo_teacher' and not resume:
            return fake_command(job, resume, exit_code=3, write_result=False)
        return fake_command(job, resume, delay=.5)

    monkeypatch.setattr(runner, 'worker_command', command)
    result = runner.run_pool(jobs, tmp_path, 3, time.monotonic() + 15)
    assert result['status'] == 'partial_failure'
    assert result['routes']['ppo_teacher']['returncode'] == 3
    assert result['routes']['ppo_structured']['status'] == 'complete'
    assert result['routes']['candidate_q']['status'] == 'complete'
    result = runner.run_pool(jobs, tmp_path, 3, time.monotonic() + 15, resume=True)
    assert result['status'] == 'complete'
    assert launched == [(r, False) for r in runner.ROUTES] + [('ppo_teacher', True)]


def test_zero_exit_without_result_is_not_reported_as_success(tmp_path, monkeypatch):
    runner = load_runner()
    jobs = prepare_jobs(runner, tmp_path)
    monkeypatch.setattr(runner, 'worker_command',
                        lambda job, resume: fake_command(job, resume, write_result=False))
    result = runner.run_pool(jobs, tmp_path, 3, time.monotonic() + 15)
    assert result['status'] == 'partial_failure'
    assert all(state['status'] == 'failed' for state in result['routes'].values())
    assert all('without route_result.json' in state['error'] for state in result['routes'].values())


@pytest.mark.parametrize('interrupt', [False, True])
def test_deadline_and_interrupt_reap_all_owned_workers(tmp_path, monkeypatch, interrupt):
    runner = load_runner()
    jobs = prepare_jobs(runner, tmp_path)
    original_popen = subprocess.Popen
    processes = []

    def launch(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(runner.subprocess, 'Popen', launch)
    monkeypatch.setattr(runner, 'worker_command',
                        lambda job, resume: fake_command(job, resume, delay=30))
    if interrupt:
        original_sleep = time.sleep
        injected = []

        def interrupted_sleep(_seconds):
            if not injected:
                injected.append(True)
                raise KeyboardInterrupt
            original_sleep(_seconds)

        monkeypatch.setattr(runner.time, 'sleep', interrupted_sleep)
    result = runner.run_pool(jobs, tmp_path, 3, time.monotonic() + (10 if interrupt else .5))
    assert len(processes) == 3 and all(process.poll() is not None for process in processes)
    assert result['status'] == ('interrupted' if interrupt else 'deadline_reached')
    events = [json.loads(line) for line in (tmp_path / 'scheduler_events.jsonl').read_text().splitlines()]
    assert {event['pid'] for event in events if event['event'] == 'started'} == {
        event['pid'] for event in events if event['event'] == 'stopped'}


def test_unmanaged_nonempty_output_is_preserved_and_rejected(tmp_path, monkeypatch):
    runner = load_runner()
    (tmp_path / 'unrelated.txt').write_text('preserve me', encoding='utf-8')
    monkeypatch.setattr(runner, 'source_identity', lambda: {'source': 'v1'})
    args = runner.parser().parse_args(['--smoke', '--output', str(tmp_path)])
    with pytest.raises(FileExistsError, match='nonempty without a runner manifest'):
        runner.run(args)
    assert (tmp_path / 'unrelated.txt').read_text() == 'preserve me'


def test_chinese_portfolio_report_and_csv_preserve_actual_paired_results(tmp_path):
    import csv

    runner = load_runner()
    settings = runner.make_settings(runner.parser().parse_args([]))
    evaluation_directory = tmp_path / 'ppo_structured' / 'evaluation' / 'identity123'
    evaluation_directory.mkdir(parents=True)
    (evaluation_directory / 'report.md').write_text('Actual route report', encoding='utf-8')

    def row(method, wins, episodes=10, scale=8):
        return {'method': method, 'scale': scale, 'episodes': episodes, 'successes': wins,
                'success_rate': wins / episodes, 'decision_p95_ms': 1.5,
                'mean_reserve_fraction': .1, 'mean_group_size': 3., 'singleton_fraction': .02}

    rows = [row('best', 3), row('latest', 4), row('static', 1), row('compact_rule', 2),
            row('dynamic_rule', 0), row('all_reserve', 0),
            row('latest', 1, episodes=5, scale=16), row('compact_rule', 1, scale=16)]
    summary = {'status': 'partial_failure', 'routes': {
        'ppo_structured': {'status': 'complete', 'result': {
            'actual_steps': 123456, 'teacher_simulation_steps': 25, 'training_seconds': 3600,
            'evaluation_directory': str(evaluation_directory),
            'evaluation': {'table': rows, 'complete_pairs': 15, 'planned_pairs': 20, 'complete': False}}},
        'ppo_teacher': {'status': 'failed', 'result': None},
        'candidate_q': {'status': 'partial', 'result': {'actual_steps': 777,
            'evaluation': {'table': [{'method': 'all_reserve', 'scale': 8, 'episodes': 2,
                                      'successes': 0, 'success_rate': 0.}],
                           'complete_pairs': 2, 'planned_pairs': 10, 'complete': False}}}}}
    runner.write_report(tmp_path, settings, summary)
    report = (tmp_path / 'portfolio_report.md').read_text(encoding='utf-8')
    assert '结构化候选 PPO' in report and '模拟教师热启动 PPO' in report and '候选动作 Double DQN' in report
    assert '计划训练时间上限 7 小时' in report
    assert '123456 | 25 | 1.0000 | 3600.00' in report
    assert 'best（验证集选优）；参考检查点 | 3 / 10 | 30.00% | +10.00' in report
    assert 'latest（最终检查点） | 4 / 10 | 40.00% | +20.00' in report
    assert '16v16 | latest（最终检查点）；参考检查点 | 1 / 5 | 20.00% | 暂无匹配对照' in report
    assert '[本路线完整评估报告](<ppo_structured/evaluation/identity123/report.md>)' in report
    assert '暂无可汇总的完整配对测试结果' in report
    assert '不会按测试结果自动部署或替换模型' in report
    with (tmp_path / 'portfolio_results.csv').open(encoding='utf-8-sig', newline='') as stream:
        exported = list(csv.DictReader(stream))
    assert len(exported) == 9
    assert not any(row['route'] == 'ppo_teacher' for row in exported)
    first = next(row for row in exported if row['route'] == 'ppo_structured' and row['method'] == 'best')
    assert first['successes'] == '3' and first['episodes'] == '10' and first['success_rate'] == '0.3'
    assert first['actual_steps'] == '123456'
    partial = next(row for row in exported if row['route'] == 'candidate_q')
    assert partial['episodes'] == '2' and partial['success_rate'] == '0.0'
    assert partial['decision_p95_ms'] == partial['mean_group_size'] == ''  # Missing is not zero.
