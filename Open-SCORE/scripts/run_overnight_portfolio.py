"""Run three independent known-opponent grouping candidates within one night.

The routes are alternative methods with one shared training seed, not three
independent training seeds. Each route owns its training and evaluation budget;
the portfolio never truncates all methods to the slowest method's step count.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
ROUTES = ('ppo_structured', 'ppo_teacher', 'candidate_q')
ROUTE_DESCRIPTIONS = {
    'ppo_structured': 'PPO over structured legal grouping candidates',
    'ppo_teacher': 'Simulation-teacher warm start, then structured PPO',
    'candidate_q': 'Replay-based Double DQN over grouping candidates',
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    """Replace atomically, retrying transient Windows antivirus/reader locks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                         encoding='utf-8')
    for attempt in range(9):
        try:
            os.replace(temporary, path)
            return
        except OSError as error:
            if getattr(error, 'winerror', None) not in (5, 32, 33) or attempt == 8:
                raise
            time.sleep(min(.05 * 2 ** attempt, 1.0))


def optional_json(path):
    try:
        return read_json(path)
    except (OSError, ValueError):
        return None


@contextmanager
def output_lock(output):
    """An OS lock guards the portfolio; the training CLI locks each route."""
    stream = (Path(output) / '.portfolio.lock').open('a+b')
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
        raise RuntimeError(f'Another portfolio runner is using {output}') from None
    try:
        yield
    finally:
        stream.close()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_identity():
    """No training-module import is needed to validate a resumed portfolio."""
    files = {}
    for directory in ('src', 'configs'):
        for path in sorted((ROOT / directory).rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.yaml', '.yml', '.json'):
                normalized = path.read_text(encoding='utf-8-sig').replace('\r\n', '\n')
                files[path.relative_to(ROOT).as_posix()] = hashlib.sha256(normalized.encode()).hexdigest()
    assets = {path.relative_to(ROOT).as_posix(): file_hash(path)
              for path in sorted((ROOT / 'assets/frozen').rglob('*')) if path.is_file()}
    return {'source_files': files, 'assets': assets, 'runner_sha256': file_hash(__file__)}


def worker_command(job, resume):
    command = [sys.executable, '-m', 'open_score.overnight.cli', 'worker',
               '--route', job['route'], '--output', job['output'],
               '--train-seconds', str(job['train_seconds']),
               '--eval-seconds', str(job['eval_seconds']),
               '--seed', str(job['seed']), '--steps', str(job['steps']),
               '--checkpoint-every', str(job['checkpoint_every']),
               '--eval-episodes', str(job['eval_episodes']), '--device', job['device']]
    if job['smoke']:
        command.append('--smoke')
    if resume:
        command.append('--resume')
    return command


def make_settings(args):
    smoke = args.smoke
    train_hours = args.train_hours if args.train_hours is not None else (1 / 60 if smoke else 7)
    eval_minutes = args.eval_minutes if args.eval_minutes is not None else (1 if smoke else 45)
    total_hours = args.total_hours if args.total_hours is not None else (.05 if smoke else 8)
    reserve_minutes = args.reserve_minutes if args.reserve_minutes is not None else (1 if smoke else 15)
    return {'routes': list(ROUTES), 'workers': args.workers, 'seed': args.seed,
            'steps': args.steps if args.steps is not None else (256 if smoke else 0),
            'train_seconds': train_hours * 3600, 'eval_seconds': eval_minutes * 60,
            'total_seconds': total_hours * 3600, 'reserve_seconds': reserve_minutes * 60,
            'checkpoint_every': args.checkpoint_every,
            'eval_episodes': args.eval_episodes if args.eval_episodes is not None else (2 if smoke else 100),
            'device': args.device, 'smoke': smoke,
            'evaluation_contract': 'Held-out paired test episodes; per-route actual training budgets',
            'seed_contract': 'Three alternative routes with one training seed; not a three-seed study'}


def validate_settings(settings):
    if settings['seed'] < 0 or settings['steps'] < 0:
        raise ValueError('Seed and steps must be nonnegative; steps=0 uses the time budget')
    for name in ('train_seconds', 'eval_seconds', 'total_seconds', 'reserve_seconds',
                 'checkpoint_every', 'eval_episodes'):
        if not math.isfinite(settings[name]) or settings[name] <= 0:
            raise ValueError(f'{name} must be finite and positive')
    waves = math.ceil(len(ROUTES) / settings['workers'])
    required = waves * (settings['train_seconds'] + settings['eval_seconds']) + settings['reserve_seconds']
    if required > settings['total_seconds'] + .001:
        raise ValueError('Training/evaluation waves plus the finish reserve exceed --total-hours')


def make_jobs(output, settings):
    names = ('seed', 'steps', 'train_seconds', 'eval_seconds', 'checkpoint_every',
             'eval_episodes', 'device', 'smoke')
    return [dict({name: settings[name] for name in names}, route=route,
                 output=str(output / route)) for route in ROUTES]


def append_event(output, event, route, **fields):
    record = {'time': time.time(), 'event': event, 'route': route, **fields}
    with (output / 'scheduler_events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()


def collect_state(job, status, **fields):
    directory = Path(job['output'])
    return {'route': job['route'], 'status': status, 'output': str(directory),
            'progress': optional_json(directory / 'progress.json'),
            'result': optional_json(directory / 'route_result.json'), **fields}


def run_pool(jobs, output, workers, deadline, resume=False, poll_seconds=.2, display_seconds=30):
    """A failed candidate is isolated; every owned process is reaped on exit."""
    pending, active, states = list(jobs), [], {}
    peak_workers, deadline_reached, interrupted = 0, False, False
    last_display = last_state = -math.inf
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', PYGAME_HIDE_SUPPORT_PROMPT='1')

    def snapshot(status='running'):
        for item in active:
            states[item['job']['route']] = collect_state(item['job'], 'running', pid=item['process'].pid)
        value = {'status': status, 'active': [item['job']['route'] for item in active],
                 'queued': [job['route'] for job in pending], 'routes': states,
                 'peak_workers': peak_workers, 'deadline_reached': deadline_reached,
                 'remaining_wall_budget_seconds': max(0, deadline - time.monotonic())}
        write_json(output / 'progress.json', value)
        return value

    try:
        while pending or active:
            if time.monotonic() >= deadline:
                deadline_reached = True
                break
            while pending and len(active) < workers:
                job = pending.pop(0)
                route = job['route']
                prior = optional_json(Path(job['output']) / 'route_result.json')
                if resume and prior and prior.get('status') == 'complete':
                    states[route] = collect_state(job, 'complete', reused=True)
                    append_event(output, 'reused', route)
                    print(f'[portfolio] reused completed {route}', flush=True)
                    continue
                command = worker_command(job, resume)
                write_json(output / 'jobs' / f'{route}.json', {'job': job, 'command': command})
                log = (output / 'logs' / f'{route}.log').open('a', encoding='utf-8')
                log.write(f'\n[portfolio invocation {time.strftime("%Y-%m-%d %H:%M:%S")}]\n')
                log.flush()
                try:
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                        stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                except Exception as error:
                    log.close()
                    states[route] = collect_state(job, 'failed', error=str(error))
                    append_event(output, 'launch_failed', route, error=str(error))
                    print(f'[portfolio] could not launch {route}: {error}', flush=True)
                    continue
                active.append({'job': job, 'process': process, 'log': log})
                peak_workers = max(peak_workers, len(active))
                append_event(output, 'started', route, pid=process.pid)
                print(f'[portfolio] started {route} (PID {process.pid})', flush=True)
            for item in list(active):
                code = item['process'].poll()
                if code is None:
                    continue
                job = item['job']
                item['log'].close()
                active.remove(item)
                result = optional_json(Path(job['output']) / 'route_result.json')
                status = 'complete' if code == 0 and result and result.get('status') == 'complete' else 'failed'
                if code == 0 and result and result.get('status') not in (None, 'complete'):
                    status = 'partial'
                states[job['route']] = collect_state(job, status, returncode=code)
                if code == 0 and not result:
                    states[job['route']]['error'] = 'Worker exited without route_result.json'
                append_event(output, 'finished', job['route'], pid=item['process'].pid,
                             returncode=code, status=status)
                print(f"[portfolio] {job['route']}: {status}; logs/{job['route']}.log", flush=True)
            now = time.monotonic()
            if now - last_state >= 1:
                snapshot()
                last_state = now
            if now - last_display >= display_seconds:
                done = sum(state['status'] == 'complete' for state in states.values())
                print(f'[portfolio] {len(active)} running, {len(pending)} queued, {done}/3 complete; '
                      f'wall-time limit remaining {max(0, deadline-now)/60:.1f} min '
                      '(a limit, not an ETA)', flush=True)
                last_display = now
            if active:
                time.sleep(poll_seconds)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        # Do not kill unrelated Python processes, or discard completed routes.
        for item in active:
            if item['process'].poll() is None:
                item['process'].terminate()
        for item in active:
            try:
                item['process'].wait(timeout=3)
            except subprocess.TimeoutExpired:
                item['process'].kill()
                item['process'].wait(timeout=3)
            item['log'].close()
            job = item['job']
            states[job['route']] = collect_state(job, 'interrupted' if interrupted else 'stopped',
                                                 returncode=item['process'].returncode)
            append_event(output, 'stopped', job['route'], pid=item['process'].pid,
                         returncode=item['process'].returncode)
        active.clear()
        for job in pending:
            states[job['route']] = collect_state(job, 'not_started')
    status = ('interrupted' if interrupted else 'complete'
              if len(states) == len(jobs) and all(s['status'] == 'complete' for s in states.values())
              else 'deadline_reached' if deadline_reached else 'partial_failure')
    return snapshot(status)


def write_report(output, settings, summary):
    import csv

    output = Path(output)
    route_names = {'ppo_structured': '结构化候选 PPO', 'ppo_teacher': '模拟教师热启动 PPO',
                   'candidate_q': '候选动作 Double DQN'}
    method_names = {'best': 'best（验证集选优）', 'latest': 'latest（最终检查点）',
                    'static': '静态规则 static', 'compact_rule': '紧凑均衡规则 compact_rule',
                    'dynamic_rule': '动态威胁规则 dynamic_rule', 'all_reserve': '全后备 all_reserve'}
    method_order = list(method_names)
    planned_hours = settings['train_seconds'] / 3600
    lines = ['# 已知对手动态分组：整夜三路线实验汇总', '',
             f"运行状态：**{summary['status']}**；共同训练种子：{settings['seed']}。",
             '三条路线是三种候选方案，不是三个独立训练随机种子。',
             f'每路计划训练时间上限 {planned_hours:g} 小时；实际训练预算分别记录，不统一截断到最慢路线的步数。',
             '训练时间包含该路线的教师采样和训练内验证；教师模拟步数单列，不冒充在线环境训练步数。', '',
             '| 路线 | 状态 | 在线训练步数 | 教师模拟步数 | 实际训练时间（小时） | 实际训练时间（秒） | 原始结果 |',
             '|---|---|---:|---:|---:|---:|---|']
    csv_rows = []
    for route in ROUTES:
        state = summary['routes'].get(route, {})
        result = state.get('result') or {}
        seconds = result.get('training_seconds')
        hours_text = f'{seconds / 3600:.4f}' if isinstance(seconds, (int, float)) else '未记录'
        seconds_text = f'{seconds:.2f}' if isinstance(seconds, (int, float)) else '未记录'
        lines.append(f"| {route_names[route]} `{route}` | {state.get('status', 'not_started')} | "
                     f"{result.get('actual_steps', '未记录')} | {result.get('teacher_simulation_steps', '未记录')} | "
                     f"{hours_text} | {seconds_text} | [结果 JSON]({route}/route_result.json) |")
    lines += ['', '## 配对测试结果', '',
        '各表只使用对应路线已有的 evaluation.table；没有完整配对时不填 0，也不计算成功率。',
        '首选检查点为验证集选出的 best；若某规模没有 best 评估，则展示该规模已有的 latest 作为参考。',
        'latest 同时保留展示，以便检查后期训练是否退化；不使用最终测试集重新挑选训练检查点。',
        '相对差值是相同规模、相同已完成配对回合数下，相对于 compact_rule 的成功率百分点差。', '']
    for route in ROUTES:
        result = summary['routes'].get(route, {}).get('result') or {}
        evaluation = result.get('evaluation') or {}
        table = evaluation.get('table') or []
        # Missing fields are not invented. Valid existing rows, including rows
        # from a partial evaluation's completed units, remain independently useful.
        required = ('scale', 'method', 'episodes', 'successes', 'success_rate')
        rows = [row for row in table if all(key in row for key in required) and row['episodes'] > 0]
        lines += [f'### {route_names[route]}（{route}）', '']
        directory = result.get('evaluation_directory')
        if directory and (Path(directory) / 'report.md').is_file():
            try:
                relative = os.path.relpath(Path(directory) / 'report.md', output).replace(os.sep, '/')
            except ValueError:
                relative = None
            if relative is not None:
                lines += [f'[本路线完整评估报告](<{relative}>)', '']
        if 'complete_pairs' in evaluation and 'planned_pairs' in evaluation:
            lines += [f"完整配对：{evaluation['complete_pairs']}/{evaluation['planned_pairs']}；"
                      f"评估完成标记：{evaluation.get('complete', '未记录')}。", '']
        if not rows:
            lines += ['暂无可汇总的完整配对测试结果。', '']
            continue
        if len(rows) != len(table):
            lines += [f'另有 {len(table)-len(rows)} 条记录缺少完整计数字段或没有已完成回合，未纳入本表。', '']
        lines += ['| 规模 | 方法 / 检查点 | 成功数 / 局数 | 成功率 | 相对紧凑规则（百分点） |',
                  '|---:|---|---:|---:|---:|']
        order = lambda row: (row['scale'], method_order.index(row['method'])
                             if row['method'] in method_order else len(method_order), row['method'])
        for row in sorted(rows, key=order):
            peers = {item['method']: item for item in rows if item['scale'] == row['scale']}
            reference = 'best' if 'best' in peers else 'latest' if 'latest' in peers else None
            method_text = method_names.get(row['method'], row['method'])
            if row['method'] == reference:
                method_text += '；参考检查点'
            compact = peers.get('compact_rule')
            delta = (f"{100*(row['success_rate']-compact['success_rate']):+.2f}"
                     if compact and compact['episodes'] == row['episodes'] else '暂无匹配对照')
            lines.append(f"| {row['scale']}v{row['scale']} | {method_text} | "
                         f"{row['successes']} / {row['episodes']} | {row['success_rate']:.2%} | {delta} |")
            csv_rows.append({'route': route, **{key: row.get(key) for key in (
                'scale', 'method', 'episodes', 'successes', 'success_rate', 'decision_p95_ms',
                'mean_reserve_fraction', 'mean_group_size', 'singleton_fraction')},
                'actual_steps': result.get('actual_steps')})
        lines.append('')
    lines += ['## 如何解释这些结果', '',
        '完成运行不等于证明方法有效。本次测试可用于观察方案差异；若测试均值有高低，也不能据此宣称跨种子可靠优势。',
        'best 由独立验证集选取。此报告不会按测试结果自动部署或替换模型。',
        '同时检查成功数、规则对照、后备比例、组大小和单人比例；详细行为与决策耗时见 CSV 和各路线报告。', '',
        '日志：`logs/<route>.log`；实时状态：`progress.json` 与 `<route>/progress.json`。',
        '恢复时使用相同指令和输出目录并加 `--resume`；改变源码、冻结资产或实验配置时需要新输出目录。', '']
    fields = ['route', 'scale', 'method', 'episodes', 'successes', 'success_rate', 'decision_p95_ms',
              'mean_reserve_fraction', 'mean_group_size', 'singleton_fraction', 'actual_steps']
    with (output / 'portfolio_results.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    (output / 'portfolio_report.md').write_text('\n'.join(lines), encoding='utf-8')
    write_json(output / 'portfolio_summary.json', {'settings': settings, **summary})


def run(args):
    started = time.monotonic()
    settings = make_settings(args)
    validate_settings(settings)
    output = args.output.resolve()
    jobs = make_jobs(output, settings)
    if args.dry_run:
        print(json.dumps({'output': str(output), 'settings': settings,
                          'plans': [{'route': job['route'], 'description': ROUTE_DESCRIPTIONS[job['route']],
                                     'command': worker_command(job, args.resume)} for job in jobs]},
                         ensure_ascii=False, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=True)
    with output_lock(output):
        manifest_path = output / 'runner_manifest.json'
        manifest = {'schema': 'overnight-portfolio-v1', 'settings': settings, **source_identity()}
        if manifest_path.exists():
            if not args.resume:
                raise FileExistsError(f'{output} contains a run; pass --resume or choose a new output directory')
            if read_json(manifest_path) != manifest:
                raise ValueError('Runner settings, source or frozen assets changed; use a new output directory')
        elif any(path.name != '.portfolio.lock' for path in output.iterdir()):
            raise FileExistsError(f'{output} is nonempty without a runner manifest; choose a new output directory')
        write_json(manifest_path, manifest)
        for folder in ('jobs', 'logs'):
            (output / folder).mkdir(exist_ok=True)
        try:
            summary = run_pool(jobs, output, settings['workers'], started + settings['total_seconds'],
                               resume=args.resume)
        except Exception as error:
            write_json(output / 'progress.json', {'status': 'failed', 'error': str(error),
                       'invocation_seconds': time.monotonic() - started})
            raise
        summary['invocation_seconds'] = time.monotonic() - started
        write_report(output, settings, summary)
        write_json(output / 'progress.json', summary)
        print(f"[portfolio] {summary['status']}; report: {output / 'portfolio_report.md'}", flush=True)
        if summary['status'] == 'interrupted':
            print('Owned workers stopped. Checkpoints retained; use --resume.', flush=True)
            return 130
        return 0 if summary['status'] == 'complete' else 2


def parser():
    options = argparse.ArgumentParser(description=__doc__)
    options.add_argument('--output', type=Path, default=ROOT / 'outputs/overnight_portfolio')
    options.add_argument('--workers', type=int, choices=(1, 2, 3), default=3)
    options.add_argument('--seed', type=int, default=20260906)
    options.add_argument('--steps', type=int, help='Per-route ceiling; 0 (default) trains to the wall-time budget')
    options.add_argument('--train-hours', type=float, help='Per-route training budget (default 7)')
    options.add_argument('--eval-minutes', type=float, help='Per-route evaluation budget (default 45)')
    options.add_argument('--total-hours', type=float, help='Invocation wall-time ceiling (default 8)')
    options.add_argument('--reserve-minutes', type=float, help='Finish reserve inside total time (default 15)')
    options.add_argument('--checkpoint-every', type=int, default=25000)
    options.add_argument('--eval-episodes', type=int, help='Paired test episodes per scale (default 100)')
    options.add_argument('--device', choices=('cpu', 'cuda', 'auto'), default='auto')
    options.add_argument('--smoke', action='store_true', help='256 steps, 2 test episodes/scale, 3-minute ceiling')
    options.add_argument('--resume', action='store_true')
    options.add_argument('--dry-run', action='store_true')
    return options


def main():
    options = parser()
    args = options.parse_args()
    try:
        return run(args)
    except (ValueError, FileExistsError, RuntimeError) as error:
        print(f'Error: {error}', file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
