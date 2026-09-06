"""Dependency-aware subprocess scheduling for the common-executor study."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROUTES = ('b1_counts', 'b2_local', 'b3_global', 'r1_ppo', 'r2_teacher_ppo', 'r3_ddqn')


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    from open_score.grouping.storage import replace_file
    replace_file(temporary, path)


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else default


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def pid_alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5  # A denied query is not evidence of death.
        try:
            code = wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


@contextmanager
def directory_lock(output, name='.runner.lock'):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / name
    token = f'{os.getpid()}:{time.time_ns()}'
    for attempt in range(2):
        try:
            with path.open('x', encoding='utf-8') as stream:
                json.dump({'pid': os.getpid(), 'token': token, 'started': now()}, stream)
            break
        except FileExistsError:
            existing = read_json(path, {})
            if pid_alive(existing.get('pid')):
                raise RuntimeError(f'Output is in use by PID {existing["pid"]}: {output}')
            if attempt:
                raise RuntimeError(f'Cannot acquire output lock: {output}')
            path.unlink(missing_ok=True)
    try:
        yield
    finally:
        if read_json(path, {}).get('token') == token:
            path.unlink(missing_ok=True)


def validate_manifest(output, identity, *, resume=False, request=None):
    """Scheduling changes and larger budgets are accepted; protocols are immutable."""
    output = Path(output)
    manifest = output / 'run_manifest.json'
    previous = read_json(manifest)
    if previous:
        if not resume:
            raise ValueError(f'Output already exists; use --resume or a new directory: {output}')
        if previous['identity'] != identity:
            raise ValueError('Source, executor, or experimental protocol changed; use a new output directory')
        history = previous.get('requests', [])
    else:
        unrelated = [p for p in output.iterdir() if p.name not in ('.runner.lock',)] if output.exists() else []
        if unrelated:
            raise ValueError(f'Output contains files without a compatible manifest: {output}')
        history = []
    result = {'schema': 'research-v4-run-v1', 'identity': identity,
              'created': previous['created'] if previous else now(),
              'requests': history + [{'time': now(), **(request or {})}]}
    atomic_json(manifest, result)
    return result


def resolve_device(requested):
    import torch
    if requested == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but this Python interpreter cannot use CUDA')
    return requested


def run_pool(jobs, *, workers, output, deadline=None, poll_seconds=.25):
    """Run only owned children; a failed route does not interrupt other routes."""
    if workers < 1:
        raise ValueError('workers must be positive')
    output = Path(output)
    logs = output / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    queue, active, finished = list(jobs), {}, []
    started = time.monotonic()
    next_print = 0.
    stopped_reason = None

    def save():
        atomic_json(output / 'scheduler_status.json', {
            'updated': now(), 'elapsed_seconds': time.monotonic() - started,
            'running': [{**job, 'pid': process.pid} for process, job, stream in active.values()],
            'queued': [job['id'] for job in queue], 'finished': finished,
            'stop_reason': stopped_reason})

    try:
        while queue or active:
            if deadline is not None and time.monotonic() >= deadline:
                stopped_reason = 'deadline'
                break
            while queue and len(active) < workers:
                job = queue.pop(0)
                stream = (logs / f'{job["id"]}.log').open('a', encoding='utf-8', buffering=1)
                stream.write(f'\n[{now()}] starting {job["id"]}\n')
                try:
                    process = subprocess.Popen(job['command'], cwd=job.get('cwd'), stdout=stream,
                        stderr=subprocess.STDOUT, env={**os.environ, 'PYTHONUNBUFFERED': '1',
                            'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'},
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                except Exception as error:
                    stream.write(repr(error) + '\n')
                    stream.close()
                    finished.append({'id': job['id'], 'state': 'failed', 'returncode': None,
                                     'error': str(error), 'finished': now()})
                    continue
                active[process.pid] = (process, job, stream)
            for pid, (process, job, stream) in list(active.items()):
                code = process.poll()
                if code is not None:
                    stream.close()
                    finished.append({'id': job['id'], 'state': 'complete' if code == 0 else 'failed',
                                     'returncode': code, 'finished': now()})
                    del active[pid]
            save()
            if time.monotonic() >= next_print:
                print(f'[v4] {len(active)} running, {len(queue)} queued, {len(finished)} finished', flush=True)
                next_print = time.monotonic() + 15
            if queue or active:
                time.sleep(poll_seconds)
    except KeyboardInterrupt:
        stopped_reason = 'interrupted'
    finally:
        for process, job, stream in active.values():
            if process.poll() is None:
                process.terminate()
        for process, job, stream in active.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            stream.close()
            finished.append({'id': job['id'], 'state': 'interrupted',
                             'returncode': process.returncode, 'finished': now()})
        active.clear()
        save()
    return {'completed': sum(item['state'] == 'complete' for item in finished),
            'failed': [item for item in finished if item['state'] == 'failed'],
            'interrupted': bool(stopped_reason), 'stop_reason': stopped_reason,
            'queued': [item['id'] for item in queue], 'jobs': finished,
            'elapsed_seconds': time.monotonic() - started}


def route_output(root, route, seed):
    return Path(root) / route / f'seed_{seed}'


def evaluation_current(output, config):
    result = read_json(Path(output) / 'evaluation_summary.json', {})
    return result.get('complete', False) and result.get('episodes_per_scale', 0) >= int(config['eval_episodes'])
