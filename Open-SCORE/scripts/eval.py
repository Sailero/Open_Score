"""Evaluate retained models: inventory scan, anchors, final, depth, mechanism, timing."""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import redirect_stderr, redirect_stdout
from multiprocessing import get_context
from pathlib import Path
from queue import Empty
import os
import signal
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from open_score.utils.logging import DEFAULT_OUTPUT, FORMAL_RUN, parse_eval_console, read_tail


def _enable_ansi():
    if os.name == "nt":
        try:
            import ctypes
            handle = ctypes.windll.kernel32.GetStdHandle(-11)
            mode = ctypes.c_uint()
            if ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                ctypes.windll.kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        except OSError:
            pass
    return True


def _rewrite_block(lines, previous, use_ansi):
    block = "\n".join(lines) + "\n"
    if use_ansi and previous:
        sys.stdout.write(f"\x1b[{previous}F\x1b[J{block}")
        sys.stdout.flush()
        return len(lines)
    sys.stdout.write(block)
    sys.stdout.flush()
    return len(lines) if use_ansi else 0


class _JobAlreadyRunning(RuntimeError):
    pass


def _exclusive_job(method, run, seed):
    from contextlib import contextmanager
    @contextmanager
    def _inner():
        if os.name != "nt":
            yield
            return
        import ctypes
        from ctypes import wintypes
        api = ctypes.WinDLL("kernel32", use_last_error=True)
        api.CreateMutexW.argtypes = (ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR)
        api.CreateMutexW.restype = wintypes.HANDLE
        api.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        api.WaitForSingleObject.restype = wintypes.DWORD
        api.ReleaseMutex.argtypes = (wintypes.HANDLE,)
        api.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = api.CreateMutexW(None, False, f"Local\\OpenScoreEval_{method}_{run}_{seed}")
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        acquired = False
        try:
            if api.WaitForSingleObject(handle, 0) not in (0, 0x80):
                raise _JobAlreadyRunning(f"{method} [{run} seed={seed}] is already evaluating")
            acquired = True
            yield
        finally:
            if acquired:
                api.ReleaseMutex(handle)
            api.CloseHandle(handle)
    return _inner()


def _worker(task, output, run, device, stop_event, results):
    kind = task["id"].split(".")[1]
    method = task["method"]
    seed = 0 if task["seed"] is None else int(task["seed"])
    log_dir = Path(output) / (method or "eval") / run / f"seed_{seed}"
    log_dir.mkdir(parents=True, exist_ok=True)

    def on_progress(progress):
        payload = dict(progress)
        payload.update(id=task["id"], kind="progress")
        try:
            results.put_nowait(payload)
        except Exception:
            pass

    with (log_dir / "eval.console.log").open("a", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                with _exclusive_job(task["id"], run, seed):
                    if kind == "anchors":
                        from open_score.eval.anchors import run_anchors
                        result = run_anchors(output, run=run, stop_requested=stop_event.is_set,
                                             on_progress=on_progress)
                    elif kind == "final":
                        from open_score.eval.protocol import evaluate_checkpoint
                        checkpoint = Path(output) / method / run / f"seed_{seed}" / "best.pt"
                        result = evaluate_checkpoint(method, checkpoint, output=output, run=run,
                                                     seed=seed, device=device, on_progress=on_progress,
                                                     stop_requested=stop_event.is_set)
                    elif kind == "depth":
                        from open_score.eval.protocol import evaluate_depth_sweep
                        checkpoint = Path(output) / method / run / f"seed_{seed}" / "best.pt"
                        result = evaluate_depth_sweep(method, checkpoint, output=output, run=run,
                                                      seed=seed, device=device, on_progress=on_progress,
                                                      stop_requested=stop_event.is_set)
                    elif kind == "mech":
                        from open_score.eval.mechanism import evaluate_mechanism
                        result = evaluate_mechanism(method, output=output, run=run, seed=seed,
                                                    device=device, stop_requested=stop_event.is_set)
                    elif kind == "timing":
                        from open_score.eval.timing import evaluate_timing
                        result = evaluate_timing(output=output, run=run, device="cuda",
                                                 stop_requested=stop_event.is_set)
                    else:
                        raise ValueError(f"unknown eval job {task['id']}")
                results.put(dict(id=task["id"], status=result.get("status", "completed"), result=result))
            except _JobAlreadyRunning as error:
                print(error, flush=True)
                results.put(dict(id=task["id"], status="skipped", error=str(error)))
            except BaseException:
                detail = traceback.format_exc()
                print(detail, flush=True)
                results.put(dict(id=task["id"], status="failed", error=detail))


def _eval_job_line(item, output, run, cached):
    task = item["task"]
    seed = 0 if task["seed"] is None else int(task["seed"])
    path = Path(output) / (task["method"] or "eval") / run / f"seed_{seed}" / "eval.console.log"
    row = parse_eval_console(read_tail(path))
    row.update({key: value for key, value in cached.items() if value is not None})
    done = int(row.get("completed", row.get("eval_completed")) or 0)
    total = int(row.get("total", row.get("eval_total")) or 0)
    phase = row.get("phase") or task["id"].split(".")[1]
    extra = f"  R={int(row['cycle_depth'])}" if row.get("cycle_depth") is not None else ""
    eta = row.get("remaining_seconds")
    eta_bit = f"  eta {int(eta) // 3600}:{int(eta) % 3600 // 60:02d}:{int(eta) % 60:02d}" if eta else ""
    progress = f"{done:,}/{total:,}" if total else "starting"
    return f"{task['id']:<36}  {phase:<11}  {progress}{extra}{eta_bit}"


def run_eval(output, *, max_concurrent=2, only=None, device="cpu", run=FORMAL_RUN):
    from open_score.eval.inventory import pending_eval_jobs
    from open_score.eval.report import refresh_report
    context = get_context("spawn")
    stop_event, queue = context.Event(), context.Queue()
    failures, live_status = [], {}
    previous = signal.getsignal(signal.SIGINT)
    last_status = 0.0
    status_height = 0
    use_ansi = _enable_ansi()

    def request_stop(signum, frame):
        nonlocal status_height
        stop_event.set()
        status_height = 0
        print("stop requested: finish current episode", flush=True)

    def emit(message):
        nonlocal status_height
        status_height = 0
        print(message, flush=True)

    def show_board(live, waiting, idle=False):
        nonlocal status_height, last_status
        now = time.monotonic()
        if now - last_status < 3 and not idle:
            return
        if idle:
            lines = ["eval idle  waiting for new checkpoints  (rescan 3 min)",
                     "this window stays up; train is a separate process"]
        else:
            lines = [f"eval live={len(live)} queue={len(waiting)}"]
            lines.extend(_eval_job_line(item, output, run, live_status.get(item["task"]["id"], {}))
                         for item in live)
            if waiting:
                lines.append("queued: " + "  ".join(task["id"] for task in list(waiting)[:6])
                             + (f" +{len(waiting) - 6}" if len(waiting) > 6 else ""))
        status_height = _rewrite_block(lines, status_height, use_ansi)
        last_status = now

    signal.signal(signal.SIGINT, request_stop)
    try:
        while not stop_event.is_set():
            if (Path(output) / "eval.stop.request").exists():
                break
            try:
                jobs = pending_eval_jobs(output, only=only)
            except (OSError, MemoryError, TimeoutError) as error:
                emit(f"inventory scan skipped: {type(error).__name__}")
                jobs = []
            if not jobs:
                show_board([], [], idle=True)
                for _ in range(18):
                    if stop_event.is_set() or (Path(output) / "eval.stop.request").exists():
                        return not failures
                    time.sleep(10)
                continue
            emit(f"launching {len(jobs)} eval jobs, max concurrent {max_concurrent}")
            for task in jobs:
                emit(f"  queue {task['id']:<42} {task.get('detail') or ''}".rstrip())
            waiting, live = deque(jobs), []

            def launch():
                while waiting and len(live) < max_concurrent and not stop_event.is_set():
                    task = waiting.popleft()
                    child = context.Process(target=_worker, args=(task, str(output), run, device, stop_event, queue))
                    child.start()
                    live.append({"child": child, "task": task})
                    emit(f"{task['id']} started")

            launch()
            while live or (waiting and not stop_event.is_set()):
                if (Path(output) / "eval.stop.request").exists():
                    stop_event.set()
                still = []
                for item in live:
                    if item["child"].is_alive():
                        still.append(item)
                        continue
                    if item["child"].exitcode not in (None, 0):
                        failures.append(dict(id=item["task"]["id"], error=f"exit {item['child'].exitcode}"))
                live[:] = still
                while True:
                    try:
                        result = queue.get_nowait()
                    except Empty:
                        break
                    if result.get("kind") == "progress":
                        live_status[result.get("id")] = result
                        continue
                    emit(f"{result.get('id')} {result.get('status')}")
                    if result.get("status") == "failed":
                        failures.append(result)
                if live or (waiting and not stop_event.is_set()):
                    launch()
                    show_board(live, waiting)
                    time.sleep(1.0)
            try:
                refresh_report(output, run=run)
            except (OSError, MemoryError, TimeoutError) as error:
                emit(f"report refresh skipped: {type(error).__name__}")
        if failures:
            print("eval jobs failed:\n" + "\n".join(item.get("error", str(item)) for item in failures), flush=True)
            return False
        return not stop_event.is_set()
    finally:
        signal.signal(signal.SIGINT, previous)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--stage", choices=("inventory", "migrate", "eval", "stop"), default="eval")
    result.add_argument("--group", default="main")
    result.add_argument("--method")
    result.add_argument("--checkpoint", type=Path)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--run", default=FORMAL_RUN)
    result.add_argument("--seed", type=int, default=None)
    result.add_argument("--device", default="cpu")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--max-concurrent", type=int, choices=(1, 2, 3, 4), default=2)
    result.add_argument("--only", default=None, help="comma list: anchors,final,depth,mech,timing")
    result.add_argument("--depth-sweep", action="store_true")
    return result


def main():
    options = parser().parse_args()
    output = options.output.resolve()
    if options.stage == "stop":
        output.mkdir(parents=True, exist_ok=True)
        (output / "eval.stop.request").write_text(
            "Stop after the current complete evaluation episode.\n", encoding="utf-8")
        print("eval.stop.request written; wait for current episode, do not kill", flush=True)
        return
    if options.stage == "eval":
        (output / "eval.stop.request").unlink(missing_ok=True)
    if options.stage == "inventory":
        from open_score.eval.inventory import scan
        scan(output)
        print(f"wrote {output / 'inventory.json'}", flush=True)
        return
    if options.stage == "migrate":
        from open_score.eval.inventory import migrate_into_main
        payload = migrate_into_main(output)
        print(f"migrated into {output}; {len(payload['tasks'])} tasks")
        return
    if options.method and options.checkpoint:
        from open_score.eval import evaluate_checkpoint, evaluate_depth_sweep
        fn = evaluate_depth_sweep if options.depth_sweep else evaluate_checkpoint
        fn(options.method, options.checkpoint, output=output, run=options.run,
           seed=options.seed, device=options.device)
        return
    only = None if not options.only else {item.strip() for item in options.only.split(",") if item.strip()}
    run_eval(output, max_concurrent=options.max_concurrent, only=only, device=options.device, run=options.run)


if __name__ == "__main__":
    main()
