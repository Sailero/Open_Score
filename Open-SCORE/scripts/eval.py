"""Evaluate retained models: inventory scan, anchors, final, depth, mechanism, timing."""
from __future__ import annotations

import argparse
import json
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


def _exclusive_job(method, run, seed, lock_dir=None):
    from contextlib import contextmanager
    @contextmanager
    def _inner():
        if os.name == "nt":
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
            return
        import fcntl
        directory = Path(lock_dir) if lock_dir else Path("/tmp")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f".eval.{method}.{run}.{seed}.lock"
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise _JobAlreadyRunning(f"{method} [{run} seed={seed}] is already evaluating")
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
    return _inner()


def _worker(task, output, run, device, stop_event, results):
    from open_score.eval.experiment import is_profile, run_directory
    from open_score.utils.resources import cpu_threads
    cpu_threads()
    profile = is_profile(output)
    kind = task.get("kind") or task["id"].split(".")[1]
    method = task["method"]
    seed = 0 if task["seed"] is None else int(task["seed"])
    log_dir = run_directory(output, method or "eval", seed, task.get("env", "had"), run)
    log_dir.mkdir(parents=True, exist_ok=True)

    def profile_stop_requested():
        # The parent may be drawing the report; workers still observe stop
        # directly at episode/transaction boundaries during that interval.
        return stop_event.is_set() or (Path(output) / "eval.stop.request").exists()

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
                with _exclusive_job(task["id"], run, seed, lock_dir=log_dir):
                    if profile and kind in ("final", "depth", "readout"):
                        from open_score.eval.protocol import evaluate_profile_checkpoint
                        result = evaluate_profile_checkpoint(method, task["checkpoint"], output=output,
                            kind=kind, env=task["env"], seed=seed, device=device,
                            on_progress=on_progress, stop_requested=profile_stop_requested)
                    elif profile and kind == "probe":
                        from open_score.eval.relationship_probe import evaluate_probe
                        result = evaluate_probe(output=output, seed=seed,
                            on_progress=on_progress, stop_requested=profile_stop_requested)
                    elif kind == "anchors":
                        from open_score.eval.anchors import run_anchors
                        result = run_anchors(output, run=run, stop_requested=stop_event.is_set,
                                             on_progress=on_progress)
                    elif kind == "final":
                        from open_score.eval.protocol import evaluate_checkpoint
                        from open_score.eval.protocol import official_eval_path
                        from open_score.utils.logging import read_latest
                        directory = Path(output) / method / run / f"seed_{seed}"
                        progress_row = read_latest(output, "progress", run=run).get((method, run, seed), {})
                        checkpoint = official_eval_path(directory, progress_row)
                        if checkpoint is None:
                            raise FileNotFoundError(f"no official last-iterate weights under {directory}")
                        result = evaluate_checkpoint(method, checkpoint, output=output, run=run,
                                                     seed=seed, device=device, on_progress=on_progress,
                                                     stop_requested=stop_event.is_set)
                    elif kind == "depth":
                        from open_score.eval.protocol import official_eval_path, evaluate_depth_sweep
                        from open_score.utils.logging import read_latest
                        directory = Path(output) / method / run / f"seed_{seed}"
                        progress_row = read_latest(output, "progress", run=run).get((method, run, seed), {})
                        checkpoint = official_eval_path(directory, progress_row)
                        if checkpoint is None:
                            raise FileNotFoundError(f"no official last-iterate weights under {directory}")
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
    from open_score.eval.experiment import run_directory, is_profile
    path = run_directory(output, task["method"] or "eval", seed, task.get("env", "had"), run) / "eval.console.log"
    # Imported console tails may describe an old best-checkpoint evaluation.
    # Until this worker publishes progress, show the qualified task inventory.
    row = (dict(completed=task.get("completed", 0), total=task.get("total", 0))
           if is_profile(output) else parse_eval_console(read_tail(path)))
    row.update({key: value for key, value in cached.items() if value is not None})
    done = int(row.get("completed", row.get("eval_completed")) or 0)
    total = int(row.get("total", row.get("eval_total")) or 0)
    phase = row.get("phase") or task["id"].split(".")[1]
    extra = f"  R={int(row['cycle_depth'])}" if row.get("cycle_depth") is not None else ""
    eta = row.get("remaining_seconds")
    eta_bit = f"  eta {int(eta) // 3600}:{int(eta) % 3600 // 60:02d}:{int(eta) % 60:02d}" if eta else ""
    progress = f"{done:,}/{total:,}" if total else "starting"
    return f"{task['id']:<36}  {phase:<11}  {progress}{extra}{eta_bit}"


def run_eval(output, *, max_concurrent=2, only=None, device="cpu", run=FORMAL_RUN,
             env="had", method=None, seed=None):
    from open_score.eval.inventory import pending_eval_jobs
    from open_score.eval.report import refresh_report
    from open_score.eval.experiment import is_profile, scan
    from open_score.utils.resources import cpu_admit
    profile = is_profile(output)
    context = get_context("spawn")
    stop_event, queue = context.Event(), context.Queue()
    live, failures, live_status, seen, retry_at = [], [], {}, set(), {}
    waiting = deque()
    selected = []
    interrupted = False
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    last_status = last_scan = last_report = 0.0
    height = 0
    use_ansi = _enable_ansi() and sys.stdout.isatty()

    def request_stop(signum, frame):
        nonlocal interrupted
        interrupted = True
        stop_event.set()
        print("stop requested: finish current episodes, save records, then join workers", flush=True)

    def emit(message):
        nonlocal height
        height = 0
        print(message, flush=True)

    def ingest():
        nonlocal selected
        if profile:
            all_tasks = scan(output, env=env, only=only, probe_collector_seed=seed or 0)["tasks"]
            selected = [t for t in all_tasks if t["kind"] in only
                        and (method is None or t["method"] == method)
                        and (seed is None or t["seed"] == seed)]
            jobs = [t for t in selected if t["status"] == "pending"]
        else:
            jobs = pending_eval_jobs(output, only=only, quiet=True)
        for task in jobs:
            tid = task["id"]
            if tid in seen or time.monotonic() < retry_at.get(tid, 0):
                continue
            seen.add(tid)
            waiting.append(task)
            emit(f"queue {tid} {task.get('completed', 0)}/{task.get('total', 0)}")

    def launch():
        while waiting and len(live) < max_concurrent and not stop_event.is_set():
            selected_index = next((i for i, task in enumerate(waiting)
                                   if not profile or cpu_admit(env, task["kind"], live, max_concurrent)), None)
            if selected_index is None:
                return
            task = waiting[selected_index]
            del waiting[selected_index]
            child = context.Process(target=_worker, args=(task, str(output), run, device, stop_event, queue))
            child.start()
            live.append(dict(child=child, task=task))
            emit(f"{task['id']} started pid={child.pid}")

    def receive():
        nonlocal last_scan
        while True:
            try:
                result = queue.get_nowait()
            except Empty:
                return
            if result.get("kind") == "progress":
                live_status[result["id"]] = result
                continue
            emit(f"{result.get('id')} {result.get('status')}")
            status = result.get("status")
            if status in ("failed", "incomplete"):
                failures.append(result)
            elif status in ("blocked", "skipped"):
                seen.discard(result["id"])
                retry_at[result["id"]] = time.monotonic() + 60
            last_scan = 0  # Resolve readout/probe dependencies after atomic result writes.

    for sig in previous:
        signal.signal(sig, request_stop)
    try:
        while not stop_event.is_set():
            if (Path(output) / "eval.stop.request").exists():
                interrupted = True
                stop_event.set()
                break
            now = time.monotonic()
            receive()
            changed = False
            for item in list(live):
                child = item["child"]
                if child.is_alive():
                    continue
                child.join()
                live.remove(item)
                changed = True
                if child.exitcode != 0:
                    failures.append(dict(id=item["task"]["id"], error=f"worker exit {child.exitcode}"))
            if changed:
                receive()
                last_scan = 0
            if now - last_scan >= 60:
                ingest()
                last_scan = now
            launch()
            if profile and not live and not waiting:
                unfinished = [t for t in selected if t["status"] != "complete"]
                if not unfinished:
                    emit("all selected evaluation tasks complete")
                    break
                if failures and all(t["id"] in seen for t in unfinished):
                    break
            if now - last_status >= (5 if live else 60):
                lines = [f"eval {env} live={len(live)} queued={len(waiting)} "
                         f"waiting_dependencies={sum(t['status'] == 'waiting' for t in selected)}"]
                lines.extend(_eval_job_line(i, output, run, live_status.get(i["task"]["id"], {})) for i in live)
                height = _rewrite_block(lines, height, use_ansi)
                last_status = now
            if now - last_report >= 300:
                refresh_report(output, run=run)
                last_report = now
            time.sleep(1)
    finally:
        stop_event.set()
        last_notice = 0
        while live:
            receive()
            for item in list(live):
                item["child"].join(timeout=.2)
                if not item["child"].is_alive():
                    live.remove(item)
            if live and time.monotonic() - last_notice > 30:
                emit(f"waiting for {len(live)} workers to finish current episodes and save")
                last_notice = time.monotonic()
        receive()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        refresh_report(output, run=run)
    if failures:
        print("Evaluation incomplete: " + "\n".join(str(x) for x in failures), flush=True)
    return not failures and not interrupted


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--profile", choices=("main0921",))
    result.add_argument("--env", choices=("had", "smacv2"), default="had")
    result.add_argument("--stage", choices=("inventory", "migrate", "eval", "stop"), default="eval")
    result.add_argument("--group", default="main")
    result.add_argument("--method")
    result.add_argument("--checkpoint", type=Path)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--run", default=FORMAL_RUN)
    result.add_argument("--seed", type=int, default=None)
    result.add_argument("--device", default="cpu")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--max-concurrent", type=int, choices=range(1, 17), default=2)
    result.add_argument("--only", default=None, help="comma list: final,depth,readout,probe; separately timing")
    result.add_argument("--depth-sweep", action="store_true")
    return result


def run_profile_timing(output, stop_requested):
    """Cost cells are atomic; an OOM retries unfinished cells on the same card."""
    from open_score.eval.experiment import initialize, atomic_json, scan
    from open_score.eval.timing import evaluate_profile_timing
    from open_score.utils.resources import gpu_memory, is_cuda_oom
    manifest = initialize(output)
    task = next(t for t in scan(output, env="had")["tasks"] if t["kind"] == "timing")
    if task["status"] == "waiting":
        return dict(status="blocked", reason="M4 needs all cost finals and the complete common state bank")
    selected = manifest["resources"].get("measurement_gpu")
    required_free, retries = 12., 0
    last_notice = 0.
    while not stop_requested():
        memories = {}
        for gpu in ((selected,) if selected is not None else (0, 1)):
            try:
                memories[gpu] = gpu_memory(gpu)
            except Exception:
                pass
        eligible = [gpu for gpu in memories if memories[gpu]["free"] >= required_free]
        if not eligible:
            if time.monotonic() - last_notice > 30:
                print(f"M4 waits for >= {required_free:.1f} GiB free on registered GPU {selected}", flush=True)
                last_notice = time.monotonic()
            time.sleep(1)
            continue
        if selected is None:
            selected = max(eligible, key=lambda gpu: memories[gpu]["free"])
            manifest["resources"]["measurement_gpu"] = selected
            atomic_json(Path(output) / "experiment.json", manifest)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(selected)
        try:
            result = evaluate_profile_timing(output=output, stop_requested=stop_requested)
            failure_path = Path(output) / "timing_failure.json"
            if failure_path.exists():
                previous = json.loads(failure_path.read_text(encoding="utf-8"))
                atomic_json(failure_path, {**previous, "status": result.get("status", "completed"),
                                          "updated_at": time.time()})
            return result
        except Exception as error:
            if not is_cuda_oom(error):
                raise
            import torch
            retries += 1
            required_free = max(required_free, 8 + 1.25 * torch.cuda.max_memory_reserved() / 1024**3)
            atomic_json(Path(output) / "timing_failure.json", dict(kind="cuda_oom", retry=retries,
                physical_gpu=selected, completed_cells_retained=True, error=str(error),
                status="retry_wait" if retries <= 3 else "failed", required_free_gib=required_free))
        # Drop the exception frame before releasing the allocator cache.
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        if retries > 3:
            return dict(status="failed", reason="M4 CUDA OOM automatic retries exhausted; completed cells retained")
        deadline = time.monotonic() + 60 * 2 ** (retries - 1)
        print(f"M4 CUDA OOM: retaining completed cells; controlled retry {retries}/3", flush=True)
        while time.monotonic() < deadline and not stop_requested():
            time.sleep(1)
    failure_path = Path(output) / "timing_failure.json"
    if failure_path.exists():
        previous = json.loads(failure_path.read_text(encoding="utf-8"))
        atomic_json(failure_path, {**previous, "status": "stopped", "updated_at": time.time()})
    return dict(status="stopped")


def profile_main(options, output):
    from open_score.eval.experiment import (initialize, scan, import_existing, SEEDS, methods,
                                          checkpoint_info, run_directory)
    from open_score.utils.resources import queue_lock, cpu_threads, clear_previous_stop, configure_workspace
    initialize(output)
    workspace = configure_workspace()
    if options.env == "smacv2":
        os.environ["SC2PATH"] = str(workspace / "envs/StarCraftII")
    cpu_threads()
    requested_at = time.time()
    stop_path = output / "eval.stop.request"
    def new_stop():
        return stop_path.exists() and stop_path.stat().st_mtime > requested_at
    if options.run != "train" or options.seed not in (None, *SEEDS):
        raise SystemExit("main0921 fixes run=train and model seeds 0,1,2")
    if options.method is not None and options.method not in methods(options.env):
        raise SystemExit("--method must belong to the fixed main0921 environment matrix")
    if options.stage == "inventory":
        inventory = scan(output)
        for task in inventory["tasks"]:
            if task["env"] == options.env:
                print(f"{task['id']:<48} {task['status']:<10} {task['completed']}/{task['total']}")
        return
    if options.stage == "migrate":
        imported = import_existing(output, output.parent / "main")
        print(f"Imported {len(imported)} final checkpoints")
        return
    only = set(options.only.split(",")) if options.only else (
        {"final", "depth", "readout", "probe"} if options.env == "had" else {"final"})
    only = {k.strip() for k in only}
    if options.depth_sweep:
        only = {"depth"}
    allowed = {"final", "depth", "readout", "probe", "timing"} if options.env == "had" else {"final"}
    if not only or not only <= allowed:
        raise SystemExit(f"Unsupported main0921 {options.env} evaluation kinds: {sorted(only)}")
    if options.method not in (None, "regir"):
        if options.only and only & {"depth", "readout", "probe"}:
            raise SystemExit("Depth/readout/probe interventions apply only to HAD Full (--method regir)")
        only &= {"final", "timing"}
    if "timing" in only:
        if only != {"timing"}:
            raise SystemExit("GPU cost measurement must run separately: --only timing")
        if options.method is not None or options.seed is not None:
            raise SystemExit("M4 uses its fixed eight-method, three-seed matrix; omit --method and --seed")
        from open_score.eval.report import refresh_report
        with queue_lock(output, "gpu", stop_requested=new_stop):
            clear_previous_stop(stop_path, requested_at)
            result = run_profile_timing(output, stop_requested=lambda: stop_path.exists())
        refresh_report(output, run="train")
        print(result)
        if result.get("status") not in ("complete", "completed"):
            raise SystemExit(1)
        return
    if options.device != "cpu":
        raise SystemExit("main0921 formal evaluation and mechanisms require --device cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if options.checkpoint:
        if not options.method:
            raise SystemExit("--checkpoint requires --method")
        info = checkpoint_info(options.checkpoint, method=options.method, seed=options.seed, env=options.env)
        expected = run_directory(output, options.method, info["seed"], options.env) / "final.pt"
        if expected.resolve() != options.checkpoint.resolve():
            raise SystemExit("Checkpoint must be the registered final in this independent output")
        options.seed = info["seed"]
    with queue_lock(output, "cpu", stop_requested=new_stop):
        clear_previous_stop(stop_path, requested_at)
        if options.env == "smacv2":
            from open_score.envs.smacv2_env import generate_registered_scenes
            generate_registered_scenes(output)
        success = run_eval(output, max_concurrent=options.max_concurrent, only=only,
                           device="cpu", run="train", env=options.env,
                           method=options.method, seed=options.seed)
    if not success:
        raise SystemExit(1)


def main():
    options = parser().parse_args()
    output = options.output.resolve()
    if options.profile and output == DEFAULT_OUTPUT.resolve():
        if any(arg == "--output" or arg.startswith("--output=") for arg in sys.argv):
            raise SystemExit("main0921 must use an independent output, not outputs/main")
        output = DEFAULT_OUTPUT.parent / "main0921"
    if options.stage == "stop":
        output.mkdir(parents=True, exist_ok=True)
        from open_score.utils.resources import write_stop
        write_stop(output / "eval.stop.request", "Stop after the current complete evaluation episode.")
        print("eval.stop.request written; wait for current episode, do not kill", flush=True)
        return
    from open_score.eval.experiment import is_profile
    if options.profile or is_profile(output):
        return profile_main(options, output)
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
