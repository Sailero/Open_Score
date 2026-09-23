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
import subprocess
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
    # Each evaluator sees only its assigned physical GPU. CPU workers never
    # initialize CUDA, and the parent keeps both cards visible for admission.
    physical_gpu = task.get("physical_gpu")
    if device == "cpu" or "assigned_jobs" in task:
        os.environ["CUDA_VISIBLE_DEVICES"] = "" if physical_gpu is None else str(physical_gpu)
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

    log_name = f"eval.{task['shard_id']}.console.log" if task.get("shard_id") else "eval.console.log"
    with (log_dir / log_name).open("a", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                from open_score.eval.experiment import EVAL_PHASES
                with _exclusive_job(task["id"], run, seed, lock_dir=log_dir):
                    if profile and kind in EVAL_PHASES:
                        from open_score.eval.protocol import evaluate_profile_checkpoint
                        result = evaluate_profile_checkpoint(method, task["checkpoint"], output=output,
                            kind=kind, env=task["env"], seed=seed, device=device,
                            on_progress=on_progress, stop_requested=profile_stop_requested,
                            assigned_jobs=task.get("assigned_jobs"), shard_id=task.get("shard_id"),
                            resume_shard=task.get("resume_shard", False))
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
            except BaseException as error:
                detail = traceback.format_exc()
                print(detail, flush=True)
                from open_score.utils.resources import is_cuda_oom
                recoverable = "assigned_jobs" in task and device == "cuda" and is_cuda_oom(error)
                results.put(dict(id=task["id"], status="retry_cpu" if recoverable else "failed", error=detail))


def _final_shards(task, output, count, index=None, read1_equivalent=False):
    """Partition the frozen episode identities of one checkpoint/kind into stable shards."""
    from open_score.eval.experiment import (checkpoint_info, evaluation_jobs,
        remaining_evaluations, result_identity)
    from open_score.utils.logging import read_records
    info = checkpoint_info(task["checkpoint"], method=task["method"], seed=task["seed"], env=task["env"])
    kind = task["kind"]
    canonical = evaluation_jobs(info, kind)
    count = max(1, min(count, (len(canonical) + 74) // 75))
    if index is None:
        existing = read_records(output, "episodes", run="train", env=task["env"],
                                method=task["method"], seed=task["seed"])
        remaining = remaining_evaluations(info, kind, existing, read1_equivalent=read1_equivalent)
    else:
        remaining = remaining_evaluations(info, kind, read1_equivalent=read1_equivalent, index=index)
    pending = {result_identity(job) for job in remaining}
    # Contiguous blocks preserve config locality (especially SC2 environment
    # startup) and provide stable IDs across stop/resume and device changes.
    width = max(1, (len(canonical) + count - 1) // count)
    shards = []
    for start in range(0, len(canonical), width):
        assigned = [j for j in canonical[start:start + width] if result_identity(j) in pending]
        if not assigned:
            continue
        shard_id = f"part{start // width:03d}"
        shards.append({**task, "id": f"{task['id']}.{shard_id}", "parent_id": task["id"],
                       "shard_id": shard_id, "assigned_jobs": assigned,
                       "completed": 0, "total": len(assigned)})
    # Expensive large-agent scenes start first, reducing the serial tail.
    return sorted(shards, key=lambda t: -t["assigned_jobs"][0]["config"]["N_R"])


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
             env="had", method=None, seed=None, final_shards=1, devices=(0, 1),
             gpu_workers_per_device=2, queue_path=None):
    from open_score.eval.inventory import pending_eval_jobs
    from open_score.eval.report import refresh_report, refresh_report_async
    from open_score.eval.experiment import is_profile, scan
    from open_score.utils.resources import cpu_admit, gpu_memory
    profile = is_profile(output)
    from open_score.eval.experiment import EpisodeIndex, EVAL_PHASES, initialize, read_json
    index = EpisodeIndex(output) if profile else None
    read1 = bool(initialize(output)["mechanisms"].get("read1_equivalence_verified")) if profile else False
    queue_state = dict(closed=queue_path is None, stamp=None)
    failure_counts = {}
    context = get_context("spawn")
    stop_event, queue = context.Event(), context.Queue()
    live, failures, live_status, seen, retry_at = [], [], {}, set(), {}
    waiting = deque()
    selected = []
    shard_tasks, parent_parts, finished_parts = {}, {}, set()
    interrupted = False
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    last_status = last_scan = last_report = 0.0
    last_checkpoint_poll = last_gpu_poll = 0.0
    checkpoint_stamp = None
    gpu_snapshots = {}
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
            allowed = None
            if queue_path is not None:
                state = read_json(queue_path, {}) or {}
                allowed = set(state.get("eval", []))
                queue_state["closed"] = bool(state.get("closed"))
            all_tasks = scan(output, env=env, only=only, index=index)["tasks"]
            selected = [t for t in all_tasks if t["kind"] in only
                        and (allowed is None or t["id"] in allowed)
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
            if profile and task["kind"] in EVAL_PHASES and final_shards > 1:
                parts = _final_shards(task, output, final_shards, index=index, read1_equivalent=read1)
                parent_parts[tid] = {part["id"] for part in parts}
                shard_tasks.update({part["id"]: part for part in parts})
                seen.update(parent_parts[tid])
                waiting.extend(parts)
                if not parts:
                    task.update(status="complete", completed=task["total"])
            else:
                waiting.append(task)
            emit(f"queue {tid} {task.get('completed', 0)}/{task.get('total', 0)}")

    def launch():
        nonlocal last_gpu_poll, gpu_snapshots
        if profile:
            # New final checkpoints take precedence over queued mechanism
            # work. Stable order preserves seeds; running episodes continue.
            priority = {"final": 0, "gate_depth": 1, "dup": 2, "depth": 3, "r1deploy": 4, "readout": 5,
                        "intent_intervention": 6, "probe": 7, "timing": 8}
            ordered = sorted(waiting, key=lambda task: (task.get("rank", 0), priority.get(task["kind"], 9)))
            waiting.clear()
            waiting.extend(ordered)
        if device == "auto" and waiting and time.monotonic() - last_gpu_poll >= 5:
            gpu_snapshots = {}
            for gpu in devices:
                try:
                    gpu_snapshots[gpu] = gpu_memory(gpu)
                except (OSError, ValueError, subprocess.SubprocessError):
                    pass
            last_gpu_poll = time.monotonic()
        while waiting and len(live) < max_concurrent and not stop_event.is_set():
            selected_index = next((i for i, task in enumerate(waiting)
                                   if time.monotonic() >= retry_at.get(task["id"], 0)
                                   and not any(item["task"]["id"] == task["id"] for item in live)
                                   and (not profile or cpu_admit(env, task["kind"], live, max_concurrent))), None)
            if selected_index is None:
                return
            task = waiting[selected_index]
            del waiting[selected_index]
            task_device = "cpu" if device == "auto" else device
            if device == "auto" and task["kind"] in EVAL_PHASES and not task.get("force_cpu"):
                for gpu in sorted(gpu_snapshots, key=lambda g: -gpu_snapshots[g]["free"]):
                    row = gpu_snapshots[gpu]
                    occupied = sum(item["task"].get("physical_gpu") == gpu for item in live)
                    # Busy training cards get one inference process; a card
                    # with compute headroom can admit the configured maximum.
                    limit = min(gpu_workers_per_device, 1 if row["utilization"] >= 90 else gpu_workers_per_device)
                    if occupied < limit and row["free"] >= 8 + 2 * (occupied + 1):
                        task["physical_gpu"] = gpu
                        task_device = "cuda"
                        break
            task["eval_device"] = task_device
            child = context.Process(target=_worker, args=(task, str(output), run, task_device, stop_event, queue))
            child.start()
            live.append(dict(child=child, task=task, started_at=time.monotonic()))
            emit(f"{task['id']} started pid={child.pid} device={task_device} physical_gpu={task.get('physical_gpu')}")

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
            tid = result["id"]
            if status == "retry_cpu":
                task = shard_tasks[tid]
                task.update(force_cpu=True, resume_shard=True)
                task.pop("physical_gpu", None)
                waiting.append(task)
                retry_at[tid] = time.monotonic() + 5
                emit(f"{tid}: CUDA OOM; resume only unfinished episodes on CPU")
            elif status in ("failed", "incomplete"):
                failure_counts[tid] = failure_counts.get(tid, 0) + 1
                if profile and queue_path is not None and failure_counts[tid] <= 2:
                    emit(f"{tid}: {status}; retry {failure_counts[tid]}/2 in 5 min")
                    if tid in shard_tasks:
                        shard_tasks[tid]["resume_shard"] = True
                        waiting.append(shard_tasks[tid])
                    else:
                        seen.discard(tid)
                    retry_at[tid] = time.monotonic() + 300
                else:
                    failures.append(result)
            elif status in ("blocked", "skipped"):
                if tid in shard_tasks:
                    shard_tasks[tid]["resume_shard"] = True
                    waiting.append(shard_tasks[tid])
                else:
                    seen.discard(tid)
                retry_at[tid] = time.monotonic() + 60
            if tid in shard_tasks:
                if status in ("complete", "completed"):
                    finished_parts.add(tid)
                parent = shard_tasks[tid]["parent_id"]
                if parent_parts[parent] <= finished_parts:
                    last_scan = 0
            else:
                last_scan = 0  # Resolve dependencies after atomic result writes.

    def publish_status(status="running"):
        if not profile:
            return
        from open_score.eval.experiment import atomic_json
        active = [{**{k: v for k, v in item["task"].items() if k != "assigned_jobs"},
                   **live_status.get(item["task"]["id"], {}), "pid": item["child"].pid,
                   "kind": item["task"]["kind"]} for item in live]
        active_ids = {row.get("parent_id", row["id"]) for row in active}
        atomic_json(Path(output) / f"scheduler.eval.{env}.json", dict(
            kind="eval", env=env, pid=os.getpid(), status=status, updated_at=time.time(),
            max_concurrent=max_concurrent, device=device, devices=list(devices), final_shards=final_shards,
            gpu_workers_per_device=gpu_workers_per_device, total=len(selected),
            completed=sum(t["status"] == "complete" for t in selected), live=active,
            completed_ids=[t["id"] for t in selected if t["status"] == "complete"],
            waiting=[t for t in selected if t["status"] != "complete" and t["id"] not in active_ids],
            failures=failures))

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
                    tid = item["task"]["id"]
                    failure_counts[tid] = failure_counts.get(tid, 0) + 1
                    if profile and queue_path is not None and failure_counts[tid] <= 2:
                        emit(f"{tid}: worker exit {child.exitcode}; retry {failure_counts[tid]}/2 in 5 min")
                        if tid in shard_tasks:
                            shard_tasks[tid]["resume_shard"] = True
                            waiting.append(shard_tasks[tid])
                        else:
                            seen.discard(tid)
                        retry_at[tid] = time.monotonic() + 300
                    else:
                        failures.append(dict(id=tid, error=f"worker exit {child.exitcode}"))
            if changed:
                receive()
                if not parent_parts:
                    last_scan = 0
            for tid, deadline in list(retry_at.items()):
                if tid not in seen and now >= deadline:
                    retry_at.pop(tid)
                    last_scan = 0
            # Poll tiny final.pt metadata frequently. Do not parse the growing
            # episode CSV every minute while nothing is ready.
            if profile and now - last_checkpoint_poll >= 5:
                from open_score.eval.experiment import methods, run_directory, method_seeds
                paths = [run_directory(output, m, s, env) / "final.pt"
                         for m in methods(env) for s in method_seeds(env, m)]
                if queue_path is not None:
                    paths.append(Path(queue_path))
                paths.append(Path(output) / "decision" / "branch.json")
                stamp = tuple((str(p), p.stat().st_mtime_ns) for p in paths if p.exists())
                if stamp != checkpoint_stamp:
                    checkpoint_stamp = stamp
                    last_scan = 0
                last_checkpoint_poll = now
            if last_scan == 0 or (not profile and now - last_scan >= 60):
                ingest()
                last_scan = now
            launch()
            if profile and not live and not waiting:
                unfinished = [t for t in selected if t["status"] != "complete"]
                if queue_path is None:
                    if not unfinished:
                        emit("all selected evaluation tasks complete")
                        break
                    if failures:
                        break
                elif queue_state["closed"] and not any(t["status"] == "pending" for t in selected):
                    emit("queue closed and no runnable evaluation remains")
                    break
            if now - last_status >= (5 if live else 60):
                lines = [f"eval {env} live={len(live)} queued={len(waiting)} "
                         f"waiting_dependencies={sum(t['status'] == 'waiting' for t in selected)}"]
                lines.extend(_eval_job_line(i, output, run, live_status.get(i["task"]["id"], {})) for i in live)
                height = _rewrite_block(lines, height, use_ansi)
                publish_status()
                last_status = now
            if now - last_report >= 300:
                refresh_report_async(output, run=run)
                last_report = now
            time.sleep(1)
    except BaseException as error:
        failures.append(dict(id=f"scheduler.eval.{env}", status="failed",
                             error=f"Parent scheduler: {type(error).__name__}: {error}"))
        raise
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
        try:
            refresh_report(output, run=run)
        except Exception as error:
            failures.append(dict(id=f"report.{env}", status="failed", error=str(error)))
            print(f"Final report refresh failed: {error}", flush=True)
        try:
            publish_status("failed" if failures else "stopped" if interrupted else "completed")
        except (OSError, ValueError) as error:
            print(f"Final scheduler status write failed: {error}", flush=True)
    if failures:
        print("Evaluation incomplete: " + "\n".join(str(x) for x in failures), flush=True)
    if queue_path is not None:
        return not interrupted
    return not failures and not interrupted


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--profile", choices=("main0923",))
    result.add_argument("--queue", type=Path, default=None, help="pipeline queue file (main0923)")
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
    result.add_argument("--max-concurrent", type=int, choices=range(1, 257), default=2)
    result.add_argument("--devices", default="0,1", help="permitted physical GPUs for --device auto")
    result.add_argument("--gpu-workers-per-device", type=int, choices=range(1, 5), default=2)
    result.add_argument("--final-shards", type=int, choices=range(1, 301), default=1,
                        help="stable episode partitions per final checkpoint")
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
    if task["status"] == "complete":
        return dict(status="complete", completed=task["total"], total=task["total"], reused=True)
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
    from open_score.eval.experiment import (initialize, atomic_json, scan, import_main0921, methods,
                                          checkpoint_info, run_directory, PROFILE, SOURCE_PROFILE,
                                          EVAL_PHASES, ALL_SEEDS)
    from open_score.utils.resources import queue_lock, cpu_threads, clear_previous_stop, configure_workspace
    initialize(output)
    workspace = configure_workspace()
    if options.env == "smacv2":
        os.environ["SC2PATH"] = str(workspace / "envs/StarCraftII")
    cpu_threads()
    requested_at = time.time()
    if not any(a.startswith("--max-concurrent") for a in sys.argv):
        options.max_concurrent = 4
    stop_path = output / "eval.stop.request"
    def new_stop():
        return stop_path.exists() and stop_path.stat().st_mtime > requested_at
    if options.run != "train" or options.seed not in (None, *ALL_SEEDS):
        raise SystemExit(f"{PROFILE} fixes run=train and model seeds 0-4")
    if options.method is not None and options.method not in methods(options.env):
        raise SystemExit(f"--method must belong to the fixed {PROFILE} environment matrix")
    if options.stage == "inventory":
        inventory = scan(output)
        for task in inventory["tasks"]:
            if task["env"] == options.env:
                print(f"{task['id']:<56} {task['status']:<10} {task['completed']}/{task['total']}")
        return
    if options.stage == "migrate":
        imported = import_main0921(output, output.parent / SOURCE_PROFILE)
        print(f"Imported {len(imported)} final checkpoints from {SOURCE_PROFILE}")
        return
    allowed = set(EVAL_PHASES) if options.env == "had" else {"final", "depth"}
    only = ({k.strip() for k in options.only.split(",")} if options.only else set(allowed))
    if not only or not only <= allowed:
        raise SystemExit(f"Unsupported {PROFILE} {options.env} evaluation kinds: {sorted(only - allowed)}")
    if options.device not in ("cpu", "auto"):
        raise SystemExit(f"{PROFILE} evaluation uses --device cpu or --device auto (CPU+GPU)")
    try:
        devices = tuple(int(x) for x in options.devices.split(","))
    except ValueError:
        raise SystemExit("--devices must select physical GPU 0 and/or 1")
    if not devices or len(set(devices)) != len(devices) or not set(devices) <= {0, 1}:
        raise SystemExit("--devices must select distinct physical GPUs from 0,1")
    if options.device == "auto" and options.final_shards == 1:
        raise SystemExit("--device auto requires --final-shards greater than one")
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if options.device == "cpu" else ",".join(map(str, devices))
    with queue_lock(output, f"cpu.{options.env}", stop_requested=new_stop):
        if stop_path.exists() and stop_path.stat().st_mtime <= requested_at:
            clear_previous_stop(stop_path, requested_at)
        manifest = initialize(output)
        manifest["resources"].setdefault("evaluation", {})[options.env] = dict(
            device=options.device, max_concurrent=options.max_concurrent,
            final_shards=options.final_shards, physical_gpus=list(devices) if options.device == "auto" else [],
            gpu_workers_per_device=options.gpu_workers_per_device,
            gpu_reserve_gib=8, checkpoint_poll_seconds=5, cpu_threads_per_worker=1,
            policy_rng="per_episode_seed_v1", updated_at=time.time())
        atomic_json(output / "experiment.json", manifest)
        if options.env == "smacv2":
            from open_score.envs.smacv2_env import generate_registered_scenes
            generate_registered_scenes(output)
        success = run_eval(output, max_concurrent=options.max_concurrent, only=only,
                           device=options.device, run="train", env=options.env,
                           method=options.method, seed=options.seed, final_shards=options.final_shards,
                           devices=devices, gpu_workers_per_device=options.gpu_workers_per_device,
                           queue_path=options.queue)
    if not success:
        raise SystemExit(1)


def main():
    options = parser().parse_args()
    output = options.output.resolve()
    if options.profile and output == DEFAULT_OUTPUT.resolve():
        if any(arg == "--output" or arg.startswith("--output=") for arg in sys.argv):
            raise SystemExit("the profile must use an independent output, not outputs/main")
        output = DEFAULT_OUTPUT.parent / options.profile
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
