"""Explicit HAD training with six training slots and a separate CPU or GPU evaluation pool."""
from __future__ import annotations
import argparse
from contextlib import redirect_stdout, redirect_stderr
from collections import deque
from multiprocessing import get_context
from pathlib import Path
from queue import Empty
import os
import math
import signal
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "1")
from open_score.algos import METHODS
from open_score.utils.logging import DEFAULT_OUTPUT
from open_score.eval.experiment import atomic_json, read_json, run_directory, process_lock, logical_devices, physical_device


def selection(text, allowed=None, integer=False):
    values = tuple(int(v.strip()) if integer else v.strip() for v in str(text).split(",") if v.strip())
    if not values or len(set(values)) != len(values):
        raise ValueError("Selection must be nonempty and contain no duplicates")
    if integer and min(values) < 0:
        raise ValueError("Seeds must be non-negative")
    if allowed is not None and any(v not in allowed for v in values):
        raise ValueError(f"Unsupported PURE learning method; choose from {', '.join(allowed)}. Rule and Random are evaluation anchors.")
    return values


def qualified_final(output, method, seed, steps):
    path = run_directory(output, method, seed) / "final.pt"
    if not path.exists():
        return None
    import torch
    saved = torch.load(path, map_location="cpu", weights_only=False)
    cfg, progress = saved.get("config", {}), saved.get("progress", {})
    if cfg.get("method") != method or cfg.get("env") != "had" or int(cfg.get("seed", -1)) != seed:
        raise ValueError(f"Model identity differs at {path}")
    if int(cfg.get("t_max", -1)) != steps:
        raise ValueError(f"Existing model has another budget at {path}; select an independent output")
    return path if int(progress.get("t_env", 0)) >= steps else None


def train_worker(options, method, seed, gpu, stop, results):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    directory = run_directory(options.output, method, seed)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        with process_lock(directory / ".train.lock") as owned:
            if not owned:
                raise RuntimeError(f"Model already has a trainer: {method}/s{seed}")
            with (directory / "console.log").open("a", encoding="utf-8", buffering=1) as stream, redirect_stdout(stream), redirect_stderr(stream):
                final = qualified_final(options.output, method, seed, options.steps)
                if final is None:
                    from open_score.algos import train, resume_files
                    recover = options.resume and any(path.exists() for path in resume_files(directory))
                    final = train(method, dict(output=str(options.output), seed=seed, t_max=options.steps,
                        run=options.run, resume=recover, batch_size_run=8, training_iters=8,
                        use_cuda=True, device="cuda", _stop_event=stop))
                final = qualified_final(options.output, method, seed, options.steps)
                results.put(("train", method, seed, "complete" if final else "stopped", None))
    except BaseException:
        results.put(("train", method, seed, "failed", traceback.format_exc()))


def eval_worker(options, method, seed, stop, results):
    directory = run_directory(options.output, method, seed)
    try:
        from open_score.eval.protocol import evaluate_checkpoint
        with (directory / "eval.console.log").open("a", encoding="utf-8", buffering=1) as stream, redirect_stdout(stream), redirect_stderr(stream):
            check_stop = lambda: stop.is_set() or (options.output / "eval.stop.request").exists()
            result = evaluate_checkpoint(method, directory / "final.pt", output=options.output,
                run=options.run, seed=seed, device=options.eval_device, stop_requested=check_stop)
            if result["status"] == "complete" and options.confirmation:
                result = evaluate_checkpoint(method, directory / "final.pt", output=options.output,
                    run=options.run, seed=seed, device=options.eval_device, stop_requested=check_stop, confirmation=True)
            results.put(("eval", method, seed, result["status"], None))
    except BaseException:
        results.put(("eval", method, seed, "failed", traceback.format_exc()))


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Worker count must be a positive integer")
    return number


def evaluation_device(value):
    value = str(value).strip().lower()
    if value in ("cpu", "cuda") or (value.startswith("cuda:") and value[5:].isdigit()):
        return value
    raise argparse.ArgumentTypeError("Device must be cpu, cuda or cuda:N")


def duration_text(value):
    if value is None or not math.isfinite(float(value)):
        return "-"
    seconds = max(0, int(value))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def damage_text(value):
    return "-" if value is None or not math.isfinite(float(value)) else f"{float(value):.3f}"


def global_eta(rows, queued):
    if queued or any(row.get("remaining_seconds") is None for row in rows):
        return None
    return max((float(row["remaining_seconds"]) for row in rows), default=0.)


def progress_lines(output, state=None):
    state = read_json(Path(output) / "scheduler.train.json", {}) if state is None else state
    lines = ["pool   method                  seed device progress        elapsed     ETA         D       status"]
    measurements = []
    for pool in ("train", "eval"):
        for item in state.get("active", []):
            if item["pool"] != pool:
                continue
            path = run_directory(output, item["method"], item["seed"])
            row = read_json(path / ("resource.json" if pool == "train" else "evaluation.json"), {})
            measurements.append(row)
            done, total = (row.get("t_env", 0), row.get("budget_steps", 0)) if pool == "train" else (row.get("completed", 0), row.get("total", 0))
            damage = row.get("latest_validation_D") if pool == "train" else row.get("D")
            device = str(item.get("gpu", state.get("eval_device", "cpu")))
            lines.append(f"{pool:<6} {item['method']:<23} {item['seed']:>4} {device:>6} {done:>7}/{total:<7} {duration_text(row.get('elapsed_seconds')):<11} {duration_text(row.get('remaining_seconds')):<11} {damage_text(damage):<7} {row.get('current_phase', row.get('phase', 'starting'))}")
    for item in state.get("finished", []):
        if item["status"] == "failed":
            lines.append(f"failed: {item['pool']} {item['method']}/s{item['seed']}; see its console log")
    done_train = sum(item["pool"] == "train" and item["status"] == "complete" for item in state.get("finished", []))
    done_eval = sum(item["pool"] == "eval" and item["status"] == "complete" for item in state.get("finished", []))
    queued = len(state.get("waiting_train", [])) + len(state.get("waiting_eval", []))
    if state.get("status", "idle") in ("failed", "stopped", "idle"):
        eta = None
    else:
        eta = global_eta(measurements, queued or done_eval < state.get("total_eval", 0) - sum(i["pool"] == "eval" for i in state.get("active", [])))
    lines.append(f"train {done_train}/{state.get('total_train', 0)} ({state.get('train_slots', 6)} slots)  eval {done_eval}/{state.get('total_eval', 0)} ({state.get('eval_slots', 2)} slots)  waiting train={len(state.get('waiting_train', []))} eval={len(state.get('waiting_eval', []))}  elapsed={duration_text(time.time() - state['started_at']) if state.get('started_at') else '-'} ETA={duration_text(eta)}  {state.get('status', 'idle')}")
    return lines


def render(lines, previous=0):
    if previous and sys.stdout.isatty():
        sys.stdout.write(f"\x1b[{previous}F\x1b[J")
    print("\n".join(lines), flush=True)
    return len(lines)


def run(options):
    from open_score.eval.experiment import check_environment, ensure_layout
    check_environment()
    ensure_layout(options.output)
    methods = selection(options.method, METHODS)
    seeds = selection(options.seeds, integer=True)
    devices = logical_devices(options.devices)
    if options.steps < 50 or options.per_gpu != 3 or len(devices) != 2:
        raise ValueError("Current training requires an explicit budget >=50 and two GPUs with three jobs each")
    ctx = get_context("spawn")
    stop, results = ctx.Event(), ctx.Queue()
    waiting_train = deque((m, s) for s in seeds for m in methods)
    waiting_eval, active_train, active_eval = deque(), {}, {}
    finished, enqueued = {}, set()
    total_tasks, started_at = len(waiting_train), time.time()
    slots = tuple(physical_device(gpu) for _ in range(3) for gpu in devices)
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in handlers:
        signal.signal(sig, lambda *_: stop.set())
    def accept(message):
        pool, method, seed, status, error = message
        finished[(pool, method, seed)] = dict(pool=pool, method=method, seed=seed, status=status, error=error)
        if error:
            print(error, file=sys.stderr, flush=True)
        if pool == "train" and status == "complete" and (method, seed) not in enqueued:
            waiting_eval.append((method, seed))
            enqueued.add((method, seed))
    def snapshot(status):
        active = [dict(pool=pool, method=i["method"], seed=i["seed"], gpu=i.get("gpu", options.eval_device), pid=i["child"].pid)
                  for pool, members in (("train", active_train), ("eval", active_eval)) for i in members.values()]
        return dict(pid=os.getpid(), updated_at=time.time(), status=status, train_slots=6,
                    eval_slots=options.eval_workers, eval_device=options.eval_device, total_train=total_tasks, total_eval=total_tasks, started_at=started_at, active=active,
                    waiting_train=[dict(method=m, seed=s) for m,s in waiting_train],
                    waiting_eval=[dict(method=m, seed=s) for m,s in waiting_eval], finished=list(finished.values()))
    try:
        with process_lock(options.output / ".train.scheduler.lock") as owned:
            if not owned:
                raise RuntimeError("This output already has a running training scheduler")
            if (options.output / "stop.request").exists() and not options.resume:
                raise ValueError("Stopped output requires --resume")
            (options.output / "stop.request").unlink(missing_ok=True)
            (options.output / "eval.stop.request").unlink(missing_ok=True)
            last_publish, last_render, height = 0., 0., 0
            while waiting_train or waiting_eval or active_train or active_eval:
                if (options.output / "stop.request").exists():
                    stop.set()
                if not stop.is_set():
                    for slot, gpu in enumerate(slots):
                        if slot not in active_train and waiting_train:
                            method, seed = waiting_train.popleft()
                            child = ctx.Process(target=train_worker, args=(options, method, seed, gpu, stop, results))
                            child.start()
                            active_train[slot] = dict(child=child, method=method, seed=seed, gpu=gpu)
                    for slot in range(options.eval_workers):
                        if slot not in active_eval and waiting_eval:
                            method, seed = waiting_eval.popleft()
                            child = ctx.Process(target=eval_worker, args=(options, method, seed, stop, results))
                            child.start()
                            active_eval[slot] = dict(child=child, method=method, seed=seed)
                try:
                    accept(results.get(timeout=.5))
                except Empty:
                    pass
                for pool, members in (("train", active_train), ("eval", active_eval)):
                    for slot, item in list(members.items()):
                        if item["child"].is_alive():
                            continue
                        item["child"].join()
                        while True:
                            try:
                                accept(results.get_nowait())
                            except Empty:
                                break
                        key = (pool, item["method"], item["seed"])
                        if key not in finished:
                            accept((*key, "failed", f"Worker exited without completion: {item['child'].exitcode}"))
                        del members[slot]
                now = time.monotonic()
                state = snapshot("stopping" if stop.is_set() else "running")
                if now - last_publish >= 3:
                    atomic_json(options.output / "scheduler.train.json", state)
                    last_publish = now
                if now - last_render >= (3 if sys.stdout.isatty() else 30):
                    height = render(progress_lines(options.output, state), height)
                    last_render = now
                if stop.is_set() and not active_train and not active_eval:
                    break
            status = "failed" if any(r["status"] == "failed" for r in finished.values()) else "stopped" if stop.is_set() or any(r["status"] == "stopped" for r in finished.values()) else "complete"
            state = snapshot(status)
            atomic_json(options.output / "scheduler.train.json", state)
            render(progress_lines(options.output, state), height)
            return 1 if status == "failed" else 130 if status == "stopped" else 0
    finally:
        stop.set()
        for item in (*active_train.values(), *active_eval.values()):
            item["child"].join()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "single", "status", "stop"), required=True)
    parser.add_argument("--method", help=f"Explicit comma-separated learning methods: {', '.join(METHODS)}")
    parser.add_argument("--seeds", help="Explicit comma-separated training seeds")
    parser.add_argument("--seed", type=int, help="Alias for one training seed")
    parser.add_argument("--steps", type=int, help="Explicit physical-step budget")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run", default="train")
    parser.add_argument("--devices", default="0,1", help="Two logical indices within inherited CUDA_VISIBLE_DEVICES")
    parser.add_argument("--per-gpu", type=int, default=3, choices=(3,))
    parser.add_argument("--eval-workers", type=positive_integer, default=2)
    parser.add_argument("--eval-device", type=evaluation_device, default="cpu", help="Independent evaluation pool device: cpu or cuda:N")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--confirmation", action="store_true", help="Also evaluate the explicitly requested independent confirmation bank")
    options = parser.parse_args()
    options.output = options.output.resolve()
    if options.stage == "stop":
        from open_score.utils.resources import write_stop
        write_stop(options.output / "stop.request", "Finish the current complete batch/episode and retain recovery state")
        return 0
    if options.stage == "status":
        render(progress_lines(options.output))
        return 0
    if options.seed is not None:
        if options.seeds is not None:
            parser.error("Use either --seed or --seeds")
        options.seeds = str(options.seed)
    if not options.method or options.seeds is None or options.steps is None:
        parser.error("Training requires explicit --method, --seed/--seeds and --steps")
    try:
        selection(options.method, METHODS)
    except ValueError as error:
        parser.error(str(error))
    return run(options)

if __name__ == "__main__":
    raise SystemExit(main())
