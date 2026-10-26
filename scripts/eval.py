"""Explicit frozen HAD evaluation in an independent CPU or GPU process pool."""
from __future__ import annotations
import argparse
from contextlib import redirect_stdout, redirect_stderr
from collections import deque
from multiprocessing import get_context
from pathlib import Path
from queue import Empty
import os
import signal
import sys
import time
import traceback

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "1")
from open_score.algos import METHODS
from open_score.utils.logging import DEFAULT_OUTPUT
from open_score.utils.experiment import atomic_json, read_json, run_directory, process_lock
from train import selection, render, positive_integer, evaluation_device, duration_text, damage_text, global_eta


def reproduce(options):
    """Keep every diagnostic log/status file in the new output, never the archive."""
    from open_score.utils.eval.protocol import run_reproduction
    options.output.mkdir(parents=True, exist_ok=True)
    stop = False
    def request_stop(*_):
        nonlocal stop
        stop = True
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in handlers:
        signal.signal(sig, request_stop)
    def stopped():
        return stop or (options.output / "eval.stop.request").exists()
    (options.output / "eval.stop.request").unlink(missing_ok=True)
    try:
        with (options.output / "reproduction.console.log").open("a", encoding="utf-8", buffering=1) as stream, redirect_stdout(stream), redirect_stderr(stream):
            result = run_reproduction(options.reference_output, options.output,
                methods=selection(options.method, METHODS) if options.method else None,
                device=options.device, stop_requested=stopped)
        print(f"reproduce {result['status']}: {result['completed']}/{result['total']}; "
              f"exact D+ep_len {result['exact_both_matches']}/{result['completed']}; "
              f"see {options.output / 'reproduction.csv'}", flush=True)
        return 0 if result["status"] == "complete" else 130
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def worker(options, task, stop, results):
    method, seed = task["method"], task["seed"]
    def stopped():
        return stop.is_set() or (options.output / "eval.stop.request").exists()
    try:
        if options.stage == "anchors":
            from open_score.utils.eval.anchors import run_anchors
            def publish(row):
                atomic_json(options.output / "evaluation.json", dict(row, elapsed_seconds=time.monotonic() - began))
            began = time.monotonic()
            result = run_anchors(options.output, run=options.run, seed=0, stop_requested=stopped, on_progress=publish)
        else:
            from open_score.utils.eval.protocol import evaluate_checkpoint, evaluate_depth_sweep
            path = Path(task["checkpoint"])
            with (path.parent / "eval.console.log").open("a", encoding="utf-8", buffering=1) as stream, redirect_stdout(stream), redirect_stderr(stream):
                if options.stage == "depth":
                    result = evaluate_depth_sweep(method, path, output=options.output, run=options.run,
                        seed=seed, device=options.device, stop_requested=stopped, depths=selection(options.depths, integer=True))
                else:
                    result = evaluate_checkpoint(method, path, output=options.output, run=options.run,
                        seed=seed, device=options.device, stop_requested=stopped, confirmation=options.stage == "confirmation")
        results.put((method, seed, result["status"], None))
    except BaseException:
        results.put((method, seed, "failed", traceback.format_exc()))


def progress_lines(output, state=None):
    state = read_json(Path(output) / "scheduler.eval.json", {}) if state is None else state
    lines = ["method                  seed device progress        elapsed     ETA         D       phase"]
    measurements = []
    for task in state.get("active", []):
        path = Path(task["checkpoint"]).parent if task.get("checkpoint") else Path(output)
        row = read_json(path / "evaluation.json", {})
        measurements.append(row)
        seed = row.get("seed", task["seed"])
        lines.append(f"{task['method']:<23} {str(seed if seed is not None else '?'):>4} {state.get('device', 'cpu'):>6} {row.get('completed', 0):>7}/{row.get('total', 0):<7} {duration_text(row.get('elapsed_seconds')):<11} {duration_text(row.get('remaining_seconds')):<11} {damage_text(row.get('D')):<7} {row.get('phase', state.get('stage', '-'))}")
    for task in state.get("finished", []):
        if task["status"] == "failed":
            lines.append(f"failed: {task['method']}/s{task['seed']}; see its console log")
    completed = sum(t["status"] == "complete" for t in state.get("finished", []))
    eta = None if state.get("status", "idle") in ("failed", "stopped", "idle") else global_eta(measurements, len(state.get("waiting", [])))
    training = read_json(Path(output) / "scheduler.train.json", {})
    trained = sum(t.get("pool") == "train" and t.get("status") == "complete" for t in training.get("finished", []))
    train_count = f"{trained}/{training['total_train']}" if "total_train" in training else "-/-"
    lines.append(f"train {train_count}  eval {completed}/{state.get('total_tasks', 0)} ({state.get('workers', 2)} slots)  waiting={len(state.get('waiting', []))}  elapsed={duration_text(time.time() - state['started_at']) if state.get('started_at') else '-'} ETA={duration_text(eta)}  {state.get('status', 'idle')}")
    return lines


def run(options, tasks):
    from open_score.utils.experiment import check_environment, ensure_layout
    check_environment()
    ensure_layout(options.output)
    ctx = get_context("spawn")
    stop, results = ctx.Event(), ctx.Queue()
    waiting, active, finished = deque(tasks), {}, {}
    total_tasks, started_at = len(tasks), time.time()
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in handlers:
        signal.signal(sig, lambda *_: stop.set())
    def accept(message):
        method, seed, status, error = message
        finished[(method, seed)] = dict(method=method, seed=seed, status=status, error=error)
        if error:
            print(error, file=sys.stderr, flush=True)
    def snapshot(status):
        return dict(pid=os.getpid(), stage=options.stage, status=status, updated_at=time.time(),
                    workers=options.workers, device=options.device, total_tasks=total_tasks, started_at=started_at, active=[dict(i["task"], pid=i["child"].pid) for i in active.values()],
                    waiting=list(waiting), finished=list(finished.values()))
    try:
        with process_lock(options.output / ".eval.scheduler.lock") as owned:
            if not owned:
                raise RuntimeError("This output already has a standalone evaluation scheduler")
            (options.output / "eval.stop.request").unlink(missing_ok=True)
            last_publish, last_render, height = 0., 0., 0
            while waiting or active:
                if (options.output / "eval.stop.request").exists():
                    stop.set()
                if not stop.is_set():
                    for slot in range(options.workers):
                        if slot not in active and waiting:
                            task = waiting.popleft()
                            child = ctx.Process(target=worker, args=(options, task, stop, results))
                            child.start()
                            active[slot] = dict(child=child, task=task)
                try:
                    accept(results.get(timeout=.5))
                except Empty:
                    pass
                for slot, item in list(active.items()):
                    if item["child"].is_alive():
                        continue
                    item["child"].join()
                    while True:
                        try:
                            accept(results.get_nowait())
                        except Empty:
                            break
                    key = (item["task"]["method"], item["task"]["seed"])
                    if key not in finished:
                        accept((*key, "failed", f"Worker exited without completion: {item['child'].exitcode}"))
                    del active[slot]
                now = time.monotonic()
                state = snapshot("stopping" if stop.is_set() else "running")
                if now - last_publish >= 3:
                    atomic_json(options.output / "scheduler.eval.json", state)
                    last_publish = now
                if now - last_render >= (3 if sys.stdout.isatty() else 30):
                    height = render(progress_lines(options.output, state), height)
                    last_render = now
                if stop.is_set() and not active:
                    break
            status = "failed" if any(r["status"] == "failed" for r in finished.values()) else "stopped" if stop.is_set() or any(r["status"] == "stopped" for r in finished.values()) else "complete"
            state = snapshot(status)
            atomic_json(options.output / "scheduler.eval.json", state)
            render(progress_lines(options.output, state), height)
            return 1 if status == "failed" else 130 if status == "stopped" else 0
    finally:
        stop.set()
        for item in active.values():
            item["child"].join()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("final", "eval", "confirmation", "depth", "anchors", "reproduce", "status", "stop"), default="final")
    parser.add_argument("--method", help=f"Explicit comma-separated learning methods: {', '.join(METHODS)}")
    parser.add_argument("--seeds", help="Explicit comma-separated training seeds")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference-output", type=Path, help="Read-only canonical output archive for the fixed 60-case reproduction")
    parser.add_argument("--run", default="train")
    parser.add_argument("--workers", type=positive_integer, default=2)
    parser.add_argument("--device", type=evaluation_device, default="cpu", help="cpu or cuda:N within inherited CUDA_VISIBLE_DEVICES")
    parser.add_argument("--depths", default="1,2,3,4,5,6", help="Explicit depth subset for --stage depth")
    parser.add_argument("--resume", action="store_true", help="Committed matching episodes are always preserved and reused")
    options = parser.parse_args()
    options.output = (options.output or (DEFAULT_OUTPUT / "smoke" if options.stage == "reproduce" else DEFAULT_OUTPUT)).resolve()
    if options.stage == "eval":
        options.stage = "final"
    if options.stage == "stop":
        from open_score.utils.resources import write_stop
        write_stop(options.output / "eval.stop.request", "Stop after the current complete episode")
        return 0
    if options.stage == "status":
        render(progress_lines(options.output))
        return 0
    if options.stage == "reproduce":
        if options.reference_output is None:
            parser.error("--stage reproduce requires --reference-output")
        if options.checkpoint or options.seed is not None or options.seeds is not None:
            parser.error("Reproduction uses current-comparison final checkpoints with training seed 0")
        options.reference_output = options.reference_output.resolve()
        if options.output.is_relative_to(options.reference_output):
            parser.error("Reproduction output must be outside the retained reference archive")
        if options.method and set(selection(options.method, METHODS)) != set(METHODS):
            parser.error("Reproduction uses the approved eight learning methods plus Rule and Random (60 cases)")
        return reproduce(options)
    if options.stage == "anchors":
        options.workers = 1
        return run(options, [dict(method="anchors", seed=0)])
    if not options.method:
        parser.error("Frozen evaluation requires explicit --method")
    try:
        methods = selection(options.method, METHODS)
    except ValueError as error:
        parser.error(str(error))
    if options.stage == "depth" and methods != ("regir_nomem",):
        parser.error("Frozen depth evaluation supports LEAF (regir_nomem) only")
    if options.checkpoint:
        if len(methods) != 1 or options.seeds is not None:
            parser.error("--checkpoint requires one method and optional --seed")
        tasks = [dict(method=methods[0], seed=options.seed, checkpoint=str(options.checkpoint.resolve()))]
    else:
        if options.seed is not None:
            if options.seeds is not None:
                parser.error("Use either --seed or --seeds")
            options.seeds = str(options.seed)
        if options.seeds is None:
            parser.error("Select explicit --seed/--seeds or --checkpoint")
        tasks = [dict(method=m, seed=s, checkpoint=str(run_directory(options.output, m, s) / "final.pt"))
                 for s in selection(options.seeds, integer=True) for m in methods]
    return run(options, tasks)

if __name__ == "__main__":
    raise SystemExit(main())
