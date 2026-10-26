"""Explicit HAD training and resumable two-session diagnostic updates."""
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

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "1")
from open_score.algos import METHODS, canonical_method
from open_score.utils.logging import DEFAULT_OUTPUT
from open_score.utils.experiment import atomic_json, read_json, run_directory, process_lock, logical_devices, physical_device


def selection(text, allowed=None, integer=False):
    values = tuple(int(v.strip()) if integer else canonical_method(v.strip()) if allowed is not None else v.strip()
                   for v in str(text).split(",") if v.strip())
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
                if options.stage == "smoke":
                    status = smoke_method(options, method, seed, stop)
                    results.put(("train", method, seed, status, None))
                    return
                final = qualified_final(options.output, method, seed, options.steps)
                if final is None:
                    from open_score.algos import train, resume_files
                    recover = options.resume and any(path.exists() for path in resume_files(directory))
                    final = train(method, dict(output=str(options.output), seed=seed, t_max=options.steps,
                        run=options.run, resume=recover,
                        use_cuda=True, device="cuda", _stop_event=stop))
                final = qualified_final(options.output, method, seed, options.steps)
                results.put(("train", method, seed, "complete" if final else "stopped", None))
    except BaseException:
        results.put(("train", method, seed, "failed", traceback.format_exc()))


def smoke_proof(output, method, seed, steps):
    """Read real retained optimizer/parameter evidence; never infer a pass from exit."""
    from open_score.algos import _load_resume, setup_runtime
    # Full resume files contain the retained PyMARL ReplayBuffer class, whose
    # pickle ABI is components.episode_buffer. Register the same upstream path
    # as train() before loading it in this otherwise independent proof process.
    setup_runtime()
    directory = run_directory(output, method, seed)
    saved, path = _load_resume(directory, require_complete=True)
    if saved is None:
        return dict(method=method, seed=seed, status="pending", passed=False, sessions=[])
    cfg, progress = saved["config"], saved["progress"]
    if cfg.get("method") != method or int(cfg.get("seed", -1)) != seed or not cfg.get("diagnostic"):
        raise ValueError(f"Diagnostic identity differs at {path}")
    if int(cfg.get("t_max", -1)) != steps:
        raise ValueError(f"Diagnostic budget differs at {path}; retained recovery is preserved")
    sessions = progress.get("diagnostic_sessions", [])
    parameter_groups = ("main", "allocation_actor", "allocation_critic") if method == "alma" else ("main",)
    optimizers = ("optimizer", "alloc_pi_optimizer", "alloc_q_optimizer") if method == "alma" else ("optimizer",)
    problems = []
    for index, row in enumerate(sessions[:2], start=1):
        before, after = row.get("optimizer_steps_before", {}), row.get("optimizer_steps_after", {})
        if not (int(row.get("session_updates", 0)) >= 1 and
                int(row.get("final_updates", 0)) > int(row.get("initial_updates", 0)) and
                int(row.get("final_t_env", 0)) > int(row.get("initial_t_env", 0))):
            problems.append(f"session_{index}: physical steps and actual updates must advance")
        if row.get("finite_training") is not True:
            problems.append(f"session_{index}: losses/gradients were not all finite")
        if any(not math.isfinite(float(row.get("parameter_changes", {}).get(group, 0))) or
               float(row.get("parameter_changes", {}).get(group, 0)) <= 0 for group in parameter_groups):
            problems.append(f"session_{index}: required parameters did not change")
        if any(int(after.get(key, 0)) <= int(before.get(key, 0)) for key in optimizers):
            problems.append(f"session_{index}: required optimizer steps did not advance")
    if len(sessions) >= 2:
        first, second = sessions[:2]
        if not (second.get("restored") is True and int(second.get("initial_t_env", 0)) > 0 and
                int(second.get("initial_updates", 0)) > 0 and
                int(second["initial_t_env"]) == int(first["final_t_env"]) and
                int(second["initial_updates"]) == int(first["final_updates"])):
            problems.append("session_2: restored training clocks do not match session_1")
        if any(int(second.get("optimizer_steps_before", {}).get(key, 0)) !=
               int(first.get("optimizer_steps_after", {}).get(key, 0)) for key in optimizers):
            problems.append("session_2: restored optimizer steps do not match session_1")
    if any(key not in saved.get("networks", {}) for key in optimizers):
        problems.append("retained checkpoint lacks required optimizer states")
    passed = len(sessions) >= 2 and not problems
    return dict(method=method, public_method="qmix" if method == "b2_qmix_atten" else method,
                seed=seed, status="passed" if passed else "failed" if problems else "pending", passed=passed,
                checkpoint=str(path), t_env=int(progress.get("t_env", 0)),
                updates=int(progress.get("updates", 0)), sessions=sessions, problems=problems)


def smoke_session(options, method, seed, resume, stop, result):
    """Each call runs in its own process; phase two reconstructs the full runtime."""
    directory = run_directory(options.output, method, seed)
    try:
        with (directory / "console.log").open("a", encoding="utf-8", buffering=1) as stream, redirect_stdout(stream), redirect_stderr(stream):
            from open_score.algos import train
            train(method, dict(output=str(options.output), seed=seed, t_max=options.steps,
                run=options.run, resume=resume, diagnostic=True, env_args=dict(train_dist="smoke_4v4"),
                batch_size_run=2, batch_size=2, buffer_size=16, training_iters=1,
                use_cuda=True, device="cuda", skip_final_eval=True,
                _session_update_limit=1, _stop_event=stop))
        result.put(None)
    except BaseException:
        result.put(traceback.format_exc())


def smoke_method(options, method, seed, stop):
    directory = run_directory(options.output, method, seed)
    from open_score.algos import resume_files
    exists = any(path.exists() for path in resume_files(directory))
    if exists and not options.resume:
        raise FileExistsError(f"Existing diagnostic run at {directory}; use --resume")
    proof = smoke_proof(options.output, method, seed, options.steps)
    if proof["passed"]:
        return "complete"
    if proof.get("problems"):
        raise RuntimeError("Retained diagnostic evidence failed: " + "; ".join(proof["problems"]))
    ctx = get_context("spawn")
    while len(proof["sessions"]) < 2:
        if stop.is_set():
            return "stopped"
        result = ctx.Queue()
        child = ctx.Process(target=smoke_session, args=(options, method, seed, exists, stop, result))
        child.start()
        child.join()
        try:
            error = result.get(timeout=2)
        except Empty:
            error = f"Diagnostic trainer exited without evidence: {child.exitcode}"
        if error:
            raise RuntimeError(error)
        before = len(proof["sessions"])
        proof = smoke_proof(options.output, method, seed, options.steps)
        exists = True
        if proof.get("problems"):
            raise RuntimeError("Diagnostic evidence failed: " + "; ".join(proof["problems"]))
        if stop.is_set():
            return "stopped"
        if len(proof["sessions"]) <= before:
            raise RuntimeError("Diagnostic trainer saved no completed-session evidence")
    return "complete" if proof["passed"] else "failed"


def publish_smoke_summary(options, methods, seeds, status):
    path = options.output / "summary.json"
    summary = read_json(path, {})
    cases = []
    for seed in seeds:
        for method in methods:
            try:
                cases.append(smoke_proof(options.output, method, seed, options.steps))
            except Exception as error:
                cases.append(dict(method=method, seed=seed, status="failed", passed=False, error=str(error)))
    summary["training"] = dict(status=status, passed=all(row["passed"] for row in cases),
        total=len(cases), completed=sum(row["passed"] for row in cases), cases=cases,
        diagnostic=True, steps_ceiling=options.steps, sessions_per_method=2,
        train_config=dict(N_R=4, N_B=4, K=2, batch_size_run=2, batch_size=2,
                          buffer_size=16, training_iters=1), updated_at=time.time())
    atomic_json(path, summary)


def eval_worker(options, method, seed, stop, results):
    directory = run_directory(options.output, method, seed)
    try:
        from open_score.utils.eval.protocol import evaluate_checkpoint
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
    lines.append(f"train {done_train}/{state.get('total_train', 0)} ({state.get('train_slots', 1)} slots)  eval {done_eval}/{state.get('total_eval', 0)} ({state.get('eval_slots', 0)} slots)  waiting train={len(state.get('waiting_train', []))} eval={len(state.get('waiting_eval', []))}  elapsed={duration_text(time.time() - state['started_at']) if state.get('started_at') else '-'} ETA={duration_text(eta)}  {state.get('status', 'idle')}")
    return lines


def render(lines, previous=0):
    if previous and sys.stdout.isatty():
        sys.stdout.write(f"\x1b[{previous}F\x1b[J")
    print("\n".join(lines), flush=True)
    return len(lines)


def run(options):
    from open_score.utils.experiment import check_environment, ensure_layout
    check_environment()
    ensure_layout(options.output)
    methods = selection(options.method, METHODS)
    seeds = selection(options.seeds, integer=True)
    devices = logical_devices(options.devices)
    if options.steps < (1 if options.stage == "smoke" else 50):
        raise ValueError("Select a positive diagnostic ceiling, or a formal budget >=50")
    if options.stage == "single" and (len(methods) != 1 or len(seeds) != 1):
        raise ValueError("--stage single requires exactly one method and one seed")
    if options.stage == "smoke" and (len(devices) != 1 or options.per_gpu > 2):
        raise ValueError("Diagnostic updates use one GPU and at most two simultaneous GPU jobs")
    ctx = get_context("spawn")
    stop, results = ctx.Event(), ctx.Queue()
    waiting_train = deque((m, s) for s in seeds for m in methods)
    waiting_eval, active_train, active_eval = deque(), {}, {}
    finished, enqueued = {}, set()
    total_tasks, started_at = len(waiting_train), time.time()
    slots = tuple(physical_device(gpu) for _ in range(options.per_gpu) for gpu in devices)
    if options.stage == "single":
        slots = slots[:1]
    evaluate_after = bool(options.eval_after_train and options.stage != "smoke")
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in handlers:
        signal.signal(sig, lambda *_: stop.set())
    def accept(message):
        pool, method, seed, status, error = message
        finished[(pool, method, seed)] = dict(pool=pool, method=method, seed=seed, status=status, error=error)
        if error:
            print(error, file=sys.stderr, flush=True)
        if pool == "train" and status == "complete" and evaluate_after and (method, seed) not in enqueued:
            waiting_eval.append((method, seed))
            enqueued.add((method, seed))
    def snapshot(status):
        active = [dict(pool=pool, method=i["method"], seed=i["seed"], gpu=i.get("gpu", options.eval_device), pid=i["child"].pid)
                  for pool, members in (("train", active_train), ("eval", active_eval)) for i in members.values()]
        return dict(pid=os.getpid(), stage=options.stage, updated_at=time.time(), status=status, train_slots=len(slots),
                    eval_slots=options.eval_workers if evaluate_after else 0, eval_device=options.eval_device, total_train=total_tasks, total_eval=total_tasks if evaluate_after else 0, started_at=started_at, active=active,
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
                    if options.stage == "smoke":
                        publish_smoke_summary(options, methods, seeds, state["status"])
                    last_publish = now
                if now - last_render >= (3 if sys.stdout.isatty() else 30):
                    height = render(progress_lines(options.output, state), height)
                    last_render = now
                if stop.is_set() and not active_train and not active_eval:
                    break
            status = "failed" if any(r["status"] == "failed" for r in finished.values()) else "stopped" if stop.is_set() or any(r["status"] == "stopped" for r in finished.values()) else "complete"
            state = snapshot(status)
            atomic_json(options.output / "scheduler.train.json", state)
            if options.stage == "smoke":
                publish_smoke_summary(options, methods, seeds, status)
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
    parser.add_argument("--stage", choices=("train", "single", "smoke", "status", "stop"), required=True)
    parser.add_argument("--method", help=f"Explicit comma-separated learning methods: {', '.join(METHODS)}")
    parser.add_argument("--seeds", help="Explicit comma-separated training seeds")
    parser.add_argument("--seed", type=int, help="Alias for one training seed")
    parser.add_argument("--steps", type=int, help="Explicit physical-step budget")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run", default="train")
    parser.add_argument("--devices", default="0", help="Logical GPU indices within inherited CUDA_VISIBLE_DEVICES")
    parser.add_argument("--per-gpu", type=positive_integer, help="Jobs per GPU; default 1 (smoke: 2)")
    parser.add_argument("--eval-workers", type=positive_integer, default=2)
    parser.add_argument("--eval-device", type=evaluation_device, default="cpu", help="Independent evaluation pool device: cpu or cuda:N")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-after-train", action="store_true", help="Explicitly opt in to the formal final-evaluation quota")
    parser.add_argument("--confirmation", action="store_true", help="Also evaluate the explicitly requested independent confirmation bank")
    options = parser.parse_args()
    options.output = (options.output or (DEFAULT_OUTPUT / "smoke" if options.stage == "smoke" else DEFAULT_OUTPUT)).resolve()
    options.per_gpu = options.per_gpu or (2 if options.stage == "smoke" else 1)
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
    if options.stage == "smoke":
        options.method = options.method or ",".join(METHODS)
        options.seeds = options.seeds if options.seeds is not None else "0"
        options.steps = options.steps if options.steps is not None else 4096
        options.run = "smoke"
        if options.eval_after_train or options.confirmation:
            parser.error("Smoke diagnostics do not run formal evaluation or confirmation")
        if options.seeds != "0":
            parser.error("The approved smoke diagnostic uses training seed 0 only")
    if not options.method or options.seeds is None or options.steps is None:
        parser.error("Training requires explicit --method, --seed/--seeds and --steps")
    try:
        selection(options.method, METHODS)
    except ValueError as error:
        parser.error(str(error))
    return run(options)

if __name__ == "__main__":
    raise SystemExit(main())
