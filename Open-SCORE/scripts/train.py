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
import shlex
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

FRIENDLY_PRESET = "sky_friendly_v1"
FRIENDLY_ARMS = ("A", "B", "C")
FRIENDLY_OUTPUT = PROJECT / "outputs/main_Sky_friendly"


class Tee:
    """Keep an interactive terminal and the existing per-run console log."""
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, value):
        written = self.terminal.write(value)
        self.log.write(value)
        self.log.flush()
        return written

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return self.terminal.isatty()

    def __getattr__(self, key):
        return getattr(self.terminal, key)


def environment_options(options):
    return {key: value for key in ("env_agent_type", "env_agent_action_type", "spatial_dim", "task_mode")
            if (value := getattr(options, key, None)) is not None}


def selection(text, allowed=None, integer=False):
    values = tuple(int(v.strip()) if integer else v.strip() for v in str(text).split(",") if v.strip())
    if not values or len(set(values)) != len(values):
        raise ValueError("Selection must be nonempty and contain no duplicates")
    if integer and min(values) < 0:
        raise ValueError("Seeds must be non-negative")
    if allowed is not None and any(v not in allowed for v in values):
        raise ValueError(f"Unknown method; choose from {', '.join(allowed)}")
    return values


def qualified_final(output, method, seed, steps, experiment_arm=None):
    path = run_directory(output, method, seed, experiment_arm=experiment_arm) / "final.pt"
    if not path.exists():
        return None
    import torch
    saved = torch.load(path, map_location="cpu", weights_only=False)
    cfg, progress = saved.get("config", {}), saved.get("progress", {})
    if cfg.get("method") != method or cfg.get("env") != "had" or int(cfg.get("seed", -1)) != seed:
        raise ValueError(f"Model identity differs at {path}")
    if int(cfg.get("t_max", -1)) != steps:
        raise ValueError(f"Existing model has another budget at {path}; select an independent output")
    if experiment_arm is not None:
        if cfg.get("experiment_arm") != experiment_arm or cfg.get("preset") != FRIENDLY_PRESET:
            raise ValueError(f"Model arm/preset differs at {path}")
        if progress.get("status") != "completed" or int(progress.get("next_eval", 0)) != 51:
            return None
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


def friendly_rows(output, state=None, experiment_arm=None):
    """Read persisted arm states; a missing run is never reported as complete."""
    output = Path(output)
    metadata = read_json(output / "experiment.json", {})
    state = read_json(output / "scheduler.train.json", {}) if state is None else state
    planned = {item["experiment_arm"]: item for item in state.get("planned", [])}
    active = {item["experiment_arm"]: item for item in state.get("active", []) if item.get("experiment_arm")}
    queued = {item["experiment_arm"] for item in state.get("waiting_train", []) if item.get("experiment_arm")}
    finished = {item["experiment_arm"]: item for item in state.get("finished", []) if item.get("experiment_arm")}
    rows = []
    for arm in FRIENDLY_ARMS:
        if experiment_arm not in (None, "all", arm):
            continue
        retained = metadata.get("arms", {}).get(arm, {})
        config = retained.get("config", planned.get(arm, {}).get("config", {}))
        method, seed = config.get("method", "regir_nomem"), int(config.get("seed", 0))
        directory = run_directory(output, method, seed, experiment_arm=arm)
        row = read_json(directory / "resource.json", {})
        status = row.get("status", "not_started")
        if arm in active:
            status = "running"
        elif arm in queued:
            status = "queued" if state.get("status") in ("running", "stopping") else "pending"
        elif arm in finished:
            status = finished[arm]["status"]
        row.update(experiment_arm=arm, method=method, seed=seed, status=status,
                   budget_steps=int(config.get("t_max", row.get("budget_steps", 1000000))),
                   gpu=active.get(arm, {}).get("gpu", row.get("physical_gpu")),
                   directory=str(directory))
        rows.append(row)
    return rows


def progress_lines(output, state=None, experiment_arm=None):
    state = read_json(Path(output) / "scheduler.train.json", {}) if state is None else state
    metadata = read_json(Path(output) / "experiment.json", {})
    if state.get("preset") == FRIENDLY_PRESET or metadata.get("preset") == FRIENDLY_PRESET:
        lines = ["arm seed GPU physical steps    safe step  validation    D       loss     stage status / phase"]
        for row in friendly_rows(output, state, experiment_arm):
            stage = row.get("curriculum_stage", -1)
            lines.append(f"{row['experiment_arm']:<3} {row['seed']:>4} {str(row.get('gpu', '-')):>3} "
                f"{row.get('observed_t_env', row.get('t_env', 0)):>7}/{row['budget_steps']:<7} "
                f"{row.get('safe_checkpoint_t_env', 0):>9}  {row.get('validation_completed_episodes', 0):>4}/5000  "
                f"{damage_text(row.get('latest_validation_D')):<7} {damage_text(row.get('loss')):<8} "
                f"{stage!s:>5} {row['status']} / {row.get('current_phase', '-')}")
        lines.append(f"{state.get('status', 'saved states')} | updated_at fields in each arm resource.json describe the last recorded heartbeat")
        return lines
    lines = ["pool   method                  seed device progress        elapsed     ETA         D       status"]
    measurements = []
    for pool in ("train", "eval"):
        for item in state.get("active", []):
            if item["pool"] != pool:
                continue
            path = run_directory(output, item["method"], item["seed"])
            row = read_json(path / ("resource.json" if pool == "train" else "evaluation.json"), {})
            measurements.append(row)
            done, total = (row.get("observed_t_env", row.get("t_env", 0)), row.get("budget_steps", 0)) if pool == "train" else (row.get("completed", 0), row.get("total", 0))
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


def friendly_worker(options, job, gpu, stop, results):
    """One locked arm, with the complete resolved configuration and no final evaluation."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    arm, cfg = job["experiment_arm"], dict(job["config"])
    method, seed = cfg["method"], int(cfg["seed"])
    directory = run_directory(options.output, method, seed, experiment_arm=arm)
    owned = False
    try:
        with process_lock(directory / ".train.lock") as owned:
            if not owned:
                raise RuntimeError(f"Arm already has a trainer: {arm}/{method}/s{seed}")
            with (directory / "console.log").open("a", encoding="utf-8", buffering=1) as stream, redirect_stdout(stream), redirect_stderr(stream):
                from open_score.algos import train
                cfg.update(progress="plain", _external_reporting=True, _stop_event=stop)
                final = train(method, cfg)
                complete = Path(final).name == "final.pt"
                results.put(dict(experiment_arm=arm, method=method, seed=seed,
                                 status="complete" if complete else "stopped", error=None))
    except BaseException:
        error = traceback.format_exc()
        if owned:
            with (directory / "console.log").open("a", encoding="utf-8") as stream:
                stream.write(error)
        results.put(dict(experiment_arm=arm, method=method, seed=seed, status="failed", error=error))


def friendly(options):
    """Schedule the selected approved arms using explicit slots per visible GPU."""
    from open_score.algos import resolve_friendly_config, resume_files, train
    arms = FRIENDLY_ARMS if options.arm == "all" else (options.arm,)
    if options.stage == "single" and len(arms) != 1:
        raise ValueError("single requires --arm A, B or C")
    devices = logical_devices(options.devices or ("0" if options.stage == "single" else "0,1"))
    if options.per_gpu != 1 or len(devices) != (1 if options.stage == "single" else 2):
        raise ValueError("Use --per-gpu 1; single selects one GPU and train selects two. Use --gpu-slots for asymmetric slots")
    if options.gpu_slots is None:
        slot_counts = (1,) * len(devices)
    else:
        try:
            slot_counts = tuple(int(value.strip()) for value in options.gpu_slots.split(","))
        except ValueError as error:
            raise ValueError("--gpu-slots requires comma-separated positive integers, e.g. 2,1") from error
        if len(slot_counts) != len(devices) or any(value < 1 for value in slot_counts):
            raise ValueError("--gpu-slots requires one positive count for each --devices entry")
    if options.stage == "single" and slot_counts != (1,):
        raise ValueError("single has one arm and requires one GPU slot")
    if options.method != "regir_nomem" or selection(options.seeds, integer=True) != (0,) or options.steps != 1000000:
        raise ValueError("sky_friendly_v1 selects LEAF, seed 0 and 1,000,000 physical steps per arm")
    if options.confirmation:
        raise ValueError("The friendly training preset does not automatically run formal evaluation")
    slots = tuple(physical_device(device) for device, count in zip(devices, slot_counts)
                  for _ in range(count))
    started_at = time.time()
    ctx = get_context("spawn")
    stop, results = ctx.Event(), ctx.Queue()
    waiting, active, finished, planned = deque(), {}, {}, []
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    bars = {}

    def snapshot(status):
        return dict(preset=FRIENDLY_PRESET, pid=os.getpid(), updated_at=time.time(), status=status,
            train_slots=len(slots), eval_slots=0, total_train=len(arms), total_eval=0, started_at=started_at,
            planned=planned, active=[dict(pool="train", experiment_arm=item["job"]["experiment_arm"],
                method="regir_nomem", seed=0, gpu=item["gpu"], pid=item["child"].pid) for item in active.values()],
            waiting_train=[dict(experiment_arm=job["experiment_arm"], method="regir_nomem", seed=0) for job in waiting],
            waiting_eval=[], finished=[dict(pool="train", **item) for item in finished.values()])

    def accept(message):
        finished[message["experiment_arm"]] = message
        if message.get("error"):
            if bars:
                from tqdm import tqdm
                tqdm.write(message["error"], file=sys.stderr)
            else:
                print(message["error"], file=sys.stderr, flush=True)

    def refresh(state, force=False):
        nonlocal last_report, report_signature, height, last_render
        rows = friendly_rows(options.output, state)
        if bars:
            for row in rows:
                bar = bars.get(row["experiment_arm"])
                if bar is None:
                    continue
                bar.n = min(row["budget_steps"], int(row.get("observed_t_env", row.get("t_env", 0))))
                bar.set_postfix(status=row["status"], phase=row.get("current_phase", "-"),
                    val=f"{row.get('validation_completed_episodes', 0)}/5000",
                    ev=f"{row.get('eval_completed', 0)}/{row.get('eval_total', 0)}",
                    sps=damage_text(row.get("training_steps_per_second")), ETA=duration_text(row.get("remaining_seconds")),
                    D=damage_text(row.get("latest_validation_D")), loss=damage_text(row.get("loss")),
                    stage=row.get("curriculum_stage", -1), safe=row.get("safe_checkpoint_t_env", 0), refresh=False)
                bar.refresh()
        elif force or time.monotonic() - last_render >= (3 if sys.stdout.isatty() else 30):
            height = render(progress_lines(options.output, state, options.arm), height)
            last_render = time.monotonic()
        signature = tuple((row["experiment_arm"], row["status"], row.get("next_eval", 1),
                           row.get("curriculum_stage", -1)) for row in rows)
        if options.live_curves and (force or signature != report_signature or time.monotonic() - last_report >= 30):
            try:
                from open_score.eval.report import render_training
                render_training(options.output)
            except Exception as error:
                print(f"Curve/report refresh failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
            last_report, report_signature = time.monotonic(), signature

    last_report = last_render = float("-inf")
    report_signature, height = None, 0
    with process_lock(options.output / ".train.scheduler.lock") as owned:
        if not owned:
            raise RuntimeError("This output already has a running training entry point")
        if (options.output / "stop.request").exists() and not options.resume:
            raise ValueError("Stopped output requires --resume")
        for arm in arms:
            overrides = dict(output=str(options.output), method="regir_nomem", seed=0, t_max=options.steps,
                run=options.run, use_cuda=True, progress=options.progress, live_curves=options.live_curves,
                launch_command=options.launch_command, launch_argv=options.launch_argv,
                launch_platform=os.name, python_executable=sys.executable)
            if environment_options(options):
                overrides["env_args"] = environment_options(options)
            cfg = resolve_friendly_config(arm, overrides)
            directory = run_directory(options.output, "regir_nomem", 0, experiment_arm=arm)
            with process_lock(directory / ".train.lock") as claimed:
                if not claimed:
                    raise RuntimeError(f"Arm already has a trainer: {arm}")
                complete = qualified_final(options.output, "regir_nomem", 0, options.steps, arm) if options.resume else None
                recovery = any(path.exists() for path in resume_files(directory))
                retained_files = any((directory / name).exists() for name in ("config.json", "final.pt", "best.pt")) or recovery
                if retained_files and not options.resume:
                    raise FileExistsError(f"Existing arm {arm}; use --resume to retain its progress")
                if options.resume and complete is None and retained_files and not recovery:
                    raise FileNotFoundError(f"Arm {arm} has existing data but no recoverable full checkpoint")
                if options.resume:
                    (directory / "pause.request").unlink(missing_ok=True)
                cfg["resume"] = bool(options.resume and recovery)
            job = dict(experiment_arm=arm, config=cfg)
            planned.append(job)
            if complete is not None:
                finished[arm] = dict(experiment_arm=arm, method="regir_nomem", seed=0, status="complete", error=None)
            else:
                waiting.append(job)
        if options.resume:
            (options.output / "stop.request").unlink(missing_ok=True)
        for sig in handlers:
            signal.signal(sig, lambda *_: stop.set())
        try:
            state = snapshot("running")
            atomic_json(options.output / "scheduler.train.json", state)
            if options.stage == "single":
                job = waiting.popleft() if waiting else None
                if job is not None:
                    arm, cfg = job["experiment_arm"], dict(job["config"])
                    directory = run_directory(options.output, "regir_nomem", 0, experiment_arm=arm)
                    with process_lock(directory / ".train.lock") as claimed:
                        if not claimed:
                            raise RuntimeError(f"Arm already has a trainer: {arm}")
                        if (directory / "pause.request").exists() or (options.output / "stop.request").exists() or stop.is_set():
                            finished[arm] = dict(experiment_arm=arm, method="regir_nomem", seed=0,
                                                 status="stopped_before_start", error=None)
                        else:
                            with (directory / "console.log").open("a", encoding="utf-8", buffering=1) as stream:
                                os.environ["CUDA_VISIBLE_DEVICES"] = str(slots[0])
                                cfg["_stop_event"] = stop
                                state = snapshot("running")
                                state["active"] = [dict(pool="train", experiment_arm=arm, method="regir_nomem",
                                                        seed=0, gpu=slots[0], pid=os.getpid())]
                                atomic_json(options.output / "scheduler.train.json", state)
                                with redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(Tee(sys.stderr, stream)):
                                    path = train("regir_nomem", cfg)
                                finished[arm] = dict(experiment_arm=arm, method="regir_nomem", seed=0,
                                    status="complete" if Path(path).name == "final.pt" else "stopped", error=None)
            else:
                if options.progress == "tqdm":
                    from tqdm import tqdm
                    for position, arm in enumerate(arms):
                        bars[arm] = tqdm(total=options.steps, desc=f"Arm {arm} seed 0", unit="step",
                            position=position, dynamic_ncols=True, mininterval=1, file=sys.stderr,
                            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}, {postfix}]")
                last_publish = last_refresh = 0.
                while waiting or active:
                    if (options.output / "stop.request").exists():
                        stop.set()
                    if not stop.is_set():
                        for slot, gpu in enumerate(slots):
                            if slot in active:
                                continue
                            while waiting:
                                job = waiting.popleft()
                                directory = run_directory(options.output, "regir_nomem", 0, experiment_arm=job["experiment_arm"])
                                if (directory / "pause.request").exists():
                                    accept(dict(experiment_arm=job["experiment_arm"], method="regir_nomem", seed=0,
                                                status="stopped_before_start", error=None))
                                    continue
                                child = ctx.Process(target=friendly_worker, args=(options, job, gpu, stop, results))
                                child.start()
                                active[slot] = dict(child=child, job=job, gpu=gpu)
                                break
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
                        arm = item["job"]["experiment_arm"]
                        if arm not in finished:
                            accept(dict(experiment_arm=arm, method="regir_nomem", seed=0, status="failed",
                                        error=f"Worker exited without completion: {item['child'].exitcode}"))
                        del active[slot]
                    now = time.monotonic()
                    state = snapshot("stopping" if stop.is_set() else "running")
                    if now - last_publish >= 3:
                        atomic_json(options.output / "scheduler.train.json", state)
                        last_publish = now
                    if now - last_refresh >= 1:
                        refresh(state)
                        last_refresh = now
                    if stop.is_set() and not active:
                        break
            status = "failed" if any(item["status"] == "failed" for item in finished.values()) else (
                "stopped" if stop.is_set() or any(item["status"].startswith("stopped") for item in finished.values()) else "complete")
            state = snapshot(status)
            atomic_json(options.output / "scheduler.train.json", state)
            refresh(state, force=True)
            return 1 if status == "failed" else 130 if status == "stopped" else 0
        except BaseException:
            stop.set()
            if options.stage == "single":
                finished[arms[0]] = dict(experiment_arm=arms[0], method="regir_nomem", seed=0,
                                         status="failed", error=traceback.format_exc())
            state = snapshot("failed")
            atomic_json(options.output / "scheduler.train.json", state)
            refresh(state, force=True)
            raise
        finally:
            stop.set()
            for item in active.values():
                item["child"].join()
            for bar in bars.values():
                bar.close()
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def single(options):
    """Train one seed in the foreground; validation stays inside train()."""
    from open_score.algos import train
    methods = selection(options.method, METHODS)
    seeds = selection(options.seeds, integer=True)
    devices = logical_devices(options.devices or "0")
    if len(methods) != 1 or len(seeds) != 1 or len(devices) != 1:
        raise ValueError("single requires one method, one seed and one GPU")
    if options.steps < 50:
        raise ValueError("The selected training protocol requires a budget >=50")
    if options.confirmation:
        raise ValueError("single performs training and validation; confirmation is a separate explicit evaluation")
    method, seed = methods[0], seeds[0]
    gpu = physical_device(devices[0])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    directory = run_directory(options.output, method, seed)
    directory.mkdir(parents=True, exist_ok=True)
    started_at = time.time()

    def publish(status):
        active = [dict(pool="train", method=method, seed=seed, gpu=gpu, pid=os.getpid())] if status == "running" else []
        finished = [] if active else [dict(pool="train", method=method, seed=seed, status=status)]
        atomic_json(options.output / "scheduler.train.json", dict(pid=os.getpid(), updated_at=time.time(),
            status=status, train_slots=1, eval_slots=0, total_train=1, total_eval=0,
            started_at=started_at, active=active, waiting_train=[], waiting_eval=[], finished=finished))

    with process_lock(options.output / ".train.scheduler.lock") as owned:
        if not owned:
            raise RuntimeError("This output already has a running trainer")
        with process_lock(directory / ".train.lock") as model_owned:
            if not model_owned:
                raise RuntimeError(f"Model already has a trainer: {method}/s{seed}")
            if (options.output / "stop.request").exists() and not options.resume:
                raise ValueError("Stopped output requires --resume")
            (options.output / "stop.request").unlink(missing_ok=True)
            with (directory / "console.log").open("a", encoding="utf-8", buffering=1) as stream:
                with redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(Tee(sys.stderr, stream)):
                    publish("running")
                    try:
                        path = train(method, dict(output=str(options.output), seed=seed, t_max=options.steps,
                            run=options.run, resume=options.resume, use_cuda=True,
                            env_args=environment_options(options), progress=options.progress,
                            live_curves=options.live_curves, launch_command=options.launch_command,
                            launch_argv=options.launch_argv, launch_platform=os.name, python_executable=sys.executable))
                        completed = Path(path).name == "final.pt"
                        publish("complete" if completed else "stopped")
                        print(f"[{method}] {'complete' if completed else 'stopped'}: {path}", flush=True)
                        return 0 if completed else 130
                    except BaseException:
                        publish("failed")
                        raise


def run(options):
    from open_score.eval.experiment import check_environment, ensure_layout
    check_environment()
    ensure_layout(options.output)
    methods = selection(options.method, METHODS)
    seeds = selection(options.seeds, integer=True)
    devices = logical_devices(options.devices or "0,1")
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
    parser.add_argument("--method", help="Explicit comma-separated current methods")
    parser.add_argument("--seeds", help="Explicit comma-separated training seeds")
    parser.add_argument("--seed", type=int, help="Alias for one training seed")
    parser.add_argument("--steps", type=int, help="Explicit physical-step budget")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preset", choices=(FRIENDLY_PRESET,))
    parser.add_argument("--arm", choices=(*FRIENDLY_ARMS, "all"), help="Named experiment arm; separate from evaluation standard/confirmation arms")
    parser.add_argument("--run", default="train")
    parser.add_argument("--devices", help="Logical GPU indices within inherited CUDA_VISIBLE_DEVICES; single defaults to 0")
    parser.add_argument("--per-gpu", type=positive_integer, help="Defaults to 1 for sky_friendly_v1 and 3 for legacy training")
    parser.add_argument("--gpu-slots", help="Friendly preset slots aligned with --devices, e.g. 2,1 for two jobs on GPU0 and one on GPU1")
    parser.add_argument("--eval-workers", type=positive_integer, default=2)
    parser.add_argument("--eval-device", type=evaluation_device, default="cpu", help="Independent evaluation pool device: cpu or cuda:N")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--confirmation", action="store_true", help="Also evaluate the explicitly requested independent confirmation bank")
    parser.add_argument("--env-agent-type", choices=("particle", "UAV_fixedwing", "UAV_quadrotor"))
    parser.add_argument("--env-agent-action-type", choices=("acceleration",))
    parser.add_argument("--spatial-dim", type=int, choices=(2, 3))
    parser.add_argument("--task-mode", choices=("damage",))
    parser.add_argument("--progress", choices=("plain", "tqdm"), default="plain")
    parser.add_argument("--live-curves", action="store_true", help="Refresh saved training curves and the experiment report every 30 seconds")
    options = parser.parse_args()
    if options.arm is not None and options.preset is None:
        options.preset = FRIENDLY_PRESET
    options.launch_argv = [sys.executable, "-X", "utf8", *sys.argv]
    options.launch_command = ("& " + " ".join("'" + str(value).replace("'", "''") + "'" for value in options.launch_argv)
                              if os.name == "nt" else shlex.join(options.launch_argv))
    options.output = (options.output or (FRIENDLY_OUTPUT if options.preset else DEFAULT_OUTPUT)).resolve()
    metadata = read_json(options.output / "experiment.json", {})
    if options.preset is None and metadata.get("preset") == FRIENDLY_PRESET:
        options.preset = FRIENDLY_PRESET
    if options.stage == "stop":
        from open_score.utils.resources import write_stop
        if options.arm not in (None, "all"):
            if options.preset != FRIENDLY_PRESET:
                parser.error("Named-arm stop requires a friendly manifest or --preset sky_friendly_v1")
            if options.seed not in (None, 0):
                parser.error("sky_friendly_v1 uses seed 0")
            path = run_directory(options.output, "regir_nomem", 0, experiment_arm=options.arm) / "pause.request"
        else:
            path = options.output / "stop.request"
        write_stop(path, "Finish the current complete batch/episode and retain recovery state")
        print(f"Stop requested: {path}", flush=True)
        return 0
    if options.stage == "status":
        state = read_json(options.output / "scheduler.train.json", {})
        if options.preset == FRIENDLY_PRESET:
            state["preset"] = FRIENDLY_PRESET
        render(progress_lines(options.output, state, options.arm))
        return 0
    if options.seed is not None:
        if options.seeds is not None:
            parser.error("Use either --seed or --seeds")
        options.seeds = str(options.seed)
    if options.preset == FRIENDLY_PRESET:
        if options.arm is None:
            parser.error("sky_friendly_v1 requires explicit --arm A, B, C or all")
        options.method = options.method or "regir_nomem"
        options.seeds = options.seeds if options.seeds is not None else "0"
        options.steps = options.steps if options.steps is not None else 1000000
        options.per_gpu = 1 if options.per_gpu is None else options.per_gpu
        return friendly(options)
    if options.arm is not None:
        parser.error("--arm requires --preset sky_friendly_v1 or its saved manifest")
    if options.gpu_slots is not None:
        parser.error("--gpu-slots requires --preset sky_friendly_v1")
    options.per_gpu = 3 if options.per_gpu is None else options.per_gpu
    if not options.method or options.seeds is None or options.steps is None:
        parser.error("Training requires explicit --method, --seed/--seeds and --steps")
    if options.stage == "single":
        return single(options)
    if environment_options(options) or options.progress != "plain" or options.live_curves:
        parser.error("Environment selection, tqdm and live curves require --stage single")
    return run(options)

if __name__ == "__main__":
    raise SystemExit(main())
