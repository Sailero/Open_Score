"""Cross-scale implementation checks, approved short runs, and stage-three launch."""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import json
from multiprocessing import get_context
import os
from pathlib import Path
from queue import Empty
import signal
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from open_score.algos import METHODS, setup_runtime
from open_score.utils.logging import DEFAULT_OUTPUT, FORMAL_RUN

BASELINE_METHODS = METHODS[:3]
MAIN_METHODS = ("dcg", "spectra")
METHOD_GROUPS = {
    "baseline": BASELINE_METHODS,
    "main": MAIN_METHODS,
    "alma": ("alma",),
    "dcg_alma": ("dcg", "alma"),
}
FORMAL_METHODS = BASELINE_METHODS
FORMAL_SEEDS = (0, 1, 2)
MAX_CONCURRENT = 3

_METHOD_ALIAS = {
    "b0_qmix": "b0",
    "b2_qmix_atten": "b2",
    "refil": "refil",
    "dcg": "dcg",
    "gnn_qmix": "gnn",
    "spectra": "spectra",
    "alma": "alma",
}
_STATUS_ALIAS = {
    "starting": "init",
    "resuming": "resume",
    "training": "train",
    "evaluating": "eval",
    "completed": "done",
    "stopped": "stop",
    "failed": "fail",
    "running": "run",
}


def _enable_ansi():
    if not sys.stdout.isatty():
        return False
    if os.name != "nt":
        return True
    try:
        import ctypes
        handle = ctypes.windll.kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint()
        if not ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(ctypes.windll.kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except OSError:
        return False


def _status_name(method, seed):
    name = _METHOD_ALIAS.get(method, method)
    return name if seed is None else f"{name} s{int(seed)}"


def _hms(seconds):
    seconds = int(max(0, float(seconds or 0)))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def _status_line(method, row, name_width, seed=None):
    name = _status_name(method, seed if seed is not None else row.get("seed"))
    status = _STATUS_ALIAS.get(row.get("status", "starting"), row.get("status", "init"))
    t_env = int(row.get("t_env") or 0)
    budget = int(row.get("budget_steps") or 0)
    sps = float(row.get("steps_per_second") or 0)
    loss, dval = row.get("loss"), row.get("latest_validation_D")
    loss_s = "-" if loss is None else f"{float(loss):.4f}"
    d_s = "-" if dval is None else f"{float(dval):.3f}"
    extra = f"  ev {row.get('eval_completed', 0)}/{row['eval_total']}" if row.get("eval_total") else ""
    return (f"{name:<{name_width}}  {status:<6}  {_hms(row.get('elapsed_seconds')):>8}  "
            f"{t_env:>9,}/{budget:<9,}  {sps:5.1f}/s  L={loss_s:<8} D={d_s}{extra}")


def _rewrite_block(lines, previous, use_ansi):
    if use_ansi and previous:
        sys.stdout.write(f"\x1b[{previous}F\x1b[J")
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()
    return len(lines)


class _JobAlreadyRunning(RuntimeError):
    pass


@contextmanager
def _exclusive_job(method, run, seed):
    """One live trainer per method/run/seed; a second copy must not write the same files."""
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
    handle = api.CreateMutexW(None, False, f"Local\\OpenScoreJob_{method}_{run}_{int(seed)}")
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    acquired = False
    try:
        if api.WaitForSingleObject(handle, 0) not in (0, 0x80):
            raise _JobAlreadyRunning(f"{method} [{run} seed={seed}] is already running")
        acquired = True
        yield
    finally:
        if acquired:
            api.ReleaseMutex(handle)
        api.CloseHandle(handle)


def _flag(command, name):
    parts = (command or "").split()
    token = f"--{name}"
    if token not in parts:
        return None
    index = parts.index(token)
    if index + 1 >= len(parts):
        return None
    return parts[index + 1]


def _occupied_train_jobs():
    """Seeds already owned by another train.py, including jobs started before the mutex."""
    occupied = set()
    try:
        import subprocess
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -match 'train.py' } | "
             "ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }"],
            capture_output=True, text=True, timeout=20)
    except Exception:
        return occupied
    mine = {str(os.getpid())}
    for line in completed.stdout.splitlines():
        pid, _, command = line.partition("\t")
        if pid in mine or "train.py" not in command:
            continue
        method, seed = _flag(command, "method"), _flag(command, "seed")
        if method is None or seed is None:
            continue
        occupied.add((method, _flag(command, "run") or FORMAL_RUN, int(seed)))
    return occupied


def _job_entry(method, options, stop_event, results):
    from open_score.algos import train
    directory = Path(options["output"]) / method / options["run"] / f"seed_{options['seed']}"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "console.log").open("a", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                with _exclusive_job(method, options["run"], options["seed"]):
                    result = train(method, {**options, "_stop_event": stop_event})
                results.put(dict(method=method, run=options["run"], seed=options["seed"],
                                 status="stopped" if stop_event.is_set() else "completed", checkpoint=result))
            except _JobAlreadyRunning as error:
                print(error, flush=True)
                results.put(dict(method=method, run=options["run"], seed=options["seed"],
                                 status="skipped", error=str(error)))
            except BaseException:
                detail = traceback.format_exc()
                print(detail, flush=True)
                results.put(dict(method=method, run=options["run"], seed=options["seed"], status="failed", error=detail))


def _anchor_entry(output, stop_event, results):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from open_score.eval.anchors import run_anchors
    try:
        result = run_anchors(output, stop_requested=stop_event.is_set)
        results.put(dict(method="anchors", status="stopped" if stop_event.is_set() else "completed", result=result))
    except BaseException:
        results.put(dict(method="anchors", status="failed", error=traceback.format_exc()))


def run_group(jobs, output, *, with_anchors=False, report_run=FORMAL_RUN, max_concurrent=MAX_CONCURRENT):
    """Run a job pool with a fixed number of live slots; a finished job frees a slot."""
    from open_score.utils.logging import read_latest
    from open_score.eval.report import refresh_report
    if max_concurrent < 1 or max_concurrent > 4:
        raise ValueError("Concurrent experiment tasks must be between 1 and 4")
    pending_jobs = []
    latest = read_latest(output, "progress")
    occupied = _occupied_train_jobs()
    for method, options in jobs:
        options = dict(options)
        directory = Path(output) / method / options["run"] / f"seed_{options['seed']}"
        row = latest.get((method, options["run"], options["seed"]), {})
        if row.get("status") in ("completed", "complete") and any((directory / name).exists() for name in ("final.pt", "best.pt")):
            print(f"{method} [{options['run']} seed={options['seed']}] already done", flush=True)
            continue
        key = (method, options["run"], options["seed"])
        if key in occupied:
            print(f"{method} [{options['run']} seed={options['seed']}] already running; skip duplicate", flush=True)
            continue
        if options.get("resume"):
            saved_config = directory / "config.json"
            if saved_config.exists():
                saved = json.loads(saved_config.read_text(encoding="utf-8"))
                for key in ("seed", "env", "batch_size_run", "t_max"):
                    if saved[key] != options[key]:
                        raise ValueError(f"Resume configuration differs: {method}/{key}")
            if not (directory / "resume.pt").exists():
                if saved_config.exists():
                    raise RuntimeError(f"Recorded run has no recoverable checkpoint: {directory}")
                options["resume"] = False
        pending_jobs.append((method, options))
    if with_anchors:
        pending_jobs.append(("anchors", None))
    if not pending_jobs:
        return True
    context = get_context("spawn")
    stop_event, queue = context.Event(), context.Queue()
    waiting = deque(pending_jobs)
    live = []
    finished = []
    use_ansi = _enable_ansi()
    status_height = 0
    previous = signal.getsignal(signal.SIGINT)

    def request_stop(signum, frame):
        nonlocal status_height
        stop_event.set()
        status_height = 0
        print("stop requested: finish batch and save resume", flush=True)

    def launch():
        while waiting and len(live) < max_concurrent and not stop_event.is_set():
            method, options = waiting.popleft()
            if method == "anchors":
                child = context.Process(target=_anchor_entry, args=(str(output), stop_event, queue))
                identity = ("anchors", None, None)
            else:
                child = context.Process(target=_job_entry, args=(method, options, stop_event, queue))
                identity = (method, options["run"], options["seed"])
            child.start()
            live.append({"child": child, "method": method, "options": options, "key": identity})

    signal.signal(signal.SIGINT, request_stop)
    try:
        launch()
        pool = len(pending_jobs)
        runs = {options["run"] for _, options in pending_jobs if options}
        header = []
        if len(runs) == 1:
            header.append(f"run={next(iter(runs))}")
        header.append(f"live={len(live)}")
        header.append(f"queue={len(waiting)}")
        header.append(f"pool={pool}")
        print("  ".join(header), flush=True)
        last_status, last_report = 0.0, time.monotonic()
        results, failures, abnormal_exits = [], [], set()

        def accept_result(result):
            nonlocal last_report, status_height
            results.append(result)
            if result["status"] == "failed":
                failures.append(result)
                stop_event.set()
                status_height = 0
                print(f"{result.get('method', 'worker')} failed:\n{result['error']}\n"
                      "other jobs will finish the current batch and save", flush=True)
                last_report = 0.0

        def reap():
            nonlocal status_height
            still = []
            progressed = False
            for item in live:
                child = item["child"]
                if child.is_alive():
                    still.append(item)
                    continue
                progressed = True
                finished.append(item)
                if child.exitcode not in (None, 0) and child.pid not in abnormal_exits:
                    abnormal_exits.add(child.pid)
                    failure = dict(status="failed", error=f"Worker {child.pid} exit code {child.exitcode}")
                    failures.append(failure)
                    stop_event.set()
                    status_height = 0
                    print(f"{failure['error']}; other jobs asked to save and stop", flush=True)
            live[:] = still
            return progressed

        while live or (waiting and not stop_event.is_set()):
            if (Path(output) / "stop.request").exists():
                stop_event.set()
            while True:
                try:
                    accept_result(queue.get_nowait())
                except Empty:
                    break
            if reap():
                launch()
            now = time.monotonic()
            if now - last_status >= 10:
                keys = {item["key"] for item in live if item["key"][0] != "anchors"}
                try:
                    latest = read_latest(output, "progress", keys=keys)
                except MemoryError:
                    latest = {}
                    print("status skipped: MemoryError", flush=True)
                if latest or keys:
                    width = max(len(_status_name(key[0], key[2])) for key in keys)
                    lines = [_status_line(key[0], latest.get(key, {}), width, seed=key[2])
                             for key in sorted(keys)]
                    status_height = _rewrite_block(lines, status_height, use_ansi)
                last_status = now
            if now - last_report >= 300:
                try:
                    refresh_report(output, run=report_run)
                except (MemoryError, TimeoutError, OSError) as error:
                    print(f"report refresh skipped: {type(error).__name__}", flush=True)
                last_report = now
            for item in live:
                item["child"].join(timeout=0.2)
        while True:
            try:
                accept_result(queue.get_nowait())
            except Empty:
                break
        failures.extend(dict(status="failed", error=f"Worker {item['child'].pid} exit code {item['child'].exitcode}")
                        for item in finished if item["child"].exitcode and item["child"].pid not in abnormal_exits)
        try:
            refresh_report(output, run=report_run)
        except (MemoryError, TimeoutError, OSError) as error:
            print(f"report refresh skipped: {type(error).__name__}", flush=True)
        if failures:
            raise RuntimeError("\n".join(item["error"] for item in failures))
        expected = len(finished) if stop_event.is_set() else pool
        if len(results) != expected:
            raise RuntimeError("An experiment process exited without reporting completion")
        return not stop_event.is_set()
    finally:
        signal.signal(signal.SIGINT, previous)


def validate(output):
    """Run the remaining approved V1-V3 checks; reuse recorded passing items."""
    setup_runtime()
    import numpy as np
    import torch
    from open_score.algos import load_config, make_scheme, LearnerLogger
    from open_score.envs.entity_env import HADEntityEnv, run_environment_checks
    from open_score.models import build_mac, run_model_checks
    from open_score.utils.logging import ExperimentLogger, read_records
    from open_score.eval.report import refresh_report
    from runners.parallel_runner import ParallelRunner
    record = ExperimentLogger(output, run="validation")
    latest = {row["id"]: row for row in read_records(output, "verification", run="validation") if row.get("id")}

    def retain(rows):
        record.verification(rows)
        latest.update((row["id"], row) for row in rows)

    def passed(identity):
        if latest.get(identity, {}).get("status") == "passed":
            print(f"[validation] {identity}: reuse recorded passed result", flush=True)
            return True
        return False

    try:
        environment_ids = ("V1-actions", "V1-physics-reward", "V1-natural-bootstrap",
                           "V1-truncation-bootstrap", "V1-shaping-invariance",
                           "V1-subtask-decomposition")
        if any(latest.get(identity, {}).get("status") != "passed" for identity in environment_ids):
            retain(run_environment_checks())
        missing_or_failed = [identity for identity in environment_ids if latest.get(identity, {}).get("status") != "passed"]
        if missing_or_failed:
            raise RuntimeError("HAD checks require repair/recorded completion: " + ", ".join(missing_or_failed))
        print("[validation] HAD semantic checks: recorded passed results retained", flush=True)
        model_rows = [row for identity, row in latest.items() if identity.startswith(("V2.", "V3."))]
        expected_model = [f"V2.{method}.variable_count" for method in METHODS]
        if "alma" in METHODS:
            expected_model += [
                "V2.alma.entity_permutation", "V2.alma.agent_equivariance",
                "V2.alma.monotonicity", "V2.alma.mixer_alignment",
                "V3.alma.death_noop", "V3.alma.episode_hidden_reset",
                "V3.alma.encoder_execution_equivalence", "V3.alma.last_action",
                "V3.alma.attention_mask", "V3.alma.allocation_gates_observation",
                "V3.alma.allocation_validity", "V3.alma.subtask_width_extrapolation",
            ]
        missing_model = [identity for identity in expected_model
                         if latest.get(identity, {}).get("status") not in ("passed", "not_applicable")]
        if missing_model or not model_rows:
            retain(run_model_checks(only=None if not model_rows else missing_model))
            model_rows = [row for identity, row in latest.items() if identity.startswith(("V2.", "V3."))]
        unresolved = [row["id"] for row in model_rows if row["status"] not in ("passed", "not_applicable")]
        if unresolved:
            raise RuntimeError("Model checks require repair before continuing: " + ", ".join(unresolved))
        print("[validation] Model checks: recorded passed/not-applicable results retained", flush=True)

        if not passed("V1.mixed_workers"):
            runner = None
            try:
                args = load_config("b2_qmix_atten", dict(batch_size_run=8, use_cuda=False, output=str(output), seed=0, t_max=20000))
                runner = ParallelRunner(args, LearnerLogger())
                for key, value in runner.get_env_info().items():
                    setattr(args, key, value)
                scheme, groups, preprocess = make_scheme(runner.get_env_info())
                import copy
                model_scheme = copy.deepcopy(scheme)
                model_scheme["entities"]["vshape"] = args.entity_shape
                model_scheme["actions_onehot"] = {"vshape": (args.n_actions,), "group": "agents"}
                runner.setup(scheme, groups, preprocess, build_mac(model_scheme, groups, args))
                scales = [(4, 4, 1), (6, 6, 2), (8, 8, 3), (10, 10, 1), (4, 4, 3), (6, 6, 1), (8, 8, 2), (10, 10, 3)]
                jobs = [dict(episode_seed=12000 + i, config=dict(zip(("N_R", "N_B", "K"), scale))) for i, scale in enumerate(scales)]
                batch, summaries = runner.run(test_mode=True, jobs=jobs)
                from open_score.envs.features import TRAIN_AGENTS, TRAIN_BLUE, TRAIN_ENTITIES
                for i, (nr, nb, k) in enumerate(scales):
                    expected = np.ones(TRAIN_ENTITIES, dtype=np.uint8)
                    expected[:nr] = 0
                    expected[TRAIN_AGENTS:TRAIN_AGENTS + nb] = 0
                    expected[TRAIN_AGENTS + TRAIN_BLUE:TRAIN_AGENTS + TRAIN_BLUE + k] = 0
                    assert np.array_equal(batch["entity_mask"][i, 0].numpy(), expected)
                    filled = batch["filled"][i, :, 0].bool()
                    mask = batch["entity_mask"][i, filled].bool()
                    assert torch.all(batch["entities"][i, filled][mask] == 0)
                    assert torch.equal(batch["agent_mask"][i, filled],
                                       1 - batch["entity_mask"][i, filled, :TRAIN_AGENTS])
                retain([dict(id="V1.mixed_workers", status="passed", detail="8 workers, 8 explicit mixed configurations; initial and filled-state masks verified")])
            except BaseException as error:
                retain([dict(id="V1.mixed_workers", status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                             detail=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())])
                raise
            finally:
                if runner is not None:
                    runner.close_env()

        failed_throughput = []
        from open_score.envs.features import MAX_AGENTS
        for config, minimum in [((8, 8, 2), 800), ((20, 20, 4), 200)]:
            identity = f"V1.throughput.{config}"
            if passed(identity):
                continue
            env = None
            try:
                env = HADEntityEnv(scale=config, seed=0)
                rng = np.random.default_rng(19001)
                steps, seed = 0, 19000
                started = time.perf_counter()
                while steps < 1000:
                    env.reset(seed=seed)
                    seed += 1
                    done = False
                    while not done and steps < 1000:
                        _, done, _ = env.step(rng.integers(0, 9, MAX_AGENTS))
                        env.get_entities(); env.get_masks(); env.get_avail_actions()
                        steps += 1
                elapsed = time.perf_counter() - started
                rate = steps / elapsed
                retain([dict(id=identity, status="passed" if rate >= minimum else "failed",
                             detail=f"{rate:.1f} physical steps/s; threshold {minimum}", value=rate)])
                if rate < minimum:
                    failed_throughput.append(identity)
            except BaseException as error:
                retain([dict(id=identity, status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                             detail=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())])
                raise
            finally:
                if env is not None:
                    env.close()
        if failed_throughput:
            raise RuntimeError("Throughput verification failed: " + ", ".join(failed_throughput))
    finally:
        refresh_report(output)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--stage", choices=("validate", "e0", "benchmark", "train", "single", "stop"), required=True)
    result.add_argument("--method", choices=METHODS)
    result.add_argument("--group", choices=tuple(METHOD_GROUPS), default="main",
                        help="train pool: main=DCG/SPECTra, baseline=B0/B2/REFIL, alma=ALMA, dcg_alma=DCG/ALMA")
    result.add_argument("--steps", type=int)
    result.add_argument("--batch-size-run", type=int, choices=(4, 8), default=4)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--cpu", action="store_true")
    result.add_argument("--env", choices=("had", "ff", "rel_overgen"), default="had")
    # Left unset so a single-method run cannot inherit the formal run name.
    result.add_argument("--run", default=None)
    return result


def record_e0_results(output):
    import math
    from open_score.utils.logging import ExperimentLogger, read_records
    progress = {row["method"]: row for row in read_records(output, "progress", run="e0", seed=0)}
    learning = read_records(output, "learning", run="e0", seed=0)
    results = []
    for method in METHODS:
        status = progress.get(method, {})
        measured = [row for row in learning if row["method"] == method and "loss" in row]
        valid = bool(measured) and all(math.isfinite(float(row[key]))
                                      for row in measured for key in ("loss", "grad_norm") if key in row)
        passed = status.get("status") == "completed" and status.get("t_env") == 20000 and status.get("updates", 0) > 0 and valid
        result = dict(id=f"V4.E0.{method}", status="passed" if passed else "failed",
                      detail=dict(env="rel_overgen" if method == "dcg" else "ff", seed=0,
                                  physical_steps=status.get("t_env", 0), updates=status.get("updates", 0),
                                  finite_learning_chain=valid, interpretation="chain/numerical validity only; no convergence claim"))
        ExperimentLogger(output, method=method, seed=0, run="e0").verification([result])
        results.append(result)
    if any(row["status"] == "failed" for row in results):
        raise RuntimeError("E0 learning-chain records are incomplete or non-finite")


def require_preparation(output):
    """Formal training follows the recorded correctness checks.

    The E0 learning-chain and throughput-benchmark gates belonged to the v2
    staged plan. Throughput is already measured and unchanged by the value
    and slot fixes, so this version gates on `--stage validate` only.
    """
    from open_score.utils.logging import read_records
    checks = {row["id"]: row for row in read_records(output, "verification", run="validation")}
    required = ("V1-actions", "V1-physics-reward", "V1-natural-bootstrap", "V1-truncation-bootstrap",
                "V1-shaping-invariance", "V1-subtask-decomposition", "V1.mixed_workers",
                "V1.throughput.(8, 8, 2)", "V1.throughput.(20, 20, 4)")
    problems = [key for key in required if checks.get(key, {}).get("status") != "passed"]
    problems += [key for key, row in checks.items() if row["status"] == "failed"]
    for method in METHODS:
        if checks.get(f"V2.{method}.variable_count", {}).get("status") != "passed":
            problems.append(f"V2.{method}.variable_count")
    if problems:
        raise SystemExit("run --stage validate first: " + "; ".join(problems))


def main():
    options = parser().parse_args()
    setup_runtime()
    output = options.output.resolve()
    if options.stage == "stop":
        output.mkdir(parents=True, exist_ok=True)
        (output / "stop.request").write_text("Stop after the current complete sampling/learning batch.\n", encoding="utf-8")
        print("stop.request written; wait for stopped + resume, do not kill", flush=True)
        return
    if options.stage in ("e0", "benchmark", "train", "single"):
        (output / "stop.request").unlink(missing_ok=True)
    base = dict(output=str(output), seed=options.seed, batch_size_run=options.batch_size_run,
                use_cuda=not options.cpu, resume=options.resume)
    if options.stage == "validate":
        validate(output)
    elif options.stage == "e0":
        # Six approved native runs, exactly 20k physical steps each, seed 0.
        for methods in (METHODS[:3], METHODS[3:]):
            jobs = [(method, {**base, "seed": 0, "t_max": 20000, "run": "e0",
                              "env": "rel_overgen" if method == "dcg" else "ff"}) for method in methods]
            if not run_group(jobs, output, report_run="e0"):
                return
        record_e0_results(output)
    elif options.stage == "benchmark":
        prefix = "benchmark" if options.run is None else options.run
        if not prefix.startswith("benchmark"):
            raise SystemExit("benchmark --run must start with 'benchmark'")
        for workers in (4, 8):
            for concurrency in (1, 4):
                run = f"{prefix}_w{workers}_c{concurrency}"
                jobs = [("refil", {**base, "seed": rank, "batch_size_run": workers, "t_max": 20000,
                                    "env": "had", "run": run, "concurrency": concurrency}) for rank in range(concurrency)]
                if not run_group(jobs, output, report_run=run):
                    return
    elif options.stage == "train":
        if options.steps is None:
            raise SystemExit("formal train needs --steps; no default budget")
        if options.steps < 50:
            raise SystemExit("formal budget must support 50 validation points")
        require_preparation(output)
        methods = METHOD_GROUPS[options.group]
        # Default group is DCG / SPECTra. GNN seed 0 is already done; later
        # GNN seeds are not queued. B0 / B2 / REFIL stay in baseline.
        jobs = [(method, {**base, "seed": seed, "t_max": options.steps, "env": "had", "run": FORMAL_RUN})
                for seed in FORMAL_SEEDS for method in methods]
        run_group(jobs, output)
    else:
        if options.method is None or options.steps is None:
            raise SystemExit("single requires --method and --steps")
        # A native sanity environment must never default into the HAD protocol.
        run = options.run or (FORMAL_RUN if options.env == "had" else f"single_{options.env}")
        run_group([(options.method, {**base, "t_max": options.steps, "env": options.env, "run": run})], output,
                  report_run=run)


if __name__ == "__main__":
    main()
