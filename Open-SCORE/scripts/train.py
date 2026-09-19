"""Cross-scale implementation checks, approved short runs, and stage-three launch."""
from __future__ import annotations

import argparse
from collections import Counter, deque
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

from open_score.algos import METHODS, PROBE_METHODS, V4_METHODS, V5_METHODS, POLICY_METHODS, resume_files, setup_runtime
from open_score.utils.logging import (
    DEFAULT_OUTPUT, FORMAL_RUN, live_console_jobs, parse_eval_console,
    parse_train_console, read_tail,
)

BASELINE_METHODS = METHODS[:3]
MAIN_METHODS = ("dcg", "spectra")
PROBE_OUTPUT = DEFAULT_OUTPUT.parent / "alma_probe_v3"
METHOD_GROUPS = {
    "baseline": BASELINE_METHODS,
    "main": MAIN_METHODS,
    "alma": ("alma",),
    "dcg_alma": ("dcg", "alma"),
    "alma_probe": PROBE_METHODS,
    "v4": V4_METHODS,
    "v5": V5_METHODS,
}
FORMAL_METHODS = BASELINE_METHODS
FORMAL_SEEDS = (0, 1, 2)
MAX_CONCURRENT = 2
# 5070 Ti 16 GiB, WDDM already holds ~1 GiB. Estimates are cuda_reserved
# from solo training; unknown methods are treated as ReGIR-sized.
GPU_BUDGET_GIB = 13.0
GPU_SAFETY_GIB = 1.2
GPU_RESERVED_GIB = {
    "regir": 10.5, "regir_norefil": 10.5, "regir_nocount": 10.5,
    "regir_r1": 10.5, "regir_last": 10.5, "refil_cycle": 10.5,
    "refil_card": 11.7, "refil_feedback": 10.7, "refil_slot": 7.5,
    "alma": 6.5, "alma_legacy": 5.8, "refil_matched": 8.0,
    "refil": 3.2, "refil_count": 2.4, "gnn_qmix": 1.8,
    "b2_qmix_atten": 1.3, "refil_local_mild": 1.2, "refil_local_mid": 1.2,
    "b0_qmix": 0.8, "spectra": 0.6, "dcg": 0.5,
}
CYCLE_METHODS = frozenset(("regir", "regir_norefil", "regir_nocount", "regir_r1", "regir_last", "refil_cycle"))
GROUP_MAX_LIVE = {"cycle": 1, "alma": 2, "matched": 1, "light": 2}

_METHOD_ALIAS = {
    "b0_qmix": "b0",
    "b2_qmix_atten": "b2",
    "refil": "refil",
    "refil_local_mild": "Lmild",
    "refil_local_mid": "Lmid",
    "refil_count": "count",
    "refil_cycle": "cycle",
    "regir": "regir",
    "regir_norefil": "noRF",
    "regir_nocount": "nozn",
    "regir_r1": "R1",
    "regir_last": "last",
    "refil_matched": "RFmat",
    "refil_card": "card",
    "refil_feedback": "fb",
    "refil_slot": "slot",
    "dcg": "dcg",
    "gnn_qmix": "gnn",
    "spectra": "spectra",
    "alma": "alma",
    "alma_fullobs": "fullobs",
    "alma_blue": "blue",
    "alma_event": "event",
    "alma_nomask": "nomask",
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
    "queued": "queue",
}


def _enable_ansi():
    """Rewrite in place even when Python reports a pipe (Cursor / ConPTY)."""
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


def _gpu_estimate(method):
    return GPU_RESERVED_GIB.get(method, 10.5)


def _gpu_free_gib():
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            encoding="utf-8", timeout=5)
        values = [float(part) for part in out.replace(",", " ").split() if part.strip()]
        return min(values) / 1024.0 if values else None
    except Exception:
        return None


def _gpu_used_gib():
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            encoding="utf-8", timeout=5)
        used, total = [float(part) for part in out.replace(",", " ").split() if part.strip()][:2]
        return used / 1024.0, total / 1024.0
    except Exception:
        return None, None


def _job_group(method):
    if method in CYCLE_METHODS:
        return "cycle"
    if method in ("alma", "alma_legacy"):
        return "alma"
    if method == "refil_matched":
        return "matched"
    return "light"


def _fits_gpu(method, live):
    group = _job_group(method)
    live_same = sum(1 for item in live if _job_group(item["method"]) == group)
    if live_same >= GROUP_MAX_LIVE.get(group, 1):
        return False
    occupied = [item["method"] for item in live if item["method"] not in ("depth_eval", "anchors")]
    if occupied and sum(_gpu_estimate(name) for name in occupied) + _gpu_estimate(method) > GPU_BUDGET_GIB:
        return False
    free = _gpu_free_gib()
    if free is not None and _gpu_estimate(method) + GPU_SAFETY_GIB > free:
        return False
    return True


def _print_train_plan(jobs):
    groups = {"cycle": [], "alma": [], "matched": [], "light": []}
    for method, options in jobs:
        seed = None if not options else options.get("seed")
        groups[_job_group(method)].append(_status_name(method, seed))
    print("train waves on 16G  (cycle×1, alma×2, matched×1)", flush=True)
    if groups["cycle"]:
        print("  1. cycle 1-wide : " + ", ".join(groups["cycle"]), flush=True)
    if groups["alma"]:
        print("  2. alma  2-wide : " + ", ".join(groups["alma"]), flush=True)
    if groups["matched"]:
        print("  3. match 1-wide : " + ", ".join(groups["matched"]), flush=True)
    if groups["light"]:
        print("  4. light 2-wide : " + ", ".join(groups["light"]), flush=True)


def _status_name(method, seed):
    name = _METHOD_ALIAS.get(method, method)
    return name if seed is None else f"{name} s{int(seed)}"


def _hms(seconds):
    seconds = int(max(0, float(seconds or 0)))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def _status_line(method, row, name_width, seed=None):
    display = row.get("target_method", method)
    if row.get("job") == "depth_eval" or method == "depth_eval":
        name = f"{_METHOD_ALIAS.get(display, display)} R"
    else:
        name = _status_name(display, seed if seed is not None else row.get("seed"))
    status = _STATUS_ALIAS.get(row.get("status", "starting"), row.get("status", "init"))
    evaluating = (row.get("job") == "depth_eval" or method == "depth_eval"
                  or row.get("phase") in ("final_eval", "depth_eval"))
    if evaluating:
        completed = int(row.get("completed", row.get("eval_completed")) or 0)
        total = int(row.get("total", row.get("eval_total")) or 0)
        depth = row.get("cycle_depth")
        rbit = f"R={int(depth)}" if depth is not None else ""
        eta = row.get("remaining_seconds")
        eta_bit = f"  eta {_hms(eta)}" if eta is not None and total else ""
        progress = f"{completed:,}/{total:,}" if total else ""
        return f"{name:<{name_width}}  {status:<6}  {rbit:<4}  {progress}{eta_bit}"
    t_env = int(row.get("t_env") or 0)
    budget = int(row.get("budget_steps") or 0)
    if status in ("queue", "queued"):
        return f"{name:<{name_width}}  {status:<6}  {t_env:>9,}"
    sps = float(row.get("steps_per_second") or 0)
    loss, dval = row.get("loss"), row.get("latest_validation_D")
    loss_s = "-" if loss is None else f"{float(loss):.4f}"
    d_s = "-" if dval is None else f"{float(dval):.3f}"
    extra = ""
    completed = row.get("completed", row.get("eval_completed"))
    total = row.get("total", row.get("eval_total"))
    if total:
        extra = f"  val {int(completed or 0)}/{int(total)}"
    reserved = row.get("cuda_reserved_gib")
    rss = row.get("process_tree_rss_gib")
    if reserved is not None:
        extra += f"  V={float(reserved):.1f}G"
    if rss is not None:
        extra += f"  ram={float(rss):.1f}G"
    elapsed = row.get("session_elapsed_seconds", row.get("elapsed_seconds"))
    return (f"{name:<{name_width}}  {status:<6}  {_hms(elapsed):>8}  "
            f"{t_env:>9,}/{budget:<9,}  {sps:5.1f}/s  L={loss_s}  D={d_s}{extra}")


def _rewrite_block(lines, previous, use_ansi):
    block = "\n".join(lines) + "\n"
    if use_ansi and previous:
        sys.stdout.write(f"\x1b[{previous}F\x1b[J{block}")
        sys.stdout.flush()
        return len(lines)
    sys.stdout.write(block)
    sys.stdout.flush()
    return len(lines) if use_ansi else 0


def _live_row_from_console(output, item):
    key = item["key"]
    if key[0] == "depth_eval":
        method, run, seed = key[3], key[1], key[2]
        path = Path(output) / method / run / f"seed_{seed}" / "console.log"
        return parse_eval_console(read_tail(path))
    method, run, seed = key[0], key[1], key[2]
    path = Path(output) / method / run / f"seed_{seed}" / "console.log"
    return parse_train_console(read_tail(path))


def run_status_board(output, interval=3.0):
    """Watch live console logs. Does not start or stop trainers."""
    use_ansi = _enable_ansi()
    height = 0
    print("live status from console.log  (Ctrl+C to close this view only)", flush=True)
    try:
        while True:
            jobs = live_console_jobs(output)
            used, total = _gpu_used_gib()
            gpu_bit = f"  gpu={used:.1f}/{total:.1f}G" if used is not None else ""
            trains = [job for job in jobs if job["kind"] == "train"]
            evals = [job for job in jobs if job["kind"] == "eval"]
            lines = [f"now  train={len(trains)}  eval={len(evals)}{gpu_bit}"]
            width = 12
            names = [(_status_name(job["method"], job["seed"]), job) for job in trains + evals]
            if names:
                width = max(width, max(len(name) for name, _ in names))
            if not names:
                lines.append("idle  no console written in the last 3 min")
            for name, job in names:
                if job["kind"] == "train":
                    row = parse_train_console(read_tail(job["path"]))
                    row.setdefault("status", "starting")
                    lines.append(_status_line(job["method"], row, width, seed=job["seed"]))
                else:
                    row = parse_eval_console(read_tail(job["path"]))
                    row.setdefault("status", "running")
                    row.setdefault("phase", "final_eval")
                    lines.append(f"{name:<{width}}  eval    {row.get('phase', 'eval'):<11}  "
                                 f"{int(row.get('completed') or 0):,}/{int(row.get('total') or 0):,}"
                                 + (f"  R={row['cycle_depth']}" if row.get("cycle_depth") else ""))
            height = _rewrite_block(lines, height, use_ansi)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nstatus view closed", flush=True)


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


def _training_finished(directory, row):
    directory = Path(directory)
    # final.pt is written only after the physical-step budget. Do not requeue
    # that arm just because progress.csv was briefly unreadable.
    if (directory / "best.pt").exists() and (directory / "final.pt").exists():
        return True
    if row.get("status") in ("completed", "complete"):
        return any((directory / name).exists() for name in ("final.pt", "best.pt"))
    return False


def _depth_eval_entry(options, stop_event, results):
    from open_score.eval.protocol import evaluate_depth_sweep
    method = options["target_method"]
    directory = Path(options["output"]) / method / options["run"] / f"seed_{options['seed']}"
    checkpoint = directory / "best.pt"

    def on_progress(progress):
        payload = dict(progress)
        payload.update(kind="progress", method=method, run=options["run"], seed=options["seed"])
        try:
            results.put_nowait(payload)
        except Exception:
            pass

    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "console.log").open("a", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                with _exclusive_job(f"{method}_depth", options["run"], options["seed"]):
                    result = evaluate_depth_sweep(
                        method, checkpoint, output=options["output"], run=options["run"],
                        seed=options["seed"], device="cpu", on_progress=on_progress,
                        stop_requested=stop_event.is_set)
                results.put(dict(kind="depth_eval", method=method, run=options["run"],
                                 seed=options["seed"], status=result.get("status", "completed"),
                                 completed=result.get("completed"), total=result.get("total")))
            except _JobAlreadyRunning as error:
                print(error, flush=True)
                results.put(dict(kind="depth_eval", method=method, run=options["run"],
                                 seed=options["seed"], status="skipped", error=str(error)))
            except BaseException:
                detail = traceback.format_exc()
                print(detail, flush=True)
                results.put(dict(kind="depth_eval", method=method, run=options["run"],
                                 seed=options["seed"], status="failed", error=detail))


def _anchor_entry(output, stop_event, results):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from open_score.eval.anchors import run_anchors
    try:
        result = run_anchors(output, stop_requested=stop_event.is_set)
        results.put(dict(method="anchors", status="stopped" if stop_event.is_set() else "completed", result=result))
    except BaseException:
        results.put(dict(method="anchors", status="failed", error=traceback.format_exc()))


def run_group(jobs, output, *, with_anchors=False, report_run=FORMAL_RUN, max_concurrent=MAX_CONCURRENT,
              with_depth_eval=True):
    """Run trainers up to max_concurrent; one cycle-depth eval may run beside them."""
    from open_score.utils.logging import read_latest
    from open_score.eval.report import refresh_report, refresh_report_async
    from open_score.eval.protocol import CYCLE_SERIES_METHODS, depth_eval_finished, infer_best_t_env
    if max_concurrent < 1 or max_concurrent > 4:
        raise ValueError("Concurrent experiment tasks must be between 1 and 4")
    pending_train, pending_eval = [], []
    live, finished, roster = [], [], []
    eval_ids = set()
    occupied = _occupied_train_jobs()
    original = [(method, dict(options)) for method, options in jobs]
    options_by_key = {(method, options["run"], options["seed"]): options for method, options in original if options}
    lookup = {(method, options["run"], options["seed"]) for method, options in original
              if options and method not in ("anchors", "depth_eval")}
    print(f"scheduler {len(original)} listed jobs", flush=True)
    latest = read_latest(output, "progress", keys=lookup) if lookup else {}

    def roster_key(method, options):
        if method == "anchors" or not options:
            return ("anchors", None, None, None)
        if method == "depth_eval":
            return ("depth_eval", options["run"], options["seed"], options["target_method"])
        return (method, options["run"], options["seed"], None)

    def add_roster(method, options):
        key = roster_key(method, options)
        if key not in roster:
            roster.append(key)

    def queue_depth(method, options, bucket):
        if not with_depth_eval:
            return
        if method not in CYCLE_SERIES_METHODS:
            return
        ident = (method, options["run"], options["seed"])
        if ident in eval_ids:
            return
        directory = Path(output) / method / options["run"] / f"seed_{options['seed']}"
        if not (directory / "best.pt").exists():
            return
        t_env = infer_best_t_env(output, method, options["run"], options["seed"], options.get("t_max"))
        if t_env is None:
            return
        if depth_eval_finished(output, method, options["run"], options["seed"], t_env):
            print(f"{method} [{options['run']} seed={options['seed']}] depth_eval already done", flush=True)
            return
        eval_ids.add(ident)
        payload = {**options, "target_method": method, "job": "depth_eval"}
        bucket.append(("depth_eval", payload))
        add_roster("depth_eval", payload)

    for method, options in original:
        directory = Path(output) / method / options["run"] / f"seed_{options['seed']}"
        row = latest.get((method, options["run"], options["seed"]), {})
        if _training_finished(directory, row) or (
                row.get("status") in ("completed", "complete")
                and any((directory / name).exists() for name in ("final.pt", "best.pt"))):
            print(f"{method} [{options['run']} seed={options['seed']}] already done", flush=True)
            queue_depth(method, options, pending_eval)
            continue
        key = (method, options["run"], options["seed"])
        if key in occupied:
            print(f"{method} [{options['run']} seed={options['seed']}] already running; skip duplicate", flush=True)
            continue
        if options.get("resume"):
            saved_config = directory / "config.json"
            if saved_config.exists():
                saved = json.loads(saved_config.read_text(encoding="utf-8"))
                for field in ("seed", "env", "batch_size_run", "t_max"):
                    if saved[field] != options[field]:
                        raise ValueError(f"Resume configuration differs: {method}/{field}")
            if not any(path.exists() for path in resume_files(directory)):
                if saved_config.exists():
                    raise RuntimeError(f"Recorded run has no recoverable checkpoint: {directory}")
                options["resume"] = False
        pending_train.append((method, options))
        add_roster(method, options)
    if with_anchors:
        pending_train.append(("anchors", None))
        add_roster("anchors", None)
    if not pending_train and not pending_eval:
        return True
    context = get_context("spawn")
    stop_event, queue = context.Event(), context.Queue()
    waiting_train, waiting_eval = deque(pending_train), deque(pending_eval)
    use_ansi = _enable_ansi()
    status_height = 0
    previous = signal.getsignal(signal.SIGINT)
    eval_live_status = {}
    pool = [len(pending_train) + len(pending_eval)]

    def request_stop(signum, frame):
        nonlocal status_height
        stop_event.set()
        status_height = 0
        print("stop requested: finish batch and save resume", flush=True)

    def train_live():
        return sum(1 for item in live if item["method"] != "depth_eval")

    def eval_live():
        return sum(1 for item in live if item["method"] == "depth_eval")

    def start_train_job(method, options):
        if method == "anchors":
            child = context.Process(target=_anchor_entry, args=(str(output), stop_event, queue))
            identity = ("anchors", None, None, None)
        else:
            child = context.Process(target=_job_entry, args=(method, options, stop_event, queue))
            identity = (method, options["run"], options["seed"], None)
        child.start()
        live.append({"child": child, "method": method, "options": options, "key": identity})

    def launch():
        while waiting_train and train_live() < max_concurrent and not stop_event.is_set():
            picked = None
            for index, (method, options) in enumerate(waiting_train):
                if method == "anchors" or _fits_gpu(method, live):
                    picked = index
                    break
            if picked is None:
                break
            method, options = waiting_train[picked]
            del waiting_train[picked]
            start_train_job(method, options)
            group = _job_group(method)
            if GROUP_MAX_LIVE.get(group, 1) < 2 or train_live() >= max_concurrent:
                continue
            for index, (other, other_options) in enumerate(waiting_train):
                if _job_group(other) == group and _fits_gpu(other, live):
                    del waiting_train[index]
                    start_train_job(other, other_options)
                    break
        while waiting_eval and eval_live() < 1 and not stop_event.is_set():
            method, options = waiting_eval.popleft()
            child = context.Process(target=_depth_eval_entry, args=(options, stop_event, queue))
            identity = ("depth_eval", options["run"], options["seed"], options["target_method"])
            child.start()
            live.append({"child": child, "method": "depth_eval", "options": options, "key": identity})
            print(f"{options['target_method']} [{options['run']} seed={options['seed']}] "
                  f"depth_eval started (1:1 R=1..6, beside training)", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    try:
        launch()
        runs = {options["run"] for _, options in original if options}
        last_status, last_report = time.monotonic() - 10, time.monotonic()
        results, failures, abnormal_exits = [], [], set()

        def progress_key(key):
            if key[0] == "depth_eval":
                return (key[3], key[1], key[2])
            return (key[0], key[1], key[2])

        def display_name(key):
            if key[0] == "depth_eval":
                return f"{_METHOD_ALIAS.get(key[3], key[3])} R"
            return _status_name(key[0], key[2])

        def accept_result(result):
            nonlocal last_report, status_height
            if result.get("kind") == "progress":
                ident = (result.get("method"), result.get("run"), result.get("seed"))
                eval_live_status[ident] = result
                return
            results.append(result)
            if result.get("kind") == "depth_eval":
                if result.get("status") == "failed":
                    status_height = 0
                    print(f"depth_eval {result.get('method')} failed:\n{result.get('error', '')}\n"
                          "training continues", flush=True)
                return
            if result.get("status") == "failed":
                failures.append(result)
                stop_event.set()
                status_height = 0
                print(f"{result.get('method', 'worker')} failed:\n{result['error']}\n"
                      "other jobs will finish the current batch and save", flush=True)
                last_report = 0.0
                return
            if result.get("status") == "completed":
                opts = options_by_key.get((result.get("method"), result.get("run"), result.get("seed")))
                if opts:
                    before = len(eval_ids)
                    queue_depth(result["method"], opts, waiting_eval)
                    if len(eval_ids) > before:
                        pool[0] += 1

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
                    failure = dict(status="failed", kind=item["method"],
                                   error=f"Worker {child.pid} exit code {child.exitcode}")
                    if item["method"] == "depth_eval":
                        status_height = 0
                        print(f"{failure['error']}; training continues", flush=True)
                    else:
                        failures.append(failure)
                        stop_event.set()
                        status_height = 0
                        print(f"{failure['error']}; other jobs asked to save and stop", flush=True)
            live[:] = still
            return progressed

        while live or ((waiting_train or waiting_eval) and not stop_event.is_set()):
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
            if now - last_status >= 3:
                live_items = [item for item in live if item["method"] != "anchors"]
                used, total = _gpu_used_gib()
                gpu_bit = f"  gpu={used:.1f}/{total:.1f}G" if used is not None else ""
                run_bit = f"run={next(iter(runs))}  " if len(runs) == 1 else ""
                pack = Counter(_job_group(item["method"]) for item in live_items)
                pack_bit = "  ".join(f"{group}×{count}/{GROUP_MAX_LIVE.get(group, 2)}"
                                     for group, count in pack.items()) if pack else "idle"
                lines = [f"{run_bit}live={train_live()}  {pack_bit}  queue={len(waiting_train)}{gpu_bit}"]
                if live_items:
                    width = max(len(display_name(item["key"])) for item in live_items)
                    for item in live_items:
                        key = item["key"]
                        row = _live_row_from_console(output, item)
                        if key[0] == "depth_eval":
                            row["job"] = "depth_eval"
                            row["target_method"] = key[3]
                            row.update({field: value for field, value
                                        in eval_live_status.get(progress_key(key), {}).items()
                                        if value is not None})
                        row.setdefault("status", "starting")
                        lines.append(_status_line(key[0], row, width, seed=key[2]))
                if waiting_train:
                    names = []
                    for method, options in list(waiting_train)[:8]:
                        names.append(_status_name(method, None if options is None else options.get("seed")))
                    extra = f" +{len(waiting_train) - 8}" if len(waiting_train) > 8 else ""
                    lines.append("queued: " + ", ".join(names) + extra)
                status_height = _rewrite_block(lines, status_height, use_ansi)
                last_status = now
            if now - last_report >= (60 if eval_live() else 300):
                refresh_report_async(output, run=report_run)
                last_report = now
            try:
                accept_result(queue.get(timeout=1.0))
                while True:
                    try:
                        accept_result(queue.get_nowait())
                    except Empty:
                        break
            except Empty:
                pass
        while True:
            try:
                accept_result(queue.get_nowait())
            except Empty:
                break
        failures.extend(dict(status="failed", error=f"Worker {item['child'].pid} exit code {item['child'].exitcode}")
                        for item in finished if item["child"].exitcode and item["child"].pid not in abnormal_exits
                        and item["method"] != "depth_eval")
        try:
            refresh_report(output, run=report_run)
        except (MemoryError, TimeoutError, OSError) as error:
            print(f"report refresh skipped: {type(error).__name__}", flush=True)
        if failures:
            raise RuntimeError("\n".join(item["error"] for item in failures))
        expected = len(finished) if stop_event.is_set() else pool[0]
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
    result.add_argument("--stage", choices=("validate", "e0", "benchmark", "train", "single", "stop", "status"), required=True)
    result.add_argument("--method", choices=POLICY_METHODS)
    result.add_argument("--group", choices=tuple(METHOD_GROUPS), default="main",
                        help="train pool: main=inventory of pending main jobs; "
                             "baseline=B0/B2/REFIL; alma=ALMA; v4/v5=archive groups")
    result.add_argument("--max-concurrent", type=int, choices=(1, 2, 3, 4), default=MAX_CONCURRENT,
                        help="live trainers in this train.py; one 1:1 cycle-depth eval may run beside them")
    result.add_argument("--steps", type=int)
    result.add_argument("--batch-size-run", type=int, choices=(4, 8), default=4)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--cpu", action="store_true")
    result.add_argument("--env", choices=("had", "ff", "rel_overgen"), default="had")
    result.add_argument("--reward-mode", choices=("damage", "friendly"), default="damage",
                        help="damage: learn -ΔD only. friendly: also subtract --friendly-penalty "
                             "per Red-Red collision pair and per extra Red self-destruct beyond "
                             "the Blues those shots hit. Reported D is always physical.")
    result.add_argument("--friendly-penalty", type=float, default=1.0,
                        help="Extra cost per collision pair or surplus intercept suicide "
                             "when --reward-mode friendly")
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
    probe_job = options.group == "alma_probe" or options.method in PROBE_METHODS
    if probe_job and output == DEFAULT_OUTPUT.resolve():
        output = PROBE_OUTPUT.resolve()
    if options.stage == "stop":
        output.mkdir(parents=True, exist_ok=True)
        (output / "stop.request").write_text("Stop after the current complete sampling/learning batch.\n", encoding="utf-8")
        print("stop.request written; wait for stopped + resume, do not kill", flush=True)
        return
    if options.stage == "status":
        run_status_board(output)
        return
    if options.stage in ("e0", "benchmark", "train", "single"):
        (output / "stop.request").unlink(missing_ok=True)
    base = dict(output=str(output), seed=options.seed, batch_size_run=options.batch_size_run,
                use_cuda=not options.cpu, resume=options.resume,
                reward_mode=options.reward_mode, friendly_penalty=options.friendly_penalty)
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
        methods = METHOD_GROUPS[options.group]
        if options.group == "main":
            from open_score.eval.inventory import pending_train_jobs
            print("main train: scan inventory, then run pending jobs", flush=True)
            pending = pending_train_jobs(output)
            jobs = [(task["method"], {**base, "seed": task["seed"], "t_max": options.steps,
                                      "env": "had", "run": FORMAL_RUN, "skip_final_eval": True})
                    for task in pending]
            print(f"launching {len(jobs)} train jobs, max concurrent {options.max_concurrent}", flush=True)
            _print_train_plan(jobs)
            if not jobs:
                print("no pending main training jobs", flush=True)
                return
            run_group(jobs, output, max_concurrent=options.max_concurrent, with_depth_eval=False)
            return
        if options.group in ("alma_probe", "v4", "v5"):
            seeds = (0,) if options.group in ("v4", "v5") else (options.seed,)
            jobs = [(method, {**base, "seed": seed, "t_max": options.steps, "env": "had", "run": FORMAL_RUN})
                    for seed in seeds for method in methods]
            run_group(jobs, output, max_concurrent=options.max_concurrent)
            return
        require_preparation(output)
        # Default group is DCG / SPECTra. GNN seed 0 is already done; later
        # GNN seeds are not queued. B0 / B2 / REFIL stay in baseline.
        jobs = [(method, {**base, "seed": seed, "t_max": options.steps, "env": "had", "run": FORMAL_RUN})
                for seed in FORMAL_SEEDS for method in methods]
        run_group(jobs, output, max_concurrent=options.max_concurrent)
    else:
        if options.method is None or options.steps is None:
            raise SystemExit("single requires --method and --steps")
        # A native sanity environment must never default into the HAD protocol.
        run = options.run or (FORMAL_RUN if options.env == "had" else f"single_{options.env}")
        if "--" in str(run):
            raise SystemExit(f"--run {run!r} looks like glued flags; write --run train --resume with a space")
        run_group([(options.method, {**base, "t_max": options.steps, "env": options.env, "run": run})], output,
                  report_run=run, max_concurrent=options.max_concurrent)


if __name__ == "__main__":
    main()
