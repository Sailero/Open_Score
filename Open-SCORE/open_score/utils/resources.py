"""Shared admission for main0921 on the user's two shared GPUs."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import time


def cpu_threads():
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"


def configure_workspace():
    """Keep this experiment's writable runtime/cache files in the user's workspace."""
    workspace = Path(__file__).resolve().parents[5]
    paths = {"TMPDIR": workspace / "tmp", "XDG_CACHE_HOME": workspace / ".cache",
             "CUDA_CACHE_PATH": workspace / ".cache/cuda",
             "TRITON_CACHE_DIR": workspace / ".cache/triton",
             "TORCHINDUCTOR_CACHE_DIR": workspace / "tmp/torchinductor"}
    for key, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    import sys
    sys.dont_write_bytecode = True
    return workspace


@contextmanager
def _stop_guard(path):
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name("." + path.name + ".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield path


def write_stop(path, message):
    with _stop_guard(path) as target:
        target.write_text(message + "\n", encoding="utf-8")


def clear_previous_stop(path, requested_at):
    """A new start consumes only the stop request that predates that command."""
    with _stop_guard(path) as target:
        if target.exists():
            if target.stat().st_mtime > requested_at:
                raise InterruptedError("A new stop request arrived before this queue started")
            target.unlink()


def gpu_memory(gpu):
    """Only inspect the explicitly permitted physical device."""
    if int(gpu) not in (0, 1):
        raise ValueError("main0921 permits physical GPUs 0 and 1")
    result = subprocess.check_output([
        "nvidia-smi", f"--id={int(gpu)}", "--query-gpu=memory.free,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True, timeout=5)
    values = result.strip().split(",")
    free, used, total = (float(x.strip()) / 1024 for x in values[:3])
    return dict(free=free, used=used, total=total, utilization=float(values[3].strip()))


@contextmanager
def queue_lock(output, kind, *, stop_requested=None):
    """OS-held locks survive stale files and release automatically on exit.

    GPU train/single/farm and cost measurement share one parent lease. A parent
    may admit at most three HAD or two SMAC trainers; another command waits outside that lease.
    CPU evaluation has one parent per experiment output.
    """
    import fcntl
    workspace = Path(__file__).resolve().parents[5]
    path = (workspace / ".cache/main0921.gpus.lock"
            if kind in ("gpu", "gpu0") else Path(output) / ".cpu-eval.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        last_message = 0.0
        while True:
            if stop_requested and stop_requested():
                raise InterruptedError("Queue stopped before acquiring its shared lease")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if stop_requested and stop_requested():
                    raise InterruptedError("Queue stopped while waiting for its shared lease")
                if time.monotonic() - last_message > 30:
                    print(f"waiting for shared {kind} scheduler lease: {path}", flush=True)
                    last_message = time.monotonic()
                time.sleep(1)
        if stop_requested and stop_requested():
            fcntl.flock(handle, fcntl.LOCK_UN)
            raise InterruptedError("Queue stopped while acquiring its shared lease")
        handle.seek(0); handle.truncate()
        json.dump(dict(pid=os.getpid(), output=str(Path(output).resolve()), kind=kind), handle)
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _resource(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if value.get("estimate_ready"):
            return value
    except (OSError, ValueError):
        pass
    return None


def gpu_admit(output, env, method, live, *, gpu=0, memory=None, minimum_peak=0., recovering=False):
    from open_score.eval.experiment import run_directory, SEEDS
    if len(live) >= (3 if env == "had" else 2):
        return False
    measurements = [_resource(run_directory(output, method, s, env) / "resource.json") for s in SEEDS]
    measurements = [r for r in measurements if r]
    peak = max(float(minimum_peak), max((float(r.get("cuda_peak_reserved_gib") or r.get("cuda_reserved_gib") or 0)
                for r in measurements), default=0.0))
    same_gpu = [item for item in live if int(item["options"].get("cuda_visible_devices", "0")) == int(gpu)]
    if len(same_gpu) >= (2 if env == "had" else 1):
        return False
    if recovering and same_gpu:
        return False
    if any(item["options"].get("_recovering") for item in same_gpu):
        return False
    try:
        observed_memory = memory if memory is not None else gpu_memory(gpu)
        free = observed_memory["free"]
        if env == "smacv2" and live and observed_memory.get("utilization", 100) >= 90:
            return False
    except (OSError, ValueError, subprocess.SubprocessError):
        return False  # Unknown free memory is not permission to launch.
    if not peak:
        return not live and free >= 9.0  # First real run is measured alone, with >=8 GiB reserve.
    growth = 0.0
    for item in live:
        options = item["options"]
        row = _resource(run_directory(output, item["method"], options["seed"], env) / "resource.json")
        if row is None or row.get("pid") != item["child"].pid:
            return False
        if int(options.get("cuda_visible_devices", "0")) != int(gpu):
            continue
        observed = float(row.get("cuda_peak_reserved_gib") or row.get("cuda_reserved_gib") or 0)
        current = float(row.get("cuda_reserved_gib") or observed)
        growth += max(0.0, 1.25 * observed - current)
    return free >= 8.0 + 1.25 * peak + growth


def choose_gpu(output, env, method, live, devices=(0, 1), *, memories=None,
               minimum_peak=0., recovering=False):
    """Choose the eligible card with most free memory, without a fixed priority."""
    if memories is None:
        memories = {}
        for gpu in devices:
            try:
                memories[int(gpu)] = gpu_memory(gpu)
            except (OSError, ValueError, subprocess.SubprocessError):
                continue
    loads = {gpu: sum(int(item["options"].get("cuda_visible_devices", "0")) == gpu for item in live)
             for gpu in memories}
    def preference(gpu):
        if env == "smacv2":
            return (loads[gpu] == 0, -memories[gpu].get("utilization", 100), memories[gpu]["free"])
        return (loads[gpu] == 0, memories[gpu]["free"])
    for gpu in sorted(memories, key=preference, reverse=True):
        if gpu_admit(output, env, method, live, gpu=gpu, memory=memories[gpu],
                     minimum_peak=minimum_peak, recovering=recovering):
            return gpu
    return None


def is_cuda_oom(error):
    message = str(error).lower()
    return (("out of memory" in message and ("cuda" in message or type(error).__name__ == "OutOfMemoryError"))
            or "cublas_status_alloc_failed" in message)


def cpu_admit(env, kind, live, maximum):
    import psutil
    if len(live) >= min(4, maximum):
        return False
    kinds = [item["task"]["kind"] for item in live]
    if env == "smacv2":
        return len(live) < min(4, maximum) and psutil.virtual_memory().available >= 12 * 1024**3
    if kind == "final" and kinds.count("final") >= 4:
        return False
    if kind != "final" and sum(k != "final" for k in kinds) >= 4:
        return False
    return psutil.virtual_memory().available >= 8 * 1024**3
