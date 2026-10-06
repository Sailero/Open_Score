"""Shared admission for the main09xx profiles on the user's two shared GPUs."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import time


GPU_PER_CARD_MAX = 4


def _resources(output):
    try:
        return json.loads(Path(output, "experiment.json").read_text(encoding="utf-8"))["resources"]
    except (OSError, ValueError, TypeError, KeyError):
        return {}


def experiment_device_caps(output, env):
    """Configured per-card maxima. GPU0 defaults to 3; GPU1 may go to 4."""
    key = "had_gpu_per_card_by_device" if env == "had" else "smac_gpu_per_card_by_device"
    raw = _resources(output).get(key)
    if not raw:
        return None
    try:
        return {int(gpu): int(cap) for gpu, cap in raw.items()}
    except (TypeError, ValueError):
        return None


def gpu0_shared_cap(output):
    """HAD + SMAC together on GPU0, regardless of which scheduler owns the job."""
    try:
        cap = int(_resources(output).get("gpu0_total_trainers", 3))
    except (TypeError, ValueError):
        cap = 3
    return max(1, min(GPU_PER_CARD_MAX, cap))


def train_scheduler_rows(output, env):
    """All train-scheduler shards for one env (legacy file plus per-device parents)."""
    root = Path(output)
    paths = [root / f"scheduler.train.{env}.json"]
    paths.extend(sorted(root.glob(f"scheduler.train.{env}.d*.json")))
    rows = []
    seen = set()
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def merge_train_scheduler(output, env):
    """Combine live trainers from every parent that is packing this environment."""
    rows = train_scheduler_rows(output, env)
    if not rows:
        return {}
    live, waiting, devices, by_device = {}, {}, [], {}
    for row in rows:
        for item in row.get("live") or []:
            try:
                pid = item.get("pid")
                if pid and not Path(f"/proc/{int(pid)}").exists():
                    continue
            except (TypeError, ValueError, OSError):
                continue
            ident = item.get("id") or f"train.{env}.{item.get('method')}.s{item.get('seed')}"
            live[ident] = item
        for item in row.get("waiting") or []:
            ident = item.get("id")
            if ident:
                waiting[ident] = item
        for gpu in row.get("devices") or []:
            if gpu not in devices:
                devices.append(gpu)
        for gpu, cap in (row.get("per_gpu_by_device") or {}).items():
            try:
                by_device[str(int(gpu))] = int(cap)
            except (TypeError, ValueError):
                continue
    newest = max(rows, key=lambda row: float(row.get("updated_at") or 0))
    merged = dict(newest)
    merged["live"] = list(live.values())
    merged["waiting"] = [item for ident, item in waiting.items() if ident not in live]
    merged["devices"] = devices or list(newest.get("devices") or [])
    if by_device:
        merged["per_gpu_by_device"] = by_device
        merged["per_gpu"] = max(by_device.values())
    return merged


def sibling_load(output, env, gpu):
    """Alive trainers from the other environment already sitting on this card."""
    other = "smacv2" if env == "had" else "had"
    row = merge_train_scheduler(output, other)
    if not row:
        return 0
    count = 0
    for item in row.get("live") or []:
        placed = item.get("physical_gpu", item.get("gpu"))
        try:
            if int(placed) != int(gpu):
                continue
        except (TypeError, ValueError):
            continue
        pid = item.get("pid")
        try:
            if pid and Path(f"/proc/{int(pid)}").exists():
                count += 1
        except (TypeError, ValueError, OSError):
            continue
    return count


# One packer for HAD and SMAC: both cards default to 3; GPU1 may climb to 4.
SHARED_GPU_ADAPT = dict(
    min_by_device={0: 1, 1: 2}, default_by_device={0: 3, 1: 3}, max_by_device={0: 3, 1: 4},
    reserve_gib=8.0, peak_multiplier=1.25, scale_down_free_gib=8.0, warmup_seconds=180,
    fallback_peak_gib=16.0, scale_up_step=1, adapt_pause_seconds=90,
)


def adapt_policy(output, env):
    """Min/default/max per card and memory headroom. HAD and SMAC share one policy."""
    policy = json.loads(json.dumps(SHARED_GPU_ADAPT))
    configured = experiment_device_caps(output, env)
    if configured:
        policy["max_by_device"] = {**policy["max_by_device"], **configured}
    raw = (_resources(output).get("gpu_adapt") or {}).get(env) or {}
    for key in ("reserve_gib", "peak_multiplier", "scale_down_free_gib", "fallback_peak_gib"):
        if key in raw:
            policy[key] = float(raw[key])
    for key in ("warmup_seconds", "scale_up_step", "adapt_pause_seconds"):
        if key in raw:
            policy[key] = int(raw[key])
    for key in ("min_by_device", "default_by_device", "max_by_device"):
        if raw.get(key):
            policy[key] = {int(gpu): int(cap) for gpu, cap in raw[key].items()}
    return policy


def _item_gpu(item):
    if item.get("gpu") is not None:
        return int(item["gpu"])
    return int(item.get("options", {}).get("cuda_visible_devices", 0))


def _item_resource(output, env, item):
    from open_score.eval.experiment import run_directory
    row = _resource(run_directory(output, item["method"], item["options"]["seed"], env) / "resource.json")
    if row is None or row.get("pid") != item["child"].pid:
        return None
    return row


def adaptive_caps(output, env, live, devices=(0, 1), memories=None):
    """Choose a live per-card target from free memory, inside the configured band.

    GPU0's setpoint is 3. GPU1 starts at 2 and may climb to 4, one slot at a
    time, after the newest trainer on that card has finished warming. A card
    below the free-memory floor sheds its newest trainer. Occupancy already
    inside the band is held until memory pressure or spare headroom says otherwise.
    experiment.json ``gpu_adapt`` is re-read on every call.
    """
    policy = adapt_policy(output, env)
    devices = tuple(int(gpu) for gpu in devices)
    if memories is None:
        memories = {}
        for gpu in devices:
            try:
                memories[int(gpu)] = gpu_memory(gpu)
            except (OSError, ValueError, subprocess.SubprocessError):
                continue
    now = time.time()
    caps = {}
    for gpu in devices:
        lo = max(1, int(policy["min_by_device"].get(gpu, 1)))
        default = max(lo, int(policy["default_by_device"].get(gpu, lo)))
        hi = min(GPU_PER_CARD_MAX, max(default, int(policy["max_by_device"].get(gpu, default))))
        same = [item for item in live if _item_gpu(item) == gpu]
        load = len(same)
        if gpu == 0:
            room = gpu0_shared_cap(output) - sibling_load(output, env, gpu)
            if room <= 0:
                caps[gpu] = 0 if load == 0 else load
                continue
            hi = min(hi, room)
            default = min(default, hi)
            lo = min(lo, hi)
        memory = memories.get(gpu)
        if memory is None:
            caps[gpu] = min(hi, load or default)
            continue
        free = float(memory.get("free") or 0)
        if free < policy["scale_down_free_gib"] and load > lo:
            caps[gpu] = load - 1
            continue
        warming = any(now - float(item.get("started_at") or now) < policy["warmup_seconds"]
                      for item in same)
        if warming:
            floor = default if load < default else load
            caps[gpu] = min(hi, max(lo, floor))
            continue
        if load == 0:
            caps[gpu] = min(hi, default)
            continue
        # Extra slots are for the next job, not a clone of the occupant.
        pack_peak = float(policy.get("pack_peak_gib") or 8.0)
        need = policy["reserve_gib"] + policy["peak_multiplier"] * pack_peak
        extra = int(free // need) if need > 0 else 0
        cap = min(hi, max(lo, load))
        if extra >= 1 and cap < hi:
            cap = min(hi, load + max(1, int(policy.get("scale_up_step", 1))))
        caps[gpu] = cap
    return caps


def adapt_overflow(live, caps):
    """Newest trainers on an over-capacity card, so scale-down can pause them."""
    victims = []
    for gpu, cap in caps.items():
        same = [item for item in live if _item_gpu(item) == gpu]
        same.sort(key=lambda item: float(item.get("started_at") or 0), reverse=True)
        victims.extend(same[: max(0, len(same) - int(cap))])
    return victims


def resolve_per_gpu(output, env, per_gpu, live=None, memories=None, devices=(0, 1)):
    """Live adaptive caps when the trainer pool is known, otherwise configured maxima."""
    if live is not None:
        return adaptive_caps(output, env, live, devices=devices, memories=memories)
    return experiment_device_caps(output, env) or per_gpu


def per_gpu_caps(per_gpu, devices=(0, 1), *, env="had"):
    """Per-card trainer caps. `per_gpu` may be one integer or {gpu: cap}."""
    devices = tuple(int(gpu) for gpu in devices)
    if not devices or len(set(devices)) != len(devices) or not set(devices) <= {0, 1}:
        raise ValueError("requires distinct physical GPUs from 0,1")
    hard = GPU_PER_CARD_MAX
    if isinstance(per_gpu, dict):
        raw = {int(key): int(value) for key, value in per_gpu.items()}
        caps = {gpu: raw.get(gpu, hard) for gpu in devices}
    else:
        caps = {gpu: int(per_gpu) for gpu in devices}
    if any(caps[gpu] not in range(0, hard + 1) for gpu in devices):
        raise ValueError(f"allows 0..{hard} trainers per physical GPU")
    return caps


def gpu_train_limit(devices=(0, 1), *, per_gpu=GPU_PER_CARD_MAX, env="had"):
    """Derive the queue limit from the explicitly permitted physical cards."""
    return sum(per_gpu_caps(per_gpu, devices, env=env).values())


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
    """Inspect one physical device. Official main0923 schedulers still only pass 0/1."""
    if int(gpu) < 0:
        raise ValueError("physical GPU id must be >= 0")
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
    may admit the configured trainers per GPU; another command waits outside that lease.
    CPU evaluation has one parent per experiment output.
    """
    import fcntl
    from open_score.eval.experiment import PROFILE
    workspace = Path(__file__).resolve().parents[5]
    base, _, env = kind.partition(".")
    suffix = f".{env}" if env else ""
    path = (workspace / f".cache/{PROFILE}.gpus{suffix}.lock"
            if base in ("gpu", "gpu0") else Path(output) / f".cpu-eval{suffix}.lock")
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


def _bootstrap_peak(output, env, method):
    """Use accepted full-episode measurements only to try an idle card.

    Sequence scaling is a provisional estimate, not a worst-case memory
    guarantee. Real process measurements and transactional OOM recovery remain
    authoritative; no synthetic per-run resource record is written.
    """
    try:
        import math
        accepted = json.loads((Path(output) / "implementation_acceptance.json").read_text())["matrix_smoke_runtime"]
        if accepted.get("status") != "passed":
            return 0.
        workers, limit = (8, 100) if env == "had" else (4, 200)
        batch_size = 128 if env == "smacv2" and method == "spectra" else 32
        rows = accepted["had_full_gpu_throughput" if env == "had" else "smac_full_gpu_throughput"]
        estimates = []
        for row in rows:
            if (row.get("method") == method and row.get("workers") == workers
                    and row.get("batch_size") == batch_size and row.get("episode_limit") == limit
                    and row.get("finite_td") is True and 1 < row.get("sequence_length", 0) <= limit + 1):
                peak = float(row["peak_reserved_gib"])
                if math.isfinite(peak) and peak > 0:
                    estimates.append(peak * (limit + 1) / row["sequence_length"])
        return max(estimates, default=0.)
    except (OSError, ValueError, KeyError, TypeError):
        return 0.


def _source_peak(output, env, method):
    """Measured peak of the same architecture in the source experiment (the SG switch does not
    change memory). Intent variants have no source measurement and start as unknown."""
    from open_score.eval.experiment import run_directory, SOURCE_PROFILE, ALL_SEEDS, INTENT_METHODS
    if method in INTENT_METHODS:
        return 0.
    source = Path(output).parent / SOURCE_PROFILE
    base = method[:-3] if method.endswith("_sg") else method
    # Relation variants share one module size; main0921 measured only its four new methods.
    family = (base, "regir_kv0", "regir_fixed4", "regir_untied4") if base.startswith("regir") else (base,)
    for name in family:
        rows = [_resource(run_directory(source, name, s, env) / "resource.json") for s in ALL_SEEDS]
        peak = max((float(r.get("cuda_peak_reserved_gib") or 0) for r in rows if r), default=0.)
        if peak:
            return peak
    return 0.


def gpu_admit(output, env, method, live, *, gpu=0, memory=None, minimum_peak=0., recovering=False,
              per_gpu=GPU_PER_CARD_MAX):
    from open_score.eval.experiment import run_directory, ALL_SEEDS
    try:
        caps = per_gpu_caps(per_gpu, env=env)
    except ValueError:
        return False
    if int(gpu) not in (0, 1) or len(live) >= gpu_train_limit(devices=(int(gpu),), per_gpu=per_gpu, env=env):
        return False
    same_gpu = [item for item in live if int(item["options"].get("cuda_visible_devices", "0")) == int(gpu)]
    occupied = len(same_gpu) + (sibling_load(output, env, gpu) if int(gpu) == 0 else 0)
    if int(gpu) == 0 and occupied >= gpu0_shared_cap(output):
        return False
    measurements = [_resource(run_directory(output, method, s, env) / "resource.json") for s in ALL_SEEDS]
    measurements = [r for r in measurements if r]
    peak = max(float(minimum_peak), max((float(r.get("cuda_peak_reserved_gib") or r.get("cuda_reserved_gib") or 0)
                for r in measurements), default=0.0))
    if not peak:
        peak = _source_peak(output, env, method)
    if len(same_gpu) >= caps[int(gpu)]:
        return False
    if recovering and same_gpu:
        return False
    if any(item["options"].get("_recovering") for item in same_gpu):
        return False
    try:
        observed_memory = memory if memory is not None else gpu_memory(gpu)
        free = observed_memory["free"]
        if (env == "smacv2" and live and observed_memory.get("utilization", 100) >= 90
                and observed_memory.get("free", 0) < 24.0):
            return False
    except (OSError, ValueError, subprocess.SubprocessError):
        return False  # Unknown free memory is not permission to launch.
    if not peak:
        bootstrap = _bootstrap_peak(output, env, method)
        if bootstrap:
            need = 8.0 + 1.25 * bootstrap
            if occupied == 0:
                return free >= need
            # Unmeasured SMAC can share a busy card only when remaining free
            # memory is clearly above the scaled smoke peak. GPU0 currently
            # hosts a 50+ GiB Ollama process and will fail this check.
            return env == "smacv2" and free >= need + 16.0
        return occupied == 0 and free >= 9.0  # First real run is measured alone, with >=8 GiB reserve.
    # nvidia-smi free already subtracts each live process's current reservation.
    # Extra headroom is only for the incoming job (peak × multiplier + reserve)
    # and for memory a live process has previously reached but later released.
    reserve, multiplier = 8.0, 1.25
    key = "had_gpu_admission" if env == "had" else "smac_gpu_admission"
    try:
        policy = _resources(output).get(key) or {}
        if policy:
            reserve = float(policy.get("reserve_gib", reserve))
            multiplier = float(policy.get("peak_multiplier", multiplier))
        if env == "had" and not (6.0 <= reserve <= 32.0 and 1.10 <= multiplier <= 2.0):
            return False
    except (OSError, ValueError, TypeError, KeyError):
        if env == "had":
            return False
    growth = 0.0
    for item in live:
        options = item["options"]
        if int(options.get("cuda_visible_devices", "0")) != int(gpu):
            continue
        row = _resource(run_directory(output, item["method"], options["seed"], env) / "resource.json")
        if row is None or row.get("pid") != item["child"].pid:
            growth += 8.0
            continue
        observed = float(row.get("cuda_peak_reserved_gib") or row.get("cuda_reserved_gib") or 0)
        current = float(row.get("cuda_reserved_gib") or observed)
        growth += max(0.0, observed - current)
    return free >= reserve + multiplier * peak + growth


def choose_gpu(output, env, method, live, devices=(0, 1), *, memories=None,
               minimum_peak=0., recovering=False, per_gpu=GPU_PER_CARD_MAX):
    """Choose the eligible card with most free memory, without a fixed priority."""
    devices = tuple(int(gpu) for gpu in devices)
    if len(live) >= gpu_train_limit(devices, per_gpu=per_gpu, env=env):
        return None
    if memories is None:
        memories = {}
        for gpu in devices:
            try:
                memories[int(gpu)] = gpu_memory(gpu)
            except (OSError, ValueError, subprocess.SubprocessError):
                continue
    memories = {int(gpu): row for gpu, row in memories.items() if int(gpu) in devices}
    loads = {gpu: sum(int(item["options"].get("cuda_visible_devices", "0")) == gpu for item in live)
             for gpu in memories}
    def preference(gpu):
        if env == "smacv2":
            return (loads[gpu] == 0, -memories[gpu].get("utilization", 100), memories[gpu]["free"])
        return (loads[gpu] == 0, memories[gpu]["free"])
    for gpu in sorted(memories, key=preference, reverse=True):
        if gpu_admit(output, env, method, live, gpu=gpu, memory=memories[gpu],
                     minimum_peak=minimum_peak, recovering=recovering, per_gpu=per_gpu):
            return gpu
    return None


def is_cuda_oom(error):
    message = str(error).lower()
    return (("out of memory" in message and ("cuda" in message or type(error).__name__ == "OutOfMemoryError"))
            or "cublas_status_alloc_failed" in message)


def cpu_admit(env, kind, live, maximum):
    import psutil
    if len(live) >= maximum:
        return False
    kinds = [item["task"]["kind"] for item in live]
    diagnostic = {"coverage", "dynamics", "deep_rounds", "global_probe",
                  "readout_attention", "intent_accuracy"}
    if kind in diagnostic:
        if sum(k in diagnostic for k in kinds) >= 8:
            return False
    elif kind != "final" and sum(k != "final" and k not in diagnostic for k in kinds) >= 4:
        return False
    # Reserve host memory for training and account for just-spawned evaluators
    # that have not loaded their model/environment yet. No synthetic benchmark.
    warming = sum(time.monotonic() - item.get("started_at", 0) < 30 for item in live)
    reserve_gib = 16 + warming * (4 if env == "smacv2" else 1.5)
    return psutil.virtual_memory().available >= reserve_gib * 1024**3
