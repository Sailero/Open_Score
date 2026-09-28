"""Shared-storage cluster: GPU discovery, mkdir claims, heartbeats, per-host queues.

Every machine mounts the same `outputs/main0928`. Claims live under
`cluster/claims/{task_id}/` (atomic `os.mkdir` on NFS). A worker file
`cluster/workers/{host}.json` is rewritten every 15 s; a claim whose host
has been silent for 120 s is released so another machine can resume from
`resume.pt`. Train/eval schedulers write per-host queue and scheduler
files so two machines never clobber each other's live lists.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import uuid

from . import experiment as X

HEARTBEAT_SECONDS = 15
CLAIM_TTL_SECONDS = 120


def hostname():
    return os.environ.get("REGIR_HOST") or socket.gethostname().split(".")[0]


def repo_root():
    return Path(os.environ.get("REGIR_ROOT") or Path(__file__).resolve().parents[3])


def saileron_root():
    configured = os.environ.get("SAILERON_ROOT")
    if configured:
        return Path(configured)
    return repo_root().parent.parent


def python_bin(smac=False):
    key = "SMAC_PY" if smac else "PY"
    fallback = os.environ.get("PY") or os.environ.get("CONDA_PREFIX", "")
    path = os.environ.get(key) or (f"{fallback}/bin/python" if fallback else "")
    if path and Path(path).exists():
        return path
    return os.environ.get("SMAC_PY" if smac else "PY") or "python"


def sc2_path():
    for candidate in (os.environ.get("SC2PATH"), repo_root() / "envs/StarCraftII",
                      saileron_root() / "envs/StarCraftII"):
        if candidate and Path(candidate).exists():
            return str(Path(candidate))
    return os.environ.get("SC2PATH") or ""


def smac_available():
    root = Path(sc2_path()) if sc2_path() else None
    return bool(root and (root / "Versions").exists() and (root / "Maps").exists())


def discover_gpus():
    """Physical GPU indices on this host (`nvidia-smi -L`). Empty if none/unavailable."""
    try:
        text = subprocess.check_output(["nvidia-smi", "-L"], text=True, timeout=8)
    except (OSError, subprocess.SubprocessError):
        return []
    ids = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("GPU ") and ":" in line:
            try:
                ids.append(int(line.split(":", 1)[0].split()[1]))
            except (IndexError, ValueError):
                continue
    return ids


def cluster_dir(out):
    path = Path(out) / "cluster"
    path.mkdir(parents=True, exist_ok=True)
    (path / "claims").mkdir(exist_ok=True)
    (path / "workers").mkdir(exist_ok=True)
    (path / "locks").mkdir(exist_ok=True)
    return path


def queue_path(out, env, host=None):
    host = host or hostname()
    return Path(out) / f"queue.{env}.{host}.json"


def scheduler_path(out, kind, env, host=None):
    host = host or hostname()
    return Path(out) / f"scheduler.{kind}.{env}.{host}.json"


def worker_path(out, host=None):
    return cluster_dir(out) / "workers" / f"{host or hostname()}.json"


def claim_dir(out, task_id):
    return cluster_dir(out) / "claims" / str(task_id)


def _claim_payload(task_id, host=None):
    return dict(task_id=task_id, host=host or hostname(), pid=os.getpid(),
                heartbeat=time.time(), claimed_at=time.time())


def read_claim(out, task_id):
    path = claim_dir(out, task_id) / "claim.json"
    return X.read_json(path, None)


def claim_age(claim):
    if not isinstance(claim, dict):
        return 1e9
    return time.time() - float(claim.get("heartbeat") or claim.get("claimed_at") or 0)


def is_stale(claim, ttl=CLAIM_TTL_SECONDS):
    if not isinstance(claim, dict):
        return True
    return claim_age(claim) > ttl


def _rmtree(path):
    path = Path(path)
    if not path.exists():
        return
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_file() or child.is_symlink():
            child.unlink(missing_ok=True)
        elif child.is_dir():
            try:
                child.rmdir()
            except OSError:
                pass
    try:
        path.rmdir()
    except OSError:
        pass


def release_claim(out, task_id, *, host=None, force=False):
    """Drop a claim directory. `force` ignores host ownership (stale reap)."""
    directory = claim_dir(out, task_id)
    claim = read_claim(out, task_id)
    if claim and host and not force and claim.get("host") != host:
        return False
    _rmtree(directory)
    return True


def try_claim(out, task_id, *, host=None, payload=None):
    """Atomically mkdir the claim directory. Returns True if this host owns it."""
    host = host or hostname()
    directory = claim_dir(out, task_id)
    directory.parent.mkdir(parents=True, exist_ok=True)
    existing = read_claim(out, task_id)
    if existing and not is_stale(existing):
        return existing.get("host") == host
    if existing and is_stale(existing):
        release_claim(out, task_id, force=True)
    try:
        os.mkdir(directory)
    except FileExistsError:
        claim = read_claim(out, task_id)
        if claim and is_stale(claim):
            release_claim(out, task_id, force=True)
            try:
                os.mkdir(directory)
            except FileExistsError:
                claim = read_claim(out, task_id)
                return bool(claim) and claim.get("host") == host
        else:
            claim = read_claim(out, task_id)
            return bool(claim) and claim.get("host") == host
    data = _claim_payload(task_id, host)
    if payload:
        data.update(payload)
    X.atomic_json(directory / "claim.json", data)
    return True


def refresh_claim(out, task_id, *, host=None):
    host = host or hostname()
    claim = read_claim(out, task_id)
    if not claim or claim.get("host") != host:
        return False
    claim["heartbeat"] = time.time()
    claim["pid"] = os.getpid()
    X.atomic_json(claim_dir(out, task_id) / "claim.json", claim)
    return True


def owned_claims(out, host=None):
    host = host or hostname()
    root = cluster_dir(out) / "claims"
    owned = []
    if not root.exists():
        return owned
    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        claim = X.read_json(directory / "claim.json", None)
        if isinstance(claim, dict) and claim.get("host") == host:
            owned.append(claim)
    return owned


def reap_stale(out, ttl=CLAIM_TTL_SECONDS):
    """Release claims whose host heartbeat is older than `ttl` seconds."""
    workers = {row["host"]: row for row in list_workers(out)}
    released = []
    root = cluster_dir(out) / "claims"
    if not root.exists():
        return released
    now = time.time()
    for directory in list(root.iterdir()):
        if not directory.is_dir():
            continue
        claim = X.read_json(directory / "claim.json", None)
        if not isinstance(claim, dict):
            _rmtree(directory)
            released.append(directory.name)
            continue
        host = claim.get("host")
        worker = workers.get(host) or {}
        worker_age = now - float(worker.get("heartbeat") or 0)
        claim_old = is_stale(claim, ttl)
        worker_old = (not worker) or worker_age > ttl
        if claim_old and worker_old:
            release_claim(out, directory.name, force=True)
            released.append(directory.name)
    return released


def heartbeat(out, *, host=None, extra=None):
    host = host or hostname()
    gpus = discover_gpus()
    payload = dict(host=host, pid=os.getpid(), heartbeat=time.time(),
                   gpus=gpus, n_gpus=len(gpus),
                   tasks=[c.get("task_id") for c in owned_claims(out, host)],
                   smac_available=smac_available())
    if extra:
        payload.update(extra)
    X.atomic_json(worker_path(out, host), payload)
    for claim in owned_claims(out, host):
        refresh_claim(out, claim["task_id"], host=host)
    return payload


def list_workers(out, ttl=CLAIM_TTL_SECONDS):
    root = cluster_dir(out) / "workers"
    rows = []
    if not root.exists():
        return rows
    now = time.time()
    for path in sorted(root.glob("*.json")):
        row = X.read_json(path, None)
        if not isinstance(row, dict):
            continue
        row["age"] = now - float(row.get("heartbeat") or 0)
        row["alive"] = row["age"] <= ttl
        rows.append(row)
    return rows


def live_ids_from_schedulers(out):
    """Union of live task ids published by every host's scheduler files."""
    ids = set()
    for path in Path(out).glob("scheduler.*.json"):
        data = X.read_json(path, {}) or {}
        for row in data.get("live") or []:
            identity = row.get("id")
            if not identity and row.get("method") is not None and row.get("seed") is not None:
                kind = data.get("kind") or path.name.split(".")[1]
                env = data.get("env") or row.get("env")
                identity = f"{kind}.{env}.{row['method']}.s{row['seed']}"
                if kind != "train":
                    identity = f"eval.{row.get('kind') or 'final'}.{env}.{row['method']}.s{row['seed']}"
            if identity:
                ids.add(identity)
    return ids


def all_live_jobs(out):
    jobs = []
    for path in sorted(Path(out).glob("scheduler.*.json")):
        data = X.read_json(path, {}) or {}
        kind = data.get("kind") or (path.name.split(".")[1] if "." in path.name else "train")
        env = data.get("env")
        host = data.get("host")
        for row in data.get("live") or []:
            job = dict(row)
            job.setdefault("kind", kind if kind == "train" else job.get("kind") or "final")
            if env:
                job.setdefault("env", env)
            if host:
                job.setdefault("host", host)
            jobs.append(job)
    jobs.sort(key=lambda r: (r.get("kind") != "train", str(r.get("host") or ""),
                             str(r.get("physical_gpu", "")), r.get("id", "")))
    return jobs


def all_queues(out):
    queues = {}
    for path in Path(out).glob("queue.*.json"):
        queues[path.name] = X.read_json(path, {}) or {}
    return queues


@contextmanager
def nfs_lock(path, *, timeout=30):
    """Directory-based exclusive lock (mkdir is atomic on NFS; flock often is not)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + f".lockdir")
    deadline = time.monotonic() + timeout
    delay = 0.02
    while True:
        try:
            os.mkdir(lock)
            break
        except FileExistsError:
            try:
                age = time.time() - lock.stat().st_mtime
            except OSError:
                age = 0
            if age > CLAIM_TTL_SECONDS:
                _rmtree(lock)
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for {lock}")
            time.sleep(delay)
            delay = min(delay * 2, 0.25)
    token = lock / f"{hostname()}.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    try:
        token.write_text(str(os.getpid()), encoding="utf-8")
        yield lock
    finally:
        token.unlink(missing_ok=True)
        try:
            lock.rmdir()
        except OSError:
            _rmtree(lock)


def default_caps(env, n_gpus):
    """Per-card default/max for 48 GiB 4090-class cards."""
    if env == "smacv2":
        default, hi = 1, 2
    else:
        default, hi = 3, 4
    n_gpus = max(0, int(n_gpus))
    return dict(per_gpu=default, per_gpu_max=hi, slots=max(1, default * n_gpus) if n_gpus else 0)
