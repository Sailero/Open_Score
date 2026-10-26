"""Paths and process ownership for the current HAD experiment."""
from __future__ import annotations
from contextlib import contextmanager
import json
import os
from pathlib import Path
import uuid

PROFILE = "main"

EXPECTED_ENV = {"had_core_version": "had-workbench-2.1.0",
                "physics_protocol": "rebuild-calibrated-v3-r7-target-initialization"}

def check_environment():
    try:
        from had_env.core.version import CORE_VERSION, PHYSICS_PROTOCOL
    except ImportError as error:
        raise RuntimeError("Install the retained had-env version exposing had_env.core.version before running; help/status/plot remain available") from error
    actual = dict(had_core_version=CORE_VERSION, physics_protocol=PHYSICS_PROTOCOL)
    if actual != EXPECTED_ENV:
        raise ValueError(f"Environment mismatch: expected {EXPECTED_ENV}, found {actual}; no sampling started")


def is_profile(output):
    root = Path(output)
    return root.name == "had" or (root / "had").is_dir()

def environment_directory(output, env="had"):
    if env != "had":
        raise ValueError("The active experiment supports HAD only")
    root = Path(output)
    return root if root.name == "had" else root / "had"

def ensure_layout(output):
    root = Path(output)
    directory = environment_directory(root)
    if root != directory and not directory.exists() and any((root / f"{stream}.csv").is_file() for stream in ("episodes", "learning", "progress", "trajectories")):
        raise RuntimeError("Legacy flat CSV output must be migrated explicitly; existing records are preserved")
    directory.mkdir(parents=True, exist_ok=True)
    return directory

def run_directory(output, method, seed, env="had", run="train"):
    return environment_directory(output, env) / str(method) / "models" / f"seed_{int(seed)}"

def new_artifact_id():
    return uuid.uuid4().hex

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.pending")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return default

@contextmanager
def process_lock(path):
    """One scheduler/evaluator owns this file until its process exits."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    locked = False
    try:
        if os.name == "nt":
            import msvcrt
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                locked = True
            except OSError:
                pass
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
            except OSError:
                pass
        yield locked
    finally:
        if locked:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()

def logical_devices(value):
    devices = tuple(int(v.strip()) for v in str(value).split(",") if v.strip())
    if not devices or len(set(devices)) != len(devices) or min(devices) < 0:
        raise ValueError("Devices must be distinct non-negative logical CUDA indices")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and max(devices) >= len(visible.split(",")):
        raise ValueError("Device index is outside inherited CUDA_VISIBLE_DEVICES")
    return devices

def physical_device(logical):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    return visible.split(",")[int(logical)].strip() if visible else str(logical)
