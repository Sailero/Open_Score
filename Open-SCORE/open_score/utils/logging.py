"""Shared CSV records for one experiment; safe for concurrent training jobs."""
from __future__ import annotations

from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import threading
import time

VERSION = "main_v5"
# The formal training run of this version. v2 called it "stage3" after a
# staged plan that no longer exists; the anchors it produced are policy- and
# version-independent and are carried over instead of being re-run.
FORMAL_RUN = "train"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "outputs" / VERSION
V3_OUTPUT = Path(__file__).resolve().parents[2] / "outputs" / "main_v3"
V4_OUTPUT = Path(__file__).resolve().parents[2] / "outputs" / "main_v4"
V5_OUTPUT = Path(__file__).resolve().parents[2] / "outputs" / "main_v5"
IDENTITY = ("version", "run", "method", "seed", "recorded_at")
EPISODE_FIELDS = (
    "train_dist", "t_env", "eval_point", "phase", "config", "episode_seed",
    "blue_upper", "blue_lower", "D", "rho", "ep_len", "terminated_naturally",
    "damage_by_target", "first_damage_step_by_target", "red_left", "blue_left",
    "red_deaths_by_cause", "blue_deaths_by_cause", "action_hist", "action_entropy",
    "noop_frac", "mean_speed", "mean_pairwise_dist", "mean_dist_to_nearest_target",
    "friendly_collisions", "min_pairwise_dist_p05", "q_tot_mean", "q_tot_std", "q_i_mean",
)
SCHEMAS = {
    "episodes": IDENTITY + EPISODE_FIELDS + ("checkpoint",),
    "learning": IDENTITY + ("t_env", "metrics"),
    "progress": IDENTITY + ("data",),
    "verification": IDENTITY + ("data",),
    "benchmarks": IDENTITY + ("data",),
    "trajectories": IDENTITY + ("phase", "config", "episode_seed", "checkpoint", "trajectory"),
}
_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()
csv.field_size_limit(2**30)


def _json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not a JSON record value: {type(value).__name__}")


def _encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                      default=_json_default, allow_nan=False)


@contextmanager
def _locked_file(path, *, create=False):
    """Lock the data file itself; all writers and readers use the same lock."""
    path = Path(path)
    key = str(path.resolve())
    with _LOCKS_GUARD:
        local_lock = _LOCKS.setdefault(key, threading.RLock())
    with local_lock:
        if create:
            path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+b" if create else "r+b") as stream:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                # LK_LOCK can raise errno 36 (deadlock avoided) when DCG,
                # SPECTra and the parent status loop contend for one byte.
                deadline = time.monotonic() + 30
                delay = 0.02
                while True:
                    try:
                        msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
                        break
                    except OSError as error:
                        if getattr(error, "errno", None) not in (11, 13, 36) or time.monotonic() >= deadline:
                            raise
                        time.sleep(delay)
                        delay = min(delay * 2, 0.5)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield stream
            finally:
                stream.seek(0)
                if os.name == "nt":
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def append_records(output, stream_name, rows):
    rows = list(rows)
    if not rows:
        return 0
    columns = SCHEMAS[stream_name]
    payload = io.StringIO(newline="")
    writer = csv.writer(payload, lineterminator="\n")
    for row in rows:
        writer.writerow([_encode(row.get(key)) for key in columns])
    path = Path(output) / f"{stream_name}.csv"
    with _locked_file(path, create=True) as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write((",".join(columns) + "\n").encode("utf-8"))
        else:
            stream.seek(0)
            if stream.readline().decode("utf-8").rstrip("\r\n").split(",") != list(columns):
                raise ValueError(f"CSV schema differs from the frozen schema: {path}")
            stream.seek(-1, os.SEEK_END)
            if stream.read(1) != b"\n":
                raise RuntimeError(f"Incomplete record at {path}; preserve it before resuming")
        stream.seek(0, os.SEEK_END)
        stream.write(payload.getvalue().encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    return len(rows)


def _decode_row(cells, path):
    if None in cells or any(value is None for value in cells.values()):
        raise ValueError(f"Malformed complete CSV row in {path}")
    row = {key: json.loads(value) for key, value in cells.items()}
    for field in ("metrics", "data"):
        if isinstance(row.get(field), dict):
            row.update(row[field])
    return row


def read_records(output, stream_name, *, run=None, method=None, seed=None):
    path = Path(output) / f"{stream_name}.csv"
    if not path.exists():
        return []
    with _locked_file(path) as stream:
        stream.seek(0)
        raw = stream.read()
    # A killed process can leave a partial final line, which is not a result.
    raw = raw[:raw.rfind(b"\n") + 1]
    rows = []
    for cells in csv.DictReader(io.StringIO(raw.decode("utf-8"), newline="")):
        row = _decode_row(cells, path)
        if run is not None and row.get("run") != run:
            continue
        if method is not None and row.get("method") != method:
            continue
        if seed is not None and row.get("seed") != int(seed):
            continue
        rows.append(row)
    return rows


def read_latest(output, stream_name, *, run=None, method=None, seed=None, keys=None):
    """Last row per (method, run, seed). Does not retain history."""
    path = Path(output) / f"{stream_name}.csv"
    if not path.exists():
        return {}
    with _locked_file(path) as stream:
        stream.seek(0)
        raw = stream.read()
    raw = raw[:raw.rfind(b"\n") + 1]
    if not raw:
        return {}
    latest = {}
    for cells in csv.DictReader(io.StringIO(raw.decode("utf-8"), newline="")):
        row = _decode_row(cells, path)
        if run is not None and row.get("run") != run:
            continue
        if method is not None and row.get("method") != method:
            continue
        if seed is not None and row.get("seed") != int(seed):
            continue
        identity = (row.get("method"), row.get("run"), row.get("seed"))
        if keys is not None and identity not in keys:
            continue
        latest[identity] = row
    return latest


def episode_key(row):
    return tuple(row.get(key) for key in ("version", "run", "method", "seed", "phase", "eval_point")) + (
        _encode(row["config"]), int(row["episode_seed"]), row.get("checkpoint"),
    )


def unique_episodes(rows):
    return list({episode_key(row): row for row in rows}.values())


class ExperimentLogger:
    def __init__(self, output=DEFAULT_OUTPUT, method="shared", seed=0, run=FORMAL_RUN):
        self.output, self.method, self.seed, self.run = Path(output), str(method), int(seed), str(run)
        self._trajectory_keys = None

    def _row(self, data):
        row = dict(version=VERSION, run=self.run, method=self.method, seed=self.seed,
                   recorded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        row.update(data)
        return row

    def episodes(self, rows):
        encoded = []
        for value in rows:
            row = dict(value)
            row.setdefault("train_dist", "mixed_le10")
            row.setdefault("blue_upper", "reactive")
            row.setdefault("blue_lower", "rush")
            for field in ("q_tot_mean", "q_tot_std", "q_i_mean"):
                row.setdefault(field, None)
            missing = set(EPISODE_FIELDS) - row.keys()
            if missing:
                raise ValueError(f"Missing episode fields: {sorted(missing)}")
            if row["phase"] not in ("train_eval", "final_eval", "depth_eval", "anchor"):
                raise ValueError(f"Not a frozen evaluation phase: {row['phase']}")
            for field in ("D", "rho", "ep_len"):
                if row[field] is None or not math.isfinite(float(row[field])):
                    raise ValueError(f"Invalid applicable episode field {field}: {row[field]}")
            metadata = {key: row.pop(key) for key in IDENTITY if key in row}
            encoded.append(dict(self._row(row), **metadata))
        return append_records(self.output, "episodes", encoded)

    def learning(self, t_env, metrics):
        return append_records(self.output, "learning", [self._row(dict(t_env=int(t_env), metrics=dict(metrics)))])

    def progress(self, **fields):
        return append_records(self.output, "progress", [self._row(dict(data=fields))])

    def verification(self, rows):
        return append_records(self.output, "verification", [self._row(dict(data=dict(row))) for row in rows])

    def benchmarks(self, rows):
        return append_records(self.output, "benchmarks", [self._row(dict(data=dict(row))) for row in rows])

    def trajectories(self, rows):
        def key(row):
            return tuple(row.get(field) for field in ("run", "method", "seed", "phase", "episode_seed", "checkpoint")) + (_encode(row["config"]),)
        if self._trajectory_keys is None:
            self._trajectory_keys = {key(row) for row in read_records(self.output, "trajectories", run=self.run,
                                                                     method=self.method, seed=self.seed)}
        pending = []
        for row in rows:
            value = self._row(dict(row))
            if key(value) not in self._trajectory_keys:
                pending.append(value)
        written = append_records(self.output, "trajectories", pending)
        self._trajectory_keys.update(key(row) for row in pending)
        return written
