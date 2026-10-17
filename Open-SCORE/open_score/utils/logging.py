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

VERSION = "main"
# Training run and provenance stay in record fields, rather than directory versions.
FORMAL_RUN = "train"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "outputs" / VERSION
IDENTITY = ("version", "run", "method", "seed", "recorded_at")
EPISODE_FIELDS = (
    "train_dist", "t_env", "eval_point", "phase", "config", "episode_seed",
    "blue_upper", "blue_lower", "D", "rho", "ep_len", "terminated_naturally",
    "damage_by_target", "first_damage_step_by_target", "red_left", "blue_left",
    "red_deaths_by_cause", "blue_deaths_by_cause", "action_hist", "action_entropy",
    "noop_frac", "mean_speed", "mean_pairwise_dist", "mean_dist_to_nearest_target",
    "friendly_collisions", "min_pairwise_dist_p05", "q_tot_mean", "q_tot_std", "q_i_mean",
)
EXPERIMENT_EPISODE_FIELDS = ("curriculum_stage", "curriculum_switches", "spawn_configuration",
    "reward_raw_sum", "reward_shaping_sum", "reward_shaped_sum")
SCHEMAS = {
    "episodes": IDENTITY + EPISODE_FIELDS + ("checkpoint",),
    "learning": IDENTITY + ("t_env", "metrics"),
    "progress": IDENTITY + ("data",),
    "trajectories": IDENTITY + ("phase", "config", "episode_seed", "checkpoint", "trajectory"),
}


def schema_for(output, stream_name):
    """All active writers use the complete HAD identity schema."""
    columns = SCHEMAS[stream_name]
    extra = ("env", "checkpoint_id", "arm", "cycle_depth", "readout", "experiment_arm")
    if stream_name == "episodes":
        extra += ("return", "source_version", "protocol", "episode_wall_seconds") + EXPERIMENT_EPISODE_FIELDS
    if stream_name in ("learning", "trajectories", "progress"):
        extra += ("source_version", "protocol")
    return columns + tuple(key for key in extra if key not in columns)
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
    from open_score.eval.experiment import ensure_layout
    if any(row.get("env", "had") != "had" for row in rows):
        raise ValueError("The active record writer supports HAD only")
    path = ensure_layout(output) / f"{stream_name}.csv"
    with _locked_file(path, create=True) as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            columns = list(schema_for(output, stream_name))
            stream.write((",".join(columns) + "\n").encode("utf-8"))
        else:
            stream.seek(0)
            columns = stream.readline().decode("utf-8").rstrip("\r\n").split(",")
            stream.seek(-1, os.SEEK_END)
            if stream.read(1) != b"\n":
                raise RuntimeError(f"Incomplete record at {path}; preserve it before resuming")
            required = {"env"}
            if stream_name == "episodes":
                required.update(("checkpoint_id", "arm", "protocol"))
            missing = required - set(columns)
            missing.update(key for row in rows for key in ("checkpoint_id", "arm", "protocol", "cycle_depth", "readout", "source_version", "experiment_arm") if row.get(key) is not None and key not in columns)
            if any(row.get("experiment_arm") is not None for row in rows) and stream_name == "episodes":
                missing.update(key for row in rows for key in EXPERIMENT_EPISODE_FIELDS if row.get(key) is not None and key not in columns)
            if missing:
                raise RuntimeError(f"Legacy CSV header at {path} lacks identity fields {sorted(missing)}; existing records are preserved")
        payload = io.StringIO(newline="")
        writer = csv.writer(payload, lineterminator="\n")
        for row in rows:
            writer.writerow([_encode(row.get(key)) for key in columns])
        stream.seek(0, os.SEEK_END)
        stream.write(payload.getvalue().encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    return len(rows)


_ARCHIVE_TEXT_FIELDS = frozenset(("archive_source", "archive_protocol", "archive_record_status"))
_OPTIONAL_CELLS = _ARCHIVE_TEXT_FIELDS | frozenset(("checkpoint", "checkpoint_id", "arm", "cycle_depth", "readout",
    "battle_won", "return", "source_version", "dup_pursuit_frac", "intent_stats", "intent_override", "protocol",
    "q_tot_mean", "q_tot_std", "q_i_mean", "episode_wall_seconds", "experiment_arm")) | frozenset(EXPERIMENT_EPISODE_FIELDS)


def _decode_row(cells, path):
    if None in cells or any(value is None for value in cells.values()):
        raise ValueError(f"Malformed complete CSV row in {path}")
    row = {}
    for field, value in cells.items():
        if value == "" and field in _OPTIONAL_CELLS:
            row[field] = None
            continue
        try:
            row[field] = json.loads(value)
        except json.JSONDecodeError as error:
            if field in _ARCHIVE_TEXT_FIELDS:
                row[field] = value
            else:
                raise ValueError(f"Invalid scientific JSON field {field} in {path}") from error
    for field in ("metrics", "data"):
        if isinstance(row.get(field), dict):
            row.update(row[field])
    return row


def _read_csv_bytes(path):
    """Read complete CSV lines without the writer lock so scans cannot stall training."""
    path = Path(path)
    delay = 0.05
    last_error = None
    for _ in range(8):
        try:
            with path.open("rb") as stream:
                raw = stream.read()
            break
        except OSError as error:
            last_error = error
            time.sleep(delay)
            delay = min(delay * 2, 0.8)
    else:
        raise last_error
    newline = raw.rfind(b"\n")
    return raw[:newline + 1] if newline >= 0 else b""


def record_paths(output, stream_name, env=None):
    """Locate a legacy shared CSV or the version's separate environment CSVs."""
    root = Path(output)
    current = root / "had" / f"{stream_name}.csv"
    if current.exists():
        return [current]
    legacy = root / f"{stream_name}.csv"
    if legacy.exists():
        return [legacy]
    if env is not None:
        from open_score.eval.experiment import environment_directory
        path = environment_directory(root, env) / f"{stream_name}.csv"
        return [path] if path.exists() else []
    return sorted(root.glob(f"*/{stream_name}.csv"))


def iter_records(output, stream_name, *, run=None, method=None, seed=None, env=None, experiment_arm=None):
    """Stream complete JSON-in-CSV rows, filtering before retaining history."""
    for path in record_paths(output, stream_name, env):
        yield from _iter_record_file(path, run=run, method=method, seed=seed, env=env, experiment_arm=experiment_arm)


def _iter_record_file(path, *, run=None, method=None, seed=None, env=None, experiment_arm=None):
    if not path.exists():
        return
    with path.open(encoding="utf-8", newline="") as stream:
        first = stream.readline()
        if not first.endswith("\n"):
            return
        header = next(csv.reader([first]))
        for line in stream:
            # Writers append one complete encoded row under a file lock. A
            # concurrent final partial write is retried on the next scan.
            if not line.endswith("\n"):
                break
            cells = next(csv.reader([line]))
            if len(cells) != len(header):
                raise ValueError(f"Malformed complete CSV record in {path}")
            raw = dict(zip(header, cells))
            filters = (("run", run), ("method", method), ("seed", seed), ("env", env), ("experiment_arm", experiment_arm))
            if any(value is not None and json.loads(raw.get(key, '"had"' if key == "env" else "null")) != value
                   for key, value in filters):
                continue
            yield _decode_row(raw, path)


def read_records(output, stream_name, *, run=None, method=None, seed=None, env=None, experiment_arm=None):
    return list(iter_records(output, stream_name, run=run, method=method, seed=seed, env=env, experiment_arm=experiment_arm))


def episode_key(row):
    return tuple(row.get(key) for key in ("version", "run", "method", "seed", "experiment_arm", "phase", "eval_point")) + (
        _encode(row["config"]), int(row["episode_seed"]), row.get("checkpoint"),
        row.get("env"), row.get("checkpoint_id"), row.get("arm"),
    )


def unique_episodes(rows):
    return list({episode_key(row): row for row in rows}.values())


class ExperimentLogger:
    def __init__(self, output=DEFAULT_OUTPUT, method="shared", seed=0, run=FORMAL_RUN, env="had", experiment_arm=None):
        if env != "had":
            raise ValueError("The active record writer supports HAD only")
        self.output, self.method, self.seed, self.run = Path(output), str(method), int(seed), str(run)
        from open_score.eval.experiment import ensure_layout, checked_experiment_arm
        self.experiment_arm = checked_experiment_arm(experiment_arm)
        ensure_layout(self.output)
        self.profile = True
        root = self.output.parent if self.output.name == "had" else self.output
        self.version = root.name
        metadata = root / "experiment.json"
        self.protocol = None
        if metadata.exists():
            retained = json.loads(metadata.read_text(encoding="utf-8"))
            self.version = retained.get("profile", self.version)
            self.protocol = retained.get("env_identity", {}).get("physics_protocol")
        self.env = env
        self._trajectory_keys = None

    def _row(self, data):
        row = dict(version=self.version, run=self.run, method=self.method, seed=self.seed,
                   recorded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        if self.profile:
            row["env"] = self.env
        if self.experiment_arm is not None:
            row["experiment_arm"] = self.experiment_arm
        if self.protocol:
            row.update(source_version=self.version, protocol=self.protocol)
        if data.get("experiment_arm", self.experiment_arm) != self.experiment_arm:
            raise ValueError("Record experiment arm differs from its writer")
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


    def trajectories(self, rows):
        def key(row):
            return tuple(row.get(field) for field in ("run", "method", "seed", "experiment_arm", "phase", "episode_seed", "checkpoint", "env", "checkpoint_id", "arm")) + (_encode(row["config"]),)
        if self._trajectory_keys is None:
            self._trajectory_keys = {key(row) for row in read_records(self.output, "trajectories", run=self.run,
                                                                     method=self.method, seed=self.seed, experiment_arm=self.experiment_arm)}
        pending = []
        for row in rows:
            value = self._row(dict(row))
            if key(value) not in self._trajectory_keys:
                pending.append(value)
        written = append_records(self.output, "trajectories", pending)
        self._trajectory_keys.update(key(row) for row in pending)
        return written
