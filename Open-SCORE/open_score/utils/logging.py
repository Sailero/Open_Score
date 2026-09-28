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
import re
import threading
import time

VERSION = "main"
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
    "timing": IDENTITY + ("config", "device", "cycle_depth", "repeat_id", "ms_per_step", "n_steps", "params"),
}


def schema_for(output, stream_name):
    from open_score.eval.experiment import is_profile
    columns = SCHEMAS[stream_name]
    if not is_profile(output):
        return columns
    extra = ("env", "checkpoint_id", "arm", "cycle_depth", "readout")
    if stream_name == "episodes":
        extra += ("battle_won", "return", "source_version", "dup_pursuit_frac", "intent_stats",
                  "intent_override", "protocol")
    if stream_name in ("learning", "trajectories", "progress"):
        extra += ("source_version", "protocol")
    if stream_name == "timing":
        extra += ("actor_params", "training_params", "p25_ms", "p75_ms", "p95_ms", "physical_gpu")
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
    columns = schema_for(output, stream_name)
    payload = io.StringIO(newline="")
    writer = csv.writer(payload, lineterminator="\n")
    for row in rows:
        writer.writerow([_encode(row.get(key)) for key in columns])
    path = Path(output) / f"{stream_name}.csv"
    try:
        from open_score.eval.cluster import nfs_lock
        _extra = nfs_lock(path)
    except Exception:
        from contextlib import nullcontext
        _extra = nullcontext()
    with _extra, _locked_file(path, create=True) as stream:
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


def iter_records(output, stream_name, *, run=None, method=None, seed=None, env=None):
    """Stream complete JSON-in-CSV rows, filtering before retaining history."""
    path = Path(output) / f"{stream_name}.csv"
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
            filters = (("run", run), ("method", method), ("seed", seed), ("env", env))
            if any(value is not None and json.loads(raw.get(key, '"had"' if key == "env" else "null")) != value
                   for key, value in filters):
                continue
            yield _decode_row(raw, path)


def read_records(output, stream_name, *, run=None, method=None, seed=None, env=None):
    return list(iter_records(output, stream_name, run=run, method=method, seed=seed, env=env))


def read_latest(output, stream_name, *, run=None, method=None, seed=None, keys=None, env=None):
    """Last row per (method, run, seed). Does not retain history."""
    from open_score.eval.experiment import is_profile
    if is_profile(output):
        latest = {}
        for row in iter_records(output, stream_name, run=run, method=method, seed=seed, env=env):
            identity = (row.get("method"), row.get("run"), row.get("seed"))
            if keys is not None and identity not in keys:
                continue
            if env is None:
                identity = (row.get("env"),) + identity
            latest[identity] = row
        return latest
    path = Path(output) / f"{stream_name}.csv"
    if not path.exists():
        return {}
    raw = _read_csv_bytes(path)
    if not raw:
        return {}
    text = raw.decode("utf-8")
    newline = text.find("\n")
    if newline < 0:
        return {}
    header_line, body = text[:newline], text[newline + 1:]
    if keys is not None:
        wanted = set(keys)
        found = {}
        header = next(csv.reader([header_line]))
        for line in reversed(body.splitlines()):
            if not line:
                continue
            cells = next(csv.reader([line]))
            if len(cells) != len(header):
                continue
            try:
                row = _decode_row(dict(zip(header, cells)), path)
            except (ValueError, json.JSONDecodeError):
                continue
            if run is not None and row.get("run") != run:
                continue
            identity = (row.get("method"), row.get("run"), row.get("seed"))
            if identity not in wanted or identity in found:
                continue
            found[identity] = row
            if len(found) >= len(wanted):
                break
        return found
    latest = {}
    for cells in csv.DictReader(io.StringIO(text, newline="")):
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
        row.get("env"), row.get("checkpoint_id"), row.get("arm"),
    )


def unique_episodes(rows):
    return list({episode_key(row): row for row in rows}.values())


def read_tail(path, nbytes=16384):
    path = Path(path)
    if not path.exists():
        return ""
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            if size > nbytes:
                stream.seek(size - nbytes)
            return stream.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _hms_seconds(text):
    hours, minutes, seconds = text.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds)


_TRAIN_CONSOLE = re.compile(
    r"\[(?P<name>[^\]]+)\]\s+(?P<hms>\d+:\d{2}:\d{2})\s+"
    r"(?P<t>[\d,]+)/(?P<b>[\d,]+)\s+(?P<sps>[\d.]+)/s"
)
_EVAL_CONSOLE = re.compile(
    r"\[(?P<who>[^\]]+)\]\s+(?P<phase>final_eval|depth_eval|mech\w*|anchor\w*)\s+"
    r"(?:R(?P<depth>\d+)\s+)?(?P<done>[\d,]+)/(?P<total>[\d,]+)"
)


def parse_train_console(text):
    for line in reversed(str(text or "").splitlines()):
        match = _TRAIN_CONSOLE.search(line)
        if match is None:
            continue
        row = dict(status="evaluating" if " ev=" in line else "training",
                   t_env=int(match.group("t").replace(",", "")),
                   budget_steps=int(match.group("b").replace(",", "")),
                   steps_per_second=float(match.group("sps")),
                   session_elapsed_seconds=_hms_seconds(match.group("hms")))
        loss = re.search(r"L=([\d.]+)", line)
        dval = re.search(r"D=([\d.-]+)", line)
        ev = re.search(r"ev=(\d+)/(\d+)", line)
        if loss:
            row["loss"] = float(loss.group(1))
        if dval is not None and dval.group(1) != "-":
            try:
                row["latest_validation_D"] = float(dval.group(1))
            except ValueError:
                pass
        if ev:
            row["eval_completed"] = int(ev.group(1))
            row["eval_total"] = int(ev.group(2))
        return row
    return {}


def parse_eval_console(text):
    for line in reversed(str(text or "").splitlines()):
        match = _EVAL_CONSOLE.search(line)
        if match is None:
            continue
        depth = match.group("depth")
        return dict(phase=match.group("phase"), status="running",
                    completed=int(match.group("done").replace(",", "")),
                    total=int(match.group("total").replace(",", "")),
                    cycle_depth=None if depth is None else int(depth))
    return {}


def live_console_jobs(output, max_age=180):
    """Train/eval jobs whose console log was written in the last max_age seconds."""
    root = Path(output)
    now = time.time()
    found = []
    for pattern, kind, filename in (
        ("*/*/seed_*/console.log", "train", "console.log"),
        ("*/*/seed_*/eval.console.log", "eval", "eval.console.log"),
    ):
        for path in root.glob(pattern):
            try:
                if now - path.stat().st_mtime > max_age:
                    continue
            except OSError:
                continue
            seed_dir, run_dir, method_dir = path.parent, path.parent.parent, path.parent.parent.parent
            try:
                seed = int(seed_dir.name.split("_")[1])
            except (IndexError, ValueError):
                continue
            found.append(dict(kind=kind, method=method_dir.name, run=run_dir.name,
                              seed=seed, path=path))
    found.sort(key=lambda item: (item["kind"], item["method"], item["seed"]))
    return found


class ExperimentLogger:
    def __init__(self, output=DEFAULT_OUTPUT, method="shared", seed=0, run=FORMAL_RUN, env="had"):
        self.output, self.method, self.seed, self.run = Path(output), str(method), int(seed), str(run)
        from open_score.eval.experiment import is_profile
        self.profile = is_profile(output)
        self.env = env
        self._trajectory_keys = None

    def _row(self, data):
        from open_score.eval.experiment import PROFILE
        row = dict(version=PROFILE if self.profile else VERSION, run=self.run, method=self.method, seed=self.seed,
                   recorded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        if self.profile:
            row["env"] = self.env
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
            if self.profile and self.env == "smacv2":
                for field in EPISODE_FIELDS:
                    row.setdefault(field, None)
            missing = set(EPISODE_FIELDS) - row.keys()
            if missing:
                raise ValueError(f"Missing episode fields: {sorted(missing)}")
            if row["phase"] not in ("train_eval", "final_eval", "depth_eval", "readout_eval", "anchor",
                                    "gate_depth_eval", "dup_eval", "r1deploy_eval", "intent_intervention_eval"):
                raise ValueError(f"Not a frozen evaluation phase: {row['phase']}")
            for field in (("battle_won", "ep_len") if self.env == "smacv2" else ("D", "rho", "ep_len")):
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

    def timing(self, rows):
        encoded = [self._row(dict(row)) for row in rows]
        return append_records(self.output, "timing", encoded)

    def trajectories(self, rows):
        def key(row):
            return tuple(row.get(field) for field in ("run", "method", "seed", "phase", "episode_seed", "checkpoint", "env", "checkpoint_id", "arm")) + (_encode(row["config"]),)
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
