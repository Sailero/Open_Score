"""Scan outputs/main, migrate v3/v4/v5 assets, and list pending train/eval jobs."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sys

from open_score.algos import (
    MAIN_ABLATION_METHODS, MAIN_METHODS, MAIN_TRAIN_METHODS, METHOD_ALIASES,
    canonical_method, resume_files,
)
from open_score.eval.protocol import (
    CYCLE_SERIES_METHODS, DEPTH_SWEEP_DEPTHS, FINAL_CONFIGS, FINAL_EPISODES_PER_CONFIG,
    MECH_CONFIG, MECH_EPISODE_SEEDS, OFFICIAL_CHECKPOINT, TIMING_METHODS, TIMING_SCALES,
    config_key, depth_eval_finished, official_eval_t_env, remaining_depth_jobs,
    remaining_jobs, final_jobs, training_finished,
)
from open_score.utils.logging import DEFAULT_OUTPUT, FORMAL_RUN, VERSION, V3_OUTPUT, V4_OUTPUT, V5_OUTPUT, SCHEMAS, read_latest, read_records, unique_episodes

FORMAL_SEEDS = (0, 1, 2)
CORE_SEEDS = (0, 1, 2)
EXTRA_SEEDS = (3, 4)
T_MAX = 1_000_000
ARCHIVE_METHODS = (
    "b0_qmix", "gnn_qmix", "refil_local_mild", "refil_local_mid", "refil_count",
    "refil_count_ln", "refil_card", "refil_feedback", "refil_slot", "alma_legacy",
)
COMPARE_METHODS = MAIN_METHODS
CORE_TRAIN_QUEUE = (
    (("regir", 1),)
    + tuple(("alma", seed) for seed in CORE_SEEDS)
    + (("regir", 2),)
    + tuple((method, seed) for method in ("regir_r1", "regir_nocount", "regir_last", "regir_norefil")
            for seed in CORE_SEEDS)
    + tuple(("refil_matched", seed) for seed in CORE_SEEDS)
)
EXTRA_TRAIN_QUEUE = tuple(
    (method, seed)
    for method in (
        "regir", "alma",
        "regir_r1", "regir_nocount", "regir_last", "regir_norefil",
        "refil_matched",
        "refil", "b2_qmix_atten", "dcg", "spectra",
    )
    for seed in EXTRA_SEEDS
)
TRAIN_QUEUE = CORE_TRAIN_QUEUE
CSV_NAMES = ("episodes", "learning", "progress", "trajectories", "verification", "benchmarks")
METHOD_REMAP = {"refil_cycle": "regir", "regia": "regir"}


def output_dir(path=None):
    return Path(DEFAULT_OUTPUT if path is None else path).resolve()


def run_dir(output, method, seed, run=FORMAL_RUN):
    return output_dir(output) / method / run / f"seed_{int(seed)}"


def _resolved_best_t_env(output, method, seed, rows, directory):
    return official_eval_t_env(directory)


def _training_finished(directory, row):
    return training_finished(directory, row)


def _copy_tree(src, dst):
    src, dst = Path(src), Path(dst)
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        completed = subprocess.run(
            ["robocopy", str(src), str(dst), "/E", "/NFL", "/NDL", "/NJH", "/NJS", "/NC", "/NS", "/NP"],
            capture_output=True, text=True)
        if completed.returncode >= 8:
            raise RuntimeError(f"robocopy {src} -> {dst} failed: {completed.returncode}\n{completed.stderr}")
        return
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _rewrite_config_method(directory, method):
    path = Path(directory) / "config.json"
    if not path.exists():
        return
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["method"] = method
    saved["name"] = method
    path.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")


def _merge_csv(output, name, sources, method_remap):
    from open_score.utils.logging import append_records, episode_key
    output = Path(output)
    path = output / f"{name}.csv"
    seen = set()
    rows = []
    if path.exists():
        for row in read_records(output, name):
            key = episode_key(row) if name == "episodes" else json.dumps(row, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    for src in sources:
        src = Path(src)
        if not (src / f"{name}.csv").exists():
            continue
        for row in read_records(src, name):
            item = dict(row)
            item["version"] = VERSION
            item["method"] = method_remap.get(item.get("method"), item.get("method"))
            if name == "episodes":
                key = episode_key(item)
            else:
                key = json.dumps(item, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            rows.append(item)
    if not rows:
        return 0
    tmp = path.with_suffix(".csv.migrating")
    fieldnames = list(SCHEMAS[name])
    with tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            encoded = {}
            for field in fieldnames:
                value = row.get(field)
                if field in ("config", "metrics", "data", "trajectory") or isinstance(value, (dict, list, tuple)):
                    encoded[field] = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str) if value is not None else ""
                else:
                    encoded[field] = "" if value is None else value
            writer.writerow(encoded)
    os.replace(tmp, path)
    return len(rows)


def migrate_into_main(output=None):
    """Copy v3/v4/v5 weights into outputs/main and merge CSV records once."""
    output = output_dir(output)
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "migration.done"
    if marker.exists():
        return scan(output)
    if V3_OUTPUT.exists():
        for method in ("b2_qmix_atten", "refil", "dcg", "spectra", "b0_qmix", "gnn_qmix"):
            _copy_tree(V3_OUTPUT / method, output / method)
        if (V3_OUTPUT / "alma").exists():
            _copy_tree(V3_OUTPUT / "alma", output / "alma_legacy")
    if V4_OUTPUT.exists():
        for method in ("refil_local_mild", "refil_local_mid"):
            _copy_tree(V4_OUTPUT / method, output / method)
        if (V4_OUTPUT / "refil_count").exists():
            _copy_tree(V4_OUTPUT / "refil_count", output / "refil_count_ln")
    if V5_OUTPUT.exists():
        _copy_tree(V5_OUTPUT / "refil_count", output / "refil_count")
        _copy_tree(V5_OUTPUT / "refil_cycle", output / "regir")
        for method in ("refil_card", "refil_feedback", "refil_slot"):
            _copy_tree(V5_OUTPUT / method, output / method)
        if (output / "regir").exists():
            for seed_dir in (output / "regir").glob("train/seed_*"):
                _rewrite_config_method(seed_dir, "regir")
        if (output / "alma_legacy").exists():
            for seed_dir in (output / "alma_legacy").glob("train/seed_*"):
                _rewrite_config_method(seed_dir, "alma_legacy")
        leftover_cycle = output / "refil_cycle"
        if leftover_cycle.exists() and leftover_cycle.resolve() != (output / "regir").resolve():
            shutil.rmtree(leftover_cycle)
    remap = dict(METHOD_REMAP)
    remap["alma"] = "alma_legacy"
    sources = [path for path in (V3_OUTPUT, V4_OUTPUT, V5_OUTPUT) if path.exists()]
    counts = {}
    for name in CSV_NAMES:
        this_remap = dict(remap)
        if name != "episodes" and name != "learning" and name != "progress" and name != "trajectories":
            this_remap = dict(METHOD_REMAP)
        # v3 alma rows become alma_legacy so they never enter the new ALMA mean.
        if name in ("episodes", "learning", "progress", "trajectories"):
            this_remap["alma"] = "alma_legacy"
        # v4 count rows are the LN arm.
        v4_only = {}
        if V4_OUTPUT.exists() and name in ("episodes", "learning", "progress"):
            v4_only["refil_count"] = "refil_count_ln"
        merged_remap = dict(this_remap)
        # Apply v4 remap only while reading v4; handled below per source.
        rows_before = counts.get(name, 0)
        counts[name] = _merge_csv_per_source(output, name, sources, this_remap, v4_only)
        _ = rows_before
    marker.write_text(json.dumps({"csv": counts}, ensure_ascii=False, indent=2), encoding="utf-8")
    return scan(output)


def _merge_csv_per_source(output, name, sources, base_remap, v4_count_remap):
    from open_score.utils.logging import episode_key
    output = Path(output)
    path = output / f"{name}.csv"
    seen = set()
    rows = []

    def remember(row):
        if name == "episodes":
            key = episode_key(row)
        else:
            key = (row.get("run"), row.get("method"), row.get("seed"), row.get("recorded_at"),
                   json.dumps(row.get("data") or row.get("metrics") or row.get("trajectory") or {},
                              sort_keys=True, default=str),
                   row.get("t_env"), row.get("phase"), row.get("checkpoint"))
        if key in seen:
            return
        seen.add(key)
        rows.append(row)

    if path.exists():
        for row in read_records(output, name):
            item = dict(row)
            item["version"] = VERSION
            item["method"] = METHOD_REMAP.get(item.get("method"), item.get("method"))
            if name in ("episodes", "learning", "progress", "trajectories") and item.get("method") == "alma":
                item["method"] = "alma_legacy"
            remember(item)
    for src in sources:
        src = Path(src)
        if not (src / f"{name}.csv").exists():
            continue
        remap = dict(base_remap)
        if src.resolve() == V4_OUTPUT.resolve():
            remap.update(v4_count_remap)
        if src.resolve() == V5_OUTPUT.resolve():
            remap.pop("alma", None)
            remap.pop("refil_count", None)
        for row in read_records(src, name):
            item = dict(row)
            item["version"] = VERSION
            item["method"] = remap.get(item.get("method"), item.get("method"))
            remember(item)
    if not rows:
        return 0
    from open_score.utils.logging import _encode
    tmp = path.with_suffix(".csv.migrating")
    fieldnames = list(SCHEMAS[name])
    with tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(fieldnames)
        for row in rows:
            writer.writerow([_encode(row.get(field)) for field in fieldnames])
    os.replace(tmp, path)
    return len(rows)


def _gpu_training_busy():
    try:
        import subprocess as sp
        if os.name == "nt":
            completed = sp.run(
                ["powershell", "-NoProfile", "-Command",
                 "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                 "Where-Object { $_.CommandLine -match 'train.py' } | Measure-Object | "
                 "Select-Object -ExpandProperty Count"],
                capture_output=True, text=True, timeout=20)
            return int((completed.stdout or "0").strip() or 0) > 0
        completed = sp.run(["ps", "-eo", "args"], capture_output=True, text=True, timeout=20)
        return any("train.py" in line and "--stage" in line for line in completed.stdout.splitlines())
    except Exception:
        return False


def core_main_train_complete(output=None):
    """True when every main method has finished seeds 0–2."""
    output = output_dir(output)
    latest = read_latest(output, "progress", run=FORMAL_RUN)
    for method in MAIN_TRAIN_METHODS:
        for seed in CORE_SEEDS:
            directory = run_dir(output, method, seed)
            row = latest.get((method, FORMAL_RUN, seed), {})
            if not _training_finished(directory, row):
                return False
    return True


def train_queue(output=None):
    queue = list(CORE_TRAIN_QUEUE)
    if core_main_train_complete(output):
        queue.extend(EXTRA_TRAIN_QUEUE)
    return queue


def _describe_train(directory, row):
    if _training_finished(directory, row):
        return "completed", ""
    if any(path.exists() for path in resume_files(directory)) or (directory / "config.json").exists():
        if row.get("status") in ("training", "evaluating", "running", "resuming"):
            return "running", ""
        return "pending", ""
    return "pending", ""


def print_roster(payload, *, kinds=None):
    tasks = list(payload.get("tasks") or [])
    if kinds:
        tasks = [task for task in tasks if task["kind"] in kinds]
    groups = defaultdict(list)
    for task in tasks:
        groups[task["status"]].append(task)
    counts = "  ".join(f"{status}={len(groups[status])}" for status in
                       ("running", "pending", "blocked", "completed", "legacy") if groups[status])
    print(f"inventory {len(tasks)} tasks  {counts}", flush=True)
    titles = (("running", "当前任务"), ("pending", "未完成"), ("blocked", "阻塞"),
              ("completed", "已完成"), ("legacy", "档案"))
    for status, title in titles:
        items = groups.get(status) or []
        if not items:
            continue
        print(f"{title} ({len(items)})", flush=True)
        for task in items:
            extra = task.get("detail") or ""
            print(f"  {task['id']:<42} {extra}".rstrip(), flush=True)
    sys.stdout.flush()


def scan(output=None, *, quiet=False):
    output = output_dir(output)
    if not quiet:
        print(f"scanning {output}", flush=True)
        print("  progress.csv ...", flush=True)
    latest = read_latest(output, "progress", run=FORMAL_RUN)
    if not quiet:
        print(f"  {len(latest)} runs in progress.csv", flush=True)
    tasks = []

    def add(task_id, kind, method, seed, status, detail=""):
        tasks.append(dict(id=task_id, kind=kind, method=method, seed=seed, status=status, detail=detail))

    for method, seed in CORE_TRAIN_QUEUE:
        directory = run_dir(output, method, seed)
        row = latest.get((method, FORMAL_RUN, seed), {})
        status, detail = _describe_train(directory, row)
        add(f"train.{method}.s{seed}", "train", method, seed, status, detail)

    extra_unlocked = all(
        _training_finished(run_dir(output, method, seed), latest.get((method, FORMAL_RUN, seed), {}))
        for method in MAIN_TRAIN_METHODS for seed in CORE_SEEDS
    )
    for method, seed in EXTRA_TRAIN_QUEUE:
        directory = run_dir(output, method, seed)
        row = latest.get((method, FORMAL_RUN, seed), {})
        if extra_unlocked:
            status, detail = _describe_train(directory, row)
        else:
            status, detail = "blocked", "after core main seeds 0-2"
        add(f"train.{method}.s{seed}", "train", method, seed, status, detail)

    for method in ("b2_qmix_atten", "refil", "dcg", "spectra"):
        for seed in FORMAL_SEEDS:
            directory = run_dir(output, method, seed)
            row = latest.get((method, FORMAL_RUN, seed), {})
            status = "completed" if _training_finished(directory, row) else "legacy"
            add(f"train.{method}.s{seed}", "train", method, seed, status)

    directory = run_dir(output, "regir", 0)
    row = latest.get(("regir", FORMAL_RUN, 0), {})
    add("train.regir.s0", "train", "regir", 0,
        "completed" if _training_finished(directory, row) else "pending")

    if not quiet:
        print("  episodes.csv ...", flush=True)
    episodes = unique_episodes(read_records(output, "episodes", run=FORMAL_RUN))
    if not quiet:
        print(f"  {len(episodes)} unique episodes", flush=True)
    by_method = defaultdict(list)
    for row in episodes:
        by_method[(row.get("method"), int(row.get("seed") or 0))].append(row)

    from open_score.eval.anchors import ANCHOR_EPISODES, anchor_jobs
    existing_anchors = {(row["method"], config_key(row["config"]), int(row["episode_seed"]))
                        for row in episodes if row.get("phase") == "anchor"}
    wanted_anchors = {(job["method"], config_key(job["config"]), job["episode_seed"]) for job in anchor_jobs()}
    missing_anchors = len(wanted_anchors - existing_anchors)
    add("eval.anchors", "eval", "anchors", None,
        "completed" if missing_anchors == 0 else "pending",
        f"missing={missing_anchors}")

    eval_seeds = CORE_SEEDS + (EXTRA_SEEDS if extra_unlocked else ())
    eval_methods = [("b2_qmix_atten", eval_seeds), ("refil", eval_seeds),
                    ("dcg", eval_seeds), ("spectra", eval_seeds),
                    ("regir", eval_seeds), ("alma", eval_seeds)]
    for method in MAIN_ABLATION_METHODS + ("refil_matched",):
        eval_methods.append((method, eval_seeds))
    for method, seeds in eval_methods:
        for seed in seeds:
            directory = run_dir(output, method, seed)
            official = directory / f"{OFFICIAL_CHECKPOINT}.pt"
            if not official.exists():
                add(f"eval.final.{method}.s{seed}", "eval", method, seed, "blocked",
                    f"no {OFFICIAL_CHECKPOINT}.pt")
                continue
            if not _training_finished(directory, latest.get((method, FORMAL_RUN, seed), {})):
                add(f"eval.final.{method}.s{seed}", "eval", method, seed, "blocked",
                    "training not finished")
                continue
            t_env = _resolved_best_t_env(output, method, seed, by_method[(method, seed)], directory)
            if t_env is None:
                add(f"eval.final.{method}.s{seed}", "eval", method, seed, "blocked",
                    f"unreadable {OFFICIAL_CHECKPOINT}.pt")
                continue
            leftover = remaining_jobs(final_jobs(t_env, OFFICIAL_CHECKPOINT, method=method),
                                      by_method[(method, seed)])
            add(f"eval.final.{method}.s{seed}", "eval", method, seed,
                "completed" if not leftover else "pending",
                f"remaining={len(leftover)}")

    for seed in eval_seeds:
        directory = run_dir(output, "regir", seed)
        if not (directory / f"{OFFICIAL_CHECKPOINT}.pt").exists():
            add(f"eval.depth.regir.s{seed}", "eval", "regir", seed, "blocked",
                f"no {OFFICIAL_CHECKPOINT}.pt")
            continue
        if not _training_finished(directory, latest.get(("regir", FORMAL_RUN, seed), {})):
            add(f"eval.depth.regir.s{seed}", "eval", "regir", seed, "blocked",
                "training not finished")
            continue
        t_env = _resolved_best_t_env(output, "regir", seed, by_method[("regir", seed)], directory)
        if t_env is None:
            add(f"eval.depth.regir.s{seed}", "eval", "regir", seed, "blocked", "no t_env")
            continue
        leftover = remaining_depth_jobs(t_env, by_method[("regir", seed)], OFFICIAL_CHECKPOINT)
        add(f"eval.depth.regir.s{seed}", "eval", "regir", seed,
            "completed" if not leftover else "pending",
            f"remaining={len(leftover)}")

    traj = read_records(output, "trajectories", run=FORMAL_RUN, method="regir", seed=0)
    mech_done = any(row.get("phase") == "mechanism" for row in traj)
    add("eval.mech.regir.s0", "eval", "regir", 0,
        "completed" if mech_done else ("pending" if (run_dir(output, "regir", 0) / f"{OFFICIAL_CHECKPOINT}.pt").exists() else "blocked"))

    timing_rows = read_records(output, "timing", run=FORMAL_RUN) if (output / "timing.csv").exists() else []
    timing_keys = {(row.get("method"), row.get("config") if not isinstance(row.get("config"), dict)
                    else tuple(int(row["config"][k]) for k in ("N_R", "N_B", "K")),
                    row.get("cycle_depth")) for row in timing_rows}
    wanted = {(method, scale, 4 if method != "regir" else None) for method in TIMING_METHODS for scale in TIMING_SCALES}
    wanted |= {("regir", scale, depth) for scale in TIMING_SCALES for depth in DEPTH_SWEEP_DEPTHS}
    gpu_busy = _gpu_training_busy()
    add("eval.timing", "eval", "timing", None,
        "blocked" if gpu_busy else ("completed" if wanted <= timing_keys else "pending"),
        "gpu training busy" if gpu_busy else "")

    payload = dict(output=str(output), version=VERSION, tasks=tasks)
    (output / "inventory.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if not quiet:
        print_roster(payload)
    return payload


def list_pending_trains(output=None):
    """Pending formal trains from checkpoints/progress only; does not read episodes.csv."""
    output = output_dir(output)
    latest = read_latest(output, "progress", run=FORMAL_RUN)
    pending = []
    for method, seed in train_queue(output):
        directory = run_dir(output, method, seed)
        row = latest.get((method, FORMAL_RUN, seed), {})
        if _training_finished(directory, row):
            continue
        pending.append(dict(id=f"train.{method}.s{seed}", kind="train",
                            method=method, seed=seed, status="pending"))
    return pending


def pending_train_jobs(output=None):
    tasks = scan(output)["tasks"]
    order = {(method, seed): index for index, (method, seed) in enumerate(train_queue(output))}
    pending = [task for task in tasks if task["kind"] == "train" and task["status"] == "pending"
               and (task["method"], task["seed"]) in order]
    pending.sort(key=lambda task: order[(task["method"], task["seed"])])
    return pending


def pending_eval_jobs(output=None, only=None, *, quiet=False):
    tasks = scan(output, quiet=quiet)["tasks"]
    rank = {"anchors": 0, "final": 1, "depth": 2, "mech": 3, "timing": 4}
    allowed = None if not only else {item.strip() for item in only}
    selected = []
    for task in tasks:
        if task["kind"] != "eval" or task["status"] != "pending":
            continue
        kind = task["id"].split(".")[1]
        if allowed and kind not in allowed:
            continue
        selected.append(task)
    method_rank = {
        "regir": 0, "alma": 1, "refil": 2, "b2_qmix_atten": 3, "dcg": 4, "spectra": 5,
        "regir_r1": 6, "regir_nocount": 7, "regir_norefil": 8, "regir_last": 9,
        "refil_matched": 10,
    }
    selected.sort(key=lambda task: (
        rank.get(task["id"].split(".")[1], 9),
        method_rank.get(task.get("method"), 50),
        -1 if task.get("seed") is None else int(task["seed"]),
        task["id"],
    ))
    return selected
