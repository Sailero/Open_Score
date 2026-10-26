"""Frozen HAD validation and final/confirmation evaluation quotas."""
from __future__ import annotations
import math
import csv
import json
import os
from pathlib import Path
import time
from itertools import count

VALIDATION_CONFIGS = ((4, 4, 2), (6, 6, 2), (8, 8, 2), (10, 10, 3))
GROUPS = {
 "ID": VALIDATION_CONFIGS,
 "K-OOD": tuple((10, 10, k) for k in (4, 6, 9, 12)),
 "N-OOD": tuple((n, n, 2) for n in (15, 20, 25, 30, 40, 50)),
 "Joint-OOD": ((15, 15, 4), (15, 15, 6), (20, 20, 4), (20, 20, 6),
               (30, 30, 4), (30, 30, 6), (30, 30, 9), (30, 30, 12)),
}
FINAL_CONFIGS = tuple(sorted({c for values in GROUPS.values() for c in values} | {(5, 5, 2), (10, 10, 2)}))
VALIDATION_EPISODES_PER_CONFIG = 25
FINAL_EPISODES_PER_CONFIG = 300
FINAL_EPISODES = len(FINAL_CONFIGS) * FINAL_EPISODES_PER_CONFIG
DEPTH_SWEEP_DEPTHS = (1, 2, 3, 4, 5, 6)
CYCLE_SERIES_METHODS = ("regir_nomem",)
OFFICIAL_CHECKPOINT = "final"
_POLICY_IDS = count()

def config_dict(config):
    return {k: int(config[k]) for k in ("N_R", "N_B", "K")} if isinstance(config, dict) else dict(zip(("N_R", "N_B", "K"), map(int, config)))

def config_key(config):
    c = config_dict(config)
    return c["N_R"], c["N_B"], c["K"]

def config_label(config):
    red, blue, targets = config_key(config)
    return f"{red}v{blue} K{targets}"

def final_configs(method=None):
    if method is not None:
        from open_score.algos import validate_method
        validate_method(method)
    return FINAL_CONFIGS

def final_episodes(method=None):
    if method is not None:
        from open_score.algos import validate_method
        validate_method(method)
    return FINAL_EPISODES

def evaluation_thresholds(budget):
    budget = int(budget)
    if budget < 50:
        raise ValueError("A formal budget must support 50 validation points")
    return [math.ceil(budget * point / 50) for point in range(1, 51)]

def validation_jobs(eval_point, t_env=0):
    if not 1 <= int(eval_point) <= 50:
        raise ValueError("eval_point must be 1..50")
    return [dict(config=config_dict(c), episode_seed=9500 + 25 * i + offset,
                 phase="train_eval", eval_point=int(eval_point), t_env=int(t_env), retain_trajectory=False)
            for i, c in enumerate(VALIDATION_CONFIGS) for offset in range(25)]

def checkpoint_tag(stem, t_env):
    return f"{stem}@{int(t_env)}"

def parse_sweep_depth(checkpoint):
    try:
        return int(str(checkpoint).rsplit("/R", 1)[1])
    except (ValueError, IndexError):
        return None

def parse_checkpoint_t_env(checkpoint):
    try:
        return int(str(checkpoint).partition("@")[2].split("/")[0])
    except (ValueError, IndexError):
        return None

def final_jobs(t_env=0, checkpoint=None, method=None, *, confirmation=False):
    if method is not None:
        from open_score.algos import validate_method
        validate_method(method)
    start = 110000 if confirmation else 9000
    tag = checkpoint_tag(checkpoint or OFFICIAL_CHECKPOINT, t_env) + ("/confirmation" if confirmation else "")
    return [dict(config=config_dict(c), episode_seed=start + offset, phase="final_eval",
                 eval_point=50, t_env=int(t_env), checkpoint=tag,
                 arm="confirmation" if confirmation else "standard", retain_trajectory=False)
            for c in FINAL_CONFIGS for offset in range(FINAL_EPISODES_PER_CONFIG)]

def validation_score(rows):
    """Return average D only for one complete, correctly seeded validation point."""
    selected = [row for row in rows if row.get("phase") == "train_eval"]
    if not selected:
        return None
    # Hard barrier against test-set leakage into checkpoint selection.
    outside = {config_key(row["config"]) for row in selected} - set(VALIDATION_CONFIGS)
    if outside:
        raise ValueError(f"Checkpoint selection saw non-training configurations: {sorted(outside)}")
    for field in ("run", "method", "seed", "eval_point"):
        identities = {row[field] for row in selected if row.get(field) is not None}
        if len(identities) > 1:
            raise ValueError("Checkpoint selection requires exactly one run/method/seed/eval_point")
    expected = {(config_key(job["config"]), job["episode_seed"]) for job in validation_jobs(selected[0]["eval_point"])}
    by_episode = {(config_key(row["config"]), int(row["episode_seed"])): row for row in selected}
    if set(by_episode) != expected:
        return None
    values = [float(row["D"]) for row in by_episode.values()]
    if not all(math.isfinite(value) for value in values):
        return None
    return sum(values) / 100


def remaining_jobs(jobs, recorded_rows):
    done = {(row.get("phase"), row.get("eval_point"), config_key(row["config"]),
             int(row["episode_seed"]), row.get("checkpoint")) for row in recorded_rows}
    return [job for job in jobs if (job.get("phase"), job.get("eval_point"), config_key(job["config"]),
            int(job["episode_seed"]), job.get("checkpoint")) not in done]


def run_directory(output, method, seed, run=None):
    from ..experiment import run_directory as directory
    return directory(output, method, seed)

def checkpoint_identity(saved, checkpoint=None, output=None):
    value = saved.get("artifact_id") or saved.get("progress", {}).get("final_artifact_id")
    if value:
        return str(value)
    if checkpoint is not None and output is not None:
        from ..experiment import read_json
        repo = Path(__file__).resolve().parents[2]
        metadata = read_json(Path(output) / "summary.json", {})
        for row in metadata.get("checkpoints", []):
            target = row.get("archived_path") or row.get("path")
            if target and row.get("artifact_id"):
                path = Path(target)
                path = path if path.is_absolute() else repo / path
                if path.resolve() == Path(checkpoint).resolve():
                    return str(row["artifact_id"])
    c = saved["config"]
    return f"{c['method']}:{int(c['seed'])}:{int(saved['progress']['t_env'])}:{c.get('implementation_revision', 'retained')}"

def _evaluate(method, checkpoint, *, output=None, run=None, seed=None, device="cpu",
              on_progress=None, stop_requested=None, confirmation=False, depths=None):
    from open_score.algos import load_policy, validate_method
    validate_method(method)
    import torch
    from open_score.algos.rules import register_end_to_end_policy, run_episode
    from open_score.utils.logging import DEFAULT_OUTPUT, FORMAL_RUN, ExperimentLogger, read_records
    from .anchors import BLUE_STRATEGY
    from ..experiment import atomic_json, process_lock, check_environment, ensure_layout, EXPECTED_ENV
    check_environment()
    output = Path(DEFAULT_OUTPUT if output is None else output)
    ensure_layout(output)
    checkpoint = Path(checkpoint)
    run = FORMAL_RUN if run is None else run
    with process_lock(checkpoint.parent / ".eval.lock") as claimed:
        if not claimed:
            raise RuntimeError(f"Checkpoint already has an evaluator: {checkpoint}")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg, progress = saved["config"], saved["progress"]
        actual_seed, t_env = int(cfg["seed"]), int(progress["t_env"])
        if cfg.get("method") != method or (seed is not None and int(seed) != actual_seed):
            raise ValueError("Evaluation method/seed must identify the retained checkpoint")
        if cfg.get("env") != "had" or checkpoint.name != "final.pt" or t_env < int(cfg.get("t_max", t_env)):
            raise ValueError("Frozen evaluation requires completed HAD final.pt")
        identity = checkpoint_identity(saved, checkpoint, output)
        protocol = cfg.get("implementation_revision", "retained")
        selected = tuple(int(d) for d in depths) if depths is not None else ()
        if depths is not None and (not selected or len(set(selected)) != len(selected) or min(selected) < 1 or max(selected) > 6):
            raise ValueError("Explicit depth selection must be a distinct subset of R1..R6")
        jobs = final_jobs(t_env, confirmation=confirmation)
        if selected:
            jobs = [dict(job, phase="depth_eval", cycle_depth=d,
                         checkpoint=checkpoint_tag("final", t_env) + f"/R{d}", arm="depth")
                    for d in selected for job in final_jobs(t_env)]
        for job in jobs:
            job.update(checkpoint_id=identity, protocol=protocol)
        existing = read_records(output, "episodes", run=run, method=method, seed=actual_seed, env="had")
        def key(row):
            return row.get("phase"), row.get("eval_point"), config_key(row["config"]), int(row["episode_seed"]), row.get("checkpoint")
        wanted = {key(job) for job in jobs}
        retained = {key(r): r for r in existing if str(r.get("checkpoint_id")) == identity and key(r) in wanted}
        compatible = list(retained.values())
        pending = remaining_jobs(jobs, compatible)
        total, completed = len(jobs), len(jobs) - len(pending)
        sum_d = sum(float(row["D"]) for row in compatible)
        phase = "depth_eval" if selected else "confirmation" if confirmation else "final_eval"
        logger = ExperimentLogger(output, method, actual_seed, run, env="had")
        started, initial, last_publish = time.monotonic(), completed, 0.
        def publish(status):
            elapsed = time.monotonic() - started
            row = dict(phase=phase, status=status, method=method, seed=actual_seed,
                       t_env=t_env, budget_steps=int(cfg.get("t_max", t_env)), completed=completed,
                       total=total, checkpoint_id=identity, checkpoint=str(checkpoint),
                       remaining_seconds=(0. if completed == total else (total - completed) * elapsed / (completed - initial) if completed > initial else None),
                       elapsed_seconds=elapsed, D=sum_d / completed if completed else None,
                       updated_at=time.time(), pid=__import__("os").getpid())
            atomic_json(checkpoint.parent / "evaluation.json", row)
            logger.progress(**{k: v for k, v in row.items() if k not in ("method", "seed")})
            if on_progress is not None:
                on_progress(dict(row))
        if not pending:
            publish("complete")
            return dict(status="complete", completed=completed, total=total)
        del saved
        policy = load_policy(method, checkpoint)
        policy.set_device(device)
        policy_name = f"had_evaluation_{next(_POLICY_IDS)}"
        register_end_to_end_policy("red", policy_name, method, lambda _: policy)
        publish("running")
        try:
            for job in pending:
                if stop_requested is not None and stop_requested():
                    publish("stopped")
                    return dict(status="stopped", completed=completed, total=total)
                if selected:
                    policy.set_eval_depth(job["cycle_depth"])
                red, blue, targets = config_key(job["config"])
                before = time.monotonic()
                result = run_episode(red=red, blue=blue, targets=targets, seed=job["episode_seed"],
                    red_strategy={"architecture": "end_to_end", "policy": policy_name},
                    blue_strategy=BLUE_STRATEGY, max_steps=int(cfg.get("episode_limit", 100)),
                    record=False, task_mode="damage", spatial_dim=2, target_initialization="random",
                    diagnostics=True, record_events=False, retain_trajectory=False)
                if result.get("physics_protocol") != EXPECTED_ENV["physics_protocol"]:
                    raise ValueError("Episode physics identity changed; result was not committed")
                row = dict(result["episode_summary"], **job, episode_wall_seconds=time.monotonic() - before)
                if hasattr(policy, "episode_q_statistics"):
                    row.update(policy.episode_q_statistics())
                logger.episodes([row])
                completed += 1
                sum_d += float(row["D"])
                now = time.monotonic()
                if now - last_publish >= 10 or completed == total:
                    publish("running")
                    print(f"{method}/s{actual_seed} {phase} {completed}/{total}", flush=True)
                    last_publish = now
            publish("complete")
        except BaseException as error:
            publish("failed")
            raise
        return dict(status="complete", completed=completed, total=total)

def evaluate_checkpoint(method, checkpoint, *, output=None, run=None, seed=None, device="cpu", on_progress=None, stop_requested=None, confirmation=False):
    return _evaluate(method, checkpoint, output=output, run=run, seed=seed, device=device,
                     on_progress=on_progress, stop_requested=stop_requested, confirmation=confirmation)

def evaluate_depth_sweep(method, checkpoint, *, output=None, run=None, seed=None, device="cpu", on_progress=None, stop_requested=None, depths=DEPTH_SWEEP_DEPTHS):
    if method not in CYCLE_SERIES_METHODS:
        raise ValueError(f"{method} has no cycle-depth interface")
    return _evaluate(method, checkpoint, output=output, run=run, seed=seed, device=device,
                     on_progress=on_progress, stop_requested=stop_requested, depths=depths)

def compute_nds(rho, rho_random, rho_rule):
    denominator = float(rho_random) - float(rho_rule)
    return None if abs(denominator) < 1e-12 else (float(rho_random) - float(rho)) / denominator


REPRODUCTION_CONFIGS = ((4, 4, 2), (15, 15, 2))
REPRODUCTION_SEEDS = (9000, 9001, 9002)


def _reference_checkpoint_path(reference, metadata):
    """Resolve archived paths against the selected read-only reference root."""
    reference = Path(reference).resolve()
    archived = str(metadata.get("archived_path") or metadata.get("path") or "").replace("\\", "/")
    marker = f"/outputs/{reference.name}/"
    normalized = "/" + archived.lstrip("/")
    suffix = normalized.partition(marker)[2] if marker in normalized else archived
    candidates = [reference / suffix,
                  reference / "had" / metadata["method"] / "models" / f"seed_{int(metadata['seed'])}" / "final.pt",
                  reference / metadata["method"] / "models" / f"seed_{int(metadata['seed'])}" / "final.pt"]
    for path in candidates:
        path = path.resolve()
        if path.is_relative_to(reference) and path.is_file() and path.name == "final.pt":
            return path
    raise FileNotFoundError(f"Retained final checkpoint is absent under {reference}: {archived}")


def reproduction_jobs(reference, methods=None):
    """Select exactly 60 current-comparison openings from retained real CSV data."""
    from open_score.algos import METHODS, validate_method
    from open_score.utils.logging import iter_records
    from ..experiment import read_json
    reference = Path(reference).resolve()
    methods = tuple(validate_method(method) for method in (METHODS if methods is None else methods))
    if len(set(methods)) != len(methods):
        raise ValueError("Reproduction methods must be distinct")
    summary = read_json(reference / "summary.json")
    if not isinstance(summary, dict):
        raise FileNotFoundError(f"Reference summary is absent or unreadable: {reference / 'summary.json'}")
    source = summary.get("current_evaluation_source", "had/aggregates.json#comparison")
    aggregate_path, _, section = str(source).partition("#")
    aggregate_file = (reference / aggregate_path).resolve()
    if not aggregate_file.is_relative_to(reference):
        raise ValueError("Current comparison source must remain inside the reference output")
    aggregate = read_json(aggregate_file, {})
    comparison = aggregate.get(section or "comparison", {})
    checkpoints = {}
    for method in methods:
        current_ids = {str(row["checkpoint_id_per_seed"][0]) for row in comparison.get("config_summary", [])
                       if row.get("method") == method and config_key(row["config"]) in REPRODUCTION_CONFIGS}
        if len(current_ids) != 1:
            raise ValueError(f"Current comparison must identify one seed-0 final checkpoint for {method}")
        identity = current_ids.pop()
        matches = [row for row in summary.get("checkpoints", []) if row.get("method") == method and
                   int(row.get("seed", -1)) == 0 and row.get("kind") == "final" and
                   str(row.get("artifact_id")) == identity]
        if len(matches) != 1:
            raise ValueError(f"Current final checkpoint metadata is ambiguous for {method}/{identity}")
        metadata = matches[0]
        if method in ("spectra", "transfqmix") and metadata.get("training_source") != "main1013_base":
            raise ValueError(f"{method} reproduction requires the current 19D main1013_base checkpoint")
        checkpoints[method] = dict(metadata, checkpoint_id=identity,
                                   checkpoint_path=str(_reference_checkpoint_path(reference, metadata)))
    wanted_methods = set(methods) | {"rule_nv1", "random"}
    selected = {}
    for row in iter_records(reference, "episodes", run="train", seed=0, env="had"):
        method = row.get("method")
        if method not in wanted_methods or int(row.get("episode_seed", -1)) not in REPRODUCTION_SEEDS:
            continue
        config = config_key(row["config"])
        if config not in REPRODUCTION_CONFIGS:
            continue
        anchor = method in ("rule_nv1", "random")
        if anchor:
            if row.get("phase") != "anchor":
                continue
        elif (row.get("phase") != "final_eval" or row.get("arm") not in ("final", "standard") or
              str(row.get("checkpoint_id")) != checkpoints[method]["checkpoint_id"] or
              not str(row.get("checkpoint", "")).startswith("final@")):
            continue
        if method in ("spectra", "transfqmix") and row.get("source_version") != "main1013_base":
            continue
        key = method, config, int(row["episode_seed"])
        if key in selected:
            fields = ("D", "ep_len", "checkpoint_id", "protocol", "archive_protocol", "source_version")
            if any(selected[key].get(field) != row.get(field) for field in fields):
                raise ValueError(f"Conflicting retained reference records: {key}")
        selected[key] = row
    jobs = []
    for method in (*methods, "rule_nv1", "random"):
        for config in REPRODUCTION_CONFIGS:
            for episode_seed in REPRODUCTION_SEEDS:
                key = method, config, episode_seed
                if key not in selected:
                    raise ValueError(f"Approved reproduction opening is absent from retained CSV: {key}")
                jobs.append(dict(method=method, seed=0, config=config_dict(config), episode_seed=episode_seed,
                                 reference=selected[key], metadata=checkpoints.get(method)))
    return jobs


def _reproduction_key(row):
    config = config_key(row["config"]) if "config" in row else tuple(int(row[key]) for key in ("N_R", "N_B", "K"))
    return row["method"], int(row.get("seed", 0)), config, int(row["episode_seed"]), row.get("checkpoint_id")


def _read_reproduction(path):
    if not Path(path).exists():
        return []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return [{key: json.loads(value) for key, value in row.items()} for row in csv.DictReader(stream)]


def _write_reproduction(path, rows):
    """Replace the single local merged CSV after every complete opening."""
    rows = list(rows)
    if not rows:
        return
    path = Path(path)
    pending = path.with_name("." + path.name + ".pending")
    try:
        with pending.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows({key: json.dumps(value, ensure_ascii=False, allow_nan=False) for key, value in row.items()}
                            for row in rows)
            stream.flush()
            os.fsync(stream.fileno())
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def run_reproduction(reference, output, *, methods=None, device="cpu", stop_requested=None, on_progress=None):
    """Run only the retained 2-config x 3-opening cases, recording actual deltas."""
    from open_score.algos import load_policy, _set_seed
    from open_score.algos.rules import register_end_to_end_policy, run_episode
    from .anchors import run_anchor_case, BLUE_STRATEGY
    from ..experiment import atomic_json, check_environment, read_json, process_lock
    import torch
    reference, output = Path(reference).resolve(), Path(output).resolve()
    if output.is_relative_to(reference):
        raise ValueError("Reproduction output must be outside the retained reference archive")
    environment = check_environment()
    jobs = reproduction_jobs(reference, methods)
    output.mkdir(parents=True, exist_ok=True)
    with process_lock(output / ".reproduction.lock") as owned:
        if not owned:
            raise RuntimeError("This output already has a reproduction worker")
        csv_path = output / "reproduction.csv"
        wanted = {}
        for job in jobs:
            meta = job["metadata"]
            identity = meta["checkpoint_id"] if meta else None
            wanted[_reproduction_key(dict(job, checkpoint_id=identity))] = job
        records = {}
        for row in _read_reproduction(csv_path):
            key = _reproduction_key(row)
            if key not in wanted:
                raise ValueError("Existing reproduction CSV contains another case selection; preserved unchanged")
            old = wanted[key]["reference"]
            if (row.get("reference_D") != float(old["D"]) or row.get("reference_ep_len") != int(old["ep_len"]) or
                row.get("reference_protocol") != old.get("protocol") or
                row.get("archive_protocol") != old.get("archive_protocol") or
                row.get("rng_policy") != "reset-python-numpy-torch-to-episode-seed-v1" or
                row.get("physics_protocol") != environment["physics_protocol"] or row.get("device") != device):
                raise ValueError("Existing reproduction case has incompatible reference/runtime identity; preserved unchanged")
            records[key] = row
        started, initial = time.monotonic(), len(records)
        def publish(status, error=None):
            rows = list(records.values())
            elapsed = time.monotonic() - started
            result = dict(status=status, completed=len(rows), total=len(jobs), reference_output=str(reference),
                csv=str(csv_path), device=device, environment=environment,
                configurations=[config_dict(c) for c in REPRODUCTION_CONFIGS], episode_seeds=list(REPRODUCTION_SEEDS),
                training_seed=0, checkpoint_kind="final", rng_policy="reset-python-numpy-torch-to-episode-seed-v1",
                protocol_known=sum(bool(r["protocol_known"]) for r in rows),
                protocol_unknown=sum(not r["protocol_known"] for r in rows),
                exact_D_matches=sum(r["exact_D_match"] for r in rows),
                exact_ep_len_matches=sum(r["exact_ep_len_match"] for r in rows),
                exact_both_matches=sum(r["exact_D_match"] and r["exact_ep_len_match"] for r in rows),
                max_abs_delta_D=max((abs(r["delta_D"]) for r in rows), default=None),
                max_abs_delta_ep_len=max((abs(r["delta_ep_len"]) for r in rows), default=None),
                elapsed_seconds=elapsed, updated_at=time.time(),
                remaining_seconds=(len(jobs)-len(rows))*elapsed/(len(rows)-initial) if len(rows)>initial else None)
            if error:
                result["error"] = error
            summary = read_json(output / "summary.json", {})
            summary["reproduction"] = result
            atomic_json(output / "summary.json", summary)
            atomic_json(output / "reproduction.evaluation.json", result)
            if on_progress is not None:
                on_progress(result)
            return result
        current_method, policy, policy_name, saved_cfg = None, None, None, None
        publish("running")
        try:
            for job in jobs:
                method, meta, ref = job["method"], job["metadata"], job["reference"]
                identity = meta["checkpoint_id"] if meta else None
                key = _reproduction_key(dict(job, checkpoint_id=identity))
                if key in records:
                    continue
                if stop_requested is not None and stop_requested():
                    return publish("stopped")
                if meta is not None and current_method != method:
                    checkpoint = Path(meta["checkpoint_path"])
                    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                    saved_cfg, progress = saved["config"], saved["progress"]
                    if (checkpoint_identity(saved, checkpoint, reference) != identity or
                        saved_cfg.get("method") != method or int(saved_cfg.get("seed", -1)) != 0 or
                        checkpoint.name != "final.pt" or int(progress["t_env"]) < int(saved_cfg["t_max"])):
                        raise ValueError(f"Retained current final identity differs: {checkpoint}")
                    if method in ("spectra", "transfqmix") and not saved_cfg.get("entity_last_action"):
                        raise ValueError(f"{method} must use the current 19D last-action input")
                    del saved
                    policy = load_policy(method, checkpoint)
                    policy.set_device(device)
                    policy_name = f"had_reproduction_{next(_POLICY_IDS)}"
                    register_end_to_end_policy("red", policy_name, method, lambda _, chosen=policy: chosen)
                    current_method = method
                # Reset after model construction, so every opening deliberately
                # uses a recorded global RNG state as well as the env seed.
                _set_seed(job["episode_seed"], str(device).startswith("cuda"))
                began = time.monotonic()
                if meta is None:
                    result = run_anchor_case(method, job["config"], job["episode_seed"])
                    protocol = environment["physics_protocol"]
                else:
                    red, blue, targets = config_key(job["config"])
                    result = run_episode(red=red, blue=blue, targets=targets, seed=job["episode_seed"],
                        red_strategy={"architecture": "end_to_end", "policy": policy_name},
                        blue_strategy=BLUE_STRATEGY, max_steps=int(saved_cfg.get("episode_limit", 100)),
                        record=False, task_mode="damage", spatial_dim=2, target_initialization="random",
                        diagnostics=True, record_events=False, retain_trajectory=False)
                    protocol = saved_cfg.get("implementation_revision")
                if result.get("physics_protocol") != environment["physics_protocol"]:
                    raise ValueError("Reproduction physics identity changed; opening was not committed")
                actual = result["episode_summary"]
                damage, length = float(actual["D"]), int(actual["ep_len"])
                if not math.isfinite(damage):
                    raise ValueError("Reproduction returned non-finite damage")
                archive_protocol = ref.get("archive_protocol")
                known = bool(ref.get("protocol") or (archive_protocol and archive_protocol != "legacy_had_protocol_unspecified"))
                reference_protocol = ref.get("protocol") or (archive_protocol if known else None)
                red, blue, targets = config_key(job["config"])
                row = dict(method=method, public_method="qmix" if method == "b2_qmix_atten" else method,
                    seed=0, N_R=red, N_B=blue, K=targets, episode_seed=job["episode_seed"],
                    phase=ref["phase"], checkpoint=ref.get("checkpoint"), checkpoint_id=identity,
                    reference_checkpoint_id=ref.get("checkpoint_id"), checkpoint_id_match=identity == ref.get("checkpoint_id"),
                    archived_checkpoint=meta["checkpoint_path"] if meta else None,
                    protocol=protocol, reference_protocol=ref.get("protocol"), archive_protocol=archive_protocol,
                    protocol_known=known, protocol_match=(protocol == reference_protocol) if known else None,
                    reference_source_version=ref.get("source_version"), physics_protocol=environment["physics_protocol"],
                    had_core_version=environment["had_core_version"], device=device,
                    rng_seed=job["episode_seed"], rng_policy="reset-python-numpy-torch-to-episode-seed-v1",
                    reference_D=float(ref["D"]), D=damage, delta_D=damage-float(ref["D"]),
                    reference_ep_len=int(ref["ep_len"]), ep_len=length, delta_ep_len=length-int(ref["ep_len"]),
                    exact_D_match=damage == float(ref["D"]), exact_ep_len_match=length == int(ref["ep_len"]),
                    episode_wall_seconds=time.monotonic()-began)
                records[key] = row
                _write_reproduction(csv_path, records.values())
                publish("running")
                print(f"reproduce {method}/s0 {red}v{blue} K{targets} opening={job['episode_seed']} "
                      f"D={damage:.12g} delta_D={row['delta_D']:.12g} delta_ep_len={row['delta_ep_len']} "
                      f"({len(records)}/{len(jobs)})", flush=True)
            return publish("complete")
        except BaseException as error:
            publish("stopped" if isinstance(error, KeyboardInterrupt) else "failed", str(error))
            raise
