"""Frozen main0921 protocol, artifact identity and version-local inventory.

This module supplies data to the existing train/eval runners. It never starts
workers or changes another experiment directory implicitly.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import uuid

_CHECKPOINT_CACHE = {}

PROFILE = "main0921"
SEEDS = (0, 1, 2)
OLD_METHODS = ("regir", "refil", "b2_qmix_atten", "dcg", "spectra", "alma",
               "regir_norefil", "regir_nocount", "regir_r1", "regir_last", "refil_matched")
NEW_METHODS = ("transfqmix", "regir_fixed4", "regir_untied4", "regir_kv0")
SMAC_METHODS = ("regir", "regir_r1", "refil", "transfqmix")
SMAC_VALIDATION = tuple((n, n, 0) for n in (4, 6, 8, 10))
SMAC_FINAL = tuple((n, n, 0) for n in (5, 10, 12, 15, 20)) + ((10, 11, 0), (10, 12, 0), (10, 15, 0))
DEPTH_CONFIGS = ((10, 10, 2), (30, 30, 2), (50, 50, 2))
READOUT_CONFIGS = ((10, 10, 2), (50, 50, 2), (30, 30, 12))
PROBE_CONFIGS = ((10, 10, 2), (10, 10, 3), (30, 30, 2), (50, 50, 2), (30, 30, 12))
READOUTS = ("learned", "read1", "read2", "read3", "read4", "uniform")
COST_METHODS = ("refil", "refil_matched", "regir_r1", "regir", "regir_fixed4",
                "regir_untied4", "regir_kv0", "transfqmix")


def is_profile(output):
    root = Path(output)
    if root.name == PROFILE:
        return True
    path = root / "experiment.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")).get("profile") == PROFILE
    return root.name == PROFILE


def methods(env="had"):
    if env not in ("had", "smacv2"):
        raise ValueError(f"Unsupported {PROFILE} environment: {env}")
    return OLD_METHODS + NEW_METHODS if env == "had" else SMAC_METHODS


def budget(env="had"):
    methods(env)
    return 1_000_000 if env == "had" else 4_000_000


def run_directory(output, method, seed, env="had", run="train"):
    root = Path(output)
    if env != "had":
        root /= env
    return root / method / run / f"seed_{int(seed)}"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.pending")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def initialize(output):
    from .protocol import FINAL_CONFIGS, VALIDATION_CONFIGS
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "experiment.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("profile") != PROFILE or saved.get("seeds") != list(SEEDS):
            raise ValueError("Existing experiment identity or seeds do not match main0921")
        return saved
    if any((root / f"{stream}.csv").exists() for stream in ("episodes", "learning", "progress")):
        raise ValueError("Cannot relabel an existing experiment as main0921; select an independent output directory")
    data = dict(profile=PROFILE, schema_version=2, seeds=list(SEEDS), run="train",
                official_checkpoint="final", episodes_per_config=300,
                environments={
                    "had": dict(methods=list(methods()), new_methods=list(NEW_METHODS), budget=budget(),
                                train_N=[4, 6, 8, 10], train_K=[1, 2, 3],
                                validation_configs=VALIDATION_CONFIGS, validation_points=50,
                                validation_episodes_per_config=25, final_configs=FINAL_CONFIGS,
                                metric="D", direction="min"),
                    "smacv2": dict(methods=list(SMAC_METHODS), budget=budget("smacv2"),
                                   validation_configs=SMAC_VALIDATION, validation_points=40,
                                   validation_episodes_per_config=32, final_configs=SMAC_FINAL,
                                   metric="battle_won", direction="max",
                                   upstream_commit="577ab5a2cff2391f8df582da5731ea9cd6adf3c6",
                                   game_version="4.10.0", state_last_action=False)},
                mechanisms=dict(depth_configs=DEPTH_CONFIGS, depths=[1, 2, 3, 4, 5, 6],
                                readout_configs=READOUT_CONFIGS, readouts=READOUTS,
                                probe_configs=PROBE_CONFIGS, probe_episodes_per_config=100,
                                probe_behavior_methods=["regir", "refil"], probe_behavior_seed=0,
                                ridge_alphas=[.0001, .001, .01, .1, 1, 10, 100],
                                read1_equivalence_verified=False,
                                timing_methods=COST_METHODS, timing_warmup=50, timing_repeats=200),
                resources=dict(gpu_ids=[0, 1], gpu_train_max=2, gpu_per_card_max=1, gpu_reserve_gib=8,
                               cuda_oom_auto_retries=3, cuda_oom_backoff_seconds=[60, 120, 240],
                               gpu_peak_multiplier=1.25, cpu_total_max=16,
                               cpu_final_max=12, cpu_mechanism_max=4, smac_cpu_max=4,
                               had_workers=8, smac_workers=4),
                sources=dict(transfqmix_commit="2ef0a0726f1f186097b4b560509ceb5a801fdafe"),
                imports=[])
    atomic_json(path, data)
    return data


def config_dict(config):
    if isinstance(config, dict):
        return {key: int(config[key]) for key in ("N_R", "N_B", "K")}
    return dict(zip(("N_R", "N_B", "K"), map(int, config)))


def configs(env="had", validation=False):
    if env == "smacv2":
        return SMAC_VALIDATION if validation else SMAC_FINAL
    from .protocol import FINAL_CONFIGS, VALIDATION_CONFIGS
    return VALIDATION_CONFIGS if validation else FINAL_CONFIGS


def validation_thresholds(env):
    count = 50 if env == "had" else 40
    return [math.ceil(budget(env) * i / count) for i in range(1, count + 1)]


def validation_jobs(env, point, t_env):
    count = 25 if env == "had" else 32
    base = 9500 if env == "had" else 100_000
    return [dict(env=env, config=config_dict(c), episode_seed=base + ci * count + ei,
                 phase="train_eval", eval_point=int(point), t_env=int(t_env),
                 retain_trajectory=False, arm="validation")
            for ci, c in enumerate(configs(env, True)) for ei in range(count)]


def validation_score(env, rows):
    expected = {(tuple(j["config"].values()), j["episode_seed"]) for j in validation_jobs(env, 1, 0)}
    selected = {(tuple(config_dict(r["config"]).values()), int(r["episode_seed"])): r for r in rows}
    if set(selected) != expected:
        return None
    metric = "D" if env == "had" else "battle_won"
    values = [float(r[metric]) for r in selected.values()]
    return sum(values) / len(values) if all(math.isfinite(v) for v in values) else None


def checkpoint_info(path, *, method=None, seed=None, env=None):
    """Read the actual retained last iterate; never fall back to best."""
    from open_score.algos import setup_runtime, canonical_method
    setup_runtime()
    import torch
    path = Path(path)
    if path.name != "final.pt" or not path.is_file():
        raise ValueError(f"Official final.pt missing: {path}")
    stat = path.stat()
    cache_key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, method, seed, env)
    if cache_key in _CHECKPOINT_CACHE:
        return dict(_CHECKPOINT_CACHE[cache_key])
    saved = torch.load(path, map_location="cpu", weights_only=False)
    cfg, progress = saved["config"], saved["progress"]
    actual_env = cfg.get("env", "had")
    actual_method = canonical_method(cfg["method"])
    actual_seed = int(cfg["seed"])
    if actual_seed not in SEEDS or actual_method not in methods(actual_env):
        raise ValueError("Checkpoint is outside the frozen method/seed matrix")
    if method is not None and canonical_method(method) != actual_method:
        raise ValueError("Checkpoint method mismatch")
    if seed is not None and int(seed) != actual_seed:
        raise ValueError("Checkpoint training seed mismatch")
    if env is not None and env != actual_env:
        raise ValueError("Checkpoint environment mismatch")
    actual = int(progress["t_env"])
    if int(cfg["t_max"]) != budget(actual_env) or actual < budget(actual_env):
        raise ValueError(f"Incomplete/incompatible final budget: {actual}/{cfg['t_max']}")
    if progress.get("status") not in ("completed", "complete"):
        raise ValueError("Final checkpoint does not contain a completed training state")
    artifact = saved.get("artifact_id")
    if not artifact:
        raise ValueError("Final checkpoint has no immutable artifact identity; import it first")
    if cfg.get("profile") == PROFILE and cfg.get("run") != "train":
        raise ValueError("Formal main0921 run must be train")
    result = dict(env=actual_env, method=actual_method, seed=actual_seed,
                  t_env=actual, checkpoint_id=artifact, checkpoint=f"final@{actual}",
                  path=str(path), config=cfg)
    _CHECKPOINT_CACHE[cache_key] = result
    return dict(result)


def evaluation_jobs(info, kind="final"):
    env = info["env"]
    if kind == "final":
        specifications = [(c, None, "learned", "final") for c in configs(env)]
    elif kind == "depth" and env == "had":
        specifications = [(c, depth, "learned", f"depth:R{depth}")
                          for depth in (1, 2, 3, 4, 5, 6) for c in DEPTH_CONFIGS]
    elif kind == "readout" and env == "had":
        specifications = [(c, 4, readout, f"readout:{readout}")
                          for readout in READOUTS for c in READOUT_CONFIGS]
    else:
        raise ValueError(f"Invalid evaluation kind: {kind}/{env}")
    base = 9000 if env == "had" else 110_000
    jobs = []
    for config, depth, readout, arm in specifications:
        for offset in range(300):
            jobs.append(dict(env=env, phase=f"{kind}_eval", eval_point=50 if env == "had" else 40,
                             config=config_dict(config), episode_seed=base + offset,
                             t_env=info["t_env"], checkpoint=info["checkpoint"],
                             checkpoint_id=info["checkpoint_id"], arm=arm,
                             cycle_depth=depth, readout=readout,
                             retain_trajectory=kind == "final" and offset < 2))
    return jobs


def result_identity(row):
    return (row.get("env", "had"), row.get("checkpoint_id"), row.get("arm"),
            tuple(config_dict(row["config"]).values()), int(row["episode_seed"]))


def remaining_evaluations(info, kind, rows, *, read1_equivalent=False):
    relevant = [r for r in rows if r.get("checkpoint_id") == info["checkpoint_id"]]
    done = {result_identity(r) for r in relevant}
    for row in relevant:
        arm = row.get("arm")
        aliases = []
        if arm == "final" and info["method"] == "regir":
            aliases = ["depth:R4", "readout:learned"]
        if arm == "depth:R1" and read1_equivalent:
            aliases.append("readout:read1")
        if arm == "readout:read1" and read1_equivalent:
            aliases.append("depth:R1")
        done.update(result_identity({**row, "arm": alias}) for alias in aliases)
    return [j for j in evaluation_jobs(info, kind) if result_identity(j) not in done]


def new_artifact_id():
    return uuid.uuid4().hex


def import_existing(output, source):
    """Copy qualified assets once; legacy best evaluations cannot satisfy final."""
    import shutil
    from open_score.algos import setup_runtime, _atomic_save
    from open_score.utils.logging import iter_records, append_records
    setup_runtime()
    import torch
    root, source = Path(output), Path(source)
    if root.resolve() == source.resolve():
        raise ValueError("Import destination must be independent of source")
    manifest = initialize(root)
    if manifest.get("imports"):
        return manifest["imports"]
    if any((root / f"{name}.csv").exists() for name in ("episodes", "learning", "progress")):
        raise ValueError("Incomplete import exists; finish/reconcile its pending files before restarting")
    infos, imported = {}, []
    for method in OLD_METHODS:
        for seed in SEEDS:
            origin = run_directory(source, method, seed)
            destination = run_directory(root, method, seed)
            final_source = origin / "final.pt"
            if not final_source.exists() and (method, seed) in (("refil", 0), ("b2_qmix_atten", 0)):
                final_source = origin / "latest.pt"
            if not final_source.exists():
                raise FileNotFoundError(f"Cannot import final: {final_source}")
            saved = torch.load(final_source, map_location="cpu", weights_only=False)
            actual = int(saved["progress"]["t_env"])
            if actual < budget() or saved["progress"].get("status") not in ("completed", "complete"):
                raise ValueError(f"Incomplete source training: {final_source}")
            destination.mkdir(parents=True, exist_ok=True)
            for filename in ("config.json", "best.pt", "console.log", "eval.console.log"):
                path = origin / filename
                if path.exists():
                    shutil.copy2(path, destination / filename)
            saved["artifact_id"] = new_artifact_id()
            saved["source"] = dict(version="main", checkpoint=str(final_source.relative_to(source)),
                                   normalization="completed latest to final" if final_source.name == "latest.pt" else "copied final")
            # Do not copy a replay buffer into a finished deployment artifact.
            for key in ("replay", "runner", "rng"):
                saved.pop(key, None)
            _atomic_save(saved, destination / "final.pt")
            info = checkpoint_info(destination / "final.pt", method=method, seed=seed, env="had")
            infos[(method, seed)] = info
            imported.append(dict(method=method, seed=seed, t_env=actual,
                                 checkpoint_id=info["checkpoint_id"], **saved["source"]))
    from .protocol import FINAL_CONFIGS
    final_configs = set(FINAL_CONFIGS)
    counts = {}
    for stream in ("episodes", "learning", "progress", "trajectories"):
        rows, count = [], 0
        for row in iter_records(source, stream, run="train"):
            key = (row.get("method"), row.get("seed"))
            anchor = stream == "episodes" and row.get("phase") == "anchor"
            if key not in infos and not anchor:
                continue
            item = dict(row, version=PROFILE, env="had", source_version=row.get("version"))
            info = infos.get(key)
            if stream == "episodes":
                config = tuple(config_dict(item["config"]).values())
                if anchor:
                    if config not in final_configs or not 9000 <= int(item["episode_seed"]) < 9300:
                        continue
                    item.update(arm="anchor", checkpoint_id=None)
                elif item.get("phase") == "train_eval":
                    item.update(arm="validation", checkpoint_id=None)
                elif item.get("phase") == "final_eval":
                    if key in (("refil", 0), ("b2_qmix_atten", 0)):
                        continue
                    if (config not in final_configs or item.get("checkpoint") != info["checkpoint"]
                            or not 9000 <= int(item["episode_seed"]) < 9300):
                        continue
                    item.update(arm="final", checkpoint_id=info["checkpoint_id"])
                else:
                    continue
            elif stream == "trajectories":
                # Only correctly identified final episodes; no old attention
                # snapshots or best-labelled depth scans become new evidence.
                if (item.get("phase") != "final_eval" or key in (("refil", 0), ("b2_qmix_atten", 0))
                        or item.get("checkpoint") != info["checkpoint"]):
                    continue
                item.update(arm="final", checkpoint_id=info["checkpoint_id"])
            rows.append(item)
            if len(rows) >= 1000:
                append_records(root, stream, rows); count += len(rows); rows.clear()
        if rows:
            append_records(root, stream, rows); count += len(rows)
        counts[stream] = count
    manifest["imports"] = imported
    manifest["import_record_counts"] = counts
    atomic_json(root / "experiment.json", manifest)
    scan(root)
    return imported


def scan(output, env=None, only=None, probe_collector_seed=0):
    """Generate a bounded queue from frozen seeds and actual artifact coverage."""
    from open_score.utils.logging import iter_records
    root = Path(output)
    manifest = initialize(root)
    rows_by_run = {}
    for row in iter_records(root, "episodes", run="train", env=env):
        if row.get("phase") not in ("final_eval", "depth_eval", "readout_eval"):
            continue
        key = (row.get("env"), row.get("method"), row.get("seed"))
        rows_by_run.setdefault(key, []).append(row)
    checkpoints, tasks = [], []
    for domain in ((env,) if env else ("had", "smacv2")):
        for method in methods(domain):
            for seed in SEEDS:
                directory = run_directory(root, method, seed, domain)
                info = None
                reason = "final checkpoint not yet available"
                if (directory / "final.pt").exists():
                    try:
                        info = checkpoint_info(directory / "final.pt", method=method, seed=seed, env=domain)
                    except (ValueError, KeyError, OSError) as error:
                        reason = str(error)
                task = dict(id=f"train.{domain}.{method}.s{seed}", kind="train", env=domain,
                            method=method, seed=seed, status="complete" if info else "pending",
                            completed=info["t_env"] if info else 0, total=budget(domain),
                            detail="qualified final" if info else reason)
                tasks.append(task)
                if info:
                    checkpoints.append({k: v for k, v in info.items() if k != "config"})
                kinds = ("final", "depth", "readout") if domain == "had" and method == "regir" else ("final",)
                depth_pending = False
                final_pending = False
                for kind in kinds:
                    if not info:
                        tasks.append(dict(id=f"eval.{kind}.{domain}.{method}.s{seed}", kind=kind,
                                          env=domain, method=method, seed=seed, status="waiting",
                                          completed=0, total=(len(configs(domain)) if kind == "final" else 18) * 300,
                                          detail=reason))
                        continue
                    previous = rows_by_run.get((domain, method, seed), [])
                    jobs = evaluation_jobs(info, kind)
                    pending = remaining_evaluations(info, kind, previous,
                        read1_equivalent=manifest["mechanisms"]["read1_equivalence_verified"])
                    if kind == "depth":
                        depth_pending = bool(pending)
                    if kind == "final":
                        final_pending = bool(pending)
                    waiting_for_depth = (kind == "readout" and depth_pending
                                         and (only is None or "depth" in only))
                    waiting_for_final = (kind in ("depth", "readout") and final_pending
                                         and (only is None or "final" in only))
                    tasks.append(dict(id=f"eval.{kind}.{domain}.{method}.s{seed}", kind=kind,
                                      env=domain, method=method, seed=seed,
                                      status=("waiting" if waiting_for_depth or waiting_for_final else "pending") if pending else "complete",
                                      completed=len(jobs)-len(pending), total=len(jobs),
                                      detail=("reuse completed final arms first" if waiting_for_final else
                                              "reuse completed depth arms first" if waiting_for_depth else "frozen final"),
                                      checkpoint=info["path"], checkpoint_id=info["checkpoint_id"]))
        if domain == "had":
            for seed in SEEDS:
                marker = root / "probe" / f"seed_{seed}.complete.json"
                complete = False
                if marker.exists():
                    finished = json.loads(marker.read_text(encoding="utf-8"))
                    actual = checkpoint_info(run_directory(root, "regir", seed) / "final.pt")
                    complete = (finished.get("status") == "complete"
                                and finished.get("checkpoint_id") == actual["checkpoint_id"])
                available = all(run_directory(root, m, s) .joinpath("final.pt").exists()
                                for m, s in (("regir", seed), ("regir", 0), ("refil", 0)))
                bank_ready = (root / "probe" / "collection_complete.json").exists()
                ready = available and (seed == probe_collector_seed or bank_ready)
                tasks.append(dict(id=f"eval.probe.had.regir.s{seed}", kind="probe", env="had",
                                  method="regir", seed=seed,
                                  status="complete" if complete else "pending" if ready else "waiting",
                                  completed=int(complete), total=1, detail="fixed common states, seed-specific probes"))
            cost_ids = {(r["method"], r["seed"]): r["checkpoint_id"] for r in checkpoints
                        if r["env"] == "had" and r["method"] in COST_METHODS}
            wanted = {(artifact, (n, n, 2)) for artifact in cost_ids.values() for n in (10, 50)}
            done = set()
            for row in iter_records(root, "timing", run="train", env="had"):
                key = (row.get("checkpoint_id"), tuple(config_dict(row["config"]).values())) if row.get("config") else None
                if (key in wanted and row.get("arm") == "cost" and row.get("device") == "cuda:0"
                        and row.get("physical_gpu") == manifest["resources"].get("measurement_gpu")
                        and row.get("physical_gpu") in (0, 1)
                        and row.get("n_steps") == 200 and all(row.get(k) is not None and math.isfinite(float(row[k]))
                            for k in ("ms_per_step", "p25_ms", "p75_ms", "p95_ms", "actor_params", "training_params"))):
                    done.add(key)
            cost_total = len(COST_METHODS) * len(SEEDS) * 2
            ready = len(cost_ids) == len(COST_METHODS) * len(SEEDS) and (root / "probe/collection_complete.json").exists()
            tasks.append(dict(id="eval.timing.had.all", kind="timing", env="had", method=None, seed=None,
                              status="complete" if len(done) == cost_total else "pending" if ready else "waiting",
                              completed=len(done), total=cost_total, detail="one registered GPU; common state bank"))
    value = dict(profile=PROFILE, seeds=list(SEEDS), checkpoints=checkpoints, tasks=tasks)
    # A filtered scan must never overwrite the other environment's inventory.
    if env is None:
        atomic_json(root / "inventory.json", value)
    return value


def validate_integration(output, env="had"):
    """Audit the frozen protocol and recorded numerical/native acceptance in place."""
    from open_score.algos import load_config
    from open_score.utils.logging import schema_for
    import csv
    root = Path(output)
    manifest = initialize(root)
    assert tuple(manifest["seeds"]) == SEEDS
    assert len(configs("had")) == 24 and len(configs("smacv2")) == 8
    assert len(validation_jobs("had", 1, 0)) == 100
    assert len(validation_jobs("smacv2", 1, 0)) == 128
    assert validation_thresholds("had")[-1] == 1_000_000
    assert validation_thresholds("smacv2")[-1] == 4_000_000
    for method in methods(env):
        args = load_config(method, dict(profile=PROFILE, output=str(root), env=env,
            t_max=budget(env), run="train", seed=0, use_cuda=False,
            batch_size_run=8 if env == "had" else 4))
        assert args.device == "cpu" and args.run == "train"
        if env == "smacv2":
            assert not any((args.entity_last_action, args.obs_last_action, args.obs_agent_id,
                            args.env_args["obs_last_action"], args.env_args["state_last_action"]))
    for stream in ("episodes", "learning", "progress", "timing", "trajectories"):
        path = root / f"{stream}.csv"
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                assert next(csv.reader(handle)) == list(schema_for(root, stream)), f"{stream} header differs"
    acceptance = json.loads((root / "implementation_acceptance.json").read_text(encoding="utf-8"))
    assert acceptance["status"] == "passed"
    if env == "had":
        assert {r["method"] for r in acceptance["results"] if r["status"] == "passed"} == set(NEW_METHODS)
        assert all(r["checkpoint_restore_exact"] and r["resumed_next_update_exact"] for r in acceptance["results"])
    else:
        native = json.loads((root / "smacv2/scenes/native_acceptance.json").read_text(encoding="utf-8"))
        if (not native.get("complete") or native.get("passed") != 33 or len(native.get("rows", [])) != 33
                or any(r.get("status") != "passed" for r in native["rows"])):
            raise ValueError("Real SC2 acceptance is incomplete")
        from open_score.envs.smacv2_env import generate_registered_scenes
        assert len(generate_registered_scenes(root)["scenes"]) == 2400
    inventory = scan(root)
    print(f"{PROFILE} {env}: protocol and recorded implementation acceptance passed; "
          f"{len(inventory['checkpoints'])} qualified finals, seeds={SEEDS}")
    return inventory
