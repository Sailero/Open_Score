"""Frozen HAD validation and final/confirmation evaluation quotas."""
from __future__ import annotations
import math
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
CYCLE_SERIES_METHODS = ("regir_nomem", "regir_sg", "regir_r0_sg", "regir_r1_sg", "regir_norefil_sg", "regir_last_sg", "regir_fixed4_sg", "regir_untied4_sg")
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
    return FINAL_CONFIGS

def final_episodes(method=None):
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
    from .experiment import run_directory as directory
    return directory(output, method, seed)

def checkpoint_identity(saved, checkpoint=None, output=None):
    value = saved.get("artifact_id") or saved.get("progress", {}).get("final_artifact_id")
    if value:
        return str(value)
    if checkpoint is not None and output is not None:
        from .experiment import read_json
        repo = Path(__file__).resolve().parents[3]
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
    import torch
    from open_score.algos import load_policy
    from open_score.utils.logging import DEFAULT_OUTPUT, FORMAL_RUN, ExperimentLogger, read_records
    from .experiment import atomic_json, process_lock, check_environment, ensure_layout, is_sky_environment, register_environment
    output = Path(DEFAULT_OUTPUT if output is None else output)
    ensure_layout(output)
    checkpoint = Path(checkpoint)
    run = FORMAL_RUN if run is None else run
    with process_lock(checkpoint.parent / ".eval.lock") as claimed:
        if not claimed:
            raise RuntimeError(f"Checkpoint already has an evaluator: {checkpoint}")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg, progress = saved["config"], saved["progress"]
        identity_env = check_environment(cfg.get("env_args"))
        fixedwing = is_sky_environment(cfg.get("env_args"))
        retained_env = cfg.get("env_identity")
        if retained_env and any(retained_env.get(key) != identity_env[key] for key in ("had_core_version", "physics_protocol")):
            raise ValueError("Checkpoint physics identity differs; no evaluation sampling started")
        actual_seed, t_env = int(cfg["seed"]), int(progress["t_env"])
        if cfg.get("method") != method or (seed is not None and int(seed) != actual_seed):
            raise ValueError("Evaluation method/seed must identify the retained checkpoint")
        if cfg.get("env") != "had" or checkpoint.name != "final.pt" or t_env < int(cfg.get("t_max", t_env)):
            raise ValueError("Frozen evaluation requires completed HAD final.pt")
        if fixedwing:
            register_environment(output, cfg, cfg, cfg.get("environment_parameters"))
        identity = checkpoint_identity(saved, checkpoint, output)
        protocol = identity_env["physics_protocol"] if fixedwing else cfg.get("implementation_revision", "retained")
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
        compatible = [row for row in retained.values() if row.get("protocol") == protocol]
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
        environment = None
        if fixedwing:
            from open_score.envs.entity_env import HADEntityEnv
            import json
            env_args = dict(cfg.get("env_args") or {})
            for key in ("seed", "scale", "config", "worker_id"):
                env_args.pop(key, None)
            env_args.update(pad="eval", diagnostics=True, retain_trajectory=False,
                            max_steps=int(cfg.get("episode_limit", 100)))
            environment = HADEntityEnv(seed=actual_seed, **env_args)
            info = environment.get_env_info()
            for key in ("entity_shape", "n_actions", "spatial_dim", "feature_names", "had_position_indices", "had_velocity_indices", "had_position_scale", "had_velocity_scale"):
                if json.dumps(cfg.get(key)) != json.dumps(info.get(key)):
                    environment.close()
                    raise ValueError(f"Frozen evaluation {key} differs from the checkpoint")
            expected_parameters = cfg.get("environment_parameters")
            actual_parameters = environment.parameter_snapshot()
            if expected_parameters is None or json.dumps(expected_parameters, sort_keys=True) != json.dumps(actual_parameters, sort_keys=True):
                environment.close()
                raise ValueError("Frozen evaluation effective parameters differ from the checkpoint")
        else:
            from open_score.rules import register_end_to_end_policy, run_episode
            from .anchors import BLUE_STRATEGY
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
                if fixedwing:
                    policy.reset()
                    environment.reset(seed=job["episode_seed"], config=job["config"], evaluate=True)
                    complete = False
                    while not complete:
                        actions = policy.act(environment.get_policy_state(), "red", environment.wrapper.native_action_ids)
                        _, complete, _ = environment.step(environment.wrapper.encode_native_actions(actions))
                    summary = environment.episode_summary()
                    actual_protocol = environment.wrapper.adapter.env.physics_protocol
                else:
                    result = run_episode(red=red, blue=blue, targets=targets, seed=job["episode_seed"],
                        red_strategy={"architecture": "end_to_end", "policy": policy_name},
                        blue_strategy=BLUE_STRATEGY, max_steps=int(cfg.get("episode_limit", 100)),
                        record=False, task_mode="damage", spatial_dim=2, target_initialization="random",
                        diagnostics=True, record_events=False, retain_trajectory=False)
                    summary, actual_protocol = result["episode_summary"], result.get("physics_protocol")
                if actual_protocol != identity_env["physics_protocol"]:
                    raise ValueError("Episode physics identity changed; result was not committed")
                row = dict(summary, **job, episode_wall_seconds=time.monotonic() - before)
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
        finally:
            if environment is not None:
                environment.close()
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
