"""Frozen validation and final-evaluation quotas; no environment construction."""
from __future__ import annotations

import math
from itertools import count
from pathlib import Path
import time

from ..envs.scales import SCALE_POOLS, TEST_POOL

# Checkpoint selection may only ever see these; they are inside the training pool.
VALIDATION_CONFIGS = ((4, 4, 2), (6, 6, 2), (8, 8, 2), (10, 10, 3))
EQUAL_SCALE_CONFIGS = tuple((s.N_R, s.N_B, s.K) for s in SCALE_POOLS["extrapolation_agents"])
RATIO_CONFIGS = tuple((s.N_R, s.N_B, s.K) for s in SCALE_POOLS["extrapolation_ratio"])
TARGET_K_VALUES = (2, 4, 6)
TARGET_SCALE_NS = (10, 15, 20, 30)
# scales.py is the source of truth for the three reported axes.
TEST_CONFIGS = tuple((scale.N_R, scale.N_B, scale.K) for scale in TEST_POOL)
# Pool-in plus the three extrapolation axes. 10v10 K2 sits on the equal-scale
# line; 10v10 K3 stays in the validation set.
FINAL_CONFIGS = tuple(dict.fromkeys(VALIDATION_CONFIGS + TEST_CONFIGS))
VALIDATION_EPISODES_PER_CONFIG = 25
FINAL_EPISODES_PER_CONFIG = 300
FINAL_EPISODES = len(FINAL_CONFIGS) * FINAL_EPISODES_PER_CONFIG
# Ordered fixed-slot baselines are sized to the training pool, so they have no
# parameters for a larger roster and are reported on the pool axis only.
POOL_ONLY_METHODS = ("b0_qmix",)
_POLICY_IDS = count()


def final_configs(method=None):
    return VALIDATION_CONFIGS if method in POOL_ONLY_METHODS else FINAL_CONFIGS


def final_episodes(method=None):
    return len(final_configs(method)) * FINAL_EPISODES_PER_CONFIG


def config_dict(config):
    if isinstance(config, dict):
        return {key: int(config[key]) for key in ("N_R", "N_B", "K")}
    red, blue, targets = config
    return {"N_R": int(red), "N_B": int(blue), "K": int(targets)}


def config_key(config):
    value = config_dict(config)
    return value["N_R"], value["N_B"], value["K"]


def config_label(config):
    red, blue, targets = config_key(config)
    return f"{red}v{blue} K{targets}"


def evaluation_thresholds(budget):
    budget = int(budget)
    if budget < 50:
        raise ValueError("A formal 50-point budget must be at least 50 physical steps")
    return [math.ceil(budget * point / 50) for point in range(1, 51)]


def validation_jobs(eval_point, t_env=0):
    if not 1 <= int(eval_point) <= 50:
        raise ValueError("eval_point must be 1..50")
    return [dict(config=config_dict(config), episode_seed=9500 + 25 * index + offset,
                 phase="train_eval", eval_point=int(eval_point), t_env=int(t_env),
                 retain_trajectory=False)
            for index, config in enumerate(VALIDATION_CONFIGS) for offset in range(25)]


def checkpoint_tag(stem, t_env):
    """Identify the retained weights, not just the reusable file name.

    ``best.pt`` is overwritten whenever validation improves, so a bare stem
    cannot distinguish two different models and would let an earlier
    evaluation satisfy the quota of a later one.
    """
    return f"{stem}@{int(t_env)}"


def final_jobs(t_env=0, checkpoint="best", method=None):
    tag = checkpoint_tag(checkpoint, t_env)
    return [dict(config=config_dict(config), episode_seed=9000 + offset, phase="final_eval",
                 eval_point=50, t_env=int(t_env), checkpoint=tag,
                 retain_trajectory=offset < 2)
            for config in final_configs(method) for offset in range(FINAL_EPISODES_PER_CONFIG)]


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


def compute_nds(rho, rho_random, rho_rule):
    if any(value is None for value in (rho, rho_random, rho_rule)):
        return None
    denominator = float(rho_random) - float(rho_rule)
    if denominator == 0:
        return None
    return (float(rho_random) - float(rho)) / denominator


def evaluate_checkpoint(method, checkpoint, *, output=None, run=None, seed=None,
                        device="cpu", on_progress=None, stop_requested=None):
    """Evaluate a retained checkpoint through the same rule-policy episode path."""
    import torch
    from open_score.algos import load_policy
    from open_score.rules import register_end_to_end_policy, run_episode
    from open_score.utils.logging import DEFAULT_OUTPUT, FORMAL_RUN, ExperimentLogger, read_records, unique_episodes
    from .anchors import BLUE_STRATEGY
    from .report import refresh_report

    run = FORMAL_RUN if run is None else run
    output = Path(DEFAULT_OUTPUT if output is None else output)
    checkpoint = Path(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actual_seed = int(saved["config"]["seed"])
    if seed is not None and int(seed) != actual_seed:
        raise ValueError("Evaluation seed must identify the checkpoint training seed")
    t_env = int(saved["progress"]["t_env"])
    if run == FORMAL_RUN and checkpoint.stem != "best":
        raise ValueError(f"The formal {FORMAL_RUN} evaluation uses best.pt; use a distinct --run for another checkpoint")
    checkpoint_label = checkpoint_tag(checkpoint.stem, t_env)
    del saved
    policy = load_policy(method, checkpoint)
    if hasattr(policy, "set_device"):
        policy.set_device(device)
    elif device != "cpu":
        raise ValueError("This policy adapter supports CPU evaluation only")
    policy_name = f"crossscale_evaluation_{next(_POLICY_IDS)}"
    register_end_to_end_policy("red", policy_name, f"{method} retained checkpoint", lambda _: policy)
    logger = ExperimentLogger(output, method, actual_seed, run)
    existing = unique_episodes(read_records(output, "episodes", run=run, method=method, seed=actual_seed))
    jobs = remaining_jobs(final_jobs(t_env, checkpoint.stem, method=method), existing)
    total = final_episodes(method)
    completed = total - len(jobs)
    started, last_report, initial = time.monotonic(), time.monotonic(), completed
    try:
        for job in jobs:
            if stop_requested is not None and stop_requested():
                logger.progress(phase="final_eval", status="stopped", completed=completed, total=total,
                                t_env=t_env, checkpoint=checkpoint_label)
                return dict(status="stopped", completed=completed, total=total)
            red, blue, targets = config_key(job["config"])
            result = run_episode(targets=targets, red=red, blue=blue, seed=job["episode_seed"],
                                 red_strategy={"architecture": "end_to_end", "policy": policy_name},
                                 blue_strategy=BLUE_STRATEGY, max_steps=100, record=False,
                                 task_mode="damage", spatial_dim=2, target_initialization="random",
                                 diagnostics=True, record_events=True,
                                 retain_trajectory=job["retain_trajectory"])
            summary = dict(result["episode_summary"])
            summary.update(job)
            if hasattr(policy, "episode_q_statistics"):
                summary.update(policy.episode_q_statistics())
            if job["retain_trajectory"]:
                if "trajectory" not in result:
                    raise RuntimeError("Selected final-evaluation episode has no retained trajectory")
                logger.trajectories([dict(phase="final_eval", config=job["config"], episode_seed=job["episode_seed"],
                                          checkpoint=checkpoint_label, trajectory=result["trajectory"])])
            logger.episodes([summary])
            completed += 1
            elapsed = time.monotonic() - started
            progress = dict(phase="final_eval", status="running", completed=completed, total=total,
                            t_env=t_env, checkpoint=checkpoint_label,
                            remaining_seconds=(total-completed)*elapsed/max(1, completed-initial))
            logger.progress(**progress)
            if on_progress is not None:
                on_progress(progress)
            if completed % 25 == 0:
                print(f"[{method}/{actual_seed}] final_eval {completed}/{total}", flush=True)
            if time.monotonic() - last_report >= 300:
                refresh_report(output, run=run)
                last_report = time.monotonic()
        logger.progress(phase="final_eval", status="complete", completed=completed, total=total,
                        t_env=t_env, checkpoint=checkpoint_label)
    except BaseException as error:
        logger.progress(phase="final_eval", status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        completed=completed, total=total, t_env=t_env, error=str(error), checkpoint=checkpoint_label)
        raise
    finally:
        refresh_report(output, run=run)
    return dict(status="complete", completed=completed, total=total)
