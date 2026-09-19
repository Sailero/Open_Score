"""Frozen validation and final-evaluation quotas; no environment construction."""
from __future__ import annotations

import math
from itertools import count
from pathlib import Path
import time

from ..envs.scales import SCALE_POOLS, TEST_POOL


def _safe_refresh(output, run=None):
    from .report import refresh_report
    try:
        refresh_report(output, run=run)
    except (OSError, MemoryError, TimeoutError) as error:
        print(f"report refresh skipped: {type(error).__name__}", flush=True)

# Checkpoint selection may only ever see these; they are inside the training pool.
VALIDATION_CONFIGS = ((4, 4, 2), (6, 6, 2), (8, 8, 2), (10, 10, 3))
EQUAL_SCALE_CONFIGS = tuple((s.N_R, s.N_B, s.K) for s in SCALE_POOLS["extrapolation_agents"])
RATIO_CONFIGS = tuple((s.N_R, s.N_B, s.K) for s in SCALE_POOLS["extrapolation_ratio"])
TARGET_K_VALUES = (2, 4, 6, 9, 12)
TARGET_SCALE_NS = (10, 15, 20, 30)
TARGET_PLOT_NS = (10, 30)
# scales.py is the source of truth. 1:2 stays named for legacy CSV rows only.
TEST_CONFIGS = tuple((scale.N_R, scale.N_B, scale.K) for scale in TEST_POOL)
# Pool-in plus equal-scale and target axes. 10v10 K2 sits on the equal-scale
# line; 10v10 K3 stays in the validation set.
FINAL_CONFIGS = tuple(dict.fromkeys(VALIDATION_CONFIGS + TEST_CONFIGS))
VALIDATION_EPISODES_PER_CONFIG = 25
FINAL_EPISODES_PER_CONFIG = 300
FINAL_EPISODES = len(FINAL_CONFIGS) * FINAL_EPISODES_PER_CONFIG
# Ordered fixed-slot baselines are sized to the training pool, so they have no
# parameters for a larger roster and are reported on the pool axis only.
POOL_ONLY_METHODS = ("b0_qmix",)
# Shared SelfAttn cycle / slot / feedback. Cardinality attention has no R.
CYCLE_SERIES_METHODS = ("regir", "refil_cycle", "refil_slot", "refil_feedback")
MECH_CONFIG = (30, 30, 2)
MECH_EPISODE_SEEDS = (9000, 9001)
TIMING_METHODS = ("regir", "refil", "refil_matched", "dcg")
TIMING_SCALES = ((10, 10, 2), (50, 50, 2))
DEPTH_SWEEP_DEPTHS = (1, 2, 3, 4, 5, 6)
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


def is_cycle_series(method):
    return method in CYCLE_SERIES_METHODS


def depth_checkpoint_tag(stem, t_env, depth):
    return f"{checkpoint_tag(stem, t_env)}/R{int(depth)}"


def parse_sweep_depth(checkpoint):
    text = str(checkpoint or "")
    marker = "/R"
    if marker not in text:
        return None
    try:
        return int(text.rsplit(marker, 1)[-1])
    except ValueError:
        return None


def depth_eval_total():
    return len(EQUAL_SCALE_CONFIGS) * FINAL_EPISODES_PER_CONFIG * len(DEPTH_SWEEP_DEPTHS)


def depth_jobs(t_env=0, checkpoint="best", depth=4):
    tag = depth_checkpoint_tag(checkpoint, t_env, depth)
    return [dict(config=config_dict(config), episode_seed=9000 + offset, phase="depth_eval",
                 eval_point=50, t_env=int(t_env), checkpoint=tag, cycle_depth=int(depth),
                 retain_trajectory=False)
            for config in EQUAL_SCALE_CONFIGS for offset in range(FINAL_EPISODES_PER_CONFIG)]


def _reuse_final_as_depth(recorded_rows, t_env, checkpoint="best"):
    """Formal 1:1 final_eval is already greedy R=4; do not rerun those episodes."""
    tag = checkpoint_tag(checkpoint, t_env)
    equal = set(EQUAL_SCALE_CONFIGS)
    reused = []
    for row in recorded_rows:
        if row.get("phase") != "final_eval" or row.get("checkpoint") != tag:
            continue
        if config_key(row["config"]) not in equal:
            continue
        reused.append(dict(row, phase="depth_eval",
                           checkpoint=depth_checkpoint_tag(checkpoint, t_env, 4)))
    return reused


def remaining_depth_jobs(t_env, recorded_rows, checkpoint="best"):
    recorded = list(recorded_rows) + _reuse_final_as_depth(recorded_rows, t_env, checkpoint)
    jobs = []
    for depth in DEPTH_SWEEP_DEPTHS:
        jobs.extend(remaining_jobs(depth_jobs(t_env, checkpoint, depth), recorded))
    return jobs


def depth_eval_finished(output, method, run, seed, t_env, checkpoint="best"):
    from open_score.utils.logging import read_records, unique_episodes
    existing = unique_episodes(read_records(output, "episodes", run=run, method=method, seed=seed))
    return not remaining_depth_jobs(t_env, existing, checkpoint)


def infer_best_t_env(output, method, run, seed, fallback=None, rows=None):
    from open_score.utils.logging import read_records
    source = rows if rows is not None else read_records(output, "episodes", run=run, method=method, seed=seed)
    for row in reversed(source):
        ckpt = str(row.get("checkpoint") or "")
        if "@" not in ckpt:
            continue
        stem, _, rest = ckpt.partition("@")
        if stem != "best":
            continue
        try:
            return int(rest.split("/")[0])
        except ValueError:
            continue
    return None if fallback is None else int(fallback)


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
                            eval_completed=completed, eval_total=total, t_env=t_env,
                            budget_steps=t_env, checkpoint=checkpoint_label,
                            remaining_seconds=(total-completed)*elapsed/max(1, completed-initial))
            logger.progress(**progress)
            if on_progress is not None:
                try:
                    on_progress(dict(progress))
                except Exception as error:
                    print(f"[{method}] final_eval progress callback: {type(error).__name__}: {error}", flush=True)
            if completed % 25 == 0:
                print(f"[{method}/{actual_seed}] final_eval {completed}/{total}", flush=True)
            if time.monotonic() - last_report >= 300:
                _safe_refresh(output, run)
                last_report = time.monotonic()
        logger.progress(phase="final_eval", status="complete", completed=completed, total=total,
                        t_env=t_env, checkpoint=checkpoint_label)
    except BaseException as error:
        logger.progress(phase="final_eval", status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        completed=completed, total=total, t_env=t_env, error=str(error), checkpoint=checkpoint_label)
        raise
    finally:
        _safe_refresh(output, run)
    return dict(status="complete", completed=completed, total=total)


def evaluate_depth_sweep(method, checkpoint, *, output=None, run=None, seed=None,
                         device="cpu", on_progress=None, stop_requested=None):
    """1:1 equal-scale sweep over cycle depths R=1..6 for a retained checkpoint."""
    import torch
    from open_score.algos import load_policy
    from open_score.rules import register_end_to_end_policy, run_episode
    from open_score.utils.logging import DEFAULT_OUTPUT, FORMAL_RUN, ExperimentLogger, read_records, unique_episodes
    from .anchors import BLUE_STRATEGY
    from .report import refresh_report, refresh_report_async

    if method not in CYCLE_SERIES_METHODS:
        raise ValueError(f"{method} has no cycle depth to sweep")
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
    del saved
    policy = load_policy(method, checkpoint)
    if hasattr(policy, "set_device"):
        policy.set_device(device)
    elif device != "cpu":
        raise ValueError("This policy adapter supports CPU evaluation only")
    policy_name = f"crossscale_depth_{next(_POLICY_IDS)}"
    register_end_to_end_policy("red", policy_name, f"{method} cycle-depth sweep", lambda _: policy)
    logger = ExperimentLogger(output, method, actual_seed, run)
    existing = unique_episodes(read_records(output, "episodes", run=run, method=method, seed=actual_seed))
    jobs = remaining_depth_jobs(t_env, existing, checkpoint.stem)
    total = depth_eval_total()
    completed = total - len(jobs)
    started, last_report, initial = time.monotonic(), time.monotonic(), completed
    last_depth = None
    print(f"[{method}/{actual_seed}] depth_eval {completed}/{total}", flush=True)

    def report(status, depth=None, extra=None, persist=True):
        progress = dict(phase="depth_eval", status=status, completed=completed, total=total,
                        eval_completed=completed, eval_total=total, t_env=t_env, budget_steps=t_env,
                        checkpoint=depth_checkpoint_tag(checkpoint.stem, t_env, depth or last_depth or 4),
                        cycle_depth=None if depth is None and last_depth is None else int(depth or last_depth))
        if extra:
            progress.update(extra)
        if persist:
            logger.progress(**progress)
            if on_progress is not None:
                try:
                    on_progress(dict(progress))
                except Exception as error:
                    print(f"[{method}] depth_eval progress callback: {type(error).__name__}: {error}", flush=True)
        return progress

    try:
        if not jobs:
            report("complete", DEPTH_SWEEP_DEPTHS[-1])
            return dict(status="complete", completed=completed, total=total)
        for job in jobs:
            if stop_requested is not None and stop_requested():
                report("stopped", job.get("cycle_depth"))
                return dict(status="stopped", completed=completed, total=total)
            depth = int(job["cycle_depth"])
            if depth != last_depth:
                if last_depth is not None:
                    refresh_report_async(output, run=run)
                    last_report = time.monotonic()
                if hasattr(policy, "set_eval_depth"):
                    policy.set_eval_depth(depth)
                last_depth = depth
                print(f"[{method}/{actual_seed}] depth_eval R{depth} {completed}/{total}", flush=True)
            red, blue, targets = config_key(job["config"])
            result = run_episode(targets=targets, red=red, blue=blue, seed=job["episode_seed"],
                                 red_strategy={"architecture": "end_to_end", "policy": policy_name},
                                 blue_strategy=BLUE_STRATEGY, max_steps=100, record=False,
                                 task_mode="damage", spatial_dim=2, target_initialization="random",
                                 diagnostics=True, record_events=True, retain_trajectory=False)
            summary = dict(result["episode_summary"])
            summary.update(job)
            summary.pop("cycle_depth", None)
            summary.pop("retain_trajectory", None)
            if hasattr(policy, "episode_q_statistics"):
                summary.update(policy.episode_q_statistics())
            logger.episodes([summary])
            completed += 1
            elapsed = time.monotonic() - started
            remaining = (total - completed) * elapsed / max(1, completed - initial)
            persist = completed % 10 == 0 or completed == total
            report("running", depth, extra=dict(remaining_seconds=remaining), persist=persist)
            if persist:
                label = config_label(job["config"])
                print(f"[{method}/{actual_seed}] depth_eval R{depth} {completed}/{total} {label}", flush=True)
            if time.monotonic() - last_report >= 60:
                refresh_report_async(output, run=run)
                last_report = time.monotonic()
        report("complete", last_depth)
    except BaseException as error:
        logger.progress(phase="depth_eval", status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        completed=completed, total=total, t_env=t_env, error=str(error),
                        eval_completed=completed, eval_total=total,
                        checkpoint=depth_checkpoint_tag(checkpoint.stem, t_env, last_depth or 4),
                        cycle_depth=last_depth)
        raise
    finally:
        _safe_refresh(output, run)
    return dict(status="complete", completed=completed, total=total)

