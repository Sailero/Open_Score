"""Frozen validation and final-evaluation quotas; no environment construction."""
from __future__ import annotations

import math
from itertools import count
from pathlib import Path
import time

from ..envs.scales import SCALE_POOLS, TEST_POOL


def _safe_refresh(output, run=None):
    try:
        import importlib
        from . import report as report_mod
        importlib.reload(report_mod)
        report_mod.refresh_report(output, run=run)
    except Exception as error:
        print(f"report refresh skipped: {type(error).__name__}: {error}", flush=True)

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
# Formal frozen eval scores last-iterate weights. Validation-best stays on disk
# as best.pt but does not enter the official tables.
OFFICIAL_CHECKPOINT = "final"
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

    ``best.pt`` / ``final.pt`` are overwritten, so a bare stem cannot
    distinguish two different models and would let an earlier evaluation
    satisfy the quota of a later one.
    """
    return f"{stem}@{int(t_env)}"


def official_checkpoint_tag(t_env, depth=None):
    tag = checkpoint_tag(OFFICIAL_CHECKPOINT, t_env)
    if depth is None:
        return tag
    return f"{tag}/R{int(depth)}"


def parse_checkpoint_stem(checkpoint):
    text = str(checkpoint or "")
    if "@" not in text:
        return None
    return text.partition("@")[0]


def final_jobs(t_env=0, checkpoint=None, method=None):
    checkpoint = OFFICIAL_CHECKPOINT if checkpoint is None else checkpoint
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


def parse_checkpoint_t_env(checkpoint):
    """`best@520055` or `best@1000000/R4` → the t_env baked into the tag."""
    text = str(checkpoint or "")
    if "@" not in text:
        return None
    stem, _, rest = text.partition("@")
    if stem not in ("best", "final"):
        return None
    try:
        return int(rest.split("/")[0])
    except ValueError:
        return None


def run_directory(output, method, seed, run=None):
    from open_score.utils.logging import FORMAL_RUN
    run = FORMAL_RUN if run is None else run
    return Path(output) / method / str(run) / f"seed_{int(seed)}"


def official_eval_path(directory, progress_row=None):
    """Only a completed last iterate is eligible; best is never a substitute."""
    directory = Path(directory)
    final = directory / f"{OFFICIAL_CHECKPOINT}.pt"
    if training_finished(directory, progress_row):
        return final
    return None


def official_eval_t_env(directory, progress_row=None):
    """t_env of the last-iterate weights. Official eval scores ``final.pt``."""
    path = official_eval_path(directory, progress_row)
    if path is None:
        return None
    try:
        import torch
        saved = torch.load(path, map_location="cpu", weights_only=False)
        return int(saved["progress"]["t_env"])
    except Exception:
        return None


def official_best_t_env(directory):
    return official_eval_t_env(directory)


def training_finished(directory, progress_row=None):
    path = Path(directory) / "final.pt"
    if not path.exists():
        return False
    try:
        from open_score.algos import setup_runtime
        setup_runtime()
        import torch
        saved = torch.load(path, map_location="cpu", weights_only=False)
        return (saved["progress"].get("status") in ("completed", "complete")
                and int(saved["progress"]["t_env"]) >= int(saved["config"]["t_max"]))
    except (OSError, ValueError, KeyError, EOFError):
        return False


def rows_for_official_checkpoint(rows, t_env, phase=None):
    if t_env is None:
        return []
    t_env = int(t_env)
    kept = []
    for row in rows:
        if phase is not None and row.get("phase") != phase:
            continue
        if parse_checkpoint_stem(row.get("checkpoint")) != OFFICIAL_CHECKPOINT:
            continue
        if parse_checkpoint_t_env(row.get("checkpoint")) != t_env:
            continue
        kept.append(row)
    return kept


def _formal_eval_allowed(directory, progress_row, t_env, run):
    from open_score.utils.logging import FORMAL_RUN
    if run != FORMAL_RUN:
        return True
    if not training_finished(directory, progress_row):
        return False
    official = official_eval_t_env(directory, progress_row)
    return official is None or int(official) == int(t_env)


def depth_eval_total():
    return len(EQUAL_SCALE_CONFIGS) * FINAL_EPISODES_PER_CONFIG * len(DEPTH_SWEEP_DEPTHS)


def depth_jobs(t_env=0, checkpoint=None, depth=4):
    checkpoint = OFFICIAL_CHECKPOINT if checkpoint is None else checkpoint
    tag = depth_checkpoint_tag(checkpoint, t_env, depth)
    return [dict(config=config_dict(config), episode_seed=9000 + offset, phase="depth_eval",
                 eval_point=50, t_env=int(t_env), checkpoint=tag, cycle_depth=int(depth),
                 retain_trajectory=False)
            for config in EQUAL_SCALE_CONFIGS for offset in range(FINAL_EPISODES_PER_CONFIG)]


def _reuse_final_as_depth(recorded_rows, t_env, checkpoint=None):
    """Formal 1:1 final_eval is already greedy R=4; do not rerun those episodes."""
    checkpoint = OFFICIAL_CHECKPOINT if checkpoint is None else checkpoint
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


def remaining_depth_jobs(t_env, recorded_rows, checkpoint=None):
    recorded = list(recorded_rows) + _reuse_final_as_depth(recorded_rows, t_env, checkpoint)
    jobs = []
    for depth in DEPTH_SWEEP_DEPTHS:
        jobs.extend(remaining_jobs(depth_jobs(t_env, checkpoint, depth), recorded))
    return jobs


def depth_eval_finished(output, method, run, seed, t_env, checkpoint=None):
    from open_score.utils.logging import read_records, unique_episodes
    existing = unique_episodes(read_records(output, "episodes", run=run, method=method, seed=seed))
    return not remaining_depth_jobs(t_env, existing, checkpoint)


def infer_best_t_env(output, method, run, seed, fallback=None, rows=None):
    from open_score.utils.logging import read_records
    source = rows if rows is not None else read_records(output, "episodes", run=run, method=method, seed=seed)
    found = [parse_checkpoint_t_env(row.get("checkpoint")) for row in source]
    found = [value for value in found if value is not None]
    if found:
        return max(found)
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
    from open_score.utils.logging import read_latest
    progress_row = read_latest(output, "progress", run=run).get((method, run, actual_seed), {})
    allowed = official_eval_path(checkpoint.parent, progress_row)
    if run == FORMAL_RUN and (allowed is None or checkpoint.resolve() != Path(allowed).resolve()):
        print(f"[{method}/{actual_seed}] refuse unofficial checkpoint {checkpoint.name} (want {OFFICIAL_CHECKPOINT}.pt)", flush=True)
        return dict(status="blocked", completed=0, total=final_episodes(method),
                    reason="not official last-iterate weights")
    if not _formal_eval_allowed(checkpoint.parent, progress_row, t_env, run):
        print(f"[{method}/{actual_seed}] refuse mid-training final_eval (file t_env={t_env})", flush=True)
        return dict(status="blocked", completed=0, total=final_episodes(method),
                    reason="training not finished")
    checkpoint_label = checkpoint_tag(OFFICIAL_CHECKPOINT if run == FORMAL_RUN else checkpoint.stem, t_env)
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
    jobs = remaining_jobs(final_jobs(t_env, OFFICIAL_CHECKPOINT, method=method), existing)
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
    from open_score.utils.logging import read_latest
    progress_row = read_latest(output, "progress", run=run).get((method, run, actual_seed), {})
    allowed = official_eval_path(checkpoint.parent, progress_row)
    if run == FORMAL_RUN and (allowed is None or checkpoint.resolve() != Path(allowed).resolve()):
        print(f"[{method}/{actual_seed}] refuse unofficial depth checkpoint {checkpoint.name}", flush=True)
        return dict(status="blocked", completed=0, total=depth_eval_total(),
                    reason="not official last-iterate weights")
    if not _formal_eval_allowed(checkpoint.parent, progress_row, t_env, run):
        print(f"[{method}/{actual_seed}] refuse mid-training depth_eval (file t_env={t_env})", flush=True)
        return dict(status="blocked", completed=0, total=depth_eval_total(),
                    reason="training not finished")
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
    jobs = remaining_depth_jobs(t_env, existing, OFFICIAL_CHECKPOINT)
    total = depth_eval_total()
    completed = total - len(jobs)
    started, last_report, initial = time.monotonic(), time.monotonic(), completed
    last_depth = None
    print(f"[{method}/{actual_seed}] depth_eval {completed}/{total}", flush=True)

    def report(status, depth=None, extra=None, persist=True):
        progress = dict(phase="depth_eval", status=status, completed=completed, total=total,
                        eval_completed=completed, eval_total=total, t_env=t_env, budget_steps=t_env,
                        checkpoint=depth_checkpoint_tag(OFFICIAL_CHECKPOINT, t_env, depth or last_depth or 4),
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
                        checkpoint=depth_checkpoint_tag(OFFICIAL_CHECKPOINT, t_env, last_depth or 4),
                        cycle_depth=last_depth)
        raise
    finally:
        _safe_refresh(output, run)
    return dict(status="complete", completed=completed, total=total)


def _smac_eval_runner(saved, config):
    """Rebuild capacity, retaining all learned parameter shapes and native inputs."""
    import copy
    from types import SimpleNamespace
    from open_score.algos import setup_runtime, make_scheme, LearnerLogger
    setup_runtime()
    from runners.parallel_runner import ParallelRunner
    from open_score.models import build_mac
    cfg = copy.deepcopy(saved["config"])
    cfg.update(device="cpu", use_cuda=False, batch_size_run=1)
    cfg["env_args"].update(pad=(config["N_R"], config["N_B"]), config=config, max_retries=1)
    args = SimpleNamespace(**cfg)
    runner = ParallelRunner(args, LearnerLogger())
    try:
        for key, value in runner.get_env_info().items():
            setattr(args, key, value)
        scheme, groups, preprocess = make_scheme(runner.get_env_info(), multi_task=False)
        model_scheme = copy.deepcopy(scheme)
        model_scheme["entities"]["vshape"] = args.entity_shape
        model_scheme["actions_onehot"] = {"vshape": (args.n_actions,), "group": "agents"}
        mac = build_mac(model_scheme, groups, args)
        mac.agent.load_state_dict(saved["networks"]["agent"])
        if "mac" in saved["networks"]:
            mac.load_state_dict(saved["networks"]["mac"])
        mac.eval()
        runner.setup(scheme, groups, preprocess, mac)
        return runner
    except BaseException:
        runner.close_env()
        raise


def evaluate_profile_checkpoint(method, checkpoint, *, output, kind="final", env="had",
                                seed=None, device="cpu", on_progress=None, stop_requested=None):
    """One fixed checkpoint/arm queue, resumed by immutable result identity."""
    import json
    import torch
    from .experiment import (checkpoint_info, initialize, evaluation_jobs,
                             remaining_evaluations, atomic_json)
    from open_score.utils.logging import ExperimentLogger, read_records
    from open_score.utils.resources import cpu_threads
    cpu_threads()
    torch.set_num_threads(1)
    if device != "cpu":
        raise ValueError("main0921 final/depth/readout evaluation uses CPU; GPU0 is reserved for training/cost")
    info = checkpoint_info(checkpoint, method=method, seed=seed, env=env)
    if kind != "final" and (method != "regir" or env != "had"):
        raise ValueError("Depth and readout interventions require the HAD Full final")
    manifest = initialize(output)
    existing = read_records(output, "episodes", run="train", env=env, method=method, seed=info["seed"])
    jobs = remaining_evaluations(info, kind, existing,
        read1_equivalent=manifest["mechanisms"]["read1_equivalence_verified"])
    total = len(evaluation_jobs(info, kind))
    completed = initial = total - len(jobs)
    logger = ExperimentLogger(output, method, info["seed"], "train", env=env)
    start = time.monotonic()
    policy = runner = None
    runner_config = None
    failed = {}
    failure_path = Path(checkpoint).parent / "evaluation_failures.json"
    if env == "had":
        from open_score.algos import load_policy
        from open_score.rules import register_end_to_end_policy, run_episode
        from .anchors import BLUE_STRATEGY
        policy = load_policy(method, checkpoint)
        policy_name = f"main0921_{next(_POLICY_IDS)}"
        register_end_to_end_policy("red", policy_name, "main0921 frozen final", lambda _: policy)
    else:
        from open_score.envs.smacv2_env import generate_registered_scenes
        scenes = generate_registered_scenes(output)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if failure_path.exists():
            old = json.loads(failure_path.read_text())
            if old.get("checkpoint_id") == info["checkpoint_id"]:
                failed = old.get("scenes", {})

    def progress(status, job=None):
        row = dict(phase=f"{kind}_eval", status=status, completed=completed, total=total,
                   t_env=info["t_env"], checkpoint=info["checkpoint"], checkpoint_id=info["checkpoint_id"],
                   remaining_seconds=(total-completed)*(time.monotonic()-start)/max(1, completed-initial))
        if job:
            row.update(arm=job["arm"], cycle_depth=job.get("cycle_depth"), readout=job.get("readout"))
        logger.progress(**row)
        if on_progress:
            on_progress(row)
        return dict(status=status, completed=completed, total=total, failed_scenes=len(failed))

    try:
        for job in jobs:
            if stop_requested and stop_requested():
                return progress("stopped", job)
            if env == "had":
                if job.get("cycle_depth") is not None:
                    policy.set_eval_depth(job["cycle_depth"])
                if kind == "readout":
                    policy.mac.agent.global_net.readout_override = job["readout"]
                result = run_episode(targets=job["config"]["K"], red=job["config"]["N_R"],
                    blue=job["config"]["N_B"], seed=job["episode_seed"],
                    red_strategy={"architecture": "end_to_end", "policy": policy_name},
                    blue_strategy=BLUE_STRATEGY, max_steps=100, record=False, task_mode="damage",
                    spatial_dim=2, target_initialization="random", diagnostics=True, record_events=True,
                    retain_trajectory=job["retain_trajectory"])
                summary, trajectory = dict(result["episode_summary"]), result.get("trajectory")
                if hasattr(policy, "episode_q_statistics"):
                    summary.update(policy.episode_q_statistics())
            else:
                scene_id = f"{job['config']['N_R']}v{job['config']['N_B']}.s{job['episode_seed']}"
                scene = scenes["scenes"][scene_id]
                job["reset_config"] = scene["reset_config"]
                job["engine_seed"] = scene["engine_seed"]
                if failed.get(scene_id, {}).get("attempts", 0) >= 3:
                    continue
                summary = trajectory = None
                for attempt in range(int(failed.get(scene_id, {}).get("attempts", 0)), 3):
                    try:
                        if runner is None or runner_config != job["config"]:
                            if runner is not None:
                                runner.close_env()
                            runner = _smac_eval_runner(saved, job["config"])
                            runner_config = dict(job["config"])
                        _, summaries = runner.run(test_mode=True, jobs=[job])
                        summary = dict(summaries[0])
                        trajectory = runner.last_trajectories[0]
                        failed.pop(scene_id, None)
                        if failure_path.exists():
                            atomic_json(failure_path, dict(checkpoint_id=info["checkpoint_id"], scenes=failed))
                        break
                    except Exception as error:
                        if runner is not None:
                            runner.close_env()
                            runner = None
                        if not (isinstance(error, (EOFError, BrokenPipeError, ConnectionError))
                                or "SMACEnvironmentError" in str(error)
                                or "Environment worker" in str(error)):
                            raise
                        # A protocol/connection failure is not a battle loss.
                        failed[scene_id] = dict(attempts=attempt+1, error=f"{type(error).__name__}: {error}")
                        atomic_json(failure_path, dict(checkpoint_id=info["checkpoint_id"], scenes=failed))
                        if stop_requested and stop_requested():
                            return progress("stopped", job)
                if summary is None:
                    continue
            summary.update({k: v for k, v in job.items() if k != "reset_config"})
            logger.episodes([summary])
            if job["retain_trajectory"] and trajectory is not None:
                logger.trajectories([{**summary, "trajectory": trajectory}])
            completed += 1
            progress("running", job)
            if completed % 25 == 0:
                print(f"[{env}/{method}/{info['seed']}] {kind}_eval {completed}/{total}", flush=True)
        return progress("complete" if completed == total else "incomplete")
    finally:
        if runner is not None:
            runner.close_env()
