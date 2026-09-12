"""The approved four-config, two-policy, 300-seed anchor run."""
from __future__ import annotations

from pathlib import Path
import time

from open_score.utils.logging import DEFAULT_OUTPUT, FORMAL_RUN, ExperimentLogger, read_records, unique_episodes
from .protocol import FINAL_CONFIGS, FINAL_EPISODES_PER_CONFIG, config_dict, config_key

ANCHOR_STRATEGIES = {
    "random": {"architecture": "end_to_end", "policy": "random_accel"},
    "rule_nv1": {"architecture": "hierarchical", "layers": {
        "grouping": "nv1", "control": "predictive_intercept"}},
}
BLUE_STRATEGY = {"architecture": "hierarchical", "layers": {"grouping": "reactive", "control": "rush"}}


# NDS needs both same-scale anchors on every reported configuration, so the
# anchors span the extrapolation axes as well, on the paired seed batch.
ANCHOR_EPISODES = len(ANCHOR_STRATEGIES) * len(FINAL_CONFIGS) * FINAL_EPISODES_PER_CONFIG


def anchor_jobs():
    return [dict(method=method, config=config_dict(config), episode_seed=seed,
                 phase="anchor", eval_point=0, t_env=0)
            for method in ANCHOR_STRATEGIES for config in FINAL_CONFIGS
            for seed in range(9000, 9000 + FINAL_EPISODES_PER_CONFIG)]


def run_anchors(output=DEFAULT_OUTPUT, *, run=FORMAL_RUN, seed=0, on_progress=None,
                stop_requested=None):
    """Run only on an explicit CLI/scheduler call; resume completed openings."""
    from open_score.rules import run_episode

    output = Path(output)
    existing = unique_episodes(read_records(output, "episodes", run=run))
    completed = {(row["method"], config_key(row["config"]), int(row["episode_seed"]))
                 for row in existing if row["phase"] == "anchor"}
    jobs = anchor_jobs()
    wanted = {(job["method"], config_key(job["config"]), job["episode_seed"]) for job in jobs}
    completed.intersection_update(wanted)
    logger = ExperimentLogger(output, "anchors", seed, run)
    started, initial = time.monotonic(), len(completed)
    total_steps = 0
    try:
        for job in jobs:
            key = (job["method"], config_key(job["config"]), job["episode_seed"])
            if key in completed:
                continue
            if stop_requested is not None and stop_requested():
                logger.progress(phase="anchor", status="stopped", completed=len(completed), total=ANCHOR_EPISODES)
                return dict(status="stopped", completed=len(completed), total=ANCHOR_EPISODES)
            red, blue, targets = key[1]
            result = run_episode(targets=targets, red=red, blue=blue, seed=job["episode_seed"],
                                 red_strategy=ANCHOR_STRATEGIES[job["method"]], blue_strategy=BLUE_STRATEGY,
                                 max_steps=100, record=False, task_mode="damage", spatial_dim=2,
                                 target_initialization="random", diagnostics=True, record_events=True,
                                 retain_trajectory=False)
            if "episode_summary" not in result:
                raise RuntimeError("rules.run_episode must return the frozen episode_summary")
            row = dict(result["episode_summary"], **job)
            row.update(q_tot_mean=None, q_tot_std=None, q_i_mean=None)
            ExperimentLogger(output, job["method"], seed, run).episodes([row])
            completed.add(key)
            total_steps += int(row["ep_len"])
            elapsed = time.monotonic() - started
            state = dict(phase="anchor", status="running", completed=len(completed), total=ANCHOR_EPISODES,
                         method_current=job["method"], config_current=job["config"],
                         steps_per_second=total_steps / max(elapsed, 1e-9),
                         remaining_seconds=(ANCHOR_EPISODES-len(completed))*elapsed/max(1, len(completed)-initial))
            logger.progress(**state)
            if on_progress is not None:
                on_progress(state)
            if len(completed) % 25 == 0:
                print(f"[anchors] {len(completed)}/{ANCHOR_EPISODES} episodes", flush=True)
        logger.progress(phase="anchor", status="complete", completed=len(completed), total=ANCHOR_EPISODES)
    except BaseException as error:
        logger.progress(phase="anchor", status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        completed=len(completed), total=ANCHOR_EPISODES, error=str(error))
        raise
    return dict(status="complete", completed=len(completed), total=ANCHOR_EPISODES)
