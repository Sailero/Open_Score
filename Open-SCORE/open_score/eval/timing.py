"""Decision-time ms/step on the current entity interface, excluding env.step."""
from __future__ import annotations

from pathlib import Path
import time

from open_score.eval.protocol import (DEPTH_SWEEP_DEPTHS, OFFICIAL_CHECKPOINT, TIMING_METHODS,
                                      TIMING_SCALES, config_dict, config_label)
from open_score.utils.logging import DEFAULT_OUTPUT, FORMAL_RUN, ExperimentLogger, read_records


class _TimedPolicy:
    def __init__(self, inner):
        self.inner = inner
        self.elapsed = 0.0
        self.steps = 0

    def reset(self):
        self.inner.reset()

    def act(self, *args, **kwargs):
        started = time.perf_counter()
        result = self.inner.act(*args, **kwargs)
        self.elapsed += time.perf_counter() - started
        self.steps += 1
        return result

    def episode_q_statistics(self):
        if hasattr(self.inner, "episode_q_statistics"):
            return self.inner.episode_q_statistics()
        return {}


def _param_count(policy):
    total = 0
    for module in (getattr(policy, "mac", None), getattr(policy, "mixer", None)):
        if module is None:
            continue
        parameters = getattr(module, "parameters", None)
        if callable(parameters):
            total += sum(item.numel() for item in parameters())
        elif hasattr(module, "agent"):
            total += sum(item.numel() for item in module.agent.parameters())
    return int(total)


def evaluate_timing(*, output=None, run=None, device="cuda", stop_requested=None,
                    warmup=1, repeats=5):
    from open_score.algos import load_policy
    from open_score.eval.anchors import BLUE_STRATEGY
    from open_score.eval.inventory import _gpu_training_busy, run_dir
    from open_score.eval.report import refresh_report
    from open_score.rules import register_end_to_end_policy, run_episode

    if device == "cuda" and _gpu_training_busy():
        return dict(status="blocked", reason="gpu training busy")
    run = FORMAL_RUN if run is None else run
    output = Path(DEFAULT_OUTPUT if output is None else output)
    existing = read_records(output, "timing", run=run) if (output / "timing.csv").exists() else []
    done = {(row.get("method"),
             tuple(int(row["config"][key]) for key in ("N_R", "N_B", "K")) if isinstance(row.get("config"), dict) else None,
             row.get("cycle_depth"))
            for row in existing}
    logger = ExperimentLogger(output, "timing", 0, run)
    written = 0
    try:
        for method in TIMING_METHODS:
            checkpoint = run_dir(output, method, 0) / f"{OFFICIAL_CHECKPOINT}.pt"
            if not checkpoint.exists():
                continue
            policy = load_policy(method, checkpoint)
            if hasattr(policy, "set_device"):
                try:
                    policy.set_device(device)
                except Exception as error:
                    print(f"[timing] {method} set_device({device}) failed: {error}; use cpu", flush=True)
                    device = "cpu"
            params = _param_count(policy)
            depths = DEPTH_SWEEP_DEPTHS if method == "regir" else (4,)
            for scale in TIMING_SCALES:
                for depth in depths:
                    if stop_requested is not None and stop_requested():
                        return dict(status="stopped", written=written)
                    key = (method, scale, depth if method == "regir" else None)
                    if key in done or (method, scale, depth) in done:
                        continue
                    if hasattr(policy, "set_eval_depth"):
                        policy.set_eval_depth(int(depth))
                    timed = _TimedPolicy(policy)
                    name = f"timing_{method}_{scale[0]}_{depth}"
                    register_end_to_end_policy("red", name, "timed policy", lambda inner=timed: inner)
                    red, blue, targets = scale
                    # Warmup is not recorded.
                    for seed in range(warmup):
                        timed.elapsed = timed.steps = 0
                        run_episode(targets=targets, red=red, blue=blue, seed=9200 + seed,
                                    red_strategy={"architecture": "end_to_end", "policy": name},
                                    blue_strategy=BLUE_STRATEGY, max_steps=100, record=False,
                                    task_mode="damage", spatial_dim=2, target_initialization="random",
                                    diagnostics=False, record_events=False, retain_trajectory=False)
                    elapsed = steps = 0
                    for seed in range(repeats):
                        timed.elapsed = timed.steps = 0
                        run_episode(targets=targets, red=red, blue=blue, seed=9210 + seed,
                                    red_strategy={"architecture": "end_to_end", "policy": name},
                                    blue_strategy=BLUE_STRATEGY, max_steps=100, record=False,
                                    task_mode="damage", spatial_dim=2, target_initialization="random",
                                    diagnostics=False, record_events=False, retain_trajectory=False)
                        elapsed += timed.elapsed
                        steps += timed.steps
                    ms = 1000.0 * elapsed / max(steps, 1)
                    row = dict(method=method, config=config_dict(scale), device=device,
                               cycle_depth=int(depth) if method == "regir" else None,
                               repeat_id=0, ms_per_step=ms, n_steps=steps, params=params)
                    logger.timing([row])
                    written += 1
                    print(f"[timing] {method} {config_label(scale)} R={depth} {ms:.2f} ms/step params={params}",
                          flush=True)
        logger.progress(phase="timing", status="complete", completed=written)
    finally:
        refresh_report(output, run=run)
    return dict(status="complete", written=written)
