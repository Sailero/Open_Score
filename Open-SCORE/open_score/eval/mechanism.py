"""One-episode attention and cross-round α for a frozen ReGIR checkpoint."""
from __future__ import annotations

from pathlib import Path
import time

from open_score.eval.protocol import MECH_CONFIG, MECH_EPISODE_SEEDS, config_dict, config_label
from open_score.utils.logging import DEFAULT_OUTPUT, FORMAL_RUN, ExperimentLogger, read_records


def _tensor_list(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    return value


def evaluate_mechanism(method="regir", *, output=None, run=None, seed=0, device="cpu",
                       stop_requested=None):
    import torch
    from open_score.algos import load_policy
    from open_score.eval.anchors import BLUE_STRATEGY
    from open_score.eval.report import refresh_report
    from open_score.rules import register_end_to_end_policy, run_episode

    run = FORMAL_RUN if run is None else run
    output = Path(DEFAULT_OUTPUT if output is None else output)
    checkpoint = output / method / run / f"seed_{int(seed)}" / "best.pt"
    if not checkpoint.exists():
        return dict(status="blocked", reason="no best.pt")
    existing = read_records(output, "trajectories", run=run, method=method, seed=seed)
    if any(row.get("phase") == "mechanism" for row in existing):
        return dict(status="complete", reused=True)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    t_env = int(saved["progress"]["t_env"])
    del saved
    policy = load_policy(method, checkpoint)
    if hasattr(policy, "set_device"):
        policy.set_device(device)
    agent = policy.mac.agent
    if not hasattr(agent, "global_net"):
        return dict(status="blocked", reason="no global_net")
    agent.capture_attention = True
    agent.global_net.capture_attention = True
    if hasattr(agent.global_net, "self_attn"):
        agent.global_net.self_attn.capture_attention = True
    policy_name = f"crossscale_mechanism_{int(time.time())}"
    register_end_to_end_policy("red", policy_name, "ReGIR attention probe", lambda _seed: policy)
    logger = ExperimentLogger(output, method, seed, run)
    red, blue, targets = MECH_CONFIG
    used = None
    try:
        for episode_seed in MECH_EPISODE_SEEDS:
            if stop_requested is not None and stop_requested():
                return dict(status="stopped")
            policy.reset()
            agent.global_net.last_self_attn = []
            result = run_episode(targets=targets, red=red, blue=blue, seed=episode_seed,
                                 red_strategy={"architecture": "end_to_end", "policy": policy_name},
                                 blue_strategy=BLUE_STRATEGY, max_steps=100, record=False,
                                 task_mode="damage", spatial_dim=2, target_initialization="random",
                                 diagnostics=True, record_events=True, retain_trajectory=True)
            if "trajectory" not in result:
                continue
            frames = result["trajectory"]
            mid = max(len(frames) // 2, 0)
            attention = {
                "round_self_attn": [_tensor_list(item) for item in getattr(agent.global_net, "last_self_attn", [])],
                "alpha": _tensor_list(getattr(agent.global_net, "last_alpha", None)),
                "step": mid,
                "episode_seed": episode_seed,
                "config": config_dict(MECH_CONFIG),
            }
            engaged = any(frame.get("blue_left", 1) < blue for frame in frames) or result["episode_summary"]["D"] > 0
            logger.trajectories([dict(phase="mechanism", config=config_dict(MECH_CONFIG),
                                      episode_seed=episode_seed, checkpoint=f"best@{t_env}",
                                      trajectory={"frames": frames, "attention": attention})])
            used = episode_seed
            if engaged or episode_seed == MECH_EPISODE_SEEDS[-1]:
                break
        logger.progress(phase="mechanism", status="complete", config=config_dict(MECH_CONFIG),
                        episode_seed=used, t_env=t_env)
    finally:
        agent.global_net.capture_attention = False
        if hasattr(agent.global_net, "self_attn"):
            agent.global_net.self_attn.capture_attention = False
        refresh_report(output, run=run)
    return dict(status="complete", episode_seed=used, config=config_label(MECH_CONFIG))
