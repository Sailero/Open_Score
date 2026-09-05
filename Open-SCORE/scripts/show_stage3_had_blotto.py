"""Render one identity-level ID-CS-BBG dynamic regrouping episode."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
for value in (PROJECT, SRC):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from open_score.envs import HADStage3Adapter
from open_score.stage3 import (
    apply_identity_joint_plan,
    build_identity_event_game,
    identity_roster_override,
    solve_identity_double_oracle,
)
from open_score.stage3.artifacts import load_yaml
from open_score.stage3.payoff import FrozenStage2Payoff
from open_score.stage3.runtime import (
    FrozenStage1GroupExecutor,
    build_observable_threat_patrol,
    load_round01_stage1_model,
    moderate_jittered_target_positions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT
        / "configs/stage3_aligned.yaml",
    )
    parser.add_argument("--targets", type=int, choices=(3, 5), default=5)
    parser.add_argument("--blue-style", choices=("rush", "split_rush"), default="split_rush")
    parser.add_argument("--seed", type=int, default=20_260_904)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--episodes", type=int, default=0, help="0 means repeat until the window closes")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT / path


def expected_hash(value: object) -> str:
    text = str(value or "")
    return "" if text.upper() == "AUTO" else text


def choose_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        print("[S3可视化] CUDA 不可用，回退到 CPU。", flush=True)
        name = "cpu"
    return torch.device(name)


def scenario(targets: int) -> dict[str, int]:
    return {"red": 18, "blue": 12, "targets": 3} if targets == 3 else {"red": 30, "blue": 20, "targets": 5}


def render_open(adapter: HADStage3Adapter, caption: str) -> bool:
    adapter.env.render()
    import pygame

    if not (pygame.get_init() and pygame.display.get_init() and pygame.display.get_surface()):
        return False
    pygame.display.set_caption(caption)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
    return True


def subgroup_map(plan):
    return {
        int(agent_id): int(group.subgroup_id)
        for group in plan.local_subgames
        for agent_id in group.blue_ids
    }


def run_episode(config, predictor, stage1_model, device, *, targets, blue_style, seed, fps):
    spec = scenario(targets)
    physical = config["physical_evaluation"]
    positions = moderate_jittered_target_positions(targets, physical["layout_generator"], seed=seed)
    adapter = HADStage3Adapter(
        spec["red"],
        spec["blue"],
        targets,
        max_steps=int(physical["max_steps"]),
        target_positions=positions,
        blue_rule_style=blue_style,
        split_spacing=float(physical["blue_split_spacing"]),
    )
    adapter.reset(seed=seed)
    executor = FrozenStage1GroupExecutor(stage1_model, device, micro_grouping="planned_groups")
    domain = config["candidate_domain"]
    solver = config["solver"]
    rng = np.random.default_rng(seed ^ 0x1DC5BB6)
    interval = int(physical["command_interval"])
    plan = None
    reserve_patrol = None
    need_plan = True
    command = 0
    done = False
    info = {"outcome_red": 0.0, "events": []}
    print(
        f"[S3可视化] Red={spec['red']} Blue={spec['blue']} 固定目标={targets} Blue={blue_style}",
        flush=True,
    )
    while not done:
        if not render_open(adapter, f"Open-SCORE ID-CS-BBG | step={adapter.step_count} | targets={targets} | command={command}"):
            return False
        red_alive = sum(row["alive"] for row in adapter.agent_states("Red").values())
        blue_alive = sum(row["alive"] for row in adapter.agent_states("Blue").values())
        if need_plan and red_alive and blue_alive:
            start = time.perf_counter()
            built = build_identity_event_game(
                adapter,
                predictor,
                blue_style=blue_style,
                full_domain_agent_threshold=int(domain["complete_domain_max_agents_per_side"]),
                neighborhood_size=int(domain["spatial_neighborhood_size"]),
                peer_count=int(domain["peer_count"]),
                batch_size=int(solver["payoff_batch_size"]),
                utility_mode=str(config["game"]["local_utility"]),
                risk_epsilon=float(config["game"]["risk_epsilon"]),
            )
            result = solve_identity_double_oracle(
                built.game,
                built.initial_red,
                built.initial_blue,
                tolerance=float(solver["tolerance"]),
                max_iterations=int(solver["max_iterations"]),
                oracle_time_limit_seconds=float(solver["oracle_time_limit_seconds"]),
                oracle_mip_relative_gap=float(solver["oracle_mip_relative_gap"]),
            )
            red_action, blue_action = result.sample_profile(rng=rng)
            plan = apply_identity_joint_plan(
                adapter, built.game, red_action, blue_action, blue_type_name=blue_style
            )
            reserve_patrol = build_observable_threat_patrol(
                adapter,
                plan.red_reserve_ids,
                dict(plan.red_assignment),
                **{
                    key: float(value)
                    for key, value in physical.get("reserve_patrol", {}).items()
                    if key
                    in {"standoff_distance", "lateral_spacing", "distance_scale"}
                },
            )
            executor.reset()
            command += 1
            print(
                f"[身份级重规划] command={command} step={adapter.step_count} "
                f"groups={len(plan.local_subgames)} gap={result.exploitability:.4f} "
                f"reserve={len(plan.red_reserve_ids)} "
                f"certificate={result.candidate_domain_exact} time={time.perf_counter()-start:.2f}s",
                flush=True,
            )
            for group in plan.local_subgames:
                print(
                    f"  target={group.target_id} channel={group.subgroup_id}: "
                    f"Red{group.red_ids} vs Blue{group.blue_ids}",
                    flush=True,
                )
            for agent_id, target_id in reserve_patrol.target_by_agent:
                waypoint = dict(reserve_patrol.waypoint_by_agent)[agent_id]
                print(
                    f"  reserve Red{agent_id}: patrol target={target_id} "
                    f"waypoint=({waypoint[0]:.1f}, {waypoint[1]:.1f}, {waypoint[2]:.1f})",
                    flush=True,
                )
            need_plan = False
        if plan is None:
            red_actions = {
                agent_id: 0
                for agent_id, row in adapter.agent_states("Red").items()
                if row["alive"]
            }
            blue_groups = None
        else:
            red_actions = executor.act(
                adapter,
                local_steps={target: adapter.step_count for target in adapter.target_ids},
                roster_override=identity_roster_override(plan),
                reserve_waypoints=(
                    None
                    if reserve_patrol is None
                    else reserve_patrol.waypoint_mapping()
                ),
            )
            blue_groups = subgroup_map(plan)
        _, _, done, info = adapter.step(
            red_actions, blue_style=blue_style, blue_subgroup_by_agent=blue_groups
        )
        events = {event["kind"] for event in info.get("events", [])}
        need_plan = not done and (
            adapter.step_count % interval == 0
            or bool(physical["replan_on_casualty"] and "agents_destroyed" in events)
        )
        time.sleep(1.0 / fps)
    render_open(adapter, "Open-SCORE ID-CS-BBG | episode complete")
    print(
        f"[S3可视化] {'Red守住' if float(info['outcome_red']) > 0 else 'Blue突破'}，steps={adapter.step_count}",
        flush=True,
    )
    return True


def main() -> None:
    args = parse_args()
    if args.fps <= 0 or args.episodes < 0:
        raise ValueError("fps must be positive and episodes non-negative")
    if not args.config.is_file():
        raise FileNotFoundError(f"找不到 {args.config}；请先启动并完成 Stage2，或指定一次已完成 run 的 resolved config")
    config = load_yaml(args.config)
    device = choose_device(args.device)
    artifacts = config["artifacts"]
    predictor = FrozenStage2Payoff(
        project_path(artifacts["stage2_checkpoint"]),
        device=device,
        expected_sha256=expected_hash(artifacts.get("stage2_sha256")),
    )
    stage1_model = load_round01_stage1_model(
        project_path(artifacts["stage1_checkpoint"]),
        device,
        expected_sha256=expected_hash(artifacts.get("stage1_sha256")),
    )
    episode = 0
    while args.episodes == 0 or episode < args.episodes:
        if not run_episode(
            config,
            predictor,
            stage1_model,
            device,
            targets=args.targets,
            blue_style=args.blue_style,
            seed=args.seed + episode,
            fps=args.fps,
        ):
            break
        episode += 1


if __name__ == "__main__":
    main()
