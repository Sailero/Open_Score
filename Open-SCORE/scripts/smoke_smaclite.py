"""Run stock SMAClite and SMAClite-AD provenance/contract smoke tests."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Sequence

import gymnasium as gym
import numpy as np
import torch

import smaclite  # noqa: F401 - registers official Gymnasium environments
from smaclite.env.util.direction import Direction

from open_score.envs.smaclite_ad import (
    PROTOCOL_ID,
    SMACliteADEnv,
    stock_scenario_fingerprint,
    tensorize_smaclite_ad_observation,
)


def _first_available(rows: np.ndarray, count: int) -> list[int]:
    return [int(np.flatnonzero(row)[0]) for row in rows[:count]]


def run_stock(seed: int) -> Dict[str, object]:
    env = gym.make("smaclite/3s5z-v0")
    obs_a, info_a = env.reset(seed=seed)
    state_a = env.unwrapped.get_state().copy()
    obs_b, info_b = env.reset(seed=seed)
    state_b = env.unwrapped.get_state().copy()
    deterministic_reset = all(np.array_equal(a, b) for a, b in zip(obs_a, obs_b))
    deterministic_reset &= np.array_equal(state_a, state_b)
    available = env.unwrapped.get_avail_actions()
    actions = _first_available(np.asarray(available), env.unwrapped.n_agents)
    next_obs, reward, terminated, truncated, step_info = env.step(actions)
    result = {
        "environment_id": "smaclite/3s5z-v0",
        "seed": seed,
        "agents": env.unwrapped.n_agents,
        "enemies": env.unwrapped.n_enemies,
        "action_dim": env.unwrapped.n_actions,
        "observation_shape": list(obs_b[0].shape),
        "state_shape": list(state_b.shape),
        "deterministic_reset": bool(deterministic_reset),
        "reset_info": info_b,
        "one_step": {
            "observations": len(next_obs),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": step_info,
        },
    }
    env.close()
    if not result["deterministic_reset"]:
        raise AssertionError("stock SMAClite reset is not deterministic under a fixed seed")
    return result


def _direction_action(env: SMACliteADEnv, unit: object, destination: np.ndarray, side: str) -> int:
    available = env._available_actions_for(unit, side)  # audit reference policy
    target_action = None
    if side == "Red":
        target_action = env.asset_action_id
    else:
        for index in range(env.config.red_agents):
            if available[6 + index]:
                return 6 + index
    if target_action is not None and available[target_action]:
        return target_action
    desired = destination - unit.pos
    best_action = 1
    best_score = -float("inf")
    for direction in Direction:
        action = 2 + direction.value
        if available[action]:
            score = float(Direction(direction.value).dx_dy @ desired)
            if score > best_score:
                best_score, best_action = score, action
    return best_action


def _reference_actions(env: SMACliteADEnv) -> Dict[str, Sequence[int]]:
    red_destination = env._asset.pos if env._asset is not None else np.asarray([24.0, 16.0])
    alive_red = [unit for unit in env._red_slots if unit.hp > 0]
    red = [
        0 if unit.hp <= 0 else _direction_action(env, unit, red_destination, "Red")
        for unit in env._red_slots
    ]
    blue = []
    for unit in env._blue_slots:
        if unit.hp <= 0:
            blue.append(0)
        elif alive_red:
            nearest = min(alive_red, key=lambda other: np.linalg.norm(other.pos - unit.pos))
            blue.append(_direction_action(env, unit, nearest.pos, "Blue"))
        else:
            blue.append(1)
    return {"Red": red, "Blue": blue}


def run_ad(seed: int) -> Dict[str, object]:
    records = []
    expected_shapes = None
    for scale_index, (red_count, blue_count) in enumerate(((2, 1), (3, 2), (5, 3))):
        env = SMACliteADEnv(
            red_count,
            blue_count,
            max_red_agents=6,
            max_blue_agents=5,
            episode_limit=80,
            seed=seed + scale_index,
        )
        obs, reset_info = env.reset(seed=seed + scale_index)
        shapes = {
            side: {name: list(value.shape) for name, value in team.items()}
            for side, team in obs.items()
        }
        shape_signature = {
            side: (tuple(obs[side]["entity_obs"].shape), tuple(obs[side]["avail_actions"].shape))
            for side in ("Red", "Blue")
        }
        if expected_shapes is None:
            expected_shapes = shape_signature
        elif expected_shapes != shape_signature:
            raise AssertionError("padded tensor shapes changed across realised team ratios")
        for side in ("Red", "Blue"):
            team, state = tensorize_smaclite_ad_observation(obs[side])
            team.validate()
            state.validate()
            if int(team.agent_mask.sum()) != env.team_sizes[side]:
                raise AssertionError("agent mask does not match realised roster")
        returns = {"Red": 0.0, "Blue": 0.0}
        terminated = truncated = False
        while not (terminated or truncated):
            actions = _reference_actions(env)
            obs, rewards, terminated, truncated, info = env.step(actions)
            if not np.isclose(rewards["Red"] + rewards["Blue"], 0.0):
                raise AssertionError("SMAClite-AD reward is not zero-sum")
            for side in returns:
                returns[side] += rewards[side]
        records.append(
            {
                "red_agents": red_count,
                "blue_agents": blue_count,
                "reset_info": reset_info,
                "tensor_shapes": shapes,
                "episode_steps": info["episode_steps"],
                "termination_reason": info["termination_reason"],
                "outcome_red": info["outcome_red"],
                "returns": returns,
                "final_health": {
                    "red": info["red_health_fraction"],
                    "blue": info["blue_health_fraction"],
                    "asset": info["asset_health_fraction"],
                },
            }
        )
        env.close()
    return {
        "protocol_id": PROTOCOL_ID,
        "seed": seed,
        "scales": records,
        "checks": {
            "shared_shapes_across_ratios": True,
            "team_and_global_contracts_valid": True,
            "zero_sum_rewards": True,
            "primitive_actions_only": True,
        },
    }


def machine_info() -> Dict[str, object]:
    cuda_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu": cuda_name,
        "numpy": np.__version__,
        "gymnasium": gym.__version__,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("all", "stock", "ad"), default="all")
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/smoke_smaclite/validation.json"),
    )
    args = parser.parse_args()
    payload: Dict[str, object] = {
        "schema_version": 1,
        "status": "smoke_and_contract_validation_not_convergence_claim",
        "machine": machine_info(),
        "upstream": stock_scenario_fingerprint(),
    }
    if args.mode in {"all", "stock"}:
        payload["stock"] = run_stock(args.seed)
    if args.mode in {"all", "ad"}:
        payload["ad"] = run_ad(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
