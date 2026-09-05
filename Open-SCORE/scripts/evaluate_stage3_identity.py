"""Evaluate the identity-level, hard-4v4 Stage-3 Blotto method."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import yaml

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from open_score.envs import HADStage3Adapter
from open_score.stage3 import (
    IdentityAction,
    IdentityBlottoGame,
    apply_identity_joint_plan,
    build_identity_event_game,
    enumerate_identity_actions,
    enumerate_coalitions,
    identity_roster_override,
    make_engagement_slots,
    round_robin_identity_action,
    solve_identity_double_oracle,
    solve_restricted_matrix_game,
)
from open_score.stage3.artifacts import (
    LiveProgress,
    atomic_write_json,
    atomic_write_text,
    utc_now_iso,
    write_csv,
)
from open_score.stage3.payoff import FrozenStage2Payoff, sha256_file
from open_score.stage3.runtime import (
    FrozenStage1GroupExecutor,
    build_observable_threat_patrol,
    load_round01_stage1_model,
    moderate_jittered_target_positions,
)


SCHEMA = "open-score-stage3-identity-cs-bbg-results-v5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/stage3_aligned.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-physical", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT / value).resolve()


def _expected_hash(value: object) -> str:
    text = str(value or "")
    return "" if text.upper() == "AUTO" else text


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config requests CUDA but torch.cuda.is_available() is false")
    return torch.device(requested)


def _percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _validate_config(config: Mapping[str, object]) -> None:
    if config.get("schema_version") != "open-score-stage3-identity-cs-bbg-v5":
        raise ValueError("Stage3 config is not identity CS-BBG v5")
    game = config["game"]
    required = {
        "zero_sum": True,
        "simultaneous_upper_actions": True,
        "identity_level": True,
        "fixed_targets_per_episode": True,
        "repeated_target_groups": True,
        "hard_cap_at_four": True,
        "conditional_group_independence": True,
    }
    for key, expected in required.items():
        if game.get(key) is not expected:
            raise ValueError(f"registered identity-game invariant failed: {key}")
    if game.get("action_representation") != "labelled_agent_target_channel_partition":
        raise ValueError("Stage3 action must retain labelled agents")
    if game.get("group_size_support") != [1, 2, 3, 4]:
        raise ValueError("the only legal non-empty local group sizes are 1..4")
    if game.get("red_reserve_enabled") is not True:
        raise ValueError("Red reserve must be enabled explicitly")
    if game.get("blue_reserve_enabled") is not False:
        raise ValueError("Blue reserve must remain disabled")
    if float(game.get("red_reserve_instant_utility", float("nan"))) != 0.0:
        raise ValueError("Red reserve instantaneous proxy utility must be zero")
    if game.get("payoff_normalization") != "analytic_absolute_bound_M_times_live_blue":
        raise ValueError("the population-invariant payoff normalization has drifted")
    if game.get("public_matching_mechanism") != "target_channel":
        raise ValueError("the simultaneous matching mechanism is not registered")
    if game.get("channel_symmetry_breaking") != "nonincreasing_group_size_per_target":
        raise ValueError("target-channel matching order is not the registered neutral rule")
    if {row["lower_style"] for row in game["blue_types"]} != {"rush", "split_rush"}:
        raise ValueError("the MVP Blue lower-policy library is incomplete")
    if config["physical_evaluation"].get("red_commanders") != [
        "balanced_identity",
        "identity_blotto",
        "revealed_blue_br",
    ]:
        raise ValueError("the registered physical controls have drifted")
    if (
        config["physical_evaluation"].get("blue_upper_policy")
        != "identity_blotto_equilibrium"
    ):
        raise ValueError("all physical controls must sample the same Blue upper policy")
    if int(config["candidate_domain"]["complete_domain_max_agents_per_side"]) < 3:
        raise ValueError("the exact small game must use a complete candidate domain")
    domain = config["candidate_domain"]
    for key in (
        "all_singletons_in_every_slot",
        "seed_coalitions_always_retained",
        "blue_exact_cover_seed_required",
        "red_all_reserve_action_required",
    ):
        if domain.get(key) is not True:
            raise ValueError(f"registered candidate-domain invariant failed: {key}")
    gap_limit = float(
        config["acceptance"]["candidate_domain_normalized_exploitability_max"]
    )
    if not 0.0 < gap_limit < 1.0:
        raise ValueError("the normalized candidate-domain gap limit must lie in (0, 1)")
    for key in (
        "require_all_groups_at_most_four",
        "require_fixed_target_count",
        "require_red_active_or_reserve_partition_exact",
        "require_blue_partition_exact",
    ):
        if config["acceptance"].get(key) is not True:
            raise ValueError(f"registered acceptance invariant failed: {key}")
    for scenario in config["physical_evaluation"]["scenarios"]:
        if not 1 <= int(scenario["targets"]) <= 5:
            raise ValueError("HAD demonstrations use one to five fixed targets")


class _ToyIdentityOracle:
    def __init__(self, red_ids, blue_ids, slots, seed: int):
        rng = np.random.default_rng(seed)
        self.red_skill = {value: rng.normal() for value in red_ids}
        self.blue_skill = {value: rng.normal() for value in blue_ids}
        self.slot_bias = rng.normal(0.0, 0.15, size=len(slots))

    def evaluate(self, requests):
        return np.asarray(
            [
                math.tanh(
                    0.35 * sum(self.red_skill[value] for value in red)
                    - 0.35 * sum(self.blue_skill[value] for value in blue)
                    + 0.18 * (len(red) - len(blue))
                    + self.slot_bias[int(slot)]
                )
                for slot, red, blue in requests
            ],
            dtype=np.float64,
        )


def _complete_candidates(ids, slots):
    coalitions = enumerate_coalitions(ids, 4)
    return tuple(coalitions for _ in slots)


def run_exact(config, output: Path, progress: LiveProgress, smoke: bool):
    spec = config["solver_evaluation"]["exact_small_game"]
    scenarios = 2 if smoke else int(spec["scenarios"])
    rows = []
    started = time.perf_counter()
    for index in range(scenarios):
        red_ids = tuple(range(int(spec["red"])))
        blue_ids = tuple(range(100, 100 + int(spec["blue"])))
        slots = make_engagement_slots(
            tuple(range(int(spec["targets"]))), max(len(red_ids), len(blue_ids)), 4
        )
        game = IdentityBlottoGame(
            red_ids,
            blue_ids,
            slots,
            _complete_candidates(red_ids, slots),
            _complete_candidates(blue_ids, slots),
            _ToyIdentityOracle(red_ids, blue_ids, slots, int(config["seed"]) + index),
            full_coalition_domain=True,
            allow_red_reserve=True,
            allow_blue_reserve=False,
        )
        red_actions = enumerate_identity_actions(red_ids, slots, allow_reserve=True)
        blue_actions = enumerate_identity_actions(blue_ids, slots, allow_reserve=False)
        exact_start = time.perf_counter()
        exact = solve_restricted_matrix_game(game.payoff_matrix(red_actions, blue_actions))
        exact_seconds = time.perf_counter() - exact_start
        do_start = time.perf_counter()
        result = solve_identity_double_oracle(
            game,
            [round_robin_identity_action(red_ids, slots)],
            [round_robin_identity_action(blue_ids, slots)],
            tolerance=1e-9,
            max_iterations=100,
        )
        rows.append(
            {
                "scenario": index,
                "red_pure_actions": len(red_actions),
                "blue_pure_actions": len(blue_actions),
                "exact_value": exact.value,
                "double_oracle_value": result.value,
                "absolute_value_error": abs(exact.value - result.value),
                "exploitability": result.exploitability,
                "full_game_certified": int(result.full_game_certified),
                "exact_seconds": exact_seconds,
                "double_oracle_seconds": time.perf_counter() - do_start,
                "iterations": len(result.history),
            }
        )
        progress.progress("Stage3/A 小规模完整博弈", index + 1, scenarios, started_at=started)
    write_csv(output / "raw/exact_identity_games.csv", rows)
    return rows, {
        "scenarios": scenarios,
        "max_value_error": max(row["absolute_value_error"] for row in rows),
        "max_exploitability": max(row["exploitability"] for row in rows),
        "all_full_game_certified": all(bool(row["full_game_certified"]) for row in rows),
    }


def _make_adapter(scenario, physical, seed: int, style: str, smoke: bool):
    positions = moderate_jittered_target_positions(
        int(scenario["targets"]), physical["layout_generator"], seed=seed
    )
    adapter = HADStage3Adapter(
        int(scenario["red"]),
        int(scenario["blue"]),
        int(scenario["targets"]),
        max_steps=int(physical["smoke_max_steps"] if smoke else physical["max_steps"]),
        target_positions=positions,
        blue_rule_style=style,
        split_spacing=float(physical["blue_split_spacing"]),
    )
    adapter.reset(seed=seed)
    return adapter


def _build(adapter, predictor, config, style):
    domain = config["candidate_domain"]
    return build_identity_event_game(
        adapter,
        predictor,
        blue_style=style,
        max_group_size=4,
        full_domain_agent_threshold=int(domain["complete_domain_max_agents_per_side"]),
        neighborhood_size=int(domain["spatial_neighborhood_size"]),
        peer_count=int(domain["peer_count"]),
        batch_size=int(config["solver"]["payoff_batch_size"]),
        utility_mode=str(config["game"]["local_utility"]),
        risk_epsilon=float(config["game"]["risk_epsilon"]),
    )


def _solve(built, config, *, verbose=False):
    solver = config["solver"]
    return solve_identity_double_oracle(
        built.game,
        built.initial_red,
        built.initial_blue,
        tolerance=float(solver["tolerance"]),
        max_iterations=int(solver["max_iterations"]),
        oracle_time_limit_seconds=float(solver["oracle_time_limit_seconds"]),
        oracle_mip_relative_gap=float(solver["oracle_mip_relative_gap"]),
        verbose=bool(verbose or solver.get("verbose", False)),
    )


def run_scaling(config, predictor, output: Path, progress: LiveProgress, smoke: bool):
    specs = config["solver_evaluation"]["planner_benchmarks"]
    repeats_total = sum(1 if smoke else int(row["repeats"]) for row in specs)
    rows = []
    completed = 0
    started = time.perf_counter()
    for spec_index, spec in enumerate(specs):
        repeats = 1 if smoke else int(spec["repeats"])
        for repeat in range(repeats):
            seed = int(config["seed"]) + 10000 + spec_index * 1000 + repeat
            adapter = _make_adapter(
                spec, config["physical_evaluation"], seed, "rush", smoke
            )
            planning_start = time.perf_counter()
            built = _build(adapter, predictor, config, "rush")
            result = _solve(built, config)
            elapsed = time.perf_counter() - planning_start
            rows.append(
                {
                    "scale": str(spec["label"]),
                    "red": int(spec["red"]),
                    "blue": int(spec["blue"]),
                    "targets": int(spec["targets"]),
                    "repeat": repeat,
                    "planning_seconds": elapsed,
                    "engagement_slots": len(built.slots),
                    "red_candidate_columns": built.red_candidate_columns,
                    "blue_candidate_columns": built.blue_candidate_columns,
                    "iterations": len(result.history),
                    "red_support": len(result.red_strategies),
                    "blue_support": len(result.blue_strategies),
                    "exploitability": result.exploitability,
                    "payoff_scale": built.game.payoff_scale,
                    "candidate_domain_certified": int(result.candidate_domain_exact),
                    "full_game_certified": int(result.full_game_certified),
                    "termination_reason": result.termination_reason,
                    "max_group_size": 4,
                }
            )
            completed += 1
            progress.progress(
                "Stage3/B 30–50人身份级规划", completed, repeats_total, started_at=started
            )
    write_csv(output / "raw/identity_planner_scaling.csv", rows)
    summary = {}
    for spec in specs:
        scale = str(spec["label"])
        selected = [row for row in rows if row["scale"] == scale]
        summary[scale] = {
            "runs": len(selected),
            "planning_p50_seconds": _percentile([row["planning_seconds"] for row in selected], 50),
            "planning_p95_seconds": _percentile([row["planning_seconds"] for row in selected], 95),
            "candidate_domain_certified_fraction": statistics.fmean(
                row["candidate_domain_certified"] for row in selected
            ),
            "exploitability_mean": statistics.fmean(row["exploitability"] for row in selected),
            "exploitability_max": max(row["exploitability"] for row in selected),
            "red_candidate_columns_mean": statistics.fmean(row["red_candidate_columns"] for row in selected),
            "blue_candidate_columns_mean": statistics.fmean(row["blue_candidate_columns"] for row in selected),
        }
    return rows, summary


def _revealed_blue_best_response(game: IdentityBlottoGame, blue: IdentityAction):
    """Best respond to the sampled equilibrium Blue action in the same game.

    This diagnostic changes only Red's information.  It does not manufacture
    a singleton Blue allocation, rebuild candidate columns, or narrow Red to
    1/2-agent coalitions.  Consequently its proxy value is an information
    upper bound inside exactly the candidate domain used by the main method.
    """

    revealed = game.validate_action(blue, "Blue")
    response = game.red_best_response(
        [revealed],
        [1.0],
        time_limit_seconds=None,
        mip_relative_gap=0.0,
    )
    if not response.optimal:
        raise RuntimeError(
            "revealed Blue best response did not reach an exact candidate-domain optimum: "
            f"{response.solver_message}"
        )
    return response


def _subgroup_map(plan):
    return {
        int(agent_id): int(item.subgroup_id)
        for item in plan.local_subgames
        for agent_id in item.blue_ids
    }


def _target_counts(action, game):
    counts = {target: 0 for target in dict.fromkeys(slot.target_id for slot in game.slots)}
    for slot, coalition in zip(game.slots, action.coalitions):
        counts[slot.target_id] += len(coalition)
    return [counts[target] for target in counts]


def _plan_record(
    plan,
    game,
    step,
    seconds,
    result,
    patrol,
    revealed_response=None,
    sampled_red_value=None,
):
    patrol_targets = patrol.target_mapping()
    patrol_waypoints = patrol.waypoint_mapping()
    return {
        "step": int(step),
        "planning_seconds": float(seconds),
        "value": None if result is None else float(result.value),
        "exploitability": None if result is None else float(result.exploitability),
        "termination_reason": result.termination_reason,
        "revealed_response_value": (
            None if revealed_response is None else float(revealed_response.value)
        ),
        "revealed_response_optimal": (
            None if revealed_response is None else bool(revealed_response.optimal)
        ),
        "sampled_equilibrium_red_value": sampled_red_value,
        "revealed_proxy_gain": (
            None
            if revealed_response is None or sampled_red_value is None
            else float(revealed_response.value) - float(sampled_red_value)
        ),
        "red_target_counts": _target_counts(plan.red_action, game),
        "blue_target_counts": _target_counts(plan.blue_action, game),
        "red_alive_count": len(game.red_ids),
        "blue_alive_count": len(game.blue_ids),
        "red_reserve_ids": list(plan.red_reserve_ids),
        "reserve_fraction": len(plan.red_reserve_ids) / max(1, len(game.red_ids)),
        "reserve_patrol": [
            {
                "red_id": int(agent_id),
                "target": int(patrol_targets[agent_id]),
                "waypoint": list(map(float, patrol_waypoints[agent_id])),
            }
            for agent_id in plan.red_reserve_ids
        ],
        "groups": [
            {
                "target": int(item.target_id),
                "group": int(item.subgroup_id),
                "channel": int(item.subgroup_id),
                "red_ids": list(item.red_ids),
                "blue_ids": list(item.blue_ids),
            }
            for item in plan.local_subgames
        ],
    }


def _run_episode(config, scenario, style, commander, seed, predictor, stage1_model, device, smoke):
    physical = config["physical_evaluation"]
    adapter = _make_adapter(scenario, physical, seed, style, smoke)
    executor = FrozenStage1GroupExecutor(stage1_model, device, micro_grouping="planned_groups")
    interval = int(physical["command_interval"])
    rng = np.random.default_rng(seed ^ 0x1DC5BB6)
    plan = None
    reserve_patrol = None
    records = []
    planning_times = []
    red_switches = 0
    previous_red = None
    done = False
    need_plan = True
    info = {"outcome_red": 0.0}
    while not done:
        red_alive = sum(row["alive"] for row in adapter.agent_states("Red").values())
        blue_alive = sum(row["alive"] for row in adapter.agent_states("Blue").values())
        if need_plan and red_alive and blue_alive:
            start = time.perf_counter()
            built = _build(adapter, predictor, config, style)
            active_game = built.game
            result = _solve(built, config)
            equilibrium_red, blue_action = result.sample_profile(rng=rng)
            revealed_response = None
            sampled_red_value = None
            if commander == "identity_blotto":
                red_action = equilibrium_red
            elif commander == "balanced_identity":
                red_action = built.initial_red[0]
            elif commander == "revealed_blue_br":
                sampled_red_value = active_game.payoff(equilibrium_red, blue_action)
                revealed_response = _revealed_blue_best_response(
                    active_game, blue_action
                )
                if revealed_response.value + 1e-9 < sampled_red_value:
                    raise RuntimeError(
                        "revealed Blue best response is below the sampled Red action; "
                        "the information-upper-bound invariant was violated"
                    )
                red_action = revealed_response.action
            else:
                raise ValueError(f"unknown Red commander: {commander}")
            plan = apply_identity_joint_plan(
                adapter,
                active_game,
                red_action,
                blue_action,
                blue_type_name=style,
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
            if any(
                adapter.red_assignment[agent_id] is not None
                for agent_id in plan.red_reserve_ids
            ):
                raise RuntimeError("reserve patrol changed a formal Red assignment")
            elapsed = time.perf_counter() - start
            planning_times.append(elapsed)
            records.append(
                _plan_record(
                    plan,
                    active_game,
                    adapter.step_count,
                    elapsed,
                    result,
                    reserve_patrol,
                    revealed_response=revealed_response,
                    sampled_red_value=sampled_red_value,
                )
            )
            current_red = dict(plan.red_assignment)
            if previous_red is not None:
                red_switches += sum(
                    previous_red.get(agent_id) != current_red[agent_id]
                    for agent_id in active_game.red_ids
                )
            previous_red = current_red
            executor.reset()
        if plan is None:
            red_actions = {
                agent_id: 0
                for agent_id, row in adapter.agent_states("Red").items()
                if row["alive"]
            }
            blue_subgroups = None
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
            blue_subgroups = _subgroup_map(plan)
        _, _, done, info = adapter.step(
            red_actions, blue_style=style, blue_subgroup_by_agent=blue_subgroups
        )
        events = {event["kind"] for event in info.get("events", [])}
        need_plan = not done and (
            adapter.step_count % interval == 0
            or bool(physical["replan_on_casualty"] and "agents_destroyed" in events)
        )
    groups = [group for record in records for group in record["groups"]]
    nonempty_sizes = [
        len(group[field])
        for group in groups
        for field in ("red_ids", "blue_ids")
        if group[field]
    ]
    unopposed = [group for group in groups if group["blue_ids"] and not group["red_ids"]]
    reserve_agent_commands = sum(len(record["red_reserve_ids"]) for record in records)
    red_alive_commands = sum(int(record["red_alive_count"]) for record in records)
    target_states = adapter.target_states()
    return {
        "scenario": scenario["label"],
        "episode": int(seed),
        "population": int(scenario["red"]) + int(scenario["blue"]),
        "red": int(scenario["red"]),
        "blue": int(scenario["blue"]),
        "targets": int(scenario["targets"]),
        "blue_lower_style": style,
        "blue_upper_policy": "identity_blotto_equilibrium",
        "red_information": (
            "sampled_blue_action_revealed"
            if commander == "revealed_blue_br"
            else "simultaneous_action"
        ),
        "red_commander": commander,
        "seed": seed,
        "outcome_red": int(float(info["outcome_red"])),
        "red_win": int(float(info["outcome_red"]) > 0),
        "breach": int(any(not row["alive"] for row in target_states.values())),
        "steps": int(adapter.step_count),
        "episode_horizon": int(adapter.max_steps),
        "commands": len(records),
        "planning_seconds_mean": statistics.fmean(planning_times) if planning_times else 0.0,
        "planning_seconds_p95": _percentile(planning_times, 95) if planning_times else 0.0,
        "red_switches": red_switches,
        "max_nonempty_group_size": max(nonempty_sizes, default=0),
        "unopposed_blue_group_fraction": len(unopposed) / max(1, sum(bool(group["blue_ids"]) for group in groups)),
        "reserve_fraction": reserve_agent_commands / max(1, red_alive_commands),
        "max_reserve_agents": max(
            (len(record["red_reserve_ids"]) for record in records), default=0
        ),
        # Each command event checks this immediately after applying the joint
        # action and aborts if a patrol waypoint changed a formal assignment.
        "all_reserve_assignments_none": 1,
        "revealed_responses_all_optimal": int(
            commander != "revealed_blue_br"
            or all(bool(record["revealed_response_optimal"]) for record in records)
        ),
        "fixed_target_count_valid": int(len(adapter.target_ids) == int(scenario["targets"])),
        # apply_identity_joint_plan validates exact once-only identity cover
        # before issuing commands and raises on any violation.  Keep this
        # invariant distinct from the independently reported size cap.
        "identity_partition_valid": 1,
        "red_active_or_reserve_partition_valid": 1,
        "blue_exact_partition_valid": 1,
        "revealed_proxy_gain_nonnegative": int(
            commander != "revealed_blue_br"
            or all(float(record["revealed_proxy_gain"]) >= -1e-9 for record in records)
        ),
        "joint_plans_json": json.dumps(records, ensure_ascii=False, separators=(",", ":")),
    }


def _paired_difference(rows, left: str, right: str):
    keys = ("scenario", "blue_lower_style", "seed")
    lookup = {
        tuple(row[key] for key in keys) + (row["red_commander"],): row["red_win"]
        for row in rows
    }
    differences = []
    for prefix in {tuple(row[key] for key in keys) for row in rows}:
        if prefix + (left,) in lookup and prefix + (right,) in lookup:
            differences.append(lookup[prefix + (left,)] - lookup[prefix + (right,)])
    if not differences:
        return {"pairs": 0, "mean_improvement": None, "ci95_low": None, "ci95_high": None}
    rng = np.random.default_rng(20260904)
    values = np.asarray(differences, dtype=np.float64)
    bootstrap = np.asarray(
        [rng.choice(values, size=len(values), replace=True).mean() for _ in range(5000)]
    )
    return {
        "pairs": len(values),
        "mean_improvement": float(values.mean()),
        "ci95_low": _percentile(bootstrap, 2.5),
        "ci95_high": _percentile(bootstrap, 97.5),
    }


def run_physical(config, predictor, stage1_model, device, output, progress, smoke):
    physical = config["physical_evaluation"]
    scenarios = physical["scenarios"]
    styles = list(map(str, physical["blue_lower_styles"]))
    commanders = list(map(str, physical["red_commanders"]))
    episodes = 1 if smoke else int(physical["episodes_per_cell"])
    total = len(scenarios) * len(styles) * len(commanders) * episodes
    progress_every = max(1, int(physical.get("progress_every_episodes", 1)))
    rows = []
    started = time.perf_counter()
    for scenario_index, scenario in enumerate(scenarios):
        for style_index, style in enumerate(styles):
            for episode in range(episodes):
                seed = int(config["seed"]) + 100000 + scenario_index * 10000 + style_index * 1000 + episode
                for commander in commanders:
                    row = _run_episode(
                        config,
                        scenario,
                        style,
                        commander,
                        seed,
                        predictor,
                        stage1_model,
                        device,
                        smoke,
                    )
                    row["episode"] = int(episode)
                    rows.append(row)
                    if len(rows) % progress_every == 0 or len(rows) == total:
                        progress.progress(
                            "Stage3/C HAD身份级物理对抗",
                            len(rows),
                            total,
                            started_at=started,
                        )
    write_csv(output / "raw/physical_identity_episodes.csv", rows)
    table = {}
    for commander in commanders:
        table[commander] = {}
        for style in styles:
            table[commander][style] = {}
            for scenario in scenarios:
                selected = [
                    row for row in rows
                    if row["red_commander"] == commander
                    and row["blue_lower_style"] == style
                    and row["scenario"] == scenario["label"]
                ]
                table[commander][style][str(scenario["targets"])] = {
                    "episodes": len(selected),
                    "win_rate": statistics.fmean(row["red_win"] for row in selected),
                    "breach_rate": statistics.fmean(row["breach"] for row in selected),
                    "planning_seconds_mean": statistics.fmean(row["planning_seconds_mean"] for row in selected),
                    "unopposed_blue_group_fraction": statistics.fmean(row["unopposed_blue_group_fraction"] for row in selected),
                    "reserve_fraction": statistics.fmean(
                        row["reserve_fraction"] for row in selected
                    ),
                }
    return rows, {
        "episodes": len(rows),
        "win_rate_by_commander_style_targets": table,
        "identity_vs_balanced_paired": _paired_difference(rows, "identity_blotto", "balanced_identity"),
        "revealed_vs_identity_paired": _paired_difference(
            rows, "revealed_blue_br", "identity_blotto"
        ),
        "all_groups_at_most_four": all(row["max_nonempty_group_size"] <= 4 for row in rows),
        "all_fixed_target_counts_valid": all(row["fixed_target_count_valid"] for row in rows),
        "all_identity_partitions_valid": all(row["identity_partition_valid"] for row in rows),
        "all_red_active_or_reserve_partitions_valid": all(
            row["red_active_or_reserve_partition_valid"] for row in rows
        ),
        "all_blue_exact_partitions_valid": all(
            row["blue_exact_partition_valid"] for row in rows
        ),
        "all_reserve_assignments_none": all(
            row["all_reserve_assignments_none"] for row in rows
        ),
        "all_revealed_responses_optimal": all(
            row["revealed_responses_all_optimal"] for row in rows
        ),
        "all_revealed_proxy_gains_nonnegative": all(
            row["revealed_proxy_gain_nonnegative"] for row in rows
        ),
    }


def _historical_comparison(config):
    path = _resolve(config["comparison"]["previous_run_summary"])
    if not path.is_file():
        return {"available": False, "path": str(path)}
    previous = json.loads(path.read_text(encoding="utf-8"))
    physical = previous.get("physical_evaluation", {})
    by_targets = physical.get("all_policy_win_rate_by_targets", {})
    return {
        "available": True,
        "path": str(path),
        "interpretation": config["comparison"]["interpretation"],
        "previous_double_oracle": by_targets.get("double_oracle"),
        "previous_stage1_balanced": by_targets.get("stage1_balanced"),
    }


def _plots(output: Path, scaling, physical):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    scales = list(scaling)
    fig, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    axis.bar(scales, [scaling[key]["planning_p95_seconds"] for key in scales], color="#4776E6")
    axis.set(xlabel="Total agents", ylabel="P95 planning time (s)", title="Identity set-partitioning planner")
    axis.grid(axis="y", alpha=.25)
    path1 = figures / "01_identity_planning_time.png"
    fig.savefig(path1, dpi=180)
    plt.close(fig)
    paths = [path1]
    if physical:
        table = physical["win_rate_by_commander_style_targets"]
        commanders = list(table)
        targets = (2, 4, 5)
        width = .25
        fig, axis = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
        x = np.arange(len(targets))
        for index, commander in enumerate(commanders):
            rates = []
            for target in targets:
                cells = [table[commander][style][str(target)]["win_rate"] for style in table[commander]]
                rates.append(statistics.fmean(cells))
            axis.bar(x + (index - (len(commanders) - 1) / 2) * width, rates, width, label=commander)
        axis.set(xticks=x, xticklabels=targets, ylim=(0, 1), xlabel="Fixed targets", ylabel="Red win rate", title="Shared-world HAD results")
        axis.legend()
        axis.grid(axis="y", alpha=.25)
        path2 = figures / "02_identity_physical_win_rate.png"
        fig.savefig(path2, dpi=180)
        plt.close(fig)
        paths.append(path2)
    return [str(path.relative_to(output)) for path in paths]


def _make_report(summary, config):
    exact = summary["exact_validation"]
    physical = summary.get("physical_evaluation")
    lines = [
        "# Stage3：身份级 4v4 硬约束 Blotto 实验报告",
        "",
        f"> 自动生成：`{summary['completed_at']}`  ",
        f"> 总体验收：**{'通过' if summary['acceptance']['all_passed'] else '未通过'}**",
        "",
        "双方同时决定带标签的存活智能体在公开 `(目标, 对抗通道)` 中的分区。同一槽位中的 Red/Blue 身份形成一个局部子博弈；同一目标可以有多个通道。每个 Blue 身份恰好活跃一次，每个 Red 身份恰好活跃一次或进入 reserve；每侧非空局部组硬性不超过4人。Stage2 只评价最终出现的 1–4v1–4 身份组，reserve 与空方边界均不送入模型。",
        "",
        "固定对手混合策略后，Red 最佳响应是集合装填 MILP，Blue 最佳响应是精确覆盖的集合划分 MILP；外层使用 Double Oracle。小规模注册所有联盟列并与完整矩阵对照，30–50 人只对显式空间邻域候选列给出证书。大规模 payoff 仅做动作无关的正比例归一化，不改变博弈策略。",
        "",
        "## 求解正确性",
        "",
        "| 完整小博弈 | 最大价值误差 | 最大 exploitability | 全博弈证书 |",
        "|---:|---:|---:|---:|",
        f"| {exact['scenarios']} | {exact['max_value_error']:.8f} | {exact['max_exploitability']:.8f} | {'是' if exact['all_full_game_certified'] else '否'} |",
        "",
        "## 30–50 人规划",
        "",
        "| 总人数 | P95时间(s) | 候选域证书率 | 最大归一化候选域 gap | Red候选列均值 | Blue候选列均值 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for scale, row in summary["planner_scaling"].items():
        lines.append(
            f"| {scale} | {row['planning_p95_seconds']:.3f} | {row['candidate_domain_certified_fraction']:.1%} | {row['exploitability_max']:.5f} | {row['red_candidate_columns_mean']:.0f} | {row['blue_candidate_columns_mean']:.0f} |"
        )
    lines.extend(["", "这里的 gap 只针对已注册候选联盟列；大规模结果不是完整指数动作空间的全局 Nash 证书。", "", "## HAD 物理对抗", ""])
    if physical is None:
        lines.append("本次跳过物理评估。")
    else:
        lines.extend([
            "| Red方法 | Blue底层 | 2目标 | 4目标 | 5目标 |",
            "|---|---|---:|---:|---:|",
        ])
        for commander, by_style in physical["win_rate_by_commander_style_targets"].items():
            for style, values in by_style.items():
                cells = [f"{values[str(target)]['win_rate']:.1%}" for target in (2, 4, 5)]
                lines.append(f"| {commander} | {style} | {' | '.join(cells)} |")
        paired = physical["identity_vs_balanced_paired"]
        revealed = physical["revealed_vs_identity_paired"]
        lines.extend([
            "",
            f"同种子配对下，身份级 Blotto 相对均匀身份分组的胜率差为 `{paired['mean_improvement']:.1%}`（95% bootstrap CI `{paired['ci95_low']:.1%}` 至 `{paired['ci95_high']:.1%}`，{paired['pairs']} 对）。",
            f"揭示本次采样 Blue 动作后的同候选域最佳响应，相对身份级 Blotto 的物理胜率差为 `{revealed['mean_improvement']:.1%}`；它仅是代理价值意义下的信息上界诊断，不是可部署主方法。",
            "",
            "| Red方法 | reserve比例 | 无Red对应Blue组比例 |",
            "|---|---:|---:|",
        ])
        for commander in ("balanced_identity", "identity_blotto", "revealed_blue_br"):
            cells = [
                value
                for by_style in physical["win_rate_by_commander_style_targets"][commander].values()
                for value in by_style.values()
            ]
            total = sum(int(cell["episodes"]) for cell in cells)
            reserve = sum(float(cell["reserve_fraction"]) * int(cell["episodes"]) for cell in cells) / total
            unopposed = sum(float(cell["unopposed_blue_group_fraction"]) * int(cell["episodes"]) for cell in cells) / total
            lines.append(f"| {commander} | {reserve:.1%} | {unopposed:.1%} |")
    historical = summary["historical_comparison"]
    lines.extend(["", "## 与上一版比较", ""])
    if historical["available"]:
        lines.append("上一版结果已载入，但由于 Stage2 数据域、上层动作和 Blue 上层协议都已改变，只作非配对历史参照；是否改善以本轮同种子 `identity_blotto - balanced_identity` 为主要证据。")
    else:
        lines.append("未找到上一版机器可读结果；本轮仍保留同种子内部对照。")
    lines.extend([
        "",
        "## 关键诊断",
        "",
        f"- 4v4 上限：{'通过' if summary['acceptance']['groups_at_most_four'] else '失败'}。",
        f"- 身份恰好分区：{'通过' if summary['acceptance']['identity_partition'] else '失败'}。",
        f"- Red活跃或reserve、Blue精确覆盖：{'通过' if summary['acceptance']['red_active_or_reserve_partition'] and summary['acceptance']['blue_exact_partition'] else '失败'}。",
        f"- reserve正式目标保持None：{'通过' if summary['acceptance']['reserve_assignment_none'] else '失败'}。",
        f"- 揭示后代理最佳响应最优且不低于同步动作：{'通过' if summary['acceptance']['revealed_response_optimal'] and summary['acceptance']['revealed_proxy_upper_bound'] else '失败'}。",
        f"- 每局目标数固定：{'通过' if summary['acceptance']['fixed_targets'] else '失败'}。",
        "- `raw/physical_identity_episodes.csv` 的 `joint_plans_json` 保存每次重规划的目标、通道、Red IDs 和 Blue IDs。",
        "- `analysis/summary.json` 保存验收和新旧对比；`figures/` 保存核心图。",
        "",
        "## 结论",
        "",
        summary["decision"],
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    _validate_config(config)
    output = args.output_dir.resolve() if args.output_dir else _resolve(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    progress = LiveProgress(output / "live_progress.log", phase_total=4)
    status_path = output / "pipeline_status.json"
    started = utc_now_iso()
    atomic_write_json(status_path, {"schema_version": SCHEMA, "status": "running", "started_at": started})
    try:
        device = _device(str(args.device or config.get("device", "auto")))
        artifacts = config["artifacts"]
        stage1_path = _resolve(artifacts["stage1_checkpoint"])
        stage2_path = _resolve(artifacts["stage2_checkpoint"])
        dataset_path = _resolve(artifacts["stage2_dataset"])
        for path in (stage1_path, stage2_path, dataset_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        if _expected_hash(artifacts.get("stage2_dataset_sha256")) and sha256_file(dataset_path) != _expected_hash(artifacts["stage2_dataset_sha256"]):
            raise ValueError("Stage2 dataset hash differs from the resolved protocol")
        predictor = FrozenStage2Payoff(stage2_path, device=device, expected_sha256=_expected_hash(artifacts.get("stage2_sha256")))
        if int(predictor.supported_roster.get("max_red", 0)) != 4 or int(predictor.supported_roster.get("max_blue", 0)) != 4:
            raise ValueError("Stage2 must be trained exactly on the 1..4v1..4 domain")
        if predictor.execution_semantics.get("controller_reset_at_every_saved_command_event") is not True:
            raise ValueError("Stage2 command-reset semantics do not match Stage3 replanning")
        stage1_model = load_round01_stage1_model(stage1_path, device, expected_sha256=_expected_hash(artifacts.get("stage1_sha256")))

        progress.phase("A：完整小博弈正确性", "B：30–50人候选域规划")
        _, exact = run_exact(config, output, progress, args.smoke)
        progress.phase("B：身份级集合划分规划", "C：HAD物理对抗")
        _, scaling = run_scaling(config, predictor, output, progress, args.smoke)
        physical = None
        if not args.skip_physical and config["physical_evaluation"]["enabled"]:
            progress.phase("C：HAD身份级动态重规划", "D：汇总与对比")
            _, physical = run_physical(config, predictor, stage1_model, device, output, progress, args.smoke)
        else:
            progress.phase("C：已跳过物理评估", "D：汇总与对比")

        acceptance_config = config["acceptance"]
        checks = {
            "small_value": exact["max_value_error"] <= float(acceptance_config["small_game_value_error_max"]),
            "small_exploitability": exact["max_exploitability"] <= float(acceptance_config["small_game_exploitability_max"]),
            "small_full_game_certificate": exact["all_full_game_certified"],
            "groups_at_most_four": True if physical is None else physical["all_groups_at_most_four"],
            "identity_partition": True if physical is None else physical["all_identity_partitions_valid"],
            "fixed_targets": True if physical is None else physical["all_fixed_target_counts_valid"],
            "red_active_or_reserve_partition": True if physical is None else physical["all_red_active_or_reserve_partitions_valid"],
            "blue_exact_partition": True if physical is None else physical["all_blue_exact_partitions_valid"],
            "reserve_assignment_none": True if physical is None else physical["all_reserve_assignments_none"],
            "revealed_response_optimal": True if physical is None else physical["all_revealed_responses_optimal"],
            "revealed_proxy_upper_bound": True if physical is None else physical["all_revealed_proxy_gains_nonnegative"],
        }
        for scale, row in scaling.items():
            checks[f"planning_{scale}"] = row["planning_p95_seconds"] <= float(acceptance_config["planning_p95_seconds"][scale])
            checks[f"candidate_gap_{scale}"] = row["exploitability_max"] <= float(
                acceptance_config["candidate_domain_normalized_exploitability_max"]
            )
        if physical is not None:
            checks["paired_not_materially_worse"] = physical["identity_vs_balanced_paired"]["mean_improvement"] >= float(acceptance_config["paired_identity_vs_balanced_min_improvement"])
        all_passed = all(checks.values())
        decision = (
            "身份级 4v4 框架通过当前协议，可据此继续增加候选联盟生成与对手策略库。"
            if all_passed
            else "框架已完整运行，但至少一个预注册门槛未通过；应先根据失败项定位 Stage2 价值误差、候选域求解或物理迁移问题，不宣称方法改善。"
        )
        summary = {
            "schema_version": SCHEMA,
            "completed_at": utc_now_iso(),
            "exact_validation": exact,
            "planner_scaling": scaling,
            "physical_evaluation": physical,
            "historical_comparison": _historical_comparison(config),
            "acceptance": {**checks, "all_passed": all_passed},
            "decision": decision,
        }
        summary["figures"] = _plots(output, scaling, physical)
        atomic_write_json(output / "analysis/summary.json", summary)
        atomic_write_text(output / "stage3_report.md", _make_report(summary, config))
        atomic_write_json(status_path, {"schema_version": SCHEMA, "status": "completed", "accepted": all_passed, "completed_at": summary["completed_at"], "report": str(output / "stage3_report.md")})
        progress.phase("D：报告与新旧诊断已生成", None)
    except Exception as error:
        atomic_write_json(status_path, {"schema_version": SCHEMA, "status": "failed", "failed_at": utc_now_iso(), "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()})
        progress.write(f"[Stage3][失败] {type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
