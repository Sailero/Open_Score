"""Evaluate the coalition-structured Stage-3 Blotto model.

The script produces three compact evidence tracks: exact toy-game agreement,
30/40/50-agent planner scaling, and shared-world HAD rollouts.  It intentionally
does not use the legacy target-count-then-chunk planner.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import traceback
from collections import Counter, defaultdict
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
from open_score.stage3.artifacts import (
    LiveProgress,
    atomic_write_json,
    atomic_write_text,
    utc_now_iso,
    write_csv,
)
from open_score.stage3.blotto import solve_restricted_matrix_game
from open_score.stage3.grouped_blotto import (
    GroupedBlottoGame,
    balanced_grouped_allocation,
    concentrated_grouped_allocation,
    enumerate_grouped_allocations,
    solve_grouped_double_oracle,
)
from open_score.stage3.payoff import FrozenStage2Payoff, sha256_file
from open_score.stage3.runtime import (
    FrozenStage1GroupExecutor,
    apply_joint_grouped_plan,
    build_grouped_event_game,
    load_round01_stage1_model,
    moderate_jittered_target_positions,
)


SCHEMA = "open-score-stage3-cs-bbg-results-v4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT / "configs/stage3_aligned.yaml"
    )
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
    if config.get("schema_version") != "open-score-stage3-cs-bbg-v4":
        raise ValueError("Stage3 config is not the registered CS-BBG v4 protocol")
    game = config["game"]
    support = list(map(int, game["group_size_support"]))
    if support != list(range(1, max(support) + 1)):
        raise ValueError("group_size_support must be contiguous from one")
    invariants = {
        "zero_sum": True,
        "simultaneous_upper_actions": True,
        "fixed_targets_per_episode": True,
        "repeated_target_groups": True,
        "hard_cap_at_four": False,
        "conditional_group_independence": True,
    }
    for key, expected in invariants.items():
        if game.get(key) is not expected:
            raise ValueError(f"registered game invariant failed: {key}={expected}")
    if game.get("action_representation") != "target_group_histogram":
        raise ValueError("Stage3 must solve group patterns, not target totals")
    if game.get("pairing_rule") != "size_assortative":
        raise ValueError("the public group pairing rule is not registered")
    if game.get("include_reserve"):
        raise ValueError("the MVP deploys every live agent")
    if int(game["preferred_group_size"]) != 4 or max(support) <= 4:
        raise ValueError("four must be a soft preference inside wider legal support")
    styles = {str(row["lower_style"]) for row in game["blue_types"]}
    if styles != {"rush", "split_rush"}:
        raise ValueError("the MVP Blue lower-policy library must contain both rules")
    for blue_type in game["blue_types"]:
        if blue_type.get("upper_family") != "strategic":
            raise ValueError("every Blue lower type must retain strategic upper actions")
    for scenario in config["physical_evaluation"]["scenarios"]:
        if not 1 <= int(scenario["targets"]) <= 5:
            raise ValueError("registered HAD demonstrations use one to five targets")


def _initial_strategies(
    game: GroupedBlottoGame, side: str, preferred_size: int
) -> list[tuple[int, ...]]:
    values = [
        balanced_grouped_allocation(
            game, side, preferred_size=min(preferred_size, game.group_size_cap)
        ),
        concentrated_grouped_allocation(
            game, side, target=0, preferred_size=min(preferred_size, game.group_size_cap)
        ),
    ]
    if game.n_targets > 1:
        values.append(
            concentrated_grouped_allocation(
                game,
                side,
                target=game.n_targets - 1,
                preferred_size=min(preferred_size, game.group_size_cap),
            )
        )
    return list(dict.fromkeys(values))


def _solve(game: GroupedBlottoGame, config: Mapping[str, object], seed: int, *, benchmark: bool):
    solver = config["solver"]
    preferred = int(config["game"]["preferred_group_size"])
    return solve_grouped_double_oracle(
        game,
        initial_defender_strategies=_initial_strategies(game, "Red", preferred),
        initial_attacker_strategies=_initial_strategies(game, "Blue", preferred),
        max_iterations=int(
            solver["benchmark_max_iterations"] if benchmark else solver["max_iterations"]
        ),
        tolerance=float(
            solver["benchmark_tolerance"] if benchmark else solver["tolerance"]
        ),
        seed=int(seed),
        verbose=bool(solver.get("verbose", False)),
    )


def _group_histogram(
    game: GroupedBlottoGame,
    strategies: Sequence[Sequence[int]],
    mixture: Sequence[float],
    side: str,
) -> dict[str, float]:
    counts: Counter[int] = Counter()
    total = 0.0
    for strategy, weight in zip(strategies, mixture):
        for target_groups in game.groups(strategy, side):
            for size in target_groups:
                counts[int(size)] += float(weight)
                total += float(weight)
    return {
        str(size): (float(counts[size]) / total if total else 0.0)
        for size in range(1, game.group_size_cap + 1)
    }


def run_exact(
    config: Mapping[str, object], output: Path, progress: LiveProgress, smoke: bool
) -> tuple[list[dict[str, object]], dict[str, object]]:
    spec = config["solver_evaluation"]["exact_small_game"]
    scenarios = 3 if smoke else int(spec["scenarios"])
    seed_base = int(config["seed"]) + 1000
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for index in range(scenarios):
        rng = np.random.default_rng(seed_base + index)
        targets = int(spec["targets"])
        cap = int(spec["group_size_cap"])
        survival = rng.uniform(0.08, 0.98, size=(targets, cap + 1, cap + 1))
        payoff = np.log(np.clip(survival, 1e-6, 1.0)) / abs(math.log(1e-6))
        payoff[:, :, 0] = 0.0
        payoff[:, 0, 1:] = -(int(spec["red"]) + int(spec["blue"]) + 1.0)
        game = GroupedBlottoGame(
            payoff,
            defender_budget=int(spec["red"]),
            attacker_budget=int(spec["blue"]),
        )
        red_actions = enumerate_grouped_allocations(game, "Red")
        blue_actions = enumerate_grouped_allocations(game, "Blue")
        exact_start = time.perf_counter()
        exact = solve_restricted_matrix_game(game.payoff_matrix(red_actions, blue_actions))
        exact_seconds = time.perf_counter() - exact_start
        do_start = time.perf_counter()
        result = _solve(game, config, seed_base + index, benchmark=False)
        do_seconds = time.perf_counter() - do_start
        rows.append(
            {
                "scenario": index,
                "exact_value": exact.value,
                "double_oracle_value": result.value,
                "absolute_value_error": abs(exact.value - result.value),
                "exploitability": result.exploitability,
                "exact_seconds": exact_seconds,
                "double_oracle_seconds": do_seconds,
                "red_pure_actions": len(red_actions),
                "blue_pure_actions": len(blue_actions),
                "red_support": len(result.defender_strategies),
                "blue_support": len(result.attacker_strategies),
                "converged": int(result.converged),
            }
        )
        if (index + 1) % max(1, min(10, scenarios)) == 0 or index + 1 == scenarios:
            progress.progress("Stage3/A 精确性", index + 1, scenarios, started_at=started)
    write_csv(output / "raw/exact_games.csv", rows)
    summary = {
        "scenarios": scenarios,
        "max_value_error": max(float(row["absolute_value_error"]) for row in rows),
        "max_exploitability": max(float(row["exploitability"]) for row in rows),
        "all_converged": all(bool(row["converged"]) for row in rows),
    }
    return rows, summary


def _make_adapter(scenario: Mapping[str, object], physical: Mapping[str, object], seed: int, style: str, *, smoke: bool) -> HADStage3Adapter:
    positions = moderate_jittered_target_positions(
        int(scenario["targets"]), physical["layout_generator"], seed=int(seed)
    )
    adapter = HADStage3Adapter(
        int(scenario["red"]),
        int(scenario["blue"]),
        int(scenario["targets"]),
        max_steps=int(physical["smoke_max_steps"] if smoke else physical["max_steps"]),
        target_positions=positions,
        blue_rule_style=str(style),
        split_spacing=float(physical["blue_split_spacing"]),
    )
    adapter.reset(seed=int(seed))
    return adapter


def run_scaling(
    config: Mapping[str, object],
    predictor: FrozenStage2Payoff,
    output: Path,
    progress: LiveProgress,
    smoke: bool,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    specs = config["solver_evaluation"]["planner_benchmarks"]
    physical = config["physical_evaluation"]
    cap = max(map(int, config["game"]["group_size_support"]))
    rows: list[dict[str, object]] = []
    total = sum(1 if smoke else int(row["repeats"]) for row in specs)
    completed = 0
    started = time.perf_counter()
    for spec_index, spec in enumerate(specs):
        repeats = 1 if smoke else int(spec["repeats"])
        for repeat in range(repeats):
            seed = int(config["seed"]) + 20000 + spec_index * 1000 + repeat
            scenario = dict(spec)
            style = "rush" if repeat % 2 == 0 else "split_rush"
            adapter = _make_adapter(scenario, physical, seed, style, smoke=smoke)
            payoff_start = time.perf_counter()
            built = build_grouped_event_game(
                adapter,
                predictor,
                group_size_cap=cap,
                blue_style=style,
                batch_size=int(config["solver"]["payoff_batch_size"]),
                utility_mode=str(config["game"]["local_utility"]),
                risk_epsilon=float(config["game"]["risk_epsilon"]),
            )
            payoff_seconds = time.perf_counter() - payoff_start
            solve_start = time.perf_counter()
            result = _solve(built.game, config, seed, benchmark=True)
            solve_seconds = time.perf_counter() - solve_start
            red_hist = _group_histogram(
                built.game, result.defender_strategies, result.defender_mixture, "Red"
            )
            blue_hist = _group_histogram(
                built.game, result.attacker_strategies, result.attacker_mixture, "Blue"
            )
            rows.append(
                {
                    "scale": str(spec["label"]),
                    "repeat": repeat,
                    "red": int(spec["red"]),
                    "blue": int(spec["blue"]),
                    "targets": int(spec["targets"]),
                    "blue_style": style,
                    "payoff_seconds": payoff_seconds,
                    "solve_seconds": solve_seconds,
                    "planning_seconds": payoff_seconds + solve_seconds,
                    "exploitability": result.exploitability,
                    "exploitability_per_target": result.exploitability / int(spec["targets"]),
                    "iterations": len(result.history),
                    "red_support": len(result.defender_strategies),
                    "blue_support": len(result.attacker_strategies),
                    "converged": int(result.converged),
                    "red_group_size_distribution_json": json.dumps(red_hist, sort_keys=True),
                    "blue_group_size_distribution_json": json.dumps(blue_hist, sort_keys=True),
                    "red_group_fraction_above_four": sum(
                        value for key, value in red_hist.items() if int(key) > 4
                    ),
                    "blue_group_fraction_above_four": sum(
                        value for key, value in blue_hist.items() if int(key) > 4
                    ),
                }
            )
            completed += 1
            progress.progress("Stage3/B 规模", completed, total, started_at=started)
    write_csv(output / "raw/planner_scaling.csv", rows)
    by_scale: dict[str, object] = {}
    for label in [str(spec["label"]) for spec in specs]:
        subset = [row for row in rows if row["scale"] == label]
        by_scale[label] = {
            "runs": len(subset),
            "planning_p50_seconds": _percentile(
                [float(row["planning_seconds"]) for row in subset], 50
            ),
            "planning_p95_seconds": _percentile(
                [float(row["planning_seconds"]) for row in subset], 95
            ),
            "max_exploitability_per_target": max(
                float(row["exploitability_per_target"]) for row in subset
            ),
            "converged_fraction": statistics.fmean(
                float(row["converged"]) for row in subset
            ),
            "red_group_fraction_above_four": statistics.fmean(
                float(row["red_group_fraction_above_four"]) for row in subset
            ),
            "blue_group_fraction_above_four": statistics.fmean(
                float(row["blue_group_fraction_above_four"]) for row in subset
            ),
        }
    return rows, by_scale


def _alive_count(adapter: HADStage3Adapter, side: str) -> int:
    return sum(bool(row["alive"]) for row in adapter.agent_states(side).values())


def _subgroup_map(plan) -> dict[int, int]:
    return {
        int(agent_id): int(subgame.subgroup_id)
        for subgame in plan.local_subgames
        for agent_id in subgame.blue_ids
    }


def _plan_record(plan, step: int, result, planning_seconds: float) -> dict[str, object]:
    return {
        "step": int(step),
        "planning_seconds": float(planning_seconds),
        "exploitability": float(result.exploitability),
        "red_target_counts": list(plan.red_allocation),
        "blue_target_counts": list(plan.blue_allocation),
        "groups": [
            {
                "target": int(item.target_id),
                "group": int(item.subgroup_id),
                "red_ids": list(item.red_ids),
                "blue_ids": list(item.blue_ids),
            }
            for item in plan.local_subgames
        ],
    }


def _run_episode(
    config: Mapping[str, object],
    scenario: Mapping[str, object],
    style: str,
    commander: str,
    seed: int,
    predictor: FrozenStage2Payoff,
    stage1_model: torch.nn.Module,
    device: torch.device,
    *,
    smoke: bool,
) -> dict[str, object]:
    physical = config["physical_evaluation"]
    adapter = _make_adapter(scenario, physical, seed, style, smoke=smoke)
    executor = FrozenStage1GroupExecutor(
        stage1_model, device, micro_grouping="planned_groups"
    )
    executor.reset()
    rng = np.random.default_rng(int(seed) ^ 0xC5BB6)
    cap = max(map(int, config["game"]["group_size_support"]))
    preferred = int(config["game"]["preferred_group_size"])
    interval = int(physical["command_interval"])
    replan_casualty = bool(physical["replan_on_casualty"])
    plan = None
    result = None
    roster_override = None
    blue_subgroups = None
    plan_records: list[dict[str, object]] = []
    planning_times: list[float] = []
    done = False
    info: dict[str, object] = {"outcome_red": 0.0}
    need_plan = True
    while not done:
        red_alive = _alive_count(adapter, "Red")
        blue_alive = _alive_count(adapter, "Blue")
        if need_plan and red_alive and blue_alive:
            plan_start = time.perf_counter()
            built = build_grouped_event_game(
                adapter,
                predictor,
                group_size_cap=cap,
                blue_style=style,
                batch_size=int(config["solver"]["payoff_batch_size"]),
                utility_mode=str(config["game"]["local_utility"]),
                risk_epsilon=float(config["game"]["risk_epsilon"]),
            )
            result = _solve(built.game, config, seed + adapter.step_count, benchmark=False)
            equilibrium_red, equilibrium_blue = result.sample_profile(rng=rng)
            red_action = (
                equilibrium_red
                if commander == "cs_bbg_double_oracle"
                else balanced_grouped_allocation(
                    built.game,
                    "Red",
                    preferred_size=min(preferred, built.game.red_group_size_cap),
                )
                if commander == "balanced_group_baseline"
                else None
            )
            if red_action is None:
                raise ValueError(f"unknown Red commander: {commander}")
            plan = apply_joint_grouped_plan(
                adapter,
                built.game,
                red_action,
                equilibrium_blue,
                blue_type_name=style,
                switch_cost=float(config["solver"]["identity_switch_cost_seconds"]),
            )
            elapsed = time.perf_counter() - plan_start
            planning_times.append(elapsed)
            plan_records.append(_plan_record(plan, adapter.step_count, result, elapsed))
            roster_override = {
                (item.target_id, item.subgroup_id): (item.red_ids, item.blue_ids)
                for item in plan.local_subgames
            }
            blue_subgroups = _subgroup_map(plan)
        if roster_override is None:
            red_actions = {
                agent_id: 0
                for agent_id, row in adapter.agent_states("Red").items()
                if bool(row["alive"])
            }
        else:
            red_actions = executor.act(
                adapter,
                local_steps={target: adapter.step_count for target in adapter.target_ids},
                roster_override=roster_override,
            )
        _, _, done, info = adapter.step(
            red_actions,
            blue_style=style,
            blue_subgroup_by_agent=blue_subgroups,
        )
        event_kinds = {str(event["kind"]) for event in info.get("events", [])}
        need_plan = (
            not done
            and (
                adapter.step_count % interval == 0
                or (replan_casualty and bool(event_kinds & {"agents_destroyed"}))
            )
        )

    all_groups = [group for record in plan_records for group in record["groups"]]
    nonempty_sizes = [
        len(group[side])
        for group in all_groups
        for side in ("red_ids", "blue_ids")
        if group[side]
    ]
    target_states = adapter.target_states()
    return {
        "scenario": str(scenario["label"]),
        "population": int(scenario["red"]) + int(scenario["blue"]),
        "red": int(scenario["red"]),
        "blue": int(scenario["blue"]),
        "targets": int(scenario["targets"]),
        "blue_lower_style": str(style),
        "blue_upper_policy": "cs_bbg_equilibrium",
        "red_commander": str(commander),
        "seed": int(seed),
        "outcome_red": int(float(info["outcome_red"])),
        "red_win": int(float(info["outcome_red"]) > 0.0),
        "breach": int(any(not bool(value["alive"]) for value in target_states.values())),
        "target_survivors": sum(bool(value["alive"]) for value in target_states.values()),
        "episode_horizon": int(adapter.max_steps),
        "steps": int(adapter.step_count),
        "commands": len(plan_records),
        "planning_seconds_mean": statistics.fmean(planning_times) if planning_times else 0.0,
        "planning_seconds_p95": _percentile(planning_times, 95) if planning_times else 0.0,
        "do_exploitability_mean": statistics.fmean(
            float(record["exploitability"]) for record in plan_records
        )
        if plan_records
        else math.nan,
        "group_count": len(all_groups),
        "max_nonempty_group_size": max(nonempty_sizes, default=0),
        "group_fraction_above_four": (
            sum(size > 4 for size in nonempty_sizes) / len(nonempty_sizes)
            if nonempty_sizes
            else 0.0
        ),
        "fixed_target_count_valid": int(len(adapter.target_ids) == int(scenario["targets"])),
        "joint_plans_json": json.dumps(plan_records, ensure_ascii=False, separators=(",", ":")),
    }


def run_physical(
    config: Mapping[str, object],
    predictor: FrozenStage2Payoff,
    stage1_model: torch.nn.Module,
    device: torch.device,
    output: Path,
    progress: LiveProgress,
    smoke: bool,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    physical = config["physical_evaluation"]
    scenarios = physical["scenarios"]
    styles = list(map(str, physical["blue_lower_styles"]))
    commanders = list(map(str, physical["red_commanders"]))
    episodes = 1 if smoke else int(physical["episodes_per_cell"])
    total = len(scenarios) * len(styles) * len(commanders) * episodes
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    completed = 0
    for scenario_index, scenario in enumerate(scenarios):
        for style_index, style in enumerate(styles):
            for episode in range(episodes):
                # The two commanders share an initial-world seed in every cell.
                world_seed = (
                    int(config["seed"])
                    + 100000
                    + scenario_index * 10000
                    + style_index * 1000
                    + episode
                )
                for commander in commanders:
                    row = _run_episode(
                            config,
                            scenario,
                            style,
                            commander,
                            world_seed,
                            predictor,
                            stage1_model,
                            device,
                            smoke=smoke,
                        )
                    row["episode"] = int(episode)
                    rows.append(row)
                    completed += 1
                    progress.progress(
                        "Stage3/C HAD物理对抗", completed, total, started_at=started
                    )
    write_csv(output / "raw/physical_episodes.csv", rows)
    table: dict[str, object] = {}
    for commander in commanders:
        table[commander] = {}
        for style in styles:
            table[commander][style] = {}
            for scenario in scenarios:
                subset = [
                    row
                    for row in rows
                    if row["red_commander"] == commander
                    and row["blue_lower_style"] == style
                    and row["scenario"] == scenario["label"]
                ]
                table[commander][style][str(scenario["targets"])] = {
                    "episodes": len(subset),
                    "win_rate": statistics.fmean(float(row["red_win"]) for row in subset),
                    "breach_rate": statistics.fmean(float(row["breach"]) for row in subset),
                    "mean_planning_seconds": statistics.fmean(
                        float(row["planning_seconds_mean"]) for row in subset
                    ),
                }
    return rows, {
        "episodes": len(rows),
        "win_rate_by_commander_style_targets": table,
        "all_fixed_target_counts_valid": all(
            bool(row["fixed_target_count_valid"]) for row in rows
        ),
        "all_groups_within_registered_support": all(
            int(row["max_nonempty_group_size"])
            <= max(map(int, config["game"]["group_size_support"]))
            for row in rows
        ),
        "observed_group_fraction_above_four": (
            statistics.fmean(float(row["group_fraction_above_four"]) for row in rows)
            if rows
            else 0.0
        ),
    }


def _make_report(summary: Mapping[str, object], config: Mapping[str, object]) -> str:
    exact = summary["exact_validation"]
    scaling = summary["planner_scaling"]
    physical = summary.get("physical_evaluation")
    lines = [
        "# Stage3 CS-BBG 自动实验报告",
        "",
        f"> 生成时间：`{summary['completed_at']}`  ",
        f"> 协议：`{SCHEMA}`  ",
        f"> 总体验收：**{'通过' if summary['acceptance']['all_passed'] else '未通过'}**",
        "",
        "## A. 小规模求解正确性",
        "",
        "| 完整矩阵场景数 | 最大价值误差 | 最大 Nash gap | 全部收敛 |",
        "|---:|---:|---:|---:|",
        f"| {exact['scenarios']} | {exact['max_value_error']:.6f} | "
        f"{exact['max_exploitability']:.6f} | {'是' if exact['all_converged'] else '否'} |",
        "",
        "## B. 30–50 人规划",
        "",
        "| 总规模 | 规划 P95(s) | 收敛率 | 最大每目标 gap | Red >4组占比 | Blue >4组占比 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for scale, row in scaling.items():
        lines.append(
            f"| {scale} | {row['planning_p95_seconds']:.3f} | "
            f"{row['converged_fraction']:.1%} | "
            f"{row['max_exploitability_per_target']:.5f} | "
            f"{row['red_group_fraction_above_four']:.1%} | "
            f"{row['blue_group_fraction_above_four']:.1%} |"
        )
    lines.extend(
        [
            "",
            "这里的 4 人是初始化偏好，不是约束；大于 4 的比例是求解结果，而不是预先设为零。",
            "",
            "## C. HAD 物理对抗",
            "",
        ]
    )
    if physical is None:
        lines.append("本次按命令跳过了物理评估。")
    else:
        lines.extend(
            [
                "| Red 上层 | Blue 底层 | 2目标胜率 | 4目标胜率 | 5目标胜率 |",
                "|---|---|---:|---:|---:|",
            ]
        )
        table = physical["win_rate_by_commander_style_targets"]
        for commander, by_style in table.items():
            for style, values in by_style.items():
                rates = [values.get(str(target), {}).get("win_rate") for target in (2, 4, 5)]
                cells = ["—" if value is None else f"{float(value):.1%}" for value in rates]
                lines.append(f"| {commander} | {style} | {' | '.join(cells)} |")
        lines.extend(
            [
                "",
                "两种 Red commander 面对相同的 CS-BBG Blue 均衡上层策略；Blue 底层分别为 rush 与 split_rush。物理结果检验的是局部独立代理能否迁移到共享世界，并非求解器正确性的替代证据。",
            ]
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            summary["decision"],
            "",
            "## 原始结果",
            "",
            "- `raw/exact_games.csv`：小规模完整矩阵与 Double Oracle 对照。",
            "- `raw/planner_scaling.csv`：30/40/50 人规划时间、gap 与组规模。",
            "- `raw/physical_episodes.csv`：逐局胜负及每次命令的实名分组 JSON。",
            "- `analysis/summary.json`：机器可读汇总与验收项。",
            "- `live_progress.log`：实时进度与失败位置。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Stage3 config must be a mapping")
    _validate_config(config)
    output = (
        args.output_dir.resolve()
        if args.output_dir
        else _resolve(str(config["output_dir"]))
    )
    output.mkdir(parents=True, exist_ok=True)
    progress = LiveProgress(output / "live_progress.log", phase_total=4)
    status_path = output / "pipeline_status.json"
    started_at = utc_now_iso()
    atomic_write_json(
        status_path,
        {"schema_version": SCHEMA, "status": "running", "started_at": started_at},
    )
    try:
        requested_device = str(args.device or config.get("device", "auto"))
        device = _device(requested_device)
        artifacts = config["artifacts"]
        stage1_path = _resolve(artifacts["stage1_checkpoint"])
        stage2_path = _resolve(artifacts["stage2_checkpoint"])
        dataset_path = _resolve(artifacts["stage2_dataset"])
        for path in (stage1_path, stage2_path, dataset_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        dataset_expected = _expected_hash(artifacts.get("stage2_dataset_sha256"))
        if dataset_expected and sha256_file(dataset_path) != dataset_expected:
            raise ValueError("Stage2 dataset hash differs from the resolved protocol")
        predictor = FrozenStage2Payoff(
            stage2_path,
            device=device,
            expected_sha256=_expected_hash(artifacts.get("stage2_sha256")),
        )
        max_cap = max(map(int, config["game"]["group_size_support"]))
        if int(predictor.supported_roster.get("max_red", 0)) < max_cap or int(
            predictor.supported_roster.get("max_blue", 0)
        ) < max_cap:
            raise ValueError("frozen Stage2 does not cover every legal Stage3 group size")
        if predictor.execution_semantics.get("local_group_semantics") != "one_target_one_group_pair":
            raise ValueError("frozen Stage2 is not the registered one-group value model")
        stage1_model = load_round01_stage1_model(
            stage1_path,
            device,
            expected_sha256=_expected_hash(artifacts.get("stage1_sha256")),
        )

        progress.phase("A：小规模完整矩阵正确性", "B：30–50 人规划")
        _, exact = run_exact(config, output, progress, args.smoke)
        progress.phase("B：30–50 人匿名组型规划", "C：HAD 物理对抗")
        _, scaling = run_scaling(config, predictor, output, progress, args.smoke)
        physical_summary = None
        if not args.skip_physical and bool(config["physical_evaluation"]["enabled"]):
            progress.phase("C：固定目标 HAD 物理对抗", "D：汇总报告")
            _, physical_summary = run_physical(
                config,
                predictor,
                stage1_model,
                device,
                output,
                progress,
                args.smoke,
            )
        else:
            progress.phase("C：物理对抗（已跳过）", "D：汇总报告")

        acceptance_config = config["acceptance"]
        checks = {
            "exact_value": exact["max_value_error"]
            <= float(acceptance_config["small_game_value_error_max"]),
            "exact_gap": exact["max_exploitability"]
            <= float(acceptance_config["small_game_exploitability_max"]),
            "planner_gap": all(
                float(row["max_exploitability_per_target"])
                <= float(acceptance_config["benchmark_exploitability_per_target_max"])
                for row in scaling.values()
            ),
            "planner_convergence": all(
                float(row["converged_fraction"]) == 1.0
                for row in scaling.values()
            ),
            "planner_time": all(
                float(row["planning_p95_seconds"])
                <= float(acceptance_config["planning_p95_seconds"][scale])
                for scale, row in scaling.items()
            ),
            "fixed_targets": physical_summary is None
            or bool(physical_summary["all_fixed_target_counts_valid"]),
            "legal_groups": physical_summary is None
            or bool(physical_summary["all_groups_within_registered_support"]),
            "physical_schedule_complete": physical_summary is None
            or int(physical_summary["episodes"])
            == len(config["physical_evaluation"]["scenarios"])
            * len(config["physical_evaluation"]["blue_lower_styles"])
            * len(config["physical_evaluation"]["red_commanders"])
            * (1 if args.smoke else int(config["physical_evaluation"]["episodes_per_cell"])),
        }
        all_passed = all(checks.values())
        decision = (
            "匿名组型 CS-BBG 的建模、精确最佳响应和运行边界均通过；可继续依据 HAD 胜率判断局部独立代理是否具有经验有效性。"
            if all_passed
            else "至少一项正确性或实时性验收未通过；应先查看失败项，不应据此宣称 Stage3 已有效。"
        )
        completed_at = utc_now_iso()
        summary: dict[str, object] = {
            "schema_version": SCHEMA,
            "status": "completed",
            "started_at": started_at,
            "completed_at": completed_at,
            "device": str(device),
            "smoke": bool(args.smoke),
            "exact_validation": exact,
            "planner_scaling": scaling,
            "physical_evaluation": physical_summary,
            "acceptance": {"checks": checks, "all_passed": all_passed},
            "decision": decision,
        }
        progress.phase("D：自动汇总与报告")
        atomic_write_json(output / "analysis/summary.json", summary)
        atomic_write_text(output / "stage3_report.md", _make_report(summary, config))
        atomic_write_json(
            status_path,
            {
                "schema_version": SCHEMA,
                "status": "completed",
                "started_at": started_at,
                "completed_at": completed_at,
                "accepted": all_passed,
                "report": str(output / "stage3_report.md"),
            },
        )
        progress.write(f"[Stage3] 完成；报告：{output / 'stage3_report.md'}")
    except Exception as error:
        atomic_write_json(
            status_path,
            {
                "schema_version": SCHEMA,
                "status": "failed",
                "started_at": started_at,
                "failed_at": utc_now_iso(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            },
        )
        progress.write(f"[Stage3] 失败：{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
