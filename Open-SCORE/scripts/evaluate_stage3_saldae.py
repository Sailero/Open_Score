"""Evaluate MILP-warm-started SALDAE-DO for identity-level Stage 3."""

from __future__ import annotations

import argparse
import json
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

# Reuse the accepted HAD protocol helpers, not its solver selection or report.
import evaluate_stage3_identity as identity_v5

from open_score.envs import HADStage3Adapter
from open_score.stage3 import (
    FrozenStage1GroupExecutor,
    FrozenStage2Payoff,
    IdentityBlottoGame,
    SALDAEConfig,
    apply_identity_joint_plan,
    build_identity_event_game,
    build_observable_threat_patrol,
    enumerate_coalitions,
    enumerate_identity_actions,
    identity_roster_override,
    load_round01_stage1_model,
    make_engagement_slots,
    round_robin_identity_action,
    solve_identity_double_oracle,
    solve_restricted_matrix_game,
    solve_saldae_double_oracle,
)
from open_score.stage3.artifacts import (
    LiveProgress,
    atomic_write_json,
    atomic_write_text,
    utc_now_iso,
    write_csv,
)
from open_score.stage3.payoff import sha256_file


SCHEMA = "open-score-stage3-saldae-do-results-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT / "configs/stage3_saldae.yaml"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-physical", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT / path).resolve()


def _expected_hash(value: object) -> str:
    text = str(value or "")
    return "" if text.upper() == "AUTO" else text


def _percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _validate_config(config: Mapping[str, object]) -> None:
    if config.get("schema_version") != "open-score-stage3-saldae-do-v1":
        raise ValueError("not a registered SALDAE-DO v1 configuration")
    method = config["method"]
    if method.get("model_changed_from_v5") is not False:
        raise ValueError("this experiment must keep the v5 game model fixed")
    if method.get("outer_solver") != "double_oracle":
        raise ValueError("SALDAE must remain a best-response oracle inside DO")
    if method.get("claims_global_nash_certificate") is not False:
        raise ValueError("an anytime SALDAE oracle cannot claim a global Nash certificate")
    game = config["game"]
    if game.get("group_size_support") != [1, 2, 3, 4]:
        raise ValueError("SALDAE experiment requires the hard 1..4 group domain")
    if game.get("fixed_targets_per_episode") is not True:
        raise ValueError("target count must remain fixed within an episode")
    if game.get("red_reserve_enabled") is not True or game.get("blue_reserve_enabled") is not False:
        raise ValueError("Red reserve / Blue exact-cover protocol drifted")
    if config["physical_evaluation"].get("matchups") != ["milp_do", "saldae_do"]:
        raise ValueError("the paired solver comparison has drifted")
    for scenario in config["physical_evaluation"]["scenarios"]:
        if not 1 <= int(scenario["targets"]) <= 5:
            raise ValueError("HAD demonstrations use at most five fixed targets")


def _saldae_config(config, *, seed: int, smoke: bool = False) -> SALDAEConfig:
    values = config["saldae_solver"]
    return SALDAEConfig(
        search_agents=int(values["search_agents"]),
        time_limit_seconds=(
            min(0.06, float(values["time_limit_seconds_per_best_response"]))
            if smoke
            else float(values["time_limit_seconds_per_best_response"])
        ),
        max_expansions=(
            min(10, int(values["max_expansions"]))
            if smoke
            else int(values["max_expansions"])
        ),
        keep_children_multiplier=float(values["keep_children_multiplier"]),
        child_sample_multiplier=float(values["child_sample_multiplier"]),
        selection_rounds=(1 if smoke else int(values["selection_rounds"])),
        omega=float(values["omega"]),
        reserve_memory=int(values["reserve_memory"]),
        bridge_path_limit=(
            min(4, int(values["bridge_path_limit"]))
            if smoke
            else int(values["bridge_path_limit"])
        ),
        random_seed=int(seed),
    )


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
        batch_size=int(config["milp_solver"]["payoff_batch_size"]),
        utility_mode=str(config["game"]["local_utility"]),
        risk_epsilon=float(config["game"]["risk_epsilon"]),
        allow_unregistered_actions=True,
    )


def _solve_milp(built, config):
    solver = config["milp_solver"]
    return solve_identity_double_oracle(
        built.game,
        built.initial_red,
        built.initial_blue,
        tolerance=float(solver["tolerance"]),
        max_iterations=int(solver["max_iterations"]),
        oracle_time_limit_seconds=float(solver["oracle_time_limit_seconds"]),
        oracle_mip_relative_gap=float(solver["oracle_mip_relative_gap"]),
    )


def _solve_saldae(built, config, milp_result, *, seed: int, smoke: bool, verbose=False):
    solver = config["saldae_solver"]
    return solve_saldae_double_oracle(
        built.game,
        milp_result.red_strategies,
        milp_result.blue_strategies,
        tolerance=float(solver["tolerance"]),
        max_iterations=(
            min(2, int(solver["max_iterations"]))
            if smoke
            else int(solver["max_iterations"])
        ),
        saldae_config=_saldae_config(config, seed=seed, smoke=smoke),
        verbose=verbose,
    )


def _complete_candidates(ids, slots):
    values = enumerate_coalitions(ids, 4)
    return tuple(values for _ in slots)


def run_exact(config, output: Path, progress: LiveProgress, smoke: bool):
    spec = config["solver_evaluation"]["exact_small_game"]
    scenarios = 2 if smoke else int(spec["scenarios"])
    rows = []
    started = time.perf_counter()
    for index in range(scenarios):
        red_ids = tuple(range(int(spec["red"])))
        blue_ids = tuple(range(100, 100 + int(spec["blue"])))
        slots = make_engagement_slots(
            tuple(range(int(spec["targets"]))),
            max(len(red_ids), len(blue_ids)),
            channels_per_target=max(len(red_ids), len(blue_ids)),
        )
        game = IdentityBlottoGame(
            red_ids,
            blue_ids,
            slots,
            _complete_candidates(red_ids, slots),
            _complete_candidates(blue_ids, slots),
            identity_v5._ToyIdentityOracle(
                red_ids, blue_ids, slots, int(config["seed"]) + index
            ),
            full_coalition_domain=True,
            allow_red_reserve=True,
            allow_blue_reserve=False,
            allow_unregistered_actions=True,
        )
        red_actions = enumerate_identity_actions(red_ids, slots, allow_reserve=True)
        blue_actions = enumerate_identity_actions(blue_ids, slots, allow_reserve=False)
        exact = solve_restricted_matrix_game(game.payoff_matrix(red_actions, blue_actions))
        red_seed = [round_robin_identity_action(red_ids, slots)]
        blue_seed = [round_robin_identity_action(blue_ids, slots)]
        search_start = time.perf_counter()
        saldae, diagnostics = solve_saldae_double_oracle(
            game,
            red_seed,
            blue_seed,
            tolerance=float(config["saldae_solver"]["tolerance"]),
            max_iterations=(8 if smoke else 12),
            saldae_config=SALDAEConfig(
                **{
                    **_saldae_config(
                        config,
                        seed=int(config["seed"]) + 5000 + index,
                        smoke=smoke,
                    ).__dict__,
                    "time_limit_seconds": 0.30 if smoke else 0.55,
                    "max_expansions": 80 if smoke else 160,
                }
            ),
        )
        rows.append(
            {
                "scenario": index,
                "red_pure_actions": len(red_actions),
                "blue_pure_actions": len(blue_actions),
                "exact_value": exact.value,
                "saldae_value": saldae.value,
                "absolute_value_error": abs(exact.value - saldae.value),
                "saldae_empirical_exploitability": saldae.exploitability,
                "saldae_seconds": time.perf_counter() - search_start,
                "saldae_iterations": len(saldae.history),
                "saldae_evaluated_nodes": sum(
                    row.evaluated_nodes
                    for row in (*diagnostics.red_searches, *diagnostics.blue_searches)
                ),
                "global_certificate_claimed": int(saldae.full_game_certified),
            }
        )
        progress.progress(
            "Stage3-SALDAE/A 小规模完整矩阵",
            index + 1,
            scenarios,
            started_at=started,
        )
    write_csv(output / "raw/small_exact_comparison.csv", rows)
    return {
        "scenarios": scenarios,
        "value_error_mean": statistics.fmean(row["absolute_value_error"] for row in rows),
        "value_error_max": max(row["absolute_value_error"] for row in rows),
        "false_global_certificates": sum(row["global_certificate_claimed"] for row in rows),
        "planning_seconds_mean": statistics.fmean(row["saldae_seconds"] for row in rows),
    }


def run_scaling(config, predictor, output: Path, progress: LiveProgress, smoke: bool):
    specs = config["solver_evaluation"]["planner_benchmarks"]
    total = sum(1 if smoke else int(spec["repeats"]) for spec in specs)
    rows = []
    completed = 0
    started = time.perf_counter()
    for spec_index, spec in enumerate(specs):
        repeats = 1 if smoke else int(spec["repeats"])
        for repeat in range(repeats):
            seed = int(config["seed"]) + 10000 + spec_index * 1000 + repeat
            adapter = identity_v5._make_adapter(
                spec, config["physical_evaluation"], seed, "rush", smoke
            )
            built = _build(adapter, predictor, config, "rush")
            milp_start = time.perf_counter()
            milp = _solve_milp(built, config)
            milp_seconds = time.perf_counter() - milp_start
            saldae_start = time.perf_counter()
            saldae, diagnostics = _solve_saldae(
                built,
                config,
                milp,
                seed=seed ^ 0x5A1DAE,
                smoke=smoke,
            )
            saldae_seconds = time.perf_counter() - saldae_start
            first = saldae.history[0]
            searches = (*diagnostics.red_searches, *diagnostics.blue_searches)
            rows.append(
                {
                    "scale": str(spec["label"]),
                    "red": int(spec["red"]),
                    "blue": int(spec["blue"]),
                    "targets": int(spec["targets"]),
                    "repeat": repeat,
                    "milp_seconds": milp_seconds,
                    "saldae_extension_seconds": saldae_seconds,
                    "saldae_total_seconds": milp_seconds + saldae_seconds,
                    "milp_value": milp.value,
                    "saldae_value": saldae.value,
                    "red_out_of_domain_response_gain": first.red_gap,
                    "blue_out_of_domain_response_gain": first.blue_gap,
                    "saldae_empirical_exploitability": saldae.exploitability,
                    "milp_registered_exploitability": milp.exploitability,
                    "saldae_iterations": len(saldae.history),
                    "saldae_nodes": sum(row.evaluated_nodes for row in searches),
                    "saldae_expansions": sum(row.expansions for row in searches),
                    "used_unregistered_coalition": int(
                        diagnostics.expanded_action_domain
                    ),
                    "global_certificate_claimed": int(saldae.full_game_certified),
                }
            )
            completed += 1
            progress.progress(
                "Stage3-SALDAE/B 30–50人规划",
                completed,
                total,
                started_at=started,
            )
    write_csv(output / "raw/planner_comparison.csv", rows)
    summary = {}
    for spec in specs:
        scale = str(spec["label"])
        selected = [row for row in rows if row["scale"] == scale]
        summary[scale] = {
            "runs": len(selected),
            "milp_seconds_p50": _percentile(
                [row["milp_seconds"] for row in selected], 50
            ),
            "saldae_total_seconds_p50": _percentile(
                [row["saldae_total_seconds"] for row in selected], 50
            ),
            "saldae_total_seconds_p95": _percentile(
                [row["saldae_total_seconds"] for row in selected], 95
            ),
            "red_response_gain_mean": statistics.fmean(
                row["red_out_of_domain_response_gain"] for row in selected
            ),
            "blue_response_gain_mean": statistics.fmean(
                row["blue_out_of_domain_response_gain"] for row in selected
            ),
            "expanded_domain_fraction": statistics.fmean(
                row["used_unregistered_coalition"] for row in selected
            ),
            "saldae_nodes_mean": statistics.fmean(
                row["saldae_nodes"] for row in selected
            ),
            "empirical_exploitability_mean": statistics.fmean(
                row["saldae_empirical_exploitability"] for row in selected
            ),
        }
    return rows, summary


def _subgroup_map(plan):
    return {
        int(agent_id): int(item.subgroup_id)
        for item in plan.local_subgames
        for agent_id in item.blue_ids
    }


def _run_episode(
    config,
    scenario,
    style,
    method,
    seed,
    predictor,
    stage1_model,
    device,
    smoke,
):
    physical = config["physical_evaluation"]
    adapter: HADStage3Adapter = identity_v5._make_adapter(
        scenario, physical, seed, style, smoke
    )
    executor = FrozenStage1GroupExecutor(
        stage1_model, device, micro_grouping="planned_groups"
    )
    interval = int(physical["command_interval"])
    rng = np.random.default_rng(seed ^ 0x5A1DAE)
    plan = None
    patrol = None
    records = []
    planning_times = []
    done = False
    need_plan = True
    info = {"outcome_red": 0.0}
    saldae_nodes = 0
    saldae_outside_events = 0
    while not done:
        red_alive = sum(row["alive"] for row in adapter.agent_states("Red").values())
        blue_alive = sum(row["alive"] for row in adapter.agent_states("Blue").values())
        if need_plan and red_alive and blue_alive:
            planning_start = time.perf_counter()
            built = _build(adapter, predictor, config, style)
            milp = _solve_milp(built, config)
            diagnostics = None
            if method == "milp_do":
                result = milp
            elif method == "saldae_do":
                result, diagnostics = _solve_saldae(
                    built,
                    config,
                    milp,
                    seed=seed + adapter.step_count * 1009,
                    smoke=smoke,
                )
                saldae_nodes += sum(
                    row.evaluated_nodes
                    for row in (*diagnostics.red_searches, *diagnostics.blue_searches)
                )
                saldae_outside_events += int(diagnostics.expanded_action_domain)
            else:
                raise ValueError(f"unknown solver method: {method}")
            red_action, blue_action = result.sample_profile(rng=rng)
            plan = apply_identity_joint_plan(
                adapter,
                built.game,
                red_action,
                blue_action,
                blue_type_name=style,
            )
            patrol = build_observable_threat_patrol(
                adapter,
                plan.red_reserve_ids,
                dict(plan.red_assignment),
                **{
                    key: float(value)
                    for key, value in physical.get("reserve_patrol", {}).items()
                    if key in {"standoff_distance", "lateral_spacing", "distance_scale"}
                },
            )
            elapsed = time.perf_counter() - planning_start
            planning_times.append(elapsed)
            record = identity_v5._plan_record(
                plan,
                built.game,
                adapter.step_count,
                elapsed,
                result,
                patrol,
            )
            record["solver"] = method
            record["saldae_expanded_domain"] = (
                None if diagnostics is None else diagnostics.expanded_action_domain
            )
            record["global_certificate_claimed"] = int(result.full_game_certified)
            records.append(record)
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
                reserve_waypoints=None if patrol is None else patrol.waypoint_mapping(),
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
    target_states = adapter.target_states()
    return {
        "scenario": scenario["label"],
        "method": method,
        "blue_lower_style": style,
        "episode": int(seed),
        "seed": int(seed),
        "population": int(scenario["red"]) + int(scenario["blue"]),
        "targets": int(scenario["targets"]),
        "red_win": int(float(info["outcome_red"]) > 0),
        "breach": int(any(not row["alive"] for row in target_states.values())),
        "steps": int(adapter.step_count),
        "commands": len(records),
        "planning_seconds_mean": statistics.fmean(planning_times) if planning_times else 0.0,
        "planning_seconds_p95": _percentile(planning_times, 95) if planning_times else 0.0,
        "max_nonempty_group_size": max(nonempty_sizes, default=0),
        "identity_partition_valid": 1,
        "red_active_or_reserve_partition_valid": 1,
        "blue_exact_partition_valid": 1,
        "fixed_target_count_valid": int(len(adapter.target_ids) == int(scenario["targets"])),
        # A full-domain MILP certificate after severe casualties can be
        # legitimate.  This diagnostic only guards against falsely promoting
        # an anytime SALDAE response to a global certificate.
        "global_certificate_claimed": int(
            method == "saldae_do"
            and any(record["global_certificate_claimed"] for record in records)
        ),
        "saldae_nodes": saldae_nodes,
        "saldae_expanded_domain_events": saldae_outside_events,
        "joint_plans_json": json.dumps(records, ensure_ascii=False, separators=(",", ":")),
    }


def _paired_difference(rows):
    lookup = {
        (row["scenario"], row["blue_lower_style"], row["seed"], row["method"]): row[
            "red_win"
        ]
        for row in rows
    }
    prefixes = {
        (row["scenario"], row["blue_lower_style"], row["seed"]) for row in rows
    }
    differences = [
        lookup[prefix + ("saldae_do",)] - lookup[prefix + ("milp_do",)]
        for prefix in prefixes
        if prefix + ("saldae_do",) in lookup and prefix + ("milp_do",) in lookup
    ]
    values = np.asarray(differences, dtype=np.float64)
    if not len(values):
        return {"pairs": 0, "mean_improvement": None, "ci95_low": None, "ci95_high": None}
    rng = np.random.default_rng(20260904)
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
    methods = list(map(str, physical["matchups"]))
    episodes = 1 if smoke else int(physical["episodes_per_cell"])
    total = len(scenarios) * len(styles) * len(methods) * episodes
    rows = []
    started = time.perf_counter()
    for scenario_index, scenario in enumerate(scenarios):
        for style_index, style in enumerate(styles):
            for episode in range(episodes):
                seed = (
                    int(config["seed"])
                    + 100000
                    + scenario_index * 10000
                    + style_index * 1000
                    + episode
                )
                for method in methods:
                    rows.append(
                        _run_episode(
                            config,
                            scenario,
                            style,
                            method,
                            seed,
                            predictor,
                            stage1_model,
                            device,
                            smoke,
                        )
                    )
                    if (
                        len(rows) % max(1, int(physical["progress_every_episodes"])) == 0
                        or len(rows) == total
                    ):
                        progress.progress(
                            "Stage3-SALDAE/C HAD配对物理对抗",
                            len(rows),
                            total,
                            started_at=started,
                        )
    write_csv(output / "raw/physical_paired_episodes.csv", rows)
    table = {}
    for method in methods:
        table[method] = {}
        for style in styles:
            table[method][style] = {}
            for scenario in scenarios:
                selected = [
                    row
                    for row in rows
                    if row["method"] == method
                    and row["blue_lower_style"] == style
                    and row["scenario"] == scenario["label"]
                ]
                table[method][style][str(scenario["targets"])] = {
                    "episodes": len(selected),
                    "win_rate": statistics.fmean(row["red_win"] for row in selected),
                    "breach_rate": statistics.fmean(row["breach"] for row in selected),
                    "planning_seconds_mean": statistics.fmean(
                        row["planning_seconds_mean"] for row in selected
                    ),
                }
    return rows, {
        "episodes": len(rows),
        "win_rate_by_method_style_targets": table,
        "saldae_vs_milp_paired": _paired_difference(rows),
        "all_groups_at_most_four": all(
            row["max_nonempty_group_size"] <= 4 for row in rows
        ),
        "all_fixed_target_counts_valid": all(row["fixed_target_count_valid"] for row in rows),
        "all_identity_partitions_valid": all(row["identity_partition_valid"] for row in rows),
        "all_red_partitions_valid": all(
            row["red_active_or_reserve_partition_valid"] for row in rows
        ),
        "all_blue_partitions_valid": all(row["blue_exact_partition_valid"] for row in rows),
        "false_global_certificates": sum(row["global_certificate_claimed"] for row in rows),
    }


def _plots(output: Path, scaling, physical):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    scales = list(scaling)
    x = np.arange(len(scales))
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), constrained_layout=True)
    width = 0.35
    axes[0].bar(
        x - width / 2,
        [scaling[key]["milp_seconds_p50"] for key in scales],
        width,
        label="MILP-DO",
        color="#5B8FF9",
    )
    axes[0].bar(
        x + width / 2,
        [scaling[key]["saldae_total_seconds_p50"] for key in scales],
        width,
        label="SALDAE-DO (warm start)",
        color="#61DDAA",
    )
    axes[0].set(xticks=x, xticklabels=scales, xlabel="Total agents", ylabel="Median planning time (s)", title="Planning cost")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(
        x - width / 2,
        [scaling[key]["red_response_gain_mean"] for key in scales],
        width,
        label="Red response gain",
        color="#F6BD16",
    )
    axes[1].bar(
        x + width / 2,
        [scaling[key]["blue_response_gain_mean"] for key in scales],
        width,
        label="Blue response gain",
        color="#E8684A",
    )
    axes[1].set(xticks=x, xticklabels=scales, xlabel="Total agents", ylabel="Proxy gain", title="Actions found beyond spatial columns")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    path1 = figures / "01_planner_comparison.png"
    fig.savefig(path1, dpi=180)
    plt.close(fig)
    paths = [path1]
    if physical:
        table = physical["win_rate_by_method_style_targets"]
        targets = [2, 4, 5]
        methods = list(table)
        fig, axis = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
        for index, method in enumerate(methods):
            rates = [
                statistics.fmean(
                    table[method][style][str(target)]["win_rate"]
                    for style in table[method]
                )
                for target in targets
            ]
            axis.bar(
                np.arange(len(targets)) + (index - 0.5) * 0.36,
                rates,
                0.36,
                label=method,
            )
        axis.set(xticks=np.arange(len(targets)), xticklabels=targets, ylim=(0, 1), xlabel="Fixed targets", ylabel="Red win rate", title="Paired HAD solver comparison")
        axis.legend()
        axis.grid(axis="y", alpha=0.25)
        path2 = figures / "02_physical_win_rate.png"
        fig.savefig(path2, dpi=180)
        plt.close(fig)
        paths.append(path2)
    return [str(path.relative_to(output)) for path in paths]


def _report(summary):
    exact = summary["exact_validation"]
    lines = [
        "# Stage3 SALDAE-DO 增量实验报告",
        "",
        f"> 自动生成：`{summary['completed_at']}`  ",
        f"> 协议与约束验收：**{'通过' if summary['acceptance']['all_passed'] else '未通过'}**",
        "",
        "本实验不改变身份级 Colonel Blotto 数学模型、Stage2价值函数、1–4人上限或HAD物理协议。变化仅在最佳响应求解器：旧方法在空间候选联盟列上求MILP；新方法以其均衡支持热启动，在按需生成的身份联盟结构图上运行SALDAE，再由外层Double Oracle求零和混合策略。",
        "",
        "实现保留SALDAE的split/merge、多个搜索体、OPEN/RESERVE/SUBSTITUTE、价值引导选择、桥接路径和anytime incumbent；目标/通道、身份交换、Red reserve、带符号零和价值及Double Oracle属于本研究扩展。因此结果应称为`SALDAE-DO扩展`，不是原论文代码的逐行复现，也不提供全局Nash证书。",
        "",
        "## 1. 小规模完整矩阵",
        "",
        "| 场景数 | 平均价值误差 | 最大价值误差 | 错误全局证书 |",
        "|---:|---:|---:|---:|",
        f"| {exact['scenarios']} | {exact['value_error_mean']:.5f} | {exact['value_error_max']:.5f} | {exact['false_global_certificates']} |",
        "",
        "## 2. 30–50人规划",
        "",
        "| 人数 | MILP中位时间(s) | SALDAE总中位时间(s) | Red域外响应增益 | Blue域外响应增益 | 发现域外联盟比例 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for scale, row in summary["planner_scaling"].items():
        lines.append(
            f"| {scale} | {row['milp_seconds_p50']:.3f} | {row['saldae_total_seconds_p50']:.3f} | {row['red_response_gain_mean']:.5f} | {row['blue_response_gain_mean']:.5f} | {row['expanded_domain_fraction']:.1%} |"
        )
    physical = summary.get("physical_evaluation")
    lines.extend(["", "## 3. HAD配对物理对抗", ""])
    if physical is None:
        lines.append("本次跳过物理评估。")
    else:
        lines.extend([
            "| 上层求解器 | Blue底层 | 2目标 | 4目标 | 5目标 |",
            "|---|---|---:|---:|---:|",
        ])
        for method, by_style in physical["win_rate_by_method_style_targets"].items():
            for style, values in by_style.items():
                rates = [values[str(target)]["win_rate"] for target in (2, 4, 5)]
                lines.append(
                    f"| {method} | {style} | {rates[0]:.1%} | {rates[1]:.1%} | {rates[2]:.1%} |"
                )
        paired = physical["saldae_vs_milp_paired"]
        lines.extend([
            "",
            f"同种子配对胜率差 `SALDAE-DO - MILP-DO = {paired['mean_improvement']:.1%}`，95% bootstrap CI `{paired['ci95_low']:.1%}` 至 `{paired['ci95_high']:.1%}`（{paired['pairs']}对）。",
        ])
    lines.extend([
        "",
        "## 4. 结论边界",
        "",
        summary["decision"],
        "",
        "- `raw/small_exact_comparison.csv`：SALDAE与完整矩阵。",
        "- `raw/planner_comparison.csv`：30/40/50人规划时间和域外响应增益。",
        "- `raw/physical_paired_episodes.csv`：逐局原始计划与结果。",
        "- `analysis/summary.json`：机器可读汇总。",
        "- `figures/`：核心对比图。",
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
        device = identity_v5._device(str(args.device or config.get("device", "auto")))
        artifacts = config["artifacts"]
        stage1_path = _resolve(artifacts["stage1_checkpoint"])
        stage2_path = _resolve(artifacts["stage2_checkpoint"])
        dataset_path = _resolve(artifacts["stage2_dataset"])
        for path in (stage1_path, stage2_path, dataset_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        expected_dataset_hash = _expected_hash(artifacts.get("stage2_dataset_sha256"))
        if expected_dataset_hash and sha256_file(dataset_path) != expected_dataset_hash:
            raise ValueError("Stage2 dataset hash differs from the registered SALDAE protocol")
        predictor = FrozenStage2Payoff(
            stage2_path,
            device=device,
            expected_sha256=_expected_hash(artifacts.get("stage2_sha256")),
        )
        if int(predictor.supported_roster.get("max_red", 0)) != 4 or int(
            predictor.supported_roster.get("max_blue", 0)
        ) != 4:
            raise ValueError("SALDAE must use the accepted 1..4v1..4 Stage2 model")
        stage1_model = load_round01_stage1_model(
            stage1_path,
            device,
            expected_sha256=_expected_hash(artifacts.get("stage1_sha256")),
        )

        progress.phase("A：SALDAE对完整小博弈", "B：30–50人同状态规划对比")
        exact = run_exact(config, output, progress, args.smoke)
        progress.phase("B：MILP-DO与SALDAE-DO规划", "C：HAD配对物理对抗")
        _, scaling = run_scaling(config, predictor, output, progress, args.smoke)
        physical = None
        if not args.skip_physical and config["physical_evaluation"]["enabled"]:
            progress.phase("C：HAD同种子配对物理对抗", "D：自动汇总")
            _, physical = run_physical(
                config, predictor, stage1_model, device, output, progress, args.smoke
            )
        else:
            progress.phase("C：已跳过物理评估", "D：自动汇总")

        acceptance_config = config["acceptance"]
        checks = {
            "small_value_quality": exact["value_error_max"]
            <= float(acceptance_config["small_game_value_error_max"]),
            "no_false_global_certificate": exact["false_global_certificates"] == 0,
        }
        for scale, row in scaling.items():
            checks[f"planning_{scale}"] = row["saldae_total_seconds_p95"] <= float(
                acceptance_config["planning_p95_seconds"][scale]
            )
        if physical is not None:
            checks.update(
                {
                    "groups_at_most_four": physical["all_groups_at_most_four"],
                    "fixed_targets": physical["all_fixed_target_counts_valid"],
                    "identity_partition": physical["all_identity_partitions_valid"],
                    "red_partition": physical["all_red_partitions_valid"],
                    "blue_partition": physical["all_blue_partitions_valid"],
                    "physical_no_false_certificate": physical["false_global_certificates"] == 0,
                }
            )
        all_passed = all(checks.values())
        response_gain = any(
            row["red_response_gain_mean"] > 1e-6
            or row["blue_response_gain_mean"] > 1e-6
            for row in scaling.values()
        )
        paired_gain = (
            None
            if physical is None
            else physical["saldae_vs_milp_paired"]["mean_improvement"]
        )
        if not all_passed:
            decision = "实现已跑通，但至少一个正确性、约束或时间门槛未通过，暂不能把它写成有效改进。"
        elif response_gain and paired_gain is not None and paired_gain > 0:
            decision = "SALDAE搜索发现了旧空间候选列之外的有利响应，并在配对HAD胜率上取得正增益；可以作为Stage3求解器优化报告。"
        elif response_gain:
            decision = "SALDAE搜索确实扩展了旧候选域并改善了代理最佳响应，但物理胜率尚未同步提升；可以称组合求解改进，不能宣称端到端性能提升。"
        else:
            decision = "SALDAE实现满足协议，但本轮未发现相对旧MILP候选域的实质收益；应作为负结果或可扩展求解基线，不宣称优化有效。"
        summary = {
            "schema_version": SCHEMA,
            "completed_at": utc_now_iso(),
            "method_change": {
                "model_changed": False,
                "old": "spatial candidate columns + set-partitioning MILP best response + DO",
                "new": "MILP-warm-started constrained SALDAE anytime best response + DO",
                "global_certificate": False,
            },
            "exact_validation": exact,
            "planner_scaling": scaling,
            "physical_evaluation": physical,
            "acceptance": {**checks, "all_passed": all_passed},
            "optimization_evidence": {
                "out_of_domain_response_gain_found": response_gain,
                "paired_physical_win_rate_gain": paired_gain,
            },
            "decision": decision,
        }
        summary["figures"] = _plots(output, scaling, physical)
        atomic_write_json(output / "analysis/summary.json", summary)
        atomic_write_text(output / "stage3_saldae_report.md", _report(summary))
        atomic_write_json(
            status_path,
            {
                "schema_version": SCHEMA,
                "status": "completed",
                "accepted": all_passed,
                "completed_at": summary["completed_at"],
                "report": str(output / "stage3_saldae_report.md"),
            },
        )
        progress.phase("D：报告与新旧方法对比已生成", None)
    except Exception as error:
        atomic_write_json(
            status_path,
            {
                "schema_version": SCHEMA,
                "status": "failed",
                "failed_at": utc_now_iso(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            },
        )
        progress.write(f"[Stage3-SALDAE][失败] {type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
