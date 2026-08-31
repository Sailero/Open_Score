"""Small, read-only behavioural audit for the SMAClite-AD protocol.

The audit establishes task separation (random, naive rush, and a two-phase
solvability rule), validates randomized spawn geometry, and checks every
available target action against the explicit entity mapping and decoded
simulator target.  It never trains a model.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from open_score.envs.smaclite_ad import stock_scenario_fingerprint
from open_score.stage1.smaclite_ad_training import (
    SMACliteADEpisodeRunner,
    SMACliteADFactory,
    SMACliteADRuleController,
    evaluate_smaclite_ad,
)


Ratio = Tuple[int, int]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
STOCK_MANIFEST = (
    PROJECT_ROOT
    / "configs"
    / "stock_reproduction"
    / "smaclite_aamas2023_epymarl_v3.json"
)
SMACLITE_CHECKOUT = PROJECT_ROOT / "upstream" / "external" / "smaclite-v2.0.0"

PYTEST_CONTRACT = {
    "executed_by_this_audit": False,
    "execution_claim": "not_executed_by_audit; must already pass in the P0 pytest gate",
    "required_nodes": [
        "tests/test_smaclite_ad.py::test_ad_dynamic_rosters_share_shapes_and_masks",
        "tests/test_smaclite_ad.py::test_asset_objective_and_stock_fingerprint_are_isolated",
        "tests/test_smaclite_ad.py::test_strict_potential_discounted_return_telescopes_to_terminal_payoff",
        "tests/test_entity_action_equivariance.py",
    ],
    "covers": [
        "dynamic shape and mask contract",
        "asset terminal branch and stock-map isolation",
        "strict discounted potential-shaping identity",
        "entity/target action permutation equivariance",
    ],
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_executable() -> Path | None:
    candidates = (
        os.environ.get("OPEN_SCORE_GIT"),
        shutil.which("git"),
        r"D:\Software\Git\cmd\git.exe",
        r"C:\Program Files\Git\cmd\git.exe",
    )
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate).resolve()
    return None


def _git_checkout_facts(path: Path) -> Dict[str, object]:
    executable = _git_executable()
    result: Dict[str, object] = {
        "path": str(path.resolve()),
        "git_executable": None if executable is None else str(executable),
        "commit": None,
        "tracked_clean": False,
        "status_porcelain": None,
    }
    if executable is None or not (path / ".git").is_dir():
        result["error"] = "git_or_checkout_unavailable"
        return result
    head = subprocess.run(
        [str(executable), "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    status = subprocess.run(
        [
            str(executable),
            "-C",
            str(path),
            "status",
            "--porcelain=v1",
            "--untracked-files=no",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if head.returncode or status.returncode:
        result["error"] = "git_checkout_probe_failed"
        return result
    rows = status.stdout.splitlines()
    result.update(
        {
            "commit": head.stdout.strip(),
            "tracked_clean": not rows,
            "status_porcelain": rows,
        }
    )
    return result


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _serialise_arguments(args: argparse.Namespace) -> Dict[str, object]:
    values: Dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            values[key] = str(value)
        elif key == "ratios":
            values[key] = [list(ratio) for ratio in value]
        else:
            values[key] = value
    return values


def collect_source_integrity(
    manifest: Mapping[str, object], *, use_cpp_rvo2: bool
) -> Tuple[Dict[str, object], Dict[str, bool]]:
    """Collect immutable stock/backend facts without resetting an environment."""

    pins = manifest["source_pins"]
    smaclite_pin = pins["smaclite"]
    rvo2_pin = pins["smaclite_python_rvo2"]
    checkout = _git_checkout_facts(SMACLITE_CHECKOUT)
    stock = stock_scenario_fingerprint()
    module_path = Path(str(stock["module_path"]))

    rvo2_facts: Dict[str, object] = {
        "requested_backend": "cpp_rvo2" if use_cpp_rvo2 else "numpy_fallback",
        "module_path": None,
        "binary_filename": None,
        "binary_sha256": None,
        "import_error": None,
    }
    if use_cpp_rvo2:
        try:
            rvo2 = importlib.import_module("rvo2")
            binary = Path(rvo2.__file__).resolve()
            rvo2_facts.update(
                {
                    "module_path": str(binary),
                    "binary_filename": binary.name,
                    "binary_sha256": _sha256_file(binary),
                }
            )
        except Exception as exc:  # recorded as a failed hard gate below
            rvo2_facts["import_error"] = f"{type(exc).__name__}: {exc}"

    evidence = {
        "manifest_path": str(STOCK_MANIFEST.resolve()),
        "manifest_sha256": _sha256_file(STOCK_MANIFEST),
        "smaclite_checkout": checkout,
        "runtime_stock_maps": stock,
        "runtime_smaclite_from_expected_checkout": _path_is_within(
            module_path, SMACLITE_CHECKOUT
        ),
        "rvo2_backend": rvo2_facts,
    }
    gates = {
        "smaclite_checkout_commit_exact": (
            checkout.get("commit") == smaclite_pin["commit"]
        ),
        "smaclite_checkout_tracked_clean": checkout.get("tracked_clean") is True,
        "runtime_smaclite_from_expected_checkout": evidence[
            "runtime_smaclite_from_expected_checkout"
        ]
        is True,
        "stock_map_file_count_exact": (
            int(stock["file_count"]) == int(smaclite_pin["stock_scenario_file_count"])
        ),
        "stock_map_combined_sha256_exact": (
            stock["combined_sha256"]
            == smaclite_pin["stock_scenario_combined_sha256"]
        ),
        "cpp_rvo2_requested": use_cpp_rvo2,
        "rvo2_binary_filename_exact": (
            rvo2_facts["binary_filename"] == rvo2_pin["binary_filename"]
        ),
        "rvo2_binary_sha256_exact": (
            rvo2_facts["binary_sha256"] == rvo2_pin["binary_sha256"]
        ),
    }
    return evidence, gates


def _source_identity(value: Mapping[str, object]) -> Dict[str, object]:
    checkout = value["smaclite_checkout"]
    maps = value["runtime_stock_maps"]
    backend = value["rvo2_backend"]
    return {
        "checkout_commit": checkout.get("commit"),
        "checkout_tracked_clean": checkout.get("tracked_clean"),
        "stock_map_file_count": maps.get("file_count"),
        "stock_map_combined_sha256": maps.get("combined_sha256"),
        "runtime_smaclite_module": maps.get("module_path"),
        "rvo2_binary_path": backend.get("module_path"),
        "rvo2_binary_sha256": backend.get("binary_sha256"),
    }


class UniformLegalRandomController:
    name = "random:uniform_legal"
    information_scope = "local_observation_action_mask"

    def reset(self) -> None:
        return None

    def act(self, env, side, observation, rng) -> np.ndarray:
        actions = []
        for row in observation["avail_actions"][: env.team_sizes[side]]:
            actions.append(int(rng.choice(np.flatnonzero(row))))
        return np.asarray(actions, dtype=np.int64)


def parse_ratios(value: str) -> Sequence[Ratio]:
    ratios = []
    for token in value.split(","):
        red, blue = (int(part) for part in token.strip().split(":"))
        ratios.append((red, blue))
    if not ratios or any(red < 1 or blue < 1 for red, blue in ratios):
        raise argparse.ArgumentTypeError("ratios must be positive, e.g. 2:1,3:2")
    return tuple(ratios)


def compact_evaluation(value) -> Dict[str, object]:
    return {
        "episodes": value.episodes,
        "win_rate": value.win_rate,
        "mean_return": value.mean_return,
        "mean_episode_length": value.mean_episode_length,
        "mean_asset_health": value.mean_asset_health,
        "per_ratio": {
            label: {
                "win_rate": cell["win_rate"],
                "mean_return": cell["mean_return"],
                "mean_episode_length": cell["mean_episode_length"],
                "mean_asset_health": cell["mean_asset_health"],
            }
            for label, cell in value.per_ratio.items()
        },
    }


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    manifest = json.loads(STOCK_MANIFEST.read_text(encoding="utf-8"))
    source_before, source_before_gates = collect_source_integrity(
        manifest, use_cpp_rvo2=args.use_cpp_rvo2
    )
    common = {
        "schema_version": "smaclite-ad-protocol-audit-v2",
        "training_performed": False,
        "arguments": _serialise_arguments(args),
        "backend": {
            "requested": "cpp_rvo2" if args.use_cpp_rvo2 else "numpy_fallback",
            "use_cpp_rvo2": bool(args.use_cpp_rvo2),
        },
        "pytest_contract": PYTEST_CONTRACT,
        "roles": {"Red": "asset attacker", "Blue": "asset defender"},
        "terminal_priority": (
            "asset destruction is checked before simultaneous Red elimination"
        ),
        "rule_information_scope": "privileged_full_environment_state",
        "reward_mode": args.reward_mode,
        "discount_gamma": args.gamma,
        "ratios": [list(ratio) for ratio in args.ratios],
    }
    if not all(source_before_gates.values()):
        return {
            **common,
            "aborted_before_behavior_audit": True,
            "source_integrity": {
                "before": source_before,
                "after": None,
                "unchanged_before_after": None,
            },
            "evaluations": {},
            "geometry": None,
            "action_mapping": None,
            "acceptance_gates": source_before_gates,
            "passed": False,
        }

    factory = SMACliteADFactory(
        max_red_agents=args.max_red_agents,
        max_blue_agents=args.max_blue_agents,
        episode_limit=args.episode_limit,
        reward_mode=args.reward_mode,
        discount_gamma=args.gamma,
        shaping_scale=args.shaping_scale,
        approach_weight=args.approach_weight,
        spawn_jitter=args.spawn_jitter,
        use_cpp_rvo2=args.use_cpp_rvo2,
    )
    runner = SMACliteADEpisodeRunner(factory)
    random_policy = UniformLegalRandomController()
    conditions = {
        "rush_vs_idle": (
            SMACliteADRuleController("rush_asset"),
            SMACliteADRuleController("idle"),
        ),
        "rush_vs_intercept": (
            SMACliteADRuleController("rush_asset"),
            SMACliteADRuleController("intercept"),
        ),
        "clear_then_asset_vs_intercept": (
            SMACliteADRuleController("clear_then_asset"),
            SMACliteADRuleController("intercept"),
        ),
        "random_vs_idle": (random_policy, SMACliteADRuleController("idle")),
        "random_vs_intercept": (
            random_policy,
            SMACliteADRuleController("intercept"),
        ),
    }
    evaluations = {
        name: compact_evaluation(
            evaluate_smaclite_ad(
                runner,
                red,
                blue,
                "Red",
                args.ratios,
                args.episodes_per_ratio,
                args.seed,
            )
        )
        for name, (red, blue) in conditions.items()
    }

    layout_rows = []
    layout_hashes: Dict[str, list[str]] = {
        f"{red}:{blue}": [] for red, blue in args.ratios
    }
    for ratio_index, ratio in enumerate(args.ratios):
        env = factory.get(ratio)
        label = f"{ratio[0]}:{ratio[1]}"
        for index in range(args.layout_seeds):
            seed = args.seed + 1_000_000 + ratio_index * 10_000 + index
            _, info = env.reset(seed=seed)
            validation = env.validate_layout()
            layout_hashes[label].append(str(info["layout_hash"]))
            layout_rows.append(
                {
                    "ratio": label,
                    "seed": seed,
                    "valid": bool(validation["valid"]),
                    "minimum_pairwise_clearance": float(
                        validation["minimum_pairwise_clearance"]
                    ),
                    "red_asset_min_distance": min(
                        float(np.linalg.norm(unit.pos - env._asset.pos))
                        for unit in env._red_slots
                    ),
                    "blue_asset_min_distance": min(
                        float(np.linalg.norm(unit.pos - env._asset.pos))
                        for unit in env._blue_slots
                    ),
                    "red_blue_min_distance": min(
                        float(np.linalg.norm(red.pos - blue.pos))
                        for red in env._red_slots
                        for blue in env._blue_slots
                    ),
                }
            )

    mapping_checks = 0
    mapping_errors = []
    for ratio_index, ratio in enumerate(args.ratios):
        env = factory.get(ratio)
        red = SMACliteADRuleController("clear_then_asset")
        blue = SMACliteADRuleController("intercept")
        for episode_index in range(args.mapping_episodes_per_ratio):
            seed = args.seed + 2_000_000 + ratio_index * 10_000 + episode_index
            observations, _ = env.reset(seed=seed)
            red.reset()
            blue.reset()
            rng = np.random.default_rng(seed + 1)
            done = False
            while not done:
                entity_slots = env._all_entity_slots()
                for side in ("Red", "Blue"):
                    controlled = env._red_slots if side == "Red" else env._blue_slots
                    for agent_index, unit in enumerate(controlled):
                        for action in np.flatnonzero(
                            observations[side]["avail_actions"][agent_index]
                        ):
                            action = int(action)
                            if action < 6:
                                continue
                            mapping_checks += 1
                            entity_index = int(
                                observations[side]["action_entity_index"][
                                    agent_index, action
                                ]
                            )
                            visible = bool(
                                entity_index >= 0
                                and observations[side]["entity_mask"][
                                    agent_index, entity_index
                                ]
                            )
                            decoded = env._decode_command(unit, side, action).target
                            if (
                                entity_index < 0
                                or not visible
                                or decoded is not entity_slots[entity_index]
                            ):
                                mapping_errors.append(
                                    {
                                        "ratio": f"{ratio[0]}:{ratio[1]}",
                                        "seed": seed,
                                        "side": side,
                                        "agent": agent_index,
                                        "action": action,
                                        "entity_index": entity_index,
                                    }
                                )
                red_actions = red.act(env, "Red", observations["Red"], rng)
                blue_actions = blue.act(env, "Blue", observations["Blue"], rng)
                observations, _, terminated, truncated, _ = env.step(
                    {"Red": red_actions, "Blue": blue_actions}
                )
                done = bool(terminated or truncated)

    layout_valid = all(row["valid"] for row in layout_rows)
    unique_layouts = {
        label: len(set(hashes)) for label, hashes in layout_hashes.items()
    }
    randomized_layouts_unique = bool(
        args.spawn_jitter <= 0.0
        or all(count == args.layout_seeds for count in unique_layouts.values())
    )
    behavior_gates = {
        "rush_solves_idle": evaluations["rush_vs_idle"]["win_rate"] >= 0.90,
        "naive_rush_is_separated_by_intercept": (
            evaluations["rush_vs_intercept"]["win_rate"] <= 0.10
        ),
        "two_phase_rule_solves_intercept": all(
            cell["win_rate"] >= 0.90
            for cell in evaluations["clear_then_asset_vs_intercept"][
                "per_ratio"
            ].values()
        ),
        "random_does_not_solve_intercept": (
            evaluations["random_vs_intercept"]["win_rate"] <= 0.10
        ),
        "all_layouts_legal": layout_valid,
        "randomized_layouts_unique": randomized_layouts_unique,
        "available_target_mapping_exact": not mapping_errors and mapping_checks > 0,
    }
    factory.close()
    source_after, source_after_gates = collect_source_integrity(
        manifest, use_cpp_rvo2=args.use_cpp_rvo2
    )
    source_unchanged = _source_identity(source_before) == _source_identity(source_after)
    source_gates = {
        **{
            f"source_before_{name}": value
            for name, value in source_before_gates.items()
        },
        **{
            f"source_after_{name}": value
            for name, value in source_after_gates.items()
        },
        "stock_checkout_maps_and_rvo2_unchanged_before_after": source_unchanged,
    }
    gates = {**behavior_gates, **source_gates}
    payload = {
        **common,
        "aborted_before_behavior_audit": False,
        "source_integrity": {
            "before": source_before,
            "after": source_after,
            "unchanged_before_after": source_unchanged,
        },
        "evaluations": evaluations,
        "geometry": {
            "layout_count": len(layout_rows),
            "unique_layouts_per_ratio": unique_layouts,
            "minimum_pairwise_clearance": min(
                row["minimum_pairwise_clearance"] for row in layout_rows
            ),
            "rows": layout_rows,
        },
        "action_mapping": {
            "available_target_actions_checked": mapping_checks,
            "error_count": len(mapping_errors),
            "errors": mapping_errors,
        },
        "acceptance_gates": gates,
        "passed": all(gates.values()),
    }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratios", type=parse_ratios, default=parse_ratios("2:1,3:2,5:3"))
    parser.add_argument("--episodes-per-ratio", type=int, default=3)
    parser.add_argument("--mapping-episodes-per-ratio", type=int, default=1)
    parser.add_argument("--layout-seeds", type=int, default=10)
    parser.add_argument("--episode-limit", type=int, default=150)
    parser.add_argument("--max-red-agents", type=int, default=6)
    parser.add_argument("--max-blue-agents", type=int, default=5)
    parser.add_argument("--spawn-jitter", type=float, default=2.0)
    parser.add_argument(
        "--reward-mode",
        choices=("terminal_only", "strict_potential", "heuristic_delta"),
        default="terminal_only",
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--shaping-scale", type=float, default=0.5)
    parser.add_argument("--approach-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--use-cpp-rvo2", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(
        args.episodes_per_ratio,
        args.mapping_episodes_per_ratio,
        args.layout_seeds,
        args.episode_limit,
    ) < 1:
        parser.error("episode and layout counts must be positive")
    return args


def main() -> None:
    args = parse_args()
    payload = run_audit(args)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    if not payload["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
