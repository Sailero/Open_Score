"""Train, calibrate and evaluate the formal Stage-2 competing-risk system."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import torch
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from open_score.stage2 import (  # noqa: E402
    PHYSICAL_TARGET_FIELDS,
    Stage2TrainingConfig,
    collect_had_records,
    evaluate_stage2_predictions,
    fit_stage2_system,
    fit_hist_gradient_boosting_baseline,
    generate_synthetic_records,
    metric_definitions_zh,
    read_records_jsonl,
    split_records_by_lineage_group,
    validate_counterfactual_design,
    validate_formal_dataset_contract,
    write_records_csv,
    write_records_jsonl,
)


_ACCEPTANCE_GATE_IDS = tuple(f"S2-{index:02d}" for index in range(1, 14))
_CRITICAL_PHYSICAL_FIELDS = {
    "payoff_red",
    "target_final_health_fraction",
    "target_min_health_fraction",
    "defender_casualties",
    "attacker_casualties",
}
_CONSTRAINED_PHYSICAL_FIELDS = set(PHYSICAL_TARGET_FIELDS) - {"payoff_red"}
_REQUIRED_SPLIT_NAMES = (
    "train",
    "validation",
    "temperature_calibration",
    "risk_calibration",
    "test",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT / "configs" / "stage2_had_formal.yaml"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in the selected Python environment")
    return torch.device(requested)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> Dict[str, object]:
    git = r"D:\Software\Git\cmd\git.exe"
    try:
        commit = subprocess.run(
            [git, "rev-parse", "HEAD"], cwd=PROJECT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                [git, "status", "--porcelain"], cwd=PROJECT, check=True,
                capture_output=True, text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _formal_collection_manifest_audit(
    data_config: Dict[str, object],
    source_dataset: Path,
    dataset_sha256: str,
    records,
    formal_data_contract: Mapping[str, object],
) -> Dict[str, object]:
    """Require the frozen JSONL to retain its passed collection/attrition audit."""

    manifest_path = _path(
        data_config.get(
            "manifest_path", source_dataset.with_name("dataset_manifest.json")
        )
    ).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"formal Stage-2 collection manifest is missing: {manifest_path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("formal Stage-2 collection manifest is invalid JSON") from error
    if not isinstance(manifest, dict):
        raise ValueError("formal Stage-2 collection manifest must be a JSON object")
    attrition = manifest.get("snapshot_attrition_audit")
    if not isinstance(attrition, dict):
        raise ValueError("formal collection manifest lacks snapshot_attrition_audit")
    integrity_checks = attrition.get("integrity_checks")
    if not isinstance(integrity_checks, dict) or not integrity_checks:
        raise ValueError("formal snapshot attrition manifest lacks integrity checks")
    roots = {record.query.root_id for record in records}
    lineages = {record.query.lineage_group_id for record in records}
    required_snapshot_steps = sorted(
        map(int, formal_data_contract.get("required_snapshot_steps", ()))
    )
    if not required_snapshot_steps:
        raise ValueError(
            "formal_data_contract.required_snapshot_steps must be non-empty"
        )
    try:
        required_planned_roots = int(
            formal_data_contract["required_planned_snapshot_roots"]
        )
        required_snapshot_zero_roots = int(
            formal_data_contract["required_snapshot_zero_roots"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "formal data contract must pin planned and snapshot-zero root counts"
        ) from error
    expected_partition = (
        int(attrition.get("realized_snapshot_roots", -1))
        + int(attrition.get("terminal_attrition_roots", -1))
        + int(attrition.get("unexplained_missing_roots", -1))
    )
    checks = {
        "protocol_v4": manifest.get("protocol") == "OpenSCORE-Stage2-HAD-v4",
        "record_schema_v4": int(manifest.get("record_schema_version", -1)) == 4,
        "dataset_sha256_matches": str(manifest.get("dataset_sha256", "")).lower()
        == dataset_sha256,
        "snapshot_attrition_passed": attrition.get("status") == "passed",
        "requested_snapshot_steps_match": sorted(
            map(int, attrition.get("requested_snapshot_steps", ()))
        )
        == required_snapshot_steps,
        "all_collection_integrity_checks_passed": all(
            value is True for value in integrity_checks.values()
        ),
        "planned_roots_partitioned": int(
            attrition.get("planned_snapshot_roots", -2)
        )
        == expected_partition,
        "planned_root_count_matches_contract": int(
            attrition.get("planned_snapshot_roots", -1)
        )
        == required_planned_roots,
        "no_unexplained_snapshot_loss": int(
            attrition.get("unexplained_missing_roots", -1)
        )
        == 0,
        "record_count_matches": int(manifest.get("records", -1)) == len(records),
        "root_count_matches": int(manifest.get("roots", -1)) == len(roots),
        "lineage_count_matches": int(manifest.get("lineage_groups", -1))
        == len(lineages),
        "attrition_record_count_matches": int(
            attrition.get("observed_records", -1)
        )
        == len(records),
        "attrition_root_count_matches": int(
            attrition.get("realized_snapshot_roots", -1)
        )
        == len(roots),
        "snapshot_zero_expected_matches_contract": int(
            attrition.get("snapshot_zero_expected_roots", -1)
        )
        == required_snapshot_zero_roots,
        "snapshot_zero_realized_matches_contract": int(
            attrition.get("snapshot_zero_realized_roots", -1)
        )
        == required_snapshot_zero_roots,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(
            "formal Stage-2 collection-manifest gate failed: " + ", ".join(failed)
        )
    return {
        "status": "passed",
        "manifest_path": _display_path(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "integrity_checks": checks,
        "snapshot_attrition_audit": attrition,
    }


def _validate_acceptance_config(value: object) -> Dict[str, object]:
    """Fail before Test is opened if the 13 preregistered gates drift."""

    if not isinstance(value, Mapping):
        raise ValueError("formal Stage-2 requires a machine-readable acceptance mapping")
    config = dict(value)
    if config.get("protocol") != "OpenSCORE-Stage2-Acceptance-v1":
        raise ValueError("formal Stage-2 acceptance protocol must be v1")
    if int(config.get("required_gate_count", -1)) != len(_ACCEPTANCE_GATE_IDS):
        raise ValueError("formal Stage-2 acceptance must preregister exactly 13 gates")
    gates = config.get("gates")
    if not isinstance(gates, Mapping) or set(gates) != set(_ACCEPTANCE_GATE_IDS):
        raise ValueError("formal Stage-2 acceptance gate IDs must be exactly S2-01..S2-13")
    gates = {str(key): dict(item) for key, item in gates.items()}

    expected_operators = {
        "S2-01": {"operator": ">"},
        "S2-02": {"brier_operator": "<", "auc_operator": ">="},
        "S2-03": {
            "joint_nll_operator": "<",
            "integrated_breach_brier_operator": "<",
        },
        "S2-04": {"operator": ">"},
        "S2-05": {"operator": ">="},
        "S2-06": {"operator": ">="},
        "S2-07": {
            "mean_upper_bound_operator": "<",
            "mean_interval_width_operator": "<=",
        },
        "S2-08": {
            "random_comparator_operator": "<",
            "fixed_global_best_comparator_operator": "<",
        },
        "S2-09": {"operator": ">"},
        "S2-10": {"skill_operator": ">"},
        "S2-11": {"operator": ">"},
        "S2-12": {"operator": "=="},
        "S2-13": {"operator": "<="},
    }
    for gate_id, expected in expected_operators.items():
        if not str(gates[gate_id].get("name", "")):
            raise ValueError(f"acceptance.{gate_id} requires a stable name")
        for key, expected_value in expected.items():
            if gates[gate_id].get(key) != expected_value:
                raise ValueError(
                    f"acceptance.{gate_id}.{key} must be {expected_value!r}"
                )

    expected_numbers = {
        ("S2-01", "value"): 0.0,
        ("S2-04", "value"): 0.0,
        ("S2-05", "value"): 0.90,
        ("S2-06", "value"): 0.90,
        ("S2-07", "mean_upper_bound_value"): 0.95,
        ("S2-07", "mean_interval_width_value"): 0.30,
        ("S2-09", "value"): 0.50,
        ("S2-10", "required_evaluated_heads"): 13,
        ("S2-10", "minimum_positive_mae_skill_heads"): 9,
        ("S2-10", "skill_value"): 0.0,
        ("S2-11", "value"): 0.0,
        ("S2-12", "value"): 1.0,
        ("S2-13", "value_agents"): 0.10,
    }
    for (gate_id, key), expected in expected_numbers.items():
        try:
            observed = float(gates[gate_id][key])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"acceptance.{gate_id}.{key} must be numeric") from error
        if not np.isfinite(observed) or observed != float(expected):
            raise ValueError(
                f"acceptance.{gate_id}.{key} must equal the protocol value {expected}"
            )

    critical_fields = tuple(map(str, gates["S2-11"].get("fields", ())))
    if (
        len(critical_fields) != len(_CRITICAL_PHYSICAL_FIELDS)
        or set(critical_fields) != _CRITICAL_PHYSICAL_FIELDS
    ):
        raise ValueError("acceptance.S2-11 must list the five critical physical heads")
    constrained_fields = tuple(map(str, gates["S2-12"].get("fields", ())))
    if (
        len(constrained_fields) != len(_CONSTRAINED_PHYSICAL_FIELDS)
        or set(constrained_fields) != _CONSTRAINED_PHYSICAL_FIELDS
    ):
        raise ValueError("acceptance.S2-12 must list all 12 constrained physical heads")
    sides = tuple(map(str, gates["S2-13"].get("sides", ())))
    if len(sides) != 2 or set(sides) != {"defender", "attacker"}:
        raise ValueError("acceptance.S2-13 must check defender and attacker identities")
    config["gates"] = gates
    return config


def _finite_number(value: object) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _metric(mapping: object, *keys: str) -> object:
    value = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _compare(left: object, operator: str, right: object) -> bool:
    first, second = _finite_number(left), _finite_number(right)
    if first is None or second is None:
        return False
    return {
        ">": first > second,
        ">=": first >= second,
        "<": first < second,
        "<=": first <= second,
        "==": first == second,
    }[operator]


def _build_stage2_acceptance(
    metrics: Mapping[str, object],
    tabular_summary: object,
    acceptance_config: Mapping[str, object],
) -> Dict[str, object]:
    """Evaluate all 13 frozen gates once on the untouched Test results."""

    config = _validate_acceptance_config(acceptance_config)
    thresholds = config["gates"]
    tabular = tabular_summary if isinstance(tabular_summary, Mapping) else {}
    gates: Dict[str, object] = {}

    def add(gate_id: str, observed: object, passed: bool) -> None:
        gate_config = thresholds[gate_id]
        gates[gate_id] = {
            "name": gate_config["name"],
            "observed": _jsonable(observed),
            "threshold": _jsonable(
                {key: value for key, value in gate_config.items() if key != "name"}
            ),
            "passed": bool(passed),
        }

    breach_skill = _metric(metrics, "breach", "climatology_brier_skill")
    add("S2-01", breach_skill, _compare(breach_skill, ">", thresholds["S2-01"]["value"]))

    main_brier = _metric(metrics, "breach", "brier")
    main_auc = _metric(metrics, "breach", "roc_auc")
    tabular_brier = _metric(tabular, "test_breach_brier")
    tabular_auc = _metric(tabular, "test_breach_roc_auc")
    add(
        "S2-02",
        {
            "main_brier": main_brier,
            "hist_gradient_boosting_brier": tabular_brier,
            "main_roc_auc": main_auc,
            "hist_gradient_boosting_roc_auc": tabular_auc,
        },
        _compare(main_brier, "<", tabular_brier)
        and _compare(main_auc, ">=", tabular_auc),
    )

    main_joint_nll = _metric(
        metrics, "joint_outcome_time", "right_censored_competing_risk_nll"
    )
    main_integrated_brier = _metric(
        metrics, "joint_outcome_time", "integrated_breach_brier"
    )
    baseline_joint_nll = _metric(
        metrics,
        "empirical_joint_train_frequency_baseline",
        "right_censored_competing_risk_nll",
    )
    baseline_integrated_brier = _metric(
        metrics,
        "empirical_joint_train_frequency_baseline",
        "integrated_breach_brier",
    )
    add(
        "S2-03",
        {
            "main_joint_censored_nll": main_joint_nll,
            "train_empirical_joint_censored_nll": baseline_joint_nll,
            "main_integrated_breach_brier": main_integrated_brier,
            "train_empirical_integrated_breach_brier": baseline_integrated_brier,
        },
        _compare(main_joint_nll, "<", baseline_joint_nll)
        and _compare(main_integrated_brier, "<", baseline_integrated_brier),
    )

    time_improvement = _metric(
        metrics, "remaining_time", "event_time_mae_improvement_over_median"
    )
    add("S2-04", time_improvement, _compare(time_improvement, ">", thresholds["S2-04"]["value"]))

    candidate_coverage = _metric(
        metrics, "calibrated_risk_bound", "candidate_rate_coverage"
    )
    add("S2-05", candidate_coverage, _compare(candidate_coverage, ">=", thresholds["S2-05"]["value"]))

    post_selection_coverage = _metric(
        metrics, "red_candidate_decision", "post_selection_coverage"
    )
    add("S2-06", post_selection_coverage, _compare(post_selection_coverage, ">=", thresholds["S2-06"]["value"]))

    mean_upper = _metric(metrics, "calibrated_risk_bound", "mean_upper_bound")
    mean_width = _metric(metrics, "calibrated_risk_bound", "mean_interval_width")
    add(
        "S2-07",
        {"mean_upper_bound": mean_upper, "mean_interval_width": mean_width},
        _compare(mean_upper, "<", thresholds["S2-07"]["mean_upper_bound_value"])
        and _compare(mean_width, "<=", thresholds["S2-07"]["mean_interval_width_value"]),
    )

    mean_regret = _metric(metrics, "red_candidate_decision", "mean_red_decision_regret")
    random_regret = _metric(
        metrics, "red_candidate_decision", "random_red_candidate_expected_regret"
    )
    global_regret = _metric(
        metrics,
        "red_candidate_decision",
        "validation_global_best_red_candidate_regret",
    )
    add(
        "S2-08",
        {
            "mean_red_decision_regret": mean_regret,
            "random_red_candidate_expected_regret": random_regret,
            "fixed_validation_global_best_regret": global_regret,
        },
        _compare(mean_regret, "<", random_regret)
        and _compare(mean_regret, "<", global_regret),
    )

    ranking_accuracy = _metric(
        metrics, "red_candidate_decision", "red_ranking_pair_accuracy"
    )
    add("S2-09", ranking_accuracy, _compare(ranking_accuracy, ">", thresholds["S2-09"]["value"]))

    physical_targets = _metric(metrics, "physical_evaluators", "targets")
    physical_targets = physical_targets if isinstance(physical_targets, Mapping) else {}
    evaluated = sorted(
        field
        for field, summary in physical_targets.items()
        if isinstance(summary, Mapping) and summary.get("status") == "evaluated"
    )
    skills = {
        field: _metric(physical_targets, field, "mae_skill_over_train_median")
        for field in PHYSICAL_TARGET_FIELDS
    }
    positive_skill = sorted(
        field for field, skill in skills.items() if _compare(skill, ">", 0.0)
    )
    add(
        "S2-10",
        {
            "evaluated_head_count": len(evaluated),
            "evaluated_heads": evaluated,
            "positive_mae_skill_head_count": len(positive_skill),
            "positive_mae_skill_heads": positive_skill,
        },
        set(evaluated) == set(PHYSICAL_TARGET_FIELDS)
        and len(evaluated) == int(thresholds["S2-10"]["required_evaluated_heads"])
        and len(positive_skill)
        >= int(thresholds["S2-10"]["minimum_positive_mae_skill_heads"]),
    )

    critical_fields = list(map(str, thresholds["S2-11"]["fields"]))
    critical_observed = {field: skills.get(field) for field in critical_fields}
    add(
        "S2-11",
        critical_observed,
        all(
            _compare(skill, ">", thresholds["S2-11"]["value"])
            for skill in critical_observed.values()
        ),
    )

    constrained_fields = list(map(str, thresholds["S2-12"]["fields"]))
    inside_rates = {
        field: _metric(
            physical_targets, field, "reasonable_range", "prediction_inside_rate"
        )
        for field in constrained_fields
    }
    add(
        "S2-12",
        inside_rates,
        all(
            _compare(rate, "==", thresholds["S2-12"]["value"])
            for rate in inside_rates.values()
        ),
    )

    redundancy = _metric(metrics, "physical_evaluators", "redundancy_consistency")
    redundancy = redundancy if isinstance(redundancy, Mapping) else {}
    residuals = {
        side: _metric(
            redundancy,
            f"{side}_survivors_plus_casualties",
            "mean_absolute_prediction_residual",
        )
        for side in map(str, thresholds["S2-13"]["sides"])
    }
    add(
        "S2-13",
        residuals,
        all(
            _compare(value, "<=", thresholds["S2-13"]["value_agents"])
            for value in residuals.values()
        ),
    )

    passed_count = sum(bool(gate["passed"]) for gate in gates.values())
    overall_passed = passed_count == len(_ACCEPTANCE_GATE_IDS)
    return {
        "protocol": config["protocol"],
        "gate_count": len(gates),
        "gates": gates,
        "overall": {
            "passed": overall_passed,
            "decision": "go" if overall_passed else "no-go",
            "passed_gates": passed_count,
            "failed_gates": len(gates) - passed_count,
            "required_gates": len(_ACCEPTANCE_GATE_IDS),
        },
        "overall_go_no_go": "go" if overall_passed else "no-go",
    }


def _formal_split_class_support_audit(split) -> Dict[str, object]:
    """Require all three censor/event classes in every frozen lineage split."""

    partitions = tuple(split.partitions)
    names = tuple(name for name, _ in partitions)
    if names != _REQUIRED_SPLIT_NAMES:
        raise ValueError(
            "formal Stage-2 split must expose Train/Validation/Temperature-Cal/"
            "Risk-Cal/Test in the registered order"
        )
    audit: Dict[str, object] = {}
    failed = []
    for name, records in partitions:
        counts = {
            "defender_win": sum(record.outcome == "defender_win" for record in records),
            "breach": sum(record.outcome == "breach" for record in records),
            "right_censored": sum(not record.event_observed for record in records),
        }
        checks = {key: int(value) > 0 for key, value in counts.items()}
        audit[name] = {
            "records": len(records),
            "counts": {key: int(value) for key, value in counts.items()},
            "checks": checks,
            "passed": all(checks.values()),
        }
        if not all(checks.values()):
            failed.append(name)
    if failed:
        raise ValueError(
            "formal Stage-2 split class-support gate failed for: "
            + ", ".join(failed)
        )
    return {"status": "passed", "required_classes": list(counts), "partitions": audit}


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT.resolve()))
    except ValueError:
        return str(path.resolve())


def _normalise_checkpoint_paths(data: Dict[str, object]) -> None:
    for key in ("defender_candidates", "attacker_threats"):
        for spec in data.get(key, []):
            if isinstance(spec, dict) and spec.get("kind") == "checkpoint":
                spec["path"] = str(_path(spec["path"]).resolve())


def _load_records(config: Dict[str, object], device: torch.device):
    data = dict(config["data"])
    source = str(data.pop("source"))
    if source == "jsonl":
        formal_contract = dict(config.get("formal_data_contract", {}))
        required_schema_version = formal_contract.get("required_schema_version")
        records = read_records_jsonl(
            _path(data["path"]),
            required_schema_version=(
                None
                if required_schema_version is None
                else int(required_schema_version)
            ),
        )
        evidence = "external_stage1_rollout_dataset"
    elif source == "had":
        data["scales"] = [tuple(scale) for scale in data["scales"]]
        data["device"] = device
        _normalise_checkpoint_paths(data)
        records = collect_had_records(**data)
        evidence = "HAD_forward_rollouts_with_root_matched_counterfactuals"
    elif source == "synthetic":
        if "scales" in data:
            data["scales"] = [tuple(scale) for scale in data["scales"]]
        records = generate_synthetic_records(**data)
        evidence = "synthetic_pipeline_test_only_not_environment_evidence"
    else:
        raise ValueError("data.source must be jsonl, had or synthetic")
    audit = validate_counterfactual_design(
        records,
        min_continuations_per_cell=int(config.get("minimum_continuations_per_cell", 2)),
        require_common_random_numbers=True,
    )
    if len(audit["red_candidates"]) < 2:
        raise ValueError("Stage-2 decision evaluation requires at least two Red candidates")
    return source, evidence, records, audit


def _physical_bounds(record, name):
    lower = None if name == "payoff_red" else 0.0
    if name in {"target_final_health_fraction", "target_min_health_fraction"}:
        upper = 1.0
    elif name in {"defender_survivors", "defender_casualties"}:
        upper = float(record.query.defender_count)
    elif name in {"attacker_survivors", "attacker_casualties"}:
        upper = float(record.query.attacker_count)
    else:
        upper = None
    return lower, upper


def _prediction_rows(records, prediction):
    rows = []
    for index, record in enumerate(records):
        rows.append(
            {
                "lineage_group_id": record.query.lineage_group_id,
                "root_id": record.query.root_id,
                "rollout_id": record.query.rollout_id,
                "continuation_id": record.query.continuation_id,
                "candidate_id": record.query.candidate_id,
                "threat_id": record.query.threat_id,
                "scale": [record.query.defender_count, record.query.attacker_count],
                "true_outcome": record.outcome,
                "event_observed": record.event_observed,
                "true_terminal_or_censor_steps": record.terminal_steps,
                "physical_outcomes": {
                    name: getattr(record, name)
                    for name in (
                        "payoff_red",
                        "target_final_health_fraction",
                        "target_min_health_fraction",
                        "defender_survivors",
                        "attacker_survivors",
                        "defender_casualties",
                        "attacker_casualties",
                        "breach_steps",
                        "attackers_neutralized_steps",
                        "minimum_threat_distance",
                        "cumulative_target_damage",
                        "cumulative_defender_damage",
                        "cumulative_attacker_damage",
                        "red_action_cost",
                        "blue_action_cost",
                    )
                },
                "physical_predictions": {
                    name: {
                        "point": float(prediction.physical_point[name][index]),
                        "ensemble_p10": float(
                            prediction.physical_interval_lower[name][index]
                        ),
                        "ensemble_p90": float(
                            prediction.physical_interval_upper[name][index]
                        ),
                        "ensemble_std": float(prediction.physical_std[name][index]),
                        "reasonable_lower": _physical_bounds(record, name)[0],
                        "reasonable_upper": _physical_bounds(record, name)[1],
                        "interval_semantics": "bootstrap_member_quantiles_not_formal_coverage",
                    }
                    for name in prediction.physical_point
                },
                "predicted_breach_probability": float(prediction.breach_probability[index]),
                "predicted_short_window_breach_probability": float(
                    prediction.short_window_breach_probability[index]
                ),
                "predicted_defender_success_probability": float(
                    prediction.defender_success_probability[index]
                ),
                "predicted_restricted_mean_steps": float(
                    prediction.expected_remaining_steps[index]
                ),
                "predicted_survival_through_horizon": float(
                    prediction.survival_through_horizon_probability[index]
                ),
                "breach_upper_marginal": float(prediction.breach_upper_marginal[index]),
                "breach_upper_selection_safe": float(
                    prediction.breach_upper_selection_safe[index]
                ),
            }
        )
    return rows


def _run_summary(payload: Dict[str, object]) -> str:
    metrics = payload["test_metrics"]
    breach = metrics["breach"]
    timing = metrics["remaining_time"]
    decision = metrics.get("red_candidate_decision", {})
    physical_targets = metrics.get("physical_evaluators", {}).get("targets", {})
    evaluated_physical = [
        value for value in physical_targets.values() if value.get("status") == "evaluated"
    ]
    improved_physical = sum(
        value.get("mae_skill_over_train_median") is not None
        and value["mae_skill_over_train_median"] > 0.0
        for value in evaluated_physical
    )
    return f"""# Stage 2 正式协议单次运行摘要

> 证据级别：`{payload['evidence_level']}`。最终科研结论必须结合多 seed 正式数据，而不是只看单次运行。

- 五路隔离：Train / Validation / Temperature-Cal / Risk-Cal / Test，lineage 泄漏检查通过。
- 决策变量：仅 Red 候选策略；Blue 策略是威胁条件，不能被推荐器选择。
- 失守预测：Brier={breach['brier']:.4f}，AUC={breach['roc_auc']}，ECE={breach['ece']:.4f}。
- 终局时间：事件样本 MAE={timing.get('event_time_mae_steps')} 步；中位数基线 MAE={timing.get('median_baseline_event_time_mae_steps')} 步。
- Red 决策：平均 regret={decision.get('mean_red_decision_regret')}，minimax regret={decision.get('minimax_red_decision_regret')}。
- 物理评估器：{len(evaluated_physical)} 个监督头完成逐指标评估，其中 {improved_physical} 个的 MAE 优于 Train 中位数常数基线；成员区间仅表示模型不确定性。
- 终止语义：30 步 continuation 预算先耗尽时右删失；HAD 原生 `max_steps` 到达时是已观察的 defender-win。

完整定义、分规模/Red候选/Blue威胁结果、cluster bootstrap 区间和运行元数据见 `metrics.json`。
"""


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Stage-2 YAML must contain a mapping")
    formal_mode = bool(config.get("formal_mode", False))
    acceptance_config = None
    if formal_mode:
        acceptance_config = _validate_acceptance_config(config.get("acceptance"))
    elif config.get("acceptance") is not None:
        acceptance_config = _validate_acceptance_config(config["acceptance"])
    git_revision = _git_revision()
    if formal_mode and git_revision.get("dirty") is not False:
        raise RuntimeError("formal Stage-2 training requires a clean committed Git worktree")
    data_config = dict(config["data"])
    if formal_mode and data_config.get("source") != "jsonl":
        raise ValueError(
            "formal Stage-2 training accepts only a separately collected frozen JSONL dataset"
        )
    source_dataset = None
    actual_dataset_sha256 = ""
    expected_dataset_sha256 = ""
    if data_config.get("source") == "jsonl":
        source_dataset = _path(data_config["path"]).resolve()
        expected_dataset_sha256 = str(data_config.get("expected_sha256", "")).lower()
        if formal_mode and len(expected_dataset_sha256) != 64:
            raise ValueError("formal frozen dataset requires a pinned expected_sha256")
        actual_dataset_sha256 = _sha256(source_dataset)
        if (
            expected_dataset_sha256
            and actual_dataset_sha256 != expected_dataset_sha256
        ):
            raise ValueError("frozen Stage-2 dataset SHA-256 differs from config")
    device = _device(args.device or str(config.get("device", "auto")))
    output_dir = args.output_dir or _path(config.get("output_dir", "outputs/stage2_formal"))
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    test_lock = output_dir / "LOCKED_TEST_EVALUATION.json"
    if formal_mode and test_lock.exists():
        raise RuntimeError(
            "locked test was already opened for this output directory; use a new run directory"
        )
    seed = int(config.get("seed", 7))
    np.random.seed(seed)
    torch.manual_seed(seed)
    started = time.perf_counter()

    source, evidence, records, design_audit = _load_records(config, device)
    formal_data_contract_audit = None
    formal_collection_manifest_audit = None
    if formal_mode:
        if not isinstance(config.get("formal_data_contract"), dict):
            raise ValueError(
                "formal Stage-2 training requires formal_data_contract"
            )
        formal_data_contract_audit = validate_formal_dataset_contract(
            records, config["formal_data_contract"]
        )
        formal_collection_manifest_audit = _formal_collection_manifest_audit(
            data_config,
            source_dataset,
            actual_dataset_sha256,
            records,
            config["formal_data_contract"],
        )
    output_dataset = output_dir / "dataset.jsonl"
    output_csv = output_dir / "dataset.csv"
    if formal_mode and source_dataset is not None:
        # The formal JSONL is already frozen and hash-pinned. Re-encoding it
        # would duplicate hundreds of MB and CSV conversion materialises every
        # row as another Python dict. Keep one immutable source of truth.
        working_dataset = source_dataset
        output_dataset = None
        output_csv = None
    else:
        write_records_jsonl(records, output_dataset)
        write_records_csv(records, output_csv)
        working_dataset = output_dataset
    published_dataset = None
    if config.get("published_dataset_path"):
        published_dataset = _path(config["published_dataset_path"])
        write_records_jsonl(records, published_dataset)
    split = split_records_by_lineage_group(
        records, seed=seed, **dict(config.get("split", {}))
    )
    formal_split_class_support_audit = None
    if formal_mode:
        # This is a preregistered data-design check only; no performance
        # statistic or threshold is fitted from Test labels here.
        formal_split_class_support_audit = _formal_split_class_support_audit(split)
    training_values = dict(config.get("training", {}))
    training_values.setdefault("seed", seed)
    training_config = Stage2TrainingConfig.from_dict(training_values)
    fit = fit_stage2_system(
        split.train,
        split.validation,
        split.temperature_calibration,
        split.risk_calibration,
        training_config,
        device=device,
    )
    if formal_mode:
        test_lock.write_text(
            json.dumps(
                {
                    "opened_utc": datetime.now(timezone.utc).isoformat(),
                    "dataset_sha256": _sha256(working_dataset),
                    "config_sha256": _sha256(config_path),
                    "git": git_revision,
                    "split_class_support_audit": formal_split_class_support_audit,
                    "semantics": "one_shot_locked_test_opened_after_fit_and_calibration",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    prediction = fit.system.predict(split.test, device=device)
    observed_train_steps = [
        record.terminal_steps for record in split.train if record.event_observed
    ]
    if not observed_train_steps:
        raise ValueError("training split has no observed terminal event for time baseline")
    evaluation = dict(config.get("evaluation", {}))
    empirical_counts = np.ones(2 * training_config.horizon_bins + 1, np.float64)
    for record in split.train:
        empirical_counts[
            record.terminal_class(
                training_config.horizon_bins, training_config.steps_per_bin
            )
        ] += 1.0
    validation_candidate_rates = {}
    for candidate in sorted(
        {record.query.candidate_id for record in split.validation}
    ):
        rows = [
            record
            for record in split.validation
            if record.query.candidate_id == candidate
        ]
        validation_candidate_rates[candidate] = float(
            np.mean([record.outcome == "breach" for record in rows])
        )
    train_physical_medians = {}
    if fit.system.physical_target_scaler is not None:
        for field in fit.system.physical_target_scaler.available_fields:
            values = [
                float(getattr(record, field))
                for record in split.train
                if getattr(record, field) is not None
            ]
            if values:
                train_physical_medians[field] = float(np.median(values))
    metrics = evaluate_stage2_predictions(
        split.test,
        prediction.probabilities,
        horizon_bins=training_config.horizon_bins,
        steps_per_bin=training_config.steps_per_bin,
        breach_upper=prediction.breach_upper_selection_safe,
        member_probabilities=prediction.member_probabilities,
        train_median_event_steps=float(np.median(observed_train_steps)),
        train_breach_rate=float(
            np.mean([record.outcome == "breach" for record in split.train])
        ),
        train_empirical_probabilities=empirical_counts / empirical_counts.sum(),
        train_candidate_breach_rates=validation_candidate_rates,
        physical_predictions=prediction.physical_point,
        physical_interval_lower=prediction.physical_interval_lower,
        physical_interval_upper=prediction.physical_interval_upper,
        physical_ensemble_std=prediction.physical_std,
        train_physical_medians=train_physical_medians,
        seen_policy_pairs={record.query.policy_pair for record in split.train},
        bootstrap_samples=int(evaluation.get("bootstrap_samples", 500)),
        bootstrap_seed=int(evaluation.get("bootstrap_seed", seed + 9001)),
    )
    tabular_config = dict(config.get("tabular_baseline", {}))
    tabular_summary = None
    if bool(tabular_config.pop("enabled", False)):
        tabular_config.setdefault("seed", seed)
        tabular_summary = fit_hist_gradient_boosting_baseline(
            split.train, split.validation, split.test, tabular_config
        ).summary
    acceptance = None
    acceptance_path = None
    if acceptance_config is not None:
        acceptance = _build_stage2_acceptance(
            metrics, tabular_summary, acceptance_config
        )
        acceptance.update(
            {
                "status": "completed",
                "formal_protocol_version": "OpenSCORE-Stage2-HAD-v4",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "provenance": {
                    "dataset_sha256": _sha256(working_dataset),
                    "config_sha256": _sha256(config_path),
                    "git": git_revision,
                    "locked_test_manifest": (
                        _display_path(test_lock) if formal_mode else None
                    ),
                },
            }
        )
        acceptance_path = output_dir / "acceptance.json"
        acceptance_path.write_text(
            json.dumps(acceptance, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
    checkpoint = output_dir / "stage2_system.pt"
    fit.system.metadata.update(
        {
            "data_source": source,
            "evidence_level": evidence,
            "split_manifest": split.manifest(),
            "counterfactual_design_audit": design_audit,
            "formal_data_contract_audit": formal_data_contract_audit,
            "formal_collection_manifest_audit": formal_collection_manifest_audit,
            "formal_split_class_support_audit": formal_split_class_support_audit,
            "acceptance": acceptance,
            "dataset_sha256": _sha256(working_dataset),
            "config_sha256": _sha256(config_path),
            "git": git_revision,
            "formal_mode": formal_mode,
        }
    )
    fit.system.save(checkpoint)
    prediction_rows = _prediction_rows(split.test, prediction)
    with (output_dir / "test_predictions.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    payload = _jsonable(
        {
            "status": "completed",
            "formal_protocol_version": "OpenSCORE-Stage2-HAD-v4",
            "evidence_level": evidence,
            "data_source": source,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "config_path": _display_path(config_path),
            "seed": seed,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
            },
            "provenance": {
                "dataset_sha256": _sha256(working_dataset),
                "source_dataset": (
                    _display_path(source_dataset) if source_dataset is not None else None
                ),
                "expected_dataset_sha256": expected_dataset_sha256 or None,
                "config_sha256": _sha256(config_path),
                "git": git_revision,
                "locked_test_manifest": (
                    _display_path(test_lock) if formal_mode else None
                ),
            },
            "dataset": {
                "records": len(records),
                "roots": len({record.query.root_id for record in records}),
                "lineage_groups": len({record.query.lineage_group_id for record in records}),
                "red_candidates": sorted({record.query.candidate_id for record in records}),
                "blue_threats": sorted({record.query.threat_id for record in records}),
                "stage1_checkpoint_sha256": sorted(
                    {
                        record.query.stage1_checkpoint_sha256
                        for record in records
                        if record.query.stage1_checkpoint_sha256
                    }
                ),
                "strict_red_superiority": all(
                    record.query.defender_count > record.query.attacker_count
                    for record in records
                ),
                "right_censored_records": sum(not record.event_observed for record in records),
            },
            "counterfactual_design_audit": design_audit,
            "formal_data_contract_audit": formal_data_contract_audit,
            "formal_collection_manifest_audit": formal_collection_manifest_audit,
            "formal_split_class_support_audit": formal_split_class_support_audit,
            "acceptance": acceptance,
            "split_manifest": split.manifest(),
            "training_config": training_values,
            "training_summary": fit.training_summary,
            "metric_definitions_zh": metric_definitions_zh(),
            "test_metrics": metrics,
            "strong_tabular_baseline": tabular_summary,
            "artifacts": {
                "checkpoint": _display_path(checkpoint),
                "dataset_jsonl": _display_path(working_dataset),
                "dataset_csv": (
                    _display_path(output_csv) if output_csv is not None else None
                ),
                "predictions_jsonl": _display_path(output_dir / "test_predictions.jsonl"),
                "acceptance_json": (
                    _display_path(acceptance_path)
                    if acceptance_path is not None
                    else None
                ),
                "published_dataset_jsonl": (
                    _display_path(published_dataset) if published_dataset is not None else None
                ),
            },
        }
    )
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (output_dir / "run_summary.md").write_text(_run_summary(payload), encoding="utf-8")
    if config.get("published_metrics_path"):
        published = _path(config["published_metrics_path"])
        published.parent.mkdir(parents=True, exist_ok=True)
        published.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "status": "completed",
                "records": len(records),
                "five_way_lineage_split": "passed",
                "counterfactual_design": design_audit["status"],
                "scientific_acceptance": (
                    acceptance["overall_go_no_go"]
                    if acceptance is not None
                    else None
                ),
                "metrics": _display_path(metrics_path),
                "elapsed_seconds": payload["elapsed_seconds"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
