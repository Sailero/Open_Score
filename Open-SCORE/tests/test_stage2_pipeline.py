import json
import hashlib
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from open_score.stage2 import (
    BootstrapOutcomeEnsemble,
    CalibratedRiskBound,
    DynamicHADWinNet,
    competing_risk_nll,
    collect_had_records,
    iter_had_records,
    HADCanonicalizer,
    HADDeepSetOutcomeNet,
    PHYSICAL_TARGET_FIELDS,
    Stage2FeatureEncoder,
    Stage2System,
    Stage2TrainingConfig,
    TemperatureScaler,
    evaluate_stage2_predictions,
    fit_stage2_system,
    generate_synthetic_records,
    read_records_jsonl,
    record_from_had_episode,
    split_records_by_lineage_group,
    validate_rollout_lineage,
    validate_counterfactual_design,
    validate_formal_dataset_contract,
    write_records_csv,
    write_records_jsonl,
)


def test_round01_dynamic_win_model_is_permutation_invariant_and_reloadable(tmp_path):
    torch.manual_seed(404)
    model = DynamicHADWinNet(entity_hidden_dim=16, hidden_dim=24)
    state = torch.randn(5, 85)
    # Presence is the final value of each 9-value unit slot.
    for offset in (8, 17, 26, 35, 44, 53, 62, 71):
        state[:, offset + 8] = 1.0
    permuted = state.clone()
    first = state[:, 8:17].clone()
    second = state[:, 17:26].clone()
    permuted[:, 8:17] = second
    permuted[:, 17:26] = first
    assert torch.allclose(model(state), model(permuted), atol=1e-6, rtol=1e-6)
    loss = model(state).square().mean()
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    path = tmp_path / "dynamic-win.pt"
    torch.save(model.state_dict(), path)
    restored = DynamicHADWinNet(entity_hidden_dim=16, hidden_dim=24)
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    assert torch.equal(model(state), restored(state))


def test_stage2_contract_roundtrip_and_lineage_group_five_way_split(tmp_path):
    records = generate_synthetic_records(n_roots=12, replicates_per_candidate=2, seed=3)
    jsonl = tmp_path / "records.jsonl"
    csv_path = tmp_path / "records.csv"
    write_records_jsonl(records, jsonl)
    write_records_csv(records, csv_path)
    restored = read_records_jsonl(jsonl)
    assert restored == records
    assert "short_window_breach" in json.loads(jsonl.read_text(encoding="utf-8").splitlines()[0])
    assert csv_path.read_text(encoding="utf-8-sig").splitlines()[0].startswith("environment_id")
    split = split_records_by_lineage_group(restored, seed=11)
    split.assert_no_lineage_leakage()
    manifest = split.manifest()
    assert manifest["leakage_check"] == "passed"
    names = ("train", "validation", "temperature_calibration", "risk_calibration", "test")
    assert sum(manifest[name]["records"] for name in names) == len(records)
    assert manifest["split_key"] == "lineage_group_id"
    design = validate_counterfactual_design(restored, min_continuations_per_cell=2)
    assert design["controllable_side"] == "Red"
    assert all("__vs__" not in record.query.candidate_id for record in restored)
    assert all(record.query.threat_id == record.query.attacker_policy_key for record in restored)
    group_sets = [set(manifest[name]["lineage_group_ids"]) for name in names]
    assert all(
        not group_sets[left] & group_sets[right]
        for left in range(len(group_sets))
        for right in range(left + 1, len(group_sets))
    )

    changed_state = list(records[1].query.canonical_state)
    changed_state[0] += 0.1
    inconsistent = replace(
        records[1],
        query=replace(
            records[1].query,
            root_id=records[0].query.root_id,
            canonical_state=tuple(changed_state),
        ),
    )
    with pytest.raises(ValueError, match="inconsistent canonical"):
        validate_rollout_lineage([records[0], inconsistent])

    # Regression: two scale-specific roots sharing a parent/master seed must
    # move together even though their root_id and scenario differ.
    first_root = records[0].query.root_id
    second_root = next(record.query.root_id for record in records if record.query.root_id != first_root)
    parent_group = records[0].query.lineage_group_id
    siblings = [
        replace(record, query=replace(record.query, lineage_group_id=parent_group))
        if record.query.root_id == second_root
        else record
        for record in records
    ]
    sibling_split = split_records_by_lineage_group(siblings, seed=13)
    locations = {}
    for name, rows in sibling_split.partitions:
        for root_id in (first_root, second_root):
            if any(record.query.root_id == root_id for record in rows):
                locations[root_id] = name
    assert locations[first_root] == locations[second_root]


def test_formal_jsonl_requires_exact_schema_v4_and_complete_data_contract(tmp_path):
    records = generate_synthetic_records(
        n_roots=5,
        scales=[(2, 1)],
        defender_policies=["guard", "adaptive"],
        attacker_policies=["rush"],
        replicates_per_candidate=2,
        seed=101,
    )
    checkpoint_sha256 = "a" * 64
    records = [
        replace(
            record,
            query=replace(
                record.query,
                stage1_checkpoint_sha256=(
                    checkpoint_sha256
                    if record.query.defender_policy_id == "adaptive"
                    else ""
                ),
            ),
        )
        for record in records
    ]
    jsonl = tmp_path / "formal.jsonl"
    write_records_jsonl(records, jsonl)
    restored = read_records_jsonl(jsonl, required_schema_version=4)
    contract = {
        "required_schema_version": 4,
        "required_lineage_groups": 5,
        "required_scales": [[2, 1]],
        "required_red_candidate_count": 2,
        "required_blue_threat_count": 1,
        "required_continuations_per_cell": 2,
        "required_nonempty_checkpoint_sha256_count": 1,
        "required_checkpoint_candidate_count": 1,
        "expected_stage1_checkpoint_sha256": checkpoint_sha256,
        "required_physical_target_fields": list(PHYSICAL_TARGET_FIELDS),
        "require_complete_physical_targets": True,
        "require_strict_red_superiority": True,
    }
    audit = validate_formal_dataset_contract(restored, contract)
    assert audit["status"] == "passed"
    assert audit["physical_target_fields"] == list(PHYSICAL_TARGET_FIELDS)
    assert audit["stage1_checkpoint_sha256"] == [checkpoint_sha256]

    old_payload = records[0].to_dict()
    old_payload["schema_version"] = 3
    legacy = tmp_path / "legacy.jsonl"
    legacy.write_text(json.dumps(old_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="requires schema_version=4"):
        read_records_jsonl(legacy, required_schema_version=4)

    incomplete = list(restored)
    incomplete[0] = replace(incomplete[0], payoff_red=None)
    with pytest.raises(ValueError, match="missing 13-head physical labels"):
        validate_formal_dataset_contract(incomplete, contract)
    wrong_checkpoint = dict(contract)
    wrong_checkpoint["expected_stage1_checkpoint_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="differs from policy lock"):
        validate_formal_dataset_contract(restored, wrong_checkpoint)
    wrong_fields = dict(contract)
    wrong_fields["required_physical_target_fields"] = list(
        PHYSICAL_TARGET_FIELDS[:-1]
    )
    with pytest.raises(ValueError, match="13 canonical Stage-2 physical labels"):
        validate_formal_dataset_contract(restored, wrong_fields)
    wrong_schema = dict(contract)
    wrong_schema["required_schema_version"] = 3
    with pytest.raises(ValueError, match="requires schema_version=4"):
        validate_formal_dataset_contract(restored, wrong_schema)


def test_stream_manifest_accounts_for_terminal_snapshot_attrition_without_rollout(
    tmp_path, monkeypatch
):
    import scripts.collect_stage2_rollouts as collection_script

    records = generate_synthetic_records(
        n_roots=5,
        scales=[(2, 1)],
        defender_policies=["guard", "adaptive"],
        attacker_policies=["rush"],
        replicates_per_candidate=2,
        seed=103,
    )

    def fake_iter_had_records(*, collection_audit, **_):
        collection_audit.update(
            {
                "status": "recording",
                "requested_snapshot_steps": [0, 10],
                "planned_snapshot_roots": 6,
                "realized_snapshot_roots": 5,
                "terminal_attrition_roots": 1,
                "unexplained_missing_roots": 0,
                "snapshot_zero_expected_roots": 5,
                "snapshot_zero_realized_roots": 5,
                "red_candidate_count": 2,
                "blue_threat_count": 1,
                "continuations_per_cell": 2,
                "expected_records_from_realized_roots": len(records),
                "by_scale_snapshot": {
                    "2v1": {
                        "snapshot-000": {
                            "planned": 5,
                            "realized": 5,
                            "terminal_attrition": 0,
                            "unexplained_missing": 0,
                        },
                        "snapshot-010": {
                            "planned": 1,
                            "realized": 0,
                            "terminal_attrition": 1,
                            "unexplained_missing": 0,
                        },
                    }
                },
                "missing_snapshot_roots": [
                    {
                        "lineage_group_id": "synthetic:lineage-terminal",
                        "root_seed": 999,
                        "scale": "2v1",
                        "snapshot_step": 10,
                        "status": "terminal_prefix_attrition",
                        "prefix_terminal_step": 7,
                        "prefix_terminal_outcome_red": -1.0,
                    }
                ],
            }
        )
        yield from records
        collection_audit["status"] = "recorded"

    monkeypatch.setattr(
        collection_script, "iter_had_records", fake_iter_had_records
    )
    dataset = tmp_path / "dataset.jsonl"
    summary = collection_script._stream_had_jsonl(
        {}, dataset, min_continuations=2
    )
    attrition = summary["snapshot_attrition_audit"]
    assert attrition["status"] == "passed"
    assert attrition["planned_snapshot_roots"] == 6
    assert attrition["realized_snapshot_roots"] == 5
    assert attrition["terminal_attrition_roots"] == 1
    assert attrition["unexplained_missing_roots"] == 0
    assert all(attrition["integrity_checks"].values())

    import scripts.train_stage2 as training_script

    manifest_path = tmp_path / "dataset_manifest.json"
    manifest = {
        "protocol": "OpenSCORE-Stage2-HAD-v4",
        "record_schema_version": 4,
        "records": len(records),
        "roots": len({record.query.root_id for record in records}),
        "lineage_groups": len(
            {record.query.lineage_group_id for record in records}
        ),
        "dataset_sha256": collection_script._sha256(dataset),
        "snapshot_attrition_audit": attrition,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_audit = training_script._formal_collection_manifest_audit(
        {"manifest_path": str(manifest_path)},
        dataset,
        collection_script._sha256(dataset),
        records,
        {
            "required_snapshot_steps": [0, 10],
            "required_planned_snapshot_roots": 6,
            "required_snapshot_zero_roots": 5,
        },
    )
    assert manifest_audit["status"] == "passed"

    invalid_manifest = dict(manifest)
    invalid_manifest["snapshot_attrition_audit"] = dict(attrition)
    invalid_manifest["snapshot_attrition_audit"]["status"] = "failed"
    manifest_path.write_text(json.dumps(invalid_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="snapshot_attrition_passed"):
        training_script._formal_collection_manifest_audit(
            {"manifest_path": str(manifest_path)},
            dataset,
            collection_script._sha256(dataset),
            records,
            {
                "required_snapshot_steps": [0, 10],
                "required_planned_snapshot_roots": 6,
                "required_snapshot_zero_roots": 5,
            },
        )

    def unexplained_iter(*, collection_audit, **_):
        yield from fake_iter_had_records(
            collection_audit=collection_audit
        )
        collection_audit["terminal_attrition_roots"] = 0
        collection_audit["unexplained_missing_roots"] = 1

    monkeypatch.setattr(
        collection_script, "iter_had_records", unexplained_iter
    )
    with pytest.raises(ValueError, match="no_unexplained_missing_roots"):
        collection_script._stream_had_jsonl(
            {}, tmp_path / "invalid.jsonl", min_continuations=2
        )


def test_had_record_distinguishes_natural_horizon_from_administrative_censoring():
    def state(remaining_horizon: float) -> np.ndarray:
        rows = []
        for type_index, count in ((8, 2), (9, 1)):
            for index in range(count):
                row = np.zeros(12, np.float32)
                row[:3] = (0.2 + 0.1 * index, 0.4, 0.3)
                row[6:8] = (1.0, 1.0)
                row[type_index] = 1.0
                row[11] = remaining_horizon
                rows.append(row)
        target = np.zeros(12, np.float32)
        target[:3] = (0.4, 0.5, 0.2)
        target[6:8] = (1.0, 1.0)
        target[10] = 1.0
        target[11] = remaining_horizon
        return np.stack(rows + [target])

    def episode(final_remaining_horizon: float):
        observations = [
            {"state_entities": state(0.1)},
            {"state_entities": state(final_remaining_horizon)},
        ]
        red = SimpleNamespace(
            scale=(2, 1),
            observations=observations,
            actions=[np.zeros(2, np.int64)],
        )
        blue = SimpleNamespace(
            observations=observations,
            actions=[np.zeros(1, np.int64)],
        )
        return SimpleNamespace(
            red=red,
            blue=blue,
            length=1,
            red_policy_name="rule:guard",
            blue_policy_name="rule:rush",
        )

    def convert(value, rollout_id):
        return record_from_had_episode(
            value,
            lineage_group_id="had:test-lineage",
            root_id="had:test-lineage:2v1:snapshot-000:state-test",
            rollout_id=rollout_id,
            seed=11,
            root_seed=7,
            scenario_id="one_target_2v1_snapshot_000",
            horizon_steps=1,
            command_steps=1,
            defender_policy_version="rule-v1",
            attacker_policy_version="rule-v1",
            capability_version="test-v1",
        )

    natural = convert(episode(0.0), "natural-horizon")
    administrative = convert(episode(0.5), "administrative-cutoff")
    assert natural.outcome == "defender_win"
    assert natural.event_observed
    assert natural.attackers_neutralized_steps is None
    assert administrative.outcome == "timeout"
    assert not administrative.event_observed
    assert administrative.attackers_neutralized_steps is None
    # payoff_red is intentionally a window-safety utility: the asset survived
    # both observation windows even though one event time remains censored.
    assert natural.payoff_red == administrative.payoff_red == 1.0


def test_stage2_machine_acceptance_has_exactly_thirteen_preregistered_gates():
    import scripts.train_stage2 as training_script

    project = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (project / "configs" / "stage2_had_formal.yaml").read_text(
            encoding="utf-8"
        )
    )
    acceptance_config = training_script._validate_acceptance_config(
        config["acceptance"]
    )
    physical_targets = {
        field: {
            "status": "evaluated",
            "mae_skill_over_train_median": 0.10,
            "reasonable_range": {"prediction_inside_rate": 1.0},
        }
        for field in PHYSICAL_TARGET_FIELDS
    }
    metrics = {
        "breach": {
            "brier": 0.10,
            "roc_auc": 0.70,
            "climatology_brier_skill": 0.10,
        },
        "joint_outcome_time": {
            "right_censored_competing_risk_nll": 0.80,
            "integrated_breach_brier": 0.10,
        },
        "empirical_joint_train_frequency_baseline": {
            "right_censored_competing_risk_nll": 1.00,
            "integrated_breach_brier": 0.15,
        },
        "remaining_time": {"event_time_mae_improvement_over_median": 1.0},
        "calibrated_risk_bound": {
            "candidate_rate_coverage": 0.91,
            "mean_upper_bound": 0.90,
            "mean_interval_width": 0.30,
        },
        "red_candidate_decision": {
            "post_selection_coverage": 0.92,
            "mean_red_decision_regret": 0.10,
            "random_red_candidate_expected_regret": 0.20,
            "validation_global_best_red_candidate_regret": 0.15,
            "red_ranking_pair_accuracy": 0.60,
        },
        "physical_evaluators": {
            "targets": physical_targets,
            "redundancy_consistency": {
                "defender_survivors_plus_casualties": {
                    "mean_absolute_prediction_residual": 0.05
                },
                "attacker_survivors_plus_casualties": {
                    "mean_absolute_prediction_residual": 0.10
                },
            },
        },
    }
    tabular = {"test_breach_brier": 0.12, "test_breach_roc_auc": 0.69}
    acceptance = training_script._build_stage2_acceptance(
        metrics, tabular, acceptance_config
    )
    assert acceptance["gate_count"] == 13
    assert set(acceptance["gates"]) == {
        f"S2-{index:02d}" for index in range(1, 14)
    }
    assert all(
        {"observed", "threshold", "passed"}.issubset(gate)
        for gate in acceptance["gates"].values()
    )
    assert acceptance["overall"]["passed"]
    assert acceptance["overall_go_no_go"] == "go"
    json.dumps(acceptance, allow_nan=False)

    failed_metrics = json.loads(json.dumps(metrics))
    failed_metrics["calibrated_risk_bound"]["mean_interval_width"] = 0.31
    failed = training_script._build_stage2_acceptance(
        failed_metrics, tabular, acceptance_config
    )
    assert not failed["gates"]["S2-07"]["passed"]
    assert failed["overall_go_no_go"] == "no-go"

    incomplete = dict(acceptance_config)
    incomplete["gates"] = dict(acceptance_config["gates"])
    incomplete["gates"].pop("S2-13")
    with pytest.raises(ValueError, match="S2-01..S2-13"):
        training_script._validate_acceptance_config(incomplete)


def test_formal_five_way_split_requires_all_three_event_classes_before_fit():
    import scripts.train_stage2 as training_script

    complete = (
        SimpleNamespace(outcome="defender_win", event_observed=True),
        SimpleNamespace(outcome="breach", event_observed=True),
        SimpleNamespace(outcome="timeout", event_observed=False),
    )
    names = (
        "train",
        "validation",
        "temperature_calibration",
        "risk_calibration",
        "test",
    )
    split = SimpleNamespace(partitions=tuple((name, complete) for name in names))
    audit = training_script._formal_split_class_support_audit(split)
    assert audit["status"] == "passed"
    assert all(
        partition["passed"] for partition in audit["partitions"].values()
    )

    invalid_partitions = list(split.partitions)
    invalid_partitions[-1] = ("test", complete[:2])
    with pytest.raises(ValueError, match="class-support gate failed for: test"):
        training_script._formal_split_class_support_audit(
            SimpleNamespace(partitions=tuple(invalid_partitions))
        )


def test_had_canonical_state_is_permutation_invariant_and_marks_padding():
    # Reconstruct a minimal valid raw state independently of generator internals.
    rows = []
    for type_index, count in ((8, 2), (9, 1)):
        for index in range(count):
            row = np.zeros(11, np.float32)
            row[:3] = (0.2 + 0.1 * index, 0.4, 0.3)
            row[6:8] = (1.0, 1.0)
            row[type_index] = 1.0
            rows.append(row)
    target = np.zeros(11, np.float32)
    target[:3] = (0.4, 0.5, 0.2)
    target[6:8] = (1.0, 1.0)
    target[10] = 1.0
    raw = np.stack(rows + [target])
    canonicalizer = HADCanonicalizer()
    expected = canonicalizer(raw)
    permuted = canonicalizer(raw[[2, 0, 3, 1]])
    assert expected.shape == (canonicalizer.state_dim,) == (85,)
    assert np.allclose(expected, permuted)
    # Present bits: two defenders + one attacker, all remaining slots are padding.
    defender_present = expected[8 + 8 :: 9][:4]
    attacker_start = 8 + 9 * 4
    attacker_present = expected[attacker_start + 8 :: 9][:4]
    assert np.array_equal(defender_present, [1.0, 1.0, 0.0, 0.0])
    assert np.array_equal(attacker_present, [1.0, 0.0, 0.0, 0.0])
    assert expected[-1] == 1.0  # legacy 11-column state migration


def test_group_max_risk_offset_is_monotone_over_point_prediction():
    predictions = [0.10, 0.20, 0.40, 0.30]
    labels = [0.20, 0.65, 0.45, 0.35]
    roots = ["a", "a", "b", "b"]
    candidates = ["x", "y", "x", "y"]
    groups = ["ga", "ga", "gb", "gb"]
    bound = CalibratedRiskBound.fit(
        predictions, labels, roots, candidates, groups, alpha=0.20
    )
    assert bound.root_max_offset >= bound.marginal_offset
    point = np.asarray([0.10, 0.60])
    assert np.all(bound.upper(point, selection_safe=True) >= point)


def test_independent_temperature_and_risk_calibration_simulated_coverage():
    rng = np.random.default_rng(41)
    temp_raw_event = rng.uniform(0.05, 0.95, 400)
    temp_probabilities = np.c_[1.0 - temp_raw_event, temp_raw_event]
    temp_labels = rng.binomial(1, temp_raw_event)
    temperature = TemperatureScaler.fit(temp_probabilities, temp_labels)

    risk_raw = rng.uniform(0.10, 0.80, 300)
    risk_probability = temperature.apply(np.c_[1.0 - risk_raw, risk_raw])[:, 1]
    risk_truth = np.clip(risk_probability + rng.normal(0.0, 0.06, len(risk_raw)), 0.0, 1.0)
    risk_groups = [f"risk-{index}" for index in range(len(risk_raw))]
    bound = CalibratedRiskBound.fit(
        risk_probability,
        risk_truth,
        risk_groups,
        ["candidate"] * len(risk_raw),
        risk_groups,
        alpha=0.10,
    )

    test_raw = rng.uniform(0.10, 0.80, 1000)
    test_probability = temperature.apply(np.c_[1.0 - test_raw, test_raw])[:, 1]
    test_truth = np.clip(test_probability + rng.normal(0.0, 0.06, len(test_raw)), 0.0, 1.0)
    coverage = np.mean(test_truth <= bound.upper(test_probability, selection_safe=True))
    assert coverage >= 0.87


def test_stage2_fit_predict_metrics_and_checkpoint_roundtrip(tmp_path):
    records = generate_synthetic_records(n_roots=24, replicates_per_candidate=2, seed=17)
    split = split_records_by_lineage_group(records, seed=19)
    config = Stage2TrainingConfig(
        horizon_bins=8,
        steps_per_bin=10,
        hidden_dim=24,
        model_kind="had_deepset",
        ensemble_members=2,
        epochs=4,
        batch_size=64,
        policy_unknown_augmentation_probability=0.50,
        seed=23,
    )
    fit = fit_stage2_system(
        split.train,
        split.validation,
        split.temperature_calibration,
        split.risk_calibration,
        config,
        device="cpu",
    )
    assert fit.training_summary["temperature_and_risk_calibration_are_disjoint"]
    assert fit.training_summary["bootstrap_unit"] == "lineage_group_id"
    augmentation = fit.training_summary["policy_unknown_augmentation"]
    assert augmentation["defender_unknown_rows"] > 0
    assert augmentation["attacker_unknown_rows"] > 0
    assert augmentation["capability_unknown_rows"] > 0
    prediction = fit.system.predict(split.test)
    assert prediction.probabilities.shape == (len(split.test), 17)
    assert np.allclose(prediction.probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(
        prediction.breach_upper_selection_safe + 1e-7 >= prediction.breach_probability
    )
    assert {
        "target_final_health_fraction",
        "target_min_health_fraction",
        "defender_survivors",
        "attacker_survivors",
        "defender_casualties",
        "attacker_casualties",
        "minimum_threat_distance",
        "cumulative_target_damage",
        "cumulative_defender_damage",
        "cumulative_attacker_damage",
        "red_action_cost",
        "blue_action_cost",
    }.issubset(prediction.physical_point)
    assert np.all(
        (prediction.physical_point["target_final_health_fraction"] >= 0.0)
        & (prediction.physical_point["target_final_health_fraction"] <= 1.0)
    )
    train_physical_medians = {
        field: float(
            np.median(
                [
                    getattr(record, field)
                    for record in split.train
                    if getattr(record, field) is not None
                ]
            )
        )
        for field in prediction.physical_point
    }
    metrics = evaluate_stage2_predictions(
        split.test,
        prediction.probabilities,
        horizon_bins=8,
        steps_per_bin=10,
        breach_upper=prediction.breach_upper_selection_safe,
        member_probabilities=prediction.member_probabilities,
        physical_predictions=prediction.physical_point,
        physical_interval_lower=prediction.physical_interval_lower,
        physical_interval_upper=prediction.physical_interval_upper,
        physical_ensemble_std=prediction.physical_std,
        train_physical_medians=train_physical_medians,
        train_median_event_steps=float(
            np.median([record.terminal_steps for record in split.train if record.event_observed])
        ),
        seen_policy_pairs={record.query.policy_pair for record in split.train},
    )
    assert {"brier", "nll", "ece", "roc_auc", "accuracy"}.issubset(metrics["breach"])
    assert {"event_time_mae_steps", "median_baseline_event_time_mae_steps"}.issubset(
        metrics["remaining_time"]
    )
    assert "by_scale" in metrics and "post_selection_coverage" in metrics["red_candidate_decision"]
    assert metrics["red_candidate_decision"]["decision_variable"] == "Red_candidate_only"
    assert metrics["protocol"]["timeout_is_right_censored"]
    physical_metrics = metrics["physical_evaluators"]
    health_metrics = physical_metrics["targets"]["target_final_health_fraction"]
    assert {"mae", "rmse", "r2", "mae_skill_over_train_median"}.issubset(
        health_metrics
    )
    assert health_metrics["reasonable_range"]["prediction_inside_rate"] == 1.0
    assert health_metrics["ensemble_interval"]["semantics"].startswith("bootstrap")
    assert physical_metrics["derived_redundant_targets"]["defender_survivors"]

    checkpoint = tmp_path / "stage2.pt"
    fit.system.save(checkpoint)
    restored = Stage2System.load(checkpoint)
    restored_prediction = restored.predict(split.test)
    assert np.allclose(prediction.probabilities, restored_prediction.probabilities, atol=1e-6)
    for field in prediction.physical_point:
        assert np.allclose(
            prediction.physical_point[field],
            restored_prediction.physical_point[field],
            atol=1e-6,
        )

    unseen_query = replace(
        split.test[0].query,
        defender_policy_id="never-seen-policy",
        defender_policy_version="unseen-v1",
        candidate_id="never-seen-policy@unseen-v1",
    )
    unseen_prediction = restored.predict([unseen_query])
    assert np.all(np.isfinite(unseen_prediction.probabilities))
    assert np.allclose(unseen_prediction.probabilities.sum(axis=1), 1.0)

    conditional = Stage2FeatureEncoder.fit(split.train, include_policy_context=True)
    unconditional = Stage2FeatureEncoder.fit(split.train, include_policy_context=False)
    assert conditional.output_dim > unconditional.output_dim


def test_right_censored_likelihood_rewards_survival_mass():
    # One timeout censored at the full 4-bin horizon: only survival-tail mass is
    # compatible with the observation.
    bad = torch.tensor([[4.0] * 8 + [-4.0]])
    good = torch.tensor([[-4.0] * 8 + [4.0]])
    event_class = torch.tensor([8])
    observed = torch.tensor([False])
    censor_bin = torch.tensor([4])
    assert competing_risk_nll(good, event_class, observed, censor_bin, 4) < competing_risk_nll(
        bad, event_class, observed, censor_bin, 4
    )


def test_had_deepset_is_invariant_to_entity_slot_permutation():
    torch.manual_seed(7)
    model = HADDeepSetOutcomeNet(input_dim=90, horizon_bins=4, hidden_dim=16)
    features = torch.randn(3, 90)
    # Valid presence bits for all eight entity slots.
    presence = [8 + slot * 9 + 8 for slot in range(8)]
    features[:, presence] = 1.0
    permuted = features.clone()
    red = features[:, 8:44].reshape(3, 4, 9)
    permuted[:, 8:44] = red[:, [2, 0, 3, 1]].reshape(3, 36)
    assert torch.allclose(model(features), model(permuted), atol=1e-6)
    ensemble = BootstrapOutcomeEnsemble(
        members=2,
        input_dim=90,
        horizon_bins=4,
        hidden_dim=16,
        model_kind="had_deepset",
        state_dim=85,
        physical_target_count=3,
    )
    assert torch.allclose(
        ensemble.physical_forward(features),
        ensemble.physical_forward(permuted),
        atol=1e-6,
    )


def test_had_collector_uses_red_candidates_blue_threats_and_crn():
    records = collect_had_records(
        scales=[(2, 1)],
        seeds=[101, 102],
        defender_candidates=["guard", "intercept"],
        attacker_threats=[
            {
                "kind": "rule",
                "style": "rush",
                "policy_id": "threat:rush",
                "version": "noisy-v1",
                "action_noise": 0.10,
            }
        ],
        continuations_per_cell=2,
        max_steps=4,
        command_steps=2,
        shaping_scale=0.5,
        device="cpu",
    )
    audit = validate_counterfactual_design(records, min_continuations_per_cell=2)
    assert audit["common_random_numbers"]
    assert len({record.query.candidate_id for record in records}) == 2
    assert {record.query.threat_id for record in records} == {"threat:rush@noisy-v1"}
    assert all(not record.event_observed for record in records if record.outcome == "timeout")


def test_had_collector_accepts_stage1_checkpoint_as_red_candidate(tmp_path):
    from open_score.envs import HADStage1Adapter
    from open_score.stage1 import VariableScaleQMIX

    dimensions = (
        HADStage1Adapter.ENTITY_DIM,
        HADStage1Adapter.SELF_DIM,
        HADStage1Adapter.TASK_DIM,
        HADStage1Adapter.STATE_ENTITY_DIM,
        HADStage1Adapter.ACTION_DIM,
    )
    model = VariableScaleQMIX(
        *dimensions, agent_hidden_dim=16, mixer_hidden_dim=16, mixing_dim=8
    )
    checkpoint = tmp_path / "stage1_qmix.pt"
    torch.save(
        {
            "online": model.state_dict(),
            "extra": {
                "algorithm": "qmix",
                "train_side": "Red",
                "registered_scales": [[2, 1]],
                "protocol_version": "had-stage1-test",
                "selection": "paired_eval_best",
                "contract": {
                    "entity_dim": dimensions[0],
                    "self_dim": dimensions[1],
                    "task_dim": dimensions[2],
                    "state_entity_dim": dimensions[3],
                    "action_dim": dimensions[4],
                },
                "environment_protocol": {"max_steps": 2},
                "reward_protocol": {"potential_shaping_scale": 0.5},
                "model_config": {
                    "agent_hidden_dim": 16,
                    "critic_or_mixer_hidden_dim": 16,
                    "mixing_dim": 8,
                },
            },
        },
        checkpoint,
    )
    records = collect_had_records(
        scales=[(2, 1)],
        seeds=[909],
        defender_candidates=[
            {
                "kind": "checkpoint",
                "path": str(checkpoint),
                "algorithm": "qmix",
                "policy_id": "stage1:qmix",
                "version": "test-v1",
            }
        ],
        attacker_threats=["rush"],
        continuations_per_cell=2,
        max_steps=2,
        command_steps=1,
        shaping_scale=0.5,
        device="cpu",
    )
    assert {record.query.candidate_id for record in records} == {"stage1:qmix@test-v1"}
    assert all(len(record.query.stage1_checkpoint_sha256) == 64 for record in records)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    strict_records = collect_had_records(
        scales=[(2, 1)],
        seeds=[910],
        defender_candidates=[
            {
                "kind": "checkpoint",
                "path": str(checkpoint),
                "algorithm": "qmix",
                "policy_id": "stage1:qmix",
                "version": "test-v1",
                "expected_sha256": digest,
            }
        ],
        attacker_threats=["rush"],
        continuations_per_cell=2,
        max_steps=2,
        command_steps=1,
        shaping_scale=0.5,
        device="cpu",
        strict_checkpoint_contract=True,
    )
    assert len(strict_records) == 2
    with pytest.raises(ValueError, match="SHA-256 differs"):
        collect_had_records(
            scales=[(2, 1)],
            seeds=[911],
            defender_candidates=[
                {
                    "kind": "checkpoint",
                    "path": str(checkpoint),
                    "algorithm": "qmix",
                    "expected_sha256": "0" * 64,
                }
            ],
            attacker_threats=["rush"],
            continuations_per_cell=2,
            max_steps=2,
            command_steps=1,
            shaping_scale=0.5,
            device="cpu",
            strict_checkpoint_contract=True,
        )


def test_stage2_formal_cli_and_plot_end_to_end(tmp_path):
    project = Path(__file__).resolve().parents[1]
    output = tmp_path / "formal_cli"
    subprocess.run(
        [
            sys.executable,
            str(project / "scripts" / "train_stage2.py"),
            "--config",
            str(project / "configs" / "stage2_formal_integration.yaml"),
            "--output-dir",
            str(output),
            "--device",
            "cpu",
        ],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert payload["formal_protocol_version"] == "OpenSCORE-Stage2-HAD-v4"
    assert payload["split_manifest"]["validation"]["lineage_groups"] > 0
    assert payload["strong_tabular_baseline"]["hyperparameter_selection_split"] == "validation"
    physical = payload["test_metrics"]["physical_evaluators"]["targets"]
    assert physical["target_final_health_fraction"]["status"] == "evaluated"
    first_prediction = json.loads(
        (output / "test_predictions.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    target_prediction = first_prediction["physical_predictions"][
        "target_final_health_fraction"
    ]
    assert {"point", "ensemble_p10", "ensemble_p90", "ensemble_std"}.issubset(
        target_prediction
    )
    figures = output / "figures"
    subprocess.run(
        [
            sys.executable,
            str(project / "scripts" / "plot_stage2_results.py"),
            "--metrics",
            str(output / "metrics.json"),
            "--predictions",
            str(output / "test_predictions.jsonl"),
            "--output-dir",
            str(figures),
        ],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    )
    assert (figures / "stage2_summary.png").stat().st_size > 10_000
    assert (figures / "stage2_summary.pdf").stat().st_size > 1_000
    assert (figures / "stage2_physical_evaluators.png").stat().st_size > 10_000


def test_had_snapshot_restore_and_prefix_roots_are_reproducible():
    from open_score.envs import HADStage1Adapter

    adapter = HADStage1Adapter(2, 1, max_steps=8)
    adapter.reset(seed=71)
    snapshot = adapter.snapshot()
    red = np.zeros(2, dtype=np.int64)
    blue = np.zeros(1, dtype=np.int64)
    adapter.restore(snapshot, continuation_seed=991)
    first = adapter.step(red, blue)[0]["Red"]["state_entities"].copy()
    adapter.restore(snapshot, continuation_seed=991)
    second = adapter.step(red, blue)[0]["Red"]["state_entities"].copy()
    assert np.array_equal(first, second)

    records = collect_had_records(
        scales=[(2, 1)],
        seeds=[71, 72, 73, 74, 75],
        defender_candidates=["guard", "intercept"],
        attacker_threats=["rush"],
        snapshot_steps=[0, 1],
        rollout_horizon_steps=2,
        continuations_per_cell=2,
        max_steps=3,
        command_steps=1,
    )
    assert any("snapshot-001" in record.query.root_id for record in records)
    assert all("state-" in record.query.root_id for record in records)
    assert any(record.query.canonical_state[-1] < 1.0 for record in records)
    assert all(record.target_final_health_fraction is not None for record in records)
    assert all(record.minimum_threat_distance is not None for record in records)


def test_had_streaming_iterator_yields_without_materialising_all_records():
    iterator = iter_had_records(
        scales=[(2, 1)],
        seeds=[303],
        defender_policies=["guard", "intercept"],
        attacker_policies=["rush"],
        continuations_per_cell=2,
        max_steps=8,
        rollout_horizon_steps=8,
        command_steps=4,
    )
    assert iter(iterator) is iterator
    rows = list(iterator)
    assert len(rows) == 4
    assert len({row.query.rollout_id for row in rows}) == 4


def test_counterfactual_audit_rejects_reused_continuation_seed():
    records = generate_synthetic_records(
        n_roots=5,
        scales=[(2, 1)],
        defender_policies=["guard", "adaptive"],
        attacker_policies=["rush"],
        replicates_per_candidate=2,
        seed=81,
    )
    first_seed = {}
    invalid = []
    for record in records:
        key = (
            record.query.root_id,
            record.query.candidate_id,
            record.query.threat_id,
        )
        seed = first_seed.setdefault(key, record.query.continuation_seed)
        invalid.append(
            replace(
                record,
                query=replace(record.query, seed=seed, continuation_seed=seed),
            )
        )
    with pytest.raises(ValueError, match="reuse a continuation seed"):
        validate_counterfactual_design(invalid, min_continuations_per_cell=2)


def test_formal_conformal_gate_rejects_too_few_lineages():
    with pytest.raises(ValueError, match="too few independent calibration lineages"):
        CalibratedRiskBound.fit(
            [0.2, 0.3],
            [0.0, 1.0],
            ["root-a", "root-b"],
            ["guard", "guard"],
            ["lineage-a", "lineage-b"],
            alpha=0.10,
            require_finite_sample_guarantee=True,
        )
