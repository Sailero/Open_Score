import json
from dataclasses import replace

import numpy as np
import pytest

from open_score.stage2 import (
    CalibratedRiskBound,
    HADCanonicalizer,
    Stage2FeatureEncoder,
    Stage2System,
    Stage2TrainingConfig,
    TemperatureScaler,
    evaluate_stage2_predictions,
    fit_stage2_system,
    generate_synthetic_records,
    read_records_jsonl,
    split_records_by_lineage_group,
    validate_rollout_lineage,
    write_records_csv,
    write_records_jsonl,
)


def test_stage2_contract_roundtrip_and_lineage_group_four_way_split(tmp_path):
    records = generate_synthetic_records(n_roots=12, replicates_per_candidate=1, seed=3)
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
    names = ("train", "temperature_calibration", "risk_calibration", "test")
    assert sum(manifest[name]["records"] for name in names) == len(records)
    assert manifest["split_key"] == "lineage_group_id"
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
    assert expected.shape == (canonicalizer.state_dim,)
    assert np.allclose(expected, permuted)
    # Present bits: two defenders + one attacker, all remaining slots are padding.
    defender_present = expected[8 + 8 :: 9][:4]
    attacker_start = 8 + 9 * 4
    attacker_present = expected[attacker_start + 8 :: 9][:4]
    assert np.array_equal(defender_present, [1.0, 1.0, 0.0, 0.0])
    assert np.array_equal(attacker_present, [1.0, 0.0, 0.0, 0.0])


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
    records = generate_synthetic_records(n_roots=24, replicates_per_candidate=1, seed=17)
    split = split_records_by_lineage_group(records, seed=19)
    config = Stage2TrainingConfig(
        horizon_bins=8,
        steps_per_bin=10,
        hidden_dim=24,
        ensemble_members=2,
        epochs=4,
        batch_size=64,
        policy_unknown_augmentation_probability=0.50,
        seed=23,
    )
    fit = fit_stage2_system(
        split.train,
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
    metrics = evaluate_stage2_predictions(
        split.test,
        prediction.probabilities,
        horizon_bins=8,
        steps_per_bin=10,
        breach_upper=prediction.breach_upper_selection_safe,
        member_probabilities=prediction.member_probabilities,
        train_mean_terminal_steps=float(np.mean([record.terminal_steps for record in split.train])),
        seen_policy_pairs={record.query.policy_pair for record in split.train},
    )
    assert {"brier", "nll", "ece", "roc_auc", "accuracy"}.issubset(metrics["breach"])
    assert {"time_mae_steps", "time_rmse_steps"}.issubset(metrics["remaining_time"])
    assert "by_scale" in metrics and "post_selection_coverage" in metrics["candidate_decision"]

    checkpoint = tmp_path / "stage2.pt"
    fit.system.save(checkpoint)
    restored = Stage2System.load(checkpoint)
    restored_prediction = restored.predict(split.test)
    assert np.allclose(prediction.probabilities, restored_prediction.probabilities, atol=1e-6)

    unseen_query = replace(
        split.test[0].query,
        defender_policy_id="never-seen-policy",
        defender_policy_version="unseen-v1",
    )
    unseen_prediction = restored.predict([unseen_query])
    assert np.all(np.isfinite(unseen_prediction.probabilities))
    assert np.allclose(unseen_prediction.probabilities.sum(axis=1), 1.0)

    conditional = Stage2FeatureEncoder.fit(split.train, include_policy_context=True)
    unconditional = Stage2FeatureEncoder.fit(split.train, include_policy_context=False)
    assert conditional.output_dim > unconditional.output_dim
