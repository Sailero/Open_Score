from dataclasses import replace

import pytest

from had_env.workbench.evaluation import EvaluationPlan, evaluate_policies, summarize_evaluation
from had_env.workbench.protocols import CUSTOM_PROTOCOL, ProtocolSpec, ScenarioSpec, scenario_family


def test_split_contract_distinguishes_new_openings_from_structural_ood():
    train = scenario_family(31, 0, split="train")
    validation = scenario_family(31, 0, split="validation")
    test_id = scenario_family(31, 0, split="test")
    test_ood = scenario_family(31, 0, split="test", distribution="ood", cells=((12, 8),))
    plan = EvaluationPlan((train,), (validation,), (test_id, test_ood))
    assert EvaluationPlan.from_dict(plan.to_dict()) == plan
    assert train.opening_seed != test_id.opening_seed
    with pytest.raises(ValueError, match="OOD"):
        EvaluationPlan((train,), (), (replace(test_id, distribution="ood", scenario_id=""),))
    with pytest.raises(ValueError, match="opening"):
        EvaluationPlan((train,), (), (replace(train, split="test", scenario_id="other"),))


def test_native_v5_semantics_cannot_be_silently_relabelled():
    with pytest.raises(ValueError, match="new registered"):
        ProtocolSpec(firing_control="learned")
    with pytest.raises(ValueError, match="protocol_id"):
        ScenarioSpec(max_steps=1)
    custom = ScenarioSpec(max_steps=1, protocol_id=CUSTOM_PROTOCOL)
    with pytest.raises(ValueError, match="relabelled"):
        custom.episode_spec()


def test_cross_seed_means_are_equal_weighted_and_missing_pairs_are_explicit():
    def row(method, seed, scenario, win):
        return dict(method_id=method, training_seed=seed, scenario_id=scenario,
                    split="test", distribution="id", success_native=win,
                    opening_seed=1, opponent_seed=2)
    rows = [row("new", 1, "a", True), row("new", 2, "a", False),
            row("new", 2, "b", False), row("rule", 1, "a", False)]
    summary = summarize_evaluation(rows)
    new = next(group for group in summary["groups"] if group["method_id"] == "new")
    assert new["mean_seed_success_rate"] == .5  # Not the pooled 1/3.
    assert not new["balanced_scenario_coverage"] and new["across_seed_t95"] is None
    assert summary["paired"][0]["pairs"] == 1
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_evaluation(rows + rows[:1])


def test_small_paired_evaluation_uses_identical_openings_and_saves_records(tmp_path):
    scenarios = [ScenarioSpec(protocol_id=CUSTOM_PROTOCOL, max_steps=1, red_count=2, blue_count=1)]
    result = evaluate_policies({"rule": "rule", "grand": "grand"}, scenarios,
                               training_seeds=(1, 2), record_dir=tmp_path)
    assert len(result["episodes"]) == 4
    assert result["paired"][0]["pairs"] == 2
    assert all(row["physical_steps"] == 1 for row in result["episodes"])
    assert len(list(tmp_path.glob("*.json.gz"))) == 4
    contents = {path: path.read_bytes() for path in tmp_path.glob("*.json.gz")}
    with pytest.raises(FileExistsError, match="new output"):
        evaluate_policies({"rule": "rule"}, scenarios, training_seeds=(1,), record_dir=tmp_path)
    assert {path: path.read_bytes() for path in contents} == contents
