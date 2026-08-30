"""Fast real-update tests for the dedicated SMAClite-AD training path."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("smaclite")

from open_score.stage1.baselines import (  # noqa: E402
    SequenceMAPPOLearner,
    VariableScaleMAPPO,
    VariableScaleVDN,
)
from open_score.stage1.entity_qmix import VariableScaleQMIX  # noqa: E402
from open_score.stage1.learner import SequenceQMIXLearner  # noqa: E402
from open_score.stage1.replay import collate_episodes  # noqa: E402
from open_score.stage1.smaclite_ad_training import (  # noqa: E402
    SMACliteADEpisodeRunner,
    SMACliteADFactory,
    SMACliteADMAPPOController,
    SMACliteADQController,
    SMACliteADRuleController,
    evaluate_smaclite_ad,
    tensor_shape_audit,
)


RATIOS = [(2, 1), (3, 2), (5, 3)]


def _dimensions(env):
    return (
        env.ENTITY_DIM,
        env.SELF_DIM,
        env.TASK_DIM,
        env.STATE_ENTITY_DIM,
        env.ACTION_DIM,
    )


def _changed(before, model):
    return any(
        not torch.equal(old, new)
        for old, new in zip(before, model.parameters())
    )


def test_smaclite_ad_runner_keeps_shapes_and_paired_seeds():
    factory = SMACliteADFactory(episode_limit=2)
    audit = tensor_shape_audit(factory, RATIOS, seed=40)
    assert audit["shared_across_ratios"]
    runner = SMACliteADEpisodeRunner(factory)
    controlled = SMACliteADRuleController("rush_asset")
    opponent = SMACliteADRuleController("idle")
    first = evaluate_smaclite_ad(runner, controlled, opponent, "Red", RATIOS, 1, 91)
    second = evaluate_smaclite_ad(runner, controlled, opponent, "Red", RATIOS, 1, 91)
    assert first.paired_returns == second.paired_returns
    assert set(first.per_ratio) == {"2:1", "3:2", "5:3"}
    factory.close()


@pytest.mark.parametrize("algorithm", ["qmix", "vdn"])
def test_qmix_and_vdn_update_once_on_all_ad_ratios(algorithm):
    torch.manual_seed(7)
    factory = SMACliteADFactory(episode_limit=2, approach_weight=1.0)
    env = factory.get(RATIOS[0])
    model = (
        VariableScaleQMIX(*_dimensions(env), agent_hidden_dim=16, mixer_hidden_dim=16, mixing_dim=8)
        if algorithm == "qmix"
        else VariableScaleVDN(*_dimensions(env), agent_hidden_dim=16)
    )
    controller = SMACliteADQController(model, torch.device("cpu"), epsilon=1.0)
    opponent = SMACliteADRuleController("idle")
    runner = SMACliteADEpisodeRunner(factory)
    episodes = [
        runner.run(ratio, controller, opponent, 100 + index).red
        for index, ratio in enumerate(RATIOS)
    ]
    learner = SequenceQMIXLearner(model, target_update_interval=1)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    metrics = learner.train_batch(collate_episodes(episodes))
    assert np.isfinite([metrics.loss, metrics.grad_norm, metrics.mean_absolute_td]).all()
    assert set(metrics.td_by_scale) == set(RATIOS)
    assert _changed(before, model)
    factory.close()


def test_mappo_updates_once_on_all_ad_ratios():
    torch.manual_seed(8)
    factory = SMACliteADFactory(episode_limit=2, approach_weight=1.0)
    env = factory.get(RATIOS[0])
    model = VariableScaleMAPPO(
        *_dimensions(env), actor_hidden_dim=16, critic_hidden_dim=16
    )
    controller = SMACliteADMAPPOController(
        model, torch.device("cpu"), deterministic=False
    )
    opponent = SMACliteADRuleController("idle")
    runner = SMACliteADEpisodeRunner(factory)
    episodes = [
        runner.run(ratio, controller, opponent, 200 + index).red
        for index, ratio in enumerate(RATIOS)
    ]
    learner = SequenceMAPPOLearner(model, epochs=1)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    metrics = learner.train_batch(collate_episodes(episodes))
    assert np.isfinite(
        [metrics.loss, metrics.policy_loss, metrics.value_loss, metrics.grad_norm]
    ).all()
    assert set(metrics.learning_signal_by_scale) == set(RATIOS)
    assert _changed(before, model)
    factory.close()


def test_randomized_evaluation_counts_effective_unique_layouts():
    factory = SMACliteADFactory(episode_limit=2, spawn_jitter=2.0)
    runner = SMACliteADEpisodeRunner(factory)
    evaluation = evaluate_smaclite_ad(
        runner,
        SMACliteADRuleController("rush_asset"),
        SMACliteADRuleController("idle"),
        "Red",
        RATIOS,
        5,
        90_000,
    )
    assert evaluation.episodes == 15
    assert evaluation.unique_layouts == 15
    assert all(values["unique_layouts"] == 5 for values in evaluation.per_ratio.values())
    factory.close()


def test_cli_saves_distinct_validation_best_and_final_checkpoints(tmp_path):
    project = Path(__file__).resolve().parents[1]
    output = tmp_path / "outputs"
    evidence = tmp_path / "evidence"
    process_environment = os.environ.copy()
    process_environment["PYTHONPATH"] = str(project / "src")
    command = [
        sys.executable,
        str(project / "scripts" / "train_smaclite_ad_baselines.py"),
        "--algorithms",
        "qmix",
        "--seeds",
        "71",
        "--episodes",
        "3",
        "--episode-limit",
        "2",
        "--eval-every",
        "3",
        "--validation-episodes-per-ratio",
        "5",
        "--heldout-episodes-per-ratio",
        "5",
        "--batch-episodes",
        "3",
        "--agent-hidden-dim",
        "8",
        "--critic-hidden-dim",
        "8",
        "--spawn-jitter",
        "2",
        "--output-dir",
        str(output),
        "--evidence-dir",
        str(evidence),
    ]
    completed = subprocess.run(
        command,
        cwd=project,
        env=process_environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(
        (evidence / "smaclite_ad_baselines_summary.json").read_text(encoding="utf-8")
    )
    result = payload["results"][0]
    checkpoints = result["checkpoints"]
    assert checkpoints["distinct_files"]
    assert checkpoints["validation_best"]["checkpoint_sha256"]
    assert checkpoints["final"]["checkpoint_sha256"]
    assert Path(checkpoints["validation_best"]["checkpoint"]).is_file()
    assert Path(checkpoints["final"]["checkpoint"]).is_file()
    assert result["layout_split_audit"]["disjoint"]
    training_hashes = result["training_layout_hashes_per_ratio"]
    assert set(training_hashes) == {"2:1", "3:2", "5:3"}
    assert sum(map(len, training_hashes.values())) == result["episodes"] == 3
    schedule = result["training_layout_schedule"]
    assert len(schedule) == result["episodes"]
    assert [record["episode_index"] for record in schedule] == list(
        range(1, result["episodes"] + 1)
    )
    assert len({record["seed"] for record in schedule}) == result["episodes"]
    assert len({record["sample_sha256"] for record in schedule}) == result["episodes"]
    assert all(
        record["seed"] == result["seed"] + record["episode_index"] * 101
        and len(record["layout_sha256"]) == 64
        and len(record["sample_sha256"]) == 64
        and len(record["randomization_config_sha256"]) == 64
        for record in schedule
    )
    assert {
        label: [
            record["layout_sha256"]
            for record in schedule
            if record["ratio"] == label
        ]
        for label in training_hashes
    } == training_hashes
    manifest = result["training_layout_manifest"]
    assert len(manifest) == result["episodes"]
    assert [record["layout_sha256"] for record in manifest] == [
        record["layout_sha256"] for record in schedule
    ]
    assert all(record["validation"]["valid"] for record in manifest)
    assert result["training_layout_evidence"] == {
        "capture_method": "inline_from_rollout_final_info",
        "seed_formula": "training_seed + episode_index * 101 (1-based)",
        "episodes_recorded": result["episodes"],
        "ordered_schedule_complete": True,
    }
    tracked_training_set = {
        layout_hash
        for ratio_hashes in training_hashes.values()
        for layout_hash in ratio_hashes
    }
    validation_set = set(result["validation"]["initial"]["layout_hashes"])
    heldout_set = set(result["heldout"]["initial"]["layout_hashes"])
    assert len(tracked_training_set) == result["layout_split_audit"][
        "training_unique_layouts"
    ]
    assert not tracked_training_set & validation_set
    assert not tracked_training_set & heldout_set
    assert result["layout_split_audit"]["validation_unique_layouts"] == 15
    assert result["layout_split_audit"]["heldout_unique_layouts"] == 15
