"""Safety tests for stock-to-SMAClite-AD initialization."""

from pathlib import Path

import pytest
import torch

from open_score.stage1.baselines import VariableScaleMAPPO, VariableScaleVDN
from open_score.stage1.entity_qmix import VariableScaleQMIX
from open_score.stage1.transfer import (
    ACTION_TARGET_CONTRACT,
    transfer_stock_checkpoint,
)


TARGET_TYPE_SEMANTICS = {
    "0": "non_target",
    "1": "enemy_damage",
    "2": "ally_heal",
    "3": "protected_asset",
}


def _model(
    algorithm: str,
    source: bool,
    encoder_kind: str = "deepset",
    attention_heads: int = 4,
):
    # Source mimics the flat stock contract; target mimics the AD contract.
    dims = (128, 1, 1, 217, 14) if source else (13, 12, 9, 12, 12)
    if algorithm == "qmix":
        return VariableScaleQMIX(
            *dims,
            agent_hidden_dim=16,
            mixer_hidden_dim=16,
            mixing_dim=8,
            encoder_kind=encoder_kind,
            attention_heads=attention_heads,
        )
    if algorithm == "vdn":
        return VariableScaleVDN(
            *dims,
            agent_hidden_dim=16,
            encoder_kind=encoder_kind,
            attention_heads=attention_heads,
        )
    return VariableScaleMAPPO(
        *dims,
        actor_hidden_dim=16,
        critic_hidden_dim=16,
        encoder_kind=encoder_kind,
        attention_heads=attention_heads,
    )


@pytest.mark.parametrize("algorithm", ["qmix", "vdn", "mappo"])
def test_transfer_copies_only_registered_compatible_latent_layers(
    algorithm: str, tmp_path: Path
):
    torch.manual_seed(3)
    source = _model(algorithm, source=True)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.fill_(0.125)
    target = _model(algorithm, source=False)
    before = {
        name: tensor.detach().clone() for name, tensor in target.state_dict().items()
    }
    checkpoint = {
        "model" if algorithm == "mappo" else "online": source.state_dict(),
        "extra": {
            "algorithm": algorithm,
            "environment_family": "SMAClite-stock",
            "environment_id": "smaclite/3s5z-v0",
            "upstream_commit": "test-commit",
            "training_learner_updates": 7,
            "selected_checkpoint_learner_updates": 7,
            "total_training_learner_updates": 9,
            "use_cpp_rvo2": False,
            "source_trained_tensor_names": list(source.state_dict()),
            "source_trained_target_type_rows": {"1": "enemy_damage"},
            "source_performance_gate": {
                "passed": True,
                "split": "heldout_after_validation_selection",
                "win_rate": 0.75,
                "minimum_win_rate": 0.20,
                "wins": 48,
                "episodes": 64,
            },
            "source_multi_seed_performance_gate": {
                "passed": True,
                "seeds": [3, 5, 7, 11, 13],
                "minimum_seed_count": 5,
                "minimum_mean_win_rate": 0.20,
                "heldout_win_rate": {"mean": 0.70},
            },
            "training_hyperparameters": {
                "learning_rate": 5e-4,
                "target_update_interval": 200,
                "td_lambda": 0.6,
                "ppo_epochs": 4,
            },
            "contract": {
                "purpose": "unit-test",
                "action_target_contract": ACTION_TARGET_CONTRACT,
                "action_target_types": TARGET_TYPE_SEMANTICS,
                "encoder_kind": "deepset",
                "attention_heads": 4,
            },
        },
    }
    path = tmp_path / f"{algorithm}.pt"
    torch.save(checkpoint, path)
    manifest = transfer_stock_checkpoint(target, path, algorithm)
    after = target.state_dict()

    assert manifest["copied_tensor_count"] > 0
    assert 0.0 < manifest["copied_parameter_fraction"] < 1.0
    assert manifest["fresh_optimizer_required"]
    assert manifest["source_target_scorer_trained"]
    assert not manifest["source_target_network_transferred"]
    assert manifest["source_selected_checkpoint_learner_updates"] == 7
    assert manifest["source_total_training_learner_updates"] == 9
    for record in manifest["copied"]:
        name = record["name"]
        assert torch.equal(after[name], source.state_dict()[name])
    excluded = {
        record["name"]
        for record in manifest["skipped"]
        if record["reason"] == "excluded_semantic_boundary"
    }
    scorer_prefix = (
        "actor.policy_head.target_scorer."
        if algorithm == "mappo"
        else "agent.q_head.target_scorer."
    )
    assert any(record["name"].startswith(scorer_prefix) for record in manifest["copied"])
    embedding_name = (
        "actor.policy_head.target_type_embedding.weight"
        if algorithm == "mappo"
        else "agent.q_head.target_type_embedding.weight"
    )
    assert manifest["partial_row_copy_count"] == 1
    assert manifest["partial_row_copies"][0]["row"] == 1
    assert manifest["partial_row_copies"][0]["semantic"] == "enemy_damage"
    assert torch.equal(after[embedding_name][1], source.state_dict()[embedding_name][1])
    for row in (0, 2, 3):
        assert torch.equal(after[embedding_name][row], before[embedding_name][row])
    head = (
        "actor.policy_head.fixed_head.bias"
        if algorithm == "mappo"
        else "agent.q_head.fixed_head.bias"
    )
    assert head in excluded
    assert torch.equal(after[head], before[head])

    del checkpoint["extra"]["selected_checkpoint_learner_updates"]
    missing_selected_audit = tmp_path / f"{algorithm}-missing-selected-update.pt"
    torch.save(checkpoint, missing_selected_audit)
    with pytest.raises(ValueError, match="selected_checkpoint_learner_updates"):
        transfer_stock_checkpoint(
            _model(algorithm, source=False), missing_selected_audit, algorithm
        )


def test_transfer_rejects_checkpoint_without_stock_provenance(tmp_path: Path):
    model = _model("qmix", source=False)
    source = _model("qmix", source=True)
    path = tmp_path / "unsafe.pt"
    torch.save(
        {
            "online": source.state_dict(),
            "extra": {
                "algorithm": "qmix",
                "environment_family": "unknown",
                "training_learner_updates": 1,
                "selected_checkpoint_learner_updates": 1,
                "total_training_learner_updates": 1,
                "source_trained_tensor_names": list(source.state_dict()),
                "source_trained_target_type_rows": {"1": "enemy_damage"},
                "contract": {
                    "action_target_contract": ACTION_TARGET_CONTRACT,
                    "action_target_types": TARGET_TYPE_SEMANTICS,
                    "encoder_kind": "deepset",
                    "attention_heads": 4,
                },
            },
        },
        path,
    )
    with pytest.raises(ValueError, match="environment_family"):
        transfer_stock_checkpoint(model, path, "qmix")


def test_transfer_rejects_untrained_target_scorer(tmp_path: Path):
    source = _model("qmix", source=True)
    target = _model("qmix", source=False)
    state = source.state_dict()
    path = tmp_path / "untrained-target-head.pt"
    torch.save(
        {
            "online": state,
            "extra": {
                "algorithm": "qmix",
                "environment_family": "SMAClite-stock",
                "training_learner_updates": 3,
                "selected_checkpoint_learner_updates": 3,
                "total_training_learner_updates": 3,
                "source_trained_tensor_names": [
                    name for name in state if "target_scorer" not in name
                ],
                "source_trained_target_type_rows": {"1": "enemy_damage"},
                "contract": {
                    "action_target_contract": ACTION_TARGET_CONTRACT,
                    "action_target_types": TARGET_TYPE_SEMANTICS,
                    "encoder_kind": "deepset",
                    "attention_heads": 4,
                },
            },
        },
        path,
    )
    with pytest.raises(ValueError, match="target-scorer"):
        transfer_stock_checkpoint(target, path, "qmix")


@pytest.mark.parametrize(
    "missing_gate, message",
    [
        ("source_performance_gate", "held-out policy-performance"),
        ("source_multi_seed_performance_gate", "multi-seed performance"),
    ],
)
def test_transfer_rejects_source_without_real_performance_gate(
    tmp_path: Path, missing_gate: str, message: str
):
    source = _model("qmix", source=True)
    target = _model("qmix", source=False)
    extra = {
        "algorithm": "qmix",
        "environment_family": "SMAClite-stock",
        "training_learner_updates": 3,
        "selected_checkpoint_learner_updates": 3,
        "total_training_learner_updates": 3,
        "source_trained_tensor_names": list(source.state_dict()),
        "source_trained_target_type_rows": {"1": "enemy_damage"},
        "contract": {
            "action_target_contract": ACTION_TARGET_CONTRACT,
            "action_target_types": TARGET_TYPE_SEMANTICS,
            "encoder_kind": "deepset",
            "attention_heads": 4,
        },
        "source_performance_gate": {
            "passed": True,
            "win_rate": 0.5,
            "minimum_win_rate": 0.2,
            "wins": 5,
            "episodes": 10,
        },
        "source_multi_seed_performance_gate": {
            "passed": True,
            "seeds": [1],
            "minimum_seed_count": 1,
            "minimum_mean_win_rate": 0.2,
            "heldout_win_rate": {"mean": 0.5},
        },
    }
    del extra[missing_gate]
    path = tmp_path / f"missing-{missing_gate}.pt"
    torch.save({"online": source.state_dict(), "extra": extra}, path)
    with pytest.raises(ValueError, match=message):
        transfer_stock_checkpoint(target, path, "qmix")


@pytest.mark.parametrize("algorithm", ["qmix", "vdn", "mappo"])
def test_saqa_transfer_requires_identical_encoder_contract(algorithm, tmp_path):
    source = _model(algorithm, source=True, encoder_kind="saqa", attention_heads=4)
    target = _model(algorithm, source=False, encoder_kind="saqa", attention_heads=4)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.fill_(0.25)
    extra = {
        "algorithm": algorithm,
        "environment_family": "SMAClite-stock",
        "training_learner_updates": 5,
        "selected_checkpoint_learner_updates": 5,
        "total_training_learner_updates": 5,
        "source_trained_tensor_names": list(source.state_dict()),
        "source_trained_target_type_rows": {"1": "enemy_damage"},
        "contract": {
            "action_target_contract": ACTION_TARGET_CONTRACT,
            "action_target_types": TARGET_TYPE_SEMANTICS,
            "encoder_kind": "saqa",
            "attention_heads": 4,
        },
        "source_performance_gate": {
            "passed": True,
            "win_rate": 0.5,
            "minimum_win_rate": 0.2,
            "wins": 5,
            "episodes": 10,
        },
        "source_multi_seed_performance_gate": {
            "passed": True,
            "seeds": [1],
            "minimum_seed_count": 1,
            "minimum_mean_win_rate": 0.2,
            "heldout_win_rate": {"mean": 0.5},
        },
    }
    key = "model" if algorithm == "mappo" else "online"
    path = tmp_path / f"saqa-{algorithm}.pt"
    torch.save({key: source.state_dict(), "extra": extra}, path)
    manifest = transfer_stock_checkpoint(target, path, algorithm)
    assert manifest["encoder_kind"] == "saqa"
    assert manifest["attention_heads"] == 4
    assert any("cross_attention" in row["name"] for row in manifest["copied"])

    mismatch = _model(
        algorithm, source=False, encoder_kind="saqa", attention_heads=2
    )
    with pytest.raises(ValueError, match="encoder contracts differ"):
        transfer_stock_checkpoint(mismatch, path, algorithm)


def test_transfer_rejects_target_type_semantic_mismatch(tmp_path: Path):
    source = _model("qmix", source=True)
    target = _model("qmix", source=False)
    extra = {
        "algorithm": "qmix",
        "environment_family": "SMAClite-stock",
        "training_learner_updates": 3,
        "selected_checkpoint_learner_updates": 3,
        "total_training_learner_updates": 3,
        "source_trained_tensor_names": list(source.state_dict()),
        "source_trained_target_type_rows": {"1": "enemy_damage"},
        "contract": {
            "action_target_contract": ACTION_TARGET_CONTRACT,
            "action_target_types": {**TARGET_TYPE_SEMANTICS, "1": "asset"},
            "encoder_kind": "deepset",
            "attention_heads": 4,
        },
    }
    path = tmp_path / "wrong-target-semantics.pt"
    torch.save({"online": source.state_dict(), "extra": extra}, path)
    with pytest.raises(ValueError, match="target-type semantic"):
        transfer_stock_checkpoint(target, path, "qmix")


def test_transfer_requires_row_level_target_type_training_evidence(tmp_path: Path):
    source = _model("qmix", source=True)
    target = _model("qmix", source=False)
    extra = {
        "algorithm": "qmix",
        "environment_family": "SMAClite-stock",
        "training_learner_updates": 3,
        "selected_checkpoint_learner_updates": 3,
        "total_training_learner_updates": 3,
        "source_trained_tensor_names": list(source.state_dict()),
        "contract": {
            "action_target_contract": ACTION_TARGET_CONTRACT,
            "action_target_types": TARGET_TYPE_SEMANTICS,
            "encoder_kind": "deepset",
            "attention_heads": 4,
        },
    }
    path = tmp_path / "missing-row-training-evidence.pt"
    torch.save({"online": source.state_dict(), "extra": extra}, path)
    with pytest.raises(ValueError, match="row-level training evidence"):
        transfer_stock_checkpoint(target, path, "qmix")
