from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts import run_stage23_pipeline as pipeline


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _evidence(tmp_path: Path) -> tuple[Path, Path]:
    stage1 = tmp_path / "stage1.pt"
    stage1.write_bytes(b"stage1")
    config = {
        "seed": 9,
        "stage1_checkpoint": str(stage1),
        "execution_semantics": {"local": "one_group"},
        "supported_roster": {"min_red": 1, "max_red": 4},
        "collection": {
            "seed_base": 100,
            "opponents": ["rush"],
            "core_scales": [[1, 1]],
            "core_episodes_per_cell": 4,
            "core_training_weight": 1.0,
            "sparse_scales": [],
            "sparse_episodes_per_cell": 0,
            "sparse_training_weight": 0.0,
            "heldout_scales": [],
            "heldout_episodes_per_cell": 0,
            "split": {
                "train_fraction": 0.25,
                "validation_fraction": 0.25,
                "calibration_fraction": 0.25,
            },
        },
        "training": {"seed": 9, "max_epochs": 2},
    }
    config_path = tmp_path / "stage2.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    root = tmp_path / "run"
    dataset = root / "stage2/data/dynamic_outcome_time.jsonl"
    checkpoint = root / "stage2/model/dynamic_outcome_time_model.pt"
    dataset.parent.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    dataset.write_text('{"sample":1}\n', encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    (root / "stage2/stage2_report.md").write_text("ok", encoding="utf-8")
    _write_json(
        root / "stage2/pipeline_status.json",
        {"status": "completed", "accepted": True},
    )
    _write_json(
        root / "stage2/data/dataset_manifest.json",
        {
            "status": "completed",
            "episodes": 4,
            "schedule_sha256": pipeline.json_sha256(pipeline._stage2_schedule(config)),
            "dataset_sha256": pipeline.sha256(dataset),
            "stage1_checkpoint_sha256": pipeline.sha256(stage1),
            "execution_semantics": config["execution_semantics"],
            "supported_roster": config["supported_roster"],
        },
    )
    _write_json(
        root / "stage2/model/metrics.json",
        {
            "status": "completed",
            "acceptance": {"all_passed": True},
            "dataset_sha256": pipeline.sha256(dataset),
            "checkpoint_sha256": pipeline.sha256(checkpoint),
            "training_config_sha256": pipeline.json_sha256(config["training"]),
            "execution_semantics": config["execution_semantics"],
            "supported_roster": config["supported_roster"],
        },
    )
    return config_path, root


def test_reuse_stage2_requires_accepted_hash_consistent_evidence(tmp_path: Path) -> None:
    config, root = _evidence(tmp_path)
    result = pipeline.validate_reusable_stage2(config, root)
    assert all(result["checks"].values())

    (root / "stage2/model/dynamic_outcome_time_model.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checkpoint_metrics_hash"):
        pipeline.validate_reusable_stage2(config, root)


def test_previous_failure_is_preserved_once_without_traceback(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write_json(
        root / "pipeline_status.json",
        {"status": "failed", "error": "stage3 failed", "traceback": "large"},
    )
    _write_json(
        root / "stage3/pipeline_status.json",
        {"status": "failed", "error": "infeasible", "traceback": "large"},
    )
    target = pipeline.preserve_previous_failure(root)
    assert target == root / "diagnostics/stage3_failed_attempt.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["stage3"]["error"] == "infeasible"
    assert "traceback" not in target.read_text(encoding="utf-8")

    _write_json(root / "stage3/pipeline_status.json", {"status": "failed", "error": "new"})
    assert pipeline.preserve_previous_failure(root) == target
    assert json.loads(target.read_text(encoding="utf-8"))["stage3"]["error"] == "infeasible"
