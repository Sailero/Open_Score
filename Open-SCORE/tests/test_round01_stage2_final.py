import hashlib
import json
from pathlib import Path

from scripts.run_round01_stage2 import _completed_pipeline_valid


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_completed_round01_pipeline_checks_dataset_and_model_hashes(tmp_path):
    output = tmp_path / "stage2" / "final"
    dataset = output / "data" / "dynamic_outcome_time.jsonl"
    checkpoint = output / "model" / "dynamic_outcome_time_model.pt"
    report = tmp_path / "round_01_report.md"
    dataset.parent.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    dataset.write_text('{"episode_id":"example"}\n', encoding="utf-8")
    checkpoint.write_bytes(b"round-01-model")
    report.write_text("# Round 01\n", encoding="utf-8")
    _write_json(
        output / "data" / "dataset_manifest.json",
        {"status": "completed", "dataset_sha256": _sha256(dataset)},
    )
    _write_json(
        output / "model" / "metrics.json",
        {
            "status": "completed",
            "dataset_sha256": _sha256(dataset),
            "checkpoint_sha256": _sha256(checkpoint),
        },
    )
    status = {"status": "completed", "report": str(report)}

    assert _completed_pipeline_valid(output, status, report)

    dataset.write_text("changed\n", encoding="utf-8")
    assert not _completed_pipeline_valid(output, status, report)
