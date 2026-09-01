"""Create a hash index for the complete round-01 experiment artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
ROUND_ROOT = PROJECT / "outputs" / "round_01_mvp"

OFFICIAL_ARTIFACTS = (
    "round_01_report.md",
    "stage1/summary.json",
    "stage1/training_curves.csv",
    "stage1/checkpoints/refil_qmix_seed20260831_best.pt",
    "evaluation/summary.json",
    "evaluation/win_rate_by_scale.csv",
    "evaluation_unseen_scales/summary.json",
    "evaluation_unseen_scales/win_rate_by_scale.csv",
    "evaluation_unseen_scales/episodes.csv",
    "figures/01_stage1_training_curve.png",
    "figures/02_stage1_scale_win_rates.png",
    "stage2/final/data/dynamic_outcome_time.jsonl",
    "stage2/final/data/dataset_manifest.json",
    "stage2/final/model/dynamic_outcome_time_model.pt",
    "stage2/final/model/metrics.json",
    "stage2/final/model/training_history.csv",
    "stage2/final/model/evaluation_predictions.csv",
    "stage2/final/figures/01_training_curves.png",
    "stage2/final/figures/02_all_scale_win_rates.png",
    "stage2/final/figures/03_all_scale_time_mae.png",
    "stage2/final/figures/04_core_calibration.png",
    "stage2/final/pipeline_status.json",
    "stage2/final/live_progress.log",
)

PRELIMINARY_ARTIFACTS = (
    "stage2/preliminary/data/win_dataset.jsonl",
    "stage2/preliminary/data/dataset_summary.json",
    "stage2/preliminary/model/dynamic_win_model.pt",
    "stage2/preliminary/model/metrics.json",
    "stage2/preliminary/model/test_predictions.csv",
    "stage2/preliminary/figures/03_stage2_scale_predictions.png",
    "stage2/preliminary/figures/04_stage2_calibration.png",
    "stage2/preliminary/legacy_pipeline_status.json",
    "stage2/preliminary/legacy_pipeline.log",
    "stage2/preliminary/legacy_round_01_report.pdf",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe(relative_path: str) -> dict[str, object]:
    path = ROUND_ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"round-01 artifact is missing: {path}")
    return {
        "path": relative_path.replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    index = {
        "schema_version": "round-01-results-index-v1",
        "round": 1,
        "status": "completed",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(
            timespec="seconds"
        ),
        "official_report": "round_01_report.md",
        "official_stage2_root": "stage2/final",
        "preliminary_stage2_root": "stage2/preliminary",
        "note": (
            "The preliminary win-only evaluator is retained for audit only. "
            "All round-01 conclusions use the official outcome-time model."
        ),
        "official_artifacts": [describe(path) for path in OFFICIAL_ARTIFACTS],
        "preliminary_artifacts": [
            describe(path)
            for path in PRELIMINARY_ARTIFACTS
            if (ROUND_ROOT / path).is_file()
        ],
    }
    output = ROUND_ROOT / "results_index.json"
    temporary = output.with_suffix(".json.partial")
    temporary.write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    print(f"Round-01 results index: {output}")


if __name__ == "__main__":
    main()
