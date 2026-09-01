import csv
import json
from pathlib import Path

from scripts.run_round01_remaining import SCALES, generate_figures, write_report


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _evaluation(win_rate: float):
    return {
        "episodes": 120,
        "win_rate": win_rate,
        "mean_payoff": 2.0 * win_rate - 1.0,
        "components": {
            opponent: {
                "per_scale_win_rate": {scale: win_rate for scale in SCALES}
            }
            for opponent in ("rush", "split_rush")
        },
    }


def _round_fixture(root: Path) -> None:
    stage1 = {
        "algorithm": "REFIL-QMIX-HAD-adaptation",
        "parameter_count": 302749,
        "seed": 20260831,
        "environment_steps": 1_000_123,
        "episodes": 41072,
        "learner_steps": 41048,
        "elapsed_seconds": 54000.0,
        "throughput_environment_steps_per_second": 18.5,
        "initial_validation": _evaluation(0.3),
        "best_validation": _evaluation(0.8),
        "heldout_evaluation": {"episodes": 1200, "win_rate": 0.81, "mean_payoff": 0.62},
        "best_environment_steps": 950000,
        "scale_episodes": {scale: 100 for scale in SCALES},
        "best_checkpoint_sha256": "a" * 64,
    }
    _write_json(root / "stage1" / "summary.json", stage1)
    curves = root / "stage1" / "training_curves.csv"
    curves.parent.mkdir(parents=True, exist_ok=True)
    with curves.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=(
                "environment_steps",
                "validation_win_rate",
                "train_win_rate_200",
                "loss",
            ),
        )
        writer.writeheader()
        writer.writerows(
            (
                {
                    "environment_steps": 0,
                    "validation_win_rate": 0.3,
                    "train_win_rate_200": "",
                    "loss": "",
                },
                {
                    "environment_steps": 1_000_000,
                    "validation_win_rate": 0.8,
                    "train_win_rate_200": 0.75,
                    "loss": 0.02,
                },
            )
        )
    evaluation = root / "evaluation" / "win_rate_by_scale.csv"
    evaluation.parent.mkdir(parents=True, exist_ok=True)
    with evaluation.open("w", encoding="utf-8", newline="") as target:
        fieldnames = (
            "scale",
            "win_rate_rush",
            "win_rate_split_rush",
            "combined_win_rate",
            "ci95_low",
            "ci95_high",
            "episodes",
        )
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        for index, scale in enumerate(SCALES):
            rate = 0.9 - 0.05 * index
            writer.writerow(
                {
                    "scale": scale,
                    "win_rate_rush": rate,
                    "win_rate_split_rush": rate - 0.02,
                    "combined_win_rate": rate - 0.01,
                    "ci95_low": rate - 0.08,
                    "ci95_high": min(1.0, rate + 0.05),
                    "episodes": 200,
                }
            )
    _write_json(
        root / "evaluation_unseen_scales" / "summary.json",
        {
            "scale_results": [
                {
                    "scale": scale,
                    "win_rate_rush": rate,
                    "win_rate_split_rush": rate - 0.02,
                    "combined_win_rate": rate - 0.01,
                    "ci95_low": rate - 0.08,
                    "ci95_high": min(1.0, rate + 0.05),
                    "episodes": 200,
                }
                for scale, rate in (("2v2", 0.51), ("3v3", 0.40), ("5v2", 0.82), ("5v3", 0.65), ("6v3", 0.72))
            ]
        },
    )
    split = {
        "train": {"episodes": 5040, "wins": 4000, "losses": 1040, "rows": 40000},
        "validation": {"episodes": 1080, "wins": 850, "losses": 230, "rows": 8500},
        "test": {"episodes": 1080, "wins": 840, "losses": 240, "rows": 8400},
    }
    _write_json(
        root / "stage2" / "data" / "dataset_summary.json",
        {"episodes": 7200, "rows": 56900, "state_dim": 85, "split_summary": split},
    )
    per_scale = {
        scale: {
            "actual_win_rate": 0.8 - 0.03 * index,
            "predicted_win_rate": 0.78 - 0.03 * index,
            "absolute_win_rate_error": 0.02,
            "auc": 0.75,
            "brier": 0.15,
        }
        for index, scale in enumerate(SCALES)
    }
    metrics = {
        "parameter_count": 15000,
        "best_epoch": 12,
        "epochs_run": 22,
        "reload_max_prediction_difference": 0.0,
        "dataset_sha256": "b" * 64,
        "test_metrics": {"accuracy": 0.8, "auc": 0.75, "brier": 0.15},
        "baselines": {"global_constant_brier": 0.2, "scale_only_brier": 0.18},
        "mean_per_scale_absolute_win_rate_error": 0.02,
        "per_scale": per_scale,
        "calibration_bins": [
            {"mean_prediction": 0.2, "observed_win_rate": 0.25, "weighted_rows": 20},
            {"mean_prediction": 0.6, "observed_win_rate": 0.58, "weighted_rows": 40},
            {"mean_prediction": 0.9, "observed_win_rate": 0.88, "weighted_rows": 60},
        ],
        "acceptance": {
            "auc_at_least_0_60": True,
            "brier_better_than_global_constant": True,
            "mean_scale_error_at_most_0_15": True,
            "all_passed": True,
        },
    }
    _write_json(root / "stage2" / "model" / "metrics.json", metrics)


def test_round01_figures_and_report_are_compact_and_complete(tmp_path):
    _round_fixture(tmp_path)
    figures = generate_figures(tmp_path)
    report = write_report(tmp_path, figures)

    assert len(figures) == 4
    assert all(path.is_file() and path.stat().st_size > 1000 for path in figures)
    text = report.read_text(encoding="utf-8")
    assert "第一轮结论：**通过**" in text
    assert "Stage 1：**通过**" in text
    assert "Stage 2：**通过**" in text
    assert "4v3" in text
    assert "训练外规模的冻结策略压力测试" in text
    assert "6v3" in text
    assert text.count("![") == 4
