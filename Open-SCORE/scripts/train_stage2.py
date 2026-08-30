"""Collect/load, train, calibrate, and evaluate the Stage-2 local game model."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from open_score.stage2 import (  # noqa: E402
    Stage2System,
    Stage2TrainingConfig,
    collect_had_records,
    evaluate_stage2_predictions,
    fit_stage2_system,
    generate_synthetic_records,
    metric_definitions_zh,
    read_records_jsonl,
    split_records_by_lineage_group,
    write_records_csv,
    write_records_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT / "configs" / "stage2_had_smoke.yaml",
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


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT.resolve()))
    except ValueError:
        return str(path.resolve())


def _load_records(config: Dict[str, object]):
    data = dict(config["data"])
    source = str(data.pop("source"))
    if source == "jsonl":
        path = Path(data["path"])
        if not path.is_absolute():
            path = PROJECT / path
        records = read_records_jsonl(path)
        evidence = "external_rollout_dataset"
    elif source == "had":
        data["scales"] = [tuple(scale) for scale in data["scales"]]
        records = collect_had_records(**data)
        source = "had_forward_rollout"
        evidence = "HAD_rule_policy_forward_rollout_pilot_not_convergence_evidence"
    elif source == "synthetic":
        if "scales" in data:
            data["scales"] = [tuple(scale) for scale in data["scales"]]
        records = generate_synthetic_records(**data)
        evidence = "synthetic_pipeline_test_only_not_environment_evidence"
    else:
        raise ValueError("data.source must be one of: jsonl, had, synthetic")
    return source, evidence, records


def _prediction_rows(records, prediction):
    rows = []
    for index, record in enumerate(records):
        rows.append(
            {
                "root_id": record.query.root_id,
                "rollout_id": record.query.rollout_id,
                "candidate_id": record.query.candidate_id,
                "scale": [record.query.defender_count, record.query.attacker_count],
                "defender_policy_id": record.query.defender_policy_id,
                "attacker_policy_id": record.query.attacker_policy_id,
                "true_outcome": record.outcome,
                "true_terminal_steps": record.terminal_steps,
                "predicted_breach_probability": float(prediction.breach_probability[index]),
                "predicted_short_window_breach_probability": float(
                    prediction.short_window_breach_probability[index]
                ),
                "predicted_defender_success_probability": float(
                    prediction.defender_success_probability[index]
                ),
                "predicted_remaining_steps": float(
                    prediction.expected_remaining_steps[index]
                ),
                "breach_upper_marginal": float(
                    prediction.breach_upper_marginal[index]
                ),
                "breach_upper_selection_safe": float(
                    prediction.breach_upper_selection_safe[index]
                ),
            }
        )
    return rows


def _run_summary(payload: Dict[str, object]) -> str:
    metrics = payload["test_metrics"]
    breach = metrics["breach"]
    early = metrics["short_window_breach"]
    time_metrics = metrics["remaining_time"]
    risk = metrics["calibrated_risk_bound"]
    decision = metrics["candidate_decision"]
    return f"""# Stage 2 单次运行摘要

> 证据级别：`{payload['evidence_level']}`。本文件自动生成，不代表正式收敛或论文结论。

- 数据源：`{payload['data_source']}`；测试 root 数：{metrics['root_count']}；测试记录数：{metrics['sample_count']}。
- 失守概率：Brier={breach['brier']:.4f}，NLL={breach['nll']:.4f}，ECE={breach['ece']:.4f}，AUC={breach['roc_auc']}。
- 短窗失守：Brier={early['brier']:.4f}，AUC={early['roc_auc']}。
- 剩余时长：MAE={time_metrics['time_mae_steps']:.2f} 步，RMSE={time_metrics['time_rmse_steps']:.2f} 步。
- 校准上界：候选覆盖率={risk['candidate_rate_coverage']:.3f}，平均宽度={risk['mean_interval_width']:.3f}。
- 选择后审计：coverage={decision['post_selection_coverage']:.3f}，平均 decision regret={decision['mean_decision_regret']:.3f}。

完整定义、分规模/策略对结果和运行元数据见 `metrics.json`。
"""


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Stage-2 YAML must contain a mapping")
    requested_device = args.device or str(config.get("device", "auto"))
    device = _device(requested_device)
    output_dir = args.output_dir or Path(config.get("output_dir", "outputs/stage2"))
    if not output_dir.is_absolute():
        output_dir = PROJECT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 7))
    np.random.seed(seed)
    torch.manual_seed(seed)
    started = time.perf_counter()

    source, evidence, records = _load_records(config)
    write_records_jsonl(records, output_dir / "dataset.jsonl")
    write_records_csv(records, output_dir / "dataset.csv")
    published_dataset = None
    if config.get("published_dataset_path"):
        published_dataset = Path(config["published_dataset_path"])
        if not published_dataset.is_absolute():
            published_dataset = PROJECT / published_dataset
        write_records_jsonl(records, published_dataset)
    split_config = dict(config.get("split", {}))
    split = split_records_by_lineage_group(records, seed=seed, **split_config)
    split.assert_no_lineage_leakage()
    training_values = dict(config.get("training", {}))
    training_values.setdefault("seed", seed)
    training_config = Stage2TrainingConfig.from_dict(training_values)
    fit = fit_stage2_system(
        split.train,
        split.temperature_calibration,
        split.risk_calibration,
        training_config,
        device=device,
    )
    prediction = fit.system.predict(split.test, device=device)
    train_mean_steps = float(np.mean([record.terminal_steps for record in split.train]))
    seen_pairs = {record.query.policy_pair for record in split.train}
    metrics = evaluate_stage2_predictions(
        split.test,
        prediction.probabilities,
        horizon_bins=training_config.horizon_bins,
        steps_per_bin=training_config.steps_per_bin,
        breach_upper=prediction.breach_upper_selection_safe,
        member_probabilities=prediction.member_probabilities,
        train_mean_terminal_steps=train_mean_steps,
        seen_policy_pairs=seen_pairs,
    )
    checkpoint = output_dir / "stage2_system.pt"
    fit.system.metadata.update(
        {
            "data_source": source,
            "evidence_level": evidence,
            "split_manifest": split.manifest(),
        }
    )
    fit.system.save(checkpoint)
    payload = _jsonable(
        {
            "status": "completed",
            "convergence_claim": False,
            "environment_performance_claim": False,
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
            "dataset": {
                "records": len(records),
                "roots": len({record.query.root_id for record in records}),
                "lineage_groups": len(
                    {record.query.lineage_group_id for record in records}
                ),
                "environments": sorted({record.query.environment_id for record in records}),
                "scenarios": sorted({record.query.scenario_id for record in records}),
                "scales": sorted(
                    {
                        f"{record.query.defender_count}v{record.query.attacker_count}"
                        for record in records
                    }
                ),
                "strict_defender_count_greater_than_attacker_count": all(
                    record.query.defender_count > record.query.attacker_count
                    for record in records
                ),
                "defender_policy_versions": sorted(
                    {
                        f"{record.query.defender_policy_id}@{record.query.defender_policy_version}"
                        for record in records
                    }
                ),
                "attacker_policy_versions": sorted(
                    {
                        f"{record.query.attacker_policy_id}@{record.query.attacker_policy_version}"
                        for record in records
                    }
                ),
            },
            "split_manifest": split.manifest(),
            "training_config": training_values,
            "training_summary": fit.training_summary,
            "metric_definitions_zh": metric_definitions_zh(),
            "test_metrics": metrics,
            "artifacts": {
                "checkpoint": _display_path(checkpoint),
                "dataset_jsonl": _display_path(output_dir / "dataset.jsonl"),
                "dataset_csv": _display_path(output_dir / "dataset.csv"),
                "predictions_jsonl": _display_path(output_dir / "test_predictions.jsonl"),
                "published_dataset_jsonl": (
                    _display_path(published_dataset) if published_dataset is not None else None
                ),
            },
        }
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    with (output_dir / "test_predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in _prediction_rows(split.test, prediction):
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    (output_dir / "run_summary.md").write_text(_run_summary(payload), encoding="utf-8")
    if config.get("published_metrics_path"):
        published = Path(config["published_metrics_path"])
        if not published.is_absolute():
            published = PROJECT / published
        published.parent.mkdir(parents=True, exist_ok=True)
        published.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "data_source": payload["data_source"],
                "evidence_level": payload["evidence_level"],
                "records": payload["dataset"]["records"],
                "roots": payload["dataset"]["roots"],
                "lineage_group_leakage_check": payload["split_manifest"][
                    "lineage_group_leakage_check"
                ],
                "elapsed_seconds": payload["elapsed_seconds"],
                "metrics": _display_path(output_dir / "metrics.json"),
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
