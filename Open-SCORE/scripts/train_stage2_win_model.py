"""Train the round-01 one-network dynamic-scale HAD win evaluator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from open_score.provenance import collect_and_require_git_provenance
from open_score.stage2.win_model import DynamicHADWinNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--entity-hidden-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            state = np.asarray(row["global_state"], dtype=np.float32)
            if state.shape != (85,) or not np.isfinite(state).all():
                raise ValueError(f"invalid state at dataset line {line_number}")
            if row["split"] not in {"train", "validation", "test"}:
                raise ValueError(f"invalid split at line {line_number}")
            rows.append(row)
    if not rows:
        raise ValueError("dataset is empty")
    episode_split = {}
    for row in rows:
        previous = episode_split.setdefault(row["episode_id"], row["split"])
        if previous != row["split"]:
            raise ValueError("one episode appears in multiple data splits")
    return rows


def _arrays(rows: Sequence[Mapping[str, object]], split: str):
    selected = [row for row in rows if row["split"] == split]
    if not selected:
        raise ValueError(f"{split} split is empty")
    state = np.asarray([row["global_state"] for row in selected], dtype=np.float32)
    label = np.asarray([row["red_win"] for row in selected], dtype=np.float32)
    weight = np.asarray([row["sample_weight"] for row in selected], dtype=np.float32)
    if set(np.unique(label)) != {0.0, 1.0}:
        raise ValueError(f"{split} must contain both Red wins and losses")
    return selected, state, label, weight


def _weighted_mean(values: np.ndarray, weight: np.ndarray) -> float:
    return float(np.sum(values * weight) / np.sum(weight))


def _weighted_auc(label: np.ndarray, probability: np.ndarray, weight: np.ndarray) -> float:
    positive = float(weight[label == 1].sum())
    negative = float(weight[label == 0].sum())
    if positive <= 0.0 or negative <= 0.0:
        return float("nan")
    order = np.argsort(probability, kind="mergesort")
    label = label[order]
    probability = probability[order]
    weight = weight[order]
    cumulative_negative = 0.0
    concordance = 0.0
    index = 0
    while index < len(label):
        stop = index + 1
        while stop < len(label) and probability[stop] == probability[index]:
            stop += 1
        group_positive = float(weight[index:stop][label[index:stop] == 1].sum())
        group_negative = float(weight[index:stop][label[index:stop] == 0].sum())
        concordance += group_positive * (cumulative_negative + 0.5 * group_negative)
        cumulative_negative += group_negative
        index = stop
    return concordance / (positive * negative)


def _metrics(label: np.ndarray, probability: np.ndarray, weight: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": _weighted_mean(((probability >= 0.5) == label).astype(np.float32), weight),
        "auc": _weighted_auc(label, probability, weight),
        "brier": _weighted_mean((probability - label) ** 2, weight),
        "actual_win_rate": _weighted_mean(label, weight),
        "predicted_win_rate": _weighted_mean(probability, weight),
    }


@torch.no_grad()
def _predict(model, state: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    result = []
    for start in range(0, len(state), batch_size):
        batch = torch.as_tensor(state[start : start + batch_size], device=device)
        result.append(model.predict_probability(batch).cpu().numpy())
    return np.concatenate(result)


def _calibration(label, probability, weight):
    rows = []
    edges = np.linspace(0.0, 1.0, 11)
    for index in range(10):
        mask = (probability >= edges[index]) & (
            probability <= edges[index + 1] if index == 9 else probability < edges[index + 1]
        )
        if not mask.any():
            continue
        rows.append(
            {
                "probability_low": float(edges[index]),
                "probability_high": float(edges[index + 1]),
                "weighted_rows": float(weight[mask].sum()),
                "mean_prediction": _weighted_mean(probability[mask], weight[mask]),
                "observed_win_rate": _weighted_mean(label[mask], weight[mask]),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    provenance = collect_and_require_git_provenance(PROJECT, formal=True)
    dataset = args.dataset.resolve()
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    device = _device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rows = _load(dataset)
    train_rows, train_state, train_label, train_weight = _arrays(rows, "train")
    validation_rows, validation_state, validation_label, validation_weight = _arrays(rows, "validation")
    test_rows, test_state, test_label, test_weight = _arrays(rows, "test")
    model = DynamicHADWinNet(args.entity_hidden_dim, args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset = TensorDataset(
        torch.as_tensor(train_state),
        torch.as_tensor(train_label),
        torch.as_tensor(train_weight),
    )
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    best_state = None
    best_brier = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        numerator = 0.0
        denominator = 0.0
        for state, label, weight in loader:
            state = state.to(device)
            label = label.to(device)
            weight = weight.to(device)
            loss_row = F.binary_cross_entropy_with_logits(
                model(state), label, reduction="none"
            )
            loss = (loss_row * weight).sum() / weight.sum().clamp_min(1e-8)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            numerator += float((loss_row.detach() * weight).sum().cpu())
            denominator += float(weight.sum().cpu())
        validation_probability = _predict(
            model, validation_state, device, args.batch_size
        )
        validation_brier = _weighted_mean(
            (validation_probability - validation_label) ** 2,
            validation_weight,
        )
        history.append(
            {
                "epoch": epoch,
                "train_bce": numerator / denominator,
                "validation_brier": validation_brier,
            }
        )
        if validation_brier < best_brier - 1e-8:
            best_brier = validation_brier
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        print(json.dumps({"event": "epoch", **history[-1], "best_epoch": best_epoch}), flush=True)
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("validation never produced a finite model")
    model.load_state_dict(best_state, strict=True)
    # Test is opened only once, after architecture/epoch selection is complete.
    test_probability = _predict(model, test_state, device, args.batch_size)
    test_metrics = _metrics(test_label, test_probability, test_weight)
    train_rate = _weighted_mean(train_label, train_weight)
    global_baseline = np.full_like(test_probability, train_rate)
    scale_rates = {}
    for scale in sorted({row["scale"] for row in train_rows}):
        mask = np.asarray([row["scale"] == scale for row in train_rows])
        scale_rates[scale] = _weighted_mean(train_label[mask], train_weight[mask])
    scale_baseline = np.asarray([scale_rates[row["scale"]] for row in test_rows])
    per_scale = {}
    for scale in sorted(scale_rates):
        mask = np.asarray([row["scale"] == scale for row in test_rows])
        per_scale[scale] = _metrics(
            test_label[mask], test_probability[mask], test_weight[mask]
        )
        per_scale[scale]["absolute_win_rate_error"] = abs(
            per_scale[scale]["predicted_win_rate"]
            - per_scale[scale]["actual_win_rate"]
        )
    mean_scale_error = float(
        np.mean([value["absolute_win_rate_error"] for value in per_scale.values()])
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "dynamic_win_model.pt"
    torch.save(
        {
            "model": best_state,
            "entity_hidden_dim": args.entity_hidden_dim,
            "hidden_dim": args.hidden_dim,
            "state_dim": 85,
            "best_epoch": best_epoch,
            "dataset_sha256": _sha256(dataset),
            "git_provenance": provenance,
        },
        checkpoint_path,
    )
    restored_payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    restored = DynamicHADWinNet(
        restored_payload["entity_hidden_dim"], restored_payload["hidden_dim"]
    ).to(device)
    restored.load_state_dict(restored_payload["model"], strict=True)
    repeated = _predict(restored, test_state, device, args.batch_size)
    reload_max_difference = float(np.max(np.abs(repeated - test_probability)))
    if reload_max_difference > 1e-7:
        raise RuntimeError("reloaded Stage-2 model changes its predictions")
    prediction_path = output_dir / "test_predictions.csv"
    with prediction_path.open("w", encoding="utf-8", newline="") as target:
        fieldnames = ["episode_id", "scale", "opponent", "step", "red_win", "sample_weight", "predicted_win_probability"]
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        for row, probability in zip(test_rows, test_probability):
            writer.writerow(
                {
                    **{name: row[name] for name in fieldnames[:-1]},
                    "predicted_win_probability": float(probability),
                }
            )
    payload = {
        "schema_version": "round-01-stage2-dynamic-win-v1",
        "status": "completed",
        "dataset": str(dataset),
        "dataset_sha256": _sha256(dataset),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "seed": args.seed,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "history": history,
        "test_metrics": test_metrics,
        "baselines": {
            "global_train_win_rate": train_rate,
            "global_constant_brier": _weighted_mean((global_baseline - test_label) ** 2, test_weight),
            "train_win_rate_by_scale": scale_rates,
            "scale_only_brier": _weighted_mean((scale_baseline - test_label) ** 2, test_weight),
        },
        "per_scale": per_scale,
        "mean_per_scale_absolute_win_rate_error": mean_scale_error,
        "calibration_bins": _calibration(test_label, test_probability, test_weight),
        "reload_max_prediction_difference": reload_max_difference,
        "acceptance": {
            "auc_at_least_0_60": bool(test_metrics["auc"] >= 0.60),
            "brier_better_than_global_constant": bool(
                test_metrics["brier"]
                < _weighted_mean((global_baseline - test_label) ** 2, test_weight)
            ),
            "mean_scale_error_at_most_0_15": bool(mean_scale_error <= 0.15),
        },
        "split_rows": {
            "train": len(train_rows),
            "validation": len(validation_rows),
            "test": len(test_rows),
        },
        "split_episodes": {
            split: len({row["episode_id"] for row in rows if row["split"] == split})
            for split in ("train", "validation", "test")
        },
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "git_provenance": provenance,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    payload["acceptance"]["all_passed"] = all(payload["acceptance"].values())
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
