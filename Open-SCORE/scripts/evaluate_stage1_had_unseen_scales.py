"""Evaluate the frozen round-01 REFIL policy on unregistered HAD rosters.

This is a post-hoc, out-of-distribution stress test.  It never trains or
fine-tunes the policy and does not alter the frozen round-01 acceptance result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from open_score.provenance import collect_git_provenance
from open_score.stage1 import (
    BatchedHADRedRunner,
    HADStage1Factory,
    QMixController,
    make_had_qmix,
)
from open_score.stage1.refil_protocol import evaluate_refil


Scale = Tuple[int, int]
ROUND01_TRAIN_SCALES = {
    (2, 1),
    (3, 1),
    (3, 2),
    (4, 1),
    (4, 2),
    (4, 3),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--scales",
        nargs="+",
        default=("2v2", "3v3", "5v2", "5v3", "6v3"),
    )
    parser.add_argument(
        "--opponents",
        nargs="+",
        choices=("rush", "split_rush"),
        default=("rush", "split_rush"),
    )
    parser.add_argument("--episodes-per-cell", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=48_260_831)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT / "outputs" / "round_01_mvp" / "evaluation_unseen_scales",
    )
    return parser.parse_args()


def parse_scale(label: str) -> Scale:
    try:
        red_text, blue_text = label.lower().split("v", maxsplit=1)
        scale = int(red_text), int(blue_text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid HAD scale: {label!r}") from exc
    if scale[0] < 1 or scale[1] < 1:
        raise ValueError("each unseen HAD scale needs at least one agent per side")
    return scale


def choose_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilson(successes: int, total: int, z: float = 1.959963984540054):
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * np.sqrt(
            probability * (1.0 - probability) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return float(center - radius), float(center + radius)


def build_scale_rows(
    records: Sequence[Mapping[str, object]],
    scales: Sequence[Scale],
    opponents: Sequence[str],
):
    rows = []
    for red, blue in scales:
        label = f"{red}v{blue}"
        combined = [row for row in records if row["scale"] == label]
        wins = sum(bool(row["red_win"]) for row in combined)
        low, high = wilson(wins, len(combined))
        row = {"scale": label}
        for opponent in opponents:
            cell = [
                value
                for value in combined
                if value["opponent"] == opponent
            ]
            row[f"win_rate_{opponent}"] = float(
                np.mean([value["red_win"] for value in cell])
            )
        row.update(
            {
                "combined_win_rate": wins / len(combined),
                "ci95_low": low,
                "ci95_high": high,
                "episodes": len(combined),
            }
        )
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    if args.episodes_per_cell < 1 or args.max_steps < 1 or args.batch_size < 1:
        raise ValueError("episode count, max steps and batch size must be positive")
    scales = tuple(parse_scale(label) for label in args.scales)
    if len(scales) != len(set(scales)):
        raise ValueError("unseen scale list contains duplicates")
    overlap = sorted(set(scales) & ROUND01_TRAIN_SCALES)
    if overlap:
        raise ValueError(f"OOD evaluation contains trained scales: {overlap}")

    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    device = choose_device(args.device)
    model = make_had_qmix(
        device,
        agent_hidden_dim=64,
        mixer_hidden_dim=128,
        mixing_dim=32,
        encoder_kind="refil",
        attention_heads=4,
        attention_embed_dim=128,
        hypernet_hidden_dim=128,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    extra = checkpoint.get("extra", {})
    if extra.get("architecture") != "REFIL-QMIX-HAD-v1":
        raise ValueError("checkpoint is not the frozen round-01 REFIL architecture")
    model.load_state_dict(checkpoint["online"], strict=True)
    model.eval()

    controller = QMixController(
        model, device, epsilon=0.0, name="refil_qmix_round01_best_frozen"
    )
    runner = BatchedHADRedRunner(
        HADStage1Factory(
            max_steps=args.max_steps,
            shaping_scale=0.5,
            allow_unregistered_roster=True,
        )
    )
    evaluation, records = evaluate_refil(
        runner,
        controller,
        scales,
        tuple(args.opponents),
        args.episodes_per_cell,
        args.seed_base,
        args.batch_size,
    )
    scale_rows = build_scale_rows(records, scales, tuple(args.opponents))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "win_rate_by_scale.csv").open(
        "w", encoding="utf-8", newline=""
    ) as target:
        writer = csv.DictWriter(target, fieldnames=list(scale_rows[0]))
        writer.writeheader()
        writer.writerows(scale_rows)
    with (output_dir / "episodes.csv").open(
        "w", encoding="utf-8", newline=""
    ) as target:
        writer = csv.DictWriter(target, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    payload = {
        "schema_version": "round-01-posthoc-unseen-scale-evaluation-v1",
        "status": "completed",
        "claim_scope": "post-hoc OOD stress test; no training or fine-tuning",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "trained_scales": [list(scale) for scale in sorted(ROUND01_TRAIN_SCALES)],
        "tested_unseen_scales": [list(scale) for scale in scales],
        "opponents": list(args.opponents),
        "scale_results": scale_rows,
        "evaluation": evaluation,
        "git_provenance": collect_git_provenance(PROJECT),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "event": "unseen_scale_evaluation_completed",
                "episodes": evaluation["episodes"],
                "combined_win_rate": evaluation["win_rate"],
                "results": scale_rows,
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
