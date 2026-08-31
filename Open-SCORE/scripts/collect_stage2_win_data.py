"""Collect the round-01 dynamic-scale HAD state-to-final-win dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from open_score.provenance import collect_and_require_git_provenance
from open_score.stage1 import (
    BatchedHADRedRunner,
    HADStage1Factory,
    QMixController,
    RuleBasedController,
    make_had_qmix,
)
from open_score.stage2.canonical import HADCanonicalizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument(
        "--scales", nargs="+", default=("2v1", "3v1", "3v2", "4v1", "4v2", "4v3")
    )
    parser.add_argument(
        "--opponents", nargs="+", choices=("rush", "split_rush"), default=("rush", "split_rush")
    )
    parser.add_argument("--episodes-per-cell", type=int, default=600)
    parser.add_argument(
        "--snapshot-steps", nargs="+", type=int, default=(0, 5, 10, 15, 20, 25, 30, 35, 40, 45)
    )
    parser.add_argument("--seed-base", type=int, default=31_000_000)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def _parse_scale(label: str):
    red, blue = label.lower().split("v")
    scale = int(red), int(blue)
    if scale not in ((2, 1), (3, 1), (3, 2), (4, 1), (4, 2), (4, 3)):
        raise ValueError(f"unregistered HAD scale: {label}")
    return scale


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    provenance = collect_and_require_git_provenance(PROJECT, formal=True)
    if args.episodes_per_cell != 600:
        raise ValueError("round-01 formal collection requires 600 episodes per cell")
    if tuple(args.snapshot_steps) != (0, 5, 10, 15, 20, 25, 30, 35, 40, 45):
        raise ValueError("snapshot steps differ from the frozen round-01 protocol")
    scales = tuple(_parse_scale(label) for label in args.scales)
    if len(scales) != 6 or len(set(scales)) != 6:
        raise ValueError("formal collection requires all six unique HAD scales")
    if set(args.opponents) != {"rush", "split_rush"}:
        raise ValueError("formal collection requires rush and split_rush")
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    device = _device(args.device)
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
    controller = QMixController(model, device, epsilon=0.0, name="refil_qmix_best")
    runner = BatchedHADRedRunner(
        HADStage1Factory(max_steps=args.max_steps, shaping_scale=0.5)
    )
    canonicalizer = HADCanonicalizer()
    schedule = []
    for scale_index, scale in enumerate(scales):
        for opponent_index, opponent in enumerate(args.opponents):
            for episode_index in range(args.episodes_per_cell):
                seed = (
                    args.seed_base
                    + scale_index * 100_000
                    + opponent_index * 10_000
                    + episode_index
                )
                split = (
                    "train"
                    if episode_index < 420
                    else ("validation" if episode_index < 510 else "test")
                )
                schedule.append((scale, opponent, episode_index, seed, split))
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    cell_summary = {
        f"{scale[0]}v{scale[1]}|{opponent}": {"episodes": 0, "wins": 0, "rows": 0}
        for scale in scales
        for opponent in args.opponents
    }
    split_summary = {
        split: {"episodes": 0, "wins": 0, "losses": 0, "rows": 0}
        for split in ("train", "validation", "test")
    }
    started = time.perf_counter()
    total_rows = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for start in range(0, len(schedule), args.num_envs):
            cells = schedule[start : start + args.num_envs]
            episodes = runner.run_batch(
                [cell[0] for cell in cells],
                controller,
                [RuleBasedController(cell[1]) for cell in cells],
                [cell[3] for cell in cells],
            )
            for cell, episode in zip(cells, episodes):
                scale, opponent, episode_index, seed, split = cell
                selected_steps = [step for step in args.snapshot_steps if step < episode.length]
                if not selected_steps:
                    raise RuntimeError("episode has no valid Stage-2 snapshot")
                weight = 1.0 / len(selected_steps)
                episode_id = f"{scale[0]}v{scale[1]}-{opponent}-{episode_index:03d}"
                red_win = int(episode.outcome_red > 0.0)
                for step in selected_steps:
                    observation = episode.red.observations[step]
                    row = {
                        "episode_id": episode_id,
                        "split": split,
                        "scale": f"{scale[0]}v{scale[1]}",
                        "opponent": opponent,
                        "step": step,
                        "episode_seed": seed,
                        "global_state": canonicalizer.from_observation(observation).tolist(),
                        "red_win": red_win,
                        "sample_weight": weight,
                    }
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                    handle.write("\n")
                key = f"{scale[0]}v{scale[1]}|{opponent}"
                cell_summary[key]["episodes"] += 1
                cell_summary[key]["wins"] += red_win
                cell_summary[key]["rows"] += len(selected_steps)
                split_summary[split]["episodes"] += 1
                split_summary[split]["wins"] += red_win
                split_summary[split]["losses"] += 1 - red_win
                split_summary[split]["rows"] += len(selected_steps)
                total_rows += len(selected_steps)
            completed = start + len(cells)
            if completed % 600 == 0 or completed == len(schedule):
                print(
                    json.dumps(
                        {
                            "event": "collection_progress",
                            "episodes": completed,
                            "rows": total_rows,
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
    temporary.replace(output)
    for value in cell_summary.values():
        value["win_rate"] = value["wins"] / value["episodes"]
    summary = {
        "schema_version": "round-01-stage2-win-data-v1",
        "status": "completed",
        "episodes": len(schedule),
        "rows": total_rows,
        "state_dim": canonicalizer.state_dim,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "dataset": str(output),
        "dataset_sha256": _sha256(output),
        "elapsed_seconds": time.perf_counter() - started,
        "cell_summary": cell_summary,
        "split_summary": split_summary,
        "git_provenance": provenance,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (output.parent / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
