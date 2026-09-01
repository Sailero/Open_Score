"""Complete round-01 Stage 2 with dynamic HAD outcome-time evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from open_score.stage1 import (  # noqa: E402
    BatchedHADRedRunner,
    HADStage1Factory,
    QMixController,
    RuleBasedController,
    make_had_qmix,
)
from open_score.stage2 import (  # noqa: E402
    DynamicHADOutcomeNet,
    HADCanonicalizer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT / "configs" / "stage2_round01_final.yaml",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run a tiny end-to-end validation in a separate output directory",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class LiveLog:
    def __init__(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        self.path = output_dir / "live_progress.log"

    def write(self, message: str) -> None:
        line = f"[{_now()}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")


def _resolve(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _git_provenance() -> Dict[str, object]:
    git = r"D:\Software\Git\cmd\git.exe"
    try:
        commit = subprocess.run(
            [git, "rev-parse", "HEAD"],
            cwd=PROJECT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            [git, "status", "--porcelain"],
            cwd=PROJECT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        return {"commit": commit, "dirty": bool(status), "status": status}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "status": []}


def _choose_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in torch310")
    return torch.device(name)


def _load_config(args: argparse.Namespace) -> Dict[str, object]:
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Round-01 Stage-2 config must be a mapping")
    if args.device is not None:
        config["device"] = args.device
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    if args.smoke:
        config = deepcopy(config)
        collection = config["collection"]
        training = config["training"]
        collection["core_scales"] = collection["core_scales"][:2]
        collection["sparse_scales"] = collection["sparse_scales"][:1]
        collection["heldout_scales"] = collection["heldout_scales"][:1]
        collection["core_episodes_per_cell"] = 4
        collection["sparse_episodes_per_cell"] = 4
        collection["heldout_episodes_per_cell"] = 4
        collection["batch_size"] = 4
        training["max_epochs"] = 3
        training["minimum_epochs"] = 1
        training["patience"] = 2
        training["batch_size"] = 32
        config["output_dir"] = "outputs/_smoke_round01_stage2"
    return config


def _validate_config(config: Mapping[str, object]) -> None:
    collection = config["collection"]
    training = config["training"]
    groups = {
        "core": collection["core_scales"],
        "sparse": collection["sparse_scales"],
        "heldout": collection["heldout_scales"],
    }
    flattened = [tuple(map(int, scale)) for values in groups.values() for scale in values]
    if len(flattened) != len(set(flattened)):
        raise ValueError("core, sparse and heldout scale lists must be disjoint")
    if any(red < 1 or blue < 1 for red, blue in flattened):
        raise ValueError("every scale needs at least one agent per side")
    steps_per_bin = int(training["steps_per_bin"])
    horizon_bins = int(training["horizon_bins"])
    if steps_per_bin * horizon_bins != int(collection["max_steps"]):
        raise ValueError("horizon_bins * steps_per_bin must equal max_steps")
    snapshots = tuple(map(int, collection["snapshot_steps"]))
    if not snapshots or snapshots[0] != 0 or tuple(sorted(set(snapshots))) != snapshots:
        raise ValueError("snapshot_steps must be sorted, unique and start at zero")
    if snapshots[-1] >= int(collection["max_steps"]):
        raise ValueError("snapshot_steps must be earlier than max_steps")
    split = collection["split"]
    train = float(split["train_fraction"])
    validation = float(split["validation_fraction"])
    if not 0.0 < train < 1.0 or not 0.0 < validation < 1.0 - train:
        raise ValueError("train/validation fractions leave no Test data")
    if float(collection["core_training_weight"]) <= float(
        collection["sparse_training_weight"]
    ):
        raise ValueError("core scales must have more training weight than sparse scales")


def _schedule(config: Mapping[str, object]) -> list[Dict[str, object]]:
    collection = config["collection"]
    seed_base = int(collection["seed_base"])
    train_fraction = float(collection["split"]["train_fraction"])
    validation_fraction = float(collection["split"]["validation_fraction"])
    specifications = (
        (
            "core",
            collection["core_scales"],
            int(collection["core_episodes_per_cell"]),
            float(collection["core_training_weight"]),
        ),
        (
            "sparse",
            collection["sparse_scales"],
            int(collection["sparse_episodes_per_cell"]),
            float(collection["sparse_training_weight"]),
        ),
        (
            "heldout",
            collection["heldout_scales"],
            int(collection["heldout_episodes_per_cell"]),
            0.0,
        ),
    )
    result: list[Dict[str, object]] = []
    serial = 0
    for group, scales, episodes_per_cell, training_weight in specifications:
        if episodes_per_cell < 3:
            raise ValueError("every collection cell needs at least three episodes")
        train_stop = max(1, min(episodes_per_cell - 2, int(episodes_per_cell * train_fraction)))
        validation_count = max(
            1,
            min(
                episodes_per_cell - train_stop - 1,
                int(episodes_per_cell * validation_fraction),
            ),
        )
        validation_stop = train_stop + validation_count
        for scale in scales:
            red, blue = map(int, scale)
            for opponent in collection["opponents"]:
                for episode_index in range(episodes_per_cell):
                    if group == "heldout":
                        split = "generalization"
                    elif episode_index < train_stop:
                        split = "train"
                    elif episode_index < validation_stop:
                        split = "validation"
                    else:
                        split = "test"
                    result.append(
                        {
                            "serial": serial,
                            "episode_id": (
                                f"{group}-{red}v{blue}-{opponent}-{episode_index:04d}"
                            ),
                            "group": group,
                            "scale": [red, blue],
                            "scale_label": f"{red}v{blue}",
                            "opponent": str(opponent),
                            "episode_index": episode_index,
                            "seed": seed_base + serial,
                            "split": split,
                            "training_weight": training_weight,
                        }
                    )
                    serial += 1
    random.Random(int(config["seed"])).shuffle(result)
    return result


def _load_stage1_controller(
    checkpoint_path: Path, device: torch.device
) -> QMixController:
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
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload.get("extra", {}).get("architecture") != "REFIL-QMIX-HAD-v1":
        raise ValueError("Stage-1 checkpoint is not REFIL-QMIX-HAD-v1")
    model.load_state_dict(payload["online"], strict=True)
    model.eval()
    return QMixController(model, device, epsilon=0.0, name="round01_refil_frozen")


def _collect_dataset(
    config: Mapping[str, object],
    output_dir: Path,
    device: torch.device,
    log: LiveLog,
) -> tuple[Path, Dict[str, object]]:
    dataset = output_dir / "data" / "dynamic_outcome_time.jsonl"
    manifest_path = output_dir / "data" / "dataset_manifest.json"
    if dataset.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("dataset_sha256") != _sha256(dataset):
            raise ValueError("existing round-01 dataset hash differs from its manifest")
        log.write(
            f"[采集] 已有完整数据，跳过仿真：{manifest['episodes']:,}局，"
            f"{manifest['rows']:,}个态势"
        )
        return dataset, manifest

    collection = config["collection"]
    schedule = _schedule(config)
    schedule_sha = _json_sha(schedule)
    checkpoint = _resolve(config["stage1_checkpoint"]).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    controller = _load_stage1_controller(checkpoint, device)
    runner = BatchedHADRedRunner(
        HADStage1Factory(
            max_steps=int(collection["max_steps"]),
            shaping_scale=0.5,
            allow_unregistered_roster=True,
        )
    )
    canonicalizer = HADCanonicalizer()
    snapshot_steps = tuple(map(int, collection["snapshot_steps"]))
    batch_size = int(collection["batch_size"])
    partial = dataset.with_suffix(".partial.jsonl")
    state_path = dataset.with_name("collection_state.json")
    dataset.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    byte_offset = 0
    if partial.is_file() and state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("schedule_sha256") != schedule_sha:
            raise ValueError("partial collection belongs to a different schedule")
        completed = int(state["completed_schedule_items"])
        byte_offset = int(state["byte_offset"])
        with partial.open("r+b") as handle:
            handle.truncate(byte_offset)
        log.write(f"[采集] 从断点继续：{completed:,}/{len(schedule):,}局")
    else:
        partial.write_bytes(b"")

    resume_completed = completed
    started = time.perf_counter()
    recent_outcomes: list[int] = []
    total_rows = 0
    split_episodes: Counter[str] = Counter()
    group_episodes: Counter[str] = Counter()
    scale_cells: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"episodes": 0, "red_wins": 0, "rows": 0}
    )
    if completed:
        # Compact reconstruction is only needed when resuming after interruption.
        with partial.open("r", encoding="utf-8") as source:
            episode_seen = set()
            for line in source:
                row = json.loads(line)
                total_rows += 1
                scale_cells[row["scale"]]["rows"] += 1
                if row["episode_id"] not in episode_seen:
                    episode_seen.add(row["episode_id"])
                    split_episodes[row["split"]] += 1
                    group_episodes[row["scale_group"]] += 1
                    scale_cells[row["scale"]]["episodes"] += 1
                    scale_cells[row["scale"]]["red_wins"] += int(row["red_win"])

    with partial.open("ab") as target:
        for start in range(completed, len(schedule), batch_size):
            cells = schedule[start : start + batch_size]
            episodes = runner.run_batch(
                [tuple(cell["scale"]) for cell in cells],
                controller,
                [RuleBasedController(cell["opponent"]) for cell in cells],
                [int(cell["seed"]) for cell in cells],
            )
            payload = bytearray()
            for cell, episode in zip(cells, episodes):
                valid_steps = [step for step in snapshot_steps if step < episode.length]
                if not valid_steps:
                    raise RuntimeError("an episode produced no non-terminal snapshot")
                red_win = int(episode.outcome_red > 0.0)
                recent_outcomes.append(red_win)
                split_episodes[cell["split"]] += 1
                group_episodes[cell["group"]] += 1
                scale_summary = scale_cells[cell["scale_label"]]
                scale_summary["episodes"] += 1
                scale_summary["red_wins"] += red_win
                episode_training_mass = float(cell["training_weight"])
                for step in valid_steps:
                    encoded = canonicalizer.entity_set_from_observation(
                        episode.red.observations[step]
                    )
                    row = {
                        "episode_id": cell["episode_id"],
                        "split": cell["split"],
                        "scale_group": cell["group"],
                        "scale": cell["scale_label"],
                        "red_count": int(cell["scale"][0]),
                        "blue_count": int(cell["scale"][1]),
                        "opponent": cell["opponent"],
                        "episode_seed": int(cell["seed"]),
                        "step": int(step),
                        "terminal_step": int(episode.length),
                        "remaining_steps": int(episode.length - step),
                        "red_win": red_win,
                        "target": encoded.target.tolist(),
                        "red_entities": encoded.red_entities.tolist(),
                        "blue_entities": encoded.blue_entities.tolist(),
                        "context": encoded.context.tolist(),
                        "training_weight": episode_training_mass / len(valid_steps),
                        "evaluation_weight": 1.0 / len(valid_steps),
                    }
                    payload.extend(
                        json.dumps(
                            row, ensure_ascii=False, separators=(",", ":")
                        ).encode("utf-8")
                    )
                    payload.extend(b"\n")
                    total_rows += 1
                    scale_summary["rows"] += 1
            target.write(payload)
            target.flush()
            completed = min(start + len(cells), len(schedule))
            byte_offset = target.tell()
            _write_json(
                state_path,
                {
                    "schedule_sha256": schedule_sha,
                    "completed_schedule_items": completed,
                    "byte_offset": byte_offset,
                    "updated_at": _now(),
                },
            )
            if completed == len(schedule) or completed % max(batch_size * 10, 1) == 0:
                elapsed = time.perf_counter() - started
                processed_now = max(1, completed - resume_completed)
                rate = processed_now / max(elapsed, 1e-6)
                eta = (len(schedule) - completed) / max(rate, 1e-8)
                recent_rate = float(np.mean(recent_outcomes[-500:]))
                log.write(
                    f"[采集] {completed:,}/{len(schedule):,}局 "
                    f"({100.0 * completed / len(schedule):.1f}%) | "
                    f"{rate:.2f}局/秒 | 最近胜率{recent_rate:.1%} | "
                    f"态势{total_rows:,} | 已用{_duration(elapsed)} | ETA {_duration(eta)}"
                )

    partial.replace(dataset)
    if state_path.exists():
        state_path.unlink()
    manifest = {
        "schema_version": "round-01-dynamic-outcome-time-data-v1",
        "status": "completed",
        "episodes": len(schedule),
        "rows": total_rows,
        "split_episodes": dict(split_episodes),
        "group_episodes": dict(group_episodes),
        "per_scale": dict(scale_cells),
        "schedule_sha256": schedule_sha,
        "dataset_sha256": _sha256(dataset),
        "stage1_checkpoint": str(checkpoint),
        "stage1_checkpoint_sha256": _sha256(checkpoint),
        "git": _git_provenance(),
        "created_at": _now(),
        "weighting": {
            "core_episode_mass": float(collection["core_training_weight"]),
            "sparse_episode_mass": float(collection["sparse_training_weight"]),
            "heldout_training_mass": 0.0,
            "within_episode": "mass divided equally across saved snapshots",
        },
    }
    _write_json(manifest_path, manifest)
    log.write(
        f"[采集] 完成：{manifest['episodes']:,}局，{manifest['rows']:,}个态势，"
        f"SHA256={manifest['dataset_sha256'][:12]}…"
    )
    return dataset, manifest


def _read_rows(path: Path) -> list[Dict[str, object]]:
    rows = []
    episode_split: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            red = np.asarray(row["red_entities"], dtype=np.float32)
            blue = np.asarray(row["blue_entities"], dtype=np.float32)
            if red.shape != (int(row["red_count"]), 9):
                raise ValueError(f"bad Red entity table at line {line_number}")
            if blue.shape != (int(row["blue_count"]), 9):
                raise ValueError(f"bad Blue entity table at line {line_number}")
            previous = episode_split.setdefault(row["episode_id"], row["split"])
            if previous != row["split"]:
                raise ValueError("one episode leaks across data splits")
            rows.append(row)
    if not rows:
        raise ValueError("round-01 dataset is empty")
    return rows


class _Rows(Dataset):
    def __init__(self, rows: Sequence[Mapping[str, object]], config: Mapping[str, object]):
        self.rows = rows
        self.horizon_bins = int(config["training"]["horizon_bins"])
        self.steps_per_bin = int(config["training"]["steps_per_bin"])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Mapping[str, object]:
        return self.rows[index]

    def collate(self, rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
        batch = len(rows)
        max_red = max(int(row["red_count"]) for row in rows)
        max_blue = max(int(row["blue_count"]) for row in rows)
        target = np.zeros((batch, 8), dtype=np.float32)
        context = np.zeros((batch, 5), dtype=np.float32)
        red = np.zeros((batch, max_red, 9), dtype=np.float32)
        blue = np.zeros((batch, max_blue, 9), dtype=np.float32)
        red_mask = np.zeros((batch, max_red), dtype=bool)
        blue_mask = np.zeros((batch, max_blue), dtype=bool)
        red_win = np.zeros(batch, dtype=np.float32)
        remaining = np.zeros(batch, dtype=np.float32)
        label = np.zeros(batch, dtype=np.int64)
        training_weight = np.zeros(batch, dtype=np.float32)
        evaluation_weight = np.zeros(batch, dtype=np.float32)
        for index, row in enumerate(rows):
            red_count, blue_count = int(row["red_count"]), int(row["blue_count"])
            target[index] = row["target"]
            context[index] = row["context"]
            red[index, :red_count] = row["red_entities"]
            blue[index, :blue_count] = row["blue_entities"]
            red_mask[index, :red_count] = True
            blue_mask[index, :blue_count] = True
            red_win[index] = float(row["red_win"])
            remaining[index] = float(row["remaining_steps"])
            time_bin = min(
                (int(row["remaining_steps"]) - 1) // self.steps_per_bin,
                self.horizon_bins - 1,
            )
            label[index] = time_bin + (
                0 if bool(row["red_win"]) else self.horizon_bins
            )
            training_weight[index] = float(row["training_weight"])
            evaluation_weight[index] = float(row["evaluation_weight"])
        tensor = torch.as_tensor
        return {
            "target": tensor(target),
            "context": tensor(context),
            "red": tensor(red),
            "blue": tensor(blue),
            "red_mask": tensor(red_mask),
            "blue_mask": tensor(blue_mask),
            "red_win": tensor(red_win),
            "remaining": tensor(remaining),
            "label": tensor(label),
            "training_weight": tensor(training_weight),
            "evaluation_weight": tensor(evaluation_weight),
            "rows": list(rows),
        }


def _to_device(batch: Mapping[str, object], device: torch.device) -> Dict[str, object]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def _weighted_auc(
    labels: np.ndarray, probabilities: np.ndarray, weights: np.ndarray
) -> float | None:
    positive = float(weights[labels == 1].sum())
    negative = float(weights[labels == 0].sum())
    if positive <= 0.0 or negative <= 0.0:
        return None
    order = np.argsort(probabilities, kind="mergesort")
    labels = labels[order]
    probabilities = probabilities[order]
    weights = weights[order]
    concordance = 0.0
    cumulative_negative = 0.0
    index = 0
    while index < len(labels):
        stop = index + 1
        while stop < len(labels) and probabilities[stop] == probabilities[index]:
            stop += 1
        group_positive = float(weights[index:stop][labels[index:stop] == 1].sum())
        group_negative = float(weights[index:stop][labels[index:stop] == 0].sum())
        concordance += group_positive * (cumulative_negative + 0.5 * group_negative)
        cumulative_negative += group_negative
        index = stop
    return concordance / (positive * negative)


@torch.no_grad()
def _predict(
    model: DynamicHADOutcomeNet,
    rows: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
    device: torch.device,
    temperature: float = 1.0,
) -> Dict[str, object]:
    dataset = _Rows(rows, config)
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate,
    )
    model.eval()
    probabilities = []
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        logits = model(
            batch["target"],
            batch["red"],
            batch["blue"],
            batch["context"],
            batch["red_mask"],
            batch["blue_mask"],
        )
        probabilities.append(torch.softmax(logits / temperature, dim=-1).cpu().numpy())
    probability = np.concatenate(probabilities)
    k = int(config["training"]["horizon_bins"])
    steps_per_bin = int(config["training"]["steps_per_bin"])
    bin_steps = (np.arange(k, dtype=np.float32) + 0.5) * steps_per_bin
    red_mass = probability[:, :k]
    blue_mass = probability[:, k:]
    red_probability = red_mass.sum(axis=1)
    blue_probability = blue_mass.sum(axis=1)
    red_time_mass = (red_mass * bin_steps).sum(axis=1)
    blue_time_mass = (blue_mass * bin_steps).sum(axis=1)
    return {
        "joint_probability": probability,
        "red_probability": red_probability,
        "blue_probability": blue_probability,
        "expected_time": red_time_mass + blue_time_mass,
        "red_conditional_time": red_time_mass / np.clip(red_probability, 1e-8, None),
        "blue_conditional_time": blue_time_mass / np.clip(blue_probability, 1e-8, None),
    }


def _metrics(
    rows: Sequence[Mapping[str, object]], prediction: Mapping[str, np.ndarray]
) -> Dict[str, object]:
    labels = np.asarray([row["red_win"] for row in rows], dtype=np.int64)
    remaining = np.asarray([row["remaining_steps"] for row in rows], dtype=np.float32)
    weight = np.asarray([row["evaluation_weight"] for row in rows], dtype=np.float32)
    red_probability = np.asarray(prediction["red_probability"])
    expected_time = np.asarray(prediction["expected_time"])
    chosen_time = np.where(
        labels == 1,
        np.asarray(prediction["red_conditional_time"]),
        np.asarray(prediction["blue_conditional_time"]),
    )
    k = prediction["joint_probability"].shape[1] // 2
    steps_per_bin = 50 // k
    time_bin = np.minimum((remaining.astype(int) - 1) // steps_per_bin, k - 1)
    joint_label = time_bin + np.where(labels == 1, 0, k)
    joint_probability = prediction["joint_probability"]
    nll = -np.log(
        np.clip(joint_probability[np.arange(len(rows)), joint_label], 1e-12, None)
    )
    start = np.asarray([int(row["step"]) == 0 for row in rows])
    red_event = labels == 1
    blue_event = ~red_event

    def event_mae(mask: np.ndarray, values: np.ndarray) -> float | None:
        if not mask.any():
            return None
        return _weighted_mean(np.abs(values[mask] - remaining[mask]), weight[mask])

    return {
        "rows": len(rows),
        "episodes": len({row["episode_id"] for row in rows}),
        "accuracy": _weighted_mean((red_probability >= 0.5) == labels, weight),
        "auc": _weighted_auc(labels, red_probability, weight),
        "brier": _weighted_mean((red_probability - labels) ** 2, weight),
        "joint_nll": _weighted_mean(nll, weight),
        "time_mae_steps": _weighted_mean(np.abs(expected_time - remaining), weight),
        "conditional_time_mae_steps": _weighted_mean(
            np.abs(chosen_time - remaining), weight
        ),
        "red_win_time_mae_steps": event_mae(
            red_event, np.asarray(prediction["red_conditional_time"])
        ),
        "blue_win_time_mae_steps": event_mae(
            blue_event, np.asarray(prediction["blue_conditional_time"])
        ),
        "actual_start_win_rate": _weighted_mean(labels[start], weight[start]),
        "predicted_start_win_rate": _weighted_mean(
            red_probability[start], weight[start]
        ),
        "start_win_rate_absolute_error": abs(
            _weighted_mean(labels[start], weight[start])
            - _weighted_mean(red_probability[start], weight[start])
        ),
        "start_time_mae_steps": _weighted_mean(
            np.abs(expected_time[start] - remaining[start]), weight[start]
        ),
    }


def _temperature(
    model: DynamicHADOutcomeNet,
    rows: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
    device: torch.device,
) -> float:
    raw = _predict(model, rows, config, device, temperature=1.0)["joint_probability"]
    logits = np.log(np.clip(raw, 1e-12, None))
    k = int(config["training"]["horizon_bins"])
    steps = int(config["training"]["steps_per_bin"])
    labels = np.asarray(
        [
            min((int(row["remaining_steps"]) - 1) // steps, k - 1)
            + (0 if bool(row["red_win"]) else k)
            for row in rows
        ],
        dtype=np.int64,
    )
    weights = np.asarray([row["evaluation_weight"] for row in rows], np.float64)

    def score(value: float) -> float:
        scaled = logits / value
        scaled -= scaled.max(axis=1, keepdims=True)
        probability = np.exp(scaled)
        probability /= probability.sum(axis=1, keepdims=True)
        loss = -np.log(np.clip(probability[np.arange(len(rows)), labels], 1e-12, None))
        return _weighted_mean(loss, weights)

    coarse = np.linspace(0.5, 2.5, 81)
    first = min(coarse, key=score)
    fine = np.linspace(max(0.25, first - 0.05), first + 0.05, 101)
    return float(min(fine, key=score))


def _train(
    config: Mapping[str, object],
    dataset_path: Path,
    output_dir: Path,
    device: torch.device,
    log: LiveLog,
) -> tuple[Path, Dict[str, object], list[Dict[str, object]]]:
    training = config["training"]
    model_dir = output_dir / "model"
    metrics_path = model_dir / "metrics.json"
    checkpoint_path = model_dir / "dynamic_outcome_time_model.pt"
    rows = _read_rows(dataset_path)
    if metrics_path.is_file() and checkpoint_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("dataset_sha256") == _sha256(dataset_path):
            metrics["dataset"] = str(dataset_path)
            metrics["checkpoint"] = str(checkpoint_path)
            metrics["checkpoint_sha256"] = _sha256(checkpoint_path)
            _write_json(metrics_path, metrics)
            log.write("[训练] 已有与当前数据匹配的完整模型，跳过重训")
            return checkpoint_path, metrics, rows

    train_rows = [
        row
        for row in rows
        if row["split"] == "train" and row["scale_group"] in {"core", "sparse"}
    ]
    validation_core = [
        row
        for row in rows
        if row["split"] == "validation" and row["scale_group"] == "core"
    ]
    if not train_rows or not validation_core:
        raise ValueError("training and core validation splits must be non-empty")
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = DynamicHADOutcomeNet(
        horizon_bins=int(training["horizon_bins"]),
        entity_hidden_dim=int(training["entity_hidden_dim"]),
        hidden_dim=int(training["hidden_dim"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    data = _Rows(train_rows, config)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        data,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=data.collate,
    )
    latest_path = model_dir / "latest_training_state.pt"
    model_dir.mkdir(parents=True, exist_ok=True)
    history: list[Dict[str, object]] = []
    best_state = None
    best_score = math.inf
    best_epoch = 0
    stale = 0
    start_epoch = 1
    if latest_path.is_file():
        latest = torch.load(latest_path, map_location=device, weights_only=False)
        if latest.get("dataset_sha256") != _sha256(dataset_path):
            raise ValueError("latest training state belongs to a different dataset")
        model.load_state_dict(latest["model"], strict=True)
        optimizer.load_state_dict(latest["optimizer"])
        history = latest["history"]
        best_state = latest["best_state"]
        best_score = float(latest["best_score"])
        best_epoch = int(latest["best_epoch"])
        stale = int(latest["stale"])
        start_epoch = int(latest["epoch"]) + 1
        generator.set_state(latest["generator_state"])
        log.write(f"[训练] 从epoch {start_epoch}继续")

    k = int(training["horizon_bins"])
    steps_per_bin = int(training["steps_per_bin"])
    bin_steps = (
        torch.arange(k, device=device, dtype=torch.float32) + 0.5
    ) * steps_per_bin
    max_steps = float(k * steps_per_bin)
    maximum_epochs = int(training["max_epochs"])
    started = time.perf_counter()
    for epoch in range(start_epoch, maximum_epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        numerator = 0.0
        denominator = 0.0
        for raw_batch in loader:
            batch = _to_device(raw_batch, device)
            logits = model(
                batch["target"],
                batch["red"],
                batch["blue"],
                batch["context"],
                batch["red_mask"],
                batch["blue_mask"],
            )
            probability = torch.softmax(logits, dim=-1)
            red_probability = probability[:, :k].sum(dim=1)
            expected_time = (
                (probability[:, :k] + probability[:, k:]) * bin_steps
            ).sum(dim=1)
            joint = F.cross_entropy(logits, batch["label"], reduction="none")
            outcome = F.binary_cross_entropy(
                red_probability.clamp(1e-7, 1.0 - 1e-7),
                batch["red_win"],
                reduction="none",
            )
            time_loss = F.smooth_l1_loss(
                expected_time / max_steps,
                batch["remaining"] / max_steps,
                reduction="none",
            )
            row_loss = (
                joint
                + float(training["outcome_bce_weight"]) * outcome
                + float(training["time_huber_weight"]) * time_loss
            )
            weight = batch["training_weight"]
            loss = (row_loss * weight).sum() / weight.sum().clamp_min(1e-8)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip"])
            )
            optimizer.step()
            numerator += float((row_loss.detach() * weight).sum().cpu())
            denominator += float(weight.sum().cpu())

        validation_prediction = _predict(model, validation_core, config, device)
        validation = _metrics(validation_core, validation_prediction)
        score = (
            float(validation["brier"])
            + float(training["selection_time_weight"])
            * float(validation["time_mae_steps"])
            / max_steps
            + float(training["selection_nll_weight"])
            * float(validation["joint_nll"])
        )
        record = {
            "epoch": epoch,
            "train_loss": numerator / denominator,
            "validation_score": score,
            "core_validation_brier": validation["brier"],
            "core_validation_auc": validation["auc"],
            "core_validation_time_mae_steps": validation["time_mae_steps"],
            "core_validation_joint_nll": validation["joint_nll"],
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history.append(record)
        if score < best_score - 1e-7:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_state": best_state,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "stale": stale,
                "history": history,
                "generator_state": generator.get_state(),
                "dataset_sha256": _sha256(dataset_path),
            },
            latest_path,
        )
        elapsed = time.perf_counter() - started
        mean_epoch = elapsed / max(1, epoch - start_epoch + 1)
        eta = mean_epoch * max(0, maximum_epochs - epoch)
        log.write(
            f"[训练] epoch {epoch:03d}/{maximum_epochs} | "
            f"loss {record['train_loss']:.4f} | 核心Brier {validation['brier']:.4f} | "
            f"AUC {validation['auc'] if validation['auc'] is not None else float('nan'):.4f} | "
            f"时间MAE {validation['time_mae_steps']:.2f}步 | 最佳epoch {best_epoch} | "
            f"已用{_duration(elapsed)} | 最长ETA {_duration(eta)}"
        )
        if (
            epoch >= int(training["minimum_epochs"])
            and stale >= int(training["patience"])
        ):
            log.write(f"[训练] Validation连续{stale}轮未改善，提前停止")
            break

    if best_state is None:
        raise RuntimeError("training never produced a finite checkpoint")
    model.load_state_dict(best_state, strict=True)
    temperature = _temperature(model, validation_core, config, device)
    test_core = [
        row for row in rows if row["split"] == "test" and row["scale_group"] == "core"
    ]
    test_sparse = [
        row for row in rows if row["split"] == "test" and row["scale_group"] == "sparse"
    ]
    heldout = [row for row in rows if row["split"] == "generalization"]
    evaluation_rows = test_core + test_sparse + heldout
    evaluation_prediction = _predict(
        model, evaluation_rows, config, device, temperature=temperature
    )
    row_index = {id(row): index for index, row in enumerate(evaluation_rows)}

    def subset_prediction(selected: Sequence[Mapping[str, object]]) -> Dict[str, np.ndarray]:
        indices = np.asarray([row_index[id(row)] for row in selected], dtype=np.int64)
        return {
            key: np.asarray(value)[indices]
            for key, value in evaluation_prediction.items()
        }

    groups = {
        "core_test": _metrics(test_core, subset_prediction(test_core)),
        "sparse_test": _metrics(test_sparse, subset_prediction(test_sparse)),
        "heldout_generalization": _metrics(heldout, subset_prediction(heldout)),
    }
    per_scale: Dict[str, Dict[str, object]] = {}
    for scale in sorted(
        {row["scale"] for row in evaluation_rows},
        key=lambda value: tuple(map(int, value.split("v"))),
    ):
        selected = [row for row in evaluation_rows if row["scale"] == scale]
        value = _metrics(selected, subset_prediction(selected))
        value["scale_group"] = selected[0]["scale_group"]
        per_scale[scale] = value
        log.write(
            f"[评估] {scale:>3} {value['scale_group']:<7} | "
            f"真实胜率{value['actual_start_win_rate']:.1%} | "
            f"预测{value['predicted_start_win_rate']:.1%} | "
            f"误差{value['start_win_rate_absolute_error']:.1%} | "
            f"剩余时间MAE {value['time_mae_steps']:.2f}步"
        )

    core_scale_error = float(
        np.mean(
            [
                value["start_win_rate_absolute_error"]
                for value in per_scale.values()
                if value["scale_group"] == "core"
            ]
        )
    )
    acceptance_config = config["acceptance"]
    core = groups["core_test"]
    acceptance = {
        "core_brier": bool(core["brier"] <= float(acceptance_config["core_brier_max"])),
        "core_auc": bool(
            core["auc"] is not None
            and core["auc"] >= float(acceptance_config["core_auc_min"])
        ),
        "core_time_mae": bool(
            core["time_mae_steps"]
            <= float(acceptance_config["core_time_mae_max_steps"])
        ),
        "core_scale_win_rate_error": bool(
            core_scale_error
            <= float(acceptance_config["core_mean_scale_win_rate_error_max"])
        ),
    }
    acceptance["all_passed"] = all(acceptance.values())

    core_train = [
        row for row in rows if row["split"] == "train" and row["scale_group"] == "core"
    ]
    train_start = [row for row in core_train if int(row["step"]) == 0]
    train_rate = float(np.mean([row["red_win"] for row in train_start]))
    train_median_time = float(np.median([row["remaining_steps"] for row in core_train]))
    core_labels = np.asarray([row["red_win"] for row in test_core], dtype=np.float32)
    core_remaining = np.asarray(
        [row["remaining_steps"] for row in test_core], dtype=np.float32
    )
    core_weights = np.asarray(
        [row["evaluation_weight"] for row in test_core], dtype=np.float32
    )
    baselines = {
        "core_train_start_win_rate": train_rate,
        "core_constant_brier": _weighted_mean(
            (core_labels - train_rate) ** 2, core_weights
        ),
        "core_train_median_remaining_steps": train_median_time,
        "core_median_time_mae_steps": _weighted_mean(
            np.abs(core_remaining - train_median_time), core_weights
        ),
    }
    calibration = []
    core_prediction = subset_prediction(test_core)
    probability = core_prediction["red_probability"]
    edges = np.linspace(0.0, 1.0, 11)
    for index in range(10):
        mask = (probability >= edges[index]) & (
            probability <= edges[index + 1]
            if index == 9
            else probability < edges[index + 1]
        )
        if mask.any():
            calibration.append(
                {
                    "low": float(edges[index]),
                    "high": float(edges[index + 1]),
                    "mean_prediction": _weighted_mean(
                        probability[mask], core_weights[mask]
                    ),
                    "observed_rate": _weighted_mean(
                        core_labels[mask], core_weights[mask]
                    ),
                    "effective_episodes": float(core_weights[mask].sum()),
                }
            )

    torch.save(
        {
            "model": best_state,
            "horizon_bins": k,
            "steps_per_bin": steps_per_bin,
            "entity_hidden_dim": int(training["entity_hidden_dim"]),
            "hidden_dim": int(training["hidden_dim"]),
            "temperature": temperature,
            "best_epoch": best_epoch,
            "dataset_sha256": _sha256(dataset_path),
            "git": _git_provenance(),
        },
        checkpoint_path,
    )
    restored_payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    restored = DynamicHADOutcomeNet(
        restored_payload["horizon_bins"],
        restored_payload["entity_hidden_dim"],
        restored_payload["hidden_dim"],
    ).to(device)
    restored.load_state_dict(restored_payload["model"], strict=True)
    repeated = _predict(
        restored,
        evaluation_rows,
        config,
        device,
        temperature=float(restored_payload["temperature"]),
    )
    reload_difference = float(
        np.max(
            np.abs(
                repeated["joint_probability"]
                - evaluation_prediction["joint_probability"]
            )
        )
    )
    if reload_difference > 1e-7:
        raise RuntimeError("reloaded outcome-time checkpoint changes predictions")

    prediction_path = model_dir / "evaluation_predictions.csv"
    with prediction_path.open("w", encoding="utf-8", newline="") as target:
        fields = [
            "episode_id",
            "split",
            "scale_group",
            "scale",
            "opponent",
            "step",
            "remaining_steps",
            "red_win",
            "predicted_red_win_probability",
            "predicted_remaining_steps",
            "predicted_steps_if_red_wins",
            "predicted_steps_if_blue_wins",
        ]
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(evaluation_rows):
            writer.writerow(
                {
                    **{field: row[field] for field in fields[:8]},
                    "predicted_red_win_probability": float(
                        evaluation_prediction["red_probability"][index]
                    ),
                    "predicted_remaining_steps": float(
                        evaluation_prediction["expected_time"][index]
                    ),
                    "predicted_steps_if_red_wins": float(
                        evaluation_prediction["red_conditional_time"][index]
                    ),
                    "predicted_steps_if_blue_wins": float(
                        evaluation_prediction["blue_conditional_time"][index]
                    ),
                }
            )
    with (model_dir / "training_history.csv").open(
        "w", encoding="utf-8", newline=""
    ) as target:
        writer = csv.DictWriter(target, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    metrics = {
        "schema_version": "round-01-dynamic-outcome-time-model-v1",
        "status": "completed",
        "dataset": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "temperature": temperature,
        "history": history,
        "group_metrics": groups,
        "per_scale": per_scale,
        "core_mean_scale_win_rate_absolute_error": core_scale_error,
        "baselines": baselines,
        "calibration": calibration,
        "acceptance": acceptance,
        "reload_max_prediction_difference": reload_difference,
        "git": _git_provenance(),
        "completed_at": _now(),
    }
    _write_json(metrics_path, metrics)
    if latest_path.exists():
        latest_path.unlink()
    log.write(
        f"[训练] 完成：最佳epoch {best_epoch}，温度{temperature:.3f}，"
        f"核心Test Brier {core['brier']:.4f}，时间MAE {core['time_mae_steps']:.2f}步"
    )
    return checkpoint_path, metrics, rows


def _plots(output_dir: Path, metrics: Mapping[str, object]) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    history = metrics["history"]
    epoch = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    axes[0].plot(epoch, [row["train_loss"] for row in history])
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Weighted loss")
    axes[1].plot(epoch, [row["core_validation_brier"] for row in history])
    axes[1].set(title="Core validation Brier", xlabel="Epoch", ylabel="Brier")
    axes[2].plot(epoch, [row["core_validation_time_mae_steps"] for row in history])
    axes[2].set(title="Core validation time error", xlabel="Epoch", ylabel="MAE (steps)")
    for axis in axes:
        axis.grid(alpha=0.25)
    training_path = figure_dir / "01_training_curves.png"
    fig.savefig(training_path, dpi=180)
    plt.close(fig)

    scales = list(metrics["per_scale"])
    actual = [metrics["per_scale"][scale]["actual_start_win_rate"] for scale in scales]
    predicted = [
        metrics["per_scale"][scale]["predicted_start_win_rate"] for scale in scales
    ]
    positions = np.arange(len(scales))
    fig, axis = plt.subplots(figsize=(15, 5), constrained_layout=True)
    axis.bar(positions - 0.2, actual, width=0.4, label="Observed")
    axis.bar(positions + 0.2, predicted, width=0.4, label="Predicted")
    axis.set(xticks=positions, xticklabels=scales, ylim=(0, 1), ylabel="Red win rate", title="Initial-state win rates by roster")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    win_path = figure_dir / "02_all_scale_win_rates.png"
    fig.savefig(win_path, dpi=180)
    plt.close(fig)

    time_mae = [metrics["per_scale"][scale]["time_mae_steps"] for scale in scales]
    colors = [
        {"core": "tab:blue", "sparse": "tab:orange", "heldout": "tab:red"}[
            metrics["per_scale"][scale]["scale_group"]
        ]
        for scale in scales
    ]
    fig, axis = plt.subplots(figsize=(15, 5), constrained_layout=True)
    axis.bar(positions, time_mae, color=colors)
    axis.set(xticks=positions, xticklabels=scales, ylabel="MAE (steps)", title="Remaining-time error by roster")
    axis.grid(axis="y", alpha=0.25)
    time_path = figure_dir / "03_all_scale_time_mae.png"
    fig.savefig(time_path, dpi=180)
    plt.close(fig)

    calibration = metrics["calibration"]
    fig, axis = plt.subplots(figsize=(6, 6), constrained_layout=True)
    axis.plot([0, 1], [0, 1], "--", color="0.5", label="Ideal")
    axis.plot(
        [row["mean_prediction"] for row in calibration],
        [row["observed_rate"] for row in calibration],
        "o-",
        label="Core test",
    )
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted Red win probability", ylabel="Observed Red win rate", title="Win-probability calibration")
    axis.legend()
    axis.grid(alpha=0.25)
    calibration_path = figure_dir / "04_core_calibration.png"
    fig.savefig(calibration_path, dpi=180)
    plt.close(fig)
    return [training_path, win_path, time_path, calibration_path]


def _number(value: object, digits: int = 3) -> str:
    if value is None:
        return "样本单一，无法计算"
    return f"{float(value):.{digits}f}"


def _stage2_report_lines(
    output_dir: Path,
    manifest: Mapping[str, object],
    metrics: Mapping[str, object],
    figures: Sequence[Path],
) -> list[str]:
    """Build the Stage-2 section without creating a second report file."""
    core = metrics["group_metrics"]["core_test"]
    sparse = metrics["group_metrics"]["sparse_test"]
    heldout = metrics["group_metrics"]["heldout_generalization"]
    baselines = metrics["baselines"]
    lines = [
        "# Open-SCORE 第一轮 Stage 2：任意规模胜率与剩余时间评估结果",
        "",
        f"> 生成时间：`{_now()}`  ",
        f"> 结果：**{'通过预设核心门槛' if metrics['acceptance']['all_passed'] else '训练完成，但有核心门槛未通过'}**  ",
        "> 本报告由统一流水线自动生成；训练外规模不参与训练或选模。",
        "",
        "## 做了什么",
        "",
        "冻结第一轮REFIL-QMIX策略，让它在不同HAD人数和两种规则Blue对手下继续推演。每个保存的全局态势同时带两个标签：最终是Red还是Blue获胜，以及从该态势到结束还剩多少步。Stage 2用一个动态实体集合网络联合预测胜负概率和剩余时间。",
        "",
        f"共采集 `{int(manifest['episodes']):,}` 局、`{int(manifest['rows']):,}` 个态势。核心规模（2v1至4v4）每局训练权重1.0；稀疏扩展规模每局权重0.25；留出规模权重0且完全不进入训练。每局的权重再平均分给该局保存的态势，避免长局被重复强调。",
        "",
        "模型使用Deep Sets共享实体编码器，并输出20个联合概率：10个Red在不同时间段获胜的概率，加10个Blue在不同时间段获胜的概率。每个时间段宽5步。将相应概率求和得到胜率，对时间段加权平均得到预计剩余步数；也可分别得到“如果Red获胜还要多久”和“如果Blue获胜还要多久”。",
        "",
        "方法依据是[Deep Sets](https://papers.nips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html)的无序集合编码，以及[DeepHit](https://ojs.aaai.org/index.php/AAAI/article/view/11842)直接学习“事件类型×发生时间”联合分布的思路。本轮采用DeepHit式离散竞争风险目标，但没有照搬其全部网络和排序损失，因此应称为借鉴而不是完整DeepHit论文复现。若实体之间的两两关系成为瓶颈，可在下一轮比较[Set Transformer](https://proceedings.mlr.press/v97/lee19d.html)。",
        "",
        "## 总体结果",
        "",
        "| 测试范围 | 局数 | Brier↓ | AUC↑ | 剩余时间MAE↓ | 胜利条件时间MAE↓ |",
        "|---|---:|---:|---:|---:|---:|",
        f"| 核心规模Test | {core['episodes']:,} | {_number(core['brier'], 4)} | {_number(core['auc'], 4)} | {_number(core['time_mae_steps'], 2)}步 | {_number(core['conditional_time_mae_steps'], 2)}步 |",
        f"| 稀疏扩展Test | {sparse['episodes']:,} | {_number(sparse['brier'], 4)} | {_number(sparse['auc'], 4)} | {_number(sparse['time_mae_steps'], 2)}步 | {_number(sparse['conditional_time_mae_steps'], 2)}步 |",
        f"| 完全留出规模 | {heldout['episodes']:,} | {_number(heldout['brier'], 4)} | {_number(heldout['auc'], 4)} | {_number(heldout['time_mae_steps'], 2)}步 | {_number(heldout['conditional_time_mae_steps'], 2)}步 |",
        "",
        f"核心规模的固定胜率基线Brier为 `{baselines['core_constant_brier']:.4f}`；固定中位剩余时间基线MAE为 `{baselines['core_median_time_mae_steps']:.2f}` 步。模型核心分规模初始胜率平均绝对误差为 `{metrics['core_mean_scale_win_rate_absolute_error']:.2%}`。",
        "",
        "![训练曲线](figures/01_training_curves.png)",
        "",
        "## 所有规模的胜率与时间",
        "",
        "表中的胜率使用每局第0步态势，因此一局只计一次；时间MAE使用该规模所有保存态势。`core`参与主要训练，`sparse`低权重参与，`heldout`完全没有进入训练。",
        "",
        "| 规模 | 类型 | 局数 | 真实Red胜率 | 预测Red胜率 | 胜率误差 | 剩余时间MAE | Red胜时MAE | Blue胜时MAE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scale, value in metrics["per_scale"].items():
        lines.append(
            f"| {scale} | {value['scale_group']} | {value['episodes']:,} | "
            f"{value['actual_start_win_rate']:.1%} | {value['predicted_start_win_rate']:.1%} | "
            f"{value['start_win_rate_absolute_error']:.1%} | "
            f"{value['time_mae_steps']:.2f}步 | {_number(value['red_win_time_mae_steps'], 2)} | "
            f"{_number(value['blue_win_time_mae_steps'], 2)} |"
        )
    lines.extend(
        [
            "",
            "![全部规模胜率](figures/02_all_scale_win_rates.png)",
            "",
            "![全部规模时间误差](figures/03_all_scale_time_mae.png)",
            "",
            "![核心规模概率校准](figures/04_core_calibration.png)",
            "",
            "## 验收与解释",
            "",
            "| 核心门槛 | 结果 |",
            "|---|---:|",
            f"| Brier不超过配置阈值 | {'通过' if metrics['acceptance']['core_brier'] else '未通过'} |",
            f"| AUC达到配置阈值 | {'通过' if metrics['acceptance']['core_auc'] else '未通过'} |",
            f"| 剩余时间MAE达到配置阈值 | {'通过' if metrics['acceptance']['core_time_mae'] else '未通过'} |",
            f"| 核心分规模胜率误差达到阈值 | {'通过' if metrics['acceptance']['core_scale_win_rate_error'] else '未通过'} |",
            "",
            "低权重扩展数据的作用是教会网络识别更大集合，但不允许它牺牲2v1至4v4的主要效果。完全留出的5v1、6v3和7v3只衡量零样本规模泛化；如果这些结果较差，应增加相邻规模覆盖或采用Set Transformer，而不能根据留出结果反复调参后仍称其为留出测试。",
            "",
            "## 结果位置",
            "",
            "- `data/dynamic_outcome_time.jsonl`：任意长度实体集合数据。",
            "- `model/dynamic_outcome_time_model.pt`：胜负—时间联合模型。",
            "- `model/metrics.json`：全部指标、训练历史和验收结果。",
            "- `model/evaluation_predictions.csv`：逐态势预测。",
            "- `figures/`：四张结果图。",
            "- `live_progress.log`：从采集到报告的完整实时日志。",
            "",
            f"模型参数量：`{metrics['parameter_count']:,}`；最佳epoch：`{metrics['best_epoch']}`；温度校准参数：`{metrics['temperature']:.3f}`；重新加载最大预测差：`{metrics['reload_max_prediction_difference']:.3g}`。",
        ]
    )
    return lines


def _merge_round01_report() -> Path:
    """Regenerate the one user-facing report for all round-01 work."""

    from run_round01_remaining import write_report

    round_root = PROJECT / "outputs" / "round_01_mvp"
    return write_report(round_root, ())


def _completed_pipeline_valid(
    output_dir: Path, status: Mapping[str, object], expected_report: Path
) -> bool:
    """Check hashes before treating an existing round-01 run as complete."""

    try:
        dataset = output_dir / "data" / "dynamic_outcome_time.jsonl"
        manifest_path = output_dir / "data" / "dataset_manifest.json"
        checkpoint = output_dir / "model" / "dynamic_outcome_time_model.pt"
        metrics_path = output_dir / "model" / "metrics.json"
        required = (dataset, manifest_path, checkpoint, metrics_path, expected_report)
        if not all(path.is_file() for path in required):
            return False
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return bool(
            status.get("status") == "completed"
            and Path(str(status.get("report", ""))).resolve()
            == expected_report.resolve()
            and manifest.get("status") == "completed"
            and metrics.get("status") == "completed"
            and str(manifest.get("dataset_sha256")) == _sha256(dataset)
            and str(metrics.get("dataset_sha256")) == _sha256(dataset)
            and str(metrics.get("checkpoint_sha256")) == _sha256(checkpoint)
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def main() -> None:
    args = parse_args()
    config = _load_config(args)
    _validate_config(config)
    output_dir = _resolve(config["output_dir"]).resolve()
    log = LiveLog(output_dir)
    status_path = output_dir / "pipeline_status.json"
    schedule = _schedule(config)
    group_counts = Counter(cell["group"] for cell in schedule)
    if args.dry_run:
        log.write(
            f"[计划] 共{len(schedule):,}局：core={group_counts['core']:,}，"
            f"sparse={group_counts['sparse']:,}，heldout={group_counts['heldout']:,}"
        )
        return
    completed_status = None
    if status_path.is_file():
        completed_status = json.loads(status_path.read_text(encoding="utf-8"))
    expected_report = PROJECT / "outputs" / "round_01_mvp" / "round_01_report.md"
    if completed_status and _completed_pipeline_valid(
        output_dir, completed_status, expected_report
    ):
        log.write(f"[流水线] 已完成。报告：{completed_status['report']}")
        return

    device = _choose_device(str(config["device"]))
    started = time.perf_counter()
    _write_json(
        status_path,
        {
            "status": "running",
            "stage": "collection",
            "started_at": _now(),
            "total_episodes": len(schedule),
            "git": _git_provenance(),
        },
    )
    log.write(
        f"[流水线] 启动第一轮Stage 2补全 | device={device} | 共{len(schedule):,}局 | "
        f"core={group_counts['core']:,} sparse={group_counts['sparse']:,} "
        f"heldout={group_counts['heldout']:,}"
    )
    try:
        dataset, manifest = _collect_dataset(config, output_dir, device, log)
        _write_json(
            status_path,
            {
                "status": "running",
                "stage": "training",
                "started_at": _now(),
                "dataset": str(dataset),
                "dataset_manifest": str(output_dir / "data" / "dataset_manifest.json"),
            },
        )
        checkpoint, metrics, _ = _train(
            config, dataset, output_dir, device, log
        )
        _write_json(
            status_path,
            {
                "status": "running",
                "stage": "report",
                "started_at": _now(),
                "checkpoint": str(checkpoint),
            },
        )
        figures = _plots(output_dir, metrics)
        report = _merge_round01_report()
        status = {
            "status": "completed",
            "stage": "completed",
            "completed_at": _now(),
            "elapsed_seconds": time.perf_counter() - started,
            "dataset": str(dataset),
            "checkpoint": str(checkpoint),
            "metrics": str(output_dir / "model" / "metrics.json"),
            "report": str(report),
            "figures": [str(path) for path in figures],
            "live_log": str(log.path),
        }
        _write_json(status_path, status)
        log.write(
            f"[流水线] 全部完成 | 用时{_duration(status['elapsed_seconds'])} | "
            f"报告：{report}"
        )
    except Exception as error:
        _write_json(
            status_path,
            {
                "status": "failed",
                "stage": "failed",
                "failed_at": _now(),
                "error": f"{type(error).__name__}: {error}",
                "live_log": str(log.path),
            },
        )
        log.write(f"[流水线] 失败：{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
