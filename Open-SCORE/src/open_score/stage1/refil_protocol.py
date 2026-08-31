"""Focused round-01 REFIL-QMIX training protocol for dynamic-scale HAD."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from open_score.stage1.curriculum import (
    Scale,
    StepLearningProgressCurriculum,
    supported_scales,
)
from open_score.stage1.learner import LearnerMetrics, SequenceQMIXLearner, linear_epsilon
from open_score.stage1.replay import EpisodeReplayBuffer
from open_score.stage1.runner import (
    BatchedHADRedRunner,
    HADStage1Factory,
    QMixController,
    RuleBasedController,
)
from open_score.stage1.training import make_had_qmix


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _scale_label(scale: Scale) -> str:
    return f"{scale[0]}v{scale[1]}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if total < 1:
        return float("nan"), float("nan")
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


def _evaluation_key(evaluation: Mapping[str, object]) -> Tuple[float, float]:
    cell_payoffs = []
    for component in evaluation["components"].values():
        cell_payoffs.extend(component["per_scale_payoff"].values())
    return min(float(value) for value in cell_payoffs), float(evaluation["mean_payoff"])


def evaluate_refil(
    runner: BatchedHADRedRunner,
    controller: QMixController,
    scales: Sequence[Scale],
    opponent_styles: Sequence[str],
    episodes_per_cell: int,
    seed_base: int,
    batch_size: int,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    schedule = []
    for style_index, style in enumerate(opponent_styles):
        for scale_index, scale in enumerate(scales):
            for episode_index in range(episodes_per_cell):
                schedule.append(
                    (
                        scale,
                        style,
                        int(
                            seed_base
                            + style_index * 1_000_000
                            + scale_index * 10_000
                            + episode_index
                        ),
                    )
                )
    records: List[Dict[str, object]] = []
    controller.epsilon = 0.0
    for start in range(0, len(schedule), batch_size):
        cells = schedule[start : start + batch_size]
        episodes = runner.run_batch(
            [cell[0] for cell in cells],
            controller,
            [RuleBasedController(cell[1]) for cell in cells],
            [cell[2] for cell in cells],
        )
        for cell, episode in zip(cells, episodes):
            records.append(
                {
                    "scale": _scale_label(cell[0]),
                    "opponent": cell[1],
                    "seed": cell[2],
                    "outcome_red": episode.outcome_red,
                    "red_win": bool(episode.outcome_red > 0.0),
                    "episode_length": episode.length,
                }
            )
    components = {}
    for style in opponent_styles:
        style_records = [row for row in records if row["opponent"] == style]
        per_scale = {}
        per_scale_win = {}
        for scale in scales:
            label = _scale_label(scale)
            cell = [row for row in style_records if row["scale"] == label]
            per_scale[label] = float(np.mean([row["outcome_red"] for row in cell]))
            per_scale_win[label] = float(np.mean([row["red_win"] for row in cell]))
        components[style] = {
            "episodes": len(style_records),
            "mean_payoff": float(
                np.mean([row["outcome_red"] for row in style_records])
            ),
            "win_rate": float(np.mean([row["red_win"] for row in style_records])),
            "per_scale_payoff": per_scale,
            "per_scale_win_rate": per_scale_win,
        }
    return {
        "episodes": len(records),
        "episodes_per_scale_opponent": episodes_per_cell,
        "mean_payoff": float(np.mean([row["outcome_red"] for row in records])),
        "win_rate": float(np.mean([row["red_win"] for row in records])),
        "mean_episode_length": float(
            np.mean([row["episode_length"] for row in records])
        ),
        "components": components,
        "selection_key": list(_evaluation_key({"components": components, "mean_payoff": float(np.mean([row["outcome_red"] for row in records]))})),
        "seed_base": seed_base,
    }, records


def _write_curves(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_evaluation(
    output_dir: Path,
    evaluation: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
    scales: Sequence[Scale],
    opponent_styles: Sequence[str],
) -> None:
    evaluation_dir = output_dir.parent / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for scale in scales:
        label = _scale_label(scale)
        by_style = {
            style: [
                row
                for row in records
                if row["scale"] == label and row["opponent"] == style
            ]
            for style in opponent_styles
        }
        combined = [row for values in by_style.values() for row in values]
        wins = sum(bool(row["red_win"]) for row in combined)
        low, high = _wilson(wins, len(combined))
        row = {"scale": label}
        for style, values in by_style.items():
            row[f"win_rate_{style}"] = float(
                np.mean([value["red_win"] for value in values])
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
    win_path = evaluation_dir / "win_rate_by_scale.csv"
    with win_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = dict(evaluation)
    payload["episode_records"] = list(records)
    (evaluation_dir / "summary.json").write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def validate_refil_args(args) -> None:
    if tuple(args.algorithms) != ("refil_qmix",):
        raise ValueError("the round-01 REFIL path runs refil_qmix alone")
    if args.train_side != "Red":
        raise ValueError("round-01 REFIL trains only Red")
    if args.total_environment_steps < 1 or args.num_envs < 1:
        raise ValueError("environment-step budget and num-envs must be positive")
    if args.batch_episodes > args.replay_episodes:
        raise ValueError("batch episodes cannot exceed replay capacity")
    if args.optimizer != "rmsprop" or args.td_lambda != 0.0:
        raise ValueError("the frozen REFIL protocol requires RMSprop and 1-step TD")
    if set(args.opponent_styles or ()) != {"rush", "split_rush"}:
        raise ValueError("the frozen opponent suite is rush and split_rush")
    if len(args.seeds) != 1:
        raise ValueError("round 01 requires exactly one training seed")
    if len(args.curriculum_boundaries) != 4:
        raise ValueError("four curriculum boundaries are required")


def run_refil_round(args, device: torch.device, git_provenance: Mapping[str, object]) -> Dict[str, object]:
    validate_refil_args(args)
    seed = int(args.seeds[0])
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    scales = tuple(supported_scales(4, 1))
    opponent_styles = tuple(args.opponent_styles)
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model = make_had_qmix(
        device,
        agent_hidden_dim=args.agent_hidden_dim,
        mixer_hidden_dim=args.hypernet_hidden_dim,
        mixing_dim=args.mixing_dim,
        encoder_kind="refil",
        attention_heads=args.attention_heads,
        attention_embed_dim=args.attention_embed_dim,
        hypernet_hidden_dim=args.hypernet_hidden_dim,
    )
    learner = SequenceQMIXLearner(
        model,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        td_lambda=args.td_lambda,
        target_update_interval=args.target_update_interval,
        max_grad_norm=10.0,
        optimizer=args.optimizer,
        rmsprop_alpha=args.rmsprop_alpha,
        rmsprop_eps=args.rmsprop_eps,
        imagine_weight=args.refil_imagine_weight,
    )
    replay = EpisodeReplayBuffer(args.replay_episodes, seed=seed + 29)
    curriculum = StepLearningProgressCurriculum(
        scales=scales,
        boundaries=args.curriculum_boundaries,
        update_steps=args.curriculum_update_steps,
        td_window=args.curriculum_td_window,
        uniform_floor=args.curriculum_uniform_floor,
        max_scale_probability=args.curriculum_max_scale_probability,
    )
    factory = HADStage1Factory(
        max_steps=args.max_steps,
        gamma=args.gamma,
        shaping_scale=args.shaping_scale,
    )
    runner = BatchedHADRedRunner(factory)
    controller = QMixController(model, device, epsilon=1.0, name="refil_qmix")
    rng = np.random.default_rng(seed + 19)
    environment_steps = 0
    episode_count = 0
    scale_steps = {scale: 0 for scale in scales}
    scale_episodes = {scale: 0 for scale in scales}
    opponent_episodes = {style: 0 for style in opponent_styles}
    curves: List[Dict[str, object]] = []
    recent_outcomes = deque(maxlen=200)
    recent_returns = deque(maxlen=200)
    latest_metrics: Optional[LearnerMetrics] = None
    best_evaluation: Optional[Dict[str, object]] = None
    best_steps: Optional[int] = None
    best_path = checkpoint_dir / f"refil_qmix_seed{seed}_best.pt"
    final_path = checkpoint_dir / f"refil_qmix_seed{seed}_final.pt"
    validation_seed = seed + 8_000_000
    heldout_seed = seed + 18_000_000
    resumed_from = None

    progress_path = output_dir / "progress.json"
    if args.resume is not None:
        resumed_from = str(Path(args.resume).resolve())
        extra = learner.load(Path(args.resume), map_location=device)
        environment_steps = int(extra["environment_steps"])
        episode_count = int(extra["episode_count"])
        curriculum.load_state_dict(extra["curriculum"])
        rng.bit_generator.state = extra["numpy_rng_state"]
        scale_steps.update(
            {tuple(map(int, key.split("v"))): int(value) for key, value in extra["scale_steps"].items()}
        )
        scale_episodes.update(
            {tuple(map(int, key.split("v"))): int(value) for key, value in extra["scale_episodes"].items()}
        )
        opponent_episodes.update(
            {str(key): int(value) for key, value in extra["opponent_episodes"].items()}
        )
        best_evaluation = extra.get("best_evaluation")
        best_steps = extra.get("best_steps")
        if progress_path.is_file():
            curves = json.loads(progress_path.read_text(encoding="utf-8"))["curves"]

    def checkpoint_extra(selection: str, evaluation=None) -> Dict[str, object]:
        return {
            "protocol": "OpenSCORE-S1S2-REFIL-1M-v2",
            "algorithm": "refil_qmix",
            "architecture": model.architecture_name,
            "selection": selection,
            "seed": seed,
            "train_side": "Red",
            "environment_steps": environment_steps,
            "episode_count": episode_count,
            "curriculum": curriculum.state_dict(),
            "numpy_rng_state": rng.bit_generator.state,
            "scale_steps": {_scale_label(key): value for key, value in scale_steps.items()},
            "scale_episodes": {_scale_label(key): value for key, value in scale_episodes.items()},
            "opponent_episodes": opponent_episodes,
            "best_evaluation": best_evaluation,
            "best_steps": best_steps,
            "selection_evaluation": evaluation,
            "git_provenance": dict(git_provenance),
            "replay_resume_semantics": "cold replay after resume; uninterrupted run is canonical",
        }

    def record_curve(evaluation: Mapping[str, object]) -> None:
        probabilities = curriculum.probabilities(environment_steps)
        curves.append(
            {
                "environment_steps": environment_steps,
                "episode": episode_count,
                "learner_step": learner.learner_step,
                "epsilon": linear_epsilon(
                    environment_steps,
                    start=1.0,
                    finish=0.05,
                    anneal_steps=args.epsilon_anneal_steps,
                ),
                "train_win_rate_200": (
                    None if not recent_outcomes else float(np.mean(recent_outcomes))
                ),
                "train_return_200": (
                    None if not recent_returns else float(np.mean(recent_returns))
                ),
                "validation_win_rate": evaluation["win_rate"],
                "validation_mean_payoff": evaluation["mean_payoff"],
                "validation_worst_cell_payoff": evaluation["selection_key"][0],
                "loss": None if latest_metrics is None else latest_metrics.loss,
                "base_loss": None if latest_metrics is None else latest_metrics.base_loss,
                "imagine_loss": None if latest_metrics is None else latest_metrics.imagine_loss,
                "mean_absolute_td": None if latest_metrics is None else latest_metrics.mean_absolute_td,
                "grad_norm": None if latest_metrics is None else latest_metrics.grad_norm,
                "per_scale_validation_json": json.dumps(
                    {
                        style: evaluation["components"][style]["per_scale_win_rate"]
                        for style in opponent_styles
                    },
                    sort_keys=True,
                ),
                "curriculum_probabilities_json": json.dumps(
                    {_scale_label(key): value for key, value in probabilities.items()},
                    sort_keys=True,
                ),
            }
        )
        _write_curves(output_dir / "training_curves.csv", curves)
        progress_path.write_text(
            json.dumps(
                _jsonable(
                    {
                        "status": "training",
                        "environment_steps": environment_steps,
                        "episode_count": episode_count,
                        "curves": curves,
                    }
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    initial, _ = evaluate_refil(
        runner,
        controller,
        scales,
        opponent_styles,
        args.eval_episodes_per_scale_opponent,
        validation_seed,
        args.num_envs,
    )
    if not curves:
        record_curve(initial)
    next_evaluation = (
        (environment_steps // args.eval_every_environment_steps) + 1
    ) * args.eval_every_environment_steps
    next_progress = ((environment_steps // 10_000) + 1) * 10_000
    started = time.perf_counter()
    while environment_steps < args.total_environment_steps:
        batch_scales = [
            curriculum.sample(environment_steps, rng)
            for _ in range(args.num_envs)
        ]
        batch_styles = [
            opponent_styles[int(rng.integers(0, len(opponent_styles)))]
            for _ in range(args.num_envs)
        ]
        batch_seeds = [seed + (episode_count + index + 1) * 101 for index in range(args.num_envs)]
        controller.epsilon = linear_epsilon(
            environment_steps,
            start=1.0,
            finish=0.05,
            anneal_steps=args.epsilon_anneal_steps,
        )
        episodes = runner.run_batch(
            batch_scales,
            controller,
            [RuleBasedController(style) for style in batch_styles],
            batch_seeds,
        )
        for style, rollout in zip(batch_styles, episodes):
            replay.add(rollout.red)
            environment_steps += rollout.length
            episode_count += 1
            scale_steps[rollout.red.scale] += rollout.length
            scale_episodes[rollout.red.scale] += 1
            opponent_episodes[style] += 1
            recent_outcomes.append(float(rollout.outcome_red > 0.0))
            recent_returns.append(float(rollout.red.rewards.sum()))
        if len(replay) >= args.batch_episodes:
            for _ in range(len(episodes) * args.updates_per_episode):
                latest_metrics = learner.train_batch(
                    replay.sample(args.batch_episodes, device)
                )
                curriculum.record_td_samples(
                    latest_metrics.td_samples_by_scale
                )
        if environment_steps >= next_progress:
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "environment_steps": environment_steps,
                        "episodes": episode_count,
                        "learner_steps": learner.learner_step,
                        "epsilon": controller.epsilon,
                        "train_win_rate_200": float(np.mean(recent_outcomes)),
                        "loss": None if latest_metrics is None else latest_metrics.loss,
                        "steps_per_second": environment_steps / max(elapsed, 1e-9),
                    }
                ),
                flush=True,
            )
            while next_progress <= environment_steps:
                next_progress += 10_000
        if environment_steps >= next_evaluation or environment_steps >= args.total_environment_steps:
            evaluation, _ = evaluate_refil(
                runner,
                controller,
                scales,
                opponent_styles,
                args.eval_episodes_per_scale_opponent,
                validation_seed,
                args.num_envs,
            )
            record_curve(evaluation)
            if best_evaluation is None or _evaluation_key(evaluation) > _evaluation_key(best_evaluation):
                best_evaluation = dict(evaluation)
                best_steps = environment_steps
                learner.save(best_path, checkpoint_extra("validation_best", evaluation))
            learner.save(
                checkpoint_dir / f"refil_qmix_seed{seed}_latest.pt",
                checkpoint_extra("latest", evaluation),
            )
            print(
                json.dumps(
                    {
                        "event": "validation",
                        "environment_steps": environment_steps,
                        "win_rate": evaluation["win_rate"],
                        "mean_payoff": evaluation["mean_payoff"],
                        "worst_cell_payoff": evaluation["selection_key"][0],
                        "best_steps": best_steps,
                    }
                ),
                flush=True,
            )
            while next_evaluation <= environment_steps:
                next_evaluation += args.eval_every_environment_steps

    final_evaluation = {
        "win_rate": curves[-1]["validation_win_rate"],
        "mean_payoff": curves[-1]["validation_mean_payoff"],
        "selection_key": [curves[-1]["validation_worst_cell_payoff"], curves[-1]["validation_mean_payoff"]],
    }
    learner.save(final_path, checkpoint_extra("final", final_evaluation))
    if best_evaluation is None or best_steps is None:
        raise RuntimeError("no trained checkpoint was eligible for validation selection")
    final_learner_steps = learner.learner_step
    learner.load(best_path, map_location=device)
    selected_checkpoint_learner_steps = learner.learner_step
    heldout, heldout_records = evaluate_refil(
        runner,
        controller,
        scales,
        opponent_styles,
        args.heldout_episodes_per_scale_opponent,
        heldout_seed,
        args.num_envs,
    )
    _write_evaluation(
        output_dir,
        heldout,
        heldout_records,
        scales,
        opponent_styles,
    )
    elapsed = time.perf_counter() - started
    payload = {
        "schema_version": "round-01-refil-qmix-had-v1",
        "status": "completed",
        "algorithm": "REFIL-QMIX-HAD-adaptation",
        "exact_upstream_claim": False,
        "upstream_basis": "REFIL official architecture/config with HAD-specific observations/actions and discrete curriculum adapter",
        "seed": seed,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "environment_steps": environment_steps,
        "episodes": episode_count,
        "learner_steps": final_learner_steps,
        "selected_checkpoint_learner_steps": selected_checkpoint_learner_steps,
        "elapsed_seconds": elapsed,
        "throughput_environment_steps_per_second": environment_steps / max(elapsed, 1e-9),
        "initial_validation": initial,
        "best_validation": best_evaluation,
        "best_environment_steps": best_steps,
        "heldout_evaluation": heldout,
        "scale_environment_steps": {_scale_label(key): value for key, value in scale_steps.items()},
        "scale_episodes": {_scale_label(key): value for key, value in scale_episodes.items()},
        "opponent_episodes": opponent_episodes,
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": _sha256(best_path),
        "final_checkpoint": str(final_path),
        "final_checkpoint_sha256": _sha256(final_path),
        "resumed_from": resumed_from,
        "curriculum": curriculum.state_dict(),
        "git_provenance": dict(git_provenance),
        "arguments": {key: value for key, value in vars(args).items() if not key.startswith("_")},
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    progress_path.write_text(
        json.dumps(_jsonable({"status": "completed", "curves": curves}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(_jsonable(payload), ensure_ascii=False), flush=True)
    return payload
