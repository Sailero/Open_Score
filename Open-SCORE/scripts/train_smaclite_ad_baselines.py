"""Real small-budget QMIX/VDN/MAPPO validation on dynamic SMAClite-AD.

This entry point is deliberately separate from the HAD reproduction CLI.  Its
tracked output is evidence of executable updates and paired short-budget
evaluation, never a formal convergence claim.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import platform
import random
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from open_score.envs.smaclite_ad import PROTOCOL_ID, UPSTREAM_COMMIT
from open_score.stage1.baselines import (
    SequenceMAPPOLearner,
    VariableScaleMAPPO,
    VariableScaleVDN,
)
from open_score.stage1.entity_qmix import VariableScaleQMIX
from open_score.stage1.learner import SequenceQMIXLearner, linear_epsilon
from open_score.stage1.replay import EpisodeReplayBuffer, collate_episodes
from open_score.stage1.smaclite_ad_training import (
    Ratio,
    SMACliteADEpisodeRunner,
    SMACliteADFactory,
    SMACliteADMAPPOController,
    SMACliteADQController,
    SMACliteADRuleController,
    evaluate_smaclite_ad,
    tensor_shape_audit,
)


def parse_ratios(value: str) -> List[Ratio]:
    result = []
    for item in value.split(","):
        try:
            red, blue = (int(part) for part in item.strip().split(":"))
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(
                "ratios must look like 2:1,3:2,5:3"
            ) from exc
        if red < 1 or blue < 1:
            raise argparse.ArgumentTypeError("ratio counts must be positive")
        result.append((red, blue))
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("ratios must be non-empty and unique")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithms", nargs="+", choices=("qmix", "vdn", "mappo"),
        default=["qmix", "vdn", "mappo"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260830])
    parser.add_argument("--ratios", type=parse_ratios, default=parse_ratios("2:1,3:2,5:3"))
    parser.add_argument("--train-side", choices=("Red", "Blue"), default="Red")
    parser.add_argument(
        "--opponent", choices=("idle", "intercept", "rush_asset"), default="idle"
    )
    parser.add_argument("--episodes", type=int, default=45)
    parser.add_argument("--episode-limit", type=int, default=60)
    parser.add_argument("--batch-episodes", type=int, default=3)
    parser.add_argument("--replay-episodes", type=int, default=96)
    parser.add_argument("--updates-per-episode", type=int, default=1)
    parser.add_argument("--validation-episodes-per-ratio", type=int, default=5)
    parser.add_argument("--heldout-episodes-per-ratio", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--agent-hidden-dim", type=int, default=32)
    parser.add_argument("--critic-hidden-dim", type=int, default=32)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--epsilon-start", type=float, default=0.90)
    parser.add_argument("--epsilon-finish", type=float, default=0.10)
    parser.add_argument("--epsilon-anneal-steps", type=int, default=2_000)
    parser.add_argument("--shaping-scale", type=float, default=0.50)
    parser.add_argument("--approach-weight", type=float, default=1.0)
    parser.add_argument("--spawn-jitter", type=float, default=2.0)
    parser.add_argument("--max-red-agents", type=int, default=6)
    parser.add_argument("--max-blue-agents", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/smaclite_ad_baselines")
    )
    parser.add_argument("--evidence-dir", type=Path, default=Path("docs/evidence"))
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "episodes",
        "episode_limit",
        "batch_episodes",
        "replay_episodes",
        "updates_per_episode",
        "validation_episodes_per_ratio",
        "heldout_episodes_per_ratio",
        "eval_every",
        "agent_hidden_dim",
        "critic_hidden_dim",
        "ppo_epochs",
    )
    if any(getattr(args, name) < 1 for name in positive):
        raise ValueError("episode, batch, evaluation and model sizes must be positive")
    if args.validation_episodes_per_ratio < 5 or args.heldout_episodes_per_ratio < 5:
        raise ValueError("validation and held-out evaluation need at least 5 layouts per ratio")
    if args.spawn_jitter <= 0.0:
        raise ValueError("training evidence must explicitly enable spawn_jitter")
    if args.batch_episodes < len(args.ratios):
        raise ValueError("batch_episodes must cover every registered ratio")
    if args.replay_episodes < args.batch_episodes:
        raise ValueError("replay_episodes must be at least batch_episodes")
    if args.max_red_agents < max(red for red, _ in args.ratios):
        raise ValueError("max_red_agents is smaller than a registered ratio")
    if args.max_blue_agents < max(blue for _, blue in args.ratios):
        raise ValueError("max_blue_agents is smaller than a registered ratio")
    if args.train_side == "Blue" and args.opponent == "intercept":
        raise ValueError("a Red opponent should use rush_asset or idle")
    if args.train_side == "Red" and args.opponent == "rush_asset":
        raise ValueError("a Blue opponent should use intercept or idle")


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(algorithm: str, env, device: torch.device, args: argparse.Namespace):
    dimensions = (
        env.ENTITY_DIM,
        env.SELF_DIM,
        env.TASK_DIM,
        env.STATE_ENTITY_DIM,
        env.ACTION_DIM,
    )
    if algorithm == "qmix":
        model = VariableScaleQMIX(
            *dimensions,
            agent_hidden_dim=args.agent_hidden_dim,
            mixer_hidden_dim=args.critic_hidden_dim,
            mixing_dim=max(8, args.critic_hidden_dim // 2),
        )
    elif algorithm == "vdn":
        model = VariableScaleVDN(
            *dimensions, agent_hidden_dim=args.agent_hidden_dim
        )
    elif algorithm == "mappo":
        model = VariableScaleMAPPO(
            *dimensions,
            actor_hidden_dim=args.agent_hidden_dim,
            critic_hidden_dim=args.critic_hidden_dim,
        )
    else:  # pragma: no cover - argparse guards this
        raise ValueError(algorithm)
    return model.to(device)


def model_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_vector(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat(
        [parameter.detach().cpu().reshape(-1) for parameter in model.parameters()]
    )


def evaluation_dict(evaluation, *, include_layouts: bool = False) -> Dict[str, object]:
    payload = asdict(evaluation)
    if not include_layouts:
        payload.pop("layout_records", None)
    return _jsonify(payload)


def learning_metric_fields(metrics) -> Dict[str, object]:
    names = (
        "loss", "policy_loss", "value_loss", "entropy", "approximate_kl",
        "clip_fraction", "mean_absolute_td", "grad_norm", "learner_step"
    )
    return {
        name: getattr(metrics, name, None) if metrics is not None else None
        for name in names
    }


def paired_comparison(initial, final) -> Dict[str, object]:
    initial_returns = np.asarray(initial.paired_returns, dtype=np.float64)
    final_returns = np.asarray(final.paired_returns, dtype=np.float64)
    if initial_returns.shape != final_returns.shape:
        raise AssertionError("paired evaluation seed sets differ")
    if initial.layout_hashes != final.layout_hashes:
        raise AssertionError("paired evaluation layouts differ")
    differences = final_returns - initial_returns
    ratio_deltas = {
        label: {
            "mean_return_delta": float(final.per_ratio[label]["mean_return"])
            - float(initial.per_ratio[label]["mean_return"]),
            "win_rate_delta": float(final.per_ratio[label]["win_rate"])
            - float(initial.per_ratio[label]["win_rate"]),
            "mean_payoff_delta": float(final.per_ratio[label]["mean_payoff"])
            - float(initial.per_ratio[label]["mean_payoff"]),
        }
        for label in initial.per_ratio
    }
    improved_ratios = sum(
        values["mean_return_delta"] > 1e-6 or values["win_rate_delta"] > 0.0
        for values in ratio_deltas.values()
    )
    learning_signal = (
        final.win_rate > initial.win_rate
        or (
            float(differences.mean()) > 0.02
            and improved_ratios >= max(1, len(ratio_deltas) - 1)
        )
    )
    return {
        "same_ordered_seeds": True,
        "same_ordered_layouts": True,
        "effective_unique_layouts": int(initial.unique_layouts),
        "mean_return_delta": float(differences.mean()),
        "median_return_delta": float(np.median(differences)),
        "positive_pair_fraction": float(np.mean(differences > 1e-6)),
        "win_rate_delta": float(final.win_rate - initial.win_rate),
        "mean_payoff_delta": float(final.mean_payoff - initial.mean_payoff),
        "per_ratio": ratio_deltas,
        "learning_status": (
            "small_budget_learning_signal_observed"
            if learning_signal
            else "learning_not_established"
        ),
        "task_success_status": (
            "win_rate_improvement_observed"
            if final.win_rate > initial.win_rate
            else "win_rate_not_improved"
        ),
    }


def train_one(
    algorithm: str,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    seed_everything(seed)
    factory = SMACliteADFactory(
        max_red_agents=args.max_red_agents,
        max_blue_agents=args.max_blue_agents,
        episode_limit=args.episode_limit,
        shaping_scale=args.shaping_scale,
        approach_weight=args.approach_weight,
        spawn_jitter=args.spawn_jitter,
    )
    runner = SMACliteADEpisodeRunner(factory)
    shape_audit = tensor_shape_audit(factory, args.ratios, seed + 700_000)
    model = make_model(algorithm, factory.get(args.ratios[0]), device, args)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    initial_hash = model_sha256(model)
    initial_parameters = parameter_vector(model)
    initial_model_state = copy.deepcopy(model.state_dict())
    opponent = SMACliteADRuleController(args.opponent)
    if algorithm == "mappo":
        learner = SequenceMAPPOLearner(
            model,
            learning_rate=args.learning_rate,
            epochs=args.ppo_epochs,
        )
        training_controller = SMACliteADMAPPOController(
            model, device, deterministic=False, name="mappo_train"
        )
        evaluation_controller = SMACliteADMAPPOController(
            model, device, deterministic=True, name="mappo_eval"
        )
        pending = []
        replay = None
    else:
        learner = SequenceQMIXLearner(
            model,
            learning_rate=args.learning_rate,
            target_update_interval=25,
        )
        training_controller = SMACliteADQController(
            model, device, epsilon=args.epsilon_start, name=f"{algorithm}_train"
        )
        evaluation_controller = SMACliteADQController(
            model, device, epsilon=0.0, name=f"{algorithm}_eval"
        )
        replay = EpisodeReplayBuffer(args.replay_episodes, seed=seed + 29)
        pending = None

    validation_seed = seed + 8_000_000
    heldout_seed = seed + 18_000_000
    validation_initial = evaluate_smaclite_ad(
        runner,
        evaluation_controller,
        opponent,
        args.train_side,
        args.ratios,
        args.validation_episodes_per_ratio,
        validation_seed,
    )
    curves: List[Dict[str, object]] = []
    recent_returns: deque = deque(maxlen=max(6, args.eval_every))
    recent_wins: deque = deque(maxlen=max(6, args.eval_every))
    ratio_episode_counts = {f"{red}:{blue}": 0 for red, blue in args.ratios}
    # Keep the ordered training schedule in the tracked result, rather than only
    # retaining an in-memory set used for the leakage assertion.  A third party
    # can therefore recompute train-vs-evaluation intersections without rerunning
    # the learner, and can reconstruct each deterministic reset from its seed.
    training_layout_hashes = {f"{red}:{blue}": [] for red, blue in args.ratios}
    training_layout_schedule: List[Dict[str, object]] = []
    training_layout_manifest: List[Dict[str, object]] = []
    updated_ratios = set()
    latest_metrics = None
    environment_steps = 0
    started = time.perf_counter()
    best_checkpoint_path = (
        args.output_dir / "checkpoints" / f"{algorithm}_seed{seed}_validation_best.pt"
    )
    best_validation_key = None
    best_validation_record = None
    best_validation_evaluation = None
    best_model_state = None
    last_validation_evaluation = None
    last_validation_episode = None

    def add_curve(episode: int, evaluation) -> None:
        row = {
            "algorithm": algorithm,
            "seed": seed,
            "episode": episode,
            "environment_steps": environment_steps,
            "learner_updates": (
                0 if latest_metrics is None else latest_metrics.learner_step
            ),
            "train_mean_return_window": (
                None if not recent_returns else float(np.mean(recent_returns))
            ),
            "train_win_rate_window": (
                None if not recent_wins else float(np.mean(recent_wins))
            ),
            "eval_mean_return": evaluation.mean_return,
            "eval_win_rate": evaluation.win_rate,
            "eval_mean_payoff": evaluation.mean_payoff,
            "eval_mean_asset_health": evaluation.mean_asset_health,
            "eval_per_ratio_json": json.dumps(evaluation.per_ratio, sort_keys=True),
            "epsilon": (
                training_controller.epsilon
                if isinstance(training_controller, SMACliteADQController)
                else None
            ),
        }
        row.update(learning_metric_fields(latest_metrics))
        curves.append(row)

    def consider_validation_best(episode: int, evaluation) -> None:
        nonlocal best_validation_key
        nonlocal best_validation_record
        nonlocal best_validation_evaluation
        nonlocal best_model_state
        if episode <= 0:
            return
        key = (float(evaluation.win_rate), float(evaluation.mean_return))
        if best_validation_key is not None and key <= best_validation_key:
            return
        best_validation_key = key
        best_validation_evaluation = copy.deepcopy(evaluation)
        best_model_state = copy.deepcopy(model.state_dict())
        metadata = {
            "checkpoint_role": "validation_best",
            "selection_metric": ["win_rate", "mean_return"],
            "selection_split": "validation",
            "algorithm": algorithm,
            "seed": seed,
            "episode": episode,
            "environment_steps": environment_steps,
            "learner_updates": (
                0 if latest_metrics is None else latest_metrics.learner_step
            ),
            "validation_seed_base": validation_seed,
            "validation_episodes_per_ratio": args.validation_episodes_per_ratio,
            "validation_unique_layouts": evaluation.unique_layouts,
            "ratios": [list(ratio) for ratio in args.ratios],
            "spawn_jitter": args.spawn_jitter,
            "train_side": args.train_side,
        }
        learner.save(best_checkpoint_path, metadata)
        best_validation_record = {
            **metadata,
            "mean_return": float(evaluation.mean_return),
            "win_rate": float(evaluation.win_rate),
            "checkpoint": str(best_checkpoint_path.resolve()),
            "checkpoint_sha256": file_sha256(best_checkpoint_path),
            "model_sha256": model_sha256(model),
        }

    add_curve(0, validation_initial)
    for episode_index in range(1, args.episodes + 1):
        ratio = args.ratios[(episode_index - 1) % len(args.ratios)]
        ratio_episode_counts[f"{ratio[0]}:{ratio[1]}"] += 1
        if isinstance(training_controller, SMACliteADQController):
            training_controller.epsilon = linear_epsilon(
                environment_steps,
                start=args.epsilon_start,
                finish=args.epsilon_finish,
                anneal_steps=args.epsilon_anneal_steps,
            )
        rollout_seed = seed + episode_index * 101
        rollout = (
            runner.run(ratio, training_controller, opponent, rollout_seed)
            if args.train_side == "Red"
            else runner.run(ratio, opponent, training_controller, rollout_seed)
        )
        team_episode = rollout.red if args.train_side == "Red" else rollout.blue
        ratio_label = f"{ratio[0]}:{ratio[1]}"
        layout = dict(rollout.final_info["layout"])
        layout_hash = str(layout["layout_sha256"])
        if layout_hash != str(rollout.final_info["layout_hash"]):
            raise AssertionError("layout hash differs between layout record and info")
        if int(layout["seed"]) != rollout_seed:
            raise AssertionError("layout record does not contain the scheduled seed")
        training_layout_hashes[ratio_label].append(layout_hash)
        training_layout_schedule.append(
            {
                "episode_index": episode_index,
                "ratio": ratio_label,
                "seed": rollout_seed,
                "accepted_attempt": int(layout["accepted_attempt"]),
                "layout_sha256": layout_hash,
                "sample_sha256": str(layout["sample_sha256"]),
                "randomization_config_sha256": str(
                    layout["randomization_config_sha256"]
                ),
            }
        )
        training_layout_manifest.append(
            {
                "episode_index": episode_index,
                "ratio": ratio_label,
                **copy.deepcopy(layout),
            }
        )
        controlled_payoff = (
            rollout.outcome_red if args.train_side == "Red" else -rollout.outcome_red
        )
        environment_steps += rollout.length
        recent_returns.append(float(team_episode.rewards.sum()))
        recent_wins.append(float(controlled_payoff > 0.0))

        if algorithm == "mappo":
            pending.append(team_episode)
            if len(pending) >= args.batch_episodes:
                latest_metrics = learner.train_batch(collate_episodes(pending, device))
                updated_ratios.update(latest_metrics.learning_signal_by_scale)
                pending.clear()
        else:
            replay.add(team_episode)
            if len(replay) >= args.batch_episodes:
                for _ in range(args.updates_per_episode):
                    if learner.learner_step == 0:
                        batch = collate_episodes(
                            replay.episodes[-args.batch_episodes :], device
                        )
                    else:
                        batch = replay.sample(args.batch_episodes, device)
                    latest_metrics = learner.train_batch(batch)
                    updated_ratios.update(latest_metrics.td_by_scale)

        if episode_index % args.eval_every == 0:
            evaluation = evaluate_smaclite_ad(
                runner,
                evaluation_controller,
                opponent,
                args.train_side,
                args.ratios,
                args.validation_episodes_per_ratio,
                validation_seed,
            )
            add_curve(episode_index, evaluation)
            consider_validation_best(episode_index, evaluation)
            last_validation_evaluation = evaluation
            last_validation_episode = episode_index

    model_changed_after_last_validation = False
    if algorithm == "mappo" and pending:
        latest_metrics = learner.train_batch(collate_episodes(pending, device))
        updated_ratios.update(latest_metrics.learning_signal_by_scale)
        pending.clear()
        model_changed_after_last_validation = True
    if last_validation_episode == args.episodes and not model_changed_after_last_validation:
        validation_final = last_validation_evaluation
    else:
        validation_final = evaluate_smaclite_ad(
            runner,
            evaluation_controller,
            opponent,
            args.train_side,
            args.ratios,
            args.validation_episodes_per_ratio,
            validation_seed,
        )
    assert validation_final is not None
    if curves[-1]["episode"] == args.episodes:
        curves.pop()
    add_curve(args.episodes, validation_final)
    consider_validation_best(args.episodes, validation_final)
    if best_model_state is None or best_validation_record is None:
        raise AssertionError("validation-best checkpoint was not created")
    training_validation_elapsed = time.perf_counter() - started
    final_model_state = copy.deepcopy(model.state_dict())
    final_hash = model_sha256(model)
    parameter_delta = float(torch.linalg.vector_norm(parameter_vector(model) - initial_parameters))
    final_checkpoint_path = (
        args.output_dir / "checkpoints" / f"{algorithm}_seed{seed}_final.pt"
    )
    final_metadata = {
        "checkpoint_role": "final",
        "environment": "OpenSCORE/SMACliteAD-Asset-v0",
        "algorithm": algorithm,
        "seed": seed,
        "episode": args.episodes,
        "environment_steps": environment_steps,
        "learner_updates": 0 if latest_metrics is None else latest_metrics.learner_step,
        "ratios": [list(ratio) for ratio in args.ratios],
        "spawn_jitter": args.spawn_jitter,
        "train_side": args.train_side,
    }
    learner.save(
        final_checkpoint_path,
        final_metadata,
    )
    final_checkpoint = {
        **final_metadata,
        "checkpoint": str(final_checkpoint_path.resolve()),
        "checkpoint_sha256": file_sha256(final_checkpoint_path),
        "model_sha256": final_hash,
    }

    # Held-out layouts are not touched during training or checkpoint selection.
    # The evaluation sweep happens once, after both checkpoint identities have
    # been frozen, using identical ordered layouts for the three policies.
    model.load_state_dict(initial_model_state)
    heldout_initial = evaluate_smaclite_ad(
        runner,
        evaluation_controller,
        opponent,
        args.train_side,
        args.ratios,
        args.heldout_episodes_per_ratio,
        heldout_seed,
    )
    model.load_state_dict(best_model_state)
    heldout_best = evaluate_smaclite_ad(
        runner,
        evaluation_controller,
        opponent,
        args.train_side,
        args.ratios,
        args.heldout_episodes_per_ratio,
        heldout_seed,
    )
    if best_validation_record["model_sha256"] == final_hash:
        heldout_final = heldout_best
    else:
        model.load_state_dict(final_model_state)
        heldout_final = evaluate_smaclite_ad(
            runner,
            evaluation_controller,
            opponent,
            args.train_side,
            args.ratios,
            args.heldout_episodes_per_ratio,
            heldout_seed,
        )
    model.load_state_dict(final_model_state)
    for split_name, evaluation, expected_per_ratio in (
        ("validation_initial", validation_initial, args.validation_episodes_per_ratio),
        ("validation_final", validation_final, args.validation_episodes_per_ratio),
        ("heldout_initial", heldout_initial, args.heldout_episodes_per_ratio),
        ("heldout_best", heldout_best, args.heldout_episodes_per_ratio),
        ("heldout_final", heldout_final, args.heldout_episodes_per_ratio),
    ):
        for ratio_label, values in evaluation.per_ratio.items():
            if int(values["unique_layouts"]) != expected_per_ratio:
                raise AssertionError(
                    f"{split_name} {ratio_label} has {values['unique_layouts']} "
                    f"unique layouts, expected {expected_per_ratio}"
                )
    validation_comparison = paired_comparison(validation_initial, validation_final)
    heldout_best_comparison = paired_comparison(heldout_initial, heldout_best)
    heldout_final_comparison = paired_comparison(heldout_initial, heldout_final)
    for label, hashes in training_layout_hashes.items():
        if len(set(hashes)) != len(hashes):
            raise AssertionError(
                f"training split {label} has duplicate randomized layouts"
            )
    training_layout_set = {
        layout_hash
        for ratio_hashes in training_layout_hashes.values()
        for layout_hash in ratio_hashes
    }
    validation_layout_set = set(validation_initial.layout_hashes)
    heldout_layout_set = set(heldout_initial.layout_hashes)
    layout_intersections = {
        "train_validation": sorted(training_layout_set & validation_layout_set),
        "train_heldout": sorted(training_layout_set & heldout_layout_set),
        "validation_heldout": sorted(validation_layout_set & heldout_layout_set),
    }
    if any(layout_intersections.values()):
        raise AssertionError(f"layout split leakage detected: {layout_intersections}")
    elapsed = time.perf_counter() - started
    result = {
        "algorithm": algorithm,
        "seed": seed,
        "train_side": args.train_side,
        "opponent": opponent.name,
        "parameter_count": parameter_count,
        "episodes": args.episodes,
        "environment_steps": environment_steps,
        "learner_updates": 0 if latest_metrics is None else latest_metrics.learner_step,
        "elapsed_seconds": elapsed,
        "training_validation_elapsed_seconds": training_validation_elapsed,
        "environment_steps_per_second": environment_steps / max(training_validation_elapsed, 1e-9),
        "ratio_episode_counts": ratio_episode_counts,
        "training_unique_layouts_per_ratio": {
            label: len(set(hashes)) for label, hashes in training_layout_hashes.items()
        },
        "training_layout_hashes_per_ratio": training_layout_hashes,
        "training_layout_schedule": training_layout_schedule,
        "training_layout_manifest": training_layout_manifest,
        "training_layout_evidence": {
            "capture_method": "inline_from_rollout_final_info",
            "seed_formula": "training_seed + episode_index * 101 (1-based)",
            "episodes_recorded": len(training_layout_schedule),
            "ordered_schedule_complete": True,
        },
        "layout_split_audit": {
            "disjoint": True,
            "intersections": layout_intersections,
            "training_unique_layouts": len(training_layout_set),
            "validation_unique_layouts": len(validation_layout_set),
            "heldout_unique_layouts": len(heldout_layout_set),
        },
        "ratios_seen_by_gradient_updates": [
            f"{ratio[0]}:{ratio[1]}" for ratio in sorted(updated_ratios)
        ],
        "tensor_shape_audit": shape_audit,
        "initial_model_sha256": initial_hash,
        "final_model_sha256": final_hash,
        "parameters_changed": final_hash != initial_hash and parameter_delta > 0.0,
        "parameter_delta_l2": parameter_delta,
        "validation": {
            "seed_base": validation_seed,
            "episodes_per_ratio": args.validation_episodes_per_ratio,
            "initial": evaluation_dict(validation_initial),
            "final": evaluation_dict(validation_final),
            "final_comparison": validation_comparison,
            "best_checkpoint_evaluation": evaluation_dict(best_validation_evaluation),
            "layout_manifest": _jsonify(validation_initial.layout_records),
        },
        "heldout": {
            "seed_base": heldout_seed,
            "episodes_per_ratio": args.heldout_episodes_per_ratio,
            "evaluated_only_after_checkpoint_selection": True,
            "initial": evaluation_dict(heldout_initial),
            "validation_best": evaluation_dict(heldout_best),
            "final": evaluation_dict(heldout_final),
            "validation_best_comparison": heldout_best_comparison,
            "final_comparison": heldout_final_comparison,
            "layout_manifest": _jsonify(heldout_initial.layout_records),
        },
        "paired_comparison": heldout_best_comparison,
        "best_recorded_evaluation": best_validation_record,
        "latest_learning_metrics": (
            None if latest_metrics is None else _jsonify(asdict(latest_metrics))
        ),
        "checkpoints": {
            "validation_best": best_validation_record,
            "final": final_checkpoint,
            "distinct_files": (
                best_checkpoint_path.resolve() != final_checkpoint_path.resolve()
            ),
        },
    }
    factory.close()
    return result, curves


def _jsonify(value):
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonify(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_curves(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("no curve rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_factory = SMACliteADFactory(
        max_red_agents=args.max_red_agents,
        max_blue_agents=args.max_blue_agents,
        episode_limit=args.episode_limit,
        shaping_scale=args.shaping_scale,
        approach_weight=args.approach_weight,
        spawn_jitter=args.spawn_jitter,
    )
    reference_runner = SMACliteADEpisodeRunner(reference_factory)
    reference_controlled = SMACliteADRuleController(
        "rush_asset" if args.train_side == "Red" else "intercept"
    )
    reference_opponent = SMACliteADRuleController(args.opponent)
    rule_reference = evaluate_smaclite_ad(
        reference_runner,
        reference_controlled,
        reference_opponent,
        args.train_side,
        args.ratios,
        args.validation_episodes_per_ratio,
        int(args.seeds[0]) + 28_000_000,
    )
    reference_factory.close()
    results = []
    curves = []
    for seed in args.seeds:
        for algorithm in args.algorithms:
            result, rows = train_one(algorithm, seed, args, device)
            results.append(result)
            curves.extend(rows)
            print(
                f"{algorithm} seed={seed}: updates={result['learner_updates']} "
                f"delta={result['paired_comparison']['mean_return_delta']:.6f} "
                f"status={result['paired_comparison']['learning_status']}"
            )
    payload = _jsonify(
        {
            "schema_version": "smaclite-ad-baseline-smoke-v2",
            "status": "randomized_layout_small_budget_training_validation",
            "formal_convergence_claim": False,
            "environment": "OpenSCORE/SMACliteAD-Asset-v0",
            "environment_provenance": {
                "protocol_id": PROTOCOL_ID,
                "upstream_commit": UPSTREAM_COMMIT,
                "use_cpp_rvo2": False,
                "environment_source_sha256": file_sha256(
                    Path(__file__).resolve().parents[1]
                    / "src"
                    / "open_score"
                    / "envs"
                    / "smaclite_ad.py"
                ),
                "layout_hash_specification": (
                    "SHA-256 of canonical sorted compact JSON over geometry; "
                    "coordinates/radii rounded to 6 decimals; seed excluded"
                ),
                "sample_hash_specification": (
                    "SHA-256 of canonical layout JSON including seed, accepted "
                    "attempt and layout_sha256"
                ),
            },
            "environment_roles": {"Red": "asset attacker", "Blue": "asset defender"},
            "shared_policy_across_ratios": True,
            "registered_ratios": [list(ratio) for ratio in args.ratios],
            "paired_evaluation": (
                "same ordered randomized layouts within a split; validation and "
                "one post-selection held-out sweep use disjoint layout hashes"
            ),
            "legacy_fixed_layout_result": {
                "status": "deterministic_pipeline_smoke_only",
                "reason": (
                    "spawn_jitter=0 produced one effective layout per ratio even when "
                    "multiple physics seeds were requested"
                ),
            },
            "rule_opponent_uses_primitive_actions_only": True,
            "rule_reference": {
                "purpose": "solvability reference, not a learned baseline",
                "controlled_policy": reference_controlled.name,
                "opponent_policy": reference_opponent.name,
                "evaluation": evaluation_dict(rule_reference, include_layouts=True),
            },
            "device": str(device),
            "machine": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "gpu": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                ),
            },
            "arguments": {
                key: (
                    str(value)
                    if isinstance(value, Path)
                    else [list(item) for item in value]
                    if key == "ratios"
                    else value
                )
                for key, value in vars(args).items()
            },
            "results": results,
            "interpretation": (
                "A changed checkpoint and finite learner metrics prove real updates. "
                "Learning status is read from the disjoint held-out evaluation of the "
                "validation-selected checkpoint. No label establishes convergence."
            ),
        }
    )
    summary_path = args.evidence_dir / "smaclite_ad_baselines_summary.json"
    curve_path = args.evidence_dir / "smaclite_ad_baselines_curve.csv"
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_curves(curve_path, curves)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_curves(args.output_dir / "training_curve.csv", curves)
    print(f"wrote {summary_path.resolve()}")
    print(f"wrote {curve_path.resolve()}")


if __name__ == "__main__":
    main()
