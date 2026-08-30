"""Train/evaluate rule, VDN, QMIX and MAPPO on strict-red-superior HAD.

The defaults are a small-budget engineering check, not paper-level convergence
evidence.  One checkpoint is shared across all six registered roster ratios.
"""

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from open_score.envs import HADStage1Adapter
from open_score.stage1 import (
    CompetitiveEpisodeRunner,
    EpisodeReplayBuffer,
    HADStage1Factory,
    LearningProgressCurriculum,
    MAPPOController,
    QMixController,
    RuleBasedController,
    SequenceMAPPOLearner,
    SequenceQMIXLearner,
    VariableScaleMAPPO,
    VariableScaleVDN,
    collate_episodes,
    evaluate_pair,
    linear_epsilon,
    make_had_qmix,
    supported_scales,
)

Scale = Tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=("qmix", "vdn", "mappo"),
        default=("qmix", "vdn", "mappo"),
    )
    parser.add_argument("--train-side", choices=("Red", "Blue"), default="Red")
    parser.add_argument(
        "--fixed-scale",
        nargs=2,
        type=int,
        metavar=("RED", "BLUE"),
        default=None,
        help="Train/evaluate one registered scale instead of the curriculum.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seeds", nargs="+", type=int, default=(20260830,))
    parser.add_argument("--episodes", type=int, default=72)
    parser.add_argument("--batch-episodes", type=int, default=4)
    parser.add_argument("--replay-episodes", type=int, default=256)
    parser.add_argument("--updates-per-episode", type=int, default=1)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--shaping-scale", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=12)
    parser.add_argument("--eval-episodes-per-scale", type=int, default=2)
    parser.add_argument("--episodes-per-stage", type=int, default=12)
    parser.add_argument("--epsilon-anneal-steps", type=int, default=2_000)
    parser.add_argument("--agent-hidden-dim", type=int, default=64)
    parser.add_argument("--critic-hidden-dim", type=int, default=64)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT / "outputs" / "stage1_had_reproduction",
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=PROJECT / "docs" / "evidence",
    )
    parser.add_argument("--evidence-prefix", default="stage1_had")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume one algorithm/seed from a checkpoint, including optimizer.",
    )
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "episodes": args.episodes,
        "batch_episodes": args.batch_episodes,
        "replay_episodes": args.replay_episodes,
        "updates_per_episode": args.updates_per_episode,
        "ppo_epochs": args.ppo_epochs,
        "max_steps": args.max_steps,
        "eval_every": args.eval_every,
        "eval_episodes_per_scale": args.eval_episodes_per_scale,
        "episodes_per_stage": args.episodes_per_stage,
    }
    invalid = [name for name, value in positive.items() if value < 1]
    if invalid:
        raise ValueError(f"positive values required for: {', '.join(invalid)}")
    if args.batch_episodes > args.replay_episodes:
        raise ValueError("batch_episodes cannot exceed replay_episodes")
    if args.shaping_scale < 0.0:
        raise ValueError("shaping_scale must be non-negative")
    if args.fixed_scale is not None and tuple(args.fixed_scale) not in supported_scales(4, 1):
        raise ValueError("fixed_scale must satisfy strict Red superiority within 1--4")
    if args.resume is not None:
        if len(args.algorithms) != 1 or len(args.seeds) != 1:
            raise ValueError("--resume requires exactly one algorithm and one seed")
        if not args.resume.is_file():
            raise ValueError(f"resume checkpoint does not exist: {args.resume}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(algorithm: str, device: torch.device, args: argparse.Namespace):
    if algorithm == "qmix":
        return make_had_qmix(
            device,
            agent_hidden_dim=args.agent_hidden_dim,
            mixer_hidden_dim=args.critic_hidden_dim,
            mixing_dim=max(8, args.critic_hidden_dim // 2),
        )
    if algorithm == "vdn":
        return VariableScaleVDN(
            HADStage1Adapter.ENTITY_DIM,
            HADStage1Adapter.SELF_DIM,
            HADStage1Adapter.TASK_DIM,
            HADStage1Adapter.STATE_ENTITY_DIM,
            HADStage1Adapter.ACTION_DIM,
            agent_hidden_dim=args.agent_hidden_dim,
        ).to(device)
    if algorithm == "mappo":
        return VariableScaleMAPPO(
            HADStage1Adapter.ENTITY_DIM,
            HADStage1Adapter.SELF_DIM,
            HADStage1Adapter.TASK_DIM,
            HADStage1Adapter.STATE_ENTITY_DIM,
            HADStage1Adapter.ACTION_DIM,
            actor_hidden_dim=args.agent_hidden_dim,
            critic_hidden_dim=args.critic_hidden_dim,
        ).to(device)
    raise ValueError(f"unknown algorithm {algorithm}")


def controlled_evaluation(
    runner: CompetitiveEpisodeRunner,
    controlled,
    side: str,
    scales: Sequence[Scale],
    episodes_per_scale: int,
    seed: int,
) -> Dict[str, object]:
    if side == "Red":
        summary = evaluate_pair(
            runner,
            controlled,
            RuleBasedController("rush"),
            scales,
            episodes_per_scale,
            seed,
        )
        sign = 1.0
        win_rate = summary.defender_win_rate
    else:
        summary = evaluate_pair(
            runner,
            RuleBasedController("guard"),
            controlled,
            scales,
            episodes_per_scale,
            seed,
        )
        sign = -1.0
        win_rate = summary.attacker_win_rate
    return {
        "controlled_mean_payoff": sign * summary.mean_defender_payoff,
        "controlled_win_rate": win_rate,
        "mean_episode_length": summary.mean_episode_length,
        "episodes": summary.episodes,
        "per_scale_controlled_payoff": {
            f"{red}v{blue}": sign * value
            for (red, blue), value in summary.per_scale_payoff.items()
        },
    }


def learning_fields(metrics: Optional[object]) -> Dict[str, object]:
    fields = {
        "loss": None,
        "policy_loss": None,
        "value_loss": None,
        "entropy": None,
        "mean_absolute_td": None,
        "grad_norm": None,
        "approximate_kl": None,
        "clip_fraction": None,
        "learner_step": 0,
    }
    if metrics is None:
        return fields
    for name in fields:
        if hasattr(metrics, name):
            fields[name] = getattr(metrics, name)
    return fields


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_one(
    algorithm: str,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    scales: Sequence[Scale],
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    seed_everything(seed)
    runner = CompetitiveEpisodeRunner(
        HADStage1Factory(
            max_steps=args.max_steps,
            shaping_scale=args.shaping_scale,
        )
    )
    model = make_model(algorithm, device, args)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    curriculum = LearningProgressCurriculum(
        max_agents=4,
        minimum_red_advantage=1,
        episodes_per_stage=args.episodes_per_stage,
        uniform_coverage=0.20,
    )
    opponent = (
        RuleBasedController("rush")
        if args.train_side == "Red"
        else RuleBasedController("guard")
    )
    if algorithm == "mappo":
        learner = SequenceMAPPOLearner(
            model,
            learning_rate=args.learning_rate,
            epochs=args.ppo_epochs,
        )
        training_controller = MAPPOController(
            model, device, deterministic=False, name=f"{algorithm}_train"
        )
        evaluation_controller = MAPPOController(
            model, device, deterministic=True, name=f"{algorithm}_eval"
        )
        pending = []
        replay = None
    else:
        learner = SequenceQMIXLearner(
            model,
            learning_rate=args.learning_rate,
            target_update_interval=50,
        )
        training_controller = QMixController(
            model, device, epsilon=1.0, name=f"{algorithm}_train"
        )
        evaluation_controller = QMixController(
            model, device, epsilon=0.0, name=f"{algorithm}_eval"
        )
        replay = EpisodeReplayBuffer(args.replay_episodes, seed=seed + 29)
        pending = None

    resume_extra: Mapping[str, object] = {}
    prior_episodes = 0
    environment_steps = 0
    if args.resume is not None:
        resume_extra = learner.load(args.resume, map_location=device)
        if resume_extra.get("algorithm") != algorithm:
            raise ValueError("resume checkpoint algorithm does not match request")
        if resume_extra.get("train_side") != args.train_side:
            raise ValueError("resume checkpoint train_side does not match request")
        reward_protocol = resume_extra.get("reward_protocol", {})
        saved_scale = reward_protocol.get("potential_shaping_scale")
        if saved_scale is not None and float(saved_scale) != args.shaping_scale:
            raise ValueError("resume checkpoint shaping scale does not match request")
        environment_protocol = resume_extra.get("environment_protocol", {})
        saved_max_steps = environment_protocol.get("max_steps")
        if saved_max_steps is not None and int(saved_max_steps) != args.max_steps:
            raise ValueError("resume checkpoint max_steps does not match request")
        saved_gamma = environment_protocol.get("gamma")
        if saved_gamma is not None and float(saved_gamma) != 0.99:
            raise ValueError("resume checkpoint gamma does not match HAD protocol")
        prior_episodes = int(
            resume_extra.get(
                "cumulative_episodes", resume_extra.get("episodes", 0)
            )
        )
        environment_steps = int(resume_extra.get("environment_steps", 0))
        if "curriculum" in resume_extra:
            curriculum.load_state_dict(resume_extra["curriculum"])
    rng = np.random.default_rng(seed + 19 + prior_episodes)
    if isinstance(training_controller, QMixController):
        training_controller.epsilon = linear_epsilon(
            environment_steps,
            start=1.0,
            finish=0.05,
            anneal_steps=args.epsilon_anneal_steps,
        )

    evaluation_seed = seed + 8_000_000
    initial = controlled_evaluation(
        runner,
        evaluation_controller,
        args.train_side,
        scales,
        args.eval_episodes_per_scale,
        evaluation_seed,
    )
    curves: List[Dict[str, object]] = []
    recent_payoffs: deque = deque(maxlen=max(6, args.eval_every))
    recent_returns: deque = deque(maxlen=max(6, args.eval_every))
    latest_metrics = None
    initial_environment_steps = environment_steps
    initial_learner_step = int(learner.learner_step)
    started = time.perf_counter()

    def checkpoint_metadata(
        cumulative_episodes: int,
        selection: str,
        evaluation: Optional[Mapping[str, object]] = None,
    ) -> Dict[str, object]:
        return {
            "algorithm": algorithm,
            "seed": seed,
            "train_side": args.train_side,
            "registered_scales": [list(scale) for scale in scales],
            "strict_red_superiority": True,
            "protocol_version": "had-stage1-strict-red-v2",
            "episodes": cumulative_episodes,
            "cumulative_episodes": cumulative_episodes,
            "environment_steps": environment_steps,
            "curriculum": curriculum.state_dict(),
            "selection": selection,
            "selection_evaluation": None if evaluation is None else dict(evaluation),
            "contract": {
                "entity_dim": HADStage1Adapter.ENTITY_DIM,
                "self_dim": HADStage1Adapter.SELF_DIM,
                "task_dim": HADStage1Adapter.TASK_DIM,
                "state_entity_dim": HADStage1Adapter.STATE_ENTITY_DIM,
                "action_dim": HADStage1Adapter.ACTION_DIM,
            },
            "model_config": {
                "agent_hidden_dim": args.agent_hidden_dim,
                "critic_or_mixer_hidden_dim": args.critic_hidden_dim,
                "mixing_dim": max(8, args.critic_hidden_dim // 2),
            },
            "environment_protocol": {
                "max_steps": args.max_steps,
                "gamma": 0.99,
                "registered_scales": [list(scale) for scale in scales],
                "strict_red_superiority": True,
            },
            "continuation_contract": {
                "restores": [
                    "online_model",
                    "target_model_when_applicable",
                    "optimizer",
                    "learner_step",
                    "curriculum_state",
                    "environment_step_counter",
                ],
                "does_not_restore": [
                    "replay_buffer",
                    "python_rng",
                    "numpy_rollout_rng",
                    "torch_rng",
                ],
                "semantics": "warm_resume_not_bit_exact_continuation",
            },
            "reward_protocol": {
                "terminal_outcome": "plus_or_minus_one",
                "potential_shaping_scale": args.shaping_scale,
                "terminal_potential_is_zero": True,
                "clearance_weight": 0.5,
                "intercept_weight": 0.25,
            },
        }

    def record_curve(episode: int, evaluation: Mapping[str, object]) -> None:
        row = {
            "algorithm": algorithm,
            "seed": seed,
            "episode": episode,
            "environment_steps": environment_steps,
            "train_mean_payoff_window": (
                None if not recent_payoffs else float(np.mean(recent_payoffs))
            ),
            "train_win_rate_window": (
                None
                if not recent_payoffs
                else float(np.mean(np.asarray(recent_payoffs) > 0.0))
            ),
            "train_mean_return_window": (
                None if not recent_returns else float(np.mean(recent_returns))
            ),
            "eval_controlled_mean_payoff": evaluation["controlled_mean_payoff"],
            "eval_controlled_win_rate": evaluation["controlled_win_rate"],
            "eval_mean_episode_length": evaluation["mean_episode_length"],
            "eval_per_scale_json": json.dumps(
                evaluation["per_scale_controlled_payoff"], sort_keys=True
            ),
            "epsilon": (
                training_controller.epsilon
                if isinstance(training_controller, QMixController)
                else None
            ),
        }
        row.update(learning_fields(latest_metrics))
        curves.append(row)

    checkpoint_directory = args.output_dir / "checkpoints"
    best_checkpoint_path = checkpoint_directory / f"{algorithm}_seed{seed}_best.pt"
    best_evaluation = dict(initial)
    best_episode = prior_episodes
    learner.save(
        best_checkpoint_path,
        checkpoint_metadata(prior_episodes, "paired_eval_best", initial),
    )

    def consider_best(episode: int, evaluation: Mapping[str, object]) -> None:
        nonlocal best_evaluation, best_episode
        score = float(evaluation["controlled_mean_payoff"])
        best_score = float(best_evaluation["controlled_mean_payoff"])
        if score > best_score:
            best_evaluation = dict(evaluation)
            best_episode = episode
            learner.save(
                best_checkpoint_path,
                checkpoint_metadata(episode, "paired_eval_best", evaluation),
            )

    record_curve(prior_episodes, initial)
    for episode_index in range(1, args.episodes + 1):
        cumulative_episode = prior_episodes + episode_index
        if args.fixed_scale is None:
            scale = curriculum.sample(rng)
        else:
            scale = tuple(args.fixed_scale)
            curriculum.record_sample(scale)
        if isinstance(training_controller, QMixController):
            training_controller.epsilon = linear_epsilon(
                environment_steps,
                start=1.0,
                finish=0.05,
                anneal_steps=args.epsilon_anneal_steps,
            )
        rollout_seed = seed + cumulative_episode * 101
        if args.train_side == "Red":
            rollout = runner.run(
                scale, training_controller, opponent, rollout_seed
            )
            team_episode = rollout.red
            controlled_payoff = rollout.outcome_red
        else:
            rollout = runner.run(
                scale, opponent, training_controller, rollout_seed
            )
            team_episode = rollout.blue
            controlled_payoff = -rollout.outcome_red
        environment_steps += rollout.length
        recent_payoffs.append(float(controlled_payoff))
        recent_returns.append(float(team_episode.rewards.sum()))
        curriculum.record_episode()

        if algorithm == "mappo":
            pending.append(team_episode)
            if len(pending) >= args.batch_episodes:
                latest_metrics = learner.train_batch(
                    collate_episodes(pending, device)
                )
                pending.clear()
                for updated_scale, signal in (
                    latest_metrics.learning_signal_by_scale.items()
                ):
                    curriculum.update(updated_scale, signal)
        else:
            replay.add(team_episode)
            if len(replay) >= args.batch_episodes:
                for _ in range(args.updates_per_episode):
                    latest_metrics = learner.train_batch(
                        replay.sample(args.batch_episodes, device)
                    )
                for updated_scale, signal in latest_metrics.td_by_scale.items():
                    curriculum.update(updated_scale, signal)

        if episode_index % args.eval_every == 0 or episode_index == args.episodes:
            evaluation = controlled_evaluation(
                runner,
                evaluation_controller,
                args.train_side,
                scales,
                args.eval_episodes_per_scale,
                evaluation_seed,
            )
            record_curve(cumulative_episode, evaluation)
            consider_best(cumulative_episode, evaluation)

    if algorithm == "mappo" and pending:
        latest_metrics = learner.train_batch(collate_episodes(pending, device))
        pending.clear()
        # Re-evaluate because the final on-policy partial batch changed weights.
        final = controlled_evaluation(
            runner,
            evaluation_controller,
            args.train_side,
            scales,
            args.eval_episodes_per_scale,
            evaluation_seed,
        )
        final_episode = prior_episodes + args.episodes
        if curves[-1]["episode"] == final_episode:
            curves.pop()
        record_curve(final_episode, final)
        consider_best(final_episode, final)
    else:
        final = controlled_evaluation(
            runner,
            evaluation_controller,
            args.train_side,
            scales,
            args.eval_episodes_per_scale,
            evaluation_seed,
        )
        final_episode = prior_episodes + args.episodes
        if curves[-1]["episode"] == final_episode:
            curves.pop()
        record_curve(final_episode, final)
        consider_best(final_episode, final)

    checkpoint_path = args.output_dir / "checkpoints" / f"{algorithm}_seed{seed}.pt"
    learner.save(
        checkpoint_path,
        checkpoint_metadata(final_episode, "final", final),
    )
    elapsed = time.perf_counter() - started
    best_eval = max(
        float(row["eval_controlled_mean_payoff"]) for row in curves
    )
    result = {
        "algorithm": algorithm,
        "seed": seed,
        "train_side": args.train_side,
        "parameter_count": parameter_count,
        "episodes_this_run": args.episodes,
        "cumulative_episodes": final_episode,
        "resumed_from": None if args.resume is None else str(args.resume.resolve()),
        "initial_environment_steps": initial_environment_steps,
        "environment_steps": environment_steps,
        "environment_steps_this_run": environment_steps - initial_environment_steps,
        "updates": 0 if latest_metrics is None else latest_metrics.learner_step,
        "initial_learner_step": initial_learner_step,
        "updates_this_run": (
            0
            if latest_metrics is None
            else int(latest_metrics.learner_step) - initial_learner_step
        ),
        "elapsed_seconds": elapsed,
        "throughput_environment_steps_per_second": (
            environment_steps - initial_environment_steps
        ) / max(elapsed, 1e-9),
        "initial_evaluation": initial,
        "final_evaluation": final,
        "best_curve_mean_payoff": best_eval,
        "best_evaluation": best_evaluation,
        "best_episode": best_episode,
        "small_budget_improvement": (
            float(final["controlled_mean_payoff"])
            - float(initial["controlled_mean_payoff"])
        ),
        "learning_status": (
            "small_budget_improvement_observed"
            if float(final["controlled_mean_payoff"])
            > float(initial["controlled_mean_payoff"])
            else "learning_not_established"
        ),
        "latest_learning_metrics": (
            None if latest_metrics is None else _jsonify(asdict(latest_metrics))
        ),
        "curriculum": curriculum.state_dict(),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
        "best_checkpoint": str(best_checkpoint_path.resolve()),
        "best_checkpoint_sha256": checkpoint_sha256(best_checkpoint_path),
    }
    return result, curves


def _jsonify(value):
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_curves(path: Path, curves: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not curves:
        raise ValueError("no curves to write")
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = choose_device(args.device)
    scales = (
        [tuple(args.fixed_scale)]
        if args.fixed_scale is not None
        else supported_scales(4, 1)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runner = CompetitiveEpisodeRunner(
        HADStage1Factory(
            max_steps=args.max_steps,
            shaping_scale=args.shaping_scale,
        )
    )
    rule_evaluation = controlled_evaluation(
        runner,
        RuleBasedController("guard"),
        "Red",
        scales,
        args.eval_episodes_per_scale,
        int(args.seeds[0]) + 8_000_000,
    )
    results = []
    curves: List[Dict[str, object]] = []
    for seed in args.seeds:
        for algorithm in args.algorithms:
            result, algorithm_curves = train_one(
                algorithm, seed, args, device, scales
            )
            results.append(result)
            curves.extend(algorithm_curves)

    payload = _jsonify(
        {
            "schema_version": "stage1-had-reproduction-v1",
            "status": "small_budget_training_check",
            "formal_convergence_claim": False,
            "reason": (
                "This run verifies executable learning and short-budget trends; "
                "formal claims require preregistered multi-seed budgets and intervals."
            ),
            "environment": "HADStage1Adapter",
            "roles": {"Red": "asset defender", "Blue": "asset attacker"},
            "firing_semantics": "any HAD AttackAgent self-destructs after firing",
            "strict_red_superiority": True,
            "registered_scales": [list(scale) for scale in scales],
            "shared_policy_architecture": True,
            "trained_across_registered_scales": args.fixed_scale is None,
            "training_mode": "fixed_scale" if args.fixed_scale is not None else "curriculum",
            "action_space": "27 acceleration primitives; no macro tactics",
            "reward_protocol": {
                "terminal_outcome": "plus_or_minus_one",
                "potential_shaping_scale": args.shaping_scale,
                "terminal_potential_is_zero": True,
                "potential_features": [
                    "target_health",
                    "team_survival",
                    "attacker_target_clearance",
                    "defender_attacker_intercept_distance",
                ],
            },
            "rule_strategy": {
                "Red": "guard",
                "Blue": "rush",
                "evaluation": rule_evaluation,
            },
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": (
                torch.cuda.get_device_name(0) if device.type == "cuda" else None
            ),
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "results": results,
        }
    )
    summary_path = args.output_dir / "summary.json"
    curves_path = args.output_dir / "training_curves.csv"
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_curves(curves_path, curves)
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence_summary = args.evidence_dir / f"{args.evidence_prefix}_summary.json"
    evidence_curves = args.evidence_dir / f"{args.evidence_prefix}_training_curves.csv"
    evidence_summary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_curves(evidence_curves, curves)
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
