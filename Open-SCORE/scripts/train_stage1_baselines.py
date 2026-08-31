"""Train/evaluate rule, VDN, QMIX and MAPPO on strict-red-superior HAD.

The defaults are a small-budget engineering check, not paper-level convergence
evidence.  One checkpoint is shared across all six registered roster ratios.
"""

import argparse
import copy
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
from open_score.formal_contracts import validate_registered_formal_contract
from open_score.provenance import collect_and_require_git_provenance
from open_score.stage1 import (
    CompetitiveEpisodeRunner,
    DEMONSTRATION_SEED_OFFSET,
    DEMONSTRATION_TRAINING_SEED_STRIDE,
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
    collect_guard_demonstrations,
    collate_episodes,
    evaluation_seed_set,
    evaluate_pair,
    linear_epsilon,
    make_had_qmix,
    model_state_sha256,
    supported_scales,
    synchronize_q_target_after_behavior_cloning,
    train_recurrent_behavior_clone,
)

Scale = Tuple[int, int]
HAD_FORMAL_CONFIG = PROJECT / "configs" / "stage1_had_reproduction.yaml"
HAD_FORMAL_PROTOCOL = "stage1-had-reproduction-v5-hybrid-preservation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=("qmix", "vdn", "mappo", "refil_qmix"),
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
    parser.add_argument("--total-environment-steps", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--batch-episodes", type=int, default=4)
    parser.add_argument("--replay-episodes", type=int, default=256)
    parser.add_argument("--updates-per-episode", type=int, default=1)
    parser.add_argument("--target-update-interval", type=int, default=200)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--td-lambda", type=float, default=0.6)
    parser.add_argument(
        "--optimizer", choices=("adam", "rmsprop"), default="adam"
    )
    parser.add_argument("--rmsprop-alpha", type=float, default=0.99)
    parser.add_argument("--rmsprop-eps", type=float, default=1e-5)
    parser.add_argument("--shaping-scale", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=12)
    parser.add_argument("--eval-episodes-per-scale", type=int, default=2)
    parser.add_argument("--heldout-episodes-per-scale", type=int, default=8)
    parser.add_argument("--eval-every-environment-steps", type=int, default=50_000)
    parser.add_argument("--eval-episodes-per-scale-opponent", type=int, default=10)
    parser.add_argument("--heldout-episodes-per-scale-opponent", type=int, default=100)
    parser.add_argument(
        "--opponent-styles",
        nargs="+",
        choices=("guard", "intercept", "rush", "split_rush", "engage"),
        default=None,
        help=(
            "Robust opponent suite. Defaults to rush+split_rush when training "
            "Red, or guard+intercept when training Blue."
        ),
    )
    parser.add_argument(
        "--formal-evidence",
        action="store_true",
        help="Enforce the preregistered minimum seed/evaluation protocol.",
    )
    parser.add_argument("--episodes-per-stage", type=int, default=12)
    parser.add_argument("--epsilon-anneal-steps", type=int, default=2_000)
    parser.add_argument("--agent-hidden-dim", type=int, default=64)
    parser.add_argument("--critic-hidden-dim", type=int, default=64)
    parser.add_argument(
        "--encoder-kind", choices=("deepset", "saqa"), default="deepset"
    )
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--attention-embed-dim", type=int, default=128)
    parser.add_argument("--mixing-dim", type=int, default=32)
    parser.add_argument("--hypernet-hidden-dim", type=int, default=128)
    parser.add_argument("--refil-imagine-weight", type=float, default=0.5)
    parser.add_argument(
        "--curriculum",
        choices=("legacy_episode", "step_learning_progress"),
        default="legacy_episode",
    )
    parser.add_argument(
        "--curriculum-boundaries",
        nargs=4,
        type=int,
        default=(100_000, 250_000, 400_000, 850_000),
    )
    parser.add_argument("--curriculum-update-steps", type=int, default=10_000)
    parser.add_argument("--curriculum-td-window", type=int, default=50)
    parser.add_argument("--curriculum-uniform-floor", type=float, default=0.30)
    parser.add_argument(
        "--curriculum-max-scale-probability", type=float, default=0.40
    )
    parser.add_argument(
        "--bc-demo-episodes-per-scale-opponent",
        type=int,
        default=0,
        help=(
            "Enable train-only rule:guard demonstration pretraining with this "
            "many episodes in every scale-by-opponent cell; zero disables BC."
        ),
    )
    parser.add_argument(
        "--bc-algorithms",
        nargs="+",
        choices=("qmix", "vdn", "mappo"),
        default=("qmix",),
        help="Algorithms receiving BC; others remain pure-RL baselines.",
    )
    parser.add_argument("--bc-epochs", type=int, default=8)
    parser.add_argument("--bc-batch-episodes", type=int, default=8)
    parser.add_argument("--bc-learning-rate", type=float, default=1e-3)
    parser.add_argument("--hybrid-rl-learning-rate", type=float, default=1e-4)
    parser.add_argument("--hybrid-epsilon-start", type=float, default=0.20)
    parser.add_argument("--hybrid-epsilon-finish", type=float, default=0.02)
    parser.add_argument(
        "--hybrid-minimum-rl-updates",
        type=int,
        default=100,
        help=(
            "Minimum cumulative RL learner updates before a BC+RL checkpoint "
            "may enter validation-best selection."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to outputs/stage1_had_formal under "
            "--formal-evidence, otherwise outputs/stage1_had_reproduction."
        ),
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=PROJECT / "outputs" / "stage1_had_evidence",
    )
    parser.add_argument(
        "--evidence-prefix",
        default=None,
        help=(
            "Evidence filename prefix. Defaults to stage1_had_formal under "
            "--formal-evidence, otherwise stage1_had."
        ),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume one algorithm/seed from a checkpoint, including optimizer.",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = PROJECT / "outputs" / (
            "stage1_had_formal"
            if args.formal_evidence
            else "stage1_had_reproduction"
        )
    if args.evidence_prefix is None:
        args.evidence_prefix = (
            "stage1_had_formal" if args.formal_evidence else "stage1_had"
        )
    return args


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def validate_args(args: argparse.Namespace) -> None:
    args._formal_contract = None
    positive = {
        "episodes": args.episodes,
        "batch_episodes": args.batch_episodes,
        "replay_episodes": args.replay_episodes,
        "updates_per_episode": args.updates_per_episode,
        "target_update_interval": args.target_update_interval,
        "ppo_epochs": args.ppo_epochs,
        "max_steps": args.max_steps,
        "eval_every": args.eval_every,
        "eval_episodes_per_scale": args.eval_episodes_per_scale,
        "heldout_episodes_per_scale": args.heldout_episodes_per_scale,
        "episodes_per_stage": args.episodes_per_stage,
        "attention_heads": args.attention_heads,
        "bc_epochs": args.bc_epochs,
        "bc_batch_episodes": args.bc_batch_episodes,
        "hybrid_minimum_rl_updates": args.hybrid_minimum_rl_updates,
    }
    invalid = [name for name, value in positive.items() if value < 1]
    if invalid:
        raise ValueError(f"positive values required for: {', '.join(invalid)}")
    if args.batch_episodes > args.replay_episodes:
        raise ValueError("batch_episodes cannot exceed replay_episodes")
    if args.encoder_kind == "saqa" and args.agent_hidden_dim % args.attention_heads:
        raise ValueError("attention_heads must divide agent_hidden_dim for SAQA")
    if args.shaping_scale < 0.0:
        raise ValueError("shaping_scale must be non-negative")
    if args.bc_demo_episodes_per_scale_opponent < 0:
        raise ValueError("bc_demo_episodes_per_scale_opponent must be non-negative")
    if args.bc_learning_rate <= 0.0:
        raise ValueError("bc_learning_rate must be positive")
    if args.hybrid_rl_learning_rate <= 0.0:
        raise ValueError("hybrid_rl_learning_rate must be positive")
    if not (
        0.0 <= args.hybrid_epsilon_finish
        <= args.hybrid_epsilon_start
        <= 1.0
    ):
        raise ValueError(
            "hybrid epsilon requires 0 <= finish <= start <= 1"
        )
    if len(args.bc_algorithms) != len(set(args.bc_algorithms)):
        raise ValueError("bc_algorithms must be unique")
    if args.bc_demo_episodes_per_scale_opponent > 0:
        if not set(args.bc_algorithms) <= set(args.algorithms):
            raise ValueError("bc_algorithms must be a subset of algorithms")
        if args.train_side != "Red":
            raise ValueError("guard-demonstration BC supports only train_side=Red")
        if args.resume is not None:
            raise ValueError("resume never repeats BC; set demo episodes to zero")
    if args.fixed_scale is not None and tuple(args.fixed_scale) not in supported_scales(4, 1):
        raise ValueError("fixed_scale must satisfy strict Red superiority within 1--4")
    if args.resume is not None:
        if len(args.algorithms) != 1 or len(args.seeds) != 1:
            raise ValueError("--resume requires exactly one algorithm and one seed")
        if not args.resume.is_file():
            raise ValueError(f"resume checkpoint does not exist: {args.resume}")
    styles = resolve_opponent_styles(args)
    defender_styles = {"guard", "intercept", "engage"}
    attacker_styles = {"rush", "split_rush"}
    allowed = attacker_styles if args.train_side == "Red" else defender_styles
    if not set(styles) <= allowed:
        raise ValueError(
            f"opponent styles for train_side={args.train_side} must be in {sorted(allowed)}"
        )
    if args.formal_evidence:
        if args.encoder_kind != "saqa":
            raise ValueError("formal evidence requires --encoder-kind saqa")
        if len(args.algorithms) != 3 or set(args.algorithms) != {
            "qmix",
            "vdn",
            "mappo",
        }:
            raise ValueError(
                "formal evidence requires qmix, vdn and mappo in one invocation"
            )
        if (
            args.train_side != "Red"
            or len(styles) != 2
            or set(styles) != {"rush", "split_rush"}
        ):
            raise ValueError(
                "formal evidence requires Red against rush and split_rush"
            )
        if len(args.seeds) != len(set(args.seeds)):
            raise ValueError("formal evidence requires unique training seeds")
        if len(set(args.seeds)) < 5:
            raise ValueError("formal evidence requires at least five unique seeds")
        if args.heldout_episodes_per_scale < 20:
            raise ValueError(
                "formal evidence requires at least 20 held-out episodes per scale/opponent"
            )
        if args.fixed_scale is not None:
            raise ValueError("formal evidence must cover every registered scale")
        if (
            args.bc_demo_episodes_per_scale_opponent > 0
            and tuple(args.bc_algorithms) != ("qmix",)
        ):
            raise ValueError(
                "formal BC designates only QMIX as hybrid primary; VDN/MAPPO "
                "must remain pure-RL replication baselines"
            )
        args._formal_contract = validate_had_formal_contract(args)


def validate_had_formal_contract(args: argparse.Namespace) -> Dict[str, object]:
    """Require an exact match to the tracked HAD formal experiment contract."""

    actual = {
        "training_seeds": list(args.seeds),
        "formal_seed_count": len(set(args.seeds)),
        "registered_scales": [
            f"{red}v{blue}" for red, blue in supported_scales(4, 1)
        ],
        "max_steps": args.max_steps,
        "reward_gamma": 0.99,
        "shaping_scale": args.shaping_scale,
        "red_rule": "guard",
        "blue_rule": "rush",
        "primary_algorithm": "qmix",
        "primary_training_regime": (
            "hybrid_rule_demonstration_pretraining_plus_RL_finetuning"
            if args.bc_demo_episodes_per_scale_opponent > 0
            else "pure_RL_without_rule_demonstration_pretraining"
        ),
        "bc_enabled": args.bc_demo_episodes_per_scale_opponent > 0,
        "bc_teacher": "guard",
        "bc_algorithms": list(args.bc_algorithms),
        "bc_demo_episodes_per_scale_opponent": (
            args.bc_demo_episodes_per_scale_opponent
        ),
        "bc_epochs": args.bc_epochs,
        "bc_batch_episodes": args.bc_batch_episodes,
        "bc_learning_rate": args.bc_learning_rate,
        "hybrid_rl_learning_rate": args.hybrid_rl_learning_rate,
        "hybrid_epsilon_start": args.hybrid_epsilon_start,
        "hybrid_epsilon_finish": args.hybrid_epsilon_finish,
        "hybrid_epsilon_anneal_steps": args.epsilon_anneal_steps,
        "hybrid_minimum_rl_updates": args.hybrid_minimum_rl_updates,
        "bc_seed_offset": DEMONSTRATION_SEED_OFFSET,
        "bc_training_seed_stride": DEMONSTRATION_TRAINING_SEED_STRIDE,
        "algorithms": list(args.algorithms),
        "opponent_styles": list(resolve_opponent_styles(args)),
        "train_side": args.train_side,
        "episodes": args.episodes,
        "batch_episodes": args.batch_episodes,
        "replay_episodes": args.replay_episodes,
        "updates_per_episode": args.updates_per_episode,
        "target_update_interval": args.target_update_interval,
        "ppo_epochs": args.ppo_epochs,
        "learning_rate": args.learning_rate,
        "eval_every": args.eval_every,
        "eval_episodes_per_scale": args.eval_episodes_per_scale,
        "heldout_episodes_per_scale_per_opponent": (
            args.heldout_episodes_per_scale
        ),
        "episodes_per_stage": args.episodes_per_stage,
        "epsilon_anneal_steps": args.epsilon_anneal_steps,
        "epsilon_start": 1.0,
        "epsilon_finish": 0.05,
        "agent_hidden_dim": args.agent_hidden_dim,
        "critic_hidden_dim": args.critic_hidden_dim,
        "encoder_kind": args.encoder_kind,
        "attention_heads": args.attention_heads,
        "curriculum_uniform_coverage": 0.20,
        "replication_algorithms": [
            algorithm
            for algorithm in args.algorithms
            if not (
                args.bc_demo_episodes_per_scale_opponent > 0
                and algorithm in args.bc_algorithms
            )
        ],
        "replication_validation_improvement_threshold": 0.0,
        "primary_gate_algorithm": "qmix",
        "primary_gate_rule": "guard",
        "primary_delta_ci_threshold": 0.0,
        "primary_every_scale_delta_threshold": 0.0,
    }
    paths = {
        "training_seeds": "seed_policy.formal",
        "formal_seed_count": "budgets.formal_minimum.seeds",
        "registered_scales": "environment.registered_scales",
        "max_steps": "environment.max_steps",
        "reward_gamma": "environment.reward.gamma",
        "shaping_scale": "environment.reward.potential_shaping_scale",
        "red_rule": "strategies.rule.Red",
        "blue_rule": "strategies.rule.Blue",
        "primary_algorithm": "strategies.intelligent.primary_algorithm",
        "primary_training_regime": "strategies.intelligent.training_regime",
        "bc_enabled": "strategies.intelligent.demonstration_pretraining.enabled",
        "bc_teacher": "strategies.intelligent.demonstration_pretraining.teacher",
        "bc_algorithms": (
            "strategies.intelligent.demonstration_pretraining.algorithms"
        ),
        "bc_demo_episodes_per_scale_opponent": (
            "budgets.formal_minimum.bc_demo_episodes_per_scale_opponent"
        ),
        "bc_epochs": "budgets.formal_minimum.bc_epochs",
        "bc_batch_episodes": "budgets.formal_minimum.bc_batch_episodes",
        "bc_learning_rate": "budgets.formal_minimum.bc_learning_rate",
        "hybrid_rl_learning_rate": (
            "budgets.formal_minimum.hybrid_preservation.rl_learning_rate"
        ),
        "hybrid_epsilon_start": (
            "budgets.formal_minimum.hybrid_preservation.epsilon_start"
        ),
        "hybrid_epsilon_finish": (
            "budgets.formal_minimum.hybrid_preservation.epsilon_finish"
        ),
        "hybrid_epsilon_anneal_steps": (
            "budgets.formal_minimum.hybrid_preservation.epsilon_anneal_steps"
        ),
        "hybrid_minimum_rl_updates": (
            "budgets.formal_minimum.hybrid_preservation."
            "minimum_rl_updates_for_checkpoint_selection"
        ),
        "bc_seed_offset": (
            "strategies.intelligent.demonstration_pretraining.seed_offset"
        ),
        "bc_training_seed_stride": (
            "strategies.intelligent.demonstration_pretraining.training_seed_stride"
        ),
        "algorithms": "executed_algorithms",
        "opponent_styles": "baselines.opponent_suite",
        "train_side": "baselines.train_side",
        "episodes": "budgets.formal_minimum.episodes",
        "batch_episodes": "budgets.formal_minimum.batch_episodes",
        "replay_episodes": "budgets.formal_minimum.replay_episodes",
        "updates_per_episode": "budgets.formal_minimum.updates_per_episode",
        "target_update_interval": "budgets.formal_minimum.target_update_interval",
        "ppo_epochs": "budgets.formal_minimum.ppo_epochs",
        "learning_rate": "budgets.formal_minimum.learning_rate",
        "eval_every": "budgets.formal_minimum.eval_every",
        "eval_episodes_per_scale": "budgets.formal_minimum.eval_episodes_per_scale",
        "heldout_episodes_per_scale_per_opponent": (
            "budgets.formal_minimum.heldout_episodes_per_scale_per_opponent"
        ),
        "episodes_per_stage": "budgets.formal_minimum.episodes_per_stage",
        "epsilon_anneal_steps": "budgets.formal_minimum.epsilon_anneal_steps",
        "epsilon_start": "budgets.formal_minimum.epsilon_start",
        "epsilon_finish": "budgets.formal_minimum.epsilon_finish",
        "agent_hidden_dim": "budgets.formal_minimum.agent_hidden_dim",
        "critic_hidden_dim": "budgets.formal_minimum.critic_hidden_dim",
        "encoder_kind": "budgets.formal_minimum.encoder_kind",
        "attention_heads": "budgets.formal_minimum.attention_heads",
        "curriculum_uniform_coverage": (
            "strategies.intelligent.curriculum.uniform_coverage"
        ),
        "replication_algorithms": (
            "budgets.formal_minimum.baseline_replication_gate.algorithms"
        ),
        "replication_validation_improvement_threshold": (
            "budgets.formal_minimum.baseline_replication_gate."
            "validation_payoff_improvement_mean_gt"
        ),
        "primary_gate_algorithm": (
            "budgets.formal_minimum.primary_strategy_gate.algorithm"
        ),
        "primary_gate_rule": (
            "budgets.formal_minimum.primary_strategy_gate.rule_reference"
        ),
        "primary_delta_ci_threshold": (
            "budgets.formal_minimum.primary_strategy_gate."
            "seed_bootstrap_delta_ci95_low_gt"
        ),
        "primary_every_scale_delta_threshold": (
            "budgets.formal_minimum.primary_strategy_gate."
            "every_scale_mean_delta_ge"
        ),
    }
    return validate_registered_formal_contract(
        project_root=PROJECT,
        config_path=HAD_FORMAL_CONFIG,
        contract_name="stage1_had_formal",
        expected_protocol_version=HAD_FORMAL_PROTOCOL,
        actual_values=actual,
        yaml_paths=paths,
        unordered_fields=(
            "algorithms",
            "opponent_styles",
            "replication_algorithms",
            "bc_algorithms",
        ),
    )


def resolve_opponent_styles(args: argparse.Namespace) -> Tuple[str, ...]:
    if args.opponent_styles is not None:
        return tuple(args.opponent_styles)
    return ("rush", "split_rush") if args.train_side == "Red" else (
        "guard",
        "intercept",
    )


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
            encoder_kind=args.encoder_kind,
            attention_heads=args.attention_heads,
        )
    if algorithm == "vdn":
        return VariableScaleVDN(
            HADStage1Adapter.ENTITY_DIM,
            HADStage1Adapter.SELF_DIM,
            HADStage1Adapter.TASK_DIM,
            HADStage1Adapter.STATE_ENTITY_DIM,
            HADStage1Adapter.ACTION_DIM,
            agent_hidden_dim=args.agent_hidden_dim,
            encoder_kind=args.encoder_kind,
            attention_heads=args.attention_heads,
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
            encoder_kind=args.encoder_kind,
            attention_heads=args.attention_heads,
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


def controlled_evaluation_suite(
    runner: CompetitiveEpisodeRunner,
    controlled,
    side: str,
    scales: Sequence[Scale],
    episodes_per_scale: int,
    seed: int,
    opponent_styles: Sequence[str],
) -> Dict[str, object]:
    """Evaluate one checkpoint against a fixed, auditable rule suite."""

    if not opponent_styles:
        raise ValueError("opponent suite cannot be empty")
    components = {}
    for index, style in enumerate(opponent_styles):
        opponent = RuleBasedController(style)
        if side == "Red":
            summary = evaluate_pair(
                runner,
                controlled,
                opponent,
                scales,
                episodes_per_scale,
                seed + index * 100_000,
            )
            sign = 1.0
            win_rate = summary.defender_win_rate
        else:
            summary = evaluate_pair(
                runner,
                opponent,
                controlled,
                scales,
                episodes_per_scale,
                seed + index * 100_000,
            )
            sign = -1.0
            win_rate = summary.attacker_win_rate
        components[style] = {
            "controlled_mean_payoff": sign * summary.mean_defender_payoff,
            "controlled_win_rate": win_rate,
            "mean_episode_length": summary.mean_episode_length,
            "episodes": summary.episodes,
            "per_scale_controlled_payoff": {
                f"{red}v{blue}": sign * value
                for (red, blue), value in summary.per_scale_payoff.items()
            },
            "seed_base": seed + index * 100_000,
        }
    return {
        "controlled_mean_payoff": float(
            np.mean([value["controlled_mean_payoff"] for value in components.values()])
        ),
        "controlled_win_rate": float(
            np.mean([value["controlled_win_rate"] for value in components.values()])
        ),
        "mean_episode_length": float(
            np.mean([value["mean_episode_length"] for value in components.values()])
        ),
        "episodes": int(sum(value["episodes"] for value in components.values())),
        "per_scale_controlled_payoff": {
            f"{red}v{blue}": float(
                np.mean(
                    [
                        value["per_scale_controlled_payoff"][f"{red}v{blue}"]
                        for value in components.values()
                    ]
                )
            )
            for red, blue in scales
        },
        "opponent_styles": list(opponent_styles),
        "components": components,
    }


def robust_validation_key(evaluation: Mapping[str, object]) -> Tuple[float, float]:
    """Worst scale-by-opponent payoff first, overall mean payoff second."""

    components = evaluation.get("components")
    if not isinstance(components, Mapping) or not components:
        raise ValueError("robust selection requires non-empty opponent components")
    cell_payoffs: List[float] = []
    for component in components.values():
        if not isinstance(component, Mapping):
            raise TypeError("opponent evaluation component must be a mapping")
        per_scale = component.get("per_scale_controlled_payoff")
        if not isinstance(per_scale, Mapping) or not per_scale:
            raise ValueError("robust selection requires per-scale payoff cells")
        cell_payoffs.extend(float(value) for value in per_scale.values())
    return (
        min(cell_payoffs),
        float(evaluation["controlled_mean_payoff"]),
    )


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


def tensor_state_is_finite(state: Mapping[str, torch.Tensor]) -> bool:
    """Return whether every stored model tensor contains only finite values."""

    return bool(state) and all(
        bool(torch.isfinite(tensor.detach()).all().item()) for tensor in state.values()
    )


def numeric_tree_is_finite(value: object) -> bool:
    """Validate numeric learner evidence while ignoring descriptive strings."""

    if isinstance(value, Mapping):
        return bool(value) and all(numeric_tree_is_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return bool(value) and all(numeric_tree_is_finite(item) for item in value)
    if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
        return bool(np.isfinite(float(value)))
    return value is not None


def rl_optimizer_state_entry_counts(learner: object, algorithm: str) -> Dict[str, int]:
    """Expose whether BC accidentally contaminated a registered RL optimizer."""

    if algorithm == "mappo":
        return {
            "actor": len(learner.actor_optimizer.state),
            "critic": len(learner.critic_optimizer.state),
        }
    return {"q_learner": len(learner.optimizer.state)}


def requested_fresh_hybrid(args: argparse.Namespace, algorithm: str) -> bool:
    return bool(
        args.resume is None
        and args.bc_demo_episodes_per_scale_opponent > 0
        and algorithm in args.bc_algorithms
    )


def q_epsilon_schedule(
    behavior_cloning: Mapping[str, object],
    environment_steps: int,
    args: argparse.Namespace,
) -> Tuple[float, Dict[str, object]]:
    hybrid = (
        behavior_cloning.get("policy_training_regime")
        == "rule_demonstration_bc_then_rl"
    )
    start = args.hybrid_epsilon_start if hybrid else 1.0
    finish = args.hybrid_epsilon_finish if hybrid else 0.05
    epsilon = linear_epsilon(
        environment_steps,
        start=start,
        finish=finish,
        anneal_steps=args.epsilon_anneal_steps,
    )
    return epsilon, {
        "schedule_role": (
            "hybrid_policy_preservation" if hybrid else "pure_rl_exploration"
        ),
        "start": start,
        "finish": finish,
        "anneal_steps": args.epsilon_anneal_steps,
        "epsilon": epsilon,
    }


def rl_optimizer_learning_rates(learner: object, algorithm: str) -> Dict[str, float]:
    if algorithm == "mappo":
        return {
            "actor": float(learner.actor_optimizer.param_groups[0]["lr"]),
            "critic": float(learner.critic_optimizer.param_groups[0]["lr"]),
        }
    return {"q_learner": float(learner.optimizer.param_groups[0]["lr"])}


def hybrid_checkpoint_selection_eligibility(
    behavior_cloning: Mapping[str, object],
    learner_step: int,
    minimum_rl_updates: int,
) -> Dict[str, object]:
    hybrid = (
        behavior_cloning.get("policy_training_regime")
        == "rule_demonstration_bc_then_rl"
    )
    required = minimum_rl_updates if hybrid else 1
    return {
        "eligible": int(learner_step) >= required,
        "policy_training_regime": behavior_cloning.get(
            "policy_training_regime", "unknown"
        ),
        "learner_step": int(learner_step),
        "minimum_cumulative_rl_updates": int(required),
        "bc_only_or_trivial_update_checkpoint_rejected": hybrid,
    }


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
    opponent_styles = resolve_opponent_styles(args)
    opponents = [RuleBasedController(style) for style in opponent_styles]
    requested_rl_learning_rate = (
        args.hybrid_rl_learning_rate
        if requested_fresh_hybrid(args, algorithm)
        else args.learning_rate
    )
    if algorithm == "mappo":
        learner = SequenceMAPPOLearner(
            model,
            learning_rate=requested_rl_learning_rate,
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
            learning_rate=requested_rl_learning_rate,
            target_update_interval=args.target_update_interval,
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
        saved_training = resume_extra.get("training_hyperparameters", {})
        saved_contract = resume_extra.get("contract", {})
        if isinstance(saved_contract, Mapping):
            saved_encoder = saved_contract.get("encoder_kind", "deepset")
            saved_heads = int(saved_contract.get("attention_heads", 4))
            if (
                saved_encoder != args.encoder_kind
                or saved_heads != args.attention_heads
            ):
                raise ValueError("resume checkpoint encoder contract does not match request")
        if isinstance(saved_training, Mapping):
            saved_target_interval = saved_training.get("target_update_interval")
            if (
                algorithm != "mappo"
                and saved_target_interval is not None
                and int(saved_target_interval) != args.target_update_interval
            ):
                raise ValueError(
                    "resume checkpoint target update interval does not match request"
                )
            saved_anneal = saved_training.get("epsilon_anneal_steps")
            if (
                algorithm != "mappo"
                and saved_anneal is not None
                and int(saved_anneal) != args.epsilon_anneal_steps
            ):
                raise ValueError(
                    "resume checkpoint epsilon anneal steps do not match request"
                )
        prior_episodes = int(
            resume_extra.get(
                "cumulative_episodes", resume_extra.get("episodes", 0)
            )
        )
        environment_steps = int(resume_extra.get("environment_steps", 0))
        if "curriculum" in resume_extra:
            curriculum.load_state_dict(resume_extra["curriculum"])
    rng = np.random.default_rng(seed + 19 + prior_episodes)
    evaluation_seed = seed + 8_000_000
    heldout_seed = seed + 18_000_000
    pre_bc_model_sha256 = model_state_sha256(model.state_dict())
    pre_bc_validation = None
    if args.resume is not None:
        saved_bc = resume_extra.get("behavior_cloning")
        behavior_cloning = (
            copy.deepcopy(dict(saved_bc))
            if isinstance(saved_bc, Mapping)
            else {
                "enabled": None,
                "policy_training_regime": "legacy_unverified_pretraining_history",
                "honest_label": "resume_checkpoint_has_no_BC_audit_metadata",
            }
        )
        behavior_cloning["resume_semantics"] = (
            "BC was not repeated; audit metadata was restored from the checkpoint"
        )
    elif (
        args.bc_demo_episodes_per_scale_opponent > 0
        and algorithm in args.bc_algorithms
    ):
        pre_bc_validation = controlled_evaluation_suite(
            runner,
            evaluation_controller,
            args.train_side,
            scales,
            args.eval_episodes_per_scale,
            evaluation_seed,
            opponent_styles,
        )
        bc_started = time.perf_counter()
        demonstrations = collect_guard_demonstrations(
            runner,
            scales,
            opponent_styles,
            args.bc_demo_episodes_per_scale_opponent,
            seed,
            reserved_splits={
                "validation": evaluation_seed_set(
                    evaluation_seed,
                    scales,
                    opponent_styles,
                    args.eval_episodes_per_scale,
                ),
                "heldout": evaluation_seed_set(
                    heldout_seed,
                    scales,
                    opponent_styles,
                    args.heldout_episodes_per_scale,
                ),
            },
        )
        rl_optimizer_before = rl_optimizer_state_entry_counts(learner, algorithm)
        bc_training = train_recurrent_behavior_clone(
            model,
            algorithm,
            demonstrations.episodes,
            epochs=args.bc_epochs,
            batch_size=args.bc_batch_episodes,
            learning_rate=args.bc_learning_rate,
            shuffle_seed=seed + 29_500_000,
            device=device,
        )
        target_sync = (
            {"performed": False, "reason": "MAPPO has no Q target network"}
            if algorithm == "mappo"
            else synchronize_q_target_after_behavior_cloning(learner)
        )
        rl_optimizer_after = rl_optimizer_state_entry_counts(learner, algorithm)
        if rl_optimizer_after != rl_optimizer_before:
            raise AssertionError("BC contaminated the registered RL optimizer state")
        behavior_cloning = {
            "enabled": True,
            "policy_training_regime": "rule_demonstration_bc_then_rl",
            "honest_label": (
                "hybrid_rule_demonstration_pretraining_plus_RL_finetuning"
            ),
            "dataset": dict(demonstrations.metadata),
            "training": bc_training,
            "target_network_sync": target_sync,
            "registered_rl_optimizer_state_entries_before": rl_optimizer_before,
            "registered_rl_optimizer_state_entries_after": rl_optimizer_after,
            "elapsed_seconds": time.perf_counter() - bc_started,
            "resume_semantics": "future resume restores metadata and never repeats BC",
        }
    else:
        behavior_cloning = {
            "enabled": False,
            "policy_training_regime": "pure_rl",
            "honest_label": "pure_RL_without_rule_demonstration_pretraining",
            "reason": (
                "BC globally disabled"
                if args.bc_demo_episodes_per_scale_opponent == 0
                else f"{algorithm} retained as a pure-RL comparison baseline"
            ),
        }
    effective_rl_learning_rates = rl_optimizer_learning_rates(learner, algorithm)
    is_hybrid = (
        behavior_cloning.get("policy_training_regime")
        == "rule_demonstration_bc_then_rl"
    )
    if args.resume is not None and isinstance(saved_training, Mapping):
        saved_preservation = saved_training.get("hybrid_preservation")
        if is_hybrid and isinstance(saved_preservation, Mapping):
            resume_contract = {
                "hybrid_epsilon_start": args.hybrid_epsilon_start,
                "hybrid_epsilon_finish": args.hybrid_epsilon_finish,
                "epsilon_anneal_steps": args.epsilon_anneal_steps,
                "minimum_cumulative_rl_updates_for_selection": (
                    args.hybrid_minimum_rl_updates
                ),
                "registered_hybrid_rl_learning_rate": (
                    args.hybrid_rl_learning_rate
                ),
            }
            mismatches = {
                key: {"checkpoint": saved_preservation.get(key), "request": value}
                for key, value in resume_contract.items()
                if saved_preservation.get(key) is not None
                and saved_preservation.get(key) != value
            }
            if mismatches:
                raise ValueError(
                    f"resume hybrid-preservation contract mismatch: {mismatches}"
                )
    behavior_cloning["rl_preservation_protocol"] = {
        "enabled": is_hybrid,
        "effective_rl_learning_rates": effective_rl_learning_rates,
        "registered_hybrid_rl_learning_rate": args.hybrid_rl_learning_rate,
        "hybrid_epsilon_start": args.hybrid_epsilon_start,
        "hybrid_epsilon_finish": args.hybrid_epsilon_finish,
        "epsilon_anneal_steps": args.epsilon_anneal_steps,
        "minimum_cumulative_rl_updates_for_selection": (
            args.hybrid_minimum_rl_updates if is_hybrid else 1
        ),
        "interleaved_demo_rehearsal": False,
        "rationale": (
            "lower-exploration/lower-step-size preservation after BC; rehearsal "
            "deferred until this minimal intervention is evaluated"
            if is_hybrid
            else "not applicable to pure RL"
        ),
    }
    if requested_fresh_hybrid(args, algorithm) and any(
        abs(value - args.hybrid_rl_learning_rate) > 1e-12
        for value in effective_rl_learning_rates.values()
    ):
        raise AssertionError("fresh hybrid learner did not use the registered RL rate")
    if isinstance(training_controller, QMixController):
        training_controller.epsilon, _ = q_epsilon_schedule(
            behavior_cloning, environment_steps, args
        )
    initial_training_model_sha256 = model_state_sha256(model.state_dict())

    initial = controlled_evaluation_suite(
        runner,
        evaluation_controller,
        args.train_side,
        scales,
        args.eval_episodes_per_scale,
        evaluation_seed,
        opponent_styles,
    )
    if behavior_cloning.get("enabled") is True:
        behavior_cloning["validation_diagnostic"] = {
            "split": "validation_not_demonstration_data",
            "pre_bc": pre_bc_validation,
            "post_bc_pre_rl": initial,
            "used_as_bc_gradient_or_label_data": False,
            "bc_only_checkpoint_eligible_for_selection": False,
        }
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
            "policy_training_regime": behavior_cloning[
                "policy_training_regime"
            ],
            "behavior_cloning": behavior_cloning,
            "git_provenance": args._git_provenance,
            "formal_contract": args._formal_contract,
            "seed": seed,
            "train_side": args.train_side,
            "registered_scales": [list(scale) for scale in scales],
            "strict_red_superiority": True,
            "protocol_version": "had-stage1-strict-red-v5-hybrid-preservation",
            "episodes": cumulative_episodes,
            "cumulative_episodes": cumulative_episodes,
            "environment_steps": environment_steps,
            "curriculum": curriculum.state_dict(),
            "selection": selection,
            "checkpoint_selection_eligibility": (
                hybrid_checkpoint_selection_eligibility(
                    behavior_cloning,
                    int(learner.learner_step),
                    args.hybrid_minimum_rl_updates,
                )
            ),
            "selection_evaluation": None if evaluation is None else dict(evaluation),
            "selection_key": (
                None if evaluation is None else list(robust_validation_key(evaluation))
            ),
            "contract": {
                "entity_dim": HADStage1Adapter.ENTITY_DIM,
                "self_dim": HADStage1Adapter.SELF_DIM,
                "task_dim": HADStage1Adapter.TASK_DIM,
                "state_entity_dim": HADStage1Adapter.STATE_ENTITY_DIM,
                "action_dim": HADStage1Adapter.ACTION_DIM,
                "encoder_kind": args.encoder_kind,
                "attention_heads": args.attention_heads,
                "policy_architecture": (
                    "OpenSCORE-SAQA-QMIX-SP"
                    if algorithm == "qmix" and args.encoder_kind == "saqa"
                    else f"OpenSCORE-{args.encoder_kind}-{algorithm.upper()}"
                ),
                "implementation_status": (
                    "Open-SCORE implementation inspired by REFIL Attention-QMIX "
                    "and SPECTra SAQA/target-action design; not an exact reproduction"
                ),
                "training_origin_label": behavior_cloning.get(
                    "honest_label", behavior_cloning["policy_training_regime"]
                ),
                "spectra_audited_commit": (
                    "ffababf6187216c9d16b2109ee8ef6fe5fdf1172"
                ),
                "spectra_code_copied": False,
                "references": {
                    "refil": "https://proceedings.mlr.press/v139/iqbal21a.html",
                    "spectra_paper": "https://arxiv.org/abs/2503.11726",
                    "spectra_repository": "https://github.com/funny-rl/SPECTra",
                    "spmarl": "https://proceedings.mlr.press/v267/zhao25o.html",
                },
            },
            "model_config": {
                "agent_hidden_dim": args.agent_hidden_dim,
                "critic_or_mixer_hidden_dim": args.critic_hidden_dim,
                "mixing_dim": max(8, args.critic_hidden_dim // 2),
                "encoder_kind": args.encoder_kind,
                "attention_heads": args.attention_heads,
            },
            "training_hyperparameters": {
                "pure_rl_baseline_learning_rate": args.learning_rate,
                "effective_rl_learning_rates": effective_rl_learning_rates,
                "hybrid_preservation": behavior_cloning[
                    "rl_preservation_protocol"
                ],
                "ppo_epochs": args.ppo_epochs,
                "target_update_interval": (
                    None if algorithm == "mappo" else args.target_update_interval
                ),
                "epsilon_anneal_steps": (
                    None if algorithm == "mappo" else args.epsilon_anneal_steps
                ),
                "epsilon_start": (
                    None
                    if algorithm == "mappo"
                    else q_epsilon_schedule(
                        behavior_cloning, environment_steps, args
                    )[1]["start"]
                ),
                "epsilon_finish": (
                    None
                    if algorithm == "mappo"
                    else q_epsilon_schedule(
                        behavior_cloning, environment_steps, args
                    )[1]["finish"]
                ),
                "epsilon_at_checkpoint": (
                    None
                    if algorithm == "mappo"
                    else q_epsilon_schedule(
                        behavior_cloning, environment_steps, args
                    )[0]
                ),
            },
            "environment_protocol": {
                "max_steps": args.max_steps,
                "gamma": 0.99,
                "registered_scales": [list(scale) for scale in scales],
                "strict_red_superiority": True,
                "episode_phase_feature": "normalized_remaining_horizon",
                "opponent_styles": list(opponent_styles),
            },
            "continuation_contract": {
                "restores": [
                    "online_model",
                    "target_model_when_applicable",
                    "optimizer",
                    "learner_step",
                    "curriculum_state",
                    "environment_step_counter",
                    "behavior_cloning_audit_metadata",
                ],
                "does_not_restore": [
                    "replay_buffer",
                    "python_rng",
                    "numpy_rollout_rng",
                    "torch_rng",
                    "discarded_behavior_cloning_optimizer",
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
        eligibility = hybrid_checkpoint_selection_eligibility(
            behavior_cloning,
            int(learner.learner_step),
            args.hybrid_minimum_rl_updates,
        )
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
            "checkpoint_selection_eligible": eligibility["eligible"],
            "minimum_cumulative_rl_updates_for_selection": eligibility[
                "minimum_cumulative_rl_updates"
            ],
        }
        row.update(learning_fields(latest_metrics))
        curves.append(row)

    checkpoint_directory = args.output_dir / "checkpoints"
    best_checkpoint_path = checkpoint_directory / f"{algorithm}_seed{seed}_best.pt"
    initial_selection_eligibility = hybrid_checkpoint_selection_eligibility(
        behavior_cloning,
        int(learner.learner_step),
        args.hybrid_minimum_rl_updates,
    )
    initial_resume_is_eligible = bool(
        prior_episodes > 0 and initial_selection_eligibility["eligible"]
    )
    best_evaluation = dict(initial) if initial_resume_is_eligible else None
    best_episode = prior_episodes if initial_resume_is_eligible else None
    if initial_resume_is_eligible:
        learner.save(
            best_checkpoint_path,
            checkpoint_metadata(prior_episodes, "paired_eval_best", initial),
        )

    def consider_best(episode: int, evaluation: Mapping[str, object]) -> None:
        nonlocal best_evaluation, best_episode
        eligibility = hybrid_checkpoint_selection_eligibility(
            behavior_cloning,
            int(learner.learner_step),
            args.hybrid_minimum_rl_updates,
        )
        if not eligibility["eligible"]:
            return
        score = robust_validation_key(evaluation)
        best_score = (
            None
            if best_evaluation is None
            else robust_validation_key(best_evaluation)
        )
        if best_score is None or score > best_score:
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
            training_controller.epsilon, _ = q_epsilon_schedule(
                behavior_cloning, environment_steps, args
            )
        rollout_seed = seed + cumulative_episode * 101
        opponent = opponents[(cumulative_episode - 1) % len(opponents)]
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
                        replay.sample_scale_balanced(args.batch_episodes, device)
                    )
                for updated_scale, signal in latest_metrics.td_by_scale.items():
                    curriculum.update(updated_scale, signal)

        if episode_index % args.eval_every == 0 or episode_index == args.episodes:
            evaluation = controlled_evaluation_suite(
                runner,
                evaluation_controller,
                args.train_side,
                scales,
                args.eval_episodes_per_scale,
                evaluation_seed,
                opponent_styles,
            )
            record_curve(cumulative_episode, evaluation)
            consider_best(cumulative_episode, evaluation)

    if algorithm == "mappo" and pending:
        latest_metrics = learner.train_batch(collate_episodes(pending, device))
        pending.clear()
        # Re-evaluate because the final on-policy partial batch changed weights.
        final = controlled_evaluation_suite(
            runner,
            evaluation_controller,
            args.train_side,
            scales,
            args.eval_episodes_per_scale,
            evaluation_seed,
            opponent_styles,
        )
        final_episode = prior_episodes + args.episodes
        if curves[-1]["episode"] == final_episode:
            curves.pop()
        record_curve(final_episode, final)
        consider_best(final_episode, final)
    else:
        final = controlled_evaluation_suite(
            runner,
            evaluation_controller,
            args.train_side,
            scales,
            args.eval_episodes_per_scale,
            evaluation_seed,
            opponent_styles,
        )
        final_episode = prior_episodes + args.episodes
        if curves[-1]["episode"] == final_episode:
            curves.pop()
        record_curve(final_episode, final)
        consider_best(final_episode, final)

    checkpoint_path = args.output_dir / "checkpoints" / f"{algorithm}_seed{seed}.pt"
    if best_evaluation is None or best_episode is None:
        raise RuntimeError(
            "no checkpoint met validation selection eligibility; increase the RL "
            "budget so cumulative learner updates reach the registered minimum"
        )
    learner.save(
        checkpoint_path,
        checkpoint_metadata(final_episode, "final", final),
    )
    final_model_state = copy.deepcopy(model.state_dict())
    final_model_sha256 = model_state_sha256(final_model_state)
    final_parameters_finite = tensor_state_is_finite(final_model_state)
    final_checkpoint_sha256 = checkpoint_sha256(checkpoint_path)
    selected_checkpoint = torch.load(
        best_checkpoint_path, map_location=device, weights_only=False
    )
    selected_key = "model" if algorithm == "mappo" else "online"
    selected_checkpoint_learner_step = int(selected_checkpoint["learner_step"])
    model.load_state_dict(selected_checkpoint[selected_key], strict=True)
    selected_model_sha256 = model_state_sha256(model.state_dict())
    selected_parameters_finite = tensor_state_is_finite(model.state_dict())
    heldout_seed = seed + 18_000_000
    heldout_best = controlled_evaluation_suite(
        runner,
        evaluation_controller,
        args.train_side,
        scales,
        args.heldout_episodes_per_scale,
        heldout_seed,
        opponent_styles,
    )
    model.load_state_dict(final_model_state, strict=True)
    elapsed = time.perf_counter() - started
    final_epsilon = (
        q_epsilon_schedule(behavior_cloning, environment_steps, args)[0]
        if isinstance(training_controller, QMixController)
        else None
    )
    best_eval = max(
        float(row["eval_controlled_mean_payoff"]) for row in curves
    )
    result = {
        "algorithm": algorithm,
        "seed": seed,
        "train_side": args.train_side,
        "policy_training_regime": behavior_cloning["policy_training_regime"],
        "behavior_cloning": behavior_cloning,
        "parameter_count": parameter_count,
        "pre_bc_or_resume_model_sha256": pre_bc_model_sha256,
        "initial_training_model_sha256": initial_training_model_sha256,
        "final_model_sha256": final_model_sha256,
        "selected_model_sha256": selected_model_sha256,
        "selected_parameters_finite": selected_parameters_finite,
        "final_parameters_finite": final_parameters_finite,
        "selected_parameters_changed": (
            selected_model_sha256 != initial_training_model_sha256
            or prior_episodes > 0
        ),
        "selected_checkpoint_learner_step": selected_checkpoint_learner_step,
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
        "selection_protocol": {
            "split": "validation",
            "seed_base": evaluation_seed,
            "metric": (
                "lexicographic: worst scale-by-opponent payoff, then overall "
                "mean payoff"
            ),
            "selected_key": list(robust_validation_key(best_evaluation)),
            "opponent_styles": list(opponent_styles),
            "minimum_cumulative_rl_updates": (
                args.hybrid_minimum_rl_updates if is_hybrid else 1
            ),
            "selected_checkpoint_minimum_updates_met": (
                selected_checkpoint_learner_step
                >= (args.hybrid_minimum_rl_updates if is_hybrid else 1)
            ),
        },
        "q_learning_schedule": {
            "epsilon_anneal_steps": (
                None if algorithm == "mappo" else args.epsilon_anneal_steps
            ),
            "final_epsilon": final_epsilon,
            "target_update_interval": (
                None if algorithm == "mappo" else args.target_update_interval
            ),
            "regime": (
                None
                if algorithm == "mappo"
                else q_epsilon_schedule(
                    behavior_cloning, environment_steps, args
                )[1]
            ),
        },
        "heldout_best_evaluation": {
            **heldout_best,
            "seed_base": heldout_seed,
            "episodes_per_scale_per_opponent": args.heldout_episodes_per_scale,
            "evaluated_only_after_checkpoint_selection": True,
        },
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
        "checkpoint_sha256": final_checkpoint_sha256,
        "best_checkpoint": str(best_checkpoint_path.resolve()),
        "best_checkpoint_sha256": checkpoint_sha256(best_checkpoint_path),
        "checkpoint_audit": {
            "best": {
                "disk_load_performed": True,
                "strict_model_load": True,
                "learner_step": selected_checkpoint_learner_step,
                "parameters_finite_after_reload": selected_parameters_finite,
            },
            "final": {
                "saved": checkpoint_path.is_file(),
                "sha256_recorded": True,
                "disk_reload_performed": False,
                "claim": "saved_and_hashed_not_disk_reload_verified",
            },
        },
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


def engineering_precheck_gate(
    results: Sequence[Mapping[str, object]], args: argparse.Namespace
) -> Dict[str, object]:
    """Build the machine-readable fixed-2v1 engineering acceptance gate.

    This gate deliberately makes no win-rate or convergence claim.  It checks
    only evidence produced by the actual learner/update and checkpoint path.
    The validation-selected checkpoint is reloaded from disk with
    ``strict=True`` inside :func:`train_one`; the final checkpoint is only
    required to be saved and hash-stable, and is never described as reloaded.
    """

    applicable = bool(
        not args.formal_evidence
        and args.fixed_scale is not None
        and tuple(args.fixed_scale) == (2, 1)
    )
    if not applicable:
        return {
            "applicable": False,
            "passed": None,
            "claim_scope": "not_a_fixed_2v1_engineering_precheck",
            "runs": [],
        }

    required_algorithms = ("qmix", "vdn", "mappo")
    expected_cells = {
        (algorithm, int(seed))
        for algorithm in required_algorithms
        for seed in args.seeds
    }
    observed_cells = {
        (str(result["algorithm"]), int(result["seed"])) for result in results
    }
    run_gates = []
    for result in results:
        metrics = result.get("latest_learning_metrics")
        metrics_mapping = metrics if isinstance(metrics, Mapping) else {}
        final_path = Path(str(result.get("checkpoint", "")))
        final_sha = str(result.get("checkpoint_sha256", ""))
        checkpoint_audit = result.get("checkpoint_audit")
        checkpoint_mapping = (
            checkpoint_audit if isinstance(checkpoint_audit, Mapping) else {}
        )
        best_audit = checkpoint_mapping.get("best")
        best_mapping = best_audit if isinstance(best_audit, Mapping) else {}
        final_audit = checkpoint_mapping.get("final")
        final_mapping = final_audit if isinstance(final_audit, Mapping) else {}
        checks = {
            "learner_updates_gt_zero": int(result.get("updates_this_run", 0)) > 0,
            "selected_parameters_changed": result.get(
                "selected_parameters_changed"
            )
            is True,
            "critical_learner_metrics_present": all(
                name in metrics_mapping for name in ("loss", "grad_norm", "learner_step")
            ),
            "critical_learner_metrics_finite": bool(metrics_mapping)
            and numeric_tree_is_finite(metrics_mapping),
            "selected_parameters_finite": result.get(
                "selected_parameters_finite"
            )
            is True,
            "final_parameters_finite": result.get("final_parameters_finite") is True,
            "best_checkpoint_disk_strict_reload": bool(
                best_mapping.get("disk_load_performed") is True
                and best_mapping.get("strict_model_load") is True
                and best_mapping.get("parameters_finite_after_reload") is True
            ),
            "final_checkpoint_saved_and_hashed": bool(
                final_mapping.get("saved") is True
                and final_mapping.get("sha256_recorded") is True
                and final_mapping.get("disk_reload_performed") is False
                and final_path.is_file()
                and len(final_sha) == 64
                and checkpoint_sha256(final_path) == final_sha
            ),
        }
        run_gates.append(
            {
                "algorithm": str(result["algorithm"]),
                "seed": int(result["seed"]),
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    coverage_exact = (
        observed_cells == expected_cells and len(results) == len(expected_cells)
    )
    return {
        "applicable": True,
        "claim_scope": (
            "gradient_finiteness_checkpoint_and_evaluation_pipeline_only; "
            "no performance_or_convergence_claim"
        ),
        "fixed_scale": [2, 1],
        "required_algorithms": list(required_algorithms),
        "expected_cells": [list(cell) for cell in sorted(expected_cells)],
        "observed_cells": [list(cell) for cell in sorted(observed_cells)],
        "algorithm_seed_coverage_exact": coverage_exact,
        "final_checkpoint_reload_claim": False,
        "runs": run_gates,
        "passed": bool(
            coverage_exact and run_gates and all(run["passed"] for run in run_gates)
        ),
    }


def write_curves(path: Path, curves: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not curves:
        raise ValueError("no curves to write")
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)


def bootstrap_mean_interval(
    values: Sequence[float], seed: int, draws: int = 20_000
) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot bootstrap an empty seed sample")
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(draws, array.size), replace=True).mean(axis=1)
    return {
        "mean": float(array.mean()),
        "std_across_seeds": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "seed_count": int(array.size),
    }


def aggregate_multi_seed_results(
    results: Sequence[Mapping[str, object]],
    rule_by_seed: Mapping[int, Mapping[str, object]],
    scales: Sequence[Scale],
) -> Dict[str, object]:
    """Aggregate independent training seeds without choosing on held-out data."""

    algorithms: Dict[str, object] = {}
    for algorithm_index, algorithm in enumerate(
        sorted({str(result["algorithm"]) for result in results})
    ):
        runs = sorted(
            [result for result in results if result["algorithm"] == algorithm],
            key=lambda result: int(result["seed"]),
        )
        training_regimes = {
            str(result.get("policy_training_regime", "legacy_unverified"))
            for result in runs
        }
        if len(training_regimes) != 1:
            raise ValueError(
                f"cannot aggregate mixed training regimes for {algorithm}: "
                f"{sorted(training_regimes)}"
            )
        training_regime = next(iter(training_regimes))
        pure_rl_replication = training_regime == "pure_rl"
        hybrid_primary = (
            algorithm == "qmix"
            and training_regime == "rule_demonstration_bc_then_rl"
        )
        seeds = [int(result["seed"]) for result in runs]
        learned_payoff = [
            float(result["heldout_best_evaluation"]["controlled_mean_payoff"])
            for result in runs
        ]
        learned_win_rate = [
            float(result["heldout_best_evaluation"]["controlled_win_rate"])
            for result in runs
        ]
        validation_improvement = [
            float(result["best_evaluation"]["controlled_mean_payoff"])
            - float(result["initial_evaluation"]["controlled_mean_payoff"])
            for result in runs
        ]
        rule_payoff = [
            float(rule_by_seed[seed]["controlled_mean_payoff"]) for seed in seeds
        ]
        payoff_delta = [
            learned - rule
            for learned, rule in zip(learned_payoff, rule_payoff)
        ]
        per_scale_delta = {}
        for red, blue in scales:
            label = f"{red}v{blue}"
            values = [
                float(result["heldout_best_evaluation"]["per_scale_controlled_payoff"][label])
                - float(rule_by_seed[int(result["seed"])]["per_scale_controlled_payoff"][label])
                for result in runs
            ]
            per_scale_delta[label] = bootstrap_mean_interval(
                values, 51_000 + algorithm_index * 1_000 + red * 10 + blue
            )
        validation_scores = [
            robust_validation_key(result["best_evaluation"]) for result in runs
        ]
        deployment_index = max(
            range(len(validation_scores)), key=validation_scores.__getitem__
        )
        delta_interval = bootstrap_mean_interval(
            payoff_delta, 49_000 + algorithm_index
        )
        formal_sample = len(seeds) >= 5
        trained_checkpoint_sample = all(
            bool(result["selected_parameters_changed"]) for result in runs
        )
        selected_minimum_updates_sample = all(
            bool(
                result.get("selection_protocol", {}).get(
                    "selected_checkpoint_minimum_updates_met", False
                )
            )
            for result in runs
        )
        validation_improvement_interval = bootstrap_mean_interval(
            validation_improvement, 46_000 + algorithm_index
        )
        replication_pre_stability = bool(
            formal_sample
            and trained_checkpoint_sample
            and selected_minimum_updates_sample
            and validation_improvement_interval["mean"] > 0.0
        )
        superiority_pre_stability = bool(
            formal_sample
            and trained_checkpoint_sample
            and selected_minimum_updates_sample
            and delta_interval["ci95_low"] > 0.0
            and all(value["mean"] >= 0.0 for value in per_scale_delta.values())
        )
        algorithms[algorithm] = {
            "seeds": seeds,
            "policy_training_regime": training_regime,
            "claim_role": (
                "hybrid_primary_intelligent_strategy"
                if hybrid_primary
                else (
                    "pure_rl_primary_and_replication_baseline"
                    if algorithm == "qmix" and pure_rl_replication
                    else "pure_rl_replication_baseline"
                )
            ),
            "heldout_payoff": bootstrap_mean_interval(
                learned_payoff, 47_000 + algorithm_index
            ),
            "heldout_win_rate": bootstrap_mean_interval(
                learned_win_rate, 48_000 + algorithm_index
            ),
            "paired_rule_payoff_delta": delta_interval,
            "per_scale_paired_rule_payoff_delta": per_scale_delta,
            "validation_best_minus_initial_payoff": (
                validation_improvement_interval
            ),
            "deployment_candidate": {
                "selection_split": "validation_only",
                "selection_metric": (
                    "lexicographic worst scale-by-opponent payoff, then overall mean"
                ),
                "seed": seeds[deployment_index],
                "checkpoint": runs[deployment_index]["best_checkpoint"],
                "checkpoint_sha256": runs[deployment_index]["best_checkpoint_sha256"],
                "validation_score": list(validation_scores[deployment_index]),
                "heldout_was_not_used_for_selection": True,
            },
            "baseline_replication_gate": {
                "passed": False,
                "eligible": pure_rl_replication,
                "pre_stability_requirements_met": bool(
                    pure_rl_replication and replication_pre_stability
                ),
                "plateau_stability_met": False,
                "requirements": [
                    "policy is pure RL without rule-demonstration pretraining",
                    "at least five independent training seeds",
                    "every selected checkpoint contains trained rather than initial weights",
                    "every selected checkpoint meets its registered minimum RL updates",
                    "mean validation-best minus initial payoff > 0",
                    "every training seed passes the preregistered validation plateau audit",
                ],
                "formal_seed_minimum_met": formal_sample,
                "trained_checkpoint_requirement_met": trained_checkpoint_sample,
                "minimum_rl_updates_requirement_met": (
                    selected_minimum_updates_sample
                ),
            },
            "rl_finetuning_gate": {
                "passed": False,
                "pre_stability_requirements_met": replication_pre_stability,
                "plateau_stability_met": False,
                "requirements": [
                    "at least five independent training seeds",
                    "every selected checkpoint changed after its RL starting point",
                    "every selected checkpoint meets the hybrid RL update minimum",
                    "mean validation-best minus pre-RL payoff > 0",
                    "every training seed passes the validation plateau audit",
                ],
                "minimum_rl_updates_requirement_met": (
                    selected_minimum_updates_sample
                ),
            },
            "robust_superiority_gate": {
                "passed": superiority_pre_stability,
                "claim_role": (
                    "hybrid_primary_strategy" if hybrid_primary else (
                    "designated_pure_rl_primary_strategy" if algorithm == "qmix" else
                    "descriptive_rule_comparison_not_a_baseline_replication_requirement"
                    )
                ),
                "requirements": [
                    "at least five independent training seeds",
                    "every selected checkpoint contains trained rather than initial weights",
                    "every selected checkpoint meets its registered minimum RL updates",
                    "95% seed-bootstrap lower bound of learned-minus-rule payoff > 0",
                    "non-negative mean learned-minus-rule payoff at every registered scale",
                ],
                "formal_seed_minimum_met": formal_sample,
                "trained_checkpoint_requirement_met": trained_checkpoint_sample,
                "minimum_rl_updates_requirement_met": (
                    selected_minimum_updates_sample
                ),
            },
        }
    return {
        "unit_of_replication": "independent training seed",
        "checkpoint_selection": (
            "within-seed validation best; cross-seed deployment candidate selected "
            "by validation only; every seed retained in aggregate"
        ),
        "heldout_reuse": "one post-selection evaluation per seed",
        "algorithms": algorithms,
    }


def plateau_stability_audit(
    curves: Sequence[Mapping[str, object]], window: int = 5
) -> Dict[str, object]:
    """Audit late validation stability without looking at held-out outcomes."""

    grouped: Dict[Tuple[str, int], List[Mapping[str, object]]] = {}
    for row in curves:
        grouped.setdefault((str(row["algorithm"]), int(row["seed"])), []).append(row)
    runs = []
    for (algorithm, seed), rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: int(row["episode"]))
        tail = ordered[-window:]
        values = np.asarray(
            [float(row["eval_controlled_mean_payoff"]) for row in tail],
            dtype=np.float64,
        )
        enough = len(tail) >= window
        slope = (
            float(np.polyfit(np.arange(len(values)), values, 1)[0])
            if len(values) >= 2
            else float("inf")
        )
        value_range = float(np.ptp(values)) if values.size else float("inf")
        runs.append(
            {
                "algorithm": algorithm,
                "seed": seed,
                "evaluations": len(tail),
                "episode_start": int(tail[0]["episode"]),
                "episode_end": int(tail[-1]["episode"]),
                "payoff_slope_per_evaluation": slope,
                "payoff_range": value_range,
                "passed": bool(
                    enough and abs(slope) <= 0.02 and value_range <= 0.10
                ),
            }
        )
    algorithms = sorted({str(run["algorithm"]) for run in runs})
    by_algorithm = {
        algorithm: {
            "passed": bool(
                [run for run in runs if run["algorithm"] == algorithm]
                and all(
                    run["passed"]
                    for run in runs
                    if run["algorithm"] == algorithm
                )
            ),
            "runs": [run for run in runs if run["algorithm"] == algorithm],
        }
        for algorithm in algorithms
    }
    return {
        "split": "validation",
        "window_evaluations": window,
        "requirements": [
            "absolute payoff slope per evaluation <= 0.02",
            "payoff range across window <= 0.10",
        ],
        "runs": runs,
        "by_algorithm": by_algorithm,
        "all_runs_passed": bool(runs and all(run["passed"] for run in runs)),
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    refil_requested = "refil_qmix" in args.algorithms
    args._git_provenance = collect_and_require_git_provenance(
        PROJECT, formal=(args.formal_evidence or refil_requested)
    )
    device = choose_device(args.device)
    if refil_requested:
        from open_score.stage1.refil_protocol import run_refil_round

        run_refil_round(args, device, args._git_provenance)
        return
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
    opponent_styles = resolve_opponent_styles(args)
    rule_controller = RuleBasedController(
        "guard" if args.train_side == "Red" else "rush"
    )
    rule_by_seed = {
        int(seed): controlled_evaluation_suite(
            runner,
            rule_controller,
            args.train_side,
            scales,
            args.heldout_episodes_per_scale,
            int(seed) + 18_000_000,
            opponent_styles,
        )
        for seed in args.seeds
    }
    results = []
    curves: List[Dict[str, object]] = []
    for seed in args.seeds:
        for algorithm in args.algorithms:
            result, algorithm_curves = train_one(
                algorithm, seed, args, device, scales
            )
            results.append(result)
            curves.extend(algorithm_curves)

    multi_seed = aggregate_multi_seed_results(results, rule_by_seed, scales)
    stability = plateau_stability_audit(curves)
    precheck_gate = engineering_precheck_gate(results, args)
    baseline_replication_claims = {}
    excluded_from_pure_rl_replication = {}
    for algorithm, value in multi_seed["algorithms"].items():
        plateau_met = bool(
            stability["by_algorithm"].get(algorithm, {}).get("passed", False)
        )
        replication_gate = value["baseline_replication_gate"]
        replication_gate["plateau_stability_met"] = plateau_met
        replication_gate["passed"] = bool(
            replication_gate["pre_stability_requirements_met"] and plateau_met
        )
        finetuning_gate = value["rl_finetuning_gate"]
        finetuning_gate["plateau_stability_met"] = plateau_met
        finetuning_gate["passed"] = bool(
            finetuning_gate["pre_stability_requirements_met"] and plateau_met
        )
        if replication_gate["eligible"]:
            baseline_replication_claims[algorithm] = bool(
                args.formal_evidence and replication_gate["passed"]
            )
        else:
            excluded_from_pure_rl_replication[algorithm] = {
                "policy_training_regime": value["policy_training_regime"],
                "reason": "hybrid/legacy policy cannot substantiate a pure-RL claim",
            }
    primary_algorithm = "qmix"
    primary_result = multi_seed["algorithms"].get(primary_algorithm)
    if primary_result is None:
        primary_strategy_gate = {
            "passed": False,
            "algorithm": primary_algorithm,
            "reason": "designated primary algorithm was not run",
        }
    else:
        primary_strategy_gate = {
            **primary_result["robust_superiority_gate"],
            "algorithm": primary_algorithm,
            "policy_training_regime": primary_result["policy_training_regime"],
            "rl_finetuning_gate_met": primary_result[
                "rl_finetuning_gate"
            ]["passed"],
        }
        primary_strategy_gate["passed"] = bool(
            primary_strategy_gate["passed"]
            and primary_strategy_gate["rl_finetuning_gate_met"]
        )
    formal_convergence_claim = bool(
        args.formal_evidence
        and primary_strategy_gate["passed"]
    )

    payload = _jsonify(
        {
            "schema_version": "stage1-had-reproduction-v5",
            "status": (
                "formal_multi_seed_protocol_executed"
                if args.formal_evidence
                else "engineering_or_pilot_training_check"
            ),
            "formal_convergence_claim": formal_convergence_claim,
            "formal_primary_strategy_claim": formal_convergence_claim,
            "formal_baseline_replication_claims": baseline_replication_claims,
            "excluded_from_pure_rl_replication": (
                excluded_from_pure_rl_replication
            ),
            "all_baseline_replication_gates_passed": bool(
                baseline_replication_claims
                and all(baseline_replication_claims.values())
            ),
            "primary_strategy_gate": primary_strategy_gate,
            "engineering_precheck_gate": precheck_gate,
            "git_provenance": args._git_provenance,
            "formal_contract": args._formal_contract,
            "reason": (
                "Each classic baseline has an independent trained-weight, validation-"
                "improvement and stability gate. A BC+RL QMIX is explicitly excluded "
                "from pure-RL replication and instead must pass an RL-finetuning gate; "
                "only the designated QMIX primary carries the superiority claim."
            ),
            "environment": "HADStage1Adapter",
            "roles": {"Red": "asset defender", "Blue": "asset attacker"},
            "firing_semantics": "any HAD AttackAgent self-destructs after firing",
            "strict_red_superiority": True,
            "registered_scales": [list(scale) for scale in scales],
            "shared_policy_architecture": True,
            "episode_phase_in_actor_and_central_state": True,
            "scale_balanced_replay": True,
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
                "controlled": rule_controller.name,
                "opponent_suite": list(opponent_styles),
                "heldout_evaluation_by_training_seed": rule_by_seed,
            },
            "intelligent_strategy": {
                "architecture": "shared recurrent entity-set QMIX/VDN/MAPPO",
                "dynamic_roster_masks": True,
                "research_basis": [
                    "REFIL variable-entity multitask representation motivation",
                    "EPC stagewise population-size curriculum principle",
                    "SPMARL TD-error learning-progress curriculum principle",
                ],
                "exact_reimplementation_claim": False,
                "training_regimes_by_algorithm": {
                    algorithm: value["policy_training_regime"]
                    for algorithm, value in multi_seed["algorithms"].items()
                },
            },
            "multi_seed_heldout": multi_seed,
            "validation_plateau_stability": stability,
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": (
                torch.cuda.get_device_name(0) if device.type == "cuda" else None
            ),
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
                if not key.startswith("_")
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
    if precheck_gate["applicable"] and not precheck_gate["passed"]:
        raise RuntimeError(
            "fixed-2v1 engineering precheck gate failed; inspect "
            "engineering_precheck_gate in the saved summary"
        )


if __name__ == "__main__":
    main()
