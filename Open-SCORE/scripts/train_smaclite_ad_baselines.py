"""Real small-budget QMIX/VDN/MAPPO validation on dynamic SMAClite-AD.

This entry point is deliberately separate from the HAD reproduction CLI.  The
defaults are a paired short-budget engineering check; ``--formal-evidence`` is
accepted only when every registered YAML argument matches exactly.
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
from open_score.formal_contracts import validate_registered_formal_contract
from open_score.provenance import collect_and_require_git_provenance
from open_score.stage1.baselines import (
    SequenceMAPPOLearner,
    VariableScaleMAPPO,
    VariableScaleVDN,
)
from open_score.stage1.entity_qmix import VariableScaleQMIX
from open_score.stage1.learner import SequenceQMIXLearner, linear_epsilon
from open_score.stage1.replay import EpisodeReplayBuffer, collate_episodes
from open_score.stage1.transfer import ACTION_TARGET_CONTRACT, transfer_stock_checkpoint
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


PROJECT = Path(__file__).resolve().parents[1]
STOCK_TO_AD_FORMAL_CONFIG = PROJECT / "configs" / "stage1_stock_to_ad_formal.yaml"
STOCK_TO_AD_FORMAL_PROTOCOL = "stock-to-ad-transfer-v3-saqa-strict-potential"


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
    parser.add_argument(
        "--initializations",
        nargs="+",
        choices=("scratch", "stock_transfer"),
        default=["scratch"],
        help=(
            "Run a scratch control, a stock-to-AD fine-tune, or both.  Both "
            "arms reuse the same training/evaluation seeds."
        ),
    )
    parser.add_argument(
        "--stock-checkpoint-template",
        default=None,
        help=(
            "Checkpoint path template for stock_transfer; supports {algorithm} "
            "and {seed}, for example outputs/stock/{algorithm}_seed{seed}.pt."
        ),
    )
    parser.add_argument("--ratios", type=parse_ratios, default=parse_ratios("2:1,3:2,5:3"))
    parser.add_argument("--train-side", choices=("Red", "Blue"), default="Red")
    parser.add_argument(
        "--opponent", choices=("idle", "intercept", "rush_asset"), default="idle"
    )
    parser.add_argument(
        "--warmup-opponent",
        choices=("idle", "intercept", "rush_asset"),
        default=None,
        help=(
            "Optional easier training opponent for the initial curriculum "
            "fraction; validation and held-out always use --opponent."
        ),
    )
    parser.add_argument("--warmup-fraction", type=float, default=0.25)
    parser.add_argument(
        "--clear-replay-on-opponent-transition",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Clear and refill Q-learning replay when the warm-up opponent is "
            "replaced by the target opponent."
        ),
    )
    parser.add_argument("--episodes", type=int, default=45)
    parser.add_argument("--episode-limit", type=int, default=60)
    parser.add_argument("--batch-episodes", type=int, default=3)
    parser.add_argument("--replay-episodes", type=int, default=96)
    parser.add_argument("--updates-per-episode", type=int, default=1)
    parser.add_argument("--target-update-interval", type=int, default=200)
    parser.add_argument("--td-lambda", type=float, default=0.6)
    parser.add_argument("--validation-episodes-per-ratio", type=int, default=5)
    parser.add_argument("--heldout-episodes-per-ratio", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--agent-hidden-dim", type=int, default=32)
    parser.add_argument("--critic-hidden-dim", type=int, default=32)
    parser.add_argument(
        "--encoder-kind", choices=("deepset", "saqa"), default="deepset"
    )
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--epsilon-start", type=float, default=0.90)
    parser.add_argument("--epsilon-finish", type=float, default=0.10)
    parser.add_argument("--epsilon-anneal-steps", type=int, default=2_000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument(
        "--reward-mode",
        choices=("terminal_only", "strict_potential", "heuristic_delta"),
        default="strict_potential",
        help=(
            "strict_potential is the formal default; heuristic_delta retains "
            "the legacy engineering reward only for an explicit ablation."
        ),
    )
    parser.add_argument("--shaping-scale", type=float, default=0.50)
    parser.add_argument("--approach-weight", type=float, default=1.0)
    parser.add_argument("--spawn-jitter", type=float, default=2.0)
    parser.add_argument("--max-red-agents", type=int, default=6)
    parser.add_argument("--max-blue-agents", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument(
        "--use-cpp-rvo2",
        action="store_true",
        help="Use the compiled SMAClite RVO2 backend for every experiment arm.",
    )
    parser.add_argument(
        "--formal-evidence",
        action="store_true",
        help="Enforce the multi-seed scratch/transfer held-out protocol.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to outputs/smaclite_ad_formal under "
            "--formal-evidence, otherwise outputs/smaclite_ad_baselines."
        ),
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=Path("outputs/smaclite_ad_evidence"),
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = Path(
            "outputs/smaclite_ad_formal"
            if args.formal_evidence
            else "outputs/smaclite_ad_baselines"
        )
    return args


def validate_args(args: argparse.Namespace) -> None:
    args._formal_contract = None
    positive = (
        "episodes",
        "episode_limit",
        "batch_episodes",
        "replay_episodes",
        "updates_per_episode",
        "target_update_interval",
        "validation_episodes_per_ratio",
        "heldout_episodes_per_ratio",
        "eval_every",
        "agent_hidden_dim",
        "critic_hidden_dim",
        "ppo_epochs",
        "attention_heads",
    )
    if any(getattr(args, name) < 1 for name in positive):
        raise ValueError("episode, batch, evaluation and model sizes must be positive")
    if args.encoder_kind == "saqa" and args.agent_hidden_dim % args.attention_heads:
        raise ValueError("attention_heads must divide agent_hidden_dim for SAQA")
    if args.validation_episodes_per_ratio < 5 or args.heldout_episodes_per_ratio < 5:
        raise ValueError("validation and held-out evaluation need at least 5 layouts per ratio")
    if args.spawn_jitter <= 0.0:
        raise ValueError("training evidence must explicitly enable spawn_jitter")
    if args.batch_episodes < len(args.ratios):
        raise ValueError("batch_episodes must cover every registered ratio")
    if args.replay_episodes < args.batch_episodes:
        raise ValueError("replay_episodes must be at least batch_episodes")
    if not 0.0 <= args.td_lambda <= 1.0:
        raise ValueError("td_lambda must be in [0, 1]")
    if not 0.0 < args.gamma <= 1.0:
        raise ValueError("gamma must lie in (0, 1]")
    if args.max_red_agents < max(red for red, _ in args.ratios):
        raise ValueError("max_red_agents is smaller than a registered ratio")
    if args.max_blue_agents < max(blue for _, blue in args.ratios):
        raise ValueError("max_blue_agents is smaller than a registered ratio")
    if args.train_side == "Blue" and args.opponent == "intercept":
        raise ValueError("a Red opponent should use rush_asset or idle")
    if args.train_side == "Red" and args.opponent == "rush_asset":
        raise ValueError("a Blue opponent should use intercept or idle")
    if not 0.0 <= args.warmup_fraction < 1.0:
        raise ValueError("warmup_fraction must be in [0, 1)")
    if args.warmup_opponent is not None:
        if args.warmup_opponent == args.opponent:
            raise ValueError("warmup_opponent must differ from target opponent")
        if args.train_side == "Blue" and args.warmup_opponent == "intercept":
            raise ValueError("a Red warmup opponent should use rush_asset or idle")
        if args.train_side == "Red" and args.warmup_opponent == "rush_asset":
            raise ValueError("a Blue warmup opponent should use intercept or idle")
    if len(set(args.initializations)) != len(args.initializations):
        raise ValueError("initializations must be unique")
    if "stock_transfer" in args.initializations and not args.stock_checkpoint_template:
        raise ValueError(
            "--stock-checkpoint-template is required for stock_transfer"
        )
    if args.formal_evidence:
        if args.reward_mode != "strict_potential":
            raise ValueError("formal evidence requires --reward-mode strict_potential")
        if args.encoder_kind != "saqa":
            raise ValueError("formal evidence requires --encoder-kind saqa")
        if not args.use_cpp_rvo2:
            raise ValueError(
                "the preregistered formal protocol requires --use-cpp-rvo2"
            )
        if len(args.algorithms) != 3 or set(args.algorithms) != {
            "qmix",
            "vdn",
            "mappo",
        }:
            raise ValueError(
                "formal evidence requires qmix, vdn and mappo in one invocation"
            )
        if args.train_side != "Red" or args.ratios != [(2, 1), (3, 2), (5, 3)]:
            raise ValueError(
                "formal evidence requires Red on ratios 2:1,3:2,5:3"
            )
        if not (
            args.opponent == "intercept"
            and args.warmup_opponent == "idle"
            and np.isclose(args.warmup_fraction, 0.25)
        ):
            raise ValueError(
                "formal evidence requires the idle-to-intercept opponent curriculum"
            )
        if not args.clear_replay_on_opponent_transition:
            raise ValueError(
                "formal evidence requires replay reset at the opponent transition"
            )
        if len(args.seeds) != len(set(args.seeds)):
            raise ValueError("formal evidence requires unique training seeds")
        if len(set(args.seeds)) < 5:
            raise ValueError("formal evidence requires at least five unique seeds")
        if set(args.initializations) != {"scratch", "stock_transfer"}:
            raise ValueError(
                "formal evidence requires scratch and stock_transfer arms"
            )
        if args.validation_episodes_per_ratio < 20:
            raise ValueError(
                "formal evidence requires at least 20 validation layouts per ratio"
            )
        if args.heldout_episodes_per_ratio < 20:
            raise ValueError(
                "formal evidence requires at least 20 held-out layouts per ratio"
            )
        args._formal_contract = validate_ad_formal_contract(args)


def validate_ad_formal_contract(args: argparse.Namespace) -> Dict[str, object]:
    """Require an exact match to the registered stock-to-AD fine-tuning YAML."""

    actual = {
        "encoder_kind": args.encoder_kind,
        "attention_heads": args.attention_heads,
        "initializations": list(args.initializations),
        "stock_checkpoint_template": args.stock_checkpoint_template,
        "paired_random_initial_draw": True,
        "paired_training_and_evaluation_layout_seeds": True,
        "ratios": [f"{red}v{blue}" for red, blue in args.ratios],
        "training_seeds": list(args.seeds),
        "train_side": args.train_side,
        "checkpoint_selection": "validation_best",
        "checkpoint_selection_metric": "lexicographic_worst_ratio_then_overall_mean",
        "heldout_used_once_after_selection": True,
        "target_opponent": args.opponent,
        "target_opponent_information_scope": "privileged_full_environment_state",
        "target_opponent_report_label": "privileged_full_state_threat",
        "solvability_rule": "clear_then_asset",
        "naive_lower_bound_rule": "rush_asset",
        "warmup_opponent": args.warmup_opponent,
        "warmup_fraction": args.warmup_fraction,
        "clear_replay_on_opponent_transition": (
            args.clear_replay_on_opponent_transition
        ),
        "mappo_pending_transition": (
            "flush_idle_pending_before_first_target_rollout"
        ),
        "validation_and_heldout_use_target_opponent_only": True,
        "scale_balanced_replay": True,
        "episodes": args.episodes,
        "episode_limit": args.episode_limit,
        "batch_episodes": args.batch_episodes,
        "replay_episodes": args.replay_episodes,
        "updates_per_episode": args.updates_per_episode,
        "eval_every": args.eval_every,
        "validation_episodes_per_ratio": args.validation_episodes_per_ratio,
        "heldout_episodes_per_ratio": args.heldout_episodes_per_ratio,
        "max_red_agents": args.max_red_agents,
        "max_blue_agents": args.max_blue_agents,
        "agent_hidden_dim": args.agent_hidden_dim,
        "critic_hidden_dim": args.critic_hidden_dim,
        "reward_mode": args.reward_mode,
        "gamma": args.gamma,
        "shaping_scale": args.shaping_scale,
        "approach_weight": args.approach_weight,
        "spawn_jitter": args.spawn_jitter,
        "use_cpp_rvo2": args.use_cpp_rvo2,
        "backend_mixed_within_comparison": False,
        "learning_rate": args.learning_rate,
        "target_update_interval": args.target_update_interval,
        "td_lambda": args.td_lambda,
        "ppo_epochs": args.ppo_epochs,
        "epsilon_start": args.epsilon_start,
        "epsilon_finish": args.epsilon_finish,
        "epsilon_anneal_steps": args.epsilon_anneal_steps,
        "minimum_training_seeds": len(set(args.seeds)),
        "task_mean_win_rate_threshold": 0.80,
        "task_every_ratio_win_rate_threshold": 0.70,
        "transfer_benefit_ci_threshold": 0.0,
        "plateau_window_evaluations": 5,
        "max_absolute_win_rate_slope_per_evaluation": 0.02,
        "max_win_rate_range": 0.10,
        "task_claim_requires_task_integrity_and_stability_gates": True,
        "transfer_benefit_is_a_separate_claim": True,
    }
    prefix = "ad_finetuning."
    paths = {
        label: prefix + label
        for label in actual
        if label
        not in {
            "stock_checkpoint_template",
            "training_seeds",
            "learning_rate",
            "target_update_interval",
            "td_lambda",
            "ppo_epochs",
            "epsilon_start",
            "epsilon_finish",
            "epsilon_anneal_steps",
            "minimum_training_seeds",
            "task_mean_win_rate_threshold",
            "task_every_ratio_win_rate_threshold",
            "transfer_benefit_ci_threshold",
            "plateau_window_evaluations",
            "max_absolute_win_rate_slope_per_evaluation",
            "max_win_rate_range",
            "task_claim_requires_task_integrity_and_stability_gates",
            "transfer_benefit_is_a_separate_claim",
        }
    }
    paths.update(
        {
            "stock_checkpoint_template": "artifacts.stock_validation_best_checkpoint",
            "training_seeds": "ad_finetuning.seeds",
            "learning_rate": "ad_finetuning.reused_stock_hyperparameters.learning_rate",
            "target_update_interval": (
                "ad_finetuning.reused_stock_hyperparameters.target_update_interval"
            ),
            "td_lambda": "ad_finetuning.reused_stock_hyperparameters.td_lambda",
            "ppo_epochs": "ad_finetuning.reused_stock_hyperparameters.ppo_epochs",
            "epsilon_start": (
                "ad_finetuning.reused_stock_hyperparameters.epsilon_start"
            ),
            "epsilon_finish": (
                "ad_finetuning.reused_stock_hyperparameters.epsilon_finish"
            ),
            "epsilon_anneal_steps": (
                "ad_finetuning.reused_stock_hyperparameters.epsilon_anneal_steps"
            ),
            "minimum_training_seeds": (
                "acceptance_gates.ad_task_per_algorithm.minimum_training_seeds"
            ),
            "task_mean_win_rate_threshold": (
                "acceptance_gates.ad_task_per_algorithm.heldout_win_rate_mean_ge"
            ),
            "task_every_ratio_win_rate_threshold": (
                "acceptance_gates.ad_task_per_algorithm."
                "every_ratio_win_rate_mean_ge"
            ),
            "transfer_benefit_ci_threshold": (
                "acceptance_gates.transfer_benefit_separate_hypothesis."
                "transfer_minus_scratch_seed_bootstrap_ci95_low_gt"
            ),
            "plateau_window_evaluations": (
                "acceptance_gates.convergence.plateau_window_evaluations"
            ),
            "max_absolute_win_rate_slope_per_evaluation": (
                "acceptance_gates.convergence."
                "max_absolute_win_rate_slope_per_evaluation"
            ),
            "max_win_rate_range": (
                "acceptance_gates.convergence.max_win_rate_range"
            ),
            "task_claim_requires_task_integrity_and_stability_gates": (
                "acceptance_gates.convergence."
                "task_claim_requires_task_integrity_and_stability_gates"
            ),
            "transfer_benefit_is_a_separate_claim": (
                "acceptance_gates.convergence.transfer_benefit_is_a_separate_claim"
            ),
        }
    )
    return validate_registered_formal_contract(
        project_root=PROJECT,
        config_path=STOCK_TO_AD_FORMAL_CONFIG,
        contract_name="stage1_smaclite_ad_finetuning",
        expected_protocol_version=STOCK_TO_AD_FORMAL_PROTOCOL,
        actual_values=actual,
        yaml_paths=paths,
        unordered_fields=("initializations", "training_seeds"),
        project_path_fields=("stock_checkpoint_template",),
    )


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


def robust_validation_key(evaluation) -> Tuple[float, float, float, float]:
    """Prefer the worst ratio before aggregate validation performance."""

    if not evaluation.per_ratio:
        raise ValueError("robust selection requires at least one ratio")
    worst_ratio = min(
        (
            float(cell["win_rate"]),
            float(cell["mean_return"]),
        )
        for cell in evaluation.per_ratio.values()
    )
    return (
        worst_ratio[0],
        worst_ratio[1],
        float(evaluation.win_rate),
        float(evaluation.mean_return),
    )


def reset_replay_at_opponent_transition(
    replay: EpisodeReplayBuffer, capacity: int, seed: int, episode: int
) -> Tuple[EpisodeReplayBuffer, Dict[str, object]]:
    """Drop warm-up trajectories and return a deterministically seeded buffer."""

    cleared = len(replay)
    return EpisodeReplayBuffer(capacity, seed=seed), {
        "performed": True,
        "transition_episode": int(episode),
        "cleared_episode_count": int(cleared),
        "minimum_refill_episodes_before_update": None,
        "first_post_transition_update_episode": None,
    }


def flush_mappo_pending_at_opponent_transition(
    pending: List[object], learner, device: torch.device, episode: int
):
    """Flush one all-warm-up PPO batch before the target opponent appears."""

    pending_count = len(pending)
    metrics = None
    if pending_count:
        metrics = learner.train_batch(collate_episodes(pending, device))
        pending.clear()
    return metrics, {
        "performed": True,
        "algorithm_path": "on_policy_pending_flush",
        "transition_episode": int(episode),
        "pending_episode_count_before_transition": pending_count,
        "flushed_idle_episode_count": pending_count,
        "discarded_idle_episode_count": 0,
        "mixed_opponent_batch_prevented": True,
    }


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
            encoder_kind=args.encoder_kind,
            attention_heads=args.attention_heads,
        )
    elif algorithm == "vdn":
        model = VariableScaleVDN(
            *dimensions,
            agent_hidden_dim=args.agent_hidden_dim,
            encoder_kind=args.encoder_kind,
            attention_heads=args.attention_heads,
        )
    elif algorithm == "mappo":
        model = VariableScaleMAPPO(
            *dimensions,
            actor_hidden_dim=args.agent_hidden_dim,
            critic_hidden_dim=args.critic_hidden_dim,
            encoder_kind=args.encoder_kind,
            attention_heads=args.attention_heads,
        )
    else:  # pragma: no cover - argparse guards this
        raise ValueError(algorithm)
    return model.to(device)


def reward_contract(args: argparse.Namespace) -> Dict[str, object]:
    """JSON-ready reward definition shared by checkpoints and run summaries."""

    return {
        "mode": args.reward_mode,
        "terminal_payoff": {
            "asset_destroyed": 1.0,
            "attackers_eliminated": -1.0,
            "asset_survived_horizon": -1.0,
        },
        "discount_gamma": float(args.gamma),
        "shaping_scale": float(args.shaping_scale),
        "approach_weight": float(args.approach_weight),
        "strict_potential_formula": "F(s,s') = gamma * Phi(s') - Phi(s)",
        "absorbing_terminal_potential": 0.0,
        "heuristic_delta_status": (
            "legacy_non_policy-invariant_ablation_only"
            if args.reward_mode == "heuristic_delta"
            else "not_active"
        ),
        "selection_primary_metric": "terminal_outcome_win_rate",
    }


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
    initialization: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    seed_everything(seed)
    factory = SMACliteADFactory(
        max_red_agents=args.max_red_agents,
        max_blue_agents=args.max_blue_agents,
        episode_limit=args.episode_limit,
        reward_mode=args.reward_mode,
        discount_gamma=args.gamma,
        shaping_scale=args.shaping_scale,
        approach_weight=args.approach_weight,
        spawn_jitter=args.spawn_jitter,
        use_cpp_rvo2=args.use_cpp_rvo2,
    )
    runner = SMACliteADEpisodeRunner(factory)
    shape_audit = tensor_shape_audit(factory, args.ratios, seed + 700_000)
    model = make_model(algorithm, factory.get(args.ratios[0]), device, args)
    random_initial_hash = model_sha256(model)
    transfer_manifest = None
    if initialization == "stock_transfer":
        source = Path(
            args.stock_checkpoint_template.format(algorithm=algorithm, seed=seed)
        )
        transfer_manifest = transfer_stock_checkpoint(model, source, algorithm)
        if bool(transfer_manifest["source_use_cpp_rvo2"]) != args.use_cpp_rvo2:
            raise ValueError(
                "stock source and AD fine-tuning must use the same RVO2 backend"
            )
        source_hyperparameters = transfer_manifest.get(
            "source_training_hyperparameters"
        )
        if not isinstance(source_hyperparameters, Mapping):
            raise ValueError("stock source lacks training_hyperparameters")
        required_hyperparameters = {
            "learning_rate": args.learning_rate,
            **(
                {
                    "ppo_epochs": args.ppo_epochs,
                }
                if algorithm == "mappo"
                else {
                    "target_update_interval": args.target_update_interval,
                    "td_lambda": args.td_lambda,
                }
            ),
        }
        for name, expected in required_hyperparameters.items():
            actual = source_hyperparameters.get(name)
            if actual is None or float(actual) != float(expected):
                raise ValueError(
                    f"stock source hyperparameter {name!r} does not match AD fine-tuning"
                )
    elif initialization != "scratch":
        raise ValueError(f"unknown initialization: {initialization}")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    initial_hash = model_sha256(model)
    initial_parameters = parameter_vector(model)
    initial_model_state = copy.deepcopy(model.state_dict())
    opponent = SMACliteADRuleController(args.opponent)
    warmup_opponent = (
        None
        if args.warmup_opponent is None
        else SMACliteADRuleController(args.warmup_opponent)
    )
    warmup_episodes = int(args.episodes * args.warmup_fraction)
    if algorithm == "mappo":
        learner = SequenceMAPPOLearner(
            model,
            learning_rate=args.learning_rate,
            epochs=args.ppo_epochs,
            gamma=args.gamma,
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
            gamma=args.gamma,
            td_lambda=args.td_lambda,
            target_update_interval=args.target_update_interval,
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
        args.output_dir
        / "checkpoints"
        / f"{algorithm}_{initialization}_seed{seed}_validation_best.pt"
    )
    best_validation_key = None
    best_validation_record = None
    best_validation_evaluation = None
    best_model_state = None
    last_validation_evaluation = None
    last_validation_episode = None
    replay_transition = {
        "enabled": bool(warmup_opponent is not None),
        "performed": False,
        "algorithm_path": (
            "on_policy_pending_flush" if algorithm == "mappo" else "off_policy_replay_reset"
        ),
        "transition_episode": (
            warmup_episodes + 1 if warmup_opponent is not None else None
        ),
        "cleared_episode_count": 0,
        "pending_episode_count_before_transition": 0,
        "flushed_idle_episode_count": 0,
        "discarded_idle_episode_count": 0,
        "mixed_opponent_batch_prevented": False,
        "minimum_refill_episodes_before_update": (
            args.batch_episodes if algorithm != "mappo" else None
        ),
        "first_post_transition_update_episode": None,
    }

    def add_curve(episode: int, evaluation) -> None:
        row = {
            "algorithm": algorithm,
            "initialization": initialization,
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
        key = robust_validation_key(evaluation)
        if best_validation_key is not None and key <= best_validation_key:
            return
        best_validation_key = key
        best_validation_evaluation = copy.deepcopy(evaluation)
        best_model_state = copy.deepcopy(model.state_dict())
        metadata = {
            "checkpoint_role": "validation_best",
            "git_provenance": args._git_provenance,
            "formal_contract": args._formal_contract,
            "selection_metric": [
                "worst_ratio_win_rate",
                "worst_ratio_mean_return",
                "overall_win_rate",
                "overall_mean_return",
            ],
            "selection_key": list(key),
            "selection_split": "validation",
            "algorithm": algorithm,
            "initialization": initialization,
            "transfer_manifest": transfer_manifest,
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
            "use_cpp_rvo2": args.use_cpp_rvo2,
            "reward_contract": reward_contract(args),
            "train_side": args.train_side,
            "target_opponent": opponent.name,
            "target_opponent_information_scope": opponent.information_scope,
            "warmup_opponent": (
                None if warmup_opponent is None else warmup_opponent.name
            ),
            "warmup_episodes": warmup_episodes,
            "replay_transition": copy.deepcopy(replay_transition),
            "opponent_transition": copy.deepcopy(replay_transition),
            "training_hyperparameters": {
                "learning_rate": args.learning_rate,
                "gamma": args.gamma,
                "target_update_interval": args.target_update_interval,
                "td_lambda": args.td_lambda,
                "ppo_epochs": args.ppo_epochs,
            },
            "contract": {
                "encoder_kind": args.encoder_kind,
                "attention_heads": args.attention_heads,
                "policy_architecture": (
                    "OpenSCORE-SAQA-QMIX-SP"
                    if algorithm == "qmix" and args.encoder_kind == "saqa"
                    else f"OpenSCORE-{args.encoder_kind}-{algorithm.upper()}"
                ),
                "action_target_contract": ACTION_TARGET_CONTRACT,
                "action_target_types": {
                    "0": "non_target",
                    "1": "enemy_damage",
                    "2": "ally_heal",
                    "3": "protected_asset",
                },
                "entity_row_order": "red_slots_then_blue_slots_then_asset",
                "target_scorer": "shared_per_entity_permutation_equivariant",
                "implementation_status": (
                    "Open-SCORE implementation inspired by REFIL Attention-QMIX "
                    "and SPECTra SAQA/target-action design; not an exact reproduction"
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
        if (
            warmup_opponent is not None
            and episode_index == warmup_episodes + 1
        ):
            if replay is not None and args.clear_replay_on_opponent_transition:
                replay, transition_record = reset_replay_at_opponent_transition(
                    replay,
                    args.replay_episodes,
                    seed + 1_000_029,
                    episode_index,
                )
                transition_record.update(
                    {
                        "algorithm_path": "off_policy_replay_reset",
                        "minimum_refill_episodes_before_update": args.batch_episodes,
                        "mixed_opponent_batch_prevented": True,
                    }
                )
                replay_transition.update(transition_record)
            elif algorithm == "mappo":
                assert pending is not None
                # All pending trajectories were generated against the idle
                # opponent. Flush them before the first intercept rollout.
                flushed_metrics, transition_record = (
                    flush_mappo_pending_at_opponent_transition(
                        pending, learner, device, episode_index
                    )
                )
                if flushed_metrics is not None:
                    latest_metrics = flushed_metrics
                    updated_ratios.update(latest_metrics.learning_signal_by_scale)
                replay_transition.update(transition_record)
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
        rollout_opponent = (
            warmup_opponent
            if warmup_opponent is not None and episode_index <= warmup_episodes
            else opponent
        )
        rollout = (
            runner.run(ratio, training_controller, rollout_opponent, rollout_seed)
            if args.train_side == "Red"
            else runner.run(ratio, rollout_opponent, training_controller, rollout_seed)
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
                if (
                    replay_transition["performed"]
                    and episode_index > warmup_episodes
                    and replay_transition["first_post_transition_update_episode"]
                    is None
                ):
                    replay_transition["first_post_transition_update_episode"] = (
                        episode_index
                    )
        else:
            replay.add(team_episode)
            if len(replay) >= args.batch_episodes:
                for _ in range(args.updates_per_episode):
                    if learner.learner_step == 0:
                        batch = collate_episodes(
                            replay.episodes[-args.batch_episodes :], device
                        )
                    else:
                        batch = replay.sample_scale_balanced(
                            args.batch_episodes, device
                        )
                    latest_metrics = learner.train_batch(batch)
                    updated_ratios.update(latest_metrics.td_by_scale)
                    if (
                        replay_transition["performed"]
                        and replay_transition["first_post_transition_update_episode"]
                        is None
                    ):
                        replay_transition["first_post_transition_update_episode"] = (
                            episode_index
                        )

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
        args.output_dir
        / "checkpoints"
        / f"{algorithm}_{initialization}_seed{seed}_final.pt"
    )
    final_metadata = {
        "checkpoint_role": "final",
        "git_provenance": args._git_provenance,
        "formal_contract": args._formal_contract,
        "environment": "OpenSCORE/SMACliteAD-Asset-v0",
        "algorithm": algorithm,
        "initialization": initialization,
        "transfer_manifest": transfer_manifest,
        "seed": seed,
        "episode": args.episodes,
        "environment_steps": environment_steps,
        "learner_updates": 0 if latest_metrics is None else latest_metrics.learner_step,
        "ratios": [list(ratio) for ratio in args.ratios],
        "spawn_jitter": args.spawn_jitter,
        "use_cpp_rvo2": args.use_cpp_rvo2,
        "reward_contract": reward_contract(args),
        "train_side": args.train_side,
        "target_opponent": opponent.name,
        "target_opponent_information_scope": opponent.information_scope,
        "warmup_opponent": (
            None if warmup_opponent is None else warmup_opponent.name
        ),
        "warmup_episodes": warmup_episodes,
        "replay_transition": copy.deepcopy(replay_transition),
        "opponent_transition": copy.deepcopy(replay_transition),
        "training_hyperparameters": {
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "target_update_interval": args.target_update_interval,
            "td_lambda": args.td_lambda,
            "ppo_epochs": args.ppo_epochs,
        },
        "contract": {
            "encoder_kind": args.encoder_kind,
            "attention_heads": args.attention_heads,
            "policy_architecture": (
                "OpenSCORE-SAQA-QMIX-SP"
                if algorithm == "qmix" and args.encoder_kind == "saqa"
                else f"OpenSCORE-{args.encoder_kind}-{algorithm.upper()}"
            ),
            "action_target_contract": ACTION_TARGET_CONTRACT,
            "action_target_types": {
                "0": "non_target",
                "1": "enemy_damage",
                "2": "ally_heal",
                "3": "protected_asset",
            },
            "entity_row_order": "red_slots_then_blue_slots_then_asset",
            "target_scorer": "shared_per_entity_permutation_equivariant",
            "implementation_status": (
                "Open-SCORE implementation inspired by REFIL Attention-QMIX "
                "and SPECTra SAQA/target-action design; not an exact reproduction"
            ),
            "spectra_audited_commit": "ffababf6187216c9d16b2109ee8ef6fe5fdf1172",
            "spectra_code_copied": False,
            "references": {
                "refil": "https://proceedings.mlr.press/v139/iqbal21a.html",
                "spectra_paper": "https://arxiv.org/abs/2503.11726",
                "spectra_repository": "https://github.com/funny-rl/SPECTra",
                "spmarl": "https://proceedings.mlr.press/v267/zhao25o.html",
            },
        },
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
        "initialization": initialization,
        "transfer_manifest": transfer_manifest,
        "random_initial_model_sha256": random_initial_hash,
        "seed": seed,
        "train_side": args.train_side,
        "opponent": opponent.name,
        "opponent_information_scope": opponent.information_scope,
        "reward_contract": reward_contract(args),
        "training_opponent_curriculum": {
            "warmup_opponent": (
                None if warmup_opponent is None else warmup_opponent.name
            ),
            "warmup_episodes": warmup_episodes,
            "target_opponent": opponent.name,
            "validation_and_heldout_use_target_only": True,
            "opponent_transition": copy.deepcopy(replay_transition),
            "off_policy_replay_transition": (
                copy.deepcopy(replay_transition) if algorithm != "mappo" else None
            ),
            "on_policy_pending_transition": (
                copy.deepcopy(replay_transition) if algorithm == "mappo" else None
            ),
        },
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
        "validation_best_parameters_changed": (
            best_validation_record["model_sha256"] != initial_hash
        ),
        "parameter_delta_l2": parameter_delta,
        "validation": {
            "seed_base": validation_seed,
            "episodes_per_ratio": args.validation_episodes_per_ratio,
            "initial": evaluation_dict(validation_initial),
            "final": evaluation_dict(validation_final),
            "final_comparison": validation_comparison,
            "best_checkpoint_evaluation": evaluation_dict(best_validation_evaluation),
            "checkpoint_selection": {
                "metric": [
                    "worst_ratio_win_rate",
                    "worst_ratio_mean_return",
                    "overall_win_rate",
                    "overall_mean_return",
                ],
                "lexicographic": True,
                "selected_key": list(best_validation_key),
            },
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


def seed_bootstrap_interval(
    values: Sequence[float], seed: int, draws: int = 20_000
) -> Dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot bootstrap an empty seed sample")
    rng = np.random.default_rng(seed)
    means = rng.choice(array, size=(draws, array.size), replace=True).mean(axis=1)
    return {
        "mean": float(array.mean()),
        "std_across_seeds": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "seed_count": int(array.size),
    }


def initialization_ablation(
    results: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    """Paired post-selection comparison of stock transfer against scratch."""

    indexed = {
        (str(result["algorithm"]), int(result["seed"]), str(result["initialization"])): result
        for result in results
    }
    comparisons = []
    for algorithm in sorted({str(result["algorithm"]) for result in results}):
        for seed in sorted({int(result["seed"]) for result in results}):
            scratch = indexed.get((algorithm, seed, "scratch"))
            transferred = indexed.get((algorithm, seed, "stock_transfer"))
            if scratch is None or transferred is None:
                continue
            scratch_eval = scratch["heldout"]["validation_best"]
            transfer_eval = transferred["heldout"]["validation_best"]
            if scratch_eval["layout_hashes"] != transfer_eval["layout_hashes"]:
                raise AssertionError("initialization ablation layouts are not paired")
            if (
                scratch["random_initial_model_sha256"]
                != transferred["random_initial_model_sha256"]
            ):
                raise AssertionError(
                    "scratch and transfer arms did not start from the same random draw"
                )
            scratch_returns = np.asarray(
                scratch_eval["paired_returns"], dtype=np.float64
            )
            transfer_returns = np.asarray(
                transfer_eval["paired_returns"], dtype=np.float64
            )
            delta = transfer_returns - scratch_returns
            comparisons.append(
                {
                    "algorithm": algorithm,
                    "seed": seed,
                    "episodes": int(delta.size),
                    "same_ordered_layouts": True,
                    "same_random_initial_draw": True,
                    "random_initial_model_sha256": scratch[
                        "random_initial_model_sha256"
                    ],
                    "transfer_minus_scratch_mean_return": float(delta.mean()),
                    "transfer_better_pair_fraction": float(np.mean(delta > 1e-6)),
                    "transfer_minus_scratch_win_rate": float(
                        transfer_eval["win_rate"] - scratch_eval["win_rate"]
                    ),
                    "scratch_mean_return": float(scratch_eval["mean_return"]),
                    "transfer_mean_return": float(transfer_eval["mean_return"]),
                }
            )
    aggregates = {}
    for algorithm_index, algorithm in enumerate(
        sorted({str(result["algorithm"]) for result in results})
    ):
        algorithm_results = [
            result for result in results if result["algorithm"] == algorithm
        ]
        arms = {}
        for initialization in ("scratch", "stock_transfer"):
            arm = [
                result
                for result in algorithm_results
                if result["initialization"] == initialization
            ]
            if not arm:
                continue
            arms[initialization] = {
                "heldout_mean_return": seed_bootstrap_interval(
                    [
                        float(result["heldout"]["validation_best"]["mean_return"])
                        for result in arm
                    ],
                    71_000 + algorithm_index * 10 + len(arms),
                ),
                "heldout_win_rate": seed_bootstrap_interval(
                    [
                        float(result["heldout"]["validation_best"]["win_rate"])
                        for result in arm
                    ],
                    72_000 + algorithm_index * 10 + len(arms),
                ),
            }
        algorithm_comparisons = [
            comparison
            for comparison in comparisons
            if comparison["algorithm"] == algorithm
        ]
        delta = None
        win_delta = None
        per_ratio_transfer_win_rate = None
        task_gate = {"passed": False, "reason": "paired arms were not both run"}
        integrity_gate = {"passed": False, "reason": "no transferred runs"}
        benefit_gate = {
            "passed": False,
            "status": "not_run",
            "reason": "paired arms were not both run",
        }
        if algorithm_comparisons:
            delta = seed_bootstrap_interval(
                [
                    float(comparison["transfer_minus_scratch_mean_return"])
                    for comparison in algorithm_comparisons
                ],
                73_000 + algorithm_index,
            )
            win_delta = seed_bootstrap_interval(
                [
                    float(comparison["transfer_minus_scratch_win_rate"])
                    for comparison in algorithm_comparisons
                ],
                73_500 + algorithm_index,
            )
            transfer_runs = [
                result
                for result in algorithm_results
                if result["initialization"] == "stock_transfer"
            ]
            ratio_labels = sorted(
                transfer_runs[0]["heldout"]["validation_best"]["per_ratio"]
            )
            per_ratio_transfer_win_rate = {
                label: seed_bootstrap_interval(
                    [
                        float(
                            result["heldout"]["validation_best"]["per_ratio"][label][
                                "win_rate"
                            ]
                        )
                        for result in transfer_runs
                    ],
                    74_000 + algorithm_index * 100 + ratio_index,
                )
                for ratio_index, label in enumerate(ratio_labels)
            }
            transfer_win = arms["stock_transfer"]["heldout_win_rate"]
            seed_minimum = int(transfer_win["seed_count"]) >= 5
            selected_trained = all(
                bool(result["validation_best_parameters_changed"])
                for result in transfer_runs
            )
            transfer_integrity = all(
                int(result["transfer_manifest"]["changed_copied_tensor_count"]) > 0
                and int(
                    result["transfer_manifest"]["copied_source_trained_tensor_count"]
                )
                > 0
                and int(
                    result["transfer_manifest"][
                        "source_selected_checkpoint_learner_updates"
                    ]
                )
                > 0
                and int(
                    result["transfer_manifest"][
                        "source_total_training_learner_updates"
                    ]
                )
                >= int(
                    result["transfer_manifest"][
                        "source_selected_checkpoint_learner_updates"
                    ]
                )
                for result in transfer_runs
            )
            task_gate = {
                "passed": bool(
                    seed_minimum
                    and selected_trained
                    and transfer_win["mean"] >= 0.80
                    and all(
                        value["mean"] >= 0.70
                        for value in per_ratio_transfer_win_rate.values()
                    )
                ),
                "minimum_training_seeds_met": seed_minimum,
                "selected_checkpoint_training_met": selected_trained,
                "requirements": [
                    "at least five independent training seeds",
                    "every selected AD checkpoint changed after initialization",
                    "stock-transfer held-out mean win rate >= 0.80",
                    "stock-transfer mean win rate >= 0.70 at every ratio",
                ],
            }
            integrity_gate = {
                "passed": transfer_integrity,
                "requirements": [
                    "every selected transfer source follows a learner update",
                    "total source updates are not earlier than selected-checkpoint updates",
                    "every copied latent set records source-training changes",
                ],
            }
            scratch_win = arms["scratch"]["heldout_win_rate"]
            ceiling_limited = bool(
                scratch_win["mean"] >= 0.95
                and transfer_win["mean"] >= 0.95
                and win_delta["mean"] >= -0.05
            )
            positive_return_evidence = bool(delta["ci95_low"] > 0.0)
            benefit_gate = {
                "passed": bool(transfer_integrity and positive_return_evidence),
                "status": (
                    "positive_transfer_supported"
                    if transfer_integrity and positive_return_evidence
                    else "ceiling_limited_noninferior"
                    if transfer_integrity and ceiling_limited
                    else "positive_transfer_not_supported"
                ),
                "transfer_integrity_met": transfer_integrity,
                "ceiling_limited_noninferiority": ceiling_limited,
                "requirements_for_positive_transfer_claim": [
                    "transfer integrity gate passes",
                    "seed-bootstrap CI95 lower bound of transfer-minus-scratch return > 0",
                ],
                "note": (
                    "This gate is reported separately and cannot invalidate a "
                    "task-success claim when both arms saturate near 100% wins."
                ),
            }
        aggregates[algorithm] = {
            "arms": arms,
            "transfer_minus_scratch_mean_return": delta,
            "transfer_minus_scratch_win_rate": win_delta,
            "stock_transfer_win_rate_by_ratio": per_ratio_transfer_win_rate,
            "task_success_gate": task_gate,
            "formal_task_gate": task_gate,
            "transfer_integrity_gate": integrity_gate,
            "transfer_benefit_gate": benefit_gate,
        }
    return {
        "status": "completed" if comparisons else "not_run",
        "unit_of_replication": "independent training seed",
        "selection": "validation-best checkpoints; held-out layouts used once",
        "paired_by": ["algorithm", "training_seed", "ordered_layout_hash"],
        "comparisons": comparisons,
        "aggregates": aggregates,
    }


def validation_plateau_audit(
    rows: Sequence[Mapping[str, object]], window: int = 5
) -> Dict[str, object]:
    grouped: Dict[Tuple[str, str, int], List[Mapping[str, object]]] = {}
    for row in rows:
        key = (
            str(row["algorithm"]),
            str(row["initialization"]),
            int(row["seed"]),
        )
        grouped.setdefault(key, []).append(row)
    runs = []
    for (algorithm, initialization, seed), values in sorted(grouped.items()):
        ordered = sorted(values, key=lambda value: int(value["episode"]))[-window:]
        win_rates = np.asarray(
            [float(value["eval_win_rate"]) for value in ordered], dtype=np.float64
        )
        enough = len(ordered) >= window
        slope = (
            float(np.polyfit(np.arange(len(win_rates)), win_rates, 1)[0])
            if len(win_rates) >= 2
            else float("inf")
        )
        win_range = float(np.ptp(win_rates)) if win_rates.size else float("inf")
        runs.append(
            {
                "algorithm": algorithm,
                "initialization": initialization,
                "seed": seed,
                "evaluations": len(ordered),
                "episode_start": int(ordered[0]["episode"]),
                "episode_end": int(ordered[-1]["episode"]),
                "win_rate_slope_per_evaluation": slope,
                "win_rate_range": win_range,
                "passed": bool(
                    enough and abs(slope) <= 0.02 and win_range <= 0.10
                ),
            }
        )
    transferred = [run for run in runs if run["initialization"] == "stock_transfer"]
    return {
        "split": "validation",
        "window_evaluations": window,
        "requirements": [
            "absolute win-rate slope per evaluation <= 0.02",
            "win-rate range across window <= 0.10",
        ],
        "runs": runs,
        "all_stock_transfer_runs_passed": bool(
            transferred and all(run["passed"] for run in transferred)
        ),
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    args._git_provenance = collect_and_require_git_provenance(
        PROJECT, formal=args.formal_evidence
    )
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_factory = SMACliteADFactory(
        max_red_agents=args.max_red_agents,
        max_blue_agents=args.max_blue_agents,
        episode_limit=args.episode_limit,
        reward_mode=args.reward_mode,
        discount_gamma=args.gamma,
        shaping_scale=args.shaping_scale,
        approach_weight=args.approach_weight,
        spawn_jitter=args.spawn_jitter,
        use_cpp_rvo2=args.use_cpp_rvo2,
    )
    reference_runner = SMACliteADEpisodeRunner(reference_factory)
    reference_controlled = SMACliteADRuleController(
        "clear_then_asset" if args.train_side == "Red" else "intercept"
    )
    naive_controlled = SMACliteADRuleController(
        "rush_asset" if args.train_side == "Red" else "idle"
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
    naive_rule_reference = evaluate_smaclite_ad(
        reference_runner,
        naive_controlled,
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
            for initialization in args.initializations:
                result, rows = train_one(
                    algorithm, seed, initialization, args, device
                )
                results.append(result)
                curves.extend(rows)
                print(
                    f"{algorithm} init={initialization} seed={seed}: "
                    f"updates={result['learner_updates']} "
                    f"delta={result['paired_comparison']['mean_return_delta']:.6f} "
                    f"status={result['paired_comparison']['learning_status']}"
                )
    ablation = initialization_ablation(results)
    stability = validation_plateau_audit(curves)
    task_gates_passed = bool(
        ablation["aggregates"]
        and all(
            aggregate["task_success_gate"]["passed"]
            for aggregate in ablation["aggregates"].values()
        )
    )
    transfer_integrity_gates_passed = bool(
        ablation["aggregates"]
        and all(
            aggregate["transfer_integrity_gate"]["passed"]
            for aggregate in ablation["aggregates"].values()
        )
    )
    transfer_benefit_claim = bool(
        ablation["aggregates"]
        and all(
            aggregate["transfer_benefit_gate"]["passed"]
            for aggregate in ablation["aggregates"].values()
        )
    )
    formal_convergence_claim = bool(
        args.formal_evidence
        and task_gates_passed
        and transfer_integrity_gates_passed
        and stability["all_stock_transfer_runs_passed"]
    )
    payload = _jsonify(
        {
            "schema_version": "smaclite-ad-baseline-v3",
            "status": (
                "formal_multi_seed_finetuning_protocol_executed"
                if args.formal_evidence
                else "randomized_layout_engineering_or_pilot_validation"
            ),
            "formal_convergence_claim": formal_convergence_claim,
            "task_success_gates_passed": task_gates_passed,
            "transfer_integrity_gates_passed": transfer_integrity_gates_passed,
            "positive_transfer_benefit_claim": transfer_benefit_claim,
            "git_provenance": args._git_provenance,
            "formal_contract": args._formal_contract,
            "environment": "OpenSCORE/SMACliteAD-Asset-v0",
            "environment_provenance": {
                "protocol_id": PROTOCOL_ID,
                "upstream_commit": UPSTREAM_COMMIT,
                "use_cpp_rvo2": args.use_cpp_rvo2,
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
            "reward_contract": reward_contract(args),
            "shared_policy_across_ratios": True,
            "episode_phase_in_actor_and_central_state": True,
            "stock_transfer": {
                "available": True,
                "unsafe_strict_false_loading": False,
                "fresh_optimizer_per_finetune": True,
                "initialization_ablation": ablation,
            },
            "validation_plateau_stability": stability,
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
            "rule_opponent_information_contract": {
                "scope": reference_opponent.information_scope,
                "label": "privileged_full_state_threat",
                "disclosure": (
                    "The deterministic rule reads simulator unit slots directly; "
                    "it is not constrained to the learned policy's local observation."
                ),
            },
            "rule_reference": {
                "purpose": "solvability reference, not a learned baseline",
                "controlled_policy": reference_controlled.name,
                "opponent_policy": reference_opponent.name,
                "evaluation": evaluation_dict(rule_reference, include_layouts=True),
            },
            "naive_rule_reference": {
                "purpose": "naive lower-bound rule, not a solvability claim",
                "controlled_policy": naive_controlled.name,
                "opponent_policy": reference_opponent.name,
                "evaluation": evaluation_dict(
                    naive_rule_reference, include_layouts=True
                ),
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
                if not key.startswith("_")
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
