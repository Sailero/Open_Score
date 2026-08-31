"""Train shared variable-entity models on unmodified stock SMAClite.

This produces auditable transfer-source checkpoints for SMAClite-AD.  It is
not a substitute for the official EPyMARL reproduction: raw stock observation
blocks are losslessly parsed into explicit ally/enemy entities while latent
recurrent/set layers share the implementation used by AD fine-tuning.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch

from open_score.formal_contracts import validate_registered_formal_contract
from open_score.provenance import collect_and_require_git_provenance
from open_score.envs.smaclite_ad import (
    UPSTREAM_COMMIT,
    SMACliteStockAdapter,
    stock_scenario_fingerprint,
)
from open_score.stage1.baselines import (
    SequenceMAPPOLearner,
    VariableScaleMAPPO,
    VariableScaleVDN,
)
from open_score.stage1.entity_qmix import VariableScaleQMIX
from open_score.stage1.learner import SequenceQMIXLearner, linear_epsilon
from open_score.stage1.replay import EpisodeReplayBuffer, collate_episodes
from open_score.stage1.smaclite_ad_training import (
    SMACliteADMAPPOController,
    SMACliteADQController,
)
from open_score.stage1.smaclite_stock_training import (
    SMACliteStockEpisodeRunner,
    evaluate_stock,
)
from open_score.stage1.transfer import (
    ACTION_TARGET_CONTRACT,
    file_sha256,
    model_state_sha256,
)


PROJECT = Path(__file__).resolve().parents[1]
STOCK_SOURCE_FORMAL_CONFIG = PROJECT / "configs" / "stage1_stock_to_ad_formal.yaml"
STOCK_TO_AD_FORMAL_PROTOCOL = "stock-to-ad-transfer-v3-saqa-strict-potential"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-name", default="3s5z")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=("qmix", "vdn", "mappo"),
        default=("qmix", "vdn", "mappo"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(20260830,))
    parser.add_argument("--episodes", type=int, default=2_000)
    parser.add_argument("--episode-limit", type=int, default=150)
    parser.add_argument("--batch-episodes", type=int, default=8)
    parser.add_argument("--replay-episodes", type=int, default=512)
    parser.add_argument("--updates-per-episode", type=int, default=1)
    parser.add_argument("--target-update-interval", type=int, default=200)
    parser.add_argument("--td-lambda", type=float, default=0.6)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--validation-episodes", type=int, default=32)
    parser.add_argument("--heldout-episodes", type=int, default=64)
    parser.add_argument(
        "--minimum-source-win-rate",
        type=float,
        default=0.20,
        help="Minimum held-out win rate required from every transfer source.",
    )
    parser.add_argument("--minimum-source-seeds", type=int, default=1)
    parser.add_argument(
        "--formal-evidence",
        action="store_true",
        help="Require at least five independent source-training seeds.",
    )
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-finish", type=float, default=0.05)
    parser.add_argument("--epsilon-anneal-steps", type=int, default=100_000)
    parser.add_argument("--agent-hidden-dim", type=int, default=128)
    parser.add_argument("--critic-hidden-dim", type=int, default=128)
    parser.add_argument(
        "--encoder-kind", choices=("deepset", "saqa"), default="deepset"
    )
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--use-cpp-rvo2",
        action="store_true",
        help="Use the compiled SMAClite RVO2 backend for every stock run.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/smaclite_stock_entity")
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args._formal_contract = None
    for name in (
        "episodes",
        "episode_limit",
        "batch_episodes",
        "replay_episodes",
        "updates_per_episode",
        "target_update_interval",
        "eval_every",
        "validation_episodes",
        "heldout_episodes",
        "ppo_epochs",
        "agent_hidden_dim",
        "critic_hidden_dim",
        "minimum_source_seeds",
        "attention_heads",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.replay_episodes < args.batch_episodes:
        raise ValueError("replay_episodes must be at least batch_episodes")
    if args.encoder_kind == "saqa" and args.agent_hidden_dim % args.attention_heads:
        raise ValueError("attention_heads must divide agent_hidden_dim for SAQA")
    if not 0.0 <= args.td_lambda <= 1.0:
        raise ValueError("td_lambda must be in [0, 1]")
    if not 0.0 < args.minimum_source_win_rate <= 1.0:
        raise ValueError("minimum_source_win_rate must be in (0, 1]")
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("source-training seeds must be unique")
    if args.formal_evidence:
        if args.encoder_kind != "saqa":
            raise ValueError("formal source evidence requires --encoder-kind saqa")
        if len(set(args.seeds)) < 5 or args.minimum_source_seeds < 5:
            raise ValueError("formal source evidence requires at least five seeds")
        if args.heldout_episodes < 64:
            raise ValueError("formal source evidence requires 64 held-out episodes")
        args._formal_contract = validate_stock_source_formal_contract(args)


def validate_stock_source_formal_contract(args: argparse.Namespace) -> Dict[str, object]:
    """Require the same-architecture stock source to match its tracked YAML."""

    actual = {
        "environment": f"smaclite/{args.map_name}-v0",
        "map_files_modified": False,
        "algorithms": list(args.algorithms),
        "training_seeds": list(args.seeds),
        "episode_limit": args.episode_limit,
        "episodes": args.episodes,
        "batch_episodes": args.batch_episodes,
        "replay_episodes": args.replay_episodes,
        "updates_per_episode": args.updates_per_episode,
        "eval_every": args.eval_every,
        "validation_episodes": args.validation_episodes,
        "heldout_episodes": args.heldout_episodes,
        "minimum_source_seeds": args.minimum_source_seeds,
        "minimum_source_win_rate": args.minimum_source_win_rate,
        "agent_hidden_dim": args.agent_hidden_dim,
        "critic_hidden_dim": args.critic_hidden_dim,
        "encoder_kind": args.encoder_kind,
        "attention_heads": args.attention_heads,
        "learning_rate": args.learning_rate,
        "target_update_interval": args.target_update_interval,
        "td_lambda": args.td_lambda,
        "ppo_epochs": args.ppo_epochs,
        "epsilon_start": args.epsilon_start,
        "epsilon_finish": args.epsilon_finish,
        "epsilon_anneal_steps": args.epsilon_anneal_steps,
        "use_cpp_rvo2": args.use_cpp_rvo2,
        "checkpoint_selection": (
            "validation_best_after_at_least_one_gradient_update"
        ),
        "checkpoint_selection_metric": ["win_rate", "mean_return"],
        "heldout_used_for_selection": False,
    }
    paths = {
        label: f"same_architecture_stock_pretraining.{label}" for label in actual
    }
    paths["training_seeds"] = "same_architecture_stock_pretraining.seeds"
    return validate_registered_formal_contract(
        project_root=PROJECT,
        config_path=STOCK_SOURCE_FORMAL_CONFIG,
        contract_name="stage1_same_architecture_stock_source",
        expected_protocol_version=STOCK_TO_AD_FORMAL_PROTOCOL,
        actual_values=actual,
        yaml_paths=paths,
        unordered_fields=("algorithms", "training_seeds"),
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


def make_model(algorithm: str, env: SMACliteStockAdapter, args, device):
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
    else:
        model = VariableScaleMAPPO(
            *dimensions,
            actor_hidden_dim=args.agent_hidden_dim,
            critic_hidden_dim=args.critic_hidden_dim,
            encoder_kind=args.encoder_kind,
            attention_heads=args.attention_heads,
        )
    return model.to(device)


def checkpoint_metadata(algorithm: str, seed: int, env, args, **extra):
    return {
        "algorithm": algorithm,
        "seed": seed,
        "environment_family": "SMAClite-stock",
        "environment_id": env.environment_id,
        "upstream_commit": UPSTREAM_COMMIT,
        "adapter": "lossless_stock_blocks_to_explicit_target_entities_v2",
        "purpose": "same-architecture stock pretraining source for AD fine-tuning",
        "official_reproduction_claim": False,
        "episode_phase_in_actor_and_central_state": True,
        "use_cpp_rvo2": args.use_cpp_rvo2,
        "git_provenance": args._git_provenance,
        "formal_contract": args._formal_contract,
        "contract": {
            "entity_dim": env.ENTITY_DIM,
            "self_dim": env.SELF_DIM,
            "task_dim": env.TASK_DIM,
            "state_entity_dim": env.STATE_ENTITY_DIM,
            "action_dim": env.ACTION_DIM,
            "agent_hidden_dim": args.agent_hidden_dim,
            "critic_hidden_dim": args.critic_hidden_dim,
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
            "stock_entity_row_order": "allies_by_id_then_enemies_by_id",
            "target_scorer": "shared_per_entity_permutation_equivariant",
            "implementation_status": (
                "Open-SCORE implementation inspired by REFIL Attention-QMIX "
                "and SPECTra SAQA/target-action design; not an exact reproduction"
            ),
            "references": {
                "refil": "https://proceedings.mlr.press/v139/iqbal21a.html",
                "spectra_paper": "https://arxiv.org/abs/2503.11726",
                "spectra_repository": "https://github.com/funny-rl/SPECTra",
                "spectra_audited_commit": "ffababf6187216c9d16b2109ee8ef6fe5fdf1172",
                "spectra_code_copied": False,
                "spmarl": "https://proceedings.mlr.press/v267/zhao25o.html",
            },
        },
        "training_hyperparameters": {
            "learning_rate": args.learning_rate,
            "target_update_interval": args.target_update_interval,
            "td_lambda": args.td_lambda,
            "ppo_epochs": args.ppo_epochs,
        },
        **extra,
    }


def source_performance_gate(evaluation, args) -> Dict[str, object]:
    """Held-out, task-level gate; structural weight changes are insufficient."""

    passed = bool(
        evaluation.episodes == args.heldout_episodes
        and evaluation.win_rate >= args.minimum_source_win_rate
        and sum(evaluation.paired_wins) > 0
    )
    return {
        "passed": passed,
        "split": "heldout_after_validation_selection",
        "episodes": int(evaluation.episodes),
        "seed_base_policy": "training_seed + 18000000",
        "win_rate": float(evaluation.win_rate),
        "mean_return": float(evaluation.mean_return),
        "wins": int(sum(evaluation.paired_wins)),
        "minimum_win_rate": float(args.minimum_source_win_rate),
        "requirements": [
            "held-out episode count matches the preregistered request",
            "held-out win rate meets the configured positive threshold",
            "at least one held-out battle is won",
        ],
    }


def _seed_bootstrap_interval(
    values: Sequence[float], seed: int, draws: int = 20_000
) -> Dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot aggregate an empty source sample")
    rng = np.random.default_rng(seed)
    means = rng.choice(array, size=(draws, array.size), replace=True).mean(axis=1)
    return {
        "mean": float(array.mean()),
        "std_across_seeds": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "seed_count": int(array.size),
    }


def aggregate_source_performance_gates(
    results: Sequence[Mapping[str, object]], args
) -> Dict[str, object]:
    algorithms: Dict[str, object] = {}
    for index, algorithm in enumerate(sorted(set(args.algorithms))):
        runs = [run for run in results if run["algorithm"] == algorithm]
        win_rates = [
            float(run["heldout_selected"]["win_rate"]) for run in runs
        ]
        seed_count = len({int(run["seed"]) for run in runs})
        interval = _seed_bootstrap_interval(win_rates, 62_000 + index)
        individual_pass = all(
            bool(run["source_performance_gate"]["passed"]) for run in runs
        )
        passed = bool(
            seed_count >= args.minimum_source_seeds
            and individual_pass
            and interval["mean"] >= args.minimum_source_win_rate
        )
        algorithms[algorithm] = {
            "passed": passed,
            "unit_of_replication": "independent training seed",
            "seeds": sorted(int(run["seed"]) for run in runs),
            "heldout_win_rate": interval,
            "minimum_seed_count": int(args.minimum_source_seeds),
            "minimum_mean_win_rate": float(args.minimum_source_win_rate),
            "every_source_checkpoint_passed": individual_pass,
            "no_zero_win_structural_only_acceptance": True,
        }
    return {
        "passed": all(bool(gate["passed"]) for gate in algorithms.values()),
        "algorithms": algorithms,
    }


def annotate_aggregate_gate(
    checkpoint_path: Path, aggregate_gate: Mapping[str, object]
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    extra = checkpoint.get("extra")
    if not isinstance(extra, dict):
        raise TypeError("source checkpoint extra metadata must be mutable mapping")
    extra["source_multi_seed_performance_gate"] = dict(aggregate_gate)
    torch.save(checkpoint, checkpoint_path)


def train_one(algorithm: str, seed: int, args, device) -> Mapping[str, object]:
    seed_everything(seed)
    env = SMACliteStockAdapter(
        args.map_name,
        episode_limit=args.episode_limit,
        seed=seed,
        use_cpp_rvo2=args.use_cpp_rvo2,
    )
    runner = SMACliteStockEpisodeRunner(env)
    model = make_model(algorithm, env, args, device)
    initial_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    initial_hash = model_state_sha256(model.state_dict())
    if algorithm == "mappo":
        learner = SequenceMAPPOLearner(
            model, learning_rate=args.learning_rate, epochs=args.ppo_epochs
        )
        train_controller = SMACliteADMAPPOController(
            model, device, deterministic=False, name="stock_mappo_train"
        )
        eval_controller = SMACliteADMAPPOController(
            model, device, deterministic=True, name="stock_mappo_eval"
        )
        pending: List[object] = []
        replay = None
    else:
        learner = SequenceQMIXLearner(
            model,
            learning_rate=args.learning_rate,
            td_lambda=args.td_lambda,
            target_update_interval=args.target_update_interval,
        )
        train_controller = SMACliteADQController(
            model, device, epsilon=args.epsilon_start, name=f"stock_{algorithm}_train"
        )
        eval_controller = SMACliteADQController(
            model, device, epsilon=0.0, name=f"stock_{algorithm}_eval"
        )
        replay = EpisodeReplayBuffer(args.replay_episodes, seed=seed + 29)
        pending = []
    validation_seed = seed + 8_000_000
    heldout_seed = seed + 18_000_000
    initial = evaluate_stock(
        runner, eval_controller, args.validation_episodes, validation_seed
    )
    # The transfer source must contain weights that received stock gradients.
    # Initial evaluation is a baseline only and is deliberately ineligible for
    # checkpoint selection.
    best_key = None
    best_state = None
    best_episode = None
    best_learner_updates = None
    environment_steps = 0
    latest_metrics = None
    curve: List[Dict[str, object]] = [
        {"episode": 0, **asdict(initial), "environment_steps": 0}
    ]
    for episode in range(1, args.episodes + 1):
        if isinstance(train_controller, SMACliteADQController):
            train_controller.epsilon = linear_epsilon(
                environment_steps,
                start=args.epsilon_start,
                finish=args.epsilon_finish,
                anneal_steps=args.epsilon_anneal_steps,
            )
        rollout = runner.run(train_controller, seed + episode * 101)
        environment_steps += rollout.length
        if algorithm == "mappo":
            pending.append(rollout.team)
            if len(pending) >= args.batch_episodes:
                latest_metrics = learner.train_batch(
                    collate_episodes(pending, device)
                )
                pending.clear()
        else:
            replay.add(rollout.team)
            if len(replay) >= args.batch_episodes:
                for _ in range(args.updates_per_episode):
                    latest_metrics = learner.train_batch(
                        replay.sample(args.batch_episodes, device)
                    )
        if episode % args.eval_every == 0 or episode == args.episodes:
            evaluation = evaluate_stock(
                runner, eval_controller, args.validation_episodes, validation_seed
            )
            curve.append(
                {
                    "episode": episode,
                    **asdict(evaluation),
                    "environment_steps": environment_steps,
                }
            )
            key = (evaluation.win_rate, evaluation.mean_return)
            if best_key is None or key > best_key:
                best_key = key
                best_state = copy.deepcopy(model.state_dict())
                best_episode = episode
                best_learner_updates = (
                    0 if latest_metrics is None else int(latest_metrics.learner_step)
                )
    if algorithm == "mappo" and pending:
        latest_metrics = learner.train_batch(collate_episodes(pending, device))
        pending.clear()
        evaluation = evaluate_stock(
            runner, eval_controller, args.validation_episodes, validation_seed
        )
        key = (evaluation.win_rate, evaluation.mean_return)
        if best_key is None or key > best_key:
            best_key = key
            best_state = copy.deepcopy(model.state_dict())
            best_episode = args.episodes
            best_learner_updates = (
                0 if latest_metrics is None else int(latest_metrics.learner_step)
            )
    if (
        best_state is None
        or best_key is None
        or best_episode is None
        or best_learner_updates is None
    ):
        raise AssertionError("no trained stock checkpoint was eligible for selection")
    if best_learner_updates < 1:
        raise AssertionError(
            "validation-selected stock checkpoint predates the first learner update"
        )
    total_training_learner_updates = (
        0 if latest_metrics is None else int(latest_metrics.learner_step)
    )
    final_path = args.output_dir / f"{algorithm}_seed{seed}_final.pt"
    learner.save(
        final_path,
        checkpoint_metadata(
            algorithm,
            seed,
            env,
            args,
            selection="final",
            episode=args.episodes,
            environment_steps=environment_steps,
            total_training_learner_updates=total_training_learner_updates,
        ),
    )
    model.load_state_dict(best_state)
    source_trained_tensor_names = sorted(
        name
        for name, tensor in model.state_dict().items()
        if not torch.equal(tensor.detach().cpu(), initial_state[name])
    )
    if not source_trained_tensor_names:
        raise AssertionError("selected stock checkpoint contains no changed tensors")
    target_embedding_name = (
        "actor.policy_head.target_type_embedding.weight"
        if algorithm == "mappo"
        else "agent.q_head.target_type_embedding.weight"
    )
    selected_embedding = model.state_dict()[target_embedding_name].detach().cpu()
    initial_embedding = initial_state[target_embedding_name]
    source_trained_target_type_rows = {
        str(row): semantic
        for row, semantic in {1: "enemy_damage", 2: "ally_heal"}.items()
        if not torch.equal(selected_embedding[row], initial_embedding[row])
    }
    if source_trained_target_type_rows.get("1") != "enemy_damage":
        raise AssertionError(
            "selected stock checkpoint has no row-level enemy-damage embedding update"
        )
    # Rebuild target/optimizer around the selected online/model weights, so the
    # transfer-source checkpoint contains no stale final-training state.
    if algorithm == "mappo":
        selected_learner = SequenceMAPPOLearner(
            model, learning_rate=args.learning_rate, epochs=args.ppo_epochs
        )
    else:
        selected_learner = SequenceQMIXLearner(
            model,
            learning_rate=args.learning_rate,
            td_lambda=args.td_lambda,
            target_update_interval=args.target_update_interval,
        )
    heldout = evaluate_stock(
        runner, eval_controller, args.heldout_episodes, heldout_seed
    )
    performance_gate = source_performance_gate(heldout, args)
    selected_path = args.output_dir / f"{algorithm}_seed{seed}.pt"
    selected_learner.save(
        selected_path,
        checkpoint_metadata(
            algorithm,
            seed,
            env,
            args,
            selection="validation_best",
            episode=best_episode,
            environment_steps=environment_steps,
            # Keep the legacy key for old readers, but make its selected-
            # checkpoint meaning explicit and separately record final totals.
            training_learner_updates=best_learner_updates,
            selected_checkpoint_learner_updates=best_learner_updates,
            total_training_learner_updates=total_training_learner_updates,
            initial_model_sha256=initial_hash,
            selected_model_sha256=model_state_sha256(model.state_dict()),
            source_trained_tensor_names=source_trained_tensor_names,
            source_trained_target_type_rows=source_trained_target_type_rows,
            validation_key=list(best_key),
            heldout_seed_base=heldout_seed,
            source_performance_gate=performance_gate,
        ),
    )
    result = {
        "algorithm": algorithm,
        "seed": seed,
        "initial_model_sha256": initial_hash,
        "selected_model_sha256": model_state_sha256(model.state_dict()),
        "parameters_changed": initial_hash != model_state_sha256(model.state_dict()),
        "source_trained_tensor_count": len(source_trained_tensor_names),
        "source_trained_tensor_names": source_trained_tensor_names,
        "source_trained_target_type_rows": source_trained_target_type_rows,
        "episodes": args.episodes,
        "environment_steps": environment_steps,
        "learner_updates": total_training_learner_updates,
        "selected_checkpoint_learner_updates": best_learner_updates,
        "total_training_learner_updates": total_training_learner_updates,
        "validation_initial": asdict(initial),
        "validation_best_key": list(best_key),
        "validation_best_episode": best_episode,
        "heldout_selected": asdict(heldout),
        "source_performance_gate": performance_gate,
        "checkpoint": str(selected_path.resolve()),
        "checkpoint_sha256": file_sha256(selected_path),
        "final_checkpoint": str(final_path.resolve()),
        "final_checkpoint_sha256": file_sha256(final_path),
        "curve": curve,
    }
    env.close()
    return result


def main() -> None:
    args = parse_args()
    validate_args(args)
    args._git_provenance = collect_and_require_git_provenance(
        PROJECT, formal=args.formal_evidence
    )
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [
        train_one(algorithm, seed, args, device)
        for seed in args.seeds
        for algorithm in args.algorithms
    ]
    source_gate = aggregate_source_performance_gates(results, args)
    for result in results:
        algorithm_gate = source_gate["algorithms"][result["algorithm"]]
        checkpoint_path = Path(result["checkpoint"])
        annotate_aggregate_gate(checkpoint_path, algorithm_gate)
        result["source_multi_seed_performance_gate"] = algorithm_gate
        result["checkpoint_sha256"] = file_sha256(checkpoint_path)
    payload = {
        "schema_version": "smaclite-stock-entity-pretraining-v2",
        "status": "transfer_source_training",
        "official_reproduction_claim": False,
        "official_reproduction_pointer": "use EPyMARL stock runs for benchmark claims",
        "upstream": stock_scenario_fingerprint(),
        "episode_phase_in_actor_and_central_state": True,
        "use_cpp_rvo2": args.use_cpp_rvo2,
        "device": str(device),
        "git_provenance": args._git_provenance,
        "formal_contract": args._formal_contract,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if not key.startswith("_")
        },
        "results": results,
        "source_performance_gate": source_gate,
    }
    summary = args.output_dir / "summary.json"
    summary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {summary.resolve()}")
    if not source_gate["passed"]:
        raise RuntimeError(
            "stock source performance gate failed; checkpoints are recorded but "
            "must not be used for AD transfer"
        )


if __name__ == "__main__":
    main()
