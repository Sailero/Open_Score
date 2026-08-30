"""Evaluate a runnable HAD rule policy or learned Stage-1 checkpoint."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from open_score.envs import HADStage1Adapter
from open_score.stage1 import (
    CompetitiveEpisodeRunner,
    HADStage1Factory,
    MAPPOController,
    QMixController,
    RuleBasedController,
    SequenceMAPPOLearner,
    SequenceQMIXLearner,
    VariableScaleMAPPO,
    VariableScaleQMIX,
    VariableScaleVDN,
    evaluate_pair,
    supported_scales,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("rule", "checkpoint"), required=True)
    parser.add_argument("--side", choices=("Red", "Blue"), default="Red")
    parser.add_argument("--rule-style", default=None)
    parser.add_argument("--algorithm", choices=("qmix", "vdn", "mappo"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=28260830)
    parser.add_argument("--episodes-per-scale", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--shaping-scale", type=float, default=0.5)
    parser.add_argument("--fixed-scale", nargs=2, type=int, default=None)
    parser.add_argument("--agent-hidden-dim", type=int, default=64)
    parser.add_argument("--critic-hidden-dim", type=int, default=64)
    parser.add_argument("--mixing-dim", type=int, default=32)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_model(args: argparse.Namespace, device: torch.device):
    dimensions = (
        HADStage1Adapter.ENTITY_DIM,
        HADStage1Adapter.SELF_DIM,
        HADStage1Adapter.TASK_DIM,
        HADStage1Adapter.STATE_ENTITY_DIM,
        HADStage1Adapter.ACTION_DIM,
    )
    if args.algorithm == "qmix":
        return VariableScaleQMIX(
            *dimensions,
            agent_hidden_dim=args.agent_hidden_dim,
            mixer_hidden_dim=args.critic_hidden_dim,
            mixing_dim=args.mixing_dim,
        ).to(device)
    if args.algorithm == "vdn":
        return VariableScaleVDN(
            *dimensions, agent_hidden_dim=args.agent_hidden_dim
        ).to(device)
    if args.algorithm == "mappo":
        return VariableScaleMAPPO(
            *dimensions,
            actor_hidden_dim=args.agent_hidden_dim,
            critic_hidden_dim=args.critic_hidden_dim,
        ).to(device)
    raise ValueError("--algorithm is required for checkpoint policies")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.episodes_per_scale < 1 or args.max_steps < 1:
        raise ValueError("episode counts and max_steps must be positive")
    scales = (
        [tuple(args.fixed_scale)]
        if args.fixed_scale is not None
        else supported_scales(4, 1)
    )
    if any(scale not in supported_scales(4, 1) for scale in scales):
        raise ValueError("every scale must satisfy strict Red superiority")
    device = choose_device(args.device)
    checkpoint_extra = None
    checkpoint_hash = None
    protocol_validation = {"status": "not_applicable_to_rule"}
    if args.policy == "rule":
        style = args.rule_style or ("guard" if args.side == "Red" else "rush")
        controlled = RuleBasedController(style)
        policy_name = controlled.name
    else:
        if args.checkpoint is None or not args.checkpoint.is_file():
            raise ValueError("a readable --checkpoint is required")
        model = make_model(args, device)
        if args.algorithm == "mappo":
            learner = SequenceMAPPOLearner(model)
            checkpoint_extra = learner.load(args.checkpoint, map_location=device)
            controlled = MAPPOController(
                model, device, deterministic=True, name="checkpoint:mappo"
            )
        else:
            learner = SequenceQMIXLearner(model)
            checkpoint_extra = learner.load(args.checkpoint, map_location=device)
            controlled = QMixController(
                model, device, epsilon=0.0, name=f"checkpoint:{args.algorithm}"
            )
        if checkpoint_extra.get("algorithm") != args.algorithm:
            raise ValueError("checkpoint metadata algorithm mismatch")
        if checkpoint_extra.get("train_side") != args.side:
            raise ValueError("checkpoint metadata train_side mismatch")
        reward_protocol = checkpoint_extra.get("reward_protocol", {})
        saved_shaping = reward_protocol.get("potential_shaping_scale")
        if saved_shaping is not None and float(saved_shaping) != args.shaping_scale:
            raise ValueError("checkpoint shaping scale does not match evaluation")
        saved_scales = {
            tuple(int(value) for value in scale)
            for scale in checkpoint_extra.get("registered_scales", [])
        }
        if saved_scales and not set(scales).issubset(saved_scales):
            raise ValueError("evaluation requests a scale outside checkpoint metadata")
        model_config = checkpoint_extra.get("model_config", {})
        expected_model_config = {
            "agent_hidden_dim": args.agent_hidden_dim,
            "critic_or_mixer_hidden_dim": args.critic_hidden_dim,
            "mixing_dim": args.mixing_dim,
        }
        for key, expected in expected_model_config.items():
            saved = model_config.get(key)
            if saved is not None and int(saved) != int(expected):
                raise ValueError(f"checkpoint model config mismatch: {key}")
        environment_protocol = checkpoint_extra.get("environment_protocol", {})
        saved_max_steps = environment_protocol.get("max_steps")
        if saved_max_steps is not None and int(saved_max_steps) != args.max_steps:
            raise ValueError("checkpoint max_steps does not match evaluation")
        missing_metadata = []
        if not model_config:
            missing_metadata.append("model_config")
        if saved_shaping is None:
            missing_metadata.append("reward_protocol.potential_shaping_scale")
        if not saved_scales:
            missing_metadata.append("registered_scales")
        if saved_max_steps is None:
            missing_metadata.append("environment_protocol.max_steps")
        protocol_validation = {
            "status": (
                "fully_checked"
                if not missing_metadata
                else "legacy_checkpoint_incomplete_metadata"
            ),
            "algorithm_side_checked": True,
            "model_config_checked": bool(model_config),
            "reward_shaping_checked": saved_shaping is not None,
            "registered_scales_checked": bool(saved_scales),
            "max_steps_checked": saved_max_steps is not None,
            "missing_metadata": missing_metadata,
            "warning": (
                None
                if not missing_metadata
                else "Legacy metadata is incomplete; unchecked CLI settings cannot be proven to match training."
            ),
        }
        policy_name = controlled.name
        checkpoint_hash = sha256(args.checkpoint)

    runner = CompetitiveEpisodeRunner(
        HADStage1Factory(
            max_steps=args.max_steps,
            shaping_scale=args.shaping_scale,
        )
    )
    opponent = RuleBasedController("rush" if args.side == "Red" else "guard")
    if args.side == "Red":
        summary = evaluate_pair(
            runner,
            controlled,
            opponent,
            scales,
            args.episodes_per_scale,
            args.seed,
        )
        sign = 1.0
        win_rate = summary.defender_win_rate
    else:
        summary = evaluate_pair(
            runner,
            opponent,
            controlled,
            scales,
            args.episodes_per_scale,
            args.seed,
        )
        sign = -1.0
        win_rate = summary.attacker_win_rate
    payload = {
        "policy": policy_name,
        "side": args.side,
        "device": str(device),
        "strict_red_superiority": True,
        "scales": [list(scale) for scale in scales],
        "episodes": summary.episodes,
        "controlled_mean_payoff": sign * summary.mean_defender_payoff,
        "controlled_win_rate": win_rate,
        "mean_episode_length": summary.mean_episode_length,
        "per_scale_controlled_payoff": {
            f"{red}v{blue}": sign * value
            for (red, blue), value in summary.per_scale_payoff.items()
        },
        "checkpoint": None if args.checkpoint is None else str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_extra": checkpoint_extra,
        "protocol_validation": protocol_validation,
        "formal_convergence_claim": False,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
