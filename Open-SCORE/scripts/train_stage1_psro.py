"""Train the executable HAD Stage-1 PSRO loop.

The default is a deliberately small 2v2 pipeline pilot.  Use
``--mode curriculum`` for the registered 1--4 population curriculum.  A pilot
checks data flow and gradient/PSRO integration; it is not convergence evidence.
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from open_score.stage1 import (
    CompetitiveEpisodeRunner,
    HADPSROTrainer,
    HADStage1Factory,
    LearningProgressCurriculum,
    RuleBasedController,
    all_evaluation_scales,
    psro_has_stabilised,
    supported_scales,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pilot2v2", "curriculum"), default="pilot2v2")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--br-episodes", type=int, default=16)
    parser.add_argument("--batch-episodes", type=int, default=4)
    parser.add_argument("--payoff-episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--episodes-per-stage", type=int, default=2000)
    parser.add_argument("--nash-conv-tolerance", type=float, default=0.10)
    parser.add_argument("--value-tolerance", type=float, default=0.03)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs" / "stage1_psro.json")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def jsonify(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonify(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    if args.iterations < 1 or args.br_episodes < 1 or args.payoff_episodes < 1:
        raise ValueError("iterations and episode counts must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = choose_device(args.device)
    fixed_scale = (2, 2) if args.mode == "pilot2v2" else None
    train_scales = [(2, 2)] if fixed_scale else supported_scales(4, True)
    evaluation_scales = [(2, 2)] if fixed_scale else all_evaluation_scales(4)
    curriculum = None if fixed_scale else LearningProgressCurriculum(
        max_agents=4,
        defender_not_outnumbered=True,
        episodes_per_stage=args.episodes_per_stage,
        uniform_coverage=0.20,
    )
    runner = CompetitiveEpisodeRunner(HADStage1Factory(max_steps=args.max_steps))
    trainer = HADPSROTrainer(
        runner,
        red_population=[RuleBasedController("guard"), RuleBasedController("intercept")],
        blue_population=[RuleBasedController("rush"), RuleBasedController("split_rush")],
        train_scales=train_scales,
        evaluation_scales=evaluation_scales,
        device=device,
        seed=args.seed,
        curriculum=curriculum,
    )
    started = time.perf_counter()
    history = []
    records = []
    for _ in range(args.iterations):
        result = trainer.run_iteration(
            br_episodes=args.br_episodes,
            payoff_episodes_per_scale=args.payoff_episodes,
            batch_episodes=args.batch_episodes,
            fixed_scale=fixed_scale,
        )
        history.append(result.metrics)
        records.append(
            {
                "metrics": asdict(result.metrics),
                "payoff_before": result.payoff_before,
                "payoff_after": result.payoff_after,
                "defender_mixture_before": result.defender_mixture_before,
                "attacker_mixture_before": result.attacker_mixture_before,
                "defender_mixture_after": result.defender_mixture_after,
                "attacker_mixture_after": result.attacker_mixture_after,
                "red_training": {
                    "episodes": result.red_training.episodes,
                    "environment_steps": result.red_training.environment_steps,
                    "mean_defender_payoff": result.red_training.mean_training_payoff,
                    "last_loss": None if result.red_training.latest_metrics is None else result.red_training.latest_metrics.loss,
                },
                "blue_training": {
                    "episodes": result.blue_training.episodes,
                    "environment_steps": result.blue_training.environment_steps,
                    "mean_defender_payoff": result.blue_training.mean_training_payoff,
                    "last_loss": None if result.blue_training.latest_metrics is None else result.blue_training.latest_metrics.loss,
                },
            }
        )
        if psro_has_stabilised(
            history,
            tolerance=args.nash_conv_tolerance,
            patience=args.patience,
            value_tolerance=args.value_tolerance,
        ):
            break
    payload = jsonify(
        {
            "status": "pipeline_pilot" if args.mode == "pilot2v2" else "training_run",
            "convergence_claim": False,
            "mode": args.mode,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "seed": args.seed,
            "train_scales": train_scales,
            "evaluation_scales": evaluation_scales,
            "elapsed_seconds": time.perf_counter() - started,
            "curriculum": None if curriculum is None else {
                "Red": trainer.red_curriculum.state_dict(),
                "Blue": trainer.blue_curriculum.state_dict(),
            },
            "iterations": records,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
