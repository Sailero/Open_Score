"""Continuously render the frozen round-01 REFIL policy in a pygame window."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from open_score.envs import HADStage1Adapter
from open_score.stage1 import QMixController, RuleBasedController, make_had_qmix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT
        / "outputs"
        / "round_01_mvp"
        / "stage1"
        / "checkpoints"
        / "refil_qmix_seed20260831_best.pt",
    )
    parser.add_argument("--scale", default="6v3")
    parser.add_argument(
        "--opponent", choices=("rush", "split_rush"), default="split_rush"
    )
    parser.add_argument("--seed", type=int, default=48_260_831)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="number of episodes to show; 0 keeps replaying until the window closes",
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    return parser.parse_args()


def parse_scale(label: str):
    try:
        red_text, blue_text = label.lower().split("v", maxsplit=1)
        scale = int(red_text), int(blue_text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid HAD scale: {label!r}") from exc
    if scale[0] < 1 or scale[1] < 1:
        raise ValueError("HAD visualization needs at least one agent per side")
    return scale


def choose_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def load_controller(checkpoint_path: Path, device: torch.device) -> QMixController:
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
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
        raise ValueError("checkpoint is not the frozen round-01 REFIL architecture")
    model.load_state_dict(payload["online"], strict=True)
    model.eval()
    return QMixController(model, device, epsilon=0.0, name="round01_best_refil")


def render_frame(adapter: HADStage1Adapter) -> bool:
    """Render once and report whether the user left the window open."""

    adapter.env.render()
    import pygame

    return bool(
        pygame.get_init()
        and pygame.display.get_init()
        and pygame.display.get_surface() is not None
    )


def main() -> None:
    args = parse_args()
    if args.fps <= 0.0 or args.episodes < 0:
        raise ValueError("fps must be positive and episodes must be non-negative")
    red_count, blue_count = parse_scale(args.scale)
    device = choose_device(args.device)
    red = load_controller(args.checkpoint, device)
    blue = RuleBasedController(args.opponent)
    frame_delay = 1.0 / args.fps

    episode_index = 0
    while args.episodes == 0 or episode_index < args.episodes:
        seed = args.seed + episode_index
        adapter = HADStage1Adapter(
            red_count,
            blue_count,
            max_steps=50,
            shaping_scale=0.5,
            allow_unregistered_roster=True,
        )
        observations = adapter.reset(seed=seed)
        red.reset()
        blue.reset()
        rng = np.random.default_rng(seed + 1_000_003)
        done = False
        info = {}
        if not render_frame(adapter):
            return
        while not done:
            with torch.inference_mode():
                red_actions = red.act(
                    adapter, "Red", observations["Red"], rng
                )
            blue_actions = blue.act(adapter, "Blue", observations["Blue"], rng)
            observations, _, done, info = adapter.step(red_actions, blue_actions)
            if not render_frame(adapter):
                return
            print(
                f"episode={episode_index + 1} step={adapter.step_count:02d} "
                f"red_alive={sum(agent.Health > 0 for agent in adapter.env.red_agents)} "
                f"blue_alive={sum(agent.Health > 0 for agent in adapter.env.blue_agents)} "
                f"target_alive={int(adapter.env.targets[0].Health > 0)}",
                flush=True,
            )
            time.sleep(frame_delay)
        outcome = "Red win" if info["outcome_red"] > 0 else "Blue win"
        print(f"episode={episode_index + 1} completed: {outcome}", flush=True)
        episode_index += 1

    import pygame

    pygame.quit()


if __name__ == "__main__":
    main()
