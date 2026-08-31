"""Whole-episode runner for stock SMAClite under the shared Stage-1 models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Protocol, Tuple

import numpy as np

from open_score.envs.smaclite_ad import SMACliteStockAdapter
from open_score.stage1.replay import TeamEpisode


class StockController(Protocol):
    name: str

    def reset(self) -> None: ...

    def act(self, env, side, observation, rng) -> np.ndarray: ...


@dataclass(frozen=True)
class StockEpisode:
    team: TeamEpisode
    battle_won: bool
    final_info: Mapping[str, object]
    policy_name: str

    @property
    def length(self) -> int:
        return self.team.length


class SMACliteStockEpisodeRunner:
    def __init__(self, adapter: SMACliteStockAdapter):
        self.adapter = adapter

    @property
    def scale(self) -> Tuple[int, int]:
        return (
            int(self.adapter.unwrapped.n_agents),
            int(self.adapter.unwrapped.n_enemies),
        )

    def run(self, controller: StockController, seed: int) -> StockEpisode:
        observation, info = self.adapter.reset(seed=seed)
        controller.reset()
        rng = np.random.default_rng(seed + 1_000_003)
        observations = [observation]
        actions = []
        rewards = []
        done_flags = []
        terminated = truncated = False
        while not (terminated or truncated):
            action = controller.act(
                self.adapter, "Allies", observation, rng
            )
            observation, reward, terminated, truncated, info = self.adapter.step(
                action
            )
            actions.append(action)
            rewards.append(reward)
            done_flags.append(float(terminated or truncated))
            observations.append(observation)
        episode = TeamEpisode(
            tuple(observations),
            np.stack(actions),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(done_flags, dtype=np.float32),
            self.scale,
            "Allies",
            seed,
        )
        return StockEpisode(
            episode,
            bool(info.get("battle_won", False)),
            dict(info),
            controller.name,
        )


@dataclass(frozen=True)
class StockEvaluation:
    episodes: int
    mean_return: float
    return_std: float
    win_rate: float
    mean_episode_length: float
    paired_returns: Tuple[float, ...]
    paired_wins: Tuple[bool, ...]
    seeds: Tuple[int, ...]


def evaluate_stock(
    runner: SMACliteStockEpisodeRunner,
    controller: StockController,
    episodes: int,
    seed_base: int,
) -> StockEvaluation:
    if episodes < 1:
        raise ValueError("stock evaluation needs at least one episode")
    rollouts = [
        runner.run(controller, seed_base + index) for index in range(episodes)
    ]
    returns = np.asarray(
        [float(rollout.team.rewards.sum()) for rollout in rollouts],
        dtype=np.float64,
    )
    wins = np.asarray([rollout.battle_won for rollout in rollouts], dtype=bool)
    return StockEvaluation(
        episodes=episodes,
        mean_return=float(returns.mean()),
        return_std=float(returns.std()),
        win_rate=float(wins.mean()),
        mean_episode_length=float(np.mean([one.length for one in rollouts])),
        paired_returns=tuple(float(value) for value in returns),
        paired_wins=tuple(bool(value) for value in wins),
        seeds=tuple(seed_base + index for index in range(episodes)),
    )


__all__ = [
    "SMACliteStockEpisodeRunner",
    "StockEpisode",
    "StockEvaluation",
    "evaluate_stock",
]
