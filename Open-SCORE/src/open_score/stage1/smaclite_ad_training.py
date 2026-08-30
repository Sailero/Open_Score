"""Rollout and evaluation utilities for small-budget SMAClite-AD learning.

The module is intentionally separate from the HAD runner.  It consumes the
same :class:`TeamObservation`, :class:`GlobalState` and :class:`TeamEpisode`
contracts, but does not overload HAD's roles or command semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np
import torch

from open_score.envs.smaclite_ad import (
    SMACliteADEnv,
    tensorize_smaclite_ad_observation,
)
from open_score.stage1.baselines import VariableScaleMAPPO
from open_score.stage1.entity_qmix import VariableScaleQMIX
from open_score.stage1.replay import TeamEpisode


Ratio = Tuple[int, int]


class SMACliteADController(Protocol):
    name: str

    def reset(self) -> None: ...

    def act(
        self,
        env: SMACliteADEnv,
        side: str,
        observation: Dict[str, np.ndarray],
        rng: np.random.Generator,
    ) -> np.ndarray: ...


def _best_move_towards(
    env: SMACliteADEnv,
    unit: object,
    destination: np.ndarray,
    side: str,
) -> int:
    """Choose one upstream primitive movement action using a direction dot product."""

    from smaclite.env.util.direction import Direction

    available = env._available_actions_for(unit, side)
    desired = np.asarray(destination, dtype=np.float32) - unit.pos
    best_action = 1  # stop is always valid for a living unit
    best_score = -float("inf")
    for direction in Direction:
        action = 2 + direction.value
        if available[action]:
            score = float(Direction(direction.value).dx_dy @ desired)
            if score > best_score:
                best_score = score
                best_action = action
    return best_action


class SMACliteADRuleController:
    """Fixed full-state audit opponent using only SMAClite primitive actions."""

    VALID_STYLES = {"rush_asset", "intercept", "idle"}

    def __init__(self, style: str):
        if style not in self.VALID_STYLES:
            raise ValueError(f"unknown SMAClite-AD rule style: {style}")
        self.style = style
        self.name = f"rule:{style}"

    def reset(self) -> None:
        return None

    def act(self, env, side, observation, rng) -> np.ndarray:
        del observation, rng
        slots = env._red_slots if side == "Red" else env._blue_slots
        opponents = env._blue_slots if side == "Red" else env._red_slots
        alive_opponents = [unit for unit in opponents if unit.hp > 0]
        result = []
        for unit in slots:
            if unit.hp <= 0:
                result.append(0)
                continue
            if self.style == "idle":
                result.append(1)
                continue
            available = env._available_actions_for(unit, side)
            if self.style == "rush_asset" and side == "Red" and env._asset is not None:
                assert env.asset_action_id is not None
                asset_action = env.asset_action_id
                if available[asset_action]:
                    result.append(asset_action)
                else:
                    result.append(_best_move_towards(env, unit, env._asset.pos, side))
                continue
            if alive_opponents:
                nearest_index, nearest = min(
                    enumerate(opponents),
                    key=lambda pair: (
                        float("inf")
                        if pair[1].hp <= 0
                        else np.linalg.norm(pair[1].pos - unit.pos)
                    ),
                )
                attack_action = 6 + nearest_index
                if available[attack_action]:
                    result.append(attack_action)
                else:
                    result.append(_best_move_towards(env, unit, nearest.pos, side))
            else:
                result.append(1)
        return np.asarray(result, dtype=np.int64)


class SMACliteADQController:
    """Decentralized controller shared by the QMIX and VDN utility networks."""

    def __init__(
        self,
        model: VariableScaleQMIX,
        device: torch.device,
        *,
        epsilon: float = 0.0,
        name: str = "q_policy",
    ):
        self.model = model
        self.device = device
        self.epsilon = float(epsilon)
        self.name = name
        self.hidden: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self.hidden = None

    def act(self, env, side, observation, rng) -> np.ndarray:
        del env, side, rng
        team, _ = tensorize_smaclite_ad_observation(observation, self.device)
        self.model.eval()
        actions, self.hidden = self.model.act(
            team, self.hidden, epsilon=self.epsilon
        )
        return actions.squeeze(0).detach().cpu().numpy().astype(np.int64)


class SMACliteADMAPPOController:
    """Decentralized categorical actor for the shared MAPPO implementation."""

    def __init__(
        self,
        model: VariableScaleMAPPO,
        device: torch.device,
        *,
        deterministic: bool,
        name: str = "mappo_policy",
    ):
        self.model = model
        self.device = device
        self.deterministic = deterministic
        self.name = name
        self.hidden: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self.hidden = None

    def act(self, env, side, observation, rng) -> np.ndarray:
        del env, side
        team, _ = tensorize_smaclite_ad_observation(observation, self.device)
        self.model.eval()
        with torch.no_grad():
            logits, self.hidden = self.model.actor_logits(team, self.hidden)
            if self.deterministic:
                actions = logits.argmax(dim=-1).squeeze(0).cpu().numpy()
            else:
                probabilities = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                actions = np.asarray(
                    [rng.choice(len(row), p=row) for row in probabilities],
                    dtype=np.int64,
                )
        return actions.astype(np.int64)


class SMACliteADFactory:
    """Cache one simulator per realised ratio while sharing tensor maxima."""

    def __init__(
        self,
        *,
        max_red_agents: int = 6,
        max_blue_agents: int = 5,
        episode_limit: int = 50,
        shaping_scale: float = 0.10,
        approach_weight: float = 0.0,
        spawn_jitter: float = 0.0,
    ):
        self.max_red_agents = max_red_agents
        self.max_blue_agents = max_blue_agents
        self.episode_limit = episode_limit
        self.shaping_scale = shaping_scale
        self.approach_weight = approach_weight
        self.spawn_jitter = spawn_jitter
        self.cache: Dict[Ratio, SMACliteADEnv] = {}

    def get(self, ratio: Ratio) -> SMACliteADEnv:
        red, blue = ratio
        if ratio not in self.cache:
            self.cache[ratio] = SMACliteADEnv(
                red,
                blue,
                max_red_agents=self.max_red_agents,
                max_blue_agents=self.max_blue_agents,
                episode_limit=self.episode_limit,
                shaping_scale=self.shaping_scale,
                approach_weight=self.approach_weight,
                spawn_jitter=self.spawn_jitter,
            )
        return self.cache[ratio]

    def close(self) -> None:
        for env in self.cache.values():
            env.close()
        self.cache.clear()


@dataclass(frozen=True)
class SMACliteADEpisode:
    red: TeamEpisode
    blue: TeamEpisode
    outcome_red: float
    final_info: Mapping[str, object]
    red_policy_name: str
    blue_policy_name: str

    @property
    def length(self) -> int:
        return self.red.length


class SMACliteADEpisodeRunner:
    def __init__(self, factory: SMACliteADFactory):
        self.factory = factory

    def run(
        self,
        ratio: Ratio,
        red_controller: SMACliteADController,
        blue_controller: SMACliteADController,
        seed: int,
    ) -> SMACliteADEpisode:
        env = self.factory.get(ratio)
        observations, _ = env.reset(seed=seed)
        red_controller.reset()
        blue_controller.reset()
        rng = np.random.default_rng(seed + 1_000_003)
        red_observations = [observations["Red"]]
        blue_observations = [observations["Blue"]]
        red_actions = []
        blue_actions = []
        red_rewards = []
        blue_rewards = []
        done_flags = []
        terminated = truncated = False
        info: Mapping[str, object] = {}
        while not (terminated or truncated):
            action_red = red_controller.act(
                env, "Red", observations["Red"], rng
            )
            action_blue = blue_controller.act(
                env, "Blue", observations["Blue"], rng
            )
            observations, rewards, terminated, truncated, info = env.step(
                {"Red": action_red, "Blue": action_blue}
            )
            red_actions.append(action_red)
            blue_actions.append(action_blue)
            red_rewards.append(rewards["Red"])
            blue_rewards.append(rewards["Blue"])
            done_flags.append(float(terminated or truncated))
            red_observations.append(observations["Red"])
            blue_observations.append(observations["Blue"])
        red_episode = TeamEpisode(
            tuple(red_observations),
            np.stack(red_actions),
            np.asarray(red_rewards, dtype=np.float32),
            np.asarray(done_flags, dtype=np.float32),
            ratio,
            "Red",
            seed,
        )
        blue_episode = TeamEpisode(
            tuple(blue_observations),
            np.stack(blue_actions),
            np.asarray(blue_rewards, dtype=np.float32),
            np.asarray(done_flags, dtype=np.float32),
            ratio,
            "Blue",
            seed,
        )
        return SMACliteADEpisode(
            red=red_episode,
            blue=blue_episode,
            outcome_red=float(info["outcome_red"]),
            final_info=dict(info),
            red_policy_name=red_controller.name,
            blue_policy_name=blue_controller.name,
        )


@dataclass(frozen=True)
class SMACliteADEvaluation:
    controlled_side: str
    episodes: int
    mean_return: float
    win_rate: float
    mean_payoff: float
    mean_episode_length: float
    mean_asset_health: float
    per_ratio: Mapping[str, Mapping[str, object]]
    paired_returns: Tuple[float, ...]
    layout_hashes: Tuple[str, ...]
    unique_layouts: int
    layout_records: Tuple[Mapping[str, object], ...]


def evaluate_smaclite_ad(
    runner: SMACliteADEpisodeRunner,
    controlled: SMACliteADController,
    opponent: SMACliteADController,
    controlled_side: str,
    ratios: Sequence[Ratio],
    episodes_per_ratio: int,
    seed: int,
) -> SMACliteADEvaluation:
    """Evaluate on a fixed ordered seed set suitable for before/after pairing."""

    if controlled_side not in {"Red", "Blue"}:
        raise ValueError("controlled_side must be Red or Blue")
    if episodes_per_ratio < 1 or not ratios:
        raise ValueError("evaluation needs ratios and episodes")
    all_returns = []
    all_wins = []
    all_payoffs = []
    all_lengths = []
    all_asset_health = []
    all_layout_hashes = []
    all_layout_records = []
    grouped: Dict[str, Dict[str, list]] = {}
    episode_index = 0
    for ratio in ratios:
        label = f"{ratio[0]}:{ratio[1]}"
        grouped[label] = {
            "returns": [],
            "wins": [],
            "payoffs": [],
            "lengths": [],
            "asset_health": [],
            "layout_hashes": [],
        }
        for _ in range(episodes_per_ratio):
            rollout_seed = seed + episode_index * 101
            if controlled_side == "Red":
                rollout = runner.run(ratio, controlled, opponent, rollout_seed)
                team = rollout.red
                payoff = rollout.outcome_red
            else:
                rollout = runner.run(ratio, opponent, controlled, rollout_seed)
                team = rollout.blue
                payoff = -rollout.outcome_red
            episode_return = float(team.rewards.sum())
            win = float(payoff > 0.0)
            asset_health = float(rollout.final_info["asset_health_fraction"])
            layout_hash = str(rollout.final_info["layout_hash"])
            layout_record = dict(rollout.final_info["layout"])
            for collection, value in (
                (all_returns, episode_return),
                (all_wins, win),
                (all_payoffs, payoff),
                (all_lengths, rollout.length),
                (all_asset_health, asset_health),
                (grouped[label]["returns"], episode_return),
                (grouped[label]["wins"], win),
                (grouped[label]["payoffs"], payoff),
                (grouped[label]["lengths"], rollout.length),
                (grouped[label]["asset_health"], asset_health),
            ):
                collection.append(float(value))
            all_layout_hashes.append(layout_hash)
            all_layout_records.append(layout_record)
            grouped[label]["layout_hashes"].append(layout_hash)
            episode_index += 1
    per_ratio = {
        label: {
            "mean_return": float(np.mean(values["returns"])),
            "win_rate": float(np.mean(values["wins"])),
            "mean_payoff": float(np.mean(values["payoffs"])),
            "mean_episode_length": float(np.mean(values["lengths"])),
            "mean_asset_health": float(np.mean(values["asset_health"])),
            "layout_hashes": tuple(values["layout_hashes"]),
            "unique_layouts": len(set(values["layout_hashes"])),
        }
        for label, values in grouped.items()
    }
    return SMACliteADEvaluation(
        controlled_side=controlled_side,
        episodes=len(all_returns),
        mean_return=float(np.mean(all_returns)),
        win_rate=float(np.mean(all_wins)),
        mean_payoff=float(np.mean(all_payoffs)),
        mean_episode_length=float(np.mean(all_lengths)),
        mean_asset_health=float(np.mean(all_asset_health)),
        per_ratio=per_ratio,
        paired_returns=tuple(all_returns),
        layout_hashes=tuple(all_layout_hashes),
        unique_layouts=len(set(all_layout_hashes)),
        layout_records=tuple(all_layout_records),
    )


def tensor_shape_audit(
    factory: SMACliteADFactory, ratios: Sequence[Ratio], seed: int
) -> Dict[str, object]:
    signatures: Dict[str, Dict[str, Tuple[int, ...]]] = {}
    reference = None
    for index, ratio in enumerate(ratios):
        observations, _ = factory.get(ratio).reset(seed=seed + index)
        signature = {
            side: (
                tuple(observations[side]["entity_obs"].shape),
                tuple(observations[side]["avail_actions"].shape),
                tuple(observations[side]["state_entities"].shape),
            )
            for side in ("Red", "Blue")
        }
        signatures[f"{ratio[0]}:{ratio[1]}"] = {
            side: tuple(dimension for shape in shapes for dimension in shape)
            for side, shapes in signature.items()
        }
        if reference is None:
            reference = signature
        elif signature != reference:
            raise AssertionError("SMAClite-AD tensor shapes changed across ratios")
    return {
        "shared_across_ratios": True,
        "ratios": signatures,
    }


__all__ = [
    "Ratio",
    "SMACliteADController",
    "SMACliteADEpisode",
    "SMACliteADEpisodeRunner",
    "SMACliteADEvaluation",
    "SMACliteADFactory",
    "SMACliteADMAPPOController",
    "SMACliteADQController",
    "SMACliteADRuleController",
    "evaluate_smaclite_ad",
    "tensor_shape_audit",
]
