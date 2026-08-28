"""Two-sided HAD rollout runner and acceleration-only reference policies."""

import copy
from typing import Dict, Optional, Protocol, Sequence, Tuple

import numpy as np
import torch

from open_score.envs import HADStage1Adapter, tensorize_had_observation
from open_score.stage1.curriculum import Scale
from open_score.stage1.entity_qmix import VariableScaleQMIX
from open_score.stage1.replay import CompetitiveEpisode, TeamEpisode


class TeamController(Protocol):
    name: str

    def reset(self) -> None: ...

    def act(
        self,
        adapter: HADStage1Adapter,
        side: str,
        observation: Dict[str, np.ndarray],
        rng: np.random.Generator,
    ) -> np.ndarray: ...


class RandomController:
    def __init__(self, name: str = "random"):
        self.name = name

    def reset(self) -> None:
        return None

    def act(self, adapter, side, observation, rng) -> np.ndarray:
        result = []
        for available in observation["avail_actions"]:
            choices = np.flatnonzero(available)
            result.append(int(rng.choice(choices)))
        return np.asarray(result, dtype=np.int64)


def _nearest_acceleration(adapter: HADStage1Adapter, direction: np.ndarray) -> int:
    norm = float(np.linalg.norm(direction))
    if norm < 1e-8:
        return 0
    unit = direction / norm
    return int(np.argmax(adapter.action_vectors @ unit))


class RuleBasedController:
    """Map simple tactics to acceleration IDs without adding macro actions.

    These policies exist only as opponents and audit baselines.  The learned
    policy still sees exactly the same 27 acceleration controls.
    """

    VALID_STYLES = {"intercept", "guard", "rush", "split_rush", "engage"}

    def __init__(self, style: str):
        if style not in self.VALID_STYLES:
            raise ValueError(f"unknown rule style: {style}")
        self.style = style
        self.name = f"rule:{style}"

    def reset(self) -> None:
        return None

    def act(self, adapter, side, observation, rng) -> np.ndarray:
        agents = adapter.env.red_agents if side == "Red" else adapter.env.blue_agents
        opponents = adapter.env.blue_agents if side == "Red" else adapter.env.red_agents
        alive_opponents = [agent for agent in opponents if agent.Health > 0]
        target = adapter.env.targets[0]
        result = []
        for index, agent in enumerate(agents):
            if agent.Health <= 0:
                result.append(0)
                continue
            position = np.asarray(agent.position, dtype=np.float32)
            if self.style in {"intercept", "engage"} and alive_opponents:
                destination = min(
                    alive_opponents,
                    key=lambda other: np.linalg.norm(np.asarray(other.position) - position),
                ).position
            elif self.style == "guard" and alive_opponents:
                threat = min(
                    alive_opponents,
                    key=lambda other: np.linalg.norm(
                        np.asarray(other.position) - np.asarray(target.position)
                    ),
                )
                destination = np.asarray(target.position) + 0.35 * (
                    np.asarray(threat.position) - np.asarray(target.position)
                )
            elif self.style == "split_rush":
                lateral = (index - (len(agents) - 1) / 2.0) * 180.0
                destination = np.asarray(target.position) + np.asarray([0.0, lateral, 0.0])
            else:
                destination = target.position
            result.append(_nearest_acceleration(adapter, np.asarray(destination) - position))
        return np.asarray(result, dtype=np.int64)


class QMixController:
    def __init__(
        self,
        model: VariableScaleQMIX,
        device: torch.device,
        epsilon: float = 0.0,
        name: str = "qmix",
    ):
        self.model = model
        self.device = device
        self.epsilon = float(epsilon)
        self.name = name
        self.hidden: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self.hidden = None

    def act(self, adapter, side, observation, rng) -> np.ndarray:
        del adapter, side, rng
        team, _ = tensorize_had_observation(observation, self.device)
        self.model.eval()
        actions, self.hidden = self.model.act(team, self.hidden, epsilon=self.epsilon)
        return actions.squeeze(0).detach().cpu().numpy().astype(np.int64)


def frozen_qmix_controller(
    model: VariableScaleQMIX,
    device: torch.device,
    name: str,
) -> QMixController:
    frozen = copy.deepcopy(model).to(device).eval()
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)
    return QMixController(frozen, device, epsilon=0.0, name=name)


class HADStage1Factory:
    """Cache one fixed-roster world per scale and reset it for each episode."""

    def __init__(
        self,
        max_steps: int = 300,
        target_region: Sequence[Sequence[float]] = (
            (-2300.0, -1900.0),
            (-1200.0, 1200.0),
            (50.0, 300.0),
        ),
        gamma: float = 0.99,
        shaping_scale: float = 0.10,
    ):
        self.max_steps = max_steps
        self.target_region = target_region
        self.gamma = gamma
        self.shaping_scale = shaping_scale
        self.cache: Dict[Scale, HADStage1Adapter] = {}

    def get(self, scale: Scale) -> HADStage1Adapter:
        if scale not in self.cache:
            self.cache[scale] = HADStage1Adapter(
                scale[0],
                scale[1],
                max_steps=self.max_steps,
                target_region=self.target_region,
                gamma=self.gamma,
                shaping_scale=self.shaping_scale,
            )
        return self.cache[scale]


class CompetitiveEpisodeRunner:
    def __init__(self, factory: HADStage1Factory):
        self.factory = factory

    def run(
        self,
        scale: Scale,
        red_controller: TeamController,
        blue_controller: TeamController,
        seed: int,
    ) -> CompetitiveEpisode:
        adapter = self.factory.get(scale)
        observations = adapter.reset(seed=seed)
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
        done = False
        info: Dict[str, object] = {}
        while not done:
            action_red = red_controller.act(adapter, "Red", observations["Red"], rng)
            action_blue = blue_controller.act(adapter, "Blue", observations["Blue"], rng)
            observations, rewards, done, info = adapter.step(action_red, action_blue)
            red_actions.append(action_red)
            blue_actions.append(action_blue)
            red_rewards.append(rewards["Red"])
            blue_rewards.append(rewards["Blue"])
            done_flags.append(float(done))
            red_observations.append(observations["Red"])
            blue_observations.append(observations["Blue"])
        red_episode = TeamEpisode(
            tuple(red_observations),
            np.stack(red_actions),
            np.asarray(red_rewards, dtype=np.float32),
            np.asarray(done_flags, dtype=np.float32),
            scale,
            "Red",
            seed,
        )
        blue_episode = TeamEpisode(
            tuple(blue_observations),
            np.stack(blue_actions),
            np.asarray(blue_rewards, dtype=np.float32),
            np.asarray(done_flags, dtype=np.float32),
            scale,
            "Blue",
            seed,
        )
        return CompetitiveEpisode(
            red_episode,
            blue_episode,
            float(info["outcome_red"]),
            tuple(float(value) for value in info["target_position"]),
            red_controller.name,
            blue_controller.name,
        )
