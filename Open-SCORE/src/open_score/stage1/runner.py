"""Two-sided HAD rollout runner and acceleration-only reference policies."""

import copy
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from open_score.contracts import TeamObservation
from open_score.envs import HADStage1Adapter, tensorize_had_observation
from open_score.stage1.baselines import VariableScaleMAPPO
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
        self.last_action: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self.hidden = None
        self.last_action = None

    def act(self, adapter, side, observation, rng) -> np.ndarray:
        del adapter, side, rng
        team, _ = tensorize_had_observation(observation, self.device)
        self.model.eval()
        actions, self.hidden = self.model.act(
            team,
            self.hidden,
            epsilon=self.epsilon,
            last_action=self.last_action,
        )
        self.last_action = F.one_hot(
            actions, num_classes=self.model.agent.action_dim
        ).to(team.self_obs.dtype)
        return actions.squeeze(0).detach().cpu().numpy().astype(np.int64)


class MAPPOController:
    """Decentralized categorical actor controller for a MAPPO checkpoint."""

    def __init__(
        self,
        model: VariableScaleMAPPO,
        device: torch.device,
        deterministic: bool = False,
        name: str = "mappo",
    ):
        self.model = model
        self.device = device
        self.deterministic = deterministic
        self.name = name
        self.hidden: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self.hidden = None

    def act(self, adapter, side, observation, rng) -> np.ndarray:
        del adapter, side
        team, _ = tensorize_had_observation(observation, self.device)
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


def frozen_qmix_controller(
    model: VariableScaleQMIX,
    device: torch.device,
    name: str,
) -> QMixController:
    frozen = copy.deepcopy(model).to(device).eval()
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)
    return QMixController(frozen, device, epsilon=0.0, name=name)


def frozen_mappo_controller(
    model: VariableScaleMAPPO,
    device: torch.device,
    name: str,
) -> MAPPOController:
    frozen = copy.deepcopy(model).to(device).eval()
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)
    return MAPPOController(frozen, device, deterministic=True, name=name)


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
        clearance_weight: float = 0.50,
        intercept_weight: float = 0.25,
        allow_unregistered_roster: bool = False,
    ):
        self.max_steps = max_steps
        self.target_region = target_region
        self.gamma = gamma
        self.shaping_scale = shaping_scale
        self.clearance_weight = clearance_weight
        self.intercept_weight = intercept_weight
        self.allow_unregistered_roster = bool(allow_unregistered_roster)
        self.cache: Dict[Scale, HADStage1Adapter] = {}

    def create(self, scale: Scale) -> HADStage1Adapter:
        return HADStage1Adapter(
            scale[0],
            scale[1],
            max_steps=self.max_steps,
            target_region=self.target_region,
            gamma=self.gamma,
            shaping_scale=self.shaping_scale,
            clearance_weight=self.clearance_weight,
            intercept_weight=self.intercept_weight,
            allow_unregistered_roster=self.allow_unregistered_roster,
        )

    def get(self, scale: Scale) -> HADStage1Adapter:
        if scale not in self.cache:
            self.cache[scale] = self.create(scale)
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


def batch_tensorize_had_observations(
    observations: Sequence[Dict[str, np.ndarray]],
    device: torch.device,
) -> TeamObservation:
    """Pad a heterogeneous list of current HAD team observations."""

    if not observations:
        raise ValueError("cannot batch an empty observation list")
    batch = len(observations)
    max_agents = max(value["entity_obs"].shape[0] for value in observations)
    max_entities = max(value["entity_obs"].shape[1] for value in observations)
    entity_dim = observations[0]["entity_obs"].shape[-1]
    self_dim = observations[0]["self_obs"].shape[-1]
    task_dim = observations[0]["task_obs"].shape[-1]
    action_dim = observations[0]["avail_actions"].shape[-1]
    entity_obs = np.zeros(
        (batch, max_agents, max_entities, entity_dim), dtype=np.float32
    )
    entity_mask = np.zeros(
        (batch, max_agents, max_entities), dtype=bool
    )
    self_obs = np.zeros((batch, max_agents, self_dim), dtype=np.float32)
    task_obs = np.zeros((batch, max_agents, task_dim), dtype=np.float32)
    agent_mask = np.zeros((batch, max_agents), dtype=bool)
    avail_actions = np.zeros((batch, max_agents, action_dim), dtype=bool)
    for index, value in enumerate(observations):
        agents, entities = value["entity_obs"].shape[:2]
        entity_obs[index, :agents, :entities] = value["entity_obs"]
        entity_mask[index, :agents, :entities] = value["entity_mask"]
        self_obs[index, :agents] = value["self_obs"]
        task_obs[index, :agents] = value["task_obs"]
        agent_mask[index, :agents] = value["agent_mask"]
        avail_actions[index, :agents] = value["avail_actions"]
    tensor = lambda value, dtype: torch.as_tensor(
        value, dtype=dtype, device=device
    )
    return TeamObservation(
        tensor(entity_obs, torch.float32),
        tensor(entity_mask, torch.bool),
        tensor(self_obs, torch.float32),
        tensor(task_obs, torch.float32),
        tensor(agent_mask, torch.bool),
        tensor(avail_actions, torch.bool),
    )


class BatchedHADRedRunner:
    """Collect several Red-policy episodes with batched neural inference.

    HAD worlds still advance independently, while all currently active Red
    teams share one padded GPU forward pass.  This preserves exact environment
    dynamics and supplies the eight-environment collection contract used by
    the round-01 REFIL experiment.
    """

    def __init__(self, factory: HADStage1Factory):
        self.factory = factory

    def run_batch(
        self,
        scales: Sequence[Scale],
        red_controller: QMixController,
        blue_controllers: Sequence[RuleBasedController],
        seeds: Sequence[int],
    ) -> List[CompetitiveEpisode]:
        if not scales or not (
            len(scales) == len(blue_controllers) == len(seeds)
        ):
            raise ValueError("scales, opponents and seeds must have equal length")
        adapters = [self.factory.create(scale) for scale in scales]
        observations = [
            adapter.reset(seed=int(seed))
            for adapter, seed in zip(adapters, seeds)
        ]
        rngs = [
            np.random.default_rng(int(seed) + 1_000_003) for seed in seeds
        ]
        for controller in blue_controllers:
            controller.reset()
        red_history = [[value["Red"]] for value in observations]
        blue_history = [[value["Blue"]] for value in observations]
        red_actions: List[List[np.ndarray]] = [[] for _ in scales]
        blue_actions: List[List[np.ndarray]] = [[] for _ in scales]
        red_rewards: List[List[float]] = [[] for _ in scales]
        blue_rewards: List[List[float]] = [[] for _ in scales]
        done_flags: List[List[float]] = [[] for _ in scales]
        final_info: List[Optional[Dict[str, object]]] = [None for _ in scales]
        hidden_states: List[Optional[torch.Tensor]] = [None for _ in scales]
        previous_actions: List[Optional[np.ndarray]] = [None for _ in scales]
        active = list(range(len(scales)))
        red_controller.model.eval()
        while active:
            team = batch_tensorize_had_observations(
                [observations[index]["Red"] for index in active],
                red_controller.device,
            )
            max_agents = team.agent_mask.shape[1]
            hidden = torch.zeros(
                len(active),
                max_agents,
                red_controller.model.agent.hidden_dim,
                device=red_controller.device,
            )
            last_action = torch.zeros(
                len(active),
                max_agents,
                red_controller.model.agent.action_dim,
                device=red_controller.device,
            )
            for local, index in enumerate(active):
                agents = observations[index]["Red"]["agent_mask"].shape[0]
                if hidden_states[index] is not None:
                    hidden[local, :agents] = hidden_states[index]
                if previous_actions[index] is not None:
                    action_tensor = torch.as_tensor(
                        previous_actions[index],
                        dtype=torch.long,
                        device=red_controller.device,
                    )
                    last_action[local, :agents] = F.one_hot(
                        action_tensor,
                        num_classes=red_controller.model.agent.action_dim,
                    ).to(last_action.dtype)
            with torch.no_grad():
                action_tensor, next_hidden = red_controller.model.act(
                    team,
                    hidden,
                    epsilon=red_controller.epsilon,
                    last_action=last_action,
                )
            action_array = action_tensor.detach().cpu().numpy()
            next_active = []
            for local, index in enumerate(active):
                red_count = scales[index][0]
                action_red = action_array[local, :red_count].astype(np.int64)
                hidden_states[index] = next_hidden[
                    local, :red_count
                ].detach()
                previous_actions[index] = action_red.copy()
                action_blue = blue_controllers[index].act(
                    adapters[index],
                    "Blue",
                    observations[index]["Blue"],
                    rngs[index],
                )
                next_observation, rewards, done, info = adapters[index].step(
                    action_red, action_blue
                )
                red_actions[index].append(action_red)
                blue_actions[index].append(action_blue)
                red_rewards[index].append(float(rewards["Red"]))
                blue_rewards[index].append(float(rewards["Blue"]))
                done_flags[index].append(float(done))
                red_history[index].append(next_observation["Red"])
                blue_history[index].append(next_observation["Blue"])
                observations[index] = next_observation
                if done:
                    final_info[index] = info
                else:
                    next_active.append(index)
            active = next_active

        episodes = []
        for index, scale in enumerate(scales):
            info = final_info[index]
            if info is None:  # pragma: no cover - every HAD rollout terminates
                raise AssertionError("batched rollout ended without terminal info")
            red_episode = TeamEpisode(
                tuple(red_history[index]),
                np.stack(red_actions[index]),
                np.asarray(red_rewards[index], dtype=np.float32),
                np.asarray(done_flags[index], dtype=np.float32),
                scale,
                "Red",
                int(seeds[index]),
            )
            blue_episode = TeamEpisode(
                tuple(blue_history[index]),
                np.stack(blue_actions[index]),
                np.asarray(blue_rewards[index], dtype=np.float32),
                np.asarray(done_flags[index], dtype=np.float32),
                scale,
                "Blue",
                int(seeds[index]),
            )
            episodes.append(
                CompetitiveEpisode(
                    red_episode,
                    blue_episode,
                    float(info["outcome_red"]),
                    tuple(float(value) for value in info["target_position"]),
                    red_controller.name,
                    blue_controllers[index].name,
                )
            )
        return episodes
