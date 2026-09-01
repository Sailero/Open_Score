"""Two-sided HAD adapter for the one-target, small-scale Stage-1 game."""

import copy
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from open_score.contracts import GlobalState, TeamObservation


def _make_acceleration_primitives() -> np.ndarray:
    """Return zero plus the 26 normalised directions in {-1, 0, 1}^3.

    These are quantised *acceleration controls*. They contain no pursuit,
    target-selection, or other hand-written tactical semantics. HAD multiplies
    them by each agent's acceleration limit before advancing the dynamics.
    """

    vectors = [np.zeros(3, dtype=np.float32)]
    for components in product((-1.0, 0.0, 1.0), repeat=3):
        value = np.asarray(components, dtype=np.float32)
        norm = float(np.linalg.norm(value))
        if norm > 0.0:
            vectors.append(value / norm)
    return np.stack(vectors)


ACCELERATION_PRIMITIVES = _make_acceleration_primitives()


@dataclass(frozen=True)
class HADSnapshot:
    """Complete in-memory branch point for counterfactual continuation rollouts.

    The public observation is intentionally insufficient for restoring HAD: it
    omits entity IDs, controller-side engine attributes and random-generator
    state.  Stage 2 therefore branches from this explicit engine snapshot while
    serialising only the canonical command-time state in its dataset.
    """

    step_count: int
    entity_states: Tuple[Mapping[str, object], ...]
    environment_rng_state: Mapping[str, object]
    adapter_rng_state: Mapping[str, object]
    numpy_random_state: Tuple[object, ...]


class HADStage1Adapter:
    """Expose HAD as a two-team zero-sum game with one random target.

    Each instantiated HAD world has a fixed roster, while the same defender and
    attacker networks are reused across all registered strict Red-superior
    worlds. Both teams submit only quantised acceleration vectors; HAD retains
    its rule-based firing.
    """

    ENTITY_DIM = 12
    SELF_DIM = 10
    # Episode phase is part of the Markov state.  The actor receives it in the
    # task vector and the centralized mixer/critic receives it on every real
    # state entity.  Repeating one global scalar is intentional: masked mean /
    # max aggregation preserves it for every roster size.
    TASK_DIM = 7
    STATE_ENTITY_DIM = 12
    ACTION_DIM = len(ACCELERATION_PRIMITIVES)

    def __init__(
        self,
        red_attackers: int,
        blue_attackers: int,
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
        task_type: str = "Training",
        allow_unregistered_roster: bool = False,
    ):
        if red_attackers < 1 or blue_attackers < 1:
            raise ValueError("HAD Stage-1 requires at least one agent on each side")
        if not allow_unregistered_roster:
            if red_attackers > 4 or blue_attackers > 4:
                raise ValueError("Stage-1 supports 1--4 agents on each side")
            if red_attackers <= blue_attackers:
                raise ValueError(
                    "HAD Stage-1 requires strict Red numerical superiority "
                    "(red_attackers > blue_attackers) because firing agents self-destruct"
                )
        if len(target_region) != 3 or any(len(bounds) != 2 for bounds in target_region):
            raise ValueError("target_region must provide low/high bounds for x, y, z")
        if shaping_scale < 0.0 or clearance_weight < 0.0 or intercept_weight < 0.0:
            raise ValueError("reward-shaping weights must be non-negative")
        from HAD_Env.config import AeroPoint
        from HAD_Env.make_env import HADEnv

        for bounds, world_bounds in zip(target_region, AeroPoint):
            if bounds[0] > bounds[1] or bounds[0] < world_bounds[0] or bounds[1] > world_bounds[1]:
                raise ValueError("target_region must stay inside the HAD world")
        self.env = HADEnv(
            red_attackers,
            blue_attackers,
            1,
            task_type=task_type,
            target_region=target_region,
        )
        self.max_steps = max_steps
        self.target_region = np.asarray(target_region, dtype=np.float32)
        self.gamma = gamma
        self.shaping_scale = shaping_scale
        self.clearance_weight = clearance_weight
        self.intercept_weight = intercept_weight
        self.allow_unregistered_roster = bool(allow_unregistered_roster)
        self.step_count = 0
        self.rng = np.random.default_rng()

    @property
    def action_vectors(self) -> np.ndarray:
        return ACCELERATION_PRIMITIVES.copy()

    def _agents(self, side: str) -> List[object]:
        if side == "Red":
            return self.env.red_agents
        if side == "Blue":
            return self.env.blue_agents
        raise ValueError("side must be Red or Blue")

    @staticmethod
    def _opponent(side: str) -> str:
        return "Blue" if side == "Red" else "Red"

    def reset(self, seed: Optional[int] = None, evaluate: bool = False) -> Dict[str, Dict[str, np.ndarray]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # Separated spawn regions (Red left, Blue right) define the actual
        # asset-defence task. `evaluate` remains accepted for API compatibility.
        self.env.reset(evaluate=True, seed=seed)
        self.step_count = 0
        return {side: self.observe(side) for side in ("Red", "Blue")}

    def snapshot(self) -> HADSnapshot:
        """Capture all mutable physical and RNG state needed for an exact fork."""

        return HADSnapshot(
            step_count=int(self.step_count),
            entity_states=tuple(copy.deepcopy(entity.__dict__) for entity in self.env.world),
            environment_rng_state=copy.deepcopy(self.env.np_random.bit_generator.state),
            adapter_rng_state=copy.deepcopy(self.rng.bit_generator.state),
            numpy_random_state=copy.deepcopy(np.random.get_state()),
        )

    def restore(
        self,
        snapshot: HADSnapshot,
        *,
        continuation_seed: Optional[int] = None,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Restore a branch point and optionally seed all continuation RNGs.

        A continuation seed deliberately replaces the captured RNG state.  The
        same seed can then be used for every Red candidate in a root/threat
        block, which makes environment and controller noise common random
        numbers rather than merely giving the rows matching metadata.
        """

        if len(snapshot.entity_states) != len(self.env.world):
            raise ValueError("HAD snapshot roster differs from adapter roster")
        for entity, state in zip(self.env.world, snapshot.entity_states):
            entity.__dict__.clear()
            entity.__dict__.update(copy.deepcopy(dict(state)))
        self.step_count = int(snapshot.step_count)
        self.env.update_alive_agents()
        if continuation_seed is None:
            self.env.np_random.bit_generator.state = copy.deepcopy(
                snapshot.environment_rng_state
            )
            self.rng.bit_generator.state = copy.deepcopy(snapshot.adapter_rng_state)
            np.random.set_state(copy.deepcopy(snapshot.numpy_random_state))
        else:
            seed = int(continuation_seed)
            self.env.np_random = np.random.default_rng(seed)
            self.rng = np.random.default_rng(seed ^ 0x5A17D0A1)
            np.random.seed(seed % (2**32))
        return {side: self.observe(side) for side in ("Red", "Blue")}

    @staticmethod
    def _decode_acceleration(action_id: int) -> np.ndarray:
        if not 0 <= action_id < len(ACCELERATION_PRIMITIVES):
            raise ValueError("action id is outside the 27 acceleration primitives")
        return ACCELERATION_PRIMITIVES[action_id]

    def _potential(self) -> float:
        """Scale-normalised defender potential used only for dense shaping."""

        from HAD_Env.config import AeroPoint, initial_health

        target = self.env.targets[0]
        target_fraction = float(target.Health) / float(initial_health)
        red_fraction = sum(float(a.Health) for a in self.env.red_agents) / len(self.env.red_agents)
        blue_fraction = sum(float(a.Health) for a in self.env.blue_agents) / len(self.env.blue_agents)
        diagonal = float(np.linalg.norm([high - low for low, high in AeroPoint]))
        alive_red = [agent for agent in self.env.red_agents if agent.Health > 0]
        alive_blue = [agent for agent in self.env.blue_agents if agent.Health > 0]
        if alive_blue:
            clearance = min(
                np.linalg.norm(np.asarray(agent.position) - np.asarray(target.position))
                for agent in alive_blue
            ) / diagonal
        else:
            clearance = 1.0
        if alive_red and alive_blue:
            intercept_distance = min(
                np.linalg.norm(np.asarray(red.position) - np.asarray(blue.position))
                for red in alive_red
                for blue in alive_blue
            ) / diagonal
        else:
            intercept_distance = 1.0
        return (
            target_fraction
            + 0.25 * (red_fraction - blue_fraction)
            + self.clearance_weight * clearance
            - self.intercept_weight * intercept_distance
        )

    def step(
        self,
        red_action_ids: Sequence[int],
        blue_action_ids: Sequence[int],
    ) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, float], bool, Dict[str, object]]:
        """Advance both learned teams by one step and return zero-sum rewards."""

        if len(red_action_ids) != len(self.env.red_agents):
            raise ValueError("one action is required for every Red roster slot")
        if len(blue_action_ids) != len(self.env.blue_agents):
            raise ValueError("one action is required for every Blue roster slot")
        before = self._potential()
        actions_by_id = {}
        for agent, action in zip(self.env.red_agents, red_action_ids):
            actions_by_id[agent.Id] = 0 if agent.Health <= 0 else int(action)
        for agent, action in zip(self.env.blue_agents, blue_action_ids):
            actions_by_id[agent.Id] = 0 if agent.Health <= 0 else int(action)
        all_actions = [self._decode_acceleration(actions_by_id[agent.Id]) for agent in self.env.agents]
        _, _, _, _, _, info = self.env.step(all_actions)
        self.step_count += 1

        terminal_sign = int(self.env.is_terminal())  # +1 Red win, -1 target breach
        terminated = terminal_sign != 0
        truncated = self.step_count >= self.max_steps and not terminated
        done = terminated or truncated
        after = self._potential()
        # Setting Phi(terminal)=0 retains the registered terminal objective
        # while providing dense geometric TD targets during an episode.
        shaped_after = 0.0 if done else after
        shaping_reward = self.shaping_scale * (self.gamma * shaped_after - before)
        red_reward = shaping_reward
        if done:
            # Surviving until the registered horizon is a defender success.
            red_reward += float(terminal_sign if terminated else 1)
        rewards = {"Red": float(red_reward), "Blue": float(-red_reward)}
        info.update(
            {
                "terminated": terminated,
                "truncated": truncated,
                "outcome_red": float(terminal_sign if terminated else (1 if truncated else 0)),
                "target_position": list(self.env.targets[0].position),
                "defender_potential_before": float(before),
                "defender_potential_after": float(shaped_after),
                "potential_shaping_reward": float(shaping_reward),
            }
        )
        observations = {side: self.observe(side) for side in ("Red", "Blue")}
        return observations, rewards, done, info

    def _entity_features(self, observer: object, entity: object) -> np.ndarray:
        from HAD_Env.config import AeroPoint, initial_health, vDomain

        span = np.asarray([high - low for low, high in AeroPoint], dtype=np.float32)
        rel_position = (np.asarray(entity.position) - np.asarray(observer.position)) / span
        rel_velocity = (np.asarray(entity.velocity) - np.asarray(observer.velocity)) / float(vDomain[1])
        health_scale = initial_health if entity.Color == "Entity" else 1.0
        relation = [
            float(entity.Color == observer.Color),
            float(entity.Color not in {observer.Color, "Entity"}),
            float(entity.Color == "Entity"),
            float(entity.Id == observer.Id),
        ]
        return np.asarray(
            list(rel_position)
            + list(rel_velocity)
            + [float(entity.Health) / health_scale, float(entity.Health > 0)]
            + relation,
            dtype=np.float32,
        )

    def _self_features(self, side: str, agent: object) -> np.ndarray:
        from HAD_Env.config import AeroPoint, vDomain

        low = np.asarray([bounds[0] for bounds in AeroPoint], dtype=np.float32)
        span = np.asarray([bounds[1] - bounds[0] for bounds in AeroPoint], dtype=np.float32)
        position = 2.0 * (np.asarray(agent.position) - low) / span - 1.0
        velocity = np.asarray(agent.velocity) / float(vDomain[1])
        counts = [
            np.log1p(sum(one.Health > 0 for one in self._agents(side))),
            np.log1p(sum(one.Health > 0 for one in self._agents(self._opponent(side)))),
            float(self.env.targets[0].Health > 0),
        ]
        return np.asarray(list(position) + list(velocity) + [float(agent.Health)] + counts, dtype=np.float32)

    def _task_features(self, side: str, agent: object) -> np.ndarray:
        from HAD_Env.config import AeroPoint, initial_health

        target = self.env.targets[0]
        span = np.asarray([high - low for low, high in AeroPoint], dtype=np.float32)
        relative = (np.asarray(target.position) - np.asarray(agent.position)) / span
        role = 1.0 if side == "Red" else -1.0
        remaining_horizon = max(0.0, 1.0 - self.step_count / self.max_steps)
        return np.asarray(
            list(relative)
            + [
                float(target.Health) / initial_health,
                float(target.Health > 0),
                role,
                remaining_horizon,
            ],
            dtype=np.float32,
        )

    def _state_features(self, entity: object) -> np.ndarray:
        from HAD_Env.config import AeroPoint, initial_health, vDomain

        low = np.asarray([bounds[0] for bounds in AeroPoint], dtype=np.float32)
        span = np.asarray([bounds[1] - bounds[0] for bounds in AeroPoint], dtype=np.float32)
        position = (np.asarray(entity.position) - low) / span
        velocity = np.asarray(entity.velocity) / float(vDomain[1])
        health_scale = initial_health if entity.Color == "Entity" else 1.0
        types = [float(entity.Color == name) for name in ("Red", "Blue", "Entity")]
        remaining_horizon = max(0.0, 1.0 - self.step_count / self.max_steps)
        return np.asarray(
            list(position)
            + list(velocity)
            + [float(entity.Health) / health_scale, float(entity.Health > 0)]
            + types
            + [remaining_horizon],
            dtype=np.float32,
        )

    def observe(self, side: str) -> Dict[str, np.ndarray]:
        controlled = self._agents(side)
        world = self.env.world
        entity_obs = np.stack(
            [np.stack([self._entity_features(agent, entity) for entity in world]) for agent in controlled]
        )
        entity_mask = np.stack(
            [np.asarray([entity.Health > 0 for entity in world], dtype=bool) for _ in controlled]
        )
        agent_mask = np.asarray([agent.Health > 0 for agent in controlled], dtype=bool)
        avail_actions = np.ones((len(controlled), self.ACTION_DIM), dtype=bool)
        avail_actions[~agent_mask] = False
        avail_actions[~agent_mask, 0] = True
        return {
            "entity_obs": entity_obs,
            "entity_mask": entity_mask,
            "self_obs": np.stack([self._self_features(side, agent) for agent in controlled]),
            "task_obs": np.stack([self._task_features(side, agent) for agent in controlled]),
            "agent_mask": agent_mask,
            "avail_actions": avail_actions,
            "state_entities": np.stack([self._state_features(entity) for entity in world]),
            # Global-state masks distinguish real world slots from batch
            # padding.  Death is already encoded by the health/alive features;
            # masking dead slots made an all-destroyed terminal state look like
            # an invalid empty set to centralized critics and mixers.
            "state_mask": np.ones(len(world), dtype=bool),
        }


def tensorize_had_observation(
    observation: Dict[str, np.ndarray], device: torch.device = torch.device("cpu")
) -> Tuple[TeamObservation, GlobalState]:
    """Add a batch axis and convert one side's observation to tensors."""

    team = TeamObservation(
        entity_obs=torch.as_tensor(observation["entity_obs"], dtype=torch.float32, device=device).unsqueeze(0),
        entity_mask=torch.as_tensor(observation["entity_mask"], dtype=torch.bool, device=device).unsqueeze(0),
        self_obs=torch.as_tensor(observation["self_obs"], dtype=torch.float32, device=device).unsqueeze(0),
        task_obs=torch.as_tensor(observation["task_obs"], dtype=torch.float32, device=device).unsqueeze(0),
        agent_mask=torch.as_tensor(observation["agent_mask"], dtype=torch.bool, device=device).unsqueeze(0),
        avail_actions=torch.as_tensor(observation["avail_actions"], dtype=torch.bool, device=device).unsqueeze(0),
    )
    state = GlobalState(
        entities=torch.as_tensor(observation["state_entities"], dtype=torch.float32, device=device).unsqueeze(0),
        entity_mask=torch.as_tensor(observation["state_mask"], dtype=torch.bool, device=device).unsqueeze(0),
    )
    return team, state
