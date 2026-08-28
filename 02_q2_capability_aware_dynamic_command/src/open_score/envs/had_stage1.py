"""Minimal attack-agent/target adapter for the first variable-scale HAD demo."""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from open_score.contracts import GlobalState, TeamObservation


class HADStage1Adapter:
    """Control one HAD team with discrete motion primitives.

    A single HAD instance still has a fixed roster.  Variable-scale training is
    obtained by creating instances with different `(red, blue, target)` counts
    while reusing the same neural parameters.  Attrition changes the active mask
    inside an episode; true reinforcement/spawn events belong to Open-HAD P1.
    """

    ENTITY_DIM = 12
    SELF_DIM = 10
    TASK_DIM = 6
    STATE_ENTITY_DIM = 11
    ACTION_DIM = 9

    def __init__(
        self,
        red_attackers: int,
        blue_attackers: int,
        targets: int,
        controlled_side: str = "Red",
        max_steps: int = 400,
        task_type: str = "Training",
    ):
        if controlled_side not in {"Red", "Blue"}:
            raise ValueError("controlled_side must be Red or Blue")
        if min(red_attackers, blue_attackers, targets) <= 0:
            raise ValueError("the stage-1 demo needs both teams and at least one target")
        # Local import keeps the neural modules usable even when HAD is absent.
        from HAD_Env.make_env import HADEnv

        self.env = HADEnv(red_attackers, blue_attackers, targets, task_type=task_type)
        self.controlled_side = controlled_side
        self.max_steps = max_steps
        self.step_count = 0
        self.task_ids = np.zeros(len(self._controlled_agents()), dtype=np.int64)

    def _controlled_agents(self) -> List[object]:
        return self.env.red_agents if self.controlled_side == "Red" else self.env.blue_agents

    def _opponent_agents(self) -> List[object]:
        return self.env.blue_agents if self.controlled_side == "Red" else self.env.red_agents

    def reset(self, seed: Optional[int] = None, evaluate: bool = False) -> Dict[str, np.ndarray]:
        if seed is not None:
            # Temporary compatibility with legacy HAD; Open-HAD P0 replaces this
            # global RNG with an environment-owned Generator and cloneable state.
            np.random.seed(seed)
        self.env.reset(evaluate=evaluate)
        self.step_count = 0
        self.task_ids = np.arange(len(self._controlled_agents()), dtype=np.int64) % len(self.env.targets)
        return self.observe()

    @staticmethod
    def _unit(vector: Sequence[float]) -> np.ndarray:
        value = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(value))
        return value / norm if norm > 1e-8 else np.zeros_like(value)

    def _target_for(self, local_agent_index: int):
        return self.env.targets[int(self.task_ids[local_agent_index]) % len(self.env.targets)]

    def _decode_action(self, action: int, agent: object, local_index: int) -> np.ndarray:
        primitives = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float32,
        )
        if action < len(primitives):
            return primitives[action]
        if action == 7:
            target = self._target_for(local_index)
            return self._unit(np.asarray(target.position) - np.asarray(agent.position))
        if action == 8:
            enemies = [one for one in self._opponent_agents() if one.Health > 0]
            if not enemies:
                return np.zeros(3, dtype=np.float32)
            nearest = min(enemies, key=lambda one: np.linalg.norm(np.asarray(one.position) - agent.position))
            return self._unit(np.asarray(nearest.position) - np.asarray(agent.position))
        raise ValueError("action id outside the nine stage-1 primitives")

    def _scripted_action(self, agent: object) -> np.ndarray:
        if agent.Health <= 0:
            return np.zeros(3, dtype=np.float32)
        if agent.Color == "Blue":
            targets = [target for target in self.env.targets if target.Health > 0]
            destination = min(
                targets,
                key=lambda target: np.linalg.norm(np.asarray(target.position) - agent.position),
            )
        else:
            enemies = [one for one in self.env.blue_agents if one.Health > 0]
            if not enemies:
                return np.zeros(3, dtype=np.float32)
            destination = min(
                enemies,
                key=lambda enemy: np.linalg.norm(np.asarray(enemy.position) - agent.position),
            )
        return self._unit(np.asarray(destination.position) - np.asarray(agent.position))

    def _health_totals(self) -> Tuple[float, float]:
        enemy = sum(float(agent.Health) for agent in self._opponent_agents())
        targets = sum(float(target.Health) for target in self.env.targets)
        return enemy, targets

    def step(self, action_ids: Sequence[int]):
        controlled = self._controlled_agents()
        if len(action_ids) != len(controlled):
            raise ValueError("one discrete action is required for each controlled roster slot")
        controlled_by_id = {agent.Id: index for index, agent in enumerate(controlled)}
        all_actions = []
        enemy_health_before, target_health_before = self._health_totals()
        for agent in self.env.agents:
            if agent.Id in controlled_by_id:
                local_index = controlled_by_id[agent.Id]
                action = 0 if agent.Health <= 0 else int(action_ids[local_index])
                all_actions.append(self._decode_action(action, agent, local_index))
            else:
                all_actions.append(self._scripted_action(agent))
        _, _, _, reward_data, _, info = self.env.step(all_actions)
        self.step_count += 1
        enemy_health_after, target_health_after = self._health_totals()
        enemy_damage = enemy_health_before - enemy_health_after
        target_damage = target_health_before - target_health_after
        terminal_reward = float(reward_data["RealReward"][self.controlled_side])
        if self.controlled_side == "Red":
            shaped_reward = enemy_damage - 2.0 * target_damage
        else:
            shaped_reward = enemy_damage + 2.0 * target_damage
        terminated = bool(abs(self.env.is_terminal()))
        truncated = self.step_count >= self.max_steps and not terminated
        reward = terminal_reward + shaped_reward - (0.01 if not terminated else 0.0)
        info.update({"terminated": terminated, "truncated": truncated})
        return self.observe(), float(reward), bool(terminated or truncated), info

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

    def _self_features(self, agent: object) -> np.ndarray:
        from HAD_Env.config import AeroPoint, vDomain

        low = np.asarray([bounds[0] for bounds in AeroPoint], dtype=np.float32)
        span = np.asarray([bounds[1] - bounds[0] for bounds in AeroPoint], dtype=np.float32)
        position = 2.0 * (np.asarray(agent.position) - low) / span - 1.0
        velocity = np.asarray(agent.velocity) / float(vDomain[1])
        counts = [
            np.log1p(sum(one.Health > 0 for one in self._controlled_agents())),
            np.log1p(sum(one.Health > 0 for one in self._opponent_agents())),
            np.log1p(sum(one.Health > 0 for one in self.env.targets)),
        ]
        return np.asarray(list(position) + list(velocity) + [float(agent.Health)] + counts, dtype=np.float32)

    def _task_features(self, agent: object, local_index: int) -> np.ndarray:
        from HAD_Env.config import AeroPoint, initial_health

        target = self._target_for(local_index)
        span = np.asarray([high - low for low, high in AeroPoint], dtype=np.float32)
        relative = (np.asarray(target.position) - np.asarray(agent.position)) / span
        target_index = float(self.task_ids[local_index]) / max(1, len(self.env.targets) - 1)
        return np.asarray(
            list(relative)
            + [float(target.Health) / initial_health, float(target.Health > 0), target_index],
            dtype=np.float32,
        )

    def _state_features(self, entity: object) -> np.ndarray:
        from HAD_Env.config import AeroPoint, initial_health, vDomain

        low = np.asarray([bounds[0] for bounds in AeroPoint], dtype=np.float32)
        span = np.asarray([bounds[1] - bounds[0] for bounds in AeroPoint], dtype=np.float32)
        position = (np.asarray(entity.position) - low) / span
        velocity = np.asarray(entity.velocity) / float(vDomain[1])
        health_scale = initial_health if entity.Color == "Entity" else 1.0
        types = [float(entity.Color == "Red"), float(entity.Color == "Blue"), float(entity.Color == "Entity")]
        return np.asarray(
            list(position)
            + list(velocity)
            + [float(entity.Health) / health_scale, float(entity.Health > 0)]
            + types,
            dtype=np.float32,
        )

    def observe(self) -> Dict[str, np.ndarray]:
        controlled = self._controlled_agents()
        world = self.env.world
        entity_obs = np.stack(
            [np.stack([self._entity_features(agent, entity) for entity in world]) for agent in controlled]
        )
        entity_mask = np.stack(
            [np.asarray([entity.Health > 0 for entity in world], dtype=bool) for _ in controlled]
        )
        self_obs = np.stack([self._self_features(agent) for agent in controlled])
        task_obs = np.stack([self._task_features(agent, index) for index, agent in enumerate(controlled)])
        agent_mask = np.asarray([agent.Health > 0 for agent in controlled], dtype=bool)
        avail_actions = np.ones((len(controlled), self.ACTION_DIM), dtype=bool)
        avail_actions[~agent_mask] = False
        avail_actions[~agent_mask, 0] = True
        state_entities = np.stack([self._state_features(entity) for entity in world])
        state_mask = np.asarray([entity.Health > 0 for entity in world], dtype=bool)
        return {
            "entity_obs": entity_obs,
            "entity_mask": entity_mask,
            "self_obs": self_obs,
            "task_obs": task_obs,
            "agent_mask": agent_mask,
            "avail_actions": avail_actions,
            "state_entities": state_entities,
            "state_mask": state_mask,
        }


def tensorize_had_observation(
    observation: Dict[str, np.ndarray], device: torch.device = torch.device("cpu")
) -> Tuple[TeamObservation, GlobalState]:
    """Add a batch axis and convert one adapter observation to torch tensors."""

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
