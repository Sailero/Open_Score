"""Simultaneous multi-agent interfaces over the unchanged native HAD world.

The default API follows PettingZoo ParallelEnv. MPEEnv is a small fixed-list
adapter for legacy consumers; it does not change physical stepping or reward.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping

import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv

from had_env.actions import ACCELERATION_PRIMITIVES, PLANAR_ACCELERATION_PRIMITIVES
from had_env.core.config import Interval
from had_env.scenarios.defense import Scenario


class HADParallelEnv(ParallelEnv):
    """Both teams submit actions from the same pre-step observations.

    Observation rows are in fixed world order excluding self:
    relative position (3), relative velocity (3), health, alive, and
    red/blue/target flags. Dead rows are zeroed; ``entity_mask`` is in info.

    By default scalar reward equals the agent's team's ``RealReward``. Set an
    explicit four-element ``reward_weights`` to take a weighted sum of native
    ``LatentReward=[hit, target, enemy, episode]`` instead. Both native values
    are always available in infos. The episode component already includes
    RealReward and must not be added a second time.

    In survival mode a killed agent receives its final native transition and
    is removed from ``agents``; its fixed entity slot remains observable.
    Damage mode keeps every agent slot until the world ends: dead agents have
    zero observations and forced no-op actions, and continue receiving the
    team's target-damage reward. Each slot receives the full team reward;
    summing rewards across teammates would count the same reward repeatedly.
    ``agent_mask`` marks participation for the returned observation/next action;
    use the preceding observation's mask for the action that caused a death.
    ``bootstrap_mask`` is the global team critic mask and remains one after a
    death or sampling truncation.
    ``max_cycles`` is a sampling truncation and never awards a new win.
    Time is available in info; observation and state dimensions are unchanged.
    Planar (default) control uses 9 discrete actions or a 2-vector; spatial
    control explicitly selects spatial_dim=3 for 27 actions or a 3-vector.
    """

    metadata = {"name": "had_defense_v1", "render_modes": ["human", "rgb_array"],
                "render_fps": 30, "is_parallelizable": True}

    def __init__(self, scenario=None, *, max_cycles=100, continuous=False,
                 render_mode=None, reward_weights=None, seed=None, record_events=False):
        if type(max_cycles) is not int or max_cycles < 1:
            raise ValueError("max_cycles must be a positive integer")
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("render_mode must be None, 'human', or 'rgb_array'")
        if not isinstance(continuous, bool):
            raise ValueError("continuous must be a boolean")
        self.scenario = scenario or Scenario()
        self.task_mode = self.scenario.task_mode
        self.spatial_dim = self.scenario.spatial_dim
        self.acceleration_primitives = (
            PLANAR_ACCELERATION_PRIMITIVES if self.spatial_dim == 2 else ACCELERATION_PRIMITIVES
        ).copy()
        self.metadata = {**type(self).metadata, "spatial_dim": self.spatial_dim}
        self.max_cycles, self.continuous, self.render_mode = max_cycles, continuous, render_mode
        self.reward_weights = None
        if reward_weights is not None:
            if self.task_mode == "damage":
                raise ValueError("damage mode uses raw target damage and does not accept reward_weights")
            weights = np.asarray(reward_weights, dtype=np.float64)
            if weights.shape != (4,) or not np.isfinite(weights).all():
                raise ValueError("reward_weights must contain four finite values")
            self.reward_weights = weights.copy()
        self.world = self.scenario.make_world(seed=seed)
        self.world.record_events = bool(record_events)
        self.possible_agents = ([f"red_{i}" for i in range(self.scenario.red_count)]
                                + [f"blue_{i}" for i in range(self.scenario.blue_count)])
        self.agent_name_mapping = {name: i for i, name in enumerate(self.possible_agents)}
        self.entity_names = self.possible_agents + [f"target_{i}" for i in range(self.scenario.target_count)]
        self._observation_entities = {
            agent: tuple(name for name in self.entity_names if name != agent)
            for agent in self.possible_agents
        }
        from had_env.core.config import OBS_ENTITY_DIM
        shape = (len(self.entity_names) - 1, OBS_ENTITY_DIM)
        self.observation_spaces = {a: spaces.Box(-np.inf, np.inf, shape, np.float32)
                                   for a in self.possible_agents}
        self.action_spaces = {a: (spaces.Box(-1., 1., (self.spatial_dim,), np.float32) if continuous
                                  else spaces.Discrete(len(self.acceleration_primitives)))
                              for a in self.possible_agents}
        self.state_space = spaces.Box(-np.inf, np.inf, (len(self.entity_names) * OBS_ENTITY_DIM,), np.float32)
        self._all_actions = (1,) * len(self.acceleration_primitives)
        self._noop_actions = (1,) + (0,) * (len(self.acceleration_primitives) - 1)
        self.agents = []
        self.num_cycles, self.outcome_red = 0, 0
        self._has_reset, self._closed = False, False
        self.seed(seed)

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def seed(self, seed=None):
        """Reseed only this environment and its spaces; call reset afterward."""
        self.np_random, value = seeding.np_random(seed)
        self.np_random_seed = int(value)
        self.world.np_random = self.np_random
        children = np.random.SeedSequence(value).spawn(2 * len(self.possible_agents))
        for i, agent in enumerate(self.possible_agents):
            self.action_spaces[agent].seed(int(children[2*i].generate_state(1)[0]))
            self.observation_spaces[agent].seed(int(children[2*i+1].generate_state(1)[0]))
        self._has_reset = False
        return [int(value)]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        native_obs, _, _ = self.scenario.reset_world(self.world, options=options)
        self.agents = self.possible_agents.copy()
        self.num_cycles, self.outcome_red = 0, 0
        self._has_reset, self._closed = True, False
        obs = self._observations(native_obs, self.agents)
        task_info = self.world.task_info()
        infos = {a: self._info(a, task_info=task_info) for a in self.agents}
        if self.render_mode == "human":
            self.render()
        return obs, infos

    def _observations(self, values, agents):
        return {
            a: (np.zeros(self.observation_spaces[a].shape, dtype=np.float32)
                if self.task_mode == "damage" and self._entity(a).Health <= 0
                else np.array(values[self.agent_name_mapping[a]], dtype=np.float32, copy=True))
            for a in agents
        }

    def _decode_action(self, agent, action):
        if not self.continuous:
            if isinstance(action, (bool, np.bool_)) or not self.action_space(agent).contains(action):
                raise ValueError(f"{agent} action must be an integer in 0..{len(self.acceleration_primitives) - 1}")
            # tolist preserves the original discrete-control arithmetic path.
            return self.acceleration_primitives[int(action)].tolist()
        try:
            value = np.asarray(action)
        except (ValueError, TypeError) as error:
            raise ValueError(f"{agent} action must be a finite {self.spatial_dim}-vector in [-1,1]") from error
        if (value.shape != (self.spatial_dim,) or value.dtype.kind not in "iuf"
                or not np.isfinite(value).all() or np.any(np.abs(value) > 1)):
            raise ValueError(f"{agent} action must be a finite {self.spatial_dim}-vector in [-1,1]")
        return value.tolist() + ([0.] if self.spatial_dim == 2 else [])

    def step(self, actions):
        if not self._has_reset or self._closed:
            raise RuntimeError("Call reset before stepping an uninitialized or closed environment")
        if not isinstance(actions, Mapping) or set(actions) != set(self.agents):
            raise ValueError("actions must contain exactly the current env.agents keys")
        if not self.agents:
            return {}, {}, {}, {}, {}
        acting = self.agents.copy()
        # Decode the complete joint action before touching physics or RNG.
        decoded = {
            a: ([0., 0., 0.] if self.task_mode == "damage" and self._entity(a).Health <= 0
                else self._decode_action(a, actions[a]))
            for a in acting
        }
        batch = [decoded.get(a, [0., 0., 0.]) for a in self.possible_agents]
        self.world.step_physics(batch)
        native_obs = self.scenario.observation(self.world)
        native_rewards = self.scenario.reward(self.world)
        self.num_cycles += 1
        self.outcome_red = self.scenario.done(self.world)
        global_terminated = bool(self.world.is_episode_done())
        global_truncated = bool(self.num_cycles >= self.max_cycles and not global_terminated)
        terminations = {
            a: bool(global_terminated or (self.task_mode == "survival" and self._entity(a).Health <= 0))
            for a in acting
        }
        truncations = {a: bool(self.num_cycles >= self.max_cycles and not terminations[a]) for a in acting}
        task_info = self.world.task_info()
        rewards, infos = {}, {}
        for a in acting:
            info = self._info(a, native_rewards, task_info=task_info,
                              global_terminated=global_terminated, global_truncated=global_truncated)
            info["terminated"], info["truncated"] = terminations[a], truncations[a]
            rewards[a] = (info["team_reward"] if self.task_mode == "damage" else
                          info["RealReward"] if self.reward_weights is None else
                          float(np.dot(info["LatentReward"], self.reward_weights)))
            infos[a] = info
        observations = self._observations(native_obs, acting)
        self.agents = [a for a in acting if not (terminations[a] or truncations[a])]
        if self.render_mode == "human":
            self.render()
        return observations, rewards, terminations, truncations, infos

    def _entity(self, agent):
        return self.world.agents[self.agent_name_mapping[agent]]

    def _info(self, agent, rewards=None, *, task_info=None,
              global_terminated=False, global_truncated=False):
        entity = self._entity(agent)
        side = entity.Color
        index = self.agent_name_mapping[agent] - (self.scenario.red_count if side == "Blue" else 0)
        real = float(rewards["RealReward"][side]) if rewards else 0.
        latent = list(map(float, rewards["LatentReward"][side][index])) if rewards else [0.] * 4
        alive = bool(entity.Health > 0)
        task = task_info if task_info is not None else self.world.task_info()
        team_reward = ((-1.0 if side == "Red" else 1.0) * float(task["step_target_damage"])
                       if self.task_mode == "damage" else real)
        mask = None
        if getattr(self.world, "entity_mask", None) is not None:
            index = self.agent_name_mapping[agent]
            if index < len(self.world.entity_mask):
                mask = tuple(self.world.entity_mask[index])
                if self.task_mode == "damage" and not alive:
                    mask = (0.0,) * len(mask)
        return {**task, "target_damage_by_target": dict(task["target_damage_by_target"]),
                "episode_returns": (None if task["episode_returns"] is None else dict(task["episode_returns"])),
                "entity_id": int(entity.Id), "side": side.lower(), "role": entity.Type.lower(),
                "spatial_dim": self.spatial_dim, "plane_altitude": float(self.world.plane_altitude),
                "alive": alive, "agent_mask": float(alive), "health": float(entity.Health),
                "action_mask": None if self.continuous else self._all_actions if alive else self._noop_actions,
                "observation_entities": self._observation_entities[agent],
                "entity_mask": mask,
                "RealReward": real, "LatentReward": latent, "outcome_red": self.outcome_red,
                "team_reward": team_reward,
                "global_terminated": bool(global_terminated), "global_truncated": bool(global_truncated),
                "bootstrap_mask": float(not global_terminated),
                "cycle": self.num_cycles, "sim_time": self.num_cycles * Interval,
                "max_cycles": self.max_cycles, "remaining_cycles": max(0, self.max_cycles - self.num_cycles),
                "events": copy.deepcopy(self.world.last_physics_events),
                "core_version": self.world.core_version, "physics_protocol": self.world.physics_protocol}

    def state(self):
        if not self._has_reset:
            raise RuntimeError("Call reset before requesting state")
        return np.asarray(self.world.get_global_state(), dtype=np.float32)

    def render(self):
        if not self._has_reset or self._closed:
            raise RuntimeError("Call reset before rendering")
        if self.render_mode == "rgb_array":
            return self.world.render_rgb_array()
        if self.render_mode == "human":
            self.world.render()
        return None

    def close(self):
        self.world.close()
        self._closed = True


class MPEEnv:
    """Fixed-order MPE-style list interface (not the old Gym package).

    ``reset`` returns obs_n; ``step`` returns obs_n, reward_n, done_n, info.
    Observations are flattened native rows. Survival dead slots are zeroed
    after their final transition. Damage dead slots immediately receive zero
    observations and keep their team reward stream until the world ends.
    Actions for already-dead slots are ignored. Discrete
    actions are integer indices, or an explicit one-hot vector matching the
    action space (9 entries in 2D, 27 in 3D).
    ``info['n']`` retains termination versus truncation for each fixed slot.
    """

    def __init__(self, env):
        self.parallel_env = env
        self.possible_agents = env.possible_agents.copy()
        self.n = len(self.possible_agents)
        self.action_space = [env.action_space(a) for a in self.possible_agents]
        self.observation_space = [spaces.flatten_space(env.observation_space(a)) for a in self.possible_agents]
        self.world, self.metadata = env.world, env.metadata
        self._done = {a: False for a in self.possible_agents}
        self._infos = {}

    @property
    def agents(self):
        return self.world.agents

    @property
    def unwrapped(self):
        return self.parallel_env

    def reset(self, seed=None, options=None):
        obs, self._infos = self.parallel_env.reset(seed=seed, options=options)
        self._done = {a: False for a in self.possible_agents}
        return [obs[a].reshape(-1) for a in self.possible_agents]

    def step(self, actions):
        if len(actions) != self.n:
            raise ValueError(f"MPE actions must have {self.n} fixed slots")
        joint = {}
        for a, value in zip(self.possible_agents, actions):
            if a not in self.parallel_env.agents:
                continue
            if self.parallel_env.task_mode == "damage" and self.parallel_env._entity(a).Health <= 0:
                joint[a] = [0.] * self.parallel_env.spatial_dim if self.parallel_env.continuous else 0
                continue
            array = np.asarray(value)
            action_count = None if self.parallel_env.continuous else self.parallel_env.action_space(a).n
            if not self.parallel_env.continuous and array.shape == (action_count,):
                if not np.all((array == 0) | (array == 1)) or np.sum(array) != 1:
                    raise ValueError(f"MPE vector actions must be explicit length-{action_count} one-hot vectors")
                value = int(np.argmax(array))
            joint[a] = value
        obs, rewards, terms, truncs, infos = self.parallel_env.step(joint)
        rows, values, dones, details = [], [], [], []
        for i, a in enumerate(self.possible_agents):
            self._done[a] = self._done[a] or terms.get(a, False) or truncs.get(a, False)
            rows.append(obs[a].reshape(-1) if a in obs else np.zeros(self.observation_space[i].shape, np.float32))
            values.append(rewards.get(a, 0.))
            dones.append(self._done[a])
            info = infos.get(a, {**self._infos.get(a, {}), "inactive": True})
            if a not in infos:
                info = {**info, "RealReward": 0., "LatentReward": [0.] * 4, "events": []}
            self._infos[a] = info
            details.append(info)
        return rows, values, dones, {"n": details}

    def seed(self, seed=None):
        return self.parallel_env.seed(seed)

    def state(self):
        return self.parallel_env.state()

    def render(self, mode=None):
        if mode is not None:
            if mode not in self.metadata["render_modes"]:
                raise ValueError("unsupported render mode")
            self.parallel_env.render_mode = mode
        return self.parallel_env.render()

    def close(self):
        self.parallel_env.close()
