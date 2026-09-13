"""ALMA entity-environment contract for HAD and its native firefighters check."""
from __future__ import annotations

import copy
import numpy as np

from .features import (MAX_AGENTS, ENTITY_DIM, N_ACTIONS, masks_from_entity_mask,
                       available_actions, blue_nearest_from_table, entities_from_state,
                       resolve_pad, task_masks)
from .had_wrapper import HADWrapper, run_environment_checks
from .scales import ScaleSampler, as_scale
from ..utils.seeding import EpisodeSeedStream, split_seeds


class HADEntityEnv:
    def __init__(self, seed=0, worker_id=0, train_dist="mixed_le10", scale=None, config=None,
                 max_steps=100, episode_limit=None, blue_upper="reactive", blue_lower="rush",
                 entity_scheme=True, gamma=0.99, **kwargs):
        if not entity_scheme:
            raise ValueError("HAD cross-scale experiments require the entity scheme")
        seeds = split_seeds(seed, worker_id)
        self.scale_sampler = ScaleSampler(kwargs.pop("scale_seed", seeds["scale_seed"]), train_dist)
        self.episode_seeds = EpisodeSeedStream(kwargs.pop("environment_seed", seeds["environment_seed"]))
        self.fixed_scale = as_scale(config if config is not None else scale) if config is not None or scale is not None else None
        self.episode_limit = int(max_steps if episode_limit is None else episode_limit)
        self.wrapper = HADWrapper(scale=self.fixed_scale or (8, 8, 2), max_steps=self.episode_limit,
                                  blue_upper=blue_upper, blue_lower=blue_lower, gamma=gamma, **kwargs)
        self.n_agents = self.wrapper.n_red
        self.n_blue = self.wrapper.n_blue
        self.n_targets = self.wrapper.n_targets
        self.n_entities = self.wrapper.n_entities
        self.n_actions = N_ACTIONS
        self.train_dist = train_dist

    def reset(self, seed=None, config=None, evaluate=False, retain_trajectory=False,
              test=False, **kwargs):
        episode_seed = self.episode_seeds.next() if seed is None else int(seed)
        scale = as_scale(config) if config is not None else self.fixed_scale or self.scale_sampler.sample()
        self.wrapper.reset(episode_seed, scale, evaluate=evaluate or test,
                           retain_trajectory=retain_trajectory, **kwargs)
        return self.get_entities(), self.get_masks()

    def step(self, actions):
        return self.wrapper.step(actions)

    def get_entities(self):
        return self.wrapper.entities.copy()

    def get_entity_size(self):
        return ENTITY_DIM

    def get_masks(self):
        return masks_from_entity_mask(self.wrapper.entity_mask)

    def get_agent_mask(self):
        return 1 - self.wrapper.entity_mask[:self.n_agents].copy()

    def get_initial_agent_mask(self):
        """One denotes initial padding, while deaths remain graph nodes."""
        mask = np.ones(self.n_agents, dtype=np.uint8)
        mask[:self.wrapper.scale.N_R] = 0
        return mask

    def get_avail_actions(self):
        return available_actions(self.wrapper.entity_mask, self.n_agents)

    def get_avail_agent_actions(self, agent_id):
        return self.get_avail_actions()[agent_id]

    def get_total_actions(self):
        return N_ACTIONS

    def get_state(self):
        env = self.wrapper.adapter.env
        fractions = np.asarray([sum(a.Health > 0 for a in env.red_agents) / len(env.red_agents),
                                sum(a.Health > 0 for a in env.blue_agents) / len(env.blue_agents),
                                1 - self.wrapper.adapter.step_count / self.episode_limit], dtype=np.float32)
        return np.concatenate((self.wrapper.entities.reshape(-1), fractions))

    def get_state_size(self):
        return self.n_entities * ENTITY_DIM + 3

    def get_obs(self):
        table = self.get_entities()
        return np.repeat(table.reshape(1, -1), self.n_agents, axis=0) * self.get_agent_mask()[:, None]

    def get_obs_agent(self, agent_id):
        return self.get_obs()[agent_id]

    def get_obs_size(self):
        return self.n_entities * ENTITY_DIM

    def get_env_info(self, args=None):
        return {"n_agents": self.n_agents, "n_entities": self.n_entities, "n_actions": N_ACTIONS,
                "entity_shape": ENTITY_DIM, "state_shape": self.get_state_size(),
                "obs_shape": self.get_obs_size(), "episode_limit": self.episode_limit,
                "gt_mask_avail": False, "feature_layout": "had",
                "n_tasks": self.wrapper.n_tasks()}

    def get_task_masks(self):
        masks = task_masks(self.wrapper.entities, self.wrapper.entity_mask, self.wrapper.scale.K,
                           self.n_agents, self.n_blue, self.n_targets,
                           subtask_set=self.wrapper.subtask_set)
        masks["hier_decision"] = np.asarray([int(self.wrapper.hier_decision)], dtype=np.uint8)
        return masks

    def episode_summary(self):
        return self.wrapper.episode_summary()

    def get_stats(self):
        return self.episode_summary() if self.wrapper.adapter is not None else {}

    def get_agg_stats(self, stats):
        return {}

    def get_policy_state(self):
        return self.wrapper.get_policy_state()

    def get_trajectory(self):
        return copy.deepcopy(self.wrapper.diagnostics.trajectory)

    def get_rng_state(self):
        return {"scale": copy.deepcopy(self.scale_sampler.state_dict()),
                "episode": copy.deepcopy(self.episode_seeds.state_dict())}

    def set_rng_state(self, state):
        self.scale_sampler.load_state_dict(state["scale"])
        self.episode_seeds.load_state_dict(state["episode"])

    def save_replay(self):
        return self.get_trajectory()

    def close(self):
        self.wrapper.close()


class NativeEntityEnv:
    """Keeps native FF features/actions; only normalizes runner calls and seeds."""
    def __init__(self, name="ff", args_dict=None, **kwargs):
        if name not in ("ff", "rel_overgen", "stag_hunt"):
            raise ValueError("NativeEntityEnv supports ff and the official rel_overgen task")
        self.name = name
        options = dict(args_dict or {})
        options.update(kwargs)
        if name == "ff":
            from ..algos.pymarl.envs.firefighters.firefighters import FireFightersEnv
            from ..algos.pymarl.envs.firefighters.scenarios import generate_single_scen_dict
            options.setdefault("scenario_dict", generate_single_scen_dict(
                agent_list=["F", "B"], building_list=["F", "S"], bld_spacing=3))
            self.env = FireFightersEnv(**options)
        else:
            from pathlib import Path
            import yaml
            from ..algos.dcg_patch.stag_hunt import StagHunt
            official = yaml.safe_load((Path(__file__).parents[1] / "algos/dcg_patch/rel_overgen.yaml").read_text(encoding="utf-8"))
            native_options = dict(official["env_args"])
            native_options.update(options)
            self.env = StagHunt(**native_options)
        self.episode_limit = self.env.episode_limit
        self._episode_seed = int(options.get("seed", 0))
        self._return = 0.0
        self._steps = 0

    def reset(self, seed=None, evaluate=False, retain_trajectory=False, config=None, test=False, **kwargs):
        if seed is not None:
            self._episode_seed = int(seed)
        np.random.seed(self._episode_seed % (2 ** 32))
        if self.name == "ff":
            self.env.seed(self._episode_seed)
            self.env.reset(test=bool(evaluate or test), **kwargs)
            self._initial_mask = np.asarray(self.env.get_masks()["entity_mask"][:self.env.max_n_agents]).copy()
        else:
            import random
            random.seed(self._episode_seed)
            self.env.reset()
            self._initial_mask = np.zeros(self.env.n_agents, dtype=np.uint8)
        self._return, self._steps = 0.0, 0
        return self.get_entities(), self.get_masks()

    def step(self, actions):
        if self.name != "ff":
            import torch
            actions = torch.as_tensor(actions)
        reward, done, raw_info = self.env.step(actions)
        self._return += float(reward)
        self._steps += 1
        info = {key: value.tolist() if isinstance(value, np.ndarray) else
                value.item() if isinstance(value, np.generic) else value for key, value in raw_info.items()}
        # FF checks natural task completion before its horizon branch. A task
        # solved exactly at the limit is terminal, not a sampling truncation.
        truncated = bool(info.get("episode_limit", False))
        info.update(terminated=bool(done and not truncated), truncated=truncated,
                    episode_limit=truncated, bootstrap_mask=float(not (done and not truncated)))
        if done:
            info["episode_summary"] = self.episode_summary()
        return float(reward), bool(done), info

    def get_entities(self):
        values = self.env.get_entities() if self.name == "ff" else self.env.get_obs()
        return np.asarray(values, dtype=np.float32)

    def get_masks(self):
        if self.name == "ff":
            masks = self.env.get_masks()
            return {key: np.asarray(masks[key], dtype=np.uint8) for key in ("obs_mask", "entity_mask")}
        absent = (1 - self.env.agents_not_frozen[:, 0]).astype(np.uint8)
        obs_mask = 1 - np.eye(self.env.n_agents, dtype=np.uint8)
        obs_mask = np.maximum(obs_mask, np.maximum(absent[:, None], absent[None, :]))
        return {"obs_mask": obs_mask, "entity_mask": absent}

    def get_initial_agent_mask(self):
        return self._initial_mask.copy()

    def get_env_info(self, args=None):
        info = self.env.get_env_info(args)
        if self.name != "ff":
            return {**info, "n_entities": self.env.n_agents, "entity_shape": self.env.obs_size,
                    "feature_layout": "native", "gt_mask_avail": False}
        return {**info, "feature_layout": "native", "gt_mask_avail": False,
                "state_shape": info["n_entities"] * info["entity_shape"]}

    def get_state(self):
        if self.name == "ff":
            return self.get_entities().reshape(-1)
        return np.asarray(self.env.get_state(), dtype=np.float32).reshape(-1)

    def episode_summary(self):
        return {"episode_seed": self._episode_seed, "episode_return": self._return,
                "ep_len": self._steps, "environment": self.name}

    def get_trajectory(self):
        return []

    def get_rng_state(self):
        import random
        return {"numpy": np.random.get_state(), "python": random.getstate(), "episode_seed": self._episode_seed}

    def set_rng_state(self, state):
        import random
        np.random.set_state(state["numpy"])
        random.setstate(state["python"])
        self._episode_seed = state["episode_seed"]

    def __getattr__(self, name):
        return getattr(self.env, name)


class FrozenPolicyAdapter:
    """Greedy checkpoint policy using the same absolute entities as training."""
    def __init__(self, mac, args, scheme, groups, preprocess, mixer=None):
        self.mac, self.args, self.scheme = mac, args, scheme
        self.groups, self.preprocess, self.mixer = groups, preprocess, mixer
        self.mac.agent.eval()
        if self.mixer is not None:
            self.mixer.eval()
        self.reset()

    def reset(self):
        from components.episode_buffer import EpisodeBatch
        self.batch = EpisodeBatch(self.scheme, self.groups, 1, int(self.args.episode_limit) + 1,
                                  preprocess=self.preprocess, device="cpu")
        self.mac.init_hidden(batch_size=1)
        self.last_step, self.last_result = -1, None
        self.q_tot, self.q_i = [], []
        self.roster = None
        self._alloc_hold = 0
        self._last_nearest = None
        self._last_alive = None

    def act(self, state, side, action_ids):
        import torch as th
        from dataclasses import replace
        if side.lower() != "red":
            raise ValueError("cross-scale checkpoints control only the Red team")
        if state.step == self.last_step:
            return dict(self.last_result)
        if state.step != self.last_step + 1:
            raise ValueError("policy needs consecutive states and reset() at every new episode")
        if self.roster is None:
            self.roster = {key: tuple(e.id for e in getattr(state, key)) for key in ("red", "blue", "targets")}
            budget = getattr(self.args, "pool_slots", None)
            counts = tuple(len(getattr(state, key)) for key in ("red", "blue", "targets"))
            if budget is not None and any(c > limit for c, limit in zip(counts, budget)):
                raise ValueError(f"configuration {counts} exceeds this checkpoint's ordered slot "
                                 f"budget {tuple(budget)}; it has no parameters for the extra entities")
        ordered = {}
        for key, ids in self.roster.items():
            rows = {e.id: e for e in getattr(state, key)}
            if set(rows) != set(ids):
                raise ValueError("policy physical roster changed within an episode")
            ordered[key] = tuple(rows[identity] for identity in ids)
        state = replace(state, **ordered)
        entities, absent = entities_from_state(state)
        masks = masks_from_entity_mask(absent)
        avail = available_actions(absent)
        initial = np.ones(MAX_AGENTS, dtype=np.uint8)
        initial[:len(state.red)] = 0
        fractions = np.asarray([sum(e.alive for e in state.red) / len(state.red),
                                sum(e.alive for e in state.blue) / len(state.blue),
                                1 - state.step / state.max_steps], dtype=np.float32)
        step = int(state.step)
        data = {"entities": entities[None], "obs_mask": masks["obs_mask"][None],
                "entity_mask": absent[None], "agent_mask": (1 - absent[:MAX_AGENTS])[None],
                "initial_agent_mask": initial[None], "state": np.concatenate((entities.reshape(-1), fractions))[None],
                "avail_actions": avail[None], "reset": [[0]]}
        if getattr(self.args, "multi_task", False):
            # Same subtask decomposition and decision clock as the training
            # runner, so a hierarchical checkpoint acts as it was trained.
            env_args = dict(getattr(self.args, "env_args", None) or {})
            n_red, n_blue, n_targets = resolve_pad("eval")
            subtask_set = env_args.get("subtask_set", "targets")
            subtasks = task_masks(entities, absent, len(state.targets), n_red, n_blue, n_targets,
                                  subtask_set=subtask_set)
            data.update(entity2task_mask=subtasks["entity2task_mask"][None],
                        task_mask=subtasks["task_mask"][None],
                        hier_decision=[[int(self._hier_decision(state, absent, step, env_args))]])
        self.batch.update(data, ts=step)
        with th.no_grad():
            actions = self.mac.select_actions(self.batch, t_ep=step, t_env=0, test_mode=True)
            self.batch.update({"actions": actions.unsqueeze(-1)}, ts=step, mark_filled=False)
            if hasattr(self.mac, "evaluation_values"):
                values = self.mac.evaluation_values(self.batch, step, actions, [0], self.mixer)
                for key, storage in (("q_tot", self.q_tot), ("q_i", self.q_i)):
                    value = values[key][0]
                    if value is not None:
                        storage.append(float(value))
        ids = tuple(map(int, action_ids))
        if len(ids) != N_ACTIONS:
            raise ValueError("checkpoint evaluation requires the 9 native planar action IDs")
        decoded = actions[0].detach().cpu().numpy()
        self.last_result = {entity.id: ids[int(decoded[i])] if entity.alive else ids[0]
                            for i, entity in enumerate(state.red)}
        self.last_step = step
        return dict(self.last_result)

    def _hier_decision(self, state, absent, step, env_args):
        length = int(self.args.hier_agent["action_length"])
        if env_args.get("allocation_clock", "interval") != "event":
            return step % length == 0
        n_red, n_blue, n_targets = resolve_pad("eval")
        nearest = blue_nearest_from_table(entities_from_state(state)[0], absent, n_red, n_blue, n_targets)
        alive = (sum(entity.alive for entity in state.red), sum(entity.alive for entity in state.blue))
        if step == 0:
            self._alloc_hold = 0
            self._last_nearest = nearest
            self._last_alive = alive
            return True
        self._alloc_hold += 1
        death = alive != self._last_alive
        geometry = self._last_nearest is not None and not np.array_equal(nearest, self._last_nearest)
        decide = death or (geometry and self._alloc_hold >= 2) or self._alloc_hold >= int(env_args.get("max_alloc_hold", 10))
        if decide:
            self._alloc_hold = 0
            self._last_nearest = nearest
            self._last_alive = alive
        return decide

    def episode_q_statistics(self):
        return {"q_tot_mean": float(np.mean(self.q_tot)) if self.q_tot else None,
                "q_tot_std": float(np.std(self.q_tot)) if self.q_tot else None,
                "q_i_mean": float(np.mean(self.q_i)) if self.q_i else None}
