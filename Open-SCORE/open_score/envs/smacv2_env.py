"""Native SMACv2 behind the existing entity-environment interface.

The actor table contains only each observer's native observation. The separate
``entities`` table contains native centralized state for the mixer. Neither
table has a feature width that depends on the number of units.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import random

import numpy as np


SMAC_COMMIT = "577ab5a2cff2391f8df582da5731ea9cd6adf3c6"
SMAC_PROTOCOL = "smacv2-protoss-mixed-v1"
TRAIN_COUNTS = (4, 6, 8, 10)
ACTOR_ENTITY_SHAPE = 32
STATE_ENTITY_SHAPE = 18


class SMACEnvironmentError(RuntimeError):
    """An invalid simulator attempt, never a scored loss or replay episode."""


def smac_counts(config):
    if isinstance(config, dict):
        if int(config.get("K", 0)) != 0:
            raise ValueError("SMAC has no HAD protected-target count; K must be 0")
        allies = config.get("N_R", config.get("n_agents", config.get("n_units")))
        enemies = config.get("N_B", config.get("n_enemies"))
    else:
        values = tuple(config)
        if len(values) not in (2, 3) or (len(values) == 3 and int(values[2]) != 0):
            raise ValueError("SMAC configuration must be (allies, enemies[, 0])")
        allies, enemies = values[:2]
    allies, enemies = int(allies), int(enemies)
    if allies < 1 or enemies < 1:
        raise ValueError("SMAC rosters must be positive")
    return allies, enemies


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _seed_distributions(wrapper, seed):
    """Seed every nested official generator, including reflect subgenerators."""
    from smacv2.env.starcraft2.distributions import Distribution

    sequence = np.random.SeedSequence([int(seed), 7319])
    seen = set()

    def visit(distribution):
        if id(distribution) in seen:
            return
        seen.add(id(distribution))
        if hasattr(distribution, "rng"):
            distribution.rng = np.random.default_rng(sequence.spawn(1)[0])
        for key, child in sorted(vars(distribution).items()):
            if isinstance(child, Distribution):
                visit(child)

    for key in sorted(wrapper.env_key_to_distribution_map):
        visit(wrapper.env_key_to_distribution_map[key])


def generate_scene(config, seed):
    """Register one official scenario without launching a StarCraft process.

    The caller writes the shared manifest once, before creating workers.
    Global generators are restored so scene registration cannot perturb
    training, evaluation policies, or the caller's subsequent random draws.
    """
    from types import SimpleNamespace
    from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

    counts = smac_counts(config)
    scene_seed = int(seed)
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    adapter = MixedScaleSMACAdapter(seed=scene_seed, pad=counts, config=counts)
    try:
        capability = copy.deepcopy(adapter._native_options["capability_config"])
        capability.update(n_units=counts[0], n_enemies=counts[1])
        # Reuse the official parser directly. Constructing StarCraft2Env for
        # every manifest row would retain thousands of atexit callbacks.
        native = SimpleNamespace(distribution_config=capability,
                                 env_key_to_distribution_map={})
        StarCraftCapabilityEnvWrapper._parse_distribution_config(native)
        random.seed(scene_seed)
        np.random.seed(scene_seed % 2**32)
        _seed_distributions(native, scene_seed)
        selected = {}
        for distribution in native.env_key_to_distribution_map.values():
            selected.update(distribution.generate())
        return dict(config={"N_R": counts[0], "N_B": counts[1], "K": 0},
                    scenario_seed=scene_seed, engine_seed=scene_seed,
                    reset_config=_plain(selected))
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)


def generate_registered_scenes(output):
    """Create once in the parent; subsequent evaluation workers only read."""
    configs = ((5, 5), (10, 10), (12, 12), (15, 15), (20, 20),
               (10, 11), (10, 12), (10, 15))
    expected = {f"{nr}v{nb}.s{seed}" for nr, nb in configs
                for seed in range(110000, 110300)}
    path = Path(output) / "smacv2" / "scenes" / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if (manifest.get("protocol") != SMAC_PROTOCOL or
                manifest.get("smacv2_commit") != SMAC_COMMIT or
                set(manifest.get("scenes", {})) != expected):
            raise ValueError(f"SMAC scene manifest does not match the registered protocol: {path}")
        for key, scene in manifest["scenes"].items():
            nr, nb = smac_counts(scene["config"])
            if (key != f"{nr}v{nb}.s{int(scene['scenario_seed'])}" or
                    scene.get("engine_seed") != scene["scenario_seed"] or
                    not scene.get("reset_config")):
                raise ValueError(f"Invalid registered SMAC scene: {key}")
        return manifest
    scenes = {f"{nr}v{nb}.s{seed}": generate_scene((nr, nb), seed)
              for nr, nb in configs for seed in range(110000, 110300)}
    manifest = dict(protocol=SMAC_PROTOCOL, smacv2_commit=SMAC_COMMIT,
                    sc2_version="4.10.0", scene_count=len(scenes), scenes=scenes)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)
    return manifest


class MixedScaleSMACAdapter:
    """Four native training environments, with fixed-capacity replay tensors."""

    def __init__(self, seed=0, pad="train", scale=None, config=None,
                 game_version="4.10.0", sc2path=None, max_retries=3,
                 env_options=None, entity_scheme=True, episode_limit=200,
                 max_steps=None, **options):
        if not entity_scheme:
            raise ValueError("SMAC adaptation requires the entity scheme")
        if int(episode_limit) != 200 or max_steps not in (None, 200):
            raise ValueError("The registered Protoss protocol has a 200-step horizon")
        if sc2path is not None:
            os.environ["SC2PATH"] = str(Path(sc2path).expanduser().resolve())
        else:
            os.environ.setdefault("SC2PATH", os.environ.get("SC2PATH") or str(
                Path(os.environ.get("REGIR_ROOT") or Path(__file__).resolve().parents[3]) / "envs/StarCraftII"))
        self.seed = int(seed)
        self.scale_rng = np.random.default_rng(np.random.SeedSequence([self.seed, 1709]))
        self.episode_rng = np.random.default_rng(np.random.SeedSequence([self.seed, 2719]))
        fixed = config if config is not None else scale
        self.fixed_scale = None if fixed is None else smac_counts(fixed)
        if pad in (None, "train"):
            self.n_agents, self.n_enemies = 10, 10
        elif pad == "eval":
            self.n_agents, self.n_enemies = self.fixed_scale or (20, 20)
        else:
            self.n_agents, self.n_enemies = smac_counts(pad)
        self.n_entities = self.n_agents + self.n_enemies
        self.n_actions = 6 + self.n_enemies
        self.episode_limit = 200
        self.max_retries = int(max_retries)
        if self.max_retries < 1:
            raise ValueError("max_retries must be positive")
        self.game_version = str(game_version)
        if self.game_version != "4.10.0":
            raise ValueError("This experiment pins SC2 4.10.0 / Base75689")
        self._native_options = self._official_options(env_options, options)
        self._pool = {}
        self.env = None
        self._episode_seed = None
        self._engine_seed = None
        self._counts = None
        self._return = 0.0
        self._steps = 0
        self._done = False
        self._timeout = False
        self._won = False
        self._reset_config = None
        self._trajectory = []
        self._retain = False

    def _official_options(self, overrides, options):
        import smacv2
        import yaml

        source = Path(smacv2.__file__).parent / "examples/configs/sc2_gen_protoss.yaml"
        if not source.exists():
            raise FileNotFoundError("Install the pinned SMACv2 source editable; official Protoss YAML is required")
        result = copy.deepcopy(yaml.safe_load(source.read_text())["env_args"])
        for update in (overrides or {}, options):
            unknown = set(update) - set(result)
            if unknown:
                raise ValueError(f"Unknown native SMAC options: {sorted(unknown)}")
            result.update(copy.deepcopy(update))
        # Shared protocol across all four algorithms. Time memory remains.
        result.update(game_version=self.game_version, obs_last_action=False,
                      state_last_action=False, obs_instead_of_state=False,
                      obs_all_health=True, obs_own_health=True, obs_own_pos=True,
                      obs_timestep_number=False, state_timestep_number=False,
                      obs_pathing_grid=False, obs_terrain_height=False,
                      conic_fov=False, continuing_episode=False)
        if result["map_name"] != "10gen_protoss":
            raise ValueError("This adapter implements the pinned native Protoss schema")
        return result

    def _new_native(self, counts, seed):
        from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

        options = copy.deepcopy(self._native_options)
        options["seed"] = int(seed)
        options["capability_config"]["n_units"] = int(counts[0])
        options["capability_config"]["n_enemies"] = int(counts[1])
        return StarCraftCapabilityEnvWrapper(**options)

    def reset(self, seed=None, config=None, evaluate=False, retain_trajectory=False,
              test=False, reset_config=None, engine_seed=None, **unused):
        episode_seed = int(self.episode_rng.integers(0, 2**31 - 1)) if seed is None else int(seed)
        counts = smac_counts(config) if config is not None else self.fixed_scale
        if counts is None:
            n = int(self.scale_rng.choice(TRAIN_COUNTS))
            counts = n, n
        if counts[0] > self.n_agents or counts[1] > self.n_enemies:
            raise ValueError(f"SMAC roster {counts} exceeds adapter capacity {(self.n_agents, self.n_enemies)}")
        self._episode_seed, self._counts = episode_seed, counts
        self._return, self._steps = 0.0, 0
        self._done = self._timeout = self._won = False
        self._trajectory, self._retain = [], bool(retain_trajectory)
        # Evaluation creates a fresh SC2 game, so the native engine seed and
        # generated scenario depend on the scene, never evaluation order.
        recreate = bool(evaluate or test)
        if engine_seed is not None and not recreate:
            raise ValueError("An explicit engine seed requires a fresh evaluation game")
        game_seed = episode_seed if recreate else int(np.random.SeedSequence(
            [self.seed, counts[0], counts[1], 4721]).generate_state(1)[0] % (2**31 - 1))
        if engine_seed is not None:
            game_seed = int(engine_seed)
        self._engine_seed = game_seed
        native = self._pool.get(counts)
        if native is not None and recreate:
            native.close()
            del self._pool[counts]
            native = None
        if native is None:
            native = self._new_native(counts, game_seed)
            self._pool[counts] = native
        random.seed(episode_seed)
        np.random.seed(episode_seed % 2**32)
        _seed_distributions(native, episode_seed)
        if reset_config is None:
            selected = {}
            for distribution in native.env_key_to_distribution_map.values():
                selected.update(distribution.generate())
        else:
            selected = copy.deepcopy(reset_config)
            for key in ("ally_start_positions", "enemy_start_positions", "attack", "health", "enemy_mask"):
                if key in selected and "item" in selected[key]:
                    selected[key]["item"] = np.asarray(selected[key]["item"])
        self._reset_config = copy.deepcopy(selected)
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        errors = []
        for attempt in range(self.max_retries):
            try:
                if attempt:
                    native = self._new_native(counts, game_seed)
                    self._pool[counts] = native
                random.setstate(python_rng)
                np.random.set_state(numpy_rng)
                self.env = native
                restarts = int(native.force_restarts)
                # Bypass wrapper.reset's recursive re-sampling on an error.
                observations, _ = native.env.reset(copy.deepcopy(selected))
                if int(native.force_restarts) != restarts:
                    raise SMACEnvironmentError("SC2 restarted while resetting the scene")
                self._refresh(observations)
                if self._retain:
                    self._capture(None)
                return self.get_entities(), self.get_masks()
            except Exception as error:
                errors.append(f"{type(error).__name__}: {error}")
                try:
                    native.close()
                finally:
                    self._pool.pop(counts, None)
        self.env = None
        raise SMACEnvironmentError(f"SMAC reset failed for {counts} seed={episode_seed}: " + " | ".join(errors))

    def _refresh(self, observations=None):
        native = self.env
        nr, nb = self._counts
        sizes = (native.get_obs_move_feats_size(), native.get_obs_own_feats_size(),
                 native.get_obs_ally_feats_size()[1], native.get_obs_enemy_feats_size()[1],
                 native.get_ally_num_attributes(), native.get_enemy_num_attributes())
        if tuple(sizes) != (4, 7, 9, 9, 8, 7):
            raise ValueError(f"Pinned Protoss feature contract changed: {sizes}")
        observations = np.asarray(native.get_obs() if observations is None else observations,
                                  dtype=np.float32)
        central = native.env.get_state_dict()
        allies, enemies = np.asarray(central["allies"]), np.asarray(central["enemies"])
        self.entities = np.zeros((self.n_entities, STATE_ENTITY_SHAPE), dtype=np.float32)
        self.observer_entities = np.zeros((self.n_agents, self.n_entities, ACTOR_ENTITY_SHAPE), dtype=np.float32)
        self.entity_mask = np.ones(self.n_entities, dtype=np.uint8)
        self.obs_mask = np.ones((self.n_entities, self.n_entities), dtype=np.uint8)
        alive_allies = allies[:, 0] > 0
        alive_enemies = enemies[:, 0] > 0
        for index in np.flatnonzero(alive_allies):
            self.entities[index, 0] = 1
            self.entities[index, 3:11] = allies[index]
            self.entity_mask[index] = 0
        for index in np.flatnonzero(alive_enemies):
            slot = self.n_agents + index
            self.entities[slot, 1] = 1
            self.entities[slot, 11:18] = enemies[index]
            self.entity_mask[slot] = 0
        for observer in np.flatnonzero(alive_allies):
            row = observations[observer]
            move = row[:4]
            enemy_end = 4 + nb * 9
            enemy_obs = row[4:enemy_end].reshape(nb, 9)
            ally_end = enemy_end + (nr - 1) * 9
            ally_obs = row[enemy_end:ally_end].reshape(nr - 1, 9)
            own = row[ally_end:]
            if own.shape != (7,):
                raise ValueError("Native own observation is not seven features")
            # Actor roles are self, ally, enemy (central roles are ally,
            # enemy, reserved because the mixer has no distinguished self).
            self.observer_entities[observer, observer, 0] = 1
            self.observer_entities[observer, observer, 3:14] = np.concatenate((move, own))
            self.obs_mask[observer, observer] = 0
            other_ids = [index for index in range(nr) if index != observer]
            for index, values in zip(other_ids, ally_obs):
                if values[0] > 0:  # native ally visibility, not attack availability
                    self.observer_entities[observer, index, 1] = 1
                    self.observer_entities[observer, index, 14:23] = values
                    self.obs_mask[observer, index] = 0
            for index, values in enumerate(enemy_obs):
                # Native obs_all_health gives positive health for every live,
                # visible enemy, including those outside attack range.
                if values[4] > 0:
                    slot = self.n_agents + index
                    self.observer_entities[observer, slot, 2] = 1
                    self.observer_entities[observer, slot, 23:32] = values
                    self.obs_mask[observer, slot] = 0
        self.avail_actions = np.zeros((self.n_agents, self.n_actions), dtype=np.int32)
        self.avail_actions[:, 0] = 1
        available = np.asarray(native.get_avail_actions(), dtype=np.int32)
        if available.shape != (nr, 6 + nb):
            raise ValueError(f"Native action shape changed: {available.shape}")
        self.avail_actions[:nr, :6 + nb] = available

    def step(self, actions):
        if self.env is None or self._done:
            raise RuntimeError("SMAC step requires an active episode")
        chosen = np.asarray(actions, dtype=np.int64).reshape(-1)
        nr, nb = self._counts
        if len(chosen) not in (nr, self.n_agents):
            raise ValueError("SMAC actions must match the real or padded ally roster")
        chosen = chosen[:nr]
        if np.any(chosen < 0) or np.any(chosen >= 6 + nb):
            raise ValueError("SMAC action selects an absent enemy slot")
        if np.any(self.avail_actions[np.arange(nr), chosen] == 0):
            raise ValueError("SMAC received an unavailable action")
        try:
            restarts = int(self.env.force_restarts)
            reward, done, raw = self.env.step(chosen.tolist())
            if int(self.env.force_restarts) != restarts or "battle_won" not in raw:
                raise SMACEnvironmentError("SC2 restarted during the episode")
            if not np.isfinite(reward):
                raise SMACEnvironmentError("SC2 returned a non-finite reward")
            self._steps += 1
            self._return += float(reward)
            natural_end = int(raw.get("dead_allies", 0)) == nr or int(raw.get("dead_enemies", 0)) == nb
            self._timeout = self._steps >= self.episode_limit and not natural_end
            self._done = bool(done or self._timeout)
            self._won = bool(raw["battle_won"])
            self._refresh()
            if self._retain:
                self._capture(chosen)
            info = dict(raw, terminated=self._done, terminal_for_learning=self._done,
                        terminated_naturally=bool(self._done and not self._timeout),
                        truncated=False, episode_limit=False, timeout=self._timeout,
                        bootstrap_mask=float(not self._done), folded_steps=0)
            return float(reward), self._done, info
        except Exception as error:
            self._done = True
            if isinstance(error, SMACEnvironmentError):
                raise
            raise SMACEnvironmentError(f"SMAC step failed at seed={self._episode_seed} t={self._steps}: {error}") from error

    def _capture(self, actions):
        self._trajectory.append(dict(step=self._steps, entities=self.entities.tolist(),
                                     entity_mask=self.entity_mask.tolist(),
                                     actions=None if actions is None else actions.tolist()))

    def get_entities(self):
        return self.entities.copy()

    def get_observer_entities(self):
        return self.observer_entities.copy()

    def get_masks(self):
        return {"obs_mask": self.obs_mask.copy(), "entity_mask": self.entity_mask.copy()}

    def get_avail_actions(self):
        return self.avail_actions.copy()

    def get_initial_agent_mask(self):
        mask = np.ones(self.n_agents, dtype=np.uint8)
        mask[:self._counts[0]] = 0
        return mask

    def get_state(self):
        return self.entities.reshape(-1).copy()

    def get_env_info(self, args=None):
        return dict(n_agents=self.n_agents, n_enemies=self.n_enemies,
                    n_entities=self.n_entities, n_actions=self.n_actions,
                    entity_shape=STATE_ENTITY_SHAPE, actor_entity_shape=ACTOR_ENTITY_SHAPE,
                    observer_entity_shape=ACTOR_ENTITY_SHAPE,
                    state_shape=self.n_entities * STATE_ENTITY_SHAPE,
                    obs_shape=self.n_entities * ACTOR_ENTITY_SHAPE,
                    episode_limit=self.episode_limit, n_tasks=0,
                    feature_layout="smacv2", action_head="common6_enemy",
                    gt_mask_avail=False, environment_protocol=SMAC_PROTOCOL,
                    smacv2_commit=SMAC_COMMIT)

    def episode_summary(self):
        nr, nb = self._counts
        return dict(environment="smacv2", config={"N_R": nr, "N_B": nb, "K": 0},
                    episode_seed=self._episode_seed, scenario_seed=self._episode_seed,
                    engine_seed=self._engine_seed, episode_return=self._return,
                    **{"return": self._return}, ep_len=self._steps,
                    battle_won=self._won, timeout=self._timeout,
                    terminated_naturally=bool(self._done and not self._timeout),
                    reset_config=_plain(self._reset_config))

    def get_trajectory(self):
        return copy.deepcopy(self._trajectory)

    def get_rng_state(self):
        # Saves future scene generation at episode boundaries. SC2 processes
        # are reconstructed; their hidden internal state is not serializable.
        return dict(scale=copy.deepcopy(self.scale_rng.bit_generator.state),
                    episode=copy.deepcopy(self.episode_rng.bit_generator.state),
                    python=random.getstate(), numpy=np.random.get_state())

    def set_rng_state(self, state):
        self.scale_rng.bit_generator.state = copy.deepcopy(state["scale"])
        self.episode_rng.bit_generator.state = copy.deepcopy(state["episode"])
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])

    def close(self):
        for native in list(self._pool.values()):
            native.close()
        self._pool.clear()
        self.env = None
