"""Auditable attack--defence extension built *around* upstream SMAClite.

This module deliberately lives outside the :mod:`smaclite` namespace.  The
official ``smaclite/*-v0`` Gymnasium registrations and stock JSON scenarios are
never changed.  ``SMACliteADEnv`` reuses the pinned upstream movement, combat,
collision and unit implementations, while adding a separately named two-sided
control protocol and an explicit protected asset.

Protocol summary
----------------
``Red`` is the attacking team and ``Blue`` is the defending team.  Both teams
submit primitive SMAClite actions (noop, stop, four moves, direct target).  In
``asset`` mode Red wins by destroying the protected asset; Blue wins by
destroying all Red units or keeping the asset alive to the horizon.  Padded
entity/action masks keep tensor shapes fixed while the realised team sizes
change between environment instances.

This is an Open-SCORE extension, not an official SMAClite benchmark scenario.
Results from it must therefore be labelled ``SMAClite-AD`` rather than
``SMAClite``.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from open_score.contracts import GlobalState, TeamObservation

try:
    import gymnasium as gym
    import smaclite
    from smaclite.env.maps.map import Group, MapInfo, get_standard_map
    from smaclite.env.smaclite import (
        MOVE_AMOUNT,
        STEP_MUL,
        SMACliteEnv,
    )
    from smaclite.env.units.combat_type import CombatType
    from smaclite.env.units.unit_command import (
        AttackUnitCommand,
        MoveCommand,
        NoopCommand,
        StopCommand,
    )
    from smaclite.env.units.unit_type import StandardUnit
    from smaclite.env.util.direction import Direction
    from smaclite.env.util.faction import Faction
except ImportError as exc:  # pragma: no cover - exercised by optional-dep users
    smaclite = None
    _SMACLITE_IMPORT_ERROR: Optional[ImportError] = exc
else:
    _SMACLITE_IMPORT_ERROR = None


PROTOCOL_ID = "OpenSCORE/SMACliteAD-Asset-v0"
UPSTREAM_REPOSITORY = "https://github.com/uoe-agents/smaclite"
UPSTREAM_TAG = "v2.0.0"
UPSTREAM_COMMIT = "e936d9dbf4f85551d6fd445a6c1150867bc79c55"


def _require_smaclite() -> None:
    if _SMACLITE_IMPORT_ERROR is not None:
        raise ImportError(
            "SMAClite-AD needs the pinned optional SMAClite dependency. "
            "Run scripts/install_smaclite.ps1 with the torch310 interpreter."
        ) from _SMACLITE_IMPORT_ERROR


@dataclass(frozen=True)
class SMACliteADConfig:
    """Configuration whose maxima define the cross-scale tensor contract."""

    red_agents: int = 3
    blue_agents: int = 2
    max_red_agents: int = 8
    max_blue_agents: int = 8
    episode_limit: int = 150
    objective: str = "asset"
    reward_mode: str = "strict_potential"
    discount_gamma: float = 0.99
    shaping_scale: float = 0.10
    approach_weight: float = 0.0
    spawn_jitter: float = 0.0
    map_width: int = 32
    map_height: int = 32

    def validate(self) -> None:
        if not 1 <= self.red_agents <= self.max_red_agents:
            raise ValueError("red_agents must be in [1, max_red_agents]")
        if not 1 <= self.blue_agents <= self.max_blue_agents:
            raise ValueError("blue_agents must be in [1, max_blue_agents]")
        if self.max_red_agents < 1 or self.max_blue_agents < 1:
            raise ValueError("team-size maxima must be positive")
        if self.episode_limit < 1:
            raise ValueError("episode_limit must be positive")
        if self.objective not in {"asset", "elimination"}:
            raise ValueError("objective must be 'asset' or 'elimination'")
        if self.reward_mode not in {
            "terminal_only",
            "strict_potential",
            "heuristic_delta",
        }:
            raise ValueError(
                "reward_mode must be terminal_only, strict_potential or heuristic_delta"
            )
        if not 0.0 < self.discount_gamma <= 1.0:
            raise ValueError("discount_gamma must lie in (0, 1]")
        if self.shaping_scale < 0 or self.approach_weight < 0:
            raise ValueError("shaping_scale and approach_weight must be non-negative")
        if not 0.0 <= self.spawn_jitter <= 3.0:
            raise ValueError("spawn_jitter must be in [0, 3] map units")


def make_smaclite_ad_map(config: SMACliteADConfig) -> "MapInfo":
    """Create an in-memory map without writing to the stock scenario folder."""

    _require_smaclite()
    config.validate()
    # Reuse the official SIMPLE terrain object.  No upstream JSON is copied or
    # edited, which keeps the stock scenario fingerprint stable.
    terrain = get_standard_map("3s5z").terrain
    groups = [
        Group(7, config.map_height // 2, Faction.ALLY, [(StandardUnit.MARINE, config.red_agents)]),
        Group(
            config.map_width - 9,
            config.map_height // 2,
            Faction.ENEMY,
            [(StandardUnit.MARINE, config.blue_agents)],
        ),
    ]
    enemy_count = config.blue_agents
    unit_types = {StandardUnit.MARINE: 0}
    if config.objective == "asset":
        groups.append(
            Group(
                config.map_width - 5,
                config.map_height // 2,
                Faction.ENEMY,
                [(StandardUnit.SPINE_CRAWLER, 1)],
            )
        )
        enemy_count += 1
        unit_types[StandardUnit.SPINE_CRAWLER] = 1
    return MapInfo(
        name=(
            f"openscore_ad_{config.objective}_"
            f"r{config.red_agents}_b{config.blue_agents}"
        ),
        num_allied_units=config.red_agents,
        num_enemy_units=enemy_count,
        groups=groups,
        attack_point=(7, config.map_height // 2),
        terrain=terrain,
        ally_has_shields=False,
        enemy_has_shields=False,
        width=config.map_width,
        height=config.map_height,
        num_unit_types=len(unit_types) if len(unit_types) > 1 else 0,
        unit_type_ids=unit_types,
    )


class SMACliteStockAdapter:
    """Non-mutating, lossless entity parser for an official stock scenario.

    Enemy blocks are keyed by upstream ``id_in_faction``; ally blocks are
    expanded back from upstream's self-omitting compact order.  Movement and
    own-unit fields remain in ``self_obs``.  Thus no environment, reward or
    action semantics change, while target action slots have an explicit entity
    row suitable for the shared permutation-equivariant scorer.
    """

    TASK_DIM = 1

    def __init__(
        self,
        map_name: str = "3s5z",
        *,
        episode_limit: int = 150,
        seed: Optional[int] = None,
        use_cpp_rvo2: bool = False,
    ) -> None:
        _require_smaclite()
        if episode_limit < 1:
            raise ValueError("episode_limit must be positive")
        self.environment_id = f"smaclite/{map_name}-v0"
        self.env = gym.make(
            self.environment_id, seed=seed, use_cpp_rvo2=use_cpp_rvo2
        )
        self.unwrapped = self.env.unwrapped
        self.n_agents = int(self.unwrapped.n_agents)
        self.episode_limit = int(episode_limit)
        self.episode_steps = 0
        self._enemy_feature_dim = int(self.unwrapped.enemy_feat_size)
        self._ally_feature_dim = int(self.unwrapped.ally_feat_size)
        self._own_feature_dim = int(
            1
            + self.unwrapped.map_info.ally_has_shields
            + self.unwrapped.map_info.num_unit_types
        )
        self._entity_payload_dim = max(
            self._enemy_feature_dim,
            self._ally_feature_dim,
            self._own_feature_dim,
        )
        # raw block plus [self, same-team, enemy, asset]
        self.ENTITY_DIM = self._entity_payload_dim + 4
        self.SELF_DIM = 4 + self._own_feature_dim
        # Add episode phase to make the feed-forward centralized critic Markov.
        # The official state vector itself remains lossless in the prefix.
        self.STATE_ENTITY_DIM = int(self.unwrapped.state_size) + 1
        self.ACTION_DIM = int(self.unwrapped.n_actions)
        self._episode_done = True

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
        observations, info = self.env.reset(seed=seed, options=options)
        self.episode_steps = 0
        self._episode_done = False
        return self._pack(observations), self._info(info)

    def _pack(self, observations: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
        alive = np.asarray(
            [index in self.unwrapped.agents for index in range(self.n_agents)],
            dtype=bool,
        )
        flat = np.asarray(observations, dtype=np.float32)
        entity_count = self.n_agents + int(self.unwrapped.n_enemies)
        entity_obs = np.zeros(
            (self.n_agents, entity_count, self.ENTITY_DIM), dtype=np.float32
        )
        entity_mask = np.zeros((self.n_agents, entity_count), dtype=bool)
        self_obs = np.zeros((self.n_agents, self.SELF_DIM), dtype=np.float32)
        enemy_offset = 4
        ally_offset = enemy_offset + self.unwrapped.n_enemies * self._enemy_feature_dim
        own_offset = ally_offset + (self.n_agents - 1) * self._ally_feature_dim
        action_entity_index = np.full(
            (self.n_agents, self.ACTION_DIM), -1, dtype=np.int64
        )
        action_target_type = np.zeros(
            (self.n_agents, self.ACTION_DIM), dtype=np.int64
        )
        for observer_index in range(self.n_agents):
            if not alive[observer_index]:
                continue
            row = flat[observer_index]
            own_block = row[own_offset : own_offset + self._own_feature_dim]
            self_obs[observer_index, :4] = row[:4]
            self_obs[observer_index, 4:] = own_block
            entity_obs[observer_index, observer_index, : self._own_feature_dim] = own_block
            entity_obs[observer_index, observer_index, -4:] = (1.0, 1.0, 0.0, 0.0)
            entity_mask[observer_index, observer_index] = True
            for ally_index in range(self.n_agents):
                if ally_index == observer_index:
                    continue
                compact_index = ally_index - int(ally_index > observer_index)
                base = ally_offset + compact_index * self._ally_feature_dim
                block = row[base : base + self._ally_feature_dim]
                entity_obs[observer_index, ally_index, : self._ally_feature_dim] = block
                entity_obs[observer_index, ally_index, -4:] = (0.0, 1.0, 0.0, 0.0)
                entity_mask[observer_index, ally_index] = bool(block[0] > 0.0)
            for enemy_index in range(self.unwrapped.n_enemies):
                base = enemy_offset + enemy_index * self._enemy_feature_dim
                block = row[base : base + self._enemy_feature_dim]
                entity_index = self.n_agents + enemy_index
                entity_obs[observer_index, entity_index, : self._enemy_feature_dim] = block
                entity_obs[observer_index, entity_index, -4:] = (0.0, 0.0, 1.0, 0.0)
                entity_mask[observer_index, entity_index] = bool(np.any(block != 0.0))
            unit = self.unwrapped.agents[observer_index]
            is_healer = unit.combat_type == CombatType.HEALING
            target_count = self.n_agents if is_healer else self.unwrapped.n_enemies
            for target_index in range(target_count):
                action_index = 6 + target_index
                if action_index >= self.ACTION_DIM:
                    break
                action_entity_index[observer_index, action_index] = (
                    target_index if is_healer else self.n_agents + target_index
                )
                action_target_type[observer_index, action_index] = 2 if is_healer else 1
        task_obs = np.full(
            (self.n_agents, 1),
            1.0 - self.episode_steps / self.episode_limit,
            dtype=np.float32,
        )
        task_obs[~alive] = 0.0
        available = np.asarray(self.unwrapped.get_avail_actions(), dtype=bool)
        remaining_horizon = max(
            0.0, 1.0 - self.episode_steps / self.episode_limit
        )
        state = np.concatenate(
            [
                np.asarray(self.unwrapped.get_state(), dtype=np.float32),
                np.asarray([remaining_horizon], dtype=np.float32),
            ]
        )[None, :]
        return {
            "entity_obs": entity_obs,
            "entity_mask": entity_mask,
            "self_obs": self_obs,
            "task_obs": task_obs,
            "agent_mask": alive,
            "avail_actions": available,
            "action_entity_index": action_entity_index,
            "action_target_type": action_target_type,
            "state_entities": state,
            "state_mask": np.ones(1, dtype=bool),
        }

    def step(
        self, actions: Sequence[int]
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, object]]:
        if self._episode_done:
            raise RuntimeError("reset() must be called before step() or after episode end")
        values = np.asarray(actions, dtype=np.int64)
        if values.shape != (self.n_agents,):
            raise ValueError(f"stock scenario requires {self.n_agents} actions")
        observations, reward, terminated, upstream_truncated, info = self.env.step(
            [int(value) for value in values]
        )
        self.episode_steps += 1
        horizon_truncated = self.episode_steps >= self.episode_limit and not terminated
        truncated = bool(upstream_truncated or horizon_truncated)
        self._episode_done = bool(terminated or truncated)
        return (
            self._pack(observations),
            float(reward),
            bool(terminated),
            truncated,
            self._info(info),
        )

    def _info(self, upstream_info: Mapping[str, object]) -> Dict[str, object]:
        return {
            **dict(upstream_info),
            "environment_id": self.environment_id,
            "episode_steps": self.episode_steps,
            "upstream_commit": UPSTREAM_COMMIT,
            "adapter": "lossless_stock_blocks_to_explicit_target_entities_v2",
        }

    def close(self) -> None:
        self.env.close()


class SMACliteADEnv(SMACliteEnv if _SMACLITE_IMPORT_ERROR is None else object):
    """Two-sided, variable-roster asset-defence environment.

    The environment intentionally uses a Parallel-style multi-team API rather
    than registering under upstream's single-team Gymnasium IDs::

        obs, info = env.reset(seed=0)
        obs, rewards, terminated, truncated, info = env.step(
            {"Red": red_actions, "Blue": blue_actions}
        )

    Action arrays may contain either the realised roster length or the padded
    maximum length.  Padding slots must use noop (0).
    """

    ENTITY_DIM = 13
    SELF_DIM = 12
    TASK_DIM = 9
    STATE_ENTITY_DIM = 12

    def __init__(
        self,
        red_agents: int = 3,
        blue_agents: int = 2,
        *,
        max_red_agents: int = 8,
        max_blue_agents: int = 8,
        episode_limit: int = 150,
        objective: str = "asset",
        reward_mode: str = "strict_potential",
        discount_gamma: float = 0.99,
        shaping_scale: float = 0.10,
        approach_weight: float = 0.0,
        spawn_jitter: float = 0.0,
        seed: Optional[int] = None,
        use_cpp_rvo2: bool = False,
    ) -> None:
        _require_smaclite()
        self.config = SMACliteADConfig(
            red_agents=red_agents,
            blue_agents=blue_agents,
            max_red_agents=max_red_agents,
            max_blue_agents=max_blue_agents,
            episode_limit=episode_limit,
            objective=objective,
            reward_mode=reward_mode,
            discount_gamma=discount_gamma,
            shaping_scale=shaping_scale,
            approach_weight=approach_weight,
            spawn_jitter=spawn_jitter,
        )
        self.config.validate()
        self.protocol_id = PROTOCOL_ID
        super().__init__(
            map_info=make_smaclite_ad_map(self.config),
            seed=seed,
            use_cpp_rvo2=use_cpp_rvo2,
        )
        self.action_dim = 6 + max(
            self.config.max_red_agents,
            self.config.max_blue_agents + (self.config.objective == "asset"),
        )
        # Instance-level aliases match the stock/HAD dimension-discovery API.
        self.ACTION_DIM = self.action_dim
        self.max_entities = (
            self.config.max_red_agents
            + self.config.max_blue_agents
            + (self.config.objective == "asset")
        )
        self.MAX_ENTITIES = self.max_entities
        self.asset_action_id = (
            6 + self.config.max_blue_agents
            if self.config.objective == "asset"
            else None
        )
        self._base_group_centers = tuple(
            (float(group.x), float(group.y)) for group in self.map_info.groups
        )
        self._layout_metadata: Dict[str, object] = {}
        self._randomization_config_hash = self._hash_json(
            {
                "protocol_id": PROTOCOL_ID,
                "upstream_commit": UPSTREAM_COMMIT,
                "spawn_jitter": self.config.spawn_jitter,
                "generator": "numpy.default_rng(seed xor 0x5A17D0A1)",
                "rejection_limit": 64,
                "constraints": ["map_bounds", "normal_terrain", "no_overlap"],
            }
        )
        self.episode_steps = 0
        self._episode_done = True
        self._red_slots: list[object] = []
        self._blue_slots: list[object] = []
        self._asset: Optional[object] = None
        self._initial_red_hp = 1.0
        self._initial_blue_hp = 1.0
        self._initial_asset_hp = 1.0

    @property
    def team_sizes(self) -> Dict[str, int]:
        return {
            "Red": self.config.red_agents,
            "Blue": self.config.blue_agents,
        }

    @property
    def max_team_sizes(self) -> Dict[str, int]:
        return {
            "Red": self.config.max_red_agents,
            "Blue": self.config.max_blue_agents,
        }

    @property
    def asset_alive(self) -> bool:
        return self._asset is not None and self._asset.hp > 0

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, object]]:
        del options
        layout_seed = (
            int(seed)
            if seed is not None
            else int(np.random.default_rng().integers(0, 2**31 - 1))
        )
        layout_rng = np.random.default_rng(layout_seed ^ 0x5A17D0A1)
        validation: Dict[str, object] = {}
        for attempt in range(64):
            for group, (base_x, base_y) in zip(
                self.map_info.groups, self._base_group_centers
            ):
                if self.config.spawn_jitter > 0.0:
                    offset = layout_rng.uniform(
                        -self.config.spawn_jitter,
                        self.config.spawn_jitter,
                        size=2,
                    )
                else:
                    offset = np.zeros(2, dtype=np.float64)
                group.x = float(base_x + offset[0])
                group.y = float(base_y + offset[1])
            # Calling the official reset constructs the world, resets RVO2
            # and seeds the upstream NumPy generator.  Rejection changes only
            # spawn centres, never the upstream physics implementation.
            super().reset(seed=layout_seed)
            validation = self._validate_current_layout()
            if validation["valid"]:
                break
            if self.config.spawn_jitter == 0.0:
                raise RuntimeError(f"fixed SMAClite-AD layout is invalid: {validation}")
        else:
            raise RuntimeError(
                "could not sample a legal SMAClite-AD layout in 64 deterministic attempts"
            )
        self._red_slots = [self.agents[index] for index in range(self.config.red_agents)]
        self._blue_slots = [self.enemies[index] for index in range(self.config.blue_agents)]
        self._asset = (
            self.enemies[self.config.blue_agents]
            if self.config.objective == "asset"
            else None
        )
        for unit in self._red_slots + self._blue_slots:
            unit.command = StopCommand()
        if self._asset is not None:
            # The protected structure is an objective, not another learned or
            # scripted weapon.  Its upstream combat stats still determine hp,
            # radius, armour, collision and damage reception.
            self._asset.command = StopCommand()
        self._initial_red_hp = self._total_hp(self._red_slots)
        self._initial_blue_hp = self._total_hp(self._blue_slots)
        self._initial_asset_hp = self._unit_total_hp(self._asset) if self._asset else 1.0
        self.episode_steps = 0
        self._episode_done = False
        self._layout_metadata = self._make_layout_metadata(
            layout_seed, attempt, validation
        )
        info = self._info(outcome_red=0, termination_reason=None)
        return self._observe_both(), info

    @staticmethod
    def _hash_json(value: Mapping[str, object]) -> str:
        canonical = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def _validate_current_layout(self) -> Dict[str, object]:
        """Audit bounds, terrain and pairwise collision constraints."""

        from smaclite.env.terrain.terrain import TerrainType

        units = list(self.all_units.values())
        boundary_violations = []
        terrain_violations = []
        overlaps = []
        for unit in units:
            x, y = (float(value) for value in unit.pos)
            if not (
                unit.radius <= x < self.map_info.width - unit.radius
                and unit.radius <= y < self.map_info.height - unit.radius
            ):
                boundary_violations.append(int(unit.id))
            elif self.map_info.terrain[int(y)][int(x)] != TerrainType.NORMAL:
                terrain_violations.append(int(unit.id))
        for left_index, left in enumerate(units):
            for right in units[left_index + 1 :]:
                distance = float(np.linalg.norm(left.pos - right.pos))
                required = float(left.radius + right.radius)
                if distance + 1e-5 < required:
                    overlaps.append(
                        {
                            "left": int(left.id),
                            "right": int(right.id),
                            "distance": distance,
                            "required": required,
                        }
                    )
        return {
            "valid": not (boundary_violations or terrain_violations or overlaps),
            "boundary_violations": boundary_violations,
            "terrain_violations": terrain_violations,
            "overlaps": overlaps,
            "minimum_pairwise_clearance": min(
                (
                    float(np.linalg.norm(left.pos - right.pos))
                    - float(left.radius + right.radius)
                    for index, left in enumerate(units)
                    for right in units[index + 1 :]
                ),
                default=float("inf"),
            ),
        }

    def _make_layout_metadata(
        self,
        seed: int,
        attempt: int,
        validation: Mapping[str, object],
    ) -> Dict[str, object]:
        def unit_record(unit: object) -> Dict[str, object]:
            return {
                "id": int(unit.id),
                "faction_id": int(unit.id_in_faction),
                "x": round(float(unit.pos[0]), 6),
                "y": round(float(unit.pos[1]), 6),
                "radius": round(float(unit.radius), 6),
            }

        asset_id = self.config.blue_agents if self.config.objective == "asset" else None
        red_units = [
            unit_record(self.agents[index]) for index in range(self.config.red_agents)
        ]
        blue_units = [
            unit_record(self.enemies[index]) for index in range(self.config.blue_agents)
        ]
        asset = unit_record(self.enemies[asset_id]) if asset_id is not None else None
        geometry = {
            "spawn_jitter": float(self.config.spawn_jitter),
            "randomization_config_sha256": self._randomization_config_hash,
            "group_centers": [
                [round(float(group.x), 6), round(float(group.y), 6)]
                for group in self.map_info.groups
            ],
            "red": red_units,
            "blue": blue_units,
            "asset": asset,
            "validation": {
                "valid": bool(validation["valid"]),
                "minimum_pairwise_clearance": round(
                    float(validation["minimum_pairwise_clearance"]), 6
                ),
            },
        }
        layout = {
            "seed": int(seed),
            "accepted_attempt": int(attempt),
            **geometry,
            "layout_sha256": self._hash_json(geometry),
        }
        layout["sample_sha256"] = self._hash_json(layout)
        return layout

    def validate_layout(self) -> Dict[str, object]:
        """Return a fresh public validation result for tests and audit tools."""

        return self._validate_current_layout()

    @staticmethod
    def _unit_total_hp(unit: Optional[object]) -> float:
        if unit is None:
            return 0.0
        return float(max(0.0, unit.hp) + max(0.0, unit.shield))

    @classmethod
    def _total_hp(cls, units: Sequence[object]) -> float:
        return float(sum(cls._unit_total_hp(unit) for unit in units))

    @staticmethod
    def _alive(unit: object) -> bool:
        return unit.hp > 0

    def _normalise_actions(
        self, side: str, actions: Sequence[int]
    ) -> Tuple[np.ndarray, Sequence[object]]:
        slots = self._red_slots if side == "Red" else self._blue_slots
        maximum = self.max_team_sizes[side]
        values = np.asarray(actions, dtype=np.int64)
        if values.ndim != 1 or len(values) not in {len(slots), maximum}:
            raise ValueError(
                f"{side} needs {len(slots)} realised or {maximum} padded actions"
            )
        if len(values) == maximum and np.any(values[len(slots) :] != 0):
            raise ValueError(f"{side} padding slots must use noop action 0")
        return values[: len(slots)], slots

    def _targets(self, side: str) -> Sequence[object]:
        if side == "Red":
            return self._blue_slots
        return self._red_slots

    def _available_actions_for(self, unit: object, side: str) -> np.ndarray:
        available = np.zeros(self.action_dim, dtype=bool)
        if not self._alive(unit):
            available[0] = True
            return available
        # SMAClite reserves noop (0) for dead agents; living agents can stop.
        available[1] = True
        for direction in Direction:
            available[2 + direction.value] = bool(
                self._SMACliteEnv__can_move(unit, direction)
            )
        for target_index, target in enumerate(self._targets(side)):
            available[6 + target_index] = bool(
                self._SMACliteEnv__can_target(unit, target)
            )
        if side == "Red" and self._asset is not None:
            assert self.asset_action_id is not None
            available[self.asset_action_id] = bool(
                self._SMACliteEnv__can_target(unit, self._asset)
            )
        return available

    def _decode_command(self, unit: object, side: str, action: int) -> object:
        available = self._available_actions_for(unit, side)
        if action < 0 or action >= self.action_dim or not available[action]:
            raise ValueError(f"invalid {side} action {action} for unit {unit.id_in_faction}")
        if action == 1:
            return StopCommand()
        if 2 <= action <= 5:
            delta = Direction(action - 2).dx_dy * MOVE_AMOUNT
            return MoveCommand(unit.pos + delta)
        if side == "Red" and action == self.asset_action_id:
            target = self._asset
        else:
            target_index = action - 6
            targets = self._targets(side)
            if target_index < 0 or target_index >= len(targets):
                raise ValueError(f"action {action} has no {side} target slot")
            target = targets[target_index]
        assert target is not None
        if unit.combat_type == CombatType.HEALING:
            raise ValueError("the first SMAClite-AD contract supports damage units only")
        return AttackUnitCommand(target)

    def _fractions(self) -> Tuple[float, float, float]:
        red = self._total_hp(self._red_slots) / self._initial_red_hp
        blue = self._total_hp(self._blue_slots) / self._initial_blue_hp
        asset = (
            self._unit_total_hp(self._asset) / self._initial_asset_hp
            if self._asset is not None
            else 1.0
        )
        return red, blue, asset

    def _approach_fraction(self) -> float:
        """Minimum live-attacker distance to the asset, normalized by map diagonal."""

        if self._asset is None:
            return 0.0
        distances = [
            float(np.linalg.norm(unit.pos - self._asset.pos))
            for unit in self._red_slots
            if self._alive(unit)
        ]
        diagonal = float(np.hypot(self.map_info.width, self.map_info.height))
        return min(distances, default=diagonal) / diagonal

    def _shaping_potential(
        self,
        red_fraction: float,
        blue_fraction: float,
        asset_fraction: float,
        approach_fraction: float,
    ) -> float:
        """Return the Red potential used by the registered shaping contract.

        Higher potential means a state is more favourable to Red.  Raw feature
        fractions are kept explicit so the strict-potential identity can be
        audited without depending on mutable simulator objects.
        """

        return float(
            self.config.shaping_scale
            * (
                -asset_fraction
                - 0.25 * blue_fraction
                + 0.25 * red_fraction
                - self.config.approach_weight * approach_fraction
            )
        )

    def _shaping_reward(
        self,
        before: Tuple[float, float, float, float],
        after: Tuple[float, float, float, float],
        episode_done: bool,
    ) -> Tuple[float, float, float]:
        """Return shaping reward and the before/after potentials.

        ``strict_potential`` implements ``gamma * Phi(s') - Phi(s)`` and sets
        the absorbing terminal potential to zero.  Consequently the discounted
        shaped return differs from the terminal-only return by the constant
        ``-Phi(s_0)`` and cannot change the optimal policy.  ``heuristic_delta``
        preserves the pre-v4 engineering reward solely as an explicit ablation.
        """

        before_phi = self._shaping_potential(*before)
        actual_after_phi = self._shaping_potential(*after)
        if self.config.reward_mode == "terminal_only":
            return 0.0, before_phi, 0.0 if episode_done else actual_after_phi
        if self.config.reward_mode == "strict_potential":
            after_phi = 0.0 if episode_done else actual_after_phi
            reward = self.config.discount_gamma * after_phi - before_phi
            return float(reward), before_phi, after_phi
        # Legacy gamma=1 delta.  It is intentionally not described as
        # policy-invariant when learners use gamma < 1.
        reward = actual_after_phi - before_phi
        return float(reward), before_phi, actual_after_phi

    def step(
        self,
        actions: Union[Mapping[str, Sequence[int]], Sequence[int]],
        blue_actions: Optional[Sequence[int]] = None,
    ) -> Tuple[
        Dict[str, Dict[str, np.ndarray]],
        Dict[str, float],
        bool,
        bool,
        Dict[str, object],
    ]:
        if self._episode_done:
            raise RuntimeError("reset() must be called before step() or after episode end")
        if isinstance(actions, Mapping):
            if blue_actions is not None or set(actions) != {"Red", "Blue"}:
                raise ValueError("action mapping must contain exactly Red and Blue")
            red_values = actions["Red"]
            blue_values = actions["Blue"]
        else:
            if blue_actions is None:
                raise ValueError("provide blue_actions with positional red actions")
            red_values = actions
            blue_values = blue_actions
        red_values, red_slots = self._normalise_actions("Red", red_values)
        blue_values, blue_slots = self._normalise_actions("Blue", blue_values)
        before_red, before_blue, before_asset = self._fractions()
        before_approach = self._approach_fraction()
        for side, values, slots in (
            ("Red", red_values, red_slots),
            ("Blue", blue_values, blue_slots),
        ):
            for unit, action in zip(slots, values):
                if self._alive(unit):
                    unit.command = self._decode_command(unit, side, int(action))
                elif int(action) != 0:
                    raise ValueError("dead agents must use noop action 0")
        if self._asset is not None and self._alive(self._asset):
            self._asset.command = StopCommand()
        for _ in range(STEP_MUL):
            self._SMACliteEnv__world_step()
        self.episode_steps += 1

        red_alive = any(self._alive(unit) for unit in self._red_slots)
        blue_alive = any(self._alive(unit) for unit in self._blue_slots)
        terminated = False
        reason: Optional[str] = None
        outcome_red = 0
        if self.config.objective == "asset" and not self.asset_alive:
            terminated, outcome_red, reason = True, 1, "asset_destroyed"
        elif not red_alive:
            terminated, outcome_red, reason = True, -1, "attackers_eliminated"
        elif self.config.objective == "elimination" and not blue_alive:
            terminated, outcome_red, reason = True, 1, "defenders_eliminated"
        truncated = not terminated and self.episode_steps >= self.config.episode_limit
        if truncated:
            outcome_red, reason = -1, "asset_survived_horizon"
        self._episode_done = terminated or truncated

        after_red, after_blue, after_asset = self._fractions()
        after_approach = self._approach_fraction()
        shaping_red, potential_before, potential_after = self._shaping_reward(
            (before_red, before_blue, before_asset, before_approach),
            (after_red, after_blue, after_asset, after_approach),
            self._episode_done,
        )
        terminal_red = float(outcome_red) if self._episode_done else 0.0
        red_reward = shaping_red + terminal_red
        rewards = {"Red": float(red_reward), "Blue": float(-red_reward)}
        info = self._info(
            outcome_red=outcome_red,
            termination_reason=reason,
            reward_components={
                "terminal_red": terminal_red,
                "shaping_red": float(shaping_red),
                "potential_before": float(potential_before),
                "potential_after": float(potential_after),
            },
        )
        return self._observe_both(), rewards, terminated, truncated, info

    def _all_entity_slots(self) -> Sequence[Optional[object]]:
        red_padding = [None] * (self.config.max_red_agents - len(self._red_slots))
        blue_padding = [None] * (self.config.max_blue_agents - len(self._blue_slots))
        result: list[Optional[object]] = list(self._red_slots) + red_padding
        result.extend(self._blue_slots)
        result.extend(blue_padding)
        if self.config.objective == "asset":
            result.append(self._asset)
        return result

    def _relation(self, observer: object, entity: object, side: str) -> Tuple[float, ...]:
        is_red = entity in self._red_slots
        is_blue = entity in self._blue_slots
        is_asset = entity is self._asset
        same_team = is_red if side == "Red" else is_blue
        enemy = is_blue or is_asset if side == "Red" else is_red
        return (
            float(entity is observer),
            float(same_team),
            float(enemy),
            float(is_asset),
        )

    def _entity_features(self, observer: object, entity: object, side: str) -> np.ndarray:
        delta = (entity.pos - observer.pos).astype(np.float32)
        diagonal = float(np.hypot(self.map_info.width, self.map_info.height))
        hp_denominator = float(max(1.0, entity.max_hp))
        shield_denominator = float(max(1.0, entity.max_shield))
        cooldown_denominator = float(max(1.0, entity.max_cooldown))
        values = (
            float(delta[0] / self.map_info.width),
            float(delta[1] / self.map_info.height),
            float(np.linalg.norm(delta) / diagonal),
            float(entity.pos[0] / self.map_info.width),
            float(entity.pos[1] / self.map_info.height),
            float(entity.hp / hp_denominator),
            float(entity.shield / shield_denominator),
            float(entity.cooldown / cooldown_denominator),
            float(self._alive(entity)),
            *self._relation(observer, entity, side),
        )
        return np.asarray(values, dtype=np.float32)

    def _self_features(self, unit: object, side: str) -> np.ndarray:
        own = self._red_slots if side == "Red" else self._blue_slots
        opponent = self._blue_slots if side == "Red" else self._red_slots
        max_own = self.max_team_sizes[side]
        max_opponent = self.max_team_sizes["Blue" if side == "Red" else "Red"]
        return np.asarray(
            [
                unit.hp / max(1.0, unit.max_hp),
                unit.shield / max(1.0, unit.max_shield),
                unit.cooldown / max(1.0, unit.max_cooldown),
                unit.pos[0] / self.map_info.width,
                unit.pos[1] / self.map_info.height,
                unit.velocity[0] / max(1.0, unit.max_velocity),
                unit.velocity[1] / max(1.0, unit.max_velocity),
                sum(self._alive(one) for one in own) / max_own,
                sum(self._alive(one) for one in opponent) / max_opponent,
                self.episode_steps / self.config.episode_limit,
                float(side == "Red"),
                float(side == "Blue"),
            ],
            dtype=np.float32,
        )

    def _task_features(self, unit: object, side: str) -> np.ndarray:
        if self._asset is None:
            destination = np.asarray([self.map_info.width / 2, self.map_info.height / 2])
            asset_fraction = 1.0
            asset_alive = 1.0
        else:
            destination = self._asset.pos
            asset_fraction = self._unit_total_hp(self._asset) / self._initial_asset_hp
            asset_alive = float(self.asset_alive)
        delta = destination - unit.pos
        diagonal = float(np.hypot(self.map_info.width, self.map_info.height))
        remaining_horizon = max(
            0.0, 1.0 - self.episode_steps / self.config.episode_limit
        )
        return np.asarray(
            [
                delta[0] / self.map_info.width,
                delta[1] / self.map_info.height,
                np.linalg.norm(delta) / diagonal,
                asset_fraction,
                asset_alive,
                self.config.red_agents / self.config.max_red_agents,
                self.config.blue_agents / self.config.max_blue_agents,
                1.0 if side == "Red" else -1.0,
                remaining_horizon,
            ],
            dtype=np.float32,
        )

    def _state_features(self, entity: object) -> np.ndarray:
        is_red = entity in self._red_slots
        is_blue = entity in self._blue_slots
        is_asset = entity is self._asset
        remaining_horizon = max(
            0.0, 1.0 - self.episode_steps / self.config.episode_limit
        )
        return np.asarray(
            [
                entity.pos[0] / self.map_info.width,
                entity.pos[1] / self.map_info.height,
                entity.velocity[0] / max(1.0, entity.max_velocity),
                entity.velocity[1] / max(1.0, entity.max_velocity),
                entity.hp / max(1.0, entity.max_hp),
                entity.shield / max(1.0, entity.max_shield),
                entity.cooldown / max(1.0, entity.max_cooldown),
                float(self._alive(entity)),
                float(is_red),
                float(is_blue),
                float(is_asset),
                remaining_horizon,
            ],
            dtype=np.float32,
        )

    def observe(self, side: str) -> Dict[str, np.ndarray]:
        if side not in {"Red", "Blue"}:
            raise ValueError("side must be Red or Blue")
        controlled = self._red_slots if side == "Red" else self._blue_slots
        max_agents = self.max_team_sizes[side]
        entity_slots = self._all_entity_slots()
        entity_obs = np.zeros(
            (max_agents, self.max_entities, self.ENTITY_DIM), dtype=np.float32
        )
        entity_mask = np.zeros((max_agents, self.max_entities), dtype=bool)
        self_obs = np.zeros((max_agents, self.SELF_DIM), dtype=np.float32)
        task_obs = np.zeros((max_agents, self.TASK_DIM), dtype=np.float32)
        agent_mask = np.zeros(max_agents, dtype=bool)
        avail_actions = np.zeros((max_agents, self.action_dim), dtype=bool)
        action_entity_index = np.full(
            (max_agents, self.action_dim), -1, dtype=np.int64
        )
        action_target_type = np.zeros(
            (max_agents, self.action_dim), dtype=np.int64
        )
        avail_actions[:, 0] = True
        for agent_index, observer in enumerate(controlled):
            if not self._alive(observer):
                continue
            agent_mask[agent_index] = True
            self_obs[agent_index] = self._self_features(observer, side)
            task_obs[agent_index] = self._task_features(observer, side)
            avail_actions[agent_index] = self._available_actions_for(observer, side)
            if side == "Red":
                for target_index in range(self.config.max_blue_agents):
                    action = 6 + target_index
                    action_entity_index[agent_index, action] = (
                        self.config.max_red_agents + target_index
                    )
                    action_target_type[agent_index, action] = 1
                if self._asset is not None:
                    assert self.asset_action_id is not None
                    action_entity_index[agent_index, self.asset_action_id] = (
                        self.config.max_red_agents + self.config.max_blue_agents
                    )
                    action_target_type[agent_index, self.asset_action_id] = 3
            else:
                for target_index in range(self.config.max_red_agents):
                    action = 6 + target_index
                    action_entity_index[agent_index, action] = target_index
                    action_target_type[agent_index, action] = 1
            for entity_index, entity in enumerate(entity_slots):
                if entity is None or not self._alive(entity):
                    continue
                distance = float(np.linalg.norm(entity.pos - observer.pos))
                # Team members, self and the objective are shared explicitly;
                # opponents obey upstream's sight range (9 map units).
                relation = self._relation(observer, entity, side)
                visible = relation[1] > 0 or relation[3] > 0 or distance < 9.0
                if visible:
                    entity_mask[agent_index, entity_index] = True
                    entity_obs[agent_index, entity_index] = self._entity_features(
                        observer, entity, side
                    )
        state_entities = np.zeros(
            (self.max_entities, self.STATE_ENTITY_DIM), dtype=np.float32
        )
        state_mask = np.zeros(self.max_entities, dtype=bool)
        for index, entity in enumerate(entity_slots):
            if entity is not None and self._alive(entity):
                state_entities[index] = self._state_features(entity)
                state_mask[index] = True
        return {
            "entity_obs": entity_obs,
            "entity_mask": entity_mask,
            "self_obs": self_obs,
            "task_obs": task_obs,
            "agent_mask": agent_mask,
            "avail_actions": avail_actions,
            "action_entity_index": action_entity_index,
            "action_target_type": action_target_type,
            "state_entities": state_entities,
            "state_mask": state_mask,
        }

    def _observe_both(self) -> Dict[str, Dict[str, np.ndarray]]:
        return {side: self.observe(side) for side in ("Red", "Blue")}

    def _info(
        self,
        outcome_red: int,
        termination_reason: Optional[str],
        reward_components: Optional[Mapping[str, float]] = None,
    ) -> Dict[str, object]:
        red_fraction, blue_fraction, asset_fraction = self._fractions()
        return {
            "protocol_id": self.protocol_id,
            "scenario_name": self.map_info.name,
            "objective": self.config.objective,
            "episode_steps": self.episode_steps,
            "outcome_red": int(outcome_red),
            "termination_reason": termination_reason,
            "red_alive": int(sum(self._alive(unit) for unit in self._red_slots)),
            "blue_alive": int(sum(self._alive(unit) for unit in self._blue_slots)),
            "red_health_fraction": float(red_fraction),
            "blue_health_fraction": float(blue_fraction),
            "asset_health_fraction": float(asset_fraction),
            "reward_mode": self.config.reward_mode,
            "discount_gamma": float(self.config.discount_gamma),
            "shaping_scale": float(self.config.shaping_scale),
            "approach_weight": float(self.config.approach_weight),
            "spawn_jitter": float(self.config.spawn_jitter),
            "randomization_config_sha256": self._randomization_config_hash,
            "layout_hash": self._layout_metadata.get("layout_sha256"),
            "layout": dict(self._layout_metadata),
            "upstream_commit": UPSTREAM_COMMIT,
            "reward_components": (
                None if reward_components is None else dict(reward_components)
            ),
        }


def tensorize_smaclite_ad_observation(
    observation: Dict[str, np.ndarray],
    device: torch.device = torch.device("cpu"),
) -> Tuple[TeamObservation, GlobalState]:
    """Convert one team observation to the shared Open-SCORE contract."""

    team = TeamObservation(
        entity_obs=torch.as_tensor(
            observation["entity_obs"], dtype=torch.float32, device=device
        ).unsqueeze(0),
        entity_mask=torch.as_tensor(
            observation["entity_mask"], dtype=torch.bool, device=device
        ).unsqueeze(0),
        self_obs=torch.as_tensor(
            observation["self_obs"], dtype=torch.float32, device=device
        ).unsqueeze(0),
        task_obs=torch.as_tensor(
            observation["task_obs"], dtype=torch.float32, device=device
        ).unsqueeze(0),
        agent_mask=torch.as_tensor(
            observation["agent_mask"], dtype=torch.bool, device=device
        ).unsqueeze(0),
        avail_actions=torch.as_tensor(
            observation["avail_actions"], dtype=torch.bool, device=device
        ).unsqueeze(0),
        action_entity_index=(
            torch.as_tensor(
                observation["action_entity_index"], dtype=torch.long, device=device
            ).unsqueeze(0)
            if "action_entity_index" in observation
            else None
        ),
        action_target_type=(
            torch.as_tensor(
                observation["action_target_type"], dtype=torch.long, device=device
            ).unsqueeze(0)
            if "action_target_type" in observation
            else None
        ),
    )
    state = GlobalState(
        entities=torch.as_tensor(
            observation["state_entities"], dtype=torch.float32, device=device
        ).unsqueeze(0),
        entity_mask=torch.as_tensor(
            observation["state_mask"], dtype=torch.bool, device=device
        ).unsqueeze(0),
    )
    team.validate()
    state.validate()
    return team, state


def tensorize_smaclite_stock_observation(
    observation: Dict[str, np.ndarray],
    device: torch.device = torch.device("cpu"),
) -> Tuple[TeamObservation, GlobalState]:
    """Tensorize the lossless stock-wrapper representation."""

    return tensorize_smaclite_ad_observation(observation, device)


def stock_scenario_fingerprint() -> Dict[str, object]:
    """Hash installed stock scenarios with the registered manifest algorithm.

    The aggregate intentionally matches ``stock_runtime_probe.py`` and the
    stock reproduction manifest: for every lexicographically sorted JSON file,
    hash ``filename + NUL + raw file bytes + LF``.  Keeping one aggregate
    definition prevents the stock-source and EPyMARL evidence paths from
    reporting different hashes for the same immutable map set.
    """

    _require_smaclite()
    root = Path(smaclite.__file__).resolve().parent / "env" / "maps" / "smaclite_maps"
    files: Dict[str, str] = {}
    combined = hashlib.sha256()
    for path in sorted(root.glob("*.json")):
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        files[path.name] = digest
        combined.update(path.name.encode("utf-8"))
        combined.update(b"\0")
        combined.update(raw)
        combined.update(b"\n")
    if not files:
        raise FileNotFoundError(
            "installed SMAClite contains no stock map JSON files; use editable install"
        )
    try:
        package_version = importlib.metadata.version("smaclite")
    except importlib.metadata.PackageNotFoundError:
        package_version = "unknown"
    return {
        "repository": UPSTREAM_REPOSITORY,
        "tag": UPSTREAM_TAG,
        "commit": UPSTREAM_COMMIT,
        "distribution_version": package_version,
        "module_path": str(Path(smaclite.__file__).resolve()),
        "file_count": len(files),
        "combined_hash_algorithm": "sorted_filename_nul_raw_bytes_lf_sha256",
        "combined_sha256": combined.hexdigest(),
        "files": files,
    }


__all__ = [
    "PROTOCOL_ID",
    "SMACliteADConfig",
    "SMACliteADEnv",
    "SMACliteStockAdapter",
    "make_smaclite_ad_map",
    "stock_scenario_fingerprint",
    "tensorize_smaclite_ad_observation",
    "tensorize_smaclite_stock_observation",
]
