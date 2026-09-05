"""Multi-target HAD adapter for the Stage-3 allocation game.

The original Stage-1 adapter deliberately exposes a single protected target.
Stage 3 needs a different contract: the two commanders allocate persistent HAD
agent IDs to targets, while the physical simulation remains one shared world.
This module implements that contract while retaining the native HAD physical
interaction rules. Collision, firing and damage still see the entire HAD world;
an upper-level assignment controls navigation and grouping, not the simulator's
damage equation.

Target IDs accepted by this adapter are stable zero-based indices into
``env.targets``.  Agent IDs are HAD's globally unique ``Entity.Id`` values.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .had_stage1 import ACCELERATION_PRIMITIVES


AssignmentInput = Union[
    Mapping[int, Optional[int]],
    Sequence[Optional[int]],
]
ActionInput = Union[Mapping[int, int], Sequence[int]]


@dataclass(frozen=True)
class HADStage3Event:
    """One event observed at a Stage-3 command boundary."""

    kind: str
    step: int
    side: Optional[str] = None
    agent_ids: Tuple[int, ...] = ()
    target_ids: Tuple[int, ...] = ()
    terminal: bool = False

    def as_dict(self) -> Dict[str, object]:
        return {
            "kind": self.kind,
            "step": int(self.step),
            "side": self.side,
            "agent_ids": list(self.agent_ids),
            "target_ids": list(self.target_ids),
            "terminal": bool(self.terminal),
        }


@dataclass(frozen=True)
class HADStage3Snapshot:
    """Exact in-memory branch point, including upper-level assignments."""

    step_count: int
    entity_states: Tuple[Mapping[str, object], ...]
    red_assignment: Tuple[Tuple[int, Optional[int]], ...]
    blue_assignment: Tuple[Tuple[int, Optional[int]], ...]
    active_target_ids: Tuple[int, ...]
    environment_rng_state: Mapping[str, object]
    adapter_rng_state: Mapping[str, object]
    numpy_random_state: Tuple[object, ...]
    last_events: Tuple[HADStage3Event, ...]


class HADStage3Adapter:
    """A shared-world, multi-target HAD command environment.

    Blue's *lower* policy is a known ``rush`` or ``split_rush`` rule.  Its
    target choice is not part of that rule: it is supplied independently via
    :meth:`set_assignments`, with both commands produced from the same
    decision-before-commit state. Red target assignments define
    target-centric policy views; Red acceleration actions are supplied by the
    caller (normally a frozen Stage-1 controller).

    The adapter deliberately retains native HAD semantics for the currently
    active protected points:

    * collision and agent-agent damage inspect the shared world;
    * Blue navigation respects the upper-level assignment while native firing
      and damage remain shared-world interactions;
    * firing attackers self-destruct;
    * destruction of any active target is an immediate Blue/global win; and
    * destruction of all Blue attackers is a Red/global win.
    """

    ENTITY_DIM = 12
    SELF_DIM = 10
    TASK_DIM = 7
    STATE_ENTITY_DIM = 12
    ACTION_DIM = len(ACCELERATION_PRIMITIVES)
    VALID_BLUE_STYLES = frozenset({"rush", "split_rush"})

    def __init__(
        self,
        red_attackers: int,
        blue_attackers: int,
        targets: int,
        *,
        max_steps: int = 50,
        target_region: Sequence[Sequence[float]] = (
            (-2300.0, -1900.0),
            (-1200.0, 1200.0),
            (50.0, 300.0),
        ),
        target_positions: Optional[Sequence[Sequence[float]]] = None,
        blue_rule_style: str = "rush",
        split_spacing: float = 180.0,
        task_type: str = "Training",
    ) -> None:
        if red_attackers < 1 or blue_attackers < 1:
            raise ValueError("Stage-3 HAD requires at least one agent on each side")
        if targets < 1:
            raise ValueError("Stage-3 HAD requires at least one target")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if blue_rule_style not in self.VALID_BLUE_STYLES:
            raise ValueError(
                f"blue_rule_style must be one of {sorted(self.VALID_BLUE_STYLES)}"
            )
        if split_spacing < 0.0:
            raise ValueError("split_spacing must be non-negative")

        from HAD_Env.config import AeroPoint
        from HAD_Env.make_env import HADEnv

        region = np.asarray(target_region, dtype=np.float64)
        if region.shape != (3, 2):
            raise ValueError("target_region must provide low/high bounds for x, y, z")
        bounds = np.asarray(AeroPoint, dtype=np.float64)
        if np.any(region[:, 0] > region[:, 1]):
            raise ValueError("target_region lower bounds must not exceed upper bounds")
        if np.any(region[:, 0] < bounds[:, 0]) or np.any(
            region[:, 1] > bounds[:, 1]
        ):
            raise ValueError("target_region must stay inside the HAD world")

        fixed_positions: Optional[np.ndarray]
        if target_positions is None:
            fixed_positions = None
        else:
            fixed_positions = np.asarray(target_positions, dtype=np.float64)
            if fixed_positions.shape != (targets, 3):
                raise ValueError("target_positions must have shape [targets, 3]")
            if np.any(fixed_positions < bounds[:, 0]) or np.any(
                fixed_positions > bounds[:, 1]
            ):
                raise ValueError("target_positions must stay inside the HAD world")

        self.env = HADEnv(
            red_attackers,
            blue_attackers,
            targets,
            task_type=task_type,
            target_region=region,
        )
        self.max_steps = int(max_steps)
        self.target_region = region.astype(np.float32)
        self.fixed_target_positions = (
            None if fixed_positions is None else fixed_positions.astype(np.float64)
        )
        self._planned_target_positions = (
            None if fixed_positions is None else fixed_positions.astype(np.float64).copy()
        )
        # The formal S3 protocol fixes this complete target set at reset.  The
        # private set is retained only so snapshots preserve an explicit
        # invariant; changing target membership during an episode is rejected.
        self._active_target_ids = set(range(targets))
        self._parking_position = np.asarray([2450.0, 2450.0, 950.0], dtype=np.float64)
        self.blue_rule_style = blue_rule_style
        self.split_spacing = float(split_spacing)
        self.step_count = 0
        self.rng = np.random.default_rng()
        self._red_assignment: Dict[int, Optional[int]] = {}
        self._blue_assignment: Dict[int, Optional[int]] = {}
        self.last_events: Tuple[HADStage3Event, ...] = ()

    @property
    def action_vectors(self) -> np.ndarray:
        return ACCELERATION_PRIMITIVES.copy()

    @property
    def red_ids(self) -> Tuple[int, ...]:
        return tuple(int(agent.Id) for agent in self.env.red_agents)

    @property
    def blue_ids(self) -> Tuple[int, ...]:
        return tuple(int(agent.Id) for agent in self.env.blue_agents)

    @property
    def target_ids(self) -> Tuple[int, ...]:
        return tuple(sorted(self._active_target_ids))

    @property
    def all_target_ids(self) -> Tuple[int, ...]:
        return tuple(range(len(self.env.targets)))

    @property
    def red_assignment(self) -> Dict[int, Optional[int]]:
        return dict(self._red_assignment)

    @property
    def blue_assignment(self) -> Dict[int, Optional[int]]:
        return dict(self._blue_assignment)

    @property
    def assignments(self) -> Dict[str, Dict[int, Optional[int]]]:
        return {
            "Red": self.red_assignment,
            "Blue": self.blue_assignment,
        }

    @staticmethod
    def _opponent(side: str) -> str:
        if side == "Red":
            return "Blue"
        if side == "Blue":
            return "Red"
        raise ValueError("side must be Red or Blue")

    def _agents(self, side: str) -> List[object]:
        if side == "Red":
            return self.env.red_agents
        if side == "Blue":
            return self.env.blue_agents
        raise ValueError("side must be Red or Blue")

    def _assignment(self, side: str) -> Dict[int, Optional[int]]:
        if side == "Red":
            return self._red_assignment
        if side == "Blue":
            return self._blue_assignment
        raise ValueError("side must be Red or Blue")

    def reset(
        self,
        seed: Optional[int] = None,
        *,
        target_positions: Optional[Sequence[Sequence[float]]] = None,
        active_target_ids: Optional[Sequence[int]] = None,
        red_assignment: Optional[AssignmentInput] = None,
        blue_assignment: Optional[AssignmentInput] = None,
    ) -> Dict[str, object]:
        """Reset physical state and install complete command assignments.

        When no allocation is supplied, IDs are assigned round-robin.  This is
        only a deterministic smoke-test default; the upper policy must always
        submit both allocations explicitly at command time.
        """

        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.env.reset(evaluate=True, seed=seed)
        positions = self.fixed_target_positions if target_positions is None else np.asarray(
            target_positions, dtype=np.float64
        )
        if positions is not None:
            self._set_target_positions(positions)
            self._planned_target_positions = np.asarray(positions, dtype=np.float64).copy()
        else:
            self._planned_target_positions = np.asarray(
                [target.position for target in self.env.targets], dtype=np.float64
            )
        requested_active = self.all_target_ids
        if active_target_ids is not None:
            requested = self._normalise_active_target_ids(active_target_ids)
            if requested != requested_active:
                raise ValueError(
                    "Stage-3 fixes every configured target for the complete episode"
                )
        self._active_target_ids = set(requested_active)
        self._apply_target_activity_positions(reset_activated=True)
        self.step_count = 0
        self._red_assignment = {
            agent_id: requested_active[index % len(requested_active)]
            for index, agent_id in enumerate(self.red_ids)
        }
        self._blue_assignment = {
            agent_id: requested_active[index % len(requested_active)]
            for index, agent_id in enumerate(self.blue_ids)
        }
        if red_assignment is not None:
            self.set_assignments("Red", red_assignment, record_event=False)
        if blue_assignment is not None:
            self.set_assignments("Blue", blue_assignment, record_event=False)
        self.last_events = ()
        return self.global_state()

    def _set_target_positions(self, positions: np.ndarray) -> None:
        from HAD_Env.config import AeroPoint

        array = np.asarray(positions, dtype=np.float64)
        if array.shape != (len(self.env.targets), 3):
            raise ValueError("target_positions must have shape [targets, 3]")
        bounds = np.asarray(AeroPoint, dtype=np.float64)
        if np.any(array < bounds[:, 0]) or np.any(array > bounds[:, 1]):
            raise ValueError("target_positions must stay inside the HAD world")
        for target, position in zip(self.env.targets, array):
            target.set_position(position.tolist())
            target.initial_position = position.tolist()

    def _normalise_active_target_ids(
        self, target_ids: Sequence[int]
    ) -> Tuple[int, ...]:
        values = tuple(sorted({int(value) for value in target_ids}))
        if not values:
            raise ValueError("at least one target must remain active")
        invalid = set(values).difference(self.all_target_ids)
        if invalid:
            raise ValueError(f"unknown target IDs: {sorted(invalid)}")
        return values

    def _apply_target_activity_positions(self, *, reset_activated: bool) -> None:
        from HAD_Env.config import initial_health

        if self._planned_target_positions is None:  # pragma: no cover - reset invariant
            raise RuntimeError("planned target positions are unavailable")
        for target_id, target in enumerate(self.env.targets):
            target.is_active_objective = target_id in self._active_target_ids
            if target_id in self._active_target_ids:
                position = self._planned_target_positions[target_id]
                target.set_position(position.tolist())
                target.initial_position = position.tolist()
                if reset_activated:
                    target.Health = float(initial_health)
            else:
                target.set_position(self._parking_position.tolist())
                target.initial_position = self._parking_position.tolist()
                # A dormant fixed-world entity is not a failed objective.
                target.Health = float(initial_health)
        self.env.update_alive_agents()

    def set_active_targets(
        self,
        target_ids: Sequence[int],
        *,
        reset_new_targets: bool = True,
    ) -> Tuple[HADStage3Event, ...]:
        """Reject target-set changes under the fixed-target S3 protocol."""

        del reset_new_targets
        requested = self._normalise_active_target_ids(target_ids)
        if requested != self.all_target_ids:
            raise ValueError(
                "Stage-3 target membership is fixed at reset and cannot change"
            )
        self.last_events = ()
        return ()

    def set_assignments(
        self,
        side: str,
        assignment: AssignmentInput,
        *,
        record_event: bool = True,
    ) -> Tuple[HADStage3Event, ...]:
        """Replace one side's allocation using global agent IDs.

        ``None`` denotes an uncommitted reserve.  A mapping may omit IDs, in
        which case those IDs become reserves; a sequence is ordered by the
        side's stable roster.  Assigning to a destroyed target is rejected.
        """

        agents = self._agents(side)
        ids = tuple(int(agent.Id) for agent in agents)
        id_set = set(ids)
        if isinstance(assignment, Mapping):
            unknown = set(int(key) for key in assignment) - id_set
            if unknown:
                raise ValueError(f"assignment contains non-{side} IDs: {sorted(unknown)}")
            candidate = {agent_id: assignment.get(agent_id) for agent_id in ids}
        else:
            values = list(assignment)
            if len(values) != len(ids):
                raise ValueError(f"{side} assignment must contain {len(ids)} entries")
            candidate = dict(zip(ids, values))

        normalised: Dict[int, Optional[int]] = {}
        for agent_id, target_id in candidate.items():
            if target_id is None:
                normalised[agent_id] = None
                continue
            target_index = int(target_id)
            self._validate_target_id(target_index)
            if self.env.targets[target_index].Health <= 0:
                raise ValueError("cannot assign an agent to a destroyed target")
            normalised[agent_id] = target_index

        old = dict(self._assignment(side))
        changed_ids = tuple(
            agent_id for agent_id in ids if old.get(agent_id) != normalised[agent_id]
        )
        if side == "Red":
            self._red_assignment = normalised
        else:
            self._blue_assignment = normalised
        if not changed_ids:
            return ()
        targets = tuple(
            sorted(
                {
                    value
                    for agent_id in changed_ids
                    for value in (old.get(agent_id), normalised[agent_id])
                    if value is not None
                }
            )
        )
        event = HADStage3Event(
            kind="assignment_changed",
            step=self.step_count,
            side=side,
            agent_ids=changed_ids,
            target_ids=targets,
        )
        if record_event:
            self.last_events = (event,)
        return (event,)

    def set_joint_assignments(
        self,
        red_assignment: AssignmentInput,
        blue_assignment: AssignmentInput,
    ) -> Tuple[HADStage3Event, ...]:
        """Install simultaneous Red/Blue upper-level allocation actions."""

        red_events = self.set_assignments("Red", red_assignment, record_event=False)
        blue_events = self.set_assignments("Blue", blue_assignment, record_event=False)
        self.last_events = red_events + blue_events
        return self.last_events

    def _validate_target_id(self, target_id: int) -> None:
        if target_id not in self.target_ids:
            raise ValueError(f"target_id must be in {self.target_ids}")

    def _agents_by_ids(self, side: str, ids: Iterable[int]) -> List[object]:
        requested = tuple(int(value) for value in ids)
        if len(set(requested)) != len(requested):
            raise ValueError(f"{side} IDs must be unique")
        table = {int(agent.Id): agent for agent in self._agents(side)}
        unknown = set(requested) - set(table)
        if unknown:
            raise ValueError(f"unknown {side} IDs: {sorted(unknown)}")
        return [table[value] for value in requested]

    def assigned_ids(
        self,
        side: str,
        target_id: int,
        *,
        alive_only: bool = False,
    ) -> Tuple[int, ...]:
        self._validate_target_id(target_id)
        health = {int(agent.Id): float(agent.Health) for agent in self._agents(side)}
        return tuple(
            agent_id
            for agent_id, assigned in self._assignment(side).items()
            if assigned == target_id and (not alive_only or health[agent_id] > 0.0)
        )

    @staticmethod
    def _decode_acceleration(action_id: int) -> np.ndarray:
        value = int(action_id)
        if not 0 <= value < len(ACCELERATION_PRIMITIVES):
            raise ValueError("action id is outside the 27 acceleration primitives")
        return ACCELERATION_PRIMITIVES[value]

    def _nearest_acceleration(self, direction: np.ndarray) -> int:
        norm = float(np.linalg.norm(direction))
        if norm < 1e-8:
            return 0
        unit = np.asarray(direction, dtype=np.float64) / norm
        return int(np.argmax(ACCELERATION_PRIMITIVES @ unit))

    def commanded_rule_actions(
        self,
        side: str = "Blue",
        *,
        style: Optional[str] = None,
        subgroup_by_agent: Optional[Mapping[int, int]] = None,
    ) -> np.ndarray:
        """Return acceleration IDs for an assignment-conditioned lower rule.

        The MVP supports the registered Blue rules.  ``rush`` aims directly at
        the assigned target.  ``split_rush`` adds the same world-y lateral
        spacing used by Stage 1, but centres slots independently within each
        target group.  An unassigned reserve receives action 0 and therefore
        continues ballistically under native HAD dynamics.
        """

        if side != "Blue":
            raise ValueError("the Stage-3 MVP only registers Blue lower rules")
        selected_style = self.blue_rule_style if style is None else style
        if selected_style not in self.VALID_BLUE_STYLES:
            raise ValueError(
                f"style must be one of {sorted(self.VALID_BLUE_STYLES)}"
            )
        agents = self.env.blue_agents
        assignment = self._blue_assignment
        if subgroup_by_agent is not None:
            unknown = set(map(int, subgroup_by_agent)).difference(self.blue_ids)
            if unknown:
                raise ValueError(f"subgroup map contains non-Blue IDs: {sorted(unknown)}")
        groups: Dict[tuple[int, int], list[object]] = {}
        for agent in agents:
            agent_id = int(agent.Id)
            target_id = assignment[agent_id]
            if target_id is None:
                continue
            subgroup_id = (
                0
                if subgroup_by_agent is None
                else int(subgroup_by_agent.get(agent_id, 0))
            )
            groups.setdefault((int(target_id), subgroup_id), []).append(agent)
        slot = {
            int(agent.Id): (index, len(group))
            for group in groups.values()
            for index, agent in enumerate(group)
        }
        result: List[int] = []
        for agent in agents:
            if agent.Health <= 0:
                result.append(0)
                continue
            target_id = assignment[int(agent.Id)]
            if target_id is None:
                result.append(0)
                continue
            destination = np.asarray(
                self.env.targets[target_id].position, dtype=np.float64
            ).copy()
            if selected_style == "split_rush":
                index, count = slot[int(agent.Id)]
                destination[1] += self._split_lateral_offset(index, count)
            direction = destination - np.asarray(agent.position, dtype=np.float64)
            result.append(self._nearest_acceleration(direction))
        return np.asarray(result, dtype=np.int64)

    def _split_lateral_offset(self, index: int, count: int) -> float:
        """Scale ``split_rush`` lanes without parking attackers out of range.

        The frozen Round-01 rule used 180 world units between lanes.  That is
        unchanged for rosters up to six (the largest S2 collection roster).
        For larger target groups, the total half-width is capped at 90% of the
        native objective attack radius, so every lane still intersects the
        assigned objective's firing footprint instead of becoming a permanent
        non-attacking trajectory.
        """

        if count < 1 or not 0 <= int(index) < int(count):
            raise ValueError("split_rush slot must lie inside a non-empty group")
        if count == 1:
            return 0.0
        from HAD_Env.config import AttackDistance

        nominal_half_width = self.split_spacing * (count - 1) / 2.0
        maximum_half_width = 0.9 * float(np.max(AttackDistance))
        half_width = min(nominal_half_width, maximum_half_width)
        return float(-half_width + 2.0 * half_width * int(index) / (count - 1))

    def _coerce_actions(
        self,
        side: str,
        actions: ActionInput,
    ) -> np.ndarray:
        agents = self._agents(side)
        ids = tuple(int(agent.Id) for agent in agents)
        if isinstance(actions, Mapping):
            unknown = set(int(key) for key in actions) - set(ids)
            if unknown:
                raise ValueError(f"actions contain non-{side} IDs: {sorted(unknown)}")
            missing_alive = [
                int(agent.Id)
                for agent in agents
                if agent.Health > 0 and int(agent.Id) not in actions
            ]
            if missing_alive:
                raise ValueError(f"actions missing live {side} IDs: {missing_alive}")
            values = [int(actions.get(agent_id, 0)) for agent_id in ids]
        else:
            values = [int(value) for value in actions]
            if len(values) != len(agents):
                raise ValueError(f"{side} actions must contain {len(agents)} entries")
        for index, agent in enumerate(agents):
            if agent.Health <= 0:
                values[index] = 0
            self._decode_acceleration(values[index])
        return np.asarray(values, dtype=np.int64)

    def step(
        self,
        red_action_ids: ActionInput,
        blue_action_ids: Optional[ActionInput] = None,
        *,
        blue_style: Optional[str] = None,
        blue_subgroup_by_agent: Optional[Mapping[int, int]] = None,
    ) -> Tuple[Dict[str, object], Dict[str, float], bool, Dict[str, object]]:
        """Advance the shared world by one physical HAD step.

        Omitting ``blue_action_ids`` executes the known assignment-conditioned
        Blue rule.  Supplying actions is useful for controlled counterfactuals.
        """

        if self._terminal_sign() != 0 or self.step_count >= self.max_steps:
            raise RuntimeError("cannot step a completed Stage-3 episode")
        red = self._coerce_actions("Red", red_action_ids)
        blue = (
            self.commanded_rule_actions(
                "Blue",
                style=blue_style,
                subgroup_by_agent=blue_subgroup_by_agent,
            )
            if blue_action_ids is None
            else self._coerce_actions("Blue", blue_action_ids)
        )
        before = self._alive_signature()
        action_by_id = {
            int(agent.Id): int(action)
            for agent, action in zip(self.env.red_agents, red)
        }
        action_by_id.update(
            {
                int(agent.Id): int(action)
                for agent, action in zip(self.env.blue_agents, blue)
            }
        )
        physical_actions = [
            self._decode_acceleration(action_by_id[int(agent.Id)])
            for agent in self.env.agents
        ]
        # This adapter supplies its own observations, events and task reward.
        self.env.step_physics(physical_actions)
        self.env.update_alive_agents()
        self.step_count += 1
        terminal_sign = self._terminal_sign()
        terminated = terminal_sign != 0
        truncated = self.step_count >= self.max_steps and not terminated
        done = terminated or truncated
        self.last_events = self._detect_events(
            before,
            terminated=terminated,
            truncated=truncated,
            terminal_sign=terminal_sign,
        )
        outcome = terminal_sign if terminated else (1 if truncated else 0)
        red_reward = float(outcome if done else 0.0)
        rewards = {"Red": red_reward, "Blue": -red_reward}
        state = self.global_state()
        info: Dict[str, object] = {
            "terminated": terminated,
            "truncated": truncated,
            "outcome_red": float(outcome),
            "step": int(self.step_count),
            "events": [event.as_dict() for event in self.last_events],
            "red_assignment": self.red_assignment,
            "blue_assignment": self.blue_assignment,
            "red_actions": red.tolist(),
            "blue_actions": blue.tolist(),
            "blue_rule_style": self.blue_rule_style if blue_style is None else blue_style,
            "targets": state["targets"],
        }
        return state, rewards, done, info

    def _terminal_sign(self) -> int:
        active_breached = any(
            self.env.targets[target_id].Health < 1e-3
            for target_id in self.target_ids
        )
        all_blue_destroyed = all(
            agent.Health < 1e-3
            for agent in self.env.blue_agents
            if agent.Type == "Attack"
        )
        if active_breached:
            return -1
        if all_blue_destroyed:
            return 1
        return 0

    def _alive_signature(self) -> Dict[str, object]:
        return {
            "Red": {
                int(agent.Id): bool(agent.Health > 0) for agent in self.env.red_agents
            },
            "Blue": {
                int(agent.Id): bool(agent.Health > 0) for agent in self.env.blue_agents
            },
            "targets": {
                target_id: bool(self.env.targets[target_id].Health > 0)
                for target_id in self.target_ids
            },
        }

    def _detect_events(
        self,
        before: Mapping[str, object],
        *,
        terminated: bool,
        truncated: bool,
        terminal_sign: int,
    ) -> Tuple[HADStage3Event, ...]:
        events: List[HADStage3Event] = []
        dead_by_side: Dict[str, Tuple[int, ...]] = {}
        for side in ("Red", "Blue"):
            old = before[side]
            if not isinstance(old, Mapping):  # pragma: no cover - internal invariant
                raise AssertionError("invalid event signature")
            dead = tuple(
                int(agent.Id)
                for agent in self._agents(side)
                if bool(old[int(agent.Id)]) and agent.Health <= 0
            )
            dead_by_side[side] = dead
            if dead:
                targets = tuple(
                    sorted(
                        {
                            value
                            for agent_id in dead
                            for value in (self._assignment(side).get(agent_id),)
                            if value is not None
                        }
                    )
                )
                events.append(
                    HADStage3Event(
                        kind="agents_destroyed",
                        step=self.step_count,
                        side=side,
                        agent_ids=dead,
                        target_ids=targets,
                    )
                )

        old_targets = before["targets"]
        if not isinstance(old_targets, Mapping):  # pragma: no cover - invariant
            raise AssertionError("invalid target event signature")
        breached = tuple(
            target_id
            for target_id in self.target_ids
            if bool(old_targets.get(target_id, False))
            and self.env.targets[target_id].Health <= 0
        )
        if breached:
            events.append(
                HADStage3Event(
                    kind="targets_breached",
                    step=self.step_count,
                    side="Blue",
                    target_ids=breached,
                    terminal=True,
                )
            )

        for target_id in self.target_ids:
            previously_active = any(
                bool(before["Blue"][agent_id])
                and assigned == target_id
                for agent_id, assigned in self._blue_assignment.items()
            )
            currently_active = any(
                agent.Health > 0
                and self._blue_assignment[int(agent.Id)] == target_id
                for agent in self.env.blue_agents
            )
            if previously_active and not currently_active:
                events.append(
                    HADStage3Event(
                        kind="battlefield_resolved",
                        step=self.step_count,
                        side="Red",
                        target_ids=(target_id,),
                    )
                )
        if terminated:
            events.append(
                HADStage3Event(
                    kind=("red_win" if terminal_sign > 0 else "blue_win"),
                    step=self.step_count,
                    side=("Red" if terminal_sign > 0 else "Blue"),
                    terminal=True,
                )
            )
        elif truncated:
            events.append(
                HADStage3Event(
                    kind="horizon_survived",
                    step=self.step_count,
                    side="Red",
                    terminal=True,
                )
            )
        return tuple(events)

    def _entity_features(self, observer: object, entity: object) -> np.ndarray:
        from HAD_Env.config import AeroPoint, initial_health, vDomain

        span = np.asarray([high - low for low, high in AeroPoint], dtype=np.float32)
        # Preserve the Stage-1 operation order (including its final float32
        # cast) so a frozen checkpoint receives bit-identical features.
        rel_position = (
            np.asarray(entity.position) - np.asarray(observer.position)
        ) / span
        rel_velocity = (
            np.asarray(entity.velocity) - np.asarray(observer.velocity)
        ) / float(vDomain[1])
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

    def _self_features(
        self,
        side: str,
        agent: object,
        red_agents: Sequence[object],
        blue_agents: Sequence[object],
        target: object,
    ) -> np.ndarray:
        from HAD_Env.config import AeroPoint, vDomain

        low = np.asarray([bounds[0] for bounds in AeroPoint], dtype=np.float32)
        span = np.asarray(
            [bounds[1] - bounds[0] for bounds in AeroPoint], dtype=np.float32
        )
        position = 2.0 * (np.asarray(agent.position) - low) / span - 1.0
        velocity = np.asarray(agent.velocity) / float(vDomain[1])
        own = red_agents if side == "Red" else blue_agents
        opponent = blue_agents if side == "Red" else red_agents
        counts = [
            np.log1p(sum(one.Health > 0 for one in own)),
            np.log1p(sum(one.Health > 0 for one in opponent)),
            float(target.Health > 0),
        ]
        return np.asarray(
            list(position) + list(velocity) + [float(agent.Health)] + counts,
            dtype=np.float32,
        )

    def _task_features(
        self,
        side: str,
        agent: object,
        target: object,
        local_step: int,
    ) -> np.ndarray:
        from HAD_Env.config import AeroPoint, initial_health

        span = np.asarray([high - low for low, high in AeroPoint], dtype=np.float32)
        relative = (
            np.asarray(target.position) - np.asarray(agent.position)
        ) / span
        role = 1.0 if side == "Red" else -1.0
        remaining_horizon = max(0.0, 1.0 - int(local_step) / self.max_steps)
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

    def _state_features(self, entity: object, local_step: int) -> np.ndarray:
        from HAD_Env.config import AeroPoint, initial_health, vDomain

        low = np.asarray([bounds[0] for bounds in AeroPoint], dtype=np.float32)
        span = np.asarray(
            [bounds[1] - bounds[0] for bounds in AeroPoint], dtype=np.float32
        )
        position = (np.asarray(entity.position) - low) / span
        velocity = np.asarray(entity.velocity) / float(vDomain[1])
        health_scale = initial_health if entity.Color == "Entity" else 1.0
        types = [float(entity.Color == name) for name in ("Red", "Blue", "Entity")]
        remaining_horizon = max(0.0, 1.0 - int(local_step) / self.max_steps)
        return np.asarray(
            list(position)
            + list(velocity)
            + [float(entity.Health) / health_scale, float(entity.Health > 0)]
            + types
            + [remaining_horizon],
            dtype=np.float32,
        )

    def _local_entities(
        self,
        target_id: int,
        red_ids: Sequence[int],
        blue_ids: Sequence[int],
    ) -> Tuple[List[object], List[object], object, List[object]]:
        self._validate_target_id(target_id)
        red_agents = self._agents_by_ids("Red", red_ids)
        blue_agents = self._agents_by_ids("Blue", blue_ids)
        target = self.env.targets[target_id]
        # Match Stage-1/HAD ordering exactly: Red, Blue, Entity.
        return red_agents, blue_agents, target, red_agents + blue_agents + [target]

    def local_state_entities(
        self,
        target_id: int,
        red_ids: Sequence[int],
        blue_ids: Sequence[int],
        local_step: int = 0,
    ) -> np.ndarray:
        """Return the Stage-1-compatible centralized state for one micro-game."""

        _, _, _, world = self._local_entities(target_id, red_ids, blue_ids)
        return np.stack(
            [self._state_features(entity, local_step) for entity in world]
        ).astype(np.float32, copy=False)

    def local_observation(
        self,
        side: str,
        target_id: int,
        red_ids: Sequence[int],
        blue_ids: Sequence[int],
        local_step: int = 0,
    ) -> Dict[str, np.ndarray]:
        """Build a virtual one-target observation for a frozen Stage-1 policy.

        This filters policy input only.  Calling :meth:`step` afterwards still
        advances every entity in the shared world and retains cross-battlefield
        firing, damage and collision effects.
        """

        red_agents, blue_agents, target, world = self._local_entities(
            target_id, red_ids, blue_ids
        )
        controlled = red_agents if side == "Red" else blue_agents if side == "Blue" else None
        if controlled is None:
            raise ValueError("side must be Red or Blue")
        count = len(controlled)
        entities = len(world)
        if count:
            entity_obs = np.stack(
                [
                    np.stack(
                        [self._entity_features(agent, entity) for entity in world]
                    )
                    for agent in controlled
                ]
            ).astype(np.float32, copy=False)
            entity_mask = np.stack(
                [
                    np.asarray([entity.Health > 0 for entity in world], dtype=bool)
                    for _ in controlled
                ]
            )
            self_obs = np.stack(
                [
                    self._self_features(
                        side, agent, red_agents, blue_agents, target
                    )
                    for agent in controlled
                ]
            ).astype(np.float32, copy=False)
            task_obs = np.stack(
                [
                    self._task_features(side, agent, target, local_step)
                    for agent in controlled
                ]
            ).astype(np.float32, copy=False)
        else:
            entity_obs = np.empty((0, entities, self.ENTITY_DIM), dtype=np.float32)
            entity_mask = np.empty((0, entities), dtype=bool)
            self_obs = np.empty((0, self.SELF_DIM), dtype=np.float32)
            task_obs = np.empty((0, self.TASK_DIM), dtype=np.float32)
        agent_mask = np.asarray([agent.Health > 0 for agent in controlled], dtype=bool)
        avail_actions = np.ones((count, self.ACTION_DIM), dtype=bool)
        avail_actions[~agent_mask] = False
        if count:
            avail_actions[~agent_mask, 0] = True
        return {
            "entity_obs": entity_obs,
            "entity_mask": entity_mask,
            "self_obs": self_obs,
            "task_obs": task_obs,
            "agent_mask": agent_mask,
            "avail_actions": avail_actions,
            "state_entities": self.local_state_entities(
                target_id, red_ids, blue_ids, local_step
            ),
            "state_mask": np.ones(entities, dtype=bool),
            "agent_ids": np.asarray(
                [int(agent.Id) for agent in controlled], dtype=np.int64
            ),
            "entity_ids": np.asarray(
                [int(entity.Id) for entity in world], dtype=np.int64
            ),
            "target_id": np.asarray(target_id, dtype=np.int64),
        }

    @staticmethod
    def _entity_record(entity: object) -> Dict[str, object]:
        return {
            "id": int(entity.Id),
            "position": np.asarray(entity.position, dtype=np.float64).copy(),
            "velocity": np.asarray(entity.velocity, dtype=np.float64).copy(),
            "health": float(entity.Health),
            "alive": bool(entity.Health > 0),
        }

    def agent_states(self, side: str) -> Dict[int, Dict[str, object]]:
        """Expose global ID/kinematic/health state for one side."""

        assignment = self._assignment(side)
        result: Dict[int, Dict[str, object]] = {}
        for agent in self._agents(side):
            record = self._entity_record(agent)
            record.update(
                {
                    "side": side,
                    "type": str(agent.Type),
                    "assigned_target": assignment[int(agent.Id)],
                }
            )
            result[int(agent.Id)] = record
        return result

    def target_states(self) -> Dict[int, Dict[str, object]]:
        """Expose stable target indices and underlying HAD entity IDs."""

        result: Dict[int, Dict[str, object]] = {}
        for target_id in self.target_ids:
            target = self.env.targets[target_id]
            record = self._entity_record(target)
            record["entity_id"] = record.pop("id")
            record["target_id"] = target_id
            record["active"] = True
            result[target_id] = record
        return result

    def global_state(self) -> Dict[str, object]:
        """Return a copy-only command state suitable for logging/planning."""

        return {
            "step": int(self.step_count),
            "red": self.agent_states("Red"),
            "blue": self.agent_states("Blue"),
            "targets": self.target_states(),
            "assignments": self.assignments,
            "terminal_sign": self._terminal_sign(),
            "horizon_reached": bool(self.step_count >= self.max_steps),
        }

    def snapshot(self) -> HADStage3Snapshot:
        """Capture an exact physical and command-level branch point."""

        return HADStage3Snapshot(
            step_count=int(self.step_count),
            entity_states=tuple(
                copy.deepcopy(entity.__dict__) for entity in self.env.world
            ),
            red_assignment=tuple(sorted(self._red_assignment.items())),
            blue_assignment=tuple(sorted(self._blue_assignment.items())),
            active_target_ids=self.target_ids,
            environment_rng_state=copy.deepcopy(
                self.env.np_random.bit_generator.state
            ),
            adapter_rng_state=copy.deepcopy(self.rng.bit_generator.state),
            numpy_random_state=copy.deepcopy(np.random.get_state()),
            last_events=copy.deepcopy(self.last_events),
        )

    def restore(
        self,
        snapshot: HADStage3Snapshot,
        *,
        continuation_seed: Optional[int] = None,
    ) -> Dict[str, object]:
        """Restore a branch, optionally replacing continuation RNG streams."""

        if len(snapshot.entity_states) != len(self.env.world):
            raise ValueError("HAD Stage-3 snapshot roster differs from adapter roster")
        expected_red = set(self.red_ids)
        expected_blue = set(self.blue_ids)
        if set(dict(snapshot.red_assignment)) != expected_red or set(
            dict(snapshot.blue_assignment)
        ) != expected_blue:
            raise ValueError("HAD Stage-3 snapshot assignment roster differs")
        for entity, state in zip(self.env.world, snapshot.entity_states):
            entity.__dict__.clear()
            entity.__dict__.update(copy.deepcopy(dict(state)))
        self.step_count = int(snapshot.step_count)
        self._red_assignment = dict(snapshot.red_assignment)
        self._blue_assignment = dict(snapshot.blue_assignment)
        if tuple(snapshot.active_target_ids) != self.all_target_ids:
            raise ValueError("snapshot violates the fixed-target Stage-3 protocol")
        self._active_target_ids = set(self.all_target_ids)
        self.last_events = copy.deepcopy(snapshot.last_events)
        self.env.update_alive_agents()
        if continuation_seed is None:
            self.env.np_random.bit_generator.state = copy.deepcopy(
                snapshot.environment_rng_state
            )
            self.rng.bit_generator.state = copy.deepcopy(
                snapshot.adapter_rng_state
            )
            np.random.set_state(copy.deepcopy(snapshot.numpy_random_state))
        else:
            seed = int(continuation_seed)
            self.env.np_random = np.random.default_rng(seed)
            self.rng = np.random.default_rng(seed ^ 0x5A17D0A1)
            np.random.seed(seed % (2**32))
        return self.global_state()


class HADCommandedRuleController:
    """Stage-1-controller-compatible wrapper around the commanded Blue rule."""

    def __init__(self, style: str = "rush") -> None:
        if style not in HADStage3Adapter.VALID_BLUE_STYLES:
            raise ValueError(
                f"style must be one of {sorted(HADStage3Adapter.VALID_BLUE_STYLES)}"
            )
        self.style = style
        self.name = f"commanded_rule:{style}"

    def reset(self) -> None:
        return None

    def act(
        self,
        adapter: HADStage3Adapter,
        side: str,
        observation: Optional[Dict[str, np.ndarray]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        del observation, rng
        return adapter.commanded_rule_actions(side, style=self.style)
