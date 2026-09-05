"""Runtime bridge from frozen S1/S2 artifacts to the Stage-3 Blotto game."""

from __future__ import annotations

import copy
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from open_score.envs import HADStage3Adapter
from open_score.stage1 import QMixController, make_had_qmix
from open_score.stage2 import HADCanonicalizer

from .blotto import Allocation, EventBlottoGame
from .grouped_blotto import (
    GroupedAllocation,
    GroupedBlottoGame,
    expand_group_histogram,
)
from .matching import AgentTaskMatching, match_agents_to_tasks
from .payoff import (
    FrozenStage2Payoff,
    build_local_payoff_tensor,
    sha256_file,
    utility_to_survival_probability,
)


def load_round01_stage1_model(
    checkpoint_path: Path,
    device: torch.device | str,
    *,
    expected_sha256: str = "",
) -> torch.nn.Module:
    """Load the immutable round-01 REFIL-QMIX model with its exact architecture."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected_sha256 and sha256_file(path).lower() != expected_sha256.lower():
        raise ValueError("Stage-1 checkpoint SHA-256 differs from Stage-3 protocol")
    target_device = torch.device(device)
    payload = torch.load(path, map_location=target_device, weights_only=False)
    if payload.get("extra", {}).get("architecture") != "REFIL-QMIX-HAD-v1":
        raise ValueError("Stage-1 checkpoint is not REFIL-QMIX-HAD-v1")
    model = make_had_qmix(
        target_device,
        agent_hidden_dim=64,
        mixer_hidden_dim=128,
        mixing_dim=32,
        encoder_kind="refil",
        attention_heads=4,
        attention_embed_dim=128,
        hypernet_hidden_dim=128,
    )
    model.load_state_dict(payload["online"], strict=True)
    model.eval()
    return model


def _nearest_acceleration(adapter: HADStage3Adapter, direction: np.ndarray) -> int:
    norm = float(np.linalg.norm(direction))
    if norm < 1e-8:
        return 0
    return int(np.argmax(adapter.action_vectors @ (direction / norm)))


class FrozenStage1GroupExecutor:
    """Execute one shared S1 model independently on target-centred Red groups.

    A target owns recurrent controller state while its ordered Red/Blue roster
    remains unchanged.  Reallocation resets that target controller, preventing
    hidden state from silently carrying information from a different option.
    No-threat assigned groups use a deterministic guard waypoint.  Red
    reserve agents may receive temporary public-state patrol waypoints.  Those
    waypoints are execution-only and never alter their formal ``None`` target
    assignment in the upper-level game.
    """

    VALID_MICRO_GROUPINGS = frozenset(
        {"balanced_chunks", "attacker_matched", "planned_groups"}
    )

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device | str,
        *,
        micro_grouping: str = "balanced_chunks",
    ) -> None:
        if micro_grouping not in self.VALID_MICRO_GROUPINGS:
            raise ValueError(
                f"micro_grouping must be one of {sorted(self.VALID_MICRO_GROUPINGS)}"
            )
        self.model = model
        self.device = torch.device(device)
        self.micro_grouping = str(micro_grouping)
        self.controllers: Dict[tuple[int, int], QMixController] = {}
        self.rosters: Dict[
            tuple[int, int], tuple[tuple[int, ...], tuple[int, ...]]
        ] = {}

    def reset(self) -> None:
        self.controllers.clear()
        self.rosters.clear()

    def _controller(
        self,
        target_id: int,
        subgroup_id: int,
        red_ids: tuple[int, ...],
        blue_ids: tuple[int, ...],
    ) -> QMixController:
        key = (int(target_id), int(subgroup_id))
        roster = (red_ids, blue_ids)
        if key not in self.controllers or self.rosters.get(key) != roster:
            controller = QMixController(
                self.model,
                self.device,
                epsilon=0.0,
                name=f"round01_refil_target_{target_id}_group_{subgroup_id}",
            )
            controller.reset()
            self.controllers[key] = controller
            self.rosters[key] = roster
        return self.controllers[key]

    @staticmethod
    def _balanced_chunks(values: Sequence[int], groups: int) -> tuple[tuple[int, ...], ...]:
        if groups < 1:
            return ()
        base, remainder = divmod(len(values), groups)
        chunks: list[tuple[int, ...]] = []
        cursor = 0
        for index in range(groups):
            size = base + int(index < remainder)
            chunks.append(tuple(int(value) for value in values[cursor : cursor + size]))
            cursor += size
        return tuple(chunks)

    def micro_rosters(
        self, adapter: HADStage3Adapter
    ) -> Dict[tuple[int, int], tuple[tuple[int, ...], tuple[int, ...]]]:
        """Partition each target allocation into Stage-1-scale subgames.

        The Blotto action is still the unrestricted target-level count.  This
        execution bridge avoids feeding a large target roster directly into
        the local policy.  attacker_matched treats each Blue attacker as a
        separate interception hypothesis whenever enough Red groups exist;
        with the registered 1.5:1 budget this naturally produces a mixture of
        1v1 and 2v1 subgames.  balanced_chunks retains the legacy fewest-groups
        split for controlled ablations.  Both styles preserve every assigned
        agent and expose true local Red shortages.
        """

        if self.micro_grouping == "planned_groups":
            raise ValueError(
                "planned_groups must be supplied through roster_override; "
                "there is no post-hoc microgroup rule"
            )
        red_state = adapter.agent_states("Red")
        blue_state = adapter.agent_states("Blue")
        target_state = adapter.target_states()
        result: Dict[
            tuple[int, int], tuple[tuple[int, ...], tuple[int, ...]]
        ] = {}
        for target_id in adapter.target_ids:
            red_ids = list(adapter.assigned_ids("Red", target_id, alive_only=False))
            blue_ids = list(adapter.assigned_ids("Blue", target_id, alive_only=False))
            if not red_ids and not blue_ids:
                continue
            target_y = float(target_state[target_id]["position"][1])

            def spatial_key(state, agent_id: int) -> tuple[float, float, int]:
                position = state[agent_id]["position"]
                return (
                    float(position[1]) - target_y,
                    float(position[0]),
                    int(agent_id),
                )

            red_ids.sort(key=lambda value: spatial_key(red_state, value))
            blue_ids.sort(key=lambda value: spatial_key(blue_state, value))
            if self.micro_grouping == "attacker_matched":
                groups = max(math.ceil(len(red_ids) / 4), len(blue_ids), 1)
            else:
                groups = max(
                    math.ceil(len(red_ids) / 4),
                    math.ceil(len(blue_ids) / 4),
                    1,
                )
            red_chunks = self._balanced_chunks(red_ids, groups)
            blue_chunks = self._balanced_chunks(blue_ids, groups)
            for subgroup_id, (red_chunk, blue_chunk) in enumerate(
                zip(red_chunks, blue_chunks)
            ):
                result[(target_id, subgroup_id)] = (red_chunk, blue_chunk)
        return result

    def _guard_action(
        self, adapter: HADStage3Adapter, agent_id: int, target_id: int
    ) -> int:
        agent = adapter.agent_states("Red")[agent_id]
        target = adapter.target_states()[target_id]
        # Hold on the threat-facing side of the protected point.  This stays
        # target-relative and does not invent a new learned macro action.
        destination = np.asarray(target["position"], dtype=np.float64) + np.asarray(
            [350.0, 0.0, 0.0]
        )
        return _nearest_acceleration(
            adapter, destination - np.asarray(agent["position"], dtype=np.float64)
        )

    def act(
        self,
        adapter: HADStage3Adapter,
        *,
        local_steps: Optional[Mapping[int, int]] = None,
        roster_override: Optional[
            Mapping[
                tuple[int, int],
                tuple[Sequence[int], Sequence[int]],
            ]
        ] = None,
        reserve_waypoints: Optional[Mapping[int, Sequence[float]]] = None,
    ) -> Dict[int, int]:
        """Execute frozen S1 controllers for current or explicitly paired groups.

        ``roster_override`` installs the exact identity coalitions selected by
        the upper-level game.  ``reserve_waypoints`` contains temporary patrol
        destinations computed from public positions at the latest replanning
        event.  Supplying either object does not change adapter assignments.
        """

        local_steps = {} if local_steps is None else local_steps
        reserve_waypoints = (
            {}
            if reserve_waypoints is None
            else {
                int(agent_id): np.asarray(destination, dtype=np.float64)
                for agent_id, destination in reserve_waypoints.items()
            }
        )
        if any(destination.shape != (3,) for destination in reserve_waypoints.values()):
            raise ValueError("reserve patrol waypoints must be three-dimensional")
        actions: Dict[int, int] = {}
        red_state = adapter.agent_states("Red")
        blue_state = adapter.agent_states("Blue")
        for agent_id in reserve_waypoints:
            if agent_id not in red_state:
                raise ValueError("reserve waypoint contains an unknown Red identity")
            if red_state[agent_id].get("assigned_target") is not None:
                raise ValueError(
                    "a formally assigned Red agent cannot receive a reserve waypoint"
                )
        alive_targets = [
            target_id
            for target_id, value in adapter.target_states().items()
            if bool(value["alive"])
        ]
        if not alive_targets:
            return {agent_id: 0 for agent_id in adapter.red_ids}

        rosters = self.micro_rosters(adapter) if roster_override is None else {
            (int(target_id), int(subgroup_id)): (
                tuple(int(value) for value in red_ids),
                tuple(int(value) for value in blue_ids),
            )
            for (target_id, subgroup_id), (red_ids, blue_ids) in roster_override.items()
        }
        for (target_id, subgroup_id), (red_ids, blue_ids) in rosters.items():
            alive_red = [agent_id for agent_id in red_ids if red_state[agent_id]["alive"]]
            alive_blue = [agent_id for agent_id in blue_ids if blue_state[agent_id]["alive"]]
            if not alive_red:
                continue
            if not alive_blue:
                for agent_id in alive_red:
                    actions[agent_id] = self._guard_action(adapter, agent_id, target_id)
                continue
            observation = adapter.local_observation(
                "Red",
                target_id,
                red_ids,
                blue_ids,
                int(local_steps.get(target_id, 0)),
            )
            controller = self._controller(
                target_id, subgroup_id, red_ids, blue_ids
            )
            selected = controller.act(adapter, "Red", observation, adapter.rng)
            if len(selected) != len(red_ids):
                raise RuntimeError("Stage-1 controller returned the wrong group action count")
            actions.update(
                {
                    agent_id: int(action)
                    for agent_id, action in zip(red_ids, selected)
                    if red_state[agent_id]["alive"]
                }
            )

        for agent_id, state in red_state.items():
            if not state["alive"] or agent_id in actions:
                continue
            if state.get("assigned_target") is None:
                destination = reserve_waypoints.get(int(agent_id))
                actions[agent_id] = (
                    0
                    if destination is None
                    else _nearest_acceleration(
                        adapter,
                        destination
                        - np.asarray(state["position"], dtype=np.float64),
                    )
                )
            else:
                if int(agent_id) in reserve_waypoints:
                    raise ValueError(
                        "a formally assigned Red agent cannot receive a reserve waypoint"
                    )
                actions[agent_id] = self._guard_action(
                    adapter, agent_id, int(state["assigned_target"])
                )
        return actions


@dataclass(frozen=True)
class ObservableThreatPatrol:
    """Execution-only Red reserve patrol computed from public kinematics.

    Tuple storage keeps the result deterministic and directly serialisable.
    In particular, this object contains no Blue target assignment, subgroup,
    hidden type, or lower-policy style.
    """

    reserve_ids: Tuple[int, ...]
    target_by_agent: Tuple[Tuple[int, int], ...]
    waypoint_by_agent: Tuple[Tuple[int, Tuple[float, float, float]], ...]
    target_threat: Tuple[Tuple[int, float], ...]

    def target_mapping(self) -> Dict[int, int]:
        return dict(self.target_by_agent)

    def waypoint_mapping(self) -> Dict[int, np.ndarray]:
        return {
            int(agent_id): np.asarray(waypoint, dtype=np.float64)
            for agent_id, waypoint in self.waypoint_by_agent
        }


def observable_threat_patrol(
    reserve_ids: Sequence[int],
    red_positions: Mapping[int, Sequence[float]],
    blue_positions: Mapping[int, Sequence[float]],
    target_positions: Mapping[int, Sequence[float]],
    active_defenders_by_target: Mapping[int, int],
    *,
    standoff_distance: float = 350.0,
    lateral_spacing: float = 90.0,
    distance_scale: float = 1000.0,
) -> ObservableThreatPatrol:
    """Assign reserves to public-state threat-facing patrol waypoints.

    Every live Blue contributes one unit of threat, normalised across targets
    by inverse relative distance.  Reserves are then assigned greedily using
    ``threat / (1 + defender load) / (1 + travel / distance_scale)``.  The
    target-facing waypoint is offset laterally within each patrol group to
    prevent all reserves from collapsing onto the same coordinate.
    """

    reserves = tuple(sorted(int(value) for value in reserve_ids))
    if len(reserves) != len(set(reserves)):
        raise ValueError("reserve_ids cannot contain duplicate identities")
    targets = tuple(sorted(int(value) for value in target_positions))
    if not targets:
        raise ValueError("observable threat patrol requires at least one target")
    if not np.isfinite(standoff_distance) or float(standoff_distance) < 0.0:
        raise ValueError("standoff_distance must be non-negative and finite")
    if not np.isfinite(lateral_spacing) or float(lateral_spacing) < 0.0:
        raise ValueError("lateral_spacing must be non-negative and finite")
    if not np.isfinite(distance_scale) or float(distance_scale) <= 0.0:
        raise ValueError("distance_scale must be positive and finite")

    target_array = np.asarray(
        [target_positions[target] for target in targets], dtype=np.float64
    )
    if target_array.shape != (len(targets), 3) or not np.all(np.isfinite(target_array)):
        raise ValueError("target positions must be finite three-dimensional vectors")
    for agent_id in reserves:
        if agent_id not in red_positions:
            raise ValueError("every reserve identity needs a public Red position")
        position = np.asarray(red_positions[agent_id], dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("Red positions must be finite three-dimensional vectors")

    blue_ids = tuple(sorted(int(value) for value in blue_positions))
    if blue_ids:
        blue_array = np.asarray(
            [blue_positions[agent_id] for agent_id in blue_ids], dtype=np.float64
        )
        if blue_array.shape != (len(blue_ids), 3) or not np.all(np.isfinite(blue_array)):
            raise ValueError("Blue positions must be finite three-dimensional vectors")
        distances = np.linalg.norm(
            blue_array[:, None, :] - target_array[None, :, :], axis=2
        )
        inverse = 1.0 / np.maximum(distances, 1e-6)
        relative_threat = inverse / inverse.sum(axis=1, keepdims=True)
        threat = relative_threat.sum(axis=0)
    else:
        blue_array = np.empty((0, 3), dtype=np.float64)
        relative_threat = np.empty((0, len(targets)), dtype=np.float64)
        # The episode normally terminates before this branch.  Uniform values
        # retain a deterministic, total execution rule for direct unit tests.
        threat = np.ones(len(targets), dtype=np.float64)

    facing: Dict[int, np.ndarray] = {}
    for target_index, target_id in enumerate(targets):
        if blue_ids:
            weights = relative_threat[:, target_index]
            centroid = np.average(blue_array, axis=0, weights=weights)
            direction = centroid - target_array[target_index]
        else:
            direction = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        facing[target_id] = (
            direction / norm
            if norm > 1e-8
            else np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        )

    load = {
        target_id: max(0, int(active_defenders_by_target.get(target_id, 0)))
        for target_id in targets
    }
    target_for_reserve: Dict[int, int] = {}
    for agent_id in reserves:
        red_position = np.asarray(red_positions[agent_id], dtype=np.float64)
        scores = []
        for target_index, target_id in enumerate(targets):
            center = (
                target_array[target_index]
                + float(standoff_distance) * facing[target_id]
            )
            travel = float(np.linalg.norm(center - red_position))
            score = (
                float(threat[target_index])
                / (1.0 + float(load[target_id]))
                / (1.0 + travel / float(distance_scale))
            )
            scores.append((score, -target_id, target_id))
        chosen = max(scores)[2]
        target_for_reserve[agent_id] = chosen
        load[chosen] += 1

    waypoints: Dict[int, Tuple[float, float, float]] = {}
    for target_index, target_id in enumerate(targets):
        assigned = sorted(
            agent_id
            for agent_id, selected in target_for_reserve.items()
            if selected == target_id
        )
        direction = facing[target_id]
        lateral = np.asarray([-direction[1], direction[0], 0.0], dtype=np.float64)
        lateral_norm = float(np.linalg.norm(lateral))
        if lateral_norm <= 1e-8:
            lateral = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            lateral /= lateral_norm
        center = target_array[target_index] + float(standoff_distance) * direction
        midpoint = (len(assigned) - 1) / 2.0
        for rank, agent_id in enumerate(assigned):
            waypoint = center + (rank - midpoint) * float(lateral_spacing) * lateral
            waypoints[agent_id] = tuple(map(float, waypoint))

    return ObservableThreatPatrol(
        reserve_ids=reserves,
        target_by_agent=tuple(sorted(target_for_reserve.items())),
        waypoint_by_agent=tuple(sorted(waypoints.items())),
        target_threat=tuple(
            (target_id, float(threat[index]))
            for index, target_id in enumerate(targets)
        ),
    )


def build_observable_threat_patrol(
    adapter: HADStage3Adapter,
    reserve_ids: Sequence[int],
    active_red_assignment: Mapping[int, Optional[int]],
    *,
    standoff_distance: float = 350.0,
    lateral_spacing: float = 90.0,
    distance_scale: float = 1000.0,
) -> ObservableThreatPatrol:
    """Build a patrol using only alive positions and Red's own assignment.

    Blue's ``assigned_target`` field is deliberately never read.  The lower
    policy style and subgroup map are not arguments, which makes accidental
    information leakage visible at the interface boundary.
    """

    red_state = adapter.agent_states("Red")
    blue_state = adapter.agent_states("Blue")
    target_state = adapter.target_states()
    reserves = tuple(sorted(int(value) for value in reserve_ids))
    alive_red = {
        int(agent_id): np.asarray(value["position"], dtype=np.float64)
        for agent_id, value in red_state.items()
        if bool(value["alive"])
    }
    alive_blue = {
        int(agent_id): np.asarray(value["position"], dtype=np.float64)
        for agent_id, value in blue_state.items()
        if bool(value["alive"])
    }
    alive_targets = {
        int(target_id): np.asarray(value["position"], dtype=np.float64)
        for target_id, value in target_state.items()
        if bool(value["alive"])
    }
    if any(agent_id not in alive_red for agent_id in reserves):
        raise ValueError("reserve patrol contains a non-live Red identity")
    if any(active_red_assignment.get(agent_id) is not None for agent_id in reserves):
        raise ValueError("reserve patrol identities must retain formal target None")
    active_counts = {target_id: 0 for target_id in alive_targets}
    for agent_id, target_id in active_red_assignment.items():
        if (
            int(agent_id) in alive_red
            and target_id is not None
            and int(target_id) in active_counts
        ):
            active_counts[int(target_id)] += 1
    return observable_threat_patrol(
        reserves,
        alive_red,
        alive_blue,
        alive_targets,
        active_counts,
        standoff_distance=standoff_distance,
        lateral_spacing=lateral_spacing,
        distance_scale=distance_scale,
    )


def _nearest_alive_ids(
    adapter: HADStage3Adapter, side: str, target_id: int, count: int
) -> tuple[int, ...]:
    if count < 1:
        return ()
    target = np.asarray(adapter.target_states()[target_id]["position"], dtype=np.float64)
    states = adapter.agent_states(side)
    alive = [
        (agent_id, np.linalg.norm(np.asarray(value["position"]) - target))
        for agent_id, value in states.items()
        if bool(value["alive"])
    ]
    if len(alive) < count:
        raise ValueError(f"not enough alive {side} agents for a {count}-agent local query")
    alive.sort(key=lambda pair: (pair[1], pair[0]))
    return tuple(agent_id for agent_id, _ in alive[:count])


@dataclass(frozen=True)
class EventGameBuild:
    game: EventBlottoGame
    local_payoff: np.ndarray
    local_survival_probability: np.ndarray
    red_alive_ids: tuple[int, ...]
    blue_alive_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    direct_red_cap: int
    direct_blue_cap: int


@dataclass(frozen=True)
class GroupedEventGameBuild:
    """S2-backed anonymous group-pattern game at one command event."""

    game: GroupedBlottoGame
    local_group_payoff: np.ndarray
    local_group_survival_probability: np.ndarray
    red_alive_ids: tuple[int, ...]
    blue_alive_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    red_group_size_cap: int
    blue_group_size_cap: int


def build_grouped_event_game(
    adapter: HADStage3Adapter,
    predictor: FrozenStage2Payoff,
    *,
    group_size_cap: int = 8,
    blue_style: Optional[str] = None,
    batch_size: int = 512,
    allow_defender_reserve: bool = False,
    allow_attacker_reserve: bool = False,
    utility_mode: str = "joint_survival_log_probability",
    risk_epsilon: float = 1e-6,
) -> GroupedEventGameBuild:
    """Build a target-pattern game that queries S2 only for one group pair.

    Unlike :func:`build_event_game`, target totals are not queried as one
    monolithic local battle.  Stage 2 supplies the payoff of one ``r v b``
    group pair for ``0 <= r,b <= group_size_cap``; Stage 3 decides how many
    such groups to create at every target.
    """

    requested_cap = int(group_size_cap)
    if requested_cap < 1:
        raise ValueError("group_size_cap must be positive")
    red_alive = tuple(
        agent_id
        for agent_id, state in adapter.agent_states("Red").items()
        if bool(state["alive"])
    )
    blue_alive = tuple(
        agent_id
        for agent_id, state in adapter.agent_states("Blue").items()
        if bool(state["alive"])
    )
    if not red_alive or not blue_alive:
        raise ValueError("a grouped event game requires live agents on both sides")
    target_ids = tuple(adapter.target_ids)
    if not target_ids:
        raise ValueError("a grouped event game requires at least one target")
    red_cap = min(requested_cap, len(red_alive))
    blue_cap = min(requested_cap, len(blue_alive))
    canonicalizer = HADCanonicalizer()

    def state_factory(target_index: int, red: int, blue: int):
        target_id = target_ids[target_index]
        red_ids = _nearest_alive_ids(adapter, "Red", target_id, red)
        blue_ids = _nearest_alive_ids(adapter, "Blue", target_id, blue)
        entities = adapter.local_state_entities(
            target_id,
            red_ids,
            blue_ids,
            local_step=adapter.step_count,
        )
        return canonicalizer.to_entity_set(entities)

    style = adapter.blue_rule_style if blue_style is None else str(blue_style)
    payoff = build_local_payoff_tensor(
        len(target_ids),
        red_cap,
        blue_cap,
        state_factory,
        predictor,
        blue_style=style,
        batch_size=int(batch_size),
        direct_red_cap=requested_cap,
        direct_blue_cap=requested_cap,
        utility_mode=str(utility_mode),
        risk_epsilon=float(risk_epsilon),
        enforce_monotonicity=False,
    )
    # In the grouped game, one target can contain many paired groups.  The
    # normalized learned log-survival contribution of every non-structural
    # pair lies in [-1, 0], and there can be at most NR+NB such pairs.  A true
    # 0vB breach represents log(0)=-infinity, so use a finite solver penalty
    # below the worst possible all-nonempty profile instead of the legacy
    # target-count penalty -(M+1).
    payoff[:, 0, 1:] = -(len(red_alive) + len(blue_alive) + 1.0)
    survival = utility_to_survival_probability(
        payoff,
        utility_mode=str(utility_mode),
        risk_epsilon=float(risk_epsilon),
    )
    if utility_mode == "joint_survival_log_probability":
        survival[:, 0, 1:] = 0.0
    game = GroupedBlottoGame(
        payoff,
        defender_budget=len(red_alive),
        attacker_budget=len(blue_alive),
        allow_defender_reserve=bool(allow_defender_reserve),
        allow_attacker_reserve=bool(allow_attacker_reserve),
        event_id=f"had-grouped-step-{adapter.step_count}",
        pairing_rule="size_assortative",
    )
    return GroupedEventGameBuild(
        game=game,
        local_group_payoff=payoff,
        local_group_survival_probability=survival,
        red_alive_ids=red_alive,
        blue_alive_ids=blue_alive,
        target_ids=target_ids,
        red_group_size_cap=red_cap,
        blue_group_size_cap=blue_cap,
    )


def build_event_game(
    adapter: HADStage3Adapter,
    predictor: FrozenStage2Payoff,
    *,
    direct_red_cap: Optional[int] = None,
    direct_blue_cap: Optional[int] = None,
    blue_style: Optional[str] = None,
    batch_size: int = 512,
    allow_defender_reserve: bool = True,
    allow_attacker_reserve: bool = True,
    utility_mode: str = "centered_red_win_probability",
    risk_epsilon: float = 0.01,
    minimum_defenders_per_target: int = 0,
    defender_reallocation_penalty: float = 0.0,
    attacker_reallocation_penalty: float = 0.0,
    enforce_payoff_monotonicity: bool = False,
    defender_balance_regularization: float = 0.0,
) -> EventGameBuild:
    """Construct the current event's S2-backed additive Blotto surrogate.

    Reserve remains available by default for backwards compatibility with the
    abstract solver experiments.  Shared-world HAD evaluations can disable it
    for either side so every live resource belongs to exactly one battlefield.
    """

    red_alive = tuple(
        agent_id
        for agent_id, state in adapter.agent_states("Red").items()
        if bool(state["alive"])
    )
    blue_alive = tuple(
        agent_id
        for agent_id, state in adapter.agent_states("Blue").items()
        if bool(state["alive"])
    )
    if not red_alive or not blue_alive:
        raise ValueError("an event game requires at least one alive agent per side")
    target_ids = tuple(adapter.target_ids)
    if not target_ids:
        raise ValueError("an event game requires at least one active target")
    # Counts are capped only by the resources that are actually alive.  The
    # S2 direct grid controls how payoff values are estimated, not which
    # Blotto actions are legal.
    local_red_cap = len(red_alive)
    local_blue_cap = len(blue_alive)
    stage2_red_support = (
        local_red_cap if direct_red_cap is None else int(direct_red_cap)
    )
    stage2_blue_support = (
        local_blue_cap if direct_blue_cap is None else int(direct_blue_cap)
    )
    queried_red_cap = local_red_cap
    queried_blue_cap = local_blue_cap
    canonicalizer = HADCanonicalizer()

    def state_factory(target_index: int, red: int, blue: int):
        target_id = target_ids[target_index]
        red_ids = _nearest_alive_ids(adapter, "Red", target_id, red)
        blue_ids = _nearest_alive_ids(adapter, "Blue", target_id, blue)
        entities = adapter.local_state_entities(
            target_id,
            red_ids,
            blue_ids,
            local_step=adapter.step_count,
        )
        return canonicalizer.to_entity_set(entities)

    style = adapter.blue_rule_style if blue_style is None else blue_style
    payoff = build_local_payoff_tensor(
        len(target_ids),
        local_red_cap,
        local_blue_cap,
        state_factory,
        predictor,
        blue_style=style,
        batch_size=batch_size,
        direct_red_cap=stage2_red_support,
        direct_blue_cap=stage2_blue_support,
        utility_mode=utility_mode,
        risk_epsilon=float(risk_epsilon),
        enforce_monotonicity=bool(enforce_payoff_monotonicity),
    )
    local_survival = utility_to_survival_probability(
        payoff,
        utility_mode=utility_mode,
        risk_epsilon=float(risk_epsilon),
    )
    # Preserve the exact registered boundary in diagnostics.  Inverting the
    # finite structural-breach penalty above yields a tiny positive numerical
    # value, but the physical assumption itself is p=0.
    if utility_mode == "joint_survival_log_probability":
        local_survival[:, 0, 1:] = 0.0
    requested_minimum_defenders = int(minimum_defenders_per_target)
    if requested_minimum_defenders < 0:
        raise ValueError("minimum_defenders_per_target must be non-negative")
    # Casualties can make full coverage infeasible mid-episode.  Relax only
    # the impossible part instead of crashing the event planner.
    minimum_defenders = min(
        requested_minimum_defenders, len(red_alive) // len(target_ids)
    )
    for name, value in (
        ("defender_reallocation_penalty", defender_reallocation_penalty),
        ("attacker_reallocation_penalty", attacker_reallocation_penalty),
        ("defender_balance_regularization", defender_balance_regularization),
    ):
        if not np.isfinite(value) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")

    # Reallocation friction is evaluated against the previous command and is
    # separable by battlefield, so the Blotto DP oracle remains exact.  The
    # first deployment is not penalized.
    red_previous = np.asarray(
        [len(adapter.assigned_ids("Red", target_id, alive_only=True)) for target_id in target_ids],
        dtype=np.float64,
    )
    blue_previous = np.asarray(
        [len(adapter.assigned_ids("Blue", target_id, alive_only=True)) for target_id in target_ids],
        dtype=np.float64,
    )
    red_has_previous = bool(red_previous.sum() > 0.0)
    blue_has_previous = bool(blue_previous.sum() > 0.0)
    if red_has_previous and float(defender_reallocation_penalty) > 0.0:
        red_counts = np.arange(local_red_cap + 1, dtype=np.float64)[None, :, None]
        payoff -= (
            float(defender_reallocation_penalty)
            * np.abs(red_counts - red_previous[:, None, None])
            / max(1, len(red_alive))
        )
    if blue_has_previous and float(attacker_reallocation_penalty) > 0.0:
        blue_counts = np.arange(local_blue_cap + 1, dtype=np.float64)[None, None, :]
        payoff += (
            float(attacker_reallocation_penalty)
            * np.abs(blue_counts - blue_previous[:, None, None])
            / max(1, len(blue_alive))
        )
    if float(defender_balance_regularization) > 0.0:
        red_counts = np.arange(local_red_cap + 1, dtype=np.float64)[None, :, None]
        equal_coverage = len(red_alive) / len(target_ids)
        payoff -= (
            float(defender_balance_regularization)
            * ((red_counts - equal_coverage) / max(1, len(red_alive))) ** 2
        )
    defender_caps = np.full(len(target_ids), local_red_cap, dtype=np.int64)
    attacker_caps = np.full(len(target_ids), local_blue_cap, dtype=np.int64)
    defender_count_mask = np.ones(
        (len(target_ids), local_red_cap + 1), dtype=bool
    )
    if minimum_defenders:
        defender_count_mask[:, :minimum_defenders] = False
    game = EventBlottoGame(
        payoff,
        defender_budget=len(red_alive),
        attacker_budget=len(blue_alive),
        defender_caps=defender_caps,
        attacker_caps=attacker_caps,
        allow_defender_reserve=bool(allow_defender_reserve),
        allow_attacker_reserve=bool(allow_attacker_reserve),
        defender_count_mask=defender_count_mask,
        event_id=f"had-step-{adapter.step_count}",
    )
    return EventGameBuild(
        game,
        payoff,
        local_survival,
        red_alive,
        blue_alive,
        target_ids,
        queried_red_cap,
        queried_blue_cap,
    )


def _cost_matrix(
    adapter: HADStage3Adapter,
    side: str,
    alive_ids: Sequence[int],
    *,
    switch_cost: float,
) -> np.ndarray:
    speed = 250.0 if side == "Red" else 300.0
    states = adapter.agent_states(side)
    targets = adapter.target_states()
    result = np.zeros((len(alive_ids), len(adapter.target_ids)), dtype=np.float64)
    for row, agent_id in enumerate(alive_ids):
        state = states[int(agent_id)]
        position = np.asarray(state["position"], dtype=np.float64)
        for column, target_id in enumerate(adapter.target_ids):
            target = np.asarray(targets[target_id]["position"], dtype=np.float64)
            result[row, column] = np.linalg.norm(target - position) / speed
            previous = state.get("assigned_target")
            if previous is not None and previous != target_id:
                result[row, column] += float(switch_cost)
    return result


def ground_count_allocation(
    adapter: HADStage3Adapter,
    side: str,
    allocation: Sequence[int],
    *,
    switch_cost: float = 0.25,
) -> tuple[Dict[int, Optional[int]], AgentTaskMatching | None]:
    """Map count allocation to stable HAD IDs using ETA-aware min-cost flow."""

    all_ids = adapter.red_ids if side == "Red" else adapter.blue_ids
    states = adapter.agent_states(side)
    alive_ids = tuple(agent_id for agent_id in all_ids if bool(states[agent_id]["alive"]))
    result: Dict[int, Optional[int]] = {agent_id: None for agent_id in all_ids}
    if not alive_ids:
        return result, None
    demand = tuple(int(value) for value in allocation)
    if len(demand) != len(adapter.target_ids):
        raise ValueError("allocation must have one count per target")
    if sum(demand) > len(alive_ids):
        raise ValueError("allocation uses more agents than are alive")
    costs = _cost_matrix(
        adapter, side, alive_ids, switch_cost=float(switch_cost)
    )
    matching = match_agents_to_tasks(
        costs,
        demand,
        capacities=demand,
        reserve_cost=0.0,
        agent_ids=alive_ids,
        task_ids=adapter.target_ids,
    )
    result.update(matching.assignment_by_agent())
    return result, matching


@dataclass(frozen=True)
class LocalSubgame:
    """One concrete target-centred S1 subgame produced by an upper action."""

    target_id: int
    subgroup_id: int
    red_ids: Tuple[int, ...]
    blue_ids: Tuple[int, ...]


@dataclass(frozen=True)
class GroundedJointBlottoPlan:
    """Executable joint profile used by simulation and scientific diagnostics."""

    blue_type_name: str
    red_allocation: Allocation
    blue_allocation: Allocation
    red_assignment: Tuple[Tuple[int, Optional[int]], ...]
    blue_assignment: Tuple[Tuple[int, Optional[int]], ...]
    local_subgames: Tuple[LocalSubgame, ...]
    red_matching: AgentTaskMatching
    blue_matching: AgentTaskMatching


@dataclass(frozen=True)
class GroundedJointGroupedPlan:
    """Executable joint profile of the coalition-structured Blotto game."""

    blue_type_name: str
    red_grouped_allocation: GroupedAllocation
    blue_grouped_allocation: GroupedAllocation
    red_allocation: Allocation
    blue_allocation: Allocation
    red_assignment: Tuple[Tuple[int, Optional[int]], ...]
    blue_assignment: Tuple[Tuple[int, Optional[int]], ...]
    local_subgames: Tuple[LocalSubgame, ...]
    red_matching: AgentTaskMatching
    blue_matching: AgentTaskMatching


def _partition_ordered_ids(
    values: Sequence[int], group_sizes: Sequence[int]
) -> Tuple[Tuple[int, ...], ...]:
    if sum(map(int, group_sizes)) != len(values):
        raise ValueError("group sizes do not cover the grounded target roster")
    chunks: list[tuple[int, ...]] = []
    cursor = 0
    for raw_size in group_sizes:
        size = int(raw_size)
        chunks.append(tuple(int(value) for value in values[cursor : cursor + size]))
        cursor += size
    return tuple(chunks)


def _ordered_target_ids(
    adapter: HADStage3Adapter, side: str, target_id: int
) -> Tuple[int, ...]:
    states = adapter.agent_states(side)
    target = np.asarray(adapter.target_states()[target_id]["position"], dtype=np.float64)
    values = list(adapter.assigned_ids(side, target_id, alive_only=True))
    values.sort(
        key=lambda agent_id: (
            float(np.asarray(states[agent_id]["position"])[1] - target[1]),
            float(np.asarray(states[agent_id]["position"])[0] - target[0]),
            int(agent_id),
        )
    )
    return tuple(values)


def apply_joint_grouped_plan(
    adapter: HADStage3Adapter,
    game: GroupedBlottoGame,
    red_grouped_allocation: Sequence[int],
    blue_grouped_allocation: Sequence[int],
    *,
    blue_type_name: str,
    switch_cost: float = 0.25,
) -> GroundedJointGroupedPlan:
    """Ground two simultaneous anonymous group actions to concrete HAD IDs.

    Each player chooses only its own target-labelled group histogram.  The
    public size-assortative rule then pairs the two canonical group sequences;
    the returned ``local_subgames`` are the requested
    ``(target, Red IDs, Blue IDs)`` joint outcome.
    """

    if tuple(adapter.target_ids) != tuple(range(game.n_targets)):
        # HAD currently numbers the fixed episode targets contiguously.  Keep
        # this assertion explicit so a future sparse-ID adapter cannot silently
        # misinterpret target-major action coordinates.
        raise ValueError("grouped game target order differs from the HAD adapter")
    red_grouped = game.validate_defender_allocation(red_grouped_allocation)
    blue_grouped = game.validate_attacker_allocation(blue_grouped_allocation)
    red_counts = game.target_counts(red_grouped, "Red")
    blue_counts = game.target_counts(blue_grouped, "Blue")
    red_assignment, red_matching = ground_count_allocation(
        adapter, "Red", red_counts, switch_cost=float(switch_cost)
    )
    blue_assignment, blue_matching = ground_count_allocation(
        adapter, "Blue", blue_counts, switch_cost=float(switch_cost)
    )
    if red_matching is None or blue_matching is None:
        raise ValueError("a grouped joint plan requires live agents on both sides")
    adapter.set_joint_assignments(red_assignment, blue_assignment)

    red_groups = game.groups(red_grouped, "Red")
    blue_groups = game.groups(blue_grouped, "Blue")
    local_subgames: list[LocalSubgame] = []
    for target_index, target_id in enumerate(adapter.target_ids):
        red_chunks = _partition_ordered_ids(
            _ordered_target_ids(adapter, "Red", target_id), red_groups[target_index]
        )
        blue_chunks = _partition_ordered_ids(
            _ordered_target_ids(adapter, "Blue", target_id), blue_groups[target_index]
        )
        total_groups = max(len(red_chunks), len(blue_chunks))
        for subgroup_id in range(total_groups):
            local_subgames.append(
                LocalSubgame(
                    target_id=int(target_id),
                    subgroup_id=int(subgroup_id),
                    red_ids=(
                        red_chunks[subgroup_id]
                        if subgroup_id < len(red_chunks)
                        else ()
                    ),
                    blue_ids=(
                        blue_chunks[subgroup_id]
                        if subgroup_id < len(blue_chunks)
                        else ()
                    ),
                )
            )

    used_red = [agent_id for item in local_subgames for agent_id in item.red_ids]
    used_blue = [agent_id for item in local_subgames for agent_id in item.blue_ids]
    if len(used_red) != len(set(used_red)) or len(used_blue) != len(set(used_blue)):
        raise RuntimeError("grouped grounding assigned one live identity more than once")
    expected_red = {
        agent_id
        for agent_id, state in adapter.agent_states("Red").items()
        if bool(state["alive"]) and dict(red_assignment)[agent_id] is not None
    }
    expected_blue = {
        agent_id
        for agent_id, state in adapter.agent_states("Blue").items()
        if bool(state["alive"]) and dict(blue_assignment)[agent_id] is not None
    }
    if set(used_red) != expected_red or set(used_blue) != expected_blue:
        raise RuntimeError("grouped grounding did not cover the deployed identities")

    return GroundedJointGroupedPlan(
        blue_type_name=str(blue_type_name),
        red_grouped_allocation=red_grouped,
        blue_grouped_allocation=blue_grouped,
        red_allocation=red_counts,
        blue_allocation=blue_counts,
        red_assignment=tuple(sorted(red_assignment.items())),
        blue_assignment=tuple(sorted(blue_assignment.items())),
        local_subgames=tuple(local_subgames),
        red_matching=red_matching,
        blue_matching=blue_matching,
    )


def apply_joint_blotto_plan(
    adapter: HADStage3Adapter,
    executor: FrozenStage1GroupExecutor,
    red_allocation: Sequence[int],
    blue_allocation: Sequence[int],
    *,
    blue_type_name: str,
    switch_cost: float = 0.25,
) -> GroundedJointBlottoPlan:
    """Ground both count actions and expose the concrete local subgames.

    Red and Blue counts are chosen before either assignment is installed.
    Identity-level min-cost matching then respects exclusivity across targets,
    and the type-conditioned executor turns the resulting joint assignment
    into explicit target/Red-ID/Blue-ID subgames.
    """

    red_counts = tuple(int(value) for value in red_allocation)
    blue_counts = tuple(int(value) for value in blue_allocation)
    red_assignment, red_matching = ground_count_allocation(
        adapter, "Red", red_counts, switch_cost=float(switch_cost)
    )
    blue_assignment, blue_matching = ground_count_allocation(
        adapter, "Blue", blue_counts, switch_cost=float(switch_cost)
    )
    if red_matching is None or blue_matching is None:
        raise ValueError("a joint Blotto plan requires live agents on both sides")
    adapter.set_joint_assignments(red_assignment, blue_assignment)
    local_subgames = tuple(
        LocalSubgame(
            target_id=int(target_id),
            subgroup_id=int(subgroup_id),
            red_ids=tuple(red_ids),
            blue_ids=tuple(blue_ids),
        )
        for (target_id, subgroup_id), (red_ids, blue_ids) in sorted(
            executor.micro_rosters(adapter).items()
        )
    )
    return GroundedJointBlottoPlan(
        blue_type_name=str(blue_type_name),
        red_allocation=red_counts,
        blue_allocation=blue_counts,
        red_assignment=tuple(sorted(red_assignment.items())),
        blue_assignment=tuple(sorted(blue_assignment.items())),
        local_subgames=local_subgames,
        red_matching=red_matching,
        blue_matching=blue_matching,
    )


def balanced_allocation(budget: int, caps: Sequence[int], offset: int = 0) -> Allocation:
    caps_array = np.asarray(caps, dtype=np.int64)
    result = np.zeros(len(caps_array), dtype=np.int64)
    remaining = min(int(budget), int(caps_array.sum()))
    while remaining > 0:
        changed = False
        for step in range(len(result)):
            index = (int(offset) + step) % len(result)
            if result[index] < caps_array[index]:
                result[index] += 1
                remaining -= 1
                changed = True
                if remaining == 0:
                    break
        if not changed:
            break
    return tuple(map(int, result))


def concentrated_allocation(
    budget: int, caps: Sequence[int], first_target: int = 0
) -> Allocation:
    caps_array = np.asarray(caps, dtype=np.int64)
    result = np.zeros(len(caps_array), dtype=np.int64)
    remaining = min(int(budget), int(caps_array.sum()))
    for step in range(len(result)):
        index = (int(first_target) + step) % len(result)
        assigned = min(remaining, int(caps_array[index]))
        result[index] = assigned
        remaining -= assigned
        if remaining == 0:
            break
    return tuple(map(int, result))


def moderate_jittered_target_positions(
    targets: int,
    generator: Mapping[str, float],
    *,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate the registered fixed, separated target layout for one episode."""

    if int(targets) < 1:
        raise ValueError("targets must be positive")
    spacing = float(generator.get("target_y_spacing", 650.0))
    jitter = float(generator.get("target_y_jitter", 50.0))
    if spacing <= 0.0 or jitter < 0.0 or 2.0 * jitter >= spacing:
        raise ValueError("moderate target spacing/jitter is invalid")
    y = spacing * (
        np.arange(int(targets), dtype=np.float64) - (int(targets) - 1) / 2.0
    )
    rng = np.random.default_rng(0 if seed is None else int(seed) ^ 0x7A63E7)
    y += rng.uniform(-jitter, jitter, size=int(targets))
    x = np.full(int(targets), float(generator["x_center"]), dtype=np.float64)
    z = np.full(int(targets), float(generator["z"]), dtype=np.float64)
    return np.stack((x, y, z), axis=1)


def eta_greedy_defender_allocation(
    adapter: HADStage3Adapter, budget: int, caps: Sequence[int]
) -> Allocation:
    """Public-state baseline using Blue distance, not Blue's new hidden command."""

    caps_array = np.asarray(caps, dtype=np.int64)
    counts = np.zeros(len(caps_array), dtype=np.int64)
    targets = adapter.target_states()
    target_to_index = {
        target_id: index for index, target_id in enumerate(adapter.target_ids)
    }
    for state in adapter.agent_states("Blue").values():
        if not state["alive"]:
            continue
        position = np.asarray(state["position"], dtype=np.float64)
        target_id = min(
            adapter.target_ids,
            key=lambda target: np.linalg.norm(
                np.asarray(targets[target]["position"], dtype=np.float64) - position
            ),
        )
        counts[target_to_index[target_id]] += 1
    allocation = np.zeros_like(caps_array)
    for _ in range(min(int(budget), int(caps_array.sum()))):
        scores = np.where(
            allocation < caps_array,
            (counts + 1.0) / (allocation + 1.0),
            -np.inf,
        )
        target = int(np.argmax(scores))
        if not np.isfinite(scores[target]):
            break
        allocation[target] += 1
    return tuple(map(int, allocation))


def enumerate_feasible_allocations(
    budget: int,
    caps: Sequence[int],
    *,
    allow_reserve: bool,
    count_mask: Optional[np.ndarray] = None,
) -> tuple[Allocation, ...]:
    """Exact enumeration for small validation/baseline games only."""

    caps_tuple = tuple(int(value) for value in caps)
    if count_mask is None:
        allowed = np.ones((len(caps_tuple), max(caps_tuple) + 1), dtype=bool)
    else:
        allowed = np.asarray(count_mask)
        if allowed.dtype != np.bool_ or allowed.ndim != 2 or allowed.shape[0] != len(
            caps_tuple
        ):
            raise ValueError("count_mask must be boolean with one row per battlefield")
        if allowed.shape[1] <= max(caps_tuple):
            raise ValueError("count_mask does not cover every local cap")
    result: list[Allocation] = []

    def visit(index: int, remaining: int, prefix: tuple[int, ...]) -> None:
        if index == len(caps_tuple):
            if allow_reserve or remaining == 0:
                result.append(prefix)
            return
        for value in range(min(caps_tuple[index], remaining) + 1):
            if not allowed[index, value]:
                continue
            visit(index + 1, remaining - value, prefix + (value,))

    visit(0, int(budget), ())
    return tuple(result)


def pure_maximin_allocation(game: EventBlottoGame) -> Allocation:
    """Pure maximin baseline for the registered 30--50-agent demonstrations."""

    defenders = enumerate_feasible_allocations(
        game.defender_budget,
        game.defender_caps,
        allow_reserve=game.allow_defender_reserve,
        count_mask=game.defender_count_mask,
    )
    attackers = enumerate_feasible_allocations(
        game.attacker_budget,
        game.attacker_caps,
        allow_reserve=game.allow_attacker_reserve,
        count_mask=game.attacker_count_mask,
    )
    matrix = game.payoff_matrix(defenders, attackers)
    values = matrix.min(axis=1)
    return defenders[int(np.argmax(values))]


__all__ = [
    "EventGameBuild",
    "FrozenStage1GroupExecutor",
    "ObservableThreatPatrol",
    "balanced_allocation",
    "build_observable_threat_patrol",
    "build_event_game",
    "concentrated_allocation",
    "enumerate_feasible_allocations",
    "eta_greedy_defender_allocation",
    "ground_count_allocation",
    "load_round01_stage1_model",
    "observable_threat_patrol",
    "pure_maximin_allocation",
]
