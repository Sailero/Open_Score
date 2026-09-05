"""Identity-aware coalition-structured Colonel Blotto.

Each player simultaneously partitions its *labelled* live agents over public
``(target, channel)`` engagement slots.  Red may leave identities in an
explicit reserve, whereas Blue identities remain exactly partitioned over
the slots.  Agents placed in the same slot are teammates; the Red and Blue
coalitions occupying the same slot form one local subgame.  Multiple channels
may share a target, so the joint profile directly produces
``(Red IDs, Blue IDs, target)`` records without letting either player choose
the opponent's action.

For a fixed opponent mixture the payoff is additive over slots.  A best
response is therefore a weighted set-partitioning MILP.  Double Oracle avoids
enumerating the exponentially many complete partitions.  Small experiments
can register every coalition up to ``K`` and obtain a full-game certificate;
large experiments may register an explicit anytime candidate family and must
report that weaker guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .blotto import solve_restricted_matrix_game


Coalition = Tuple[int, ...]
LocalPayoffRequest = Tuple[int, Coalition, Coalition]


@dataclass(frozen=True, order=True)
class EngagementSlot:
    target_id: int
    channel_id: int


@dataclass(frozen=True)
class IdentityAction:
    """One side's labelled slot assignment and explicit reserve identities."""

    coalitions: Tuple[Coalition, ...]
    reserve_ids: Coalition = ()


class IdentityLocalPayoffOracle(Protocol):
    """Batched payoff interface for non-empty local coalition pairs."""

    def evaluate(self, requests: Sequence[LocalPayoffRequest]) -> np.ndarray:
        ...


@dataclass(frozen=True)
class IdentityBestResponse:
    action: IdentityAction
    value: float
    optimal: bool
    solver_status: int
    solver_message: str
    mip_gap: float
    mip_node_count: int
    variables: int


@dataclass(frozen=True)
class IdentityDoubleOracleIteration:
    iteration: int
    red_support: int
    blue_support: int
    restricted_value: float
    red_best_response_value: float
    blue_best_response_value: float
    red_gap: float
    blue_gap: float
    exploitability: float
    red_oracle_optimal: bool
    blue_oracle_optimal: bool
    added_red: bool
    added_blue: bool


@dataclass(frozen=True)
class IdentityDoubleOracleResult:
    red_strategies: Tuple[IdentityAction, ...]
    blue_strategies: Tuple[IdentityAction, ...]
    red_mixture: np.ndarray
    blue_mixture: np.ndarray
    value: float
    exploitability: float
    converged: bool
    candidate_domain_exact: bool
    full_game_certified: bool
    termination_reason: str
    history: Tuple[IdentityDoubleOracleIteration, ...]

    def sample_profile(
        self, *, rng: np.random.Generator
    ) -> tuple[IdentityAction, IdentityAction]:
        red = int(rng.choice(len(self.red_strategies), p=self.red_mixture))
        blue = int(rng.choice(len(self.blue_strategies), p=self.blue_mixture))
        return self.red_strategies[red], self.blue_strategies[blue]


def _coalition(values: Iterable[int]) -> Coalition:
    result = tuple(sorted(int(value) for value in values))
    if len(result) != len(set(result)):
        raise ValueError("a coalition cannot contain a duplicate identity")
    return result


def enumerate_coalitions(
    agent_ids: Sequence[int], max_group_size: int = 4
) -> Tuple[Coalition, ...]:
    """Enumerate every non-empty labelled coalition up to ``max_group_size``."""

    ids = _coalition(agent_ids)
    cap = int(max_group_size)
    if cap < 1:
        raise ValueError("max_group_size must be positive")
    return tuple(
        tuple(group)
        for size in range(1, min(cap, len(ids)) + 1)
        for group in combinations(ids, size)
    )


def make_engagement_slots(
    target_ids: Sequence[int],
    maximum_side_agents: int,
    max_group_size: int = 4,
    *,
    channels_per_target: Optional[int] = None,
) -> Tuple[EngagementSlot, ...]:
    """Create the complete public engagement-channel set.

    ``N`` channels per target are necessary because a legal pure action may
    concentrate all ``N`` agents at one target as singleton coalitions.  Using
    only ``ceil(N / 4)`` channels would silently exclude such identity-level
    partitions even though every individual coalition respects the 4-agent
    cap.
    """

    targets = tuple(int(value) for value in target_ids)
    if not targets or len(targets) != len(set(targets)):
        raise ValueError("target_ids must be a non-empty unique sequence")
    if int(maximum_side_agents) < 1 or int(max_group_size) < 1:
        raise ValueError("agent and group limits must be positive")
    channels = (
        int(maximum_side_agents)
        if channels_per_target is None
        else int(channels_per_target)
    )
    if channels < 1:
        raise ValueError("channels_per_target must be positive")
    return tuple(
        EngagementSlot(target_id, channel)
        for target_id in targets
        for channel in range(channels)
    )


def round_robin_identity_action(
    agent_ids: Sequence[int],
    slots: Sequence[EngagementSlot],
    *,
    max_group_size: int = 4,
    concentrated_target: Optional[int] = None,
    offset: int = 0,
) -> IdentityAction:
    """Construct a deterministic feasible seed action."""

    ids = list(_coalition(agent_ids))
    cap = int(max_group_size)
    slot_values = tuple(slots)
    target_ids = tuple(dict.fromkeys(slot.target_id for slot in slot_values))
    if concentrated_target is not None and int(concentrated_target) not in target_ids:
        raise ValueError("concentrated_target is not registered")
    buckets: Dict[int, list[int]] = {target: [] for target in target_ids}
    if concentrated_target is None:
        for index, agent_id in enumerate(ids):
            target = target_ids[(int(offset) + index) % len(target_ids)]
            buckets[target].append(agent_id)
    else:
        buckets[int(concentrated_target)].extend(ids)
    groups_by_target: Dict[int, list[Coalition]] = {}
    for target, values in buckets.items():
        groups_by_target[target] = [
            _coalition(values[start : start + cap])
            for start in range(0, len(values), cap)
        ]
    coalitions: list[Coalition] = []
    used_channel: Dict[int, int] = {target: 0 for target in target_ids}
    for slot in slot_values:
        index = used_channel[slot.target_id]
        groups = groups_by_target[slot.target_id]
        coalitions.append(groups[index] if index < len(groups) else ())
        used_channel[slot.target_id] += 1
    if any(used_channel[target] < len(groups) for target, groups in groups_by_target.items()):
        raise ValueError("registered slots cannot hold this seed action")
    return IdentityAction(tuple(coalitions))


def enumerate_identity_actions(
    agent_ids: Sequence[int],
    slots: Sequence[EngagementSlot],
    *,
    max_group_size: int = 4,
    allow_reserve: bool = False,
) -> Tuple[IdentityAction, ...]:
    """Enumerate the complete pure-action domain for a *small* identity game.

    Each labelled agent chooses exactly one public slot, or (when
    ``allow_reserve`` is true) the explicit reserve, and every slot has
    capacity ``max_group_size``.  The routine is intentionally exponential
    and exists only for toy-game verification of the Double-Oracle solver.
    Large HAD events use the set-partitioning oracle directly.
    """

    ids = _coalition(agent_ids)
    slot_values = tuple(slots)
    cap = int(max_group_size)
    if cap != 4:
        raise ValueError("the identity-aware MVP is registered at max_group_size=4")
    if not slot_values:
        raise ValueError("at least one engagement slot is required")
    if not allow_reserve and len(ids) > len(slot_values) * cap:
        raise ValueError("the slot set cannot hold every identity")
    buckets: list[list[int]] = [[] for _ in slot_values]
    reserve: list[int] = []
    result: list[IdentityAction] = []

    def visit(index: int) -> None:
        if index == len(ids):
            for target in dict.fromkeys(slot.target_id for slot in slot_values):
                target_indices = sorted(
                    (
                        slot_index
                        for slot_index, slot in enumerate(slot_values)
                        if slot.target_id == target
                    ),
                    key=lambda slot_index: slot_values[slot_index].channel_id,
                )
                sizes = [len(buckets[slot_index]) for slot_index in target_indices]
                if any(
                    sizes[position] > sizes[position - 1]
                    for position in range(1, len(sizes))
                ):
                    return
            result.append(
                IdentityAction(
                    tuple(_coalition(values) for values in buckets),
                    _coalition(reserve),
                )
            )
            return
        agent_id = ids[index]
        for slot in range(len(slot_values)):
            if len(buckets[slot]) >= cap:
                continue
            buckets[slot].append(agent_id)
            visit(index + 1)
            buckets[slot].pop()
        if allow_reserve:
            reserve.append(agent_id)
            visit(index + 1)
            reserve.pop()

    visit(0)
    return tuple(result)


def _position_array(
    agent_ids: Sequence[int], positions: Mapping[int, Sequence[float]]
) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for agent_id in agent_ids:
        if int(agent_id) not in positions:
            raise ValueError(f"missing position for agent {agent_id}")
        value = np.asarray(positions[int(agent_id)], dtype=np.float64)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError("agent positions must be finite three-vectors")
        result[int(agent_id)] = value
    return result


def spatial_candidate_coalitions(
    agent_ids: Sequence[int],
    slots: Sequence[EngagementSlot],
    agent_positions: Mapping[int, Sequence[float]],
    target_positions: Mapping[int, Sequence[float]],
    *,
    max_group_size: int = 4,
    neighborhood_size: int = 10,
    peer_count: int = 6,
    full_domain: bool = False,
    seed_actions: Sequence[IdentityAction] = (),
) -> Tuple[Tuple[Coalition, ...], ...]:
    """Build deterministic identity-aware coalition columns for every slot.

    In full-domain mode every labelled coalition up to size four is available
    in every slot.  The scalable mode contains all singletons, all coalitions
    among agents nearest each target, and local peer coalitions around every
    agent.  Seed-action coalitions are always retained, guaranteeing that the
    registered initial strategies remain feasible.
    """

    ids = _coalition(agent_ids)
    cap = int(max_group_size)
    if cap != 4:
        raise ValueError("the identity-aware MVP is registered at max_group_size=4")
    if not full_domain and (
        neighborhood_size < min(cap, len(ids))
        or peer_count < min(cap - 1, max(0, len(ids) - 1))
    ):
        raise ValueError("candidate neighborhoods are too small for 4-agent groups")
    slot_values = tuple(slots)
    position = _position_array(ids, agent_positions)
    target_position = {
        int(target): np.asarray(value, dtype=np.float64)
        for target, value in target_positions.items()
    }
    if full_domain:
        complete = enumerate_coalitions(ids, cap)
        return tuple(complete for _ in slot_values)

    seeded: dict[int, set[Coalition]] = {
        index: set() for index in range(len(slot_values))
    }
    for action in seed_actions:
        if len(action.coalitions) != len(slot_values):
            raise ValueError("seed action has the wrong slot count")
        for index, coalition in enumerate(action.coalitions):
            if coalition:
                seeded[index].add(_coalition(coalition))

    target_pools: dict[int, set[Coalition]] = {}
    for target in dict.fromkeys(slot.target_id for slot in slot_values):
        if target not in target_position or target_position[target].shape != (3,):
            raise ValueError(f"missing three-vector for target {target}")
        pool: set[Coalition] = {(agent_id,) for agent_id in ids}
        nearest = sorted(
            ids,
            key=lambda agent_id: (
                float(np.linalg.norm(position[agent_id] - target_position[target])),
                agent_id,
            ),
        )[: min(len(ids), int(neighborhood_size))]
        for size in range(2, min(cap, len(nearest)) + 1):
            pool.update(tuple(value) for value in combinations(nearest, size))
        # Large domains deliberately price peer-local coalitions only around
        # agents that are themselves near this target.  Registering the same
        # peer family for every identity in every channel creates thousands of
        # symmetric MILP columns without adding a distinct spatial hypothesis.
        # All singletons and every seed-action coalition remain available, so
        # the candidate domain always has a complete feasible partition.
        for anchor in nearest:
            peers = sorted(
                (value for value in ids if value != anchor),
                key=lambda value: (
                    float(np.linalg.norm(position[value] - position[anchor])),
                    value,
                ),
            )[: min(len(ids) - 1, int(peer_count))]
            for size in range(2, min(cap, len(peers) + 1) + 1):
                pool.update(
                    _coalition((anchor, *others))
                    for others in combinations(peers, size - 1)
                )
        target_pools[target] = pool

    result: list[Tuple[Coalition, ...]] = []
    for index, slot in enumerate(slot_values):
        pool = set(target_pools[slot.target_id])
        pool.update(seeded[index])
        result.append(tuple(sorted(pool, key=lambda value: (len(value), value))))
    return tuple(result)


class IdentityBlottoGame:
    """Finite candidate-domain identity-aware zero-sum Blotto game.

    By default Red may reserve agents and Blue must assign every agent.  The
    side-specific switches are explicit to make diagnostic variants possible
    without weakening Blue's exact-cover constraint accidentally.
    """

    def __init__(
        self,
        red_ids: Sequence[int],
        blue_ids: Sequence[int],
        slots: Sequence[EngagementSlot],
        red_candidates: Sequence[Sequence[Coalition]],
        blue_candidates: Sequence[Sequence[Coalition]],
        payoff_oracle: IdentityLocalPayoffOracle,
        *,
        max_group_size: int = 4,
        structural_breach_penalty: Optional[float] = None,
        payoff_scale: float = 1.0,
        full_coalition_domain: bool = False,
        red_fallback_action: Optional[IdentityAction] = None,
        blue_fallback_action: Optional[IdentityAction] = None,
        allow_red_reserve: bool = True,
        allow_blue_reserve: bool = False,
        allow_unregistered_actions: bool = False,
    ) -> None:
        self.red_ids = _coalition(red_ids)
        self.blue_ids = _coalition(blue_ids)
        self.slots = tuple(slots)
        self.max_group_size = int(max_group_size)
        if self.max_group_size != 4:
            raise ValueError("the registered identity game requires max_group_size=4")
        if not self.red_ids or not self.blue_ids or not self.slots:
            raise ValueError("both sides and the slot set must be non-empty")
        self.red_candidates = self._normalise_candidates(
            red_candidates, set(self.red_ids), "Red"
        )
        self.blue_candidates = self._normalise_candidates(
            blue_candidates, set(self.blue_ids), "Blue"
        )
        self.payoff_oracle = payoff_oracle
        self.structural_breach_penalty = float(
            structural_breach_penalty
            if structural_breach_penalty is not None
            else len(self.red_ids) + len(self.blue_ids) + 1
        )
        if not np.isfinite(self.structural_breach_penalty) or self.structural_breach_penalty <= 0:
            raise ValueError("structural_breach_penalty must be positive and finite")
        self.payoff_scale = float(payoff_scale)
        if not np.isfinite(self.payoff_scale) or self.payoff_scale <= 0.0:
            raise ValueError("payoff_scale must be positive and finite")
        self.full_coalition_domain = bool(full_coalition_domain)
        self.allow_red_reserve = bool(allow_red_reserve)
        self.allow_blue_reserve = bool(allow_blue_reserve)
        # MILP best responses remain restricted to ``*_candidates``.  An
        # anytime graph-search oracle such as SALDAE may, however, construct
        # a legal labelled coalition on demand without registering every
        # O(n^4) coalition-slot column first.  Keeping this switch explicit
        # prevents the old restricted-domain method from silently changing.
        self.allow_unregistered_actions = bool(allow_unregistered_actions)
        self._cache: dict[LocalPayoffRequest, float] = {}
        self.red_fallback_action = (
            None
            if red_fallback_action is None
            else self.validate_action(red_fallback_action, "Red")
        )
        self.blue_fallback_action = (
            None
            if blue_fallback_action is None
            else self.validate_action(blue_fallback_action, "Blue")
        )

    def _normalise_candidates(
        self,
        candidates: Sequence[Sequence[Coalition]],
        allowed_ids: set[int],
        side: str,
    ) -> Tuple[Tuple[Coalition, ...], ...]:
        if len(candidates) != len(self.slots):
            raise ValueError(f"{side} candidates must align with slots")
        result: list[Tuple[Coalition, ...]] = []
        for values in candidates:
            groups = tuple(dict.fromkeys(_coalition(value) for value in values))
            if not groups:
                raise ValueError(f"every slot needs at least one {side} candidate")
            for coalition in groups:
                if not coalition or len(coalition) > self.max_group_size:
                    raise ValueError(f"{side} coalition size is outside 1..4")
                if not set(coalition).issubset(allowed_ids):
                    raise ValueError(f"{side} coalition contains an unknown identity")
            result.append(groups)
        return tuple(result)

    def validate_action(self, action: IdentityAction, side: str) -> IdentityAction:
        ids = self.red_ids if side == "Red" else self.blue_ids if side == "Blue" else None
        candidates = self.red_candidates if side == "Red" else self.blue_candidates if side == "Blue" else None
        allow_reserve = (
            self.allow_red_reserve
            if side == "Red"
            else self.allow_blue_reserve
            if side == "Blue"
            else None
        )
        if ids is None or candidates is None or allow_reserve is None:
            raise ValueError("side must be Red or Blue")
        if not isinstance(action, IdentityAction) or len(action.coalitions) != len(self.slots):
            raise ValueError(f"{side} action must provide one coalition per slot")
        used: list[int] = []
        normalised: list[Coalition] = []
        for index, raw in enumerate(action.coalitions):
            coalition = _coalition(raw)
            if len(coalition) > self.max_group_size:
                raise ValueError(f"{side} group exceeds 4 agents")
            if (
                coalition
                and not self.allow_unregistered_actions
                and coalition not in candidates[index]
            ):
                raise ValueError(f"{side} action uses an unregistered coalition column")
            normalised.append(coalition)
            used.extend(coalition)
        reserve = _coalition(action.reserve_ids)
        known = set(ids)
        if not set(reserve).issubset(known):
            raise ValueError(f"{side} reserve contains an unknown identity")
        if reserve and not allow_reserve:
            raise ValueError(f"{side} reserve is disabled")
        if len(used) != len(set(used)):
            raise ValueError(
                f"{side} active partition must not repeat an identity"
            )
        if set(used).intersection(reserve):
            raise ValueError(f"{side} identity cannot be both active and reserve")
        if set(used).union(reserve) != known:
            if allow_reserve:
                raise ValueError(
                    f"{side} action must place every live identity exactly once "
                    "in an active coalition or reserve"
                )
            raise ValueError(f"{side} action must partition every live identity exactly once")
        for target in dict.fromkeys(slot.target_id for slot in self.slots):
            indices = sorted(
                (
                    index
                    for index, slot in enumerate(self.slots)
                    if slot.target_id == target
                ),
                key=lambda index: self.slots[index].channel_id,
            )
            sizes = [len(normalised[index]) for index in indices]
            if any(
                sizes[position] > sizes[position - 1]
                for position in range(1, len(sizes))
            ):
                raise ValueError(
                    f"{side} channels at each target must use non-increasing group sizes"
                )
        return IdentityAction(tuple(normalised), reserve)

    def _local_values(
        self, requests: Sequence[LocalPayoffRequest]
    ) -> np.ndarray:
        values = np.empty(len(requests), dtype=np.float64)
        missing: list[LocalPayoffRequest] = []
        missing_indices: list[int] = []
        for index, raw in enumerate(requests):
            slot, red, blue = int(raw[0]), _coalition(raw[1]), _coalition(raw[2])
            key = (slot, red, blue)
            if not red and not blue:
                values[index] = 0.0
            elif not blue:
                values[index] = 0.0
            elif not red:
                values[index] = -self.structural_breach_penalty
            elif key in self._cache:
                values[index] = self._cache[key]
            else:
                missing.append(key)
                missing_indices.append(index)
        if missing:
            unique = tuple(dict.fromkeys(missing))
            predicted = np.asarray(self.payoff_oracle.evaluate(unique), dtype=np.float64)
            if predicted.shape != (len(unique),) or not np.all(np.isfinite(predicted)):
                raise ValueError("identity payoff oracle returned invalid values")
            self._cache.update(zip(unique, map(float, predicted)))
            for index, key in zip(missing_indices, missing):
                values[index] = self._cache[key]
        return values

    def local_values(
        self, requests: Sequence[LocalPayoffRequest]
    ) -> np.ndarray:
        """Evaluate local requests through the shared, batched payoff cache.

        This public read-only facade is used by on-demand coalition-search
        oracles.  Empty-side analytic boundaries and Stage2 queries therefore
        remain exactly identical to the MILP oracle.
        """

        return self._local_values(requests)

    def payoff(self, red: IdentityAction, blue: IdentityAction) -> float:
        red = self.validate_action(red, "Red")
        blue = self.validate_action(blue, "Blue")
        requests = [
            (index, red.coalitions[index], blue.coalitions[index])
            for index in range(len(self.slots))
        ]
        return float(self._local_values(requests).sum() / self.payoff_scale)

    def payoff_matrix(
        self,
        red_actions: Sequence[IdentityAction],
        blue_actions: Sequence[IdentityAction],
    ) -> np.ndarray:
        red = [self.validate_action(value, "Red") for value in red_actions]
        blue = [self.validate_action(value, "Blue") for value in blue_actions]
        if not red or not blue:
            raise ValueError("both restricted supports must be non-empty")
        requests: list[LocalPayoffRequest] = []
        for red_action in red:
            for blue_action in blue:
                requests.extend(
                    (slot, red_action.coalitions[slot], blue_action.coalitions[slot])
                    for slot in range(len(self.slots))
                )
        values = self._local_values(requests).reshape(
            len(red), len(blue), len(self.slots)
        )
        return values.sum(axis=2) / self.payoff_scale

    @staticmethod
    def _mixture(values: Sequence[float], expected: int) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64)
        if result.shape != (expected,) or np.any(result < -1e-12) or not np.all(np.isfinite(result)):
            raise ValueError("invalid opponent mixture")
        result = np.clip(result, 0.0, None)
        if float(result.sum()) <= 0.0:
            raise ValueError("opponent mixture has no probability mass")
        return result / result.sum()

    def _infeasible_diagnostic(
        self,
        *,
        side: str,
        own_ids: Sequence[int],
        candidates: Sequence[Sequence[Coalition]],
        allow_reserve: bool,
        solver_message: str,
    ) -> str:
        """Describe candidate coverage without hiding a modelling failure."""

        covered = {
            agent_id
            for slot_candidates in candidates
            for coalition in slot_candidates
            for agent_id in coalition
        }
        missing = tuple(agent_id for agent_id in own_ids if agent_id not in covered)
        per_slot = tuple(len(values) for values in candidates)
        return (
            f"side={side}, reserve_enabled={allow_reserve}, "
            f"identities={len(own_ids)}, slots={len(self.slots)}, "
            f"candidate_columns={sum(per_slot)}, candidates_per_slot={per_slot}, "
            f"uncovered_identities={missing}, solver={solver_message}"
        )

    def _best_response(
        self,
        opponent_actions: Sequence[IdentityAction],
        opponent_mixture: Sequence[float],
        *,
        red_player: bool,
        time_limit_seconds: Optional[float],
        mip_relative_gap: float,
    ) -> IdentityBestResponse:
        opponent_side = "Blue" if red_player else "Red"
        opponents = [self.validate_action(value, opponent_side) for value in opponent_actions]
        probability = self._mixture(opponent_mixture, len(opponents))
        own_ids = self.red_ids if red_player else self.blue_ids
        candidates = self.red_candidates if red_player else self.blue_candidates
        allow_reserve = self.allow_red_reserve if red_player else self.allow_blue_reserve
        variables = [
            (slot, coalition)
            for slot, groups in enumerate(candidates)
            for coalition in groups
        ]

        baseline_requests: list[LocalPayoffRequest] = []
        candidate_requests: list[LocalPayoffRequest] = []
        for opponent in opponents:
            for slot in range(len(self.slots)):
                other = opponent.coalitions[slot]
                baseline_requests.append(
                    (slot, (), other) if red_player else (slot, other, ())
                )
            for slot, coalition in variables:
                other = opponent.coalitions[slot]
                candidate_requests.append(
                    (slot, coalition, other)
                    if red_player
                    else (slot, other, coalition)
                )
        baseline_values = self._local_values(baseline_requests).reshape(
            len(opponents), len(self.slots)
        )
        candidate_values = self._local_values(candidate_requests).reshape(
            len(opponents), len(variables)
        )
        baseline_by_opponent = baseline_values.sum(axis=1)
        baseline = float(probability @ baseline_by_opponent)
        # Selecting a group replaces the empty-side payoff in its slot.
        baseline_for_variable = np.asarray(
            [baseline_values[:, slot] for slot, _ in variables], dtype=np.float64
        ).T
        delta = probability @ (candidate_values - baseline_for_variable)

        agent_index = {agent_id: index for index, agent_id in enumerate(own_ids)}
        rows: list[int] = []
        columns: list[int] = []
        data: list[float] = []
        agent_rows = len(own_ids)
        for column, (slot, coalition) in enumerate(variables):
            for agent_id in coalition:
                rows.append(agent_index[agent_id])
                columns.append(column)
                data.append(1.0)
            rows.append(agent_rows + slot)
            columns.append(column)
            data.append(1.0)
        target_prefix_pairs: list[tuple[int, int]] = []
        for target in dict.fromkeys(slot.target_id for slot in self.slots):
            indices = sorted(
                (
                    index
                    for index, slot in enumerate(self.slots)
                    if slot.target_id == target
                ),
                key=lambda index: self.slots[index].channel_id,
            )
            target_prefix_pairs.extend(zip(indices[:-1], indices[1:]))
        prefix_offset = agent_rows + len(self.slots)
        for pair_index, (previous_slot, next_slot) in enumerate(target_prefix_pairs):
            row = prefix_offset + pair_index
            for column, (slot, coalition_value) in enumerate(variables):
                if slot == next_slot:
                    rows.append(row)
                    columns.append(column)
                    data.append(float(len(coalition_value)))
                elif slot == previous_slot:
                    rows.append(row)
                    columns.append(column)
                    data.append(-float(len(coalition_value)))
        matrix = coo_matrix(
            (data, (rows, columns)),
            shape=(
                agent_rows + len(self.slots) + len(target_prefix_pairs),
                len(variables),
            ),
        ).tocsr()
        identity_lower = (
            np.zeros(agent_rows, dtype=np.float64)
            if allow_reserve
            else np.ones(agent_rows, dtype=np.float64)
        )
        lower = np.concatenate(
            [
                identity_lower,
                np.zeros(len(self.slots)),
                np.full(len(target_prefix_pairs), -np.inf),
            ]
        )
        upper = np.concatenate(
            [
                np.ones(agent_rows + len(self.slots), dtype=np.float64),
                np.zeros(len(target_prefix_pairs)),
            ]
        )
        options: dict[str, object] = {
            "disp": False,
            "mip_rel_gap": float(mip_relative_gap),
        }
        if time_limit_seconds is not None:
            options["time_limit"] = float(time_limit_seconds)
        result = milp(
            -delta if red_player else delta,
            integrality=np.ones(len(variables), dtype=np.uint8),
            bounds=Bounds(np.zeros(len(variables)), np.ones(len(variables))),
            constraints=LinearConstraint(matrix, lower, upper),
            options=options,
        )
        if result.x is None:
            diagnostic = self._infeasible_diagnostic(
                side="Red" if red_player else "Blue",
                own_ids=own_ids,
                candidates=candidates,
                allow_reserve=allow_reserve,
                solver_message=str(result.message),
            )
            if int(result.status) == 2 or "infeasible" in str(result.message).lower():
                raise RuntimeError(
                    "identity set-partitioning oracle is infeasible; " + diagnostic
                )
            fallback = (
                self.red_fallback_action if red_player else self.blue_fallback_action
            )
            if fallback is None and red_player and self.allow_red_reserve:
                fallback = self.validate_action(
                    IdentityAction(
                        tuple(() for _ in self.slots),
                        self.red_ids,
                    ),
                    "Red",
                )
            if fallback is None:
                raise RuntimeError(
                    "identity set-partitioning oracle failed; " + diagnostic
                )
            if red_player:
                fallback_value = float(
                    self.payoff_matrix([fallback], opponents)[0] @ probability
                )
            else:
                fallback_value = float(
                    probability @ self.payoff_matrix(opponents, [fallback])[:, 0]
                )
            return IdentityBestResponse(
                action=fallback,
                value=fallback_value,
                optimal=False,
                solver_status=int(result.status),
                solver_message=str(result.message),
                mip_gap=float("inf"),
                mip_node_count=0,
                variables=len(variables),
            )
        selected = np.flatnonzero(np.asarray(result.x) > 0.5)
        groups: list[Coalition] = [()] * len(self.slots)
        for index in selected:
            slot, coalition = variables[int(index)]
            if groups[slot]:
                raise RuntimeError("MILP selected two coalitions for one engagement slot")
            groups[slot] = coalition
        active_ids = {agent_id for group in groups for agent_id in group}
        reserve = (
            tuple(agent_id for agent_id in own_ids if agent_id not in active_ids)
            if allow_reserve
            else ()
        )
        action = self.validate_action(
            IdentityAction(tuple(groups), reserve),
            "Red" if red_player else "Blue",
        )
        if red_player:
            values = self.payoff_matrix([action], opponents)[0]
        else:
            values = self.payoff_matrix(opponents, [action])[:, 0]
        value = float(probability @ values)
        gap = getattr(result, "mip_gap", np.nan)
        nodes = getattr(result, "mip_node_count", 0)
        return IdentityBestResponse(
            action=action,
            value=value,
            optimal=int(result.status) == 0,
            solver_status=int(result.status),
            solver_message=str(result.message),
            mip_gap=float(gap) if gap is not None else np.nan,
            mip_node_count=int(nodes) if nodes is not None else 0,
            variables=len(variables),
        )

    def red_best_response(
        self,
        blue_actions: Sequence[IdentityAction],
        blue_mixture: Sequence[float],
        *,
        time_limit_seconds: Optional[float] = None,
        mip_relative_gap: float = 0.0,
    ) -> IdentityBestResponse:
        return self._best_response(
            blue_actions,
            blue_mixture,
            red_player=True,
            time_limit_seconds=time_limit_seconds,
            mip_relative_gap=mip_relative_gap,
        )

    def blue_best_response(
        self,
        red_actions: Sequence[IdentityAction],
        red_mixture: Sequence[float],
        *,
        time_limit_seconds: Optional[float] = None,
        mip_relative_gap: float = 0.0,
    ) -> IdentityBestResponse:
        return self._best_response(
            red_actions,
            red_mixture,
            red_player=False,
            time_limit_seconds=time_limit_seconds,
            mip_relative_gap=mip_relative_gap,
        )


def solve_identity_double_oracle(
    game: IdentityBlottoGame,
    initial_red: Sequence[IdentityAction],
    initial_blue: Sequence[IdentityAction],
    *,
    tolerance: float = 1e-3,
    max_iterations: int = 40,
    oracle_time_limit_seconds: Optional[float] = None,
    oracle_mip_relative_gap: float = 0.0,
    verbose: bool = False,
) -> IdentityDoubleOracleResult:
    """Run Double Oracle with weighted set-partitioning best responses."""

    red = list(dict.fromkeys(game.validate_action(value, "Red") for value in initial_red))
    blue = list(dict.fromkeys(game.validate_action(value, "Blue") for value in initial_blue))
    if not red or not blue:
        raise ValueError("initial supports must be non-empty")
    history: list[IdentityDoubleOracleIteration] = []
    final_solution = None
    converged = False
    reason = "max_iterations"
    for iteration in range(1, int(max_iterations) + 1):
        matrix = game.payoff_matrix(red, blue)
        solution = solve_restricted_matrix_game(matrix)
        red_br = game.red_best_response(
            blue,
            solution.attacker_mixture,
            time_limit_seconds=oracle_time_limit_seconds,
            mip_relative_gap=oracle_mip_relative_gap,
        )
        blue_br = game.blue_best_response(
            red,
            solution.defender_mixture,
            time_limit_seconds=oracle_time_limit_seconds,
            mip_relative_gap=oracle_mip_relative_gap,
        )
        red_gap = max(0.0, red_br.value - solution.value)
        blue_gap = max(0.0, solution.value - blue_br.value)
        exploitability = max(0.0, red_br.value - blue_br.value)
        oracles_exact = red_br.optimal and blue_br.optimal
        certified = oracles_exact and exploitability <= float(tolerance)
        add_red = (
            not certified
            and red_gap > float(tolerance) / 2.0
            and red_br.action not in red
            and iteration < int(max_iterations)
        )
        add_blue = (
            not certified
            and blue_gap > float(tolerance) / 2.0
            and blue_br.action not in blue
            and iteration < int(max_iterations)
        )
        history.append(
            IdentityDoubleOracleIteration(
                iteration=iteration,
                red_support=len(red),
                blue_support=len(blue),
                restricted_value=float(solution.value),
                red_best_response_value=red_br.value,
                blue_best_response_value=blue_br.value,
                red_gap=red_gap,
                blue_gap=blue_gap,
                exploitability=exploitability,
                red_oracle_optimal=red_br.optimal,
                blue_oracle_optimal=blue_br.optimal,
                added_red=add_red,
                added_blue=add_blue,
            )
        )
        final_solution = solution
        if verbose:
            print(
                f"[Stage3][IdentityDO] iter={iteration:02d} "
                f"support=({len(red)},{len(blue)}) value={solution.value:+.5f} "
                f"gap={exploitability:.3e} exact_oracles={oracles_exact}",
                flush=True,
            )
        if certified:
            converged = True
            reason = "converged"
            break
        if not red_br.optimal or not blue_br.optimal:
            reason = "oracle_time_limit"
        if not add_red and not add_blue:
            if iteration >= int(max_iterations) and not certified:
                reason = "max_iterations"
            else:
                reason = "oracle_stalled" if oracles_exact else "oracle_time_limit"
            break
        if add_red:
            red.append(red_br.action)
        if add_blue:
            blue.append(blue_br.action)
    if final_solution is None:
        raise RuntimeError("identity Double Oracle did not solve a restricted game")
    last = history[-1]
    return IdentityDoubleOracleResult(
        red_strategies=tuple(red),
        blue_strategies=tuple(blue),
        red_mixture=final_solution.defender_mixture.copy(),
        blue_mixture=final_solution.attacker_mixture.copy(),
        value=float(final_solution.value),
        exploitability=float(last.exploitability),
        converged=converged,
        candidate_domain_exact=converged,
        full_game_certified=converged and game.full_coalition_domain,
        termination_reason=reason,
        history=tuple(history),
    )


__all__ = [
    "Coalition",
    "EngagementSlot",
    "IdentityAction",
    "IdentityBestResponse",
    "IdentityBlottoGame",
    "IdentityDoubleOracleResult",
    "IdentityLocalPayoffOracle",
    "enumerate_coalitions",
    "enumerate_identity_actions",
    "make_engagement_slots",
    "round_robin_identity_action",
    "solve_identity_double_oracle",
    "spatial_candidate_coalitions",
]
