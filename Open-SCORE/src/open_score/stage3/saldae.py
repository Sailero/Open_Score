"""SALDAE-style anytime best responses for identity coalition Blotto.

The original SALDAE algorithm searches a graph whose nodes are coalition
structures and whose edges split or merge coalitions.  This module preserves
that core and its multi-search-agent OPEN/RESERVE/SUBSTITUTE memory scheme,
then adds the minimum operations required by Open-SCORE's action space:
target/channel relocation, identity swaps, and Red reserve transitions.

SALDAE is used *inside* Double Oracle.  It is not treated as a replacement
for the two-player zero-sum game: after fixing the opponent mixture, the
expected additive slot utility is a constrained CSG objective and can be
searched by SALDAE.  Since the search is anytime, the returned response is
never labelled an exact oracle unless a separate exhaustive check is made.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
import time
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from .blotto import solve_restricted_matrix_game
from .identity_blotto import (
    Coalition,
    IdentityAction,
    IdentityBestResponse,
    IdentityBlottoGame,
    IdentityDoubleOracleIteration,
    IdentityDoubleOracleResult,
)


@dataclass(frozen=True)
class SALDAEConfig:
    """Bounded anytime-search settings.

    ``keep_children_multiplier=2`` and ``omega=0.995`` follow the best
    settings reported in the SALDAE paper.  Search agents are interleaved in
    one process so Stage2's batched neural payoff oracle remains deterministic
    and is never called concurrently.
    """

    search_agents: int = 4
    time_limit_seconds: float = 0.35
    max_expansions: int = 80
    keep_children_multiplier: float = 2.0
    child_sample_multiplier: float = 4.0
    selection_rounds: int = 2
    omega: float = 0.995
    reserve_memory: int = 512
    bridge_path_limit: int = 16
    random_seed: int = 0

    def validate(self) -> "SALDAEConfig":
        if self.search_agents < 1:
            raise ValueError("SALDAE needs at least one search agent")
        if self.time_limit_seconds <= 0.0 or not math.isfinite(self.time_limit_seconds):
            raise ValueError("SALDAE time limit must be positive and finite")
        if self.max_expansions < 1:
            raise ValueError("SALDAE max_expansions must be positive")
        if self.keep_children_multiplier <= 0.0 or self.child_sample_multiplier <= 0.0:
            raise ValueError("SALDAE child multipliers must be positive")
        if self.selection_rounds < 1:
            raise ValueError("SALDAE selection_rounds must be positive")
        if not 0.0 < self.omega <= 1.0:
            raise ValueError("SALDAE omega must lie in (0, 1]")
        if self.reserve_memory < 1 or self.bridge_path_limit < 0:
            raise ValueError("SALDAE memory settings are invalid")
        return self


@dataclass(frozen=True)
class SALDAESearchDiagnostics:
    side: str
    value: float
    objective_score: float
    elapsed_seconds: float
    expansions: int
    evaluated_nodes: int
    generated_nodes: int
    duplicate_conflicts: int
    open_promotions: int
    substitute_promotions: int
    reserve_promotions: int
    bridge_nodes: int
    incumbent_updates: int
    search_agents: int
    termination_reason: str
    used_unregistered_coalition: bool


@dataclass(frozen=True)
class SALDAEDoubleOracleDiagnostics:
    red_searches: Tuple[SALDAESearchDiagnostics, ...]
    blue_searches: Tuple[SALDAESearchDiagnostics, ...]
    empirical_converged: bool
    expanded_action_domain: bool


@dataclass
class _SearchAgentMemory:
    open: list[tuple[float, int, IdentityAction]]
    substitute: list[tuple[float, int, IdentityAction]]
    reserve: list[tuple[float, int, IdentityAction]]


def _push(
    heap: list[tuple[float, int, IdentityAction]],
    action: IdentityAction,
    score: float,
    serial: int,
) -> None:
    heapq.heappush(heap, (-float(score), int(serial), action))


def _side_data(game: IdentityBlottoGame, red_player: bool):
    return (
        (game.red_ids, game.allow_red_reserve, "Red")
        if red_player
        else (game.blue_ids, game.allow_blue_reserve, "Blue")
    )


def _canonical_action(
    game: IdentityBlottoGame,
    coalitions: Sequence[Iterable[int]],
    reserve: Iterable[int],
    *,
    red_player: bool,
) -> IdentityAction:
    """Apply only the registered channel-size symmetry breaker.

    Equal-sized groups retain their order because their identities can matter
    against an opponent's equally-sized channel group.
    """

    groups = [tuple(sorted(map(int, group))) for group in coalitions]
    for target in dict.fromkeys(slot.target_id for slot in game.slots):
        indices = [
            index for index, slot in enumerate(game.slots) if slot.target_id == target
        ]
        ordered = sorted((groups[index] for index in indices), key=len, reverse=True)
        for index, group in zip(indices, ordered):
            groups[index] = group
    _, _, side = _side_data(game, red_player)
    return game.validate_action(
        IdentityAction(tuple(groups), tuple(sorted(map(int, reserve)))), side
    )


def _group_sizes(total: int, count: int, rng: np.random.Generator) -> list[int]:
    if count < 1 or count > total or total > 4 * count:
        raise ValueError("cannot form requested 1..4 group-size pattern")
    sizes = [1] * count
    remaining = total - count
    while remaining:
        choices = [index for index, size in enumerate(sizes) if size < 4]
        index = int(rng.choice(choices))
        sizes[index] += 1
        remaining -= 1
    rng.shuffle(sizes)
    return sizes


def _random_action(
    game: IdentityBlottoGame,
    *,
    red_player: bool,
    rng: np.random.Generator,
    mode: str = "random",
) -> IdentityAction:
    ids, allow_reserve, _ = _side_data(game, red_player)
    values = list(ids)
    rng.shuffle(values)
    slot_count = len(game.slots)
    reserve_count = 0
    if allow_reserve:
        if mode == "reserve":
            reserve_count = len(values)
        elif mode == "random" and len(values) > 1:
            reserve_count = int(rng.integers(0, max(1, len(values) // 5) + 1))
        elif mode == "singleton" and len(values) > slot_count:
            reserve_count = len(values) - slot_count
    reserve = values[:reserve_count]
    active = values[reserve_count:]
    coalitions: list[Coalition] = [()] * slot_count
    if active:
        minimum = int(math.ceil(len(active) / 4.0))
        maximum = min(len(active), slot_count)
        if mode == "packed":
            group_count = minimum
        elif mode == "singleton":
            group_count = maximum
        else:
            group_count = int(rng.integers(minimum, maximum + 1))
        sizes = _group_sizes(len(active), group_count, rng)
        slots = list(map(int, rng.choice(slot_count, size=group_count, replace=False)))
        cursor = 0
        for slot, size in zip(slots, sizes):
            coalitions[slot] = tuple(sorted(active[cursor : cursor + size]))
            cursor += size
    return _canonical_action(
        game, coalitions, reserve, red_player=red_player
    )


def _try_action(
    game: IdentityBlottoGame,
    groups: Sequence[Iterable[int]],
    reserve: Iterable[int],
    *,
    red_player: bool,
) -> Optional[IdentityAction]:
    try:
        return _canonical_action(
            game, groups, reserve, red_player=red_player
        )
    except ValueError:
        return None


def _sample_neighbors(
    game: IdentityBlottoGame,
    action: IdentityAction,
    *,
    red_player: bool,
    rng: np.random.Generator,
    requested: int,
) -> Tuple[IdentityAction, ...]:
    """Sample legal graph neighbors.

    Split and merge are the original SALDAE edges.  Relocate, member-shift,
    swap and reserve transitions are explicit task-labelled CSG extensions;
    each is also expressible as a short split/merge path when unconstrained.
    """

    groups0 = [tuple(group) for group in action.coalitions]
    reserve0 = list(action.reserve_ids)
    allow_reserve = _side_data(game, red_player)[1]
    output: list[IdentityAction] = []
    seen = {action}
    maximum_attempts = max(64, int(requested) * 12)
    operations = ["split", "merge", "relocate", "shift", "swap_member", "swap_group"]
    if allow_reserve:
        operations.extend(["to_reserve", "from_reserve", "reserve_swap"])

    for _ in range(maximum_attempts):
        if len(output) >= int(requested):
            break
        groups = list(groups0)
        reserve = list(reserve0)
        nonempty = [index for index, group in enumerate(groups) if group]
        empty = [index for index, group in enumerate(groups) if not group]
        operation = str(rng.choice(operations))
        changed = False

        if operation == "split":
            sources = [index for index in nonempty if len(groups[index]) >= 2]
            if sources and empty:
                source = int(rng.choice(sources))
                destination = int(rng.choice(empty))
                members = list(groups[source])
                rng.shuffle(members)
                cut = int(rng.integers(1, len(members)))
                groups[source] = tuple(sorted(members[:cut]))
                groups[destination] = tuple(sorted(members[cut:]))
                changed = True
        elif operation == "merge":
            pairs = [
                (left, right)
                for offset, left in enumerate(nonempty)
                for right in nonempty[offset + 1 :]
                if len(groups[left]) + len(groups[right]) <= 4
            ]
            if pairs:
                left, right = pairs[int(rng.integers(len(pairs)))]
                keep, clear = (left, right) if rng.random() < 0.5 else (right, left)
                groups[keep] = tuple(sorted((*groups[left], *groups[right])))
                groups[clear] = ()
                changed = True
        elif operation == "relocate":
            if nonempty and empty:
                source = int(rng.choice(nonempty))
                destination = int(rng.choice(empty))
                groups[destination], groups[source] = groups[source], ()
                changed = True
        elif operation == "shift":
            destinations = [index for index, group in enumerate(groups) if len(group) < 4]
            if nonempty and destinations:
                source = int(rng.choice(nonempty))
                destinations = [value for value in destinations if value != source]
                if destinations:
                    destination = int(rng.choice(destinations))
                    member = int(rng.choice(groups[source]))
                    groups[source] = tuple(value for value in groups[source] if value != member)
                    groups[destination] = tuple(sorted((*groups[destination], member)))
                    changed = True
        elif operation == "swap_member":
            if len(nonempty) >= 2:
                left, right = map(int, rng.choice(nonempty, size=2, replace=False))
                left_member = int(rng.choice(groups[left]))
                right_member = int(rng.choice(groups[right]))
                groups[left] = tuple(
                    sorted(right_member if value == left_member else value for value in groups[left])
                )
                groups[right] = tuple(
                    sorted(left_member if value == right_member else value for value in groups[right])
                )
                changed = True
        elif operation == "swap_group":
            if len(groups) >= 2:
                left, right = map(int, rng.choice(len(groups), size=2, replace=False))
                groups[left], groups[right] = groups[right], groups[left]
                changed = True
        elif operation == "to_reserve":
            if nonempty:
                source = int(rng.choice(nonempty))
                member = int(rng.choice(groups[source]))
                groups[source] = tuple(value for value in groups[source] if value != member)
                reserve.append(member)
                changed = True
        elif operation == "from_reserve":
            destinations = [index for index, group in enumerate(groups) if len(group) < 4]
            if reserve and destinations:
                member = int(rng.choice(reserve))
                destination = int(rng.choice(destinations))
                reserve.remove(member)
                groups[destination] = tuple(sorted((*groups[destination], member)))
                changed = True
        elif operation == "reserve_swap":
            if reserve and nonempty:
                source = int(rng.choice(nonempty))
                active_member = int(rng.choice(groups[source]))
                reserve_member = int(rng.choice(reserve))
                groups[source] = tuple(
                    sorted(reserve_member if value == active_member else value for value in groups[source])
                )
                reserve.remove(reserve_member)
                reserve.append(active_member)
                changed = True

        if not changed:
            continue
        candidate = _try_action(
            game, groups, reserve, red_player=red_player
        )
        if candidate is not None and candidate not in seen:
            seen.add(candidate)
            output.append(candidate)
    return tuple(output)


def _role_map(action: IdentityAction) -> Dict[int, int]:
    result = {
        int(agent_id): int(slot)
        for slot, coalition in enumerate(action.coalitions)
        for agent_id in coalition
    }
    result.update({int(agent_id): -1 for agent_id in action.reserve_ids})
    return result


def _action_distance(left: IdentityAction, right: IdentityAction) -> int:
    left_roles = _role_map(left)
    right_roles = _role_map(right)
    return sum(left_roles.get(agent_id) != role for agent_id, role in right_roles.items())


def _targeted_neighbors(
    game: IdentityBlottoGame,
    current: IdentityAction,
    target: IdentityAction,
    *,
    red_player: bool,
) -> Tuple[IdentityAction, ...]:
    current_roles = _role_map(current)
    target_roles = _role_map(target)
    groups0 = [tuple(group) for group in current.coalitions]
    reserve0 = list(current.reserve_ids)
    output: list[IdentityAction] = []
    seen = {current}
    for agent_id, desired in target_roles.items():
        source = current_roles[agent_id]
        if source == desired:
            continue
        groups = list(groups0)
        reserve = list(reserve0)
        changed = False
        if desired == -1:
            if source >= 0:
                groups[source] = tuple(value for value in groups[source] if value != agent_id)
                reserve.append(agent_id)
                changed = True
        elif source == -1:
            destination = list(groups[desired])
            if len(destination) < 4:
                reserve.remove(agent_id)
                destination.append(agent_id)
                groups[desired] = tuple(sorted(destination))
                changed = True
            else:
                wrong = next(
                    (value for value in destination if target_roles[value] != desired), None
                )
                if wrong is not None:
                    reserve.remove(agent_id)
                    reserve.append(wrong)
                    groups[desired] = tuple(
                        sorted(agent_id if value == wrong else value for value in destination)
                    )
                    changed = True
        else:
            destination = list(groups[desired])
            if len(destination) < 4:
                groups[source] = tuple(value for value in groups[source] if value != agent_id)
                destination.append(agent_id)
                groups[desired] = tuple(sorted(destination))
                changed = True
            else:
                wrong = next(
                    (value for value in destination if target_roles[value] != desired), None
                )
                if wrong is not None:
                    groups[source] = tuple(
                        sorted(wrong if value == agent_id else value for value in groups[source])
                    )
                    groups[desired] = tuple(
                        sorted(agent_id if value == wrong else value for value in destination)
                    )
                    changed = True
        if changed:
            candidate = _try_action(
                game, groups, reserve, red_player=red_player
            )
            if candidate is not None and candidate not in seen:
                seen.add(candidate)
                output.append(candidate)
    return tuple(output)


def _bridge_path(
    game: IdentityBlottoGame,
    start: IdentityAction,
    target: IdentityAction,
    *,
    red_player: bool,
    limit: int,
) -> Tuple[IdentityAction, ...]:
    """Build an APPROACH-THEN-SWAP-style feasible path toward a new best."""

    current = start
    path: list[IdentityAction] = []
    for _ in range(int(limit)):
        distance = _action_distance(current, target)
        if distance == 0:
            break
        candidates = _targeted_neighbors(
            game, current, target, red_player=red_player
        )
        improving = [
            value for value in candidates if _action_distance(value, target) < distance
        ]
        if not improving:
            break
        current = min(
            improving,
            key=lambda value: (_action_distance(value, target), repr(value)),
        )
        if current != target:
            path.append(current)
    return tuple(path)


class _ExpectedValueEvaluator:
    """Incrementally price labelled coalitions against a fixed mixture."""

    def __init__(
        self,
        game: IdentityBlottoGame,
        opponent_actions: Sequence[IdentityAction],
        opponent_mixture: Sequence[float],
        *,
        red_player: bool,
    ) -> None:
        self.game = game
        self.red_player = bool(red_player)
        opponent_side = "Blue" if red_player else "Red"
        self.opponents = tuple(
            game.validate_action(action, opponent_side) for action in opponent_actions
        )
        self.probability = game._mixture(opponent_mixture, len(self.opponents))
        baseline_requests = []
        for opponent in self.opponents:
            for slot, other in enumerate(opponent.coalitions):
                baseline_requests.append(
                    (slot, (), other) if red_player else (slot, other, ())
                )
        self.baseline_by_opponent_slot = game.local_values(baseline_requests).reshape(
            len(self.opponents), len(game.slots)
        )
        self.baseline = float(
            self.probability @ self.baseline_by_opponent_slot.sum(axis=1)
        ) / game.payoff_scale
        self.delta_cache: dict[tuple[int, Coalition], float] = {}
        self.action_cache: dict[IdentityAction, float] = {}

    def evaluate(self, actions: Sequence[IdentityAction]) -> np.ndarray:
        _, _, side = _side_data(self.game, self.red_player)
        checked = tuple(self.game.validate_action(action, side) for action in actions)
        missing_keys = tuple(
            dict.fromkeys(
                (slot, coalition)
                for action in checked
                for slot, coalition in enumerate(action.coalitions)
                if coalition and (slot, coalition) not in self.delta_cache
            )
        )
        if missing_keys:
            requests = []
            for slot, coalition in missing_keys:
                for opponent in self.opponents:
                    other = opponent.coalitions[slot]
                    requests.append(
                        (slot, coalition, other)
                        if self.red_player
                        else (slot, other, coalition)
                    )
            values = self.game.local_values(requests).reshape(
                len(missing_keys), len(self.opponents)
            )
            for key, candidate_values in zip(missing_keys, values):
                slot, _ = key
                delta = self.probability @ (
                    candidate_values - self.baseline_by_opponent_slot[:, slot]
                )
                self.delta_cache[key] = float(delta) / self.game.payoff_scale
        result = np.empty(len(checked), dtype=np.float64)
        for index, action in enumerate(checked):
            if action not in self.action_cache:
                self.action_cache[action] = self.baseline + sum(
                    self.delta_cache[(slot, coalition)]
                    for slot, coalition in enumerate(action.coalitions)
                    if coalition
                )
            result[index] = self.action_cache[action]
        return result

    def score(self, values: np.ndarray) -> np.ndarray:
        return values if self.red_player else -values


def _uses_unregistered(
    game: IdentityBlottoGame, action: IdentityAction, *, red_player: bool
) -> bool:
    candidates = game.red_candidates if red_player else game.blue_candidates
    return any(
        coalition and coalition not in candidates[slot]
        for slot, coalition in enumerate(action.coalitions)
    )


def saldae_best_response(
    game: IdentityBlottoGame,
    opponent_actions: Sequence[IdentityAction],
    opponent_mixture: Sequence[float],
    *,
    red_player: bool,
    initial_actions: Sequence[IdentityAction] = (),
    config: SALDAEConfig = SALDAEConfig(),
) -> tuple[IdentityBestResponse, SALDAESearchDiagnostics]:
    """Return an anytime SALDAE response to a fixed opponent mixture."""

    settings = config.validate()
    if not game.allow_unregistered_actions:
        raise ValueError(
            "SALDAE requires allow_unregistered_actions=True so graph moves are not "
            "silently reduced to the old registered-column domain"
        )
    started = time.perf_counter()
    rng = np.random.default_rng(int(settings.random_seed))
    evaluator = _ExpectedValueEvaluator(
        game, opponent_actions, opponent_mixture, red_player=red_player
    )
    _, _, side = _side_data(game, red_player)
    starts = list(
        dict.fromkeys(game.validate_action(action, side) for action in initial_actions)
    )
    for mode in ("singleton", "packed", "reserve"):
        if mode == "reserve" and not _side_data(game, red_player)[1]:
            continue
        try:
            starts.append(
                _random_action(game, red_player=red_player, rng=rng, mode=mode)
            )
        except ValueError:
            pass
    attempts = 0
    while (
        len(dict.fromkeys(starts)) < settings.search_agents
        and attempts < 20 * settings.search_agents
    ):
        starts.append(_random_action(game, red_player=red_player, rng=rng))
        attempts += 1
    starts = list(dict.fromkeys(starts))
    start_values = evaluator.evaluate(starts)
    start_scores = evaluator.score(start_values)
    best_index = int(np.argmax(start_scores))
    incumbent = starts[best_index]
    incumbent_value = float(start_values[best_index])
    incumbent_score = float(start_scores[best_index])
    min_score_seen = float(np.min(start_scores))

    memories = [
        _SearchAgentMemory([], [], []) for _ in range(settings.search_agents)
    ]
    serial = 0
    discovered: set[IdentityAction] = set()
    for index, (action, score) in enumerate(zip(starts, start_scores)):
        owner = index % len(memories)
        if action in discovered:
            continue
        discovered.add(action)
        _push(memories[owner].open, action, float(score), serial)
        serial += 1

    expansions = 0
    generated = 0
    conflicts = 0
    substitute_promotions = 0
    reserve_promotions = 0
    open_promotions = 0
    bridge_nodes = 0
    incumbent_updates = 0
    cursor = 0
    termination = "search_exhausted"
    own_agents = len(_side_data(game, red_player)[0])
    keep_children = max(2, int(math.ceil(settings.keep_children_multiplier * own_agents)))
    sample_children = max(
        keep_children,
        int(math.ceil(settings.child_sample_multiplier * own_agents)),
    )

    while expansions < settings.max_expansions:
        if time.perf_counter() - started >= settings.time_limit_seconds:
            termination = "time_limit"
            break
        available = False
        memory = None
        for _ in range(len(memories)):
            memory = memories[cursor % len(memories)]
            cursor += 1
            if not memory.open:
                if memory.substitute:
                    memory.open, memory.substitute = memory.substitute, []
                    substitute_promotions += 1
                elif memory.reserve:
                    memory.open, memory.reserve = memory.reserve, []
                    reserve_promotions += 1
            if memory.open:
                available = True
                break
        if not available or memory is None:
            break
        _, _, node = heapq.heappop(memory.open)
        expansions += 1

        pool: list[IdentityAction] = []
        gamma = incumbent_score
        for _ in range(settings.selection_rounds):
            children = _sample_neighbors(
                game,
                node,
                red_player=red_player,
                rng=rng,
                requested=sample_children,
            )
            generated += len(children)
            fresh = []
            for child in children:
                if child in discovered or child in pool:
                    conflicts += 1
                else:
                    fresh.append(child)
            if not fresh:
                continue
            values = evaluator.evaluate(fresh)
            scores = evaluator.score(values)
            pool.extend(fresh)
            min_score_seen = min(min_score_seen, float(np.min(scores)))
            best_child = float(np.max(scores))
            if best_child > gamma:
                break
            gamma -= (gamma - best_child) / 2.0
            if time.perf_counter() - started >= settings.time_limit_seconds:
                break
        if not pool:
            continue
        pool_values = evaluator.evaluate(pool)
        pool_scores = evaluator.score(pool_values)
        order = np.argsort(-pool_scores)[:keep_children]
        old_incumbent = incumbent
        improved = False
        for raw_index in order:
            index = int(raw_index)
            child = pool[index]
            value = float(pool_values[index])
            score = float(pool_scores[index])
            if child in discovered:
                conflicts += 1
                continue
            discovered.add(child)
            if score > incumbent_score + 1e-12:
                incumbent = child
                incumbent_value = value
                incumbent_score = score
                incumbent_updates += 1
                improved = True
            # The paper uses a multiplicative threshold for non-negative CSG
            # values.  Our zero-sum utilities can be negative, so we apply the
            # same omega threshold after an affine shift by the worst observed
            # score.  This preserves score ordering and avoids sign reversal.
            floor = min_score_seen - max(1e-9, abs(min_score_seen) * 1e-6)
            threshold = floor + settings.omega * (incumbent_score - floor)
            if score >= threshold:
                _push(memory.open, child, score, serial)
                open_promotions += 1
            else:
                _push(memory.reserve, child, score, serial)
                if len(memory.reserve) > settings.reserve_memory:
                    # Retain the highest-scoring reserve entries.
                    memory.reserve = heapq.nsmallest(
                        settings.reserve_memory, memory.reserve
                    )
                    heapq.heapify(memory.reserve)
            serial += 1
        if improved and settings.bridge_path_limit:
            path = _bridge_path(
                game,
                old_incumbent,
                incumbent,
                red_player=red_player,
                limit=settings.bridge_path_limit,
            )
            bridge_fresh = [value for value in path if value not in discovered]
            if bridge_fresh:
                bridge_values = evaluator.evaluate(bridge_fresh)
                bridge_scores = evaluator.score(bridge_values)
                for child, value, score in zip(
                    bridge_fresh, bridge_values, bridge_scores
                ):
                    discovered.add(child)
                    bridge_nodes += 1
                    if float(score) > incumbent_score + 1e-12:
                        incumbent = child
                        incumbent_value = float(value)
                        incumbent_score = float(score)
                        incumbent_updates += 1
                    _push(memory.substitute, child, float(score), serial)
                    serial += 1
    else:
        termination = "max_expansions"

    elapsed = time.perf_counter() - started
    response = IdentityBestResponse(
        action=incumbent,
        value=incumbent_value,
        optimal=False,
        solver_status=10,
        solver_message=f"SALDAE anytime search stopped by {termination}",
        mip_gap=float("nan"),
        mip_node_count=expansions,
        variables=len(evaluator.action_cache),
    )
    diagnostics = SALDAESearchDiagnostics(
        side=side,
        value=incumbent_value,
        objective_score=incumbent_score,
        elapsed_seconds=elapsed,
        expansions=expansions,
        evaluated_nodes=len(evaluator.action_cache),
        generated_nodes=generated,
        duplicate_conflicts=conflicts,
        open_promotions=open_promotions,
        substitute_promotions=substitute_promotions,
        reserve_promotions=reserve_promotions,
        bridge_nodes=bridge_nodes,
        incumbent_updates=incumbent_updates,
        search_agents=settings.search_agents,
        termination_reason=termination,
        used_unregistered_coalition=_uses_unregistered(
            game, incumbent, red_player=red_player
        ),
    )
    return response, diagnostics


def solve_saldae_double_oracle(
    game: IdentityBlottoGame,
    initial_red: Sequence[IdentityAction],
    initial_blue: Sequence[IdentityAction],
    *,
    tolerance: float = 1e-3,
    max_iterations: int = 12,
    saldae_config: SALDAEConfig = SALDAEConfig(),
    verbose: bool = False,
) -> tuple[IdentityDoubleOracleResult, SALDAEDoubleOracleDiagnostics]:
    """Solve the restricted game while SALDAE searches both best responses."""

    red = list(dict.fromkeys(game.validate_action(value, "Red") for value in initial_red))
    blue = list(dict.fromkeys(game.validate_action(value, "Blue") for value in initial_blue))
    if not red or not blue:
        raise ValueError("SALDAE-DO initial supports must be non-empty")
    history: list[IdentityDoubleOracleIteration] = []
    red_diagnostics: list[SALDAESearchDiagnostics] = []
    blue_diagnostics: list[SALDAESearchDiagnostics] = []
    final_solution = None
    empirical_converged = False
    reason = "max_iterations"
    base_seed = int(saldae_config.random_seed)

    for iteration in range(1, int(max_iterations) + 1):
        solution = solve_restricted_matrix_game(game.payoff_matrix(red, blue))
        red_response, red_diag = saldae_best_response(
            game,
            blue,
            solution.attacker_mixture,
            red_player=True,
            initial_actions=red,
            config=SALDAEConfig(
                **{
                    **saldae_config.__dict__,
                    "random_seed": base_seed + 2 * iteration,
                }
            ),
        )
        blue_response, blue_diag = saldae_best_response(
            game,
            red,
            solution.defender_mixture,
            red_player=False,
            initial_actions=blue,
            config=SALDAEConfig(
                **{
                    **saldae_config.__dict__,
                    "random_seed": base_seed + 2 * iteration + 1,
                }
            ),
        )
        red_diagnostics.append(red_diag)
        blue_diagnostics.append(blue_diag)
        red_gap = max(0.0, red_response.value - solution.value)
        blue_gap = max(0.0, solution.value - blue_response.value)
        empirical_gap = max(0.0, red_response.value - blue_response.value)
        add_red = (
            red_gap > float(tolerance) / 2.0
            and red_response.action not in red
            and iteration < int(max_iterations)
        )
        add_blue = (
            blue_gap > float(tolerance) / 2.0
            and blue_response.action not in blue
            and iteration < int(max_iterations)
        )
        history.append(
            IdentityDoubleOracleIteration(
                iteration=iteration,
                red_support=len(red),
                blue_support=len(blue),
                restricted_value=float(solution.value),
                red_best_response_value=red_response.value,
                blue_best_response_value=blue_response.value,
                red_gap=red_gap,
                blue_gap=blue_gap,
                exploitability=empirical_gap,
                red_oracle_optimal=False,
                blue_oracle_optimal=False,
                added_red=add_red,
                added_blue=add_blue,
            )
        )
        final_solution = solution
        if verbose:
            print(
                f"[Stage3][SALDAE-DO] iter={iteration:02d} "
                f"support=({len(red)},{len(blue)}) value={solution.value:+.5f} "
                f"empirical_gap={empirical_gap:.3e} "
                f"nodes=({red_diag.evaluated_nodes},{blue_diag.evaluated_nodes})",
                flush=True,
            )
        if empirical_gap <= float(tolerance) and not add_red and not add_blue:
            empirical_converged = True
            reason = "empirical_saldae_convergence"
            break
        if not add_red and not add_blue:
            reason = "saldae_stalled"
            break
        if add_red:
            red.append(red_response.action)
        if add_blue:
            blue.append(blue_response.action)
    if final_solution is None:
        raise RuntimeError("SALDAE-DO did not solve a restricted matrix game")
    last = history[-1]
    result = IdentityDoubleOracleResult(
        red_strategies=tuple(red),
        blue_strategies=tuple(blue),
        red_mixture=final_solution.defender_mixture.copy(),
        blue_mixture=final_solution.attacker_mixture.copy(),
        value=float(final_solution.value),
        exploitability=float(last.exploitability),
        converged=empirical_converged,
        # Anytime SALDAE responses do not certify either the old candidate
        # domain or the complete exponentially-large action domain.
        candidate_domain_exact=False,
        full_game_certified=False,
        termination_reason=reason,
        history=tuple(history),
    )
    diagnostics = SALDAEDoubleOracleDiagnostics(
        red_searches=tuple(red_diagnostics),
        blue_searches=tuple(blue_diagnostics),
        empirical_converged=empirical_converged,
        expanded_action_domain=any(
            row.used_unregistered_coalition
            for row in (*red_diagnostics, *blue_diagnostics)
        ),
    )
    return result, diagnostics


__all__ = [
    "SALDAEConfig",
    "SALDAEDoubleOracleDiagnostics",
    "SALDAESearchDiagnostics",
    "saldae_best_response",
    "solve_saldae_double_oracle",
]
