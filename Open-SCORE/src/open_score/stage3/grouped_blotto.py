"""Target-labelled coalition-structured Colonel Blotto.

The classical count action ``(agents at target 0, ..., agents at target M)``
cannot distinguish one 8-agent engagement from two 4-agent engagements at the
same target.  This module lifts that action to a target-by-group-size
histogram.  For a group-size support ``1..K`` a pure action is

``x[t, k-1] = number of own groups of size k assigned to target t``.

Agent identities are deliberately not part of this strategic action.  HAD's
agents are exchangeable at the game layer; identities are grounded after a
joint profile is sampled, using the current physical state.  This symmetry
reduction changes an intractable identity partition into bounded integer
partitions while still allowing several independently executed groups to
share one target.

At one target the two independently selected group multisets are paired by a
public, permutation-invariant rule: sizes are sorted from large to small and
paired by rank, padding the shorter side with empty groups.  Consequently a
player never selects the opponent's identities and cannot exploit arbitrary
unlabelled slot names.  The rule is an explicit modelling approximation; it
can later be replaced by a spatial or learned target-local matching game.

Under an additive target/engagement surrogate, a best response to an opponent
mixture is exact: enumerate the bounded integer partitions available at one
target, retain the best partition for each local resource total, and run a
resource dynamic program across targets.  Double Oracle therefore returns a
full-domain exploitability certificate for this registered anonymous game,
without enumerating all global pure strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import zip_longest
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .blotto import (
    Allocation,
    BestResponse,
    DoubleOracleIteration,
    DoubleOracleResult,
    MatrixGameSolution,
    solve_restricted_matrix_game,
)


GroupHistogram = Tuple[int, ...]
GroupedAllocation = Allocation


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _mixture(values: Sequence[float], expected: int, name: str) -> np.ndarray:
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.shape != (expected,):
        raise ValueError(f"{name} must have shape ({expected},)")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < -1e-12):
        raise ValueError(f"{name} must contain finite non-negative values")
    probabilities = np.clip(probabilities, 0.0, None)
    total = float(probabilities.sum())
    if total <= 0.0:
        raise ValueError(f"{name} must have positive mass")
    return probabilities / total


def histogram_resources(histogram: Sequence[int]) -> int:
    """Return the number of agents represented by a group histogram."""

    values = np.asarray(histogram)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise TypeError("group histogram must be a one-dimensional integer sequence")
    if np.any(values < 0):
        raise ValueError("group histogram counts must be non-negative")
    sizes = np.arange(1, len(values) + 1, dtype=np.int64)
    return int(values.astype(np.int64, copy=False) @ sizes)


def expand_group_histogram(histogram: Sequence[int]) -> Tuple[int, ...]:
    """Expand a histogram into a canonical descending sequence of group sizes."""

    values = tuple(int(value) for value in histogram)
    histogram_resources(values)  # validation
    return tuple(
        size
        for size in range(len(values), 0, -1)
        for _ in range(values[size - 1])
    )


@lru_cache(maxsize=None)
def group_histograms_for_total(total: int, group_size_cap: int) -> Tuple[GroupHistogram, ...]:
    """Enumerate integer partitions of ``total`` with parts in ``1..K``.

    Histograms remove permutations such as ``(3, 1)`` versus ``(1, 3)``.
    This is the key exchangeability reduction used by the best-response oracle.
    """

    total = _nonnegative_integer(total, "total")
    cap = _nonnegative_integer(group_size_cap, "group_size_cap")
    if cap < 1:
        raise ValueError("group_size_cap must be positive")
    result: List[GroupHistogram] = []
    counts = [0] * cap

    def visit(size: int, remaining: int) -> None:
        if size == 0:
            if remaining == 0:
                result.append(tuple(counts))
            return
        maximum = remaining // size
        for count in range(maximum + 1):
            counts[size - 1] = count
            visit(size - 1, remaining - count * size)
        counts[size - 1] = 0

    visit(cap, total)
    # Stable scientific artifacts: few groups first, then larger groups first.
    result.sort(
        key=lambda item: (
            sum(item),
            tuple(-value for value in reversed(item)),
            item,
        )
    )
    return tuple(result)


def pair_group_histograms(
    red_histogram: Sequence[int],
    blue_histogram: Sequence[int],
) -> Tuple[Tuple[int, int], ...]:
    """Pair canonical groups by descending size, padding with empty groups."""

    red = expand_group_histogram(red_histogram)
    blue = expand_group_histogram(blue_histogram)
    return tuple(
        (int(r), int(b))
        for r, b in zip_longest(red, blue, fillvalue=0)
    )


def histogram_from_group_sizes(
    group_sizes: Sequence[int], group_size_cap: int
) -> GroupHistogram:
    cap = _nonnegative_integer(group_size_cap, "group_size_cap")
    if cap < 1:
        raise ValueError("group_size_cap must be positive")
    histogram = [0] * cap
    for raw_size in group_sizes:
        size = _nonnegative_integer(raw_size, "group size")
        if not 1 <= size <= cap:
            raise ValueError(f"group size must lie in [1, {cap}]")
        histogram[size - 1] += 1
    return tuple(histogram)


def compact_partition(total: int, group_size_cap: int, preferred_size: int = 4) -> GroupHistogram:
    """Deterministically split a local total for baseline/initial strategies."""

    total = _nonnegative_integer(total, "total")
    cap = _nonnegative_integer(group_size_cap, "group_size_cap")
    preferred = _nonnegative_integer(preferred_size, "preferred_size")
    if cap < 1 or not 1 <= preferred <= cap:
        raise ValueError("preferred_size must lie inside the group-size support")
    groups: List[int] = [preferred] * (total // preferred)
    if total % preferred:
        groups.append(total % preferred)
    return histogram_from_group_sizes(groups, cap)


@dataclass(frozen=True)
class GroupedBlottoGame:
    """Finite zero-sum Blotto with target-labelled group-pattern actions.

    ``local_payoff[t, r, b]`` is Red utility for one paired engagement at
    target ``t`` with group sizes ``r`` and ``b``.  Index zero is an exact
    empty-side boundary supplied by Stage 2.  A global action concatenates
    Red and Blue histograms may have different caps at a late event when one
    side has fewer live agents than the registered maximum.
    """

    local_payoff: np.ndarray
    defender_budget: int
    attacker_budget: int
    allow_defender_reserve: bool = False
    allow_attacker_reserve: bool = False
    event_id: str = "grouped-event"
    pairing_rule: str = "size_assortative"

    def __post_init__(self) -> None:
        payoff = np.asarray(self.local_payoff, dtype=np.float64)
        if payoff.ndim != 3 or payoff.shape[0] < 1:
            raise ValueError("local_payoff must have shape [targets, K+1, K+1]")
        if payoff.shape[1] < 2 or payoff.shape[2] < 2:
            raise ValueError("both group-size axes must have a positive cap")
        if not np.all(np.isfinite(payoff)):
            raise ValueError("local_payoff must be finite")
        defender_budget = _nonnegative_integer(self.defender_budget, "defender_budget")
        attacker_budget = _nonnegative_integer(self.attacker_budget, "attacker_budget")
        if self.pairing_rule != "size_assortative":
            raise ValueError("the registered grouped game uses size_assortative pairing")
        payoff = payoff.copy()
        payoff.setflags(write=False)
        object.__setattr__(self, "local_payoff", payoff)
        object.__setattr__(self, "defender_budget", defender_budget)
        object.__setattr__(self, "attacker_budget", attacker_budget)
        object.__setattr__(self, "event_id", str(self.event_id))
        object.__setattr__(
            self,
            "_defender_patterns",
            tuple(
                group_histograms_for_total(total, self.red_group_size_cap)
                for total in range(defender_budget + 1)
            ),
        )
        object.__setattr__(
            self,
            "_attacker_patterns",
            tuple(
                group_histograms_for_total(total, self.blue_group_size_cap)
                for total in range(attacker_budget + 1)
            ),
        )
        object.__setattr__(self, "_target_payoff_cache", {})

    @property
    def n_targets(self) -> int:
        return int(self.local_payoff.shape[0])

    @property
    def n_battlefields(self) -> int:
        """Compatibility alias: the physical battlefields are the targets."""

        return self.n_targets

    @property
    def red_group_size_cap(self) -> int:
        return int(self.local_payoff.shape[1] - 1)

    @property
    def blue_group_size_cap(self) -> int:
        return int(self.local_payoff.shape[2] - 1)

    @property
    def group_size_cap(self) -> int:
        """Largest group size represented on either side."""

        return max(self.red_group_size_cap, self.blue_group_size_cap)

    @property
    def defender_action_dimension(self) -> int:
        return self.n_targets * self.red_group_size_cap

    @property
    def attacker_action_dimension(self) -> int:
        return self.n_targets * self.blue_group_size_cap

    @property
    def defender_patterns(self) -> Tuple[Tuple[GroupHistogram, ...], ...]:
        return self._defender_patterns

    @property
    def attacker_patterns(self) -> Tuple[Tuple[GroupHistogram, ...], ...]:
        return self._attacker_patterns

    def _validate_allocation(
        self,
        allocation: Sequence[int],
        budget: int,
        group_size_cap: int,
        allow_reserve: bool,
        name: str,
    ) -> GroupedAllocation:
        values = np.asarray(allocation)
        action_dimension = self.n_targets * int(group_size_cap)
        if values.shape != (action_dimension,):
            raise ValueError(f"{name} must have shape ({action_dimension},)")
        if not np.issubdtype(values.dtype, np.integer):
            raise TypeError(f"{name} must contain integers")
        values = values.astype(np.int64, copy=False)
        if np.any(values < 0):
            raise ValueError(f"{name} group counts must be non-negative")
        matrix = values.reshape(self.n_targets, int(group_size_cap))
        used = int(
            np.sum(matrix * np.arange(1, int(group_size_cap) + 1)[None, :])
        )
        if used > budget or (not allow_reserve and used != budget):
            relation = "equal" if not allow_reserve else "not exceed"
            raise ValueError(f"{name} resource total must {relation} its budget")
        return tuple(int(value) for value in values)

    def validate_defender_allocation(self, allocation: Sequence[int]) -> GroupedAllocation:
        return self._validate_allocation(
            allocation,
            self.defender_budget,
            self.red_group_size_cap,
            bool(self.allow_defender_reserve),
            "defender grouped allocation",
        )

    def validate_attacker_allocation(self, allocation: Sequence[int]) -> GroupedAllocation:
        return self._validate_allocation(
            allocation,
            self.attacker_budget,
            self.blue_group_size_cap,
            bool(self.allow_attacker_reserve),
            "attacker grouped allocation",
        )

    def target_histogram(
        self, allocation: Sequence[int], target: int, side: str
    ) -> GroupHistogram:
        values = tuple(int(value) for value in allocation)
        if side == "Red":
            cap = self.red_group_size_cap
        elif side == "Blue":
            cap = self.blue_group_size_cap
        else:
            raise ValueError("side must be Red or Blue")
        if len(values) != self.n_targets * cap:
            raise ValueError("grouped allocation has the wrong dimension")
        index = _nonnegative_integer(target, "target")
        if index >= self.n_targets:
            raise ValueError("target index is outside this game")
        start = index * cap
        return values[start : start + cap]

    def target_counts(self, allocation: Sequence[int], side: str) -> Tuple[int, ...]:
        values = tuple(int(value) for value in allocation)
        dimension = (
            self.defender_action_dimension
            if side == "Red"
            else self.attacker_action_dimension
            if side == "Blue"
            else -1
        )
        if len(values) != dimension:
            raise ValueError("grouped allocation has the wrong dimension")
        return tuple(
            histogram_resources(self.target_histogram(values, target, side))
            for target in range(self.n_targets)
        )

    def groups(
        self, allocation: Sequence[int], side: str
    ) -> Tuple[Tuple[int, ...], ...]:
        values = tuple(int(value) for value in allocation)
        return tuple(
            expand_group_histogram(self.target_histogram(values, target, side))
            for target in range(self.n_targets)
        )

    def target_payoff(
        self,
        target: int,
        red_histogram: Sequence[int],
        blue_histogram: Sequence[int],
    ) -> float:
        red = tuple(int(value) for value in red_histogram)
        blue = tuple(int(value) for value in blue_histogram)
        if len(red) != self.red_group_size_cap:
            raise ValueError("Red target histogram has the wrong group-size cap")
        if len(blue) != self.blue_group_size_cap:
            raise ValueError("Blue target histogram has the wrong group-size cap")
        histogram_resources(red)
        histogram_resources(blue)
        key = (int(target), red, blue)
        cached = self._target_payoff_cache.get(key)
        if cached is not None:
            return cached
        value = float(
            sum(
                self.local_payoff[int(target), red_size, blue_size]
                for red_size, blue_size in pair_group_histograms(red, blue)
            )
        )
        self._target_payoff_cache[key] = value
        return value

    def payoff(self, defender: Sequence[int], attacker: Sequence[int]) -> float:
        red = self.validate_defender_allocation(defender)
        blue = self.validate_attacker_allocation(attacker)
        return float(
            sum(
                self.target_payoff(
                    target,
                    self.target_histogram(red, target, "Red"),
                    self.target_histogram(blue, target, "Blue"),
                )
                for target in range(self.n_targets)
            )
        )

    def payoff_matrix(
        self,
        defender_strategies: Sequence[Sequence[int]],
        attacker_strategies: Sequence[Sequence[int]],
    ) -> np.ndarray:
        red = [self.validate_defender_allocation(item) for item in defender_strategies]
        blue = [self.validate_attacker_allocation(item) for item in attacker_strategies]
        if not red or not blue:
            raise ValueError("both restricted strategy sets must be non-empty")
        matrix = np.empty((len(red), len(blue)), dtype=np.float64)
        for row, defender in enumerate(red):
            for column, attacker in enumerate(blue):
                matrix[row, column] = self.payoff(defender, attacker)
        return matrix

    def _best_response(
        self,
        opponent_strategies: Sequence[Sequence[int]],
        opponent_mixture: Sequence[float],
        *,
        defender: bool,
    ) -> BestResponse:
        validate = (
            self.validate_attacker_allocation
            if defender
            else self.validate_defender_allocation
        )
        opponents = [validate(item) for item in opponent_strategies]
        probabilities = _mixture(
            opponent_mixture, len(opponents), "opponent grouped mixture"
        )
        budget = self.defender_budget if defender else self.attacker_budget
        allow_reserve = (
            bool(self.allow_defender_reserve)
            if defender
            else bool(self.allow_attacker_reserve)
        )
        patterns_by_total = (
            self.defender_patterns if defender else self.attacker_patterns
        )

        # Additivity means a correlated global opponent mixture is represented
        # exactly by its target-wise pattern marginals for best-response use.
        marginals: List[Dict[GroupHistogram, float]] = []
        for target in range(self.n_targets):
            distribution: Dict[GroupHistogram, float] = {}
            for allocation, probability in zip(opponents, probabilities):
                pattern = self.target_histogram(
                    allocation, target, "Blue" if defender else "Red"
                )
                distribution[pattern] = distribution.get(pattern, 0.0) + float(
                    probability
                )
            marginals.append(distribution)

        # Flatten all integer partitions once, then evaluate them as a dense
        # [pattern, rank] array.  A Python loop over every partition/opponent
        # pair made the exact oracle unnecessarily slow at 30 agents even
        # though the underlying operation is only a table lookup and sum.
        flat_patterns: List[GroupHistogram] = []
        total_slices: List[slice] = []
        for patterns in patterns_by_total:
            start = len(flat_patterns)
            flat_patterns.extend(patterns)
            total_slices.append(slice(start, len(flat_patterns)))
        maximum_ranks = max(self.defender_budget, self.attacker_budget, 1)
        own_sizes = np.zeros(
            (len(flat_patterns), maximum_ranks), dtype=np.int16
        )
        for index, pattern in enumerate(flat_patterns):
            expanded = expand_group_histogram(pattern)
            own_sizes[index, : len(expanded)] = expanded

        # For a fixed target and local resource total, only the locally best
        # partition can matter to the across-target resource dynamic program.
        local_values = np.full(
            (self.n_targets, budget + 1),
            -np.inf if defender else np.inf,
            dtype=np.float64,
        )
        local_patterns: List[List[Optional[GroupHistogram]]] = [
            [None] * (budget + 1) for _ in range(self.n_targets)
        ]
        for target in range(self.n_targets):
            expected = np.zeros(len(flat_patterns), dtype=np.float64)
            table = self.local_payoff[target]
            for other, probability in marginals[target].items():
                other_sizes = np.zeros(maximum_ranks, dtype=np.int16)
                expanded = expand_group_histogram(other)
                other_sizes[: len(expanded)] = expanded
                if defender:
                    pair_values = table[own_sizes, other_sizes[None, :]]
                else:
                    pair_values = table[other_sizes[None, :], own_sizes]
                # Canonical pairing stops after the final non-empty group;
                # dense padding must not invent arbitrary 0v0 payoffs.
                pair_values = np.where(
                    (own_sizes == 0) & (other_sizes[None, :] == 0),
                    0.0,
                    pair_values,
                )
                expected += float(probability) * np.sum(pair_values, axis=1)
            for total, pattern_slice in enumerate(total_slices):
                values = expected[pattern_slice]
                local_index = int(np.argmax(values) if defender else np.argmin(values))
                index = int(pattern_slice.start) + local_index
                local_values[target, total] = float(expected[index])
                local_patterns[target][total] = flat_patterns[index]

        unreachable = -np.inf if defender else np.inf
        previous = np.full(budget + 1, unreachable, dtype=np.float64)
        previous[0] = 0.0
        choices = np.full((self.n_targets, budget + 1), -1, dtype=np.int32)
        for target in range(self.n_targets):
            current = np.full(budget + 1, unreachable, dtype=np.float64)
            for local_total in range(budget + 1):
                candidates = previous[: budget + 1 - local_total] + local_values[
                    target, local_total
                ]
                destination = current[local_total:]
                better = candidates > destination if defender else candidates < destination
                if np.any(better):
                    destination[better] = candidates[better]
                    choices[target, local_total:][better] = local_total
            previous = current

        if allow_reserve:
            feasible = np.flatnonzero(np.isfinite(previous))
            if feasible.size == 0:
                raise RuntimeError("grouped best response is infeasible")
            values = previous[feasible]
            offset = int(np.argmax(values) if defender else np.argmin(values))
            used = int(feasible[offset])
        else:
            used = budget
            if not np.isfinite(previous[used]):
                raise RuntimeError("grouped best response cannot spend the full budget")

        selected: List[GroupHistogram] = [()] * self.n_targets
        remaining = used
        for target in range(self.n_targets - 1, -1, -1):
            local_total = int(choices[target, remaining])
            if local_total < 0:
                raise RuntimeError("failed to reconstruct grouped best response")
            pattern = local_patterns[target][local_total]
            if pattern is None:
                raise RuntimeError("grouped best response omitted a local pattern")
            selected[target] = pattern
            remaining -= local_total
        if remaining != 0:
            raise RuntimeError("grouped best-response reconstruction left resources")
        allocation = tuple(value for pattern in selected for value in pattern)
        return BestResponse(
            allocation=allocation,
            value=float(previous[used]),
            assigned_resources=used,
            reserve_resources=budget - used,
        )

    def defender_best_response(
        self,
        attacker_strategies: Sequence[Sequence[int]],
        attacker_mixture: Sequence[float],
    ) -> BestResponse:
        return self._best_response(
            attacker_strategies, attacker_mixture, defender=True
        )

    def attacker_best_response(
        self,
        defender_strategies: Sequence[Sequence[int]],
        defender_mixture: Sequence[float],
    ) -> BestResponse:
        return self._best_response(
            defender_strategies, defender_mixture, defender=False
        )


def grouped_allocation_from_target_counts(
    counts: Sequence[int],
    group_size_cap: int,
    *,
    preferred_size: int = 4,
) -> GroupedAllocation:
    """Convert a target-count baseline into an explicit group action."""

    return tuple(
        value
        for raw_count in counts
        for value in compact_partition(
            _nonnegative_integer(raw_count, "target count"),
            group_size_cap,
            preferred_size,
        )
    )


def _balanced_target_counts(budget: int, targets: int, offset: int = 0) -> Tuple[int, ...]:
    values = [0] * targets
    for resource in range(int(budget)):
        values[(int(offset) + resource) % targets] += 1
    return tuple(values)


def balanced_grouped_allocation(
    game: GroupedBlottoGame,
    side: str,
    *,
    offset: int = 0,
    preferred_size: int = 4,
) -> GroupedAllocation:
    if side not in {"Red", "Blue"}:
        raise ValueError("side must be Red or Blue")
    budget = game.defender_budget if side == "Red" else game.attacker_budget
    cap = game.red_group_size_cap if side == "Red" else game.blue_group_size_cap
    result = grouped_allocation_from_target_counts(
        _balanced_target_counts(budget, game.n_targets, offset),
        cap,
        preferred_size=min(int(preferred_size), cap),
    )
    return (
        game.validate_defender_allocation(result)
        if side == "Red"
        else game.validate_attacker_allocation(result)
    )


def concentrated_grouped_allocation(
    game: GroupedBlottoGame,
    side: str,
    *,
    target: int = 0,
    preferred_size: int = 4,
) -> GroupedAllocation:
    if side not in {"Red", "Blue"}:
        raise ValueError("side must be Red or Blue")
    target = int(target) % game.n_targets
    budget = game.defender_budget if side == "Red" else game.attacker_budget
    cap = game.red_group_size_cap if side == "Red" else game.blue_group_size_cap
    counts = [0] * game.n_targets
    counts[target] = budget
    result = grouped_allocation_from_target_counts(
        counts, cap, preferred_size=min(int(preferred_size), cap)
    )
    return (
        game.validate_defender_allocation(result)
        if side == "Red"
        else game.validate_attacker_allocation(result)
    )


def enumerate_grouped_allocations(
    game: GroupedBlottoGame, side: str
) -> Tuple[GroupedAllocation, ...]:
    """Enumerate the full grouped action domain for toy exact validation only."""

    if side not in {"Red", "Blue"}:
        raise ValueError("side must be Red or Blue")
    budget = game.defender_budget if side == "Red" else game.attacker_budget
    allow_reserve = (
        game.allow_defender_reserve if side == "Red" else game.allow_attacker_reserve
    )
    patterns = game.defender_patterns if side == "Red" else game.attacker_patterns
    result: List[GroupedAllocation] = []

    def visit(target: int, remaining: int, prefix: Tuple[int, ...]) -> None:
        if target == game.n_targets:
            if allow_reserve or remaining == 0:
                result.append(prefix)
            return
        for used in range(remaining + 1):
            for pattern in patterns[used]:
                visit(target + 1, remaining - used, prefix + pattern)

    visit(0, budget, ())
    return tuple(result)


def _unique(items: Iterable[GroupedAllocation]) -> List[GroupedAllocation]:
    result: List[GroupedAllocation] = []
    seen = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def solve_grouped_double_oracle(
    game: GroupedBlottoGame,
    *,
    initial_defender_strategies: Optional[Sequence[Sequence[int]]] = None,
    initial_attacker_strategies: Optional[Sequence[Sequence[int]]] = None,
    max_iterations: int = 60,
    tolerance: float = 1e-6,
    seed: int = 0,
    verbose: bool = False,
) -> DoubleOracleResult:
    """Solve the anonymous grouped game with exact pattern-DP best responses."""

    if not isinstance(game, GroupedBlottoGame):
        raise TypeError("game must be a GroupedBlottoGame")
    max_iterations = _nonnegative_integer(max_iterations, "max_iterations")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    if initial_defender_strategies is None:
        defender_support = [balanced_grouped_allocation(game, "Red")]
    else:
        defender_support = [
            game.validate_defender_allocation(item)
            for item in initial_defender_strategies
        ]
    if initial_attacker_strategies is None:
        attacker_support = [balanced_grouped_allocation(game, "Blue")]
    else:
        attacker_support = [
            game.validate_attacker_allocation(item)
            for item in initial_attacker_strategies
        ]
    defender_support = _unique(defender_support)
    attacker_support = _unique(attacker_support)
    history: List[DoubleOracleIteration] = []
    final_solution: Optional[MatrixGameSolution] = None
    used_fallback = False
    converged = False
    reason = "max_iterations"
    if verbose:
        print(
            f"[Stage3][GroupedDO] event={game.event_id} targets={game.n_targets} "
            f"R={game.defender_budget} B={game.attacker_budget} K={game.group_size_cap}",
            flush=True,
        )
    for iteration in range(1, max_iterations + 1):
        matrix = game.payoff_matrix(defender_support, attacker_support)
        solution = solve_restricted_matrix_game(matrix)
        used_fallback = used_fallback or solution.used_fallback
        red_br = game.defender_best_response(
            attacker_support, solution.attacker_mixture
        )
        blue_br = game.attacker_best_response(
            defender_support, solution.defender_mixture
        )
        red_gap = max(0.0, red_br.value - solution.value)
        blue_gap = max(0.0, solution.value - blue_br.value)
        exploitability = max(0.0, red_br.value - blue_br.value)
        certified = exploitability <= tolerance
        add_red = (
            red_gap > tolerance / 2.0
            and red_br.allocation not in defender_support
            and not certified
            and iteration < max_iterations
        )
        add_blue = (
            blue_gap > tolerance / 2.0
            and blue_br.allocation not in attacker_support
            and not certified
            and iteration < max_iterations
        )
        history.append(
            DoubleOracleIteration(
                iteration=iteration,
                defender_support_size=len(defender_support),
                attacker_support_size=len(attacker_support),
                restricted_value=solution.value,
                defender_best_response_value=red_br.value,
                attacker_best_response_value=blue_br.value,
                defender_gap=red_gap,
                attacker_gap=blue_gap,
                exploitability=exploitability,
                defender_best_response=red_br.allocation,
                attacker_best_response=blue_br.allocation,
                added_defender_strategy=add_red,
                added_attacker_strategy=add_blue,
            )
        )
        final_solution = solution
        if verbose:
            print(
                f"[Stage3][GroupedDO] iter={iteration:03d} "
                f"support=({len(defender_support)},{len(attacker_support)}) "
                f"value={solution.value:+.6f} gap={exploitability:.3e}",
                flush=True,
            )
        if certified:
            converged = True
            reason = "converged"
            break
        if not add_red and not add_blue:
            reason = "oracle_stalled"
            break
        if add_red:
            defender_support.append(red_br.allocation)
        if add_blue:
            attacker_support.append(blue_br.allocation)
    if final_solution is None:
        raise RuntimeError("grouped double oracle produced no restricted solution")
    last = history[-1]
    return DoubleOracleResult(
        defender_strategies=tuple(defender_support),
        attacker_strategies=tuple(attacker_support),
        defender_mixture=final_solution.defender_mixture.copy(),
        attacker_mixture=final_solution.attacker_mixture.copy(),
        value=final_solution.value,
        exploitability=last.exploitability,
        defender_gap=last.defender_gap,
        attacker_gap=last.attacker_gap,
        converged=converged,
        termination_reason=reason,
        history=tuple(history),
        seed=int(seed),
        used_matrix_game_fallback=used_fallback,
    )


__all__ = [
    "GroupHistogram",
    "GroupedAllocation",
    "GroupedBlottoGame",
    "balanced_grouped_allocation",
    "compact_partition",
    "concentrated_grouped_allocation",
    "enumerate_grouped_allocations",
    "expand_group_histogram",
    "group_histograms_for_total",
    "grouped_allocation_from_target_counts",
    "histogram_from_group_sizes",
    "histogram_resources",
    "pair_group_histograms",
    "solve_grouped_double_oracle",
]
