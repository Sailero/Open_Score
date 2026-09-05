"""Bayesian event-driven Colonel Blotto over explicit Blue grouping types.

Nature draws a Blue type.  Blue observes its own type and chooses a legal
target-count allocation for that type; Red knows the type prior but chooses a
single allocation without seeing the draw.  A Blue pure strategy is therefore
a type-contingent tuple of allocations.  The restricted game is solved by
Double Oracle, while both full-space best responses remain dynamic programs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .blotto import (
    Allocation,
    BestResponse,
    EventBlottoGame,
    MatrixGameSolution,
    solve_restricted_matrix_game,
)


@dataclass(frozen=True)
class BlueGroupingType:
    """One private Blue grouping type in the Bayesian Blotto game.

    ``candidate_allocations=None`` means that the type may use the complete
    attacker allocation domain and receives an exact DP best response.
    A finite tuple represents a policy family such as balanced-only or
    concentrated-on-one-target.  ``micro_grouping`` is carried into physical
    grounding so the same type also determines how target groups become local
    subgames.
    """

    name: str
    prior: float
    game: EventBlottoGame
    allocation_family: str = "strategic"
    candidate_allocations: Optional[Tuple[Allocation, ...]] = None
    lower_style: str = "rush"
    micro_grouping: str = "balanced_chunks"

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("Blue grouping type needs a non-empty name")
        prior = float(self.prior)
        if not np.isfinite(prior) or prior <= 0.0:
            raise ValueError("Blue grouping type prior must be finite and positive")
        if not isinstance(self.game, EventBlottoGame):
            raise TypeError("Blue grouping type game must be an EventBlottoGame")
        family = str(self.allocation_family)
        if family not in {"strategic", "balanced", "concentrated"}:
            raise ValueError(
                "allocation_family must be strategic, balanced or concentrated"
            )
        candidates = self.candidate_allocations
        if candidates is not None:
            validated = tuple(
                self.game.validate_attacker_allocation(allocation)
                for allocation in candidates
            )
            if not validated:
                raise ValueError("a finite Blue type needs at least one allocation")
            validated = tuple(dict.fromkeys(validated))
            object.__setattr__(self, "candidate_allocations", validated)
        micro_grouping = str(self.micro_grouping)
        if micro_grouping not in {"balanced_chunks", "attacker_matched"}:
            raise ValueError(
                "micro_grouping must be balanced_chunks or attacker_matched"
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "prior", prior)
        object.__setattr__(self, "allocation_family", family)
        object.__setattr__(self, "lower_style", str(self.lower_style))
        object.__setattr__(self, "micro_grouping", micro_grouping)


@dataclass(frozen=True)
class TypedBluePolicy:
    """One Blue pure policy: one target allocation for every private type."""

    allocations: Tuple[Allocation, ...]


@dataclass(frozen=True)
class BayesianProfileSample:
    defender_allocation: Allocation
    blue_type_index: int
    blue_type_name: str
    attacker_allocation: Allocation
    lower_style: str
    micro_grouping: str


@dataclass(frozen=True)
class BayesianDoubleOracleIteration:
    iteration: int
    defender_support_size: int
    attacker_policy_support_size: int
    restricted_value: float
    defender_best_response_value: float
    attacker_best_response_value: float
    exploitability: float
    defender_best_response: Allocation
    attacker_best_response: TypedBluePolicy
    added_defender_strategy: bool
    added_attacker_policy: bool


@dataclass(frozen=True)
class BayesianDoubleOracleResult:
    defender_strategies: Tuple[Allocation, ...]
    attacker_policies: Tuple[TypedBluePolicy, ...]
    defender_mixture: np.ndarray
    attacker_policy_mixture: np.ndarray
    blue_type_names: Tuple[str, ...]
    blue_type_prior: np.ndarray
    value: float
    exploitability: float
    defender_gap: float
    attacker_gap: float
    converged: bool
    termination_reason: str
    history: Tuple[BayesianDoubleOracleIteration, ...]
    seed: int
    used_matrix_game_fallback: bool
    lower_styles: Tuple[str, ...]
    micro_groupings: Tuple[str, ...]

    def sample_profile(
        self,
        *,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        blue_type_name: Optional[str] = None,
    ) -> BayesianProfileSample:
        """Sample Red and Blue equilibrium strategies, then realise a Blue type.

        ``blue_type_name`` conditions an evaluation on one registered type
        without revealing that type to Red: Red is still sampled from the same
        ex-ante Bayesian equilibrium.  Omitting it samples Nature from the
        registered prior.
        """

        if seed is not None and rng is not None:
            raise ValueError("pass either seed or rng, not both")
        generator = rng if rng is not None else np.random.default_rng(
            self.seed if seed is None else int(seed)
        )
        defender_index = int(
            generator.choice(len(self.defender_strategies), p=self.defender_mixture)
        )
        if blue_type_name is None:
            type_index = int(
                generator.choice(len(self.blue_type_names), p=self.blue_type_prior)
            )
        else:
            try:
                type_index = self.blue_type_names.index(str(blue_type_name))
            except ValueError as error:
                raise ValueError(
                    f"unknown Blue type {blue_type_name!r}; "
                    f"expected one of {self.blue_type_names}"
                ) from error
        policy_index = int(
            generator.choice(
                len(self.attacker_policies), p=self.attacker_policy_mixture
            )
        )
        return BayesianProfileSample(
            defender_allocation=self.defender_strategies[defender_index],
            blue_type_index=type_index,
            blue_type_name=self.blue_type_names[type_index],
            attacker_allocation=self.attacker_policies[policy_index].allocations[
                type_index
            ],
            lower_style=self.lower_styles[type_index],
            micro_grouping=self.micro_groupings[type_index],
        )


@dataclass(frozen=True)
class BayesianEventBlottoGame:
    """Zero-sum Harsanyi transformation of a typed dynamic Blotto event."""

    blue_types: Tuple[BlueGroupingType, ...]
    event_id: str = "bayesian-event"

    def __post_init__(self) -> None:
        types = tuple(self.blue_types)
        if not types:
            raise ValueError("Bayesian Blotto needs at least one Blue type")
        if len({item.name for item in types}) != len(types):
            raise ValueError("Blue grouping type names must be unique")
        reference = types[0].game
        for item in types[1:]:
            game = item.game
            common_scalars = (
                game.n_battlefields == reference.n_battlefields
                and game.defender_budget == reference.defender_budget
                and game.allow_defender_reserve == reference.allow_defender_reserve
                and game.max_defenders_per_battlefield
                == reference.max_defenders_per_battlefield
            )
            common_arrays = np.array_equal(
                game.defender_caps, reference.defender_caps
            ) and np.array_equal(
                game.defender_count_mask, reference.defender_count_mask
            )
            if not common_scalars or not common_arrays:
                raise ValueError(
                    "all Blue types must share the same Red allocation domain"
                )
        prior = np.asarray([item.prior for item in types], dtype=np.float64)
        prior /= prior.sum()
        prior.setflags(write=False)
        object.__setattr__(self, "blue_types", types)
        object.__setattr__(self, "_prior", prior)
        object.__setattr__(self, "event_id", str(self.event_id))

    @property
    def reference_game(self) -> EventBlottoGame:
        return self.blue_types[0].game

    @property
    def prior(self) -> np.ndarray:
        return self._prior

    def validate_defender_allocation(self, allocation: Sequence[int]) -> Allocation:
        return self.reference_game.validate_defender_allocation(allocation)

    def validate_attacker_policy(self, policy: TypedBluePolicy) -> TypedBluePolicy:
        if not isinstance(policy, TypedBluePolicy):
            raise TypeError("attacker policy must be a TypedBluePolicy")
        if len(policy.allocations) != len(self.blue_types):
            raise ValueError("typed attacker policy must have one allocation per type")
        validated = []
        for item, allocation in zip(self.blue_types, policy.allocations):
            value = item.game.validate_attacker_allocation(allocation)
            if (
                item.candidate_allocations is not None
                and value not in item.candidate_allocations
            ):
                raise ValueError(
                    f"attacker allocation is outside Blue type {item.name!r}"
                )
            validated.append(value)
        return TypedBluePolicy(tuple(validated))

    def payoff(self, defender: Sequence[int], attacker: TypedBluePolicy) -> float:
        red = self.validate_defender_allocation(defender)
        blue = self.validate_attacker_policy(attacker)
        return float(
            sum(
                probability * item.game.payoff(red, allocation)
                for probability, item, allocation in zip(
                    self.prior, self.blue_types, blue.allocations
                )
            )
        )

    def payoff_matrix(
        self,
        defenders: Sequence[Sequence[int]],
        attackers: Sequence[TypedBluePolicy],
    ) -> np.ndarray:
        red = [self.validate_defender_allocation(item) for item in defenders]
        blue = [self.validate_attacker_policy(item) for item in attackers]
        matrix = np.zeros((len(red), len(blue)), dtype=np.float64)
        for type_index, (probability, item) in enumerate(
            zip(self.prior, self.blue_types)
        ):
            allocations = [policy.allocations[type_index] for policy in blue]
            matrix += probability * item.game.payoff_matrix(red, allocations)
        return matrix

    def defender_best_response(
        self,
        attacker_policies: Sequence[TypedBluePolicy],
        mixture: Sequence[float],
    ) -> BestResponse:
        policies = [self.validate_attacker_policy(item) for item in attacker_policies]
        weights = _probabilities(mixture, len(policies), "attacker policy mixture")
        reference = self.reference_game
        expected = np.zeros(
            (reference.n_battlefields, reference.max_defenders_per_battlefield + 1),
            dtype=np.float64,
        )
        battlefield = np.arange(reference.n_battlefields)
        red_counts = np.arange(reference.max_defenders_per_battlefield + 1)
        for policy_weight, policy in zip(weights, policies):
            for type_weight, type_index, item in zip(
                self.prior, range(len(self.blue_types)), self.blue_types
            ):
                blue = np.asarray(policy.allocations[type_index], dtype=np.int64)
                expected += (
                    policy_weight
                    * type_weight
                    * item.game.local_payoff[
                        battlefield[:, None], red_counts[None, :], blue[:, None]
                    ]
                )
        synthetic = EventBlottoGame(
            expected[:, :, None],
            defender_budget=reference.defender_budget,
            attacker_budget=0,
            defender_caps=reference.defender_caps,
            attacker_caps=np.zeros(reference.n_battlefields, dtype=np.int64),
            allow_defender_reserve=reference.allow_defender_reserve,
            allow_attacker_reserve=False,
            defender_count_mask=reference.defender_count_mask,
            event_id=f"{self.event_id}:red-br",
        )
        return synthetic.defender_best_response(
            [(0,) * reference.n_battlefields], [1.0]
        )

    def attacker_best_response(
        self,
        defender_strategies: Sequence[Sequence[int]],
        mixture: Sequence[float],
    ) -> tuple[TypedBluePolicy, float]:
        defenders = [self.validate_defender_allocation(item) for item in defender_strategies]
        weights = _probabilities(mixture, len(defenders), "defender mixture")
        allocations = []
        total_value = 0.0
        for probability, item in zip(self.prior, self.blue_types):
            if item.candidate_allocations is None:
                response = item.game.attacker_best_response(defenders, weights)
                allocation = response.allocation
                value = response.value
            else:
                candidates = item.candidate_allocations
                matrix = item.game.payoff_matrix(defenders, candidates)
                values = weights @ matrix
                index = int(np.argmin(values))
                allocation = candidates[index]
                value = float(values[index])
            allocations.append(allocation)
            total_value += float(probability) * float(value)
        return TypedBluePolicy(tuple(allocations)), float(total_value)


def _probabilities(
    values: Sequence[float], expected: int, name: str
) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (expected,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector of length {expected}")
    if np.any(result < 0.0) or result.sum() <= 0.0:
        raise ValueError(f"{name} must be non-negative with positive mass")
    return result / result.sum()


def _balanced_feasible_allocation(
    budget: int,
    count_mask: np.ndarray,
    *,
    allow_reserve: bool,
) -> Allocation:
    """Find the feasible allocation closest to equal coverage by small DP."""

    target = float(budget) / len(count_mask)
    states: dict[int, tuple[float, Allocation]] = {0: (0.0, ())}
    for mask in count_mask:
        next_states: dict[int, tuple[float, Allocation]] = {}
        for used, (score, prefix) in states.items():
            for count in np.flatnonzero(mask):
                total = used + int(count)
                if total > budget:
                    continue
                candidate = (score - abs(float(count) - target), prefix + (int(count),))
                previous = next_states.get(total)
                if previous is None or candidate[0] > previous[0] or (
                    candidate[0] == previous[0] and candidate[1] < previous[1]
                ):
                    next_states[total] = candidate
        states = next_states
    if not states:
        raise ValueError("allocation domain contains no feasible balanced seed")
    if allow_reserve:
        _, allocation = max(
            states.values(), key=lambda item: (item[0], sum(item[1]), tuple(-x for x in item[1]))
        )
    else:
        if budget not in states:
            raise ValueError("allocation domain cannot spend the exact budget")
        _, allocation = states[budget]
    return allocation


def make_blue_grouping_type(
    game: EventBlottoGame,
    *,
    name: str,
    prior: float,
    allocation_family: str,
    lower_style: str = "rush",
    micro_grouping: str = "balanced_chunks",
) -> BlueGroupingType:
    """Build a preregisterable Blue type from a compact policy-family name."""

    family = str(allocation_family)
    if family == "strategic":
        candidates = None
    elif family == "balanced":
        candidates = (
            _balanced_feasible_allocation(
                game.attacker_budget,
                np.asarray(game.attacker_count_mask),
                allow_reserve=game.allow_attacker_reserve,
            ),
        )
    elif family == "concentrated":
        candidates = []
        for first_target in range(game.n_battlefields):
            remaining = int(game.attacker_budget)
            allocation = np.zeros(game.n_battlefields, dtype=np.int64)
            for step in range(game.n_battlefields):
                target = (first_target + step) % game.n_battlefields
                legal = np.flatnonzero(game.attacker_count_mask[target])
                legal = legal[legal <= remaining]
                if legal.size:
                    allocation[target] = int(legal[-1])
                    remaining -= int(legal[-1])
            candidate = tuple(map(int, allocation))
            try:
                candidate = game.validate_attacker_allocation(candidate)
            except ValueError:
                continue
            candidates.append(candidate)
        candidates = tuple(dict.fromkeys(candidates))
        if not candidates:
            raise ValueError("concentrated Blue family has no feasible allocation")
    else:
        raise ValueError(
            "allocation_family must be strategic, balanced or concentrated"
        )
    return BlueGroupingType(
        name=name,
        prior=prior,
        game=game,
        allocation_family=family,
        candidate_allocations=candidates,
        lower_style=lower_style,
        micro_grouping=micro_grouping,
    )


def solve_bayesian_double_oracle(
    game: BayesianEventBlottoGame,
    *,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    seed: int = 0,
    verbose: bool = False,
) -> BayesianDoubleOracleResult:
    """Solve the typed simultaneous Blotto game with exact DP best responses."""

    if not isinstance(game, BayesianEventBlottoGame):
        raise TypeError("game must be a BayesianEventBlottoGame")
    if int(max_iterations) < 1:
        raise ValueError("max_iterations must be positive")
    if not np.isfinite(tolerance) or float(tolerance) < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    max_iterations = int(max_iterations)
    tolerance = float(tolerance)
    seed = int(seed)
    reference = game.reference_game

    red_seed = _balanced_feasible_allocation(
        reference.defender_budget,
        np.asarray(reference.defender_count_mask),
        allow_reserve=reference.allow_defender_reserve,
    )
    blue_seed = []
    for item in game.blue_types:
        if item.candidate_allocations is not None:
            candidates = item.candidate_allocations
            values = item.game.payoff_matrix([red_seed], candidates)[0]
            blue_seed.append(candidates[int(np.argmin(values))])
        else:
            blue_seed.append(
                item.game.attacker_best_response([red_seed], [1.0]).allocation
            )
    defender_support = [red_seed]
    attacker_support = [TypedBluePolicy(tuple(blue_seed))]
    history = []
    final_solution: MatrixGameSolution | None = None
    converged = False
    termination_reason = "max_iterations"
    used_fallback = False

    for iteration in range(1, max_iterations + 1):
        matrix = game.payoff_matrix(defender_support, attacker_support)
        solution = solve_restricted_matrix_game(matrix)
        used_fallback = used_fallback or solution.used_fallback
        red_br = game.defender_best_response(
            attacker_support, solution.attacker_mixture
        )
        blue_br, blue_value = game.attacker_best_response(
            defender_support, solution.defender_mixture
        )
        defender_gap = max(0.0, red_br.value - solution.value)
        attacker_gap = max(0.0, solution.value - blue_value)
        exploitability = max(0.0, red_br.value - blue_value)
        certified = exploitability <= tolerance
        unilateral = tolerance / 2.0
        add_red = (
            defender_gap > unilateral
            and red_br.allocation not in defender_support
            and not certified
            and iteration < max_iterations
        )
        add_blue = (
            attacker_gap > unilateral
            and blue_br not in attacker_support
            and not certified
            and iteration < max_iterations
        )
        history.append(
            BayesianDoubleOracleIteration(
                iteration=iteration,
                defender_support_size=len(defender_support),
                attacker_policy_support_size=len(attacker_support),
                restricted_value=float(solution.value),
                defender_best_response_value=float(red_br.value),
                attacker_best_response_value=float(blue_value),
                exploitability=float(exploitability),
                defender_best_response=red_br.allocation,
                attacker_best_response=blue_br,
                added_defender_strategy=add_red,
                added_attacker_policy=add_blue,
            )
        )
        final_solution = solution
        if verbose:
            print(
                f"[BayesianBlotto] iter={iteration:03d} "
                f"support=({len(defender_support)},{len(attacker_support)}) "
                f"value={solution.value:+.6f} exploitability={exploitability:.3e}",
                flush=True,
            )
        if certified:
            converged = True
            termination_reason = "converged"
            break
        if not add_red and not add_blue:
            termination_reason = "oracle_stalled"
            break
        if add_red:
            defender_support.append(red_br.allocation)
        if add_blue:
            attacker_support.append(blue_br)

    if final_solution is None:  # pragma: no cover - loop invariant
        raise RuntimeError("Bayesian Double Oracle produced no restricted solution")
    red_br = game.defender_best_response(
        attacker_support, final_solution.attacker_mixture
    )
    blue_br, blue_value = game.attacker_best_response(
        defender_support, final_solution.defender_mixture
    )
    defender_gap = max(0.0, red_br.value - final_solution.value)
    attacker_gap = max(0.0, final_solution.value - blue_value)
    exploitability = max(0.0, red_br.value - blue_value)
    defender_mixture = np.asarray(final_solution.defender_mixture, dtype=np.float64)
    attacker_mixture = np.asarray(final_solution.attacker_mixture, dtype=np.float64)
    defender_mixture.setflags(write=False)
    attacker_mixture.setflags(write=False)
    prior = np.asarray(game.prior, dtype=np.float64).copy()
    prior.setflags(write=False)
    return BayesianDoubleOracleResult(
        defender_strategies=tuple(defender_support),
        attacker_policies=tuple(attacker_support),
        defender_mixture=defender_mixture,
        attacker_policy_mixture=attacker_mixture,
        blue_type_names=tuple(item.name for item in game.blue_types),
        blue_type_prior=prior,
        value=float(final_solution.value),
        exploitability=float(exploitability),
        defender_gap=float(defender_gap),
        attacker_gap=float(attacker_gap),
        converged=bool(converged or exploitability <= tolerance),
        termination_reason=(
            "converged" if exploitability <= tolerance else termination_reason
        ),
        history=tuple(history),
        seed=seed,
        used_matrix_game_fallback=bool(used_fallback),
        lower_styles=tuple(item.lower_style for item in game.blue_types),
        micro_groupings=tuple(item.micro_grouping for item in game.blue_types),
    )


__all__ = [
    "BayesianDoubleOracleIteration",
    "BayesianDoubleOracleResult",
    "BayesianEventBlottoGame",
    "BayesianProfileSample",
    "BlueGroupingType",
    "TypedBluePolicy",
    "make_blue_grouping_type",
    "solve_bayesian_double_oracle",
]
