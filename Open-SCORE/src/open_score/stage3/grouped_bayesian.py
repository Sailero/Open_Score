"""Bayesian Double Oracle for target-pattern Blotto games.

Blue's latent type denotes a lower rule/model (for the MVP: ``rush`` or
``split_rush``).  Every type retains the strategic grouped upper action; the
balanced and concentrated patterns are seeds/baselines, not opponent types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .blotto import Allocation, BestResponse, MatrixGameSolution, solve_restricted_matrix_game
from .grouped_blotto import (
    GroupHistogram,
    GroupedAllocation,
    GroupedBlottoGame,
    balanced_grouped_allocation,
    concentrated_grouped_allocation,
)


def _probabilities(values: Sequence[float], expected: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (expected,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector of length {expected}")
    if np.any(result < 0.0) or float(result.sum()) <= 0.0:
        raise ValueError(f"{name} must be non-negative with positive mass")
    return result / result.sum()


@dataclass(frozen=True)
class GroupedBlueType:
    name: str
    prior: float
    game: GroupedBlottoGame
    lower_style: str
    candidate_allocations: Optional[Tuple[GroupedAllocation, ...]] = None

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("a grouped Blue type needs a name")
        prior = float(self.prior)
        if not np.isfinite(prior) or prior <= 0.0:
            raise ValueError("a grouped Blue type prior must be positive")
        if not isinstance(self.game, GroupedBlottoGame):
            raise TypeError("game must be a GroupedBlottoGame")
        candidates = self.candidate_allocations
        if candidates is not None:
            candidates = tuple(
                dict.fromkeys(
                    self.game.validate_attacker_allocation(value)
                    for value in candidates
                )
            )
            if not candidates:
                raise ValueError("a restricted grouped Blue type needs candidates")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "prior", prior)
        object.__setattr__(self, "lower_style", str(self.lower_style))
        object.__setattr__(self, "candidate_allocations", candidates)


@dataclass(frozen=True)
class GroupedTypedBluePolicy:
    allocations: Tuple[GroupedAllocation, ...]


@dataclass(frozen=True)
class GroupedBayesianProfileSample:
    defender_allocation: GroupedAllocation
    blue_type_index: int
    blue_type_name: str
    attacker_allocation: GroupedAllocation
    lower_style: str


@dataclass(frozen=True)
class GroupedBayesianIteration:
    iteration: int
    defender_support_size: int
    attacker_support_size: int
    restricted_value: float
    defender_best_response_value: float
    attacker_best_response_value: float
    exploitability: float
    added_defender: bool
    added_attacker: bool


@dataclass(frozen=True)
class GroupedBayesianDoubleOracleResult:
    defender_strategies: Tuple[GroupedAllocation, ...]
    attacker_policies: Tuple[GroupedTypedBluePolicy, ...]
    defender_mixture: np.ndarray
    attacker_policy_mixture: np.ndarray
    blue_type_names: Tuple[str, ...]
    blue_type_prior: np.ndarray
    lower_styles: Tuple[str, ...]
    value: float
    exploitability: float
    defender_gap: float
    attacker_gap: float
    converged: bool
    termination_reason: str
    history: Tuple[GroupedBayesianIteration, ...]
    seed: int
    used_matrix_game_fallback: bool

    def sample_profile(
        self,
        *,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        blue_type_name: Optional[str] = None,
    ) -> GroupedBayesianProfileSample:
        if seed is not None and rng is not None:
            raise ValueError("pass either seed or rng, not both")
        generator = rng if rng is not None else np.random.default_rng(
            self.seed if seed is None else int(seed)
        )
        red_index = int(
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
                raise ValueError(f"unknown Blue type {blue_type_name!r}") from error
        policy_index = int(
            generator.choice(
                len(self.attacker_policies), p=self.attacker_policy_mixture
            )
        )
        return GroupedBayesianProfileSample(
            defender_allocation=self.defender_strategies[red_index],
            blue_type_index=type_index,
            blue_type_name=self.blue_type_names[type_index],
            attacker_allocation=self.attacker_policies[policy_index].allocations[
                type_index
            ],
            lower_style=self.lower_styles[type_index],
        )


def _red_domain_signature(game: GroupedBlottoGame) -> tuple[object, ...]:
    return (
        game.n_targets,
        game.defender_budget,
        game.red_group_size_cap,
        bool(game.allow_defender_reserve),
        game.pairing_rule,
    )


@dataclass(frozen=True)
class GroupedBayesianBlottoGame:
    blue_types: Tuple[GroupedBlueType, ...]
    event_id: str = "grouped-bayesian-event"

    def __post_init__(self) -> None:
        values = tuple(self.blue_types)
        if not values:
            raise ValueError("a grouped Bayesian game needs at least one Blue type")
        if len({value.name for value in values}) != len(values):
            raise ValueError("grouped Blue type names must be unique")
        signature = _red_domain_signature(values[0].game)
        if any(_red_domain_signature(value.game) != signature for value in values[1:]):
            raise ValueError("all Blue types must share the same Red action domain")
        prior = np.asarray([value.prior for value in values], dtype=np.float64)
        prior /= prior.sum()
        prior.setflags(write=False)
        object.__setattr__(self, "blue_types", values)
        object.__setattr__(self, "_prior", prior)

    @property
    def reference_game(self) -> GroupedBlottoGame:
        return self.blue_types[0].game

    @property
    def prior(self) -> np.ndarray:
        return self._prior

    def validate_defender_allocation(self, value: Sequence[int]) -> GroupedAllocation:
        return self.reference_game.validate_defender_allocation(value)

    def validate_attacker_policy(
        self, policy: GroupedTypedBluePolicy
    ) -> GroupedTypedBluePolicy:
        if not isinstance(policy, GroupedTypedBluePolicy):
            raise TypeError("attacker policy must be GroupedTypedBluePolicy")
        if len(policy.allocations) != len(self.blue_types):
            raise ValueError("typed policy must contain one group action per type")
        allocations = []
        for blue_type, raw in zip(self.blue_types, policy.allocations):
            action = blue_type.game.validate_attacker_allocation(raw)
            if (
                blue_type.candidate_allocations is not None
                and action not in blue_type.candidate_allocations
            ):
                raise ValueError("typed action is outside its registered family")
            allocations.append(action)
        return GroupedTypedBluePolicy(tuple(allocations))

    def payoff(
        self, defender: Sequence[int], attacker: GroupedTypedBluePolicy
    ) -> float:
        red = self.validate_defender_allocation(defender)
        blue = self.validate_attacker_policy(attacker)
        return float(
            sum(
                probability * blue_type.game.payoff(red, action)
                for probability, blue_type, action in zip(
                    self.prior, self.blue_types, blue.allocations
                )
            )
        )

    def payoff_matrix(
        self,
        defenders: Sequence[Sequence[int]],
        attackers: Sequence[GroupedTypedBluePolicy],
    ) -> np.ndarray:
        red = [self.validate_defender_allocation(value) for value in defenders]
        blue = [self.validate_attacker_policy(value) for value in attackers]
        if not red or not blue:
            raise ValueError("both restricted strategy sets must be non-empty")
        matrix = np.empty((len(red), len(blue)), dtype=np.float64)
        for row, defender in enumerate(red):
            for column, attacker in enumerate(blue):
                matrix[row, column] = self.payoff(defender, attacker)
        return matrix

    def defender_best_response(
        self,
        attacker_policies: Sequence[GroupedTypedBluePolicy],
        mixture: Sequence[float],
    ) -> BestResponse:
        policies = [self.validate_attacker_policy(value) for value in attacker_policies]
        policy_weights = _probabilities(mixture, len(policies), "Blue policy mixture")
        reference = self.reference_game
        budget = reference.defender_budget
        local_values = np.full((reference.n_targets, budget + 1), -np.inf)
        local_patterns: List[List[Optional[GroupHistogram]]] = [
            [None] * (budget + 1) for _ in range(reference.n_targets)
        ]
        for target in range(reference.n_targets):
            for total, patterns in enumerate(reference.defender_patterns):
                for red_pattern in patterns:
                    value = 0.0
                    for policy_weight, policy in zip(policy_weights, policies):
                        for type_weight, type_index, blue_type in zip(
                            self.prior, range(len(self.blue_types)), self.blue_types
                        ):
                            blue_pattern = blue_type.game.target_histogram(
                                policy.allocations[type_index], target, "Blue"
                            )
                            value += (
                                float(policy_weight)
                                * float(type_weight)
                                * blue_type.game.target_payoff(
                                    target, red_pattern, blue_pattern
                                )
                            )
                    if value > local_values[target, total]:
                        local_values[target, total] = value
                        local_patterns[target][total] = red_pattern
        return _resource_pattern_response(
            local_values,
            local_patterns,
            budget,
            reference.n_targets,
            maximise=True,
            allow_reserve=bool(reference.allow_defender_reserve),
        )

    def attacker_best_response(
        self,
        defender_strategies: Sequence[Sequence[int]],
        mixture: Sequence[float],
    ) -> tuple[GroupedTypedBluePolicy, float]:
        defenders = [self.validate_defender_allocation(value) for value in defender_strategies]
        weights = _probabilities(mixture, len(defenders), "Red mixture")
        allocations: List[GroupedAllocation] = []
        value = 0.0
        for probability, blue_type in zip(self.prior, self.blue_types):
            if blue_type.candidate_allocations is None:
                response = blue_type.game.attacker_best_response(defenders, weights)
                action = response.allocation
                local_value = response.value
            else:
                candidates = blue_type.candidate_allocations
                candidate_values = weights @ blue_type.game.payoff_matrix(
                    defenders, candidates
                )
                index = int(np.argmin(candidate_values))
                action = candidates[index]
                local_value = float(candidate_values[index])
            allocations.append(action)
            value += float(probability) * float(local_value)
        return GroupedTypedBluePolicy(tuple(allocations)), value


def _resource_pattern_response(
    local_values: np.ndarray,
    local_patterns: Sequence[Sequence[Optional[GroupHistogram]]],
    budget: int,
    targets: int,
    *,
    maximise: bool,
    allow_reserve: bool,
) -> BestResponse:
    unreachable = -np.inf if maximise else np.inf
    previous = np.full(budget + 1, unreachable)
    previous[0] = 0.0
    choices = np.full((targets, budget + 1), -1, dtype=np.int32)
    for target in range(targets):
        current = np.full(budget + 1, unreachable)
        for local_total in range(budget + 1):
            candidates = previous[: budget + 1 - local_total] + local_values[
                target, local_total
            ]
            destination = current[local_total:]
            better = candidates > destination if maximise else candidates < destination
            if np.any(better):
                destination[better] = candidates[better]
                choices[target, local_total:][better] = local_total
        previous = current
    if allow_reserve:
        feasible = np.flatnonzero(np.isfinite(previous))
        values = previous[feasible]
        used = int(feasible[int(np.argmax(values) if maximise else np.argmin(values))])
    else:
        used = budget
    if not np.isfinite(previous[used]):
        raise RuntimeError("Bayesian grouped best response is infeasible")
    selected: List[GroupHistogram] = [()] * targets
    remaining = used
    for target in range(targets - 1, -1, -1):
        total = int(choices[target, remaining])
        pattern = local_patterns[target][total]
        if total < 0 or pattern is None:
            raise RuntimeError("failed to reconstruct Bayesian grouped response")
        selected[target] = pattern
        remaining -= total
    action: Allocation = tuple(value for pattern in selected for value in pattern)
    return BestResponse(action, float(previous[used]), used, budget - used)


def make_grouped_blue_type(
    game: GroupedBlottoGame,
    *,
    name: str,
    prior: float,
    lower_style: str,
    upper_family: str = "strategic",
    preferred_group_size: int = 4,
) -> GroupedBlueType:
    if upper_family == "strategic":
        candidates = None
    elif upper_family == "balanced":
        candidates = (
            balanced_grouped_allocation(
                game, "Blue", preferred_size=preferred_group_size
            ),
        )
    elif upper_family == "concentrated":
        candidates = tuple(
            concentrated_grouped_allocation(
                game,
                "Blue",
                target=target,
                preferred_size=preferred_group_size,
            )
            for target in range(game.n_targets)
        )
    else:
        raise ValueError("upper_family must be strategic, balanced or concentrated")
    return GroupedBlueType(name, prior, game, lower_style, candidates)


def solve_grouped_bayesian_double_oracle(
    game: GroupedBayesianBlottoGame,
    *,
    max_iterations: int = 60,
    tolerance: float = 1e-6,
    seed: int = 0,
    verbose: bool = False,
) -> GroupedBayesianDoubleOracleResult:
    if not isinstance(game, GroupedBayesianBlottoGame):
        raise TypeError("game must be GroupedBayesianBlottoGame")
    reference = game.reference_game
    red_support = [balanced_grouped_allocation(reference, "Red")]
    blue_seed = []
    for blue_type in game.blue_types:
        if blue_type.candidate_allocations is None:
            action = blue_type.game.attacker_best_response(red_support, [1.0]).allocation
        else:
            values = blue_type.game.payoff_matrix(
                red_support, blue_type.candidate_allocations
            )[0]
            action = blue_type.candidate_allocations[int(np.argmin(values))]
        blue_seed.append(action)
    blue_support = [GroupedTypedBluePolicy(tuple(blue_seed))]
    history: List[GroupedBayesianIteration] = []
    solution: Optional[MatrixGameSolution] = None
    used_fallback = False
    converged = False
    reason = "max_iterations"
    for iteration in range(1, int(max_iterations) + 1):
        solution = solve_restricted_matrix_game(
            game.payoff_matrix(red_support, blue_support)
        )
        used_fallback = used_fallback or solution.used_fallback
        red_br = game.defender_best_response(blue_support, solution.attacker_mixture)
        blue_br, blue_value = game.attacker_best_response(
            red_support, solution.defender_mixture
        )
        red_gap = max(0.0, red_br.value - solution.value)
        blue_gap = max(0.0, solution.value - blue_value)
        exploitability = max(0.0, red_br.value - blue_value)
        certified = exploitability <= float(tolerance)
        add_red = (
            red_gap > float(tolerance) / 2
            and red_br.allocation not in red_support
            and not certified
            and iteration < int(max_iterations)
        )
        add_blue = (
            blue_gap > float(tolerance) / 2
            and blue_br not in blue_support
            and not certified
            and iteration < int(max_iterations)
        )
        history.append(
            GroupedBayesianIteration(
                iteration,
                len(red_support),
                len(blue_support),
                solution.value,
                red_br.value,
                blue_value,
                exploitability,
                add_red,
                add_blue,
            )
        )
        if verbose:
            print(
                f"[Stage3][GroupedBayesianDO] iter={iteration:03d} "
                f"support=({len(red_support)},{len(blue_support)}) "
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
            red_support.append(red_br.allocation)
        if add_blue:
            blue_support.append(blue_br)
    if solution is None or not history:
        raise RuntimeError("grouped Bayesian Double Oracle produced no solution")
    last = history[-1]
    return GroupedBayesianDoubleOracleResult(
        defender_strategies=tuple(red_support),
        attacker_policies=tuple(blue_support),
        defender_mixture=solution.defender_mixture.copy(),
        attacker_policy_mixture=solution.attacker_mixture.copy(),
        blue_type_names=tuple(value.name for value in game.blue_types),
        blue_type_prior=game.prior.copy(),
        lower_styles=tuple(value.lower_style for value in game.blue_types),
        value=solution.value,
        exploitability=last.exploitability,
        defender_gap=max(0.0, last.defender_best_response_value - last.restricted_value),
        attacker_gap=max(0.0, last.restricted_value - last.attacker_best_response_value),
        converged=converged,
        termination_reason=reason,
        history=tuple(history),
        seed=int(seed),
        used_matrix_game_fallback=used_fallback,
    )


__all__ = [
    "GroupedBayesianBlottoGame",
    "GroupedBayesianDoubleOracleResult",
    "GroupedBayesianIteration",
    "GroupedBayesianProfileSample",
    "GroupedBlueType",
    "GroupedTypedBluePolicy",
    "make_grouped_blue_type",
    "solve_grouped_bayesian_double_oracle",
]
