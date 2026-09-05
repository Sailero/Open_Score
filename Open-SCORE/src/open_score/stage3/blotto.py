"""Scalable event-level Colonel Blotto solvers.

The upper-level action of each player is a count allocation over the active
battlefields.  The lower-level policies are assumed to be fixed while an
event is resolved, so their outcome model can be represented by a local
defender-payoff tensor ``U[m, r, b]``.  Stage 3 uses a separable surrogate

    U(x, y) = sum_m U[m, x_m, y_m]

and solves the resulting simultaneous zero-sum game with a double oracle.
``U`` may be a centered local outcome or a normalized log-survival utility;
the latter makes this sum equivalent to maximizing the product of local
survival probabilities under the registered conditional-independence model.
Crucially, a best response is a resource-constrained dynamic program; the
implementation never enumerates all count compositions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np


Allocation = Tuple[int, ...]


def _as_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _normalise_mixture(mixture: Sequence[float], size: int, name: str) -> np.ndarray:
    probabilities = np.asarray(mixture, dtype=np.float64)
    if probabilities.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < -1e-12):
        raise ValueError(f"{name} must contain finite non-negative probabilities")
    probabilities = np.clip(probabilities, 0.0, None)
    total = float(probabilities.sum())
    if total <= 0.0:
        raise ValueError(f"{name} must have positive mass")
    return probabilities / total


def payoff_from_breach_risk(
    breach_probability: np.ndarray,
    *,
    target_weights: Optional[Sequence[float]] = None,
    early_breach_pressure: Optional[np.ndarray] = None,
    breach_loss: float = 1.0,
    survival_reward: float = 0.0,
    time_weight: float = 0.0,
) -> np.ndarray:
    """Convert Stage-2 local risk outputs to a zero-sum defender payoff.

    Args:
        breach_probability: Array ``[M, R_max + 1, B_max + 1]``.  Entry
            ``(m, r, b)`` is the predicted probability that Blue breaches
            battlefield ``m`` when the two players allocate ``r`` and ``b``
            resources respectively.
        target_weights: Optional non-negative importance of each battlefield.
        early_breach_pressure: Optional tensor with the same shape.  It can be
            a probability of an early breach or another non-negative urgency
            statistic emitted by Stage 2.
        breach_loss: Defender loss for a breach.
        survival_reward: Defender reward for preventing a breach.
        time_weight: Additional penalty applied to early-breach pressure.

    The constant survival term is retained because event sets and target
    weights can change over time, even though constants do not affect a
    single fixed matrix game's equilibrium.
    """

    breach = np.asarray(breach_probability, dtype=np.float64)
    if breach.ndim != 3 or min(breach.shape) <= 0:
        raise ValueError("breach_probability must have shape [M, R, B]")
    if not np.all(np.isfinite(breach)) or np.any((breach < 0.0) | (breach > 1.0)):
        raise ValueError("breach_probability must be finite and lie in [0, 1]")
    if not np.isfinite(breach_loss) or breach_loss < 0.0:
        raise ValueError("breach_loss must be finite and non-negative")
    if not np.isfinite(survival_reward) or survival_reward < 0.0:
        raise ValueError("survival_reward must be finite and non-negative")
    if not np.isfinite(time_weight) or time_weight < 0.0:
        raise ValueError("time_weight must be finite and non-negative")

    if target_weights is None:
        weights = np.ones(breach.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(target_weights, dtype=np.float64)
        if weights.shape != (breach.shape[0],):
            raise ValueError("target_weights must have one entry per battlefield")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("target_weights must be finite and non-negative")

    payoff = weights[:, None, None] * (
        survival_reward * (1.0 - breach) - breach_loss * breach
    )
    if early_breach_pressure is not None:
        pressure = np.asarray(early_breach_pressure, dtype=np.float64)
        if pressure.shape != breach.shape:
            raise ValueError("early_breach_pressure must match breach_probability")
        if not np.all(np.isfinite(pressure)) or np.any(pressure < 0.0):
            raise ValueError("early_breach_pressure must be finite and non-negative")
        payoff -= time_weight * weights[:, None, None] * pressure
    return np.asarray(payoff, dtype=np.float64)


@dataclass(frozen=True)
class BestResponse:
    """A pure resource allocation that best responds to a mixed opponent."""

    allocation: Allocation
    value: float
    assigned_resources: int
    reserve_resources: int


@dataclass(frozen=True)
class EventBlottoGame:
    """One event state's additive zero-sum Colonel Blotto surrogate.

    ``local_payoff[m, r, b]`` is Red/defender utility.  Blue/attacker utility
    is its negative.  A resource not assigned to an active battlefield is an
    implicit reserve when the corresponding ``allow_*_reserve`` flag is true.
    Optional boolean ``*_count_mask`` arrays have the same battlefield/count
    axes as the payoff tensor and remove unsupported local roster counts from
    validation and both best-response oracles.  With no mask, every count up
    to the corresponding battlefield cap remains legal for compatibility.
    """

    local_payoff: np.ndarray
    defender_budget: int
    attacker_budget: int
    defender_caps: Optional[Sequence[int]] = None
    attacker_caps: Optional[Sequence[int]] = None
    allow_defender_reserve: bool = True
    allow_attacker_reserve: bool = True
    event_id: str = "event"
    defender_count_mask: Optional[np.ndarray] = None
    attacker_count_mask: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        payoff = np.asarray(self.local_payoff, dtype=np.float64)
        if payoff.ndim != 3 or min(payoff.shape) <= 0:
            raise ValueError("local_payoff must have shape [M, R, B]")
        if not np.all(np.isfinite(payoff)):
            raise ValueError("local_payoff must be finite")
        defender_budget = _as_nonnegative_int(self.defender_budget, "defender_budget")
        attacker_budget = _as_nonnegative_int(self.attacker_budget, "attacker_budget")
        n_battlefields = payoff.shape[0]

        defender_caps = self._validate_caps(
            self.defender_caps, n_battlefields, payoff.shape[1] - 1, "defender_caps"
        )
        attacker_caps = self._validate_caps(
            self.attacker_caps, n_battlefields, payoff.shape[2] - 1, "attacker_caps"
        )
        defender_count_mask = self._validate_count_mask(
            self.defender_count_mask,
            n_battlefields,
            payoff.shape[1],
            defender_caps,
            "defender_count_mask",
        )
        attacker_count_mask = self._validate_count_mask(
            self.attacker_count_mask,
            n_battlefields,
            payoff.shape[2],
            attacker_caps,
            "attacker_count_mask",
        )
        if not self.allow_defender_reserve and defender_budget > int(defender_caps.sum()):
            raise ValueError("defender budget exceeds capacity while reserve is disabled")
        if not self.allow_attacker_reserve and attacker_budget > int(attacker_caps.sum()):
            raise ValueError("attacker budget exceeds capacity while reserve is disabled")
        if not self._has_feasible_allocation(
            defender_count_mask, defender_budget, self.allow_defender_reserve
        ):
            raise ValueError("defender count mask admits no budget-feasible allocation")
        if not self._has_feasible_allocation(
            attacker_count_mask, attacker_budget, self.allow_attacker_reserve
        ):
            raise ValueError("attacker count mask admits no budget-feasible allocation")

        # Defensive copies make a frozen game stable while it is being solved.
        payoff = payoff.copy()
        payoff.setflags(write=False)
        defender_caps.setflags(write=False)
        attacker_caps.setflags(write=False)
        defender_count_mask.setflags(write=False)
        attacker_count_mask.setflags(write=False)
        object.__setattr__(self, "local_payoff", payoff)
        object.__setattr__(self, "defender_budget", defender_budget)
        object.__setattr__(self, "attacker_budget", attacker_budget)
        object.__setattr__(self, "defender_caps", defender_caps)
        object.__setattr__(self, "attacker_caps", attacker_caps)
        object.__setattr__(self, "defender_count_mask", defender_count_mask)
        object.__setattr__(self, "attacker_count_mask", attacker_count_mask)
        object.__setattr__(self, "event_id", str(self.event_id))

    @staticmethod
    def _validate_caps(
        caps: Optional[Sequence[int]], n_battlefields: int, tensor_cap: int, name: str
    ) -> np.ndarray:
        if caps is None:
            return np.full(n_battlefields, tensor_cap, dtype=np.int64)
        values = np.asarray(caps)
        if values.shape != (n_battlefields,):
            raise ValueError(f"{name} must have one entry per battlefield")
        if not np.issubdtype(values.dtype, np.integer):
            raise TypeError(f"{name} must contain integers")
        values = values.astype(np.int64, copy=True)
        if np.any(values < 0) or np.any(values > tensor_cap):
            raise ValueError(f"{name} entries must lie in [0, {tensor_cap}]")
        return values

    @staticmethod
    def _validate_count_mask(
        mask: Optional[np.ndarray],
        n_battlefields: int,
        n_counts: int,
        caps: np.ndarray,
        name: str,
    ) -> np.ndarray:
        count_axis = np.arange(n_counts, dtype=np.int64)[None, :]
        within_caps = count_axis <= caps[:, None]
        if mask is None:
            return within_caps.copy()
        values = np.asarray(mask)
        if values.shape != (n_battlefields, n_counts):
            raise ValueError(
                f"{name} must have shape ({n_battlefields}, {n_counts})"
            )
        if values.dtype != np.bool_:
            raise TypeError(f"{name} must be a boolean array")
        values = values.astype(bool, copy=True)
        if np.any(values & ~within_caps):
            raise ValueError(f"{name} enables a count above its battlefield cap")
        if np.any(~values.any(axis=1)):
            raise ValueError(f"{name} must enable at least one count per battlefield")
        return values

    @staticmethod
    def _has_feasible_allocation(
        count_mask: np.ndarray, budget: int, allow_reserve: bool
    ) -> bool:
        reachable = np.zeros(budget + 1, dtype=bool)
        reachable[0] = True
        for battlefield_mask in count_mask:
            current = np.zeros_like(reachable)
            for count in np.flatnonzero(battlefield_mask):
                if count > budget:
                    continue
                current[count:] |= reachable[: budget + 1 - count]
            reachable = current
        return bool(reachable.any() if allow_reserve else reachable[budget])

    @property
    def n_battlefields(self) -> int:
        return int(self.local_payoff.shape[0])

    @property
    def max_defenders_per_battlefield(self) -> int:
        return int(self.local_payoff.shape[1] - 1)

    @property
    def max_attackers_per_battlefield(self) -> int:
        return int(self.local_payoff.shape[2] - 1)

    def validate_defender_allocation(self, allocation: Sequence[int]) -> Allocation:
        return self._validate_allocation(
            allocation,
            self.defender_budget,
            np.asarray(self.defender_caps),
            np.asarray(self.defender_count_mask),
            self.allow_defender_reserve,
            "defender allocation",
        )

    def validate_attacker_allocation(self, allocation: Sequence[int]) -> Allocation:
        return self._validate_allocation(
            allocation,
            self.attacker_budget,
            np.asarray(self.attacker_caps),
            np.asarray(self.attacker_count_mask),
            self.allow_attacker_reserve,
            "attacker allocation",
        )

    def _validate_allocation(
        self,
        allocation: Sequence[int],
        budget: int,
        caps: np.ndarray,
        count_mask: np.ndarray,
        allow_reserve: bool,
        name: str,
    ) -> Allocation:
        values = np.asarray(allocation)
        if values.shape != (self.n_battlefields,):
            raise ValueError(f"{name} must have shape ({self.n_battlefields},)")
        if not np.issubdtype(values.dtype, np.integer):
            raise TypeError(f"{name} must contain integers")
        values = values.astype(np.int64, copy=False)
        if np.any(values < 0) or np.any(values > caps):
            raise ValueError(f"{name} violates a per-battlefield capacity")
        if not np.all(count_mask[np.arange(self.n_battlefields), values]):
            raise ValueError(f"{name} contains an unsupported local count")
        used = int(values.sum())
        if used > budget or (not allow_reserve and used != budget):
            relation = "equal" if not allow_reserve else "not exceed"
            raise ValueError(f"{name} resource sum must {relation} its budget")
        return tuple(int(value) for value in values)

    def payoff(self, defender: Sequence[int], attacker: Sequence[int]) -> float:
        defender_allocation = self.validate_defender_allocation(defender)
        attacker_allocation = self.validate_attacker_allocation(attacker)
        indices = np.arange(self.n_battlefields)
        return float(
            self.local_payoff[
                indices,
                np.asarray(defender_allocation, dtype=np.int64),
                np.asarray(attacker_allocation, dtype=np.int64),
            ].sum()
        )

    def payoff_matrix(
        self,
        defender_strategies: Sequence[Sequence[int]],
        attacker_strategies: Sequence[Sequence[int]],
    ) -> np.ndarray:
        defenders = [self.validate_defender_allocation(item) for item in defender_strategies]
        attackers = [self.validate_attacker_allocation(item) for item in attacker_strategies]
        if not defenders or not attackers:
            raise ValueError("both restricted strategy sets must be non-empty")
        defender_array = np.asarray(defenders, dtype=np.int64)
        attacker_array = np.asarray(attackers, dtype=np.int64)
        matrix = np.zeros((len(defenders), len(attackers)), dtype=np.float64)
        for battlefield in range(self.n_battlefields):
            matrix += self.local_payoff[battlefield][
                defender_array[:, battlefield, None],
                attacker_array[None, :, battlefield],
            ]
        return matrix

    def defender_best_response(
        self,
        attacker_strategies: Sequence[Sequence[int]],
        attacker_mixture: Sequence[float],
    ) -> BestResponse:
        strategies = [self.validate_attacker_allocation(item) for item in attacker_strategies]
        marginals = allocation_marginals(
            strategies,
            attacker_mixture,
            np.asarray(self.attacker_caps),
            n_counts=self.local_payoff.shape[2],
        )
        expected_local = np.einsum("mb,mrb->mr", marginals, self.local_payoff)
        return _resource_dp(
            expected_local,
            self.defender_budget,
            np.asarray(self.defender_caps),
            maximise=True,
            allow_reserve=self.allow_defender_reserve,
            count_mask=np.asarray(self.defender_count_mask),
        )

    def attacker_best_response(
        self,
        defender_strategies: Sequence[Sequence[int]],
        defender_mixture: Sequence[float],
    ) -> BestResponse:
        strategies = [self.validate_defender_allocation(item) for item in defender_strategies]
        marginals = allocation_marginals(
            strategies,
            defender_mixture,
            np.asarray(self.defender_caps),
            n_counts=self.local_payoff.shape[1],
        )
        expected_local = np.einsum("mr,mrb->mb", marginals, self.local_payoff)
        return _resource_dp(
            expected_local,
            self.attacker_budget,
            np.asarray(self.attacker_caps),
            maximise=False,
            allow_reserve=self.allow_attacker_reserve,
            count_mask=np.asarray(self.attacker_count_mask),
        )


def allocation_marginals(
    strategies: Sequence[Sequence[int]],
    mixture: Sequence[float],
    caps: Sequence[int],
    *,
    n_counts: Optional[int] = None,
) -> np.ndarray:
    """Return per-battlefield count marginals of a mixed allocation strategy.

    Correlations between battlefields remain present in the restricted matrix
    game.  Additivity means only these marginals are required by a full-space
    best-response oracle.
    """

    caps_array = np.asarray(caps)
    if caps_array.ndim != 1 or not np.issubdtype(caps_array.dtype, np.integer):
        raise ValueError("caps must be a one-dimensional integer sequence")
    caps_array = caps_array.astype(np.int64, copy=False)
    if np.any(caps_array < 0):
        raise ValueError("caps must be non-negative")
    strategy_array = np.asarray(strategies)
    if strategy_array.ndim != 2 or strategy_array.shape[0] == 0:
        raise ValueError("strategies must have shape [K, M] with K > 0")
    if strategy_array.shape[1] != len(caps_array):
        raise ValueError("strategy battlefield dimension does not match caps")
    if not np.issubdtype(strategy_array.dtype, np.integer):
        raise TypeError("strategies must contain integer counts")
    strategy_array = strategy_array.astype(np.int64, copy=False)
    if np.any(strategy_array < 0) or np.any(strategy_array > caps_array[None, :]):
        raise ValueError("a strategy violates its battlefield capacity")
    probabilities = _normalise_mixture(mixture, strategy_array.shape[0], "mixture")
    minimum_count_dimension = int(caps_array.max(initial=0)) + 1
    if n_counts is None:
        count_dimension = minimum_count_dimension
    else:
        count_dimension = _as_nonnegative_int(n_counts, "n_counts")
        if count_dimension < minimum_count_dimension:
            raise ValueError("n_counts is too small for caps")
    marginals = np.zeros((len(caps_array), count_dimension), dtype=np.float64)
    for strategy, probability in zip(strategy_array, probabilities):
        marginals[np.arange(len(caps_array)), strategy] += probability
    return marginals


def _resource_dp(
    local_values: np.ndarray,
    budget: int,
    caps: np.ndarray,
    *,
    maximise: bool,
    allow_reserve: bool,
    count_mask: Optional[np.ndarray] = None,
) -> BestResponse:
    """Solve a separable integer resource allocation without enumeration."""

    values = np.asarray(local_values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(caps):
        raise ValueError("local_values and caps have incompatible shapes")
    if not np.all(np.isfinite(values)):
        raise ValueError("local_values must be finite")
    budget = _as_nonnegative_int(budget, "budget")
    caps = np.asarray(caps, dtype=np.int64)
    if np.any(caps < 0) or np.any(caps >= values.shape[1]):
        raise ValueError("caps exceed local_values count dimension")
    if not allow_reserve and budget > int(caps.sum()):
        raise ValueError("budget exceeds total capacity while reserve is disabled")
    if count_mask is None:
        allowed = np.arange(values.shape[1])[None, :] <= caps[:, None]
    else:
        allowed = np.asarray(count_mask)
        if allowed.shape != values.shape or allowed.dtype != np.bool_:
            raise ValueError("count_mask must be boolean and match local_values")
        if np.any(
            allowed
            & (np.arange(values.shape[1], dtype=np.int64)[None, :] > caps[:, None])
        ):
            raise ValueError("count_mask enables a count above its cap")

    unreachable = -np.inf if maximise else np.inf
    previous = np.full(budget + 1, unreachable, dtype=np.float64)
    previous[0] = 0.0
    choices = np.full((len(caps), budget + 1), -1, dtype=np.int32)

    for battlefield, cap in enumerate(caps):
        current = np.full(budget + 1, unreachable, dtype=np.float64)
        max_local = min(int(cap), budget)
        # Only the local-count loop is in Python.  Each resource-axis shift is
        # vectorised, which keeps a 200-cell, 600-resource oracle comfortably
        # below the cost of a HAD rollout.
        for local_count in np.flatnonzero(allowed[battlefield, : max_local + 1]):
            candidates = (
                previous[: budget + 1 - local_count]
                + values[battlefield, local_count]
            )
            destination = current[local_count:]
            better = candidates > destination if maximise else candidates < destination
            if np.any(better):
                destination[better] = candidates[better]
                choices[battlefield, local_count:][better] = local_count
            # Strict comparison and ascending local_count give deterministic
            # ties, preferring fewer new resources at the latest battlefield.
        previous = current

    if allow_reserve:
        finite_indices = np.flatnonzero(np.isfinite(previous))
        if finite_indices.size == 0:
            raise RuntimeError("resource allocation dynamic program is infeasible")
        finite_values = previous[finite_indices]
        best_offset = int(np.argmax(finite_values) if maximise else np.argmin(finite_values))
        used = int(finite_indices[best_offset])
    else:
        used = budget
        if not np.isfinite(previous[used]):
            raise RuntimeError("resource allocation dynamic program is infeasible")

    allocation = np.zeros(len(caps), dtype=np.int64)
    remaining = used
    for battlefield in range(len(caps) - 1, -1, -1):
        local_count = int(choices[battlefield, remaining])
        if local_count < 0:
            raise RuntimeError("failed to reconstruct resource allocation")
        allocation[battlefield] = local_count
        remaining -= local_count
    if remaining != 0:
        raise RuntimeError("resource allocation reconstruction left unused state")
    return BestResponse(
        allocation=tuple(int(value) for value in allocation),
        value=float(previous[used]),
        assigned_resources=used,
        reserve_resources=budget - used,
    )


@dataclass(frozen=True)
class MatrixGameSolution:
    """Nash solution of a finite defender-payoff zero-sum matrix game."""

    defender_mixture: np.ndarray
    attacker_mixture: np.ndarray
    value: float
    defender_security_value: float
    attacker_security_value: float
    duality_gap: float
    used_fallback: bool

    def sample_indices(
        self,
        *,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[int, int]:
        if seed is not None and rng is not None:
            raise ValueError("pass either seed or rng, not both")
        generator = rng if rng is not None else np.random.default_rng(seed)
        defender = int(generator.choice(len(self.defender_mixture), p=self.defender_mixture))
        attacker = int(generator.choice(len(self.attacker_mixture), p=self.attacker_mixture))
        return defender, attacker


def solve_restricted_matrix_game(defender_payoff: np.ndarray) -> MatrixGameSolution:
    """Solve both primal and dual LPs of a restricted zero-sum game."""

    matrix = np.asarray(defender_payoff, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) <= 0 or not np.all(np.isfinite(matrix)):
        raise ValueError("defender_payoff must be a finite non-empty 2-D matrix")
    n_defender, n_attacker = matrix.shape
    try:
        from scipy.optimize import linprog

        defender_objective = np.r_[np.zeros(n_defender), -1.0]
        defender_result = linprog(
            defender_objective,
            A_ub=np.c_[-matrix.T, np.ones(n_attacker)],
            b_ub=np.zeros(n_attacker),
            A_eq=np.r_[np.ones(n_defender), 0.0][None, :],
            b_eq=np.ones(1),
            bounds=[(0.0, 1.0)] * n_defender + [(None, None)],
            method="highs",
        )

        attacker_objective = np.r_[np.zeros(n_attacker), 1.0]
        attacker_result = linprog(
            attacker_objective,
            A_ub=np.c_[matrix, -np.ones(n_defender)],
            b_ub=np.zeros(n_defender),
            A_eq=np.r_[np.ones(n_attacker), 0.0][None, :],
            b_eq=np.ones(1),
            bounds=[(0.0, 1.0)] * n_attacker + [(None, None)],
            method="highs",
        )
        if defender_result.success and attacker_result.success:
            defender_mixture = np.clip(defender_result.x[:n_defender], 0.0, None)
            attacker_mixture = np.clip(attacker_result.x[:n_attacker], 0.0, None)
            defender_mixture /= defender_mixture.sum()
            attacker_mixture /= attacker_mixture.sum()
            defender_security = float(np.min(defender_mixture @ matrix))
            attacker_security = float(np.max(matrix @ attacker_mixture))
            value = float(defender_mixture @ matrix @ attacker_mixture)
            return MatrixGameSolution(
                defender_mixture=defender_mixture,
                attacker_mixture=attacker_mixture,
                value=value,
                defender_security_value=defender_security,
                attacker_security_value=attacker_security,
                duality_gap=max(0.0, attacker_security - defender_security),
                used_fallback=False,
            )
    except (ImportError, RuntimeError, ValueError):
        pass

    # A deterministic conservative fallback keeps online experiments alive if
    # SciPy/HiGHS is unavailable.  It is explicitly marked as non-equilibrium.
    defender_index = int(np.argmax(matrix.min(axis=1)))
    attacker_index = int(np.argmin(matrix.max(axis=0)))
    defender_mixture = np.zeros(n_defender, dtype=np.float64)
    attacker_mixture = np.zeros(n_attacker, dtype=np.float64)
    defender_mixture[defender_index] = 1.0
    attacker_mixture[attacker_index] = 1.0
    defender_security = float(matrix[defender_index].min())
    attacker_security = float(matrix[:, attacker_index].max())
    return MatrixGameSolution(
        defender_mixture=defender_mixture,
        attacker_mixture=attacker_mixture,
        value=float(matrix[defender_index, attacker_index]),
        defender_security_value=defender_security,
        attacker_security_value=attacker_security,
        duality_gap=max(0.0, attacker_security - defender_security),
        used_fallback=True,
    )


@dataclass(frozen=True)
class DoubleOracleIteration:
    iteration: int
    defender_support_size: int
    attacker_support_size: int
    restricted_value: float
    defender_best_response_value: float
    attacker_best_response_value: float
    defender_gap: float
    attacker_gap: float
    exploitability: float
    defender_best_response: Allocation
    attacker_best_response: Allocation
    added_defender_strategy: bool
    added_attacker_strategy: bool


@dataclass(frozen=True)
class DoubleOracleResult:
    """Restricted equilibrium and full-space exploitability certificate."""

    defender_strategies: Tuple[Allocation, ...]
    attacker_strategies: Tuple[Allocation, ...]
    defender_mixture: np.ndarray
    attacker_mixture: np.ndarray
    value: float
    exploitability: float
    defender_gap: float
    attacker_gap: float
    converged: bool
    termination_reason: str
    history: Tuple[DoubleOracleIteration, ...]
    seed: int
    used_matrix_game_fallback: bool

    def sample_profile(
        self,
        *,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[Allocation, Allocation]:
        """Sample simultaneous allocations reproducibly from the equilibrium."""

        if seed is not None and rng is not None:
            raise ValueError("pass either seed or rng, not both")
        generator = rng if rng is not None else np.random.default_rng(
            self.seed if seed is None else seed
        )
        defender_index = int(
            generator.choice(len(self.defender_strategies), p=self.defender_mixture)
        )
        attacker_index = int(
            generator.choice(len(self.attacker_strategies), p=self.attacker_mixture)
        )
        return self.defender_strategies[defender_index], self.attacker_strategies[
            attacker_index
        ]


ProgressCallback = Callable[[DoubleOracleIteration], None]


def _balanced_allocation(budget: int, caps: np.ndarray, offset: int = 0) -> Allocation:
    allocation = np.zeros(len(caps), dtype=np.int64)
    remaining = min(int(budget), int(caps.sum()))
    if len(caps) == 0:
        return tuple()
    while remaining > 0:
        changed = False
        for step in range(len(caps)):
            battlefield = (offset + step) % len(caps)
            if allocation[battlefield] < caps[battlefield]:
                allocation[battlefield] += 1
                remaining -= 1
                changed = True
                if remaining == 0:
                    break
        if not changed:
            break
    return tuple(int(value) for value in allocation)


def _default_initial_allocation(
    budget: int,
    caps: np.ndarray,
    count_mask: np.ndarray,
    *,
    allow_reserve: bool,
) -> Allocation:
    """Return a valid deterministic seed allocation without enumerating."""

    balanced = _balanced_allocation(budget, caps)
    balanced_array = np.asarray(balanced, dtype=np.int64)
    balanced_sum_valid = int(balanced_array.sum()) <= budget and (
        allow_reserve or int(balanced_array.sum()) == budget
    )
    if balanced_sum_valid and np.all(
        count_mask[np.arange(len(caps)), balanced_array]
    ):
        return balanced

    # Prefer using resources when the ordinary balanced allocation hits a
    # masked count.  This is only a seed for the double oracle; equilibrium
    # values still come from the actual local payoff tensor.
    resource_value = np.broadcast_to(
        np.arange(count_mask.shape[1], dtype=np.float64)[None, :],
        count_mask.shape,
    )
    return _resource_dp(
        resource_value,
        budget,
        caps,
        maximise=True,
        allow_reserve=allow_reserve,
        count_mask=count_mask,
    ).allocation


def _unique_allocations(items: Iterable[Allocation]) -> List[Allocation]:
    result: List[Allocation] = []
    seen = set()
    for item in items:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def solve_double_oracle(
    game: EventBlottoGame,
    *,
    initial_defender_strategies: Optional[Sequence[Sequence[int]]] = None,
    initial_attacker_strategies: Optional[Sequence[Sequence[int]]] = None,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    seed: int = 0,
    verbose: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
) -> DoubleOracleResult:
    """Solve an event-level Blotto game using DP best-response oracles.

    The restricted matrix is the only object whose size grows with double-
    oracle iterations.  The pure allocation spaces are never materialised.
    ``exploitability`` is the full-game saddle gap

        max_x E[U(x, sigma_B)] - min_y E[U(sigma_R, y)].
    """

    if not isinstance(game, EventBlottoGame):
        raise TypeError("game must be an EventBlottoGame")
    max_iterations = _as_nonnegative_int(max_iterations, "max_iterations")
    if max_iterations == 0:
        raise ValueError("max_iterations must be positive")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    seed = int(seed)

    if initial_defender_strategies is None:
        defender_support = [
            _default_initial_allocation(
                game.defender_budget,
                np.asarray(game.defender_caps),
                np.asarray(game.defender_count_mask),
                allow_reserve=game.allow_defender_reserve,
            )
        ]
        if game.allow_defender_reserve and np.all(
            np.asarray(game.defender_count_mask)[:, 0]
        ):
            defender_support.append((0,) * game.n_battlefields)
    else:
        defender_support = [
            game.validate_defender_allocation(item) for item in initial_defender_strategies
        ]
    if initial_attacker_strategies is None:
        attacker_support = [
            _default_initial_allocation(
                game.attacker_budget,
                np.asarray(game.attacker_caps),
                np.asarray(game.attacker_count_mask),
                allow_reserve=game.allow_attacker_reserve,
            )
        ]
        if game.allow_attacker_reserve and np.all(
            np.asarray(game.attacker_count_mask)[:, 0]
        ):
            attacker_support.append((0,) * game.n_battlefields)
    else:
        attacker_support = [
            game.validate_attacker_allocation(item) for item in initial_attacker_strategies
        ]
    defender_support = _unique_allocations(defender_support)
    attacker_support = _unique_allocations(attacker_support)
    if not defender_support or not attacker_support:
        raise ValueError("initial strategy sets must be non-empty")

    if verbose:
        print(
            f"[Stage3][DoubleOracle] event={game.event_id} start "
            f"M={game.n_battlefields} R={game.defender_budget} "
            f"B={game.attacker_budget} max_iter={max_iterations}",
            flush=True,
        )

    history: List[DoubleOracleIteration] = []
    final_solution: Optional[MatrixGameSolution] = None
    used_matrix_game_fallback = False
    termination_reason = "max_iterations"
    converged = False

    for iteration in range(1, max_iterations + 1):
        matrix = game.payoff_matrix(defender_support, attacker_support)
        solution = solve_restricted_matrix_game(matrix)
        used_matrix_game_fallback = (
            used_matrix_game_fallback or solution.used_fallback
        )
        defender_br = game.defender_best_response(
            attacker_support, solution.attacker_mixture
        )
        attacker_br = game.attacker_best_response(
            defender_support, solution.defender_mixture
        )
        defender_gap = max(0.0, defender_br.value - solution.value)
        attacker_gap = max(0.0, solution.value - attacker_br.value)
        exploitability = max(0.0, defender_br.value - attacker_br.value)
        # Exploitability is the *sum* of the two unilateral gaps.  Testing
        # each oracle against the full tolerance can otherwise stall when,
        # for example, both gaps are 0.6 * tolerance.  Half the total budget
        # guarantees that a non-certified profile adds at least one oracle.
        unilateral_tolerance = tolerance / 2.0
        add_defender = (
            defender_gap > unilateral_tolerance
            and defender_br.allocation not in defender_support
        )
        add_attacker = (
            attacker_gap > unilateral_tolerance
            and attacker_br.allocation not in attacker_support
        )
        certified = exploitability <= tolerance
        will_add_defender = add_defender and not certified and iteration < max_iterations
        will_add_attacker = add_attacker and not certified and iteration < max_iterations
        record = DoubleOracleIteration(
            iteration=iteration,
            defender_support_size=len(defender_support),
            attacker_support_size=len(attacker_support),
            restricted_value=solution.value,
            defender_best_response_value=defender_br.value,
            attacker_best_response_value=attacker_br.value,
            defender_gap=defender_gap,
            attacker_gap=attacker_gap,
            exploitability=exploitability,
            defender_best_response=defender_br.allocation,
            attacker_best_response=attacker_br.allocation,
            added_defender_strategy=will_add_defender,
            added_attacker_strategy=will_add_attacker,
        )
        history.append(record)
        final_solution = solution
        if verbose:
            print(
                f"[Stage3][DoubleOracle] iter={iteration:03d} "
                f"support=({len(defender_support)},{len(attacker_support)}) "
                f"value={solution.value:+.6f} exploitability={exploitability:.3e} "
                f"gaps=({defender_gap:.3e},{attacker_gap:.3e})",
                flush=True,
            )
        if progress_callback is not None:
            progress_callback(record)

        if certified:
            converged = True
            termination_reason = "converged"
            break
        if not add_defender and not add_attacker:
            termination_reason = "oracle_stalled"
            break
        if iteration == max_iterations:
            break
        if will_add_defender:
            defender_support.append(defender_br.allocation)
        if will_add_attacker:
            attacker_support.append(attacker_br.allocation)

    if final_solution is None or not history:
        raise RuntimeError("double oracle produced no solution")
    last = history[-1]
    result = DoubleOracleResult(
        defender_strategies=tuple(defender_support),
        attacker_strategies=tuple(attacker_support),
        defender_mixture=final_solution.defender_mixture.copy(),
        attacker_mixture=final_solution.attacker_mixture.copy(),
        value=final_solution.value,
        exploitability=last.exploitability,
        defender_gap=last.defender_gap,
        attacker_gap=last.attacker_gap,
        converged=converged,
        termination_reason=termination_reason,
        history=tuple(history),
        seed=seed,
        used_matrix_game_fallback=used_matrix_game_fallback,
    )
    if verbose:
        print(
            f"[Stage3][DoubleOracle] event={game.event_id} done "
            f"reason={termination_reason} iterations={len(history)} "
            f"support=({len(defender_support)},{len(attacker_support)}) "
            f"exploitability={result.exploitability:.3e}",
            flush=True,
        )
    return result


__all__ = [
    "Allocation",
    "BestResponse",
    "DoubleOracleIteration",
    "DoubleOracleResult",
    "EventBlottoGame",
    "MatrixGameSolution",
    "allocation_marginals",
    "payoff_from_breach_risk",
    "solve_double_oracle",
    "solve_restricted_matrix_game",
]
