"""Receding-horizon zero-sum grouping game for the Stage-3 commander."""

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class CommandDecision:
    defender_mixture: np.ndarray
    representative_index: int
    attacker_worst_response: int
    worst_case_value: float
    used_fallback: bool

    def sample_index(self, rng: Optional[np.random.Generator] = None) -> int:
        """Draw the defender allocation actually executed this command period."""

        generator = rng if rng is not None else np.random.default_rng()
        return int(generator.choice(len(self.defender_mixture), p=self.defender_mixture))


def _compositions(total: int, parts: int) -> Iterable[Tuple[int, ...]]:
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first,) + rest


def enumerate_count_allocations(
    n_agents: int,
    n_tasks: int,
    include_reserve: bool = True,
    max_per_task: int = 4,
    max_candidates: int = 256,
) -> List[Tuple[int, ...]]:
    """Enumerate task counts; the optional last entry is a reserve group."""

    if n_agents < 0 or n_tasks <= 0 or max_per_task <= 0:
        raise ValueError("invalid allocation dimensions")
    parts = n_tasks + int(include_reserve)
    candidates = [
        allocation
        for allocation in _compositions(n_agents, parts)
        if all(value <= max_per_task for value in allocation[:n_tasks])
    ]
    if len(candidates) <= max_candidates:
        return candidates
    indices = np.linspace(0, len(candidates) - 1, max_candidates).round().astype(int)
    return [candidates[index] for index in np.unique(indices)]


def build_defender_payoff(
    breach_upper95: np.ndarray,
    early_breach_pressure: Optional[np.ndarray] = None,
    switch_cost: Optional[Sequence[float]] = None,
    time_weight: float = 0.10,
    switch_weight: float = 0.02,
) -> np.ndarray:
    """Convert per-target S2 outputs into a conservative global payoff matrix.

    `breach_upper95` has shape [defender candidates, attacker candidates,
    targets].  A union bound avoids assuming that target breaches are
    independent.  Higher returned values are better for the defender.
    """

    breach = np.asarray(breach_upper95, dtype=np.float64)
    if breach.ndim != 3 or breach.size == 0:
        raise ValueError("breach_upper95 must have shape [K_D, K_A, M]")
    global_breach_upper = np.clip(breach.sum(axis=-1), 0.0, 1.0)
    payoff = 1.0 - global_breach_upper
    if early_breach_pressure is not None:
        pressure = np.asarray(early_breach_pressure, dtype=np.float64)
        if pressure.shape != breach.shape:
            raise ValueError("early_breach_pressure must match breach_upper95")
        payoff -= time_weight * pressure.mean(axis=-1)
    if switch_cost is not None:
        costs = np.asarray(switch_cost, dtype=np.float64)
        if costs.shape != (breach.shape[0],):
            raise ValueError("switch_cost must have one value per defender candidate")
        payoff -= switch_weight * costs[:, None]
    return payoff


def solve_defender_maximin(defender_payoff: np.ndarray) -> CommandDecision:
    """Maximise the defender's worst response value in a zero-sum matrix game."""

    matrix = np.asarray(defender_payoff, dtype=np.float64)
    if matrix.ndim != 2 or matrix.size == 0 or not np.all(np.isfinite(matrix)):
        raise ValueError("defender_payoff must be a finite, non-empty 2-D matrix")
    rows, cols = matrix.shape
    try:
        from scipy.optimize import linprog

        # Minimise -v subject to x^T U[:,j] >= v for every attacker j.
        objective = np.r_[np.zeros(rows), -1.0]
        a_ub = np.c_[-matrix.T, np.ones(cols)]
        result = linprog(
            objective,
            A_ub=a_ub,
            b_ub=np.zeros(cols),
            A_eq=np.r_[np.ones(rows), 0.0][None, :],
            b_eq=np.array([1.0]),
            bounds=[(0.0, 1.0)] * rows + [(None, None)],
            method="highs",
        )
        if result.success:
            mixture = np.clip(result.x[:rows], 0.0, 1.0)
            mixture /= mixture.sum()
            responses = mixture @ matrix
            worst_response = int(np.argmin(responses))
            return CommandDecision(
                mixture,
                int(np.argmax(mixture)),
                worst_response,
                float(responses[worst_response]),
                False,
            )
    except (ImportError, RuntimeError, ValueError):
        pass

    row_values = matrix.min(axis=1)
    representative = int(np.argmax(row_values))
    worst_response = int(np.argmin(matrix[representative]))
    mixture = np.zeros(rows, dtype=np.float64)
    mixture[representative] = 1.0
    return CommandDecision(
        mixture,
        representative,
        worst_response,
        float(matrix[representative, worst_response]),
        True,
    )


def needs_replan(
    step: int,
    last_command_step: int,
    previous_roster: Tuple[int, int],
    current_roster: Tuple[int, int],
    command_interval: int,
) -> bool:
    """Replan on a macro boundary or immediately after a population event."""

    return current_roster != previous_roster or step - last_command_step >= command_interval
