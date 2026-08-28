"""Count-level candidates and a standard zero-sum minimax commander."""

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class CommandDecision:
    defender_mixture: np.ndarray
    # Deterministic representative for logging/fallback, not a mixed-strategy draw.
    selected_index: int
    worst_case_breach: float
    used_fallback: bool

    def sample_index(self, rng: np.random.Generator = None) -> int:
        """Draw the allocation to execute from the minimax mixture."""

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
    min_per_task: int = 0,
    max_candidates: int = 256,
) -> List[Tuple[int, ...]]:
    """Enumerate integer task counts; individual IDs are assigned afterwards."""

    if n_agents < 0 or n_tasks <= 0 or min_per_task < 0:
        raise ValueError("invalid allocation dimensions")
    remaining = n_agents - n_tasks * min_per_task
    if remaining < 0:
        return []
    candidates = [
        tuple(value + min_per_task for value in composition)
        for composition in _compositions(remaining, n_tasks)
    ]
    if len(candidates) <= max_candidates:
        return candidates
    # Deterministic spread across the ordered candidate set; no hidden randomness.
    indices = np.linspace(0, len(candidates) - 1, max_candidates).round().astype(int)
    return [candidates[index] for index in np.unique(indices)]


def solve_defender_minimax(
    predicted_breach: np.ndarray,
    empirical_correction: np.ndarray = None,
    switch_cost: Sequence[float] = None,
    switch_weight: float = 0.0,
) -> CommandDecision:
    """Minimise worst-case breach across a registered attacker candidate set."""

    matrix = np.asarray(predicted_breach, dtype=np.float64)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError("predicted_breach must be a non-empty 2-D matrix")
    if empirical_correction is not None:
        matrix = np.clip(matrix + np.asarray(empirical_correction, dtype=np.float64), 0.0, 1.0)
    if switch_cost is not None:
        costs = np.asarray(switch_cost, dtype=np.float64)
        if costs.shape != (matrix.shape[0],):
            raise ValueError("switch_cost must have one value per defender candidate")
        matrix = matrix + switch_weight * costs[:, None]

    rows, cols = matrix.shape
    try:
        from scipy.optimize import linprog

        # Variables are defender mixture x_1...x_rows and upper value v.
        objective = np.r_[np.zeros(rows), 1.0]
        a_ub = np.c_[matrix.T, -np.ones(cols)]
        b_ub = np.zeros(cols)
        a_eq = np.r_[np.ones(rows), 0.0][None, :]
        result = linprog(
            objective,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=a_eq,
            b_eq=np.array([1.0]),
            bounds=[(0.0, 1.0)] * rows + [(0.0, None)],
            method="highs",
        )
        if result.success:
            mixture = np.clip(result.x[:rows], 0.0, 1.0)
            mixture /= mixture.sum()
            selected = int(np.argmax(mixture))
            value = float(np.max(mixture @ matrix))
            return CommandDecision(mixture, selected, value, False)
    except (ImportError, RuntimeError, ValueError):
        pass

    row_risk = matrix.max(axis=1)
    selected = int(np.argmin(row_risk))
    mixture = np.zeros(rows, dtype=np.float64)
    mixture[selected] = 1.0
    return CommandDecision(mixture, selected, float(row_risk[selected]), True)


def needs_replan(
    step: int,
    last_command_step: int,
    previous_roster: Tuple[int, int],
    current_roster: Tuple[int, int],
    command_interval: int,
) -> bool:
    """Replan on a macro boundary or immediately after a join/leave/death event."""

    return current_roster != previous_roster or step - last_command_step >= command_interval
