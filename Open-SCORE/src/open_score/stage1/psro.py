"""Minimal PSRO bookkeeping for two cooperative QMIX teams."""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from open_score.stage3.commander import solve_defender_maximin


@dataclass(frozen=True)
class MetaNash:
    defender_mixture: np.ndarray
    attacker_mixture: np.ndarray
    value: float


@dataclass(frozen=True)
class PSROIterationMetrics:
    iteration: int
    population_shape: tuple
    meta_value: float
    defender_response_value: float
    attacker_response_value: float
    estimated_nash_conv: float
    archive_mean_payoff: float = float("nan")
    meta_value_change: float = float("inf")
    oracle_budget_met: bool = False


def solve_zero_sum_meta_game(payoff: np.ndarray) -> MetaNash:
    """Solve the empirical game; rows are defender team policies."""

    matrix = np.asarray(payoff, dtype=np.float64)
    defender = solve_defender_maximin(matrix)
    attacker_as_row = solve_defender_maximin(-matrix.T)
    attacker_mixture = attacker_as_row.defender_mixture
    value = float(defender.defender_mixture @ matrix @ attacker_mixture)
    return MetaNash(defender.defender_mixture, attacker_mixture, value)


def estimated_nash_conv(
    meta_value: float,
    defender_response_value: float,
    attacker_response_value: float,
) -> float:
    """Approximate two-player NashConv from independently evaluated responses.

    `attacker_response_value` is still expressed as defender payoff, so a lower
    value is a stronger attacker response.
    """

    defender_gap = max(0.0, defender_response_value - meta_value)
    attacker_gap = max(0.0, meta_value - attacker_response_value)
    return defender_gap + attacker_gap


def psro_has_stabilised(
    history: Sequence[PSROIterationMetrics],
    tolerance: float = 0.10,
    patience: int = 3,
    max_iterations: Optional[int] = None,
    value_tolerance: float = 0.03,
) -> bool:
    """Require both exploitability and empirical-game value to stabilise.

    Average win rate against an archive or a scripted policy is diagnostic but
    is not a convergence certificate: a new exploiter can remain hidden behind
    a stable mean.  Estimated NashConv is therefore the primary condition.
    """

    if max_iterations is not None and history and history[-1].iteration >= max_iterations:
        return True
    recent = history[-patience:]
    return len(history) >= patience and all(
        item.estimated_nash_conv <= tolerance
        and item.meta_value_change <= value_tolerance
        and item.oracle_budget_met
        for item in recent
    )
