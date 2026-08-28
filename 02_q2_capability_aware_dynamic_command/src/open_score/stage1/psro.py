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
) -> bool:
    """Use a patience rule because deep-RL responses and payoffs are noisy."""

    if max_iterations is not None and history and history[-1].iteration >= max_iterations:
        return True
    return len(history) >= patience and all(
        item.estimated_nash_conv <= tolerance for item in history[-patience:]
    )
