"""Shape-constrained calibration for Stage-2 roster-response surfaces."""

from __future__ import annotations

import numpy as np


def _isotonic_increasing(values: np.ndarray) -> np.ndarray:
    """Unweighted pool-adjacent-violators projection onto increasing values."""

    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 1:
        raise ValueError("isotonic input must be one-dimensional")
    means: list[float] = []
    weights: list[int] = []
    starts: list[int] = []
    for index, value in enumerate(source):
        means.append(float(value))
        weights.append(1)
        starts.append(index)
        while len(means) >= 2 and means[-2] > means[-1]:
            total = weights[-2] + weights[-1]
            merged = (means[-2] * weights[-2] + means[-1] * weights[-1]) / total
            means[-2:] = [merged]
            weights[-2:] = [total]
            starts[-2:] = [starts[-2]]
    result = np.empty_like(source)
    for block, (mean, weight) in enumerate(zip(means, weights)):
        start = starts[block]
        result[start : start + weight] = mean
    return result


def _project_red_curve(values: np.ndarray) -> np.ndarray:
    """Project Red-count logits to nondecreasing diminishing increments."""

    if len(values) < 2:
        return values.copy()
    slopes = np.diff(values)
    slopes = -_isotonic_increasing(-slopes)
    slopes = np.maximum(slopes, 0.0)
    reconstructed = np.concatenate(([0.0], np.cumsum(slopes)))
    reconstructed += float(np.mean(values - reconstructed))
    return reconstructed


def _project_blue_curve(values: np.ndarray) -> np.ndarray:
    """Project Blue-count logits to nonincreasing diminishing attack effects."""

    if len(values) < 2:
        return values.copy()
    slopes = _isotonic_increasing(np.diff(values))
    slopes = np.minimum(slopes, 0.0)
    reconstructed = np.concatenate(([0.0], np.cumsum(slopes)))
    reconstructed += float(np.mean(values - reconstructed))
    return reconstructed


def project_roster_probability_surface(
    probability: np.ndarray,
    *,
    max_iterations: int = 50,
    tolerance: float = 1e-7,
) -> np.ndarray:
    """Project ``[target, red>=1, blue>=1]`` probabilities in log-odds space.

    The frozen Stage-2 estimator uses this operator after neural inference.  It
    is a shape-constrained calibration step over directly queried cells, never
    an extrapolator: every output cell has a corresponding neural prediction.
    """

    values = np.asarray(probability, dtype=np.float64)
    if values.ndim != 3 or min(values.shape) < 1:
        raise ValueError("roster probability surface must be three-dimensional")
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("roster probabilities must be finite and lie in [0,1]")
    if max_iterations < 1 or tolerance <= 0.0:
        raise ValueError("invalid shape-projection convergence settings")
    clipped = np.clip(values, 1e-5, 1.0 - 1e-5)
    logits = np.log(clipped) - np.log1p(-clipped)
    for _ in range(max_iterations):
        previous = logits.copy()
        for target in range(logits.shape[0]):
            for blue in range(logits.shape[2]):
                logits[target, :, blue] = _project_red_curve(
                    logits[target, :, blue]
                )
            for red in range(logits.shape[1]):
                logits[target, red, :] = _project_blue_curve(
                    logits[target, red, :]
                )
        if float(np.max(np.abs(logits - previous))) <= tolerance:
            break
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


__all__ = ["project_roster_probability_surface"]
