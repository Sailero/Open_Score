"""Independent probability calibration and one-sided breach-risk bounds."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=-1, keepdims=True)


@dataclass(frozen=True)
class TemperatureScaler:
    """One scalar fitted only on calibration roots for joint-class NLL."""

    temperature: float = 1.0

    @classmethod
    def fit(cls, probabilities: np.ndarray, labels: np.ndarray) -> "TemperatureScaler":
        values = np.asarray(probabilities, dtype=np.float64)
        targets = np.asarray(labels, dtype=np.int64)
        if values.ndim != 2 or targets.shape != (len(values),):
            raise ValueError("probabilities/labels have incompatible shapes")
        if len(values) < 1 or np.any(targets < 0) or np.any(targets >= values.shape[1]):
            raise ValueError("calibration labels are empty or outside the class range")
        values = np.clip(values, 1e-12, 1.0)
        values /= values.sum(axis=1, keepdims=True)
        log_probability = np.log(values)

        def search(low: float, high: float, count: int) -> Tuple[float, float]:
            temperatures = np.exp(np.linspace(np.log(low), np.log(high), count))
            losses = []
            for temperature in temperatures:
                calibrated = _softmax(log_probability / temperature)
                losses.append(-np.log(np.clip(calibrated[np.arange(len(targets)), targets], 1e-12, 1.0)).mean())
            index = int(np.argmin(losses))
            return float(temperatures[index]), float(losses[index])

        coarse, _ = search(0.20, 5.0, 161)
        refined, _ = search(max(0.05, coarse / 1.20), min(20.0, coarse * 1.20), 121)
        return cls(refined)

    def apply(self, probabilities: np.ndarray) -> np.ndarray:
        values = np.asarray(probabilities, dtype=np.float64)
        if values.ndim != 2 or self.temperature <= 0.0:
            raise ValueError("invalid probability matrix or temperature")
        values = np.clip(values, 1e-12, 1.0)
        values /= values.sum(axis=1, keepdims=True)
        return _softmax(np.log(values) / self.temperature).astype(np.float32)

    def to_dict(self) -> Dict[str, float]:
        return {"temperature": float(self.temperature)}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "TemperatureScaler":
        return cls(float(value["temperature"]))


def _finite_sample_quantile(values: Sequence[float], alpha: float) -> float:
    scores = np.sort(np.asarray(values, dtype=np.float64))
    if len(scores) < 1:
        raise ValueError("cannot calibrate an empty score collection")
    # Split-conformal 'higher' order statistic: ceil((n+1)*(1-alpha)).
    rank = min(len(scores), int(np.ceil((len(scores) + 1) * (1.0 - alpha))))
    return float(scores[max(0, rank - 1)])


def aggregate_candidate_rates(
    predictions: Sequence[float],
    labels: Sequence[float],
    root_ids: Sequence[str],
    candidate_ids: Sequence[str],
) -> Dict[Tuple[str, str], Tuple[float, float, int]]:
    """Average rollout replicates for each physical root/candidate cell."""

    if not (len(predictions) == len(labels) == len(root_ids) == len(candidate_ids)):
        raise ValueError("candidate aggregation inputs must have equal length")
    groups: Dict[Tuple[str, str], list] = {}
    for prediction, label, root_id, candidate_id in zip(
        predictions, labels, root_ids, candidate_ids
    ):
        if not np.isfinite(prediction) or not np.isfinite(label):
            raise ValueError("candidate predictions and labels must be finite")
        groups.setdefault((str(root_id), str(candidate_id)), []).append(
            (float(prediction), float(label))
        )
    return {
        key: (
            float(np.mean([row[0] for row in rows])),
            float(np.mean([row[1] for row in rows])),
            len(rows),
        )
        for key, rows in groups.items()
    }


@dataclass(frozen=True)
class CalibratedRiskBound:
    """Additive one-sided breach-risk bound fitted on independent lineages.

    ``root_max_offset`` is fitted from one maximum residual per independent
    lineage group (covering all sibling roots/candidates).  It is a conservative
    empirical bound, not an unconditional deployment guarantee.
    """

    alpha: float
    marginal_offset: float
    root_max_offset: float
    calibration_cells: int
    calibration_roots: int
    calibration_lineage_groups: int

    @classmethod
    def fit(
        cls,
        predictions: Sequence[float],
        labels: Sequence[float],
        root_ids: Sequence[str],
        candidate_ids: Sequence[str],
        lineage_group_ids: Sequence[str],
        alpha: float = 0.10,
    ) -> "CalibratedRiskBound":
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie strictly between zero and one")
        if len(lineage_group_ids) != len(root_ids):
            raise ValueError("lineage_group_ids must align with root_ids")
        cells = aggregate_candidate_rates(predictions, labels, root_ids, candidate_ids)
        root_to_group: Dict[str, str] = {}
        for root_id, group_id in zip(root_ids, lineage_group_ids):
            previous = root_to_group.setdefault(str(root_id), str(group_id))
            if previous != str(group_id):
                raise ValueError("one calibration root maps to multiple lineage groups")
        residuals = {
            key: true_rate - prediction
            for key, (prediction, true_rate, _) in cells.items()
        }
        marginal = max(0.0, _finite_sample_quantile(list(residuals.values()), alpha))
        root_scores: Dict[str, float] = {}
        for (root_id, _), residual in residuals.items():
            root_scores[root_id] = max(root_scores.get(root_id, -np.inf), residual)
        lineage_scores: Dict[str, float] = {}
        for root_id, residual in root_scores.items():
            group_id = root_to_group[root_id]
            lineage_scores[group_id] = max(
                lineage_scores.get(group_id, -np.inf), residual
            )
        root_max = max(
            marginal,
            0.0,
            _finite_sample_quantile(list(lineage_scores.values()), alpha),
        )
        return cls(
            alpha,
            marginal,
            root_max,
            len(cells),
            len(root_scores),
            len(lineage_scores),
        )

    def upper(self, breach_probability: Sequence[float], selection_safe: bool = True) -> np.ndarray:
        values = np.asarray(breach_probability, dtype=np.float64)
        offset = self.root_max_offset if selection_safe else self.marginal_offset
        return np.clip(values + offset, 0.0, 1.0).astype(np.float32)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "CalibratedRiskBound":
        return cls(
            alpha=float(value["alpha"]),
            marginal_offset=float(value["marginal_offset"]),
            root_max_offset=float(value["root_max_offset"]),
            calibration_cells=int(value["calibration_cells"]),
            calibration_roots=int(value["calibration_roots"]),
            calibration_lineage_groups=int(value["calibration_lineage_groups"]),
        )
