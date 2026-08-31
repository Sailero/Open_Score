"""Strong tabular baselines selected only on the validation lineage split."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from .data import Stage2FeatureEncoder, Stage2Record


@dataclass(frozen=True)
class TabularBaselineResult:
    summary: Dict[str, object]
    breach_probability: np.ndarray
    event_time_prediction: np.ndarray


def _lineage_weights(records: Sequence[Stage2Record]) -> np.ndarray:
    counts: Dict[str, int] = {}
    for record in records:
        counts[record.query.lineage_group_id] = counts.get(record.query.lineage_group_id, 0) + 1
    return np.asarray(
        [1.0 / counts[record.query.lineage_group_id] for record in records], np.float64
    )


def fit_hist_gradient_boosting_baseline(
    train_records: Sequence[Stage2Record],
    validation_records: Sequence[Stage2Record],
    test_records: Sequence[Stage2Record],
    config: Optional[Mapping[str, object]] = None,
) -> TabularBaselineResult:
    """Fit sklearn HistGradientBoosting classification and event-time heads.

    This is a strong non-neural tabular comparator, not the calibrated deployed
    model. Hyperparameters are selected on validation only; test is touched once.
    """

    try:
        import sklearn
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ImportError as error:  # pragma: no cover - depends on optional environment.
        raise RuntimeError(
            "scikit-learn is required for the optional strong tabular baseline"
        ) from error
    values = dict(config or {})
    candidates = tuple(int(value) for value in values.get("max_iter_candidates", (50, 100, 200)))
    if not candidates or any(value < 1 for value in candidates):
        raise ValueError("max_iter_candidates must contain positive integers")
    encoder = Stage2FeatureEncoder.fit(train_records, include_policy_context=True)
    x_train = encoder.transform(train_records)
    x_validation = encoder.transform(validation_records)
    x_test = encoder.transform(test_records)
    y_train = np.asarray([record.outcome == "breach" for record in train_records], int)
    y_validation = np.asarray([record.outcome == "breach" for record in validation_records], int)
    y_test = np.asarray([record.outcome == "breach" for record in test_records], int)
    weights = _lineage_weights(train_records)
    classifier_rows = []
    best_classifier = None
    for max_iter in candidates:
        classifier = HistGradientBoostingClassifier(
            learning_rate=float(values.get("learning_rate", 0.08)),
            max_iter=max_iter,
            max_leaf_nodes=int(values.get("max_leaf_nodes", 31)),
            l2_regularization=float(values.get("l2_regularization", 1.0)),
            early_stopping=False,
            random_state=int(values.get("seed", 20260831)),
        ).fit(x_train, y_train, sample_weight=weights)
        validation_probability = classifier.predict_proba(x_validation)[:, 1]
        validation_brier = float(np.mean((validation_probability - y_validation) ** 2))
        classifier_rows.append({"max_iter": max_iter, "validation_brier": validation_brier})
        if best_classifier is None or validation_brier < best_classifier[0]:
            best_classifier = (validation_brier, max_iter, classifier)
    breach_probability = best_classifier[2].predict_proba(x_test)[:, 1]

    train_event = np.asarray([record.event_observed for record in train_records], bool)
    validation_event = np.asarray([record.event_observed for record in validation_records], bool)
    test_event = np.asarray([record.event_observed for record in test_records], bool)
    train_time = np.asarray([record.terminal_steps for record in train_records], np.float64)
    validation_time = np.asarray(
        [record.terminal_steps for record in validation_records], np.float64
    )
    test_time = np.asarray([record.terminal_steps for record in test_records], np.float64)
    if not train_event.any() or not validation_event.any():
        raise ValueError("tabular event-time head needs observed events in train and validation")
    regressor_rows = []
    best_regressor = None
    event_weights = weights[train_event]
    for max_iter in candidates:
        regressor = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=float(values.get("learning_rate", 0.08)),
            max_iter=max_iter,
            max_leaf_nodes=int(values.get("max_leaf_nodes", 31)),
            l2_regularization=float(values.get("l2_regularization", 1.0)),
            early_stopping=False,
            random_state=int(values.get("seed", 20260831)),
        ).fit(x_train[train_event], train_time[train_event], sample_weight=event_weights)
        validation_prediction = regressor.predict(x_validation[validation_event])
        validation_mae = float(
            np.mean(np.abs(validation_prediction - validation_time[validation_event]))
        )
        regressor_rows.append({"max_iter": max_iter, "validation_event_mae": validation_mae})
        if best_regressor is None or validation_mae < best_regressor[0]:
            best_regressor = (validation_mae, max_iter, regressor)
    event_time_prediction = best_regressor[2].predict(x_test)
    median = float(np.median(train_time[train_event]))
    event_mae = (
        float(np.mean(np.abs(event_time_prediction[test_event] - test_time[test_event])))
        if test_event.any()
        else None
    )
    median_mae = (
        float(np.mean(np.abs(median - test_time[test_event]))) if test_event.any() else None
    )
    summary: Dict[str, object] = {
        "name": "sklearn_hist_gradient_boosting",
        "role": "strong_tabular_comparator_not_deployed_calibrated_model",
        "sklearn_version": sklearn.__version__,
        "lineage_equalised_training_weights": True,
        "hyperparameter_selection_split": "validation",
        "classifier_candidates": classifier_rows,
        "selected_classifier_max_iter": best_classifier[1],
        "regressor_candidates": regressor_rows,
        "selected_regressor_max_iter": best_regressor[1],
        "test_breach_brier": float(np.mean((breach_probability - y_test) ** 2)),
        "test_breach_roc_auc": (
            float(roc_auc_score(y_test, breach_probability))
            if len(np.unique(y_test)) == 2
            else None
        ),
        "test_breach_average_precision": (
            float(average_precision_score(y_test, breach_probability)) if y_test.any() else None
        ),
        "test_event_time_mae_steps": event_mae,
        "test_median_baseline_event_time_mae_steps": median_mae,
        "test_event_time_mae_improvement_over_median": (
            None if event_mae is None else float(median_mae - event_mae)
        ),
        "right_censored_rows_excluded_from_time_mae": int((~test_event).sum()),
    }
    return TabularBaselineResult(summary, breach_probability, event_time_prediction)


__all__ = ["TabularBaselineResult", "fit_hist_gradient_boosting_baseline"]
