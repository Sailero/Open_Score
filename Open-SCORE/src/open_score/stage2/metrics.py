"""Auditable and decision-relevant evaluation metrics for Stage 2."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from .calibration import aggregate_candidate_rates
from .data import Stage2Record


METRIC_DEFINITIONS_ZH: Dict[str, str] = {
    "joint_nll": "真实‘终局类型×时间桶’被分配概率的平均负对数，越低越好。",
    "joint_brier": "全部联合类别的平方概率误差，越低越好。",
    "joint_accuracy": "概率最高的联合类别命中率，越高越好。",
    "top3_accuracy": "真实联合类别落在前三个预测类别中的比例，越高越好。",
    "brier": "二元事件概率与0/1真值的均方误差，越低越好。",
    "nll": "二元事件概率的负对数似然，越低越好。",
    "accuracy": "以0.5为阈值的事件分类正确率，越高越好。",
    "balanced_accuracy": "正负样本召回率的平均，类别不平衡时比普通正确率更直观。",
    "roc_auc": "随机正例风险高于随机负例的概率，0.5近似随机，越高越好。",
    "average_precision": "按风险从高到低检索时的平均精确率，越高越好。",
    "ece": "分桶后预测概率与真实频率之差的加权平均，越低越好。",
    "mce": "所有概率桶中最大的校准偏差，越低越好。",
    "row_event_rate_wilson95_descriptive": "按候选行计算的描述性95% Wilson区间；同一root内候选相关，不能当作cluster-level置信区间。",
    "time_mae_steps": "预计剩余步数与真实持续步数的平均绝对误差，越低越好。",
    "time_rmse_steps": "剩余步数误差的均方根，对大误差更敏感，越低越好。",
    "time_median_ae_steps": "剩余步数绝对误差的中位数，越低越好。",
    "baseline_mean_time_mae_steps": "始终预测训练集平均时长的MAE，供判断时间头是否有信息增益。",
    "candidate_rate_coverage": "真实重复rollout失守率不超过校准上界的候选单元比例。",
    "post_selection_coverage": "每个root经上界选出的候选仍被风险上界覆盖的比例。",
    "mean_interval_width": "校准风险上界减去点预测的平均宽度，越窄越有决策信息。",
    "mean_decision_regret": "所选候选真实失守率减去同root最优候选失守率，越低越好。",
    "ranking_pair_accuracy": "同一root内两候选的预测风险顺序与真实顺序一致的比例。",
    "ensemble_breach_std": "不同ensemble成员对失守率预测的平均标准差，仅表示模型分歧。",
    "mutual_information": "ensemble预测熵减去成员平均熵，表示认知分歧，不是覆盖保证。",
}


def metric_definitions_zh() -> Dict[str, str]:
    return dict(METRIC_DEFINITIONS_ZH)


def _safe_probability(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float64), 1e-8, 1.0 - 1e-8)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _roc_auc(labels: np.ndarray, probabilities: np.ndarray) -> Optional[float]:
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = _average_ranks(probabilities)
    value = (ranks[labels == 1].sum() - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )
    return float(value)


def _average_precision(labels: np.ndarray, probabilities: np.ndarray) -> Optional[float]:
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = np.argsort(-probabilities, kind="mergesort")
    ordered = labels[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(precision[ordered == 1].sum() / positives)


def _calibration_errors(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> Tuple[float, float]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    mce = 0.0
    for index in range(bins):
        if index == bins - 1:
            selected = (probabilities >= edges[index]) & (probabilities <= edges[index + 1])
        else:
            selected = (probabilities >= edges[index]) & (probabilities < edges[index + 1])
        if selected.any():
            gap = abs(float(probabilities[selected].mean() - labels[selected].mean()))
            ece += selected.mean() * gap
            mce = max(mce, gap)
    return float(ece), float(mce)


def _wilson_interval(successes: int, count: int, z: float = 1.959963984540054) -> list:
    if count < 1:
        raise ValueError("Wilson interval needs at least one observation")
    proportion = successes / count
    denominator = 1.0 + z * z / count
    centre = (proportion + z * z / (2.0 * count)) / denominator
    radius = z * np.sqrt(
        proportion * (1.0 - proportion) / count + z * z / (4.0 * count * count)
    ) / denominator
    return [float(max(0.0, centre - radius)), float(min(1.0, centre + radius))]


def _binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, object]:
    y = np.asarray(labels, dtype=np.int64)
    p = _safe_probability(probabilities)
    if y.shape != p.shape or y.ndim != 1 or len(y) < 1:
        raise ValueError("binary metric inputs must be equal non-empty vectors")
    predicted = p >= 0.5
    positive_recall = float(predicted[y == 1].mean()) if np.any(y == 1) else None
    negative_recall = float((~predicted[y == 0]).mean()) if np.any(y == 0) else None
    balanced = (
        float((positive_recall + negative_recall) / 2.0)
        if positive_recall is not None and negative_recall is not None
        else None
    )
    ece, mce = _calibration_errors(y, p)
    return {
        "count": int(len(y)),
        "event_rate": float(y.mean()),
        "row_event_rate_wilson95_descriptive": _wilson_interval(
            int(y.sum()), len(y)
        ),
        "mean_probability": float(p.mean()),
        "mean_probability_minus_event_rate": float(p.mean() - y.mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "nll": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1.0 - p))),
        "accuracy": float(np.mean(predicted == y)),
        "balanced_accuracy": balanced,
        "positive_recall": positive_recall,
        "negative_recall": negative_recall,
        "roc_auc": _roc_auc(y, p),
        "average_precision": _average_precision(y, p),
        "ece": ece,
        "mce": mce,
    }


def _joint_metrics(labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, object]:
    targets = np.asarray(labels, dtype=np.int64)
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or targets.shape != (len(values),):
        raise ValueError("joint metric inputs have incompatible shapes")
    values = np.clip(values, 1e-12, 1.0)
    values /= values.sum(axis=1, keepdims=True)
    one_hot = np.eye(values.shape[1], dtype=np.float64)[targets]
    top_count = min(3, values.shape[1])
    top = np.argpartition(values, -top_count, axis=1)[:, -top_count:]
    return {
        "joint_nll": float(-np.log(values[np.arange(len(targets)), targets]).mean()),
        "joint_brier": float(np.sum((values - one_hot) ** 2, axis=1).mean()),
        "joint_accuracy": float(np.mean(values.argmax(axis=1) == targets)),
        "top3_accuracy": float(np.mean(np.any(top == targets[:, None], axis=1))),
    }


def _probability_views(
    records: Sequence[Stage2Record], probabilities: np.ndarray, horizon_bins: int, steps_per_bin: int
) -> Dict[str, np.ndarray]:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (len(records), 2 * horizon_bins + 1):
        raise ValueError("probability matrix does not match records/horizon bins")
    defender = values[:, :horizon_bins]
    breach_bins = values[:, horizon_bins : 2 * horizon_bins]
    timeout = values[:, -1]
    midpoints = (np.arange(horizon_bins, dtype=np.float64) + 0.5) * steps_per_bin
    expected_steps = (defender + breach_bins) @ midpoints
    expected_steps += timeout * (horizon_bins * steps_per_bin)
    early = np.zeros(len(records), dtype=np.float64)
    for index, record in enumerate(records):
        command_bins = min(
            horizon_bins, max(1, int(np.ceil(record.query.command_steps / steps_per_bin)))
        )
        early[index] = breach_bins[index, :command_bins].sum()
    return {
        "defender_success": defender.sum(axis=1) + timeout,
        "breach": breach_bins.sum(axis=1),
        "timeout": timeout,
        "early_breach": early,
        "expected_steps": expected_steps,
    }


def _time_metrics(
    records: Sequence[Stage2Record], expected_steps: np.ndarray, baseline_steps: Optional[float]
) -> Dict[str, object]:
    truth = np.asarray([record.terminal_steps for record in records], dtype=np.float64)
    error = np.asarray(expected_steps, dtype=np.float64) - truth
    terminal = np.asarray([record.outcome != "timeout" for record in records])
    result: Dict[str, object] = {
        "time_mae_steps": float(np.abs(error).mean()),
        "time_rmse_steps": float(np.sqrt(np.mean(error**2))),
        "time_median_ae_steps": float(np.median(np.abs(error))),
        "mean_true_steps": float(truth.mean()),
        "mean_predicted_steps": float(np.mean(expected_steps)),
        "terminal_only_mae_steps": float(np.abs(error[terminal]).mean()) if terminal.any() else None,
    }
    if baseline_steps is not None:
        result["baseline_mean_time_mae_steps"] = float(np.abs(truth - baseline_steps).mean())
        result["mae_improvement_over_baseline_steps"] = float(
            result["baseline_mean_time_mae_steps"] - result["time_mae_steps"]
        )
    return result


def _basic_subset(
    records: Sequence[Stage2Record],
    views: Mapping[str, np.ndarray],
    indices: np.ndarray,
) -> Dict[str, object]:
    breach = np.asarray([record.outcome == "breach" for record in records], dtype=np.int64)
    early = np.asarray([record.short_window_breach for record in records], dtype=np.int64)
    truth_steps = np.asarray([record.terminal_steps for record in records], dtype=np.float64)
    predicted_steps = np.asarray(views["expected_steps"])
    return {
        "count": int(indices.sum()),
        "breach": _binary_metrics(breach[indices], np.asarray(views["breach"])[indices]),
        "short_window_breach": _binary_metrics(
            early[indices], np.asarray(views["early_breach"])[indices]
        ),
        "time_mae_steps": float(np.abs(predicted_steps[indices] - truth_steps[indices]).mean()),
        "time_rmse_steps": float(
            np.sqrt(np.mean((predicted_steps[indices] - truth_steps[indices]) ** 2))
        ),
    }


def _uncertainty_metrics(member_probabilities: np.ndarray, horizon_bins: int) -> Dict[str, float]:
    members = np.asarray(member_probabilities, dtype=np.float64)
    if members.ndim != 3 or members.shape[-1] != 2 * horizon_bins + 1:
        raise ValueError("member probabilities need shape [members, samples, classes]")
    breach = members[:, :, horizon_bins : 2 * horizon_bins].sum(axis=-1)
    clipped = np.clip(members, 1e-12, 1.0)
    member_entropy = -np.sum(clipped * np.log(clipped), axis=-1)
    mean_probability = members.mean(axis=0)
    predictive_entropy = -np.sum(
        np.clip(mean_probability, 1e-12, 1.0) * np.log(np.clip(mean_probability, 1e-12, 1.0)),
        axis=-1,
    )
    mutual_information = np.maximum(0.0, predictive_entropy - member_entropy.mean(axis=0))
    return {
        "ensemble_breach_std": float(breach.std(axis=0).mean()),
        "ensemble_breach_std_p90": float(np.quantile(breach.std(axis=0), 0.90)),
        "predictive_entropy": float(predictive_entropy.mean()),
        "mutual_information": float(mutual_information.mean()),
    }


def _risk_and_decision_metrics(
    records: Sequence[Stage2Record], breach_probability: np.ndarray, breach_upper: np.ndarray
) -> Tuple[Dict[str, object], Dict[str, object]]:
    labels = np.asarray([record.outcome == "breach" for record in records], dtype=np.float64)
    roots = [record.query.root_id for record in records]
    candidates = [record.query.candidate_id for record in records]
    point_cells = aggregate_candidate_rates(breach_probability, labels, roots, candidates)
    upper_cells = aggregate_candidate_rates(breach_upper, labels, roots, candidates)
    coverage = []
    widths = []
    shortfalls = []
    by_root: Dict[str, list] = {}
    for key, (point, truth, count) in point_cells.items():
        upper = upper_cells[key][0]
        coverage.append(float(truth <= upper + 1e-12))
        widths.append(upper - point)
        shortfalls.append(max(0.0, truth - upper))
        by_root.setdefault(key[0], []).append((key[1], point, upper, truth, count))
    risk = {
        "candidate_cells": len(point_cells),
        "mean_rollouts_per_candidate_cell": float(
            np.mean([cell[2] for cell in point_cells.values()])
        ),
        "candidate_cells_with_replicates": int(
            sum(cell[2] > 1 for cell in point_cells.values())
        ),
        "candidate_rate_coverage": float(np.mean(coverage)),
        "mean_interval_width": float(np.mean(widths)),
        "median_interval_width": float(np.median(widths)),
        "mean_upper_bound": float(np.mean([cell[0] for cell in upper_cells.values()])),
        "worst_uncovered_shortfall": float(np.max(shortfalls)),
    }
    regrets = []
    selected_coverage = []
    ranking = []
    for cells in by_root.values():
        selected = min(cells, key=lambda cell: (cell[2], cell[1], cell[0]))
        oracle_truth = min(cell[3] for cell in cells)
        regrets.append(selected[3] - oracle_truth)
        selected_coverage.append(float(selected[3] <= selected[2] + 1e-12))
        for left in range(len(cells)):
            for right in range(left + 1, len(cells)):
                predicted_delta = cells[left][1] - cells[right][1]
                true_delta = cells[left][3] - cells[right][3]
                if true_delta == 0.0:
                    continue
                ranking.append(float(predicted_delta * true_delta > 0.0) + 0.5 * float(predicted_delta == 0.0))
    decision = {
        "roots": len(by_root),
        "mean_decision_regret": float(np.mean(regrets)),
        "worst_decision_regret": float(np.max(regrets)),
        "zero_regret_root_rate": float(np.mean(np.asarray(regrets) <= 1e-12)),
        "post_selection_coverage": float(np.mean(selected_coverage)),
        "ranking_pair_accuracy": float(np.mean(ranking)) if ranking else None,
        "ranking_pairs": len(ranking),
    }
    return risk, decision


def evaluate_stage2_predictions(
    records: Sequence[Stage2Record],
    probabilities: np.ndarray,
    *,
    horizon_bins: int,
    steps_per_bin: int,
    breach_upper: Optional[np.ndarray] = None,
    member_probabilities: Optional[np.ndarray] = None,
    train_mean_terminal_steps: Optional[float] = None,
    seen_policy_pairs: Optional[Set[Tuple[str, str]]] = None,
) -> Dict[str, object]:
    """Evaluate predictions without fitting or selecting thresholds on test data."""

    if not records:
        raise ValueError("cannot evaluate an empty Stage-2 dataset")
    values = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(
        [record.terminal_class(horizon_bins, steps_per_bin) for record in records], dtype=np.int64
    )
    views = _probability_views(records, values, horizon_bins, steps_per_bin)
    defender_success = np.asarray([record.outcome != "breach" for record in records], dtype=np.int64)
    breach = 1 - defender_success
    timeout = np.asarray([record.outcome == "timeout" for record in records], dtype=np.int64)
    early = np.asarray([record.short_window_breach for record in records], dtype=np.int64)
    result: Dict[str, object] = {
        "sample_count": len(records),
        "root_count": len({record.query.root_id for record in records}),
        "class_balance": {
            "defender_win_rate": float(np.mean([record.outcome == "defender_win" for record in records])),
            "breach_rate": float(breach.mean()),
            "timeout_rate": float(timeout.mean()),
            "short_window_breach_rate": float(early.mean()),
        },
        "joint_outcome_time": _joint_metrics(labels, values),
        "defender_success": _binary_metrics(defender_success, views["defender_success"]),
        "breach": _binary_metrics(breach, views["breach"]),
        "timeout": _binary_metrics(timeout, views["timeout"]),
        "short_window_breach": _binary_metrics(early, views["early_breach"]),
        "remaining_time": _time_metrics(records, views["expected_steps"], train_mean_terminal_steps),
    }
    if member_probabilities is not None:
        result["ensemble_disagreement"] = _uncertainty_metrics(member_probabilities, horizon_bins)
    if breach_upper is not None:
        risk, decision = _risk_and_decision_metrics(records, views["breach"], breach_upper)
        result["calibrated_risk_bound"] = risk
        result["candidate_decision"] = decision

    by_scale: Dict[str, object] = {}
    for scale in sorted({(record.query.defender_count, record.query.attacker_count) for record in records}):
        selected = np.asarray(
            [
                (record.query.defender_count, record.query.attacker_count) == scale
                for record in records
            ],
            dtype=bool,
        )
        by_scale[f"{scale[0]}v{scale[1]}"] = _basic_subset(records, views, selected)
    result["by_scale"] = by_scale
    scale_briers = [group["breach"]["brier"] for group in by_scale.values()]
    scale_maes = [group["time_mae_steps"] for group in by_scale.values()]
    result["worst_scale_summary"] = {
        "max_breach_brier": float(max(scale_briers)),
        "max_time_mae_steps": float(max(scale_maes)),
    }

    by_pair: Dict[str, object] = {}
    for pair in sorted({record.query.policy_pair for record in records}):
        selected = np.asarray([record.query.policy_pair == pair for record in records], dtype=bool)
        by_pair[f"{pair[0]}__vs__{pair[1]}"] = _basic_subset(records, views, selected)
    result["by_policy_pair"] = by_pair

    if seen_policy_pairs is not None:
        seen = np.asarray([record.query.policy_pair in seen_policy_pairs for record in records])
        generalization: Dict[str, object] = {}
        if seen.any():
            generalization["seen_policy_pair"] = _basic_subset(records, views, seen)
        if (~seen).any():
            generalization["held_out_policy_pair"] = _basic_subset(records, views, ~seen)
        result["policy_pair_generalization"] = generalization
    return result
