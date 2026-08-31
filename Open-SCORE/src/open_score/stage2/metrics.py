"""Censor-aware, decision-relevant metrics for formal Stage 2."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from .calibration import aggregate_candidate_rates
from .data import PHYSICAL_REDUNDANT_RELATIONS, PHYSICAL_TARGET_FIELDS, Stage2Record


METRIC_DEFINITIONS_ZH: Dict[str, str] = {
    "right_censored_competing_risk_nll": "竞争风险离散分布的右删失负对数似然，越低越好。timeout 只表示至少存活到截尾时刻。",
    "observed_event_joint_accuracy": "仅在真实终局已观察样本上，终局类型×时间箱的最高概率命中率。",
    "brier": "预测事件概率与0/1真值的均方误差，越低越好。",
    "roc_auc": "随机正例的风险评分高于随机负例的概率；0.5接近随机。",
    "average_precision": "按风险从高到低检索时的平均精确率。",
    "ece": "分箱预测概率与真实频率差异的加权平均，越低越好。",
    "event_time_mae_steps": "只对已观察到终局的轨迹计算的剩余步数平均绝对误差。",
    "median_baseline_event_time_mae_steps": "始终预测训练集已观察终局时间中位数时的MAE；MAE的合法常数基线。",
    "event_time_mae_improvement_over_median": "中位数基线MAE减模型MAE；大于0才表示时间头有增益。",
    "mean_predicted_survival_at_censor": "右删失样本在删失时刻仍未发生终局的预测概率，越高表示越符合删失观测。",
    "candidate_rate_coverage": "真实重复推演失守率不超过校准上界的Red候选单元比例。",
    "mean_red_decision_regret": "每个root所选Red策略的真实平均失守率减去同root最优Red策略失守率。",
    "minimax_red_decision_regret": "按最坏Blue威胁选Red后的真实最坏失守率，相对最优minimax Red策略的差值。",
    "red_selection_accuracy": "模型所选Red策略与真实最优Red策略一致的root比例。",
    "red_ranking_pair_accuracy": "同root内Red候选两两风险排序与真实排序一致的比例。",
    "lineage_cluster_bootstrap95": "以lineage为抽样单位的描述性95% bootstrap区间，保留root内相关性。",
    "integrated_breach_brier": "逐时间箱累计失守概率的平均Brier分数；早于该时刻删失的样本不参与。",
    "conditional_event_time_mae_steps": "在已观察终局样本上，条件于预测期内发生终局的中位时间MAE。",
    "climatology_brier_skill": "相对训练集常数失守率的Brier技能分数；大于0表示优于气候基线。",
    "physical_target_mae": "直观物理指标预测值与真实值的平均绝对误差，越低越好。",
    "physical_target_rmse": "直观物理指标预测值与真实值的均方根误差，对大误差更敏感，越低越好。",
    "physical_target_r2": "直观物理指标的决定系数；1为完美，0等同测试均值常数预测，负值更差。",
    "physical_mae_skill_over_train_median": "1-MAE/训练集标签中位数常数基线MAE；大于0表示监督头有效。",
    "physical_ensemble_interval": "bootstrap成员预测分位数区间，仅表示模型不稳定性，不是有限样本覆盖保证。",
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
    return float(
        (ranks[labels == 1].sum() - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def _average_precision(labels: np.ndarray, probabilities: np.ndarray) -> Optional[float]:
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = np.argsort(-probabilities, kind="mergesort")
    ordered = labels[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(precision[ordered == 1].sum() / positives)


def _wilson_interval(successes: int, count: int, z: float = 1.959963984540054) -> list:
    proportion = successes / count
    denominator = 1.0 + z * z / count
    centre = (proportion + z * z / (2.0 * count)) / denominator
    radius = z * np.sqrt(
        proportion * (1.0 - proportion) / count + z * z / (4.0 * count * count)
    ) / denominator
    return [float(max(0.0, centre - radius)), float(min(1.0, centre + radius))]


def _binary_metrics(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> Dict[str, object]:
    y = np.asarray(labels, dtype=np.int64)
    p = _safe_probability(probabilities)
    if y.shape != p.shape or y.ndim != 1 or len(y) < 1:
        raise ValueError("binary metric inputs must be equal non-empty vectors")
    predicted = p >= 0.5
    positive_recall = float(predicted[y == 1].mean()) if np.any(y == 1) else None
    negative_recall = float((~predicted[y == 0]).mean()) if np.any(y == 0) else None
    reliability = []
    ece, mce = 0.0, 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        selected = (p >= edges[index]) & (
            p <= edges[index + 1] if index == bins - 1 else p < edges[index + 1]
        )
        if selected.any():
            confidence = float(p[selected].mean())
            frequency = float(y[selected].mean())
            gap = abs(confidence - frequency)
            ece += float(selected.mean()) * gap
            mce = max(mce, gap)
            reliability.append(
                {
                    "lower": float(edges[index]),
                    "upper": float(edges[index + 1]),
                    "count": int(selected.sum()),
                    "mean_probability": confidence,
                    "event_rate": frequency,
                }
            )
    return {
        "count": len(y),
        "event_rate": float(y.mean()),
        "event_rate_wilson95_descriptive": _wilson_interval(int(y.sum()), len(y)),
        "mean_probability": float(p.mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "nll": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "accuracy": float(np.mean(predicted == y)),
        "balanced_accuracy": (
            float((positive_recall + negative_recall) / 2.0)
            if positive_recall is not None and negative_recall is not None
            else None
        ),
        "roc_auc": _roc_auc(y, p),
        "average_precision": _average_precision(y, p),
        "ece": float(ece),
        "mce": float(mce),
        "reliability_bins": reliability,
    }


def _probability_views(
    records: Sequence[Stage2Record],
    probabilities: np.ndarray,
    horizon_bins: int,
    steps_per_bin: int,
) -> Dict[str, np.ndarray]:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (len(records), 2 * horizon_bins + 1):
        raise ValueError("probability matrix shape differs from Stage-2 contract")
    values = np.clip(values, 1e-12, 1.0)
    values /= values.sum(axis=1, keepdims=True)
    defender = values[:, :horizon_bins]
    breach = values[:, horizon_bins : 2 * horizon_bins]
    survival = values[:, -1]
    midpoint = (np.arange(horizon_bins) + 0.5) * steps_per_bin
    restricted_mean = (defender + breach) @ midpoint + survival * (
        horizon_bins * steps_per_bin
    )
    event_mass = defender + breach
    event_total = event_mass.sum(axis=1)
    conditional_cdf = np.cumsum(event_mass, axis=1) / np.clip(
        event_total[:, None], 1e-12, None
    )
    conditional_median_bin = np.argmax(conditional_cdf >= 0.5, axis=1)
    conditional_median = (conditional_median_bin + 0.5) * steps_per_bin
    conditional_median[event_total <= 1e-12] = horizon_bins * steps_per_bin
    early = np.asarray(
        [
            breach[index, : min(horizon_bins, int(np.ceil(record.query.command_steps / steps_per_bin)))].sum()
            for index, record in enumerate(records)
        ]
    )
    survival_at_censor = np.ones(len(records), dtype=np.float64)
    for index, record in enumerate(records):
        if not record.event_observed:
            start = min(horizon_bins, int(record.terminal_steps // steps_per_bin))
            survival_at_censor[index] = (
                defender[index, start:].sum()
                + breach[index, start:].sum()
                + survival[index]
            )
    return {
        "probabilities": values,
        "defender_win": defender.sum(axis=1),
        "breach": breach.sum(axis=1),
        "survival": survival,
        "defender_success": defender.sum(axis=1) + survival,
        "early_breach": early,
        "restricted_mean_steps": restricted_mean,
        "conditional_event_median_steps": conditional_median,
        "survival_at_censor": survival_at_censor,
    }


def _right_censored_joint_metrics(
    records: Sequence[Stage2Record],
    values: np.ndarray,
    horizon_bins: int,
    steps_per_bin: int,
) -> Dict[str, object]:
    labels = np.asarray(
        [record.terminal_class(horizon_bins, steps_per_bin) for record in records], np.int64
    )
    observed = np.asarray([record.event_observed for record in records], bool)
    likelihood = values[np.arange(len(values)), labels].copy()
    for index in np.flatnonzero(~observed):
        start = min(horizon_bins, int(records[index].terminal_steps // steps_per_bin))
        likelihood[index] = (
            values[index, start:horizon_bins].sum()
            + values[index, horizon_bins + start : 2 * horizon_bins].sum()
            + values[index, -1]
        )
    event_accuracy = (
        float(np.mean(values[observed].argmax(axis=1) == labels[observed]))
        if observed.any()
        else None
    )
    top3 = None
    if observed.any():
        top = np.argpartition(values[observed], -3, axis=1)[:, -3:]
        top3 = float(np.mean(np.any(top == labels[observed, None], axis=1)))
    time_brier = []
    for time_bin in range(horizon_bins):
        time_step = (time_bin + 1) * steps_per_bin
        eligible = np.asarray(
            [
                record.event_observed or record.terminal_steps >= time_step
                for record in records
            ],
            bool,
        )
        if eligible.any():
            truth = np.asarray(
                [
                    record.outcome == "breach" and record.terminal_steps <= time_step
                    for record in records
                ],
                float,
            )
            predicted = values[:, horizon_bins : horizon_bins + time_bin + 1].sum(axis=1)
            time_brier.append(float(np.mean((predicted[eligible] - truth[eligible]) ** 2)))
    return {
        "right_censored_competing_risk_nll": float(
            -np.log(np.clip(likelihood, 1e-12, 1.0)).mean()
        ),
        "observed_events": int(observed.sum()),
        "right_censored": int((~observed).sum()),
        "observed_event_joint_accuracy": event_accuracy,
        "observed_event_top3_accuracy": top3,
        "integrated_breach_brier": float(np.mean(time_brier)) if time_brier else None,
        "time_bins_scored": len(time_brier),
    }


def _time_metrics(
    records: Sequence[Stage2Record],
    predicted_rmst_steps: np.ndarray,
    train_median_event_steps: Optional[float],
    survival_at_censor: Optional[np.ndarray] = None,
    conditional_event_median_steps: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    truth = np.asarray([record.terminal_steps for record in records], np.float64)
    predicted_rmst = np.asarray(predicted_rmst_steps, np.float64)
    predicted_event = np.asarray(
        predicted_rmst_steps
        if conditional_event_median_steps is None
        else conditional_event_median_steps,
        np.float64,
    )
    observed = np.asarray([record.event_observed for record in records], bool)
    result: Dict[str, object] = {
        "interpretation": "restricted mean time; timeout is right-censored",
        "observed_event_count": int(observed.sum()),
        "right_censored_count": int((~observed).sum()),
        "mean_predicted_restricted_mean_steps": float(predicted_rmst.mean()),
    }
    if observed.any():
        error = predicted_event[observed] - truth[observed]
        result.update(
            {
                "conditional_event_time_mae_steps": float(np.abs(error).mean()),
                "conditional_event_time_rmse_steps": float(np.sqrt(np.mean(error**2))),
                "conditional_event_time_median_ae_steps": float(np.median(np.abs(error))),
            }
        )
        # Compatibility aliases now use the coherent conditional-event estimand.
        result["event_time_mae_steps"] = result["conditional_event_time_mae_steps"]
        result["event_time_rmse_steps"] = result["conditional_event_time_rmse_steps"]
        result["event_time_median_ae_steps"] = result[
            "conditional_event_time_median_ae_steps"
        ]
        if train_median_event_steps is not None:
            baseline = float(np.abs(truth[observed] - train_median_event_steps).mean())
            result["train_median_event_steps"] = float(train_median_event_steps)
            result["median_baseline_event_time_mae_steps"] = baseline
            result["conditional_event_time_mae_improvement_over_median"] = float(
                baseline - result["conditional_event_time_mae_steps"]
            )
            result["event_time_mae_improvement_over_median"] = result[
                "conditional_event_time_mae_improvement_over_median"
            ]
    if (~observed).any() and survival_at_censor is not None:
        survival = np.asarray(survival_at_censor, np.float64)[~observed]
        result["mean_predicted_survival_at_censor"] = float(survival.mean())
        result["censoring_log_score"] = float(-np.log(np.clip(survival, 1e-12, 1.0)).mean())
        result["censor_lower_bound_violation_rate"] = float(
            np.mean(predicted_event[~observed] < truth[~observed])
        )
    return result


def _uncertainty_metrics(member_probabilities: np.ndarray, horizon_bins: int) -> Dict[str, float]:
    members = np.asarray(member_probabilities, dtype=np.float64)
    breach = members[:, :, horizon_bins : 2 * horizon_bins].sum(axis=-1)
    clipped = np.clip(members, 1e-12, 1.0)
    member_entropy = -np.sum(clipped * np.log(clipped), axis=-1)
    mean_probability = members.mean(axis=0)
    predictive_entropy = -np.sum(
        np.clip(mean_probability, 1e-12, 1.0)
        * np.log(np.clip(mean_probability, 1e-12, 1.0)),
        axis=-1,
    )
    mutual_information = np.maximum(0.0, predictive_entropy - member_entropy.mean(axis=0))
    return {
        "ensemble_breach_std": float(breach.std(axis=0).mean()),
        "ensemble_breach_std_p90": float(np.quantile(breach.std(axis=0), 0.90)),
        "predictive_entropy": float(predictive_entropy.mean()),
        "mutual_information": float(mutual_information.mean()),
    }


def _risk_and_red_decision_metrics(
    records: Sequence[Stage2Record],
    breach_probability: np.ndarray,
    breach_upper: np.ndarray,
    train_candidate_breach_rates: Optional[Mapping[str, float]] = None,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    labels = np.asarray([record.outcome == "breach" for record in records], np.float64)
    roots = [record.query.root_id for record in records]
    red_candidates = [record.query.candidate_id for record in records]
    point_cells = aggregate_candidate_rates(breach_probability, labels, roots, red_candidates)
    upper_cells = aggregate_candidate_rates(breach_upper, labels, roots, red_candidates)
    threat_cells: Dict[Tuple[str, str, str], list] = {}
    for index, record in enumerate(records):
        key = (record.query.root_id, record.query.candidate_id, record.query.threat_id)
        threat_cells.setdefault(key, []).append(
            (float(breach_probability[index]), float(breach_upper[index]), float(labels[index]))
        )
    coverage, widths, shortfalls = [], [], []
    by_root: Dict[str, list] = {}
    for key, (point, truth, count) in point_cells.items():
        upper = upper_cells[key][0]
        coverage.append(float(truth <= upper + 1e-12))
        widths.append(max(0.0, upper - point))
        shortfalls.append(max(0.0, truth - upper))
        by_root.setdefault(key[0], []).append((key[1], point, upper, truth, count))
    risk = {
        "unit": "root_x_red_candidate_averaged_over_blue_threats_and_continuations",
        "candidate_cells": len(point_cells),
        "mean_rollouts_per_candidate_cell": float(np.mean([cell[2] for cell in point_cells.values()])),
        "candidate_cells_with_replicates": int(sum(cell[2] > 1 for cell in point_cells.values())),
        "candidate_rate_coverage": float(np.mean(coverage)),
        "mean_interval_width": float(np.mean(widths)),
        "median_interval_width": float(np.median(widths)),
        "mean_upper_bound": float(np.mean([cell[0] for cell in upper_cells.values()])),
        "worst_uncovered_shortfall": float(np.max(shortfalls)),
    }
    regrets, minimax_regrets, selection_hits, selected_coverage, ranking = [], [], [], [], []
    validation_global_regrets, random_expected_regrets = [], []
    root_rows = []
    for root_id, cells in by_root.items():
        selected = min(cells, key=lambda cell: (cell[2], cell[1], cell[0]))
        oracle = min(cells, key=lambda cell: (cell[3], cell[0]))
        regrets.append(selected[3] - oracle[3])
        random_expected_regrets.append(float(np.mean([cell[3] for cell in cells]) - oracle[3]))
        if train_candidate_breach_rates:
            eligible = [
                cell for cell in cells if cell[0] in train_candidate_breach_rates
            ]
            if eligible:
                baseline_selected = min(
                    eligible,
                    key=lambda cell: (
                        float(train_candidate_breach_rates[cell[0]]),
                        cell[0],
                    ),
                )
                validation_global_regrets.append(baseline_selected[3] - oracle[3])
        selection_hits.append(float(selected[0] == oracle[0]))
        selected_coverage.append(float(selected[3] <= selected[2] + 1e-12))
        threat_summary = {}
        for candidate, _, _, _, _ in cells:
            candidate_threats = {
                threat: rows
                for (root, red, threat), rows in threat_cells.items()
                if root == root_id and red == candidate
            }
            threat_summary[candidate] = {
                "predicted_worst": max(np.mean([row[0] for row in rows]) for rows in candidate_threats.values()),
                "upper_worst": max(np.mean([row[1] for row in rows]) for rows in candidate_threats.values()),
                "true_worst": max(np.mean([row[2] for row in rows]) for rows in candidate_threats.values()),
            }
        robust_selected = min(
            threat_summary, key=lambda candidate: (threat_summary[candidate]["upper_worst"], candidate)
        )
        robust_oracle = min(
            threat_summary, key=lambda candidate: (threat_summary[candidate]["true_worst"], candidate)
        )
        minimax_regrets.append(
            threat_summary[robust_selected]["true_worst"]
            - threat_summary[robust_oracle]["true_worst"]
        )
        root_rows.append(
            {
                "root_id": root_id,
                "selected_red_candidate": selected[0],
                "oracle_red_candidate": oracle[0],
                "decision_regret": float(selected[3] - oracle[3]),
                "minimax_selected_red_candidate": robust_selected,
                "minimax_oracle_red_candidate": robust_oracle,
            }
        )
        for left in range(len(cells)):
            for right in range(left + 1, len(cells)):
                predicted_delta = cells[left][1] - cells[right][1]
                true_delta = cells[left][3] - cells[right][3]
                if true_delta != 0.0:
                    ranking.append(
                        float(predicted_delta * true_delta > 0.0)
                        + 0.5 * float(predicted_delta == 0.0)
                    )
    decision = {
        "decision_variable": "Red_candidate_only",
        "conditioned_on": "Blue_threat_distribution",
        "roots": len(by_root),
        "red_candidates": sorted(set(red_candidates)),
        "blue_threats": sorted({record.query.threat_id for record in records}),
        "mean_red_decision_regret": float(np.mean(regrets)),
        "worst_red_decision_regret": float(np.max(regrets)),
        "minimax_red_decision_regret": float(np.mean(minimax_regrets)),
        "red_selection_accuracy": float(np.mean(selection_hits)),
        "zero_regret_root_rate": float(np.mean(np.asarray(regrets) <= 1e-12)),
        "post_selection_coverage": float(np.mean(selected_coverage)),
        "red_ranking_pair_accuracy": float(np.mean(ranking)) if ranking else None,
        "random_red_candidate_expected_regret": float(np.mean(random_expected_regrets)),
        "validation_global_best_red_candidate_regret": (
            float(np.mean(validation_global_regrets))
            if validation_global_regrets
            else None
        ),
        "ranking_pairs": len(ranking),
        "per_root_selection": root_rows,
    }
    return risk, decision


def _physical_outcome_metrics(records: Sequence[Stage2Record]) -> Dict[str, object]:
    """Summarise directly interpretable rollout outcomes without model inference."""

    fields = (
        "payoff_red",
        "target_final_health_fraction",
        "target_min_health_fraction",
        "defender_survivors",
        "attacker_survivors",
        "defender_casualties",
        "attacker_casualties",
        "minimum_threat_distance",
        "cumulative_target_damage",
        "cumulative_defender_damage",
        "cumulative_attacker_damage",
        "red_action_cost",
        "blue_action_cost",
    )
    direction = {
        "payoff_red": "higher_is_better_for_Red",
        "target_final_health_fraction": "higher_is_better_for_Red",
        "target_min_health_fraction": "higher_is_better_for_Red",
        "defender_survivors": "higher_is_better_for_Red",
        "attacker_survivors": "lower_is_better_for_Red",
        "defender_casualties": "lower_is_better_for_Red",
        "attacker_casualties": "higher_is_better_for_Red",
        "minimum_threat_distance": "higher_is_better_for_Red",
        "cumulative_target_damage": "lower_is_better_for_Red",
        "cumulative_defender_damage": "lower_is_better_for_Red",
        "cumulative_attacker_damage": "higher_is_better_for_Red",
        "red_action_cost": "lower_is_better_at_equal_outcome",
        "blue_action_cost": "diagnostic",
    }
    result: Dict[str, object] = {}
    for field in fields:
        values = np.asarray(
            [float(getattr(record, field)) for record in records if getattr(record, field) is not None],
            np.float64,
        )
        result[field] = {
            "available": int(len(values)),
            "missing": int(len(records) - len(values)),
            "direction": direction[field],
            "mean": float(values.mean()) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "p10": float(np.quantile(values, 0.10)) if len(values) else None,
            "p90": float(np.quantile(values, 0.90)) if len(values) else None,
        }
        if field == "payoff_red":
            result[field]["semantics"] = (
                "local_window_safety_utility: +1 iff the asset survives the "
                "observation window, including administrative right-censoring; "
                "not an uncensored eventual-game payoff"
            )
    losses = np.asarray(
        [-float(record.payoff_red) for record in records if record.payoff_red is not None],
        np.float64,
    )
    if len(losses):
        threshold = float(np.quantile(losses, 0.90))
        result["red_payoff_loss_cvar90"] = {
            "value": float(losses[losses >= threshold].mean()),
            "direction": "lower_is_better_for_Red",
            "tail_threshold": threshold,
            "semantics": "CVaR90 of local-window safety-utility loss",
        }
    result["breach_time_steps"] = {
        "mean": (
            float(np.mean([record.breach_steps for record in records if record.breach_steps is not None]))
            if any(record.breach_steps is not None for record in records)
            else None
        ),
        "direction": "higher_is_better_for_Red",
    }
    result["attackers_neutralized_time_steps"] = {
        "mean": (
            float(
                np.mean(
                    [
                        record.attackers_neutralized_steps
                        for record in records
                        if record.attackers_neutralized_steps is not None
                    ]
                )
            )
            if any(record.attackers_neutralized_steps is not None for record in records)
            else None
        ),
        "direction": "lower_is_better_for_Red",
    }
    return result


def _physical_reasonable_bounds(
    field: str, records: Sequence[Stage2Record]
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(len(records), -np.inf, dtype=np.float64)
    upper = np.full(len(records), np.inf, dtype=np.float64)
    if field != "payoff_red":
        lower[:] = 0.0
    if field in {"target_final_health_fraction", "target_min_health_fraction"}:
        upper[:] = 1.0
    elif field.startswith("defender_") and field.endswith(
        ("survivors", "casualties")
    ):
        upper = np.asarray([record.query.defender_count for record in records], np.float64)
    elif field.startswith("attacker_") and field.endswith(
        ("survivors", "casualties")
    ):
        upper = np.asarray([record.query.attacker_count for record in records], np.float64)
    return lower, upper


def _physical_prediction_metrics(
    records: Sequence[Stage2Record],
    predictions: Mapping[str, np.ndarray],
    train_medians: Optional[Mapping[str, float]],
    interval_lower: Optional[Mapping[str, np.ndarray]],
    interval_upper: Optional[Mapping[str, np.ndarray]],
    ensemble_std: Optional[Mapping[str, np.ndarray]],
) -> Dict[str, object]:
    """Evaluate every learned physical head without fitting on test labels."""

    result: Dict[str, object] = {
        "normalization": "target z-score fitted on train split only",
        "ensemble_interval_semantics": "bootstrap_member_quantiles_not_formal_coverage",
        "derived_redundant_targets": dict(PHYSICAL_REDUNDANT_RELATIONS),
        "targets": {},
    }
    for field in PHYSICAL_TARGET_FIELDS:
        values = [getattr(record, field) for record in records]
        available = np.asarray([value is not None for value in values], dtype=bool)
        target_result: Dict[str, object] = {
            "available_test_labels": int(available.sum()),
            "missing_test_labels": int((~available).sum()),
            "derived_redundant": field in PHYSICAL_REDUNDANT_RELATIONS,
        }
        if field not in predictions or not available.any():
            target_result["status"] = "prediction_or_label_unavailable"
            result["targets"][field] = target_result
            continue
        truth = np.asarray(
            [float(value) if value is not None else np.nan for value in values], np.float64
        )
        point = np.asarray(predictions[field], np.float64)
        if point.shape != (len(records),):
            raise ValueError(f"physical prediction {field} has the wrong shape")
        selected = available & np.isfinite(point)
        if not selected.any():
            target_result["status"] = "no_finite_predictions"
            result["targets"][field] = target_result
            continue
        error = point[selected] - truth[selected]
        mae = float(np.mean(np.abs(error)))
        rmse = float(np.sqrt(np.mean(error**2)))
        denominator = float(np.sum((truth[selected] - truth[selected].mean()) ** 2))
        r2 = float(1.0 - np.sum(error**2) / denominator) if denominator > 0.0 else None
        median = float(train_medians[field]) if train_medians and field in train_medians else None
        baseline_mae = None
        baseline_rmse = None
        if median is not None:
            baseline_error = median - truth[selected]
            baseline_mae = float(np.mean(np.abs(baseline_error)))
            baseline_rmse = float(np.sqrt(np.mean(baseline_error**2)))
        lower_bound, upper_bound = _physical_reasonable_bounds(field, records)
        target_result.update(
            {
                "status": "evaluated",
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "train_median_baseline": median,
                "train_median_baseline_mae": baseline_mae,
                "train_median_baseline_rmse": baseline_rmse,
                "mae_improvement_over_train_median": (
                    float(baseline_mae - mae) if baseline_mae is not None else None
                ),
                "mae_skill_over_train_median": (
                    float(1.0 - mae / baseline_mae)
                    if baseline_mae is not None and baseline_mae > 0.0
                    else None
                ),
                "rmse_skill_over_train_median": (
                    float(1.0 - rmse / baseline_rmse)
                    if baseline_rmse is not None and baseline_rmse > 0.0
                    else None
                ),
                "reasonable_range": {
                    "lower_rule": "unbounded" if np.all(np.isneginf(lower_bound)) else "0",
                    "upper_rule": (
                        "1"
                        if field in {"target_final_health_fraction", "target_min_health_fraction"}
                        else "initial_roster"
                        if field in {
                            "defender_survivors",
                            "defender_casualties",
                            "attacker_survivors",
                            "attacker_casualties",
                        }
                        else "unbounded"
                    ),
                    "prediction_inside_rate": float(
                        np.mean(
                            (point[selected] >= lower_bound[selected])
                            & (point[selected] <= upper_bound[selected])
                        )
                    ),
                },
            }
        )
        if interval_lower and interval_upper and field in interval_lower and field in interval_upper:
            lower = np.asarray(interval_lower[field], np.float64)[selected]
            upper = np.asarray(interval_upper[field], np.float64)[selected]
            target_result["ensemble_interval"] = {
                "empirical_coverage": float(
                    np.mean((truth[selected] >= lower) & (truth[selected] <= upper))
                ),
                "mean_width": float(np.mean(upper - lower)),
                "semantics": "bootstrap_member_quantiles_not_formal_prediction_interval",
            }
        if ensemble_std and field in ensemble_std:
            target_result["mean_ensemble_std"] = float(
                np.mean(np.asarray(ensemble_std[field], np.float64)[selected])
            )
        result["targets"][field] = target_result

    consistency: Dict[str, object] = {}
    for side in ("defender", "attacker"):
        survivors = predictions.get(f"{side}_survivors")
        casualties = predictions.get(f"{side}_casualties")
        if survivors is not None and casualties is not None:
            roster = np.asarray(
                [
                    record.query.defender_count
                    if side == "defender"
                    else record.query.attacker_count
                    for record in records
                ],
                np.float64,
            )
            residual = np.asarray(survivors) + np.asarray(casualties) - roster
            consistency[f"{side}_survivors_plus_casualties"] = {
                "identity": f"{side}_survivors + {side}_casualties = initial_{side}_count",
                "mean_absolute_prediction_residual": float(np.mean(np.abs(residual))),
            }
    result["redundancy_consistency"] = consistency
    return result


def _basic_subset(
    records: Sequence[Stage2Record], views: Mapping[str, np.ndarray], selected: np.ndarray
) -> Dict[str, object]:
    breach = np.asarray([record.outcome == "breach" for record in records], np.int64)
    early = np.asarray([record.short_window_breach for record in records], np.int64)
    subset_records = [record for record, keep in zip(records, selected) if keep]
    return {
        "count": int(selected.sum()),
        "breach": _binary_metrics(breach[selected], np.asarray(views["breach"])[selected]),
        "short_window_breach": _binary_metrics(early[selected], np.asarray(views["early_breach"])[selected]),
        "remaining_time": _time_metrics(
            subset_records,
            np.asarray(views["restricted_mean_steps"])[selected],
            None,
            np.asarray(views["survival_at_censor"])[selected],
            np.asarray(views["conditional_event_median_steps"])[selected],
        ),
        "physical_outcomes": _physical_outcome_metrics(subset_records),
    }


def _cluster_bootstrap(
    records: Sequence[Stage2Record],
    breach_probability: np.ndarray,
    predicted_steps: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> Dict[str, object]:
    if samples < 1:
        return {"samples": 0}
    group_to_indices: Dict[str, list] = {}
    for index, record in enumerate(records):
        group_to_indices.setdefault(record.query.lineage_group_id, []).append(index)
    groups = np.asarray(sorted(group_to_indices), dtype=object)
    rng = np.random.default_rng(seed)
    breach_truth = np.asarray([record.outcome == "breach" for record in records], np.float64)
    observed = np.asarray([record.event_observed for record in records], bool)
    terminal = np.asarray([record.terminal_steps for record in records], np.float64)
    # Pre-compute one Red-choice regret per root; bootstrap then samples whole
    # lineage clusters so sibling roots/counterfactual rows stay together.
    root_cells: Dict[Tuple[str, str], list] = {}
    root_lineage: Dict[str, str] = {}
    for index, record in enumerate(records):
        root_cells.setdefault((record.query.root_id, record.query.candidate_id), []).append(index)
        root_lineage[record.query.root_id] = record.query.lineage_group_id
    root_regret: Dict[str, float] = {}
    for root_id in sorted({key[0] for key in root_cells}):
        candidates = []
        for (cell_root, candidate), indices in root_cells.items():
            if cell_root == root_id:
                candidates.append(
                    (
                        candidate,
                        float(np.mean(breach_probability[indices])),
                        float(np.mean(breach_truth[indices])),
                    )
                )
        selected = min(candidates, key=lambda row: (row[1], row[0]))
        oracle = min(candidates, key=lambda row: (row[2], row[0]))
        root_regret[root_id] = selected[2] - oracle[2]
    lineage_regrets: Dict[str, list] = {}
    for root_id, regret in root_regret.items():
        lineage_regrets.setdefault(root_lineage[root_id], []).append(regret)
    values = {
        "breach_brier": [],
        "event_time_mae_steps": [],
        "point_red_decision_regret": [],
    }
    for _ in range(samples):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([np.asarray(group_to_indices[str(group)], int) for group in sampled])
        values["breach_brier"].append(float(np.mean((breach_probability[indices] - breach_truth[indices]) ** 2)))
        event_indices = indices[observed[indices]]
        if len(event_indices):
            values["event_time_mae_steps"].append(
                float(np.mean(np.abs(predicted_steps[event_indices] - terminal[event_indices])))
            )
        sampled_regrets = [
            regret
            for group in sampled
            for regret in lineage_regrets.get(str(group), [])
        ]
        if sampled_regrets:
            values["point_red_decision_regret"].append(float(np.mean(sampled_regrets)))
    return {
        "unit": "lineage_group_id",
        "samples": samples,
        "seed": seed,
        **{
            name: {
                "lower": float(np.quantile(rows, 0.025)),
                "median": float(np.quantile(rows, 0.5)),
                "upper": float(np.quantile(rows, 0.975)),
            }
            for name, rows in values.items()
            if rows
        },
    }


def evaluate_stage2_predictions(
    records: Sequence[Stage2Record],
    probabilities: np.ndarray,
    *,
    horizon_bins: int,
    steps_per_bin: int,
    breach_upper: Optional[np.ndarray] = None,
    member_probabilities: Optional[np.ndarray] = None,
    train_median_event_steps: Optional[float] = None,
    train_mean_terminal_steps: Optional[float] = None,
    train_breach_rate: Optional[float] = None,
    train_empirical_probabilities: Optional[np.ndarray] = None,
    train_candidate_breach_rates: Optional[Mapping[str, float]] = None,
    physical_predictions: Optional[Mapping[str, np.ndarray]] = None,
    physical_interval_lower: Optional[Mapping[str, np.ndarray]] = None,
    physical_interval_upper: Optional[Mapping[str, np.ndarray]] = None,
    physical_ensemble_std: Optional[Mapping[str, np.ndarray]] = None,
    train_physical_medians: Optional[Mapping[str, float]] = None,
    seen_policy_pairs: Optional[Set[Tuple[str, str]]] = None,
    bootstrap_samples: int = 200,
    bootstrap_seed: int = 20260831,
) -> Dict[str, object]:
    """Evaluate once on untouched test lineages; no threshold is fitted here."""

    if not records:
        raise ValueError("cannot evaluate an empty Stage-2 dataset")
    # Compatibility: old callers supplied a mean.  It is intentionally ignored
    # for formal MAE; the median is the risk-minimising constant predictor.
    del train_mean_terminal_steps
    views = _probability_views(records, probabilities, horizon_bins, steps_per_bin)
    breach = np.asarray([record.outcome == "breach" for record in records], np.int64)
    defender_win = np.asarray([record.outcome == "defender_win" for record in records], np.int64)
    censored = np.asarray([not record.event_observed for record in records], np.int64)
    horizon_known = np.asarray(
        [record.event_observed or record.terminal_steps == record.query.horizon_steps for record in records],
        bool,
    )
    survived_horizon = np.asarray(
        [
            (not record.event_observed)
            and record.terminal_steps == record.query.horizon_steps
            for record in records
        ],
        np.int64,
    )
    early = np.asarray([record.short_window_breach for record in records], np.int64)
    result: Dict[str, object] = {
        "sample_count": len(records),
        "root_count": len({record.query.root_id for record in records}),
        "lineage_group_count": len({record.query.lineage_group_id for record in records}),
        "protocol": {
            "candidate_is_red_only": True,
            "blue_is_threat_condition": True,
            "timeout_is_right_censored": True,
        },
        "class_balance": {
            "defender_win_rate": float(defender_win.mean()),
            "breach_rate": float(breach.mean()),
            "right_censored_rate": float(censored.mean()),
            "short_window_breach_rate": float(early.mean()),
        },
        "joint_outcome_time": _right_censored_joint_metrics(
            records, views["probabilities"], horizon_bins, steps_per_bin
        ),
        "defender_win": _binary_metrics(defender_win, views["defender_win"]),
        "breach": _binary_metrics(breach, views["breach"]),
        "survival_through_horizon": (
            _binary_metrics(
                survived_horizon[horizon_known], views["survival"][horizon_known]
            )
            if horizon_known.any()
            else None
        ),
        "early_censored_rows_excluded_from_horizon_survival_metric": int(
            (~horizon_known).sum()
        ),
        "short_window_breach": _binary_metrics(early, views["early_breach"]),
        "remaining_time": _time_metrics(
            records,
            views["restricted_mean_steps"],
            train_median_event_steps,
            views["survival_at_censor"],
            views["conditional_event_median_steps"],
        ),
        "physical_outcomes": _physical_outcome_metrics(records),
    }
    if physical_predictions is not None:
        result["physical_evaluators"] = _physical_prediction_metrics(
            records,
            physical_predictions,
            train_physical_medians,
            physical_interval_lower,
            physical_interval_upper,
            physical_ensemble_std,
        )
    if train_breach_rate is not None:
        climatology = float(np.mean((breach - float(train_breach_rate)) ** 2))
        result["breach"]["climatology_probability_from_train"] = float(
            train_breach_rate
        )
        result["breach"]["climatology_brier"] = climatology
        result["breach"]["climatology_brier_skill"] = (
            float(1.0 - result["breach"]["brier"] / climatology)
            if climatology > 0.0
            else None
        )
    if train_empirical_probabilities is not None:
        empirical = np.asarray(train_empirical_probabilities, np.float64)
        if empirical.shape != (2 * horizon_bins + 1,):
            raise ValueError("train empirical joint baseline has the wrong dimension")
        empirical = empirical / empirical.sum()
        result["empirical_joint_train_frequency_baseline"] = (
            _right_censored_joint_metrics(
                records,
                np.tile(empirical, (len(records), 1)),
                horizon_bins,
                steps_per_bin,
            )
        )
    if member_probabilities is not None:
        result["ensemble_disagreement"] = _uncertainty_metrics(
            member_probabilities, horizon_bins
        )
    if breach_upper is not None:
        risk, decision = _risk_and_red_decision_metrics(
            records,
            views["breach"],
            np.asarray(breach_upper),
            train_candidate_breach_rates,
        )
        result["calibrated_risk_bound"] = risk
        result["red_candidate_decision"] = decision
        result["candidate_decision"] = decision  # compatibility alias
    result["lineage_cluster_bootstrap95"] = _cluster_bootstrap(
        records,
        views["breach"],
        views["conditional_event_median_steps"],
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    by_scale: Dict[str, object] = {}
    for scale in sorted({(record.query.defender_count, record.query.attacker_count) for record in records}):
        selected = np.asarray(
            [(record.query.defender_count, record.query.attacker_count) == scale for record in records],
            bool,
        )
        by_scale[f"{scale[0]}v{scale[1]}"] = _basic_subset(records, views, selected)
    result["by_scale"] = by_scale
    scale_time_errors = {
        name: group["remaining_time"].get("conditional_event_time_mae_steps")
        for name, group in by_scale.items()
    }
    available_time_errors = [
        value for value in scale_time_errors.values() if value is not None
    ]
    result["worst_scale_summary"] = {
        "max_breach_brier": float(max(group["breach"]["brier"] for group in by_scale.values())),
        "max_conditional_event_time_mae_steps": (
            float(max(available_time_errors)) if available_time_errors else None
        ),
        "scales_without_observed_event_time_metric": sorted(
            name for name, value in scale_time_errors.items() if value is None
        ),
    }
    by_threat: Dict[str, object] = {}
    for threat in sorted({record.query.threat_id for record in records}):
        selected = np.asarray([record.query.threat_id == threat for record in records], bool)
        by_threat[threat] = _basic_subset(records, views, selected)
    result["by_blue_threat"] = by_threat
    by_candidate: Dict[str, object] = {}
    for candidate in sorted({record.query.candidate_id for record in records}):
        selected = np.asarray([record.query.candidate_id == candidate for record in records], bool)
        by_candidate[candidate] = _basic_subset(records, views, selected)
    result["by_red_candidate"] = by_candidate
    if seen_policy_pairs is not None:
        seen = np.asarray([record.query.policy_pair in seen_policy_pairs for record in records])
        generalization: Dict[str, object] = {}
        if seen.any():
            generalization["seen_policy_pair"] = _basic_subset(records, views, seen)
        if (~seen).any():
            generalization["held_out_policy_pair"] = _basic_subset(records, views, ~seen)
        result["policy_pair_generalization"] = generalization
    return result


__all__ = ["evaluate_stage2_predictions", "metric_definitions_zh"]
