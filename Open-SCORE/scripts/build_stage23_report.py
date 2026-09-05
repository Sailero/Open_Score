"""Build the reader-facing report for the registered identity CS-BBG v5 pipeline."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Mapping


STAGE3_SCHEMA = "open-score-stage3-identity-cs-bbg-results-v5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected one JSON object in {path}")
    return value


def _average_target_rate(
    table: Mapping[str, object], commander: str, target: str
) -> float | None:
    by_style = table.get(commander)
    if not isinstance(by_style, Mapping):
        return None
    values = [
        float(cells[target]["win_rate"])
        for cells in by_style.values()
        if isinstance(cells, Mapping)
        and target in cells
        and isinstance(cells[target], Mapping)
    ]
    return None if not values else sum(values) / len(values)


def _average_cell_metric(
    table: Mapping[str, object], commander: str, metric: str
) -> float | None:
    by_style = table.get(commander)
    if not isinstance(by_style, Mapping):
        return None
    weighted_sum = 0.0
    episodes = 0
    for cells in by_style.values():
        if not isinstance(cells, Mapping):
            continue
        for cell in cells.values():
            if not isinstance(cell, Mapping) or cell.get(metric) is None:
                continue
            count = int(cell.get("episodes", 1))
            weighted_sum += float(cell[metric]) * count
            episodes += count
    return None if episodes == 0 else weighted_sum / episodes


def build_report(
    root: Path,
    stage2: Mapping[str, object],
    stage3: Mapping[str, object],
) -> Path:
    """Build the concise v5 report with paired and historical comparisons."""

    if stage3.get("schema_version") != STAGE3_SCHEMA:
        raise ValueError("Stage3 result is not identity CS-BBG v5")
    core = stage2["group_metrics"]["core_test"]
    exact = stage3["exact_validation"]
    physical = stage3.get("physical_evaluation")
    table = (
        physical.get("win_rate_by_commander_style_targets", {})
        if isinstance(physical, Mapping)
        else {}
    )
    lines = [
        "# Stage2–Stage3 身份级 4v4 联合实验报告",
        "",
        f"> 自动生成：`{datetime.now().astimezone().isoformat(timespec='seconds')}`  ",
        f"> Stage2：**{'通过' if stage2.get('acceptance', {}).get('all_passed') else '未通过'}**  ",
        f"> Stage3：**{'通过' if stage3.get('acceptance', {}).get('all_passed') else '未通过'}**",
        "",
        "## 本轮改变",
        "",
        "Stage2 的合法域收缩为完整的 1–4×1–4，并对 16 个初始配比各自直接采样；固定指挥点和伤亡点均重置冻结 Stage1 控制器。Stage3 直接输出每个 `(目标, 通道)` 中的 Red IDs 与 Blue IDs。Blue 身份必须恰好覆盖一次；每个 Red 身份则恰好位于一个 1–4 人活跃组或 reserve，不能重复。",
        "",
        "## Stage2 关键指标",
        "",
        "| Brier↓ | ECE↓ | AUC↑ | 时间MAE↓ | 16配比胜率误差↓ |",
        "|---:|---:|---:|---:|---:|",
        f"| {float(core['brier']):.4f} | {float(core['ece_10']):.4f} | {float(core['auc']):.4f} | {float(core['time_mae_steps']):.2f}步 | {float(stage2['core_mean_scale_win_rate_absolute_error']):.2%} |",
        "",
        "## Stage3 求解与规模",
        "",
        f"完整小博弈最大价值误差为 `{float(exact['max_value_error']):.8f}`，最大 exploitability 为 `{float(exact['max_exploitability']):.8f}`，全博弈证书：`{bool(exact['all_full_game_certified'])}`。",
        "",
        "| 总人数 | P95规划时间(s) | 候选域证书率 | 最大归一化候选域gap |",
        "|---:|---:|---:|---:|",
    ]
    if (root / "reused_stage2_validation.json").is_file():
        lines.extend(
            [
                "",
                "本次恢复运行已对 Stage2 的 accepted 状态、配置语义、调度、数据集、checkpoint 和训练配置哈希逐项复核；未重新采集或训练 Stage2。",
            ]
        )
    for scale, row in stage3["planner_scaling"].items():
        lines.append(
            f"| {scale} | {float(row['planning_p95_seconds']):.3f} | "
            f"{float(row['candidate_domain_certified_fraction']):.1%} | "
            f"{float(row['exploitability_max']):.5f} |"
        )
    lines.extend(
        [
            "",
            "30–50 人采用显式活动通道和候选联盟列。代理 payoff 除以其解析绝对上界 `M·|B|`；这是正比例缩放，不改变最佳响应或均衡，只让容差和 gap 可跨规模解释。表中证书与 gap 只针对该受限候选域；若 MILP 限时，gap 是当前可行偏离诊断，不能冒充完整指数动作空间的全局证书。",
            "",
            "## 物理胜率与关键诊断",
            "",
        ]
    )
    if isinstance(physical, Mapping):
        lines.extend(
            [
                "| 方法（两种Blue底层取平均） | 2目标 | 4目标 | 5目标 |",
                "|---|---:|---:|---:|",
            ]
        )
        for commander in (
            "balanced_identity",
            "identity_blotto",
            "revealed_blue_br",
        ):
            cells = []
            for target in ("2", "4", "5"):
                rate = _average_target_rate(table, commander, target)
                cells.append("—" if rate is None else f"{rate:.1%}")
            lines.append(f"| {commander} | {' | '.join(cells)} |")
        paired = physical["identity_vs_balanced_paired"]
        revealed = physical["revealed_vs_identity_paired"]
        lines.extend(
            [
                "",
                f"主要内部对照是相同场景、Blue底层和初始种子的配对差：`identity_blotto - balanced_identity = {float(paired['mean_improvement']):.1%}`，95% bootstrap CI 为 `{float(paired['ci95_low']):.1%}` 至 `{float(paired['ci95_high']):.1%}`（{int(paired['pairs'])} 对）。",
                "",
                f"观察同一 Blue 均衡动作后，Red 在相同候选域内用 1–4 人组与 reserve 求最佳响应；其相对身份级 Blotto 的物理胜率差为 `{float(revealed['mean_improvement']):.1%}`。`revealed_blue_br` 只在当前局部价值代理和注册候选域中构成信息上界，不是 HAD 真实胜率的理论上界，也不是可部署主方法。",
                "",
                "| 方法 | Red reserve比例 | 无Red对应的Blue组比例 |",
                "|---|---:|---:|",
            ]
        )
        for commander in (
            "balanced_identity",
            "identity_blotto",
            "revealed_blue_br",
        ):
            reserve = _average_cell_metric(table, commander, "reserve_fraction")
            unopposed = _average_cell_metric(
                table, commander, "unopposed_blue_group_fraction"
            )
            reserve_text = "—" if reserve is None else f"{reserve:.1%}"
            unopposed_text = "—" if unopposed is None else f"{unopposed:.1%}"
            lines.append(f"| {commander} | {reserve_text} | {unopposed_text} |")
        lines.extend(
            [
                "",
                "reserve 的即时代理价值固定为0，不送入 Stage2；执行时仅依据公开的存活位置和固定目标进行威胁巡逻，并在下一次重规划重新参与分配。Blue 人数无法被 Red 活跃组覆盖时，允许合法出现 `0v1`–`0v4` 槽位。",
            ]
        )
    else:
        lines.append("本次跳过了物理评估。")
    historical = stage3.get("historical_comparison", {})
    lines.extend(["", "## 与上一版的关系", ""])
    if historical.get("available"):
        previous = historical.get("previous_double_oracle") or {}
        cells = [
            "—"
            if str(target) not in previous
            else f"{float(previous[str(target)]['win_rate']):.1%}"
            for target in (2, 4, 5)
        ]
        lines.extend(
            [
                "上一版匿名/count-based Double Oracle 作为历史量级参照：",
                "",
                "| 协议 | 2目标 | 4目标 | 5目标 |",
                "|---|---:|---:|---:|",
                f"| 上一版匿名Double Oracle | {' | '.join(cells)} |",
                "",
                "该历史表不是配对消融：Stage2 样本域、动作表示和对手上层协议均已改变，因此不能把数值差直接归因于身份建模。改进判断以本轮配对结果、4v4/身份约束检查和小博弈求解正确性共同决定。",
            ]
        )
    else:
        lines.append("上一版机器可读结果不可用；仅报告本轮内部配对。")
    lines.extend(
        [
            "",
            "## 验收结论",
            "",
            str(stage3["decision"]),
            "",
            "## 查看位置",
            "",
            "- `pipeline.log`：完整实时进度。",
            "- `stage2/stage2_report.md`：Stage2 数据、训练和校准。",
            "- `stage3/stage3_report.md`：身份级 Stage3 结果。",
            "- `stage3/raw/physical_identity_episodes.csv`：逐局和逐次实名分组。",
            "- `stage3/analysis/summary.json`：机器可读验收与新旧对比。",
            "",
        ]
    )
    target = root / "stage23_report.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def main() -> None:
    args = parse_args()
    root = args.run_root.resolve()
    target = build_report(
        root,
        read_json(root / "stage2/model/metrics.json"),
        read_json(root / "stage3/analysis/summary.json"),
    )
    print(f"[报告] ID-CS-BBG 联合报告已生成：{target}", flush=True)


if __name__ == "__main__":
    main()
