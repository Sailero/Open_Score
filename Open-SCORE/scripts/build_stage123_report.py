"""One report for frozen stages and explicitly labelled protocol comparisons."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "outputs/stage123_unknown_upper_v1"


def build(root=ROOT):
    def read(name):
        return json.loads((root / name).read_text(encoding="utf8"))
    def pct(value):
        return f"{100 * value:.1f}%"
    s1 = read("stage1/summary.json")
    s2 = read("stage2/model/metrics.json")
    lines = ["# Stage1–3 综合实验报告", "",
             "当前证据由原始数据及 SHA256 清单生成。旧实验与未知上层新协议分别解释；尚未运行的结果不填入估计值。",
             "", "## Stage1：冻结局部控制器", "",
             f"REFIL-QMIX-HAD adaptation，单训练种子 `{s1['seed']}`，{s1['environment_steps']:,} 环境步。"
             f"验证胜率 {pct(s1['initial_validation']['win_rate'])} → {pct(s1['best_validation']['win_rate'])}；"
             f"独立测试 {s1['heldout_evaluation']['episodes']:,} 局，总胜率 {pct(s1['heldout_evaluation']['win_rate'])}。",
             "", "| 局部规模 | rush | split_rush |", "|---|---:|---:|"]
    components = s1["heldout_evaluation"]["components"]
    for scale in components["rush"]["per_scale_win_rate"]:
        lines.append(f"| {scale} | {pct(components['rush']['per_scale_win_rate'][scale])} | {pct(components['split_rush']['per_scale_win_rate'][scale])} |")
    lines += ["", "训练外规模的事后压力测试（没有微调）：", "",
              "| 规模 | rush | split_rush | 合并胜率 |", "|---|---:|---:|---:|"]
    for row in read("stage1/generalization_summary.json")["scale_results"]:
        lines.append(f"| {row['scale']} | {pct(row['win_rate_rush'])} | {pct(row['win_rate_split_rush'])} | {pct(row['combined_win_rate'])} |")
    lines += ["", "5v2/5v3/6v3 仅为历史泛化证据，不进入本轮 4v4 上层动作域。"
              "现存 Stage1 没有同协议其他算法对照，因此只报告内部训练与规模分析，不虚构算法优势。原已删除的泛化原始证据从 Git 历史恢复，来源见 manifest。",
              "", "## Stage2：冻结的联合胜负与时间模型", "",
              "正式数据共 19,200 局、101,366 个状态，覆盖全部 1–4 × 1–4 配比；按局划分，保留本地 JSONL，不重新采集或训练 Stage2。",
              "", "| 方法/测试切片 | Brier↓ | ECE↓ | 时间 MAE（步）↓ |", "|---|---:|---:|---:|"]
    for key, label in [("core_test", "联合模型：全部状态"), ("core_initial_test", "联合模型：初始状态"), ("core_trajectory_test", "联合模型：轨迹状态")]:
        row = s2["group_metrics"][key]
        lines.append(f"| {label} | {row['brier']:.4f} | {row['ece_10']:.4f} | {row['time_mae_steps']:.2f} |")
    lines.append(f"| 训练集常数胜率 / 中位时间基线 | {s2['baselines']['core_constant_brier']:.4f} | — | {s2['baselines']['core_median_time_mae_steps']:.2f} |")
    lines += ["", "Stage2 的概率与时间误差是本阶段的必要指标，不受 Stage3 正文四类指标限制。总体校准不能替代困难配比或新分组反事实上的校准。",
              "", "| 困难单元（按 Brier 降序） | Brier↓ | ECE↓ | 时间 MAE↓ |", "|---|---:|---:|---:|"]
    for scale, row in sorted(s2["per_scale"].items(), key=lambda item: -item[1]["brier"])[:4]:
        lines.append(f"| {scale} | {row['brier']:.4f} | {row['ece_10']:.4f} | {row['time_mae_steps']:.2f} |")
    lines += ["", "![Stage2 训练曲线](figures/stage2_01_training_curves.png)", "",
              "![Stage2 校准](figures/stage2_04_core_calibration.png)", "",
              "## Stage3：历史证据与局限", "",
              "### 原 identity 协议（180 局）", "",
              "Blue 每次事件从均衡重新采样；这里没有整局隐藏上层策略。revealed 方法看到当前完整动作，不能代表已知策略但动作隐藏的 oracle。",
              "", "| Red 方法 | rush | split_rush | 总胜率 |", "|---|---:|---:|---:|"]
    old = read("stage3/baselines/legacy_idb/analysis/summary.json")
    for method, styles in old["physical_evaluation"]["win_rate_by_commander_style_targets"].items():
        means = [sum(row["episodes"] * row["win_rate"] for row in styles[style].values()) / sum(row["episodes"] for row in styles[style].values()) for style in ["rush", "split_rush"]]
        lines.append(f"| {method} | {pct(means[0])} | {pct(means[1])} | {pct(sum(means)/2)} |")
    effect = old["physical_evaluation"]["identity_vs_balanced_paired"]
    lines += ["", f"identity − balanced：**{pct(effect['mean_improvement'])}**，配对 95% CI [{pct(effect['ci95_low'])}, {pct(effect['ci95_high'])}]。负结果保留；原验收未通过。",
              "", "| 总人数 | P95 规划秒数 |", "|---|---:|"]
    for size, row in old["planner_scaling"].items():
        value = row.get("planning_seconds_p95", row.get("p95_planning_seconds"))
        if value is None:
            value = next((v for k, v in row.items() if "p95" in k), None)
        lines.append(f"| {size} | {value:.3f} |" if value is not None else f"| {size} | 见原始 JSON |")
    sal = read("stage3/baselines/saldae/analysis/summary.json")
    effect = sal["physical_evaluation"]["saldae_vs_milp_paired"]
    lines += ["", "### SALDAE 求解器消融（60 局，30 对）", "",
              f"SALDAE − MILP 胜率差 {pct(effect['mean_improvement'])}，95% CI [{pct(effect['ci95_low'])}, {pct(effect['ci95_high'])}]。"
              "置信区间包含 0；只支持发现候选域外响应、物理胜率点估计上升，不支持统计显著改善。",
              "", "| 求解器 | rush | split_rush | 总胜率 |", "|---|---:|---:|---:|"]
    for method, styles in sal["physical_evaluation"]["win_rate_by_method_style_targets"].items():
        means = [sum(row["win_rate"] for row in styles[style].values()) / len(styles[style]) for style in ["rush", "split_rush"]]
        lines.append(f"| {method} | {pct(means[0])} | {pct(means[1])} | {pct(sum(means)/2)} |")
    lines += ["", "### 匿名人数分配 Bayesian v3（历史协议）", "",
              "此协议的对手、动作、局部上限与 Stage2 均不同，不与 identity 或新协议合并估计。以下是各自历史单元按局数加权的胜率。", "",
              "| 人数分配方法 | rush | split_rush | 总胜率 |", "|---|---:|---:|---:|"]
    with (root / "stage3/baselines/count_bayesian_v3/analysis/physical_summary.csv").open(encoding="utf8") as stream:
        records = list(csv.DictReader(stream))
    for method in sorted({r["red_commander"] for r in records}):
        rates = []
        for lower in ["rush", "split_rush"]:
            subset = [r for r in records if r["red_commander"] == method and r["blue_lower_style"] == lower]
            rates.append(sum(int(r["episodes"])*float(r["red_win_rate"]) for r in subset)/sum(int(r["episodes"]) for r in subset))
        subset = [r for r in records if r["red_commander"] == method]
        overall = sum(int(r["episodes"])*float(r["red_win_rate"]) for r in subset)/sum(int(r["episodes"]) for r in subset)
        lines.append(f"| {method} | {pct(rates[0])} | {pct(rates[1])} | {pct(overall)} |")
    lines += ["", "## Stage3：未知上层策略的统一协议", "",
              "论文主问题是未知上层意图下的长期身份分组，以及相对已知上层 oracle 的差距。Stage1/2 是全方法共用的冻结实验条件，不作为本轮新增贡献。", ""]
    result_path = root / "stage3/unknown_upper/summary.json"
    ready = False
    if result_path.exists():
        new = read("stage3/unknown_upper/summary.json")
        lines.extend(new["report_lines"])
        ready = bool(new["all_passed"])
        lines += ["", "![未知上层的同协议胜率](figures/unknown_upper_win_rates.png)"]
    else:
        lines += ["**正式未知策略评估尚未完成。** 不能据旧 identity、SALDAE 或单元测试推断 BA-DIB 接近已知上层 oracle。",
                  "", "注册比较：balanced_identity、legacy_idb、event_risk_idb、finite_type_bbr、qom_bbr、qom_mcp、known_upper_mcp、revealed_blue_action_br。",
                  "", "3 个规模 × 2 个已知底层 × 4 个整局固定的隐藏上层 × 8 个 Red 方法 × 10 个共同种子 = 1,920 局。"
                  "另注册 finite_type_mcp、prior_mcp 两个同预算机制消融，共 480 局；物理评估共 2,400 局。"
                  "不包含局中策略类型切换或开放集检测，扩展中间诊断默认关闭。正文报告胜率、oracle gap、分配推断、规划时间；其他记录保存为 JSON/CSV。"]
    lines += ["", "## 失败归因与 Stage4 决策", "",
              ("本轮预注册效果门槛已通过，可评估后续扩展；论文投稿前仍需跨训练种子及更强候选域 oracle 检查。" if ready else
               "目前不能判定未知上层目标达成，不以现有证据直接推进 Stage4。需分别检查公开轨迹识别、候选覆盖、5 步代理排序和长期规划。")+
              "信息 oracle 是同预算经验参照，有限搜索的物理胜率不保证单调；不宣称完整指数动作空间的全局 Nash 证书。",
              "", "可复核入口：`manifest.json`、`stage1/`、`stage2/model/metrics.json`、`stage3/baselines/`、"
              "`stage3/unknown_upper/`、`pipeline.log`、`status.json`。旧报告文字按原哈希保存在 `stage3/baselines/source_reports.json`。", ""]
    (root / "experiment_report.md").write_text("\n".join(lines), encoding="utf8")
    print(root / "experiment_report.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    build(parser.parse_args().root)
