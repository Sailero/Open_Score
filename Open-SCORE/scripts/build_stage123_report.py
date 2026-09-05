"""One report for frozen stages and explicitly labelled protocol comparisons."""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "outputs/stage123_unknown_upper_v1"

OLD_NAMES = {
    "balanced_identity":"均分防守", "legacy_idb":"旧版分组", "event_risk_idb":"短期稳健防守",
    "finite_type_bbr":"规则推断＋短期防守", "qom_bbr":"学习推断＋短期防守",
    "qom_mcp":"学习推断＋多步规划（旧主方法）", "known_upper_mcp":"知道策略的多步参照",
    "revealed_blue_action_br":"看见当前分组的短期参照",
}
HISTORICAL_NAMES = {**OLD_NAMES, "identity_blotto":"旧版身份分组", "revealed_blue_br":"读取当前完整分组的参照",
                    "milp_do":"混合整数求解器", "saldae_do":"联盟搜索改进求解器（SALDAE）",
                    "clairvoyant_count_oracle":"读取目标人数的参照", "double_oracle":"人数博弈求解",
                    "eta_greedy":"按到达时间贪心分配", "known_blue_1v1_2v1":"已知对手的固定小组规则",
                    "s2_greedy":"按局部预测贪心分配", "stage1_balanced":"均分人数并用底层执行"}


def stopped_evidence(root):
    """Only complete matched cells from the superseded run enter this table."""
    rows=[]
    for path in sorted((root/"stage3/unknown_upper/units").glob("*.json.gz")):
        with gzip.open(path,"rt",encoding="utf-8") as stream:
            envelope=json.load(stream)
        row=envelope["result"]
        raw=json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")
        if hashlib.sha256(raw).hexdigest()!=envelope["result_sha256"]:
            raise ValueError(f"Historical result hash mismatch: {path}")
        rows.append(row)
    grouped={}
    for row in rows:
        c=row["cell"]
        grouped.setdefault((c["scenario"]["label"],c["lower"],c["upper"]),[]).append(row)
    complete=[]
    for key, group in sorted(grouped.items()):
        seeds=[{r["cell"]["seed"] for r in group if r["cell"]["method"]==m} for m in OLD_NAMES]
        if len(group)==80 and all(len(s)==10 and s==seeds[0] for s in seeds):
            complete.append((key,group))
    lines=["### 已停止的学习推断方案：保留负结果", "",
           f"旧长时实验已停止，保存 {len(rows)} 个通过内容哈希检查的结果；其中 {len(complete)} 个条件完成了全部八方法的十个共同种子。"
           "下面只比较这些完整条件，不把未配齐的记录混入总胜率。这不是完整 30/40/50 人结论。", ""]
    for (scenario,lower,upper), group in complete:
        types={"balanced":"均衡攻击", "concentrated_nearest":"集中攻击最近有利目标"}
        lines += [f"条件：`{scenario}`，底层 `{lower}`，上层 {types.get(upper,upper)}。", "",
                  "| 方法 | 获胜局数 | 平均决策秒数 |", "|---|---:|---:|"]
        for method,name in OLD_NAMES.items():
            chosen=[r for r in group if r["cell"]["method"]==method]
            times=[e["planning_seconds"] for r in chosen for e in r["events"]]
            lines.append(f"| {name} | {sum(r['red_win'] for r in chosen)}/10 | {sum(times)/len(times):.2f} |")
    lines += ["", "这些局部结果尚未支持旧主方法接近知情参照，或稳定优于旧版分组。"
              "它的额外计算成本同样属于负结果。知情参照知道整局规则，但看不到当前随机动作；它仍是有限候选与有限计算下的经验参照。", ""]
    return lines


def build(root=ROOT):
    def read(name):
        return json.loads((root / name).read_text(encoding="utf-8-sig"))
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
        lines.append(f"| {HISTORICAL_NAMES.get(method,method)} | {pct(means[0])} | {pct(means[1])} | {pct(sum(means)/2)} |")
    effect = old["physical_evaluation"]["identity_vs_balanced_paired"]
    lines += ["", f"旧版身份分组 − 均分防守：**{100*effect['mean_improvement']:.1f} 个百分点**，配对 95% 区间 [{100*effect['ci95_low']:.1f}, {100*effect['ci95_high']:.1f}] 个百分点。负结果保留；原验收未通过。",
              "", "| 总人数 | P95 规划秒数 |", "|---|---:|"]
    for size, row in old["planner_scaling"].items():
        value = row.get("planning_seconds_p95", row.get("p95_planning_seconds"))
        if value is None:
            value = next((v for k, v in row.items() if "p95" in k), None)
        lines.append(f"| {size} | {value:.3f} |" if value is not None else f"| {size} | 见原始 JSON |")
    sal = read("stage3/baselines/saldae/analysis/summary.json")
    effect = sal["physical_evaluation"]["saldae_vs_milp_paired"]
    lines += ["", "### SALDAE 求解器消融（60 局，30 对）", "",
              f"联盟搜索改进 − 混合整数求解器：胜率差 {100*effect['mean_improvement']:.1f} 个百分点，95% 区间 [{100*effect['ci95_low']:.1f}, {100*effect['ci95_high']:.1f}] 个百分点。"
              "置信区间包含 0；只支持发现候选域外响应、物理胜率点估计上升，不支持统计显著改善。",
              "", "| 求解器 | rush | split_rush | 总胜率 |", "|---|---:|---:|---:|"]
    for method, styles in sal["physical_evaluation"]["win_rate_by_method_style_targets"].items():
        means = [sum(row["win_rate"] for row in styles[style].values()) / len(styles[style]) for style in ["rush", "split_rush"]]
        lines.append(f"| {HISTORICAL_NAMES.get(method,method)} | {pct(means[0])} | {pct(means[1])} | {pct(sum(means)/2)} |")
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
        lines.append(f"| {HISTORICAL_NAMES.get(method,method)} | {pct(rates[0])} | {pct(rates[1])} | {pct(overall)} |")
    lines += ["", "## Stage3：未知上层策略的统一协议", "",
              "论文主问题是未知上层意图下的长期身份分组，以及相对已知上层 oracle 的差距。Stage1/2 是全方法共用的冻结实验条件，不作为本轮新增贡献。", ""]
    result_path = root / "stage3/unknown_upper/summary.json"
    ready = False
    fast_path = root / "stage3/fast_compare_v2"
    if (root/"stage3/unknown_upper/superseded_run.json").exists():
        lines.extend(stopped_evidence(root))
        if (fast_path/"summary.json").exists():
            new=read("stage3/fast_compare_v2/summary.json")
            lines.extend(new["report_lines"])
            ready=new["all_identity_valid"] and any(all(g.values()) for g in new["acceptance"].values())
        else:
            lines += ["### 当前验证：两个轻量方案，共六种方法", "",
                      "**新的胜率结果尚未完成，不能把运行加速当作防守能力改善。**", "",
                      "均分防守是唯一简单规则基线；旧版分组在公开的快速求解预算下重跑。"
                      "方案一根据公开运动推断蓝方，再模拟短期结果；方案二再考虑下一次观察后的重新分组。"
                      "两种方案各有一个知道蓝方规则的参照，使用对应的候选规则与计算预算。", "",
                      "3 个规模 × 2 种已知底层 × 4 种整局固定上层 × 5 个共同种子 × 6 方法 = **720 局**。"
                      "每种方法都跑完同样的 120 个条件与种子。蓝方分配可随公开状态变化，但整局不切换规则类型。", "",
                      "这是重新登记的集中验证版本，替代旧长时实验；不将缩小后的 720 局表述为完成原 2,400 局验收。"
                      "两种新方案不训练新模型，Stage1/2 保持冻结。软决策预算为 2.5 秒，在计算段之间检查；"
                      "超时与退回初选动作均记录。启动后实际进度见 `status.json`。", "",
                      "判断顺序：先看是否比两个基线更能守住目标，再看与对应知情参照的差距，最后检查较慢决策时间。"
                      "胜率差用百分点表示；配对 95% 区间跨过零时不宣称稳定提升。两种蓝方底层分别报告。"]
    elif result_path.exists():
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
              "`stage3/unknown_upper/`、`stage3/fast_compare_v2/`、`pipeline.log`、`status.json`。旧报告文字按原哈希保存在 `stage3/baselines/source_reports.json`。", ""]
    (root / "experiment_report.md").write_text("\n".join(lines), encoding="utf8")
    print(root / "experiment_report.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    build(parser.parse_args().root)
