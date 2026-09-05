"""Paired, stratified evaluation; never invent rows for missing experiments."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from .planner import ABLATION_METHODS, METHODS
from .storage import atomic_json,read_unit


def paired_effect(rows,left,right,repeats=10000):
    lookup = {(r["cell"]["seed"],r["cell"]["method"]):r for r in rows}
    seeds = sorted({seed for seed,method in lookup if method==left})
    paired = [(lookup[s,left],lookup[s,right]) for s in seeds if (s,right) in lookup]
    if not paired:
        return {"pairs":0,"difference":None,"ci95":None,"p_value":None}
    groups = {}
    for a,b in paired:
        key=(a["cell"]["scenario"]["label"],a["cell"]["lower"],a["cell"]["upper"])
        groups.setdefault(key,[]).append(a["red_win"]-b["red_win"])
    rng = np.random.default_rng(3102026)
    draws = np.zeros(repeats)
    differences = []
    for values in groups.values():
        values=np.asarray(values,float);differences.extend(values)
        draws += values[rng.integers(len(values),size=(repeats,len(values)))].sum(-1)/len(paired)
    differences=np.asarray(differences)
    wins=int(np.sum(differences>0));losses=int(np.sum(differences<0))
    p=binomtest(wins,wins+losses,.5).pvalue if wins+losses else 1.0
    return {"pairs":len(paired),"difference":float(differences.mean()),"ci95":np.quantile(draws,[.025,.975]).tolist(),
            "p_value":float(p),"resampling":"paired episode seed within scenario/lower/upper strata"}


def inference_metrics(rows):
    selected=[r for r in rows if r["cell"]["method"]=="qom_mcp"]
    correct,total,eligible,forecast_correct,forecast_total=0,0,0,0,0
    oracle_ceiling_sum=0.0
    for row in selected:
        observations=[e for e in row["filtered_assignments"] if e["event_index"]>=1]
        if observations:
            eligible+=1
            for event in observations:
                correct+=sum(a==b for a,b in zip(event["truth_targets"],event["predicted_targets"]))
                total+=len(event["truth_targets"])
        for event in row["events"]:
            if event["event_index"]>=1:
                forecast_correct+=sum(a==b for a,b in zip(event["truth_targets"],event["predicted_targets"]))
                count=len(event["truth_targets"])
                forecast_total+=count
                oracle_ceiling_sum+=count*event["forecast_oracle_accuracy_ceiling"]
    return {"target_allocation_micro_f1":correct/total if total else None,"identity_predictions":total,
            "target_allocation_semantics":"current persistent action filtered from public transitions after the second command",
            "next_action_forecast_micro_f1":forecast_correct/forecast_total if forecast_total else None,
            "same_state_known_program_forecast_accuracy_ceiling":oracle_ceiling_sum/forecast_total if forecast_total else None,
            "closed_eligible_episodes":eligible,"closed_total_episodes":len(selected),
            "scope":"four episode-fixed upper types; no mid-episode type switches or open-set claims"}


def summarize(paths,config,root,diagnostics,ablation_paths=()):
    rows=[read_unit(p) for p in paths]
    if any(row is None for row in rows):
        raise ValueError("Formal result failed its content hash")
    if len(rows)!=config["acceptance"]["expected_episodes"] or len({r["cell"]["id"] for r in rows})!=len(rows):
        raise ValueError("Cannot report completion without all 1,920 distinct core units")
    ablations=[read_unit(p) for p in ablation_paths]
    if any(r is None for r in ablations) or len(ablations)!=config["paper_ablations"]["expected_episodes"]:
        raise ValueError("Cannot draw paper conclusions with missing matched controls")
    if len({r["cell"]["id"] for r in ablations})!=len(ablations):
        raise ValueError("Duplicate ablation units")
    core_episodes=len(rows)
    rows.extend(ablations)
    method_names=METHODS+ABLATION_METHODS
    rows.sort(key=lambda r:r["cell"]["id"])
    methods={}
    for method in method_names:
        subset=[r for r in rows if r["cell"]["method"]==method]
        times=[e["planning_seconds"] for r in subset for e in r["events"]]
        methods[method]={"episodes":len(subset),"win_rate":float(np.mean([r["red_win"] for r in subset])),
                         "win_rate_by_lower":{lower:float(np.mean([r["red_win"] for r in subset if r["cell"]["lower"]==lower])) for lower in config["lower_policies"]},
                         "planning_seconds_mean":float(np.mean(times)),"planning_seconds_p95":float(np.quantile(times,.95)),
                         "planning_events":len(times)}
    repeats=config["acceptance"]["bootstrap_replicates"]
    effects={m:paired_effect(rows,"qom_mcp",m,repeats) for m in ["known_upper_mcp","legacy_idb","event_risk_idb","qom_bbr","finite_type_bbr"]}
    controls={"learned_vs_finite_belief":paired_effect(rows,"qom_mcp","finite_type_mcp",repeats),
              "finite_belief_vs_prior":paired_effect(rows,"finite_type_mcp","prior_mcp",repeats),
              "finite_mcp_vs_bbr":paired_effect(rows,"finite_type_mcp","finite_type_bbr",repeats),
              "finite_mcp_vs_oracle":paired_effect(rows,"finite_type_mcp","known_upper_mcp",repeats)}
    lower_effects={lower:paired_effect([r for r in rows if r["cell"]["lower"]==lower],"qom_mcp","known_upper_mcp",repeats)
                   for lower in config["lower_policies"]}
    scale_effects={s["label"]:paired_effect([r for r in rows if r["cell"]["scenario"]["label"]==s["label"]],"qom_mcp","known_upper_mcp",repeats)
                   for s in config["scenarios"]}
    # Two pre-registered superiority comparisons: Holm family-wise correction.
    ordered=sorted(["legacy_idb","event_risk_idb"],key=lambda k:effects[k]["p_value"])
    previous=0.
    for rank,key in enumerate(ordered):
        previous=max(previous,min(1.,(2-rank)*effects[key]["p_value"]))
        effects[key]["holm_adjusted_p"]=previous
        effects[key]["significant_improvement"]=(effects[key]["ci95"][0]>0 and previous<.05)
    inference=inference_metrics(rows)
    cfg=config["acceptance"]
    oracle=effects["known_upper_mcp"]
    gates={"all_registered_completed":True,"identity_and_fixed_targets":all(r["identity_valid"] and r["fixed_targets"] and not r["infeasible"] for r in rows),
           "public_interface_diagnostics":bool(diagnostics["hard_checks_passed"]),
           "inference_f1":inference["target_allocation_micro_f1"] is not None and inference["target_allocation_micro_f1"]>=cfg["allocation_f1_after_second_event"],
           "oracle_point_gap":oracle["difference"]>=cfg["qom_minus_oracle_min"],
           "oracle_ci_gap":oracle["ci95"][0]>=cfg["qom_minus_oracle_ci_lower_min"],
           "oracle_gap_each_lower":all(e["difference"]>=cfg["qom_minus_oracle_min"] for e in lower_effects.values()),
           "legacy_point_gain":effects["legacy_idb"]["difference"]>=cfg["improvement_vs_legacy_min"],
           "er_point_gain":effects["event_risk_idb"]["difference"]>=cfg["improvement_vs_event_risk_min"]}
    def pct(x):return f"{100*x:.1f}%"
    lines=["四种上层类型均整局固定：1,920 局核心网格及 480 局同预算消融已完成；每个方法 240 局，场景/种子按局配对。", "",
           "| 方法 | rush 胜率 | split_rush 胜率 | 总胜率 | 规划 P95（秒） |", "|---|---:|---:|---:|---:|"]
    for method,row in methods.items():
        lines.append(f"| {method} | {pct(row['win_rate_by_lower']['rush'])} | {pct(row['win_rate_by_lower']['split_rush'])} | {pct(row['win_rate'])} | {row['planning_seconds_p95']:.3f} |")
    lines += ["", "legacy_idb 使用原 channel 代理求解，再通过共同无 channel 匹配器执行；是统一协议重跑的旧规划器基线，"
              "不是把历史胜率直接并入本表。QOM-MCP 与 known_upper_mcp 共用候选生成、32 次 PUCT、两层事件和终局续演；"
              "其他基线记录实际成本，不能声称所有方法都同计算量。", "",
              "| QOM-MCP 相对方法 | 配对胜率差 | 配对 95% CI |", "|---|---:|---:|"]
    for method,effect in effects.items():
        lines.append(f"| {method} | {pct(effect['difference'])} | [{pct(effect['ci95'][0])}, {pct(effect['ci95'][1])}] |")
    lines += ["", "两种已知底层分别检验 QOM-MCP − known_upper_mcp，防止总体均值掩盖某一种底层的退化：", "",
              "| 已知 Blue 底层 | 配对胜率差 | 配对 95% CI |", "|---|---:|---:|"]
    for lower,effect in lower_effects.items():
        lines.append(f"| {lower} | {pct(effect['difference'])} | [{pct(effect['ci95'][0])}, {pct(effect['ci95'][1])}] |")
    lines += ["", "上层机制消融（相同候选规则、物理重排与 32 次 PUCT 预算）：", "",
              "| 对比 | 配对胜率差 | 配对 95% CI |", "|---|---:|---:|"]
    for name,effect in controls.items():
        lines.append(f"| {name} | {pct(effect['difference'])} | [{pct(effect['ci95'][0])}, {pct(effect['ci95'][1])}] |")
    lines += ["", "prior_mcp 在每次真实决策前恢复均匀类型先验，用于检验跨事件信念积累的作用；"
              "它仍能读取当前公开物理状态。上述机制对比报告效应及区间，不能根据测试结果倒换预注册主方法。"
              "若 finite_type_mcp 更强，必须报告 QOM 未带来额外优势。"]
    lines += ["", f"第二个命令起、消费公开运动后的当前分配 micro-F1：`{inference['target_allocation_micro_f1']}`；"
              f"下一次命令预测 micro-F1：`{inference['next_action_forecast_micro_f1']}`，同状态已知程序的预测准确率上限："
              f"`{inference['same_state_known_program_forecast_accuracy_ceiling']}`。"
              f"可评分局数 {inference['closed_eligible_episodes']}/{inference['closed_total_episodes']}。本轮不包含策略类型切换或开放集检测。", "",
              "正式效果门槛："+("全部通过。" if all(gates.values()) else "未全部通过；保留负结果，不能声称任务目标已经达到。"), "",
              "统计显著改善仅在配对 CI 排除 0 且两个预注册比较的 Holm 校正检验通过时声明。"
              "细分 24 个场景单元、约束和单训练种子限制见 JSON/CSV。扩展代理/候选诊断本轮默认未运行，不能据空记录排除相关瓶颈。"]
    failure={"inference":not gates["inference_f1"],
             "candidate_coverage":"Only eight shared actions: quantify using diagnostics; oracle gap alone cannot exclude a shared candidate bottleneck.",
             "stage2_ranking":[s["rank_spearman"] for s in diagnostics["snapshots"]],
             "planning_delta_vs_qom_bbr":effects["qom_bbr"],
             "causal_limit":"These diagnostics locate possible bottlenecks; associations do not prove a unique failure cause."}
    summary={"episodes":core_episodes,"ablation_episodes":len(ablations),"methods":methods,"paired_effects":effects,
             "matched_controls":controls,"oracle_effect_by_lower":lower_effects,"oracle_effect_by_scale":scale_effects,
             "inference":inference,"acceptance":gates,
             "all_passed":all(gates.values()),"failure_attribution":failure,"report_lines":lines}
    root.mkdir(parents=True,exist_ok=True)
    atomic_json(root/"summary.json",summary)
    with (root/"episodes.csv").open("w",newline="",encoding="utf8") as stream:
        writer=csv.DictWriter(stream,fieldnames=["id","scenario","population","lower","upper","method","seed","win","steps","planning_seconds_mean","planning_seconds_p95"])
        writer.writeheader()
        for row in rows:
            cell=row["cell"];times=[e["planning_seconds"] for e in row["events"]]
            writer.writerow({"id":cell["id"],"scenario":cell["scenario"]["label"],"population":cell["scenario"]["red"]+cell["scenario"]["blue"],
                             "lower":cell["lower"],"upper":cell["upper"],"method":cell["method"],"seed":cell["seed"],
                             "win":row["red_win"],"steps":row["steps"],"planning_seconds_mean":float(np.mean(times)),"planning_seconds_p95":float(np.quantile(times,.95))})
    cell_rows=[]
    for scenario in config["scenarios"]:
        for lower in config["lower_policies"]:
            for upper in config["closed_upper"]+config["open_upper"]:
                for method in method_names:
                    subset=[r for r in rows if r["cell"]["scenario"]==scenario and r["cell"]["lower"]==lower and r["cell"]["upper"]==upper and r["cell"]["method"]==method]
                    cell_rows.append({"scenario":scenario["label"],"lower":lower,"upper":upper,"method":method,"episodes":len(subset),"win_rate":float(np.mean([r["red_win"] for r in subset]))})
    atomic_json(root/"cell_metrics.json",cell_rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4))
    ax.bar(range(len(method_names)),[methods[m]["win_rate"] for m in method_names])
    ax.set_xticks(range(len(method_names)),method_names,rotation=25,ha="right")
    ax.set_ylabel("Red win rate");ax.set_ylim(0,1);fig.tight_layout()
    figures=root.parents[1]/"figures";figures.mkdir(exist_ok=True)
    fig.savefig(figures/"unknown_upper_win_rates.png",dpi=160);plt.close(fig)
    return summary
