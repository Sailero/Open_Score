"""Reader-oriented paired results for the six-method fast comparison."""
from pathlib import Path
import csv
import numpy as np
from .fast_planner import METHODS, METHOD_NAMES
from .storage import atomic_json
from .analysis import paired_effect


def summarize(rows, config, root):
    if len(rows) != config["expected_episodes"] or len({r["cell"]["id"] for r in rows}) != len(rows):
        raise ValueError("Missing or duplicate fast comparison units")
    groups = {}
    for row in rows:
        groups.setdefault(row["cell"]["seed"], []).append(row)
    if any(len(group)!=len(METHODS) or {r["cell"]["method"] for r in group} != set(METHODS) or
           len({(r["cell"]["scenario"]["label"],r["cell"]["lower"],r["cell"]["upper"]) for r in group})!=1
           for group in groups.values()):
        raise ValueError("Each seed must have all six methods")
    from .fast_evaluation import registry
    if sorted([r["cell"] for r in rows],key=lambda c:c["id"]) != sorted(registry(config),key=lambda c:c["id"]):
        raise ValueError("Results differ from the registered conditions")
    results = {}
    for method in METHODS:
        selected = [r for r in rows if r["cell"]["method"] == method]
        events = [e for r in selected for e in r["events"]]
        times = np.array([e["planning_seconds"] for e in events])
        filtered = [e for r in selected for e in r["filtered_assignments"] if e["event_index"]>=1]
        labels = [(a,b) for e in filtered for a,b in zip(e["truth_targets"],e["predicted_targets"])]
        results[method] = {
            "name":METHOD_NAMES[method], "episodes":len(selected),
            "win_rate":float(np.mean([r["red_win"] for r in selected])),
            "planning_mean_seconds":float(times.mean()),"planning_p95_seconds":float(np.quantile(times,.95)),
            "budget_overrun_rate":float(np.mean(times>config["planning"]["decision_seconds"])),
            "analytic_fallback_rate":float(np.mean([e.get("analytic_fallback",False) for e in events])),
            "current_target_accuracy":float(np.mean([a==b for a,b in labels])) if labels else None,
            "scored_identity_records":len(labels),
            "win_rate_by_lower":{l:float(np.mean([r["red_win"] for r in selected if r["cell"]["lower"]==l])) for l in config["lower_policies"]},
            "planning_p95_by_scale":{s["label"]:float(np.quantile([e["planning_seconds"] for r in selected if r["cell"]["scenario"]["label"]==s["label"] for e in r["events"]],.95)) for s in config["scenarios"]},
        }
    effects = {}
    for method, oracle in [("belief_short","known_short"),("belief_recourse","known_recourse")]:
        for reference in ["balanced","legacy",oracle]:
            effects[f"{method}:{reference}"] = paired_effect(rows, method, reference, config["acceptance"]["bootstrap_replicates"])
    effects["belief_recourse:belief_short"] = paired_effect(rows,"belief_recourse","belief_short",config["acceptance"]["bootstrap_replicates"])
    lower_effects = {m:{lower:paired_effect([r for r in rows if r["cell"]["lower"]==lower],m,o,
                       config["acceptance"]["bootstrap_replicates"]) for lower in config["lower_policies"]}
                     for m,o in [("belief_short","known_short"),("belief_recourse","known_recourse")]}
    # Two proposed methods x two ordinary baselines, all predeclared.
    family=[key for key in effects if key.split(":")[1] in {"balanced","legacy"}]
    previous=0.
    for rank,key in enumerate(sorted(family,key=lambda k:effects[k]["p_value"])):
        previous=max(previous,min(1.,(len(family)-rank)*effects[key]["p_value"]))
        effects[key]["holm_adjusted_p"]=previous
        effects[key]["significant_gain"]=effects[key]["ci95"][0]>0 and previous<.05
    gates={}
    for method, oracle in [("belief_short","known_short"),("belief_recourse","known_recourse")]:
        e=effects[f"{method}:{oracle}"]
        gains=[effects[f"{method}:{ref}"] for ref in ("balanced","legacy")]
        gates[method]={
            "oracle_point_gap":e["difference"]>=-config["acceptance"]["oracle_gap_pp"]/100,
            "oracle_ci_gap":e["ci95"][0]>=config["acceptance"]["oracle_ci_lower_pp"]/100,
            "oracle_gap_each_lower":all(x["difference"]>=-config["acceptance"]["oracle_gap_pp"]/100 for x in lower_effects[method].values()),
            "gain_at_least_five_pp":all(x["difference"]>=config["acceptance"]["improvement_pp"]/100 for x in gains),
            "significant_gain":all(x["significant_gain"] for x in gains),
            "planning_p95":all(t<=config["acceptance"]["decision_p95_seconds"] for t in results[method]["planning_p95_by_scale"].values()),
        }
    def pct(x):return f"{100*x:.1f}%"
    lines=["## Stage3：六方法同场景快速比较","",
           f"完成 {len(rows)} 局；每个方法 {len(groups)} 局。每一个场景、底层规则、上层类型和种子都包含全部六个方法。",
           "两种方案都只根据公开运动推断蓝方规则。方案一作短期验证，方案二再模拟一次看到新运动后的重新分组。Stage1/2 冻结，不重新训练 QOM。","",
           "| 方法 | rush 胜率 | split_rush 胜率 | 总胜率 | 平均决策秒数 | P95 秒数 |",
           "|---|---:|---:|---:|---:|---:|"]
    for m,r in results.items():
        lines.append(f"| {r['name']} | {pct(r['win_rate_by_lower']['rush'])} | {pct(r['win_rate_by_lower']['split_rush'])} | {pct(r['win_rate'])} | {r['planning_mean_seconds']:.3f} | {r['planning_p95_seconds']:.3f} |")
    lines += ["","差值为前者减后者，单位为百分点。置信区间跨过 0 时，不据点估计宣称稳定改善。","",
              "| 对比 | 配对胜率差（pp） | 95% 区间（pp） |","|---|---:|---:|"]
    for key,e in effects.items():
        a,b=key.split(":")
        lines.append(f"| {METHOD_NAMES[a]} − {METHOD_NAMES[b]} | {100*e['difference']:.1f} | [{100*e['ci95'][0]:.1f}, {100*e['ci95'][1]:.1f}] |")
    lines += ["", "两种蓝方底层分别检查，避免合并胜率掩盖其中一种的退化：", "",
              "| 方案 | 蓝方底层 | 相对知情参照差值（百分点） | 95% 区间 |", "|---|---|---:|---:|"]
    for m, by_lower in lower_effects.items():
        for lower, e in by_lower.items():
            lines.append(f"| {METHOD_NAMES[m]} | {lower} | {100*e['difference']:.1f} | [{100*e['ci95'][0]:.1f}, {100*e['ci95'][1]:.1f}] |")
    lines += ["", "识别准确率表示：从第二次分组起，观察到新的运动后，每名存活蓝方的当前攻击目标猜对了多少。"
              "这里每名成员只有一个目标，因此它等于目标分配的 micro-F1；并不是未知下一次随机动作的预测准确率。", "",
              "| 方法 | 当前目标识别准确率 |", "|---|---:|"]
    for m,r in results.items():
        if r['current_target_accuracy'] is not None:
            lines.append(f"| {r['name']} | {pct(r['current_target_accuracy'])} |")
    lines += ["","### 如何判断是否有效",""]
    for m,g in gates.items():
        lines.append(f"- {METHOD_NAMES[m]}：知情差距门槛 {'通过' if g['oracle_point_gap'] else '未通过'}；"
                     f"知情差距置信区间门槛 {'通过' if g['oracle_ci_gap'] else '未通过'}；"
                     f"两种底层各自差距门槛 {'通过' if g['oracle_gap_each_lower'] else '未通过'}；"
                     f"相对两基线至少 +5 pp {'通过' if g['gain_at_least_five_pp'] else '未通过'}；"
                     f"校正后显著改善 {'通过' if g['significant_gain'] else '未通过'}；"
                     f"各规模 P95 ≤3 秒 {'通过' if g['planning_p95'] else '未通过'}。")
    lines += ["","这是一轮集中验证，不替代多训练种子和外部最先进算法对照。旧方法使用公开的快速求解预算，"
              "不是把旧协议的历史胜率直接放进新表。知情参照与对应方案共用实现、候选规则和预算；"
              "预算在计算边界检查，超时率与退回代理方案的比例保存在 JSON 中，不声称硬实时保证。",
              "环境在决策计算时等待，胜率尚未计入观测过期造成的损失。未达到门槛时保留负结果，"
              "不把局部误差、候选覆盖和未来价值估计统称为训练不足。"]
    summary={"complete":True,"episodes":len(rows),"pairs_per_method":len(groups),"methods":results,
             "paired_effects":effects,"oracle_effect_by_lower":lower_effects,"acceptance":gates,
             "all_identity_valid":all(r["identity_valid"] and r["fixed_targets"] and not r["infeasible"] for r in rows),
             "report_lines":lines}
    root=Path(root);atomic_json(root/"summary.json",summary)
    with (root/"episodes.csv").open("w",newline="",encoding="utf-8-sig") as stream:
        fields=["id","scenario","lower","upper","seed","method","name","win","steps","wall_seconds","planning_mean_seconds"]
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in sorted(rows,key=lambda x:x["cell"]["id"]):
            c=r["cell"];writer.writerow({"id":c["id"],"scenario":c["scenario"]["label"],"lower":c["lower"],"upper":c["upper"],"seed":c["seed"],
                                        "method":c["method"],"name":METHOD_NAMES[c["method"]],"win":r["red_win"],"steps":r["steps"],"wall_seconds":r["wall_seconds"],
                                        "planning_mean_seconds":float(np.mean([e["planning_seconds"] for e in r["events"]]))})
    cells=[]
    for s in config["scenarios"]:
        for l in config["lower_policies"]:
            for u in config["closed_upper"]:
                for m in METHODS:
                    selected=[r for r in rows if r["cell"]["scenario"]==s and r["cell"]["lower"]==l and r["cell"]["upper"]==u and r["cell"]["method"]==m]
                    cells.append({"scenario":s["label"],"lower":l,"upper":u,"method":m,"episodes":len(selected),"wins":sum(r["red_win"] for r in selected)})
    atomic_json(root/"cell_metrics.json",cells)
    return summary
