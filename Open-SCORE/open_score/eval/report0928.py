"""main0928 formal report: Looped (no count) as the main method.

Reuses identity-checked IQM helpers from report0923. Table 1 is baselines vs
Looped; table 2 is ablations and +count; figure 3 is mechanism.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from . import experiment as X
from .report0921 import HAD_GROUPS, _num, _table
from .report0923 import (
    FORMAL_GROUPS, SMAC_KEY, Results, _atomic_text, _cost_table, _figure_lines,
    _fmt_wins, _json, _label, _probe_pointer, _quota, _snapshot, paper_name as _paper0923,
    render_figures, PROBE_FILES,
)

PAPER_NAMES = {
    "regir_sg": "Looped", "regir_norefil_sg": "NoREFIL", "regir_r0_sg": "R0",
    "regir_r1_sg": "1-round", "regir_nomem": "NoMem", "regir_last_sg": "Last",
    "regir_fixed4_sg": "Fixed4", "regir_kv0_sg": "ARR", "regir_untied4_sg": "Untied4",
    "regir_count_sg": "Looped+count",
    "refil": "REFIL", "refil_matched": "REFIL-matched", "b2_qmix_atten": "QMIX-Atten",
    "dcg": "DCG", "spectra": "SPECTra", "alma": "ALMA", "transfqmix": "TransfQMix",
    "rule_nv1": "Rule", "random": "Random",
}


def paper_name(method, branch=None):
    return PAPER_NAMES.get(method, _paper0923(method, branch))


TABLE1 = ("regir_sg", "refil", "refil_matched", "b2_qmix_atten", "dcg", "spectra",
          "alma", "transfqmix", "rule_nv1")
TABLE2 = ("regir_sg", "regir_norefil_sg", "regir_r0_sg", "regir_r1_sg", "regir_nomem",
          "regir_last_sg", "regir_fixed4_sg", "regir_kv0_sg", "regir_untied4_sg",
          "regir_count_sg")


def render_markdown(results, *, paper=False):
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    ok, checks = results.integrity()
    mismatches = [c for c in checks if c["observed"] is not None and not c["match"]]
    pending_import = [c for c in checks if c["pending"]]
    groups = FORMAL_GROUPS
    lines = []
    if paper:
        lines += [f"# main0928 论文版（{now}）", "",
                  "Looped（无 count）为主方法。表1 基线，表2 消融与 +count，图3 机制。"
                  "统计量为组内等权后的跨种子 IQM 与分层 bootstrap 95% 区间。缺格显示 —，不插值。", ""]
    else:
        lines += [f"# main0928：无 count 的 Looped 战役（{now}）", "",
                  "主方法 `regir_sg`（Looped，`skip_count_inject=True`，读出记忆梯度截断）。"
                  "唯一 +count 变体是 `regir_count_sg`。没有试训闸门，没有 IAR。"
                  "导入 7 个 main0921 外部基线，不重训。", ""]
    if mismatches:
        lines += ["**身份核对失败：导入基线与 main0921 发表格不一致。**", ""]
        lines += _table(["方法", "组", "0921", "本次"],
                        [[c["method"], c["group"], c["expected"], _num(c["observed"])] for c in mismatches])
    elif pending_import:
        lines += ["导入基线仍有未完成的正式格；未完成项显示 —。", ""]
    else:
        lines += ["导入基线四组 IQM 与 main0921 表 1 三位小数一致（若已展开 imported/）。", ""]

    lines += ["## 图1：方法结构" if not paper else "### 图 1", ""]
    stem = "paper_fig1_architecture" if paper else "fig1_architecture"
    lines += _figure_lines(stem, "图1 方法结构",
                           "REFIL 宿主 + 循环关系旁路（实体间注意力、未锚定 K/V、记忆读出、跨轮加权、零初始化残差）。")

    lines += ["## 表1：主比较（Looped vs 基线）" if not paper else "### 表 1", "",
              "HAD：IQM [bootstrap 2.5%, 97.5%]。SMAC 为胜率 IQM。", ""]
    rows = []
    for method in TABLE1:
        if method == "rule_nv1":
            line = [paper_name(method),
                    *[f"{_num(results.anchor_mean(method, HAD_GROUPS[g]))}（共同参照）" for g in groups],
                    "—", "—"]
        else:
            wins = results.ood_wins(method)
            smac = [results.cell("smacv2", method, [cfg]) if method in X.methods("smacv2") else "—"
                    for cfg in SMAC_KEY]
            line = [paper_name(method),
                    *[results.cell("had", method, HAD_GROUPS[g]) for g in groups],
                    "—" if method == "refil" else _fmt_wins(wins),
                    " / ".join(smac)]
        rows.append(line)
    lines += _table(["方法", *groups, "相对 REFIL $O$", "SMAC 10/15/20/10v15"], rows)

    lines += ["## 图2：规模曲线" if not paper else "### 图 2", ""]
    stem = "paper_fig2_scale" if paper else "fig2_scale"
    lines += _figure_lines(stem, "图2 规模曲线", "HAD 对数纵轴 IQM；SMACv2 胜率线性轴。缺格断开。")

    lines += ["## 表2：消融与设计空间" if not paper else "### 表 2", "",
              "−宿主 = NoREFIL；−实体交互 = R0；−多轮 = R1；−记忆 = NoMem；"
              "−跨轮加权 = Last；−随机深度 = Fixed4；锚定替换 = ARR；解绑参数 = Untied4；"
              "+count 单独一行。", ""]
    rows = []
    for method in TABLE2:
        rows.append([
            paper_name(method),
            *[results.cell("had", method, HAD_GROUPS[g]) for g in groups],
            results.cell("had", method, [(50, 50, 2)], field=1),
            results.cell("had", method, [(50, 50, 2)], field=2),
        ])
    lines += _table(["方法", *groups, "50v50 相撞率", "50v50 重复追击"], rows)

    lines += ["## 图3：机制" if not paper else "### 图 3", ""]
    stem = "paper_fig3_mechanism" if paper else "fig3_mechanism"
    lines += _figure_lines(stem, "图3 机制",
                           "执行深度（Looped / ARR / Fixed4 / Untied4）、读出干预（Looped）、"
                           "探针 coverage / dynamics / deep_rounds / global_probe。")

    if paper:
        return "\n".join(lines).rstrip() + "\n"

    lines += ["## 成本 M4", "",
              "仅保留匹配当前 final 身份、登记物理 GPU、float32、200 次同步计时的行。", ""]
    lines += _cost_table(results)
    lines += ["## 探针文件", ""]
    lines += [f"- {_probe_pointer(results, kind)}" for kind in PROBE_FILES if kind in (
        "coverage", "dynamics", "deep_rounds", "global_probe")]
    lines += ["", "原始逐局：[episodes.csv](episodes.csv)；冻结协议：[experiment.json](experiment.json)。", ""]
    return "\n".join(lines).rstrip() + "\n"


def _load(output, run="train"):
    if run != "train":
        raise ValueError("main0928 正式报告只读 train run")
    output = Path(output)
    metadata = _json(output / "experiment.json", {})
    if metadata.get("profile") not in (None, X.PROFILE):
        raise ValueError(f"report0928 refuses profile={metadata.get('profile')}")
    inventory = _json(output / "inventory.json", {})
    return Results(output, inventory, metadata)


def render_report(output, *, run="train"):
    output = Path(output)
    results = _load(output, run=run)
    X.atomic_json(output / "aggregates.json", _snapshot(results))
    try:
        render_figures(results, output, paper=False)
    except Exception as error:
        print(f"formal figures skipped: {type(error).__name__}: {error}", flush=True)
    return _atomic_text(output / "实验报告.md", render_markdown(results, paper=False))


def render_paper(output, *, run="train"):
    output = Path(output)
    results = _load(output, run=run)
    X.atomic_json(output / "aggregates.json", _snapshot(results))
    try:
        render_figures(results, output, paper=True)
    except Exception as error:
        print(f"paper figures skipped: {type(error).__name__}: {error}", flush=True)
    return _atomic_text(output / "实验报告_论文版.md", render_markdown(results, paper=True))
