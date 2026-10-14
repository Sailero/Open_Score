"""Render v3 display figures and tables from the saved publication statistics.

Reuse the existing pure plotting functions without importing the original
script's top-level recomputation or output writes. No CSVs or models are read.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
import ast
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parents[1]
SOURCE = REPO / "Open-SCORE/scripts/plot_paper1007.py"
METHODS = ("regir_nomem", "refil", "b2_qmix_atten", "dcg", "spectra", "alma", "rule_nv1")
CORE = ("regir_nomem", "regir_sg", "regir_norefil_sg", "regir_r0_sg", "regir_r1_sg")
VARIANTS = CORE + ("regir_last_sg", "regir_fixed4_sg", "regir_kv0_sg", "regir_untied4_sg", "regir_count_sg")
NAMES = dict(zip(METHODS, ("LEAF", "REFIL", "QMIX-Atten", "DCG", "SPECTra", "ALMA", "Rule")))
NAMES.update({"regir_sg": "Memory query", "regir_norefil_sg": "No host",
    "regir_r0_sg": "Direct read", "regir_r1_sg": "One round",
    "regir_last_sg": "Last read", "regir_fixed4_sg": "Fixed depth",
    "regir_kv0_sg": "Initial keys", "regir_untied4_sg": "Unshared rounds",
    "regir_count_sg": "Count inputs", "refil_matched": "REFIL (matched)"})
COLORS = dict(zip(METHODS, ("#087f8c", "#df8f44", "#4775b4", "#9768aa", "#be5962", "#799d54", "#6c727a")))
COLORS.update({"regir_sg": "#7388b7", "regir_norefil_sg": "#c79532",
    "regir_r0_sg": "#ba5671", "regir_r1_sg": "#745399"})


def plotting_functions(p):
    selected = {"style", "curve", "validation_curve", "joint_heatmap", "legend_handles"}
    source = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    nodes = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in selected]
    if {n.name for n in nodes} != selected:
        raise RuntimeError("The existing plotting function interface changed")
    scope = {"np": np, "plt": plt, "Line2D": Line2D, "COLORS": COLORS, "NAMES": NAMES,
        "GROUPS": {g: tuple(tuple(c) for c in cs) for g, cs in p["groups"].items()},
        "CONFIG": {(r["method"], tuple(r["config"])): r for r in p["config_summary"] + p["rule_config"]},
        "VALIDATION": {(r["method"], r["eval_point"]): r for r in p["validation_curves"]}}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), scope)
    return scope


def figures(p):
    f = plotting_functions(p)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .65, "savefig.dpi": 220})
    for basename, methods, main in (("fig_had_generalization", METHODS, True),
                                    ("fig_had_ablation", CORE, False)):
        fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.15))
        fig.subplots_adjust(left=.095, right=.918, bottom=.235,
                            top=.935, hspace=.62, wspace=.43)
        ax = axes[0, 0]
        for method in methods:
            if method != "rule_nv1": f["validation_curve"](ax, method)
        if main:
            ax.axhline(p["rule_ID_reference"]["reward"]["mean"], color=COLORS["rule_nv1"], ls="--", lw=1)
            ax.text(.04, .05, "Rule: final ID reference", transform=ax.transAxes, fontsize=8)
        ax.set(title="(a) Training validation", xlabel="Environment steps (millions)",
            ylabel=r"Reward $R=-D$ $\uparrow$", xlim=(0, 1.01), xticks=(0, .25, .5, .75, 1))
        f["style"](ax)
        ks = (4, 6, 9, 12)
        ax = axes[0, 1]
        for method in methods: f["curve"](ax, method, [(10, 10, k) for k in ks], ks)
        ax.set(title="(b) Target extrapolation", xlabel="Targets $K$ (10 vehicles per team)",
               ylabel=r"Reward $R$ $\uparrow$", xticks=ks)
        f["style"](ax)
        ns = (15, 20, 25, 30, 40, 50)
        ax = axes[1, 0]
        for method in methods: f["curve"](ax, method, [(n, n, 2) for n in ns], ns)
        ax.set(title="(c) Vehicle extrapolation", xlabel="Vehicles per team $N$ ($K=2$)",
               ylabel=r"Reward $R$ $\uparrow$", xticks=(15, 25, 40, 50))
        f["style"](ax)
        f["joint_heatmap"](fig, axes[1, 1], methods)
        # Keep all one-decimal cell values at 8pt without adjacent labels
        # touching in the two-column heatmap's double-digit negative cells.
        for text in axes[1, 1].texts:
            text.set_fontstretch("condensed")
        fig.legend(handles=f["legend_handles"](methods), loc="lower center",
                   bbox_to_anchor=(.52, .025), ncol=4 if main else 3,
                   frameon=False, columnspacing=1.2, handlelength=1.8)
        for ext in ("pdf", "png"):
            fig.savefig(PAPER/"figures"/f"{basename}.{ext}", facecolor="white")
        plt.close(fig)


def tables(p):
    groups = ("ID", "K-OOD", "N-OOD", "Joint-OOD")
    rows = {(r["method"], r["group"]): r for r in p["group_summary"]}
    output = [r"% Numerical source: ../data/paper1007.json; no scientific statistics recomputed.",
        r"% Display names map existing saved designs; SD unit is stated in each caption."]

    def table(macro, label, caption, methods, collision=True, note="", bold=True):
        out = ["", "\\newcommand{\\"+macro+"}{%", r"\begin{table*}[!t]", r"\centering\footnotesize",
            r"\setlength{\tabcolsep}{5pt}", "\\caption{"+caption+"}", "\\label{"+label+"}",
            r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}"+("llrrrr" if collision else "lrrrr")+"@{}}",
            r"\toprule", "Method"+(" & Metric" if collision else "")+r" & ID & $K$-OOD & $N$-OOD & Joint OOD \\", r"\midrule"]
        for i, method in enumerate(methods):
            name = NAMES[method]
            if method == "regir_nomem": name = r"\textbf{LEAF}"
            if method == "rule_nv1": name += r"$^{\dagger}$"
            for metric, digits in (("reward", 3), ("collision_percent", 2)) if collision else (("reward", 3),):
                cells = []
                for group in groups:
                    s = rows[method, group][metric]
                    value = f"{s['mean']:.{digits}f}\\pm{s['sample_sd']:.{digits}f}"
                    candidates = [rows[m, group][metric]["mean"] for m in methods]
                    best = max(candidates) if metric == "reward" else min(candidates)
                    if bold and s["mean"] == best: value = r"\mathbf{"+value+"}"
                    cells.append("$"+value+"$")
                prefix = name if metric == "reward" else ""
                if collision: prefix += " & "+(r"$R\uparrow$" if metric == "reward" else r"$C\downarrow$ (\%)")
                out.append(prefix+" & "+" & ".join(cells)+r" \\")
            if collision and i < len(methods)-1: out.append(r"\addlinespace[2pt]")
        out += [r"\bottomrule", r"\end{tabular*}"]
        if note: out += [r"\par\vspace{2pt}\parbox{\textwidth}{\footnotesize", note+"}"]
        out.append(r"\end{table*}}")
        output.extend(out)

    table("HadMainTable", "tab:had-main",
        r"Formal comparison: mean $\pm$ sample SD. Reward $R=-D$ is higher-is-better; friendly-collision mortality $C$ is lower-is-better. Each learned method has three training seeds and 300 common scenes per configuration. Bold marks the best group mean for each metric.", METHODS,
        note=r"$\dagger$ Rule has no training seeds. Its SD is across the 300 scene seeds, after averaging configurations within each scene. All other SDs measure training-seed variation. Collision SDs are percentage points.")
    table("HadCoreAblationTable", "tab:had-core-ablation",
        r"Trained design comparisons: absolute reward $R=-D$ and friendly-collision mortality $C$, mean $\pm$ sample SD across three training seeds. Each configuration uses 300 common scenes. Bold marks the best mean among these five designs.", CORE,
        note=r"Configurations are weighted equally within each training seed. Memory query adds incoming GRU history to LEAF's reader query. No host removes the local host, imagination, and associated losses from the memory-query design. Direct read and One round retain query history and execute zero and one contextual update, respectively. Collision SDs are percentage points.")
    table("HadFullVariantTable", "tab:had-full-variants",
        r"All ten completed trained designs: reward mean $\pm$ sample SD across three training seeds on the same formal matrix. Bold marks the best group mean. All designs share the evaluation configurations.", VARIANTS, collision=False,
        note=r"The five core names follow Table~\ref{tab:had-core-ablation}. Last read uses only the final read; Fixed depth trains and executes four rounds; Initial keys anchors keys and values to the initial entity set; Unshared rounds trains four independent update blocks; Count inputs adds typed-count features. These five designs also retain query history. Unshared rounds is compared with Fixed depth to separate sharing under a matched depth schedule.")
    table("HadCapacityTable", "tab:had-capacity",
        r"Capacity reference: reward and friendly-collision mortality, mean $\pm$ sample SD across three training seeds. The enlarged REFIL reference approximately matches LEAF's actor parameter budget; all methods use the same formal evaluation scenes.",
        ("regir_nomem", "refil", "refil_matched"),
        note=r"Actor parameter counts are 175,754 for LEAF, 102,025 for REFIL, and 176,393 for REFIL (matched). Collision SDs are percentage points.")
    (PAPER/"sections/publication_tables.tex").write_text("\n".join(output)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--figures-only", action="store_true")
    group.add_argument("--tables-only", action="store_true")
    opts = parser.parse_args()
    p = json.loads((PAPER/"data/paper1007.json").read_text(encoding="utf-8"))
    if not opts.tables_only: figures(p)
    if not opts.figures_only: tables(p)
    print("Rendered v3 display artifacts from saved statistics only.")


if __name__ == "__main__":
    main()
