"""Section V figures of the 1015 draft, sized for the IEEE two-column page.

Run ``python figures.py`` to render every figure into ``Papers/1015/figures``,
or ``python figures.py generalization depth`` to render selected figures.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator

import figdata

OUT = Path(__file__).resolve().parents[1]
COLUMN, PAGE = 3.5, 7.16
LEAF = figdata.LEAF

INK = "#222222"
SPINE = "#555555"
GRID = "#EBEBEB"
SHADE = "#F3F3F3"
MUTED = "#8C8C8C"

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "font.size": 7,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "axes.edgecolor": SPINE,
    "axes.labelcolor": INK,
    "axes.labelpad": 2.5,
    "axes.titlepad": 5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": SPINE,
    "ytick.color": SPINE,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.pad": 2,
    "ytick.major.pad": 2,
    "text.color": INK,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.2,
    "legend.borderaxespad": 0.2,
    "lines.solid_capstyle": "round",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 400,
}

COLORS = {
    "b2_qmix_atten": "#A7BDD8",
    "refil": "#3B6FA8",
    "dcg": "#93C2A3",
    "alma": "#E6B96A",
    "transfqmix": "#ADA0CF",
    "spectra": "#DDA7C2",
    "rule": "#4A4A4A",
    LEAF: "#D1342F",
    "regir_sg": "#F08C2E",
    "regir_fixed4_sg": "#4F78B0",
    "regir_untied4_sg": "#9DB3CF",
    "regir_r1_sg": "#EBA096",
    "regir_r0_sg": "#B4B9C0",
    "refil_matched": "#B4B9C0",
}
NAMES = {
    "b2_qmix_atten": "QMIX", "refil": "REFIL", "dcg": "DCG", "alma": "ALMA", "transfqmix": "TransfQMix",
    "spectra": "SPECTra", "rule": "Rule", LEAF: "LEAF",
    "regir_sg": "Memory query", "regir_fixed4_sg": "Fixed depth", "regir_untied4_sg": "Unshared rounds",
    "regir_r1_sg": "One round", "regir_r0_sg": "Direct read", "refil_matched": "REFIL-matched",
}
ORDER = ("b2_qmix_atten", "refil", "dcg", "alma", "transfqmix", "spectra", "rule", LEAF)
DASH = (0, (4, 2))


# ---------------------------------------------------------------- helpers
def canvas(width, height):
    fig = plt.figure(figsize=(width, height))
    return fig, lambda l, b, w, h: fig.add_axes((l / width, b / height, w / width, h / height))


def title(ax, letter, text, x=0.0):
    ax.set_title(rf"$\mathbf{{({letter})}}$ {text}", loc="left", x=x)


def log_y(ax, low, high, ticks):
    ax.set_yscale("log")
    ax.set_ylim(low, high)
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())


def log_x(ax, ticks, minor=()):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(FixedLocator(minor) if minor else NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())


def grid(ax, ticks, axis="y"):
    for t in ticks:
        (ax.axhline if axis == "y" else ax.axvline)(t, color=GRID, lw=0.5, zorder=0)


def smooth(values, width):
    values = np.asarray(values, dtype=float)
    out = np.empty_like(values)
    half = width // 2
    for i in range(values.shape[-1]):
        out[..., i] = values[..., max(0, i - half):i + half + 1].mean(axis=-1)
    return out


def validation(data, method, width=5):
    val = data["validation"][method]
    seeds = smooth(np.asarray(val["seeds"]), width)
    return np.asarray(val["t"]) / 1e6, seeds.mean(0), seeds.std(0, ddof=1)


def save(fig, name):
    paths = []
    for ext in ("pdf", "png"):
        path = OUT / f"{name}.{ext}"
        fig.savefig(path, facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def curve(ax, data, method, cfgs, xs):
    cell = data["configs"][method]
    mean = np.array([cell[c]["mean"] for c in cfgs])
    if method != "rule":
        sd = np.array([cell[c]["sd"] for c in cfgs])
        ax.fill_between(xs, np.maximum(mean - sd, 1e-3), mean + sd, color=COLORS[method],
                        alpha=0.16 if method == LEAF else 0.10, lw=0, zorder=4 if method == LEAF else 2)
    if method == LEAF:
        ax.plot(xs, mean, color=COLORS[LEAF], lw=1.8, marker="o", ms=2.8, mew=0, zorder=6)
    elif method == "rule":
        ax.plot(xs, mean, color=COLORS["rule"], lw=1.0, ls=DASH, zorder=5)
    else:
        ax.plot(xs, mean, color=COLORS[method], lw=1.0, zorder=3)


# ---------------------------------------------------------------- Fig. 4, zero-shot generalization
SWEEPS = (
    ("b", "More targets, $N$ = 10", ("10-2", "10-3", "10-4", "10-6", "10-9", "10-12"), (2, 3, 4, 6, 9, 12),
     r"Targets $K$", (2, 4, 6, 12), (3, 9), np.sqrt(12)),
    ("c", "Larger teams, $K$ = 2", ("4-2", "5-2", "6-2", "8-2", "10-2", "15-2", "20-2", "25-2", "30-2", "40-2",
                                     "50-2"), (4, 5, 6, 8, 10, 15, 20, 25, 30, 40, 50),
     r"Aircraft per side $N$", (4, 10, 20, 50), (5, 6, 8, 15, 25, 30, 40), np.sqrt(150)),
    ("d", "More targets, $N$ = 30", ("30-2", "30-4", "30-6", "30-9", "30-12"), (2, 4, 6, 9, 12),
     r"Targets $K$", (2, 4, 6, 12), (9,), None),
)


def fig_generalization(data):
    fig, add = canvas(PAGE, 1.84)
    bottom, width, height = 0.32, 1.40, 1.10
    axes = [add(left, bottom, width, height) for left in (0.40, 2.35, 3.99, 5.63)]
    ticks = (0.3, 1, 3, 10, 30)
    for ax in axes[1:]:
        if ax is not axes[1]:
            ax.sharey(axes[1])
        log_y(ax, 0.15, 30, ticks)
        grid(ax, ticks)

    ax = axes[0]
    ax.set_ylim(0, 8)
    ax.set_yticks((0, 2, 4, 6, 8))
    grid(ax, (2, 4, 6, 8))
    for method in ORDER:
        if method == "rule":
            continue
        x, y, sd = validation(data, method)
        lead = method == LEAF
        ax.fill_between(x, y - sd, y + sd, color=COLORS[method], alpha=0.16 if lead else 0.10,
                        lw=0, zorder=4 if lead else 2)
        ax.plot(x, y, color=COLORS[method], lw=1.8 if lead else 1.0, zorder=6 if lead else 3)
    ax.axhline(data["rule_id_reference"]["mean"], color=COLORS["rule"], lw=1.0, ls=DASH, zorder=5)
    ax.set_xlim(0, 1)
    ax.set_xticks((0, 0.5, 1), ("0", "0.5", "1"))
    ax.set_xlabel("Environment steps (M)")
    ax.set_ylabel(r"Validation damage $D$")
    title(ax, "a", "Training on ID scales")

    for ax, (letter, name, cfgs, xs, xlabel, major, minor, train) in zip(axes[1:], SWEEPS):
        if train:
            ax.axvspan(0.1, train, color=SHADE, lw=0, zorder=0)
            ax.text(xs[0] / 1.07, 0.94, "training", transform=ax.get_xaxis_transform(),
                    ha="left", va="top", fontsize=6.2, color=MUTED)
        for method in ORDER:
            curve(ax, data, method, cfgs, xs)
        log_x(ax, major, minor)
        ax.set_xlim(xs[0] / 1.12, xs[-1] * 1.12)
        ax.set_xlabel(xlabel)
        ax.tick_params(axis="y", labelleft=ax is axes[1])
        if ax is axes[1]:
            ax.set_ylabel(r"Target damage $D$")
        title(ax, letter, name)

    handles = [Line2D([], [], color=COLORS[m], lw=1.8 if m == LEAF else 1.0, ls=DASH if m == "rule" else "-",
                      label=NAMES[m]) for m in ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=8, bbox_to_anchor=(0.5, 1.0), columnspacing=1.8,
               handlelength=2.0)
    return save(fig, "exp_generalization")


# ---------------------------------------------------------------- Fig. 5, organization versus size
SAME_SIZE = ("refil_matched", "regir_r0_sg", "regir_r1_sg", "regir_fixed4_sg", LEAF)
NO_LOOP, LOOP = "#B8BDC4", "#EBA096"


def fig_capacity(data):
    fig, add = canvas(COLUMN, 1.70)
    ax = add(0.80, 0.33, 0.98, 1.08)
    ys = np.arange(len(SAME_SIZE))[::-1]
    values = [data["group"][m]["Joint-OOD"]["mean"] for m in SAME_SIZE]
    colors = [NO_LOOP, NO_LOOP, LOOP, LOOP, COLORS[LEAF]]
    ax.barh(ys, values, height=0.64, color=colors, lw=0, zorder=3)
    for y, v, c in zip(ys, values, colors):
        ax.text(v - 0.1, y, f"{v:.1f}", va="center", ha="right", fontsize=6.5,
                color="white" if c == COLORS[LEAF] else INK, zorder=5)
    refil = data["group"]["refil"]["Joint-OOD"]["mean"]
    ax.axvline(refil, color=COLORS["refil"], lw=0.9, ls=DASH, zorder=2)
    ax.text(refil + 0.25, 2.5, "REFIL", color=COLORS["refil"], ha="left", va="center", fontsize=6.5)
    ax.set_ylim(-0.55, ys[0] + 0.95)
    ax.set_yticks(ys, [NAMES[m] for m in SAME_SIZE])
    ax.get_yticklabels()[-1].set_color(COLORS[LEAF])
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.tick_params(axis="y", length=0, pad=3)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 5)
    ax.set_xticks((0, 2, 4))
    grid(ax, (2, 4), axis="x")
    ax.set_xlabel(r"Joint-OOD damage $D$")
    title(ax, "a", "Same size as LEAF", x=-0.78)

    ax = add(2.30, 0.33, 1.12, 1.08)
    ticks = (0.5, 1, 2, 4)
    log_y(ax, 0.35, 6.5, ticks)
    grid(ax, ticks)
    lines = (("refil", "#8F969F", "REFIL"), ("refil_matched", "#B8BDC4", "REFIL-matched"),
             ("regir_r0_sg", "#A3A9B1", "Direct read"), ("regir_r1_sg", LOOP, "One round"), (LEAF, COLORS[LEAF], "LEAF"))
    for method, color, _ in lines:
        x, y, _ = validation(data, method)
        lead = method == LEAF
        ax.plot(x, y, color=color, lw=1.8 if lead else 1.0, zorder=5 if lead else 3)
    handles = [Line2D([], [], color=COLORS[LEAF], lw=1.8, label="LEAF"),
               Line2D([], [], color=LOOP, lw=1.0, label="One round"),
               Line2D([], [], color="#A3A9B1", lw=1.0, label="No context loop")]
    ax.legend(handles=handles, loc="lower left")
    ax.set_xlim(0, 1)
    ax.set_xticks((0, 0.5, 1), ("0", "0.5", "1"))
    ax.set_xlabel("Environment steps (M)")
    ax.set_ylabel(r"ID validation $D$")
    title(ax, "b", "Learning speed")
    return save(fig, "exp_capacity")


# ---------------------------------------------------------------- Fig. 6, memory after reading
CURRENT, MEMORY = COLORS[LEAF], COLORS["regir_sg"]


def fig_memory(data):
    fig, add = canvas(COLUMN, 1.70)
    auc = data["memory_drift"]["auc"]
    ax = add(0.42, 0.40, 1.16, 1.01)
    width = 0.34
    for i, shifted in enumerate(("50-2", "30-12")):
        for j, (feature, color) in enumerate((("self", CURRENT), ("memory", MEMORY))):
            values = np.asarray(auc[shifted][feature]["linear"])
            x = i + (j - 0.5) * (width + 0.03)
            ax.bar(x, values.mean() - 0.5, bottom=0.5, width=width, color=color, lw=0, zorder=3)
            ax.errorbar(x, values.mean(), yerr=values.std(ddof=1), color=INK, lw=0.7, capsize=1.6, capthick=0.7,
                        zorder=4)
            ax.text(x, values.mean() + values.std(ddof=1) + 0.012, f"{values.mean():.2f}", ha="center",
                    va="bottom", fontsize=6.3)
    ax.set_xticks((0, 1), ("$N$ = 50\n$K$ = 2", "$N$ = 30\n$K$ = 12"))
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0.5, 1.08)
    ax.set_yticks((0.5, 0.75, 1.0), ("0.5", "0.75", "1"))
    ax.spines["left"].set_bounds(0.5, 1.0)
    grid(ax, (0.75, 1.0))
    ax.set_ylabel("Scale-detection AUC")
    ax.legend(handles=[Patch(color=CURRENT, label="Current state"), Patch(color=MEMORY, label="Memory")],
              loc="upper center", handlelength=0.9, handleheight=0.9, ncol=2, columnspacing=0.8,
              bbox_to_anchor=(0.5, 1.04))
    title(ax, "a", "Scale information")

    ax = add(2.10, 0.40, 1.32, 1.01)
    group = data["group"]
    rise = [100 * (group["regir_sg"][g]["mean"] / group[LEAF][g]["mean"] - 1) for g in figdata.GROUPS]
    ax.bar(range(4), rise, width=0.58, color=MEMORY, lw=0, zorder=3)
    for i, v in enumerate(rise):
        ax.text(i, v + 2, f"+{v:.0f}%", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(range(4), ("ID", "$K$-OOD", "$N$-OOD", "Joint"))
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylim(0, 95)
    ax.set_yticks((0, 25, 50, 75))
    grid(ax, (25, 50, 75))
    ax.set_ylabel("Extra damage (%)")
    title(ax, "b", "Memory in the query")
    return save(fig, "exp_memory")


# ---------------------------------------------------------------- Fig. 7, any execution depth
DEPTH_MODELS = (LEAF, "regir_sg", "regir_fixed4_sg", "regir_untied4_sg")
SAMPLED = {LEAF, "regir_sg"}
DEPTH_CONFIGS = ("10-2", "30-2", "50-2")


def depth_curve(data, method):
    prof = data["depth"][method]
    depths = [d for d in range(1, 7) if all(str(d) in prof[c] for c in DEPTH_CONFIGS)]
    seeds = np.array([[np.mean([prof[c][str(d)][s] for c in DEPTH_CONFIGS]) for d in depths] for s in range(3)])
    return np.array(depths), seeds.mean(0)


def fig_depth(data):
    fig, add = canvas(PAGE, 1.78)
    bottom, height = 0.32, 1.17
    ax = add(0.40, bottom, 1.95, height)
    ax.axvspan(4.5, 6.4, color=SHADE, lw=0, zorder=0)
    ax.text(5.45, 0.33, "beyond training", ha="center", va="bottom", fontsize=6.2, color=MUTED)
    ticks = (0.3, 1, 3, 10)
    log_y(ax, 0.3, 25, ticks)
    grid(ax, ticks)
    for method in DEPTH_MODELS[::-1]:
        depths, mean = depth_curve(data, method)
        lead = method == LEAF
        ax.plot(depths, mean, color=COLORS[method], lw=1.8 if lead else 1.2,
                ls="-" if method in SAMPLED else DASH, marker="o", ms=3.0 if lead else 2.6, mew=0,
                zorder=6 if lead else 4)
    ax.set_xlim(0.7, 6.4)
    ax.set_xticks(range(1, 7))
    ax.set_xlabel(r"Execution depth $L$")
    ax.set_ylabel(r"Target damage $D$")
    handles = [Line2D([], [], color=COLORS[m], lw=1.8 if m == LEAF else 1.2, ls="-" if m in SAMPLED else DASH,
                      label=NAMES[m]) for m in DEPTH_MODELS]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.02, 1.02), handlelength=2.2,
              labelspacing=0.3, frameon=True, facecolor="white", framealpha=1, edgecolor="none")
    title(ax, "a", "Damage at every exit")

    ax = add(2.95, bottom, 1.55, height)
    fusion = data["fusion"]
    alpha = 100 * np.mean([fusion[c]["alpha"] for c in fusion], axis=0)
    ax.bar(range(1, 5), alpha, width=0.6, color=COLORS[LEAF], lw=0, zorder=3)
    for r, v in enumerate(alpha, start=1):
        ax.text(r, v + 2, f"{v:.0f}%" if v >= 1 else "<1%", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(range(1, 5))
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(0.4, 4.6)
    ax.set_ylim(0, 92)
    ax.set_yticks((0, 25, 50, 75))
    grid(ax, (25, 50, 75))
    ax.set_xlabel(r"Round $r$")
    ax.set_ylabel(r"Fusion weight $\alpha_r$ (%)")
    title(ax, "b", "Weight per round")

    ax = add(4.95, bottom, 1.68, height)
    depth_cost_panel(ax, data)
    return save(fig, "exp_depth")


def depth_cost_panel(ax, data):
    """Panel 7(c), shared by the paper and its standalone preview."""
    damage_ax = ax.twinx()
    timing = data["timing"]["closed_loop"]["50-2"]
    depths = np.arange(1, 5)
    positions = np.arange(1, 6)
    times = [timing[f"LEAF R{d}"]["ms"] for d in depths] + [timing["One round"]["ms"]]
    damage = [np.mean(data["depth"][LEAF]["50-2"][str(d)]) for d in depths]
    one_damage = data["configs"]["regir_r1_sg"]["50-2"]["mean"]

    ax.bar(positions, times, width=0.56, color="#D6DADF", lw=0, zorder=2)
    for x, value in zip(positions, times):
        ax.text(x, value + 0.22, f"{value:.1f}", ha="center", va="bottom", fontsize=6.5)
    ax.axvline(4.5, color="#B8BDC4", ls=DASH, lw=0.7, zorder=1)
    damage_ax.plot(depths, damage, color=COLORS[LEAF], marker="o", ms=3.8, mew=0, lw=1.5, zorder=5)
    damage_ax.plot(5, one_damage, color=COLORS[LEAF], marker="^", ms=5.0, mfc="white", mew=1, ls="", zorder=5)
    for x, value in zip(positions, damage + [one_damage]):
        damage_ax.text(x, value - 0.12, f"{value:.2f}", color=COLORS[LEAF],
                       ha="center", va="top", fontsize=6.5)

    ax.annotate("", xy=(1, 12.15), xytext=(4, 12.15),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.7, mutation_scale=7))
    saving = 100 * (1 - times[0] / times[3])
    ax.text(2.5, 12.5, f"{saving:.0f}% less time", ha="center", va="bottom", fontsize=6.5)
    ax.set_xlim(0.45, 5.55)
    ax.set_ylim(0, 14)
    ax.set_yticks((0, 5, 10))
    grid(ax, (5, 10))
    ax.set_xticks(positions, [rf"$L$={d}" for d in depths] + ["One\nround"])
    ax.set_ylabel("Team decision time (ms)")
    damage_ax.set_ylim(0, 2.25)
    damage_ax.set_yticks((0, 0.5, 1, 1.5, 2), ("0", "0.5", "1", "1.5", "2"))
    damage_ax.spines["right"].set_visible(True)
    damage_ax.spines["right"].set_color(SPINE)
    damage_ax.tick_params(axis="y", colors=SPINE)
    damage_ax.set_ylabel(r"Target damage $D$", color=INK)
    title(ax, "c", "Depth, time, and protection")


def fig_depth_preview(data):
    """Standalone preview of the paper's depth, time, and damage panel."""
    fig, add = canvas(COLUMN, 2.10)
    ax = add(0.46, 0.32, 2.52, 1.48)
    depth_cost_panel(ax, data)
    return save(fig, "backfig/exp_depth_alternative")


FIGURES = {
    "generalization": fig_generalization,
    "capacity": fig_capacity,
    "memory": fig_memory,
    "depth": fig_depth,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    renderers = {**FIGURES, "depth-preview": fig_depth_preview}
    parser.add_argument("names", nargs="*", help=f"subset of {', '.join(renderers)}")
    parser.add_argument("--rebuild", action="store_true", help="recompute statistics from the saved records")
    args = parser.parse_args()
    unknown = set(args.names) - set(renderers)
    if unknown:
        parser.error(f"unknown figures: {', '.join(sorted(unknown))}")
    data = figdata.load(args.rebuild)
    with plt.rc_context(RC):
        for name in args.names or FIGURES:
            for path in renderers[name](data):
                print(path)


if __name__ == "__main__":
    main()
