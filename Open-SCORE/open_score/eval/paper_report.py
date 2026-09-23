"""Paper report and figures from retained main0921 records, without running experiments.

All learned-policy intervals use the same three-seed, 95% Student-t summaries
as the formal report. Rendering never refreshes the inventory or loads models.
"""
from __future__ import annotations

from pathlib import Path
import statistics

from . import experiment as protocol
from .report0921 import HAD_GROUPS, LABELS, PROBE_LABELS, Results, _mean_zoom, _stats


_STYLES = {
    "regir": ("#004C78", "o", "-"),
    "regir_r1": ("#D55E00", "s", "--"),
    "refil": ("#009E73", "^", "-."),
    "alma": ("#CC79A7", "D", ":"),
    "transfqmix": ("#E69F00", "v", (0, (5, 2))),
    "rule_nv1": ("#666666", "x", (0, (2, 2))),
    "regir_norefil": ("#7B61A8", "D", "-."),
    "regir_nocount": ("#56B4E9", "^", ":"),
    "regir_last": ("#8C564B", "v", (0, (5, 2))),
    "refil_matched": ("#009E73", "P", (0, (3, 1, 1, 1))),
}
_METHODS = ("regir", "regir_r1", "refil", "alma", "transfqmix", "rule_nv1")
_ABLATIONS = ("regir", "regir_r1", "regir_norefil", "regir_nocount",
              "regir_last", "refil_matched")
_NAN = float("nan")


def _style_axes(ax, *, horizontal=False):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E6E8EB", linewidth=.55)
    if horizontal:
        ax.grid(axis="x", color="#EFF0F2", linewidth=.45)
    ax.tick_params(length=3, width=.65, color="#65707A")
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#8A939B")
        ax.spines[side].set_linewidth(.65)
    ax.margins(x=.055, y=.09)


def _title(ax, index, text):
    ax.set_title(f"({chr(97 + index)}) {text}", loc="left", pad=10,
                 fontsize=10, fontweight="medium")


def _shared_legend(fig, axes, *, ncol=6):
    entries = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            entries.setdefault(label, handle)
    if entries:
        fig.legend(entries.values(), entries.keys(), loc="lower center",
                   bbox_to_anchor=(.5, .01), ncol=min(ncol, len(entries)),
                   frameon=False, columnspacing=1.35, handlelength=2.6,
                   handletextpad=.55, fontsize=8.2)


def _save_figure(fig, directory, stem, plt):
    for extension in ("png", "pdf"):
        target = directory / f"{stem}.{extension}"
        temporary = directory / f".{stem}.pending.{extension}"
        fig.savefig(temporary, dpi=300, bbox_inches="tight", pad_inches=.06,
                    facecolor="white")
        temporary.replace(target)
    plt.close(fig)
    return stem


def _series(results, method, configs, *, arm="final"):
    if method == "rule_nv1":
        return [(statistics.mean(results.anchors[method, cfg].values()), 0., ())
                if len(results.anchors.get((method, cfg), {})) == 300
                and (method, cfg) not in results.conflicts else None
                for cfg in configs]
    return [results.summary("had", method, [cfg], arm) for cfg in configs]


def _draw_series(ax, x, summaries, style, label, *, emphasis=False, seeds=False):
    if not any(summaries):
        return False
    color, marker, linestyle = style
    # NaNs deliberately break both curves and intervals at incomplete cells.
    means = [s[0] if s else _NAN for s in summaries]
    errors = [s[1] if s else _NAN for s in summaries]
    line_options = dict(color=color, marker=marker, linestyle=linestyle,
                        linewidth=1.8 if emphasis else 1.25,
                        markersize=4.0 if emphasis else 3.5, markeredgewidth=.7,
                        label=label, zorder=6 if emphasis else 3)
    if any(s[2] for s in summaries if s):
        ax.errorbar(x, means, yerr=errors, elinewidth=.8, capsize=2, capthick=.7,
                    **line_options)
    else:
        # The Rule anchor is one shared scenario sample, not three model seeds.
        ax.plot(x, means, **line_options)
    if seeds:
        for seed_index in range(len(protocol.SEEDS)):
            ax.scatter(x, [s[2][seed_index] if s and s[2] else _NAN
                           for s in summaries], s=9, color=color,
                       alpha=.38, linewidths=0, zorder=2)
    return True


def _share_limits(axes, *, coordinate="y"):
    limits = [ax.get_ylim() if coordinate == "y" else ax.get_xlim() for ax in axes]
    lower = min(min(pair) for pair in limits)
    upper = max(max(pair) for pair in limits)
    for ax in axes:
        if coordinate == "y":
            ax.set_ylim(lower, upper)
        else:
            ax.set_xlim(lower, upper)


def _generalization(results, directory, plt):
    specifications = (
        ("Team-size transfer", tuple((n, n, 2) for n in (5, 10, 15, 20, 25, 30, 40, 50)),
         "Agents per team N (K = 2)", 0),
        ("Target transfer: N = 10", tuple((10, 10, k) for k in (2, 4, 6, 9, 12)),
         "Targets K", 2),
        ("Target transfer: N = 30", tuple((30, 30, k) for k in (2, 4, 6, 9, 12)),
         "Targets K", 2),
    )
    panels = []
    for title, configs, xlabel, dimension in specifications:
        data = {method: _series(results, method, configs) for method in _METHODS}
        if any(any(values) for values in data.values()):
            panels.append((title, configs, xlabel, dimension, data))
    if not panels:
        return None
    fig, grid = plt.subplots(2, len(panels), figsize=(3.35 * len(panels), 5.6),
                             squeeze=False, gridspec_kw={"height_ratios": (1, .6)})
    axes, zoom_axes = list(grid[0]), list(grid[1])
    fig.subplots_adjust(left=.07, right=.99, top=.92, bottom=.15,
                        wspace=.27, hspace=.47)
    for index, (ax, panel) in enumerate(zip(axes, panels)):
        title, configs, xlabel, dimension, data = panel
        x = [cfg[dimension] for cfg in configs]
        for method in _METHODS:
            _draw_series(ax, x, data[method], _STYLES[method], LABELS[method],
                         emphasis=method == "regir")
        _style_axes(ax)
        _title(ax, index, title)
        ax.set_xlabel(xlabel)
        ax.set_xticks((5, 10, 20, 30, 40, 50) if dimension == 0 else x)
        zoom = zoom_axes[index]
        _mean_zoom(ax, zoom, ylim=(0, 3.5))
        _style_axes(zoom)
        zoom.set_xlabel(xlabel)
        zoom.set_xticks(ax.get_xticks())
    axes[0].set_ylabel("Damage D ↓")
    _shared_legend(fig, axes)
    return _save_figure(fig, directory, "paper_generalization", plt)


def _learning_series(results, method):
    summaries, x = [], []
    thresholds = protocol.validation_thresholds("had")
    quota = len(protocol.validation_jobs("had", 1, 0))
    for point, threshold in enumerate(thresholds, 1):
        means, steps = [], []
        for seed in protocol.SEEDS:
            key = ("had", method, seed, point)
            records = results.validation.get(key, {})
            means.append(statistics.mean(records.values())
                         if len(records) == quota and key not in results.conflicts else None)
            steps.append(results.validation_steps.get(key, 0))
        summaries.append(_stats(means))
        x.append((statistics.mean(steps) if all(steps) else threshold) / 1e6)
    return x, summaries


def _learning(results, directory, plt):
    specifications = (
        ("Algorithm comparison", ("regir", "refil", "alma", "transfqmix")),
        ("Structure comparison", ("regir", "regir_r1", "regir_norefil",
                                  "regir_nocount", "regir_last")),
    )
    data = {method: _learning_series(results, method)
            for _, methods in specifications for method in methods}
    panels = [(title, methods) for title, methods in specifications
              if any(any(data[method][1]) for method in methods)]
    if not panels:
        return None
    fig, grid = plt.subplots(2, len(panels), figsize=(4.3 * len(panels), 5.8),
                             squeeze=False, gridspec_kw={"height_ratios": (1, .6)})
    axes, zoom_axes = list(grid[0]), list(grid[1])
    fig.subplots_adjust(left=.08, right=.99, top=.92, bottom=.16,
                        wspace=.22, hspace=.47)
    for index, (ax, (title, methods)) in enumerate(zip(axes, panels)):
        for method in methods:
            x, summaries = data[method]
            if not any(summaries):
                continue
            color, marker, linestyle = _STYLES[method]
            ax.plot(x, [s[0] if s else _NAN for s in summaries], color=color,
                    marker=marker, markevery=5, markersize=3,
                    linestyle=linestyle, linewidth=1.65 if method == "regir" else 1.1,
                    label=LABELS[method], zorder=5 if method == "regir" else 3)
            ax.fill_between(x, [s[0] - s[1] if s else _NAN for s in summaries],
                            [s[0] + s[1] if s else _NAN for s in summaries],
                            color=color, alpha=.105, linewidth=0, zorder=1)
        _style_axes(ax)
        _title(ax, index, title)
        ax.set_xlabel("Training environment steps (million)")
        ax.set_xlim(0, max(1.01, ax.get_xlim()[1]))
        ax.set_xticks((0, .2, .4, .6, .8, 1.0))
        zoom = zoom_axes[index]
        _mean_zoom(ax, zoom, ylim=(0, 2.5), xlim=(.4, 1.0))
        _style_axes(zoom)
        zoom.set_xlabel("Training environment steps (million)")
        zoom.set_xticks((.4, .6, .8, 1.0))
    axes[0].set_ylabel("Validation damage D ↓")
    _shared_legend(fig, axes, ncol=4)
    return _save_figure(fig, directory, "paper_learning", plt)


def _draw_point_interval(ax, position, summary, style, *, label=None, seeds=True):
    color, marker, _ = style
    ax.errorbar(summary[0], position, xerr=summary[1], color=color,
                fmt=marker, markersize=5, markeredgewidth=.8, elinewidth=1.15,
                capsize=3, capthick=.8, label=label, zorder=4)
    if seeds and summary[2]:
        offsets = (-.12, 0., .12)
        ax.scatter(summary[2], [position + offset for offset in offsets], s=15,
                   color=color, alpha=.45, edgecolors="white", linewidths=.3, zorder=3)


def _ablation(results, directory, plt):
    data = {group: {method: results.summary("had", method, HAD_GROUPS[group])
                    for method in _ABLATIONS}
            for group in ("N-OOD", "K-OOD", "联合 OOD")}
    groups = [group for group, values in data.items() if any(values.values())]
    methods = [method for method in _ABLATIONS
               if any(data[group][method] for group in groups)]
    if not groups:
        return None
    fig, axes = plt.subplots(1, len(groups), figsize=(3.15 * len(groups) + .45, 3.3),
                             squeeze=False, sharey=True)
    axes = list(axes[0])
    fig.subplots_adjust(left=.135, right=.99, top=.85, bottom=.19, wspace=.23)
    for index, (ax, group) in enumerate(zip(axes, groups)):
        for position, method in enumerate(methods):
            summary = data[group][method]
            if summary:
                _draw_point_interval(ax, position, summary, _STYLES[method])
        _style_axes(ax, horizontal=True)
        _title(ax, index, "Joint OOD" if group == "联合 OOD" else group)
        ax.set_xlabel("Damage D ↓")
        ax.set_yticks(range(len(methods)), [LABELS[method] for method in methods])
        ax.set_ylim(len(methods) - .5, -.5)
    _share_limits(axes, coordinate="x")
    return _save_figure(fig, directory, "paper_ablation", plt)


def _mechanism(results, directory, plt):
    depth = {}
    for cfg in protocol.DEPTH_CONFIGS:
        depth[cfg] = [results.summary("had", "regir", [cfg], f"depth:R{r}")
                      for r in range(1, 7)]
    readout = {name: results.summary("had", "regir", [(50, 50, 2)], f"readout:{name}")
               for name in protocol.READOUTS}
    panels = []
    if any(any(values) for values in depth.values()):
        panels.append("depth")
    if any(readout.values()):
        panels.append("readout")
    if not panels:
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(4.15 * len(panels), 3.5),
                             squeeze=False)
    axes = list(axes[0])
    fig.subplots_adjust(left=.085, right=.99, top=.85, bottom=.24, wspace=.43)
    for index, (ax, panel) in enumerate(zip(axes, panels)):
        if panel == "depth":
            styles = (("#004C78", "o", "-"), ("#D55E00", "s", "--"),
                      ("#009E73", "^", "-."))
            for cfg, style in zip(protocol.DEPTH_CONFIGS, styles):
                _draw_series(ax, list(range(1, 7)), depth[cfg], style,
                             f"N = {cfg[0]}")
            _style_axes(ax)
            _title(ax, index, "Frozen Full: execution depth")
            ax.set(xlabel="Execution rounds R (K = 2)", ylabel="Damage D ↓",
                   xticks=range(1, 7))
        else:
            names = [name for name in protocol.READOUTS if readout[name]]
            labels = {"learned": "Learned", "read1": "Read-1", "read2": "Read-2",
                      "read3": "Read-3", "read4": "Read-4", "uniform": "Uniform"}
            for position, name in enumerate(names):
                style = ("#004C78", "o", "-") if name == "learned" else ("#657A89", "s", "-")
                _draw_point_interval(ax, position, readout[name], style)
            _style_axes(ax, horizontal=True)
            _title(ax, index, "Frozen Full: readout at N = 50")
            ax.set(xlabel="Damage D ↓ (K = 2, R = 4)",
                   yticks=range(len(names)), yticklabels=[labels[name] for name in names],
                   ylim=(len(names) - .5, -.5))
            zoom = ax.inset_axes((.03, .63, .25, .30))
            for position, name in enumerate(("learned", "read1", "read2")):
                summary = readout.get(name)
                if summary:
                    zoom.scatter(summary[0], position, color="#004C78" if position == 0 else "#657A89",
                                 s=16)
                    zoom.annotate(f"{summary[0]:.2f}", (summary[0], position), xytext=(3, 0),
                                  textcoords="offset points", va="center", fontsize=6)
            zoom.set(xlim=(.5, 3.1), ylim=(2.5, -.5))
            zoom.set_yticks(range(3), ("Learned", "Read-1", "Read-2"))
            zoom.set_xticks((1, 2, 3))
            zoom.set_title("Means only; CI in main", loc="left", fontsize=6.5)
            zoom.tick_params(labelsize=6, pad=1)
            zoom.grid(axis="x", alpha=.2)
    _shared_legend(fig, axes, ncol=3)
    return _save_figure(fig, directory, "paper_mechanism", plt)


def _probes(results, directory, plt):
    controls = ("trained", "random_init", "shuffled_labels")
    data = {label: {control: [results.probe_summary((50, 50, 2), label, layer,
                                                   control=control, split="ood_confirm")
                              for layer in range(5)]
                    for control in controls} for label in PROBE_LABELS}
    labels = [label for label in PROBE_LABELS if any(any(series) for series in data[label].values())]
    if not labels:
        return None
    titles = {"nearest_target_distance": "Nearest-target distance",
              "nearest_other_defender_distance": "Nearest-other-defender distance",
              "nearest_target_region_imbalance": "Target-region imbalance"}
    styles = {"trained": ("#004C78", "o", "-"),
              "random_init": ("#009E73", "^", "--"),
              "shuffled_labels": ("#D55E00", "s", ":")}
    names = {"trained": "Trained", "random_init": "Random initialization",
             "shuffled_labels": "Shuffled labels"}
    fig, axes = plt.subplots(1, len(labels), figsize=(3.45 * len(labels), 3.35),
                             squeeze=False)
    axes = list(axes[0])
    fig.subplots_adjust(left=.075, right=.99, top=.85, bottom=.25, wspace=.24)
    for index, (ax, label) in enumerate(zip(axes, labels)):
        ax.axhline(0., color="#9CA4AA", linewidth=.75, linestyle="--", zorder=0)
        for control in controls:
            _draw_series(ax, list(range(5)), data[label][control], styles[control],
                         names[control], emphasis=control == "trained")
        _style_axes(ax)
        _title(ax, index, titles[label])
        ax.set(xlabel="Representation layer", xticks=range(5),
               xticklabels=[f"H{layer}" for layer in range(5)])
    axes[0].set_ylabel("Confirmation-set R² ↑")
    _share_limits(axes)
    _shared_legend(fig, axes, ncol=3)
    return _save_figure(fig, directory, "paper_probes", plt)


def render_figures(results: Results, output) -> list[str]:
    """Write 300-dpi PNG and vector PDF pairs; return generated figure stems.

    Missing seed/configuration cells remain NaN, so lines cannot bridge them.
    Entirely unavailable panels are omitted. Axes include every confidence
    interval endpoint, including negative damage bounds and negative probe R².
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    directory = Path(output) / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    style = {"font.family": "DejaVu Sans", "font.size": 9,
             "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
             "axes.spines.top": False, "axes.spines.right": False,
             "axes.facecolor": "white", "figure.facecolor": "white",
             "text.color": "#202832", "axes.labelcolor": "#202832",
             "xtick.color": "#35414C", "ytick.color": "#35414C",
             "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300}
    with plt.rc_context(style):
        stems = [render(results, directory, plt) for render in
                 (_generalization, _learning, _ablation, _mechanism, _probes)]
    return [stem for stem in stems if stem is not None]


def _paper_markdown(results, created_at):
    """A concise experimental section; every reported number uses Results."""
    from .report0921 import HAD_GROUPS, LABELS, _table
    from . import experiment as protocol

    groups = ("ID", "N-OOD", "K-OOD", "联合 OOD")
    def mean(method, group):
        value = results.summary("had", method, HAD_GROUPS[group])
        return value[0] if value else float("nan")

    def cell(method, configs, arm="final"):
        return results.cell("had", method, configs, arm).replace("（共同参照）", "†")

    def figure(stem, number, title, caption):
        return [f"![{number} {title}](figures/{stem}.png)", "",
                f"**{number}｜{title}。** {caption} [矢量 PDF](figures/{stem}.pdf)。", ""]

    baseline_methods = ("regir", "regir_r1", "refil", "b2_qmix_atten", "dcg", "spectra",
                        "alma", "transfqmix", "refil_matched", "rule_nv1")
    reduction = {g: 100 * (1 - mean("regir", g) / mean("refil", g)) for g in groups}
    lines = [
        "# 正文", "", "## 1. 实验设置", "",
        "我们考察策略仅在小规模任务上训练后，能否直接泛化至更多智能体、更多保护目标以及二者同时增加的场景，"
        "并通过结构消融与冻结模型干预分析循环表征的作用。当前已完成的实验来自 HAD 协同防御环境："
        "红方智能体拦截按固定规则行动的蓝方攻击机，以减小保护目标的整局累计损伤 $D$；该指标越低越好。", "",
        "训练时，双方人数相等，$N\\in\\{4,6,8,10\\}$，目标数 $K\\in\\{1,2,3\\}$，"
        "二者独立均匀采样。每次训练使用 $10^6$ 个环境物理步，测试冻结最终模型，人数最多扩展至 $50\\mathrm{v}50$，"
        "目标数最多扩展至 12。测试分为分布内（ID）、人数外推（N-OOD）、目标数外推（K-OOD）和联合外推四组；"
        "$5\\mathrm{v}5$ 属于人数插值，$10\\mathrm{v}10,K=2$ 属于训练支持，两者均不计入 OOD 汇总。", "",
        "比较方法包括 REFIL、QMIX-Atten、DCG、SPECTra、ALMA、TransfQMix 和规则策略。"
        "本文方法记为 Full，训练深度在 1–4 轮间采样，正式测试固定为 4 轮；Single 为从头训练的一轮版本，"
        "REFIL-matched 为容量匹配对照。学习方法均使用三个训练种子，每个种子、每个测试配置评估 300 局。"
        "表格报告跨种子均值及 95% $t$ 置信区间半宽，分组内配置等权；缺少完整三种子评估的结果记为“—”。", "",
        "## 2. 跨规模泛化", "",
        "表 1 汇总各方法在四类测试分布上的表现。Full 相对 REFIL 的分组平均损伤分别降低 "
        f"{reduction['ID']:.1f}%、{reduction['N-OOD']:.1f}%、{reduction['K-OOD']:.1f}% 和 {reduction['联合 OOD']:.1f}%。"
        "这四组中，Full 在三个对应训练种子上的损伤也均低于 REFIL，说明性能改善不仅来自单个种子的较好结果。"
        "人数外推中的差距最为突出：Full 的平均损伤为 "
        f"{mean('regir', 'N-OOD'):.3f}，REFIL 为 {mean('refil', 'N-OOD'):.3f}。", "",
        "**表 1｜HAD 主比较。** $D$ 越低越好；加粗表示本表学习方法中各列最低的平均损伤，不代表统计显著性。"
        "† 为同一公共场景集上的规则参照，不具有训练种子置信区间。", "",
    ]
    minima = {g: min(mean(m, g) for m in baseline_methods if m != "rule_nv1"
                     and results.summary("had", m, HAD_GROUPS[g])) for g in groups}
    rows = []
    for method in baseline_methods:
        values = []
        for group in groups:
            text = cell(method, HAD_GROUPS[group])
            if method != "rule_nv1" and mean(method, group) == minima[group]:
                text = f"**{text}**"
            values.append(text)
        rows.append([LABELS[method], *values])
    lines += _table(["方法", "ID", "人数外推", "目标数外推", "联合外推"], rows)
    lines += figure("paper_generalization", "图 1", "人数与目标数变化下的泛化表现",
                     "(a) 固定 $K=2$ 改变双方人数；(b–c) 分别固定双方人数为 10 和 30，改变目标数。"
                     "上排曲线为三个训练种子的均值，误差线为 95% $t$ 置信区间；下排仅放大低损伤均值，"
                     "完整数据与置信区间以上排为准；Rule 仅显示公共参照均值。"
                     "$N=5$ 为人数插值，$N=10,K=2$ 为训练支持；缺失配置处断开曲线")
    lines += [
        "图 1 表明，随着人数或目标数增加，Full 的损伤整体保持在 REFIL 与 ALMA 之下；"
        "容量匹配的 REFIL 也未消除表 1 中的差距。不过，循环版本并未在所有外推轴上优于 Single："
        f"Single 的人数外推损伤为 {mean('regir_r1', 'N-OOD'):.3f}，低于 Full 的 {mean('regir', 'N-OOD'):.3f}。"
        "Full 在 ID、目标数外推和联合外推上的均值较低，但这些比较仍存在明显种子差异，"
        "不能据此断言多轮训练稳定优于单轮训练。", "",
        "## 3. 结构消融", "",
        "为区分循环结构、局部分支和数量条件的贡献，我们比较 Single、No-REFIL、No-count 与 Last。"
        "No-REFIL 同时去掉原 REFIL 局部注意力路径和想象分组训练；No-count 取消循环查询中的数量编码注入；"
        "Last 在训练和测试时均只保留最后一轮读出。上述变体均独立训练，结果见图 2。", "",
    ]
    lines += figure("paper_ablation", "图 2", "不同泛化轴上的结构消融",
                     "点与横向误差线表示分组均值及 95% $t$ 置信区间，小点表示三个训练种子。"
                     "三个面板分别对应人数、目标数和联合外推")
    lines += [
        f"Last 在人数、目标数和联合外推上的损伤分别为 {mean('regir_last', 'N-OOD'):.3f}、"
        f"{mean('regir_last', 'K-OOD'):.3f} 和 {mean('regir_last', '联合 OOD'):.3f}，均高于 Full，"
        "与保留跨轮读出有助于当前任务的解释一致。另一方面，No-REFIL 的部分分组均值与 Full 相近，"
        "表明现有实验不能确立该局部分支及想象分组训练的必要性，且这一消融同时改变了两个因素。", "",
        "数量编码也未表现出一致收益。No-count 在目标数和联合外推上的损伤分别为 "
        f"{mean('regir_nocount', 'K-OOD'):.3f} 和 {mean('regir_nocount', '联合 OOD'):.3f}，"
        f"低于 Full 的 {mean('regir', 'K-OOD'):.3f} 和 {mean('regir', '联合 OOD'):.3f}，"
        "且三个对应训练种子均呈现这一方向。因而，当前性能优势应归于整体方法在本实验协议下的表现，"
        "不能解释为每个组件都提供了独立且稳定的增益。附录 C–D 进一步分析执行深度、读出干预及关系可读性。", "",
        "# 附录", "", "## A. 评估协议与统计口径", "",
        "**表 A1｜HAD 测试配置。** 各分组互不重叠，共 24 个配置；表 1 汇总其中四个主要分组。", "",
    ]
    lines += _table(["分组", "配置（双方人数相等）", "配置数"], [
        ["ID", "$N=4,6,8$ 时 $K=2$；$N=10$ 时 $K=3$", 4],
        ["训练支持", "$N=10,K=2$", 1],
        ["人数插值", "$N=5,K=2$", 1],
        ["人数外推", "$N=15,20,25,30,40,50$，$K=2$", 6],
        ["目标数外推", "$N=10$，$K=4,6,9,12$", 4],
        ["联合外推", "$N=15,20,30$ 与 $K=4,6$ 的组合；另加 $N=30,K=9,12$", 8],
    ])
    lines += [
        "终评仅使用达到训练预算的最终模型，不依据外推表现选取检查点。各方法共享 300 个评估场景种子"
        "（9000–9299），训练种子为 0、1、2。对每个训练种子，先计算每个配置的回合平均损伤，"
        "再对该分组内配置等权平均，最后跨三个训练种子计算均值。95% 区间半宽为 "
        "$4.30265\\,s/\\sqrt{3}$，其中 $s$ 为三个种子汇总值的样本标准差；不将评估回合视为独立训练重复。"
        "对称区间可能跨过零，图中保留其完整范围。", "",
        "Full 的循环表征维度为 64，注意力头数为 4，前馈层宽度为 128；经验池包含 5000 个回合，"
        "批大小为 32，学习率为 $5\\times10^{-4}$，折扣因子为 0.99。"
        "比较方法保留各自登记的网络与优化设置，不假定相同训练步数等价于相同计算成本。", "",
        f"本次整理时间为 {created_at}。数值来自本实验目录中的"
        "[评估记录](episodes.csv)、[探针结果](probe/results.csv)和[实验配置](experiment.json)，"
        "仅纳入与已登记最终模型匹配、场景配额完整的结果。", "",
        "## B. 学习过程与种子差异", "",
    ]
    lines += figure("paper_learning", "图 A1", "训练期间的分布内验证",
                     "左图比较 Full 与主要学习基线，右图比较结构变体。每个验证点在四个 ID 配置上各评估 25 局；"
                     "上排曲线与阴影为三个训练种子的均值及 95% $t$ 置信区间，未作平滑；"
                     "下排仅放大后 0.4–1.0M 步的低损伤均值，完整区间以上排为准")
    lines += [
        "为便于区分总体均值与训练随机性，表 A2 列出 Full、Single 与 REFIL 的逐种子分组结果。"
        "Full 相比 Single 的四组结果均呈现相同分化：种子 0 更好，而种子 1、2 更差。"
        "因此，主表中部分均值优势并不意味着优势在所有训练重复中成立。", "",
        "**表 A2｜逐训练种子的平均损伤。** 每个数值已在对应分组内对配置等权平均。", "",
    ]
    rows = [[LABELS[m], s, *(f"{results.seed_mean('had', m, s, HAD_GROUPS[g]):.3f}" for g in groups)]
            for m in ("regir", "regir_r1", "refil") for s in protocol.SEEDS]
    lines += _table(["方法", "种子", "ID", "人数外推", "目标数外推", "联合外推"], rows)
    lines += ["## C. 执行深度与读出干预", "",
              "冻结同一个 Full 模型，仅改变推理轮数，可将执行计算量的影响与重训效应分开。"
              "图 A2(a) 中，$30\\mathrm{v}30$ 和 $50\\mathrm{v}50$ 的平均损伤均在两轮时最低，"
              "超过四轮后也未观察到持续改善。例如，$50\\mathrm{v}50,K=2$ 下，"
              "两轮与四轮的平均损伤分别为 "
              f"{results.summary('had', 'regir', [(50, 50, 2)], 'depth:R2')[0]:.3f} 和 "
              f"{results.summary('had', 'regir', [(50, 50, 2)], 'depth:R4')[0]:.3f}。"
              "这些结果不支持“规模越大便需要更多循环轮数”的单调解释；正文仍按预先设定的四轮模型报告。", ""]
    lines += figure("paper_mechanism", "图 A2", "冻结 Full 的执行深度与读出干预",
                     "(a) 同一最终模型在 1–6 轮下的表现；(b) 固定计算四轮，在 $50\\mathrm{v}50,K=2$ 下替换读出方式。"
                     "误差表示三个训练种子的 95% $t$ 置信区间；(b) 内嵌小图仅展示前三种读出的低损伤均值，"
                     "完整区间以主图为准。Read-4 是冻结后的干预，不同于图 2 中重新训练的 Last")
    lines += ["**表 A3｜固定四轮时的读出干预。** Learned 使用原学习权重，Read-$r$ 只读取第 $r$ 轮，"
              "Uniform 对四轮读出等权平均。", ""]
    readout_names = ("Learned", "Read-1", "Read-2", "Read-3", "Read-4", "Uniform")
    lines += _table(["读出方式", "$10\\mathrm{v}10,K=2$", "$50\\mathrm{v}50,K=2$", "$30\\mathrm{v}30,K=12$"],
                    [[name, *(cell("regir", [cfg], f"readout:{arm}") for cfg in protocol.READOUT_CONFIGS)]
                     for name, arm in zip(readout_names, protocol.READOUTS)])
    lines += [
        "Learned 在三个配置上的平均损伤均低于其他读出方式；仅使用较晚轮次或改用均匀权重时，"
        "部分种子的性能下降很大。这说明已训策略依赖其学习到的读取方式。"
        "但冻结干预也改变了策略头接收的输入分布，因而不能据此断言晚轮表示本身无效，"
        "或将性能下降直接解释为轮次间的因果分工。四轮和 Learned 复用相同正式评估；"
        "Read-1 与一轮执行只在已验证等价的相交场景中复用，不增加样本量。", "",
        "## D. 关系可读性与控制实验", "",
        "关系探针使用五个配置的 500 条公共轨迹，Full 和 REFIL 各提供一半。"
        "对三个 Full 最终模型分别重建相同状态历史，在 $H_0$–$H_4$ 上拟合岭回归，针对观察者可见敌人，预测其到最近可见目标的距离、"
        "到除观察者外最近可见防守者的距离，以及最近目标分区内可见敌我数量的归一化差值 "
        "$(n_B-n_R)/(n_B+n_R)$（防守者计数排除观察者）。"
        "ID 数据按场景划分为拟合、验证和测试集，标准化与正则强度仅依据 ID 拟合及验证集确定；"
        "OOD 确认集不参与选择，并同时报告随机初始化和置乱训练标签控制。", "",
    ]
    lines += figure("paper_probes", "图 A3", "$50\\mathrm{v}50,K=2$ 确认集上的关系探针",
                     "三个面板依次为目标距离、其他防守者距离及目标分区人数不平衡。"
                     "曲线为三个模型的平均测试 $R^2$，误差线为 95% $t$ 置信区间；负值按原值保留。"
                     "随机初始化网络和置乱标签使用相同数据划分与拟合流程")
    lines += [
        "目标距离的可读性从 $H_0$ 的 0.697 上升至 $H_1$ 的 0.858，但随后下降至 $H_4$ 的 0.817，"
        "并未随轮次持续增强。其他防守者距离的 $H_4$ 得分为 0.597，随机初始化控制为 0.655；"
        "人数不平衡的 $H_4$ 得分为 −0.032，随机初始化控制反而达到 0.231。"
        "因此，当前结果支持部分几何信息可从循环表征中线性读取，"
        "尚不能证明晚轮持续形成更强、或专属于训练所得策略的关系表征。", "",
        "## E. 结果覆盖范围", "",
    ]
    completed = {m: sum(results.summary("had", m, [cfg]) is not None for cfg in protocol.configs("had"))
                 for m in protocol.methods("had")}
    complete_methods = sum(n == len(protocol.configs("had")) for n in completed.values())
    pending = [LABELS[m] for m, count in completed.items() if count == 0]
    scope = f"当前 {complete_methods} 个 HAD 学习方法已完成全部 24 个配置的三种子终评。"
    partial = [f"{LABELS[m]} 完成 {count}/24 个配置" for m, count in completed.items() if 0 < count < 24]
    if partial:
        scope += "；".join(partial) + "；任一配置缺少完整三种子评估时，该分组均值不予报告。"
    if pending:
        scope += "、".join(pending) + " 尚无完整三种子终评结果。"
    smac_count = sum(results.summary("smacv2", m, [cfg]) is not None
                     for m in protocol.SMAC_METHODS for cfg in protocol.configs("smacv2"))
    if smac_count == 0:
        scope += "SMACv2 尚无完整正式结果，本文结论限于 HAD。"
    else:
        scope += f"SMACv2 已有 {smac_count} 个完整正式格，其跨域分析仍需另行核对。"
    if not results.costs:
        scope += "参数与时延比较尚无合格测量，因此不作计算效率优于基线的结论。"
    lines += [scope, ""]
    return "\n".join(lines).rstrip() + "\n"


def render_report(output, *, run="train"):
    """Create the requested paper edition without refreshing training state."""
    import json
    from datetime import datetime
    from pathlib import Path
    from .report0921 import Results

    if run != "train":
        raise ValueError("The paper edition uses the registered train run only")
    output = Path(output)
    metadata = json.loads((output / "experiment.json").read_text(encoding="utf-8"))
    if metadata.get("profile") != "main0921":
        raise ValueError("The paper edition is specific to the main0921 experiment")
    inventory = json.loads((output / "inventory.json").read_text(encoding="utf-8"))
    created_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    results = Results(output, inventory, metadata)
    text = _paper_markdown(results, created_at)
    render_figures(results, output)
    target = output / "实验报告_论文版.md"
    temporary = target.with_suffix(".md.pending")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(target)
    return target
