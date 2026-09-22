"""One main0921 report: complete three-seed statistics and explicit pending data.

The parent refreshes the inventory from actual final metadata, then renders
retained records. It does not construct models or run environments. Main-text
figures and the full appendix share the same aggregates.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import statistics

from . import experiment as protocol
from ..utils.logging import iter_records

LABELS = {
    "regir": "Full", "regir_r1": "Single", "refil": "REFIL",
    "b2_qmix_atten": "QMIX-Atten", "dcg": "DCG", "spectra": "SPECTra",
    "alma": "ALMA", "transfqmix": "TransfQMix", "rule_nv1": "Rule",
    "random": "Random", "regir_norefil": "No-REFIL", "regir_nocount": "No-count",
    "regir_last": "Last", "refil_matched": "REFIL-matched",
    "regir_fixed4": "Fixed4", "regir_untied4": "Untied4", "regir_kv0": "KV0",
}
BODY_HAD = ("regir", "regir_r1", "refil", "transfqmix", "alma", "rule_nv1")
TABLE_HAD = ("regir", "regir_r1", "refil", "b2_qmix_atten", "dcg", "spectra",
             "alma", "transfqmix", "refil_matched", "rule_nv1")
ABLATIONS = ("regir", "regir_r1", "regir_norefil", "regir_nocount", "regir_last",
             "refil_matched", "regir_fixed4", "regir_untied4", "regir_kv0")
HAD_GROUPS = {
    "ID": ((4, 4, 2), (6, 6, 2), (8, 8, 2), (10, 10, 3)),
    "训练范围支持点": ((10, 10, 2),),
    "人数插值": ((5, 5, 2),),
    "N-OOD": tuple((n, n, 2) for n in (15, 20, 25, 30, 40, 50)),
    "K-OOD": tuple((10, 10, k) for k in (4, 6, 9, 12)),
    "联合 OOD": tuple((n, n, k) for n in (15, 20, 30) for k in (4, 6))
    + ((30, 30, 9), (30, 30, 12)),
}
PROBE_LABELS = ("nearest_target_distance", "nearest_other_defender_distance",
                "nearest_target_region_imbalance")
T95 = 4.302652729911275


def _json(path, default):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _cfg(value):
    if isinstance(value, str):
        if value.startswith("{") or value.startswith("["):
            return _cfg(json.loads(value))
        teams, _, targets = value.replace(" ", "_").partition("_K")
        red, blue = teams.split("v")
        return int(red), int(blue), int(targets or 0)
    if isinstance(value, dict):
        return tuple(int(value[k]) for k in ("N_R", "N_B", "K"))
    return tuple(map(int, value))


def _label(config):
    r, b, k = config
    return f"{r}v{b}" + (f" K{k}" if k else "")


def _finite(value):
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _num(value, digits=3):
    return f"{float(value):.{digits}f}" if _finite(value) else "—"


def _escape(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers, rows):
    lines = ["| " + " | ".join(map(_escape, headers)) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(map(_escape, row)) + " |" for row in rows)
    return lines + [""]


def _stats(values):
    if len(values) != 3 or any(not _finite(v) for v in values):
        return None
    values = tuple(map(float, values))
    return statistics.mean(values), T95 * statistics.stdev(values) / math.sqrt(3), values


class Results:
    def __init__(self, output, inventory, metadata):
        self.output, self.inventory, self.metadata = Path(output), inventory, metadata
        self.checkpoints = {}
        for info in inventory.get("checkpoints", []):
            key = (info.get("env", "had"), info.get("method"), int(info.get("seed", -1)))
            if (key[0] in ("had", "smacv2") and key[1] in protocol.methods(key[0])
                    and key[2] in protocol.SEEDS and info.get("checkpoint_id")
                    and int(info.get("t_env", 0)) >= protocol.budget(key[0])):
                self.checkpoints[key] = info
        self.episodes = defaultdict(dict)
        self.validation = defaultdict(dict)
        self.validation_steps = {}
        self.anchors = defaultdict(dict)
        self.conflicts = set()
        self.excluded = Counter()
        self.sources = Counter()
        self.progress = {}
        self.probes, self.costs = [], []
        self.measurement_gpu = metadata.get("resources", {}).get("measurement_gpu")
        self.probe_metadata = {}
        self._read_episodes()
        for row in iter_records(output, "progress", run="train"):
            self.progress[(row.get("env", "had"), row.get("method"), row.get("seed"),
                           row.get("phase"), row.get("arm"),
                           _cfg(row["config"]) if row.get("phase") == "timing" and row.get("config") else None)] = row
        costs = {}
        for row in iter_records(output, "timing", run="train"):
            info = self.checkpoints.get((row.get("env", "had"), row.get("method"), row.get("seed")))
            if (info and row.get("checkpoint_id") == info["checkpoint_id"]
                    and row.get("env") == "had" and row.get("method") in protocol.COST_METHODS
                    and row.get("arm") == "cost" and row.get("device") == "cuda:0"
                    and type(self.measurement_gpu) is int and self.measurement_gpu in (0, 1)
                    and row.get("physical_gpu") == self.measurement_gpu
                    and row.get("n_steps") == 200 and row.get("config")
                    and _cfg(row["config"]) in ((10, 10, 2), (50, 50, 2))
                    and all(_finite(row.get(k)) for k in ("ms_per_step", "p25_ms", "p75_ms", "p95_ms",
                                                         "actor_params", "training_params"))):
                costs[row["checkpoint_id"], _cfg(row["config"])] = row
        self.costs = list(costs.values())
        path = self.output / "probe" / "results.csv"
        if path.exists():
            for seed in protocol.SEEDS:
                marker = self.output / "probe" / f"seed_{seed}.complete.json"
                if not marker.exists():
                    marker = self.output / "probe" / f"seed_{seed}.json"
                summary = _json(marker, {})
                info = self.checkpoints.get(("had", "regir", seed), {})
                if (summary.get("status") in ("complete", "completed") and info.get("checkpoint_id")
                        and summary.get("checkpoint_id") == info["checkpoint_id"]):
                    self.probe_metadata[seed] = summary
            with path.open(encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    if not all(row.get(k) is not None for k in ("model_seed", "round", "config", "r2")):
                        continue
                    row = dict(row, model_seed=int(row["model_seed"]), round=int(row["round"]),
                               config=_cfg(row["config"]), split=row.get("split", "").lower())
                    if row["model_seed"] not in self.probe_metadata:
                        self.excluded["探针缺合格checkpoint完成记录"] += 1
                        continue
                    row["r2"] = float(row["r2"]) if _finite(row["r2"]) else None
                    self.probes.append(row)

    def _put(self, mapping, key, episode, value):
        old = mapping[key].get(episode)
        if old is not None and old != value:
            self.conflicts.add(key)
        mapping[key][episode] = value

    def _read_episodes(self):
        expected_validation = {
            env: {(tuple(job["config"].values()), job["episode_seed"])
                  for job in protocol.validation_jobs(env, 1, 0)}
            for env in ("had", "smacv2")}
        for row in iter_records(self.output, "episodes", run="train"):
            env, method = row.get("env", "had"), row.get("method")
            if env not in expected_validation or not row.get("config"):
                self.excluded["非协议环境/缺配置"] += 1
                continue
            cfg, episode = _cfg(row["config"]), int(row["episode_seed"])
            phase, seed = row.get("phase"), int(row.get("seed") or 0)
            metric = row.get("D" if env == "had" else "battle_won")
            if not _finite(metric):
                self.excluded["指标非有限值/缺失"] += 1
                continue
            metric = float(metric)
            if env == "smacv2" and metric not in (0.0, 1.0):
                self.excluded["battle_won 非0/1"] += 1
                continue
            if phase == "anchor" and env == "had" and method in ("random", "rule_nv1"):
                if cfg in protocol.configs("had") and 9000 <= episode < 9300:
                    self._put(self.anchors, (method, cfg), episode, metric)
                continue
            if method not in protocol.methods(env) or seed not in protocol.SEEDS:
                self.excluded["非登记方法/种子"] += 1
                continue
            if phase == "train_eval":
                point = int(row.get("eval_point") or 0)
                if 1 <= point <= (50 if env == "had" else 40) and (cfg, episode) in expected_validation[env]:
                    key = (env, method, seed, point)
                    self._put(self.validation, key, (cfg, episode), metric)
                    self.validation_steps[key] = max(self.validation_steps.get(key, 0), int(row.get("t_env") or 0))
                else:
                    self.excluded["验证配置/随机流/点位不符"] += 1
                continue
            if phase not in ("final_eval", "depth_eval", "readout_eval"):
                continue
            info = self.checkpoints.get((env, method, seed))
            if not info or row.get("checkpoint_id") != info["checkpoint_id"]:
                self.excluded["无合格final身份/身份不符"] += 1
                continue
            if int(row.get("t_env") or 0) != int(info["t_env"]):
                self.excluded["final步数不符"] += 1
                continue
            base = 9000 if env == "had" else 110000
            if episode not in range(base, base + 300):
                self.excluded["终评随机流不符"] += 1
                continue
            arm = row.get("arm")
            accepted = ((phase == "final_eval" and arm == "final" and cfg in protocol.configs(env))
                        or (env == "had" and method == "regir" and phase == "depth_eval"
                            and arm in {f"depth:R{d}" for d in range(1, 7)} and cfg in protocol.DEPTH_CONFIGS)
                        or (env == "had" and method == "regir" and phase == "readout_eval"
                            and arm in {f"readout:{r}" for r in protocol.READOUTS} and cfg in protocol.READOUT_CONFIGS))
            if not accepted:
                self.excluded["阶段/实验臂/配置不符"] += 1
                continue
            self._put(self.episodes, (env, method, seed, arm, cfg), episode, metric)
            self.sources[row.get("source_version") or row.get("version") or "未标识"] += 1

    def values(self, env, method, seed, cfg, arm="final"):
        key = (env, method, seed, arm, cfg)
        if key in self.conflicts:
            return {}
        rows = self.episodes.get(key, {})
        if (env == "had" and method == "regir" and arm in ("depth:R1", "readout:read1")
                and self.metadata.get("mechanisms", {}).get("read1_equivalence_verified")):
            other = "readout:read1" if arm == "depth:R1" else "depth:R1"
            other_key = (env, method, seed, other, cfg)
            source = self.episodes.get(other_key, {})
            if other_key in self.conflicts or any(ep in rows and rows[ep] != value for ep, value in source.items()):
                self.conflicts.add(key)
                return {}
            rows = source | rows
        if len(rows) == 300:
            return rows
        alias = None
        if env == "had" and method == "regir":
            if arm in ("depth:R4", "readout:learned"):
                alias = "final"
        if alias:
            source = self.values(env, method, seed, cfg, alias)
            if any(ep in rows and rows[ep] != value for ep, value in source.items()):
                self.conflicts.add(key)
                return {}
            rows = source | rows
        return rows

    def seed_mean(self, env, method, seed, configs, arm="final"):
        means = []
        for cfg in configs:
            values = self.values(env, method, seed, cfg, arm)
            if len(values) != 300:
                return None
            means.append(statistics.mean(values.values()))
        return statistics.mean(means) if means else None

    def summary(self, env, method, configs, arm="final"):
        return _stats([self.seed_mean(env, method, seed, configs, arm) for seed in protocol.SEEDS])

    def cell(self, env, method, configs, arm="final", *, points=False):
        if method in ("random", "rule_nv1"):
            if any(len(self.anchors.get((method, cfg), {})) != 300 or (method, cfg) in self.conflicts for cfg in configs):
                return "—"
            return _num(statistics.mean(statistics.mean(self.anchors[method, cfg].values()) for cfg in configs)) + "（共同参照）"
        result = self.summary(env, method, configs, arm)
        per_seed = [self.seed_mean(env, method, seed, configs, arm) for seed in protocol.SEEDS]
        prefix = f"{_num(result[0])} ± {_num(result[1])}" if result else "—"
        if points:
            prefix += " [" + "; ".join(_num(v) for v in per_seed) + "]"
            if not result:
                prefix += " n(0/1/2)=" + "/".join(str(sum(len(self.values(env, method, s, c, arm)) for c in configs))
                                                 for s in protocol.SEEDS) + f"; quota={300 * len(configs)}"
        return prefix

    def probe_summary(self, cfg, label, layer, control="trained", split="ood_confirm"):
        values = {row["model_seed"]: row["r2"] for row in self.probes
                  if row["config"] == cfg and row.get("label") == label and row["round"] == layer
                  and row.get("control") == control and row.get("split") == split
                  and row["model_seed"] in protocol.SEEDS}
        return _stats([values.get(seed) for seed in protocol.SEEDS])


def _save(fig, directory, stem, plt):
    for extension in ("png", "pdf"):
        target = directory / f"{stem}.{extension}"
        temporary = directory / f".{stem}.pending.{extension}"
        fig.savefig(temporary, dpi=180, bbox_inches="tight")
        temporary.replace(target)
    plt.close(fig)


def _empty(ax, message="待完成 / Pending"):
    ax.text(.5, .5, message, transform=ax.transAxes, ha="center", va="center", color="#707070")


def _curve(ax, results, env, method, configs, x, color, *, arm="final", linestyle="-", label=None):
    if method == "rule_nv1":
        y = [statistics.mean(results.anchors[method, cfg].values())
             if len(results.anchors.get((method, cfg), {})) == 300 and (method, cfg) not in results.conflicts
             else float("nan") for cfg in configs]
        if any(math.isfinite(v) for v in y):
            ax.plot(x, y, color=color, linestyle=linestyle, label=label or LABELS[method])
            return True
        return False
    summaries = [results.summary(env, method, [cfg], arm) for cfg in configs]
    y = [s[0] if s else float("nan") for s in summaries]
    errors = [s[1] if s else 0 for s in summaries]
    if not any(summaries):
        return False
    ax.errorbar(x, y, yerr=errors, color=color, marker="o", markersize=3,
                linestyle=linestyle, linewidth=1.2, capsize=2, label=label or LABELS[method])
    for si in range(3):
        points = [s[2][si] if s else float("nan") for s in summaries]
        ax.scatter(x, points, s=8, color=color, alpha=.35)
    return True


def _figures(results, output):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt, font_manager
    from matplotlib.patches import FancyBboxPatch
    import numpy as np
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                        "axes.spines.right": False, "pdf.fonttype": 42})
    directory = output / "figures"
    directory.mkdir(exist_ok=True)
    palette = plt.get_cmap("tab10")
    colors = {m: palette(i) for i, m in enumerate(dict.fromkeys(BODY_HAD + protocol.SMAC_METHODS))}

    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.set(xlim=(0, 12), ylim=(0, 4))
    ax.axis("off")
    boxes = [(.1, 2.3, 1.6, .8, "Entities + masks\n允许的实体集合"),
             (2.1, 2.3, 1.4, .8, "Encoder\nH0"),
             (4.0, 2.3, 2.4, .8, "Shared entity update F\nH1 → H2 → H3 → H4"),
             (6.9, 2.3, 2.1, .8, "Agent readouts\nu1, u2, u3, u4"),
             (9.5, 2.3, 2.0, .8, "Learned depth fusion\n" + r"$\sum_r \alpha_r u_r$"),
             (2.1, .7, 2.2, .8, "Local REFIL path"),
             (5.1, .7, 2.3, .8, "Previous agent hidden\nh(t−1)"),
             (9.5, .7, 2.0, .8, "GRU + action values\n" + r"$Q_i, h(t)$")]
    for x, y, w, h, label in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.05", linewidth=.9,
                                   edgecolor="#366480", facecolor="#edf4f8"))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9)
    def arrow(a, b, **kw):
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color="#4b5861", lw=1.2, **kw))
    for a, b in [((1.75, 2.7), (2.04, 2.7)), ((3.55, 2.7), (3.94, 2.7)),
                 ((6.45, 2.7), (6.84, 2.7)), ((9.05, 2.7), (9.44, 2.7)),
                 ((10.5, 2.25), (10.5, 1.56)), ((4.35, 1.1), (9.44, 1.1)),
                 ((6.25, 1.55), (7.4, 2.24))]:
        arrow(a, b)
    arrow((2.8, 2.25), (2.8, 1.56))
    arrow((3.3, 3.15), (7.8, 3.15), connectionstyle="arc3,rad=-.2")
    ax.text(5.5, 3.7, "Own initial representation contributes to each read query", ha="center", fontsize=8)
    ax.text(6, .1, "Actor schematic · Full: random R=1–4 during training, R=4 at final evaluation · mixer/auxiliary losses used during training",
            ha="center", fontsize=8)
    _save(fig, directory, "fig1_architecture", plt)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for env, ax in (("had", axes[0, 0]), ("smacv2", axes[1, 0])):
        shown = False
        for method in BODY_HAD if env == "had" else protocol.SMAC_METHODS:
            if method == "rule_nv1":
                cfgs = HAD_GROUPS["ID"]
                if all(len(results.anchors.get((method, cfg), {})) == 300 for cfg in cfgs):
                    value = statistics.mean(statistics.mean(results.anchors[method, c].values()) for c in cfgs)
                    ax.axhline(value, color=colors[method], linestyle=":", label="Rule")
                continue
            x, y, lo, hi = [], [], [], []
            points = []
            for point in range(1, (50 if env == "had" else 40) + 1):
                means = []
                steps = []
                for seed in protocol.SEEDS:
                    key = (env, method, seed, point)
                    values = results.validation.get(key, {})
                    means.append(statistics.mean(values.values()) if len(values) == (100 if env == "had" else 128)
                                 and key not in results.conflicts else None)
                    steps.append(results.validation_steps.get(key, 0))
                summary = _stats(means)
                x.append(statistics.mean(steps) / 1e6 if all(steps) else protocol.validation_thresholds(env)[point-1] / 1e6)
                y.append(summary[0] if summary else float("nan"))
                lo.append(summary[0] - summary[1] if summary else float("nan"))
                hi.append(summary[0] + summary[1] if summary else float("nan"))
                points.append(summary)
            if any(points):
                ax.plot(x, y, color=colors[method], label=LABELS[method])
                ax.fill_between(x, lo, hi, color=colors[method], alpha=.12)
                for si in range(3):
                    ax.scatter(x, [p[2][si] if p else float("nan") for p in points], s=3, color=colors[method], alpha=.23)
                shown = True
        ax.set(xlabel="Training steps (million)", ylabel="Validation D ↓" if env == "had" else "Validation win rate ↑")
        if not shown:
            _empty(ax)
    equal = tuple((n, n, 2) for n in (5, 10, 15, 20, 25, 30, 40, 50))
    shown = False
    for method in BODY_HAD:
        shown |= _curve(axes[0, 1], results, "had", method, equal, [c[0] for c in equal], colors[method])
    if not shown:
        _empty(axes[0, 1])
    axes[0, 1].set(xlabel="N (equal teams, K=2)", ylabel="D ↓")
    shown = False
    for method in BODY_HAD:
        for n, style in ((10, "-"), (30, "--")):
            cfgs = tuple((n, n, k) for k in (2, 4, 6, 9, 12))
            shown |= _curve(axes[0, 2], results, "had", method, cfgs, [c[2] for c in cfgs], colors[method],
                            linestyle=style, label=f"{LABELS[method]} N={n}")
    if not shown:
        _empty(axes[0, 2])
    axes[0, 2].set(xlabel="K (solid N=10; dashed N=30)", ylabel="D ↓")
    for ax, cfgs, xlabel in (
            (axes[1, 1], tuple((n, n, 0) for n in (5, 10, 12, 15, 20)), "N (equal teams)"),
            (axes[1, 2], tuple((10, n, 0) for n in (10, 11, 12, 15)), "Enemies (10 allies)")):
        shown = False
        for method in protocol.SMAC_METHODS:
            shown |= _curve(ax, results, "smacv2", method, cfgs, [c[1] for c in cfgs], colors[method])
        if not shown:
            _empty(ax)
        ax.set(xlabel=xlabel, ylabel="Win rate ↑")
    for index, (ax, title) in enumerate(zip(axes.flat, ("HAD learning", "HAD team-size transfer", "HAD target transfer",
                                                      "SMACv2 learning", "SMACv2 team-size transfer", "SMACv2 enemy pressure"))):
        ax.set_title(f"{chr(97 + index)}. {title}", loc="left")
        ax.grid(alpha=.2)
        if index in (0, 1, 3, 4):
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, fontsize=7, ncol=2)
    _save(fig, directory, "fig2_comparison", plt)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.8), constrained_layout=True)
    ax = axes[0, 0]
    shown = False
    for gi, group in enumerate(("N-OOD", "K-OOD")):
        for mi, method in enumerate(ABLATIONS):
            summary = results.summary("had", method, HAD_GROUPS[group])
            if not summary:
                if gi == 0:
                    ax.text(mi, .025, "待完成", transform=ax.get_xaxis_transform(), ha="center",
                            va="bottom", rotation=90, color="#777777", fontsize=7)
                continue
            x = mi + (gi - .5) * .34
            ax.bar(x, summary[0], .32, yerr=summary[1], color=palette(gi), alpha=.65,
                   label=group if mi == 0 else None, capsize=2)
            ax.scatter([x - .06, x, x + .06], summary[2], color="#333333", s=10, zorder=3)
            shown = True
    ax.set_xticks(range(len(ABLATIONS)), [LABELS[m] for m in ABLATIONS], rotation=40, ha="right")
    ax.set(title="a. Retrained structure comparisons", ylabel="D ↓")
    if shown:
        ax.legend(fontsize=8)
    else:
        _empty(ax)
    ax = axes[0, 1]
    shown = False
    for ci, cfg in enumerate(protocol.DEPTH_CONFIGS):
        summaries = [results.summary("had", "regir", [cfg], f"depth:R{r}") for r in range(1, 7)]
        if any(summaries):
            x = list(range(1, 7))
            ax.errorbar(x, [s[0] if s else float("nan") for s in summaries],
                        yerr=[s[1] if s else 0 for s in summaries], marker="o", capsize=2,
                        color=palette(ci), label=_label(cfg))
            for si in range(3):
                ax.scatter(x, [s[2][si] if s else float("nan") for s in summaries], s=10, color=palette(ci), alpha=.4)
            shown = True
    ax.set(title="b. Frozen Full: execution depth", xlabel="R", ylabel="D ↓", xticks=range(1, 7))
    missing_depths = [r for r in range(1, 7) if not all(
        results.summary("had", "regir", [cfg], f"depth:R{r}") for cfg in protocol.DEPTH_CONFIGS)]
    if shown and missing_depths:
        ax.text(.02, .02, "待完成: " + ", ".join(f"R{r}" for r in missing_depths),
                transform=ax.transAxes, fontsize=8, color="#707070")
    if shown:
        ax.legend(fontsize=8)
    else:
        _empty(ax)
    ax = axes[1, 0]
    data = np.full((3, 5), np.nan)
    for li, label in enumerate(PROBE_LABELS):
        for layer in range(5):
            summary = results.probe_summary((50, 50, 2), label, layer)
            if summary:
                data[li, layer] = summary[0]
    if np.isfinite(data).any():
        plot = ax.imshow(np.ma.masked_invalid(data), cmap="viridis", aspect="auto")
        fig.colorbar(plot, ax=ax, label="Mean test R² (3 checkpoints)", shrink=.8)
        for li in range(3):
            for layer in range(5):
                ax.text(layer, li, _num(data[li, layer], 2), ha="center", va="center", fontsize=9,
                        bbox=dict(facecolor="white", alpha=.65, edgecolor="none", pad=1))
    else:
        _empty(ax)
    ax.set(title="c. Relation probes: 50v50 K2 confirmation", xticks=range(5), xticklabels=[f"H{r}" for r in range(5)],
           yticks=range(3), yticklabels=["Enemy–target distance", "Enemy–other defender", "Region imbalance"])
    ax = axes[1, 1]
    shown = False
    for ri, readout in enumerate(protocol.READOUTS):
        summary = results.summary("had", "regir", [(50, 50, 2)], f"readout:{readout}")
        if summary:
            ax.bar(ri, summary[0], yerr=summary[1], capsize=3, color=palette(ri), alpha=.7)
            ax.scatter([ri - .07, ri, ri + .07], summary[2], color="#333333", s=14)
            shown = True
        else:
            ax.text(ri, .03, "待完成", transform=ax.get_xaxis_transform(), ha="center", color="#777777", fontsize=8)
    ax.set(title="d. Fixed R=4 readout: 50v50 K2", ylabel="D ↓",
           xticks=range(6), xticklabels=["Learned", "Read-1", "Read-2", "Read-3", "Read-4", "Uniform"])
    if not shown:
        _empty(ax)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 1]):
        ax.grid(axis="y", alpha=.2)
    _save(fig, directory, "fig3_mechanism", plt)


def _cost_table(results):
    rows = []
    for method in protocol.COST_METHODS:
        records = [r for r in results.costs if r.get("method") == method and r.get("env", "had") == "had"]
        actor = sorted({int(r["actor_params"]) for r in records if _finite(r.get("actor_params"))})
        total = sorted({int(r["training_params"]) for r in records if _finite(r.get("training_params"))})
        latencies = []
        for n in (10, 50):
            seed_cells = []
            for seed in protocol.SEEDS:
                selected = [r for r in records if r.get("seed") == seed and _cfg(r["config"]) == (n, n, 2)]
                devices = {(r.get("physical_gpu"), r.get("device")) for r in selected}
                if len(devices) != 1:
                    seed_cells.append("—")
                    continue
                last = selected[-1]
                seed_cells.append(f"{_num(last['ms_per_step'])} [{_num(last.get('p25_ms'))}, {_num(last.get('p75_ms'))}]; p95={_num(last.get('p95_ms'))}")
            latencies.append("<br>".join(f"s{s}: {v}" for s, v in zip(protocol.SEEDS, seed_cells)))
        device = ",".join(sorted({f"物理GPU{r['physical_gpu']} ({r['device']})" for r in records})) or "—"
        rows.append([LABELS[method], "/".join(map(str, actor)) or "—", "/".join(map(str, total)) or "—", device, *latencies])
    return _table(["方法", "actor 参数", "训练总参数", "设备", "N=10 ms/全队决策", "N=50 ms/全队决策"], rows)


def _figure_lines(stem, title, caption):
    return [f"![{title}](figures/{stem}.png)", "", caption,
            f"[PDF](figures/{stem}.pdf) · [PNG](figures/{stem}.png)", ""]


def _appendix(results):
    lines = ["## 附录", "", "### A. HAD全部24配置与15个学习方法", "",
             "每格为三seed均值 ± 95% t半宽；方括号依次为seed0、seed1、seed2。"
             "不完整格保留已完成种子值，并显示三个种子的有效局数；这些值不进入正式均值。"
             "Rule和Random是同一场景库的单组共同参照。", ""]
    for group, cfgs in HAD_GROUPS.items():
        lines += [f"#### A.{list(HAD_GROUPS).index(group) + 1} {group}（{len(cfgs)}配置）", ""]
        lines += _table(["方法", *map(_label, cfgs)], [
            [LABELS[method], *(results.cell("had", method, [cfg], points=True) for cfg in cfgs)]
            for method in protocol.methods("had") + ("rule_nv1", "random")])
    lines += [f"### B. SMACv2全部{len(protocol.SMAC_FINAL)}配置与{len(protocol.SMAC_METHODS)}个方法", "",
              "10v10在两条展示轴交叉，但正式唯一配置和统计局数只计一次。每格记法同附录A。", ""]
    for title, cfgs in (("等人数", protocol.SMAC_FINAL[:5]), ("固定己方10", protocol.SMAC_FINAL[5:])):
        lines += [f"#### B. {title}", ""] + _table(["方法", *map(_label, cfgs)], [
            [LABELS[method], *(results.cell("smacv2", method, [cfg], points=True) for cfg in cfgs)]
            for method in protocol.SMAC_METHODS])
    lines += ["### C. 全部执行深度与固定计算读出", "",
              "主执行轮数始终R4，不依据外推结果选择深度；所有登记点保留，包含降低表现的深度或干预。", ""]
    lines += _table(["M1执行轮数", *map(_label, protocol.DEPTH_CONFIGS)], [
        [f"R{r}", *(results.cell("had", "regir", [cfg], f"depth:R{r}", points=True)
                    for cfg in protocol.DEPTH_CONFIGS)] for r in range(1, 7)])
    lines += ["Single从头训练R1的对应格，用于与Full同权重R1/R4并列辨别训练与执行效应。", ""]
    lines += _table(["方法", *map(_label, protocol.DEPTH_CONFIGS)], [
        ["Single", *(results.cell("had", "regir_r1", [cfg], points=True) for cfg in protocol.DEPTH_CONFIGS)]])
    lines += _table(["M2读出", *map(_label, protocol.READOUT_CONFIGS)], [
        [readout, *(results.cell("had", "regir", [cfg], f"readout:{readout}", points=True)
                    for cfg in protocol.READOUT_CONFIGS)] for readout in protocol.READOUTS])
    equivalent = results.metadata.get("mechanisms", {}).get("read1_equivalence_verified", False)
    lines += [f"Read-1等价复用登记：**{'已验证' if equivalent else '尚未验证，不据此复用'}**。"
              "Learned复用相同final场景；R4复用相同final场景；复用不复制回合或扩大样本量。", ""]
    lines += ["### D. 关系探针与必要控制", "",
              "公共状态集固定为5配置×100条轨迹；Full seed0与REFIL seed0各贡献一半。"
              "三Full checkpoint分别重建同一前缀并拟合probe，不把隐空间合并。"
              "H0为输入控制；random_init与shuffled_labels分别控制随机特征和标签记忆。"
              "标准化和ridge强度只由ID训练/验证确定，OOD确认集不参与选择。"
              "保留负R²；近零方差或标签无定义不填零。", ""]
    sections = [(cfg, split) for cfg in protocol.PROBE_CONFIGS for split in (
        ("id_test",) if cfg in protocol.PROBE_CONFIGS[:2] else ("ood_explore", "ood_confirm"))]
    if not results.probes:
        lines += _table(["配置", "预登记拆分", "每checkpoint的报告矩阵", "状态"],
            [[_label(cfg), split, "3标签 × H0–H4 × 3控制；seeds0/1/2", "待完成"] for cfg, split in sections])
    else:
        for cfg, split in sections:
            lines += [f"#### D. {_label(cfg)} / {split}", ""]
            rows = []
            for control in ("trained", "random_init", "shuffled_labels"):
                for label in PROBE_LABELS:
                    for layer in range(5):
                        summary = results.probe_summary(cfg, label, layer, control, split)
                        values = {r["model_seed"]: r for r in results.probes if r["config"] == cfg
                                  and r.get("split") == split and r.get("control") == control
                                  and r.get("label") == label and r["round"] == layer}
                        seeds = [values.get(s, {}) for s in protocol.SEEDS]
                        rows.append([control, label, f"H{layer}",
                                     f"{_num(summary[0])} ± {_num(summary[1])}" if summary else "—",
                                     *[_num(r.get("r2")) for r in seeds],
                                     "/".join(str(r.get("n", "—")) for r in seeds),
                                     "/".join(str(r.get("alpha", "—")) for r in seeds)])
            lines += _table(["控制", "标签", "轮次", "R²均值±95% t", "s0", "s1", "s2", "n(0/1/2)", "α(0/1/2)"], rows)
    probe_metadata = results.probe_metadata
    if probe_metadata:
        lines += ["探针场景划分与无定义标签的排除记录：", "", "```json",
                  json.dumps(probe_metadata, ensure_ascii=False, indent=2), "```", ""]
    lines += ["### E. 协议、来源和实现边界", "",
              "原始逐局：[episodes.csv](episodes.csv)；训练指标：[learning.csv](learning.csv)；"
              "进度：[progress.csv](progress.csv)；权重资格与任务清单：[inventory.json](inventory.json)；"
              "冻结协议：[experiment.json](experiment.json)。这些是同一实验根下的原始数据，不另造第二套汇总CSV。", "",
              "HAD训练只见N∈{4,6,8,10}、K∈{1,2,3}，预算1M物理步；每20k步验证4配置×25局。"
              "SMACv2为4M联合环境决策步，每100k步验证4规模×32局；并行环境的步数累加，不乘智能体数。"
              "正式选用完成预算的final，best只保留为训练内验证最优档案。", ""]
    lines += ["SMAC新增对照：QMIX-Atten使用已有局部实体输入与共享敌人动作头，关闭imagined。"
              "SPECTra依据[作者SMAC agent](https://github.com/funny-rl/SPECTra/blob/ffababf6187216c9d16b2109ee8ef6fe5fdf1172/SPECTra_SMACv2/modules/agents/ss_rnn_agent.py)"
              "及其mixer/config适配，采用typed projections、SAQA、GRU、六基础动作＋敌人QueryKey评分、SMAC ST-HyperNet及batch128。"
              "与作者固定规模实现的差异包括显式局部可见性/死亡/padding mask、四采集worker及每四回合四次更新；"
              "Adam/lr=.001、TD(λ)=.6、hidden64/head4保持作者配置。HAD SPECTra的batch32及原权重路径保留。"
              "完整原始来源、适配边界及验收在[实验计划](实验计划.md#10-smac六方法与复现边界)和既有验收文件中。", ""]
    lines += _table(["字段", "登记值"], [[key, json.dumps(results.metadata.get(key, {}), ensure_ascii=False)]
                                            for key in ("profile", "seeds", "sources", "resources")])
    cost_contracts = []
    for row in results.progress.values():
        info = results.checkpoints.get((row.get("env"), row.get("method"), row.get("seed")), {})
        contract = row.get("measurement_contract")
        if (row.get("phase") == "timing" and row.get("arm") == "cost" and contract
                and row.get("checkpoint_id") == info.get("checkpoint_id")
                and type(results.measurement_gpu) is int and results.measurement_gpu in (0, 1)
                and contract.get("physical_gpu") == results.measurement_gpu):
            cost_contracts.append([LABELS.get(row.get("method"), row.get("method")), row.get("seed"),
                                   _label(_cfg(row["config"])), json.dumps(contract, ensure_ascii=False)])
    lines += ["M4测量条件（仅保留匹配当前final身份的登记值；总参数指在线可训练actor+mixer，排除target复制）：", ""]
    lines += _table(["方法", "seed", "配置", "实际测量条件"],
                    cost_contracts or [["—", "—", "—", "公共状态库或登记物理GPU上的正式测量待完成"]])
    imports = results.metadata.get("imports", [])
    if imports:
        lines += ["来源与迁入补丁记录（以登记值为准）：", "", "```json",
                  json.dumps(imports, ensure_ascii=False, indent=2), "```", ""]
    else:
        lines += ["来源与迁入补丁记录尚未登记；报告不推断迁入过程已完成。", ""]
    lines += _table(["已有合格终评来源", "记录数（实际回合按唯一键去重）"],
                    sorted(results.sources.items()) or [["—", 0]])
    artifact_rows = []
    for env in ("had", "smacv2"):
        for method in protocol.methods(env):
            for seed in protocol.SEEDS:
                info = results.checkpoints.get((env, method, seed), {})
                artifact_rows.append([env, LABELS[method], seed, info.get("t_env", "—"),
                                      info.get("checkpoint_id", "—"), info.get("path", "—")])
    lines += _table(["环境", "方法", "seed", "final实际步数", "artifact ID", "来源权重"], artifact_rows)
    lines += ["### F. 当前缺口、失败、冲突与排除", "",
              "运行上限：HAD每卡最多3个进程、两卡最多6个；SMAC每卡最多2个、两卡最多4个。仍须通过显存准入，SMAC还检查目标卡利用率；"
              "CPU终评与机制合计4个任务。HAD优先，CPU按终评→深度→读出→探针排序；HAD和M4完成后进入SMAC。报告每5分钟自动刷新，阶段结束再刷新。统一恢复使用`bash Open-SCORE/outputs/main0921/run_all.sh resume`。实时任务数、已完成/剩余量及条件ETA可通过"
              "`train.py --profile main0921 --stage status --output Open-SCORE/outputs/main0921`查看。"
              "该面板默认精简，--details显示完整明细；只读查看，关闭面板不会停止训练。具体时间依据与区间见[实验计划](实验计划.md)，"
              "smoke与受控恢复证据归并在[implementation_acceptance.json](implementation_acceptance.json)。", "",
              "任务状态来自本版本inventory；completed文字本身不能使结果进入统计。"
              "计入与否仍由权重身份、配置、随机流、完整回合和三种子规则决定。", ""]
    tasks = results.inventory.get("tasks", [])
    lines += _table(["任务", "环境", "方法", "seed", "状态", "完成/总量", "详情"], [
        [t.get("id", "—"), t.get("env", "—"), t.get("method", "—"), t.get("seed", "—"),
         t.get("status", "—"), f"{t.get('completed', '—')}/{t.get('total', '—')}", t.get("detail", "")]
        for t in tasks] or [["inventory尚未生成", "—", "—", "—", "待完成", "—", "未推断任务已完成"]])
    lines += _table(["排除原因", "读取到的记录数"], sorted(results.excluded.items()) or [["本次读取未发现排除记录", 0]])
    lines += [f"同一唯一回合出现不同指标的冲突格：**{len(results.conflicts)}**；冲突格不进入统计。", ""]
    if results.conflicts:
        lines += _table(["冲突身份"], [[str(k)] for k in sorted(results.conflicts, key=str)])
    failure_rows = [r for r in results.progress.values() if r.get("status") in ("failed", "interrupted", "stopped")]
    lines += ["以下为各已记录阶段最后的失败/暂停历史，包含迁入历史；不覆盖上方inventory当前任务状态。", ""]
    lines += _table(["时间", "环境", "方法", "seed", "阶段/实验臂", "已记录状态", "原因/进度"], [
        [r.get("recorded_at", "—"), r.get("env", "had"), r.get("method"), r.get("seed"), r.get("arm") or r.get("phase") or "旧记录未标阶段", r.get("status"),
         r.get("error") or r.get("detail") or f"t_env={r.get('t_env', '—')} completed={r.get('completed', '—')}"]
        for r in failure_rows] or [["—", "—", "—", "—", "—", "无此类阶段状态记录", "不代表所有任务已完成"]])
    return lines


def refresh_report(output, *, run="train", report_stream=None):
    """Render only retained observations, without initiating any experiment."""
    if run != "train":
        raise ValueError("The main0921 report reads the registered train run only")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = _json(output / "experiment.json", {})
    inventory = protocol.scan(output)
    results = Results(output, inventory, metadata)
    _figures(results, output)
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    qualified = sum(results.summary(env, method, [cfg]) is not None
                    for env in ("had", "smacv2") for method in protocol.methods(env) for cfg in protocol.configs(env))
    planned = sum(len(protocol.methods(env)) * len(protocol.configs(env)) for env in ("had", "smacv2"))
    runs = sum(len(protocol.methods(env)) * len(protocol.SEEDS) for env in ("had", "smacv2"))
    smac_runs = len(protocol.SMAC_METHODS) * len(protocol.SEEDS)
    lines = ["# main0921：跨规模泛化与循环机制", "", f"更新时间：{now}。唯一正式报告；正文与附录共用同一数据源。",
             f"当前三种子完整正式格：**{qualified}/{planned}**。合格 final 权重身份：**{len(results.checkpoints)}/{runs}**。"
             "本页展示已有结果与明确缺口，不把计划任务写成结论。", "",
             f"SMACv2固定矩阵：{'、'.join(LABELS[m] for m in protocol.SMAC_METHODS)}；"
             f"**{len(protocol.SMAC_METHODS)}方法×{len(protocol.SEEDS)}种子＝{smac_runs}次训练**，"
             f"每次{protocol.budget('smacv2') // 1_000_000}M步，总计{smac_runs * protocol.budget('smacv2') // 1_000_000}M步；"
             f"训练内验证{smac_runs * 40 * len(protocol.SMAC_VALIDATION) * 32:,}局，"
             f"正式终评{smac_runs * len(protocol.SMAC_FINAL) * 300:,}局。", "",
             "## 正文", "", "### 数据口径与展示安排", "",
             "HAD 的 D 越低越好；SMACv2 的 battle_won 胜率越高越好。所有训练种子固定为0、1、2。"
             "正式格必须匹配 inventory 中已验证的 final 权重身份和实际预算，且每种子完成规定随机流的300个唯一回合。"
             "HAD使用9000–9299，SMACv2使用110000–110299。缺种子或缺回合的正式均值显示 —；局数和已有种子值留在附录。", "",
             "统计顺序：每种子每配置求回合均值 → 固定配置等权 → 三个训练种子等权。"
             "± 为95% t区间半宽（df=2，t=4.30265273）；散点为三个种子值，不能当作独立回合样本。"
             "规则只有一组公共场景结果，没有虚构三个训练种子或跨种子误差条。R²保留负值；未定义值显示 —。", ""]
    if metadata.get("runtime_estimate", {}).get("status") == "six_method_throughput_pending":
        lines += ["工期修正：此前5.4–6.8天只估算了12次SMAC训练，不能代表当前18次矩阵。"
                  "新增QMIX-Atten/SPECTra尚无GPU实测吞吐，完整训练ETA待补测；7天仍是目标，尚不能确认达成。"
                  "HAD继续优先执行，所有三种子、预算和评估配额保持登记值。", ""]
    elif metadata.get("runtime_estimate", {}).get("status") == "conditional_full_two_gpu_six_method_estimate":
        estimate = metadata["runtime_estimate"]
        span = estimate["deadline_review"]["training_target_days"]
        lines += [f"工期登记（{estimate['updated_at']}）：按六方法全部实测架构吞吐与当时剩余量，"
                  f"两张物理GPU全速条件下，剩余训练、训练内验证及HAD阶段尾部约**{span[0]:.1f}–{span[1]:.1f}天**。"
                  "这是初始化策略的短时测量估算，终端按各方法剩余步数更新；不含外部资源等待、故障重试和最终SMAC终评尾部。"
                  "此前12次SMAC对应的5.4–6.8天已作废；当前区间跨过7天，不能承诺7天内完成。", ""]
    lines += _table(["位置 / ID", "问题与固定展示", "轴 / 指标与不确定性", "数据源 / 完成条件"], [
        ["正文 图1", "Full结构；共享更新与逐轮读取", "结构图，无结果数值", "当前方法实现；不是机制实验证据"],
        ["正文 图2a/d", "HAD与SMAC训练内学习", "训练步→D/胜率；三seed均值和95% t带", "episodes.csv/train_eval；每点100/128局×3seed"],
        ["正文 图2b/e", "等人数外推", "N→D/胜率；95% t、seed点", "episodes.csv/final_eval；每格300×3"],
        ["正文 图2c/f", "HAD目标数／SMAC敌人数压力", "K或敌人数→D/胜率；95% t、seed点", "同一final；10v10交叉点复用"],
        ["正文 图3a", "重训结构对照", "方法→N-OOD/K-OOD D；95% t、seed点", "正式终评固定分组；不隐藏无收益的变体"],
        ["正文 图3b", "同一Full增加执行轮数", "R=1–6，N=10/30/50；D、95% t", "depth_eval；R4复用final"],
        ["正文 图3c", "后轮的关系可读性", "50v50 K2确认集；3标签×H0–H4 R²", "probe/results.csv；每格3个checkpoint"],
        ["正文 图3d", "固定四轮下改变读取", "50v50 K2；六读出D、95% t、seed点", "readout_eval；Learned复用final"],
        ["正文 表1", "核心算法在不同泛化轴的表现", "HAD四分组D；SMAC四关键格胜率", "完整配置集合×3seed；禁止局数加权"],
        ["正文 表2", "参数与计算成本", "八方法actor/训练总参数、10/50全队时延", "timing.csv实际身份；逐seed中位数/IQR/p95"],
        ["附录 A–F", "全部格、种子、机制与控制、来源和失败", "不按结果筛选；缺口与排除原因", "同源完整表；保留负结果和全部登记臂"],
    ])
    lines += ["### 图1：方法结构", ""] + _figure_lines("fig1_architecture", "图1 方法结构",
        "共享块重复更新实体工作区；每轮结果分别被个体读取，再融合进入策略。图示不声称已经证明压缩、收敛或语义分工。")
    lines += ["### 图2：两域主比较", ""] + _figure_lines("fig2_comparison", "图2 两域主比较",
        "上排HAD、下排SMACv2。HAD登记的方法为Full、Single、REFIL、TransfQMix、ALMA及规则；只有完整格形成曲线。完整算法表见表1和附录。"
        "HAD N=5是插值，N=10属于训练范围；目标轴实线N=10、虚线N=30。SMAC N=5是插值，10是训练边界，12/15/20为人数外推；"
        "固定己方10的敌人数变化同时改变兵力比。缺完整三种子的数据不连成趋势。")
    lines += ["### 表1：核心算法", "", "HAD：分组严格互斥；训练范围支持点和人数插值另列于附录。", ""]
    groups = ("ID", "N-OOD", "K-OOD", "联合 OOD")
    lines += _table(["方法", *groups], [[LABELS[m], *(results.cell("had", m, HAD_GROUPS[g]) for g in groups)] for m in TABLE_HAD])
    smac_key = ((10, 10, 0), (15, 15, 0), (20, 20, 0), (10, 15, 0))
    lines += ["SMACv2：胜率。", ""] + _table(["方法", *map(_label, smac_key)],
        [[LABELS[m], *(results.cell("smacv2", m, [c]) for c in smac_key)] for m in protocol.SMAC_METHODS])
    lines += ["### 图3：循环机制", ""] + _figure_lines("fig3_mechanism", "图3 循环机制",
        "重训、同权重执行深度、同状态关系探针和固定计算读出回答不同问题，不能互相替代。"
        "Read-4是冻结Full的干预，Last是重新训练的方法；Read-1仅在登记等价核查通过后复用M1。"
        "图3c颜色显示三checkpoint平均R²，完整种子值、95% t区间及随机初始化/置乱标签控制见附录D。")
    lines += ["### 表2：参数与全队决策成本", "",
              "中位数 [p25, p75] 与p95均来自实际时延记录，逐seed保留；—表示尚无合格测量。"
              "全部48格使用experiment.json登记的同一物理GPU（0或1），"
              "float32（关闭TF32）、batch=1全队，50次预热后200次同步计时。"
              "各模型从同一公共真实动作前缀重建各自h_prev，每次计时前恢复；计入actor张量编码，"
              "不计环境、状态重建、动作解码或mixer。训练总参数为在线可训练actor+mixer，不含target复制。"
              "配置表示初始物理队伍数；实际设备型号、当时其他计算进程显存占用、"
              "状态编号、时刻、存活数及共同padding见附录E。", ""] + _cost_table(results)
    lines += _appendix(results)
    text = "\n".join(lines).rstrip() + "\n"
    path = output / "实验报告.md"
    if report_stream is not None:
        report_stream.seek(0)
        report_stream.write(text.encode("utf-8"))
        report_stream.truncate()
        report_stream.flush()
        os.fsync(report_stream.fileno())
    else:
        temporary = path.with_suffix(".md.pending")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    return path
