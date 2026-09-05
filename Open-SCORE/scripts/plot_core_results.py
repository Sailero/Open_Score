"""Plot logged core training and paired evaluation without changing source/data."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np


METHODS = ("selective", "full", "random", "rule", "alma", "static", "dlom")
LABELS = {"selective": "B4 Selective", "full": "B2 Full", "random": "B3 Random",
          "rule": "B6 Rule", "alma": "B5 ALMA style", "static": "B0 Static", "dlom": "B1 DLOM"}
COLORS = dict(zip(METHODS, ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#777777", "#56B4E9")))


def read_training(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if row.get("recent_success_rate") is not None]


def plot(csv_path):
    evaluation = csv_path.parent
    budget_path = evaluation / "training_budget.json"
    budget = json.loads(budget_path.read_text(encoding="utf-8"))["requested_steps"] if budget_path.exists() else None
    run_root = evaluation.parent.parent if evaluation.parent.name == "comparison" else evaluation.parent
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        results = list(csv.DictReader(stream))
    if not results:
        raise ValueError(f"No complete evaluation rows: {csv_path}")
    curves = {method: read_training(run_root / method / "training.jsonl")
              for method in METHODS[:5] if (run_root / method / "training.jsonl").exists()}
    if budget is not None:
        curves = {method: [row for row in rows if row["physical_steps"] <= budget + 4]
                  for method, rows in curves.items()}
    curves = {method: rows for method, rows in curves.items() if rows}
    if not curves:
        raise ValueError(f"No actual training curves below {run_root}")
    manifest = json.loads((run_root / next(iter(curves)) / "manifest.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    scales = sorted({int(row["scale"]) for row in results})
    episode_counts = sorted({int(row["episodes"]) for row in results})
    training_scales = set(config["train_scales"])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.labelcolor": "#334155",
                         "text.color": "#172033", "axes.edgecolor": "#CBD5E1"})
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    layout = fig.add_gridspec(2, 1, left=.065, right=.982, top=.81, bottom=.21,
                             height_ratios=(1, 1.3), hspace=.57)
    top = layout[0].subgridspec(1, len(curves), wspace=.2)
    upper = max(.2, max(row["recent_success_rate"] for rows in curves.values() for row in rows) * 1.2)
    max_steps = max(row["physical_steps"] for rows in curves.values() for row in rows)
    for index, (method, rows) in enumerate(curves.items()):
        axis = fig.add_subplot(top[index])
        axis.plot([row["physical_steps"] for row in rows], [row["recent_success_rate"] for row in rows],
                  color=COLORS[method], linewidth=2, marker="o", markersize=3.4)
        axis.set(title=LABELS[method], xlim=(0, max_steps * 1.02), ylim=(-.008, upper),
                 xlabel="Physical training steps")
        axis.title.set_color(COLORS[method])
        axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        axis.xaxis.set_major_locator(MaxNLocator(3))
        axis.yaxis.set_major_locator(MaxNLocator(4))
        axis.grid(axis="y", color="#E2E8F0", linewidth=.7)
        axis.set_axisbelow(True)
        if index == 0:
            axis.set_ylabel("Recent training success")
        else:
            axis.tick_params(labelleft=False)
    bottom = layout[1].subgridspec(1, len(scales), wspace=.2)
    evaluation_upper = max(.25, max(float(row["wilson_high"]) for row in results) * 1.12)
    for index, scale in enumerate(scales):
        axis = fig.add_subplot(bottom[index])
        selected = {row["method"]: row for row in results if int(row["scale"]) == scale}
        methods = [method for method in METHODS if method in selected]
        means = np.asarray([float(selected[method]["success_rate"]) for method in methods])
        low = np.asarray([float(selected[method]["wilson_low"]) for method in methods])
        high = np.asarray([float(selected[method]["wilson_high"]) for method in methods])
        x = np.arange(len(methods))
        axis.bar(x, means, width=.67, color=[COLORS[method] for method in methods], alpha=.9, zorder=3)
        axis.errorbar(x, means, yerr=np.maximum(0, np.vstack((means-low, high-means))),
                      fmt="none", ecolor="#475569", capsize=3, linewidth=1.1, zorder=4)
        labels = [LABELS[method].replace(" ", "\n", 1) +
                  f"\n{selected[method]['successes']}/{selected[method]['episodes']}" for method in methods]
        axis.set_xticks(x, labels, fontsize=9)
        status = "training size" if scale in training_scales else "unseen initial size"
        axis.set(title=f"{scale} vs {scale}  |  {status}", ylim=(-.006, evaluation_upper))
        axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        axis.yaxis.set_major_locator(MaxNLocator(4))
        axis.grid(axis="y", color="#E2E8F0", linewidth=.7)
        axis.set_axisbelow(True)
        if index == 0:
            axis.set_ylabel("Evaluation success")
        else:
            axis.tick_params(labelleft=False)
    fig.text(.065, .955, "Known-opponent dynamic grouping", fontsize=22, weight="bold")
    fig.text(.065, .912, f"Development validation  |  single training seed {config['seed']}  |  "
             f"frozen {config['opponent']} opponent  |  {config['max_steps']}-step task horizon", fontsize=12, color="#475569")
    fig.text(.065, .856, "A  Observed training curves", fontsize=12, weight="bold")
    fig.text(.065, .505, "B  Independent paired evaluation by initial team size", fontsize=12, weight="bold")
    counts = str(episode_counts[0]) if len(episode_counts) == 1 else "/".join(map(str, episode_counts))
    budget_label = f"Checkpoint budget: {budget:,} physical steps (+0 to 4 at event boundary). " if budget is not None else ""
    fig.text(.065, .055, f"{budget_label}Training points: recent episode success; no added smoothing.\n"
             f"Evaluation: {counts} episodes per method and scale; labels show successes/episodes. "
             "Whiskers: 95% Wilson episode intervals.\n"
             "Single training seed: these intervals do not describe variation across independently trained models.",
             fontsize=10, color="#475569", linespacing=1.55)
    destination = evaluation / "training_and_evaluation.png"
    temporary = destination.with_suffix(".tmp.png")
    fig.savefig(temporary, dpi=180, facecolor="white")
    plt.close(fig)
    temporary.replace(destination)
    print(destination.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Core comparison output root.")
    args = parser.parse_args()
    paths = [path for path in sorted(args.input.resolve().rglob("results.csv"))
             if path.parent.name == "comparison" or path.parent.parent.name == "comparison"]
    if not paths:
        parser.error("No comparison results.csv found below --input")
    for path in paths:
        plot(path)
