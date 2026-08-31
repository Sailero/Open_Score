"""Create auditable Stage-2 calibration, risk, time and decision figures."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.metrics.read_text(encoding="utf-8"))
    metrics = payload.get("test_metrics", payload)
    rows = _read_jsonl(args.predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    reliability = metrics["breach"]["reliability_bins"]
    axes[0, 0].plot([0, 1], [0, 1], "--", color="0.5", label="ideal")
    if reliability:
        axes[0, 0].plot(
            [row["mean_probability"] for row in reliability],
            [row["event_rate"] for row in reliability],
            "o-",
            label="model",
        )
    axes[0, 0].set(xlabel="Predicted breach probability", ylabel="Observed breach rate", title="Reliability")
    axes[0, 0].legend()

    cells = defaultdict(list)
    for row in rows:
        cells[(row["root_id"], row["candidate_id"], row["threat_id"])].append(row)
    point, truth, upper = [], [], []
    for values in cells.values():
        point.append(np.mean([row["predicted_breach_probability"] for row in values]))
        truth.append(np.mean([row["true_outcome"] == "breach" for row in values]))
        upper.append(np.mean([row["breach_upper_selection_safe"] for row in values]))
    axes[0, 1].scatter(point, truth, alpha=0.65, label="cell")
    axes[0, 1].hlines(
        truth, point, upper, color="tab:orange", alpha=0.35, label="calibrated upper gap"
    )
    axes[0, 1].plot([0, 1], [0, 1], "--", color="0.5")
    axes[0, 1].set(xlabel="Predicted breach rate", ylabel="Observed breach rate", title="Root × Red × Blue cells")
    axes[0, 1].legend(fontsize=8)

    observed_rows = [row for row in rows if row["event_observed"]]
    if observed_rows:
        true_time = [row["true_terminal_or_censor_steps"] for row in observed_rows]
        predicted_time = [row["predicted_restricted_mean_steps"] for row in observed_rows]
        axes[1, 0].scatter(true_time, predicted_time, alpha=0.55)
        limit = max(max(true_time), max(predicted_time))
        axes[1, 0].plot([0, limit], [0, limit], "--", color="0.5")
    axes[1, 0].set(xlabel="Observed terminal steps", ylabel="Predicted restricted mean steps", title="Observed-event time")

    decisions = metrics.get("red_candidate_decision", {}).get("per_root_selection", [])
    regrets = sorted([row["decision_regret"] for row in decisions], reverse=True)
    axes[1, 1].bar(np.arange(len(regrets)), regrets, color="tab:red", alpha=0.75)
    if regrets and max(regrets) <= 1e-12:
        axes[1, 1].text(
            0.5,
            0.5,
            "All test-root regrets = 0",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
        )
        axes[1, 1].set_ylim(0.0, 0.05)
    axes[1, 1].set(xlabel="Test root (sorted)", ylabel="Breach-rate regret", title="Red policy selection regret")

    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure_png = args.output_dir / "stage2_summary.png"
    figure_pdf = args.output_dir / "stage2_summary.pdf"
    fig.savefig(figure_png, dpi=180)
    fig.savefig(figure_pdf)
    plt.close(fig)
    physical_fields = sorted(
        {
            field
            for row in rows
            for field in row.get("physical_predictions", {})
            if row.get("physical_outcomes", {}).get(field) is not None
        }
    )
    physical_figures = []
    physical_plot_data = {}
    if physical_fields:
        columns = 3
        row_count = int(np.ceil(len(physical_fields) / columns))
        physical_fig, physical_axes = plt.subplots(
            row_count,
            columns,
            figsize=(4.6 * columns, 3.8 * row_count),
            constrained_layout=True,
            squeeze=False,
        )
        for axis, field in zip(physical_axes.flat, physical_fields):
            selected = [
                row
                for row in rows
                if row.get("physical_outcomes", {}).get(field) is not None
                and field in row.get("physical_predictions", {})
            ]
            truth_values = np.asarray(
                [row["physical_outcomes"][field] for row in selected], float
            )
            point_values = np.asarray(
                [row["physical_predictions"][field]["point"] for row in selected], float
            )
            lower_values = np.asarray(
                [row["physical_predictions"][field]["ensemble_p10"] for row in selected],
                float,
            )
            upper_values = np.asarray(
                [row["physical_predictions"][field]["ensemble_p90"] for row in selected],
                float,
            )
            axis.errorbar(
                truth_values,
                point_values,
                yerr=np.vstack(
                    (
                        np.maximum(0.0, point_values - lower_values),
                        np.maximum(0.0, upper_values - point_values),
                    )
                ),
                fmt="o",
                alpha=0.35,
                markersize=3,
                linewidth=0.7,
            )
            plot_min = float(min(truth_values.min(), lower_values.min()))
            plot_max = float(max(truth_values.max(), upper_values.max()))
            if plot_max <= plot_min:
                plot_max = plot_min + 1.0
            axis.plot([plot_min, plot_max], [plot_min, plot_max], "--", color="0.5")
            axis.set(title=field, xlabel="Observed", ylabel="Predicted")
            axis.grid(alpha=0.2)
            physical_plot_data[field] = {
                "observed": truth_values.tolist(),
                "predicted": point_values.tolist(),
                "ensemble_p10": lower_values.tolist(),
                "ensemble_p90": upper_values.tolist(),
            }
        for axis in physical_axes.flat[len(physical_fields) :]:
            axis.set_visible(False)
        physical_png = args.output_dir / "stage2_physical_evaluators.png"
        physical_pdf = args.output_dir / "stage2_physical_evaluators.pdf"
        physical_fig.savefig(physical_png, dpi=180)
        physical_fig.savefig(physical_pdf)
        plt.close(physical_fig)
        physical_figures = [str(physical_png.resolve()), str(physical_pdf.resolve())]
    plot_data = {
        "breach_reliability_bins": reliability,
        "candidate_threat_cells": [
            {"predicted": float(p), "observed": float(y), "upper": float(u)}
            for p, y, u in zip(point, truth, upper)
        ],
        "red_decision_regrets": regrets,
        "physical_evaluators": physical_plot_data,
        "figures": [str(figure_png.resolve()), str(figure_pdf.resolve())]
        + physical_figures,
    }
    (args.output_dir / "plot_data.json").write_text(
        json.dumps(plot_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {"status": "completed", "png": str(figure_png), "pdf": str(figure_pdf)}
        )
    )


if __name__ == "__main__":
    main()
