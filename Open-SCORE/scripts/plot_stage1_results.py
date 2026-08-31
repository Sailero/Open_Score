"""Generate auditable Stage-1 figures from tracked experiment artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260831


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--had-summary", type=Path)
    parser.add_argument("--had-curves", type=Path)
    parser.add_argument("--ad-summary", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/stage1_figures")
    )
    parser.add_argument(
        "--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def save_figure(fig, stem: Path, title: str, outputs: list) -> None:
    fig.suptitle(title)
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        path = stem.with_suffix(suffix)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)


def derived_bootstrap_seed(base_seed: int, *cell: object) -> int:
    """Derive a stable RNG stream without relying on Python's salted hash."""

    token = "|".join([str(base_seed), *(str(value) for value in cell)])
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def bootstrap_mean_interval(
    values: Sequence[float], *, samples: int, seed: int
) -> Dict[str, object]:
    """Percentile interval over independent training-seed means."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 1:
        raise ValueError("bootstrap requires at least one scalar seed value")
    if not np.isfinite(array).all():
        raise ValueError("bootstrap seed values must be finite")
    if samples < 100:
        raise ValueError("bootstrap requires at least 100 samples")
    mean = float(array.mean())
    if array.size == 1:
        low = high = mean
    else:
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, array.size, size=(samples, array.size))
        bootstrap_means = array[indices].mean(axis=1)
        low, high = (
            float(value)
            for value in np.quantile(bootstrap_means, (0.025, 0.975))
        )
    return {
        "mean": mean,
        "ci95_low": low,
        "ci95_high": high,
        "seed_count": int(array.size),
        "bootstrap_seed": int(seed),
    }


def symmetric_offsets(count: int, span: float) -> np.ndarray:
    if count < 1:
        return np.empty(0, dtype=np.float64)
    if count == 1:
        return np.zeros(1, dtype=np.float64)
    return np.linspace(-span, span, count, dtype=np.float64)


def draw_asymmetric_error_bars(
    axis,
    positions: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    cap_width: float = 0.045,
) -> None:
    """Draw percentile bounds without Matplotlib's single-bar yerr ambiguity."""

    x = np.asarray(positions, dtype=np.float64)
    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    axis.vlines(x, low, high, color="black", linewidth=1.0, zorder=5)
    axis.hlines(
        low, x - cap_width, x + cap_width, color="black", linewidth=1.0, zorder=5
    )
    axis.hlines(
        high, x - cap_width, x + cap_width, color="black", linewidth=1.0, zorder=5
    )


def plot_had_curves(
    path: Path,
    output_dir: Path,
    outputs: list,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> Dict[str, object]:
    with path.open("r", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    grouped: Dict[str, Dict[int, Dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        algorithm = str(row["algorithm"])
        seed = int(row["seed"])
        episode = int(row["episode"])
        if episode in grouped[algorithm][seed]:
            raise ValueError(
                f"duplicate HAD curve cell: algorithm={algorithm} seed={seed} "
                f"episode={episode}"
            )
        grouped[algorithm][seed][episode] = float(
            row["eval_controlled_mean_payoff"]
        )
    if not grouped:
        raise ValueError("HAD curve CSV contains no rows")

    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    bootstrap_cells = []
    individual_seed_line_count = 0
    for algorithm_index, algorithm in enumerate(sorted(grouped)):
        color = f"C{algorithm_index % 10}"
        seed_series = grouped[algorithm]
        for seed in sorted(seed_series):
            points = sorted(seed_series[seed].items())
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                color=color,
                alpha=0.18,
                linewidth=0.9,
            )
            individual_seed_line_count += 1

        episodes = sorted(
            {episode for values in seed_series.values() for episode in values}
        )
        means, lower, upper = [], [], []
        for episode in episodes:
            values = [
                seed_series[seed][episode]
                for seed in sorted(seed_series)
                if episode in seed_series[seed]
            ]
            cell_seed = derived_bootstrap_seed(
                bootstrap_seed,
                "had_validation_learning_curves",
                algorithm,
                episode,
            )
            interval = bootstrap_mean_interval(
                values, samples=bootstrap_samples, seed=cell_seed
            )
            means.append(interval["mean"])
            lower.append(interval["ci95_low"])
            upper.append(interval["ci95_high"])
            bootstrap_cells.append(
                {"algorithm": algorithm, "episode": episode, **interval}
            )
        x = np.asarray(episodes, dtype=np.int64)
        axis.plot(
            x,
            np.asarray(means, dtype=np.float64),
            color=color,
            marker="o",
            linewidth=2.0,
            label=algorithm.upper(),
        )
        axis.fill_between(
            x,
            np.asarray(lower, dtype=np.float64),
            np.asarray(upper, dtype=np.float64),
            color=color,
            alpha=0.20,
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Training episodes")
    axis.set_ylabel("Validation controlled payoff")
    axis.set_ylim(-1.05, 1.05)
    axis.legend()
    axis.grid(alpha=0.25)
    save_figure(
        fig,
        output_dir / "had_validation_learning_curves",
        "HAD validation learning curves (training-seed bootstrap 95% CI)",
        outputs,
    )
    return {
        "individual_seed_line_count": individual_seed_line_count,
        "bootstrap_cells": bootstrap_cells,
    }


def plot_had_heldout(
    summary: Mapping[str, object],
    output_dir: Path,
    outputs: list,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> Dict[str, object]:
    grouped: Dict[str, Dict[int, float]] = defaultdict(dict)
    for result in summary["results"]:
        algorithm = str(result["algorithm"])
        seed = int(result["seed"])
        if seed in grouped[algorithm]:
            raise ValueError(
                f"duplicate HAD held-out seed: algorithm={algorithm} seed={seed}"
            )
        grouped[algorithm][seed] = float(
            result["heldout_best_evaluation"]["controlled_mean_payoff"]
        )
    rule_by_seed = {
        int(seed): float(value["controlled_mean_payoff"])
        for seed, value in summary["rule_strategy"][
            "heldout_evaluation_by_training_seed"
        ].items()
    }
    if not grouped or not rule_by_seed:
        raise ValueError("HAD held-out summary is missing learned or rule seed results")

    cells = [(label, grouped[label]) for label in sorted(grouped)]
    cells.append(("rule", rule_by_seed))
    intervals = []
    for label, seed_values in cells:
        cell_seed = derived_bootstrap_seed(
            bootstrap_seed, "had_heldout_payoff", label
        )
        interval = bootstrap_mean_interval(
            [seed_values[seed] for seed in sorted(seed_values)],
            samples=bootstrap_samples,
            seed=cell_seed,
        )
        intervals.append({"label": label, **interval})

    plot_labels = [
        interval["label"].upper() if interval["label"] != "rule" else "RULE"
        for interval in intervals
    ]
    plot_means = [float(interval["mean"]) for interval in intervals]
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    positions = np.arange(len(cells), dtype=np.float64)
    axis.bar(positions, plot_means, alpha=0.78)
    draw_asymmetric_error_bars(
        axis,
        positions,
        [float(interval["ci95_low"]) for interval in intervals],
        [float(interval["ci95_high"]) for interval in intervals],
    )
    individual_seed_point_count = 0
    for position, (_, seed_values) in zip(positions, cells):
        seeds = sorted(seed_values)
        offsets = symmetric_offsets(len(seeds), 0.08)
        axis.scatter(
            position + offsets,
            [seed_values[seed] for seed in seeds],
            color="black",
            alpha=0.65,
            s=18,
            zorder=3,
        )
        individual_seed_point_count += len(seeds)
    axis.set_xticks(positions, plot_labels)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("Held-out controlled payoff")
    axis.set_ylim(-1.05, 1.05)
    axis.grid(axis="y", alpha=0.25)
    save_figure(
        fig,
        output_dir / "had_heldout_payoff",
        "HAD held-out payoff (training-seed bootstrap 95% CI)",
        outputs,
    )
    return {
        "individual_seed_point_count": individual_seed_point_count,
        "bootstrap_cells": intervals,
    }


def plot_ad_ablation(
    summary: Mapping[str, object],
    output_dir: Path,
    outputs: list,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> Dict[str, object]:
    results = summary["results"]
    grouped: Dict[tuple[str, str], Dict[int, float]] = defaultdict(dict)
    for result in results:
        key = (str(result["algorithm"]), str(result["initialization"]))
        seed = int(result["seed"])
        if seed in grouped[key]:
            raise ValueError(
                f"duplicate AD held-out seed: algorithm={key[0]} "
                f"initialization={key[1]} seed={seed}"
            )
        grouped[key][seed] = float(
            result["heldout"]["validation_best"]["mean_return"]
        )
    algorithms = sorted({key[0] for key in grouped})
    initializations = [
        value
        for value in ("scratch", "stock_transfer")
        if any((algorithm, value) in grouped for algorithm in algorithms)
    ]
    width = 0.8 / max(1, len(initializations))
    x = np.arange(len(algorithms), dtype=np.float64)
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    bootstrap_cells = []
    arm_positions: Dict[tuple[str, str], float] = {}
    seed_jitter: Dict[str, Dict[int, float]] = {}
    for algorithm in algorithms:
        all_seeds = sorted(
            {
                seed
                for initialization in initializations
                for seed in grouped.get((algorithm, initialization), {})
            }
        )
        seed_jitter[algorithm] = dict(
            zip(all_seeds, symmetric_offsets(len(all_seeds), width * 0.16))
        )
    individual_seed_point_count = 0
    for index, initialization in enumerate(initializations):
        means, ci_lower, ci_upper = [], [], []
        for algorithm in algorithms:
            seed_values = grouped.get((algorithm, initialization), {})
            if not seed_values:
                raise ValueError(
                    f"missing AD arm: algorithm={algorithm} "
                    f"initialization={initialization}"
                )
            cell_seed = derived_bootstrap_seed(
                bootstrap_seed,
                "smaclite_ad_initialization_ablation",
                algorithm,
            )
            interval = bootstrap_mean_interval(
                [seed_values[seed] for seed in sorted(seed_values)],
                samples=bootstrap_samples,
                seed=cell_seed,
            )
            means.append(float(interval["mean"]))
            ci_lower.append(float(interval["ci95_low"]))
            ci_upper.append(float(interval["ci95_high"]))
            bootstrap_cells.append(
                {
                    "algorithm": algorithm,
                    "initialization": initialization,
                    **interval,
                }
            )
        offset = (index - (len(initializations) - 1) / 2.0) * width
        axis.bar(
            x + offset,
            means,
            width=width,
            alpha=0.72,
            label=initialization,
        )
        draw_asymmetric_error_bars(
            axis, x + offset, ci_lower, ci_upper, cap_width=width * 0.10
        )
        for algorithm_index, algorithm in enumerate(algorithms):
            position = float(x[algorithm_index] + offset)
            arm_positions[(algorithm, initialization)] = position
            seed_values = grouped[(algorithm, initialization)]
            seeds = sorted(seed_values)
            axis.scatter(
                [position + seed_jitter[algorithm][seed] for seed in seeds],
                [seed_values[seed] for seed in seeds],
                color="black",
                alpha=0.70,
                s=18,
                zorder=4,
            )
            individual_seed_point_count += len(seeds)

    paired_seed_line_count = 0
    if {"scratch", "stock_transfer"} <= set(initializations):
        for algorithm in algorithms:
            scratch = grouped[(algorithm, "scratch")]
            transfer = grouped[(algorithm, "stock_transfer")]
            for seed in sorted(set(scratch) & set(transfer)):
                jitter = seed_jitter[algorithm][seed]
                axis.plot(
                    [
                        arm_positions[(algorithm, "scratch")] + jitter,
                        arm_positions[(algorithm, "stock_transfer")] + jitter,
                    ],
                    [scratch[seed], transfer[seed]],
                    color="0.35",
                    alpha=0.32,
                    linewidth=0.8,
                    zorder=2,
                )
                paired_seed_line_count += 1
    axis.set_xticks(x, [name.upper() for name in algorithms])
    axis.set_ylabel("Held-out mean return")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    save_figure(
        fig,
        output_dir / "smaclite_ad_initialization_ablation",
        "SMAClite-AD scratch vs stock-transfer (training-seed bootstrap 95% CI)",
        outputs,
    )
    return {
        "individual_seed_point_count": individual_seed_point_count,
        "paired_seed_line_count": paired_seed_line_count,
        "paired_bootstrap_stream_by_algorithm": True,
        "bootstrap_cells": bootstrap_cells,
    }


def main() -> None:
    args = parse_args()
    if not any((args.had_summary, args.had_curves, args.ad_summary)):
        raise ValueError("provide at least one Stage-1 input artifact")
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples must be at least 100")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = []
    outputs = []
    figure_statistics = {}
    if args.had_curves is not None:
        figure_statistics["had_validation_learning_curves"] = plot_had_curves(
            args.had_curves,
            args.output_dir,
            outputs,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        inputs.append(args.had_curves)
    if args.had_summary is not None:
        summary = load_json(args.had_summary)
        if summary.get("schema_version") != "stage1-had-reproduction-v5":
            raise ValueError("HAD plots require stage1-had-reproduction-v5")
        figure_statistics["had_heldout_payoff"] = plot_had_heldout(
            summary,
            args.output_dir,
            outputs,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        inputs.append(args.had_summary)
    if args.ad_summary is not None:
        summary = load_json(args.ad_summary)
        if summary.get("schema_version") != "smaclite-ad-baseline-v3":
            raise ValueError("AD plots require smaclite-ad-baseline-v3")
        figure_statistics["smaclite_ad_initialization_ablation"] = (
            plot_ad_ablation(
                summary,
                args.output_dir,
                outputs,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
            )
        )
        inputs.append(args.ad_summary)
    manifest = {
        "schema_version": "stage1-figure-manifest-v2",
        "inputs": [
            {"path": str(path.resolve()), "sha256": sha256(path)} for path in inputs
        ],
        "outputs": [
            {"path": str(path.resolve()), "sha256": sha256(path)} for path in outputs
        ],
        "uncertainty_unit": "independent training seed",
        "uncertainty": {
            "method": "nonparametric_percentile_bootstrap_of_seed_mean",
            "confidence_level": 0.95,
            "bootstrap_samples": args.bootstrap_samples,
            "base_seed": args.bootstrap_seed,
            "deterministic_stream_derivation": (
                "SHA-256(base_seed|figure|cell), first 64 bits"
            ),
        },
        "figure_statistics": figure_statistics,
        "generated_from_artifacts_only": True,
    }
    target = args.output_dir / "figure_manifest.json"
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {target.resolve()}")


if __name__ == "__main__":
    main()
