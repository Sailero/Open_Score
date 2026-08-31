"""Collect only preregistered EPyMARL/SMAClite stock reproduction runs.

The collector follows run records produced by ``run_stock_reproduction.ps1``;
it does not glob arbitrary Sacred history and call it evidence.  A formal run
must be completed, protocol-compliant, source-clean, config-exact, and contain a
100-episode evaluation at or beyond the preregistered terminal threshold.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_score.provenance import collect_and_require_git_provenance
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "configs"
    / "stock_reproduction"
    / "smaclite_aamas2023_epymarl_v3.json"
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


CHECKPOINT_REQUIRED_FILES = {
    "qmix": {"agent.th", "mixer.th", "opt.th"},
    "vdn": {"agent.th", "mixer.th", "opt.th"},
    "mappo": {"agent.th", "critic.th", "agent_opt.th", "critic_opt.th"},
}


def checkpoint_fingerprint(root: Path | None) -> dict[str, Any] | None:
    if root is None or not root.is_dir():
        return None
    numbered = sorted(
        (child for child in root.iterdir() if child.is_dir() and child.name.isdigit()),
        key=lambda path: int(path.name),
    )
    if not numbered:
        return None
    latest = numbered[-1]
    files = sorted(path for path in latest.rglob("*") if path.is_file())
    file_hashes = {
        str(path.relative_to(latest)).replace("\\", "/"): sha256_file(path)
        for path in files
    }
    combined = hashlib.sha256()
    for name, digest in file_hashes.items():
        combined.update(name.encode("utf-8"))
        combined.update(b"\0")
        combined.update(digest.encode("ascii"))
        combined.update(b"\n")
    return {
        "checkpoint_root": str(root.resolve()),
        "latest_step": int(latest.name),
        "latest_directory": str(latest.resolve()),
        "file_sha256": file_hashes,
        "combined_sha256": combined.hexdigest(),
    }


def committed_blob_sha256(
    repository_root: Path, commit: str, repository_relative_path: str
) -> str:
    candidates = (
        shutil.which("git"),
        r"D:\Software\Git\cmd\git.exe",
        r"C:\Program Files\Git\cmd\git.exe",
        r"C:\Program Files (x86)\Git\cmd\git.exe",
    )
    git = next((candidate for candidate in candidates if candidate and Path(candidate).is_file()), None)
    if git is None:
        raise ValueError("Git executable is unavailable")
    result = subprocess.run(
        [git, "-C", str(repository_root), "cat-file", "blob", f"{commit}:{repository_relative_path}"],
        capture_output=True,
        check=False,
        timeout=30,
    )
    if result.returncode:
        raise ValueError("manifest blob is not present in the recorded Open-SCORE commit")
    return hashlib.sha256(result.stdout).hexdigest()


def same_value(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return actual == expected


def metric_points(metrics: dict[str, Any], name: str) -> list[tuple[int, float]]:
    metric = metrics.get(name)
    if not isinstance(metric, dict):
        return []
    steps = metric.get("steps", [])
    values = metric.get("values", [])
    if not isinstance(steps, list) or not isinstance(values, list) or len(steps) != len(values):
        return []
    points: list[tuple[int, float]] = []
    for step, value in zip(steps, values):
        try:
            parsed_step = int(step)
            parsed_value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed_value):
            points.append((parsed_step, parsed_value))
    return points


def bootstrap_mean_ci(
    values: list[float], *, samples: int, seed: int
) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    means = sorted(
        statistics.fmean(rng.choice(values) for _ in values) for _ in range(samples)
    )
    lo_index = max(0, math.floor(0.025 * (samples - 1)))
    hi_index = min(samples - 1, math.ceil(0.975 * (samples - 1)))
    return means[lo_index], means[hi_index]


def paper_reference_map(manifest: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(entry["algorithm"]), str(entry["map"])): dict(entry["paper_reference"])
        for entry in manifest["suites"]["primary"]
    }


def expected_config(
    manifest: dict[str, Any], profile_name: str, record: dict[str, Any]
) -> dict[str, Any]:
    profile = manifest["profiles"][profile_name]
    algorithm = str(record["algorithm"])
    expected: dict[str, Any] = {
        "name": algorithm,
        "label": manifest["protocol_id"],
        "seed": int(record["seed"]),
        "t_max": int(profile["training_steps"]),
        "test_interval": int(profile["test_interval"]),
        "test_nepisode": int(profile["test_episodes"]),
        "log_interval": int(profile["log_interval"]),
        "runner_log_interval": int(profile["runner_log_interval"]),
        "learner_log_interval": int(profile["learner_log_interval"]),
        "save_model": True,
        "save_model_interval": int(profile["save_model_interval"]),
        "use_cuda": bool(profile["use_cuda"]),
    }
    expected.update(profile["algorithm_overrides"][algorithm])
    return expected


def validate_record(
    record_path: Path,
    *,
    manifest: dict[str, Any],
    profile_name: str,
    sacred_root: Path,
    manifest_path: Path = DEFAULT_MANIFEST,
    verify_git_blob: bool = False,
) -> tuple[dict[str, Any] | None, list[str]]:
    reasons: list[str] = []
    try:
        record = load_json(record_path)
    except Exception as exc:  # evidence collector reports malformed records
        return None, [f"invalid_record:{exc}"]

    if record.get("status") != "completed":
        reasons.append("record_not_completed")
    if record.get("protocol_compliance") != "registered":
        reasons.append("record_not_registered")
    if record.get("profile") != profile_name:
        reasons.append("wrong_profile")
    if record.get("protocol_id") != manifest.get("protocol_id"):
        reasons.append("wrong_protocol_id")
    if record.get("source_pins") != manifest.get("source_pins"):
        reasons.append("source_pins_mismatch")
    if record.get("source_checkout_tracked_clean") is not True:
        reasons.append("source_checkout_not_clean")
    open_score_git = record.get("open_score_git", {})
    open_score_commit = str(open_score_git.get("commit", ""))
    if (
        len(open_score_commit) != 40
        or any(character not in "0123456789abcdefABCDEF" for character in open_score_commit)
    ):
        reasons.append("open_score_git_commit_invalid")
    if open_score_git.get("dirty") is not False:
        reasons.append("open_score_git_not_clean")
    if open_score_git.get("status_porcelain") not in ([], None):
        reasons.append("open_score_git_status_not_empty")
    manifest_provenance = record.get("manifest_provenance", {})
    recorded_manifest_sha = manifest_provenance.get("sha256")
    if recorded_manifest_sha != sha256_file(manifest_path.resolve()):
        reasons.append("manifest_sha256_mismatch")
    try:
        recorded_manifest_path = Path(str(manifest_provenance.get("path", ""))).resolve()
    except OSError:
        recorded_manifest_path = Path()
    if recorded_manifest_path != manifest_path.resolve():
        reasons.append("manifest_path_mismatch")
    if manifest_provenance.get("outer_commit") != open_score_commit:
        reasons.append("manifest_outer_commit_mismatch")
    if verify_git_blob:
        try:
            blob_sha = committed_blob_sha256(
                Path(str(open_score_git.get("repository_root"))),
                open_score_commit,
                str(manifest_provenance.get("repository_relative_path", "")),
            )
        except (OSError, ValueError, subprocess.SubprocessError):
            reasons.append("manifest_commit_blob_unverifiable")
        else:
            if blob_sha != recorded_manifest_sha:
                reasons.append("manifest_commit_blob_sha256_mismatch")
        try:
            launcher_blob_sha = committed_blob_sha256(
                Path(str(open_score_git.get("repository_root"))),
                open_score_commit,
                "Open-SCORE/scripts/run_stock_reproduction.ps1",
            )
        except (OSError, ValueError, subprocess.SubprocessError):
            reasons.append("launcher_commit_blob_unverifiable")
        else:
            if launcher_blob_sha != record.get("launcher_sha256"):
                reasons.append("launcher_commit_blob_sha256_mismatch")
    post_audit = record.get("post_run_source_audit", {})
    if post_audit.get("passed") is not True:
        reasons.append("post_run_source_audit_failed")
    if post_audit.get("outer_commit") != open_score_commit:
        reasons.append("post_run_outer_commit_mismatch")
    post_upstream = post_audit.get("upstream", {})
    for name, pin_name in (
        ("epymarl", "epymarl"),
        ("smaclite", "smaclite"),
        ("rvo2", "smaclite_python_rvo2"),
    ):
        audit = post_upstream.get(name, {}) if isinstance(post_upstream, dict) else {}
        if audit.get("commit") != manifest["source_pins"][pin_name]["commit"]:
            reasons.append(f"post_run_{name}_commit_mismatch")
        if audit.get("tracked_clean") is not True:
            reasons.append(f"post_run_{name}_not_clean")
    if record.get("algorithm_label") != f"EPyMARL-{str(record.get('algorithm', '')).upper()}":
        reasons.append("algorithm_label_mismatch")
    if int(record.get("seed", -1)) not in manifest["profiles"][profile_name]["seeds"]:
        reasons.append("seed_not_registered")

    profile = manifest["profiles"][profile_name]
    expected_cpu_threads = int(profile.get("cpu_threads_per_run", 1))
    if int(record.get("cpu_threads_per_run", -1)) != expected_cpu_threads:
        reasons.append("cpu_threads_per_run_mismatch")
    require_single_core = bool(profile.get("require_single_core_affinity", False))
    if bool(record.get("require_single_core_affinity", False)) != require_single_core:
        reasons.append("cpu_affinity_requirement_mismatch")
    affinity_value = record.get("cpu_affinity_mask")
    if require_single_core:
        try:
            affinity_mask = int(affinity_value)
        except (TypeError, ValueError):
            affinity_mask = 0
        if affinity_mask <= 0 or affinity_mask & (affinity_mask - 1):
            reasons.append("cpu_affinity_not_single_core")
    runtime_cpu_probe = record.get("runtime_cpu_probe", {})
    runtime_probe_pin = manifest["source_pins"].get("runtime_cpu_probe", {})
    if runtime_cpu_probe.get("sha256") != runtime_probe_pin.get("sha256"):
        reasons.append("runtime_cpu_probe_sha256_mismatch")
    runtime_probe_result = runtime_cpu_probe.get("result", {})
    if int(runtime_probe_result.get("torch_num_threads", -1)) != expected_cpu_threads:
        reasons.append("runtime_torch_threads_mismatch")
    if require_single_core and int(
        runtime_probe_result.get("process_affinity_mask", 0)
    ) != int(record.get("cpu_affinity_mask", 0)):
        reasons.append("runtime_cpu_affinity_mismatch")
    smaclite_pin = manifest["source_pins"]["smaclite"]
    if runtime_probe_result.get("smaclite_imported_from_expected_checkout") is not True:
        reasons.append("runtime_smaclite_import_path_mismatch")
    if int(runtime_probe_result.get("stock_scenario_file_count", -1)) != int(
        smaclite_pin["stock_scenario_file_count"]
    ):
        reasons.append("runtime_stock_scenario_count_mismatch")
    if runtime_probe_result.get("stock_scenario_combined_sha256") != smaclite_pin[
        "stock_scenario_combined_sha256"
    ]:
        reasons.append("runtime_stock_scenario_sha256_mismatch")
    dependency_versions = runtime_probe_result.get("dependency_versions", {})
    for dependency in ("gymnasium", "numpy", "sacred", "smaclite", "torch", "pyyaml"):
        if not str(dependency_versions.get(dependency, "")):
            reasons.append(f"runtime_dependency_version_missing:{dependency}")
    thread_environment = record.get("thread_environment", {})
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        if str(thread_environment.get(variable, "")) != str(expected_cpu_threads):
            reasons.append(f"thread_environment_mismatch:{variable}")
        if str(
            runtime_probe_result.get("thread_environment", {}).get(variable, "")
        ) != str(expected_cpu_threads):
            reasons.append(f"runtime_thread_environment_mismatch:{variable}")
    expected_backend = "cpp" if profile["use_cpp_rvo2"] else "numpy"
    runtime_rvo2 = record.get("runtime_rvo2", {})
    rvo_pin = manifest["source_pins"]["smaclite_python_rvo2"]
    if runtime_rvo2.get("backend") != expected_backend:
        reasons.append("rvo2_backend_mismatch")
    if runtime_rvo2.get("source_commit") != rvo_pin["commit"]:
        reasons.append("rvo2_source_commit_mismatch")
    if runtime_rvo2.get("source_checkout_tracked_clean") is not True:
        reasons.append("rvo2_source_checkout_not_clean")
    if expected_backend == "cpp":
        if runtime_rvo2.get("binary_filename") != rvo_pin["binary_filename"]:
            reasons.append("rvo2_binary_filename_mismatch")
        if runtime_rvo2.get("binary_sha256") != rvo_pin["binary_sha256"]:
            reasons.append("rvo2_binary_sha256_mismatch")
    overlay = record.get("windows_compatibility_overlay", {})
    overlay_pin = manifest["source_pins"]["windows_compatibility_overlay"]
    if overlay.get("sha256") != overlay_pin["sha256"]:
        reasons.append("windows_overlay_sha256_mismatch")
    if overlay.get("sacred_capture_mode") != overlay_pin["sacred_capture_mode"]:
        reasons.append("sacred_capture_mode_mismatch")
    command = record.get("command", [])
    expected_capture_argument = f"--capture={overlay_pin['sacred_capture_mode']}"
    if not isinstance(command, list) or expected_capture_argument not in command:
        reasons.append("command_missing_registered_capture_mode")

    sacred_value = record.get("sacred_run_path")
    if not sacred_value:
        reasons.append("missing_sacred_run_path")
        return None, reasons
    sacred_path = Path(str(sacred_value)).resolve()
    try:
        sacred_path.relative_to(sacred_root.resolve())
    except ValueError:
        reasons.append("sacred_path_outside_root")
        return None, reasons

    config_path = sacred_path / "config.json"
    metrics_path = sacred_path / "metrics.json"
    run_path = sacred_path / "run.json"
    if not all(path.is_file() for path in (config_path, metrics_path, run_path)):
        reasons.append("incomplete_sacred_artifacts")
        return None, reasons
    try:
        config = load_json(config_path)
        metrics = load_json(metrics_path)
        sacred_run = load_json(run_path)
    except Exception as exc:
        reasons.append(f"invalid_sacred_json:{exc}")
        return None, reasons

    if sacred_run.get("status") != "COMPLETED":
        reasons.append("sacred_not_completed")
    expected = expected_config(manifest, profile_name, record)
    for key, expected_value in expected.items():
        if not same_value(config.get(key), expected_value):
            reasons.append(f"config_mismatch:{key}")
    env_args = config.get("env_args", {})
    env_expected = {
        "map_name": record.get("map"),
        "time_limit": int(profile["time_limit"]),
        "use_cpp_rvo2": bool(profile["use_cpp_rvo2"]),
    }
    for key, expected_value in env_expected.items():
        if not same_value(env_args.get(key), expected_value):
            reasons.append(f"env_config_mismatch:{key}")

    expected_commit = manifest["source_pins"]["epymarl"]["commit"]
    repositories = sacred_run.get("experiment", {}).get("repositories", [])
    matching_repositories = [
        repo for repo in repositories if repo.get("commit") == expected_commit
    ]
    if not matching_repositories:
        reasons.append("sacred_missing_epymarl_commit")
    elif any(repo.get("dirty") for repo in matching_repositories):
        reasons.append("sacred_epymarl_dirty")

    returns = metric_points(metrics, "test_return_mean")
    threshold = float(profile["training_steps"]) * 0.975
    terminal_returns = [(step, value) for step, value in returns if step >= threshold]
    if not terminal_returns:
        reasons.append("missing_terminal_evaluation")
        return None, reasons
    terminal_step, terminal_return = terminal_returns[-1]
    win_points = metric_points(metrics, "test_battle_won_mean")
    win_matches = [value for step, value in win_points if step == terminal_step]
    win_at_terminal = win_matches[-1] if win_matches else None
    if win_at_terminal is None:
        reasons.append("missing_terminal_battle_won_metric")
    std_points = metric_points(metrics, "test_return_std")
    std_matches = [value for step, value in std_points if step == terminal_step]
    std_at_terminal = std_matches[-1] if std_matches else None
    if std_at_terminal is None:
        reasons.append("missing_terminal_return_std")

    checkpoint_root = record.get("produced_checkpoint_root")
    checkpoint = checkpoint_fingerprint(Path(checkpoint_root) if checkpoint_root else None)
    if checkpoint is None:
        reasons.append("missing_checkpoint")
    else:
        if int(checkpoint["latest_step"]) < threshold:
            reasons.append("checkpoint_below_terminal_threshold")
        required_files = CHECKPOINT_REQUIRED_FILES[str(record.get("algorithm"))]
        missing_files = sorted(required_files - set(checkpoint["file_sha256"]))
        if missing_files:
            reasons.append("checkpoint_required_files_missing:" + ",".join(missing_files))
        recorded_checkpoint = record.get("produced_checkpoint", {})
        if int(recorded_checkpoint.get("latest_step", -1)) != int(checkpoint["latest_step"]):
            reasons.append("checkpoint_recorded_step_mismatch")
        if recorded_checkpoint.get("file_sha256") != checkpoint["file_sha256"]:
            reasons.append("checkpoint_file_sha256_drift")
        if recorded_checkpoint.get("combined_sha256") != checkpoint["combined_sha256"]:
            reasons.append("checkpoint_combined_sha256_drift")
    if reasons:
        return None, reasons
    return {
        "record_path": str(record_path.resolve()),
        "sacred_run_path": str(sacred_path),
        "algorithm_label": record["algorithm_label"],
        "algorithm": record["algorithm"],
        "map": record["map"],
        "seed": int(record["seed"]),
        "finished_at": record.get("finished_at"),
        "runtime_rvo2": runtime_rvo2,
        "open_score_git": open_score_git,
        "windows_compatibility_overlay": overlay,
        "terminal_step": terminal_step,
        "terminal_test_return_mean": terminal_return,
        "terminal_test_return_std": std_at_terminal,
        "terminal_test_battle_won_mean": win_at_terminal,
        "test_return_curve": [
            {"step": step, "value": value} for step, value in returns
        ],
        "test_win_curve": [
            {"step": step, "value": value} for step, value in win_points
        ],
        "checkpoint": checkpoint,
        "artifact_sha256": {
            "config.json": sha256_file(config_path),
            "metrics.json": sha256_file(metrics_path),
            "run.json": sha256_file(run_path),
        },
    }, []


def choose_latest_unique(runs: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[(run["algorithm"], run["map"], run["seed"])].append(run)
    selected: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for candidates in grouped.values():
        ordered = sorted(
            candidates,
            key=lambda run: str(run.get("finished_at") or ""),
            reverse=True,
        )
        selected.append(ordered[0])
        duplicates.extend(ordered[1:])
    return sorted(selected, key=lambda run: (run["algorithm"], run["map"], run["seed"])), duplicates


def aggregate_groups(
    runs: list[dict[str, Any]],
    *,
    manifest: dict[str, Any],
    profile_name: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    profile = manifest["profiles"][profile_name]
    registered_seeds = list(map(int, profile["seeds"]))
    references = paper_reference_map(manifest)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[(run["algorithm"], run["map"])].append(run)

    output: list[dict[str, Any]] = []
    for (algorithm, map_name), members in sorted(grouped.items()):
        members.sort(key=lambda run: run["seed"])
        returns = [float(run["terminal_test_return_mean"]) for run in members]
        wins = [
            float(run["terminal_test_battle_won_mean"])
            for run in members
            if run["terminal_test_battle_won_mean"] is not None
        ]
        group_bootstrap_seed = bootstrap_seed + sum(map(ord, algorithm + map_name))
        ci_low, ci_high = bootstrap_mean_ci(
            returns,
            samples=bootstrap_samples,
            seed=group_bootstrap_seed,
        )
        win_ci_low, win_ci_high = bootstrap_mean_ci(
            wins,
            samples=bootstrap_samples,
            seed=group_bootstrap_seed,
        )
        reference = references.get((algorithm, map_name))
        mean_return = statistics.fmean(returns)
        complete_seeds = sorted(run["seed"] for run in members)
        completed = complete_seeds == registered_seeds
        comparison_status = "no_paper_numeric_reference"
        delta = None
        if reference is not None:
            comparison_status = "context_only_non_gateable"
        output.append(
            {
                "algorithm_label": f"EPyMARL-{algorithm.upper()}",
                "algorithm": algorithm,
                "map": map_name,
                "expected_seeds": registered_seeds,
                "completed_seeds": complete_seeds,
                "seed_complete": completed,
                "n_seeds": len(members),
                "terminal_test_return_mean_by_seed": {
                    str(run["seed"]): run["terminal_test_return_mean"] for run in members
                },
                "terminal_test_return_mean": mean_return,
                "terminal_test_return_sample_std": (
                    statistics.stdev(returns) if len(returns) > 1 else None
                ),
                "terminal_test_return_bootstrap_95_ci": [ci_low, ci_high],
                "terminal_test_battle_won_mean": statistics.fmean(wins) if wins else None,
                "terminal_test_battle_won_bootstrap_95_ci": [
                    win_ci_low,
                    win_ci_high,
                ],
                "paper_reference": reference,
                "delta_from_paper_reference": delta,
                "delta_not_computed_reason": (
                    "non-comparable estimands: selected single model versus "
                    "five-training-seed aggregate"
                    if reference is not None
                    else None
                ),
                "comparison_status": comparison_status,
            }
        )
    return output


def write_csvs(output_path: Path, runs: list[dict[str, Any]], groups: list[dict[str, Any]]) -> tuple[Path, Path]:
    curve_path = output_path.with_name(output_path.stem + "_curves.csv")
    summary_path = output_path.with_name(output_path.stem + "_summary.csv")
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    with curve_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["algorithm", "map", "seed", "step", "test_return_mean"])
        writer.writeheader()
        for run in runs:
            for point in run["test_return_curve"]:
                writer.writerow(
                    {
                        "algorithm": run["algorithm"],
                        "map": run["map"],
                        "seed": run["seed"],
                        "step": point["step"],
                        "test_return_mean": point["value"],
                    }
                )
    fields = [
        "algorithm",
        "map",
        "n_seeds",
        "seed_complete",
        "terminal_test_return_mean",
        "terminal_test_return_sample_std",
        "terminal_test_return_bootstrap_95_ci",
        "terminal_test_battle_won_mean",
        "terminal_test_battle_won_bootstrap_95_ci",
        "paper_reference_value",
        "delta_from_paper_reference",
        "comparison_status",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for group in groups:
            reference = group.get("paper_reference")
            writer.writerow(
                {
                    **{field: group.get(field) for field in fields},
                    "paper_reference_value": reference.get("value") if reference else None,
                }
            )
    return curve_path, summary_path


def write_plot(path: Path, runs: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # optional output should fail explicitly
        raise SystemExit("matplotlib is required for --plot") from exc

    references = paper_reference_map(manifest)
    pairs = [(entry["algorithm"], entry["map"]) for entry in manifest["suites"]["primary"]]
    figure, axes = plt.subplots(1, len(pairs), figsize=(15, 4.5), squeeze=False)
    for axis, pair in zip(axes[0], pairs):
        pair_runs = [run for run in runs if (run["algorithm"], run["map"]) == pair]
        for run in pair_runs:
            x = [point["step"] for point in run["test_return_curve"]]
            y = [point["value"] for point in run["test_return_curve"]]
            axis.plot(x, y, alpha=0.35, linewidth=1, label=f"seed {run['seed']}")
        if pair_runs:
            common_length = min(len(run["test_return_curve"]) for run in pair_runs)
            mean_x = [
                statistics.fmean(run["test_return_curve"][index]["step"] for run in pair_runs)
                for index in range(common_length)
            ]
            mean_y = [
                statistics.fmean(run["test_return_curve"][index]["value"] for run in pair_runs)
                for index in range(common_length)
            ]
            axis.plot(mean_x, mean_y, color="black", linewidth=2.2, label="seed mean")
        reference = references[pair]
        axis.axhline(
            float(reference["value"]),
            color="tab:red",
            linestyle="--",
            label="paper selected single-model context (non-gating)",
        )
        axis.set_title(f"EPyMARL-{pair[0].upper()} · {pair[1]}")
        axis.set_xlabel("environment steps")
        axis.set_ylabel("test return")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    figure.suptitle("Stock SMAClite preregistered reproduction")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--records-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "stock_reproduction" / "run_records",
    )
    parser.add_argument(
        "--sacred-root",
        type=Path,
        default=PROJECT_ROOT / "upstream" / "external" / "epymarl" / "results" / "sacred",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "stock_reproduction" / "stock_results.json",
    )
    parser.add_argument("--profile", default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260831)
    parser.add_argument("--plot", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples < 100:
        raise SystemExit("--bootstrap-samples must be at least 100")
    manifest = load_json(args.manifest.resolve())
    collector_provenance = collect_and_require_git_provenance(
        PROJECT_ROOT, formal=True
    )
    profile_name = args.profile or manifest["default_profile"]
    if profile_name not in manifest["profiles"]:
        raise SystemExit(f"unknown profile: {profile_name}")

    record_paths = sorted(args.records_root.resolve().rglob("run_record.json")) if args.records_root.exists() else []
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for record_path in record_paths:
        run, reasons = validate_record(
            record_path,
            manifest=manifest,
            profile_name=profile_name,
            sacred_root=args.sacred_root.resolve(),
            manifest_path=args.manifest.resolve(),
            verify_git_blob=True,
        )
        if run is None:
            rejected.append({"record_path": str(record_path.resolve()), "reasons": reasons})
        else:
            accepted.append(run)

    selected, duplicates = choose_latest_unique(accepted)
    selected_source_commits = sorted(
        {str(run["open_score_git"]["commit"]) for run in selected}
    )
    if len(selected_source_commits) > 1:
        raise SystemExit(
            "accepted stock runs span multiple Open-SCORE commits; collect each "
            "frozen source revision separately"
        )
    if selected_source_commits and collector_provenance.get("head_commit") != selected_source_commits[0]:
        raise SystemExit(
            "formal collection must use the same clean Open-SCORE commit as the runs"
        )
    groups = aggregate_groups(
        selected,
        manifest=manifest,
        profile_name=profile_name,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    primary_pairs = set(paper_reference_map(manifest))
    primary_groups = [
        group for group in groups if (group["algorithm"], group["map"]) in primary_pairs
    ]
    expected_run_count = len(primary_pairs) * len(
        manifest["profiles"][profile_name]["seeds"]
    )
    protocol_gate_passed = (
        len(selected) == expected_run_count
        and len(primary_groups) == len(primary_pairs)
        and all(group["seed_complete"] for group in primary_groups)
    )
    result = {
        "schema_version": 1,
        "protocol_id": manifest["protocol_id"],
        "profile": profile_name,
        "claim_label": manifest["claim_label"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest.resolve()),
        "collector_git_provenance": collector_provenance,
        "open_score_git_commit": (
            selected_source_commits[0] if selected_source_commits else None
        ),
        "discovery": {
            "record_count": len(record_paths),
            "accepted_before_deduplication": len(accepted),
            "selected_run_count": len(selected),
            "rejected_count": len(rejected),
            "duplicate_count": len(duplicates),
        },
        "formal_runs": selected,
        "rejected_records": rejected,
        "excluded_duplicate_runs": duplicates,
        "groups": groups,
        "protocol_completeness_gate": {
            "passed": protocol_gate_passed,
            "kind": "15-run source/config/terminal/checkpoint integrity gate",
            "performance_claim": False,
            "required_primary_pairs": [list(pair) for pair in sorted(primary_pairs)],
            "required_seed_count_per_pair": len(
                manifest["profiles"][profile_name]["seeds"]
            ),
            "required_run_count": expected_run_count,
        },
        "paper_numeric_gate": {
            "passed": None,
            "gateable": False,
            "reason": (
                "16.22/20/24.63 are selected single-model examples, not "
                "five-training-seed aggregate estimands"
            ),
        },
        "claim_exclusions": manifest["claim_exclusions"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    curve_path, summary_path = write_csvs(args.output, selected, groups)
    if args.plot is not None:
        write_plot(args.plot.resolve(), selected, manifest)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "curve_csv": str(curve_path.resolve()),
                "summary_csv": str(summary_path.resolve()),
                "selected_runs": len(selected),
                "rejected_records": len(rejected),
                "protocol_gate_passed": protocol_gate_passed,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
