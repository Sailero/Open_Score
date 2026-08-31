"""Collect a reusable Stage-2 HAD dataset from Stage-1 policies/checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from open_score.stage2 import (  # noqa: E402
    iter_had_records,
    read_records_jsonl,
    write_records_csv,
)


def _resolve(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> dict:
    git = r"D:\Software\Git\cmd\git.exe"
    try:
        commit = subprocess.run(
            [git, "rev-parse", "HEAD"], cwd=PROJECT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                [git, "status", "--porcelain"], cwd=PROJECT, check=True,
                capture_output=True, text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _stream_had_jsonl(data: dict, output: Path, min_continuations: int) -> dict:
    """Write records once while retaining only compact validation indices."""

    rollout_hashes = set()
    roots = {}
    cells = {}
    root_candidates = {}
    root_threats = {}
    lineages = set()
    right_censored = 0
    record_count = 0
    snapshot_attrition_audit = {}
    started = time.perf_counter()
    tracemalloc.start()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in iter_had_records(
            **data, collection_audit=snapshot_attrition_audit
        ):
            query = record.query
            rollout_hash = hashlib.sha256(query.rollout_id.encode("utf-8")).digest()
            if rollout_hash in rollout_hashes:
                raise ValueError("rollout_id must be globally unique")
            rollout_hashes.add(rollout_hash)
            static = (
                query.lineage_group_id,
                query.environment_id,
                query.scenario_id,
                query.defender_count,
                query.attacker_count,
                query.horizon_steps,
                query.command_steps,
                int(query.root_seed),
                query.canonical_state,
            )
            previous = roots.setdefault(query.root_id, static)
            if previous != static:
                raise ValueError("one streamed root contains inconsistent physical metadata")
            cell = (query.root_id, query.candidate_id, query.threat_id)
            continuations = cells.setdefault(cell, {})
            if query.continuation_id in continuations:
                raise ValueError("duplicate continuation_id in streamed candidate cell")
            continuations[query.continuation_id] = int(query.continuation_seed)
            root_candidates.setdefault(query.root_id, set()).add(query.candidate_id)
            root_threats.setdefault(query.root_id, set()).add(query.threat_id)
            lineages.add(query.lineage_group_id)
            right_censored += int(not record.event_observed)
            handle.write(
                json.dumps(record.to_dict(), ensure_ascii=False, separators=(",", ":"))
            )
            handle.write("\n")
            record_count += 1
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if not record_count:
        raise ValueError("Stage-2 streamed dataset is empty")
    snapshot_checks = {
        "audit_completed": snapshot_attrition_audit.get("status") == "recorded",
        "planned_partition_complete": (
            int(snapshot_attrition_audit.get("planned_snapshot_roots", -1))
            == int(snapshot_attrition_audit.get("realized_snapshot_roots", -2))
            + int(snapshot_attrition_audit.get("terminal_attrition_roots", -3))
            + int(snapshot_attrition_audit.get("unexplained_missing_roots", -4))
        ),
        "no_unexplained_missing_roots": int(
            snapshot_attrition_audit.get("unexplained_missing_roots", -1)
        )
        == 0,
        "all_snapshot_zero_roots_realized": int(
            snapshot_attrition_audit.get("snapshot_zero_realized_roots", -1)
        )
        == int(snapshot_attrition_audit.get("snapshot_zero_expected_roots", -2)),
        "streamed_root_count_matches_realized": len(roots)
        == int(snapshot_attrition_audit.get("realized_snapshot_roots", -1)),
        "record_count_matches_realized_grid": record_count
        == int(
            snapshot_attrition_audit.get(
                "expected_records_from_realized_roots", -1
            )
        ),
    }
    snapshot_attrition_audit["observed_records"] = record_count
    snapshot_attrition_audit["integrity_checks"] = snapshot_checks
    snapshot_attrition_audit["status"] = (
        "passed" if all(snapshot_checks.values()) else "failed"
    )
    if snapshot_attrition_audit["status"] != "passed":
        failed = [name for name, passed in snapshot_checks.items() if not passed]
        raise ValueError(
            "formal snapshot attrition audit failed: " + ", ".join(failed)
        )
    undersampled = [key for key, rows in cells.items() if len(rows) < min_continuations]
    if undersampled:
        raise ValueError(f"{len(undersampled)} streamed cells are undersampled")
    reused = [key for key, rows in cells.items() if len(set(rows.values())) != len(rows)]
    if reused:
        raise ValueError(f"{len(reused)} streamed cells reuse continuation seeds")
    blocks = {}
    for (root_id, candidate, threat), continuations in cells.items():
        blocks.setdefault((root_id, threat), {})[candidate] = continuations
    for (root_id, threat), candidates in blocks.items():
        reference_candidate, reference = next(iter(candidates.items()))
        for candidate, continuations in candidates.items():
            if continuations != reference:
                raise ValueError(
                    "streamed CRN mismatch for "
                    f"root={root_id}, threat={threat}, "
                    f"candidates={reference_candidate}/{candidate}"
                )
    candidate_sets = {tuple(sorted(values)) for values in root_candidates.values()}
    threat_sets = {tuple(sorted(values)) for values in root_threats.values()}
    if len(candidate_sets) != 1 or len(threat_sets) != 1:
        raise ValueError("streamed roots do not expose one complete candidate/threat grid")
    continuation_counts = [len(rows) for rows in cells.values()]
    elapsed = time.perf_counter() - started
    return {
        "records": record_count,
        "roots": len(roots),
        "lineage_groups": len(lineages),
        "right_censored_records": right_censored,
        "red_candidates": list(next(iter(candidate_sets))),
        "blue_threats": list(next(iter(threat_sets))),
        "snapshot_attrition_audit": snapshot_attrition_audit,
        "counterfactual_design_audit": {
            "status": "passed",
            "controllable_side": "Red",
            "blue_role": "fixed_or_sampled_threat",
            "candidate_cells": len(cells),
            "root_threat_blocks": len(blocks),
            "minimum_continuations_per_cell": min(continuation_counts),
            "mean_continuations_per_cell": sum(continuation_counts) / len(continuation_counts),
            "common_random_numbers": True,
            "red_candidates": list(next(iter(candidate_sets))),
            "blue_threats": list(next(iter(threat_sets))),
            "complete_candidate_threat_grid": True,
            "validation_mode": "single_pass_compact_indices",
        },
        "collection_performance": {
            "elapsed_seconds": elapsed,
            "records_per_second": record_count / elapsed,
            "python_tracemalloc_current_bytes": current_bytes,
            "python_tracemalloc_peak_bytes": peak_bytes,
            "streaming": True,
            "all_records_materialized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT / "configs" / "stage2_had_formal_collect.yaml",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    git_revision = _git_revision()
    if bool(config.get("formal_collection", False)) and git_revision.get("dirty") is not False:
        raise RuntimeError(
            "formal Stage-2 collection requires a clean committed Git worktree"
        )
    data = dict(config["data"])
    if data.pop("source") != "had":
        raise ValueError("collection script requires data.source=had")
    data["scales"] = [tuple(scale) for scale in data["scales"]]
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    data["device"] = device
    if "seed_range" in data:
        values = dict(data.pop("seed_range"))
        start, count = int(values["start"]), int(values["count"])
        data["seeds"] = list(range(start, start + count))
    write_csv = bool(data.pop("write_csv", not bool(config.get("formal_collection", False))))
    for key in ("defender_candidates", "attacker_threats"):
        for spec in data.get(key, []):
            if isinstance(spec, dict) and spec.get("kind") == "checkpoint":
                spec["path"] = str(_resolve(spec["path"]).resolve())
    output = (
        args.output
        or _resolve(
            config.get(
                "output", "outputs/stage2_had_formal_frozen/dataset.jsonl"
            )
        )
    ).resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"refusing to overwrite frozen Stage-2 dataset without --overwrite: {output}"
        )
    streamed = _stream_had_jsonl(
        data,
        output,
        min_continuations=int(config.get("minimum_continuations_per_cell", 2)),
    )
    csv_path = None
    if write_csv:
        # Optional compatibility artifact for small runs only.  Formal v4 uses
        # JSONL as its sole frozen source to avoid a second materialised copy.
        csv_path = output.with_suffix(".csv")
        write_records_csv(read_records_jsonl(output), csv_path)
    file_bytes = output.stat().st_size
    manifest = {
        "protocol": "OpenSCORE-Stage2-HAD-v4",
        "record_schema_version": 4,
        "records": streamed["records"],
        "roots": streamed["roots"],
        "lineage_groups": streamed["lineage_groups"],
        "red_candidates": streamed["red_candidates"],
        "blue_threats": streamed["blue_threats"],
        "right_censored_records": streamed["right_censored_records"],
        "counterfactual_design_audit": streamed["counterfactual_design_audit"],
        "snapshot_attrition_audit": streamed["snapshot_attrition_audit"],
        "label_semantics": {
            "payoff_red": (
                "local_window_safety_utility: +1 iff the asset survives the "
                "observation window, including administrative right-censoring; "
                "not an uncensored eventual-game payoff"
            ),
            "timeout": (
                "administrative right-censoring only; reaching the natural HAD "
                "max_steps horizon is an observed defender terminal"
            ),
        },
        "collection_performance": streamed["collection_performance"],
        "storage": {
            "jsonl_bytes": file_bytes,
            "mean_jsonl_bytes_per_record": file_bytes / streamed["records"],
            "csv_written": write_csv,
            "encoding": "UTF-8 JSONL, compact separators, one record per line",
        },
        "dataset_sha256": _sha256(output),
        "config_sha256": _sha256(args.config.resolve()),
        "git": git_revision,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "jsonl": str(output),
        "csv": str(csv_path) if csv_path is not None else None,
    }
    output.with_name("dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
