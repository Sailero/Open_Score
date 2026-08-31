"""Run independent registered SMAClite stock seeds on isolated CPU cores.

The SMAClite paper reports one CPU core per run.  This orchestrator assigns a
one-bit Windows affinity mask to each child launcher while allowing independent
seeds to run concurrently on different cores.  The authoritative protocol and
per-run evidence remain in ``run_stock_reproduction.ps1``; this file only
schedules those single-run launchers and records their logs.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "configs"
    / "stock_reproduction"
    / "smaclite_aamas2023_epymarl_v3.json"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Job:
    algorithm: str
    map_name: str
    seed: int

    @property
    def label(self) -> str:
        return f"{self.algorithm}/{self.map_name}/seed={self.seed}"


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("manifest must be a JSON object")
    return value


def select_jobs(
    manifest: dict[str, Any],
    *,
    profile_name: str,
    algorithm: str,
    map_name: str | None,
    seeds: list[int] | None,
) -> list[Job]:
    profile = manifest["profiles"][profile_name]
    selected_seeds = list(map(int, seeds or profile["seeds"]))
    registered_seeds = set(map(int, profile["seeds"]))
    if any(seed not in registered_seeds for seed in selected_seeds):
        raise ValueError("all scheduler seeds must be registered in the selected profile")
    jobs: list[Job] = []
    for entry in manifest["suites"]["primary"]:
        alg = str(entry["algorithm"])
        scenario = str(entry["map"])
        if algorithm != "all" and alg != algorithm:
            continue
        if map_name and scenario.lower() != map_name.lower():
            continue
        jobs.extend(Job(alg, scenario, seed) for seed in selected_seeds)
    if not jobs:
        raise ValueError("filters selected no primary-suite jobs")
    return jobs


def write_orchestrator_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--algorithm", choices=["all", "qmix", "vdn", "mappo"], default="all")
    parser.add_argument("--map-name", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--max-parallel", type=int, default=5)
    parser.add_argument(
        "--cpu-indices",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Required logical CPU indices used as one-bit affinity masks. "
            "Choose one hardware thread per physical core (for this i7-14700KF, "
            "the audited starting recommendation is 0 2 4 6 8)."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "stock_reproduction")
    parser.add_argument("--python-path", default=r"D:\Software\Anaconda\envs\torch310\python.exe")
    parser.add_argument("--git-path", default=r"D:\Software\Git\cmd\git.exe")
    parser.add_argument("--powershell", default="powershell.exe")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    profile_name = args.profile or str(manifest["default_profile"])
    profile = manifest["profiles"].get(profile_name)
    if not isinstance(profile, dict) or not profile.get("runnable", False):
        raise ValueError(f"profile is not runnable: {profile_name}")
    if not profile.get("require_single_core_affinity", False):
        raise ValueError("parallel paper scheduler requires a single-core-affinity profile")
    if int(profile.get("cpu_threads_per_run", 0)) != 1:
        raise ValueError("parallel paper scheduler requires cpu_threads_per_run=1")
    if profile.get("use_cuda", True):
        raise ValueError("parallel paper scheduler only accepts the registered CPU profile")

    jobs = select_jobs(
        manifest,
        profile_name=profile_name,
        algorithm=args.algorithm,
        map_name=args.map_name,
        seeds=args.seeds,
    )
    if args.max_parallel < 1:
        raise ValueError("max-parallel must be positive")
    logical_count = os.cpu_count() or 1
    if args.cpu_indices is None:
        raise ValueError(
            "--cpu-indices is required: automatic logical numbering cannot prove "
            "that simultaneous jobs do not share a physical core"
        )
    cpu_indices = args.cpu_indices
    if len(set(cpu_indices)) != len(cpu_indices):
        raise ValueError("cpu-indices must be unique")
    if any(index < 0 or index >= logical_count for index in cpu_indices):
        raise ValueError(f"cpu-indices must be in [0, {logical_count - 1}]")
    parallelism = min(args.max_parallel, len(cpu_indices), len(jobs))
    if parallelism < 1:
        raise ValueError("no CPU index is available")

    output_root = args.output_root.resolve()
    log_root = output_root / "orchestrator_logs" / profile_name
    launcher = PROJECT_ROOT / "scripts" / "run_stock_reproduction.ps1"
    record_path = output_root / "orchestrator_record.json"
    record: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": manifest["protocol_id"],
        "profile": profile_name,
        "manifest_path": str(manifest_path),
        "started_at": utc_now(),
        "finished_at": None,
        "status": "running",
        "machine": {
            "platform": platform.platform(),
            "logical_cpu_count": logical_count,
        },
        "max_parallel": parallelism,
        "cpu_indices": cpu_indices[:parallelism],
        "jobs": [
            {**asdict(job), "status": "pending", "exit_code": None, "log_path": None}
            for job in jobs
        ],
    }
    write_orchestrator_record(record_path, record)

    pending: deque[tuple[int, Job]] = deque(enumerate(jobs))
    available_cpus: deque[int] = deque(cpu_indices[:parallelism])
    running: list[dict[str, Any]] = []
    failures = 0
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    environment = os.environ.copy()
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        environment[variable] = "1"

    try:
        while pending or running:
            while pending and available_cpus:
                job_index, job = pending.popleft()
                cpu_index = available_cpus.popleft()
                log_path = log_root / job.algorithm / job.map_name / f"seed_{job.seed}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_handle = log_path.open("w", encoding="utf-8", buffering=1)
                command = [
                    args.powershell,
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(launcher),
                    "-ManifestPath",
                    str(manifest_path),
                    "-Profile",
                    profile_name,
                    "-Suite",
                    "primary",
                    "-Algorithm",
                    job.algorithm,
                    "-MapName",
                    job.map_name,
                    "-Seeds",
                    str(job.seed),
                    "-OutputRoot",
                    str(output_root),
                    "-PythonPath",
                    args.python_path,
                    "-GitPath",
                    args.git_path,
                    "-CpuAffinityMask",
                    str(1 << cpu_index),
                ]
                if args.dry_run:
                    command.append("-DryRun")
                process = subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    creationflags=creation_flags,
                )
                item = record["jobs"][job_index]
                item.update(
                    {
                        "status": "running",
                        "started_at": utc_now(),
                        "cpu_logical_index": cpu_index,
                        "cpu_affinity_mask": 1 << cpu_index,
                        "pid": process.pid,
                        "log_path": str(log_path),
                        "command": command,
                    }
                )
                running.append(
                    {
                        "job_index": job_index,
                        "job": job,
                        "cpu_index": cpu_index,
                        "process": process,
                        "log_handle": log_handle,
                    }
                )
                print(f"[started cpu={cpu_index} pid={process.pid}] {job.label}", flush=True)
                write_orchestrator_record(record_path, record)

            completed: list[dict[str, Any]] = []
            for state in running:
                exit_code = state["process"].poll()
                if exit_code is None:
                    continue
                state["log_handle"].close()
                available_cpus.append(state["cpu_index"])
                item = record["jobs"][state["job_index"]]
                item.update(
                    {
                        "status": "completed" if exit_code == 0 else "failed",
                        "exit_code": exit_code,
                        "finished_at": utc_now(),
                    }
                )
                if exit_code != 0:
                    failures += 1
                print(
                    f"[{item['status']} exit={exit_code}] {state['job'].label} -> {item['log_path']}",
                    flush=True,
                )
                completed.append(state)
                write_orchestrator_record(record_path, record)
            for state in completed:
                running.remove(state)
            if running and not completed:
                time.sleep(2.0)
    except BaseException:
        record["status"] = "interrupted"
        for state in running:
            state["process"].terminate()
            state["log_handle"].close()
        raise
    finally:
        record["finished_at"] = utc_now()
        if record["status"] != "interrupted":
            record["status"] = "completed" if failures == 0 else "completed_with_failures"
        write_orchestrator_record(record_path, record)

    print(f"orchestrator finished: jobs={len(jobs)} failures={failures}", flush=True)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
