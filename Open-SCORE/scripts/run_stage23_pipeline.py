"""Run aligned Stage 2, resolve its frozen artifacts, then run Stage 3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Mapping

import yaml


PROJECT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "formal"), default="formal")
    parser.add_argument(
        "--stage2-config", type=Path, default=PROJECT / "configs/stage2_aligned.yaml"
    )
    parser.add_argument(
        "--stage3-config", type=Path, default=PROJECT / "configs/stage3_aligned.yaml"
    )
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument(
        "--reuse-stage2",
        action="store_true",
        help=(
            "reuse the accepted Stage2 evidence already stored below --run-root; "
            "validate its protocol and hashes, then run only Stage3 and the joint report"
        ),
    )
    return parser.parse_args()


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _resolve_project_path(value: object) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (PROJECT / path).resolve()


def _stage2_schedule(config: Mapping[str, object]) -> list[dict[str, object]]:
    """Rebuild the deterministic Stage2 schedule solely for reuse validation."""

    collection = config["collection"]
    assert isinstance(collection, Mapping)
    split = collection["split"]
    assert isinstance(split, Mapping)
    seed_base = int(collection["seed_base"])
    train_fraction = float(split["train_fraction"])
    validation_fraction = float(split["validation_fraction"])
    calibration_fraction = float(split.get("calibration_fraction", 0.0))
    specifications = (
        (
            "core",
            collection["core_scales"],
            int(collection["core_episodes_per_cell"]),
            float(collection["core_training_weight"]),
        ),
        (
            "sparse",
            collection["sparse_scales"],
            int(collection["sparse_episodes_per_cell"]),
            float(collection["sparse_training_weight"]),
        ),
        (
            "heldout",
            collection["heldout_scales"],
            int(collection["heldout_episodes_per_cell"]),
            0.0,
        ),
    )
    result: list[dict[str, object]] = []
    serial = 0
    for group, scales, episodes_per_cell, training_weight in specifications:
        if not scales:
            continue
        reserved = 3 if calibration_fraction > 0.0 else 2
        train_stop = max(
            1,
            min(episodes_per_cell - reserved, int(episodes_per_cell * train_fraction)),
        )
        validation_count = max(
            1,
            min(
                episodes_per_cell
                - train_stop
                - (2 if calibration_fraction > 0.0 else 1),
                int(episodes_per_cell * validation_fraction),
            ),
        )
        validation_stop = train_stop + validation_count
        calibration_count = (
            max(
                1,
                min(
                    episodes_per_cell - validation_stop - 1,
                    int(episodes_per_cell * calibration_fraction),
                ),
            )
            if calibration_fraction > 0.0
            else 0
        )
        calibration_stop = validation_stop + calibration_count
        for scale in scales:
            red, blue = map(int, scale)
            for opponent in collection["opponents"]:
                for episode_index in range(episodes_per_cell):
                    if group == "heldout":
                        subset = "generalization"
                    elif episode_index < train_stop:
                        subset = "train"
                    elif episode_index < validation_stop:
                        subset = "validation"
                    elif episode_index < calibration_stop:
                        subset = "calibration"
                    else:
                        subset = "test"
                    result.append(
                        {
                            "serial": serial,
                            "episode_id": f"{group}-{red}v{blue}-{opponent}-{episode_index:04d}",
                            "group": group,
                            "scale": [red, blue],
                            "scale_label": f"{red}v{blue}",
                            "opponent": str(opponent),
                            "episode_index": episode_index,
                            "seed": seed_base + serial,
                            "split": subset,
                            "training_weight": training_weight,
                        }
                    )
                    serial += 1
    random.Random(int(config["seed"])).shuffle(result)
    return result


def validate_reusable_stage2(config_path: Path, root: Path) -> dict[str, object]:
    """Fail closed unless existing Stage2 evidence is accepted and immutable."""

    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Stage2 config must be a mapping")
    stage2 = root / "stage2"
    paths = {
        "status": stage2 / "pipeline_status.json",
        "manifest": stage2 / "data/dataset_manifest.json",
        "dataset": stage2 / "data/dynamic_outcome_time.jsonl",
        "checkpoint": stage2 / "model/dynamic_outcome_time_model.pt",
        "metrics": stage2 / "model/metrics.json",
        "report": stage2 / "stage2_report.md",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "cannot reuse Stage2; missing required evidence: " + ", ".join(missing)
        )
    status = json.loads(paths["status"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    schedule = _stage2_schedule(config)
    expected_schedule = json_sha256(schedule)
    expected_training = json_sha256(config["training"])
    dataset_hash = sha256(paths["dataset"])
    checkpoint_hash = sha256(paths["checkpoint"])
    stage1 = _resolve_project_path(config["stage1_checkpoint"])
    if not stage1.is_file():
        raise FileNotFoundError(f"Stage1 checkpoint required by Stage2 is missing: {stage1}")
    stage1_hash = sha256(stage1)
    checks = {
        "status_completed": status.get("status") == "completed",
        "status_accepted": status.get("accepted") is True,
        "metrics_completed": metrics.get("status") == "completed",
        "metrics_accepted": metrics.get("acceptance", {}).get("all_passed") is True,
        "manifest_completed": manifest.get("status") == "completed",
        "episode_count": int(manifest.get("episodes", -1)) == len(schedule),
        "schedule_hash": manifest.get("schedule_sha256") == expected_schedule,
        "dataset_manifest_hash": manifest.get("dataset_sha256") == dataset_hash,
        "dataset_metrics_hash": metrics.get("dataset_sha256") == dataset_hash,
        "checkpoint_metrics_hash": metrics.get("checkpoint_sha256") == checkpoint_hash,
        "training_config_hash": metrics.get("training_config_sha256") == expected_training,
        "stage1_hash": manifest.get("stage1_checkpoint_sha256") == stage1_hash,
        "execution_semantics": (
            manifest.get("execution_semantics") == config.get("execution_semantics")
            and metrics.get("execution_semantics") == config.get("execution_semantics")
        ),
        "supported_roster": (
            manifest.get("supported_roster") == config.get("supported_roster")
            and metrics.get("supported_roster") == config.get("supported_roster")
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "cannot reuse Stage2; validation failed: " + ", ".join(failed)
        )
    return {
        "validated_at": now(),
        "stage2_config": str(config_path),
        "stage2_config_sha256": sha256(config_path),
        "checks": checks,
        "dataset_sha256": dataset_hash,
        "checkpoint_sha256": checkpoint_hash,
        "stage1_sha256": stage1_hash,
        "schedule_sha256": expected_schedule,
        "training_config_sha256": expected_training,
    }


def preserve_previous_failure(root: Path) -> Path | None:
    """Keep one concise, traceback-free record before recovery overwrites status."""

    target = root / "diagnostics/stage3_failed_attempt.json"
    if target.is_file():
        return target
    values: dict[str, object] = {}
    for label, path in (
        ("pipeline", root / "pipeline_status.json"),
        ("stage3", root / "stage3/pipeline_status.json"),
    ):
        if path.is_file():
            current = json.loads(path.read_text(encoding="utf-8"))
            if current.get("status") == "failed":
                values[label] = {
                    key: current.get(key)
                    for key in ("status", "phase", "started_at", "failed_at", "error")
                    if current.get(key) is not None
                }
    if not values:
        return None
    write_json(target, {"preserved_at": now(), **values})
    return target


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class PipelineLog:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def write(self, message: str) -> None:
        line = f"[{now()}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as target:
            target.write(line + "\n")

    def run(self, label: str, command: list[str]) -> None:
        self.write(f"[{label}] 启动：{' '.join(command)}")
        environment = os.environ.copy()
        environment.update(
            {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
        )
        process = subprocess.Popen(
            command,
            cwd=PROJECT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=environment,
        )
        assert process.stdout is not None
        for raw in process.stdout:
            line = raw.rstrip("\r\n")
            print(line, flush=True)
            with self.path.open("a", encoding="utf-8") as target:
                target.write(line + "\n")
        code = process.wait()
        if code != 0:
            raise RuntimeError(f"{label} failed with exit code {code}")
        self.write(f"[{label}] 完成")


def resolved_stage3_config(template_path: Path, root: Path) -> Path:
    config = yaml.safe_load(template_path.resolve().read_text(encoding="utf-8"))
    s1 = PROJECT / config["artifacts"]["stage1_checkpoint"]
    s2 = root / "stage2/model/dynamic_outcome_time_model.pt"
    dataset = root / "stage2/data/dynamic_outcome_time.jsonl"
    if not all(path.is_file() for path in (s1, s2, dataset)):
        raise FileNotFoundError("one or more frozen Stage1/Stage2 artifacts are missing")
    config["output_dir"] = str((root / "stage3").relative_to(PROJECT)).replace("\\", "/")
    config["artifacts"].update(
        {
            "stage1_sha256": sha256(s1),
            "stage2_checkpoint": str(s2.relative_to(PROJECT)).replace("\\", "/"),
            "stage2_sha256": sha256(s2),
            "stage2_dataset": str(dataset.relative_to(PROJECT)).replace("\\", "/"),
            "stage2_dataset_sha256": sha256(dataset),
        }
    )
    target = root / "resolved_stage3_config.yaml"
    target.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return target


def main() -> None:
    args = parse_args()
    if args.reuse_stage2 and args.mode != "formal":
        raise ValueError("--reuse-stage2 is only valid with --mode formal")
    python = args.python.resolve()
    if not python.is_file():
        raise FileNotFoundError(python)
    if args.run_root:
        root = args.run_root.resolve()
    elif args.mode == "smoke":
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = PROJECT / f"outputs/_stage23_aligned_smoke_{stamp}"
    else:
        root = PROJECT / "outputs/mvp/stage23_identity_blotto_v5"
    root.mkdir(parents=True, exist_ok=True)
    previous_failure = preserve_previous_failure(root) if args.reuse_stage2 else None
    log = PipelineLog(root / "pipeline.log")
    status_path = root / "pipeline_status.json"
    started = now()
    write_json(
        status_path,
        {
            "status": "running",
            "phase": "initialization",
            "mode": args.mode,
            "started_at": started,
            "reuse_stage2": bool(args.reuse_stage2),
            "previous_failure": str(previous_failure) if previous_failure else None,
        },
    )
    try:
        reuse_validation: dict[str, object] | None = None
        if args.reuse_stage2:
            write_json(
                status_path,
                {
                    "status": "running",
                    "phase": "stage2_reuse_validation",
                    "mode": args.mode,
                    "started_at": started,
                    "reuse_stage2": True,
                },
            )
            reuse_validation = validate_reusable_stage2(args.stage2_config, root)
            validation_path = root / "reused_stage2_validation.json"
            write_json(validation_path, reuse_validation)
            log.write(
                "[Stage2] 已验证并冻结复用：accepted=true，"
                f"dataset={reuse_validation['dataset_sha256'][:12]}…，"
                f"checkpoint={reuse_validation['checkpoint_sha256'][:12]}…"
            )
        if not args.skip_tests:
            write_json(
                status_path,
                {
                    "status": "running",
                    "phase": "tests",
                    "started_at": started,
                    "reuse_stage2": bool(args.reuse_stage2),
                },
            )
            log.run(
                "测试",
                [
                    str(python),
                    "-u",
                    "-m",
                    "pytest",
                    "-q",
                    "tests/test_stage2_aligned.py",
                    "tests/test_stage3_blotto.py",
                    "tests/test_stage3_belief.py",
                    "tests/test_had_stage3.py",
                    "tests/test_stage3_protocol.py",
                    "tests/test_stage3_identity_blotto.py",
                    "tests/test_stage3_identity_reserve.py",
                    "tests/test_stage3_reserve_runtime.py",
                    "tests/test_stage23_resume.py",
                ],
            )
        if not args.reuse_stage2:
            write_json(
                status_path,
                {"status": "running", "phase": "stage2", "started_at": started},
            )
            stage2_command = [
                str(python),
                "-u",
                "scripts/run_stage2_aligned.py",
                "--config",
                str(args.stage2_config.resolve()),
                "--output-dir",
                str(root / "stage2"),
            ]
            if args.mode == "smoke":
                stage2_command.append("--smoke")
            log.run("Stage2", stage2_command)
        else:
            log.write("[Stage2] 跳过采集与训练；仅恢复 Stage3 和联合报告")

        resolved = resolved_stage3_config(args.stage3_config, root)
        s2_status = json.loads(
            (root / "stage2/pipeline_status.json").read_text(encoding="utf-8")
        )
        log.write(
            f"[Stage2] 验收结果 accepted={s2_status.get('accepted')}；继续运行 Stage3 以保留诊断证据"
        )
        write_json(
            status_path,
            {
                "status": "running",
                "phase": "stage3_evaluation",
                "started_at": started,
                "resolved_stage3_config": str(resolved),
                "reuse_stage2": bool(args.reuse_stage2),
            },
        )
        evaluate = [
            str(python),
            "-u",
            "scripts/evaluate_stage3_identity.py",
            "--config",
            str(resolved),
            "--output-dir",
            str(root / "stage3"),
        ]
        if args.mode == "smoke":
            evaluate.append("--smoke")
        log.run("Stage3-评估", evaluate)

        write_json(status_path, {"status": "running", "phase": "report", "started_at": started})
        log.run(
            "联合报告",
            [str(python), "-u", "scripts/build_stage23_report.py", "--run-root", str(root)],
        )
        s3_status = json.loads(
            (root / "stage3/pipeline_status.json").read_text(encoding="utf-8")
        )
        write_json(
            status_path,
            {
                "status": "completed",
                "phase": "completed",
                "mode": args.mode,
                "started_at": started,
                "completed_at": now(),
                "stage2_accepted": bool(s2_status.get("accepted")),
                "stage3_accepted": bool(s3_status.get("accepted")),
                "stage2_reused": bool(args.reuse_stage2),
                "stage2_reuse_validation": (
                    str(root / "reused_stage2_validation.json")
                    if args.reuse_stage2
                    else None
                ),
                "report": str(root / "stage23_report.md"),
                "log": str(log.path),
                "resolved_stage3_config": str(resolved),
            },
        )
        log.write(f"[联合流水线] 全部完成；报告：{root / 'stage23_report.md'}")
    except Exception as error:
        write_json(
            status_path,
            {
                "status": "failed",
                "phase": "failed",
                "mode": args.mode,
                "started_at": started,
                "failed_at": now(),
                "error": f"{type(error).__name__}: {error}",
                "log": str(log.path),
                "stage2_reused": bool(args.reuse_stage2),
                "previous_failure": str(previous_failure) if previous_failure else None,
            },
        )
        log.write(f"[联合流水线] 失败：{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
