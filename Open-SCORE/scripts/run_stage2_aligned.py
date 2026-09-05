"""Train the Stage-3-aligned Stage-2 local outcome/duration model.

Each rollout executes one actual Red/Blue group at one target.  Saved casualty
states contain only the currently alive roster and retain the terminal label of
their own episode: they are behavior-policy, on-trajectory continuation-value
samples, not freshly initialized smaller matchups.  Model selection,
probability calibration and final testing use disjoint episode splits.  Empty
side outcomes remain exact analytical boundaries rather than learned queries.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

import stage2_pipeline_common as common  # noqa: E402
from open_score.envs import HADStage3Adapter  # noqa: E402
from open_score.stage2 import HADCanonicalizer, HADVariableSetState  # noqa: E402
from open_score.stage3.payoff import FrozenStage2Payoff  # noqa: E402
from open_score.stage3.runtime import FrozenStage1GroupExecutor  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT / "configs" / "stage2_aligned.yaml",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--reuse-data-from",
        type=Path,
        help=(
            "reuse a completed aligned Stage2 run (or its JSONL dataset) and "
            "run only training, calibration, shape audit, and reporting"
        ),
    )
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--minimum-epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def load_config(args: argparse.Namespace) -> dict[str, object]:
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("aligned Stage-2 config must be a mapping")
    config = deepcopy(config)
    if args.device:
        config["device"] = args.device
    if args.output_dir:
        config["output_dir"] = str(args.output_dir)
    if args.smoke:
        collection = config["collection"]
        # Exercise a singleton, the full small grid and the maximum supported
        # roster without spending formal-run time.
        collection["core_scales"] = [[1, 1], [2, 1], [4, 4]]
        collection["sparse_scales"] = []
        collection["heldout_scales"] = []
        collection["core_episodes_per_cell"] = 4
        collection["sparse_episodes_per_cell"] = 0
        collection["heldout_episodes_per_cell"] = 0
        config["training"]["max_epochs"] = 2
        config["training"]["minimum_epochs"] = 1
        config["training"]["patience"] = 1
        config["training"]["batch_size"] = 32
        if not args.output_dir:
            config["output_dir"] = "outputs/_stage2_aligned_smoke"
    training = config["training"]
    for argument, key in (
        (args.training_seed, "seed"),
        (args.max_epochs, "max_epochs"),
        (args.minimum_epochs, "minimum_epochs"),
        (args.patience, "patience"),
        (args.learning_rate, "learning_rate"),
    ):
        if argument is not None:
            training[key] = argument
    return config


def _alive_ids(adapter: HADStage3Adapter, side: str) -> tuple[int, ...]:
    return tuple(
        agent_id
        for agent_id, state in adapter.agent_states(side).items()
        if bool(state["alive"])
    )


def _episode_state(adapter: HADStage3Adapter) -> HADVariableSetState:
    """Encode only the currently alive and therefore deployable agents."""

    red_ids = _alive_ids(adapter, "Red")
    blue_ids = _alive_ids(adapter, "Blue")
    if not red_ids or not blue_ids:
        raise ValueError("a learned Stage2 query requires one live agent per side")
    entities = adapter.local_state_entities(
        0, red_ids, blue_ids, local_step=adapter.step_count
    )
    return HADCanonicalizer().to_entity_set(entities)


def _run_episode(
    cell: Mapping[str, object],
    stage1_model: torch.nn.Module,
    device: torch.device,
    config: Mapping[str, object],
) -> tuple[
    list[tuple[int, str, HADVariableSetState]],
    int,
    int,
    list[tuple[int, int]],
]:
    red_count, blue_count = map(int, cell["scale"])
    collection = config["collection"]
    adapter = HADStage3Adapter(
        red_count,
        blue_count,
        1,
        max_steps=int(collection["max_steps"]),
        target_positions=[[-2100.0, 0.0, 100.0]],
        blue_rule_style=str(cell["opponent"]),
        split_spacing=float(collection.get("blue_split_spacing", 180.0)),
    )
    adapter.reset(
        seed=int(cell["seed"]),
        red_assignment={agent_id: 0 for agent_id in adapter.red_ids},
        blue_assignment={agent_id: 0 for agent_id in adapter.blue_ids},
    )
    executor = FrozenStage1GroupExecutor(
        stage1_model,
        device,
        micro_grouping=str(
            config["execution_semantics"].get(
                "micro_grouping", "attacker_matched"
            )
        ),
    )
    executor.reset()
    snapshot_steps = set(map(int, collection["snapshot_steps"]))
    snapshots: list[tuple[int, str, HADVariableSetState]] = []
    snapshot_rosters: list[tuple[int, int]] = []
    previous_roster: tuple[tuple[int, ...], tuple[int, ...]] | None = None
    done = False
    info: dict[str, object] = {"outcome_red": 0.0}
    while not done:
        current_roster = (
            _alive_ids(adapter, "Red"),
            _alive_ids(adapter, "Blue"),
        )
        current_alive = tuple(len(roster) for roster in current_roster)
        fixed_time = adapter.step_count in snapshot_steps
        casualty_event = (
            previous_roster is not None and current_roster != previous_roster
        )
        # Empty-side states are exact analytical boundaries in Stage2 and
        # must never be sent through the learned non-empty roster encoder.
        if (fixed_time or casualty_event) and all(value > 0 for value in current_alive):
            reason = (
                "initial"
                if adapter.step_count == 0
                else "casualty_and_time"
                if fixed_time and casualty_event
                else "casualty"
                if casualty_event
                else "time_anchor"
            )
            snapshots.append((adapter.step_count, reason, _episode_state(adapter)))
            snapshot_rosters.append(current_alive)
        # A Stage2 datum represents one current group pair.  When casualties
        # change the roster, explicitly discard recurrent state before acting
        # with the survivor roster.  FrozenStage1GroupExecutor also guards this
        # invariant internally by rebuilding a controller for a changed roster.
        # This is still an on-trajectory survivor state, not a freshly sampled
        # initial condition and not an enumeration of counterfactual groups.
        if (fixed_time or casualty_event) and adapter.step_count > 0:
            executor.reset()
        red_actions = executor.act(
            adapter,
            local_steps={0: adapter.step_count},
            roster_override={(0, 0): current_roster},
        )
        previous_roster = current_roster
        _, _, done, info = adapter.step(
            red_actions, blue_style=str(cell["opponent"])
        )
    return snapshots, int(float(info["outcome_red"]) > 0.0), adapter.step_count, snapshot_rosters


def collect_dataset(
    config: Mapping[str, object],
    output_dir: Path,
    device: torch.device,
    log: common.LiveLog,
) -> tuple[Path, dict[str, object]]:
    dataset = output_dir / "data" / "dynamic_outcome_time.jsonl"
    manifest_path = output_dir / "data" / "dataset_manifest.json"
    schedule = common._schedule(config)
    schedule_sha = common._json_sha(schedule)
    if dataset.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schedule_sha256") == schedule_sha
            and manifest.get("dataset_sha256") == common._sha256(dataset)
            and manifest.get("execution_semantics")
            == config["execution_semantics"]
        ):
            log.write(
                f"[Stage2][采集] 复用已完成数据：{manifest['episodes']:,} 局，"
                f"{manifest['rows']:,} 个状态"
            )
            return dataset, manifest
        raise ValueError("existing aligned dataset does not match this protocol")

    stage1_checkpoint = common._resolve(config["stage1_checkpoint"]).resolve()
    if not stage1_checkpoint.is_file():
        raise FileNotFoundError(stage1_checkpoint)
    stage1_model = common._load_stage1_controller(
        stage1_checkpoint, device
    ).model
    dataset.parent.mkdir(parents=True, exist_ok=True)
    partial = dataset.with_suffix(".partial.jsonl")
    state_path = dataset.with_name("collection_state.json")
    completed = 0
    byte_offset = 0
    if partial.is_file() and state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("schedule_sha256") != schedule_sha:
            raise ValueError("partial dataset belongs to another schedule")
        completed = int(state["completed_schedule_items"])
        byte_offset = int(state["byte_offset"])
        with partial.open("r+b") as handle:
            handle.truncate(byte_offset)
        log.write(f"[Stage2][采集] 从断点继续：{completed:,}/{len(schedule):,}")
    else:
        partial.write_bytes(b"")

    total_rows = 0
    split_episodes: Counter[str] = Counter()
    group_episodes: Counter[str] = Counter()
    per_scale: dict[str, dict[str, int]] = defaultdict(
        lambda: {"episodes": 0, "red_wins": 0, "rows": 0}
    )
    micro_counts: Counter[str] = Counter()
    if completed:
        with partial.open("r", encoding="utf-8") as source:
            seen: set[str] = set()
            for line in source:
                row = json.loads(line)
                origin_scale = str(row.get("origin_scale", row["scale"]))
                total_rows += 1
                # Manifest per-scale statistics describe the independently
                # initialized cell, not a survivor count reached mid-episode.
                per_scale[origin_scale]["rows"] += 1
                if row["episode_id"] not in seen:
                    seen.add(row["episode_id"])
                    split_episodes[row["split"]] += 1
                    group_episodes[row["scale_group"]] += 1
                    per_scale[origin_scale]["episodes"] += 1
                    per_scale[origin_scale]["red_wins"] += int(row["red_win"])
                    initial_red = int(row.get("initial_red_count", row["red_count"]))
                    initial_blue = int(
                        row.get("initial_blue_count", row["blue_count"])
                    )
                    micro_counts[f"{initial_red}v{initial_blue}"] += 1

    resume_completed = completed
    started = time.perf_counter()
    stride = max(1, int(config["collection"].get("progress_every_episodes", 50)))
    checkpoint_stride = max(
        1, int(config["collection"].get("checkpoint_every_episodes", 10))
    )
    with partial.open("ab") as target:
        for schedule_index in range(completed, len(schedule)):
            cell = schedule[schedule_index]
            snapshots, red_win, terminal_step, micro = _run_episode(
                cell, stage1_model, device, config
            )
            if not snapshots:
                raise RuntimeError("aligned episode produced no non-terminal snapshot")
            episode_mass = float(cell["training_weight"])
            payload = bytearray()
            for step, snapshot_reason, encoded in snapshots:
                current_red = int(len(encoded.red_entities))
                current_blue = int(len(encoded.blue_entities))
                row = {
                    "episode_id": cell["episode_id"],
                    "split": cell["split"],
                    "scale_group": cell["group"],
                    "scale": f"{current_red}v{current_blue}",
                    "origin_scale": cell["scale_label"],
                    "initial_red_count": int(cell["scale"][0]),
                    "initial_blue_count": int(cell["scale"][1]),
                    "current_red_count": current_red,
                    "current_blue_count": current_blue,
                    # Compatibility fields now mean the current alive roster.
                    "red_count": current_red,
                    "blue_count": current_blue,
                    "opponent": cell["opponent"],
                    "episode_seed": int(cell["seed"]),
                    "step": int(step),
                    "snapshot_reason": snapshot_reason,
                    "terminal_step": int(terminal_step),
                    "remaining_steps": int(terminal_step - step),
                    "red_win": int(red_win),
                    "target": encoded.target.tolist(),
                    "red_entities": encoded.red_entities.tolist(),
                    "blue_entities": encoded.blue_entities.tolist(),
                    "context": encoded.context.tolist(),
                    "training_weight": episode_mass / len(snapshots),
                    "evaluation_weight": 1.0 / len(snapshots),
                    "execution_semantics": "single_group_replanned_continuation",
                }
                payload.extend(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode(
                        "utf-8"
                    )
                )
                payload.extend(b"\n")
            target.write(payload)
            target.flush()
            total_rows += len(snapshots)
            split_episodes[str(cell["split"])] += 1
            group_episodes[str(cell["group"])] += 1
            scale_value = per_scale[str(cell["scale_label"])]
            scale_value["episodes"] += 1
            scale_value["red_wins"] += red_win
            scale_value["rows"] += len(snapshots)
            for red, blue in micro:
                micro_counts[f"{red}v{blue}"] += 1
            completed = schedule_index + 1
            if completed % checkpoint_stride == 0 or completed == len(schedule):
                common._write_json(
                    state_path,
                    {
                        "schedule_sha256": schedule_sha,
                        "completed_schedule_items": completed,
                        "byte_offset": target.tell(),
                        "updated_at": common._now(),
                    },
                )
            if completed % stride == 0 or completed == len(schedule):
                elapsed = time.perf_counter() - started
                rate = max(1, completed - resume_completed) / max(elapsed, 1e-8)
                eta = (len(schedule) - completed) / max(rate, 1e-8)
                log.write(
                    f"[Stage2][采集] {completed:,}/{len(schedule):,} "
                    f"({completed / len(schedule):.1%}) | 状态 {total_rows:,} | "
                    f"{rate:.2f} 局/秒 | ETA {common._duration(eta)}"
                )

    partial.replace(dataset)
    if state_path.exists():
        state_path.unlink()
    manifest = {
        "schema_version": "stage2-identity-local-value-data-v4",
        "status": "completed",
        "episodes": len(schedule),
        "rows": total_rows,
        "split_episodes": dict(split_episodes),
        "group_episodes": dict(group_episodes),
        "per_scale": dict(per_scale),
        "snapshot_rosters": dict(sorted(micro_counts.items())),
        "schedule_sha256": schedule_sha,
        "dataset_sha256": common._sha256(dataset),
        "stage1_checkpoint": str(stage1_checkpoint),
        "stage1_checkpoint_sha256": common._sha256(stage1_checkpoint),
        "execution_semantics": config["execution_semantics"],
        "supported_roster": config["supported_roster"],
        "weighting": {
            "unit": "initial_episode",
            "within_episode": "mass divided equally across saved snapshots",
            "trajectory_snapshot_role": (
                "current survivor-group value with controller reset at every saved command event; "
                "not an independently initialised or counterfactual regrouping sample"
            ),
        },
        "git": common._git_provenance(),
        "created_at": common._now(),
    }
    common._write_json(manifest_path, manifest)
    log.write(
        f"[Stage2][采集] 完成：{len(schedule):,} 局，{total_rows:,} 个状态，"
        f"SHA256={manifest['dataset_sha256'][:12]}…"
    )
    return dataset, manifest


def reuse_completed_dataset(
    source: Path,
    config: Mapping[str, object],
    output_dir: Path,
    log: common.LiveLog,
) -> tuple[Path, dict[str, object]]:
    """Validate and reuse immutable aligned data without copying or recollecting it."""

    resolved = common._resolve(source).resolve()
    if resolved.is_file():
        dataset = resolved
        manifest_path = resolved.parent / "dataset_manifest.json"
    else:
        direct = resolved / "dynamic_outcome_time.jsonl"
        nested = resolved / "data" / "dynamic_outcome_time.jsonl"
        dataset = direct if direct.is_file() else nested
        manifest_path = dataset.parent / "dataset_manifest.json"
    if not dataset.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "--reuse-data-from must identify a completed Stage2 run or its "
            "dynamic_outcome_time.jsonl"
        )

    source_root = dataset.parent.parent if dataset.parent.name == "data" else dataset.parent
    output_resolved = output_dir.resolve()
    if output_resolved == source_root or source_root in output_resolved.parents:
        raise ValueError(
            "training-only output must be outside the source Stage2 run; choose "
            "a new --output-dir so existing evidence is not overwritten"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_schedule = common._json_sha(common._schedule(config))
    actual_dataset_sha = common._sha256(dataset)
    checks = {
        "completed": manifest.get("status") == "completed",
        "dataset_sha256": manifest.get("dataset_sha256") == actual_dataset_sha,
        "schedule_sha256": manifest.get("schedule_sha256") == expected_schedule,
        "execution_semantics": manifest.get("execution_semantics")
        == config["execution_semantics"],
        "supported_roster": manifest.get("supported_roster")
        == config["supported_roster"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "reused Stage2 data does not match the aligned protocol: "
            + ", ".join(failed)
        )

    reference = {
        "schema_version": "stage2-aligned-dataset-reference-v1",
        "source_dataset": str(dataset),
        "source_manifest": str(manifest_path),
        "dataset_sha256": actual_dataset_sha,
        "schedule_sha256": expected_schedule,
        "created_at": common._now(),
    }
    common._write_json(output_dir / "data" / "dataset_reference.json", reference)
    reused_manifest = dict(manifest)
    reused_manifest["reused_dataset"] = str(dataset)
    reused_manifest["reused_manifest"] = str(manifest_path)
    log.write(
        f"[Stage2][数据] 跳过采集，复用 {manifest['episodes']:,} 局、"
        f"{manifest['rows']:,} 个状态 | SHA256={actual_dataset_sha[:12]}…"
    )
    return dataset, reused_manifest


def _counterfactual(template: HADVariableSetState, red: int, blue: int) -> HADVariableSetState:
    red_entities = np.asarray(template.red_entities[:red], dtype=np.float32).copy()
    blue_entities = np.asarray(template.blue_entities[:blue], dtype=np.float32).copy()
    return HADVariableSetState(
        target=np.asarray(template.target, dtype=np.float32).copy(),
        red_entities=red_entities,
        blue_entities=blue_entities,
        context=np.asarray(
            [
                red / 4.0,
                blue / 4.0,
                float((red_entities[:, 7] > 0.5).sum()) / 4.0,
                float((blue_entities[:, 7] > 0.5).sum()) / 4.0,
                float(template.context[4]),
            ],
            dtype=np.float32,
        ),
    )


def shape_diagnostics(
    dataset: Path,
    checkpoint: Path,
    config: Mapping[str, object],
    device: torch.device,
) -> dict[str, object]:
    max_red = int(config["supported_roster"]["max_red"])
    max_blue = int(config["supported_roster"]["max_blue"])
    roots: list[tuple[str, HADVariableSetState]] = []
    with dataset.open("r", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if (
                row["split"] == "test"
                and int(row["step"]) == 0
                and int(row["red_count"]) == max_red
                and int(row["blue_count"]) == max_blue
            ):
                roots.append(
                    (
                        str(row["opponent"]),
                        HADVariableSetState(
                            target=np.asarray(row["target"], dtype=np.float32),
                            red_entities=np.asarray(row["red_entities"], dtype=np.float32),
                            blue_entities=np.asarray(row["blue_entities"], dtype=np.float32),
                            context=np.asarray(row["context"], dtype=np.float32),
                        ),
                    )
                )
            if len(roots) >= 12:
                break
    if not roots:
        raise ValueError("shape audit needs a max-roster Test root")
    predictor = FrozenStage2Payoff(checkpoint, device=device)
    monotonic_checks = 0
    monotonic_violations = 0
    diminishing_checks = 0
    diminishing_violations = 0
    absolute_adjustment = 0.0
    adjusted_cells = 0
    tolerance = 1e-4
    for style, root in roots:
        states = [
            _counterfactual(root, red, blue)
            for red in range(1, max_red + 1)
            for blue in range(1, max_blue + 1)
        ]
        roster_values = [
            (red, blue)
            for red in range(1, max_red + 1)
            for blue in range(1, max_blue + 1)
        ]
        raw_probabilities = predictor.predict_red_win(
            states,
            styles=[style] * len(states),
            rosters=roster_values,
            batch_size=int(config["training"]["batch_size"]),
        ).reshape(max_red, max_blue)
        probabilities = predictor.predict_roster_surface(
            states,
            target_count=1,
            red_cap=max_red,
            blue_cap=max_blue,
            styles=[style] * len(states),
            rosters=roster_values,
            batch_size=int(config["training"]["batch_size"]),
        ).reshape(max_red, max_blue)
        absolute_adjustment += float(
            np.abs(probabilities - raw_probabilities).sum()
        )
        adjusted_cells += probabilities.size
        red_difference = np.diff(probabilities, axis=0)
        blue_difference = np.diff(probabilities, axis=1)
        monotonic_checks += red_difference.size + blue_difference.size
        monotonic_violations += int((red_difference < -tolerance).sum())
        monotonic_violations += int((blue_difference > tolerance).sum())
        logits = np.log(np.clip(probabilities, 1e-5, 1.0 - 1e-5)) - np.log1p(
            -np.clip(probabilities, 1e-5, 1.0 - 1e-5)
        )
        red_second = np.diff(logits, n=2, axis=0)
        blue_second = np.diff(logits, n=2, axis=1)
        diminishing_checks += red_second.size + blue_second.size
        diminishing_violations += int((red_second > tolerance).sum())
        diminishing_violations += int((blue_second < -tolerance).sum())
    monotonic_rate = monotonic_violations / max(1, monotonic_checks)
    diminishing_rate = diminishing_violations / max(1, diminishing_checks)
    acceptance = config["acceptance"]
    return {
        "roots": len(roots),
        "grid": [max_red, max_blue],
        "structural_boundaries": {
            "p_red_survival_r_vs_0": 1.0,
            "p_red_survival_0_vs_b": 0.0,
            "passed": True,
        },
        "monotonic_checks": monotonic_checks,
        "monotonic_violations": monotonic_violations,
        "monotonic_violation_rate": monotonic_rate,
        "diminishing_checks": diminishing_checks,
        "diminishing_violations": diminishing_violations,
        "diminishing_violation_rate": diminishing_rate,
        "shape_projection_mean_absolute_adjustment": absolute_adjustment
        / max(1, adjusted_cells),
        "acceptance": {
            "monotonic": monotonic_rate
            <= float(acceptance["monotonic_violation_rate_max"]),
            "diminishing": diminishing_rate
            <= float(acceptance["diminishing_violation_rate_max"]),
            "projection_size": absolute_adjustment / max(1, adjusted_cells)
            <= float(acceptance["shape_projection_mae_max"]),
        },
    }


def _legacy_write_report(
    output: Path,
    manifest: Mapping[str, object],
    metrics: Mapping[str, object],
    shape: Mapping[str, object],
) -> Path:
    def metric(value: object, digits: int = 4) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "—"
        return "—" if not np.isfinite(number) else f"{number:.{digits}f}"

    groups = metrics["group_metrics"]
    accepted = bool(metrics["acceptance"]["all_passed"]) and all(
        bool(value) for value in shape["acceptance"].values()
    )
    lines = [
        "# Stage2：与 Stage3 执行语义对齐的局部结果模型",
        "",
        f"> 自动生成时间：`{common._now()}`  ",
        f"> 总体判定：**{'通过' if accepted else '未通过'}**",
        "",
        "本轮把每个 rollout 定义为一个真实局部组对，而不是先采集目标总人数再事后切块。模型选择、概率校准和最终测试按 episode lineage 分离；1–5 为主域，6–8 仅作稀疏边界锚点。",
        "",
        f"所用固定数据集包含 `{manifest['episodes']:,}` 局、`{manifest['rows']:,}` 个非终止状态。支持范围为 Red 1–{manifest['supported_roster']['max_red']}、Blue 1–{manifest['supported_roster']['max_blue']}；空方组合由任务边界精确给定。",
        "",
        "| 测试集合 | Brier↓ | ECE↓ | AUC↑ | 剩余时间 MAE↓ |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in (
        ("core_test", "1–5 主域（含轨迹态）"),
        ("sparse_test", "6–8 边界锚点（含轨迹态）"),
        ("heldout_generalization", "未参与训练的规模"),
    ):
        value = groups[key]
        lines.append(
            f"| {label} | {metric(value.get('brier'))} | "
            f"{metric(value.get('ece_10'))} | {metric(value.get('auc'))} | "
            f"{metric(value.get('time_mae_steps'), 2)} 步 |"
        )
    lines.extend(
        [
            "",
            "| 结构检查 | 结果 |",
            "|---|---:|",
            "| `p(r,0)=1`、`p(0,b)=0` | 通过 |",
            f"| 单调性违反率 | {shape['monotonic_violation_rate']:.2%} |",
            f"| 边际递减违反率 | {shape['diminishing_violation_rate']:.2%} |",
            f"| 形状校准平均绝对修正 | {shape['shape_projection_mean_absolute_adjustment']:.4f} |",
            "",
            "## 文件入口",
            "",
            "- `live_progress.log`：采集、训练和评估实时进度。",
            f"- 训练数据：`{metrics['dataset']}`。",
            "- `model/dynamic_outcome_time_model.pt`：冻结模型、温度和规则风格校准。",
            "- `model/metrics.json`：完整测试指标与验收结果。",
            "- `model/shape_diagnostics.json`：8×8 局部组人数网格结构诊断。",
            "- `figures/`：训练、分规模与校准图。",
            "",
        ]
    )
    report = output / "stage2_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def write_report(
    output: Path,
    manifest: Mapping[str, object],
    metrics: Mapping[str, object],
    shape: Mapping[str, object],
) -> Path:
    """Write the concise reader-facing identity-local Stage2 report."""

    core = metrics["group_metrics"]["core_test"]
    accepted = bool(metrics["acceptance"]["all_passed"]) and all(
        bool(value) for value in shape["acceptance"].values()
    )

    def value(raw: object, digits: int = 4) -> str:
        if raw is None:
            return "—"
        return f"{float(raw):.{digits}f}"

    lines = [
        "# Stage2：身份级局部对抗价值模型",
        "",
        f"> 自动生成：`{common._now()}`  ",
        f"> 验收：**{'通过' if accepted else '未通过'}**",
        "",
        "本轮只定义和训练 1v1 至 4v4 的单目标、单局部组对抗。16 个非空人数配比均由独立初始化 episode 直接采样；不存在 4 人以上的训练、外推或合法动作。空方边界使用解析值 `p(r,0)=1`、`p(0,b)=0`。",
        "",
        "每局在固定指挥时刻和伤亡时刻保存当前存活身份的状态，并重置冻结 Stage1 控制器。伤亡后的状态是该真实态势的重规划续局样本，不冒充独立初始化的小规模对局。数据按原始 episode 划分，避免同一局轨迹泄漏到训练与测试两侧。",
        "",
        f"共采集 `{int(manifest['episodes']):,}` 个独立 episode、`{int(manifest['rows']):,}` 个非终止状态。模型使用置换不变的实体集合编码，因此保留成员的物理状态与阵营身份，同时不学习没有物理含义的固定编号嵌入。",
        "",
        "| 测试指标 | 结果 |",
        "|---|---:|",
        f"| Brier↓ | {value(core.get('brier'))} |",
        f"| ECE↓ | {value(core.get('ece_10'))} |",
        f"| AUC↑ | {value(core.get('auc'))} |",
        f"| 剩余时间 MAE↓ | {value(core.get('time_mae_steps'), 2)} 步 |",
        f"| 16 配比初始胜率平均绝对误差↓ | {float(metrics['core_mean_scale_win_rate_absolute_error']):.2%} |",
        "",
        "| 结构诊断 | 结果 |",
        "|---|---:|",
        "| 空方解析边界 | 通过 |",
        f"| 单调性违反率 | {float(shape['monotonic_violation_rate']):.2%} |",
        f"| 边际递减违反率 | {float(shape['diminishing_violation_rate']):.2%} |",
        f"| 形状投影平均修正量 | {float(shape['shape_projection_mean_absolute_adjustment']):.4f} |",
        "",
        "## 文件入口",
        "",
        "- `live_progress.log`：采集和训练实时进度。",
        "- `data/dynamic_outcome_time.jsonl`：逐状态训练数据。",
        "- `model/dynamic_outcome_time_model.pt`：冻结模型与独立校准参数。",
        "- `model/metrics.json`：完整测试指标。",
        "- `model/shape_diagnostics.json`：1–4×1–4 结构诊断。",
        "- `figures/`：训练、分配比和校准图。",
        "",
    ]
    report = output / "stage2_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def validate_group_value_protocol(config: Mapping[str, object]) -> None:
    """Reject silent drift from the complete 1..4 identity-local domain."""

    if config.get("schema_version") != "open-score-stage2-identity-local-value-v4":
        raise ValueError("the registered Stage2 schema must be identity-local-value-v4")
    collection = config["collection"]
    support = config["supported_roster"]
    core = {tuple(map(int, value)) for value in collection["core_scales"]}
    expected_core = {(red, blue) for red in range(1, 5) for blue in range(1, 5)}
    if core != expected_core:
        raise ValueError("Stage2 cells must be the complete 1..4 x 1..4 grid")
    anchors = {tuple(map(int, value)) for value in collection["sparse_scales"]}
    if anchors or collection["heldout_scales"]:
        raise ValueError("identity Stage2 does not collect any roster above 4v4")
    if (
        int(support["min_red"]) != 1
        or int(support["min_blue"]) != 1
        or int(support["max_red"]) != 4
        or int(support["max_blue"]) != 4
        or int(support["primary_max_red"]) != 4
        or int(support["primary_max_blue"]) != 4
    ):
        raise ValueError("the registered Stage2 local support must be 1..4 per side")
    semantics = config["execution_semantics"]
    required_semantics = {
        "local_group_semantics": "one_target_one_group_pair",
        "trajectory_snapshots": "current_alive_roster_replanned_continuation",
        "controller_reset_on_roster_change": True,
        "controller_reset_at_every_saved_command_event": True,
        "freshly_regrouped_counterfactuals": False,
        "target_count": 1,
        "exact_empty_side_boundaries": True,
    }
    drifted = {
        key: (semantics.get(key), expected)
        for key, expected in required_semantics.items()
        if semantics.get(key) != expected
    }
    if drifted:
        raise ValueError(
            "Stage2 one-group trajectory semantics drifted: " + repr(drifted)
        )
    if collection.get("snapshot_on_alive_count_change") is not True:
        raise ValueError("Stage2 must snapshot every non-empty casualty roster")


def main() -> None:
    args = parse_args()
    if args.reuse_data_from and args.smoke:
        raise ValueError("--reuse-data-from cannot be combined with --smoke")
    if args.reuse_data_from and not args.output_dir:
        raise ValueError("training-only mode requires a new --output-dir")
    config = load_config(args)
    common._validate_config(config)
    if not args.smoke:
        validate_group_value_protocol(config)
    output = common._resolve(config["output_dir"]).resolve()
    log = common.LiveLog(output)
    status_path = output / "pipeline_status.json"
    schedule = common._schedule(config)
    if args.dry_run:
        groups = Counter(str(cell["group"]) for cell in schedule)
        log.write(
            f"[Stage2][计划] 共 {len(schedule):,} 局：core={groups['core']:,}，"
            f"sparse={groups['sparse']:,}，heldout={groups['heldout']:,}"
        )
        return
    device = common._choose_device(str(config["device"]))
    started = time.perf_counter()
    common._write_json(
        status_path,
        {
            "status": "running",
            "stage": "data_reuse" if args.reuse_data_from else "collection",
            "started_at": common._now(),
            "episodes": len(schedule),
            "reuse_data_from": str(args.reuse_data_from)
            if args.reuse_data_from
            else None,
        },
    )
    try:
        log.write(
            f"[Stage2] 启动对齐训练 | device={device} | episode={len(schedule):,}"
        )
        if args.reuse_data_from:
            dataset, manifest = reuse_completed_dataset(
                args.reuse_data_from, config, output, log
            )
        else:
            dataset, manifest = collect_dataset(config, output, device, log)
        common._write_json(
            status_path,
            {"status": "running", "stage": "training", "dataset": str(dataset)},
        )
        checkpoint, metrics, _ = common._train(
            config, dataset, output, device, log
        )
        common._write_json(
            status_path,
            {"status": "running", "stage": "shape_audit", "checkpoint": str(checkpoint)},
        )
        shape = shape_diagnostics(dataset, checkpoint, config, device)
        metrics["shape_diagnostics"] = shape
        metrics["acceptance"]["shape_monotonic"] = bool(
            shape["acceptance"]["monotonic"]
        )
        metrics["acceptance"]["shape_diminishing"] = bool(
            shape["acceptance"]["diminishing"]
        )
        metrics["acceptance"]["all_passed"] = all(
            bool(value)
            for key, value in metrics["acceptance"].items()
            if key != "all_passed"
        )
        common._write_json(output / "model" / "metrics.json", metrics)
        common._write_json(output / "model" / "shape_diagnostics.json", shape)
        figures = common._plots(output, metrics)
        report = write_report(output, manifest, metrics, shape)
        status = {
            "status": "completed",
            "stage": "completed",
            "accepted": bool(metrics["acceptance"]["all_passed"]),
            "completed_at": common._now(),
            "elapsed_seconds": time.perf_counter() - started,
            "dataset": str(dataset),
            "reused_dataset": manifest.get("reused_dataset"),
            "checkpoint": str(checkpoint),
            "metrics": str(output / "model" / "metrics.json"),
            "report": str(report),
            "figures": [str(path) for path in figures],
            "live_log": str(log.path),
        }
        common._write_json(status_path, status)
        log.write(
            f"[Stage2] 完成 | accepted={status['accepted']} | "
            f"用时 {common._duration(status['elapsed_seconds'])} | 报告 {report}"
        )
    except Exception as error:
        common._write_json(
            status_path,
            {
                "status": "failed",
                "stage": "failed",
                "failed_at": common._now(),
                "error": f"{type(error).__name__}: {error}",
                "live_log": str(log.path),
            },
        )
        log.write(f"[Stage2] 失败：{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
