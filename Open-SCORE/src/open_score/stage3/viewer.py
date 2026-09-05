"""Reusable 16:9 viewer for one physical Stage-3 HAD episode.

The evaluator stores every upper-level joint plan. This module deterministically
replays those commands through the frozen Stage-1 controller and the native HAD
physics, then places real render snapshots beside a compact allocation view.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.animation import FuncAnimation
from matplotlib.colors import TwoSlopeNorm
from matplotlib.widgets import Slider

from open_score.envs import HADStage3Adapter

from .runtime import (
    FrozenStage1GroupExecutor,
    load_round01_stage1_model,
    moderate_jittered_target_positions,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CSV_CANDIDATES = (
    REPOSITORY_ROOT
    / "outputs/stage123_unknown_upper_v1/stage3/baselines/legacy_idb/raw/physical_identity_episodes.csv",
)

# This is deliberately the viewer's consumer contract, rather than every
# diagnostic column emitted by the evaluator.  Extra analysis columns may be
# added without breaking replay, while a legacy target-count CSV fails early
# with an actionable message instead of a later KeyError.
REQUIRED_GROUPED_RAW_COLUMNS = frozenset(
    {
        "scenario",
        "red",
        "blue",
        "targets",
        "blue_lower_style",
        "blue_upper_policy",
        "red_commander",
        "seed",
        "outcome_red",
        "red_win",
        "episode_horizon",
        "steps",
        "commands",
        "joint_plans_json",
        "episode",
    }
)


@dataclass(frozen=True)
class EpisodeReplay:
    frames: tuple[np.ndarray, ...]
    steps: tuple[int, ...]
    group_labels: tuple[tuple[str, ...], ...]
    reserve_counts: tuple[int, ...]
    reproduced_outcome: int


def _default_csv() -> Path:
    for candidate in DEFAULT_CSV_CANDIDATES:
        if candidate.is_file():
            return candidate
    return DEFAULT_CSV_CANDIDATES[0]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay and visualize one raw Stage-3 HAD episode."
    )
    parser.add_argument("--csv", type=Path, default=_default_csv())
    parser.add_argument("--config", type=Path)
    parser.add_argument("--targets", type=int, choices=range(1, 11), default=5)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--scenario", default="")
    parser.add_argument("--commander", default="identity_blotto")
    parser.add_argument(
        "--blue-upper", dest="blue_upper", default="identity_blotto_equilibrium"
    )
    parser.add_argument("--blue-style", dest="blue_style", default="rush")
    parser.add_argument("--interval-ms", type=int, default=1200)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--no-play", action="store_true", help="Open paused at the first command."
    )
    parser.add_argument(
        "--list", action="store_true", help="List matching episodes and exit."
    )
    return parser.parse_args(argv)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"raw episode CSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        columns = set(reader.fieldnames or ())
        missing = sorted(REQUIRED_GROUPED_RAW_COLUMNS - columns)
        if missing:
            raise ValueError(
                "CSV is not a replayable grouped Stage-3 raw file; missing "
                f"columns: {', '.join(missing)}"
            )
        return list(reader)


def _base_match(row: Mapping[str, str], args: argparse.Namespace) -> bool:
    return (
        int(row["targets"]) == int(args.targets)
        and row["red_commander"] == args.commander
        and row["blue_upper_policy"] == args.blue_upper
        and row["blue_lower_style"] == args.blue_style
        and (not args.scenario or row["scenario"] == args.scenario)
    )


def select_row(
    rows: Iterable[dict[str, str]], args: argparse.Namespace
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if _base_match(row, args) and int(row["episode"]) == int(args.episode)
    ]
    if not matches:
        raise ValueError(
            "no episode matches: "
            f"targets={args.targets}, episode={args.episode}, "
            f"commander={args.commander}, blue_upper={args.blue_upper}, "
            f"blue_style={args.blue_style}"
        )
    if len(matches) > 1:
        choices = ", ".join(row["scenario"] for row in matches)
        raise ValueError(f"episode filter is ambiguous; add --scenario: {choices}")
    return matches[0]


def allocation_arrays(row: Mapping[str, str]) -> tuple[np.ndarray, np.ndarray]:
    plans = _stored_joint_plans(row)
    targets = int(row["targets"])
    if not plans:
        raise ValueError("joint_plans_json has no grouped commands")
    red = np.asarray([plan["red_target_counts"] for plan in plans], dtype=np.int64)
    blue = np.asarray([plan["blue_target_counts"] for plan in plans], dtype=np.int64)
    if red.shape != blue.shape or red.shape != (len(plans), targets):
        raise ValueError("stored grouped target counts have the wrong shape")
    return red.T, blue.T


def _compressed_group_label(groups: Sequence[tuple[int, int]]) -> str:
    runs: list[list[object]] = []
    for pair in groups:
        if runs and runs[-1][0] == pair:
            runs[-1][1] = int(runs[-1][1]) + 1
        else:
            runs.append([pair, 1])
    labels = []
    for (red, blue), count in runs:
        item = f"{red}v{blue}"
        labels.append(f"{item}×{count}" if int(count) > 1 else item)
    return " + ".join(labels) if labels else "no Red group"


def _resolve_config(csv_path: Path, explicit: Optional[Path]) -> Path:
    if explicit is not None:
        path = explicit.resolve()
    else:
        run_config = csv_path.resolve().parents[2] / "resolved_stage3_config.yaml"
        path = (
            run_config
            if run_config.is_file()
            else REPOSITORY_ROOT / "configs/stage3_aligned.yaml"
        )
    if not path.is_file():
        raise FileNotFoundError(f"resolved Stage-3 config does not exist: {path}")
    return path


def _repository_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        converted = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be an integer") from error
    try:
        exact = float(value) == float(converted)
    except (TypeError, ValueError):
        exact = False
    if not exact or converted < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return converted


def _count_vector(value: object, targets: int, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != targets:
        raise ValueError(f"{label} must contain exactly {targets} target counts")
    return tuple(
        _integer(item, f"{label}[{index}]") for index, item in enumerate(value)
    )


def _id_vector(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON list")
    result = tuple(
        _integer(item, f"{label}[{index}]") for index, item in enumerate(value)
    )
    if len(set(result)) != len(result):
        raise ValueError(f"{label} contains duplicate identities")
    return result


def _stored_joint_plans(row: Mapping[str, str]) -> list[dict[str, object]]:
    """Parse and validate the evaluator's grouped command schema."""

    value = row.get("joint_plans_json", "")
    if not value:
        return []
    try:
        decoded = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("joint_plans_json is not valid JSON") from error
    if not isinstance(decoded, list):
        raise ValueError("joint_plans_json must be a JSON list")

    targets = _integer(row.get("targets"), "targets", minimum=1)
    recorded_steps = _integer(row.get("steps"), "steps")
    expected_commands = _integer(row.get("commands"), "commands")
    if len(decoded) != expected_commands:
        raise ValueError(
            "joint_plans_json command count differs from the raw commands column"
        )

    previous_step = -1
    plans: list[dict[str, object]] = []
    required_plan = {
        "step",
        "red_target_counts",
        "blue_target_counts",
        "red_alive_count",
        "blue_alive_count",
        "red_reserve_ids",
        "reserve_patrol",
        "groups",
    }
    required_group = {"target", "red_ids", "blue_ids"}
    for command_index, raw_plan in enumerate(decoded):
        label = f"joint plan {command_index}"
        if not isinstance(raw_plan, dict):
            raise ValueError(f"{label} must be a JSON object")
        missing = sorted(required_plan - set(raw_plan))
        if missing:
            raise ValueError(f"{label} is missing fields: {', '.join(missing)}")
        step = _integer(raw_plan["step"], f"{label}.step")
        if step <= previous_step:
            raise ValueError("joint plan steps must be strictly increasing")
        if step >= recorded_steps:
            raise ValueError("a joint plan step must precede the recorded terminal step")
        previous_step = step
        red_counts = _count_vector(
            raw_plan["red_target_counts"], targets, f"{label}.red_target_counts"
        )
        blue_counts = _count_vector(
            raw_plan["blue_target_counts"], targets, f"{label}.blue_target_counts"
        )
        raw_groups = raw_plan["groups"]
        if not isinstance(raw_groups, list):
            raise ValueError(f"{label}.groups must be a JSON list")

        observed_red = [0] * targets
        observed_blue = [0] * targets
        used_red: set[int] = set()
        used_blue: set[int] = set()
        group_keys: set[tuple[int, int]] = set()
        subgroup_ids: list[list[int]] = [[] for _ in range(targets)]
        for group_index, raw_group in enumerate(raw_groups):
            group_label = f"{label}.groups[{group_index}]"
            if not isinstance(raw_group, dict):
                raise ValueError(f"{group_label} must be a JSON object")
            missing = sorted(required_group - set(raw_group))
            if missing:
                raise ValueError(
                    f"{group_label} is missing fields: {', '.join(missing)}"
                )
            target = _integer(raw_group["target"], f"{group_label}.target")
            subgroup = _integer(
                raw_group.get("channel", raw_group.get("group")),
                f"{group_label}.channel",
            )
            if target >= targets:
                raise ValueError(f"{group_label}.target is outside the episode")
            key = (target, subgroup)
            if key in group_keys:
                raise ValueError(f"duplicate target/group key in {label}: {key}")
            group_keys.add(key)
            subgroup_ids[target].append(subgroup)
            red_ids = _id_vector(raw_group["red_ids"], f"{group_label}.red_ids")
            blue_ids = _id_vector(raw_group["blue_ids"], f"{group_label}.blue_ids")
            if not red_ids and not blue_ids:
                raise ValueError(f"{group_label} cannot have two empty sides")
            if len(red_ids) > 4 or len(blue_ids) > 4:
                raise ValueError(f"{group_label} exceeds the hard 4v4 cap")
            if used_red.intersection(red_ids) or used_blue.intersection(blue_ids):
                raise ValueError(f"{label} assigns one identity to multiple groups")
            used_red.update(red_ids)
            used_blue.update(blue_ids)
            observed_red[target] += len(red_ids)
            observed_blue[target] += len(blue_ids)

        reserve_ids = _id_vector(
            raw_plan["red_reserve_ids"], f"{label}.red_reserve_ids"
        )
        if used_red.intersection(reserve_ids):
            raise ValueError(f"{label} puts one Red identity in a group and reserve")
        raw_patrol = raw_plan["reserve_patrol"]
        if not isinstance(raw_patrol, list):
            raise ValueError(f"{label}.reserve_patrol must be a JSON list")
        patrol_ids: set[int] = set()
        for patrol_index, raw_item in enumerate(raw_patrol):
            patrol_label = f"{label}.reserve_patrol[{patrol_index}]"
            if not isinstance(raw_item, dict):
                raise ValueError(f"{patrol_label} must be a JSON object")
            agent_id = _integer(raw_item.get("red_id"), f"{patrol_label}.red_id")
            target = _integer(raw_item.get("target"), f"{patrol_label}.target")
            if target >= targets:
                raise ValueError(f"{patrol_label}.target is outside the episode")
            waypoint = np.asarray(raw_item.get("waypoint"), dtype=np.float64)
            if waypoint.shape != (3,) or not np.all(np.isfinite(waypoint)):
                raise ValueError(f"{patrol_label}.waypoint must be a finite 3-vector")
            if agent_id in patrol_ids:
                raise ValueError(f"{label} repeats a reserve patrol identity")
            patrol_ids.add(agent_id)
        if patrol_ids != set(reserve_ids):
            raise ValueError(
                f"{label} reserve patrol must cover exactly the Red reserve identities"
            )

        # Empty public channels are omitted from the raw log, so used channel
        # IDs need only be unique; they do not need to be contiguous.
        if tuple(observed_red) != red_counts or tuple(observed_blue) != blue_counts:
            raise ValueError(f"{label} group identities disagree with target counts")
        red_alive_count = _integer(raw_plan["red_alive_count"], f"{label}.red_alive_count")
        blue_alive_count = _integer(raw_plan["blue_alive_count"], f"{label}.blue_alive_count")
        if len(used_red) + len(reserve_ids) != red_alive_count:
            raise ValueError(f"{label} does not cover every live Red exactly once")
        if len(used_blue) != blue_alive_count:
            raise ValueError(f"{label} does not cover every live Blue exactly once")
        plans.append(raw_plan)
    return plans


def replay_episode(
    row: Mapping[str, str],
    red: np.ndarray,
    blue: np.ndarray,
    *,
    config_path: Path,
    device: str = "cpu",
) -> EpisodeReplay:
    """Reproduce command-boundary HAD frames from one raw evaluation row."""

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    physical = config["physical_evaluation"]
    targets = int(row["targets"])
    commands = red.shape[1]
    if blue.shape != red.shape:
        raise ValueError("Red and Blue command arrays must share one shape")
    target_positions = moderate_jittered_target_positions(
        targets,
        physical["layout_generator"],
        seed=int(row["seed"]),
    )
    adapter = HADStage3Adapter(
        int(row["red"]),
        int(row["blue"]),
        targets,
        max_steps=int(row.get("episode_horizon") or physical["max_steps"]),
        target_positions=target_positions,
        blue_rule_style=row["blue_lower_style"],
        split_spacing=float(physical["blue_split_spacing"]),
    )
    adapter.reset(
        seed=int(row["seed"]),
    )
    checkpoint = _repository_path(config["artifacts"]["stage1_checkpoint"])
    expected_hash = str(config["artifacts"].get("stage1_sha256", ""))
    if expected_hash == "AUTO":
        expected_hash = ""
    model = load_round01_stage1_model(
        checkpoint,
        torch.device(device),
        expected_sha256=expected_hash,
    )
    executor = FrozenStage1GroupExecutor(
        model,
        torch.device(device),
        micro_grouping="planned_groups",
    )
    executor.reset()
    recorded_steps = int(row["steps"])
    stored_plans = _stored_joint_plans(row)
    if stored_plans and len(stored_plans) != commands:
        raise ValueError("joint_plans_json does not match the command count")

    frames: list[np.ndarray] = []
    steps: list[int] = []
    command_groups: list[tuple[str, ...]] = []
    command_reserves: list[int] = []
    done = False
    info: Mapping[str, object] = {"outcome_red": 0.0}
    roster_override = None
    reserve_waypoints = None
    blue_subgroups = None

    def advance(stop: int) -> None:
        nonlocal done, info
        while adapter.step_count < stop and not done:
            if roster_override is None:
                red_actions = {
                    agent_id: 0
                    for agent_id, value in adapter.agent_states("Red").items()
                    if value["alive"]
                }
            else:
                red_actions = executor.act(
                    adapter,
                    local_steps={target: adapter.step_count for target in adapter.target_ids},
                    roster_override=roster_override,
                    reserve_waypoints=reserve_waypoints,
                )
            _, _, done, info = adapter.step(
                red_actions,
                blue_style=row["blue_lower_style"],
                blue_subgroup_by_agent=blue_subgroups,
            )

    for plan in stored_plans:
        plan_step = int(plan["step"])
        advance(plan_step)
        if done or adapter.step_count != plan_step:
            raise RuntimeError("stored command step cannot be reached during replay")
        red_assignment = {agent_id: None for agent_id in adapter.red_ids}
        blue_assignment = {agent_id: None for agent_id in adapter.blue_ids}
        live_red = {
            int(agent_id)
            for agent_id, state in adapter.agent_states("Red").items()
            if state["alive"]
        }
        live_blue = {
            int(agent_id)
            for agent_id, state in adapter.agent_states("Blue").items()
            if state["alive"]
        }
        roster_override = {}
        reserve_waypoints = {
            int(item["red_id"]): np.asarray(item["waypoint"], dtype=np.float64)
            for item in plan["reserve_patrol"]
        }
        blue_subgroups = {}
        groups_by_target: list[list[tuple[int, int]]] = [
            [] for _ in range(targets)
        ]
        for group in plan["groups"]:
            target = int(group["target"])
            subgroup = int(group.get("channel", group.get("group")))
            red_ids = tuple(map(int, group["red_ids"]))
            blue_ids = tuple(map(int, group["blue_ids"]))
            for agent_id in red_ids:
                red_assignment[agent_id] = target
            for agent_id in blue_ids:
                blue_assignment[agent_id] = target
                blue_subgroups[agent_id] = subgroup
            roster_override[(target, subgroup)] = (red_ids, blue_ids)
            groups_by_target[target].append((len(red_ids), len(blue_ids)))
        active_red = {
            int(agent_id) for group in plan["groups"] for agent_id in group["red_ids"]
        }
        active_blue = {
            int(agent_id) for group in plan["groups"] for agent_id in group["blue_ids"]
        }
        reserve_ids = set(map(int, plan["red_reserve_ids"]))
        if active_red.intersection(reserve_ids) or active_red.union(reserve_ids) != live_red:
            raise RuntimeError("replay plan does not partition live Red into active or reserve")
        if active_blue != live_blue:
            raise RuntimeError("replay plan does not partition every live Blue exactly once")
        adapter.set_joint_assignments(red_assignment, blue_assignment)
        if any(adapter.red_assignment[int(value)] is not None for value in plan["red_reserve_ids"]):
            raise RuntimeError("replay reserve identity acquired a formal target")
        command_groups.append(
            tuple(_compressed_group_label(value) for value in groups_by_target)
        )
        command_reserves.append(len(plan["red_reserve_ids"]))
        frames.append(adapter.env.render_rgb_array(include_result=True))
        steps.append(adapter.step_count)

    advance(recorded_steps)

    if len(frames) != commands or adapter.step_count != recorded_steps:
        raise RuntimeError(
            "the raw row could not be reproduced exactly: "
            f"commands={len(frames)}/{commands}, steps={adapter.step_count}/{recorded_steps}"
        )
    reproduced = int(float(info.get("outcome_red", 0.0)))
    expected = int(row["outcome_red"])
    if reproduced != expected:
        raise RuntimeError(
            f"replayed outcome {reproduced} differs from raw outcome {expected}"
        )
    return EpisodeReplay(
        frames=tuple(frames),
        steps=tuple(steps),
        group_labels=tuple(command_groups),
        reserve_counts=tuple(command_reserves),
        reproduced_outcome=reproduced,
    )


def _target_labels(targets: int) -> list[str]:
    return [f"Target {index + 1}" for index in range(targets)]


def _style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor("#ffffff")
    for spine in axis.spines.values():
        spine.set_color("#d9dee8")


def show_episode(
    row: Mapping[str, str],
    red: np.ndarray,
    blue: np.ndarray,
    replay: EpisodeReplay,
    *,
    interval_ms: int = 1200,
    no_play: bool = False,
) -> None:
    """Open the compact 16:9 scientific episode dashboard."""

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    targets, commands = red.shape
    labels = _target_labels(targets)
    gap = red - blue
    active_gap = np.ma.masked_where(blue == 0, gap)
    extent = max(1, int(np.max(np.abs(gap))))
    max_count = max(1, int(red.max()), int(blue.max()))

    figure = plt.figure(figsize=(16, 9), facecolor="#f4f6fa")
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.62, 1.0),
        height_ratios=(1.0, 1.0),
        left=0.035,
        right=0.975,
        top=0.88,
        bottom=0.12,
        wspace=0.14,
        hspace=0.30,
    )
    render_axis = figure.add_subplot(grid[:, 0])
    allocation_axis = figure.add_subplot(grid[0, 1])
    heat_axis = figure.add_subplot(grid[1, 1])
    for axis in (render_axis, allocation_axis, heat_axis):
        _style_axis(axis)

    render_image = render_axis.imshow(replay.frames[0])
    render_axis.set_title("HAD physical snapshot", loc="left", fontsize=13, pad=10)
    render_axis.set_xticks([])
    render_axis.set_yticks([])

    heat_axis.set_facecolor("#e9edf4")
    heat = heat_axis.imshow(
        active_gap,
        aspect="auto",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-extent, vcenter=0, vmax=extent),
    )
    heat_axis.set_title("Allocation history  (cell: Red / Blue)", loc="left", fontsize=12)
    heat_axis.set_xticks(np.arange(commands), [f"C{i + 1}" for i in range(commands)])
    heat_axis.set_yticks(np.arange(targets), labels)
    heat_axis.set_xlabel("Upper-level command")
    for target in range(targets):
        for command in range(commands):
            color = "#626b78" if int(blue[target, command]) == 0 else "#172033"
            heat_axis.text(
                command,
                target,
                f"{int(red[target, command])}/{int(blue[target, command])}",
                ha="center",
                va="center",
                fontsize=8.5,
                color=color,
                fontweight="medium",
            )
    marker = heat_axis.axvline(0, color="#f6c344", linewidth=3, alpha=0.95)
    colorbar = figure.colorbar(heat, ax=heat_axis, fraction=0.045, pad=0.03)
    colorbar.set_label("Red − Blue")

    result = "RED WIN" if int(row["red_win"]) else "TARGET BREACHED"
    title = (
        f"{row['scenario']}  ·  {row['red_commander']} vs "
        f"{row['blue_upper_policy']} / {row['blue_lower_style']}"
    )
    figure.suptitle(title, x=0.035, ha="left", fontsize=16, fontweight="semibold")
    status = figure.text(
        0.975,
        0.94,
        result,
        ha="right",
        va="center",
        fontsize=12,
        color="#b42318" if int(row["red_win"]) else "#175cd3",
        fontweight="bold",
    )
    state = {"frame": 0, "paused": bool(no_play)}

    def draw_current(command: int) -> None:
        state["frame"] = int(command)
        render_image.set_data(replay.frames[command])
        allocation_axis.clear()
        _style_axis(allocation_axis)
        y = np.arange(targets)
        red_now = red[:, command]
        blue_now = blue[:, command]
        allocation_axis.barh(y, -red_now, color="#d94b4b", height=0.62, label="Red")
        allocation_axis.barh(y, blue_now, color="#3f7ad6", height=0.62, label="Blue")
        allocation_axis.axvline(0, color="#8d96a5", linewidth=0.9)
        allocation_axis.set_yticks(y, labels)
        allocation_axis.invert_yaxis()
        allocation_axis.set_xlim(-1.22 * max_count, 1.72 * max_count)
        allocation_axis.set_title(
            f"Command {command + 1}  ·  physical step {replay.steps[command]}",
            loc="left",
            fontsize=12,
        )
        allocation_axis.set_xlabel("Red  ←  assigned agents  →  Blue")
        allocation_axis.grid(axis="x", color="#e4e8ef", linewidth=0.8)
        allocation_axis.legend(frameon=False, ncol=2, loc="lower right")
        for target in range(targets):
            allocation_axis.text(
                1.08 * max_count,
                target,
                replay.group_labels[command][target],
                va="center",
                fontsize=8.5,
                color="#344054",
            )
        marker.set_xdata([command, command])
        status.set_text(
            f"{result}  ·  C{command + 1}/{commands}  ·  step {replay.steps[command]}"
            f"  ·  Red reserve {replay.reserve_counts[command]}"
        )
        figure.canvas.draw_idle()

    slider_axis = figure.add_axes([0.19, 0.047, 0.62, 0.025], facecolor="#e5e9f1")
    slider = Slider(
        slider_axis,
        "Command",
        valmin=1,
        valmax=commands,
        valinit=1,
        valstep=1,
        color="#3f7ad6",
    )
    slider.on_changed(lambda value: draw_current(int(value) - 1))

    def animate(frame: int):
        if not state["paused"]:
            slider.set_val((frame % commands) + 1)
        return render_image, marker

    animation = FuncAnimation(
        figure,
        animate,
        frames=range(commands),
        interval=max(100, int(interval_ms)),
        repeat=True,
        blit=False,
    )

    def key_press(event) -> None:
        key = str(event.key).lower()
        if key == " ":
            state["paused"] = not state["paused"]
            if state["paused"]:
                animation.event_source.stop()
            else:
                animation.event_source.start()
        elif key in {"left", "right", "home", "end"}:
            state["paused"] = True
            animation.event_source.stop()
            if key == "left":
                selected = max(0, state["frame"] - 1)
            elif key == "right":
                selected = min(commands - 1, state["frame"] + 1)
            elif key == "home":
                selected = 0
            else:
                selected = commands - 1
            slider.set_val(selected + 1)

    figure.canvas.mpl_connect("key_press_event", key_press)
    draw_current(0)
    if no_play:
        animation.event_source.stop()
    print("Controls: Space play/pause; Left/Right previous/next; Home/End first/last")
    plt.show()


def list_matching(rows: Iterable[dict[str, str]], args: argparse.Namespace) -> None:
    matches = [row for row in rows if _base_match(row, args)]
    if not matches:
        print("No episodes match the requested strategy filters.")
        return
    print("episode  scenario             result           steps  commands")
    for row in matches:
        result = "Red-win" if int(row["red_win"]) else "breached"
        print(
            f"{int(row['episode']):7d}  {row['scenario'][:20]:20s} "
            f"{result:15s}  {int(row['steps']):5d}  {int(row['commands']):8d}"
        )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    csv_path = args.csv.resolve()
    rows = read_rows(csv_path)
    if args.list:
        list_matching(rows, args)
        return
    row = select_row(rows, args)
    red, blue = allocation_arrays(row)
    config_path = _resolve_config(csv_path, args.config)
    print(
        f"Replaying {row['scenario']} episode {row['episode']} "
        f"({red.shape[1]} commands, {row['steps']} physical steps)...",
        flush=True,
    )
    replay = replay_episode(
        row,
        red,
        blue,
        config_path=config_path,
        device=args.device,
    )
    print("Replay verified against the recorded outcome. Opening viewer...", flush=True)
    show_episode(
        row,
        red,
        blue,
        replay,
        interval_ms=args.interval_ms,
        no_play=args.no_play,
    )


if __name__ == "__main__":
    main()
