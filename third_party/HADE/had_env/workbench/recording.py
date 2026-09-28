"""Complete physical frames and honest, read-only imports of historical traces."""
from __future__ import annotations

from bisect import bisect_right
import copy
from dataclasses import asdict, dataclass, field, is_dataclass
import gzip
import json
from pathlib import Path
import uuid

import numpy as np

SCHEMA = "had-workbench-episode-v1"


def serializable(value):
    if is_dataclass(value):
        return serializable(asdict(value))
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _groups(value):
    if value is None:
        return []
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if isinstance(value, dict):
        value = value.get("groups", [])
    return [{"target_id": row.get("target_id", row.get("target")), "members": list(row["members"])}
            for row in value]


def capture_frame(adapter, *, groups=None, info=None):
    """Copy a Stage-3 adapter after a physical step, without changing its state."""
    from had_env.core.config import Interval
    info = info or {}
    entities = []
    for side, rows in (("red", adapter.agent_states("Red")), ("blue", adapter.agent_states("Blue")),
                       ("targets", adapter.target_states())):
        for entity_id, row in sorted(rows.items()):
            entity = dict(id=int(entity_id), side=side, role=row.get("type", "target" if side == "targets" else "Attack"),
                          position=list(row["position"]), velocity=list(row["velocity"]),
                          health=float(row["health"]), alive=bool(row["alive"]))
            if "entity_id" in row:
                entity["entity_id"] = int(row["entity_id"])
            entity["max_health"] = float(adapter.env.targets[entity_id].initial_health) if side == "targets" else 1.
            if side == "targets":
                target = adapter.env.targets[entity_id]
                entity.update(step_damage=float(target.step_damage), cumulative_damage=float(target.cumulative_damage))
            entity["attack_range"] = adapter.env.attack_distance[1] if entity["role"] == "Attack" else None
            entities.append(entity)
    native_events = copy.deepcopy(getattr(adapter.env, "last_physics_events", []))
    events = [{**event, "phase": "physics"} for event in native_events]
    events.extend({**event, "phase": "protocol"} for event in info.get("events", []))
    assignment = adapter.assignments
    task = adapter.env.task_info()
    if adapter.step_count >= adapter.max_steps and not task["episode_done"]:
        task.update(episode_done=True, termination_reason="time_limit")
    frame = dict(step=int(adapter.step_count), sim_time=float(adapter.step_count * Interval), entities=entities,
                 groups={side: _groups((groups or {}).get(side)) for side in ("red", "blue")},
                 assignments={side.lower(): values for side, values in assignment.items()},
                 events=events, actions={"red": {}, "blue": {}}, **task)
    for side in ("red", "blue"):
        values = info.get(f"{side}_actions", [])
        ids = getattr(adapter, f"{side}_ids")
        frame["actions"][side] = dict(values) if isinstance(values, dict) else dict(zip(ids, values))
    return serializable(frame)


@dataclass
class ReplayEpisode:
    metadata: dict = field(default_factory=dict)
    frames: list[dict] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)
    sparse: bool = False

    @property
    def max_step(self):
        recorded = max((int(frame["step"]) for frame in self.frames), default=0)
        return max(recorded, int(self.metadata.get("physical_steps", recorded)))

    def frame_at(self, step):
        """Nearest preceding *recorded* frame; no interpolation or invented states."""
        if not self.frames:
            raise ValueError("Episode contains no spatial frames")
        index = max(0, bisect_right([int(frame["step"]) for frame in self.frames], int(step)) - 1)
        return copy.deepcopy(self.frames[index])

    def event_steps(self):
        return sorted({int(f["step"]) for f in self.frames if f.get("events")}
                      | {int(d["step"]) for d in self.decisions})

    def to_dict(self):
        return serializable(dict(schema=SCHEMA, metadata={**self.metadata, "sparse": self.sparse},
                                 frames=self.frames, decisions=self.decisions, sparse=self.sparse))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
        try:
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(temporary, "wt", encoding="utf-8") as stream:
                json.dump(self.to_dict(), stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return path


class EpisodeRecorder:
    def __init__(self, metadata=None):
        self.episode = ReplayEpisode(metadata=serializable(copy.deepcopy(metadata or {})))
        self.episode.metadata.setdefault("schema", SCHEMA)
        self.episode.metadata.setdefault("complete", False)

    def append_frame(self, frame):
        frame = serializable(copy.deepcopy(frame))
        if self.episode.frames and int(frame["step"]) <= int(self.episode.frames[-1]["step"]):
            raise ValueError("Physical frames must advance strictly")
        self.episode.frames.append(frame)
        for key in ("task_mode", "spatial_dim", "plane_altitude", "step_target_damage", "target_damage",
                    "target_damage_by_target", "episode_returns", "episode_done", "termination_reason"):
            if key in frame:
                self.episode.metadata[key] = copy.deepcopy(frame[key])

    def append_decision(self, decision):
        self.episode.decisions.append(serializable(copy.deepcopy(decision)))

    def finish(self, outcome):
        self.episode.metadata.update(serializable(copy.deepcopy(outcome)))
        self.episode.metadata["complete"] = True
        if self.episode.frames:
            self.episode.metadata["physical_steps"] = self.episode.frames[-1]["step"]

    def save(self, path):
        return self.episode.save(path)


def _legacy_frame(state, *, step=None, grouping=None):
    from had_env.core.config import Interval
    step = int(state.get("step", 0) if step is None else step)
    entities = []
    source = state.get("entities", state)
    for side in ("red", "blue", "targets"):
        values = source.get(side, [])
        if isinstance(values, dict):
            values = [dict(id=i, position=p) for i, p in values.items()]
        for row in values:
            entity = dict(id=int(row["id"]), side=side, role=row.get("role", "unknown"),
                          position=list(row["position"]), velocity=row.get("velocity"),
                          health=row.get("health"), alive=row.get("alive"))
            if entity["alive"] is None and entity["health"] is not None:
                entity["alive"] = entity["health"] > 0
            entities.append(entity)
    # v2/v4 grouping traces used positions for Red and blue_positions for Blue.
    for side, key in (("red", "positions"), ("blue", "blue_positions")):
        if not any(e["side"] == side for e in entities):
            entities.extend(dict(id=int(i), side=side, role="unknown", position=list(p),
                                 velocity=None, health=None, alive=None)
                            for i, p in state.get(key, {}).items())
    return dict(step=step, sim_time=step * float(Interval), entities=entities,
                groups={"red": _groups(grouping or state.get("previous")), "blue": []},
                assignments={"red": {}, "blue": {}}, events=[], actions={"red": {}, "blue": {}},
                sparse=True)


def load_episode(path):
    """Load portable recordings or sparse v5 family / v4 decision traces.

    Historical data is never rewritten and missing physical positions are never
    manufactured from target health or command-time states.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        text = stream.read()
    def invalid(value):
        raise ValueError(f"Non-finite value {value} in recording")
    try:
        data = json.loads(text, parse_constant=invalid)
    except json.JSONDecodeError:
        data = [json.loads(line, parse_constant=invalid) for line in text.splitlines() if line.strip()]
    if isinstance(data, dict) and data.get("schema") == SCHEMA:
        frames = data.get("frames", [])
        steps = [int(frame["step"]) for frame in frames]
        if steps != sorted(set(steps)):
            raise ValueError("Recording has duplicate or unordered physical frames")
        return ReplayEpisode(data.get("metadata", {}), frames, data.get("decisions", []),
                             bool(data.get("sparse", data.get("metadata", {}).get("sparse", False))))
    if isinstance(data, dict) and data.get("record_type"):
        data = [data]
    if not isinstance(data, list) or not data:
        raise ValueError("Unsupported or empty episode recording")
    if any(row.get("record_type") == "episode" for row in data):
        episodes = [row for row in data if row.get("record_type") == "episode"]
        if len(episodes) != 1:
            raise ValueError("Select one v5 family shard, not a multi-episode index")
        metadata = {**episodes[0], "legacy_schema": "v5-family", "sparse": True,
                    "complete_spatial_replay": False, "source_path": str(path)}
        frames, decisions = [], []
        for row in data:
            if row.get("record_type") != "event":
                continue
            if row.get("family_id") != episodes[0].get("family_id"):
                raise ValueError("Mixed families in historical recording")
            step = int(row.get("physical_step", row.get("state", {}).get("step", 0)))
            if row.get("state"):
                frames.append(_legacy_frame(row["state"], step=step, grouping=row.get("selected_plan_raw")))
            decisions.append({**row, "step": step, "selected_plan": row.get("selected_plan_raw"),
                              "trace": row.get("decision_trace", {})})
        metadata.setdefault("protocol_id", metadata.get("protocol_version", "legacy-v5"))
    elif all("step" in row for row in data):
        metadata = dict(legacy_schema="v4-trace", sparse=True, complete_spatial_replay=False,
                        complete=False, source_path=str(path), protocol_id="legacy-v4")
        frames = [_legacy_frame(row, grouping=row.get("grouping")) for row in data]
        decisions = [dict(step=int(row["step"]), selected_plan=row.get("grouping"), trace={}) for row in data]
    else:
        raise ValueError("Unsupported historical trace schema")
    frames.sort(key=lambda row: row["step"])
    if len({row["step"] for row in frames}) != len(frames):
        raise ValueError("Historical recording contains duplicate state frames")
    return ReplayEpisode(metadata, frames, decisions, sparse=True)
