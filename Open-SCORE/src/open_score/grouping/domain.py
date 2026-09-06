"""Identity-preserving, label-free actions and observable decision states."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Mapping


@dataclass(frozen=True)
class Entity:
    id: int
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    health: float

    def __post_init__(self):
        object.__setattr__(self, "id", int(self.id))
        object.__setattr__(self, "position", tuple(map(float, self.position)))
        object.__setattr__(self, "velocity", tuple(map(float, self.velocity)))
        object.__setattr__(self, "health", float(self.health))
        if len(self.position) != 3 or len(self.velocity) != 3:
            raise ValueError("HAD positions and velocities have three coordinates")

    @property
    def alive(self) -> bool:
        return self.health > 0


@dataclass(frozen=True, order=True)
class Group:
    target: int
    members: tuple[int, ...]

    def __post_init__(self):
        object.__setattr__(self, "target", int(self.target))
        object.__setattr__(self, "members", tuple(sorted(map(int, self.members))))


@dataclass(frozen=True)
class Grouping:
    groups: tuple[Group, ...]
    reserve: tuple[int, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "groups", tuple(sorted(self.groups)))
        object.__setattr__(self, "reserve", tuple(sorted(map(int, self.reserve))))

    def canonical(self) -> Grouping:
        """Groups have no numeric labels; construction already canonicalizes."""
        return self

    def validate(self, live_ids: Iterable[int], target_ids: Iterable[int], *,
                 max_members: int | None = 4) -> Grouping:
        live, targets = tuple(map(int, live_ids)), set(map(int, target_ids))
        used = [i for group in self.groups for i in group.members] + list(self.reserve)
        if len(live) != len(set(live)):
            raise ValueError("Live identities must be unique")
        if len(used) != len(set(used)) or set(used) != set(live):
            raise ValueError("Every live identity must occur exactly once")
        if max_members is not None and int(max_members) < 1:
            raise ValueError("max_members must be positive or None")
        if any(group.target not in targets or len(group.members) < 1
               or (max_members is not None and len(group.members) > max_members)
               for group in self.groups):
            raise ValueError("Every group needs a valid target and a supported positive size")
        return self

    def prune(self, live_ids: Iterable[int]) -> Grouping:
        """Remove casualties while preserving all surviving group relations."""
        live = set(map(int, live_ids))
        groups = tuple(Group(group.target, tuple(i for i in group.members if i in live))
                       for group in self.groups if any(i in live for i in group.members))
        return Grouping(groups, tuple(i for i in self.reserve if i in live))

    def assignment(self) -> dict[int, int | None]:
        return {**{i: group.target for group in self.groups for i in group.members},
                **{i: None for i in self.reserve}}

    def to_dict(self) -> dict:
        return {"groups": [{"target": group.target, "members": list(group.members)}
                           for group in self.groups], "reserve": list(self.reserve)}

    @classmethod
    def from_dict(cls, data: Mapping) -> Grouping:
        return cls(tuple(Group(row["target"], tuple(row["members"]))
                         for row in data["groups"]), tuple(data.get("reserve", ())))


@dataclass(frozen=True)
class DecisionState:
    step: int
    max_steps: int
    opponent: str
    red: tuple[Entity, ...]
    blue: tuple[Entity, ...]
    targets: tuple[Entity, ...]
    previous: Grouping
    memory: dict[int, tuple[float, ...]] = field(default_factory=dict)
    last_actions: dict[int, int] = field(default_factory=dict)

    def alive(self, side: str) -> tuple[Entity, ...]:
        key = side.lower()
        if key == "target":
            key = "targets"
        if key not in {"red", "blue", "targets"}:
            raise ValueError("side must be red, blue or targets")
        return tuple(entity for entity in getattr(self, key) if entity.alive)

    def ids(self, side: str) -> tuple[int, ...]:
        return tuple(entity.id for entity in self.alive(side))

    def to_dict(self) -> dict:
        return {"step": int(self.step), "max_steps": int(self.max_steps),
                "opponent": self.opponent,
                "red": [asdict(entity) for entity in self.red],
                "blue": [asdict(entity) for entity in self.blue],
                "targets": [asdict(entity) for entity in self.targets],
                "previous": self.previous.to_dict(),
                "memory": {str(i): list(values) for i, values in self.memory.items()},
                "last_actions": {str(i): int(value) for i, value in self.last_actions.items()}}

    @classmethod
    def from_dict(cls, data: Mapping) -> DecisionState:
        return cls(int(data["step"]), int(data["max_steps"]), str(data["opponent"]),
                   *(tuple(Entity(**row) for row in data[side]) for side in ("red", "blue", "targets")),
                   Grouping.from_dict(data["previous"]),
                   {int(i): tuple(map(float, row)) for i, row in data.get("memory", {}).items()},
                   {int(i): int(value) for i, value in data.get("last_actions", {}).items()})
