"""Immutable public observations, unlabelled groups, and neutral matching."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Iterable

import numpy as np


def seed_for(*parts) -> int:
    data = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:4], "little")


@dataclass(frozen=True)
class Entity:
    id: int
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    health: float

    @property
    def alive(self):
        return self.health > 0


@dataclass(frozen=True)
class PublicState:
    step: int
    max_steps: int
    lower: str
    red: tuple[Entity, ...]
    blue: tuple[Entity, ...]
    targets: tuple[Entity, ...]
    # Own previous commands, including reserves, are public to the commander.
    red_history_counts: tuple[float, ...]

    def alive(self, side):
        return tuple(x for x in getattr(self, side.lower()) if x.alive)

    def ids(self, side):
        return tuple(x.id for x in self.alive(side))

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        values = dict(data)
        for key in ["red", "blue", "targets"]:
            values[key] = tuple(Entity(int(x["id"]), tuple(x["position"]), tuple(x["velocity"]), float(x["health"])) for x in values[key])
        values["red_history_counts"] = tuple(values["red_history_counts"])
        return cls(**values)


def observe(adapter, red_history_counts=None) -> PublicState:
    """Only explicit kinematics/health fields; never global_state or snapshot.

    IDs have no learned numeric embedding and no relation to policy names.
    This is the only official-world -> planner boundary.
    """
    def entities(records):
        return tuple(Entity(int(i), tuple(map(float, r["position"])),
                            tuple(map(float, r["velocity"])), float(r["health"]))
                     for i, r in sorted(records.items()))
    return PublicState(adapter.step_count, adapter.max_steps, adapter.blue_rule_style,
                       entities(adapter.agent_states("Red")), entities(adapter.agent_states("Blue")),
                       entities(adapter.target_states()),
                       tuple(red_history_counts or [0.0] * len(adapter.target_ids)))


@dataclass(frozen=True)
class IdentityUpperAction:
    groups: tuple[tuple[int, tuple[int, ...]], ...]
    reserve_ids: tuple[int, ...] = ()
    # Optional Red intent: (own group IDs, public Blue IDs to intercept).
    # These are physical identities, never hidden Blue group/target labels.
    intercepts: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "groups", tuple(sorted((int(t), tuple(sorted(map(int, ids)))) for t, ids in self.groups)))
        object.__setattr__(self, "reserve_ids", tuple(sorted(map(int, self.reserve_ids))))
        object.__setattr__(self, "intercepts", tuple(sorted((tuple(sorted(map(int,r))),tuple(sorted(map(int,b)))) for r,b in self.intercepts)))

    def validate(self, ids: Iterable[int], targets: Iterable[int], side="Red", blue_ids=None):
        targets, ids = set(targets), tuple(ids)
        used = [i for t, group in self.groups for i in group] + list(self.reserve_ids)
        if len(used) != len(ids) or len(set(used)) != len(used) or set(used) != set(ids):
            raise ValueError("Every live identity must occur exactly once")
        if any(t not in targets or not 1 <= len(group) <= 4 for t, group in self.groups):
            raise ValueError("Invalid target or group size outside 1..4")
        if side.lower() == "blue" and self.reserve_ids:
            raise ValueError("Blue reserves are not legal")
        if side.lower() == "blue" and self.intercepts:
            raise ValueError("Blue does not submit Red interception intents")
        own_groups = {group for _,group in self.groups}
        keys = [r for r,_ in self.intercepts]
        intended = [i for _,b in self.intercepts for i in b]
        if len(set(keys))!=len(keys) or any(r not in own_groups or not 1<=len(b)<=4 for r,b in self.intercepts):
            raise ValueError("An interception intent must name one active own group and 1..4 Blue identities")
        if len(set(intended))!=len(intended) or (blue_ids is not None and not set(intended).issubset(blue_ids)):
            raise ValueError("Interception intents repeat or reference unknown/dead Blue identities")
        return self

    def assignment(self):
        return {**{i: t for t, group in self.groups for i in group}, **{i: None for i in self.reserve_ids}}

    def counts(self, targets):
        return tuple(sum(len(ids) for t, ids in self.groups if t == target) for target in targets)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(tuple((t, tuple(ids)) for t, ids in data["groups"]), tuple(data.get("reserve_ids", ())),
                   tuple((tuple(r),tuple(b)) for r,b in data.get("intercepts",())))


def prune(action, live_ids, live_blue_ids=None):
    live = set(live_ids)
    intents = []
    for r,b in action.intercepts:
        r = tuple(i for i in r if i in live)
        b = tuple(i for i in b if live_blue_ids is None or i in live_blue_ids)
        if r and b:
            intents.append((r,b))
    return IdentityUpperAction(tuple((t, tuple(i for i in group if i in live)) for t, group in action.groups if any(i in live for i in group)),
                               tuple(i for i in action.reserve_ids if i in live),tuple(intents))


def eta(entity, target):
    distance = max(0.0, float(np.linalg.norm(np.asarray(entity.position) - target.position)) - 500.0)
    return distance / max(20.0, float(np.linalg.norm(entity.velocity)))


def canonical_blue(state, action):
    action.validate(state.ids("blue"), [x.id for x in state.targets], "Blue")
    if state.lower != "rush":
        return action
    lookup = {x.id: x for x in state.blue}
    groups = []
    assignment = action.assignment()
    for target in state.targets:
        ids = sorted((i for i, t in assignment.items() if t == target.id), key=lambda i: (eta(lookup[i], target), i))
        groups.extend((target.id, tuple(ids[k:k + 4])) for k in range(0, len(ids), 4))
    return IdentityUpperAction(tuple(groups))


def match(state, red, blue):
    """Target, intended physical IDs, ETA, centroid, IDs; no virtual channel.

    Intents are preferences, not a claim to know Blue's true coalition. The
    arbiter resolves all preferences deterministically after simultaneous commit.
    """
    red.validate(state.ids("red"), [x.id for x in state.targets],blue_ids=state.ids("blue"))
    blue = canonical_blue(state, blue)
    lookups = {side: {x.id: x for x in getattr(state, side)} for side in ["red", "blue"]}
    pairs = []
    for target in state.targets:
        def order(action, side):
            def key(group):
                entities = [lookups[side][i] for i in group]
                return (sum(eta(x, target) for x in entities)/len(entities),
                        tuple(np.mean([x.position for x in entities], axis=0)), group)
            return sorted([group for t, group in action.groups if t == target.id], key=key)
        reds, blues = order(red, "red"), order(blue, "blue")
        intents = dict(red.intercepts)
        edges=[]
        for ri,r in enumerate(reds):
            for bi,b in enumerate(blues):
                overlap=len(set(intents.get(r,())).intersection(b))
                re=[lookups["red"][i] for i in r];be=[lookups["blue"][i] for i in b]
                arrival=abs(np.mean([eta(x,target) for x in re])-np.mean([eta(x,target) for x in be]))
                distance=float(np.linalg.norm(np.mean([x.position for x in re],axis=0)-np.mean([x.position for x in be],axis=0)))
                edges.append((-overlap,arrival,distance,r,b,ri,bi))
        used_r,used_b=set(),set()
        for *_,ri,bi in sorted(edges):
            if ri not in used_r and bi not in used_b:
                pairs.append((target.id,reds[ri],blues[bi]));used_r.add(ri);used_b.add(bi)
        pairs.extend((target.id,r,()) for i,r in enumerate(reds) if i not in used_r)
        pairs.extend((target.id,(),b) for i,b in enumerate(blues) if i not in used_b)
    return tuple(pairs)


def action_from_counts(state, side, counts, *, group_size=4, reserve=0, rotation=0):
    """Exact identity cover with distance-aware assignment and nearby grouping."""
    entities = {x.id: x for x in state.alive(side)}
    if sum(counts) + reserve != len(entities) or any(x < 0 for x in counts):
        raise ValueError("Counts do not cover the live roster")
    available = set(entities)
    groups = []
    order = list(range(len(state.targets)))
    rotation %= max(1, len(order))
    order = order[rotation:] + order[:rotation]
    for k in order:
        target = state.targets[k]
        ids = sorted(available, key=lambda i: (eta(entities[i], target), i))[:counts[k]]
        # Chunk locally, preserving the explicit identities.
        while ids:
            anchor = ids.pop(0)
            near = sorted(ids, key=lambda i: (np.linalg.norm(np.asarray(entities[i].position)-entities[anchor].position), i))[:group_size-1]
            members = (anchor, *near)
            ids = [i for i in ids if i not in near]
            groups.append((target.id, tuple(members)))
            available.difference_update(members)
    result = IdentityUpperAction(tuple(groups), tuple(available))
    result.validate(state.ids(side), [x.id for x in state.targets], side)
    return canonical_blue(state, result) if side.lower() == "blue" else result


def quotas(n, weights):
    weights = np.asarray(weights, float)
    weights = weights / weights.sum()
    fractional = n * weights
    counts = np.floor(fractional).astype(int)
    for k in sorted(range(len(weights)), key=lambda k: (-(fractional[k]-counts[k]), k))[:n-int(counts.sum())]:
        counts[k] += 1
    return tuple(map(int, counts))


def red_candidates(state, previous=None, limit=32):
    n, m = len(state.ids("red")), len(state.targets)
    candidates = []
    def add(action):
        if action not in candidates:
            candidates.append(action)
    add(action_from_counts(state, "red", quotas(n, np.ones(m))))
    if previous is not None:
        add(prune(previous, state.ids("red"),state.ids("blue")))
    add(IdentityUpperAction((), state.ids("red")))
    blue = state.alive("blue")
    threats = [sum(1/(1+eta(x,t)) for x in blue)+.01 for t in state.targets]
    for size in [4, 2, 3, 1]:
        for rotation in range(m):
            for reserve in [0, min(4, n)]:
                add(action_from_counts(state, "red", quotas(n-reserve, threats), group_size=size, reserve=reserve, rotation=rotation))
    for k in range(m):
        for strength in [2.0, 6.0]:
            weights = np.ones(m)
            weights[k] = strength
            add(action_from_counts(state, "red", quotas(n, weights), rotation=k))
    # Round-robin selection preserves different allocation patterns, not just group sizes.
    if len(candidates) > limit:
        indices = [0, 1] + np.linspace(2, len(candidates)-1, limit-2, dtype=int).tolist()
        candidates = [candidates[k] for k in indices]
    return tuple(candidates)
