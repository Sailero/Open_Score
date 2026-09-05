"""Deterministic, observable-state proposals for bounded dynamic grouping.

This is a restricted action domain, not exhaustive coalition enumeration.
The unchanged previous action is deliberately retained even when degenerate.
All newly constructed deployments use spatial groups of two to four members
when possible; a single surviving member is necessarily a singleton.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.optimize import linear_sum_assignment

from open_score.grouping.domain import DecisionState, Group, Grouping


def _key(entity):
    # IDs only break exact physical ties, never encode learned capabilities.
    return (*entity.position, *entity.velocity, entity.health, entity.id)


def _previous(state):
    live, targets = set(state.ids('red')), set(state.ids('targets'))
    groups, reserve = [], set(state.previous.reserve) & live
    for group in state.previous.groups:
        members = tuple(i for i in group.members if i in live)
        if group.target in targets and members:
            groups.append(Group(group.target, members))
        else:
            reserve.update(members)
    used = {i for group in groups for i in group.members} | reserve
    reserve.update(live - used)
    return Grouping(tuple(groups), tuple(reserve)).validate(live, targets)


def _sizes(n, preferred):
    if n == 0:
        return []
    if n == 1:
        return [1]
    count = max(1, int(round(n / preferred)))
    count = max(math.ceil(n / 4), min(count, n // 2))
    return [n // count + int(i < n % count) for i in range(count)]


def _threat_weights(state, targets):
    if not targets:
        return np.zeros(0)
    weights = np.ones(len(targets), dtype=np.float64)
    for enemy in state.alive('blue'):
        distances = np.asarray([np.linalg.norm(np.subtract(enemy.position, target.position))
                                for target in targets])
        proximity = np.exp(-distances / 1200.0)
        weights += max(0.0, enemy.health) * proximity / max(float(proximity.sum()), 1e-9)
    # Vulnerable targets need more protection; all weights stay positive.
    weights *= np.asarray([1.0 + .5 * (1.0 - min(target.health / 1.2, 1.0)) for target in targets])
    return weights / weights.sum()


def _prepare(state):
    red = sorted(state.alive('red'), key=_key)
    targets = sorted(state.alive('targets'), key=_key)
    if not red or not targets:
        return red, targets, None, None, None, None
    red_pos = np.asarray([entity.position for entity in red])
    target_pos = np.asarray([entity.position for entity in targets])
    distances = np.linalg.norm(red_pos[:, None] - target_pos[None], axis=2)
    pair_distances = np.linalg.norm(red_pos[:, None] - red_pos[None], axis=2)
    return red, targets, distances, pair_distances, target_pos, _threat_weights(state, targets)


def _construct(state, preferred=4, allocation='threat', orientation=0, reserve_fraction=0.0, prepared=None):
    red, targets, distances, pair_distances, target_pos, threat_weights = prepared or _prepare(state)
    if not red or not targets:
        return Grouping((), tuple(entity.id for entity in red))
    reserve_n = min(max(0, len(red) - 2), int(len(red) * reserve_fraction))
    # Keep only a small distant reserve. No all-reserve proposal is created.
    ranked = sorted(range(len(red)), key=lambda i: (float(distances[i].min()), i))
    reserved = set(ranked[-reserve_n:]) if reserve_n else set()
    reserve = tuple(red[i].id for i in reserved)
    remaining = [i for i in range(len(red)) if i not in reserved]
    weights = threat_weights if allocation == 'threat' else np.ones(len(targets)) / len(targets)
    assigned = np.zeros(len(targets), dtype=np.float64)
    groups = []
    axis = orientation % 3
    direction = 1 if orientation < 3 else -1
    for size in _sizes(len(remaining), preferred):
        anchor = min(remaining, key=lambda i: (direction * red[i].position[axis], i))
        members = sorted(remaining, key=lambda i: (pair_distances[anchor, i], i))[:size]
        center = np.asarray([red[i].position for i in members]).mean(axis=0)
        group_distances = np.linalg.norm(center - target_pos, axis=1) / 2500.0
        if allocation == 'nearest':
            costs = group_distances
        else:
            # Marginal quadratic deviation from desired total allocation.
            desired = weights * (len(red) - reserve_n)
            costs = group_distances + .7 * (((assigned + size - desired) ** 2 - (assigned - desired) ** 2)
                                      / max(1, len(red)))
        chosen = int(costs.argmin())
        assigned[chosen] += size
        groups.append(Group(targets[chosen].id, tuple(red[i].id for i in members)))
        selected = set(members)
        remaining = [i for i in remaining if i not in selected]
    return Grouping(tuple(groups), reserve).validate(state.ids('red'), state.ids('targets'))


def _local_repair(state):
    previous = _previous(state)
    positions = {entity.id: np.asarray(entity.position) for entity in state.alive('red')}
    target_positions = {entity.id: np.asarray(entity.position) for entity in state.alive('targets')}
    if not positions or not target_positions:
        return previous
    groups = [group for group in previous.groups if len(group.members) >= 2]
    free = set(previous.reserve) | {i for group in previous.groups if len(group.members) == 1 for i in group.members}
    for i in sorted(free, key=lambda i: (*positions[i], i)):
        available = [(np.linalg.norm(positions[i] - np.mean([positions[j] for j in group.members], axis=0)), index)
                     for index, group in enumerate(groups) if len(group.members) < 4]
        if available:
            index = min(available)[1]
            groups[index] = Group(groups[index].target, groups[index].members + (i,))
        else:
            # Defer the remaining members to a spatial reconstruction below.
            continue
        free.remove(i)
    if free:
        from dataclasses import replace
        subset = replace(state, red=tuple(entity for entity in state.red if entity.id in free),
                         previous=Grouping((), tuple(free)))
        groups.extend(_construct(subset).groups)
    return Grouping(tuple(groups)).validate(state.ids('red'), state.ids('targets'))


def rule_action(state: DecisionState) -> Grouping:
    """Pure dynamic threat-balanced spatial rule; no learned components."""
    return _construct(state, preferred=4, allocation='threat', orientation=0)


def compact_action(state: DecisionState) -> Grouping:
    """Reproduce the audited compact-balanced rule without changing its tail.

    First solve a balanced target-slot distance assignment, then repeatedly
    cluster up to four spatially close members at each target. Per-target
    remainder singletons are intentional: merging them would change the
    exact rule whose executor sensitivity was empirically measured.
    """
    red, targets = state.alive('red'), state.alive('targets')
    if not red or not targets:
        return Grouping((), tuple(entity.id for entity in red))
    slots = [targets[i % len(targets)] for i in range(len(red))]
    cost = np.asarray([[np.linalg.norm(np.asarray(entity.position) - target.position)
                        for target in slots] for entity in red])
    rows, columns = linear_sum_assignment(cost)
    mapping = {red[i].id: slots[j].id for i, j in zip(rows, columns)}
    records = {entity.id: entity for entity in red}
    groups = []
    for target in targets:
        remaining = [i for i in records if mapping[i] == target.id]
        while remaining:
            first = min(remaining, key=lambda i: np.linalg.norm(np.asarray(records[i].position) - target.position))
            cluster = sorted(remaining, key=lambda i: np.linalg.norm(np.asarray(records[i].position) - records[first].position))[:4]
            groups.append(Group(target.id, tuple(cluster)))
            selected = set(cluster)
            remaining = [i for i in remaining if i not in selected]
    return Grouping(tuple(groups)).validate(state.ids('red'), state.ids('targets'))


def candidate_actions(state: DecisionState, limit: int = 24) -> list[Grouping]:
    """Return a stable, deduplicated candidate list from public state only."""
    if int(limit) != limit or limit < 1:
        raise ValueError('candidate limit must be a positive integer')
    pool, seen = [], set()
    def add(action):
        action.validate(state.ids('red'), state.ids('targets'))
        if action not in seen and len(pool) < limit:
            seen.add(action)
            pool.append(action)
    # step zero's reserve is an unassigned initialization placeholder, not a
    # previously executed grouping. Keeping it would permit an idle fixed point.
    prepared = _prepare(state)
    if state.step > 0:
        add(_previous(state))
    add(_construct(state, prepared=prepared))
    add(compact_action(state))
    add(_local_repair(state))
    for preferred in (3, 4):
        add(_construct(state, preferred, 'threat', 1, .125, prepared))
    for orientation in (0, 1, 3):
        for preferred in (2, 3, 4):
            for allocation in ('threat', 'balanced', 'nearest'):
                add(_construct(state, preferred, allocation, orientation, prepared=prepared))
                if len(pool) >= limit:
                    return pool
    return pool
