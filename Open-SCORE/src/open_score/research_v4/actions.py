"""Unbounded identity partitions, transparent controls and budgeted search.

The budget limits computation, not the legal action domain. Single-member
moves to existing/new groups or reserve connect every labelled-target partition.
Swaps and whole-group target moves are accelerators, not required constraints.
"""
from __future__ import annotations

import heapq
import itertools
import json
import hashlib
import numpy as np
from scipy.optimize import linear_sum_assignment

from open_score.grouping.domain import Group, Grouping
from open_score.grouping.opponents import distribution


def partition_key(grouping):
    return (tuple((g.target, g.members) for g in grouping.groups), grouping.reserve)


def legal(state, grouping):
    return grouping.validate(state.ids('red'), state.ids('targets'), max_members=None)


def current_grouping(state):
    live = set(state.ids('red'))
    old = state.previous.prune(live)
    valid = set(state.ids('targets'))
    groups = tuple(g for g in old.groups if g.target in valid)
    used = {i for g in groups for i in g.members}
    return legal(state, Grouping(groups, tuple(live-used)))


def count_allocations(total, targets=2, reserve=True):
    """Enumerate weak compositions; two targets + reserve has C(N+2,2)."""
    total, targets = int(total), int(targets)
    if total < 0 or targets < 1:
        raise ValueError('nonnegative count and positive targets required')
    def compositions(n, k):
        if k == 1:
            yield (n,)
        else:
            for first in range(n+1):
                for tail in compositions(n-first, k-1):
                    yield (first, *tail)
    for counts in compositions(total, targets+int(reserve)):
        yield counts if reserve else (*counts, 0)


def decode_counts(state, counts):
    """Same deterministic minimum-travel identity decoder for B1/B2-count."""
    targets, reds = state.alive('targets'), state.alive('red')
    counts = tuple(map(int, counts))
    if len(counts) != len(targets)+1 or min(counts) < 0 or sum(counts) != len(reds):
        raise ValueError('counts must include each target and reserves, summing to live Red')
    if not reds:
        return Grouping(())
    destinations = [np.asarray(t.position) for t, n in zip(targets, counts) for _ in range(n)]
    labels = [t.id for t, n in zip(targets, counts) for _ in range(n)]
    standby = np.mean([t.position for t in targets], axis=0)
    destinations.extend([standby]*counts[-1])
    labels.extend([None]*counts[-1])
    costs = np.linalg.norm(np.asarray([e.position for e in reds])[:, None]-np.asarray(destinations)[None], axis=-1)
    rows, cols = linear_sum_assignment(costs)
    assigned = {t.id: [] for t in targets}
    reserve = []
    for row, col in zip(rows, cols):
        identity, target = reds[int(row)].id, labels[int(col)]
        (reserve if target is None else assigned[target]).append(identity)
    return legal(state, Grouping(tuple(Group(t, tuple(ids)) for t, ids in assigned.items() if ids), tuple(reserve)))


def _rule_counts(state):
    n, targets = len(state.ids('red')), state.alive('targets')
    blues, probabilities = distribution(state, state.opponent)
    expected = {t.id: 0.0 for t in targets}
    for blue, probability in zip(blues, probabilities):
        for group in blue.groups:
            if group.target in expected:
                expected[group.target] += probability*len(group.members)
    weights = np.asarray([expected[t.id] for t in targets], float)
    if weights.sum() == 0:
        weights[:] = 1.0
    raw = n*weights/weights.sum()
    counts = np.floor(raw).astype(int)
    for index in sorted(range(len(targets)), key=lambda k: (-(raw[k]-counts[k]), targets[k].id))[:n-int(counts.sum())]:
        counts[index] += 1
    return (*map(int, counts), 0)


def grand_grouping(state):
    """One group per target; known opponent expected-force allocation."""
    return decode_counts(state, _rule_counts(state))


def rule_grouping(state):
    """Allocate to expected threats, group spatially connected teammates.

    The connection radius is the native interaction radius. Components can
    contain any number of members; this rule creates no four-person capacity.
    """
    from HAD_Env.config import AttackDistance
    result = []
    positions = {e.id: np.asarray(e.position) for e in state.alive('red')}
    radius = float(np.max(AttackDistance))
    for group in grand_grouping(state).groups:
        remaining = set(group.members)
        while remaining:
            reached = {min(remaining)}
            remaining -= reached
            frontier = list(reached)
            while frontier:
                identity = frontier.pop()
                connected = {i for i in remaining if np.linalg.norm(positions[i]-positions[identity]) <= radius}
                reached.update(connected)
                remaining -= connected
                frontier.extend(sorted(connected))
            result.append(Group(group.target, tuple(reached)))
    return legal(state, Grouping(tuple(result)))


def _remove(grouping, identity):
    groups = [Group(g.target, tuple(i for i in g.members if i != identity))
              for g in grouping.groups if any(i != identity for i in g.members)]
    return groups, [i for i in grouping.reserve if i != identity]


def neighbors(state, grouping, *, mode='full'):
    """Lazy exhaustive elementary edits, deduplicated by callers."""
    grouping = legal(state, grouping)
    targets = state.ids('targets')
    old_assignments = grouping.assignment()
    for identity in state.ids('red'):
        groups, reserve = _remove(grouping, identity)
        if mode != 'target_only':
            for index, group in enumerate(groups):
                if mode == 'group_only' and group.target != old_assignments[identity]:
                    continue
                changed = list(groups)
                changed[index] = Group(group.target, (*group.members, identity))
                yield Grouping(tuple(changed), tuple(reserve))
            for target in targets:
                if mode == 'group_only' and target != old_assignments[identity]:
                    continue
                yield Grouping(tuple(groups)+(Group(target, (identity,)),), tuple(reserve))
            if mode != 'group_only':
                yield Grouping(tuple(groups), (*reserve, identity))
    if mode != 'group_only':
        for index, group in enumerate(grouping.groups):
            for target in targets:
                if target != group.target:
                    changed = list(grouping.groups)
                    changed[index] = Group(target, group.members)
                    yield Grouping(tuple(changed), grouping.reserve)
    if mode == 'target_only':
        return
    for first, second in itertools.combinations(state.ids('red'), 2):
        if mode == 'group_only' and old_assignments[first] != old_assignments[second]:
            continue
        def swap(i):
            return second if i == first else first if i == second else i
        yield Grouping(tuple(Group(g.target, tuple(swap(i) for i in g.members)) for g in grouping.groups),
                       tuple(swap(i) for i in grouping.reserve))


def _rng(state):
    encoded = json.dumps(state.to_dict(), sort_keys=True, separators=(',', ':')).encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(encoded).digest()[:8], 'little'))


def candidate_pool(state, budget=32, rng=None):
    """General connected random-walk proposals, without per-edit quotas.

    Always include keep and the common rule when budget permits. A random walk
    can compose arbitrarily many changes; the budget controls unique proposals.
    """
    if int(budget) < 1:
        raise ValueError('candidate budget must be positive')
    budget = int(budget)
    rng = _rng(state) if rng is None else rng
    pool, seen = [], set()
    def add(grouping):
        key = partition_key(grouping)
        if key not in seen and len(pool) < budget:
            pool.append(legal(state, grouping))
            seen.add(key)
    for initial in (current_grouping(state), rule_grouping(state), grand_grouping(state)):
        add(initial)
    if not state.ids('red'):
        return pool
    current = pool[-1]
    for _ in range(max(32, budget*12)):
        if len(pool) >= budget:
            break
        # Sample a connected elementary relocation directly, without building
        # O(N^2) discarded Grouping objects for each random-walk step.
        identity = int(rng.choice(state.ids('red')))
        groups, reserve = _remove(current, identity)
        destinations = len(groups)+len(state.ids('targets'))+1
        destination = int(rng.integers(destinations))
        if destination < len(groups):
            group = groups[destination]
            groups[destination] = Group(group.target, (*group.members, identity))
        elif destination < destinations-1:
            groups.append(Group(state.ids('targets')[destination-len(groups)], (identity,)))
        else:
            reserve.append(identity)
        current = Grouping(tuple(groups), tuple(reserve))
        add(current)
    return pool


def search(state, scorer, budget=64, start=None, *, mode='full', return_trace=False):
    """Best-first partition search with identical mechanics for B2 and B3.

    Scorer is callable(state, list[Grouping])->sequence[float]. Non-improving
    intermediate partitions remain expandable, so this is not greedy descent.
    The score evaluation budget includes the incumbent and rule proposal.
    """
    if int(budget) < 1:
        raise ValueError('search budget must be positive')
    budget = int(budget)
    initial = current_grouping(state) if start is None else legal(state, start)
    roots = [initial]
    if mode == 'full':
        roots.extend((rule_grouping(state), grand_grouping(state)))
    seen, frontier, evaluated = set(), [], []
    rng = _rng(state)
    def evaluate(proposals):
        fresh = []
        for action in proposals:
            key = partition_key(action)
            if key not in seen and len(evaluated)+len(fresh) < budget:
                seen.add(key)
                fresh.append(action)
        if not fresh:
            return
        values = np.asarray(scorer(state, fresh), float)
        if values.shape != (len(fresh),) or not np.isfinite(values).all():
            raise ValueError('scorer must return one finite value per partition')
        for action, value in zip(fresh, values):
            serial = len(evaluated)
            evaluated.append((float(value), action))
            heapq.heappush(frontier, (-float(value), serial, action))
    evaluate(roots)
    while frontier and len(evaluated) < budget:
        _, _, action = heapq.heappop(frontier)
        proposals = list(neighbors(state, action, mode=mode))
        # No bias toward small numeric identities if budget truncates a level.
        rng.shuffle(proposals)
        # Expand multiple levels even when one node has thousands of legal
        # neighbors. This branching width is a search-computation parameter,
        # shared across scorers, and does not forbid any legal edit.
        unseen = [p for p in proposals if partition_key(p) not in seen]
        evaluate(unseen[:max(1, int(np.sqrt(budget)))])
    best_index = max(range(len(evaluated)), key=lambda i: (evaluated[i][0], -i))
    value, action = evaluated[best_index]
    trace = {'evaluations': len(evaluated), 'best_value': value,
             'initial_value': evaluated[0][0], 'budget': budget, 'mode': mode}
    return (action, trace) if return_trace else action
