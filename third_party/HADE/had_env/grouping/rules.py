"""Shared, stateless, group-conditioned rule execution in unchanged HAD physics.

No neural lower model, identity cap, privileged Blue command, or synthetic fire
action is used. Grouping determines interception responsibilities. The physical
simulator retains its own firing, movement, damage and native terminal rules.
"""
from __future__ import annotations

import copy
import numpy as np
from scipy.optimize import linear_sum_assignment

from had_env.grouping.domain import DecisionState, Entity, Grouping
from had_env.grouping.environment import KnownOpponentEnv

RULE_VERSION = "rule_group_v1"


def _records_state(adapter, grouping):
    def entities(rows):
        return tuple(Entity(i, r['position'], r['velocity'], r['health'])
                     for i, r in sorted(rows.items()))
    return DecisionState(adapter.step_count, adapter.max_steps, 'reactive',
                         entities(adapter.agent_states('Red')),
                         entities(adapter.agent_states('Blue')),
                         entities(adapter.target_states()), grouping, spatial_dim=adapter.spatial_dim)


def threat_targets(state):
    """Infer destinations from PUBLIC motion only, not committed Blue actions."""
    targets = state.alive('targets')
    result = {}
    for blue in state.alive('blue'):
        if not targets:
            break
        def cost(target):
            displacement = np.asarray(target.position) - blue.position
            distance = float(np.linalg.norm(displacement))
            closing = float(np.dot(displacement, blue.velocity)) / max(distance, 1e-9)
            # A geometrical arrival-time estimate, not a fitted threat bonus.
            return (distance / max(50.0, closing), distance, target.id)
        result[blue.id] = min(targets, key=cost).id
    return result


def responsibilities(state, grouping):
    """Allocate enemy contacts to existing same-target groups, never regroup Red.

    Each contact has one owner. Group quotas proportional to live member count
    prevent every group chasing the same closest contact. No identity can leave
    its commanded group. An undefended target remains undefended.
    """
    grouping = grouping.prune(state.ids('red'))
    grouping.validate(state.ids('red'), state.ids('targets'), max_members=None)
    reds = {e.id: e for e in state.alive('red')}
    blues = {e.id: e for e in state.alive('blue')}
    inferred = threat_targets(state)
    assigned = {group: [] for group in grouping.groups}
    for target in state.alive('targets'):
        groups = [g for g in grouping.groups if g.target == target.id]
        enemies = [i for i in blues if inferred.get(i) == target.id]
        if not groups or not enemies:
            continue
        weights = np.asarray([len(g.members) for g in groups], float)
        raw = len(enemies) * weights / weights.sum()
        quotas = np.floor(raw).astype(int)
        for index in sorted(range(len(groups)), key=lambda k: (-(raw[k]-quotas[k]), groups[k]))[:len(enemies)-int(quotas.sum())]:
            quotas[index] += 1
        slots = [k for k, quota in enumerate(quotas) for _ in range(int(quota))]
        centers = np.asarray([np.mean([reds[i].position for i in g.members], axis=0) for g in groups])
        positions = np.asarray([blues[i].position for i in enemies])
        costs = np.linalg.norm(positions[:, None, :] - centers[slots][None, :, :], axis=-1)
        rows, columns = linear_sum_assignment(costs)
        for row, col in zip(rows, columns):
            assigned[groups[slots[int(col)]]].append(enemies[int(row)])
    return [(g, tuple(sorted(assigned[g]))) for g in grouping.groups]


class RuleExecutor:
    """Intercept publicly visible assigned threats, otherwise guard the target.

    Fixed parameters are part of the executor protocol, shared by every upper
    method. Reserves keep station at the midpoint and do not repair allocations.
    """
    def __init__(self, lookahead=2.0, guard_distance=700.0):
        if lookahead < 0 or guard_distance < 0:
            raise ValueError('rule distances and lookahead must be nonnegative')
        self.config = dict(version=RULE_VERSION, lookahead=float(lookahead),
                           guard_distance=float(guard_distance))
        self.last_actions = {}

    def reset(self, ids):
        self.last_actions = {int(i): -1 for i in ids}

    def prune(self, ids):
        live = set(ids)
        self.last_actions = {i: a for i, a in self.last_actions.items() if i in live}

    def memory(self):
        return {}

    def snapshot(self):
        return {'config': copy.deepcopy(self.config), 'last_actions': dict(self.last_actions)}

    def restore(self, snapshot):
        if snapshot['config'] != self.config:
            raise ValueError('Rule executor parameters differ from snapshot')
        self.last_actions = {int(i): int(a) for i, a in snapshot['last_actions'].items()}

    @staticmethod
    def _nearest_action(adapter, direction):
        return adapter._nearest_acceleration(np.asarray(direction, dtype=np.float64))

    def act(self, adapter, grouping):
        state = _records_state(adapter, grouping)
        grouping = grouping.prune(state.ids('red'))
        reds, blues = ({e.id: e for e in state.alive(side)} for side in ('red', 'blue'))
        targets = {e.id: e for e in state.alive('targets')}
        self.prune(reds)
        result = {i: 0 for i in adapter.red_ids}
        if not targets:
            return result
        standby = np.mean([e.position for e in targets.values()], axis=0) + np.asarray([self.config['guard_distance'], 0., 0.])
        destinations = {i: standby for i in grouping.reserve}
        for group, enemy_ids in responsibilities(state, grouping):
            guard = np.asarray(targets[group.target].position) + np.asarray([self.config['guard_distance'], 0., 0.])
            if not enemy_ids:
                destinations.update({i: guard for i in group.members})
                continue
            predicted = np.asarray([np.asarray(blues[i].position) + self.config['lookahead']*np.asarray(blues[i].velocity) for i in enemy_ids])
            positions = np.asarray([reds[i].position for i in group.members])
            costs = np.linalg.norm(positions[:, None] - predicted[None], axis=-1)
            rows, columns = linear_sum_assignment(costs)
            matched = {group.members[int(r)]: int(c) for r, c in zip(rows, columns)}
            for row, identity in enumerate(group.members):
                destinations[identity] = predicted[matched.get(identity, int(np.argmin(costs[row])))].copy()
        for identity, destination in destinations.items():
            displacement = np.asarray(destination) - reds[identity].position
            distance = float(np.linalg.norm(displacement))
            # Velocity matching avoids repeatedly overshooting a guard point.
            desired = displacement * min(250.0/max(distance, 1e-9), 1.0)
            result[identity] = self._nearest_action(adapter, desired - reds[identity].velocity)
        self.last_actions.update({i: result[i] for i in reds})
        return result


def make_env(red, blue=None, opponent='reactive', seed=0, max_steps=100,
             command_interval=5, targets=2, target_positions=None, **kwargs):
    executor = kwargs.pop('executor', None)
    if executor is None:
        executor = RuleExecutor(kwargs.pop('lookahead', 2.0), kwargs.pop('guard_distance', 700.0))
    # Device has no effect on the pure deterministic rule lower controller.
    kwargs.pop('device', None)
    return KnownOpponentEnv(red=red, blue=red if blue is None else blue,
                            opponent=opponent, seed=seed, max_steps=max_steps,
                            command_interval=command_interval, executor=executor,
                            group_max_size=None, targets=targets,
                            target_positions=target_positions, **kwargs)
