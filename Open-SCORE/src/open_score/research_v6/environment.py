"""V6 public event states, fixed shaping and genuine single-target projections.

The shared HAD physics and ``rule_group_v1`` executor are unchanged. Only this
wrapper retains zero-Red decision events, so replay can represent every actual
event without treating a casualty as a team terminal.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np

from open_score.grouping.domain import DecisionState, Entity, Group, Grouping
from had_env.grouping.environment import KnownOpponentEnv
from had_env.grouping.opponents import sample
from had_env.grouping.rules import RuleExecutor, responsibilities, threat_targets


def rule_grouping(state):
    from had_env.grouping.actions import rule_grouping as native_rule
    return Grouping.from_dict(native_rule(state).to_dict())


def grand_grouping(state):
    from had_env.grouping.actions import grand_grouping as native_grand
    return Grouping.from_dict(native_grand(state).to_dict())


def decode_counts(state, counts):
    from had_env.grouping.actions import decode_counts as native_decode
    return Grouping.from_dict(native_decode(state, counts).to_dict())


def partition_key(grouping):
    from had_env.grouping.actions import partition_key as native_key
    return native_key(grouping)


@dataclass(frozen=True)
class V6State(DecisionState):
    initial_red_count: int = 0
    initial_blue_count: int = 0
    initial_blue_health: float = 0.0
    initial_target_count: int = 0
    command_interval: int = 5

    def to_dict(self):
        return {**super().to_dict(),
                'initial_red_count': self.initial_red_count,
                'initial_blue_count': self.initial_blue_count,
                'initial_blue_health': self.initial_blue_health,
                'initial_target_count': self.initial_target_count,
                'command_interval': self.command_interval}

    @classmethod
    def from_dict(cls, data: Mapping):
        base = DecisionState.from_dict(data)
        return cls(**base.__dict__,
                   initial_red_count=int(data.get('initial_red_count', len(base.red))),
                   initial_blue_count=int(data.get('initial_blue_count', len(base.blue))),
                   initial_blue_health=float(data.get('initial_blue_health', len(base.blue))),
                   initial_target_count=int(data.get('initial_target_count', len(base.targets))),
                   command_interval=int(data.get('command_interval', 5)))


def native_terminal(state):
    """Public equivalent of Stage3's terminal priority; horizon is a true end."""
    if any(target.health < 1e-3 for target in state.targets):
        return True, False, 'targets_breached'
    if all(blue.health < 1e-3 for blue in state.blue):
        return True, True, 'blue_destroyed'
    if state.step >= state.max_steps:
        return True, True, 'horizon'
    return False, False, ''


def blue_health_loss(state):
    initial = float(state.initial_blue_health)
    remaining = sum(float(np.clip(e.health, 0., 1.)) for e in state.blue)
    return float(np.clip(1.-remaining/initial, 0., 1.)) if initial > 0 else 0.


def potential(state):
    return 0. if native_terminal(state)[0] else blue_health_loss(state)


class V6Env(KnownOpponentEnv):
    def __init__(self, red, blue, targets, seed=0, opponent_seed=None,
                 max_steps=50, command_interval=5, target_positions=None,
                 reward_coefficient=.5):
        if min(int(red), int(blue)) < 0 or int(targets) < 1:
            raise ValueError('Nonnegative rosters and at least one target are required')
        if target_positions is None:
            ys = [0.] if int(targets) == 1 else np.linspace(-650., 650., int(targets))
            target_positions = [[-2100., float(y), 100.] for y in ys]
        self.initial_red_count = int(red)
        self.initial_blue_count = int(blue)
        self.initial_target_count = int(targets)
        self.initial_blue_health = float(blue)
        self.opponent_seed = None if opponent_seed is None else int(opponent_seed)
        self.reward_coefficient = float(reward_coefficient)
        self.projection_id_map = None
        super().__init__(red=max(1, int(red)), blue=max(1, int(blue)), targets=targets, opponent='reactive',
                         seed=seed, max_steps=max_steps,
                         command_interval=command_interval, executor=RuleExecutor(),
                         group_max_size=None, target_positions=target_positions)
        if int(red) == 0 or int(blue) == 0:
            # The exported adapter restricts its constructor to positive counts,
            # while its native core handles empty sides. Replace the unexecuted
            # constructor world before reset; no placeholder enters any physics.
            from had_env.core.make_env import HADEnv
            self.adapter.env.close()
            self.adapter.env = HADEnv(int(red), int(blue), int(targets),
                                      task_type='Training', target_region=self.adapter.target_region)
            self.previous = Grouping((), self.adapter.red_ids)
            self.set_rng(self.seed)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.initial_blue_health = sum(float(np.clip(r['health'], 0., 1.))
                                       for r in self.adapter.agent_states('Blue').values())
        if self.opponent_seed is not None:
            self.set_rng(self.opponent_seed)
        result = self.state()
        self.done = native_terminal(result)[0]
        return result

    def state(self):
        # had_env's domain types and old checkpoint types deliberately remain
        # separate; normalize the public wire representation for learning.
        base = DecisionState.from_dict(super().state().to_dict())
        return V6State(**base.__dict__, initial_red_count=self.initial_red_count,
                       initial_blue_count=self.initial_blue_count,
                       initial_blue_health=self.initial_blue_health,
                       initial_target_count=self.initial_target_count,
                       command_interval=self.command_interval)

    @property
    def native_success(self):
        return native_terminal(self.state())[1]

    def step(self, grouping):
        if self.done:
            raise RuntimeError('Cannot step a completed episode; call reset')
        before = self.state()
        grouping.validate(before.ids('red'), before.ids('targets'), max_members=None)
        blue = sample(before, 'reactive', self.opponent_rng)
        blue.validate(before.ids('blue'), before.ids('targets'))
        self.previous, self.blue_grouping = grouping, blue
        red_assignment, blue_assignment = grouping.assignment(), blue.assignment()
        self.adapter.set_joint_assignments(
            {i: red_assignment.get(i) for i in self.adapter.red_ids},
            {i: blue_assignment.get(i) for i in self.adapter.blue_ids})
        start = self.adapter.step_count
        events = []
        event_reason = 'periodic'
        while True:
            actions = self.executor.act(self.adapter, self.previous)
            with self._legacy_rng():
                _, _, self.done, physical_info = self.adapter.step(actions, blue_style='rush')
            events.extend(physical_info['events'])
            if self.done:
                event_reason = ('horizon' if physical_info['truncated'] else 'terminal')
                break
            if any(e['kind'] == 'agents_destroyed' for e in physical_info['events']):
                event_reason = 'casualty'
                break
            if self.adapter.step_count % self.command_interval == 0:
                break
        after = self.state()
        self.previous = after.previous
        self.blue_grouping = self.blue_grouping.prune(after.ids('blue'))
        success = bool(self.done and physical_info['outcome_red'] > 0)
        native_reward = float(success)
        phi_before, phi_after = potential(before), potential(after)
        correction = self.reward_coefficient*(phi_after-phi_before)
        info = dict(delta=self.adapter.step_count-start, success=success,
                    physical_steps=self.adapter.step_count, event_reason=event_reason,
                    termination_reason=native_terminal(after)[2], events=events,
                    terminated=bool(self.done), truncated=False,
                    initial_red=self.initial_red_count, initial_blue=self.initial_blue_count,
                    remaining_red=len(after.ids('red')), remaining_blue=len(after.ids('blue')),
                    no_red_continuation=not bool(before.ids('red')),
                    automatic_blue_events=0, native_reward=native_reward,
                    shaped_reward=native_reward+correction,
                    phi_before=phi_before, phi_after=phi_after,
                    terminal_compensation=correction if self.done else 0.,
                    nonterminal_positive_shaping=max(0., correction) if not self.done else 0.,
                    blue_health_loss=blue_health_loss(after))
        return after, native_reward, bool(self.done), info

    def snapshot(self):
        return {**super().snapshot(), 'v6_metadata': dict(
            initial_red_count=self.initial_red_count,
            initial_blue_count=self.initial_blue_count,
            initial_blue_health=self.initial_blue_health,
            initial_target_count=self.initial_target_count,
            opponent_seed=self.opponent_seed, reward_coefficient=self.reward_coefficient,
            projection_id_map=copy.deepcopy(self.projection_id_map)),
                'native_core': dict(core_version=self.adapter.env.core_version,
                    physics_protocol=self.adapter.env.physics_protocol,
                    physics_step_count=self.adapter.env.physics_step_count,
                    last_physics_events=copy.deepcopy(self.adapter.env.last_physics_events),
                    record_events=bool(self.adapter.env.record_events))}

    def restore(self, snapshot):
        metadata = snapshot.get('v6_metadata')
        if metadata is None:
            raise ValueError('V6 restoration requires the original public initial metadata')
        if float(metadata['reward_coefficient']) != self.reward_coefficient:
            raise ValueError('Snapshot reward coefficient differs')
        core = snapshot.get('native_core', {})
        if (core.get('core_version') != self.adapter.env.core_version
                or core.get('physics_protocol') != self.adapter.env.physics_protocol):
            raise ValueError('Snapshot was generated with a different HAD native core')
        for key, value in metadata.items():
            setattr(self, key, copy.deepcopy(value))
        # A projected world's public agent IDs can be non-contiguous.
        physical = snapshot['physical']
        for entity, source in zip(self.adapter.env.world, physical.entity_states):
            entity.Id = int(source['Id'])
        result = super().restore(snapshot)
        for key in ('physics_step_count', 'last_physics_events', 'record_events'):
            setattr(self.adapter.env, key, copy.deepcopy(core[key]))
        return result

    def close(self):
        self.adapter.env.close()


def public_projection(state, target_id, red_ids, grouping):
    """Filter public entities, fix Blue threat ownership, normalize target to 0.

    Agent identities are unchanged. The grouping argument supplies the candidate
    within-target partition; it cannot alter which Blue entities are projected.
    The projection is an independent local approximation, never a global win
    label for an unbreached target after another target has already been lost.
    """
    selected = {int(i) for i in red_ids}
    live_red = {entity.id: entity for entity in state.alive('red')}
    if not selected.issubset(live_red):
        raise ValueError('Local Red roster must contain live public identities')
    target = next((e for e in state.targets if e.id == int(target_id)), None)
    if target is None:
        raise ValueError('Unknown projection target')
    ownership = threat_targets(state)
    reds = tuple(live_red[i] for i in sorted(selected))
    blues = tuple(e for e in state.alive('blue') if ownership.get(e.id) == int(target_id))
    groups = tuple(Group(0, tuple(i for i in g.members if i in selected))
                   for g in grouping.groups if g.target == int(target_id)
                   and any(i in selected for i in g.members))
    previous = Grouping(groups)
    previous.validate(selected, (0,), max_members=None)
    return V6State(state.step, state.max_steps, 'reactive', reds, blues,
                   (replace(target, id=0),), previous,
                   initial_red_count=len(reds), initial_blue_count=len(blues),
                   initial_blue_health=sum(float(np.clip(e.health, 0., 1.)) for e in blues),
                   initial_target_count=1,
                   command_interval=getattr(state, 'command_interval', 5))


def local_env_from_state(local_state, seed):
    """Build and populate a new native world using public state only.

    HAD recalculates firing and acceleration before each physical transition;
    their old values do not enter this restoration. Physical constants are the
    common unchanged engine configuration. No removed target or agent exists in
    the new world's physics, and Blue chooses a fresh local reactive command.
    """
    if len(local_state.targets) != 1 or local_state.targets[0].id != 0:
        raise ValueError('Expected a canonical single-target public projection')
    env = V6Env(len(local_state.red), len(local_state.blue), 1, seed=int(seed),
                opponent_seed=int(seed), max_steps=local_state.max_steps,
                command_interval=local_state.command_interval,
                target_positions=[local_state.targets[0].position])
    env.reset()
    source_rows = (*local_state.red, *local_state.blue, *local_state.targets)
    physical_rows = (*env.adapter.env.red_agents, *env.adapter.env.blue_agents,
                     *env.adapter.env.targets)
    ids = [e.id for e in (*local_state.red, *local_state.blue)]
    if len(set(ids)) != len(ids):
        raise ValueError('Projected agent identities must remain globally unique')
    target_entity_id = max(ids, default=-1)+1
    for physical, source in zip(physical_rows, source_rows):
        physical.Id = target_entity_id if physical.Type == 'Entity' else source.id
        physical.position = list(source.position)
        physical.initial_position = list(source.position)
        physical.velocity = list(source.velocity)
        physical.Health = float(source.health)
        if physical.Type != 'Entity':
            physical.acceleration = [0., 0., 0.]
            physical.IsFire = False
    env.adapter.step_count = int(local_state.step)
    env.adapter.env.physics_step_count = int(local_state.step)
    env.adapter.env.last_physics_events = []
    env.adapter.env.update_alive_agents()
    assignment = local_state.previous.assignment()
    env.adapter._red_assignment = {i: assignment.get(i) for i in env.adapter.red_ids}
    env.adapter._blue_assignment = {i: None for i in env.adapter.blue_ids}
    env.previous = local_state.previous
    env.blue_grouping = Grouping((), env.adapter.blue_ids)
    env.executor.reset(env.adapter.red_ids)
    env.initial_red_count = local_state.initial_red_count
    env.initial_blue_count = local_state.initial_blue_count
    env.initial_blue_health = local_state.initial_blue_health
    env.projection_id_map = dict(red={i: i for i in env.adapter.red_ids},
                                 blue={i: i for i in env.adapter.blue_ids},
                                 target={0: target_entity_id})
    env.done = native_terminal(env.state())[0]
    return env
