"""A public-state generative model, never a clone of hidden official commands."""
from __future__ import annotations
from contextlib import contextmanager
import numpy as np

from open_score.envs import HADStage3Adapter
from open_score.envs.had_stage1 import ACCELERATION_PRIMITIVES
from open_score.stage3.runtime import FrozenStage1GroupExecutor, build_observable_threat_patrol
from .domain import PublicState, canonical_blue, match, observe, prune


@contextmanager
def isolated_rng(seed):
    state = np.random.get_state()
    np.random.seed(int(seed) % 2**32)
    try:
        yield
    finally:
        np.random.set_state(state)


def from_public(state: PublicState, seed=0):
    """Construct from a whitelist, with fresh independent RNG and no commands.

    The HAD attack-only world has no hidden cooldown or ammunition state. Fire
    decisions are recomputed each physical step. Recurrent controllers reset at
    every upper event in both the real and simulated execution protocols.
    """
    with isolated_rng(seed):
        adapter = HADStage3Adapter(len(state.red), len(state.blue), len(state.targets),
                                  max_steps=state.max_steps, target_positions=[x.position for x in state.targets],
                                  blue_rule_style=state.lower)
        adapter.reset(seed=seed)
    for entities, records in [(adapter.env.red_agents, state.red), (adapter.env.blue_agents, state.blue), (adapter.env.targets, state.targets)]:
        for entity, row in zip(entities, records):
            # HAD IDs are stable by side/order; target IDs are public indices.
            if entity.Color != "Entity" and entity.Id != row.id:
                raise ValueError("Public identity ordering differs from HAD")
            entity.position = list(row.position)
            entity.velocity = list(row.velocity)
            entity.Health = row.health
            if hasattr(entity, "IsFire"):
                entity.IsFire = False
    adapter.step_count = state.step
    adapter.env.update_alive_agents()
    adapter.set_joint_assignments({i: None for i in adapter.red_ids}, {i: None for i in adapter.blue_ids})
    adapter.last_events = ()
    return adapter


def blue_destinations(state, action):
    action = canonical_blue(state, action)
    targets = {x.id: x for x in state.targets}
    destinations = {}
    for t, group in action.groups:
        for index, i in enumerate(sorted(group)):
            dest = np.asarray(targets[t].position, float).copy()
            if state.lower == "split_rush":
                half = min(180.0 * (len(group)-1)/2, 450.0)
                dest[1] += -half + 2*half*index/(len(group)-1) if len(group)>1 else 0
            destinations[i] = dest
    return destinations


def predicted_velocities(state, action):
    """Known attack-only kinematics; wMax=pi, so turning is unrestricted."""
    destinations = blue_destinations(state, action)
    result = {}
    for entity in state.alive("blue"):
        direction = destinations[entity.id] - entity.position
        norm = np.linalg.norm(direction)
        action_id = 0 if norm < 1e-8 else int(np.argmax(ACCELERATION_PRIMITIVES @ (direction/norm)))
        velocity = np.asarray(entity.velocity) + ACCELERATION_PRIMITIVES[action_id]*36.0
        speed = np.linalg.norm(velocity)
        if speed < 1e-3:
            velocity = -np.asarray(entity.velocity)/10.0
            speed = np.linalg.norm(velocity)
        result[entity.id] = velocity * np.clip(speed, 20.0, 300.0)/max(speed, 1e-12)
    return result


def motion_log_likelihood(before, after, action, sigma=8.0):
    """Incremental likelihood of new public velocities; no target labels.

    Gaussian discrepancy is deliberately robust to model/sensor error. Dead
    agents have no movement evidence. Covariance is N*sigma^2 per coordinate,
    where N is the observed survivor count, to temper population-dependent
    certainty. The common, type-independent normalizer is omitted; this is
    valid for filtering. Reported surprise is NLL up to that common constant.
    """
    action = prune(action, before.ids("blue"))
    predicted = predicted_velocities(before, action)
    errors = [np.sum((np.asarray(x.velocity)-predicted[x.id])**2)/(2*sigma*sigma)
              for x in after.alive("blue") if x.id in predicted]
    return -float(np.mean(errors)) if errors else 0.0


def install(adapter, state, red, blue):
    blue = canonical_blue(state, blue)
    pairs = match(state, red, blue)
    adapter.set_joint_assignments({i: red.assignment().get(i) for i in adapter.red_ids},
                                  {i: blue.assignment().get(i) for i in adapter.blue_ids})
    rosters = {(t, k): (r, b) for k, (t, r, b) in enumerate(pairs)}
    # Blue split lanes depend exclusively on Blue's own action, never on Red.
    subgroups = {i: k for k, (_, ids) in enumerate(blue.groups) for i in ids}
    return rosters, subgroups


class Execution:
    def __init__(self, adapter, state, red, blue, stage1, device, reserve_mode="patrol"):
        self.red, self.blue = red, blue
        self.live_ids = (state.ids("red"),state.ids("blue"))
        self.rosters, self.subgroups = install(adapter, state, red, blue)
        self.executor = FrozenStage1GroupExecutor(stage1, device, micro_grouping="planned_groups")
        self.patrol = None
        if reserve_mode == "patrol" and red.reserve_ids:
            self.patrol = build_observable_threat_patrol(adapter, red.reserve_ids, red.assignment()).waypoint_mapping()

    def step(self, adapter):
        state = observe(adapter)
        live_ids = (state.ids("red"),state.ids("blue"))
        if live_ids != self.live_ids:
            self.red,self.blue = prune(self.red,live_ids[0],live_ids[1]),prune(self.blue,live_ids[1])
            self.rosters,self.subgroups = install(adapter,state,self.red,self.blue)
            self.executor.reset()
            self.live_ids = live_ids
        actions = self.executor.act(adapter, local_steps={t: adapter.step_count for t in adapter.target_ids},
                                    roster_override=self.rosters, reserve_waypoints=self.patrol)
        return adapter.step(actions, blue_subgroup_by_agent=self.subgroups)


def branch(state, red, blue, stage1, device, seed, *, steps=5, reserve_mode="patrol", stop_on_event=True):
    with isolated_rng(seed):
        adapter = from_public(state, seed)
        execution = Execution(adapter, state, red, blue, stage1, device, reserve_mode)
        observations = [state]
        outcome = 0
        if stop_on_event:
            steps = min(steps,5-state.step%5)
        for _ in range(min(steps, state.max_steps-state.step)):
            if adapter._terminal_sign() != 0:
                outcome = adapter._terminal_sign()
                break
            _, _, done, info = execution.step(adapter)
            observations.append(observe(adapter, state.red_history_counts))
            if done:
                outcome = int(info["outcome_red"])
                break
            if stop_on_event and any(x["kind"] == "agents_destroyed" for x in info["events"]):
                break
        return observations[-1], outcome, observations
