"""The only project module importing HAD; physics is never copied or patched."""
from __future__ import annotations

from dataclasses import dataclass
import copy
import numpy as np

from had_env import make_env, parallel_env
from had_env.actions import ACCELERATION_PRIMITIVES
from had_env.core import config as had_config
from had_env.core.make_env import HADEnv
from had_env.core.version import PHYSICS_PROTOCOL
from had_env.grouping.adapter import HADStage3Adapter
from had_env.grouping.actions import decode_counts, grand_grouping, rule_grouping
from had_env.grouping.domain import DecisionState as NativeDecisionState, Entity as NativeEntity, Group, Grouping
from had_env.grouping.environment import KnownOpponentEnv
from had_env.grouping.opponents import sample
from had_env.grouping.policies import RulePolicy
from had_env.grouping.rules import RuleExecutor, make_env as make_grouping_env

from .features import (ENTITY_DIM, POSITION_SCALE, VELOCITY_SCALE,
                       planar_action_mapping, resolve_pad, task_masks)
from .scales import as_scale

PLANAR_NATIVE_IDS, NATIVE_TO_PLANAR = planar_action_mapping(ACCELERATION_PRIMITIVES)


@dataclass(frozen=True)
class Entity(NativeEntity):
    initial_health: float = 1.0
    cumulative_damage: float = 0.0


@dataclass(frozen=True)
class DecisionState(NativeDecisionState):
    """Adds physical feature data without changing the existing policy signature."""
    @classmethod
    def from_dict(cls, data):
        return cls(int(data["step"]), int(data["max_steps"]), str(data["opponent"]),
                   *(tuple(Entity(**row) for row in data[side]) for side in ("red", "blue", "targets")),
                   Grouping.from_dict(data["previous"]),
                   {int(i): tuple(map(float, row)) for i, row in data.get("memory", {}).items()},
                   {int(i): int(value) for i, value in data.get("last_actions", {}).items()},
                   int(data.get("spatial_dim", 3)))


def build_decision_state(adapter, grouping=None, opponent="reactive", last_actions=None):
    def row(entity, identity=None):
        return Entity(int(entity.Id if identity is None else identity),
                      tuple(entity.position), tuple(entity.velocity), float(entity.Health),
                      float(entity.initial_health), float(entity.cumulative_damage))
    reds = tuple(row(agent) for agent in adapter.env.red_agents)
    live = tuple(agent.id for agent in reds if agent.alive)
    grouping = Grouping((), live) if grouping is None else grouping.prune(live)
    return DecisionState(int(adapter.step_count), int(adapter.max_steps), opponent,
                         reds, tuple(row(agent) for agent in adapter.env.blue_agents),
                         tuple(row(target, i) for i, target in enumerate(adapter.env.targets)),
                         grouping, {}, dict(last_actions if last_actions is not None else
                                            getattr(adapter, "_policy_last_actions", {})),
                         spatial_dim=adapter.spatial_dim)


def native_actions_to_planar(actions):
    values = np.asarray(actions, dtype=np.int64)
    if np.any(values < 0) or np.any(values >= len(NATIVE_TO_PLANAR)):
        raise ValueError("invalid native acceleration ID")
    result = NATIVE_TO_PLANAR[values]
    if np.any(result < 0):
        raise ValueError("a 3D action was supplied to a 2D policy")
    return result


def trajectory_frame(adapter, native_actions):
    def rows(entities):
        return [{"id": int(e.Id), "position": list(map(float, e.position)),
                 "velocity": list(map(float, e.velocity)), "alive": bool(e.Health > 0),
                 "health": float(e.Health), "step_damage": float(e.step_damage),
                 "cumulative_damage": float(e.cumulative_damage)} for e in entities]
    return {"step": int(adapter.step_count), "red": rows(adapter.env.red_agents),
            "blue": rows(adapter.env.blue_agents), "targets": rows(adapter.env.targets),
            "actions": [int(native_actions.get(int(e.Id), 0)) for e in adapter.env.red_agents],
            "step_damage": float(adapter.env.step_target_damage)}


def red_friendly_collision_pairs(env):
    """Unique Red-Red collision incidents this physics step."""
    red_ids = {int(agent.Id) for agent in env.red_agents}
    pairs = set()
    for event in env.last_physics_events:
        if event.get("kind") != "collision":
            continue
        target = event.get("target_id")
        if target is None:
            continue
        pair = tuple(sorted((int(event["source_id"]), int(target))))
        if all(identity in red_ids for identity in pair):
            pairs.add(pair)
    return pairs


def red_overkill_self_destructs(env):
    """Extra Red intercept suicides beyond the Blues those shots hit.

    Splash does not damage allies. A Red inside fire_range of a live Blue
    attacker auto-fires and self-destructs, so a cluster all die together.
    One suicide per engaged Blue is the intended intercept; extras are waste.
    """
    red_ids = {int(agent.Id) for agent in env.red_agents}
    blue_ids = {int(agent.Id) for agent in env.blue_agents}
    fired, engaged = set(), set()
    for event in env.last_physics_events:
        kind = event.get("kind")
        source = event.get("source_id")
        if source is None:
            continue
        source = int(source)
        if source not in red_ids:
            continue
        if kind == "self_destruct":
            fired.add(source)
        elif kind == "attack_damage":
            target = event.get("target_id")
            if target is not None and int(target) in blue_ids:
                engaged.add(int(target))
    return max(0, len(fired) - len(engaged))


def red_friendly_waste(env):
    """Collision pairs plus surplus intercept suicides this physics step."""
    return len(red_friendly_collision_pairs(env)) + red_overkill_self_destructs(env)


class EpisodeDiagnostics:
    """Accumulate evaluation diagnostics from actual actions and native events."""
    def __init__(self, adapter, seed, blue_upper="reactive", blue_lower="rush", enabled=True,
                 retain_trajectory=False):
        self.adapter, self.seed = adapter, int(seed)
        self.blue_upper, self.blue_lower = blue_upper, blue_lower
        self.enabled, self.retain_trajectory = bool(enabled), bool(retain_trajectory)
        self.action_hist = np.zeros(9, dtype=np.int64)
        self.first_damage = [None] * len(adapter.env.targets)
        self.red_deaths = dict.fromkeys(("shot_down", "friendly_collision", "enemy_collision", "boundary", "self_destruct"), 0)
        self.blue_deaths = dict.fromkeys(("intercepted", "self_destruct", "collision"), 0)
        self.friendly_collisions = 0
        self.friendly_overkill = 0
        # Steps where Red is wiped out but Blue survives: the entity-based
        # critics structurally output zero there (see diff_log). Counted
        # during training sampling too, so the real rate is on record.
        self.wipeout_steps = 0
        self.speed_sum = self.speed_count = self.distance_sum = self.distance_count = 0
        self.target_distance_sum = self.target_distance_count = 0
        self.minimum_distances = []
        self.trajectory = []

    def before_step(self, native_actions):
        env = self.adapter.env
        self.before_alive = {int(a.Id): bool(a.Health > 0) for a in env.agents}
        if not self.enabled:
            return
        red = [a for a in env.red_agents if a.Health > 0]
        if red:
            action_values = [native_actions.get(int(a.Id), 0) for a in red]
            self.action_hist += np.bincount(native_actions_to_planar(action_values), minlength=9)
            positions = np.asarray([a.position[:2] for a in red], dtype=np.float64)
            speeds = np.linalg.norm(np.asarray([a.velocity[:2] for a in red]), axis=-1)
            self.speed_sum += float(speeds.sum())
            self.speed_count += len(red)
            target_positions = np.asarray([t.position[:2] for t in env.targets])
            nearest = np.linalg.norm(positions[:, None] - target_positions[None], axis=-1).min(axis=1)
            self.target_distance_sum += float(nearest.sum())
            self.target_distance_count += len(red)
            if len(red) > 1:
                distances = np.linalg.norm(positions[:, None] - positions[None], axis=-1)
                pairs = distances[np.triu_indices(len(red), 1)]
                self.distance_sum += float(pairs.sum())
                self.distance_count += len(pairs)
                self.minimum_distances.append(float(pairs.min()))

    def after_step(self, native_actions):
        env = self.adapter.env
        for i, target in enumerate(env.targets):
            if target.step_damage > 0 and self.first_damage[i] is None:
                self.first_damage[i] = int(self.adapter.step_count)
        if not any(a.Health > 0 for a in env.red_agents) and any(a.Health > 0 for a in env.blue_agents):
            self.wipeout_steps += 1
        if self.enabled and env.record_events:
            red_ids = {int(a.Id) for a in env.red_agents}
            by_target = {}
            for event in env.last_physics_events:
                target = event.get("target_id")
                if target is not None:
                    by_target.setdefault(int(target), []).append(event)
            self.friendly_collisions += len(red_friendly_collision_pairs(env))
            self.friendly_overkill += red_overkill_self_destructs(env)
            for agent in env.agents:
                identity = int(agent.Id)
                if not self.before_alive[identity] or agent.Health > 0:
                    continue
                events = by_target.get(identity, ())
                collision = next((e for e in events if e["kind"] == "collision"), None)
                is_red = identity in red_ids
                self_destruct = next((e for e in events if e["kind"] == "self_destruct"
                                      and e["health_before"] > 0), None)
                if collision:
                    cause = ("friendly_collision" if int(collision["source_id"]) in red_ids
                             else "enemy_collision") if is_red else "collision"
                elif self_destruct:
                    cause = "self_destruct"
                else:
                    cause = "shot_down" if is_red else "intercepted"
                (self.red_deaths if is_red else self.blue_deaths)[cause] += 1
        if self.retain_trajectory:
            self.trajectory.append(trajectory_frame(self.adapter, native_actions))

    def summary(self):
        env = self.adapter.env
        damage = float(env.target_damage)
        total_actions = int(self.action_hist.sum())
        probabilities = self.action_hist[self.action_hist > 0] / total_actions if total_actions else []
        mean = lambda value, count: float(value / count) if count else None
        known_causes = self.enabled and env.record_events
        return {
            "config": {"N_R": len(env.red_agents), "N_B": len(env.blue_agents), "K": len(env.targets)},
            "episode_seed": self.seed, "blue_upper": self.blue_upper, "blue_lower": self.blue_lower,
            # Undiscounted physical return, unaffected by any folded tail.
            "return": -damage,
            "D": damage, "rho": damage / len(env.blue_agents),
            "ep_len": int(self.adapter.step_count), "terminated_naturally": bool(env.is_episode_done()),
            "damage_by_target": [float(t.cumulative_damage) for t in env.targets],
            "first_damage_step_by_target": self.first_damage.copy(),
            "red_left": int(sum(a.Health > 0 for a in env.red_agents)),
            "blue_left": int(sum(a.Health > 0 for a in env.blue_agents)),
            "red_deaths_by_cause": dict(self.red_deaths) if known_causes else None,
            "blue_deaths_by_cause": dict(self.blue_deaths) if known_causes else None,
            "action_hist": self.action_hist.tolist() if self.enabled else None,
            "action_entropy": float(-np.sum(np.asarray(probabilities) * np.log(probabilities))) if total_actions else None,
            "noop_frac": float(self.action_hist[0] / total_actions) if total_actions else None,
            "mean_speed": mean(self.speed_sum, self.speed_count),
            "mean_pairwise_dist": mean(self.distance_sum, self.distance_count),
            "mean_dist_to_nearest_target": mean(self.target_distance_sum, self.target_distance_count),
            "friendly_collisions": int(self.friendly_collisions) if known_causes else None,
            "friendly_overkill": int(self.friendly_overkill) if known_causes else None,
            "wipeout_steps": int(self.wipeout_steps),
            "min_pairwise_dist_p05": float(np.quantile(self.minimum_distances, .05)) if self.minimum_distances else None,
        }


class HADWrapper:
    """Fast physical stepping with the same Blue event schedule as rule episodes."""
    def __init__(self, scale=(8, 8, 2), max_steps=100, blue_upper="reactive", blue_lower="rush",
                 command_interval=5, diagnostics=False, retain_trajectory=False,
                 gamma=0.99, fold_wipeout_tail=True, pool_slots=None,
                 shaping_coef=0.0, shaping_range=4000.0, pad="eval",
                 subtask_set="targets", allocation_clock="interval",
                 action_length=5, max_alloc_hold=10, reward_mode="damage",
                 friendly_penalty=1.0, **kwargs):
        self.scale = as_scale(scale)
        self.max_steps = int(max_steps)
        self.blue_upper, self.blue_lower = blue_upper, blue_lower
        self.command_interval = int(command_interval)
        self.gamma = float(gamma)
        self.fold_wipeout_tail = bool(fold_wipeout_tail)
        self.shaping_coef = float(shaping_coef)
        self.shaping_range = float(shaping_range)
        if self.shaping_range <= 0:
            raise ValueError("shaping_range must be positive")
        if reward_mode not in ("damage", "friendly"):
            raise ValueError("reward_mode must be 'damage' or 'friendly'")
        if friendly_penalty < 0:
            raise ValueError("friendly_penalty must be non-negative")
        self.reward_mode = reward_mode
        self.friendly_penalty = float(friendly_penalty)
        if subtask_set not in ("targets", "blues"):
            raise ValueError("subtask_set must be 'targets' or 'blues'")
        if allocation_clock not in ("interval", "event"):
            raise ValueError("allocation_clock must be 'interval' or 'event'")
        self.subtask_set = subtask_set
        self.allocation_clock = allocation_clock
        self.action_length = int(action_length)
        self.max_alloc_hold = int(max_alloc_hold)
        if self.action_length < 1 or self.max_alloc_hold < 1:
            raise ValueError("action_length and max_alloc_hold must be positive")
        self.n_red, self.n_blue, self.n_targets = resolve_pad(pad)
        self.n_entities = self.n_red + self.n_blue + self.n_targets
        self.pool_slots = None if pool_slots is None else tuple(int(value) for value in pool_slots)
        if self.command_interval < 1:
            raise ValueError("command_interval must be positive")
        self.diagnostics_enabled, self.retain_trajectory = diagnostics, retain_trajectory
        self.adapter_kwargs = {key: kwargs[key] for key in ("target_positions", "target_health", "target_initialization") if key in kwargs}
        self.adapter = None
        self.hier_decision = 1
        self.steps_since_alloc = 0
        self.last_nearest = None
        self.last_alive = None

    def reset(self, seed=0, scale=None, evaluate=False, diagnostics=None, retain_trajectory=None):
        self.scale = self.scale if scale is None else as_scale(scale)
        roster = (self.scale.N_R, self.scale.N_B, self.scale.K)
        pad = (self.n_red, self.n_blue, self.n_targets)
        if any(value > limit for value, limit in zip(roster, pad)):
            raise ValueError(f"configuration {roster} exceeds this environment's slot pad {pad}")
        if self.pool_slots is not None:
            if any(value > limit for value, limit in zip(roster, self.pool_slots)):
                raise ValueError(
                    f"configuration {roster} exceeds this method's ordered slot budget "
                    f"{self.pool_slots}; it has no parameters for the extra entities")
        self.adapter = HADStage3Adapter(self.scale.N_R, self.scale.N_B, self.scale.K,
                                       max_steps=self.max_steps, task_mode="damage", spatial_dim=2,
                                       blue_rule_style=self.blue_lower, **self.adapter_kwargs)
        adapter = self.adapter
        adapter.reset(seed=int(seed), red_assignment={i: None for i in adapter.red_ids},
                      blue_assignment={i: None for i in adapter.blue_ids})
        enabled = (bool(evaluate) or bool(self.diagnostics_enabled)) if diagnostics is None else bool(diagnostics)
        retain = self.retain_trajectory if retain_trajectory is None else retain_trajectory
        adapter.env.record_events = enabled or self.reward_mode == "friendly"
        adapter._policy_last_actions = {}
        self.opponent_rng = np.random.default_rng(int(seed) ^ 0x375AC18F)
        self.red_grouping = Grouping((), adapter.red_ids)
        self.blue_grouping = Grouping((), adapter.blue_ids)
        self.episode_seed, self.return_sum = int(seed), 0.0
        self._decide_blue()
        self.diagnostics = EpisodeDiagnostics(adapter, seed, self.blue_upper, self.blue_lower,
                                              enabled, retain)
        self._refresh_features()
        self.hier_decision = 1
        self.steps_since_alloc = 0
        self.last_nearest = self._blue_nearest_ids()
        self.last_alive = self._alive_counts()
        return self.entities.copy()

    def _decide_blue(self):
        state = build_decision_state(self.adapter, self.red_grouping, self.blue_upper)
        self.blue_grouping = sample(state, self.blue_upper, self.opponent_rng)
        self.adapter.set_joint_assignments({}, self.blue_grouping.assignment())

    def _refresh_features(self):
        env = self.adapter.env
        self.entities = np.zeros((self.n_entities, ENTITY_DIM), dtype=np.float32)
        self.entity_mask = np.ones(self.n_entities, dtype=np.uint8)
        for group, start, kind in ((env.red_agents, 0, 0), (env.blue_agents, self.n_red, 1),
                                    (env.targets, self.n_red + self.n_blue, 2)):
            indices = np.asarray([start + i for i, e in enumerate(group) if kind == 2 or e.Health > 0])
            rows = [e for e in group if kind == 2 or e.Health > 0]
            if not rows:
                continue
            self.entity_mask[indices] = 0
            self.entities[indices, :2] = np.asarray([e.position[:2] for e in rows]) / POSITION_SCALE
            self.entities[indices, 7 + kind] = 1
            if kind == 2:
                self.entities[indices, 6] = [e.cumulative_damage / self.scale.N_B for e in rows]
            else:
                self.entities[indices, 2:4] = np.asarray([e.velocity[:2] for e in rows]) / VELOCITY_SCALE
                self.entities[indices, 4] = 1
                self.entities[indices, 5] = [e.Health / e.initial_health for e in rows]

    def _potential_by_target(self):
        """The shaping potential split over the target each Blue is closing on.

        Summing this array is the team potential, so ALMA's per-subtask
        rewards add up to the scalar reward the other arms learn from. The
        attachment is the same nearest-target geometry the entity table
        exposes to every method, not the Blue side's private assignment.
        """
        env = self.adapter.env
        values = np.zeros(self.n_targets, dtype=np.float64)
        live = [agent for agent in env.blue_agents if agent.Health > 0]
        if not self.shaping_coef or not live:
            return values
        targets = np.asarray([target.position[:2] for target in env.targets])
        blue = np.asarray([agent.position[:2] for agent in live])
        distances = np.linalg.norm(blue[:, None] - targets[None], axis=-1)
        closed = np.clip(1.0 - distances.min(axis=1) / self.shaping_range, 0.0, 1.0)
        values[:] = -self.shaping_coef * np.bincount(distances.argmin(axis=1), weights=closed,
                                                     minlength=self.n_targets)
        return values

    def n_tasks(self):
        return self.n_blue if self.subtask_set == "blues" else self.n_targets

    def _alive_counts(self):
        env = self.adapter.env
        return (sum(agent.Health > 0 for agent in env.red_agents),
                sum(agent.Health > 0 for agent in env.blue_agents))

    def _blue_nearest_ids(self):
        env = self.adapter.env
        nearest = np.full(self.n_blue, -1, dtype=np.int32)
        live = [(index, agent) for index, agent in enumerate(env.blue_agents) if agent.Health > 0]
        if not live or not env.targets:
            return nearest
        targets = np.asarray([target.position[:2] for target in env.targets])
        blue = np.asarray([agent.position[:2] for _, agent in live])
        chosen = np.linalg.norm(blue[:, None] - targets[None], axis=-1).argmin(axis=1)
        for (index, _), target_id in zip(live, chosen):
            nearest[index] = int(target_id)
        return nearest

    def _potential_by_blue(self):
        values = np.zeros(self.n_blue, dtype=np.float64)
        live = [(index, agent) for index, agent in enumerate(self.adapter.env.blue_agents) if agent.Health > 0]
        if not self.shaping_coef or not live:
            return values
        targets = np.asarray([target.position[:2] for target in self.adapter.env.targets])
        blue = np.asarray([agent.position[:2] for _, agent in live])
        distances = np.linalg.norm(blue[:, None] - targets[None], axis=-1)
        closed = np.clip(1.0 - distances.min(axis=1) / self.shaping_range, 0.0, 1.0)
        for (index, _), value in zip(live, closed):
            values[index] = -self.shaping_coef * value
        return values

    def _potential_parts(self):
        return self._potential_by_blue() if self.subtask_set == "blues" else self._potential_by_target()

    def _friendly_penalty_parts(self, n_incidents):
        """Optional extra cost; not potential-based, so it can change the optimum."""
        parts = np.zeros(self.n_tasks(), dtype=np.float64)
        if self.reward_mode != "friendly" or n_incidents <= 0 or not self.friendly_penalty:
            return parts
        width = self.scale.N_B if self.subtask_set == "blues" else self.scale.K
        parts[:width] = (self.friendly_penalty * n_incidents) / width
        return parts

    def _attribute_damage_to_blues(self, damage_by_target, nearest):
        values = np.zeros(self.n_blue, dtype=np.float64)
        nearest = np.asarray(nearest)
        for target_id, damage in enumerate(damage_by_target):
            if damage == 0:
                continue
            owners = np.flatnonzero(nearest == target_id)
            if len(owners):
                values[owners] += damage / len(owners)
        leftover = float(np.asarray(damage_by_target).sum() - values.sum())
        if abs(leftover) > 1e-9:
            live = np.flatnonzero(nearest >= 0)
            if len(live):
                values[live] += leftover / len(live)
            else:
                values[0] += leftover
        return values

    def _damage_parts(self, nearest=None):
        env = self.adapter.env
        damage = np.zeros(self.n_targets, dtype=np.float64)
        damage[:len(env.targets)] = [float(target.step_damage) for target in env.targets]
        if self.subtask_set != "blues":
            return damage
        return self._attribute_damage_to_blues(damage, nearest)

    def _update_hier_decision(self, terminated, truncated):
        if terminated or truncated:
            self.hier_decision = 0
            return
        if self.allocation_clock != "event":
            self.hier_decision = int(self.adapter.step_count % self.action_length == 0)
            return
        nearest = self._blue_nearest_ids()
        alive = self._alive_counts()
        self.steps_since_alloc += 1
        death = alive != self.last_alive
        geometry = self.last_nearest is not None and not np.array_equal(nearest, self.last_nearest)
        decide = death or (geometry and self.steps_since_alloc >= 2) or self.steps_since_alloc >= self.max_alloc_hold
        self.hier_decision = int(decide)
        if decide:
            self.steps_since_alloc = 0
            self.last_nearest = nearest
            self.last_alive = alive

    def potential(self):
        """Shaping potential: how far the live Blue force has closed in.

        Negative and bounded by -shaping_coef * N_B, zero once no Blue is
        left. Only the timing of the signal changes: the discounted sum of
        gamma*potential(s') - potential(s) telescopes to -potential(s_0),
        a constant of the initial state, so the optimal policy and the
        reported damage D are unchanged (Ng, Harada & Russell 1999).
        """
        return float(self._potential_by_target().sum())

    def _advance(self, native_red):
        """One physical step with the same Blue schedule as a rule episode."""
        adapter, env = self.adapter, self.adapter.env
        blue = adapter.commanded_rule_actions("Blue", style=self.blue_lower)
        all_actions = dict(native_red)
        all_actions.update({int(a.Id): int(value) for a, value in zip(env.blue_agents, blue)})
        before = [a.Health > 0 for a in env.agents]
        nearest_before = self._blue_nearest_ids()
        potential_parts = self._potential_parts()
        potential = float(potential_parts.sum())
        self.diagnostics.before_step(native_red)
        env.step_physics([ACCELERATION_PRIMITIVES[all_actions[int(a.Id)]].copy() for a in env.agents])
        adapter.step_count += 1
        adapter._policy_last_actions = native_red
        reward = -float(env.step_target_damage)
        damage = np.zeros(self.n_targets, dtype=np.float64)
        damage[:len(env.targets)] = [float(target.step_damage) for target in env.targets]
        if not np.isclose(damage.sum(), -reward, atol=1e-6, rtol=0):
            raise AssertionError("per-target damage does not sum to the step damage")
        damage_parts = self._damage_parts(nearest_before)
        # return_sum stays the physical -D, so D and the episode summary never
        # see shaping or the optional friendly-waste cost.
        self.return_sum += reward
        incidents = red_friendly_waste(env) if self.reward_mode == "friendly" else 0
        ff_parts = self._friendly_penalty_parts(incidents)
        reward = reward - float(ff_parts.sum())
        terminated = bool(env.is_episode_done())
        truncated = bool(adapter.step_count >= self.max_steps and not terminated)
        self.diagnostics.after_step(native_red)
        self._refresh_features()
        if not (terminated or truncated):
            casualty = any(old and a.Health <= 0 for old, a in zip(before, env.agents))
            if casualty or adapter.step_count % self.command_interval == 0:
                self._decide_blue()
        # A terminal state has no future, so its potential is zero by
        # convention; truncation keeps the real one because it bootstraps.
        successor_parts = np.zeros(self.n_tasks()) if terminated else self._potential_parts()
        shaped = reward + self.gamma * float(successor_parts.sum()) - potential
        by_task = -damage_parts - ff_parts + self.gamma * successor_parts - potential_parts
        if not np.isclose(by_task.sum(), shaped, atol=1e-6, rtol=0):
            raise AssertionError("per-subtask rewards do not sum to the team reward")
        self._update_hier_decision(terminated, truncated)
        return shaped, by_task, terminated, truncated

    def step(self, actions):
        adapter, env = self.adapter, self.adapter.env
        if env.is_episode_done() or adapter.step_count >= self.max_steps:
            raise RuntimeError("cannot step a completed HAD episode")
        values = np.asarray(actions, dtype=np.int64).reshape(-1)
        if not (self.scale.N_R <= len(values) <= self.n_red) or np.any((values < 0) | (values > 8)):
            raise ValueError("red actions must contain 0..8 IDs for the actual or padded red roster")
        native_red = {int(a.Id): int(PLANAR_NATIVE_IDS[values[i]]) if a.Health > 0 else 0
                      for i, a in enumerate(env.red_agents)}
        reward, task_rewards, terminated, truncated = self._advance(native_red)
        folded_steps = 0
        if self.fold_wipeout_tail and not any(a.Health > 0 for a in env.red_agents):
            # Red has no live unit left, so no later Red action can change the
            # remaining damage. Play the tail out and fold its discounted
            # return into this transition, then treat Red's decision process as
            # ended. Physics, the Blue schedule and D are untouched; the critic
            # gets the real tail instead of bootstrapping a state whose team Q
            # the entity mixers can only represent as zero.
            noop = {int(a.Id): 0 for a in env.red_agents}
            discount = 1.0
            while not (terminated or truncated) and any(a.Health > 0 for a in env.blue_agents):
                discount *= self.gamma
                tail, task_tail, terminated, truncated = self._advance(noop)
                reward += discount * tail
                task_rewards = task_rewards + discount * task_tail
                folded_steps += 1
            if not terminated:
                # Declaring a still-running state terminal drops its future,
                # so take back the successor potential the last step credited.
                refund = self._potential_parts()
                reward -= discount * self.gamma * float(refund.sum())
                task_rewards = task_rewards - discount * self.gamma * refund
            terminated, truncated = True, False
            self.hier_decision = 0
        info = {"terminated": terminated, "truncated": truncated, "episode_limit": truncated,
                "bootstrap_mask": float(not terminated), "target_damage": float(env.target_damage),
                "step_target_damage": float(env.step_target_damage), "step": int(adapter.step_count),
                "n_agents_init": self.scale.N_R, "config": self.scale.as_dict(),
                "episode_seed": self.episode_seed, "folded_steps": folded_steps,
                "task_rewards": task_rewards.tolist(),
                "tasks_terminated": self._tasks_terminated(terminated).tolist()}
        if terminated or truncated:
            if not np.isclose(self.return_sum, -env.target_damage, atol=1e-9, rtol=0):
                raise AssertionError("damage reward does not equal the native episode return")
            info["episode_summary"] = self.diagnostics.summary()
            if self.diagnostics.retain_trajectory:
                info["trajectory"] = self.diagnostics.trajectory
        return reward, terminated or truncated, info

    def _tasks_terminated(self, episode_over):
        if self.subtask_set == "blues":
            flags = np.zeros(self.n_blue, dtype=np.int64)
            for index, agent in enumerate(self.adapter.env.blue_agents):
                if episode_over or agent.Health <= 0:
                    flags[index] = 1
            return flags
        if episode_over:
            return np.asarray([1] * self.scale.K + [0] * (self.n_targets - self.scale.K), dtype=np.int64)
        return np.zeros(self.n_targets, dtype=np.int64)

    def get_policy_state(self):
        return build_decision_state(self.adapter, self.red_grouping, self.blue_upper)

    def episode_summary(self):
        return self.diagnostics.summary()

    def snapshot(self):
        """Explicit recovery/validation path; never called during physical stepping."""
        return {"native": self.adapter.snapshot(), "opponent_rng": copy.deepcopy(self.opponent_rng.bit_generator.state),
                "last_actions": dict(self.adapter._policy_last_actions), "return_sum": self.return_sum,
                "diagnostics": copy.deepcopy({k: v for k, v in vars(self.diagnostics).items() if k != "adapter"}),
                "hier_decision": int(self.hier_decision), "steps_since_alloc": int(self.steps_since_alloc),
                "last_nearest": None if self.last_nearest is None else np.asarray(self.last_nearest).copy(),
                "last_alive": self.last_alive}

    def restore(self, state):
        self.adapter.restore(state["native"])
        self.opponent_rng.bit_generator.state = state["opponent_rng"]
        self.adapter._policy_last_actions = dict(state["last_actions"])
        self.return_sum = state["return_sum"]
        vars(self.diagnostics).update(copy.deepcopy(state["diagnostics"]))
        self._refresh_features()
        self.hier_decision = int(state.get("hier_decision", 0))
        self.steps_since_alloc = int(state.get("steps_since_alloc", 0))
        self.last_nearest = state.get("last_nearest")
        self.last_alive = state.get("last_alive")

    def close(self):
        if self.adapter is not None:
            self.adapter.env.close()


def run_environment_checks():
    """The approved V1 semantic checks, called once by the unified verifier."""
    rows = []

    def record(identity, function):
        try:
            detail = function()
            rows.append({"id": identity, "status": "passed", "detail": detail})
        except Exception as error:
            rows.append({"id": identity, "status": "failed", "detail": f"{type(error).__name__}: {error}"})

    def action_mapping():
        native, inverse = planar_action_mapping(ACCELERATION_PRIMITIVES)
        assert np.array_equal(inverse[native], np.arange(9))
        assert np.all(ACCELERATION_PRIMITIVES[native, 2] == 0)
        return {"model_to_native": native.tolist(), "noop": int(native[0])}

    def physics_equivalence():
        # Per-step comparison against the native reference, so the wipeout
        # tail must not be folded into one of the compared rewards.
        wrapper = HADWrapper((4, 4, 2), max_steps=100, fold_wipeout_tail=False)
        reference = HADStage3Adapter(4, 4, 2, max_steps=100, task_mode="damage", spatial_dim=2)
        wrapper.reset(seed=173)
        reference.reset(seed=173, red_assignment={i: None for i in reference.red_ids},
                        blue_assignment={i: None for i in reference.blue_ids})
        total = 0.0
        try:
            for step in range(100):
                actions = (np.arange(wrapper.n_red) + 3 * step) % 9
                red = {int(a.Id): int(PLANAR_NATIVE_IDS[actions[i]]) if a.Health > 0 else 0
                       for i, a in enumerate(wrapper.adapter.env.red_agents)}
                blue = wrapper.adapter.commanded_rule_actions("Blue")
                reference.set_joint_assignments(wrapper.adapter.red_assignment, wrapper.adapter.blue_assignment)
                _, native_reward, native_done, native_info = reference.step(red, blue)
                reward, done, info = wrapper.step(actions)
                assert abs(reward - native_reward["Red"]) <= 1e-9, (step, reward, native_reward)
                assert abs(wrapper.adapter.env.target_damage - reference.env.target_damage) <= 1e-9
                for a, b in zip(wrapper.adapter.env.world, reference.env.world):
                    np.testing.assert_allclose(a.position, b.position, atol=1e-9, rtol=0)
                    np.testing.assert_allclose(a.velocity, b.velocity, atol=1e-9, rtol=0)
                    assert abs(a.Health - b.Health) <= 1e-9
                assert done == native_done
                assert info["terminated"] == native_info["terminated"]
                assert info["truncated"] == native_info["truncated"]
                total += reward
                if done:
                    break
            task = reference.env.task_info()
            assert abs(total - task["episode_returns"]["Red"]) <= 1e-9
            assert abs(total + task["target_damage"]) <= 1e-9
            return {"seed": 173, "steps": step + 1, "return": total, "D": task["target_damage"],
                    "tolerance": 1e-9, "terminated": info["terminated"], "truncated": info["truncated"]}
        finally:
            wrapper.close()
            reference.env.close()

    def shaping_invariance():
        """Shaping may only add a constant that depends on the start state."""
        gamma, coef = 0.99, 1.0
        plain = HADWrapper((6, 6, 2), max_steps=100, gamma=gamma, fold_wipeout_tail=False)
        shaped = HADWrapper((6, 6, 2), max_steps=100, gamma=gamma, fold_wipeout_tail=False,
                            shaping_coef=coef)
        try:
            for seed in (211, 212, 213):
                plain.reset(seed=seed)
                shaped.reset(seed=seed)
                start = shaped.potential()
                assert plain.potential() == 0.0
                totals, discount = [0.0, 0.0], 1.0
                for step in range(100):
                    actions = (np.arange(plain.n_red) + 5 * step) % 9
                    rewards = [wrapper.step(actions)[0] for wrapper in (plain, shaped)]
                    for i, value in enumerate(rewards):
                        totals[i] += discount * value
                    discount *= gamma
                    if plain.adapter.env.is_episode_done():
                        break
                assert abs(plain.return_sum - shaped.return_sum) <= 1e-9
                assert abs((totals[1] - totals[0]) - (-start)) <= 1e-6, (seed, totals, start)
            return {"configuration": "6v6 K2", "seeds": [211, 212, 213], "coefficient": coef,
                    "identity": "discounted shaped return - plain return = -potential(s0)",
                    "tolerance": 1e-6}
        finally:
            plain.close()
            shaped.close()

    def natural_completion():
        wrapper = HADWrapper((4, 4, 1), max_steps=100)
        wrapper.reset(seed=174)
        env = wrapper.adapter.env
        try:
            for agent in env.agents:
                agent.Health = 0.0
            attacker = env.blue_agents[0]
            attacker.Health = attacker.initial_health
            attacker.position = (np.asarray(env.targets[0].position) + np.array([50., 0., 0.])).tolist()
            attacker.velocity = [35., 0., 0.]
            env.update_alive_agents()
            wrapper._refresh_features()
            reward, done, info = wrapper.step(np.zeros(20, dtype=np.int64))
            assert done and info["terminated"] and not info["truncated"]
            assert info["bootstrap_mask"] == 0
            assert abs(reward + env.target_damage) <= 1e-9
            assert abs(reward - env.task_info()["episode_returns"]["Red"]) <= 1e-9
            return {"bootstrap_mask": 0, "D": float(env.target_damage), "return": reward}
        finally:
            wrapper.close()

    def time_limit():
        wrapper = HADWrapper((4, 4, 1), max_steps=1)
        wrapper.reset(seed=175)
        env = wrapper.adapter.env
        try:
            for side, x in ((env.red_agents, -1000.), (env.blue_agents, 1000.)):
                for i, agent in enumerate(side):
                    agent.position = [x, 200. * i, 100.]
                    agent.velocity = [35., 0., 0.]
            env.update_alive_agents()
            wrapper._refresh_features()
            reward, done, info = wrapper.step(np.zeros(20, dtype=np.int64))
            assert done and info["truncated"] and not info["terminated"]
            assert info["bootstrap_mask"] == 1
            return {"bootstrap_mask": 1, "steps": 1, "return": reward}
        finally:
            wrapper.close()

    def subtask_decomposition():
        """ALMA's per-subtask signal must add up to the team reward.

        Runs the training configuration, including shaping and the folded
        Red-wipeout tail, so the decomposition is checked on the same
        transitions the hierarchical learner receives.
        """
        wrapper = HADWrapper((8, 8, 3), max_steps=100, gamma=0.99, shaping_coef=1.0)
        try:
            worst, steps = 0.0, 0
            for seed in (221, 222, 223):
                wrapper.reset(seed=seed)
                blue = slice(wrapper.n_red, wrapper.n_red + wrapper.n_blue)
                targets = slice(wrapper.n_red + wrapper.n_blue, wrapper.n_entities)
                for step in range(100):
                    subtasks = task_masks(wrapper.entities, wrapper.entity_mask, wrapper.scale.K,
                                          wrapper.n_red, wrapper.n_blue, wrapper.n_targets)
                    live = wrapper.entity_mask == 0
                    belongs = (1 - subtasks["entity2task_mask"]).sum(axis=1)
                    assert np.all(belongs[blue][live[blue]] == 1)
                    assert np.all(belongs[targets][live[targets]] == 1)
                    assert np.all(belongs[~live] == 0)
                    assert np.all(subtasks["task_mask"] == [0] * wrapper.scale.K
                                  + [1] * (wrapper.n_targets - wrapper.scale.K))
                    reward, done, info = wrapper.step((np.arange(wrapper.n_red) + 7 * step) % 9)
                    worst = max(worst, abs(float(np.sum(info["task_rewards"])) - reward))
                    steps += 1
                    if done:
                        break
                summary = wrapper.episode_summary()
                assert abs(sum(summary["damage_by_target"]) - summary["D"]) <= 1e-9
                assert np.all(np.asarray(info["tasks_terminated"][wrapper.scale.K:]) == 0)
            return {"configuration": "8v8 K3", "seeds": [221, 222, 223], "steps": steps,
                    "identity": "sum of per-subtask rewards = team reward; damage by target = D",
                    "max_abs_deviation": worst, "tolerance": 1e-6}
        finally:
            wrapper.close()

    record("V1-actions", action_mapping)
    record("V1-physics-reward", physics_equivalence)
    record("V1-natural-bootstrap", natural_completion)
    record("V1-truncation-bootstrap", time_limit)
    record("V1-shaping-invariance", shaping_invariance)
    record("V1-subtask-decomposition", subtask_decomposition)
    return rows
