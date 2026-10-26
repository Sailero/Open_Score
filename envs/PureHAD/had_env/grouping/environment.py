"""Event-driven known-opponent SMDP over the shared HAD physical simulator."""
from __future__ import annotations

from contextlib import contextmanager
import copy

import numpy as np

from .adapter import HADStage3Adapter

from .domain import DecisionState, Entity, Grouping
from .opponents import OPPONENTS, sample


class KnownOpponentEnv:
    """A fixed Blue rule and a stateless Red group executor.

    Survival preserves the legacy terminal-success reward (gamma=1).
    Damage sums physical-step target damage with a negative sign; horizon
    expiry is a sampling truncation in that mode. Decision opportunities are
    absolute periodic ticks and nonterminal casualty events; a macro step
    always executes at least one physical step.
    """

    def __init__(self, red=8, blue=8, max_steps=100, opponent="reactive", seed=0,
                 command_interval=5, executor=None, group_max_size=None,
                 targets=2, target_positions=None, horizon_policy="red_win",
                 task_mode="survival", target_health=None, spatial_dim=2, plane_altitude=None):
        if opponent not in OPPONENTS:
            raise ValueError(f"opponent must be one of {OPPONENTS}")
        if not 1 <= int(max_steps) <= 500 or not 1 <= int(command_interval) <= int(max_steps):
            raise ValueError("max_steps must be in 1..500 and command_interval in 1..max_steps")
        self.opponent = str(opponent)
        self.seed = int(seed)
        self.max_steps = int(max_steps)
        self.command_interval = int(command_interval)
        self.group_max_size = group_max_size
        self.horizon_policy = str(horizon_policy)
        if target_positions is None:
            if int(targets) == 1:
                target_positions = [[-2100.0, 0.0, 100.0]]
            elif int(targets) == 2:
                target_positions = [[-2100.0, -650.0, 100.0], [-2100.0, 650.0, 100.0]]
            else:
                ys = np.linspace(-650.0, 650.0, int(targets))
                target_positions = [[-2100.0, float(y), 100.0] for y in ys]
        self.adapter = HADStage3Adapter(int(red), int(blue), int(targets), max_steps=self.max_steps,
                                        target_positions=target_positions,
                                        blue_rule_style="rush",
                                        horizon_policy=self.horizon_policy,
                                        task_mode=task_mode, target_health=target_health,
                                        spatial_dim=spatial_dim, plane_altitude=plane_altitude)
        self.task_mode = self.adapter.task_mode
        self.target_health = self.adapter.target_health
        self.spatial_dim = self.adapter.spatial_dim
        self.plane_altitude = self.adapter.plane_altitude
        if executor is None:
            from .rules import RuleExecutor
            executor = RuleExecutor()
        self.executor = executor
        self.previous = Grouping((), self.adapter.red_ids)
        self.blue_grouping = Grouping(())
        self.done = False
        self.set_rng(self.seed)

    @property
    def physical(self):
        """Expose the shared adapter to controlled offline diagnostics."""
        return self.adapter

    @contextmanager
    def _legacy_rng(self):
        external = np.random.get_state()
        np.random.set_state(self._numpy_state)
        try:
            yield
        finally:
            self._numpy_state = copy.deepcopy(np.random.get_state())
            np.random.set_state(external)

    def set_rng(self, seed: int):
        """Independent continuation streams; policy rules and state stay fixed."""
        value = int(seed) % 2**32
        self.opponent_rng = np.random.default_rng(value ^ 0x375AC18F)
        self.adapter.env.np_random = np.random.default_rng(value ^ 0x1900CBE7)
        self.adapter.rng = np.random.default_rng(value ^ 0x5A17D0A1)
        self._numpy_state = np.random.RandomState(value ^ 0x619BF029).get_state()

    def reset(self, seed=None) -> DecisionState:
        if seed is not None:
            self.seed = int(seed)
        self.set_rng(self.seed)
        with self._legacy_rng():
            self.adapter.reset(seed=self.seed,
                               red_assignment={i: None for i in self.adapter.red_ids},
                               blue_assignment={i: None for i in self.adapter.blue_ids})
        self.previous = Grouping((), self.adapter.red_ids)
        self.blue_grouping = Grouping(())
        self.executor.reset(self.adapter.red_ids)
        self.done = False
        return self.state()

    def state(self) -> DecisionState:
        def entities(records):
            return tuple(Entity(i, tuple(row["position"]), tuple(row["velocity"]), row["health"])
                         for i, row in sorted(records.items()))
        red = entities(self.adapter.agent_states("Red"))
        live_red = tuple(entity.id for entity in red if entity.alive)
        self.executor.prune(live_red)
        return DecisionState(self.adapter.step_count, self.max_steps, self.opponent, red,
                             entities(self.adapter.agent_states("Blue")),
                             entities(self.adapter.target_states()), self.previous.prune(live_red),
                             self.executor.memory(), dict(self.executor.last_actions), self.adapter.spatial_dim)

    def step(self, grouping: Grouping) -> tuple[DecisionState, float, bool, dict]:
        if self.done:
            raise RuntimeError("Cannot step a completed episode; call reset")
        before = self.state()
        grouping.validate(before.ids("red"), (target.id for target in before.targets),
                          max_members=self.group_max_size)
        # Both choices use precisely `before`; Blue cannot read `grouping`.
        blue = sample(before, self.opponent, self.opponent_rng)
        blue.validate(before.ids("blue"), (target.id for target in before.targets))
        self.previous, self.blue_grouping = grouping, blue
        red_assignment, blue_assignment = grouping.assignment(), blue.assignment()
        self.adapter.set_joint_assignments(
            {i: red_assignment.get(i) for i in self.adapter.red_ids},
            {i: blue_assignment.get(i) for i in self.adapter.blue_ids})
        start = self.adapter.step_count
        all_events = []
        event_reason = "periodic"
        physical_info = {}
        no_red_continuation = False
        automatic_blue_events = 0
        macro_reward = 0.0
        while True:
            actions = self.executor.act(self.adapter, self.previous)
            with self._legacy_rng():
                _, rewards, self.done, physical_info = self.adapter.step(actions, blue_style="rush")
            macro_reward += float(rewards["Red"])
            events = physical_info["events"]
            all_events.extend(events)
            casualty = any(event["kind"] == "agents_destroyed" for event in events)
            if self.done:
                event_reason = ("horizon" if self.adapter.step_count >= self.max_steps
                                and bool(physical_info["truncated"]) else "terminal")
                break
            periodic = self.adapter.step_count % self.command_interval == 0
            live_red = any(row["alive"] for row in self.adapter.agent_states("Red").values())
            if not live_red:
                # There is no remaining Red choice, but no invented early
                # terminal either. Continue the real world, including every
                # known Blue command opportunity, until its native outcome.
                no_red_continuation = True
                self.previous = Grouping(())
                if casualty or periodic:
                    blue_state = self.state()
                    self.blue_grouping = sample(blue_state, self.opponent, self.opponent_rng)
                    assignment = self.blue_grouping.assignment()
                    self.adapter.set_joint_assignments(
                        {i: None for i in self.adapter.red_ids},
                        {i: assignment.get(i) for i in self.adapter.blue_ids})
                    automatic_blue_events += 1
                continue
            if casualty:
                event_reason = "casualty"
                break
            if periodic:
                break
        after = self.state()
        self.previous = after.previous
        self.blue_grouping = self.blue_grouping.prune(after.ids("blue"))
        success = (None if self.task_mode == "damage" else
                   bool(self.done and float(physical_info.get("outcome_red", 0.0)) > 0.0))
        delta = int(self.adapter.step_count - start)
        info = {**self.adapter.env.task_info(),
                "spatial_dim": self.spatial_dim, "plane_altitude": self.plane_altitude,
                "delta": delta, "success": success, "physical_steps": self.adapter.step_count,
                "event_reason": event_reason, "events": all_events,
                "terminated": bool(self.done), "truncated": False,
                "initial_red": len(self.adapter.red_ids), "initial_blue": len(self.adapter.blue_ids),
                "remaining_red": len(after.ids("red")), "remaining_blue": len(after.ids("blue")),
                "no_red_continuation": no_red_continuation,
                "automatic_blue_events": automatic_blue_events}
        if self.task_mode == "damage":
            info.update(terminated=bool(physical_info["terminated"]),
                        truncated=bool(physical_info["truncated"]),
                        bootstrap_mask=float(not physical_info["terminated"]),
                        macro_target_damage=-macro_reward)
        return after, macro_reward if self.task_mode == "damage" else float(success), bool(self.done), info

    def snapshot(self) -> dict:
        with self._legacy_rng():
            physical = self.adapter.snapshot()
        return {"physical": physical, "executor": self.executor.snapshot(),
                "previous": self.previous, "blue_grouping": self.blue_grouping,
                "opponent_rng": copy.deepcopy(self.opponent_rng.bit_generator.state),
                "numpy_state": copy.deepcopy(self._numpy_state), "done": self.done,
                "opponent": self.opponent, "max_steps": self.max_steps,
                "command_interval": self.command_interval, "seed": self.seed,
                "group_max_size": self.group_max_size,
                "task_mode": self.task_mode, "target_health": self.target_health,
                "spatial_dim": self.spatial_dim, "plane_altitude": self.plane_altitude,
                "horizon_policy": self.horizon_policy,
                "executor_type": type(self.executor).__name__}

    def restore(self, snapshot: dict) -> DecisionState:
        if (snapshot["opponent"] != self.opponent or snapshot["max_steps"] != self.max_steps
                or snapshot.get("task_mode") != self.task_mode
                or snapshot.get("spatial_dim") != self.spatial_dim
                or snapshot.get("plane_altitude") != self.plane_altitude
                or snapshot.get("target_health") != self.target_health
                or snapshot.get("horizon_policy") != self.horizon_policy
                or snapshot["command_interval"] != self.command_interval
                or snapshot.get("group_max_size", 4) != self.group_max_size
                or snapshot.get("executor_type", type(self.executor).__name__) != type(self.executor).__name__):
            raise ValueError("Snapshot and environment protocols differ")
        with self._legacy_rng():
            self.adapter.restore(snapshot["physical"])
        self.executor.restore(snapshot["executor"])
        self.previous, self.blue_grouping = snapshot["previous"], snapshot["blue_grouping"]
        self.opponent_rng.bit_generator.state = copy.deepcopy(snapshot["opponent_rng"])
        self._numpy_state = copy.deepcopy(snapshot["numpy_state"])
        self.done, self.seed = bool(snapshot["done"]), int(snapshot["seed"])
        return self.state()

    def close(self):
        """No window is opened and no worker process is owned by this env."""
        return None
