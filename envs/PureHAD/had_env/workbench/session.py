"""Native grouping sessions with isolated policy inference and physical recording."""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import inspect
import time

from had_env.grouping.domain import Grouping
from had_env.grouping.rules import make_env
from had_env.grouping.policies import RulePolicy
from had_env.grouping.protocol import stable_seed
from .identity import source_identity
from .protocols import ScenarioSpec, ProtocolSpec
from .recording import EpisodeRecorder, capture_frame, serializable
from .rng import random_state, restore_random_state, seed_everything, policy_import_guard
from .snapshot import decode_snapshot, encode_snapshot

_POLICIES = {name: (lambda seed=0, name=name: RulePolicy(name, seed))
             for name in ("rule", "grand", "static_rule", "random")}


def register_policy(name, factory, *, replace_existing=False):
    """Register a factory(seed=...) returning act(state) or act_env(branch)."""
    if not name or not callable(factory):
        raise ValueError("A policy needs a name and a callable factory")
    if name in _POLICIES and not replace_existing:
        raise ValueError(f"Policy {name!r} is already registered")
    _POLICIES[name] = factory


def policy_names():
    return tuple(sorted(_POLICIES))


def branch_from_snapshot(snapshot, *, continuation_seed=None):
    """Create an independent rule environment; never mutate the live branch or ambient RNG."""
    ambient = random_state()
    env = None
    try:
        snapshot = copy.deepcopy(snapshot)
        if snapshot.get("executor_type") != "RuleExecutor":
            raise ValueError("Only registered rule executor snapshots are supported here")
        if not {"task_mode", "target_health", "horizon_policy", "spatial_dim", "plane_altitude"} <= snapshot.keys():
            raise ValueError("Legacy snapshots lack task configuration and cannot be resumed")
        physical = snapshot["physical"]
        count = len(physical.active_target_ids)
        positions = [row["position"] for row in physical.entity_states[-count:]]
        executor = snapshot["executor"]["config"]
        env = make_env(len(physical.red_assignment), len(physical.blue_assignment),
                       opponent=snapshot["opponent"], max_steps=snapshot["max_steps"],
                       command_interval=snapshot["command_interval"], targets=count,
                       target_positions=positions, lookahead=executor["lookahead"],
                       guard_distance=executor["guard_distance"], task_mode=snapshot["task_mode"],
                       target_health=snapshot["target_health"], horizon_policy=snapshot["horizon_policy"],
                       spatial_dim=snapshot["spatial_dim"], plane_altitude=snapshot["plane_altitude"])
        env.reset(seed=snapshot["seed"])
        env.restore(snapshot)
        if continuation_seed is not None:
            env.set_rng(int(continuation_seed))
        env.adapter.env.physics_step_count = env.adapter.step_count
        return env
    except BaseException:
        if env is not None:
            env.close()
        raise
    finally:
        restore_random_state(ambient)


class PolicyAdapter:
    """Policies see immutable observations or a private, independently seeded simulator.

    act_env is a trusted local plugin contract, not a security sandbox. It never
    receives the live environment or its future random stream.
    Import PyTorch before constructing an adapter that uses it; first-time
    imports inside a policy's constructor, reset, or act method are rejected.
    CUDA policies must also initialize CUDA before adapter construction.
    """
    def __init__(self, policy="rule", *, seed=0):
        self.seed = int(seed)
        self.name = policy if isinstance(policy, str) else type(policy).__name__
        ambient = random_state()
        try:
            with policy_import_guard():
                seed_everything(self.seed)
                if isinstance(policy, str):
                    if policy not in _POLICIES:
                        raise ValueError(f"Unknown policy {policy!r}; available: {policy_names()}")
                    self.policy = _POLICIES[policy](seed=self.seed)
                else:
                    self.policy = copy.deepcopy(policy)
                self._reset_policy()
                self._random_state = random_state()
        finally:
            restore_random_state(ambient)
        self.last_trace = {}

    def _reset_policy(self):
        reset = getattr(self.policy, "reset", None)
        if reset is not None:
            parameters = inspect.signature(reset).parameters
            reset(seed=self.seed) if "seed" in parameters else reset()

    def act(self, env, *, planning_seed):
        ambient = random_state()
        if ambient["torch"] is not None and self._random_state["torch"] is None:
            raise RuntimeError("PyTorch was loaded after this PolicyAdapter; recreate the adapter after importing PyTorch to establish its private stream")
        if ambient["cuda"] and not self._random_state["cuda"]:
            raise RuntimeError("CUDA was initialized after this PolicyAdapter; initialize CUDA before constructing the adapter to establish its private stream")
        branch = None
        try:
            restore_random_state(self._random_state)
            with policy_import_guard():
                if hasattr(self.policy, "act_env"):
                    branch = branch_from_snapshot(env.snapshot(), continuation_seed=planning_seed)
                    value = self.policy.act_env(branch)
                else:
                    state = copy.deepcopy(env.state())
                    value = self.policy.act(state) if hasattr(self.policy, "act") else self.policy(state)
                self.last_trace = serializable(copy.deepcopy(getattr(self.policy, "last_trace", {}) or {}))
                return value.action if hasattr(value, "action") else value
        finally:
            self._random_state = random_state()
            restore_random_state(ambient)
            if branch is not None:
                branch.close()


@dataclass(frozen=True)
class Transition:
    state: object
    reward: float
    done: bool
    delta: int
    native_outcome: int | None
    terminated: bool
    truncated: bool
    info: dict

    def __iter__(self):
        return iter((self.state, self.reward, self.done, self.info))


class SimulationSession:
    def __init__(self, scenario=None, policy="rule", *, record=True, policy_seed=0):
        from had_env.core.config import HorizonPolicy
        self.scenario = ScenarioSpec.from_dict(scenario) if isinstance(scenario, dict) else scenario or ScenarioSpec()
        self.policy = PolicyAdapter(policy, seed=policy_seed)
        self.record = bool(record)
        self.env = make_env(self.scenario.red_count, self.scenario.blue_count, opponent=self.scenario.opponent,
                            seed=self.scenario.opening_seed, max_steps=self.scenario.max_steps,
                            command_interval=self.scenario.command_interval,
                            targets=len(self.scenario.target_positions), target_positions=self.scenario.target_positions,
                            task_mode=self.scenario.task_mode, target_health=self.scenario.target_health,
                            horizon_policy=HorizonPolicy if self.scenario.horizon_policy is None else self.scenario.horizon_policy,
                            spatial_dim=self.scenario.spatial_dim,
                            plane_altitude=self.scenario.plane_altitude)
        self.env.reset(seed=self.scenario.opening_seed)
        self.env.set_rng(self.scenario.opponent_seed)
        self.env.adapter.env.record_events = self.record
        self._closed = False
        self._start_recorder()

    def _start_recorder(self, parent=None):
        from had_env.core.config import Interval
        from had_env.core.version import CORE_VERSION, PHYSICS_PROTOCOL
        identity = source_identity()
        metadata = dict(protocol_id=self.scenario.protocol_id, protocol=ProtocolSpec(self.scenario.protocol_id).to_dict(),
                        physics_version=CORE_VERSION, physics_protocol=PHYSICS_PROTOCOL, source_identity=identity,
                        scenario=self.scenario.to_dict(), policies={"red": self.policy.name, "blue": self.scenario.opponent},
                        policy_seed=self.policy.seed, dt=float(Interval), complete=False,
                        start_step=self.env.adapter.step_count, physical_events_available=hasattr(self.env.adapter.env, "last_physics_events"),
                        **self.env.adapter.env.task_info())
        if parent is not None:
            metadata["branch"] = parent
        if self.record:
            metadata["initial_snapshot"] = encode_snapshot(self.snapshot(), source=identity)
        self.recorder = EpisodeRecorder(metadata)
        self.recorder.episode.sparse = not self.record
        # Initial/branch frames have no action or event: no transition has occurred yet.
        initial = capture_frame(self.env.adapter, groups=self._groups())
        initial["events"] = []
        self.recorder.append_frame(initial)

    def _groups(self):
        return {"red": self.env.previous, "blue": self.env.blue_grouping}

    @property
    def done(self):
        return self.env.done

    @property
    def state(self):
        return self.env.state()

    @property
    def episode(self):
        return self.recorder.episode

    def step(self, action=None):
        if self._closed or self.done:
            raise RuntimeError("Cannot step a closed or terminal session")
        if self.episode.metadata.get("execution_status") == "interrupted":
            raise RuntimeError("Interrupted macro execution requires an explicit snapshot branch")
        before = self.env.state()
        snapshot = (encode_snapshot(self.snapshot(), source=self.episode.metadata["source_identity"])
                    if self.record else None)
        started = time.perf_counter()
        intervention = action is not None
        if action is None:
            planning_seed = stable_seed("workbench-planning", self.scenario.scenario_id, before.step, self.policy.seed)
            action = self.policy.act(self.env, planning_seed=planning_seed)
        if isinstance(action, dict):
            action = Grouping.from_dict(action)
        if not isinstance(action, Grouping):
            raise ValueError("Grouping policy must return a Grouping")
        action.validate(before.ids("red"), before.ids("targets"), max_members=None)
        decision = dict(step=before.step, selected_plan=action.to_dict(), state=before.to_dict(),
                        trace={} if intervention else copy.deepcopy(self.policy.last_trace),
                        decision_seconds=time.perf_counter()-started, intervention=intervention,
                        execution_status="in_progress", delta=0)
        if snapshot is not None:
            decision["snapshot"] = snapshot
        # Commit the selected command before its first physical transition. The
        # owned live worker may stop inside this macro, after recording frames.
        self.recorder.append_decision(decision)
        recorded_decision = self.episode.decisions[-1]
        self.episode.metadata["execution_status"] = "running"
        if intervention:
            self.episode.metadata["contains_interventions"] = True
        old_step = self.env.adapter.step
        def observed(*args, **kwargs):
            self.env.adapter.env.physics_step_count = self.env.adapter.step_count
            result = old_step(*args, **kwargs)
            if self.record:
                self.recorder.append_frame(capture_frame(self.env.adapter, groups=self._groups(), info=result[3]))
            return result
        self.env.adapter.step = observed
        try:
            state, reward, done, info = self.env.step(action)
        except BaseException as error:
            recorded_decision.update(execution_status="interrupted",
                                     delta=int(self.env.adapter.step_count-before.step),
                                     interruption_reason=type(error).__name__)
            self.episode.metadata.update(complete=False, execution_status="interrupted",
                                         physical_steps=int(self.env.adapter.step_count))
            raise
        finally:
            self.env.adapter.step = old_step
        recorded_decision.update(delta=int(info["delta"]), event_reason=info["event_reason"],
                                 execution_status="completed")
        damage_mode = self.scenario.task_mode == "damage"
        native_outcome = None if damage_mode else (1 if info["success"] else -1) if done else 0
        if done:
            if not self.record:
                self.recorder.append_frame(capture_frame(self.env.adapter, groups=self._groups()))
            self.recorder.finish(dict(**self.env.adapter.env.task_info(),
                                      success_native=None if damage_mode else bool(info["success"]), outcome_red=native_outcome,
                                      event_reason=info["event_reason"], reward=float(reward),
                                      terminated=bool(info["terminated"]), truncated=bool(info["truncated"]),
                                      execution_status="completed"))
            self.recorder.episode.metadata.update(episode_done=True, termination_reason=info["event_reason"])
        return Transition(state, float(reward), bool(done), int(info["delta"]), native_outcome,
                          bool(info["terminated"]), bool(info["truncated"]), copy.deepcopy(info))

    def run(self, max_decisions=None):
        count = 0
        while not self.done and (max_decisions is None or count < max_decisions):
            self.step()
            count += 1
        return self.episode

    def snapshot(self):
        return self.env.snapshot()

    @classmethod
    def from_snapshot(cls, document, scenario, policy="rule", *, continuation_seed=None, policy_seed=0, record=True):
        """Resume a portable physical branch with an explicitly selected fresh policy."""
        snapshot = decode_snapshot(document)
        scenario = ScenarioSpec.from_dict(scenario) if isinstance(scenario, dict) else scenario
        physical = snapshot["physical"]
        from had_env.core.config import initial_health, HorizonPolicy, PlanarAltitude
        scenario_health = float(initial_health if scenario.target_health is None else scenario.target_health)
        scenario_horizon = HorizonPolicy if scenario.horizon_policy is None else scenario.horizon_policy
        scenario_altitude = float(PlanarAltitude if scenario.plane_altitude is None else scenario.plane_altitude)
        target_positions = tuple(tuple(float(x) for x in row["position"])
                                 for row in physical.entity_states[-len(physical.active_target_ids):])
        if (scenario.red_count != len(physical.red_assignment) or scenario.blue_count != len(physical.blue_assignment)
                or len(scenario.target_positions) != len(physical.active_target_ids)
                or scenario.target_positions != target_positions
                or scenario.max_steps != snapshot["max_steps"] or scenario.command_interval != snapshot["command_interval"]
                or scenario.opponent != snapshot["opponent"] or scenario.task_mode != snapshot["task_mode"]
                or scenario_health != snapshot["target_health"] or scenario_horizon != snapshot["horizon_policy"]
                or scenario.spatial_dim != snapshot["spatial_dim"] or scenario_altitude != snapshot["plane_altitude"]):
            raise ValueError("Scenario and portable snapshot protocols differ")
        if snapshot["done"]:
            raise ValueError("Cannot continue a terminal snapshot")
        result = object.__new__(cls)
        result.scenario = (replace(scenario, opponent_seed=int(continuation_seed), scenario_id="")
                           if continuation_seed is not None else scenario)
        result.policy = PolicyAdapter(policy, seed=policy_seed)
        result.record = bool(record)
        result.env = branch_from_snapshot(snapshot, continuation_seed=continuation_seed)
        result.env.adapter.env.record_events = result.record
        result._closed = False
        result._start_recorder(dict(parent_scenario_id=scenario.scenario_id,
                                   parent_step=physical.step_count, continuation_seed=continuation_seed,
                                   policy_initialization="fresh_selected_policy"))
        return result

    def branch(self, continuation_seed=None):
        if self._closed or self.done:
            raise RuntimeError("Cannot branch a closed or terminal session")
        result = object.__new__(type(self))
        result.scenario = self.scenario
        if continuation_seed is not None:
            result.scenario = replace(self.scenario, opponent_seed=int(continuation_seed), scenario_id="")
        result.policy = copy.deepcopy(self.policy)
        result.record = self.record
        result.env = branch_from_snapshot(self.snapshot(), continuation_seed=continuation_seed)
        result.env.adapter.env.record_events = self.record
        result._closed = False
        result._start_recorder(dict(parent_scenario_id=self.scenario.scenario_id,
                                   parent_step=self.state.step, continuation_seed=continuation_seed))
        return result

    def save(self, path):
        return self.recorder.save(path)

    def close(self):
        if not self._closed:
            self.env.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
