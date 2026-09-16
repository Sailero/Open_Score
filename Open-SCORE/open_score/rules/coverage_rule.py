"""Per-step n-versus-one integer assignment with linear Blue prediction."""
from __future__ import annotations

import copy
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from open_score.envs import (
    ACCELERATION_PRIMITIVES, HADStage3Adapter, DecisionState, Entity, Group, Grouping,
    sample, build_decision_state, EpisodeDiagnostics, trajectory_frame,
    had_config as config, HADEnv,
)


def env_constants(spatial_dim=2):
    inner, outer = config.attack_distance_for(spatial_dim)
    fire_range = inner
    lethal_ratio = 1.0 / max(float(config.AttackIntensity), 1e-9)
    if lethal_ratio >= 1.0:
        kill_radius = float(inner)
    else:
        kill_radius = float(inner + (1.0 - lethal_ratio) * (outer - inner))
    return {
        "dt": float(config.Interval),
        "dim": int(config.EnvDim),
        "plane_altitude": float(getattr(config, "PlanarAltitude", 100.0)),
        "vmin": float(config.vDomain[0]),
        "vmax_red": float(config.vDomain[1]),
        "vmax_blue": float(config.vDomain[1]) * float(config.BlueVmaxCoef),
        "amax_red": float(config.aMax),
        "amax_blue": float(config.aMax) * float(config.BlueAmaxCoef),
        "fire_range": fire_range,
        "kill_radius": kill_radius,
        "attack_inner": float(inner),
        "attack_outer": float(outer),
        "attack_intensity": float(config.AttackIntensity),
        "target_hp": float(config.initial_health),
        "avoid": float(config.AvoidanceDistance),
        "bounds": np.asarray(config.AeroPoint, dtype=np.float64),
        "horizon": int(getattr(config, "DefaultMaxSteps", 100)),
    }


CONST = env_constants()
PRIMITIVES = np.asarray(ACCELERATION_PRIMITIVES, dtype=np.float64)
RED_RULE_VERSION = "rule_nv1_v1"


def refresh_constants():
    global CONST
    CONST = env_constants()
    return CONST


def apply_live_params(updates):
    for name, value in updates.items():
        setattr(config, name, value)
    return refresh_constants()


def clipspeed(velocity, vmin, vmax):
    value = np.asarray(velocity, dtype=np.float64)
    speed = float(np.linalg.norm(value))
    if speed < 1e-3:
        fallback = np.zeros(3, dtype=np.float64)
        fallback[0] = vmin
        return fallback
    return value * (float(np.clip(speed, vmin, vmax)) / speed)


def clamp_position(position, bounds=None):
    point = np.asarray(position, dtype=np.float64).copy()
    box = CONST["bounds"] if bounds is None else bounds
    axes = []
    for dim in range(3):
        if point[dim] > box[dim][1]:
            point[dim] = box[dim][1]
            axes.append((dim, 1))
        elif point[dim] < box[dim][0]:
            point[dim] = box[dim][0]
            axes.append((dim, -1))
    return point, axes


def step_kinematics(position, velocity, accel_dir, vmin, vmax, amax, dt=None):
    dt = CONST["dt"] if dt is None else dt
    position = np.asarray(position, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    accel = np.asarray(accel_dir, dtype=np.float64) * amax
    next_position, axes = clamp_position(position + velocity * dt)
    next_velocity = clipspeed(velocity + accel * dt, vmin, vmax)
    for axis, sign in axes:
        if sign > 0 and next_velocity[axis] > 0:
            next_velocity[axis] = 0.0
        elif sign < 0 and next_velocity[axis] < 0:
            next_velocity[axis] = 0.0
    return next_position, next_velocity


def nearest_primitive(direction):
    norm = float(np.linalg.norm(direction))
    if norm < 1e-8:
        return 0
    return int(np.argmax(PRIMITIVES @ (np.asarray(direction, dtype=np.float64) / norm)))


def unit(vector):
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm < 1e-9:
        fallback = np.zeros(3, dtype=np.float64)
        fallback[0] = 1.0
        return fallback
    return value / norm


def classify_targets(blue_states, target_positions):
    targets = np.asarray(target_positions, dtype=np.float64)
    mapping = {}
    for blue_id, position, velocity in blue_states:
        best_k, best_key = 0, None
        for index, goal in enumerate(targets):
            offset = np.asarray(position, dtype=np.float64) - goal
            vel = np.asarray(velocity, dtype=np.float64)
            speed2 = float(np.dot(vel, vel))
            tau = 0.0 if speed2 < 1e-9 else max(0.0, -float(np.dot(offset, vel)) / speed2)
            distance = float(np.linalg.norm(offset + tau * vel))
            key = (distance, tau, index)
            if best_key is None or key < best_key:
                best_key, best_k = key, index
        mapping[int(blue_id)] = int(best_k)
    return mapping


def predict_blue(blue_states, horizon=1):
    """Extrapolate p + v * dt * step without acceleration or boundary rollout."""
    offsets = np.arange(int(horizon) + 1, dtype=np.float64)[:, None] * CONST["dt"]
    return {
        int(blue_id): np.asarray(position, dtype=np.float64)
        + offsets * np.asarray(velocity, dtype=np.float64)
        for blue_id, position, velocity in blue_states
    }


def steer(position, velocity, aim, remaining_steps, vmin=None, vmax=None, amax=None,
          loiter=False, forbidden=None, action_ids=None, fire_range=None):
    vmin = CONST["vmin"] if vmin is None else vmin
    vmax = CONST["vmax_red"] if vmax is None else vmax
    amax = CONST["amax_red"] if amax is None else amax
    dt = CONST["dt"]
    fire = CONST["fire_range"] if fire_range is None else fire_range
    position = np.asarray(position, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    aim = np.asarray(aim, dtype=np.float64)
    offset = aim - position
    distance = float(np.linalg.norm(offset))
    heading = unit(velocity)
    desired = unit(offset)
    cosine = float(np.clip(np.dot(heading, desired), -1.0, 1.0))
    theta = float(np.arccos(cosine))
    speed_turn = amax / max(theta, 1e-3)
    speed_time = distance / max(float(remaining_steps) * dt, dt)
    command = float(np.clip(min(vmax, speed_time, speed_turn), vmin, vmax))
    if loiter and distance < 200.0:
        tangent = np.cross(offset if distance > 1e-6 else np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
        reference = vmin * unit(tangent)
    else:
        reference = command * desired

    def next_state(primitive):
        return step_kinematics(position, velocity, primitive, vmin, vmax, amax, dt)

    def blocked(next_p, next_v):
        if not forbidden:
            return False
        future = next_p + next_v * dt
        return any(float(np.linalg.norm(future - point)) < fire for point in forbidden)

    best_id, best_err = 0, float("inf")
    found = False
    next_velocities = {}
    candidates = range(len(PRIMITIVES)) if action_ids is None else action_ids
    for index in candidates:
        primitive = PRIMITIVES[index]
        next_p, next_v = next_state(primitive)
        next_velocities[index] = next_v
        if blocked(next_p, next_v):
            continue
        error = float(np.sum((next_v - reference) ** 2))
        if error < best_err - 1e-12 or (abs(error - best_err) <= 1e-12 and index < best_id):
            best_id, best_err, found = index, error, True
    if found:
        return best_id
    for index, next_v in next_velocities.items():
        error = float(np.sum((next_v - reference) ** 2))
        if error < best_err - 1e-12 or (abs(error - best_err) <= 1e-12 and index < best_id):
            best_id, best_err = index, error
    return best_id


class CoveragePolicy:
    """Every step: binary n-versus-one assignment covering all selected Blue."""

    def __init__(self, name=RED_RULE_VERSION, seed=0):
        self.name = name
        self.seed = int(seed)
        self.reset()

    def reset(self):
        self.last_plan = {}
        self.last_grouping = None
        self.last_state_key = None

    @staticmethod
    def state_key(state):
        return (state.step, state.red, state.blue, state.targets)

    def act(self, state: DecisionState) -> Grouping:
        reds = sorted(state.alive("red"), key=lambda entity: entity.id)
        blues = sorted(state.alive("blue"), key=lambda entity: entity.id)
        targets = sorted(state.alive("targets"), key=lambda entity: entity.id)
        assignments = {}
        predicted = {}
        selected = []
        objective = 0.0
        grouping = Grouping((), tuple(entity.id for entity in reds))
        if reds and blues and targets:
            red_positions = np.asarray([entity.position for entity in reds], dtype=np.float64)
            blue_positions = np.asarray([entity.position for entity in blues], dtype=np.float64)
            # When outnumbered, rank Blue by current distance to its nearest Red.
            distances = np.linalg.norm(red_positions[:, None, :] - blue_positions[None, :, :], axis=-1)
            nearest = np.min(distances, axis=0)
            chosen = sorted(range(len(blues)), key=lambda j: (nearest[j], blues[j].id))[:len(reds)]
            selected = sorted((blues[j] for j in chosen), key=lambda entity: entity.id)
            paths = predict_blue([(entity.id, entity.position, entity.velocity) for entity in selected], horizon=1)
            predicted = {entity.id: paths[entity.id][1].tolist() for entity in selected}
            costs = np.linalg.norm(red_positions[:, None, :] - np.asarray(list(predicted.values()))[None, :, :], axis=-1)
            n_red, n_blue = costs.shape
            size = n_red * n_blue
            variables = np.arange(size)
            # x[i,b] is binary: one Blue per Red; at least one Red per Blue.
            rows = np.concatenate((np.repeat(np.arange(n_red), n_blue),
                                   n_red + np.tile(np.arange(n_blue), n_red)))
            columns = np.concatenate((variables, variables))
            matrix = coo_matrix((np.ones(2 * size), (rows, columns)),
                                shape=(n_red + n_blue, size)).tocsc()
            lower = np.ones(n_red + n_blue)
            upper = np.concatenate((np.ones(n_red), np.full(n_blue, n_red)))
            solution = milp(
                c=costs.ravel(), integrality=np.ones(size, dtype=np.uint8),
                bounds=Bounds(0.0, 1.0), constraints=LinearConstraint(matrix, lower, upper),
                options={"mip_rel_gap": 0.0},
            )
            if not solution.success or solution.x is None:
                raise RuntimeError(f"nv1 整数规划未求得最优解（第 {state.step} 步）：{solution.message}")
            binary = np.rint(solution.x).reshape(n_red, n_blue)
            if (np.max(np.abs(solution.x.reshape(n_red, n_blue) - binary)) > 1e-6
                    or np.any((binary < 0) | (binary > 1))
                    or np.any(binary.sum(axis=1) != 1) or np.any(binary.sum(axis=0) < 1)):
                raise RuntimeError(f"nv1 整数规划返回了不满足覆盖约束的解（第 {state.step} 步）。")
            assignments = {entity.id: selected[int(np.argmax(binary[i]))].id for i, entity in enumerate(reds)}
            # Native Group.target still denotes a protected target, never a Blue ID.
            inferred = classify_targets([(entity.id, entity.position, entity.velocity) for entity in selected],
                                        [entity.position for entity in targets])
            grouping = Grouping(tuple(
                Group(targets[inferred[blue.id]].id, tuple(red.id for red in reds if assignments[red.id] == blue.id))
                for blue in selected
            ))
            grouping.validate((entity.id for entity in reds), (entity.id for entity in targets), max_members=None)
            objective = float(np.sum(costs * binary))
        selected_ids = [entity.id for entity in selected]
        self.last_plan = {
            "rule_version": RED_RULE_VERSION, "step": int(state.step),
            "assignment": assignments, "selected_blue": selected_ids,
            "ignored_blue": [entity.id for entity in blues if entity.id not in selected_ids],
            "predicted_blue": predicted, "prediction_steps": 1, "total_distance": objective,
        }
        self.last_grouping = grouping
        self.last_state_key = self.state_key(state)
        return grouping


class CoverageExecutor:
    """Follow the current nv1 assignment; reuse only this step's exact solution."""

    def __init__(self, guard_distance=1100.0, *, policy=None):
        self.config = {"version": RED_RULE_VERSION, "prediction_steps": 1,
                       "guard_distance": float(guard_distance)}
        self.policy = policy if policy is not None else CoveragePolicy()
        self.reset(())

    def reset(self, ids):
        self.last_actions = {int(i): -1 for i in ids}
        self.last_stations = {}
        self.last_roles = {}
        self.last_interception = {}
        self.selected_blue = []
        self.ignored_blue = []
        self.forced_contact = 0
        self.switches = 0
        self.policy.reset()

    def prune(self, ids):
        live = set(map(int, ids))
        self.last_actions = {i: a for i, a in self.last_actions.items() if i in live}

    def memory(self):
        return {}

    def snapshot(self):
        return {"config": dict(self.config), "last_actions": dict(self.last_actions),
                "switches": self.switches}

    def restore(self, snapshot):
        if snapshot["config"] != self.config:
            raise ValueError("nv1 executor parameters differ from snapshot")
        self.reset(snapshot["last_actions"])
        self.last_actions = {int(i): int(a) for i, a in snapshot["last_actions"].items()}
        self.switches = int(snapshot.get("switches", 0))

    def plan_stations(self, state: DecisionState, grouping: Grouping):
        if self.policy.last_state_key != self.policy.state_key(state):
            self.policy.act(state)
        if self.policy.last_grouping != grouping:
            raise ValueError("nv1 执行器必须使用同一步整数规划产生的分组。")
        plan = self.policy.last_plan
        stations = {red_id: np.asarray(plan["predicted_blue"][blue_id], dtype=np.float64)
                    for red_id, blue_id in plan["assignment"].items()}
        roles = {red_id: {"kind": "intercept", "tau": 1, "cover": [blue_id]}
                 for red_id, blue_id in plan["assignment"].items()}
        targets = state.alive("targets")
        for red in state.alive("red"):
            if red.id not in stations:
                stations[red.id] = (np.mean([entity.position for entity in targets], axis=0)
                                    + np.array([self.config["guard_distance"], 0.0, 0.0])
                                    if targets else np.asarray(red.position, dtype=np.float64))
                roles[red.id] = {"kind": "guard", "tau": 1, "cover": []}
        self.last_stations = {i: point.tolist() for i, point in stations.items()}
        self.last_roles = roles
        self.last_interception = dict(plan["assignment"])
        self.selected_blue = list(plan["selected_blue"])
        self.ignored_blue = list(plan["ignored_blue"])
        return stations, roles, list(stations.values())

    def act(self, adapter, grouping):
        state = _adapter_state(adapter, grouping)
        grouping = grouping.prune(state.ids("red"))
        reds = {entity.id: entity for entity in state.alive("red")}
        self.prune(reds)
        stations, roles, _ = self.plan_stations(state, grouping)
        result = {int(i): 0 for i in adapter.red_ids}
        for red_id, red in reds.items():
            action = steer(red.position, red.velocity, stations[red_id], 1,
                           loiter=roles[red_id]["kind"] == "guard",
                           action_ids=adapter.valid_action_ids, fire_range=adapter.env.fire_range)
            if self.last_actions.get(red_id, action) != action:
                self.switches += 1
            result[red_id] = int(action)
        self.last_actions.update({i: result[i] for i in reds})
        return result


class DirectActionPolicy:
    """Observation-to-action rule baselines, with no grouping stage or MILP."""

    def __init__(self, kind, seed=0):
        self.kind, self.seed = kind, int(seed)
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.seed)

    def act(self, state, side, action_ids):
        fire_range = config.attack_distance_for(state.spatial_dim)[0]
        friends = state.red if side == "red" else state.blue
        goals = state.alive("blue" if side == "red" else "targets")
        actions = {entity.id: 0 for entity in friends}
        for entity in friends:
            if not entity.alive:
                continue
            if self.kind == "random_accel":
                actions[entity.id] = int(self.rng.choice(action_ids))
            elif goals:
                target = min(goals, key=lambda goal: (
                    float(np.linalg.norm(np.asarray(goal.position) - entity.position)), goal.id))
                actions[entity.id] = steer(
                    entity.position, entity.velocity, target.position, 1,
                    vmax=CONST["vmax_red" if side == "red" else "vmax_blue"],
                    amax=CONST["amax_red" if side == "red" else "amax_blue"],
                    action_ids=action_ids, fire_range=fire_range,
                )
        return actions


def _adapter_state(adapter, grouping, opponent="reactive"):
    return build_decision_state(adapter, grouping, opponent)


STRATEGY_CATALOG = {
    "red": {
        "end_to_end": {"label": "端到端", "policies": {
            "nearest_intercept": "最近威胁直追 · 规则", "random_accel": "随机动作基线"}},
        "hierarchical": {"label": "分层", "layers": [
            {"id": "grouping", "label": "上层 · 拦截分组", "policies": {"nv1": "nv1 整数规划"}},
            {"id": "control", "label": "下层 · 飞行控制", "policies": {"predictive_intercept": "一步预测拦截"}},
        ]},
    },
    "blue": {
        "end_to_end": {"label": "端到端", "policies": {
            "nearest_target": "最近目标直冲 · 规则", "random_accel": "随机动作基线"}},
        "hierarchical": {"label": "分层", "layers": [
            {"id": "grouping", "label": "上层 · 目标分组", "policies": {
                "reactive": "响应防守分配", "concentrated": "集中攻击", "balanced": "均衡分配"}},
            {"id": "control", "label": "下层 · 飞行控制", "policies": {
                "rush": "直接趋近目标", "split_rush": "分散航道趋近"}},
        ]},
    },
}
DEFAULT_STRATEGIES = {
    "red": {"architecture": "hierarchical", "layers": {"grouping": "nv1", "control": "predictive_intercept"}},
    "blue": {"architecture": "hierarchical", "layers": {"grouping": "reactive", "control": "rush"}},
}
_ACTION_FACTORIES = {
    "red": {"nearest_intercept": lambda seed: DirectActionPolicy("nearest_intercept", seed),
            "random_accel": lambda seed: DirectActionPolicy("random_accel", seed)},
    "blue": {"nearest_target": lambda seed: DirectActionPolicy("nearest_target", seed),
             "random_accel": lambda seed: DirectActionPolicy("random_accel", seed)},
}


def register_end_to_end_policy(side, name, label, factory):
    """Register before serving. factory(seed) -> reset()/act(state, side, action_ids).

    act returns global agent IDs mapped to the supplied valid native action IDs;
    state is the immutable public DecisionState, never an opponent's pending action.
    Models must map their local action indices through action_ids in 2D.
    """
    if side not in _ACTION_FACTORIES or not isinstance(name, str) or not name.isidentifier():
        raise ValueError("Policy requires side red/blue and an identifier name")
    if name in _ACTION_FACTORIES[side] or not callable(factory) or not isinstance(label, str) or not label.strip():
        raise ValueError("Policy must have a unique name, label and callable factory")
    _ACTION_FACTORIES[side][name] = factory
    STRATEGY_CATALOG[side]["end_to_end"]["policies"][name] = label


def validate_strategy(side, spec=None):
    """Accept exactly one active architecture; no legacy policy aliases."""
    if spec is None:
        return copy.deepcopy(DEFAULT_STRATEGIES[side])
    if not isinstance(spec, dict) or not isinstance(spec.get("architecture"), str):
        raise ValueError(f"{side} 策略必须明确选择端到端或分层。")
    architecture = spec["architecture"]
    if architecture not in STRATEGY_CATALOG[side]:
        raise ValueError(f"不支持的 {side} 策略架构。")
    catalog = STRATEGY_CATALOG[side][architecture]
    if architecture == "end_to_end":
        policy = spec.get("policy")
        if set(spec) != {"architecture", "policy"} or not isinstance(policy, str) or policy not in catalog["policies"]:
            raise ValueError(f"{side} 端到端架构只接受一个已注册的动作策略。")
    else:
        layers = spec.get("layers")
        if set(spec) != {"architecture", "layers"} or not isinstance(layers, dict) or set(layers) != {row["id"] for row in catalog["layers"]}:
            raise ValueError(f"{side} 分层架构需要完整的上层、下层策略。")
        for row in catalog["layers"]:
            value = layers[row["id"]]
            if not isinstance(value, str) or value not in row["policies"]:
                raise ValueError(f"{side} 的 {row['label']} 未注册。")
    return copy.deepcopy(spec)


TASK_MODES = {"damage": "累计目标伤害", "survival": "目标存活 / 胜负"}


def run_episode(targets=2, red=8, blue=8, seed=0, *, red_strategy=None,
                blue_strategy=None, max_steps=None,
                command_interval=5, record=True, target_positions=None,
                task_mode="survival", target_health=None, spatial_dim=2,
                target_initialization="random", diagnostics=False, record_events=None,
                retain_trajectory=False, on_step=None):
    """nv1 Red decides every step; command_interval retains Blue's cadence."""
    if int(command_interval) < 1:
        raise ValueError("command_interval must be positive")
    red_strategy = validate_strategy("red", red_strategy)
    blue_strategy = validate_strategy("blue", blue_strategy)
    red_layered = red_strategy["architecture"] == "hierarchical"
    blue_layered = blue_strategy["architecture"] == "hierarchical"
    blue_opponent = blue_strategy["layers"]["grouping"] if blue_layered else "direct"
    blue_style = blue_strategy["layers"]["control"] if blue_layered else "rush"
    horizon = int(config.DefaultMaxSteps if max_steps is None else max_steps)
    adapter = HADStage3Adapter(int(red), int(blue), int(targets), max_steps=horizon,
                               target_positions=target_positions, blue_rule_style=blue_style,
                               target_initialization=target_initialization,
                               task_mode=task_mode, target_health=target_health,
                               spatial_dim=spatial_dim)
    policy = CoveragePolicy(seed=seed) if red_layered else None
    executor = (CoverageExecutor(policy=policy) if red_layered else
                _ACTION_FACTORIES["red"][red_strategy["policy"]](int(seed)))
    blue_direct = (None if blue_layered else
                   _ACTION_FACTORIES["blue"][blue_strategy["policy"]](int(seed) ^ 0x375AC18F))
    adapter.reset(seed=int(seed), red_assignment={i: None for i in adapter.red_ids},
                  blue_assignment={i: None for i in adapter.blue_ids})
    adapter.env.record_events = bool(diagnostics if record_events is None else record_events)
    adapter._policy_last_actions = {}
    collector = EpisodeDiagnostics(adapter, seed, blue_opponent, blue_style,
                                   diagnostics, retain_trajectory)
    positions = [list(target.position) for target in adapter.env.targets]
    if red_layered:
        executor.reset(adapter.red_ids)
    else:
        executor.reset()
    if blue_direct is not None:
        blue_direct.reset()
    opponent_rng = np.random.default_rng(int(seed) ^ 0x375AC18F)
    grouping = Grouping((), adapter.red_ids)
    blue_grouping = Grouping((), adapter.blue_ids)
    frames = []
    events = []
    fires = []
    clips = 0
    command_events = 0

    def decide(reason, update_blue=True):
        nonlocal grouping, blue_grouping, command_events
        state = _adapter_state(adapter, grouping, blue_opponent)
        grouping = policy.act(state) if red_layered else Grouping((), state.ids("red"))
        grouping.validate(state.ids("red"), state.ids("targets"), max_members=None)
        if update_blue:
            blue_grouping = (sample(state, blue_opponent, opponent_rng) if blue_layered
                             else Grouping((), state.ids("blue")))
        blue_grouping.validate(state.ids("blue"), state.ids("targets"))
        adapter.set_joint_assignments(grouping.assignment(), blue_grouping.assignment())
        command_events += 1
        return reason

    reason = decide("reset")
    while True:
        before_health = {int(agent.Id): float(agent.Health) for agent in adapter.env.agents}
        before_targets = {i: float(adapter.env.targets[i].Health) for i in adapter.target_ids}
        # Direct policies receive only physical state, without this step's
        # newly committed Red grouping or the opponent's pending action.
        state = _adapter_state(adapter, Grouping((), adapter.red_ids), blue_opponent)
        actions = (executor.act(adapter, grouping) if red_layered else
                   executor.act(state, "red", adapter.valid_action_ids))
        blue_actions = (adapter.commanded_rule_actions("Blue", style=blue_style) if blue_layered else
                        blue_direct.act(state, "blue", adapter.valid_action_ids))
        collector.before_step(actions)
        _, rewards, done, info = adapter.step(actions, blue_actions, blue_style=blue_style)
        adapter._policy_last_actions = dict(actions)
        collector.after_step(actions)
        if on_step is not None:
            on_step(adapter.step_count, state, actions, float(rewards["Red"]), info)
        clips += int(getattr(adapter.env, "boundary_clips", 0))
        fired = []
        for agent in adapter.env.red_agents:
            if getattr(agent, "IsFire", False):
                fired.append(int(agent.Id))
        kills = []
        for agent in adapter.env.blue_agents:
            if before_health.get(int(agent.Id), 0) > 0 and agent.Health <= 0:
                kills.append(int(agent.Id))
        if fired:
            fires.append({"step": adapter.step_count, "red": fired, "kills": kills})
        if record:
            frames.append(_frame(adapter, grouping, blue_grouping, executor, reason, info, fired))
        events.extend(info.get("events", []))
        if done:
            outcome = int(info.get("outcome", info.get("outcome_red", 0)))
            return {
                **adapter.env.task_info(),
                "spatial_dim": adapter.spatial_dim,
                "plane_altitude": adapter.plane_altitude,
                "physics_protocol": adapter.env.physics_protocol,
                "bounds": CONST["bounds"].tolist(),
                "target_damage_by_target": [float(adapter.env.targets[i].cumulative_damage)
                                            for i in adapter.target_ids],
                "seed": int(seed),
                "success": None if task_mode == "damage" else bool(outcome > 0),
                "outcome": outcome,
                "terminated": bool(info.get("terminated")),
                "truncated": bool(info.get("truncated")),
                "steps": int(adapter.step_count),
                "event_reason": "horizon" if info.get("truncated") else "terminal",
                "frames": frames,
                "events": events,
                "fires": fires,
                "command_events": command_events,
                "boundary_clips": clips,
                "forced_contact": int(getattr(executor, "forced_contact", 0)),
                "switches": int(getattr(executor, "switches", 0)),
                "target_health": [float(adapter.env.targets[i].Health) for i in adapter.target_ids],
                "remaining_red": sum(1 for agent in adapter.env.red_agents if agent.Health > 0),
                "remaining_blue": sum(1 for agent in adapter.env.blue_agents if agent.Health > 0),
                "geometry": _geometry_score(adapter, positions, horizon),
                "red_strategy": red_strategy,
                "blue_strategy": blue_strategy,
                "targets": int(targets),
                "red": int(red),
                "blue": int(blue),
                "target_health_before_end": before_targets,
                "episode_summary": collector.summary(),
                "trajectory": collector.trajectory,
            }
        casualty = any(event.get("kind") == "agents_destroyed" for event in info.get("events", []))
        periodic = adapter.step_count % int(command_interval) == 0
        reason = decide("casualty" if casualty else "periodic" if periodic else "physical_step",
                        update_blue=not blue_layered or casualty or periodic)


def _frame(adapter, grouping, blue_grouping, executor, reason, info, fired):
    def pack(rows):
        return {int(i): {
            "p": [float(x) for x in row["position"]],
            "v": [float(x) for x in row["velocity"]],
            "h": float(row["health"]),
            "alive": bool(row["alive"]),
            "initial_health": float(row["initial_health"]),
            "step_damage": float(row["step_damage"]),
            "cumulative_damage": float(row["cumulative_damage"]),
        } for i, row in rows.items()}

    return {
        **adapter.env.task_info(),
        "spatial_dim": adapter.spatial_dim,
        "plane_altitude": adapter.plane_altitude,
        "target_damage_by_target": [float(adapter.env.targets[i].cumulative_damage)
                                    for i in adapter.target_ids],
        "t": int(adapter.step_count),
        "command": reason,
        "red": pack(adapter.agent_states("Red")),
        "blue": pack(adapter.agent_states("Blue")),
        "targets": pack(adapter.target_states()),
        "stations": getattr(executor, "last_stations", {}),
        "assignment": grouping.assignment(),
        "blue_assignment": {int(identity): (None if target is None else int(target))
                            for identity, target in blue_grouping.assignment().items()},
        "interception": dict(getattr(executor, "last_interception", {})),
        "selected_blue": (list(executor.selected_blue) if hasattr(executor, "selected_blue") else None),
        "ignored_blue": (list(executor.ignored_blue) if hasattr(executor, "ignored_blue") else None),
        "fired": list(fired),
        "outcome": info.get("outcome"),
    }


def _geometry_score(adapter, positions, horizon):
    reds = [np.asarray(agent.position) for agent in adapter.env.red_agents]
    blues = [np.asarray(agent.position) for agent in adapter.env.blue_agents]
    goals = [np.asarray(pos) for pos in positions]
    fire = adapter.env.fire_range
    vmax_r, vmax_b = CONST["vmax_red"], CONST["vmax_blue"]

    def eta(points, goal, speed):
        if not points:
            return horizon + 1
        return min(max(0.0, (float(np.linalg.norm(point - goal)) - fire) / max(speed, 1e-6)) for point in points)

    gaps = [eta(blues, goal, vmax_b) - eta(reds, goal, vmax_r) for goal in goals]
    return float(min(gaps) if gaps else 0.0)


def wilson_interval(successes, n, z=1.96):
    if n <= 0:
        return 0.0, 0.0, 0.0
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    margin = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return float(p), float(max(0.0, center - margin)), float(min(1.0, center + margin))


def roc_auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if len(positive) == 0 or len(negative) == 0:
        return 0.5
    wins = 0.0
    for value in positive:
        wins += float(np.sum(value > negative) + 0.5 * np.sum(value == negative))
    return float(wins / (len(positive) * len(negative)))


def evaluate(seeds, targets=2, red=8, blue=8, *, red_strategy=None,
             blue_strategy=None, record_first=0, max_steps=None,
             on_episode=None, task_mode="survival", target_health=None, spatial_dim=2,
             target_initialization="random", diagnostics=False, record_events=None,
             retain_trajectory=False, on_step=None):
    episodes = []
    display = []
    for index, seed in enumerate(seeds):
        keep = index < int(record_first)
        result = run_episode(targets=targets, red=red, blue=blue, seed=int(seed),
                             red_strategy=red_strategy, blue_strategy=blue_strategy,
                             record=keep, max_steps=max_steps,
                             task_mode=task_mode, target_health=target_health, spatial_dim=spatial_dim,
                             target_initialization=target_initialization,
                             diagnostics=diagnostics, record_events=record_events,
                             retain_trajectory=retain_trajectory, on_step=on_step)
        episodes.append(result)
        if keep:
            display.append(result)
        if on_episode is not None:
            on_episode(index + 1, result, keep)
    return _summarize(episodes, display)


def _summarize(episodes, display=None):
    n = len(episodes)
    task_mode = episodes[0].get("task_mode", "survival") if episodes else "survival"
    if any(item.get("task_mode", "survival") != task_mode for item in episodes):
        raise ValueError("Cannot aggregate results from different task modes")
    damage = [float(item.get("target_damage", 0.0)) for item in episodes]
    red_returns = [item["episode_returns"]["Red"] for item in episodes] if task_mode == "damage" else []
    blue_returns = [-value for value in red_returns]
    wins = sum(1 for item in episodes if item["success"])
    rate, low, high = wilson_interval(wins, n)
    breached = sum(1 for item in episodes if item["outcome"] < 0)
    horizon = sum(1 for item in episodes if item["truncated"])
    wiped = sum(1 for item in episodes if item["terminated"] and item["outcome"] > 0)
    fire_count = sum(len(item["fires"]) for item in episodes)
    kills = sum(len(event["kills"]) for item in episodes for event in item["fires"])
    engage = sum(1 for item in episodes if item["fires"])
    steps = sum(item["steps"] for item in episodes)
    clips = sum(item["boundary_clips"] for item in episodes)
    agents_steps = max(1, steps * max(1, episodes[0]["red"] + episodes[0]["blue"]) if episodes else 1)
    return {
        "task_mode": task_mode,
        "mean_target_damage": float(np.mean(damage)) if n else 0.0,
        "mean_red_return": float(np.mean(red_returns)) if red_returns else None,
        "mean_blue_return": float(np.mean(blue_returns)) if blue_returns else None,
        "target_damage_by_episode": damage,
        "red_returns": red_returns,
        "blue_returns": blue_returns,
        "mean_damage_by_target": (np.mean([item["target_damage_by_target"] for item in episodes], axis=0).tolist()
                                  if n else []),
        "zero_damage_count": sum(value == 0.0 for value in damage),
        "n": n,
        "wins": None if task_mode == "damage" else wins,
        "win_rate": None if task_mode == "damage" else rate,
        "wilson": None if task_mode == "damage" else [low, high],
        "breached": None if task_mode == "damage" else breached,
        "horizon": horizon,
        "wiped": None if task_mode == "damage" else wiped,
        "engage_rate": engage / n if n else 0.0,
        "kills_per_fire": kills / fire_count if fire_count else 0.0,
        "clip_rate": clips / agents_steps,
        "mean_steps": steps / n if n else 0.0,
        "geometry": [item["geometry"] for item in episodes],
        "labels": [1 if item["success"] else 0 for item in episodes],
        "forced_contact": sum(item["forced_contact"] for item in episodes),
        "episodes": display if display is not None else episodes,
        "outcomes": [item["outcome"] for item in episodes],
        "breach_steps": [item["steps"] for item in episodes if item["outcome"] < 0],
        "raw": episodes,
    }


def physics_throughput(red=8, blue=8, targets=2, steps=400):
    import time
    env = HADEnv(red, blue, targets, seed=0)
    env.reset(evaluate=True, seed=0)
    zeros = [[0.0, 0.0, 0.0] for _ in env.agents]
    done = 0
    t0 = time.perf_counter()
    while done < steps:
        if env.is_terminal() or env.physics_step_count >= 200:
            env.reset(evaluate=True, seed=done)
        env.step_physics(zeros)
        done += 1
    elapsed = max(time.perf_counter() - t0, 1e-6)
    env.close()
    return done / elapsed


def calibrate(seeds, targets=2, red=8, blue=8, *, blue_strategy=None, max_steps=None):
    refresh_constants()
    rule = evaluate(seeds, targets, red, blue, blue_strategy=blue_strategy,
                    record_first=0, max_steps=max_steps)
    rand = evaluate(seeds, targets, red, blue, blue_strategy=blue_strategy,
                    red_strategy={"architecture": "end_to_end", "policy": "random_accel"},
                    record_first=0, max_steps=max_steps)
    throughput = physics_throughput(red, blue, targets)
    auc = roc_auc(rule["labels"], rule["geometry"])
    first = run_episode(targets, red, blue, int(seeds[0]), record=False, max_steps=max_steps)
    second = run_episode(targets, red, blue, int(seeds[0]), record=False, max_steps=max_steps)
    gates = {
        "G1": 0.25 <= rule["win_rate"] <= 0.75,
        "G2": (rule["win_rate"] - rand["win_rate"] >= 0.20) and rand["win_rate"] >= 0.02,
        "G3": auc <= 0.75,
        "G4": rule["engage_rate"] >= 0.90,
        "G5": 0.20 <= (rule["breached"] / rule["n"] if rule["n"] else 0) <= 0.70,
        "G6": 0.50 <= rule["kills_per_fire"] <= 3.0,
        "G7": rule["clip_rate"] <= 0.05,
        "G8": throughput >= 500.0,
        "G9": first["outcome"] == second["outcome"] and first["steps"] == second["steps"],
    }
    return {
        "rule": {k: rule[k] for k in ("n", "wins", "win_rate", "wilson", "breached", "horizon",
                                      "wiped", "engage_rate", "kills_per_fire", "clip_rate", "mean_steps")},
        "random": {k: rand[k] for k in ("n", "wins", "win_rate", "wilson", "engage_rate")},
        "auc": auc,
        "throughput": throughput,
        "gates": gates,
        "passed": all(gates.values()),
        "failed": [name for name, ok in gates.items() if not ok],
        "constants": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in CONST.items()},
        "repeatable": gates["G9"],
    }


def parameter_snapshot(spatial_dim=2):
    names = [
        "vDomain", "aMax", "BlueAmaxCoef", "BlueVmaxCoef",
        "AttackIntensity", "initial_health", "AvoidanceDistance",
        "AeroPoint", "SpawnAltitude", "RedSpawnAnnulus", "BlueSpawnX",
        "DefaultMaxSteps", "HorizonPolicy",
        "PlanarAltitude", "PlanarSpawnSeparation",
        "Interval", "wMax", "Boundary", "DefaultTargetRegion",
        "DisturbAngleMax", "DisturbDistanceMax", "DisturbIntensity", "GaussSigma",
        "DisturbStopDistanceRatio", "ScoutAngleMax", "ScoutDistanceMax",
        "reward_disturb_single", "reward_scout_single", "reward_attack_single",
        "reward_boundary", "reward_episode",
    ]
    ranges = config.attack_distance_for(spatial_dim)
    values = {"AttackDistance": list(ranges), "FireRange": ranges[0]}
    for name in names:
        value = getattr(config, name)
        values[name] = value.tolist() if hasattr(value, "tolist") else value
    return values
