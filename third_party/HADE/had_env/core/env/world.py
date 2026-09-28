from had_env.core.config import *
from had_env.core.function.Function import (
    distance, distances_from, attack_intensity_ratio, disturb_intensity_ratio,
    closest_segment_distance, WorldKinematics,
)
from had_env.core.version import CORE_VERSION, PHYSICS_PROTOCOL
from had_env.core.agents.base import Entity
from had_env.core.agents.attack import AttackAgent
from had_env.core.agents.disturb import DisturbAgent
from had_env.core.agents.scout import ScoutAgent


class World:
    def __init__(self, red_scout_n, red_disturb_n, red_attack_n,
                 blue_scout_n, blue_disturb_n, blue_attack_n,
                 target_n, task_mode='survival', target_health=None,
                 spatial_dim=2, plane_altitude=None):
        # 读取世界信息
        self.red_scout_n = red_scout_n
        self.red_disturb_n = red_disturb_n
        self.red_attack_n = red_attack_n
        self.blue_scout_n = blue_scout_n
        self.blue_disturb_n = blue_disturb_n
        self.blue_attack_n = blue_attack_n
        self.target_n = target_n
        if isinstance(spatial_dim, bool) or spatial_dim not in (2, 3):
            raise ValueError("spatial_dim must be 2 or 3")
        self.spatial_dim = int(spatial_dim)
        self.attack_distance = attack_distance_for(self.spatial_dim)
        self.fire_range = self.attack_distance[0]
        self.plane_altitude = float(PlanarAltitude if plane_altitude is None else plane_altitude)
        if not np.isfinite(self.plane_altitude) or not AeroPoint[2][0] <= self.plane_altitude <= AeroPoint[2][1]:
            raise ValueError("plane_altitude must be finite and inside the world height bounds")
        if task_mode not in ('survival', 'damage'):
            raise ValueError("task_mode must be survival or damage")
        self.task_mode = task_mode
        self.target_initialization = 'random'
        self.target_health = float(initial_health if target_health is None else target_health)
        if not np.isfinite(self.target_health) or self.target_health <= 0:
            raise ValueError("target_health must be finite and positive")

        # 创建智能体（这里的智能体是否跟决策的智能体保持一致，如何保持顺序一致的问题呢？还是说不需要保持顺序一致呢？）
        self.world = self.create_world()
        for entity in self.world:
            entity.spatial_dim = self.spatial_dim
            entity.attack_distance = self.attack_distance
            entity.fire_range = self.fire_range
            entity.plane_altitude = self.plane_altitude
            if self.spatial_dim == 2:
                entity.position[2] = self.plane_altitude
                entity.pre_position[2] = self.plane_altitude
        self.red_agent_n = self.red_disturb_n + self.red_attack_n + self.red_scout_n
        self.blue_agent_n = self.blue_disturb_n + self.blue_attack_n + self.blue_scout_n

        self.agents = [agent for agent in self.world if agent.Type != 'Entity']
        self.targets = [entity for entity in self.world if entity.Type == 'Entity']
        self.red_agents = [agent for agent in self.agents if agent.Color == "Red"]
        self.blue_agents = [agent for agent in self.agents if agent.Color == "Blue"]
        self.update_alive_agents()
        self.core_version = CORE_VERSION
        self.physics_protocol = PHYSICS_PROTOCOL
        self.record_events = False
        self.physics_step_count = 0
        self.last_physics_events = []
        color_map = {"Red": 0, "Blue": 1, "Entity": 2}
        type_map = {"Entity": 0, "Attack": 1, "Disturb": 2, "Scout": 3}
        self._color_code = np.asarray([color_map[entity.Color] for entity in self.world], dtype=np.int8)
        self._type_code = np.asarray([type_map[entity.Type] for entity in self.world], dtype=np.int8)
        self.entity_mask = None
        self.boundary_clips = 0

    def reset_episode_state(self):
        """Refresh derived state after every native random or uniform reset."""
        for target in self.targets:
            target.step_damage = 0.0
            target.cumulative_damage = 0.0
        self.update_alive_agents()
        self.physics_step_count = 0
        self.last_physics_events = []
        self.boundary_clips = 0
        self.entity_mask = None
        if hasattr(self, "_render_closed"):
            self._render_closed = False

    def update_alive_agents(self):
        self.alive_agents = [agent for agent in self.agents if agent.Health > 0]
        self.alive_targets = [target for target in self.targets if target.Health > 0]

    @property
    def step_target_damage(self):
        return float(sum(target.step_damage for target in self.targets))

    @property
    def target_damage(self):
        return float(sum(target.cumulative_damage for target in self.targets))

    def _episode_termination_reason(self):
        if self.task_mode == 'survival':
            if any(target.Health < 1e-3 for target in self.targets):
                return 'target_destroyed'
            blue_eliminated = all(agent.Health < 1e-3 for agent in self.blue_agents
                                  if agent.Type == 'Attack')
        else:
            blue_eliminated = all(agent.Health <= 0 for agent in self.blue_agents
                                  if agent.Type == 'Attack')
        return 'blue_attackers_destroyed' if blue_eliminated else None

    def is_episode_done(self):
        """Natural completion only; the caller owns the time limit."""
        return self._episode_termination_reason() is not None

    def task_info(self):
        """Expose task state without accumulating rewards on repeated reads.

        Per-target counters live in entity state so native deep copies and
        grouping entity snapshots retain the complete damage branch point.
        Survival returns remain governed by the existing reward contract.
        """
        by_target = {int(target.Id): float(target.cumulative_damage) for target in self.targets}
        total = float(sum(by_target.values()))
        reason = self._episode_termination_reason()
        return {
            'task_mode': self.task_mode,
            'target_initialization': self.target_initialization,
            'spatial_dim': self.spatial_dim,
            'plane_altitude': self.plane_altitude,
            'attack_distance': list(self.attack_distance),
            'fire_range': self.fire_range,
            'step_target_damage': self.step_target_damage,
            'target_damage': total,
            'target_damage_by_target': by_target,
            'episode_returns': {'Red': -total, 'Blue': total} if self.task_mode == 'damage' else None,
            'episode_done': reason is not None,
            'termination_reason': reason,
        }

    def get_agents_dim_info(self):
        n_entities = self.red_agent_n + self.blue_agent_n + self.target_n
        agents_dim_info = {
            "target_n": self.target_n,
            "red_agent_n": self.red_agent_n,
            "blue_agent_n": self.blue_agent_n,
            "red_scout_n": self.red_scout_n,
            "red_disturb_n": self.red_disturb_n,
            "red_attack_n": self.red_attack_n,
            "blue_scout_n": self.blue_scout_n,
            "blue_disturb_n": self.blue_disturb_n,
            "blue_attack_n": self.blue_attack_n,
            # 局部观测维度：相对位置 + 相对速度 + health + alive + 阵营标志
            "scout_obs_dim": OBS_ENTITY_DIM,
            "disturb_obs_dim": OBS_ENTITY_DIM,
            "attack_obs_dim": OBS_ENTITY_DIM,
            "scout_action_dim": EnvDim,
            "disturb_action_dim": EnvDim,
            "attack_action_dim": EnvDim,
            # 全局状态维度：每个实体的 [绝对位置 + 绝对速度 + is_alive]
            "global_state_dim": n_entities * OBS_ENTITY_DIM,
        }
        return agents_dim_info

    def create_world(self):
        world = []

        # 按照次序初始化智能体
        Id = 0
        render_id = 0

        # 初始化红方侦查智能体
        for _ in range(self.red_scout_n):
            world.append(ScoutAgent('Red', Id, render_id))
            Id += 1
            render_id += 1

        # 初始化红方软杀伤智能体
        for _ in range(self.red_disturb_n):
            world.append(DisturbAgent('Red', Id, render_id))
            Id += 1
            render_id += 1

        # 初始化红方打击智能体
        for _ in range(self.red_attack_n):
            world.append(AttackAgent('Red', Id, render_id))
            Id += 1
            render_id += 1

        render_id = 0

        # 初始化蓝方侦查智能体
        for _ in range(self.blue_scout_n):
            world.append(ScoutAgent('Blue', Id, render_id))
            Id += 1
            render_id += 1

        # 初始化蓝方软杀伤智能体
        for _ in range(self.blue_disturb_n):
            world.append(DisturbAgent('Blue', Id, render_id))
            Id += 1
            render_id += 1

        # 初始化蓝方打击智能体
        for _ in range(self.blue_attack_n):
            world.append(AttackAgent('Blue', Id, render_id))
            Id += 1
            render_id += 1

        render_id = 0

        # 初始化保护目标点
        for _ in range(self.target_n):
            world.append(Entity(Id, render_id, task_mode=self.task_mode,
                                target_health=self.target_health))
            Id += 1
            render_id += 1

        return world

    def get_agents(self):
        return self.agents

    def get_world(self):
        return self.world

    def get_agents_flying_actions(self):
        return [one.get_flying_action() for one in self.agents]

    def get_agents_actions(self):
        return [one.get_action() for one in self.agents]

    def get_status(self):
        return [agent.get_status() for agent in self.world]

    def step(self, flying_action_n):
        self.physics_step_count += 1
        self.last_physics_events = []
        self.boundary_clips = 0
        # Swept collision on the upcoming displacement. Skip agents already
        # killed earlier in this same i-major/j-minor pass.
        for i, first in enumerate(self.alive_agents):
            if first.Health <= 0:
                continue
            p1 = np.asarray(first.position, dtype=np.float64)
            v1 = np.asarray(first.velocity, dtype=np.float64)
            q1 = p1 + v1 * Interval
            for j in range(i):
                second = self.alive_agents[j]
                if second.Health <= 0:
                    continue
                p2 = np.asarray(second.position, dtype=np.float64)
                v2 = np.asarray(second.velocity, dtype=np.float64)
                if float(np.linalg.norm(p1 - p2)) > AvoidanceDistance + float(np.linalg.norm(v1 - v2)) * Interval:
                    continue
                q2 = p2 + v2 * Interval
                separation = closest_segment_distance(p1, q1, p2, q2)
                if separation <= AvoidanceDistance:
                    if self.record_events:
                        for source, target in ((first, second), (second, first)):
                            if target.Health > 0:
                                self._record_physics_event(
                                    "collision", source, target, target.Health, 0,
                                    target.Health, separation,
                                )
                    first.Health = 0
                    second.Health = 0

        self._apply_automatic_fire(flying_action_n)

        snapshot = WorldKinematics(self.world, self._color_code, self._type_code,
                                   attack_distance=self.attack_distance)
        for agent in self.world:
            health_before = float(agent.Health) if self.record_events else None
            agent.update_status(snapshot)
            if getattr(agent, "_clamped_axes", None):
                self.boundary_clips += len(agent._clamped_axes)
            if self.record_events and health_before is not None and health_before > 0:
                self._record_damage_events(agent, snapshot, health_before)

        self.update_alive_agents()

    def _apply_automatic_fire(self, flying_action_n):
        positions = np.asarray([entity.position for entity in self.world], dtype=np.float64)
        health = np.asarray([entity.Health for entity in self.world], dtype=np.float64)
        fire_range = self.fire_range
        for i, agent in enumerate(self.agents):
            agent.set_flying_action(flying_action_n[i])
            if agent.Type != "Attack":
                agent.set_function_action(agent.choose_function_ruled_action(self.world))
                continue
            fire = False
            if agent.Health > 0:
                origin = positions[i]
                if agent.Color == "Blue":
                    mask = (self._type_code == 0) & (health > 0)
                else:
                    mask = (self._color_code == 1) & (self._type_code == 1) & (health > 0)
                if np.any(mask):
                    fire = bool(np.min(np.linalg.norm(positions[mask] - origin, axis=1)) < fire_range)
            agent.set_function_action(fire)
            if self.record_events and agent.Health > 0 and fire:
                self._record_physics_event("fire", agent, None, agent.Health, agent.Health, 0, None)

    def _record_physics_event(self, kind, source, target, before, after, damage, separation, **extra):
        self.last_physics_events.append({
            "kind": kind,
            "step": self.physics_step_count,
            "source_id": int(source.Id),
            "target_id": None if target is None else int(target.Id),
            "health_before": float(before),
            "health_after": float(after),
            "damage": float(damage),
            "distance": None if separation is None else float(separation),
            "source_position": [float(value) for value in source.position],
            "target_position": None if target is None else [float(value) for value in target.position],
            **extra,
        })

    def _record_damage_events(self, target, previous_world, health_before):
        """Record simultaneous combat using the same pre-update positions as hits."""
        victim_pos = getattr(target, "pre_position", target.position)
        if isinstance(previous_world, WorldKinematics):
            if target.Type == "Entity":
                combat_after = float(target.Health)
            else:
                combat_after = float(np.max([
                    0.0,
                    health_before
                    - previous_world.attack_loss(victim_pos, target.Color, target.Type)
                    - previous_world.disturb_loss(victim_pos),
                ]))
            fire_mask = (previous_world.type_code == 1) & (previous_world.health > 0) & previous_world.is_fire
            if target.Type == "Entity":
                fire_mask &= previous_world.color_code == 1
            else:
                victim_code = 0 if target.Color == "Red" else 1
                fire_mask &= previous_world.color_code != victim_code
            for index in np.flatnonzero(fire_mask):
                source = self.world[int(index)]
                separation = distance(victim_pos, previous_world.positions[int(index)])
                damage = AttackIntensity * attack_intensity_ratio(float(separation), previous_world.attack_distance)
                if damage > 0:
                    self._record_physics_event(
                        "attack_damage", source, target, health_before, combat_after,
                        damage, separation, damage_scope="source_unclipped", simultaneous=True,
                    )
            if target.Type == "Attack" and getattr(target, "IsFire", False):
                self._record_physics_event(
                    "self_destruct", target, target, combat_after, target.Health,
                    combat_after - float(target.Health), 0,
                )
            return
        attacks = [source for source in previous_world if source.Type == "Attack"
                   and source.Health > 0
                   and (source.Color == "Blue" if target.Type == "Entity"
                        else source.Color != target.Color)]
        disturbers = [] if target.Type == "Entity" else [
            source for source in previous_world if source.Type == "Disturb" and source.Health > 0
        ]
        if target.Type == "Entity":
            combat_after = float(target.Health)
        else:
            combat_after = float(np.max([
                0, health_before - target.calculate_attack_health_loss(attacks, victim_position=victim_pos)
                - target.calculate_disturb_health_loss(disturbers, victim_position=victim_pos),
            ]))
        for source in attacks:
            separation = distance(victim_pos, source.position)
            damage = AttackIntensity * attack_intensity_ratio(float(separation), self.attack_distance) * source.IsFire
            if damage > 0:
                self._record_physics_event(
                    "attack_damage", source, target, health_before, combat_after,
                    damage, separation, damage_scope="source_unclipped", simultaneous=True,
                )
        for source in disturbers:
            damage = DisturbIntensity * disturb_intensity_ratio(
                source.position, victim_pos, source.velocity,
            ) * source.IsDisturb
            if damage > 0:
                self._record_physics_event(
                    "disturb_damage", source, target, health_before, combat_after,
                    damage, distance(victim_pos, source.position),
                    damage_scope="source_unclipped", simultaneous=True,
                )
        if target.Type == "Attack" and target.IsFire:
            self._record_physics_event(
                "self_destruct", target, target, combat_after, target.Health,
                combat_after - float(target.Health), 0,
            )

    def get_red_blue_relative_distance(self):
        red_agents = [agent for agent in self.agents if agent.Color == "Red"]
        blue_agents = [agent for agent in self.agents if agent.Color == "Blue"]
        red_agents_mean_position = np.mean([agent.get_position() for agent in red_agents], axis=0)
        blue_agents_mean_position = np.mean([agent.get_position() for agent in blue_agents], axis=0)
        return np.linalg.norm(red_agents_mean_position - blue_agents_mean_position)

    def get_blue_target_relative_distance(self):
        blue_agents = [agent for agent in self.agents if agent.Color == "Blue"]
        targets_mean_position = np.mean([agent.get_position() for agent in self.targets], axis=0)
        blue_agents_mean_position = np.mean([agent.get_position() for agent in blue_agents], axis=0)
        return np.linalg.norm(targets_mean_position - blue_agents_mean_position)
