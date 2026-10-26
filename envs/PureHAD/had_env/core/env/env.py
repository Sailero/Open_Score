from had_env.core.env.world import World
from had_env.core.config import *
from had_env.core.function.Function import distance, world_diagonal, velocity_scale
import numpy as np


class Env(World):
    def __init__(self, red_scout_n, red_disturb_n, red_attack_n,
                 blue_scout_n, blue_disturb_n, blue_attack_n,
                 target_n, target_region=None, seed=None,
                 red_spawn_annulus=None, blue_spawn_x=None, spawn_altitude=None,
                 fire_range=None, horizon_policy=None, reward_config=None,
                 task_mode='survival', target_health=None, spatial_dim=2, plane_altitude=None,
                 target_initialization='random', target_positions=None):
        super(Env, self).__init__(red_scout_n, red_disturb_n, red_attack_n,
                                  blue_scout_n, blue_disturb_n, blue_attack_n,
                                  target_n, task_mode=task_mode, target_health=target_health,
                                  spatial_dim=spatial_dim, plane_altitude=plane_altitude)

        # 定义各个智能体数量以便于reset和render
        self.red_agent_n = self.red_disturb_n + self.red_attack_n + self.red_scout_n
        self.blue_agent_n = self.blue_disturb_n + self.blue_attack_n + self.blue_scout_n

        # 读取区域限制
        self.area = AeroPoint
        self.target_region = np.asarray(
            DefaultTargetRegion if target_region is None else target_region,
            dtype=np.float64,
        )
        self.red_spawn_annulus = tuple(RedSpawnAnnulus if red_spawn_annulus is None else red_spawn_annulus)
        self.blue_spawn_x = tuple(BlueSpawnX if blue_spawn_x is None else blue_spawn_x)
        self.spawn_altitude = tuple(SpawnAltitude if spawn_altitude is None else spawn_altitude)
        self.fire_range = float(self.attack_distance[0] if fire_range is None else fire_range)
        if not np.isfinite(self.fire_range) or not 0 < self.fire_range <= self.attack_distance[0]:
            raise ValueError("fire_range must be positive and no larger than the full-damage radius")
        for entity in self.world:
            entity.fire_range = self.fire_range
        self.horizon_policy = str(HorizonPolicy if horizon_policy is None else horizon_policy)
        if self.horizon_policy not in ("red_win", "draw", "blue_win"):
            raise ValueError("horizon_policy must be red_win, draw, or blue_win")
        self.reward_config = {
            "disturb_single": reward_disturb_single,
            "scout_single": reward_scout_single,
            "attack_single": reward_attack_single,
            "boundary": reward_boundary,
            "episode": reward_episode,
        }
        if reward_config:
            self.reward_config.update(reward_config)
        if self.target_region.shape != (3, 2):
            raise ValueError("target_region must contain [low, high] for x, y, z")
        world_bounds = np.asarray(AeroPoint, dtype=np.float64)
        if np.any(self.target_region[:, 0] > self.target_region[:, 1]):
            raise ValueError("target_region lower bounds must not exceed upper bounds")
        if np.any(self.target_region[:, 0] < world_bounds[:, 0]) or np.any(
            self.target_region[:, 1] > world_bounds[:, 1]
        ):
            raise ValueError("target_region must stay inside AeroPoint")
        if target_initialization not in ('random', 'fixed'):
            raise ValueError("target_initialization must be random or fixed")
        self.target_initialization = 'fixed' if target_positions is not None else target_initialization
        self.fixed_target_positions = None
        if self.target_initialization == 'fixed':
            if target_positions is None:
                positions = np.tile(self.target_region.mean(axis=1), (target_n, 1))
                positions[:, 1] = np.linspace(*self.target_region[1], target_n + 2)[1:-1]
            else:
                positions = np.asarray(target_positions, dtype=np.float64).copy()
            if positions.shape != (target_n, EnvDim) or not np.isfinite(positions).all():
                raise ValueError("target_positions must contain one finite [x, y, z] per target")
            if self.spatial_dim == 2:
                positions[:, 2] = self.plane_altitude
            if np.any(positions < world_bounds[:, 0]) or np.any(positions > world_bounds[:, 1]):
                raise ValueError("target_positions must stay inside AeroPoint")
            self.fixed_target_positions = positions
        self.np_random = np.random.default_rng(seed)

        self.surface_show = False if self.red_agent_n + self.blue_agent_n > 10 else True
        self._display_player = None
        self._rgb_player = None
        self._render_closed = False


    def env_random_reset(self, evaluate=False):
        # Target layout is independent of the agents' random/uniform reset mode.
        self._reset_targets()

        # 随机初始化情形
        for agent_id in range(len(self.agents)):
            initial_position, initial_velocity = self.random_reset(self.agents[agent_id].Color, evaluate=evaluate)
            self.agents[agent_id].reset(initial_position, initial_velocity)

        self._separate_planar_spawns(evaluate=evaluate)
        self.reset_episode_state()

    def env_uniform_reset(self):
        # 均匀初始化情形；初始化的顺序和World中create_agents的顺序一致
        # 首先初始化目标点位置
        self._reset_targets()

        agent_id = 0

        # 初始化红方侦查智能体
        for type_agent_id in range(self.red_scout_n):
            initial_position, initial_velocity = self.uniform_reset(type_agent_id, self.red_scout_n, ScoutRatio, 'Red')
            self.agents[agent_id].reset(initial_position, initial_velocity)
            agent_id += 1

        # 初始化红方软杀伤智能体
        for type_agent_id in range(self.red_disturb_n):
            initial_position, initial_velocity = self.uniform_reset(type_agent_id, self.red_disturb_n, DisturbRatio, 'Red')
            self.agents[agent_id].reset(initial_position, initial_velocity)
            agent_id += 1

        # 初始化红方打击智能体
        for type_agent_id in range(self.red_attack_n):
            initial_position, initial_velocity = self.uniform_reset(type_agent_id, self.red_attack_n, AttackRatio, 'Red')
            self.agents[agent_id].reset(initial_position, initial_velocity)
            agent_id += 1

        # 初始化蓝方侦查智能体
        for type_agent_id in range(self.blue_scout_n):
            initial_position, initial_velocity = self.uniform_reset(type_agent_id, self.blue_scout_n, ScoutRatio, 'Blue')
            self.agents[agent_id].reset(initial_position, initial_velocity)
            agent_id += 1

        # 初始化蓝方软杀伤智能体
        for type_agent_id in range(self.blue_disturb_n):
            initial_position, initial_velocity = self.uniform_reset(type_agent_id, self.blue_disturb_n, DisturbRatio, 'Blue')
            self.agents[agent_id].reset(initial_position, initial_velocity)
            agent_id += 1

        # 初始化蓝方打击智能体
        for type_agent_id in range(self.blue_attack_n):
            initial_position, initial_velocity = self.uniform_reset(type_agent_id, self.blue_attack_n, AttackRatio, 'Blue')
            self.agents[agent_id].reset(initial_position, initial_velocity)
            agent_id += 1

        self._separate_planar_spawns()
        self.reset_episode_state()

    def _reset_targets(self):
        if self.target_initialization == 'fixed':
            for target, position in zip(self.targets, self.fixed_target_positions):
                target.reset(position.tolist(), [0.0] * EnvDim)
            return
        placed = []
        spacing = max(float(PlanarSpawnSeparation), float(AvoidanceDistance) + 1e-6)
        for target in self.targets:
            for _ in range(4096):
                point = self.np_random.uniform(self.target_region[:self.spatial_dim, 0],
                                               self.target_region[:self.spatial_dim, 1]).tolist()
                if self.spatial_dim == 2:
                    point.append(self.plane_altitude)
                if self.spatial_dim == 3 or all(np.linalg.norm(np.asarray(point) - previous) > spacing for previous in placed):
                    break
            else:
                raise ValueError("Target region cannot provide the required planar spawn separation")
            target.reset(point, [0.0] * EnvDim)
            placed.append(np.asarray(point))

    def _separate_planar_spawns(self, evaluate=False):
        """Reject close planar openings using only this environment's RNG.

        The 3D path and its random draws remain unchanged. This prevents initial
        overlaps from height projection; it does not prevent later collisions.
        """
        if self.spatial_dim != 2:
            return
        spacing = max(float(PlanarSpawnSeparation), float(AvoidanceDistance) + 1e-6)
        placed = [np.asarray(target.position) for target in self.targets]
        for agent in self.agents:
            point, velocity = agent.position, agent.velocity
            for _ in range(4096):
                if all(np.linalg.norm(np.asarray(point) - previous) > spacing for previous in placed):
                    break
                point, velocity = self.random_reset(agent.Color, evaluate=evaluate)
            else:
                raise ValueError("Spawn regions cannot provide the required planar separation")
            if point is not agent.position:
                agent.reset(point, velocity)
            placed.append(np.asarray(agent.position))

    def step(self, action_n):
        # Validate the complete batch before collision checks or other mutation.
        # Values are deliberately not clipped: retain legacy control semantics.
        try:
            actions = np.asarray(action_n)
        except (TypeError, ValueError) as error:
            raise ValueError("actions must be a numeric [agents, EnvDim] array") from error
        expected = (len(self.agents), EnvDim)
        if actions.shape != expected:
            raise ValueError(f"actions must have shape {expected}, got {actions.shape}")
        if actions.dtype.kind not in "biuf":
            raise ValueError("actions must contain real numeric values")
        if not np.all(np.isfinite(actions)):
            raise ValueError("actions must contain only finite values")
        if self.spatial_dim == 2:
            action_n = actions.copy()
            action_n[:, 2] = 0
        # Preserve incoming scalar dtypes in the existing scaling calculation.
        flying_action_n = [(np.array(action_n[i]) * self.agents[i].aMax).tolist() for i in range(len(action_n))]
        if not np.all(np.isfinite(flying_action_n)):
            raise ValueError("scaled accelerations must contain only finite values")
        super().step(flying_action_n)

    def _render_information(self):
        targets_info_n, red_agents_info_n, blue_agents_info_n = [], [], []
        for target in self.targets:
            position = [(target.position[0] - AeroPoint[0][0]) / (AeroPoint[0][1] - AeroPoint[0][0]),
                        (target.position[1] - AeroPoint[1][0]) / (AeroPoint[1][1] - AeroPoint[1][0]),
                        (target.position[2] - AeroPoint[2][0]) / (AeroPoint[2][1] - AeroPoint[2][0])]
            targets_info_n.append({
                "position": position,
                "alive": target.Health > 0
            })
        for red_agent in self.red_agents:
            position = [(red_agent.position[0] - AeroPoint[0][0]) / (AeroPoint[0][1] - AeroPoint[0][0]),
                        (red_agent.position[1] - AeroPoint[1][0]) / (AeroPoint[1][1] - AeroPoint[1][0]),
                        (red_agent.position[2] - AeroPoint[2][0]) / (AeroPoint[2][1] - AeroPoint[2][0])]
            red_agents_info_n.append({
                "position": position,
                "velocity": red_agent.velocity,
                "type": red_agent.Type,
                "alive": red_agent.Health > 0
            })
        for blue_agent in self.blue_agents:
            position = [(blue_agent.position[0] - AeroPoint[0][0]) / (AeroPoint[0][1] - AeroPoint[0][0]),
                        (blue_agent.position[1] - AeroPoint[1][0]) / (AeroPoint[1][1] - AeroPoint[1][0]),
                        (blue_agent.position[2] - AeroPoint[2][0]) / (AeroPoint[2][1] - AeroPoint[2][0])]
            blue_agents_info_n.append({
                "position": position,
                "velocity": blue_agent.velocity,
                "type": blue_agent.Type,
                "alive": blue_agent.Health > 0
            })

        return targets_info_n, red_agents_info_n, blue_agents_info_n

    def render_rgb_array(self, include_result=True):
        """Return one native HAD frame as an ``H x W x 3`` RGB array."""
        import pygame
        if not pygame.font.get_init():
            pygame.font.init()
        if self._rgb_player is None:
            from had_env.core.render.render import DisplayPlayer
            surface = pygame.Surface((int(ScreenLength), int(ScreenWidth + ScreenHeight)))
            self._rgb_player = DisplayPlayer(surface)
        player = self._rgb_player
        surface = player.screen
        surface.fill(SurfaceColor)
        player.draw(
            self._render_information(),
            self.is_terminal() if include_result else 0,
        )
        # pygame exposes W x H x C; Matplotlib and image writers expect H x W x C.
        return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2)).copy()

    def render(self):
        """Draw one interactive frame; return False after the window is closed."""
        if self._render_closed:
            return False
        import pygame
        if self._display_player is None or not pygame.display.get_init():
            from had_env.core.render.render import DisplayPlayer
            pygame.display.init()
            if not pygame.font.get_init():
                pygame.font.init()
            screen = pygame.display.set_mode((int(ScreenLength), int(ScreenWidth + ScreenHeight)))
            self._display_player = DisplayPlayer(screen)
        player = self._display_player
        player.screen.fill(SurfaceColor)
        player.update(self._render_information(), self.is_terminal())
        if not player.running:
            self.close()
            return False
        return True

    def close(self):
        """Release this environment's render resources without ending Python."""
        if self._display_player is not None:
            self._display_player.close_window()
        self._display_player = None
        self._rgb_player = None
        self._render_closed = True

    def uniform_reset(self, type_id, type_agent_n, ratio, color):
        # ratio是各个阵营智能体在x轴方向的初始位置比例
        # type_agent_n为y轴等分的份数；type_id是第几等分的位置
        # color决定着初始速度方向和x的位置

        # 计算y坐标
        length_interval = (self.area[1][1] - self.area[1][0]) / (type_agent_n + 1)
        agent_y = length_interval * (type_id + 1) + self.area[1][0]

        # 计算x坐标和初始速度方向
        if color == 'Blue':
            agent_x = self.area[0][1] - (self.area[0][1] - self.area[0][0]) * ratio
            initial_velocity = [-vDomain[0]] + [0.0] * (EnvDim - 1)
        else:
            agent_x = (self.area[0][1] - self.area[0][0]) * ratio + self.area[0][0]
            initial_velocity = [vDomain[0]] + [0.0] * (EnvDim - 1)
        agent_z = (self.plane_altitude if self.spatial_dim == 2 else
                   float(self.np_random.uniform(self.spawn_altitude[0], self.spawn_altitude[1])))
        return [agent_x, agent_y, agent_z], initial_velocity

    def _sample_speed(self):
        sample = self.np_random.uniform(low=vDomain[0], high=vDomain[1], size=self.spatial_dim)
        return float(np.linalg.norm(sample) / np.sqrt(self.spatial_dim))

    def _sample_direction(self):
        direction = np.zeros(EnvDim, dtype=np.float64)
        direction[:self.spatial_dim] = self.np_random.normal(size=self.spatial_dim)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            direction = np.zeros(EnvDim, dtype=np.float64)
            direction[0] = 1.0
            return direction
        return direction / norm

    def random_reset(self, color, evaluate=False):
        altitude = (self.plane_altitude if self.spatial_dim == 2 else
                    float(self.np_random.uniform(self.spawn_altitude[0], self.spawn_altitude[1])))
        if color == 'Red':
            target = self.targets[int(self.np_random.integers(0, max(len(self.targets), 1)))]
            inner, outer = self.red_spawn_annulus
            for _ in range(4096 if self.spatial_dim == 2 else 1):
                radius = float(np.sqrt(self.np_random.uniform(inner ** 2, outer ** 2)))
                angle = float(self.np_random.uniform(0.0, 2.0 * np.pi))
                random_x = float(target.position[0] + radius * np.cos(angle))
                random_y = float(target.position[1] + radius * np.sin(angle))
                if self.spatial_dim == 3:
                    random_x = float(np.clip(random_x, self.area[0][0], self.area[0][1]))
                    random_y = float(np.clip(random_y, self.area[1][0], self.area[1][1]))
                    break
                if (self.area[0][0] <= random_x <= self.area[0][1]
                        and self.area[1][0] <= random_y <= self.area[1][1]):
                    break
            else:
                raise ValueError("Red spawn annulus cannot provide an in-bounds planar position after 4096 attempts")
        else:
            low, high = self.blue_spawn_x
            if not evaluate:
                low, high = self.blue_spawn_x
            random_x = float(self.np_random.uniform(low, high))
            random_y = float(self.np_random.uniform(self.area[1][0], self.area[1][1]))
        random_z = float(np.clip(altitude, self.area[2][0], self.area[2][1]))
        velocity = (self._sample_direction() * self._sample_speed()).tolist()
        return [random_x, random_y, random_z], velocity

    def get_observed_agents(self, color):
        # 获取红蓝双方观测的智能体集合
        red_observation = []
        blue_observation = []
        for agent in self.world:
            if agent.Color == 'Red':
                red_observation += agent.own_observation(self.world)
            elif agent.Color == 'Blue':
                blue_observation += agent.own_observation(self.world)
            else:
                # 目标点默认被观测到
                red_observation.append(agent)
                blue_observation.append(agent)

        # 去除重复id的智能体
        red_unique_agents = [one for j, one in enumerate(red_observation)
                             if one.Id not in [a.Id for a in red_observation[:j]]]
        blue_unique_agents = [one for j, one in enumerate(blue_observation)
                              if one.Id not in [a.Id for a in blue_observation[:j]]]

        if color == 'Red':
            return red_unique_agents
        elif color == 'Blue':
            return blue_unique_agents
        else:
            raise ValueError('Wrong color input')

    # 为不完全信息下的对抗留出接口
    def get_observed_agents_bool_list(self, agent, drop_agent_obs_id=False):
        # 获取color阵营观测到智能体的Id集合
        agent_observed_agents = self.get_observed_agents(agent.Color)
        agent_observed_agent_id = [observed_agent.Id for observed_agent in agent_observed_agents]

        # 获取color阵营是否观测到world中每个智能体的bool集合
        observed_agents_bool_list = []
        for observed_agent in self.world:
            # 如果drop_agent_obs_id为True，则丢弃自身的观测数据，与obs维度对应；否则与world的维度对应。
            if observed_agent is agent and drop_agent_obs_id:
                continue
            if observed_agent.Id in agent_observed_agent_id:
                observed_agents_bool_list.append(True)
            else:
                observed_agents_bool_list.append(False)

        return observed_agents_bool_list

    def get_observation(self, normalization=True, is_relative_observation=True):
        """Return per-agent rows of [rel_pos, rel_vel, health, alive, side flags].

        Dead entities are zeroed. ``self.entity_mask`` holds the matching alive bits.
        """
        observation_n = []
        mask_n = []
        diagonal = world_diagonal()
        speed = velocity_scale()
        for agent in self.agents:
            observation = []
            mask = []
            agent_pos = np.asarray(agent.position, dtype=np.float64)
            agent_vel = np.asarray(agent.velocity, dtype=np.float64)
            for other in self.world:
                if other.Id == agent.Id and is_relative_observation:
                    continue
                alive = float(other.Health > 0)
                if alive <= 0:
                    observation.append([0.0] * OBS_ENTITY_DIM)
                    mask.append(0.0)
                    continue
                if is_relative_observation:
                    pos = np.asarray(other.position, dtype=np.float64) - agent_pos
                    vel = np.asarray(other.velocity, dtype=np.float64) - agent_vel
                    if normalization:
                        pos = pos / diagonal
                        vel = vel / speed
                else:
                    pos = np.asarray(other.position, dtype=np.float64)
                    vel = np.asarray(other.velocity, dtype=np.float64)
                    if normalization:
                        pos = np.array([(pos[dim] - AeroPoint[dim][0]) / (AeroPoint[dim][1] - AeroPoint[dim][0])
                                        for dim in range(EnvDim)])
                        vel = vel / speed
                row = pos.tolist() + vel.tolist() + [
                    float(other.Health),
                    1.0,
                    1.0 if other.Color == "Red" else 0.0,
                    1.0 if other.Color == "Blue" else 0.0,
                    1.0 if other.Color == "Entity" else 0.0,
                ]
                observation.append(row)
                mask.append(1.0)
            observation_n.append(observation)
            mask_n.append(mask)
        self.entity_mask = mask_n
        return observation_n

    def get_world_alive(self):
        world_alive = [agent.Health > 0 for agent in self.world]
        return world_alive

    def get_global_state(self, normalization=True):
        """
        返回全局绝对状态，用于 Critic 输入（CTDE 的"集中训练"部分）。
        对死亡实体：位置和速度归零，is_alive=0。
        对存活实体：保留绝对位置和速度，is_alive=1。
        返回: 一维 list，长度 = n_entities * OBS_ENTITY_DIM
        """
        global_state = []
        speed = velocity_scale()
        for entity in self.world:
            is_alive = float(entity.Health > 0)
            flags = [
                1.0 if entity.Color == "Red" else 0.0,
                1.0 if entity.Color == "Blue" else 0.0,
                1.0 if entity.Color == "Entity" else 0.0,
            ]
            if not is_alive:
                global_state.extend([0.0] * OBS_ENTITY_DIM)
                continue
            pos = list(entity.position)
            vel = list(entity.velocity)
            if normalization:
                pos = [(pos[dim] - AeroPoint[dim][0]) / (AeroPoint[dim][1] - AeroPoint[dim][0]) for dim in range(EnvDim)]
                vel = (np.asarray(vel, dtype=np.float64) / speed).tolist()
            global_state.extend(pos + vel + [float(entity.Health), is_alive] + flags)

        return global_state


    def get_reward(self):
        """
        返回每个智能体的个体奖励结构。
        return {
            "RealReward": {"Red": r, "Blue": r}, 
            "LatentReward": {
                "Red": [[r1,r2,r3], [r1,r2,r3], ...], # 每个红方智能体的个体奖励
                "Blue": [[r1,r2,r3], ...]             # 每个蓝方智能体的个体奖励
            }
        }
        """
        # 初始化奖励结构: 4维潜在奖励 [r_hit, r_target, r_enemy, r_episode]
        reward_n = {
            "RealReward": {
                "Red": 0,
                "Blue": 0
            },
            "LatentReward": {
                "Red": [[0.0, 0.0, 0.0, 0.0] for _ in range(self.red_agent_n)],
                "Blue": [[0.0, 0.0, 0.0, 0.0] for _ in range(self.blue_agent_n)]
            }
        }

        if self.task_mode == 'damage':
            # Only this physical step's new target damage is rewarded. Reading
            # rewards or reaching termination never adds cumulative damage again.
            damage = self.step_target_damage
            reward_n['RealReward'] = {'Red': -damage, 'Blue': damage}
            return reward_n

        # 计算红方智能体的个体奖励: [r_hit, r_target, r_enemy]
        for i, agent in enumerate(self.red_agents):
            if agent.Health > 0:
                indiv_reward = agent.get_attack_reward(self.world)
                for dim in range(3):
                    reward_n["LatentReward"]["Red"][i][dim] = indiv_reward[dim]

        # 计算蓝方智能体的个体奖励: [r_hit, r_target, r_enemy]
        for i, agent in enumerate(self.blue_agents):
            if agent.Health > 0:
                indiv_reward = agent.get_attack_reward(self.world)
                for dim in range(3):
                    reward_n["LatentReward"]["Blue"][i][dim] = indiv_reward[dim]

        # 回合结束奖励（对应公式中的 r^4_episode，全队共享）
        episode_scale = self.reward_config.get("episode", reward_episode)
        episode_reward = 0.0 if abs(episode_scale) < 1e-12 else episode_scale * self.is_terminal()
        reward_n["RealReward"]["Red"] = episode_reward
        reward_n["RealReward"]["Blue"] = -episode_reward

        for i in range(self.red_agent_n):
            reward_n["LatentReward"]["Red"][i][3] = reward_n["RealReward"]["Red"]
        for i in range(self.blue_agent_n):
            reward_n["LatentReward"]["Blue"][i][3] = reward_n["RealReward"]["Blue"]
        
        return reward_n

    def get_done(self):
        """
        修改：返回每个智能体的 done 状态。
        如果全场结束 (is_terminal)，则所有人 done。
        如果个体死亡 (Health <= 0)，则该个体 done。
        """
        is_terminal = self.is_episode_done()
        if self.task_mode == 'damage':
            # Fixed team slots keep receiving later team damage rewards; death
            # remains observable through each entity's health/alive mask.
            return [is_terminal] * len(self.agents)
        
        # 红方每个个体的 done
        red_done = [is_terminal or (agent.Health <= 0) for agent in self.red_agents]
        # 蓝方每个个体的 done
        blue_done = [is_terminal or (agent.Health <= 0) for agent in self.blue_agents]
        
        return red_done + blue_done

    def get_info(self):
        info = {"core_version": self.core_version, "physics_protocol": self.physics_protocol,
                **self.task_info()}
        for agent_id in range(len(self.targets)):
            info['target_%d' % agent_id] = {
                "position": self.targets[agent_id].get_position(),
                "health": float(self.targets[agent_id].Health),
                "is_alive": bool(self.targets[agent_id].Health > 0),
                "step_damage": float(self.targets[agent_id].step_damage),
                "cumulative_damage": float(self.targets[agent_id].cumulative_damage),
            }
        for agent_id in range(len(self.agents)):
            info['agent_%d' % agent_id] = {
                "velocity": self.agents[agent_id].get_velocity(),
                "position": self.agents[agent_id].get_position(),
                "is_alive": self.agents[agent_id].Health > 0,
                "Color": self.agents[agent_id].Color
            }
        return info

    def is_terminal(self):
        """Legacy survival outcome; use is_episode_done for task completion.

        A damage score is not a win/loss sign. Returning zero in that mode also
        keeps the native renderer from drawing the survival result banner.
        """
        if self.task_mode == 'damage':
            return 0
        reason = self._episode_termination_reason()
        if reason == 'target_destroyed':
            return -1
        elif reason == 'blue_attackers_destroyed':
            return 1
        else:
            return 0
