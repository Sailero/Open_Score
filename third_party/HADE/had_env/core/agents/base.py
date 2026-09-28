from had_env.core.function.Function import *


class Entity:
    # 定义预保护目标点
    def __init__(self, id_, render_id, task_mode='survival', target_health=None):
        # 定义状态空间中的坐标与速度
        self.position = [0] * EnvDim
        self.initial_position = [0] * EnvDim
        self.velocity = [0] * EnvDim

        # 定义状态空间中的健康值
        if task_mode not in ('survival', 'damage'):
            raise ValueError("task_mode must be survival or damage")
        self.task_mode = task_mode
        self.initial_health = float(initial_health if target_health is None else target_health)
        if not np.isfinite(self.initial_health) or self.initial_health <= 0:
            raise ValueError("target_health must be finite and positive")
        self.Health = self.initial_health
        self.step_damage = 0.0
        self.cumulative_damage = 0.0
        self.spatial_dim = 3
        self.attack_distance = attack_distance_for(self.spatial_dim)
        self.fire_range = self.attack_distance[0]
        self.plane_altitude = float(PlanarAltitude)
        self.pre_position = [0] * EnvDim
        self._clamped_axes = []

        # 定义预保护目标点编号
        self.Id = id_
        self.render_id = render_id
        self.Color = 'Entity'
        self.Type = 'Entity'

    def reset(self, initial_position, initial_velocity):
        # 重置预保护目标点的位置
        self.position = list(initial_position)
        if self.spatial_dim == 2:
            self.position[2] = self.plane_altitude
        self.initial_position = list(self.position)
        self.velocity = [0] * EnvDim
        self.Health = self.initial_health
        self.step_damage = 0.0
        self.cumulative_damage = 0.0
        self.pre_position = list(self.position)
        self._clamped_axes = []

    def update_status(self, world):
        # In the asset-defence task, targets belong to Red and are damaged only
        # by Blue attackers. Fire and hit distances both use pre-update positions.
        self.pre_position = list(self.position)
        if isinstance(world, WorldKinematics):
            loss = world.attack_loss(self.pre_position, self.Color, self.Type)
        else:
            AttackAgents = [agent for agent in world
                            if agent.Type == 'Attack' and agent.Color == 'Blue' and agent.Health > 0]
            DistanceList = distances_from(self.pre_position, [Agent.get_position() for Agent in AttackAgents])
            IsFireArray = np.array([AttackAgent.IsFire for AttackAgent in AttackAgents])
            AttackIntensityArray = attack_intensity_ratio(DistanceList, self.attack_distance)
            loss = AttackIntensity * np.sum(AttackIntensityArray * IsFireArray)
        # Damage is measured before HP clipping, including simultaneous overkill.
        # Damage-mode targets remain ordinary finite-health observable entities.
        self.step_damage = float(loss)
        self.cumulative_damage += self.step_damage
        if self.task_mode == 'damage':
            self.Health = self.initial_health
        else:
            self.Health = np.max([0, self.Health - loss])

    def set_position(self, position_list):
        self.position = position_list

    def get_position(self):
        return self.position

    def get_status(self):
        # 给颜色编号
        if self.Color == 'Red':
            color = 1
        elif self.Color == 'Blue':
            color = 2
        else:
            color = 0

        # 给type编号
        if self.Type == 'Attack':
            ty = 1
        elif self.Type == 'Disturb':
            ty = 2
        elif self.Type == 'Scout':
            ty = 3
        else:
            ty = 0
        return self.position + self.velocity #  + [self.Health]  # + [color, ty, self.Health]  # [p, v, c, t, h]


# 定义智能体基类
class BaseAgent(Entity):
    def __init__(self, color, id_, render_id):
        super(BaseAgent, self).__init__(id_, render_id)
        # 定义状态空间的属性
        self.acceleration = [0] * EnvDim
        self.initial_health = 1.0
        self.Health = 1

        # 定义智能体的其他特征
        self.Color = color  # 智能体阵营，阵营为Red或Blue
        self.vMax = vDomain[1] if self.Color == "Red" else vDomain[1] * BlueVmaxCoef  # 最大速度
        self.vMin = vDomain[0]  # 最小速度
        self.wMax = wMax  # 最大角速度
        self.aMax = aMax if self.Color == "Red" else BlueAmaxCoef * aMax

        # 训练时需要调用的接口
        self.Boundary = Boundary
        self.initial_position = None

    def reset(self, initial_position, initial_velocity):
        # 初始化智能体的状态空间
        self.position = list(initial_position)
        self.velocity = list(initial_velocity)
        if self.spatial_dim == 2:
            self.position[2] = self.plane_altitude
            planar = np.asarray(self.velocity[:2], dtype=np.float64)
            speed = float(np.linalg.norm(planar))
            if speed < 1e-3:
                planar = np.array([self.vMin if self.Color == 'Red' else -self.vMin, 0.0])
            else:
                planar *= float(np.clip(speed, self.vMin, self.vMax)) / speed
            self.velocity = [float(planar[0]), float(planar[1]), 0.0]
        self.acceleration = [0.0] * EnvDim
        self.Health = 1
        self.initial_position = list(self.position)
        self.pre_position = list(self.position)
        self._clamped_axes = []

    def get_attack_reward(self, world):
        """Adapt legacy Scout/Disturb rewards to the shared three-slot contract.

        AttackAgent retains its existing geometric reward implementation.
        Non-attack roles keep their own scalar reward in the first slot; target
        and enemy distance shaping remain zero for these roles.
        """
        return [float(self.get_reward(world)), 0.0, 0.0]

    def get_velocity(self):
        return self.velocity

    def get_flying_action(self):
        # 获取当前智能体的飞行动作空间
        return self.acceleration

    def set_flying_action(self, action_list):
        if self.spatial_dim == 2:
            self.acceleration = [action_list[0], action_list[1], 0.0]
        else:
            self.acceleration = action_list

    def update_status(self, world):
        # 根据场上的状态与动作对下一步状态进行更新
        # all_agents表示场上所有智能体的类集合

        if self.Health > 0:
            self.pre_position = list(self.position)
            self.update_position()
            self.update_velocity()
            if isinstance(world, WorldKinematics):
                attack_health_loss = world.attack_loss(self.pre_position, self.Color, self.Type)
                disturb_health_loss = world.disturb_loss(self.pre_position)
            else:
                AttackAgents = [Agent for Agent in world if Agent.Type == 'Attack' and Agent.Health > 0 and Agent.Color != self.Color]
                DisturbAgents = [Agent for Agent in world if Agent.Type == 'Disturb' and Agent.Health > 0]
                attack_health_loss = self.calculate_attack_health_loss(AttackAgents, victim_position=self.pre_position)
                disturb_health_loss = self.calculate_disturb_health_loss(DisturbAgents, victim_position=self.pre_position)
            self.Health = np.max([0, self.Health - attack_health_loss - disturb_health_loss])
        else:
            self.velocity = [0] * EnvDim

    def update_position(self):
        if self.Color == "Entity":
            return
            
        # 获取当前飞行状态
        now_position = np.array(self.get_position())
        now_velocity = np.array(self.get_velocity())

        # 获取下一时刻的位置
        next_position = now_position + now_velocity * Interval
        if self.spatial_dim == 2:
            next_position[2] = self.plane_altitude

        # 基于地图边界对位置进行限制，这里AeroPoint是地图边界点
        self._clamped_axes = []
        if self.Boundary:
            for i in range(len(next_position)):
                if next_position[i] > AeroPoint[i][1]:
                    next_position[i] = AeroPoint[i][1]
                    self._clamped_axes.append((i, 1))
                elif next_position[i] < AeroPoint[i][0]:
                    next_position[i] = AeroPoint[i][0]
                    self._clamped_axes.append((i, -1))

        # 更新智能体状态
        self.position = next_position.tolist()

    def update_velocity(self):
        # 获取该时刻的速度和动作
        if self.Color == "Entity":
            return

        now_velocity = np.array(self.get_velocity())
        now_flying_action = np.array(self.get_flying_action())
        if self.spatial_dim == 2:
            now_velocity[2] = 0.0
            now_flying_action[2] = 0.0
        # 获取下一时刻的预测速度
        next_velocity = now_velocity + now_flying_action * Interval
        speed = float(np.linalg.norm(next_velocity))
        # Keep heading instead of reversing when the commanded increment cancels.
        # Scaling back up to vMin can add at most vMin beyond the aMax*dt budget.
        if speed < 1e-3:
            previous = float(np.linalg.norm(now_velocity))
            if previous < 1e-3:
                next_velocity = np.zeros(EnvDim, dtype=np.float64)
                next_velocity[0] = self.vMin
                speed = float(self.vMin)
            else:
                next_velocity = now_velocity * (self.vMin / previous)
                speed = float(self.vMin)
        clipped = float(np.clip(speed, self.vMin, self.vMax))
        next_velocity = next_velocity * (clipped / max(speed, 1e-12))
        for axis, sign in self._clamped_axes:
            if sign > 0 and next_velocity[axis] > 0:
                next_velocity[axis] = 0.0
            elif sign < 0 and next_velocity[axis] < 0:
                next_velocity[axis] = 0.0
        if self.spatial_dim == 2:
            # Constrain heading on the circle, including deterministic 180-degree
            # turns, before any 3D rotation could introduce a vertical component.
            before_speed = float(np.linalg.norm(now_velocity[:2]))
            after_speed = float(np.linalg.norm(next_velocity[:2]))
            if before_speed * after_speed > 1e-3:
                old_angle = float(np.arctan2(now_velocity[1], now_velocity[0]))
                new_angle = float(np.arctan2(next_velocity[1], next_velocity[0]))
                turn = (new_angle - old_angle + np.pi) % (2 * np.pi) - np.pi
                if abs(turn) > self.wMax:
                    angle = old_angle + float(np.clip(turn, -self.wMax, self.wMax))
                    next_velocity[:2] = after_speed * np.array([np.cos(angle), np.sin(angle)])
            next_velocity[2] = 0.0
            next_velocity = next_velocity.tolist()
        else:
            next_velocity = rotate_restrict_velocity(now_velocity, next_velocity)
        self.velocity = next_velocity

    def calculate_attack_health_loss(self, AttackAgents, victim_position=None):
        AttackAgents = [agent for agent in AttackAgents if agent.Color != self.Color]
        position = self.get_position() if victim_position is None else victim_position
        if len(AttackAgents) > 0:
            AttackDistanceList = distances_from(
                position, [AttackAgent.get_position() for AttackAgent in AttackAgents],
            )

            # 根据上述距离与打击成功概率函数获取实际的打击强度
            AttackIntensityArray = attack_intensity_ratio(AttackDistanceList, self.attack_distance)

            # 获取打击智能体是否开火的数组
            IsFireArray = np.array([AttackAgent.IsFire for AttackAgent in AttackAgents])

            # 计算打击智能体造成的损伤
            health_loss = AttackIntensity * np.sum(IsFireArray * AttackIntensityArray)
        else:
            health_loss = 0

        return health_loss

    def calculate_disturb_health_loss(self, DisturbAgents, victim_position=None):
        position = self.get_position() if victim_position is None else victim_position
        if len(DisturbAgents) > 0:
            DisturbIntensityArray = np.array(
                [disturb_intensity_ratio(DisturbAgent.get_position(), position,
                                         DisturbAgent.get_velocity()) for DisturbAgent in DisturbAgents])

            # 获取干扰智能体是否开启干扰的列表
            IsDisturbArray = np.array([DisturbAgent.IsDisturb for DisturbAgent in DisturbAgents])

            # 计算干扰智能体造成的损伤
            health_loss = DisturbIntensity * np.sum(IsDisturbArray * DisturbIntensityArray)
        else:
            health_loss = 0

        return health_loss
