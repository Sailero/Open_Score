from HAD_Env.function.Function import *


class Entity:
    # 定义预保护目标点
    def __init__(self, id_, render_id):
        # 定义状态空间中的坐标与速度
        self.position = [0] * EnvDim
        self.initial_position = [0] * EnvDim
        self.velocity = [0] * EnvDim

        # 定义状态空间中的健康值
        self.Health = initial_health

        # 定义预保护目标点编号
        self.Id = id_
        self.render_id = render_id
        self.Color = 'Entity'
        self.Type = 'Entity'

    def reset(self, initial_position, initial_velocity):
        # 重置预保护目标点的位置
        self.position = initial_position
        self.velocity = [0] * EnvDim
        self.Health = initial_health

    def update_status(self, world):
        # AttackAgents是所有打击智能体类的列表集合
        AttackAgents = [agent for agent in world if agent.Type == 'Attack' and agent.Health > 0]
        DistanceList = [distance(self.position, Agent.get_position()) for Agent in AttackAgents]

        # 假设智能体打击具有友伤属性
        IsFireArray = np.array([AttackAgent.IsFire for AttackAgent in AttackAgents])
        AttackIntensityArray = attack_intensity_ratio(DistanceList)

        self.Health = np.max([0, self.Health - AttackIntensity * np.sum(AttackIntensityArray * IsFireArray)])

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
        self.Health = 1

        # 定义智能体的其他特征
        self.Color = color  # 智能体阵营，阵营为Red或Blue
        self.vMax = vDomain[1] if self.Color == "Red" else vDomain[1] + 50  # 最大速度
        self.vMin = vDomain[0]  # 最小速度
        self.wMax = wMax  # 最大角速度
        self.aMax = aMax if self.Color == "Red" else BlueAmaxCoef * aMax

        # 训练时需要调用的接口
        self.Boundary = Boundary
        self.initial_position = None

    def reset(self, initial_position, initial_velocity):
        # 初始化智能体的状态空间
        self.position = initial_position
        self.velocity = initial_velocity
        self.Health = 1
        self.initial_position = initial_position.copy()

    def get_velocity(self):
        return self.velocity

    def get_flying_action(self):
        # 获取当前智能体的飞行动作空间
        return self.acceleration

    def set_flying_action(self, action_list):
        self.acceleration = action_list

    def update_status(self, world):
        # 根据场上的状态与动作对下一步状态进行更新
        # all_agents表示场上所有智能体的类集合

        if self.Health > 0:
            # 更新位置信息
            self.update_position()

            # 更新速度信息
            self.update_velocity()

            # 根据仍然存活的敌我双方，打击和干扰智能体的信息，更新健康值状态
            AttackAgents = [Agent for Agent in world if Agent.Type == 'Attack' and Agent.Health > 0 and Agent.Color != self.Color]
            DisturbAgents = [Agent for Agent in world if Agent.Type == 'Disturb' and Agent.Health > 0]

            # 进行四个维度的健康值计算
            # 这里建模的地方缺少了打击后自身健康值变成0的过程
            # 这里建模的地方最好加上干扰智能体开启干扰后不会影响自身（在示性函数的式子中加一个判定即可）（上述函数中已经体现）

            attack_health_loss = self.calculate_attack_health_loss(AttackAgents)
            disturb_health_loss = self.calculate_disturb_health_loss(DisturbAgents)

            # 将上述结果汇总
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

        # 基于地图边界对位置进行限制，这里AeroPoint是地图边界点
        if self.Boundary:
            for i in range(len(next_position)):
                if next_position[i] > AeroPoint[i][1]:
                    next_position[i] = AeroPoint[i][1]
                if next_position[i] < AeroPoint[i][0]:
                    next_position[i] = AeroPoint[i][0]

        # 更新智能体状态
        self.position = next_position.tolist()

    def update_velocity(self):
        # 获取该时刻的速度和动作
        if self.Color == "Entity":
            return

        now_velocity = np.array(self.get_velocity())
        now_flying_action = np.array(self.get_flying_action())
        # 获取下一时刻的预测速度
        next_velocity = now_velocity + now_flying_action * Interval

        # 避免0/0程序出错。后续需要完善
        if np.linalg.norm(next_velocity) < 1e-3:
            next_velocity = - now_velocity / 10

        # 基于最大速度对智能体限制
        next_velocity = next_velocity * np.clip(np.linalg.norm(next_velocity),
                                                self.vMin, self.vMax) / np.linalg.norm(next_velocity)
        # 基于最大偏转角度对智能体限制
        next_velocity = rotate_restrict_velocity(now_velocity, next_velocity)

        # 更新智能体状态
        self.velocity = next_velocity

    def calculate_attack_health_loss(self, AttackAgents):
        if len(AttackAgents) > 0:
            # 获取打击智能体与自己的距离列表
            AttackDistanceList = [distance(self.get_position(), AttackAgent.get_position())
                                  for AttackAgent in AttackAgents]

            # 根据上述距离与打击成功概率函数获取实际的打击强度
            AttackIntensityArray = attack_intensity_ratio(AttackDistanceList)

            # 获取打击智能体是否开火的数组
            IsFireArray = np.array([AttackAgent.IsFire for AttackAgent in AttackAgents])

            # 计算打击智能体造成的损伤
            health_loss = AttackIntensity * np.sum(IsFireArray * AttackIntensityArray)
        else:
            health_loss = 0

        return health_loss

    def calculate_disturb_health_loss(self, DisturbAgents):
        # 计算干扰智能体对自己的影响
        if len(DisturbAgents) > 0:
            # 获取干扰智能体如果开启干扰的话，对自己的干扰杀伤强度列表
            DisturbIntensityArray = np.array(
                [disturb_intensity_ratio(DisturbAgent.get_position(), self.get_position(),
                                         DisturbAgent.get_velocity()) for DisturbAgent in DisturbAgents])

            # 获取干扰智能体是否开启干扰的列表
            IsDisturbArray = np.array([DisturbAgent.IsDisturb for DisturbAgent in DisturbAgents])

            # 计算干扰智能体造成的损伤
            health_loss = DisturbIntensity * np.sum(IsDisturbArray * DisturbIntensityArray)
        else:
            health_loss = 0

        return health_loss
