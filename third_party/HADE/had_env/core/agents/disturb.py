from had_env.core.agents.base import BaseAgent
from had_env.core.function.Function import within_sector_area, disturb_intensity_ratio, boundary_loss
from had_env.core.config import *


# 定义Disturb智能体类
class DisturbAgent(BaseAgent):
    def __init__(self, color, id_, render_id):
        # 继承基础智能体中的特征
        super(DisturbAgent, self).__init__(color, id_, render_id)

        # 设立干扰智能体专有特征
        self.Type = 'Disturb'
        self.IsDisturb = False

    def reset(self, initial_position, initial_velocity):
        super(DisturbAgent, self).reset(initial_position, initial_velocity)
        self.IsDisturb = False

    def get_function_action(self):
        return [self.IsDisturb]

    def set_function_action(self, is_disturb):
        self.IsDisturb = is_disturb

    def get_action(self):
        return super(DisturbAgent, self).get_flying_action() + [self.IsDisturb]

    def own_observation(self, world):
        # 满足两个条件，第一是角度满足，第二是距离满足
        if self.Health > 0:
            return [one for one in world if
                    within_sector_area(self.get_position(), one.get_position(), self.get_velocity(),
                                       DisturbAngleMax, DisturbDistanceMax) and one.Health > 0]
        else:
            return []

    def get_reward(self, world):
        if self.IsDisturb and self.Health > 0:
            DisturbIntensityRatio = 0
            for one in world:
                # 对保护点不造成伤害
                if one.Color == 'Entity':
                    continue

                # 友伤则惩罚，敌伤则奖励
                if one.Color == self.Color:
                    DisturbIntensityRatio -= disturb_intensity_ratio(self.get_position(), one.get_position(),
                                                                     self.get_velocity())
                else:
                    DisturbIntensityRatio += disturb_intensity_ratio(self.get_position(), one.get_position(),
                                                                     self.get_velocity())

            return reward_disturb_single * DisturbIntensityRatio * DisturbIntensity - boundary_loss(self.get_position())
        else:
            return 0

    # 决定是否开启干扰
    def choose_function_ruled_action(self, all_agents):
        # 这里假设，是在智能体未开启干扰时，仍然可以观察到干扰区域内的敌我方智能体情况。且智能体的干扰不对目标带你产生软杀伤。
        own_obs = self.own_observation(all_agents)

        RedDisturbRatio = [disturb_intensity_ratio(self.get_position(), one.get_position(), self.get_velocity())
                           for one in own_obs if one.Color == 'Red']
        BlueDisturbRatio = [disturb_intensity_ratio(self.get_position(), one.get_position(), self.get_velocity())
                            for one in own_obs if one.Color == 'Blue']

        IsDisturb = False

        # 干扰智能体只有一个干扰逻辑，即对敌方智能体软杀伤强度大于对己方智能体的软杀伤强度时，进行干扰。
        if self.Color == 'Red':
            IsDisturb = np.sum(RedDisturbRatio) < np.sum(BlueDisturbRatio)
        elif self.Color == 'Blue':
            IsDisturb = np.sum(RedDisturbRatio) >= np.sum(BlueDisturbRatio)

        return IsDisturb
