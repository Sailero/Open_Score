from had_env.core.agents.base import BaseAgent
from had_env.core.function.Function import within_sector_area, boundary_loss
from had_env.core.config import *


# 定义Scout智能体类
class ScoutAgent(BaseAgent):
    def __init__(self, color, id_, render_id):
        # 继承基础智能体中的特征
        super(ScoutAgent, self).__init__(color, id_, render_id)
        # 设立侦察智能体专有特征
        self.Type = 'Scout'

    def get_function_action(self):
        return []

    def set_function_action(self, action):
        pass

    def get_action(self):
        return self.get_flying_action()

    def own_observation(self, world):
        # 满足两个条件，第一是角度满足，第二是距离满足
        if self.Health > 0:
            return [one for one in world if
                    within_sector_area(self.get_position(), one.get_position(), self.get_velocity(),
                                       ScoutAngleMax, ScoutDistanceMax) and one.Health > 0]
        else:
            return []

    def get_reward(self, world):
        scout_obs = self.own_observation(world)
        return reward_scout_single * len(scout_obs) - boundary_loss(self.get_position())

    def choose_function_ruled_action(self, all_agents):
        return []
