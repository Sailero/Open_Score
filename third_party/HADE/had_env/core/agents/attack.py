from had_env.core.agents.base import BaseAgent
from had_env.core.function.Function import distance, distances_from, attack_intensity_ratio, boundary_loss
from had_env.core.config import *
import numpy as np


# 定义Attack智能体类
class AttackAgent(BaseAgent):
    def __init__(self, color, id_, render_id):
        # 继承基础智能体中的特征
        super(AttackAgent, self).__init__(color, id_, render_id)

        # 设立打击智能体专有特征
        self.Type = 'Attack'
        self.IsFire = False

    def reset(self, initial_position, initial_velocity):
        super(AttackAgent, self).reset(initial_position, initial_velocity)
        self.IsFire = False

    def update_status(self, all_agents):
        super(AttackAgent, self).update_status(all_agents)
        # 如果自身进行打击，那么立刻消亡
        if self.IsFire:
            self.Health = 0

    def get_function_action(self):
        return [self.IsFire]

    def set_function_action(self, is_fire):
        self.IsFire = is_fire

    def get_action(self):
        return super(AttackAgent, self).get_flying_action() + [self.IsFire]

    def own_observation(self, world):
        # 获取攻击距离内仍然存活的其他智能体信息，不含自身
        if self.Health > 0:
            others = [one for one in world if one is not self]
            separations = distances_from(self.get_position(), [one.get_position() for one in others])
            attack_range = self.fire_range
            return [one for one, separation in zip(others, separations)
                    if separation < attack_range and one.Health > 0]
        else:
            return []

    def choose_function_ruled_action(self, all_agents):  # 射程内有对应目标即开火
        fire_range = self.fire_range
        origin = self.get_position()
        if self.Color == 'Blue':
            return any(
                one.Color == 'Entity' and one.Health > 0 and distance(origin, one.get_position()) < fire_range
                for one in all_agents
            )
        if self.Color == 'Red':
            return any(
                one.Color == 'Blue' and one.Health > 0 and distance(origin, one.get_position()) < fire_range
                for one in all_agents
            )
        return False

    def calculate_normalized_distance_to_targets(self, targets):
        """计算智能体到最近目标的归一化距离"""
        if len(targets) == 0:
            return 0  # 如果没有目标，返回最大距离
        
        min_distance = float('inf')
        for target in targets:
            if target.Health > 0:  # 只考虑存活的目标
                dist = distance(self.get_position(), target.get_position())
                min_distance = min(min_distance, dist)
        
        if min_distance == float('inf'):
            return 0
        
        # 使用地图对角线长度作为归一化因子
        d_max = np.sqrt(sum([(AeroPoint[i][1] - AeroPoint[i][0])**2 for i in range(3)]))
        return min_distance / d_max

    def calculate_normalized_distance_to_color(self, agents, target_color):
        """计算智能体到指定颜色最近智能体的归一化距离"""
        target_agents = [a for a in agents if a.Color == target_color and a != self and a.Health > 0]
        if len(target_agents) == 0:
            return 0.0  # 如果没有目标颜色的智能体，返回最大距离

        min_distance = float('inf')
        for target_agent in target_agents:
            dist = distance(self.get_position(), target_agent.get_position())
            min_distance = min(min_distance, dist)

        # 使用地图对角线长度作为归一化因子
        d_max = np.sqrt(sum([(AeroPoint[i][1] - AeroPoint[i][0]) ** 2 for i in range(3)]))

        if self.Color == "Blue":
            return min_distance / d_max - 1
        else:
            return min_distance / d_max

    def calculate_health_damage(self, agents):
        """计算智能体对敌方造成的健康值损耗"""
        # 获取敌方智能体
        enemy_agents = [a for a in agents if a.Color != self.Color and a.Color != 'Entity' and a.Health > 0]
        if len(enemy_agents) == 0:
            return 0.0

        # 计算对敌方造成的伤害（基于攻击范围和强度）
        total_damage = 0.0
        for enemy in enemy_agents:
            dist = distance(self.get_position(), enemy.get_position())
            if dist <= self.attack_distance[1]:  # 在当前场景的攻击范围内
                damage_ratio = max(0, 1 - dist / self.attack_distance[1])
                total_damage += damage_ratio

        return total_damage

    def get_attack_reward(self, world):
        """
        计算单个智能体的个体奖励，返回三个维度的潜在奖励。
        对应论文公式 (22): r = r^1 + λ·r^2 + (1-λ)·r^3 + r^4
        本函数返回前三维 [r^1_hit, r^2_target, r^3_enemy]，r^4_episode 由 env.get_reward() 填充。
        """
        agents = [agent for agent in world if hasattr(agent, 'Color') and agent.Color in ['Red', 'Blue']]
        targets = [target for target in world if hasattr(target, 'Color') and target.Color == 'Entity']

        # r^1: 对敌方造成的健康值损害（不受 λ 加权，始终正向激励）
        r_hit = self.calculate_health_damage(agents)

        # r^2: 与目标距离相关（对应 λ 项）
        r_target = -self.calculate_normalized_distance_to_targets(targets)

        # r^3: 与敌方距离相关（对应 1-λ 项）
        if self.Color == 'Red':
            r_enemy = -self.calculate_normalized_distance_to_color(agents, 'Blue')
        else:
            r_enemy = self.calculate_normalized_distance_to_color(agents, 'Red')

        return [r_hit, r_target, r_enemy]

