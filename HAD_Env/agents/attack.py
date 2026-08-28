from HAD_Env.agents.base import BaseAgent
from HAD_Env.function.Function import distance, attack_intensity_ratio, boundary_loss
from HAD_Env.config import *
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
        # 获取攻击距离内仍然存活的所有智能体信息
        if self.Health > 0:
            return [one for one in world if
                    distance(self.get_position(), one.get_position()) < np.max(AttackDistance) and one.Health > 0]
        else:
            return []

    def choose_function_ruled_action(self, all_agents):  # 决定是否打击
        own_obs = self.own_observation(all_agents)

        RedAttackRatio = [attack_intensity_ratio(distance(self.get_position(), one.get_position()))
                          for one in own_obs if one.Color == 'Red']
        BlueAttackRatio = [attack_intensity_ratio(distance(self.get_position(), one.get_position()))
                           for one in own_obs if one.Color == 'Blue']
        EntityAttackRatio = [attack_intensity_ratio(distance(self.get_position(), one.get_position()))
                             for one in own_obs if one.Color == 'Entity']
        IsFire = False

        # 对于蓝方智能体，打击逻辑有两个，第一是没有遇到目标点时尽可能杀伤红方智能体；第二是遇到目标点时尽可能对目标点进行杀伤
        if self.Color == 'Blue':  # 对于蓝方战机
            if len(EntityAttackRatio) > 0:
                if len(RedAttackRatio) > 0:  # 目标点附近如果存在红方智能体立即打击
                    IsFire = True
                elif np.sum(EntityAttackRatio) >= attack_intensity_ratio(AvoidanceDistance + 1e-3):
                    # 在安全情况下打击强度尽可能达到最大，即到达即将碰撞时打击。
                    IsFire = True
            # else:
            #     if np.sum(RedAttackRatio) >= np.sum(BlueAttackRatio):
            #         # 在没有遇到目标点的情况下，当对敌方智能体可以造成更多杀伤的时候，选择开火
            #         IsFire = True

        # 对于红方智能体，打击逻辑只有一个，就是对蓝方智能体进行更多杀伤的情况下进行开火
        elif self.Color == 'Red':  # 对于红方战机
            if np.sum(BlueAttackRatio) >= (np.sum(RedAttackRatio) + np.sum(EntityAttackRatio)) * RedAttackCoef:
                IsFire = True

        return IsFire

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
            # 使用与原有攻击奖励相似的逻辑，使用AttackDistance[1]作为最大攻击范围
            if dist <= AttackDistance[1]:  # 在攻击范围内
                damage_ratio = max(0, 1 - dist / AttackDistance[1])  # 距离越近伤害越大
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


