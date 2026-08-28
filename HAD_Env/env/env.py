import pygame
from HAD_Env.env.world import World
from HAD_Env.render.render import DisplayPlayer
from HAD_Env.config import *
from HAD_Env.function.Function import distance, normalized_obs_n
import numpy as np


class Env(World):
    def __init__(self, red_scout_n, red_disturb_n, red_attack_n,
                 blue_scout_n, blue_disturb_n, blue_attack_n,
                 target_n):
        super(Env, self).__init__(red_scout_n, red_disturb_n, red_attack_n,
                                  blue_scout_n, blue_disturb_n, blue_attack_n,
                                  target_n)

        # 定义各个智能体数量以便于reset和render
        self.red_agent_n = self.red_disturb_n + self.red_attack_n + self.red_scout_n
        self.blue_agent_n = self.blue_disturb_n + self.blue_attack_n + self.blue_scout_n

        # 读取区域限制
        self.area = AeroPoint

        self.surface_show = False if self.red_agent_n + self.blue_agent_n > 10 else True


    def env_random_reset(self, evaluate=False):
        # 首先初始化目标点位置
        for type_target_id in range(self.target_n):
            initial_position, _ = self.uniform_reset(type_target_id, self.target_n, ProtectRatio, color='Entity')
            self.targets[type_target_id].reset(initial_position, _)

        # 随机初始化情形
        for agent_id in range(len(self.agents)):
            initial_position, initial_velocity = self.random_reset(self.agents[agent_id].Color, evaluate=evaluate)
            self.agents[agent_id].reset(initial_position, initial_velocity)

    def env_uniform_reset(self):
        # 均匀初始化情形；初始化的顺序和World中create_agents的顺序一致
        # 首先初始化目标点位置
        for type_target_id in range(self.target_n):
            initial_position, _ = self.uniform_reset(type_target_id, self.target_n, ProtectRatio, color='Entity')
            self.targets[type_target_id].reset(initial_position, _)

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

    def step(self, action_n):
        # 更新世界状态
        flying_action_n = [(np.array(action_n[i]) * self.agents[i].aMax).tolist() for i in range(len(action_n))]
        super().step(flying_action_n)

    def render(self):
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

        agent_info_list = targets_info_n, red_agents_info_n, blue_agents_info_n

        # 可视化设置与接口调用
        pygame.init()
        pygame.font.init()
        screen = pygame.display.set_mode((ScreenLength, ScreenWidth + ScreenHeight))
        all_agent = DisplayPlayer(screen)
        screen.fill(SurfaceColor)

        all_agent.update(agent_info_list, self.is_terminal())
        pygame.display.update()

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
        return [agent_x, agent_y] + [0.0] * (EnvDim - 2), initial_velocity

    def random_reset(self, color, evaluate=False):
        if color == 'Red':
            area_x_high = 0  # self.area[0][0] / 4
            random_x = np.random.uniform(low=self.area[0][0], high=area_x_high)
            random_y = np.random.uniform(low=self.area[1][0], high=self.area[1][1])
        else:
            area_x_low = self.area[0][0] if not evaluate else 0 #self.area[0][1] / 4
            random_x = np.random.uniform(low=area_x_low, high=self.area[0][1])
            random_y = np.random.uniform(low=self.area[1][0], high=self.area[1][1])
        initial_velocity = (np.random.uniform(low=vDomain[0], high=vDomain[1], size=EnvDim) / np.sqrt(EnvDim)).tolist()
        return [random_x, random_y] + [0] * (EnvDim - 2), initial_velocity

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
        """
        获取每个智能体的局部观测，包含所有消亡和未消亡的实体信息。
        每个被观测实体的向量末尾追加 is_alive 位（0.0/1.0），该位在归一化之后追加。
        返回: observation_n，每个元素 shape = (n_entities-1, obs_dim)，其中 obs_dim = EnvDim*2+1
        """
        observation_n = []
        for agent in self.agents:
            observation = []
            for observation_agent in self.world:
                if observation_agent.Id == agent.Id and is_relative_observation:
                    continue
                else:
                    observation_agent_status = observation_agent.get_status()
                    agent_status = agent.get_status()

                    if is_relative_observation:
                        relative_observation = np.array(observation_agent_status) - np.array(agent_status)
                        if len(agent_status) > 2 * EnvDim:
                            relative_observation[-1] += agent_status[-1]
                    else:
                        relative_observation = np.array(observation_agent_status)

                    observation.append(relative_observation.tolist())
            observation_n.append(observation)

        # 先归一化位置和速度维度
        if normalization:
            observation_n = normalized_obs_n(observation_n)

        # 在归一化之后追加 is_alive 位（绝对值，不参与归一化）
        world_alive_map = {entity.Id: float(entity.Health > 0) for entity in self.world}
        for agent_idx, agent in enumerate(self.agents):
            entity_idx = 0
            for observation_agent in self.world:
                if observation_agent.Id == agent.Id and is_relative_observation:
                    continue
                alive_bit = world_alive_map[observation_agent.Id]
                observation_n[agent_idx][entity_idx].append(alive_bit)
                entity_idx += 1

        return observation_n

    def get_world_alive(self):
        world_alive = [agent.Health > 0 for agent in self.world]
        return world_alive

    def get_global_state(self, normalization=True):
        """
        返回全局绝对状态，用于 Critic 输入（CTDE 的"集中训练"部分）。
        对死亡实体：位置和速度归零，is_alive=0。
        对存活实体：保留绝对位置和速度，is_alive=1。
        返回: 一维 list，长度 = n_entities * (EnvDim*2 + 1)
        """
        global_state = []
        for entity in self.world:
            is_alive = float(entity.Health > 0)
            if is_alive:
                pos = list(entity.position)
                vel = list(entity.velocity)
            else:
                pos = [0.0] * EnvDim
                vel = [0.0] * EnvDim

            if normalization and is_alive:
                norm_pos = [(pos[d] - AeroPoint[d][0]) / (AeroPoint[d][1] - AeroPoint[d][0]) for d in range(EnvDim)]
                norm_vel = [(vel[d] - vDomain[0]) / (vDomain[1] - vDomain[0]) for d in range(EnvDim)]
                global_state.extend(norm_pos + norm_vel + [is_alive])
            else:
                global_state.extend(pos + vel + [is_alive])

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
        episode_reward = reward_episode * self.is_terminal()
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
        is_terminal = bool(np.abs(self.is_terminal()))
        
        # 红方每个个体的 done
        red_done = [is_terminal or (agent.Health <= 0) for agent in self.red_agents]
        # 蓝方每个个体的 done
        blue_done = [is_terminal or (agent.Health <= 0) for agent in self.blue_agents]
        
        return red_done + blue_done

    def get_info(self):
        info = {}
        for agent_id in range(len(self.targets)):
            info['target_%d' % agent_id] = {
                "position": self.targets[agent_id].get_position()
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
        any_entity_die = any(agent.Health < 1e-3 for agent in [one for one in self.world
                                                                 if one.Color == 'Entity'])
        all_agent_die_b = all(agent.Health < 1e-3 for agent in [one for one in self.agents
                                                                if one.Color == 'Blue' and one.Type == 'Attack'])
        if any_entity_die:
            # print('There is a target destroyed. The episode ends and blue agents win!')
            return -1
        elif all_agent_die_b:
            # print('All blue algorithms have been destroyed. The episode ends and red agents win!')
            return 1
        else:
            return 0


if __name__ == '__main__':
    from Agent.agents.agent import RLAgent as Agent
    from common.arguments import get_args

    args = get_args()
    args.red_attack_n = 1
    args.blue_attack_n = 1
    args.target_n = 1

    Agents = []
    for i in range(args.red_attack_n):
        Agents.append(Agent(args, 'Red', 'Attack', i))
    for i in range(args.red_attack_n, args.red_attack_n + args.blue_attack_n):
        Agents.append(Agent(args, 'Blue', 'Attack', i))

    env = Env(0, 0, 1, 0, 0, 1, 1)

    env.reset(random_position=True)
    obs_n, _, _ = env.get_observation()
    for _ in range(1000):
        env.render()
        action_n = []
        for i in range(len(env.agents)):
            action_n.append(Agents[i].choose_action(obs_n[i]))
        obs_n_, reward_n, done_n, info = env.step(action_n)
        obs_n = obs_n_

        import time
        time.sleep(0.01)
        if np.abs(env.is_terminal()):
            print(env.is_terminal())
            env.reset(random_position=True)


# 1. 先修改成定长的，只改变obs的维度，不改变其他任何东西
# 2. 不改变env的前提下，更改obs中死亡智能体的观测为0，并记录mask
