import copy
from HAD_Env.config import *
from HAD_Env.function.Function import distance
from HAD_Env.agents.base import Entity
from HAD_Env.agents.attack import AttackAgent
from HAD_Env.agents.disturb import DisturbAgent
from HAD_Env.agents.scout import ScoutAgent


class World:
    def __init__(self, red_scout_n, red_disturb_n, red_attack_n,
                 blue_scout_n, blue_disturb_n, blue_attack_n,
                 target_n):
        # 读取世界信息
        self.red_scout_n = red_scout_n
        self.red_disturb_n = red_disturb_n
        self.red_attack_n = red_attack_n
        self.blue_scout_n = blue_scout_n
        self.blue_disturb_n = blue_disturb_n
        self.blue_attack_n = blue_attack_n
        self.target_n = target_n

        # 创建智能体（这里的智能体是否跟决策的智能体保持一致，如何保持顺序一致的问题呢？还是说不需要保持顺序一致呢？）
        self.world = self.create_world()
        self.red_agent_n = self.red_disturb_n + self.red_attack_n + self.red_scout_n
        self.blue_agent_n = self.blue_disturb_n + self.blue_attack_n + self.blue_scout_n

        self.agents = [agent for agent in self.world if agent.Type != 'Entity']
        self.targets = [entity for entity in self.world if entity.Type == 'Entity']
        self.red_agents = [agent for agent in self.agents if agent.Color == "Red"]
        self.blue_agents = [agent for agent in self.agents if agent.Color == "Blue"]
        self.update_alive_agents()

    def update_alive_agents(self):
        self.alive_agents = [agent for agent in self.agents if agent.Health > 0]
        self.alive_targets = [target for target in self.targets if target.Health > 0]

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
            # 局部观测维度：相对位置 + 相对速度 + is_alive
            "scout_obs_dim": EnvDim * 2 + 1,
            "disturb_obs_dim": EnvDim * 2 + 1,
            "attack_obs_dim": EnvDim * 2 + 1,
            "scout_action_dim": EnvDim,
            "disturb_action_dim": EnvDim,
            "attack_action_dim": EnvDim,
            # 全局状态维度：每个实体的 [绝对位置 + 绝对速度 + is_alive]
            "global_state_dim": n_entities * (EnvDim * 2 + 1),
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
            world.append(Entity(Id, render_id))
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
        # 碰撞检测
        for i in range(0, len(self.alive_agents)):
            for j in range(0, len(self.alive_agents)):
                if i > j:
                    if distance(self.alive_agents[i].get_position(), self.alive_agents[j].get_position()) \
                            <= AvoidanceDistance:
                        self.alive_agents[i].Health = 0
                        self.alive_agents[j].Health = 0

        # 智能体根据action_n设定动作
        for i, agent in enumerate(self.agents):
            agent.set_flying_action(flying_action_n[i])
            function_action = agent.choose_function_ruled_action(self.world)
            agent.set_function_action(function_action)

        # 更新状态
        agents_copy = copy.deepcopy(self.world)
        for agent in self.world:
            agent.update_status(agents_copy)

        # 更新还在存活的agent列表
        self.update_alive_agents()

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
