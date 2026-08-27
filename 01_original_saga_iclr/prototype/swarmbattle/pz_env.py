"""SwarmArena 的 PettingZoo ParallelEnv 标准封装.

开源与可复现性的关键交付物: 自研环境通过社区标准 API 暴露,
任何人可用 PettingZoo 生态工具 (wrappers, 训练库, api test) 直接使用,
并可通过 pettingzoo.test.parallel_api_test 官方合规测试验证接口正确性.

设计说明:
- 双方全部单位均为可控 agent (self-play ready), id 形如 red_0 / blue_3;
- possible_agents 按容量 capacity 预声明 (PettingZoo 规范), 实际激活子集由
  reset(options={"n_red":..., "n_blue":...}) 决定 —— 变规模在标准 API 内表达;
- 观测 = (max_entities, ENTITY_DIM+1) 定长数组, 末列为有效位 (padding mask);
- 动作 = MultiDiscrete([9, capacity+1]): 9 个机动 bin x (目标索引 | capacity=不开火);
- 中途增援在标准 API 下默认关闭 (PettingZoo 不支持已终止 agent 复活);
  增援语义由研究用原生接口 (envs/adapters.py) 提供, 论文评测用原生接口,
  开源发布两种接口并存并在文档中说明差异.
"""
from __future__ import annotations

import functools

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from .env import SwarmBattleEnv, Config, ENTITY_DIM

_TURNS = (-1.0, 0.0, 1.0)
_ACCELS = (0.0, 0.5, 1.0)
N_MOVE_BINS = 9


class SwarmArenaPZEnv(ParallelEnv):
    metadata = {"name": "swarmarena_v0", "render_modes": [], "is_parallelizable": True}

    def __init__(self, n_red=8, n_blue=8, capacity=16, max_steps=200, seed=0):
        assert n_red <= capacity and n_blue <= capacity
        self.capacity = capacity
        self.n_red0, self.n_blue0 = n_red, n_blue
        self.env = SwarmBattleEnv(Config(seed=seed, max_steps=max_steps,
                                         reinforce_rate=0.0))
        self.possible_agents = ([f"red_{i}" for i in range(capacity)]
                                + [f"blue_{i}" for i in range(capacity)])
        self.max_entities = 2 * capacity
        self.agents = []

    # ---- spaces (PettingZoo 要求按 agent 提供且稳定) ----
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Box(-np.inf, np.inf,
                          (self.max_entities, ENTITY_DIM + 1), np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.MultiDiscrete([N_MOVE_BINS, self.capacity + 1])

    # ---- helpers ----
    def _team(self, agent):
        side, idx = agent.split("_")
        return side, int(idx)

    def _live_agents(self):
        out = [f"red_{i}" for i in np.nonzero(self.env.red.alive)[0]]
        out += [f"blue_{i}" for i in np.nonzero(self.env.blue.alive)[0]]
        return out

    def _obs_for(self, agent):
        side, i = self._team(agent)
        own, opp = ((self.env.red, self.env.blue) if side == "red"
                    else (self.env.blue, self.env.red))
        toks, mask = self.env._entity_tokens(own.pos[i], own.vel[i],
                                             own.type_mat[i, 5], own, opp)
        out = np.zeros((self.max_entities, ENTITY_DIM + 1), np.float32)
        k = min(len(toks), self.max_entities)
        out[:k, :ENTITY_DIM] = toks[:k]
        out[:k, ENTITY_DIM] = mask[:k].astype(np.float32)
        return out

    def _all_obs(self):
        return {a: self._obs_for(a) for a in self.agents}

    # ---- ParallelEnv API ----
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.env.rng = np.random.default_rng(seed)
        options = options or {}
        n_red = min(options.get("n_red", self.n_red0), self.capacity)
        n_blue = min(options.get("n_blue", self.n_blue0), self.capacity)
        self.env.reset(n_red=n_red, n_blue=n_blue)
        self.agents = self._live_agents()
        return self._all_obs(), {a: {} for a in self.agents}

    def _build_team_action(self, team, side, actions):
        n = team.n
        act = dict(turn=np.zeros(n, np.float32), accel=np.zeros(n, np.float32),
                   fire=np.zeros(n, bool), target=np.full(n, -1, int))
        for a in self.agents:
            s, i = self._team(a)
            if s != side or a not in actions:
                continue
            move, tgt = int(actions[a][0]), int(actions[a][1])
            act["turn"][i] = _TURNS[move // 3]
            act["accel"][i] = _ACCELS[move % 3]
            if tgt < self.capacity:          # capacity = 不开火
                act["fire"][i] = True
                act["target"][i] = tgt
        return act

    def step(self, actions):
        prev_agents = list(self.agents)
        red_act = self._build_team_action(self.env.red, "red", actions)
        blue_act = self._build_team_action(self.env.blue, "blue", actions)
        _, (r_red, r_blue), done, info = self.env.step(red_act, blue_act)

        annihilated = self.env.red.n_alive == 0 or self.env.blue.n_alive == 0
        truncated = done and not annihilated   # 到时限
        rewards, terms, truncs, infos = {}, {}, {}, {}
        for a in prev_agents:
            side, i = self._team(a)
            team = self.env.red if side == "red" else self.env.blue
            rewards[a] = float(r_red if side == "red" else r_blue)
            dead = not team.alive[i]
            terms[a] = bool(dead or (done and not truncated))
            truncs[a] = bool(truncated and not dead)
            infos[a] = {}
        self.agents = [] if done else self._live_agents()
        obs = {a: self._obs_for(a) for a in prev_agents}
        return obs, rewards, terms, truncs, infos

    def render(self):
        return None

    def close(self):
        pass
