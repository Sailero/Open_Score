"""统一环境适配层: 把不同环境全部转成 SAGA 的"实体 token"接口.

支持三个环境 (均已在本机 gpu_py_310 验证):
- swarmbattle:  自研两队对抗环境 (变规模/中途增援, 连续动作)
- mpe_tag:      PettingZoo MPE simple_tag (公认基准, 3 追捕者 vs 1 逃跑者, 离散动作)
- magent_battle: MAgent2 battle_v4 (公认大规模两队对抗基准, 离散动作)

统一接口 (TeamBattleAdapter):
    reset() -> obs                obs = dict(tokens=[(n_i, ENTITY_DIM)...], unit_ids=[...])
    step(actions) -> obs, team_reward, done, info
    actions: dict(move=(n_units,) int 离散移动/机动 bin, target=(n_units,) int 开火目标实体索引, 其中 -1 = 不开火)
    n_move_bins: 离散机动动作数 (各环境不同, 网络侧用同一个头 + mask)

设计意图: 论文中的"实体场编码器"只依赖此 token 接口, 因此同一网络
(同一组参数) 可以在三个环境间共享架构, 这正是数量无关/环境无关的卖点.
"""
from __future__ import annotations

import numpy as np

ENTITY_DIM = 16  # 与 swarmbattle.env.ENTITY_DIM 一致


def _token(rel_pos, rel_vel=(0, 0), heading=None, hp=1.0, is_ally=0.0,
           alive=1.0, type_vec=None, scale=1.0):
    """构造统一实体 token: 16 维布局与 SwarmBattle 一致."""
    cos_h, sin_h = (np.cos(heading), np.sin(heading)) if heading is not None else (0.0, 0.0)
    tv = np.zeros(6, np.float32) if type_vec is None else np.asarray(type_vec, np.float32)
    return np.array([rel_pos[0] / scale, rel_pos[1] / scale, rel_vel[0], rel_vel[1],
                     cos_h, sin_h, hp, is_ally, 1.0 - is_ally, alive, *tv], np.float32)


class TeamBattleAdapter:
    """基类: 我方 = 被控队, 敌方 = 脚本/内置对手."""
    n_move_bins: int = 5

    def reset(self, **kwargs):
        raise NotImplementedError

    def step(self, actions):
        raise NotImplementedError

    def enemy_tokens(self):
        """指挥层用: 全体敌实体 token (n_enemy, ENTITY_DIM), 以我方质心为参考系."""
        raise NotImplementedError

    def render_frame(self):
        return None


# --------------------------------------------------------------------------
# 1) SwarmBattle (自研)
# --------------------------------------------------------------------------
class SwarmBattleAdapter(TeamBattleAdapter):
    """我方=红, 敌方=脚本采样. 连续机动离散化为 9 个 bin (3转向 x 3加速)."""
    n_move_bins = 9
    _TURNS = (-1.0, 0.0, 1.0)
    _ACCELS = (0.0, 0.5, 1.0)

    def __init__(self, n_red=6, n_blue=6, reinforce_rate=0.0, opponent="random",
                 opponent_params=None, seed=0):
        import os, sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from swarmbattle.env import SwarmBattleEnv, Config
        from swarmbattle import scripts
        self._scripts = scripts
        self.env = SwarmBattleEnv(Config(seed=seed, reinforce_rate=reinforce_rate,
                                         max_steps=200))
        self.n_red, self.n_blue = n_red, n_blue
        self.opponent = opponent
        self.opponent_params = opponent_params or {}
        self.rng = np.random.default_rng(seed)
        self._blue_policy = None

    def reset(self, n_red=None, n_blue=None):
        self.env.reset(n_red=n_red or self.n_red, n_blue=n_blue or self.n_blue)
        if self.opponent == "random":
            _, _, self._blue_policy = self._scripts.sample_script(self.rng)
        else:
            fn = self._scripts.SCRIPT_LIBRARY[self.opponent]
            params = self.opponent_params
            self._blue_policy = lambda env, side: fn(env, side, **params)
        return self._obs()

    def _obs(self):
        raw = self.env._observations()["red"]
        # 敌实体全局 token (供开火目标索引对齐): 按 blue 全局索引排列
        blue = self.env.blue
        return dict(tokens=raw["tokens"], unit_ids=raw["unit_indices"],
                    enemy_alive=blue.alive.copy())

    def step(self, actions):
        red = self.env.red
        n = red.n
        turn = np.zeros(n, np.float32)
        accel = np.zeros(n, np.float32)
        fire = np.zeros(n, bool)
        target = np.full(n, -1, int)
        ids = np.nonzero(red.alive)[0]
        for k, i in enumerate(ids):
            m = int(actions["move"][k])
            turn[i] = self._TURNS[m // 3]
            accel[i] = self._ACCELS[m % 3]
            t = int(actions["target"][k])
            if t >= 0:
                fire[i], target[i] = True, t
        red_action = dict(turn=turn, accel=accel, fire=fire, target=target)
        blue_action = self._blue_policy(self.env, "blue")
        _, (r, _), done, info = self.env.step(red_action, blue_action)
        info["win"] = done and self.env.red.n_alive > self.env.blue.n_alive
        return self._obs(), float(r), done, info

    def enemy_tokens(self):
        red, blue = self.env.red, self.env.blue
        center = red.pos[red.alive].mean(axis=0) if red.n_alive else np.zeros(2)
        toks, mask = self.env._entity_tokens(center, np.zeros(2), 1e9, red, blue)
        n_own = red.n
        return toks[n_own:], (mask[n_own:] & blue.alive)

    def snapshot(self):
        """导出当前帧状态 (回放前端用)."""
        def team_state(t):
            return dict(pos=t.pos.round(2).tolist(), alive=t.alive.tolist(),
                        hp=(t.hp / np.maximum(t.type_mat[:, 2], 1e-6)).round(2).tolist(),
                        heading=t.heading.round(2).tolist())
        return dict(red=team_state(self.env.red), blue=team_state(self.env.blue),
                    t=self.env.t, arena=self.env.cfg.arena_size)


# --------------------------------------------------------------------------
# 2) MPE simple_tag (PettingZoo)
# --------------------------------------------------------------------------
class MPETagAdapter(TeamBattleAdapter):
    """我方=3个追捕者(adversary), 敌方=1个逃跑者(启发式逃跑).

    无开火概念 -> target 恒为 -1; 接触即得分 (环境自带 reward).
    离散动作 5: noop/left/right/down/up.
    """
    n_move_bins = 5

    def __init__(self, seed=0, render_mode=None):
        from pettingzoo.mpe import simple_tag_v3
        self.env = simple_tag_v3.parallel_env(num_good=1, num_adversaries=3,
                                              num_obstacles=2, max_cycles=100,
                                              continuous_actions=False,
                                              render_mode=render_mode)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self, **kwargs):
        self.env.reset(seed=int(self.rng.integers(1 << 30)))
        return self._obs()

    def _world(self):
        return self.env.unwrapped.world

    def _obs(self):
        world = self._world()
        preds = [a for a in world.agents if a.adversary]
        prey = [a for a in world.agents if not a.adversary]
        tokens = []
        for me in preds:
            toks = []
            for a in preds:
                if a is me:
                    continue
                toks.append(_token(a.state.p_pos - me.state.p_pos, a.state.p_vel,
                                   is_ally=1.0, type_vec=[a.max_speed or 1.0] + [0] * 5))
            for a in prey:
                toks.append(_token(a.state.p_pos - me.state.p_pos, a.state.p_vel,
                                   is_ally=0.0, type_vec=[a.max_speed or 1.3] + [0] * 5))
            for lm in world.landmarks:
                toks.append(_token(lm.state.p_pos - me.state.p_pos, hp=0.0,
                                   is_ally=0.0, alive=0.0, type_vec=[0, 1, 0, 0, 0, 0]))
            tokens.append(np.stack(toks))
        return dict(tokens=tokens, unit_ids=np.arange(len(preds)),
                    enemy_alive=np.ones(len(prey), bool))

    def enemy_tokens(self):
        world = self._world()
        preds = [a for a in world.agents if a.adversary]
        prey = [a for a in world.agents if not a.adversary]
        center = np.mean([p.state.p_pos for p in preds], axis=0)
        toks = np.stack([_token(a.state.p_pos - center, a.state.p_vel, is_ally=0.0,
                                type_vec=[a.max_speed or 1.3] + [0] * 5) for a in prey])
        return toks, np.ones(len(prey), bool)

    def _prey_heuristic(self):
        """逃跑者: 远离最近追捕者 (含边界斥力)."""
        world = self._world()
        preds = [a for a in world.agents if a.adversary]
        prey = [a for a in world.agents if not a.adversary][0]
        vecs = [prey.state.p_pos - p.state.p_pos for p in preds]
        d = [np.linalg.norm(v) for v in vecs]
        away = vecs[int(np.argmin(d))] - 0.3 * prey.state.p_pos
        if abs(away[0]) > abs(away[1]):
            return 2 if away[0] > 0 else 1   # right / left
        return 4 if away[1] > 0 else 3       # up / down

    def step(self, actions):
        act = {}
        names = [a for a in self.env.agents]
        adv_names = [n for n in names if "adversary" in n]
        good = [n for n in names if "agent" in n]
        for k, name in enumerate(adv_names):
            act[name] = int(actions["move"][k])
        for name in good:
            act[name] = self._prey_heuristic()
        _, rew, term, trunc, _ = self.env.step(act)
        r = sum(rew[n] for n in adv_names)
        done = all(term.values()) or all(trunc.values()) or not self.env.agents
        return self._obs(), float(r), done, dict(win=r > 0)

    def render_frame(self):
        return self.env.render()


# --------------------------------------------------------------------------
# 3) MAgent2 battle_v4
# --------------------------------------------------------------------------
class MAgentBattleAdapter(TeamBattleAdapter):
    """我方=红队, 敌方=蓝队(脚本: 朝最近红者移动/攻击).

    从全局 state 的 presence/hp 通道提取实体列表 -> token.
    离散动作 21: 0-12 移动 (含 noop), 13-20 攻击 8 邻域.
    统一接口映射: move bin 0-12 -> 移动; target>=0 -> 转为攻击最近方向.
    """
    n_move_bins = 13

    def __init__(self, map_size=20, seed=0, render_mode=None):
        from magent2.environments import battle_v4
        self.env = battle_v4.parallel_env(map_size=map_size, max_cycles=200,
                                          attack_penalty=-0.05, attack_opponent_reward=1.0,
                                          render_mode=render_mode)
        self.map_size = map_size
        self.rng = np.random.default_rng(seed)
        self._obs_cache = None

    def reset(self, **kwargs):
        self._raw_obs, _ = self.env.reset(seed=int(self.rng.integers(1 << 30)))
        return self._obs()

    def _positions(self):
        """从全局 state 提取双方位置与血量: 通道 [wall, red, red_hp, blue, blue_hp]."""
        s = self.env.state()  # (H, W, 5)
        red_yx = np.argwhere(s[:, :, 1] > 0)
        blue_yx = np.argwhere(s[:, :, 3] > 0)
        red_hp = s[red_yx[:, 0], red_yx[:, 1], 2] if len(red_yx) else np.zeros(0)
        blue_hp = s[blue_yx[:, 0], blue_yx[:, 1], 4] if len(blue_yx) else np.zeros(0)
        return red_yx[:, ::-1].astype(float), red_hp, blue_yx[:, ::-1].astype(float), blue_hp

    def _obs(self):
        red_pos, red_hp, blue_pos, blue_hp = self._positions()
        tokens = []
        for i in range(len(red_pos)):
            toks = []
            for j in range(len(red_pos)):
                if j == i:
                    continue
                toks.append(_token(red_pos[j] - red_pos[i], hp=float(red_hp[j]),
                                   is_ally=1.0, scale=self.map_size))
            for j in range(len(blue_pos)):
                toks.append(_token(blue_pos[j] - red_pos[i], hp=float(blue_hp[j]),
                                   is_ally=0.0, scale=self.map_size))
            if not toks:
                toks = [np.zeros(ENTITY_DIM, np.float32)]
            tokens.append(np.stack(toks))
        self._red_pos, self._blue_pos = red_pos, blue_pos
        return dict(tokens=tokens, unit_ids=np.arange(len(red_pos)),
                    enemy_alive=np.ones(len(blue_pos), bool))

    def enemy_tokens(self):
        red_pos, _, blue_pos, blue_hp = self._positions()
        center = red_pos.mean(axis=0) if len(red_pos) else np.zeros(2)
        if not len(blue_pos):
            return np.zeros((1, ENTITY_DIM), np.float32), np.zeros(1, bool)
        toks = np.stack([_token(blue_pos[j] - center, hp=float(blue_hp[j]),
                                is_ally=0.0, scale=self.map_size)
                         for j in range(len(blue_pos))])
        return toks, np.ones(len(blue_pos), bool)

    def _attack_action(self, my_pos, target_pos):
        """把"攻击目标实体"翻译为 8 邻域攻击动作 (13-20)."""
        d = target_pos - my_pos
        dx, dy = int(np.sign(round(d[0]))), int(np.sign(round(d[1])))
        offsets = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
        if (dx, dy) in offsets:
            return 13 + offsets.index((dx, dy))
        return 13 + offsets.index((int(np.sign(d[0])) or 1, 0))

    def _blue_script(self):
        acts = {}
        blue_names = [n for n in self.env.agents if n.startswith("blue")]
        for k, name in enumerate(blue_names):
            if k < len(self._blue_pos) and len(self._red_pos):
                me = self._blue_pos[k]
                j = np.argmin(np.linalg.norm(self._red_pos - me, axis=1))
                d = self._red_pos[j] - me
                if np.abs(d).max() <= 1.5:
                    acts[name] = self._attack_action(me, self._red_pos[j])
                else:  # 简化移动: 12 邻域中最接近目标方向的 bin
                    acts[name] = int(self.rng.integers(0, 13))
            else:
                acts[name] = 0
        return acts

    def step(self, actions):
        red_names = [n for n in self.env.agents if n.startswith("red")]
        act = self._blue_script()
        for k, name in enumerate(red_names):
            t = int(actions["target"][k]) if k < len(actions["target"]) else -1
            if 0 <= t < len(self._blue_pos) and k < len(self._red_pos):
                act[name] = self._attack_action(self._red_pos[k], self._blue_pos[t])
            elif k < len(actions["move"]):
                act[name] = int(actions["move"][k]) % 13
            else:
                act[name] = 0
        _, rew, term, trunc, _ = self.env.step(act)
        r = sum(rew.get(n, 0.0) for n in red_names)
        done = not self.env.agents or all(term.values()) or all(trunc.values())
        obs = self._obs()
        n_red, n_blue = len(obs["unit_ids"]), int(obs["enemy_alive"].sum())
        return obs, float(r), done or n_red == 0 or n_blue == 0, dict(win=n_red > n_blue)

    def render_frame(self):
        return self.env.render()


def make_env(name: str, **kwargs) -> TeamBattleAdapter:
    return {"swarmbattle": SwarmBattleAdapter,
            "mpe_tag": MPETagAdapter,
            "magent_battle": MAgentBattleAdapter}[name](**kwargs)
