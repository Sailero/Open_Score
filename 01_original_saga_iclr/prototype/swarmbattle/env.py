"""SwarmBattle: 最小可运行的两队集群对抗环境原型 (NumPy 版).

设计要点 (与研究方案 §1.2 对应):
- 二维连续平面, 质点运动学: 动作 = (角速度, 加速度, 是否开火, 目标索引)
- 双方数量 N, M 任意且可在 episode 中途变化 (增援事件 / 减员)
- 部分可观测: 每单位只见视野半径内实体
- 单位由连续属性向量参数化 (速度上限/射程/血量/伤害), 支持异构与属性外推

注: 正式实验将迁移到 JAX 向量化实现; 本文件仅为接口与规则的参考原型,
    用于验证观测/动作接口、脚本对手与 SAGA 网络骨架的联通性.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class UnitType:
    """连续属性向量定义的单位类型 (支持"换型号"外推)."""
    max_speed: float = 1.0
    attack_range: float = 3.0
    max_hp: float = 1.0
    damage: float = 0.34
    cooldown: int = 3          # 开火间隔 (steps)
    sight_range: float = 8.0

    def as_vector(self) -> np.ndarray:
        return np.array([self.max_speed, self.attack_range, self.max_hp,
                         self.damage, self.cooldown, self.sight_range], dtype=np.float32)


DEFAULT_TYPE = UnitType()
# "重载远程打击机": 慢、远射程、厚血、高伤 —— 属性外推评测用
HEAVY_TYPE = UnitType(max_speed=0.6, attack_range=5.0, max_hp=2.0, damage=0.6,
                      cooldown=5, sight_range=10.0)


@dataclass
class Config:
    arena_size: float = 40.0
    max_steps: int = 400
    dt: float = 1.0
    max_turn: float = 0.5          # 最大角速度 (rad/step)
    max_accel: float = 0.3
    reinforce_rate: float = 0.0    # 每步触发增援事件的概率 (泊松近似)
    reinforce_size: tuple = (2, 6) # 每次增援的单位数范围
    seed: int = 0


# 实体特征布局 (观测中每个实体 token 的维度)
#  [rel_x, rel_y, rel_vx, rel_vy, cos(heading), sin(heading), hp_frac,
#   is_ally, is_enemy, alive, *type_vector(6)] -> 16 维
ENTITY_DIM = 16


class Team:
    """一方兵力的结构化数组容器, 支持动态增删."""

    def __init__(self, positions, headings, types):
        n = len(positions)
        self.pos = np.asarray(positions, dtype=np.float32)
        self.vel = np.zeros((n, 2), dtype=np.float32)
        self.heading = np.asarray(headings, dtype=np.float32)
        self.types = list(types)
        self.type_mat = np.stack([t.as_vector() for t in types]) if n else np.zeros((0, 6), np.float32)
        self.hp = np.array([t.max_hp for t in types], dtype=np.float32)
        self.cool = np.zeros(n, dtype=np.int32)
        self.alive = np.ones(n, dtype=bool)

    @property
    def n(self):
        return len(self.pos)

    @property
    def n_alive(self):
        return int(self.alive.sum())

    def add_units(self, positions, headings, types):
        other = Team(positions, headings, types)
        self.pos = np.concatenate([self.pos, other.pos])
        self.vel = np.concatenate([self.vel, other.vel])
        self.heading = np.concatenate([self.heading, other.heading])
        self.types += other.types
        self.type_mat = np.concatenate([self.type_mat, other.type_mat])
        self.hp = np.concatenate([self.hp, other.hp])
        self.cool = np.concatenate([self.cool, other.cool])
        self.alive = np.concatenate([self.alive, other.alive])


class SwarmBattleEnv:
    """两队对抗环境. step 接收双方动作, 返回双方观测.

    动作格式 (每队): dict(
        turn:   (n,)  in [-1, 1]   -> 乘 max_turn
        accel:  (n,)  in [-1, 1]   -> 乘 max_accel
        fire:   (n,)  bool
        target: (n,)  int, 指向对方 *存活* 单位的全局索引 (无效则不开火)
    )
    """

    def __init__(self, cfg: Config = None):
        self.cfg = cfg or Config()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.red: Team = None
        self.blue: Team = None
        self.t = 0

    # ---------------- lifecycle ----------------
    def reset(self, n_red=8, n_blue=8, red_types=None, blue_types=None):
        cfg = self.cfg
        self.t = 0
        half = cfg.arena_size / 2

        def spawn(n, x_lo, x_hi, types):
            pos = np.stack([self.rng.uniform(x_lo, x_hi, n),
                            self.rng.uniform(-half * 0.8, half * 0.8, n)], axis=1)
            heading = self.rng.uniform(-np.pi, np.pi, n)
            types = types or [DEFAULT_TYPE] * n
            return Team(pos, heading, types)

        self.red = spawn(n_red, -half * 0.9, -half * 0.4, red_types)
        self.blue = spawn(n_blue, half * 0.4, half * 0.9, blue_types)
        return self._observations()

    def step(self, red_action, blue_action):
        cfg = self.cfg
        self.t += 1
        self._move(self.red, red_action)
        self._move(self.blue, blue_action)
        # 同时结算开火: 双方基于本帧移动后的同一状态快照选择与命中,
        # 阵亡在两侧都结算完后统一生效 —— 消除先手优势
        # (镜像对局实测: 顺序结算时先手方胜率 18/20, 修复后应回归 ~50%)
        red_alive_snap = self.red.alive.copy()
        blue_alive_snap = self.blue.alive.copy()
        r_dmg = self._resolve_fire(self.red, self.blue, red_action,
                                   shooter_alive=red_alive_snap)
        b_dmg = self._resolve_fire(self.blue, self.red, blue_action,
                                   shooter_alive=blue_alive_snap)
        self.red.alive &= self.red.hp > 0
        self.blue.alive &= self.blue.hp > 0
        self._maybe_reinforce()

        done = (self.red.n_alive == 0 or self.blue.n_alive == 0
                or self.t >= cfg.max_steps)
        # 零和塑形奖励: 造成伤害差 + 终局歼灭奖励
        reward_red = r_dmg - b_dmg
        if done:
            reward_red += 5.0 * np.sign(self.red.n_alive - self.blue.n_alive)
        obs = self._observations()
        info = dict(t=self.t, n_red=self.red.n_alive, n_blue=self.blue.n_alive)
        return obs, (reward_red, -reward_red), done, info

    # ---------------- dynamics ----------------
    def _move(self, team: Team, action):
        cfg = self.cfg
        alive = team.alive
        turn = np.clip(action["turn"], -1, 1) * cfg.max_turn
        accel = np.clip(action["accel"], -1, 1) * cfg.max_accel
        team.heading[alive] = (team.heading[alive] + turn[alive] + np.pi) % (2 * np.pi) - np.pi
        dir_vec = np.stack([np.cos(team.heading), np.sin(team.heading)], axis=1)
        team.vel[alive] += accel[alive, None] * dir_vec[alive]
        speed = np.linalg.norm(team.vel, axis=1, keepdims=True) + 1e-8
        cap = team.type_mat[:, 0:1]
        team.vel = np.where(speed > cap, team.vel / speed * cap, team.vel)
        team.pos[alive] += team.vel[alive] * cfg.dt
        half = cfg.arena_size / 2
        team.pos = np.clip(team.pos, -half, half)
        team.cool = np.maximum(team.cool - 1, 0)

    def _resolve_fire(self, shooter: Team, target_team: Team, action,
                      shooter_alive=None) -> float:
        """结算 shooter 方开火. 只扣 hp 不置 alive=False (由 step 统一生效,
        保证双方同时结算的对称性)."""
        total = 0.0
        fire = np.asarray(action["fire"], dtype=bool)
        targets = np.asarray(action["target"], dtype=int)
        alive = shooter_alive if shooter_alive is not None else shooter.alive
        for i in np.nonzero(alive & fire & (shooter.cool == 0))[0]:
            j = targets[i]
            if j < 0 or j >= target_team.n or not target_team.alive[j]:
                continue
            dist = np.linalg.norm(shooter.pos[i] - target_team.pos[j])
            if dist <= shooter.type_mat[i, 1]:  # attack_range
                dmg = shooter.type_mat[i, 3]
                target_team.hp[j] -= dmg
                total += dmg
                shooter.cool[i] = int(shooter.type_mat[i, 4])
        return total

    def _maybe_reinforce(self):
        cfg = self.cfg
        if cfg.reinforce_rate <= 0 or self.rng.random() >= cfg.reinforce_rate:
            return
        team = self.red if self.rng.random() < 0.5 else self.blue
        k = int(self.rng.integers(*cfg.reinforce_size))
        half = cfg.arena_size / 2
        side = -1 if team is self.red else 1
        pos = np.stack([np.full(k, side * half * 0.95),
                        self.rng.uniform(-half * 0.8, half * 0.8, k)], axis=1)
        team.add_units(pos, self.rng.uniform(-np.pi, np.pi, k), [DEFAULT_TYPE] * k)

    # ---------------- observations ----------------
    def _entity_tokens(self, observer_pos, observer_vel, sight, own: Team, opp: Team):
        """返回 (tokens, mask): 观察者视野内所有实体的变长 token 集合."""
        tokens, masks = [], []
        for team, is_ally in ((own, 1.0), (opp, 0.0)):
            rel = team.pos - observer_pos
            dist = np.linalg.norm(rel, axis=1)
            visible = team.alive & (dist <= sight)
            feat = np.concatenate([
                rel / max(self.cfg.arena_size, 1e-6),
                (team.vel - observer_vel),
                np.stack([np.cos(team.heading), np.sin(team.heading)], axis=1),
                (team.hp / np.maximum(team.type_mat[:, 2], 1e-6))[:, None],
                np.full((team.n, 1), is_ally, np.float32),
                np.full((team.n, 1), 1.0 - is_ally, np.float32),
                team.alive[:, None].astype(np.float32),
                team.type_mat,
            ], axis=1).astype(np.float32)
            tokens.append(feat)
            masks.append(visible)
        return np.concatenate(tokens), np.concatenate(masks)

    def _observations(self):
        """返回 dict(red=..., blue=...), 每方为变长实体 token 列表.

        obs[side]["tokens"][i]: (n_entities, ENTITY_DIM) 第 i 个存活单位的视野内实体
        置换不变、数量无关 —— 与 SAGA 编码器接口直接对齐.
        """
        out = {}
        for side, own, opp in (("red", self.red, self.blue), ("blue", self.blue, self.red)):
            per_unit, idxs = [], []
            for i in np.nonzero(own.alive)[0]:
                toks, mask = self._entity_tokens(own.pos[i], own.vel[i],
                                                own.type_mat[i, 5], own, opp)
                per_unit.append(toks[mask])
                idxs.append(i)
            out[side] = dict(tokens=per_unit, unit_indices=np.array(idxs, dtype=int))
        return out
