"""参数化脚本对手库 (研究方案 §3.4 的脚本成分原型).

每个脚本 = 一个从 (env, side) 到动作 dict 的策略函数, 带连续风格参数,
参数随机化可产生从"笨"到"有章法"的连续对手谱系.
正式版本约 12 个脚本; 原型先实现 4 个代表性风格.
"""
from __future__ import annotations

import numpy as np


def _teams(env, side):
    return (env.red, env.blue) if side == "red" else (env.blue, env.red)


def _steer_towards(team, desired_vec, max_turn_frac=1.0):
    """把期望方向向量转换为 (turn, accel) 动作."""
    desired_angle = np.arctan2(desired_vec[:, 1], desired_vec[:, 0])
    diff = (desired_angle - team.heading + np.pi) % (2 * np.pi) - np.pi
    turn = np.clip(diff / 0.5, -1, 1) * max_turn_frac
    accel = np.ones(team.n, dtype=np.float32)
    return turn.astype(np.float32), accel


def _nearest_enemy(own, opp):
    """返回每个己方单位最近存活敌人的索引与距离 (无敌人时索引 -1)."""
    if opp.n_alive == 0:
        return np.full(own.n, -1), np.full(own.n, np.inf)
    alive_idx = np.nonzero(opp.alive)[0]
    d = np.linalg.norm(own.pos[:, None, :] - opp.pos[None, alive_idx, :], axis=-1)
    j = np.argmin(d, axis=1)
    return alive_idx[j], d[np.arange(own.n), j]


def greedy_chase(env, side, aggression=1.0):
    """贪心追击: 每个单位冲向最近敌人并开火. aggression 越低越犹豫."""
    own, opp = _teams(env, side)
    tgt, dist = _nearest_enemy(own, opp)
    vec = np.where(tgt[:, None] >= 0, opp.pos[np.maximum(tgt, 0)] - own.pos, 0.0)
    turn, accel = _steer_towards(own, vec + 1e-6)
    fire = (dist <= own.type_mat[:, 1]) & (np.random.random(own.n) < aggression)
    return dict(turn=turn, accel=accel * aggression, fire=fire, target=tgt)


def kite(env, side, keep_frac=0.85):
    """放风筝: 保持在射程边缘, 敌近则退, 敌远则进."""
    own, opp = _teams(env, side)
    tgt, dist = _nearest_enemy(own, opp)
    desired = own.type_mat[:, 1] * keep_frac
    to_enemy = np.where(tgt[:, None] >= 0, opp.pos[np.maximum(tgt, 0)] - own.pos, 0.0)
    vec = np.where((dist < desired)[:, None], -to_enemy, to_enemy)
    turn, accel = _steer_towards(own, vec + 1e-6)
    fire = dist <= own.type_mat[:, 1]
    return dict(turn=turn, accel=accel, fire=fire, target=tgt)


def focus_fire(env, side, focus_lowest_hp=True):
    """集火: 全队攻击同一个目标 (最低血量或最近质心的敌人)."""
    own, opp = _teams(env, side)
    if opp.n_alive == 0:
        z = np.zeros(own.n, np.float32)
        return dict(turn=z, accel=z, fire=np.zeros(own.n, bool), target=np.full(own.n, -1))
    alive_idx = np.nonzero(opp.alive)[0]
    if focus_lowest_hp:
        j = alive_idx[np.argmin(opp.hp[alive_idx])]
    else:
        centroid = own.pos[own.alive].mean(axis=0)
        j = alive_idx[np.argmin(np.linalg.norm(opp.pos[alive_idx] - centroid, axis=1))]
    vec = opp.pos[j] - own.pos
    dist = np.linalg.norm(vec, axis=1)
    turn, accel = _steer_towards(own, vec)
    fire = dist <= own.type_mat[:, 1]
    return dict(turn=turn, accel=accel, fire=fire, target=np.full(own.n, j))


def coward(env, side, panic_range=6.0):
    """消极逃避: 远离最近敌人 (极端退化风格, 用于测试分簇层的过度反应)."""
    own, opp = _teams(env, side)
    tgt, dist = _nearest_enemy(own, opp)
    to_enemy = np.where(tgt[:, None] >= 0, opp.pos[np.maximum(tgt, 0)] - own.pos, 0.0)
    vec = -to_enemy
    turn, accel = _steer_towards(own, vec + 1e-6)
    accel = accel * np.clip(panic_range / (dist + 1e-6), 0.3, 1.0).astype(np.float32)
    fire = dist <= own.type_mat[:, 1] * 0.9
    return dict(turn=turn, accel=accel, fire=fire, target=tgt)


def focus_greedy(env, side, regroup_dist=6.0):
    """强脚本 (Heuristic-strong 基线): 抱团推进 + 射程内机会主义集火.

    每单位: 机动 = 冲向 (最近敌 + 队伍质心聚拢项) 的合成方向 (保持队形避免被各个击破);
    开火 = 自身射程内血量最低的敌人 (射程内无敌则锁定最近敌).
    注: 同构单位设定下风筝无优势 (同速同射程), 强脚本的正确构成是队形 + 集火."""
    own, opp = _teams(env, side)
    if opp.n_alive == 0:
        z = np.zeros(own.n, np.float32)
        return dict(turn=z, accel=z, fire=np.zeros(own.n, bool), target=np.full(own.n, -1))
    alive_idx = np.nonzero(opp.alive)[0]
    d_all = np.linalg.norm(own.pos[:, None, :] - opp.pos[None, alive_idx, :], axis=-1)
    in_range = d_all <= own.type_mat[:, 1:2]             # (n_own, n_alive_opp)
    hp_masked = np.where(in_range, opp.hp[alive_idx][None, :], np.inf)
    tgt_local = np.where(in_range.any(1),
                         np.argmin(hp_masked, axis=1), np.argmin(d_all, axis=1))
    tgt = alive_idx[tgt_local]
    # 机动: 冲最近敌 + 聚拢 (离质心远于 regroup_dist 时向质心分量加权)
    nearest = alive_idx[np.argmin(d_all, axis=1)]
    to_enemy = opp.pos[nearest] - own.pos
    centroid = own.pos[own.alive].mean(axis=0) if own.n_alive else np.zeros(2)
    to_center = centroid - own.pos
    far = (np.linalg.norm(to_center, axis=1, keepdims=True) > regroup_dist)
    vec = to_enemy / (np.linalg.norm(to_enemy, axis=1, keepdims=True) + 1e-6) \
        + 0.8 * far * to_center / (np.linalg.norm(to_center, axis=1, keepdims=True) + 1e-6)
    turn, accel = _steer_towards(own, vec + 1e-6)
    fire = d_all[np.arange(own.n), tgt_local] <= own.type_mat[:, 1]
    return dict(turn=turn, accel=accel, fire=fire, target=tgt)


SCRIPT_LIBRARY = {
    "greedy_chase": greedy_chase,
    "kite": kite,
    "focus_fire": focus_fire,
    "coward": coward,
    "focus_greedy": focus_greedy,   # Heuristic-strong 基线
}


def sample_script(rng: np.random.Generator):
    """随机采样 (脚本, 参数) 组合, 供 SxS 课程的风格轴使用."""
    name = rng.choice(list(SCRIPT_LIBRARY))
    params = {
        "greedy_chase": dict(aggression=float(rng.uniform(0.3, 1.0))),
        "kite": dict(keep_frac=float(rng.uniform(0.6, 0.95))),
        "focus_fire": dict(focus_lowest_hp=bool(rng.random() < 0.5)),
        "coward": dict(panic_range=float(rng.uniform(3.0, 10.0))),
        "focus_greedy": dict(regroup_dist=float(rng.uniform(4.0, 9.0))),
    }[name]
    fn = SCRIPT_LIBRARY[name]
    return name, params, (lambda env, side, _fn=fn, _p=params: _fn(env, side, **_p))
