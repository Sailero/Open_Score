"""SAGA 智能体: 模块 A+B+C 组装为适配层统一接口上的 actor-critic.

动作接口 (与 envs/adapters.py 对齐, 三个环境共用同一网络):
- move:   每单位离散机动 bin (各环境 bin 数不同, 用 mask 处理)
- target: 每单位开火目标 = 对"可见敌实体逐个打分" + 1 个不开火选项
          -> 动作空间维度随敌数量变化, 网络参数不变 (数量无关)

价值函数: 队级 (团队奖励), 由单位嵌入的置换不变池化给出.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .encoder import EntityFieldEncoder
from .commander import Commander

ENTITY_DIM = 16
STAT_DIM = 8
MAX_MOVE_BINS = 16


def pad_stack(token_list, entity_dim=ENTITY_DIM):
    """变长 token 列表 -> (U, n_max, D) 与 mask (U, n_max)."""
    U = len(token_list)
    n_max = max(1, max((t.shape[0] for t in token_list), default=1))
    out = np.zeros((U, n_max, entity_dim), np.float32)
    mask = np.zeros((U, n_max), bool)
    for i, t in enumerate(token_list):
        if t.shape[0]:
            out[i, : t.shape[0]] = t
            mask[i, : t.shape[0]] = True
    return out, mask


def team_stats(enemy_toks: np.ndarray, enemy_mask: np.ndarray) -> np.ndarray:
    """队级交战统计 (STAT_DIM 维), 供对手风格编码器积累上下文."""
    m = enemy_mask.astype(bool)
    if not m.any():
        return np.zeros(STAT_DIM, np.float32)
    e = enemy_toks[m]
    rel = e[:, 0:2]
    d = np.linalg.norm(rel, axis=1)
    return np.array([
        m.sum() / 32.0,                # 敌数量 (归一)
        d.mean(), d.min(), d.std(),    # 距离统计 (压上程度)
        rel.std(axis=0).mean(),        # 空间散布 (集中/分散)
        e[:, 6].mean(),                # 平均血量
        e[:, 2:4].std(axis=0).mean(),  # 速度散布
        1.0,
    ], np.float32)


class SAGAAgent(nn.Module):
    """commander_mode: "dynamic"=完整SAGA | "fixed_k"=固定K基线 | "none"=SetPolicy-flat基线."""

    def __init__(self, d_model=64, d_style=32, k_max=6, use_commander=True,
                 commander_mode="dynamic"):
        super().__init__()
        if not use_commander:
            commander_mode = "none"
        self.use_commander = commander_mode != "none"
        self.commander_mode = commander_mode
        self.encoder = EntityFieldEncoder(ENTITY_DIM, d_model)
        self.commander = Commander(ENTITY_DIM, d_model, d_style, STAT_DIM, k_max,
                                   dynamic=(commander_mode == "dynamic"))
        goal_dim = d_model if self.use_commander else 0
        self.trunk = nn.Sequential(
            nn.Linear(d_model + goal_dim, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.ReLU())
        self.move_head = nn.Linear(d_model, MAX_MOVE_BINS)
        self.no_fire_head = nn.Linear(d_model, 1)
        self.target_key = nn.Linear(ENTITY_DIM, d_model)
        self.target_query = nn.Linear(d_model, d_model)
        self.value_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(),
                                        nn.Linear(d_model, 1))
        self.d_model = d_model

    def forward(self, tokens, mask, enemy_toks, enemy_mask, stats_seq, n_move_bins):
        """tokens: (U, n_ent, D), enemy_toks: (n_enemy, D), stats_seq: (T, STAT_DIM).

        返回 move/target 的 Categorical 分布与队级 value.
        """
        unit_emb = self.encoder(tokens, mask)                      # (U, d)
        if self.use_commander:
            cmd = self.commander(unit_emb.unsqueeze(0), enemy_toks.unsqueeze(0),
                                 enemy_mask.unsqueeze(0), stats_seq.unsqueeze(0))
            goal = cmd["goal"].squeeze(0)                          # (U, d)
            h = self.trunk(torch.cat([unit_emb, goal], dim=-1))
        else:
            cmd = None
            h = self.trunk(unit_emb)

        move_logits = self.move_head(h)[:, :n_move_bins]
        q = self.target_query(h)                                   # (U, d)
        k = self.target_key(enemy_toks)                            # (n_enemy, d)
        tgt_logits = q @ k.t() / self.d_model ** 0.5               # (U, n_enemy)
        tgt_logits = tgt_logits.masked_fill(~enemy_mask.unsqueeze(0), -1e9)
        # index 0 = 不开火, index j+1 = 攻击敌实体 j
        tgt_logits = torch.cat([self.no_fire_head(h), tgt_logits], dim=-1)
        value = self.value_head(unit_emb.mean(dim=0, keepdim=True)).squeeze()
        return Categorical(logits=move_logits), Categorical(logits=tgt_logits), value, cmd

    @torch.no_grad()
    def act(self, obs_pack, device, greedy=False):
        """obs_pack = prepare_obs(...) 的输出. 返回动作与 log_prob/value."""
        t = {k: torch.as_tensor(v, device=device) for k, v in obs_pack.items()
             if k != "n_move_bins"}
        dist_m, dist_t, value, _ = self(t["tokens"], t["mask"], t["enemy_toks"],
                                        t["enemy_mask"], t["stats_seq"],
                                        obs_pack["n_move_bins"])
        move = dist_m.probs.argmax(-1) if greedy else dist_m.sample()
        tgt = dist_t.probs.argmax(-1) if greedy else dist_t.sample()
        logp = dist_m.log_prob(move).sum() + dist_t.log_prob(tgt).sum()
        return (move.cpu().numpy(), tgt.cpu().numpy() - 1,  # -1 = 不开火
                float(logp), float(value))


def prepare_obs(obs, env_adapter, stats_buf):
    """把适配层观测打包为网络输入 (numpy)."""
    tokens, mask = pad_stack(obs["tokens"])
    enemy_toks, enemy_mask = env_adapter.enemy_tokens()
    stats_buf.append(team_stats(enemy_toks, enemy_mask))
    T = 8
    seq = np.stack(list(stats_buf)[-T:])
    if len(seq) < T:
        seq = np.concatenate([np.zeros((T - len(seq), STAT_DIM), np.float32), seq])
    return dict(tokens=tokens.astype(np.float32), mask=mask,
                enemy_toks=enemy_toks.astype(np.float32),
                enemy_mask=enemy_mask.astype(bool),
                stats_seq=seq.astype(np.float32),
                n_move_bins=env_adapter.n_move_bins)
