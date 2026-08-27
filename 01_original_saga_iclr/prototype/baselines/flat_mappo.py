"""基线 1: MAPPO-flat — 固定维度扁平策略 (代表主流做法).

设计要点 (与 paper/05 §5.0 基线 1 对应):
- 每单位观测 = 前 N_MAX 个实体 token 按固定顺序拼接成定长向量 (不足 padding,
  超出截断) -> MLP. 这正是"网络结构与规模绑定"的典型做法;
- 目标选择头输出固定 E_MAX+1 维 logits (含不开火), 敌数超过 E_MAX 时后续敌人
  永远不可选 —— 规模外推失效的机制在架构层面即可见;
- forward/act 接口与 SAGAAgent 完全一致, 训练循环与评测 harness 零改动复用.

公平性协议: 参数量与 SAGA 匹配 (±10%), 同 PPO 预算, 同调参预算 (见 docs/12).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

ENTITY_DIM = 16
N_MAX = 32          # 拼接的最大实体数 (训练域 4-16 内充足; 外推时截断)
E_MAX = 16          # 可选目标的最大敌数
MAX_MOVE_BINS = 16


class FlatMAPPOAgent(nn.Module):
    """固定维度扁平 actor-critic. 接口与 SAGAAgent 对齐."""

    def __init__(self, d_hidden=128):  # d_hidden=128 使参数量与 SAGA 匹配 (~110K, 公平协议)
        super().__init__()
        in_dim = N_MAX * ENTITY_DIM
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, 128), nn.ReLU())
        self.move_head = nn.Linear(128, MAX_MOVE_BINS)
        self.target_head = nn.Linear(128, E_MAX + 1)   # 0 = 不开火
        self.value_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(),
                                        nn.Linear(64, 1))

    @staticmethod
    def _flatten(tokens, mask):
        """(U, n_ent, D) 变长 -> (U, N_MAX*D) 定长: pad/截断 + 无效位清零."""
        U, n, D = tokens.shape
        out = tokens.new_zeros(U, N_MAX, D)
        k = min(n, N_MAX)
        out[:, :k] = tokens[:, :k] * mask[:, :k].unsqueeze(-1)
        return out.reshape(U, N_MAX * D)

    def forward(self, tokens, mask, enemy_toks, enemy_mask, stats_seq, n_move_bins):
        """签名与 SAGAAgent.forward 一致; stats_seq 未使用 (扁平基线无对手建模)."""
        h = self.trunk(self._flatten(tokens, mask))
        move_logits = self.move_head(h)[:, :n_move_bins]
        tgt_logits = self.target_head(h)                       # (U, E_MAX+1)
        n_enemy = enemy_mask.shape[0]
        valid = torch.zeros(E_MAX + 1, dtype=torch.bool, device=tokens.device)
        valid[0] = True                                        # 不开火恒可选
        k = min(n_enemy, E_MAX)
        valid[1:1 + k] = enemy_mask[:k]
        tgt_logits = tgt_logits.masked_fill(~valid.unsqueeze(0), -1e9)
        value = self.value_head(h.mean(dim=0, keepdim=True)).squeeze()
        return (Categorical(logits=move_logits), Categorical(logits=tgt_logits),
                value, None)

    @torch.no_grad()
    def act(self, obs_pack, device, greedy=False):
        t = {k: torch.as_tensor(v, device=device) for k, v in obs_pack.items()
             if k != "n_move_bins"}
        dist_m, dist_t, value, _ = self(t["tokens"], t["mask"], t["enemy_toks"],
                                        t["enemy_mask"], t["stats_seq"],
                                        obs_pack["n_move_bins"])
        move = dist_m.probs.argmax(-1) if greedy else dist_m.sample()
        tgt = dist_t.probs.argmax(-1) if greedy else dist_t.sample()
        logp = dist_m.log_prob(move).sum() + dist_t.log_prob(tgt).sum()
        return (move.cpu().numpy(), tgt.cpu().numpy() - 1,
                float(logp), float(value))
