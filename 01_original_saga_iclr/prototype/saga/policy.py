"""模块 C: 目标条件化底层交战策略 (共享参数, 每步执行) + 置换不变 critic."""
from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import EntityFieldEncoder


class MicroPolicy(nn.Module):
    """输入 = 局部实体编码 + 任务向量 g_i, 输出 = 机动 (连续) + 开火/目标 (离散).

    目标选择通过"对每个可见敌实体打分"实现 -> 动作空间也数量无关.
    """

    def __init__(self, entity_dim: int = 16, d_model: int = 64, d_goal: int = 64):
        super().__init__()
        self.encoder = EntityFieldEncoder(entity_dim, d_model)
        self.trunk = nn.Sequential(
            nn.Linear(d_model + d_goal, d_model), nn.ReLU(), nn.Linear(d_model, d_model), nn.ReLU())
        self.maneuver_head = nn.Linear(d_model, 4)      # (turn_mu, turn_logstd, acc_mu, acc_logstd)
        self.fire_head = nn.Linear(d_model, 1)
        self.target_key = nn.Linear(entity_dim, d_model)
        self.target_query = nn.Linear(d_model, d_model)
        self.d_model = d_model

    def forward(self, entity_tokens, entity_mask, enemy_tokens, enemy_mask, goal):
        """entity_tokens: (B, n_ent, D) 单位视野实体; enemy_tokens: (B, n_enemy, D);
        goal: (B, d_goal). 返回动作分布参数与目标 logits (数量无关)."""
        h = self.encoder(entity_tokens, entity_mask)          # (B, d)
        h = self.trunk(torch.cat([h, goal], dim=-1))
        maneuver = self.maneuver_head(h)
        fire_logit = self.fire_head(h).squeeze(-1)
        q = self.target_query(h).unsqueeze(1)                 # (B, 1, d)
        k = self.target_key(enemy_tokens)                     # (B, n_enemy, d)
        target_logits = (q * k).sum(-1) / self.d_model ** 0.5
        target_logits = target_logits.masked_fill(~enemy_mask, -1e9)
        return dict(maneuver=maneuver, fire_logit=fire_logit, target_logits=target_logits)


class SetCritic(nn.Module):
    """置换不变全局 critic (CTDE 训练用): Deep Sets 池化全体实体."""

    def __init__(self, entity_dim: int = 16, d_model: int = 64):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(entity_dim, d_model), nn.ReLU(),
                                 nn.Linear(d_model, d_model))
        self.rho = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))

    def forward(self, all_tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.phi(all_tokens)
        m = mask.float().unsqueeze(-1)
        pooled = (h * m).sum(-2) / m.sum(-2).clamp(min=1)
        return self.rho(pooled).squeeze(-1)
