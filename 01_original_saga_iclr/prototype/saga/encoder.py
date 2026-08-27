"""模块 A: 实体场编码器 (置换不变, 数量无关).

SAQA 式交叉注意力: 每个决策单位只作为 query, 视野内实体作为 key/value,
复杂度 O(n_entities * d), 参数量与双方数量完全解耦.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EntityFieldEncoder(nn.Module):
    def __init__(self, entity_dim: int = 16, d_model: int = 64, n_heads: int = 4):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(entity_dim, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )
        self.self_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.ReLU())

    def forward(self, entity_tokens: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """entity_tokens: (B, n_entities, entity_dim), mask: (B, n_entities) True=有效.

        返回 (B, d_model): 每个决策单位的态势嵌入. B 可以是"所有存活单位"展平后的批,
        n_entities 每批可不同 (padding + mask), 网络结构与数量无关.
        """
        kv = self.embed(entity_tokens)
        q = self.self_token.expand(kv.shape[0], 1, -1)
        key_padding_mask = ~mask if mask is not None else None
        h, _ = self.cross_attn(q, kv, kv, key_padding_mask=key_padding_mask)
        return self.out(h.squeeze(1))


class MeanFieldSummary(nn.Module):
    """队级平均场摘要: 双方实体特征的置换不变池化, 作为廉价全局上下文."""

    def __init__(self, entity_dim: int = 16, d_model: int = 64):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(entity_dim, d_model), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.ReLU())

    def forward(self, all_tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """all_tokens: (n_entities, entity_dim) 全局实体集合 -> (d_model,)"""
        h = self.phi(all_tokens)
        m = mask.float().unsqueeze(-1)
        mean_pool = (h * m).sum(0) / m.sum().clamp(min=1)
        max_pool = (h.masked_fill(~mask.unsqueeze(-1), -1e9)).max(0).values
        return self.rho(torch.cat([mean_pool, max_pool], dim=-1))
