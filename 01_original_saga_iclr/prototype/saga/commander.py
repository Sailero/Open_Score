"""模块 B: 对手锚定的动态分簇指挥层 (每 k 步运行). 全模块支持批处理.

三步流水线:
1. 敌情软聚类: slot attention 风格的可微聚类, 簇数自适应 (空 slot 退化)
2. 对手风格嵌入 z_opp: in-context GRU 编码最近 T 步队级交战统计
3. 兵力-任务匹配: 己方单位嵌入 x 任务 slot 的交叉注意力软分配

批处理约定: B = 并行的"队级决策时刻"数 (跨时间步/跨环境实例展平).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnemySoftClustering(nn.Module):
    """slot attention 简化版: K_max 个可学习 slot 竞争解释敌实体.

    实际激活簇数由数据决定: 无实体分配的 slot 权重自动接近 0.
    """

    def __init__(self, entity_dim: int = 16, d_model: int = 64, k_max: int = 6, n_iter: int = 3):
        super().__init__()
        self.k_max, self.n_iter = k_max, n_iter
        self.slots_init = nn.Parameter(torch.randn(1, k_max, d_model) * 0.1)
        self.embed = nn.Linear(entity_dim, d_model)
        self.to_q = nn.Linear(d_model, d_model)
        self.to_k = nn.Linear(d_model, d_model)
        self.to_v = nn.Linear(d_model, d_model)
        self.gru = nn.GRUCell(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, enemy_tokens: torch.Tensor, mask: torch.Tensor):
        """enemy_tokens: (B, n_enemy, entity_dim), mask: (B, n_enemy) True=有效.

        返回 slots: (B, K, d) 敌簇描述子, assign: (B, n_enemy, K) 软分配,
        occupancy: (B, K) 每簇分到的实体质量.
        """
        B = enemy_tokens.shape[0]
        h = self.embed(enemy_tokens)                            # (B, n, d)
        slots = self.slots_init.expand(B, -1, -1).contiguous()  # (B, K, d)
        k, v = self.to_k(h), self.to_v(h)
        assign = None
        for _ in range(self.n_iter):
            q = self.to_q(self.norm(slots))                     # (B, K, d)
            logits = q @ k.transpose(1, 2) / self.d_model ** 0.5  # (B, K, n)
            logits = logits.masked_fill(~mask.unsqueeze(1), -1e9)
            assign = F.softmax(logits, dim=1)                   # 实体在 slot 间竞争
            w = assign / assign.sum(2, keepdim=True).clamp(min=1e-6)
            updates = w @ v                                     # (B, K, d)
            slots = self.gru(updates.reshape(B * self.k_max, -1),
                             slots.reshape(B * self.k_max, -1)).reshape(B, self.k_max, -1)
        occupancy = (assign * mask.unsqueeze(1)).sum(dim=2)     # (B, K)
        return slots, assign.transpose(1, 2), occupancy


class OpponentStyleEncoder(nn.Module):
    """in-context 对手风格嵌入: 输入最近 T 步的队级交战统计序列."""

    def __init__(self, stat_dim: int = 8, d_style: int = 32):
        super().__init__()
        self.gru = nn.GRU(stat_dim, d_style, batch_first=True)

    def forward(self, stats_seq: torch.Tensor) -> torch.Tensor:
        """stats_seq: (B, T, stat_dim) -> z_opp: (B, d_style)"""
        _, h = self.gru(stats_seq)
        return h.squeeze(0)


class ForceTaskMatcher(nn.Module):
    """兵力-任务匹配: 每个敌簇派生 n_task_types 种任务 + n_free 个自由任务 slot."""

    def __init__(self, d_model: int = 64, d_style: int = 32,
                 n_task_types: int = 3, n_free: int = 2):
        super().__init__()
        self.n_task_types, self.n_free = n_task_types, n_free
        self.task_type_emb = nn.Parameter(torch.randn(n_task_types, d_model) * 0.1)
        self.free_task_emb = nn.Parameter(torch.randn(n_free, d_model) * 0.1)
        self.task_proj = nn.Sequential(
            nn.Linear(d_model * 2 + d_style, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.unit_proj = nn.Linear(d_model, d_model)
        self.d_model = d_model

    def forward(self, unit_emb: torch.Tensor, cluster_slots: torch.Tensor,
                occupancy: torch.Tensor, z_opp: torch.Tensor):
        """unit_emb: (B, U, d), cluster_slots: (B, K, d), occupancy: (B, K),
        z_opp: (B, d_style).

        返回 goal: (B, U, d) 每单位任务向量 g_i, assign: (B, U, n_tasks) 软分配.
        """
        B, K, d = cluster_slots.shape
        c = cluster_slots.unsqueeze(2).expand(B, K, self.n_task_types, d)
        t = self.task_type_emb.view(1, 1, self.n_task_types, d).expand(B, K, -1, -1)
        z = z_opp.view(B, 1, 1, -1).expand(B, K, self.n_task_types, -1)
        tasks = self.task_proj(torch.cat([c, t, z], dim=-1)).reshape(B, K * self.n_task_types, d)
        tasks = torch.cat([tasks, self.free_task_emb.unsqueeze(0).expand(B, -1, -1)], dim=1)
        # 空簇的任务被 occupancy 门控抑制
        gate = torch.cat([occupancy.repeat_interleave(self.n_task_types, dim=1),
                          torch.ones(B, self.n_free, device=occupancy.device)], dim=1)
        logits = self.unit_proj(unit_emb) @ tasks.transpose(1, 2) / self.d_model ** 0.5
        logits = logits + torch.log(gate.clamp(min=1e-6)).unsqueeze(1)
        assign = F.softmax(logits, dim=-1)                      # (B, U, n_tasks)
        goal = assign @ tasks                                   # (B, U, d)
        return goal, assign


class Commander(nn.Module):
    """模块 B 整体封装 (批处理).

    dynamic=True  : 完整版 —— 簇由敌方实时结构决定 (对手锚定, 论文方法);
    dynamic=False : Fixed-K 基线 —— 簇为可学习静态嵌入, 与敌结构无关
                    (自锚定固定 K 分层, 隔离"对手锚定动态簇"的贡献,
                     对应 paper/03 假设 (b) 与 paper/05 基线 3 / 消融 (a)).
    """

    def __init__(self, entity_dim=16, d_model=64, d_style=32, stat_dim=8,
                 k_max=6, n_task_types=3, n_free=2, dynamic=True):
        super().__init__()
        self.dynamic = dynamic
        if dynamic:
            self.clustering = EnemySoftClustering(entity_dim, d_model, k_max)
        else:
            # 静态"簇"槽: 参数量与动态版同阶, 但不消费敌实体
            self.static_slots = nn.Parameter(torch.randn(k_max, d_model) * 0.1)
        self.k_max = k_max
        self.style = OpponentStyleEncoder(stat_dim, d_style)
        self.matcher = ForceTaskMatcher(d_model, d_style, n_task_types, n_free)

    def forward(self, unit_emb, enemy_tokens, enemy_mask, stats_seq):
        """unit_emb: (B, U, d), enemy_tokens: (B, n_enemy, entity_dim),
        enemy_mask: (B, n_enemy), stats_seq: (B, T, stat_dim)."""
        B = unit_emb.shape[0]
        if self.dynamic:
            slots, cluster_assign, occ = self.clustering(enemy_tokens, enemy_mask)
        else:
            slots = self.static_slots.unsqueeze(0).expand(B, -1, -1)
            cluster_assign = None
            # 固定均匀 occupancy: 门控退化为常数, 任务结构与敌结构解耦
            occ = torch.ones(B, self.k_max, device=unit_emb.device)
        z_opp = self.style(stats_seq)
        goal, task_assign = self.matcher(unit_emb, slots, occ, z_opp)
        return dict(goal=goal, task_assign=task_assign,
                    cluster_assign=cluster_assign, occupancy=occ, z_opp=z_opp)
