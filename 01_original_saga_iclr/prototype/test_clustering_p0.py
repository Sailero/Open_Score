"""P0 单元测试: 对手锚定动态分簇的可学习性 (技术路线图 0.1 / 决策门 G1).

问题: 原型冒烟测试显示 slot attention 的 6 个 slot 恒全激活, "空 slot 自动退化"
      未兑现. 本测试验证: 加入空间一致性正则 L_coh + 簇数稀疏正则后,
      EnemySoftClustering 能否在合成态势上恢复真实群数.

设置: 合成敌方态势, 真实群数 g ∈ {1,2,3,4}, 每群位置服从各群中心的高斯分布.
      训练目标 = L_coh (软分配加权的簇内位置方差) + β·Σ_κ √(o_κ/M) (稀疏项,
      凹函数惩罚碎片化分簇). 无任何群数标签 —— 完全无监督.

判据 (G1): 测试集上 |有效簇数 − 真实群数| ≤ 1 的比例 ≥ 80%, 且群数与有效簇数
      的相关性显著 (不是恒输出同一个数).
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from saga.commander import EnemySoftClustering

ENTITY_DIM = 16
ARENA = 20.0


def make_scene(rng, g=None, m_range=(8, 24)):
    """生成一个合成敌方态势: g 个空间群. 返回 tokens (M,16), 真实群数."""
    g = g or int(rng.integers(1, 5))
    M = int(rng.integers(*m_range))
    # 群中心两两距离至少 8, 保证可分
    centers = []
    while len(centers) < g:
        c = rng.uniform(-ARENA * 0.8, ARENA * 0.8, 2)
        if all(np.linalg.norm(c - c2) > 8.0 for c2 in centers):
            centers.append(c)
    assign = rng.integers(0, g, M)
    pos = np.stack(centers)[assign] + rng.normal(0, 1.2, (M, 2))
    tok = np.zeros((M, ENTITY_DIM), np.float32)
    tok[:, 0:2] = pos / ARENA                      # 相对位置 (归一化)
    tok[:, 2:4] = rng.normal(0, 0.1, (M, 2))       # 速度噪声
    tok[:, 6] = rng.uniform(0.5, 1.0, M)           # 血量
    tok[:, 8] = 1.0                                # enemy flag
    tok[:, 9] = 1.0                                # alive
    tok[:, 10] = 1.0                               # 类型占位
    return tok, g


def batch_scenes(rng, B, **kw):
    toks, gs, masks = [], [], []
    m_max = 0
    scenes = [make_scene(rng, **kw) for _ in range(B)]
    m_max = max(t.shape[0] for t, _ in scenes)
    for t, g in scenes:
        pad = np.zeros((m_max, ENTITY_DIM), np.float32)
        pad[: t.shape[0]] = t
        mask = np.zeros(m_max, bool)
        mask[: t.shape[0]] = True
        toks.append(pad), gs.append(g), masks.append(mask)
    return (torch.from_numpy(np.stack(toks)), torch.from_numpy(np.stack(masks)),
            np.array(gs))


def losses(module, toks, masks, beta):
    slots, assign, occ = module(toks, masks)       # assign: (B, M, K)
    pos = toks[:, :, 0:2]                          # 归一化位置
    m = masks.float().unsqueeze(-1)                # (B, M, 1)
    w = assign * m                                 # 掩码后的软分配 (B, M, K)
    wsum = w.sum(1).clamp(min=1e-6)                # (B, K)
    mu = torch.einsum("bmk,bmd->bkd", w, pos) / wsum.unsqueeze(-1)   # 软质心
    var = torch.einsum("bmk,bmkd->bk",
                       w, (pos.unsqueeze(2) - mu.unsqueeze(1)) ** 2)
    M_count = m.sum(1)                             # (B, 1)
    l_coh = (var.sum(1) / M_count.squeeze(-1)).mean()
    # 凹稀疏项: sqrt 在 0 处梯度无穷, 必须 clamp (P0 首轮 NaN 的教训)
    l_sparse = torch.sqrt((occ / M_count).clamp(min=1e-4)).sum(1).mean()
    return l_coh + beta * l_sparse, l_coh, l_sparse, occ


@torch.no_grad()
def effective_k(module, toks, masks, thresh=0.5):
    _, _, occ = module(toks, masks)
    return (occ > thresh).sum(1).cpu().numpy()


def run(beta=0.02, k_max=8, steps=600, lr=3e-3, seed=0, verbose=True):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    module = EnemySoftClustering(ENTITY_DIM, d_model=64, k_max=k_max)
    opt = torch.optim.Adam(module.parameters(), lr=lr)
    for it in range(steps):
        toks, masks, _ = batch_scenes(rng, 32)
        # 稀疏项 warmup: 先让 L_coh 分化出簇, 再逐步施加稀疏压力,
        # 防止训练早期全实体塌缩进单个 slot (seed 1 的失败模式)
        beta_t = beta * min(1.0, it / (steps * 0.4))
        loss, l_coh, l_sp, _ = losses(module, toks, masks, beta_t)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
        opt.step()
        if verbose and (it + 1) % 150 == 0:
            print(f"  step {it+1}: L_coh={l_coh:.4f} L_sparse={l_sp:.3f}")
    # ---- 评测 ----
    rng_t = np.random.default_rng(seed + 1000)
    all_k, all_g = [], []
    for g in (1, 2, 3, 4):
        toks, masks, gs = batch_scenes(rng_t, 64, g=g)
        all_k.append(effective_k(module, toks, masks))
        all_g.append(gs)
    ks, gs = np.concatenate(all_k), np.concatenate(all_g)
    acc1 = float(np.mean(np.abs(ks - gs) <= 1))
    corr = float(np.corrcoef(ks, gs)[0, 1])
    per_g = {g: (float(ks[gs == g].mean()), float(ks[gs == g].std()))
             for g in (1, 2, 3, 4)}
    return acc1, corr, per_g


def main():
    print("[P0] 动态分簇可学习性测试 (无监督, L_coh + 稀疏正则)")
    results = []
    for seed in (0, 1, 2):
        acc1, corr, per_g = run(seed=seed, verbose=(seed == 0))
        results.append((acc1, corr))
        pg = ", ".join(f"g={g}: Keff={m:.2f}+-{s:.2f}" for g, (m, s) in per_g.items())
        print(f"  seed {seed}: |Keff-g|<=1 准确率={acc1:.0%}, corr(Keff,g)={corr:.3f}")
        print(f"    {pg}")
    accs = [a for a, _ in results]
    corrs = [c for _, c in results]
    passed = np.mean(accs) >= 0.8 and np.mean(corrs) > 0.5
    print(f"\n[G1 判定] 平均准确率={np.mean(accs):.0%} (判据>=80%), "
          f"平均相关={np.mean(corrs):.3f} (判据>0.5) -> "
          f"{'通过: 动态分簇可保留' if passed else '未通过: 触发 Sinkhorn 备选方案'}")


if __name__ == "__main__":
    main()
