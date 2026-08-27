"""冒烟测试: 验证 环境 <-> 脚本对手 <-> SAGA 三模块 的接口联通性.

1. 8v12 非对称对局, 双方脚本对手互打, 中途开启增援 -> 验证环境规则与动态规模;
2. 用红方观测跑一遍 SAGA 前向 (编码器 -> 指挥层 -> 底层策略) -> 验证数量无关性:
   同一网络在 8v12 与 24v6 两种规模下前向均成功且参数不变;
3. 渲染一局对抗轨迹图保存到 docs/figures/.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from swarmbattle.env import SwarmBattleEnv, Config, ENTITY_DIM
from swarmbattle.scripts import greedy_chase, kite, sample_script
from saga.encoder import EntityFieldEncoder
from saga.commander import Commander
from saga.policy import MicroPolicy, SetCritic


def pad_tokens(token_list):
    """变长实体 token 列表 -> (B, n_max, D) padding + mask."""
    B = len(token_list)
    n_max = max(1, max(t.shape[0] for t in token_list))
    out = np.zeros((B, n_max, ENTITY_DIM), np.float32)
    mask = np.zeros((B, n_max), bool)
    for i, t in enumerate(token_list):
        out[i, : t.shape[0]] = t
        mask[i, : t.shape[0]] = True
    return torch.from_numpy(out), torch.from_numpy(mask)


def global_enemy_tokens(env, side):
    """指挥层用: 全体敌实体 token (原型简化为全局可见)."""
    own, opp = (env.red, env.blue) if side == "red" else (env.blue, env.red)
    center = own.pos[own.alive].mean(axis=0) if own.n_alive else np.zeros(2)
    toks, mask = env._entity_tokens(center, np.zeros(2), 1e9, own, opp)
    n_own = own.n
    return (torch.from_numpy(toks[n_own:]).float(),
            torch.from_numpy(mask[n_own:] & opp.alive).bool())


def saga_forward(env, encoder, commander, policy, critic, side="red"):
    obs = env._observations()[side]
    if len(obs["tokens"]) == 0:
        return None
    tok, mask = pad_tokens(obs["tokens"])
    unit_emb = encoder(tok, mask)                                    # (n_units, d)
    enemy_tok, enemy_mask = global_enemy_tokens(env, side)
    stats_seq = torch.randn(16, 8)                                   # 占位: 队级交战统计
    cmd = commander(unit_emb, enemy_tok, enemy_mask, stats_seq)
    B = unit_emb.shape[0]
    out = policy(tok, mask, enemy_tok.unsqueeze(0).expand(B, -1, -1),
                 enemy_mask.unsqueeze(0).expand(B, -1), cmd["goal"])
    all_tok = torch.cat([t for t in map(torch.from_numpy, obs["tokens"])])
    value = critic(all_tok.float(), torch.ones(all_tok.shape[0], dtype=bool))
    return dict(n_units=B, n_enemy_visible=int(enemy_mask.sum()),
                active_clusters=int((cmd["occupancy"] > 0.5).sum()),
                goal_shape=tuple(cmd["goal"].shape),
                target_logits_shape=tuple(out["target_logits"].shape),
                value=float(value))


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    encoder = EntityFieldEncoder(ENTITY_DIM)
    commander = Commander(ENTITY_DIM)
    policy = MicroPolicy(ENTITY_DIM)
    critic = SetCritic(ENTITY_DIM)
    n_params = sum(p.numel() for m in (encoder, commander, policy, critic)
                   for p in m.parameters())
    print(f"[SAGA] 总参数量: {n_params:,} (与双方数量无关)")

    # ---- 测试 1: 数量无关前向 ----
    for n_red, n_blue in [(8, 12), (24, 6), (48, 48)]:
        env = SwarmBattleEnv(Config(seed=1))
        env.reset(n_red=n_red, n_blue=n_blue)
        info = saga_forward(env, encoder, commander, policy, critic)
        print(f"[前向] {n_red}v{n_blue}: {info}")

    # ---- 测试 2: 完整对局 (含中途增援) + 轨迹渲染 ----
    env = SwarmBattleEnv(Config(seed=7, reinforce_rate=0.01, max_steps=300))
    env.reset(n_red=8, n_blue=12)
    name, params, blue_script = sample_script(np.random.default_rng(3))
    print(f"[对局] 红=greedy_chase vs 蓝={name}{params}, 初始 8v12, 增援开启")

    traj_red, traj_blue = [], []
    done, t, ret = False, 0, 0.0
    while not done:
        ra = greedy_chase(env, "red", aggression=0.9)
        ba = blue_script(env, "blue")
        _, (r, _), done, info = env.step(ra, ba)
        ret += r
        traj_red.append((env.red.pos.copy(), env.red.alive.copy()))
        traj_blue.append((env.blue.pos.copy(), env.blue.alive.copy()))
        t += 1
    print(f"[对局] 结束于 t={info['t']}, 存活 红={info['n_red']} 蓝={info['n_blue']}, "
          f"红累计回报={ret:.2f}, 期末规模 红={env.red.n} 蓝={env.blue.n} (含增援)")

    # ---- 渲染 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 7))
    stride = max(1, len(traj_red) // 120)
    for traj, cmap in ((traj_red, plt.cm.Reds), (traj_blue, plt.cm.Blues)):
        T = len(traj)
        for k in range(0, T, stride):
            pos, alive = traj[k]
            ax.scatter(pos[alive, 0], pos[alive, 1], s=6,
                       color=cmap(0.3 + 0.7 * k / T), alpha=0.5, linewidths=0)
    for pos, alive, c in ((traj_red[-1][0], traj_red[-1][1], "darkred"),
                          (traj_blue[-1][0], traj_blue[-1][1], "navy")):
        ax.scatter(pos[alive, 0], pos[alive, 1], s=60, color=c, marker="^",
                   edgecolors="white", zorder=5)
    half = env.cfg.arena_size / 2
    ax.set_xlim(-half, half); ax.set_ylim(-half, half)
    ax.set_aspect("equal")
    ax.set_title(f"SwarmBattle prototype: red(greedy) vs blue({name}), "
                 f"8v12 start, reinforcements on\n(color: light=early, dark=late; "
                 f"triangles=final survivors)")
    fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    out_path = os.path.abspath(os.path.join(fig_dir, "smoke_test_rollout.png"))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[渲染] 轨迹图已保存: {out_path}")


if __name__ == "__main__":
    main()
