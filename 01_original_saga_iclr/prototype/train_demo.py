"""SAGA 训练 demo (gpu_py_310 环境, GPU).

流程:
1. 在 SwarmBattle 6v6 (对手=greedy_chase 脚本) 上用轻量 PPO 训练 SAGA 智能体;
2. 零样本规模迁移评测: 同一组参数直接测 6v6 / 12v12 / 24v24 / 12v24(非对称+增援);
3. 跨环境架构冒烟: 同一网络类在 MPE simple_tag 与 MAgent2 battle_v4 上前向+rollout;
4. 输出: 学习曲线 PNG, 训练前后回放 JSON (供 HTML 前端), checkpoint.

用法:
    python train_demo.py                 # 完整训练 (默认 150 episodes, ~15min GPU)
    python train_demo.py --episodes 30   # 快速验证
    python train_demo.py --no-commander  # 消融: 去掉模块B
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from envs.adapters import make_env, SwarmBattleAdapter
from saga.agent import SAGAAgent, prepare_obs

OUT_DIR = os.path.join(ROOT, "outputs")
FIG_DIR = os.path.abspath(os.path.join(ROOT, "..", "docs", "figures"))


# ----------------------------------------------------------------------
# rollout & replay
# ----------------------------------------------------------------------
def run_episode(env, agent, device, stats_buf=None, greedy=False,
                record=None, collect=None):
    obs = env.reset()
    stats_buf = deque(maxlen=32) if stats_buf is None else stats_buf
    done, ep_ret, steps = False, 0.0, 0
    while not done:
        if len(obs["tokens"]) == 0:
            break
        pack = prepare_obs(obs, env, stats_buf)
        move, tgt, logp, value = agent.act(pack, device, greedy=greedy)
        obs, r, done, info = env.step(dict(move=move, target=tgt))
        ep_ret += r
        steps += 1
        if collect is not None:
            collect.append(dict(pack=pack, move=move, tgt=tgt, logp=logp,
                                value=value, reward=r, done=done))
        if record is not None and isinstance(env, SwarmBattleAdapter):
            record.append(env.snapshot())
    return ep_ret, info.get("win", False), steps


# ----------------------------------------------------------------------
# 轻量 PPO (变长观测 -> 逐步前向累积损失; demo 规模下足够快)
# ----------------------------------------------------------------------
def ppo_update(agent, optimizer, buffer, device, epochs=3, clip=0.2,
               gamma=0.99, lam=0.95, vf_coef=0.5, ent_coef=0.01):
    # GAE
    adv, ret = [], []
    gae, next_v = 0.0, 0.0
    for tr in reversed(buffer):
        delta = tr["reward"] + gamma * next_v * (1 - tr["done"]) - tr["value"]
        gae = delta + gamma * lam * (1 - tr["done"]) * gae
        adv.append(gae)
        ret.append(gae + tr["value"])
        next_v = tr["value"]
    adv, ret = np.array(adv[::-1]), np.array(ret[::-1])
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    idx = np.arange(len(buffer))
    losses = []
    for _ in range(epochs):
        np.random.shuffle(idx)
        for start in range(0, len(idx), 64):
            batch = idx[start:start + 64]
            loss = torch.zeros((), device=device)
            for i in batch:
                tr = buffer[i]
                p = tr["pack"]
                t = {k: torch.as_tensor(v, device=device) for k, v in p.items()
                     if k != "n_move_bins"}
                dist_m, dist_t, value, _ = agent(t["tokens"], t["mask"],
                                                 t["enemy_toks"], t["enemy_mask"],
                                                 t["stats_seq"], p["n_move_bins"])
                move = torch.as_tensor(tr["move"], device=device)
                tgt = torch.as_tensor(tr["tgt"] + 1, device=device)
                logp = dist_m.log_prob(move).sum() + dist_t.log_prob(tgt).sum()
                ratio = torch.exp((logp - tr["logp"]).clamp(-20, 20))
                a = torch.as_tensor(adv[i], dtype=torch.float32, device=device)
                pg = -torch.min(ratio * a,
                                torch.clamp(ratio, 1 - clip, 1 + clip) * a)
                vf = (value - torch.as_tensor(ret[i], dtype=torch.float32,
                                              device=device)) ** 2
                ent = dist_m.entropy().mean() + dist_t.entropy().mean()
                loss = loss + pg + vf_coef * vf - ent_coef * ent
            loss = loss / len(batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()
            losses.append(float(loss))
    return float(np.mean(losses))


# ----------------------------------------------------------------------
def evaluate_scale_transfer(agent, device, scales, n_eval=10, seed=100):
    """零样本规模迁移: 同一组参数在不同 (N, M, 增援) 下的胜率."""
    results = {}
    for name, kw in scales.items():
        wins, rets = [], []
        for e in range(n_eval):
            env = SwarmBattleAdapter(opponent=OPP, opponent_params=OPP_PARAMS,
                                     seed=seed + e, **kw)
            ret, win, _ = run_episode(env, agent, device, greedy=True)
            wins.append(win)
            rets.append(ret)
        results[name] = dict(win_rate=float(np.mean(wins)),
                             mean_return=float(np.mean(rets)))
        print(f"  [零样本] {name}: 胜率={np.mean(wins):.0%} 回报={np.mean(rets):+.2f}")
    return results


def cross_env_smoke(agent, device):
    """同一网络类在公开基准环境上的接口冒烟 (架构通用性验证)."""
    print("[跨环境] 同一 SAGA 网络在公开基准上 rollout ...")
    for name in ("mpe_tag", "magent_battle"):
        try:
            env = make_env(name, seed=0)
            ret, win, steps = run_episode(env, agent, device)
            n_units = "3(追捕者)" if name == "mpe_tag" else "12(红队)"
            print(f"  {name}: {steps} steps 完成, 控制单位数={n_units}, 回报={ret:+.2f}")
        except Exception as ex:  # 公开环境版本差异兜底, 不阻塞 demo 主线
            print(f"  {name}: 失败 ({type(ex).__name__}: {ex})")


def export_replay(agent, device, path, greedy, seed=7, **env_kw):
    env = SwarmBattleAdapter(opponent=OPP, opponent_params=OPP_PARAMS,
                             seed=seed, **env_kw)
    frames = []
    ret, win, steps = run_episode(env, agent, device, greedy=greedy, record=frames)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(frames=frames, result=dict(ret=ret, win=bool(win))), f)
    print(f"[回放] {os.path.basename(path)}: {steps} 帧, win={win}, ret={ret:+.2f}")


# ----------------------------------------------------------------------
OPP = "greedy_chase"
OPP_PARAMS = dict(aggression=0.45)  # demo 用降低攻击性的对手, 使学习曲线在百局内可见


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--update-every", type=int, default=4, help="每N局做一次PPO更新")
    ap.add_argument("--no-commander", action="store_true")
    ap.add_argument("--agent", default="saga",
                    choices=["saga", "fixed_k", "set_flat", "mappo_flat"],
                    help="saga=完整方法 | fixed_k/set_flat/mappo_flat=基线")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    torch.manual_seed(0)
    np.random.seed(0)

    if args.no_commander:
        args.agent = "set_flat"
    if args.agent == "mappo_flat":
        from baselines.flat_mappo import FlatMAPPOAgent
        agent = FlatMAPPOAgent().to(device)
    else:
        mode = {"saga": "dynamic", "fixed_k": "fixed_k", "set_flat": "none"}[args.agent]
        agent = SAGAAgent(commander_mode=mode).to(device)
    n_params = sum(p.numel() for p in agent.parameters())
    print(f"[{args.agent}] device={device}, 参数量={n_params:,}")

    export_replay(agent, device, os.path.join(OUT_DIR, "replay_before.json"),
                  greedy=False, n_red=6, n_blue=6)

    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr)
    history, buffer = [], []
    t0 = time.time()
    for ep in range(1, args.episodes + 1):
        env = SwarmBattleAdapter(n_red=6, n_blue=6, opponent=OPP,
                                 opponent_params=OPP_PARAMS, seed=ep)
        collect = []
        ret, win, steps = run_episode(env, agent, device, collect=collect)
        buffer += collect
        history.append(dict(ep=ep, ret=ret, win=int(win), steps=steps))
        if ep % args.update_every == 0:
            try:
                loss = ppo_update(agent, optimizer, buffer, device)
            except RuntimeError as ex:
                if "CUDA" in str(ex) and device.type == "cuda":
                    # 笔记本 GPU 偶发驱动错误: 回退 CPU 继续训练 (网络很小, CPU 足够)
                    print(f"[警告] CUDA 错误, 回退 CPU 继续: {ex}")
                    device = torch.device("cpu")
                    agent.cpu()
                    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr)
                    loss = ppo_update(agent, optimizer, buffer, device)
                else:
                    raise
            buffer = []
            recent = history[-20:]
            print(f"ep {ep:4d} | ret {np.mean([h['ret'] for h in recent]):+7.2f} "
                  f"| win {np.mean([h['win'] for h in recent]):.0%} "
                  f"| loss {loss:+.3f} | {time.time()-t0:.0f}s")

    torch.save(agent.state_dict(), os.path.join(OUT_DIR, "saga_demo.pt"))

    # 学习曲线
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rets = np.array([h["ret"] for h in history])
    wins = np.array([h["win"] for h in history], float)
    k = min(15, len(rets))
    smooth = np.convolve(rets, np.ones(k) / k, "valid")
    wsmooth = np.convolve(wins, np.ones(k) / k, "valid")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(rets, alpha=0.25, color="tab:blue")
    ax[0].plot(np.arange(k - 1, len(rets)), smooth, color="tab:blue")
    ax[0].set_title("Episode return (SwarmBattle 6v6 vs greedy)")
    ax[0].set_xlabel("episode")
    ax[1].plot(np.arange(k - 1, len(wins)), wsmooth, color="tab:red")
    ax[1].set_ylim(0, 1)
    ax[1].set_title(f"Win rate (moving avg, k={k})")
    ax[1].set_xlabel("episode")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "demo_learning_curve.png"), dpi=150)
    print(f"[图] 学习曲线 -> docs/figures/demo_learning_curve.png")

    # 零样本规模迁移 (训练只见 6v6)
    print("[评测] 零样本规模迁移 (训练域=6v6):")
    transfer = evaluate_scale_transfer(agent, device, scales={
        "6v6 (train)": dict(n_red=6, n_blue=6),
        "12v12": dict(n_red=12, n_blue=12),
        "24v24": dict(n_red=24, n_blue=24),
        "12v24+增援": dict(n_red=12, n_blue=24, reinforce_rate=0.02),
    })
    with open(os.path.join(OUT_DIR, "scale_transfer.json"), "w", encoding="utf-8") as f:
        json.dump(transfer, f, ensure_ascii=False, indent=2)

    cross_env_smoke(agent, device)

    export_replay(agent, device, os.path.join(OUT_DIR, "replay_after.json"),
                  greedy=True, n_red=6, n_blue=6)
    export_replay(agent, device, os.path.join(OUT_DIR, "replay_after_24v24.json"),
                  greedy=True, n_red=24, n_blue=24)

    # 生成回放前端
    from viewer.build_viewer import build
    build()
    print(f"[完成] 总耗时 {time.time()-t0:.0f}s. 打开 prototype/viewer/replay.html 查看回放.")


if __name__ == "__main__":
    main()
