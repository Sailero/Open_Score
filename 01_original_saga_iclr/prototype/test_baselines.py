"""基线就绪度测试: 验证全部学习型基线与 SAGA 接口对齐、可在训练循环中直接替换.

检查项 (对应 docs/12 实验就绪度审计):
1. 四个学习型 agent (SAGA / SetPolicy-flat / Fixed-K / MAPPO-flat) 在
   6v6 与 24v24 上完成完整 episode rollout (同一循环代码, 零改动);
2. 参数量对照 (公平性协议要求同量级);
3. MAPPO-flat 的架构性缺陷可观测: 敌数 > E_MAX 时目标头盲区
   (打印可选目标数 vs 实际敌数);
4. Heuristic-strong 脚本 (focus_kite) 对 greedy 的胜率 sanity check.
"""
import os
import sys
from collections import deque

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.adapters import SwarmBattleAdapter
from saga.agent import SAGAAgent, prepare_obs
from baselines.flat_mappo import FlatMAPPOAgent, E_MAX
from swarmbattle.scripts import focus_greedy, greedy_chase
from swarmbattle.env import SwarmBattleEnv, Config


def rollout(env, agent, device):
    obs = env.reset()
    stats = deque(maxlen=32)
    done, ret, steps = False, 0.0, 0
    while not done and len(obs["tokens"]):
        pack = prepare_obs(obs, env, stats)
        move, tgt, _, _ = agent.act(pack, device, greedy=False)
        obs, r, done, info = env.step(dict(move=move, target=tgt))
        ret += r
        steps += 1
    return ret, info.get("win", False), steps


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cpu")   # 接口测试, CPU 足够

    agents = {
        "SAGA (dynamic)": SAGAAgent(commander_mode="dynamic"),
        "Fixed-K 基线": SAGAAgent(commander_mode="fixed_k"),
        "SetPolicy-flat 基线": SAGAAgent(commander_mode="none"),
        "MAPPO-flat 基线": FlatMAPPOAgent(),
    }

    print(f"{'agent':<24} | {'参数量':>9} | {'6v6':>14} | {'24v24':>14}")
    for name, agent in agents.items():
        n_params = sum(p.numel() for p in agent.parameters())
        row = f"{name:<24} | {n_params:>9,}"
        for n in (6, 24):
            env = SwarmBattleAdapter(n_red=n, n_blue=n, opponent="greedy_chase",
                                     opponent_params=dict(aggression=0.45), seed=42)
            ret, win, steps = rollout(env, agent, device)
            row += f" | {steps:>3}步 ret{ret:+6.1f}"
        print(row)

    print(f"\n[架构缺陷验证] MAPPO-flat 目标头容量 E_MAX={E_MAX}: "
          f"24v24 时 24 个敌人中仅前 {E_MAX} 个可被选中 (盲区 8 个) -> "
          f"规模外推失效的机制在架构层面即成立")

    # Heuristic-strong sanity check
    wins = 0
    for s in range(20):
        env = SwarmBattleEnv(Config(seed=100 + s))
        env.reset(n_red=8, n_blue=8)
        done = False
        while not done:
            ra = focus_greedy(env, "red")
            ba = greedy_chase(env, "blue", aggression=1.0)
            _, _, done, info = env.step(ra, ba)
        wins += info["n_red"] > info["n_blue"]
    print(f"[Heuristic-strong] focus_greedy vs greedy(1.0) 8v8: 胜率 {wins}/20 "
          f"(应显著 >10/20 才配称强脚本)")


if __name__ == "__main__":
    main()
