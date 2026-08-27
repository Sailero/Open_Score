"""开源合规测试: PettingZoo 官方 API 测试 + 同种子确定性 + 变规模 reset.

对应 docs/13 开源与可复现性审计的验收项.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from swarmbattle.pz_env import SwarmArenaPZEnv


def rollout(seed, n_red=6, n_blue=8):
    env = SwarmArenaPZEnv(n_red=n_red, n_blue=n_blue, capacity=16, max_steps=60)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(7)  # 动作序列固定, 只变环境种子
    trace = []
    while env.agents:
        acts = {a: np.array([rng.integers(9), rng.integers(17)]) for a in env.agents}
        obs, rew, term, trunc, _ = env.step(acts)
        key = (tuple(sorted(obs)), round(sum(rew.values()), 6))
        trace.append(key)
    return trace


def main():
    # 1) 官方 API 合规
    from pettingzoo.test import parallel_api_test
    parallel_api_test(SwarmArenaPZEnv(n_red=6, n_blue=8, capacity=16, max_steps=100),
                      num_cycles=300)
    print("[1] PettingZoo parallel_api_test: PASSED")

    # 2) 同种子确定性 / 异种子可区分
    t1, t2, t3 = rollout(42), rollout(42), rollout(43)
    same = len(t1) == len(t2) and all(a == b for a, b in zip(t1, t2))
    diff = t1 != t3
    print(f"[2] 同种子确定性: {'PASSED' if same else 'FAILED'} (轨迹长 {len(t1)})")
    print(f"[3] 异种子可区分: {'PASSED' if diff else 'FAILED'}")

    # 3) 变规模 reset (options 接口)
    env = SwarmArenaPZEnv(capacity=16)
    for nr, nb in ((4, 12), (16, 16), (2, 3)):
        obs, _ = env.reset(options=dict(n_red=nr, n_blue=nb))
        assert len([a for a in env.agents if a.startswith("red")]) == nr
        assert len([a for a in env.agents if a.startswith("blue")]) == nb
    print("[4] 变规模 reset (4v12/16v16/2v3): PASSED")

    if not (same and diff):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
