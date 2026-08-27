"""推理延迟基准: 部署可行性论证用.

测量 SAGA 完整决策 (编码器+指挥层+底层策略) 在不同规模下的单次前向耗时,
分别在 CPU 与 GPU 上测, 对应游戏服务器 tick 预算的评估.
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from saga.agent import SAGAAgent, STAT_DIM
from envs.adapters import ENTITY_DIM


def bench(device, n_own, n_enemy, n_repeat=50):
    agent = SAGAAgent().to(device).eval()
    n_ent = n_own - 1 + n_enemy
    tokens = torch.randn(n_own, n_ent, ENTITY_DIM, device=device)
    mask = torch.ones(n_own, n_ent, dtype=torch.bool, device=device)
    etoks = torch.randn(n_enemy, ENTITY_DIM, device=device)
    emask = torch.ones(n_enemy, dtype=torch.bool, device=device)
    stats = torch.randn(8, STAT_DIM, device=device)
    with torch.no_grad():
        for _ in range(5):  # warmup
            agent(tokens, mask, etoks, emask, stats, 9)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_repeat):
            agent(tokens, mask, etoks, emask, stats, 9)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / n_repeat * 1000
    return dt


def main():
    torch.set_num_threads(4)
    print(f"{'规模':>12} | {'CPU (ms)':>9} | {'GPU (ms)':>9}")
    for n in (8, 16, 32, 64, 128, 256):
        cpu = bench(torch.device("cpu"), n, n)
        gpu = bench(torch.device("cuda"), n, n) if torch.cuda.is_available() else float("nan")
        print(f"{n:>5}v{n:<5} | {cpu:9.2f} | {gpu:9.2f}")


if __name__ == "__main__":
    main()
