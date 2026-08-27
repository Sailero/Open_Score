# Baseline policy

主实验采用“统一下层、只比较上层”的公平协议。baseline 输出、检查点和复现日志可放在此目录，但不提交大体积训练产物。

## 必须比较

1. nearest/type-matching heuristic；
2. constrained Hungarian/greedy heuristic；
3. ALMA-style allocation；
4. PLATO-style pointer MAPPO；
5. DT-GAT-style dual-frequency allocator；
6. latest-opponent self-play；
7. proposed history-opponent-pool training。

现有 `src/q2marl/models.py` 的 `HPNPolicy` 和 `OpponentConditionedHPN` 是旧阶段留下的可变实体结构基线，可以复用 encoder，但二者不再代表论文主方法。所有独立重实现必须记录来源论文、实现差异、commit SHA、环境版本和训练预算。
