# 共同规则底层：六条上层路线比较

所有路线使用同一个规则执行器、已知对手及原生终局奖励。B1/B2/B3 使用共享准备阶段产出的支付模型；B1 没有伪造的策略训练曲线。

best 仅由独立验证选出；latest 与 initialized 单列。单个训练种子只能用于方向筛选。尚未完成的评估不能当作零胜率。

| 路线 | 种子 | 模型 | 规模 | 成功/局数 | 成功率 |
|---|---:|---|---:|---:|---:|
| b1_counts | 20260906 | policy | 8 | 0/2 | 0.0% |
| b1_counts | 20260906 | policy | 12 | 0/2 | 0.0% |
| b1_counts | 20260906 | policy | 16 | 0/2 | 0.0% |
| b1_counts | 20260906 | policy | 24 | 1/2 | 50.0% |
| b1_counts | 20260906 | policy | 32 | 0/2 | 0.0% |
| b2_count | 20260906 | policy | 8 | 0/2 | 0.0% |
| b2_count | 20260906 | policy | 12 | 0/2 | 0.0% |
| b2_count | 20260906 | policy | 16 | 0/2 | 0.0% |
| b2_count | 20260906 | policy | 24 | 0/2 | 0.0% |
| b2_count | 20260906 | policy | 32 | 0/2 | 0.0% |
| b2_local | 20260906 | policy | 8 | 1/2 | 50.0% |
| b2_local | 20260906 | policy | 12 | 0/2 | 0.0% |
| b2_local | 20260906 | policy | 16 | 0/2 | 0.0% |
| b2_local | 20260906 | policy | 24 | 0/2 | 0.0% |
| b2_local | 20260906 | policy | 32 | 0/2 | 0.0% |
| b2_min | 20260906 | policy | 8 | 0/2 | 0.0% |
| b2_min | 20260906 | policy | 12 | 0/2 | 0.0% |
| b2_min | 20260906 | policy | 16 | 0/2 | 0.0% |
| b2_min | 20260906 | policy | 24 | 0/2 | 0.0% |
| b2_min | 20260906 | policy | 32 | 0/2 | 0.0% |
| b3_global | 20260906 | policy | 8 | 0/2 | 0.0% |
| b3_global | 20260906 | policy | 12 | 0/2 | 0.0% |
| b3_global | 20260906 | policy | 16 | 0/2 | 0.0% |
| b3_global | 20260906 | policy | 24 | 0/2 | 0.0% |
| b3_global | 20260906 | policy | 32 | 0/2 | 0.0% |
| b3_rebuild | 20260906 | policy | 8 | 0/2 | 0.0% |
| b3_rebuild | 20260906 | policy | 12 | 0/2 | 0.0% |
| b3_rebuild | 20260906 | policy | 16 | 0/2 | 0.0% |
| b3_rebuild | 20260906 | policy | 24 | 0/2 | 0.0% |
| b3_rebuild | 20260906 | policy | 32 | 0/2 | 0.0% |
| grand | 20260906 | policy | 8 | 0/2 | 0.0% |
| grand | 20260906 | policy | 12 | 0/2 | 0.0% |
| grand | 20260906 | policy | 16 | 0/2 | 0.0% |
| grand | 20260906 | policy | 24 | 0/2 | 0.0% |
| grand | 20260906 | policy | 32 | 0/2 | 0.0% |
| group_only | 20260906 | policy | 8 | 1/2 | 50.0% |
| group_only | 20260906 | policy | 12 | 0/2 | 0.0% |
| group_only | 20260906 | policy | 16 | 1/2 | 50.0% |
| group_only | 20260906 | policy | 24 | 0/2 | 0.0% |
| group_only | 20260906 | policy | 32 | 0/2 | 0.0% |
| r1_ppo | 20260906 | best | 8 | 0/2 | 0.0% |
| r1_ppo | 20260906 | best | 12 | 0/2 | 0.0% |
| r1_ppo | 20260906 | best | 16 | 0/2 | 0.0% |
| r1_ppo | 20260906 | best | 24 | 0/2 | 0.0% |
| r1_ppo | 20260906 | best | 32 | 0/2 | 0.0% |
| r1_ppo | 20260906 | latest | 8 | 0/2 | 0.0% |
| r1_ppo | 20260906 | latest | 12 | 0/2 | 0.0% |
| r1_ppo | 20260906 | latest | 16 | 0/2 | 0.0% |
| r1_ppo | 20260906 | latest | 24 | 0/2 | 0.0% |
| r1_ppo | 20260906 | latest | 32 | 0/2 | 0.0% |
| r1_ppo | 20260906 | initialized | 8 | 0/2 | 0.0% |
| r1_ppo | 20260906 | initialized | 12 | 0/2 | 0.0% |
| r1_ppo | 20260906 | initialized | 16 | 0/2 | 0.0% |
| r1_ppo | 20260906 | initialized | 24 | 0/2 | 0.0% |
| r1_ppo | 20260906 | initialized | 32 | 0/2 | 0.0% |
| r2_teacher_ppo | 20260906 | best | 8 | 0/2 | 0.0% |
| r2_teacher_ppo | 20260906 | best | 12 | 0/2 | 0.0% |
| r2_teacher_ppo | 20260906 | best | 16 | 0/2 | 0.0% |
| r2_teacher_ppo | 20260906 | best | 24 | 0/2 | 0.0% |
| r2_teacher_ppo | 20260906 | best | 32 | 0/2 | 0.0% |
| r2_teacher_ppo | 20260906 | latest | 8 | 0/2 | 0.0% |
| r2_teacher_ppo | 20260906 | latest | 12 | 0/2 | 0.0% |
| r2_teacher_ppo | 20260906 | latest | 16 | 0/2 | 0.0% |
| r2_teacher_ppo | 20260906 | latest | 24 | 0/2 | 0.0% |
| r2_teacher_ppo | 20260906 | latest | 32 | 0/2 | 0.0% |
| r2_teacher_ppo | 20260906 | initialized | 8 | 0/2 | 0.0% |
| r2_teacher_ppo | 20260906 | initialized | 12 | 0/2 | 0.0% |
| r2_teacher_ppo | 20260906 | initialized | 16 | 0/2 | 0.0% |
| r2_teacher_ppo | 20260906 | initialized | 24 | 0/2 | 0.0% |
| r2_teacher_ppo | 20260906 | initialized | 32 | 0/2 | 0.0% |
| r3_ddqn | 20260906 | best | 8 | 0/2 | 0.0% |
| r3_ddqn | 20260906 | best | 12 | 0/2 | 0.0% |
| r3_ddqn | 20260906 | best | 16 | 0/2 | 0.0% |
| r3_ddqn | 20260906 | best | 24 | 0/2 | 0.0% |
| r3_ddqn | 20260906 | best | 32 | 0/2 | 0.0% |
| r3_ddqn | 20260906 | latest | 8 | 0/2 | 0.0% |
| r3_ddqn | 20260906 | latest | 12 | 0/2 | 0.0% |
| r3_ddqn | 20260906 | latest | 16 | 0/2 | 0.0% |
| r3_ddqn | 20260906 | latest | 24 | 0/2 | 0.0% |
| r3_ddqn | 20260906 | latest | 32 | 0/2 | 0.0% |
| r3_ddqn | 20260906 | initialized | 8 | 0/2 | 0.0% |
| r3_ddqn | 20260906 | initialized | 12 | 0/2 | 0.0% |
| r3_ddqn | 20260906 | initialized | 16 | 0/2 | 0.0% |
| r3_ddqn | 20260906 | initialized | 24 | 0/2 | 0.0% |
| r3_ddqn | 20260906 | initialized | 32 | 0/2 | 0.0% |
| random | 20260906 | policy | 8 | 0/2 | 0.0% |
| random | 20260906 | policy | 12 | 0/2 | 0.0% |
| random | 20260906 | policy | 16 | 0/2 | 0.0% |
| random | 20260906 | policy | 24 | 0/2 | 0.0% |
| random | 20260906 | policy | 32 | 0/2 | 0.0% |
| rule | 20260906 | policy | 8 | 0/2 | 0.0% |
| rule | 20260906 | policy | 12 | 0/2 | 0.0% |
| rule | 20260906 | policy | 16 | 0/2 | 0.0% |
| rule | 20260906 | policy | 24 | 0/2 | 0.0% |
| rule | 20260906 | policy | 32 | 0/2 | 0.0% |
| static_rule | 20260906 | policy | 8 | 1/2 | 50.0% |
| static_rule | 20260906 | policy | 12 | 0/2 | 0.0% |
| static_rule | 20260906 | policy | 16 | 0/2 | 0.0% |
| static_rule | 20260906 | policy | 24 | 0/2 | 0.0% |
| static_rule | 20260906 | policy | 32 | 0/2 | 0.0% |
| target_only | 20260906 | policy | 8 | 0/2 | 0.0% |
| target_only | 20260906 | policy | 12 | 0/2 | 0.0% |
| target_only | 20260906 | policy | 16 | 0/2 | 0.0% |
| target_only | 20260906 | policy | 24 | 0/2 | 0.0% |
| target_only | 20260906 | policy | 32 | 0/2 | 0.0% |

## 与规则上层的配对比较

以相同训练种子对应的相同测试开局配对；下表区间为局级配对 bootstrap，不能替代跨训练种子不确定性。

| 路线 | 种子 | 模型 | 配对局数 | 胜率差（百分点） | 95% 区间 |
|---|---:|---|---:|---:|---|
| b1_counts | 20260906 | policy | 10 | +10.0 | +0.0, +30.0 |
| b2_count | 20260906 | policy | 10 | +0.0 | +0.0, +0.0 |
| b2_local | 20260906 | policy | 10 | +10.0 | +0.0, +30.0 |
| b2_min | 20260906 | policy | 10 | +0.0 | +0.0, +0.0 |
| b3_global | 20260906 | policy | 10 | +0.0 | +0.0, +0.0 |
| b3_rebuild | 20260906 | policy | 10 | +0.0 | +0.0, +0.0 |
| grand | 20260906 | policy | 10 | +0.0 | +0.0, +0.0 |
| group_only | 20260906 | policy | 10 | +20.0 | +0.0, +50.0 |
| r1_ppo | 20260906 | best | 10 | +0.0 | +0.0, +0.0 |
| r1_ppo | 20260906 | initialized | 10 | +0.0 | +0.0, +0.0 |
| r1_ppo | 20260906 | latest | 10 | +0.0 | +0.0, +0.0 |
| r2_teacher_ppo | 20260906 | best | 10 | +0.0 | +0.0, +0.0 |
| r2_teacher_ppo | 20260906 | initialized | 10 | +0.0 | +0.0, +0.0 |
| r2_teacher_ppo | 20260906 | latest | 10 | +0.0 | +0.0, +0.0 |
| r3_ddqn | 20260906 | best | 10 | +0.0 | +0.0, +0.0 |
| r3_ddqn | 20260906 | initialized | 10 | +0.0 | +0.0, +0.0 |
| r3_ddqn | 20260906 | latest | 10 | +0.0 | +0.0, +0.0 |
| random | 20260906 | policy | 10 | +0.0 | +0.0, +0.0 |
| static_rule | 20260906 | policy | 10 | +10.0 | +0.0, +30.0 |
| target_only | 20260906 | policy | 10 | +0.0 | +0.0, +0.0 |

逐局数据位于各路线 seed 目录的 evaluation.jsonl；训练曲线位于 training_curves.png。
S2 的训练与留出审计位于 shared；模型训练成本与实际执行交互成本须分别报告。
