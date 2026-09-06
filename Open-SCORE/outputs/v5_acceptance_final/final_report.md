# v5.1 实验进度与结果

更新时间：2026-09-06T12:33:12.752434+00:00。
执行状态：全部完成；14 个预定主方法与对照臂，完整测试完成 14 臂。

**仅一个训练种子。执行完成和方法有效分别判定；训练、验证、最终测试分开报告。**
主表使用预先指定的末期模型或冻结策略，没有依据测试结果选择检查点。不同配比总胜率不能与旧 v4 等人数总胜率直接比较。

**本目录是短测/集成验证，不能作为正式有效性证据。**

| 方法 | 检查点 | 成功/局数 | 样本成功率 | 规模配比宏平均 | 等人数宏平均 | 全程未部署 | 测试完整 | 有效性状态 |
|---|---|---:|---:|---:|---:|---:|---|---|
| T1_rollout | latest | 1/3 | 33.33% | 33.33% | 0.00% | 0.00% | True | smoke_only |
| T2_mcts_dpw | latest | 1/3 | 33.33% | 33.33% | 0.00% | 0.00% | True | smoke_only |
| frozen_b1 | frozen | 1/3 | 33.33% | 33.33% | 0.00% | 0.00% | True | smoke_only |
| frozen_b3 | frozen | 0/3 | 0.00% | 0.00% | 0.00% | 0.00% | True | smoke_only |
| grand | frozen | 1/3 | 33.33% | 33.33% | 0.00% | 0.00% | True | smoke_only |
| rule | frozen | 1/3 | 33.33% | 33.33% | 0.00% | 0.00% | True | smoke_only |
| singleton | frozen | 0/3 | 0.00% | 0.00% | 0.00% | 0.00% | True | smoke_only |
| t3_autoregressive | latest | 0/3 | 0.00% | 0.00% | 0.00% | 0.00% | True | smoke_only |
| t3_candidate | latest | 0/3 | 0.00% | 0.00% | 0.00% | 0.00% | True | smoke_only |
| t4_exit | latest | 0/3 | 0.00% | 0.00% | 0.00% | 0.00% | True | smoke_only |
| t5_bridge_grouping | final_construction | 1/3 | 33.33% | 33.33% | 0.00% | 0.00% | True | smoke_only |
| t6_adv | final_epoch | 1/3 | 33.33% | 33.33% | 0.00% | 0.00% | True | smoke_only |
| t6_adv_gated | final_epoch | 1/3 | 33.33% | 33.33% | 0.00% | 0.00% | True | smoke_only |
| t6_bce | final_epoch | 0/3 | 0.00% | 0.00% | 0.00% | 0.00% | True | smoke_only |

未覆盖全部格子时，宏平均仅针对已有记录，不能当作完整矩阵成绩。

## 配对规则对照

| 方法 | 配对局数 | 比规则多赢/多输 | 差值（百分点） | 保守 95% 区间 |
|---|---:|---:|---:|---|
| T1_rollout | 3 | 0/0 | +0.00 | [-76.79, +76.79] |
| T2_mcts_dpw | 3 | 0/0 | +0.00 | [-76.79, +76.79] |
| frozen_b1 | 3 | 0/0 | +0.00 | [-76.79, +76.79] |
| frozen_b3 | 3 | 0/1 | -33.33 | [-93.40, +76.37] |
| grand | 3 | 0/0 | +0.00 | [-76.79, +76.79] |
| rule | 3 | 0/0 | +0.00 | [-76.79, +76.79] |
| singleton | 3 | 0/1 | -33.33 | [-93.40, +76.37] |
| t3_autoregressive | 3 | 0/1 | -33.33 | [-93.40, +76.37] |
| t3_candidate | 3 | 0/1 | -33.33 | [-93.40, +76.37] |
| t4_exit | 3 | 0/1 | -33.33 | [-93.40, +76.37] |
| t5_bridge_grouping | 3 | 0/0 | +0.00 | [-76.79, +76.79] |
| t6_adv | 3 | 0/0 | +0.00 | [-76.79, +76.79] |
| t6_adv_gated | 3 | 0/0 | +0.00 | [-76.79, +76.79] |
| t6_bce | 3 | 0/1 | -33.33 | [-93.40, +76.37] |

区间用 Bonferroni 组合两个 97.5% Clopper–Pearson 区间，只描述当前模型在测试开局上的差异，不能体现跨训练种子的波动。
`no_observed_gain`：完整配对结果没有正点估计；`inconclusive`：尚未完成、缺少对照或区间未支持正差；`observed_gain_single_seed` 仍只限当前训练种子。

## 分难度结果

| 方法 | 蓝/红比例 | 成功/局数 | 规模宏平均 | 完整 |
|---|---:|---:|---:|---|
| T1_rollout | 0.5 | 1/1 | 100.00% | True |
| T1_rollout | 0.75 | 0/1 | 0.00% | True |
| T1_rollout | 1 | 0/1 | 0.00% | True |
| T2_mcts_dpw | 0.5 | 1/1 | 100.00% | True |
| T2_mcts_dpw | 0.75 | 0/1 | 0.00% | True |
| T2_mcts_dpw | 1 | 0/1 | 0.00% | True |
| frozen_b1 | 0.5 | 1/1 | 100.00% | True |
| frozen_b1 | 0.75 | 0/1 | 0.00% | True |
| frozen_b1 | 1 | 0/1 | 0.00% | True |
| frozen_b3 | 0.5 | 0/1 | 0.00% | True |
| frozen_b3 | 0.75 | 0/1 | 0.00% | True |
| frozen_b3 | 1 | 0/1 | 0.00% | True |
| grand | 0.5 | 1/1 | 100.00% | True |
| grand | 0.75 | 0/1 | 0.00% | True |
| grand | 1 | 0/1 | 0.00% | True |
| rule | 0.5 | 1/1 | 100.00% | True |
| rule | 0.75 | 0/1 | 0.00% | True |
| rule | 1 | 0/1 | 0.00% | True |
| singleton | 0.5 | 0/1 | 0.00% | True |
| singleton | 0.75 | 0/1 | 0.00% | True |
| singleton | 1 | 0/1 | 0.00% | True |
| t3_autoregressive | 0.5 | 0/1 | 0.00% | True |
| t3_autoregressive | 0.75 | 0/1 | 0.00% | True |
| t3_autoregressive | 1 | 0/1 | 0.00% | True |
| t3_candidate | 0.5 | 0/1 | 0.00% | True |
| t3_candidate | 0.75 | 0/1 | 0.00% | True |
| t3_candidate | 1 | 0/1 | 0.00% | True |
| t4_exit | 0.5 | 0/1 | 0.00% | True |
| t4_exit | 0.75 | 0/1 | 0.00% | True |
| t4_exit | 1 | 0/1 | 0.00% | True |
| t5_bridge_grouping | 0.5 | 1/1 | 100.00% | True |
| t5_bridge_grouping | 0.75 | 0/1 | 0.00% | True |
| t5_bridge_grouping | 1 | 0/1 | 0.00% | True |
| t6_adv | 0.5 | 1/1 | 100.00% | True |
| t6_adv | 0.75 | 0/1 | 0.00% | True |
| t6_adv | 1 | 0/1 | 0.00% | True |
| t6_adv_gated | 0.5 | 1/1 | 100.00% | True |
| t6_adv_gated | 0.75 | 0/1 | 0.00% | True |
| t6_adv_gated | 1 | 0/1 | 0.00% | True |
| t6_bce | 0.5 | 0/1 | 0.00% | True |
| t6_bce | 0.75 | 0/1 | 0.00% | True |
| t6_bce | 1 | 0/1 | 0.00% | True |

每格分子、分母、Wilson 区间和配对差：[comparison_cells.csv](comparison_cells.csv)。

## 决策与计算成本

| 方法 | 逐事件 p50（ms） | 逐事件 p95（ms） | 时延样本 | 正式测试真实步 | 正式测试规划模拟步 |
|---|---:|---:|---:|---:|---:|
| T1_rollout | 483.569 | 2154.244 | 35 | 84 | 4007 |
| T2_mcts_dpw | 388.555 | 1329.381 | 29 | 65 | 1736 |
| frozen_b1 | 16.889 | 65.292 | 31 | 65 | 0 |
| frozen_b3 | 39.953 | 55.924 | 27 | 78 | 0 |
| grand | 0.545 | 1.366 | 31 | 63 | 0 |
| rule | 0.599 | 1.695 | 35 | 85 | 0 |
| singleton | 0.617 | 1.114 | 31 | 86 | 0 |
| t3_autoregressive | 5.282 | 10.763 | 35 | 86 | 0 |
| t3_candidate | 5.829 | 11.224 | 32 | 76 | 0 |
| t4_exit | 3.476 | 6.064 | 35 | 77 | 0 |
| t5_bridge_grouping | 2.425 | 8.370 | 32 | 62 | 0 |
| t6_adv | 12.742 | 24.581 | 35 | 85 | 0 |
| t6_adv_gated | 11.414 | 18.113 | 35 | 85 | 0 |
| t6_bce | 14.283 | 42.546 | 35 | 83 | 0 |

时延是运行时逐事件观测值，不能当作独占部署基准；未记录的时延留空。其余已记录训练、离线标签及验证/控制成本见 [costs.csv](costs.csv)。

串行部署测量：六任务工作进程全部退出后，在同一组预冻结状态上使用 CPU 单数值线程逐一测量。包含首次调用，排除环境恢复；外部系统负载只记录、未控制。
测量状态：已完成。

| 方法 | 串行 p50（ms） | 串行 p95（ms） | 串行样本 |
|---|---:|---:|---:|
| T1_rollout | 1088.855 | 1734.096 | 3 |
| T2_mcts_dpw | 626.706 | 1111.202 | 3 |
| frozen_b1 | 29.038 | 78.551 | 3 |
| frozen_b3 | 33.533 | 35.021 | 3 |
| grand | 0.623 | 0.985 | 3 |
| rule | 0.851 | 1.393 | 3 |
| singleton | 0.778 | 1.137 | 3 |
| t3_autoregressive | 6.286 | 8.393 | 3 |
| t3_candidate | 4.849 | 6.209 | 3 |
| t4_exit | 4.071 | 5.527 | 3 |
| t5_bridge_grouping | 4.406 | 6.781 | 3 |
| t6_adv | 11.405 | 14.383 | 3 |
| t6_adv_gated | 10.879 | 14.092 | 3 |
| t6_bce | 12.207 | 13.769 | 3 |

[串行部署原始测量](reports/serial_latency.json)。

## 六路线当前判断

[逐路线的已观察、已排除与仍无法区分](task_analysis.md)。

| 路线 | 执行 | 仍无法区分 |
|---|---|---|
| T1 | complete | 有限候选覆盖、有限分支误差与持续重规划效应仍需结合预算和单次/持续对照区分。 |
| T2 | complete | 搜索深度收益与真实模拟工作量不同；必须依据实际模拟步对照区分深度和预算作用。 |
| T3 | complete | 直接策略优化、候选覆盖和状态访问分布的影响，不能只凭 PPO 损失或两臂最终胜率区分。 |
| T4 | complete | 教师质量、蒸馏误差与学生状态分布变化仍需逐轮教师/学生配对验证区分。 |
| T5 | complete | 只在规则确定的目标和身份归属内学分区；不能据此判断目标分配学习是否有效。 |
| T6 | complete | 价值预测/排序误差、候选覆盖和门控作用仍需结合独立 verification 及三臂配对结果区分。 |

## 诊断、曲线与复现范围

[同状态独立复核](diagnostics.csv)、[预算诊断](budget_diagnostics.csv)、[复现状态](reproduction_status.csv)、[完整 JSON](comparison.json)。
- T1 预算与独立复核曲线：[查看](T1/budget_curves.png)
- T2 预算与独立复核曲线：[查看](T2/budget_curves.png)
- T3 训练与验证曲线：[查看](T3/training_curves.png)
- T3 分难度训练成功率：[查看](T3/success_by_difficulty.png)
- T4 训练与验证曲线：[查看](T4/training_curves.png)
- T5 训练与验证曲线：[查看](T5/training_curves.png)
- T6 训练与验证曲线：[查看](T6/training_curves.png)

损失下降、部署率提高或候选标签更好，均不能替代原生任务成功率证据。预算结果采用独立 verification 分支；T4 教师复核比较的是该轮冻结续行策略，不应统一标成相对规则。
