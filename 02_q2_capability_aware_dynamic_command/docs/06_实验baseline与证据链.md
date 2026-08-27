# 实验、baseline 与证据链

## 1. 研究问题

- **RQ1：**EGOM 是否在随机候选和 commander 真正选中的候选上都具有可靠的成功概率、伤亡和耗时预测？
- **RQ2：**同态势干预数据是否比自然日志训练显著降低 false-safe 和选择遗憾？
- **RQ3：**经校准的战果风险是否在相同下层、候选集和计算预算下改善最终关键点任务？
- **RQ4：**改善是否能迁移到未见兵力比例、增援时序、单位组合和对手策略？
- **RQ5：**双边博弈是否比只对固定敌方做 greedy/best response 更不容易被利用？
- **RQ6：**效果是否值得额外的决策延迟和重编次数？

## 2. 场景与数据划分

以下是 P0 默认值，第一周校准难度后冻结协议：

- 三个关键点，初始双方 6 或 8 个单位；
- 局部训练覆盖 1–4 vs 1–4；
- 一个 episode 中双方各有 0–2 波增援，每波 1–3 个单位；
- commander 固定周期 \(H\in\{10,20\}\)，关键事件可提前触发；
- 训练对手池含 focus-fire、kite、rush、balanced 和历史 learned checkpoint；
- 每个 candidate snapshot pair 用 16–32 个共同随机数 rollout 估计标签。

测试集按源生成因素整体 hold out：

| Split | 变化 | 目的 |
|---|---|---|
| ID | 训练范围内的新种子/位置 | 基本拟合与校准 |
| Scale-OOD | 初始 4、10、12 个单位或未见 4v8 / 8v4 比例 | 数量/比例泛化 |
| Event-OOD | 更早/更晚、两波连续或突发增援，不同死亡率 | 回合内开放事件 |
| Type-OOD | 留出一种单位组合或能力比例 | 构成泛化 |
| Geometry-OOD | 未见关键点间距、入口与遮挡 | 空间泛化 |
| Opponent-OOD | 完整留出算法/训练谱系，而非只留出 seed | 对手策略泛化 |
| Joint-OOD | Scale + Event + Opponent 同时变化 | 最困难实际测试 |

同一快照的候选分配、重复 rollout 和同谱系对手检查点必须放在同一 split，避免数据泄漏。

## 3. 最小充分 baseline

### 3.1 不学习的分配

1. **Static uniform：**开局平均分兵，之后不重编；
2. **Event-uniform：**事件后按人数重新平均；
3. **Nearest / threat-first：**按距离、关键点威胁和总血量分配；
4. **Force-ratio：**保持每条战线预设敌我优势比；
5. **Min-cost flow / Hungarian：**一对一或容量约束匹配。

### 3.2 学习 score / value 的结构化方法

6. **BVR-style XGBoost：**扁平统计量预测结果，再分配；
7. **Fu-style linear capability：**可加能力和任务阈值；
8. **Carion-style LP/QUAD：**学习 unary/pairwise score 后结构化求解；
9. **REDA-style additive value：**学习 assignment value 后优化；
10. **Uncalibrated critic：**MAPPO/allocator 的 scalar \(V/Q\) 直接作为收益。

### 3.3 层次与对抗方法

11. **ALMA-style AQL allocator：**共享冻结下层或按论文接口；
12. **HHMARL-style commander：**预训练低层 + 上层 target/task policy；
13. **PLATO/pointer-style MAPPO：**集合 encoder + pointer decoder，不输入 EGOM；
14. **Double-IC：**手工态势效用上的双方联盟博弈；
15. **Hierarchical self-play without EGOM：**与 proposed 相同上层容量和对手池，仅移除战果接口；
16. **Flat MAPPO/QMIX：**无显式 commander，作为非层次参照。

### 3.4 上下界与 proposed

17. **Monte Carlo oracle：**用大量在线 rollout 评估候选，只作小规模不可部署上界；
18. **COGAR-mean：**EGOM 预测均值 + 同一矩阵求解器；
19. **COGAR-risk：**完整校准、ensemble 风险、switch cost 和事件触发。

资源不足时不能随意删最关键对照。最终主表至少保留：force-ratio、flow/Hungarian、Carion 或 REDA、Double-IC、ALMA、same-capacity no-EGOM、COGAR 和 oracle。

## 4. 结果模型指标

### 概率质量

- Brier score、negative log-likelihood；
- AUROC/AUPRC 只作排序补充，不能代替校准；
- reliability diagram、adaptive ECE、calibration slope/intercept；
- 按兵力比、任务、关键点和对手分别分层报告。

### 连续战果

- 双方伤亡 MAE / RMSE；
- 耗时 MAE、截尾任务的 concordance 或分位数覆盖；
- 资产损伤/进度 MAE；
- 预测区间宽度与经验覆盖。

### 安全和选择指标

\[
\operatorname{FalseSafe}_\tau
=P(Y=0\mid \widehat P_{\rm lower}(Y=1)\ge\tau).
\]

\[
\operatorname{Regret}
=U_{\rm MC}(g^\star)-U_{\rm MC}(\hat g).
\]

还需报告 top-k regret、候选排序相关性以及**被 commander 最终选中方案**的 Brier/NLL/false-safe。随机候选上校准良好但选中候选过度自信，仍是方法失败。

## 5. 端到端指标

- 进攻方攻破至少两个关键点的概率 / 防守方守住至少两个的概率；
- 团队终局回报和最坏对手回报；
- 双方伤亡、资产损失、完成时间和资源效率；
- 伤亡/增援事件后的恢复时间，以及事件后回报/关键点控制 AUC；
- empirical best-response gain（固定预算代理，不能称精确 NashConv）；
- 分配切换数、单位平均转场距离、通信 token 数；
- 上层平均/p95 延迟、吞吐与效果—成本 Pareto。

## 6. 关键消融

| 消融 | 要回答的问题 |
|---|---|
| 自然日志 vs paired interventions | 提升是否来自主动覆盖未执行候选 |
| 单次标签 vs 16–32 重复 rollout | 分布标签是否必要 |
| 未校准 vs temperature scaling vs ensemble | 校准和认知不确定性的独立作用 |
| local-only vs full partition context | 战线外部性是否会破坏可加联盟价值 |
| mean vs LCB/CVaR | 风险策略是否减少 false-safe |
| 无 switch cost | 改善是否以不可接受抖动为代价 |
| 固定 \(H\) vs 每步 vs 事件触发 | 哪个重规划频率最有效 |
| 单一敌方 vs 对手池 vs 双边矩阵博弈 | 对抗训练/求解的实际贡献 |
| frozen vs naive joint vs alternating refresh | 阶段 4 是否值得保留 |
| true EGOM vs shuffled outputs | commander 是否真正利用正确能力信息 |

## 7. 统计协议

- 开发阶段 3 seeds，正式结果至少 5 个独立训练 seeds；
- 每个固定 matchup/event suite 至少 500 个评估 episode，或通过功效分析确定等价样本量；
- 对相同初始状态与随机种子做 paired evaluation；
- 主指标给 bootstrap 95% confidence interval 和 paired effect size；
- 多重 OOD 比较采用预先指定的主终点，避免事后只挑有利 split；
- 报告所有失败 run、超参数范围、wall-clock、GPU 型号与总仿真步数。

## 8. 预期结果不是虚构结果

当前没有 COGAR 实验结果。以下只是继续投入的 go/no-go 门槛：

- 校准后 ID 或 OOD ECE 小于 0.05，或相对未校准模型下降至少 20%；
- 选中候选的 false-safe 和 oracle regret 均显著下降；
- 至少两个未见兵力比上，最坏对手胜率/任务成功率提高约 5 个百分点，或规范化回报提高 8%；
- 固定预算 best-response gain 相对 no-EGOM 下降至少 15%；
- ID 性能下降不超过 2 个百分点；
- 上层 p95 延迟低于低层周期的 10%；
- full 优于同容量 no-EGOM，而不只是优于弱启发式。

这些门槛未通过时应降级路线，而不是把预期数字写成结果。

## 9. 主表模板

| Method | ID success ↑ | Scale-OOD ↑ | Opponent-OOD worst ↑ | False-safe ↓ | Regret ↓ | Event recovery ↓ | p95 ms ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Force-ratio | TBD | TBD | TBD | n/a | TBD | TBD | TBD |
| Carion-style | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Double-IC | TBD | TBD | TBD | n/a | TBD | TBD | TBD |
| ALMA-style | TBD | TBD | TBD | n/a | TBD | TBD | TBD |
| Same-capacity no-EGOM | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| COGAR-mean | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| COGAR-risk | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| MC oracle | TBD | TBD | TBD | 0-ish | 0 | TBD | very high |

## 10. 失败解释优先级

若 COGAR 不优于 baseline，依次检查：

1. 环境是否真的存在资源瓶颈；
2. 下层是否能执行所有任务，且不同编组有可分离结果；
3. 数据是否覆盖势均力敌边界，而非全是必胜/必败；
4. 是否发生同快照泄漏或对手谱系泄漏；
5. commander 是否利用模型外推高估；
6. 多战线外部性是否使局部结果不可加；
7. 候选生成是否先于模型把最优分配过滤掉。

只有定位失败原因后，才决定补数据、改模型或终止方向。
