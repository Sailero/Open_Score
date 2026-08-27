# Baseline protocol

本目录记录 baseline 的统一接口与公平性约束；当前不表示这些方法已经全部实现。

## 公平性原则

- 所有上层方法共享同一冻结下层 checkpoint；
- 使用相同全局观测、任务定义、活动单位 mask 和事件序列；
- 使用相同候选分配集与候选预算；
- learned 方法的训练步数、环境交互、超参数搜索次数和对手池预算等价；
- 同一评估 episode 使用配对初始状态和随机数；
- clean-room 复现要记录论文公式到代码模块的映射；
- oracle 只能作为上界，不能计入部署时延对比。

## 计划实现顺序

| ID | Baseline | 目的 | 状态 |
|---|---|---|---|
| B0 | Static / event-uniform | 最弱资源分配下界 | TODO |
| B1 | Nearest / threat / force-ratio | 强可解释启发式 | TODO |
| B2 | Capacity min-cost / Hungarian | 经典匹配 | TODO |
| B3 | XGBoost / linear capability | 普通结果预测与 Fu-style 能力 | TODO |
| B4 | Carion-style unary/pairwise + LP/QUAD | learned score + structured inference | TODO |
| B5 | Double-IC with hand-crafted utility | 双方联盟博弈核心对照 | TODO |
| B6 | ALMA-style AQL allocator | 顶会层次分配对照 | TODO |
| B7 | Pointer/MAPPO commander without EGOM | 同容量 learned commander | TODO |
| B8 | Flat MAPPO/QMIX | 无显式层次 | TODO |
| B9 | Uncalibrated scalar critic + same solver | 区分 value 与校准结果 | TODO |
| B10 | Monte Carlo rollout oracle | 小规模不可部署上界 | TODO |
| P | COGAR-risk | Proposed | TODO |

## 必做因果消融

- shuffled EGOM；
- natural logs instead of paired interventions；
- no calibration；
- no uncertainty penalty；
- no full-partition context；
- no switch cost；
- fixed-period vs event-triggered；
- frozen vs naive joint vs alternating refresh。

## 复现验收

每个实现都需包含最小单元测试、固定 seed smoke test、版本/许可证记录、默认配置和一个可比较的结果文件。若原论文代码无法运行，报告失败原因，不得用未经说明的简化版本冒充原方法。
