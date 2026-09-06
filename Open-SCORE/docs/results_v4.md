# v4.0 实现与验证记录

验证日期：2026-09-06。已实现共享规则底层、六条上层路线、局部与完整编组 S2、终局反事实采集、配对评估及恢复机制。本次完成了运行验证和小预算研究诊断，**尚未完成正式十万步训练，也没有证明 B3 优于局部 S2 或强规则上层**。

## 实现范围

- 红方所有路线使用同一 `rule_group_v1` 执行器；底层接受现有成员和目标指令，仅在组内分配拦截责任。v4 不加载 S1，没有四人上限、手工 24 候选配额、冷却或后备配额。
- 保持原生 50 步任务，每 5 步及非终止伤亡事件决策；蓝方完整规则已知。没有添加用于制造分组优势的容量限制或额外奖励。
- R1 是独立 actor/critic 的 PPO；R2 使用完整终局教师初始化后运行 PPO；R3 是 Double DQN。B1 枚举人数配置，B2/B3 使用相同分区搜索，分别由局部/全局 S2 评分。
- 全局 S2 标签为“候选执行至下一事件，随后固定规则续行到真正终局”；局部标签来自独立单目标规则环境。同回合族的状态和分支只属于一个数据划分。旧 S2 保留并单独报告覆盖范围。
- 模型和数据记录执行器、对手、续行版本及源码指纹。学习器恢复优化器、随机状态、采样状态及回放；预算可延长，并发数可调整。不同源码或实验协议需新输出目录。

正式网络规模：PPO actor 和 critic 各 **432,642** 参数，全局 S2 **664,706** 参数，局部 S2 **102,145** 参数。网络接受可变数量实体及组；支持 32 人运行不等于已证明大规模策略质量。

## 自动测试与完整链路

全量 `python -m pytest --junitxml=outputs/v4_validation/tests.xml`：**208 passed**，62.19 秒；一个来自 pygame 的依赖弃用警告。覆盖规则执行、分区可达性、同状态恢复、反事实终局与随机配对、回合族隔离、PPO/DDQN 更新、教师阶段隔离、数据版本以及调度恢复。

正式网络短试跑目录为 [full_scale_pilot](../outputs/v4_validation/full_scale_pilot/)。六路线和九个对照共 **15/15 个任务成功**；五种规模为 8/12/16/24/32，各策略版本每规模 2 局，共 **210 局独立测试、5,238 个测试物理步**。每条 RL 另完成 500 局独立选优验证，这些不计入独立测试局数。

| 路线 | 实际学习物理步 | 完成训练回合 / 成功 | 本次更新情况 | 选优验证 |
|---|---:|---:|---|---:|
| R1 PPO | 2,049 | 79 / 3 | 1 个 rollout；actor 32、critic 32 次优化 | 6/500 |
| R2 教师 PPO | 2,048 | 78 / 4 | 教师 1,000 次；随后 actor 1、critic 32 次优化 | 4/500 |
| R3 DDQN | 2,048 | 78 / 3 | 15 次 TD 更新 | 13/500 |

R2 的共享训练集有 54 个教师局面，其中仅 6 个存在非平局标签；重复优化次数不能当作独立教师样本数。首个 PPO rollout 出现 KL 提前停止，日志公开记录，没有继续用 critic 更新改变 actor。后续正式实验需要关注该现象。

这轮 PPO 仅有一个完整更新批次，优化图只有一个记录点；不能据此判断收敛。全局 S2 的训练 NLL 从约 0.419 降到 0.121，说明参数更新和拟合链路工作，但拟合训练标签不等于决策能力改善。

完整路线调度阶段首次耗时 **294.1 秒**，另有共享数据准备。保持相同协议并将 `--workers 6` 改成 `--workers 3 --resume` 后，**34 个检查点及训练/评估记录的 SHA-256 完全不变**，没有重复训练或重复计数。详见 [恢复验证](../outputs/v4_validation/resume_verification.json)。

## 第一关：分组是否影响真实结果

[扩展规则诊断](../outputs/v4_validation/rule_gate_extended/diagnostics.md) 在 4/8/16/32 规模各做 20 个开局，比较同目标大组、单体组、空间配对组及规则组，共 320 局动态执行；另从真实决策状态做 1,920 次配对终局续行。

**240 个同状态比较中，43 个在保持成员目标归属、仅改变组划分时出现不同终局结果。** 因此本规则执行器下的组关系不只是标签。该检查尚不能证明细分组优于大组，也不能证明学习方法优于规则。

| 规模 | 每目标大组 | 单体组 | 空间配对组 | 规则组 |
|---|---:|---:|---:|---:|
| 4 | 1/20 | 4/20 | 1/20 | 4/20 |
| 8 | 1/20 | 1/20 | 0/20 | 0/20 |
| 16 | 1/20 | 1/20 | 0/20 | 3/20 |
| 32 | 0/20 | 2/20 | 1/20 | 1/20 |

总体成功仍稀疏，规则底层本身的任务可行性和强度需要继续用独立对照评估。不能通过改动某一路的底层来获得上层优势。

## 第二关：新旧 S2 的客观比较

本次全局数据 30 个回合族，局部数据 50 个回合族；每个全局状态最多 6 个候选、2 个配对随机分支，最多采初始/伤亡/受威胁三类真实状态。全局采集使用 18,449 个物理步，局部 4,939 个物理步，**共享合计 23,388 步**，各拟合 8 个 epoch。

全局独立测试有 6 个回合族、18 个状态、105 个候选记录。下表在**完全相同的全局候选集合与终局标签**上比较，局部值乘积是待检验的独立性近似。

| 评估器 | Brier ↓ | ECE ↓ | 非平局排序正确 | 样本最佳方案 regret ↓ |
|---|---:|---:|---:|---:|
| 新局部 S2 乘积代理 | 0.09339 | 0.04154 | 9/13（69.2%） | 0.02778 |
| 新完整编组 S2 | 0.07697 | 0.06198 | 8/13（61.5%） | 0.02778 |

**B3 目前没有通过“排序优于局部 S2”这一关。** 完整编组 S2 的 Brier 较低，但 ECE 更高，排序少对一对，所选方案损失相同。18 个状态中 **16 个全部候选平局**，有效差异只来自 2 个状态；13 对候选不是 13 个独立回合。2 个分支也不足以精确估计真实成功概率，regret 的参照仅为采样候选中的经验最佳。

按规模报告同样必要：完整编组 S2 在 12:12 这一个测试回合族中的 Brier 约 **0.421**，总体小误差不能掩盖困难子集。局部独立单目标测试 Brier 为 **0.20051**，它与上表全局组合代理不是同一指标分布。

旧 S2 没有删除。本轮只有 **5/18 个完整候选池**的全部局部查询落在旧模型 1～4 人支持范围，才能做三方直接比较。这 5 个状态都为终局平局，无法评价排序；该子集的旧 S2 / 新局部 / 新全局 Brier 分别约 **0.14030 / 0.20937 / 0.19599**。既不能把不支持的大组截成四人后声称公平，也不能据此否定旧 S2 的原分布能力。

原始依据：[全局/局部/旧模型同候选比较](../outputs/v4_validation/full_scale_pilot/shared/seed_20260906/cross_evaluator_comparison.json)、[旧模型迁移报告](../outputs/v4_validation/full_scale_pilot/shared/seed_20260906/historical_transfer.json)、[全局 S2 报告](../outputs/v4_validation/full_scale_pilot/shared/seed_20260906/global_model/report.json)。

## 第三关：在线结果与下一步判断

本次各在线策略只有 10 局测试：B1 和 B2 各 1/10，B3 与规则上层均 0/10；静态规则 1/10，固定目标只改组 2/10，其余对照及三条 RL 的 best/latest/initialized 均为 0/10。完整分规模结果见 [比较报告](../outputs/v4_validation/full_scale_pilot/comparison_report.md) 和 [CSV](../outputs/v4_validation/full_scale_pilot/comparison_results.csv)。

这些结果说明链路可运行，不能支持方法优势。全失败的小样本配对 bootstrap 会退化为零宽区间，不代表总体差异被精确确定；正式报告应结合逐策略二项区间及多训练种子复现。当前优先事项是检查反事实候选的有效差异与分布覆盖，再看评价器能否稳定改进同状态排序。保持所有平局数据及原生终局目标，不默认用更多 PPO 步数解决。

## 算力、时间及搜索预算

本机 i7-14700KF（20 核 / 28 线程）、约 32 GB 内存、RTX 5070 Ti 16 GB；当前 Torch 环境可用 CUDA。每进程数值线程默认 1。

三任务/六任务 CPU 短测各重复两次，每任务 1,024 步：总吞吐分别为 **307.7～317.1** 和 **567.6～588.3 步/秒**，六并发约为三并发的 **1.85 倍**。这是小网络、短任务的实测，不能直接当作正式网络十万步速度。正式网络这轮三条 RL 在排除最终 500 局选优验证后，约 38～50 秒完成约 2,048 步；验证另用约 200～227 秒。

以此作为粗略预算参考：默认单种子 B3 加规则对照可先预留 **20～60 分钟**；六路线十万步加默认五规模测试可先预留 **1～3 小时**。这是短测外推，不是已完成的长跑测量；搜索预算、候选数、终局长度及并发干扰会影响实际耗时。默认每条 RL 的 6.5 小时是兜底预算，并不要求跑满。

补充 [搜索预算脚本](../outputs/v4_validation/search_budget_study.py) 在独立新开局上固定模型，使用 B2/B3 相同的 8/32/64 候选评价预算，以真实终局续行验证所选方案，并输出 [效率图](../outputs/v4_validation/search_budget_study/budget_curves.png) 和 [统计](../outputs/v4_validation/search_budget_study/summary.json)。该离线图的目标是 Q^rule，一次改进后固定规则续行，不是在线反复调用 B3 的胜率曲线。复现：

```powershell
& 'D:\Software\Anaconda\envs\torch310\python.exe' outputs/v4_validation/search_budget_study.py --help
```

这项补测覆盖五规模、10 个独立新开局、20 个真实决策状态，实际进行 230 次去重后的完整终局模拟，合计 5,521 个物理步。预算为 8/32/64 时，B2 终局成功均值均为 2.5%，B3 均为 0%，共同规则参考为 5%；B2 平均决策耗时为 24.9/46.7/72.1 ms，B3 为 28.2/52.0/71.0 ms。**本轮增加搜索预算只增加了耗时，没有改善真实续行结果。** 状态与分支有关联，这些均值仅作描述，不当作独立大样本显著性结论。

## 可直接运行的命令与产物

在 `E:\Code\Open_Score\Open-SCORE` 下，单独运行主方法和规则对照，准备数据阶段最多六个进程：

```powershell
& 'D:\Software\Anaconda\envs\torch310\python.exe' scripts/run_research_comparison.py run --routes b3_global --workers 6 --device auto --output outputs/v4_main
```

六路线首轮，一个训练种子，最多六个并行任务：

```powershell
& 'D:\Software\Anaconda\envs\torch310\python.exe' scripts/run_research_comparison.py run --seeds 20260906 --workers 6 --steps 100000 --device auto --output outputs/v4_comparison
```

上述 B3 默认采全局 120 / 局部 160 个回合族、每全局状态 8 候选、4 分支、最多 3 状态，各训练最多 40 epoch。`--steps` 只控制 RL 的物理交互预算；B3 不训练 PPO。默认每规模 100 局独立测试。需要全部消融时加：

```powershell
--controls rule grand static_rule random b2_count b2_min b3_rebuild group_only target_only
```

正式跨训练种子复现将 `--seeds` 改成 `20260905 20260906 20260907`；每个种子分别采集/拟合 S2，同种子六路线共享其数据。先按本方案完成单种子路线筛选，再运行这个更长的实验。

意外中断后原命令加 `--resume`。从头重训用新的输出目录。主要文件：

| 位置（相对输出目录） | 用途 |
|---|---|
| `comparison_report.md`、`comparison_results.csv`、`comparison_summary.json` | 六路线与规则的分规模、配对结果 |
| `scheduler_status.json`、`logs/` | 正在运行、排队、已完成及失败原因 |
| `shared/seed_<seed>/global_data/`、`local_data/` | 回合族反事实数据及分割、成本、版本 |
| `shared/seed_<seed>/global_model/`、`local_model/` | `best.pt`、`latest.pt`、训练曲线及校准报告 |
| `shared/seed_<seed>/cross_evaluator_comparison.json` | 相同全局候选上的排序、平局、regret 与旧模型覆盖 |
| `<route>/seed_<seed>/` | 学习检查点、原始训练和独立评估日志、轨迹及曲线 |

Git 保留源码、测试及精选可审计结果；大体积数据及检查点留在本机输出目录。用户修改的旧研究文档与历史成果保留。v4 的研究假设、最终结果及后续核查已归并到 [实验记录](../../实验记录.md)。

本次正式网络短试跑的复现命令如下；它是小预算验证，不是正式效果实验：

```powershell
& 'D:\Software\Anaconda\envs\torch310\python.exe' scripts/run_research_comparison.py run --workers 6 --device cuda --steps 2048 --train-hours 0.0833333333 --s2-families 30 --s2-local-families 50 --s2-candidates 6 --s2-branches 2 --s2-states 3 --s2-epochs 8 --scales 8 12 16 24 32 --eval-episodes 2 --search-budget 32 --controls rule grand static_rule random b2_count b2_min b3_rebuild group_only target_only --output outputs/v4_validation/full_scale_pilot_new
```
