# Open-SCORE v2.0：已知对手下的动态分组

本项目研究：在对手完整规则已知、红方下层执行器冻结的条件下，如何利用当前物理态势与已有编组，学习成员减员后的长期分组决策。主方法 B4 依次选择需要释放的成员，再通过合法掩码重建编组，使用真实共享环境的最终成功回报进行 PPO 训练。

每局两个固定目标，最多 **50 个物理步**；每 **5 步及非终止减员事件**获得重决策机会。红方组容量至多 4，允许同目标多组及后备队。守住时限或消灭全部蓝方为成功；任一关键目标被毁为失败。终局成功奖励为 1，其余为 0，`gamma=1`。

## 先运行主算法

在本目录执行。当前机器可用解释器为 `D:\Software\Anaconda\envs\torch310\python.exe`；默认 Anaconda Python 没有 Torch。该环境已有 CUDA 版 Torch 及项目依赖。其他机器需要 Python 3.10 或更新版本，先安装适合硬件的 Torch，再执行 `python -m pip install -e ".[test]"`。

已交付验证结果及其限制见 [小预算验证报告](docs/results_v2.md)。这些结果用于检查训练和评估链路，当前不能支持算法优势。继续训练已交付的 B4 checkpoint：

```powershell
.\scripts\run_main_algorithm.ps1 -TrainMinutes 20 -TrainSteps 0 -EvalEpisodes 100 -EvalMinutes 4 -Output outputs/v2_validation/main_ready -Resume
```

该命令从已完成恢复验证的最新模型（2,204 个物理步、11 次更新）**新增 20 分钟训练**，之后自动评估；`TrainSteps 0` 表示只按时间限制，不设置额外交互步数上限。预计整条命令约 21–25 分钟，取决于实际决策和评估耗时；这是一段追加预算，不是收敛保证。任务时限始终为 50 步。

评估为训练前模型、训练后模型和“训练后相同初始编组、随后保持”的静态对照，每个策略计划 100 局。评估另有 4 分钟预算，只比较完整配对单元并公开实际完成数。旧步数的评估目录保留，训练日志追加；`latest.pt`、当前状态及最新报告索引更新。按需保存的编号快照用于回看历史模型。

要从头创建独立的小预算验证，使用新输出目录，最多训练 2,000 个物理交互步及 5 分钟，以先达到者为准：

```powershell
.\scripts\run_main_algorithm.ps1 -Python 'D:\Software\Anaconda\envs\torch310\python.exe' -TrainMinutes 5 -TrainSteps 2000 -EvalEpisodes 20 -Output outputs/v2_main
```

等价 Python 入口：

```powershell
& D:\Software\Anaconda\envs\torch310\python.exe -m open_score run --config configs/known_opponent_v2.yaml --profile briefing --method selective --steps 2000 --train-seconds 300 --eval-episodes 20 --output outputs/v2_main
```

已有输出目录继续训练时，在上述命令末尾增加 `-Resume` 或 `--resume`。恢复读取最近完成更新的模型和优化器，从新回合继续；未完成的采样批次不恢复。源代码、配置和冻结资产哈希必须一致。修改实现后使用新的输出目录，不能混用已有结果。

时间预算在安全计算边界检查，记录实际耗时和超时量，不能当作硬实时保证。后续训练规模与总预算根据诊断、吞吐和学习曲线确定；训练时间与评估时间分别记录。

`train`、`evaluate`、`diagnose`、`compare`、`report` 可单独运行，完整选项查看：

```powershell
& D:\Software\Anaconda\envs\torch310\python.exe -m open_score --help
```

## 三个种子、四个并发任务：B4 与 B6

以下专用脚本只训练主算法 `selective` 和规则释放 `rule`。三个种子共六个独立任务，最多同时运行四个；每个进程使用一个 CPU 计算线程。空槽自动启动后续任务，训练后按种子并行进行配对评估并汇总。

```powershell
& 'D:\Software\Anaconda\envs\torch310\python.exe' scripts/run_parallel_comparison.py --seeds 20260905 20260906 20260907 --workers 4 --steps 150000 --checkpoint-every 25000 --train-minutes 80 --eval-minutes 10 --eval-episodes 100 --total-minutes 180 --output outputs/v2_parallel_3seeds --resume
```

默认混合 8v8/12v12 训练，在 8v8、12v12、16v16 上每方法每种子计划评估 100 局。每局最多 50 步。B6 以规则选择释放成员，但重建器仍训练；红方下层与完整对手始终冻结。

当前 i7-14700KF、32 GB 内存机器的四进程短试跑测得每任务约 36–52 物理步/秒。15 万步是依据该吞吐和时间窗口选取的请求上限，不是收敛要求；按短跑速度外推整轮约 1.7–2.5 小时，尚未实测完整长训练。以完成步数为主，不在 120 分钟截停。每任务最多累计 80 分钟，整个调用以 180 分钟兜底，提前预留评估和收尾时间。达到调度截止时只停止该脚本启动的子进程；已完成更新保留。

每 25,000 步保存一个固定预算节点，节点之间从新回合继续。若有模型未到请求上限，所有模型统一使用共同完成的最高节点评估，例如全部采用 125,000 步节点；禁止把不同训练预算的最新模型直接混入对照。图中训练曲线也截至统一评估节点。单次宏观转移使实际步数最多多出 4 步，报告同时公开实际节点与请求上限。

`progress.json` 显示阶段、统一评估步数和完成情况；`logs/` 保存各任务日志，`scheduler_events.jsonl` 记录进程启动/结束。汇总见 `comparison_report.md`、`comparison_summary.json` 和 `paired_differences.json`，各种子的曲线及评估图自动生成。三种子的结果属于探索证据，不自动判定方法有效。`--resume` 补齐相同设置下未完成的任务，不重复训练已完成节点；如需改变预算或设置，使用新输出目录。`--dry-run` 可只显示配置。

当前 `batch_size: 16` 是一次 PPO 优化使用的上层事件数；每次采样 64 个上层事件。实现逐事件调用 `evaluate_action` 后合并损失，尚未把环境和变长解码整理为批量张量。因此增大该数值主要改变梯度统计和更新次数，不能保证 GPU 吞吐提升。128 维、2 层、4 头来自根目录 v2 方案的第一版建议；这些参数尚未调优，也不能保证大规模策略质量。

2026-09-06 在本机 RTX 5070 Ti 上对主算法进行约 1,024 步的顺序短测：batch 16 时 CPU 约 63 物理步/秒，上层 CUDA/下层 CPU 约 43，两者 CUDA 约 48；batch 64 时依次约 60、28、25。该测量是当前逐事件实现的短时吞吐，不是收敛比较或并行性能保证。复测速率可运行 `python scripts/benchmark_training_devices.py --output outputs/device_benchmark_new --steps 1024`，结果写入 `benchmark.json`，独立于正式训练输出。后续 GPU 优化应先批量化编码、采样和解码，再评估较大 rollout、minibatch 和网络的效果。

Windows 写入 checkpoint 或 JSON 时，对文件替换的 WinError 5/32/33 增加有限退避重试，始终保留上一份有效文件。仅对本次保存重试修复，`python scripts/repair_windows_checkpoint_run.py --output <旧并行输出目录>` 可迁移旧版运行后再 `--resume`；脚本限定精确的新旧源码哈希，备份原文件，并核对模型、优化器、随机状态和计数器全部不变。它不允许跨训练逻辑或配置变更恢复。

## 方法与验证范围

| 编号 | CLI 方法名 | 用途 |
| --- | --- | --- |
| B0 | `static` | 开局均衡分组，此后只删除死亡成员 |
| B1 | `dlom` | 已知蓝方分布下的 DLOM 期望代理搜索 |
| B2 | `full` | 与 B4 共用编码、修复、critic、PPO，每次全量重构 |
| B3 | `random` | 随机选择释放对象，使用共同学习型修复器 |
| B4 | `selective` | 学习选择释放成员和合法重建，主方法 |
| B5 | `alma` | ALMA 风格适配：候选生成、动作 Q 选择、重放 TD 与赢家轨迹训练 |
| B6 | `rule` | 基于受损组和空间邻近关系的规则释放，共用修复器 |

B5 是针对本项目接口的 **ALMA 风格适配**，不称为原论文的严格复现。B3 的正式归因比较需要使用训练数据估计并冻结与 B4 接近的释放人数分布；同时报告真实释放比例。B0 的静态规则比较不能代替“相同初始编组、是否允许后续调整”的配对诊断。

配置提供四种运行范围：`smoke` 检查端到端链路，`briefing` 优先主算法及有限评估，`minimal` 用于小规模对照和诊断，`formal` 用于正式协议。正式协议预留五个独立训练种子及每核心测试配置 200 回合；训练总物理步数根据最小验证确定并冻结。默认不自动启动正式全套实验。

开发阶段先使用已知 `reactive` 对手；`concentrated` 和 `balanced` 用于分别固定的复核实验。三个对手的规则、参数和 `rush` 下层均不在训练中更新。规则可以依据公开态势产生随机动作，红方不读取尚未公开的动作样本或未来随机状态。

## 结果怎样解读

输出目录保存初始化模型、最近完成更新的 checkpoint、训练/回合日志、评估与报告。训练与评估可分别执行，即使后续诊断尚未完成，也可读取已有完整回合和最新模型。

最小有效证据包括真实交互步数、完成更新数、参数变化、checkpoint 重载、实际完成回合数与成功数、决策耗时、释放比例、任务与队友关系变化、约束违规数和冻结资产哈希。有限评估遇到时间上限时，只将完整配对单元纳入比较，并公开完成数；未完成回合不当作失败或成功。

单种子最小验证只说明实现能训练和评估，不等价于收敛、不证明优于基线，也不替代 E1 组关系有效性、E2 候选与排序、E3 更新范围的诊断和正式多种子结果。实际指标以对应输出目录的运行报告为准，不借用旧实验胜率。

## 冻结资产与源码

`assets/frozen/lcl.pt` 是已有 REFIL-QMIX 下层；`assets/frozen/dlom.pt` 是已有局部结果模型。两者从备份提交对应的本地资产迁移，来源和 SHA-256 见 [资产清单](assets/frozen/manifest.json)。v2 不重新训练它们。

DLOM 的直接训练支持是 **1–4 红方对 1–4 蓝方、50 步任务时限**。它提供局部存活代理，未用重新编组反事实或全局长期结果训练；B1 的配对仅用于评分，物理执行仍是共享环境。下层支持变数量实体输入不保证在所有实际接触人数下都有良好能力，需要执行诊断。

源码主要位于 `src/open_score/grouping`，保留 HAD 物理环境和冻结推理所需模块。研究推导见 [根研究方案](../固定对手下动态分组研究方案_v2.md)，接口、训练与验收细节见 [method_v2.md](docs/method_v2.md)。

运行测试：

```powershell
& D:\Software\Anaconda\envs\torch310\python.exe -m pytest
```
