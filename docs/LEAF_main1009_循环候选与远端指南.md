# LEAF main1009：五种循环候选与远端执行指南

2026-10-09。本文是代码与运行指南；实验结果统一更新到 `Open-SCORE/outputs/main1009/实验报告.md`。本机准备代码，使用既有保存状态执行离线验收及用户要求的临时 smoke；不启动正式训练或 HAD 对局，不安装依赖，不保留 smoke 代码。

## 本轮固定范围

五种候选，各从头训练 1,000,000 环境步，训练种子为 0/1/2：共 **15 个训练任务**。不增加旧方法重训、初始化扫描、锚定系数扫描或各候选的训练消融。所有新权重、恢复点、数据、日志和图片只进入 `Open-SCORE/outputs/main1009`；main0921/main0923/main0928 只读。

保留原 `had_env.core` 环境接口、动作、观测、物理规则和奖励。沿用 main0928 NoMem 的学习配置、REFIL 局部编码、下游时序 GRU 和 mixer。无需升级 HAD 或安装新包。

## 五种实际不同的循环结构

| ID | 注册名 | 循环的状态与反馈路径 | 最接近的旧变体及区别 |
|---|---|---|---|
| M1 | `regir_loop_nomem` | 实体上下文循环；本轮读取更新下一轮私有 query | 原 LEAF/refine 的各轮 reader 复用固定 query；新融合进入循环状态 |
| M2 | `regir_bidirectional_nomem` | M1 加私有 query→实体 K/V 的反向通路 | 旧 feedback/IAR 回注动作偏好或队友动作预测；本方法只追加当前 observer 的连续 query，不跨 observer 传递 |
| M3 | `regir_requery_nomem` | 实体上下文编码一次，缓存 K/V；只循环 query | KV0/ARR 虽固定初始 K/V，仍更新每个实体；本方法不反复执行实体自注意力 |
| M4 | `regir_entitygru_nomem` | 原观测与注意力消息驱动每个实体的本步 GRU | 原 entity cycle 是注意力/FFN 残差；下游 Temporal GRU 不是实体状态循环 |
| M5 | `regir_slotgru_nomem` | 四个自身条件 slots，通过竞争式注意力与 slot GRU 反复更新 | 旧 slot/K10 是普通 cross-MHA、slot self-attention 与 FFN；本方法改变归一化及状态更新 |

原先拟议的 query-GRU M2 暂缓，用双向反馈替换；不要同时启动第六种方法。早期 feedback、KV0 和 IAR 确实已有实验，不能把它们换名后再当成新方法。main1009 原有十四种 refine 配置保留兼容，但不在本轮训练矩阵内。

统一工作空间维度 64、四个注意力 heads、FFN 宽度 128，最多四轮、轮间共享参数。循环状态在每个环境步重新初始化，不直接接收跨时刻 GRU 历史。选中轮次的读取送入现有局部残差与时序 head。

M1/M2 的实体状态每轮以 `0.75 * Z_prev + 0.25 * E` 注入原观测，再执行稳定化 attention/FFN 更新。读取为 `u = Read(LN(s), Z)`，继续下一轮时：

```text
c = b0 + Wf LN(u)
g = sigmoid(Wg [LN(s), LN(c)])
s_next = (1-g) s + g c
```

M3 使用同一 query 更新，context 只形成一次。M2 中 `s` 仅作为当前 observer 的一个额外 K/V，实体 Q 仍来自真实实体；虚拟 token 不计入环境实体数，也不越过 imagined 分组传递信息。

M4 的实体 GRU 输入为 `[LN(E), LN(message)]`（128 维），hidden 为上轮实体状态（64 维），后接残差 FFN/LN，每轮重新屏蔽无效实体；自身读取 query 固定。M5 每个实体先在 slots 之间竞争，再对每个 slot 的实体权重归一化，聚合消息进入 slot GRU；没有额外 slot self-attention。四个 slots 使用不同可学习 seed 与共同自身条件，避免完全相同的初始化。

删除此前的固定 `0.5 * Wf` 增益。query 门权重/偏置零初始化使初始门为 `sigmoid(0)=0.5`，这是初始化选择；`0.25` 输入锚定及四个 slots 均是本轮冻结默认值，没有最优性保证。门控/LN 不能被表述为收敛或奖励提升的定理。

## 训练、提前停止与记录

训练时按每个智能体、每次环境决策独立均匀抽样 1/2/3/4 轮。深度进入 replay 的 `loop_depth:[B,T,A,1]`，包含 bootstrap 观测。online、target、Double Q 与 REFIL imagined 分支使用同一记录；独立深度随机流随恢复点保存，不消耗探索随机流。

正式使用 1M 的 `final.pt`，验证最优 `best.pt` 另外保留。沿用 50 次 ID 验证、每配置 25 场；不通过 OOD 选择 checkpoint。短出口训练并不保证深轮收益，深度收益属于筛选条件。

推理从第二轮起检查：合法贪心动作相同，合法中心化 Q 的单位向量变化与候选时序 hidden 的相对变化都小于阈值；最多四轮。零优势和近并列动作使用数值保护。各轮候选时序 hidden 都从同一个进入本步的 hidden 计算，最后只提交选中状态一次。退出的 observer 不再计算后续循环。

同时记录循环深度、attention 矩阵规模、读出/GRU 调用、矩阵乘加计数及正常评估耗时。矩阵乘加计数有明确覆盖范围，不等同于全部 FLOPs；不能仅凭平均轮数宣称总计算节省。

## 校准、完整评估与论文主线选择

| 阶段 | 场景与配额 |
|---|---|
| 校准 | 四个训练 ID 配置，各 40 场，42000–42039；阈值 `{0.01,0.03,0.1,0.3,1}` |
| 选型 | 四个 ID 配置，各 100 场，40000–40099；与校准独立 |
| 完整评估 | 原 24 配置，各 300 场，9000–9299；固定 1/2/3/4 轮、自适应、预算匹配随机深度 |
| OOD 随机预算匹配 | 阈值冻结后，每配置 10 场，43000–43009；只用计算计数匹配，不用回报选择规则 |
| 反事实 | 五配置各 10 场；步数 0/5/10；每快照最多两个存活智能体，各自只改变当前决策深度为 1/2/4 |
| 独立确认 | 锁定候选与旧 LEAF/REFIL/TransfQMix；24 配置各 300 场，110000–110299 |

反事实配置固定为 `(4,4,2)`、`(10,10,3)`、`(30,30,2)`、`(50,50,2)`、`(30,30,12)`。恢复环境、随机流、策略 batch/hidden/cache 与必要 grouping 状态，其他智能体和未来决策均执行四轮。R4 直接复用同场景的事实轨迹最终 D，不重复运行；全轮最多 750 场事实轨迹及 9,000 条 R1/R2 干预尾段。以可见数量及最近攻击者竞争程度作独立物理复杂度指标，在同一 N/K 内分层。缺失或提前终止的快照如实记录，不补充额外场景。

以 `D=-reward`、`delta(D)=max(0.01,0.02*D)` 为预先约定的实际意义容差：

1. 循环资格：四轮相对一轮的平均伤害下降超过 `delta(D1)`，至少两个训练种子改善，任何种子的恶化不超过对应容差。
2. 自适应资格：相对四轮伤害增加必须在容差内，并且矩阵计算量降低至少 10%，或实际预算与随机深度相差不超过 5% 且收益超过容差。
3. 在合格方法中按 ID 四轮伤害选择；差异在容差内时优先低计算成本，再比较参数量，仍相同按 M1/M4/M3/M2/M5。只选一个统一方法，不按种子挑赢家。
4. 方法与阈值锁定后公开所有 OOD 结果，不通过 OOD 重排候选。没有合格方法时确认性能最佳候选，但明确标记循环动机未验证。

同 checkpoint 深度比较验证冻结策略的深度效用，不能替代重新训练单轮的贡献对照。胜出方法的训练消融、参数匹配和额外 Set Transformer 等训练基线后置。完整故事还需要反事实证明复杂决策从深度中获益更多，以及自适应比预算匹配的随机分配更有效。

## 远端拉取与运行

轻量发布分支为 `leaf/main1009-lite`，运行代码与 `leaf/main1009-dual-route` 的 `dc04965` 相同。此分支只有一次无父提交，不携带旧实验、权重、论文、网页和 Git 历史；HAD 不使用的 SC2 地图也已去掉。训练、校准、选型、全部评估和图表报告生成均保留。原 HAD 环境接口和依赖保持不变，继续使用远端已配置好的环境。

从你希望保存项目的父目录运行下面命令。`LEAF1009-lite` 是新建的代码目录；原目录和已有结果保留。共享存储只需一台服务器克隆一次，其他服务器进入同一个目录。

```bash
git clone --depth 1 --single-branch --branch leaf/main1009-lite https://github.com/Sailero/Open_Score.git LEAF1009-lite
cd LEAF1009-lite/Open-SCORE

python scripts/leaf1009.py plan --suite loop_candidates --steps 1000000 --seeds 0,1,2
python scripts/leaf1009.py train --suite loop_candidates --steps 1000000 --seeds 0,1,2 --devices all --jobs-per-gpu 4 --resume

python scripts/leaf1009.py calibrate --suite loop_candidates --devices all
python scripts/leaf1009.py select --suite loop_candidates --devices all
python scripts/leaf1009.py eval --suite loop_candidates --devices all
python scripts/leaf1009.py eval --suite references --devices all --source-output /absolute/path/to/main0928
python scripts/leaf1009.py adaptive --suite loop_candidates --devices all
python scripts/leaf1009.py counterfactual --suite loop_candidates --devices all
python scripts/leaf1009.py confirm --suite selected --devices all --source-output /absolute/path/to/main0928
python scripts/leaf1009.py report
```

上面的 `/absolute/path/to/main0928` 必须替换成已有旧基线所在目录。五个新候选的训练与候选评估不需要旧数据。只有 `eval --suite references` 与 `confirm --suite selected` 需要旧 LEAF (`regir_nomem`)、REFIL (`refil`) 和 TransfQMix (`transfqmix`) 的种子 0/1/2 权重及对应配置/完成记录。轻量仓库不打包这些大权重，也不新增旧基线训练。

轻量版以后更新代码时，在仓库根目录执行 `git pull --ff-only origin leaf/main1009-lite`；无需切换回大分支或下载全部分支。

输出目录由运行指令生成，所有阶段始终合并写入一个 `Open-SCORE/outputs/main1009`（或显式指定的共享 `main1009`）：

```text
outputs/main1009/
  experiment.json, runner_state.json
  had/<method>/train/seed_<0|1|2>/
    config.json, final.pt, best.pt, resume.pt, 日志与训练记录
  loop_records.jsonl, loop_summary.csv, summary.json
  calibration.json, selection.json, budgets.json
  had/figures/*.png
  实验报告.md
```

权重、完整数据和图表在对应阶段实际完成后生成；克隆时不会预置未训练的结果。若已在旧代码目录开始训练，新代码的**每个阶段**都加 `--output /已有绝对路径/main1009`，训练再加 `--resume`，继续使用同一份进度。

`--devices all` 使用 `CUDA_VISIBLE_DEVICES` 下全部可见卡，`--jobs-per-gpu 4` 让每张 **48GB 魔改 4090** 同时执行四个独立任务。也可指定 `--devices 0,1,2` 等可见逻辑编号；本轮所有服务器合计只有 15 个训练任务。评估按方法、种子、配置分片。

多台服务器使用同一个共享仓库及 `outputs/main1009`，每台在自己的已配置环境中运行下面训练命令。进程持有共享任务锁，其他服务器跳过已领取任务；权重完成后也会跳过。某台服务器退出不能代表全体 15 个任务均已完成。以下命令中，`hostname` 只区分主控日志，不创建结果副本。

如果代码在各服务器的独立 checkout 中，共享输出路径必须相同：在各阶段指令加 `--output /shared/path/main1009`。共享存储需支持文件锁；Linux 使用 `fcntl.flock`，本机 Windows smoke 使用 `msvcrt`。锁在进程退出时释放，不需要手动领取或维护心跳。完成任务在主控末尾显示 `coverage.complete/total`；`global_complete=true` 且 `15/15` 才进入后续阶段。

```bash
mkdir -p outputs/main1009
nohup python scripts/leaf1009.py train --suite loop_candidates --steps 1000000 --devices all --jobs-per-gpu 4 --resume >> "outputs/main1009/train_$(hostname).log" 2>&1 &
```

`--resume` 可用于首次启动和恢复；全新任务正常开始，已有恢复点继续执行。所有 15 个 final 通过完成判定之后，再在任意一台协调服务器执行校准、选择和后续评估。可将上述训练命令后的各阶段用 `&&` 连起来；某阶段失败立即停止。不要让各服务器各自把训练与后续阶段连成一条命令，以免其他服务器的权重尚未完成。

后续阶段读取已完成的 final 权重，不重训旧基线；主控运行日志和方法种子日志均留在同一个 main1009 目录。

候选各阶段配额已经冻结，不能传 `--episodes` 或自定义反事实步数。`confirm --suite selected` 自动运行锁定候选的 fixed4/adaptive/random 和三个旧基线。旧权重默认读取 `outputs/main0928`。若旧结果位于其他位置，给 references 和 confirm 阶段加 `--source-output /absolute/path/to/main0928`。训练默认内置冻结配置，无需复制旧输出目录；显式 `--base-config` 只读加载原 NoMem config 并记录来源。

训练中按 Ctrl+C 请求完整 batch 边界保存，之后在原训练指令加 `--resume`。评估原指令重跑，完整记录跳过。不要改变同一方法已经开始运行的架构或预算；需要中断时等待恢复点落盘。完成和失败必须按实际权重/记录判定，不把旧 failure 文件直接当最终状态。

## 文献依据与执行状态

- [Recurrent Relational Networks](https://arxiv.org/html/1711.08028v2)：节点状态、消息、原输入的迭代更新；也展示部分任务单轮足够。
- [Perceiver](https://arxiv.org/abs/2103.03206)：迭代注意力与潜在状态。
- [Slot Attention](https://arxiv.org/abs/2006.15055)：竞争式注意力与 slot GRU。
- [GTrXL](https://arxiv.org/abs/1910.06764)：RL 中的门控与稳定化。
- [Universal Transformers](https://arxiv.org/abs/1807.03819)、[ACT](https://arxiv.org/abs/1603.08983)、[PonderNet](https://arxiv.org/abs/2107.05407)：共享循环及计算分配。本文的停止启发式不是这些方法的原样复现。

文献支持可检验设计，不证明 HAD 中必然有效。M2 私有 query 写回为本项目的组合设计。

离线验收使用已有 main0928 保存状态：seed 0、`(10,10,2)`、场景 9000、步数 0。旧 NoMem 严格加载原权重，重算 Q 与保存值最大误差 `9.54e-7`。五个新核心的 fixed4 与逐 observer 执行四轮、退化随机四轮均对齐，override2 与 fixed2 对齐；全 key 屏蔽和混合深度反传均为有限值。反传未执行优化器更新，以上只验证实现路径。

执行矩阵离线检查为五种不同 `leaf_loop_core`、种子 0/1/2、15 个训练任务、360 个配置级正式固定深度评估分片。正式 HAD 包的导入和对局留给已配置原环境的云端。

按新增要求完成 CPU smoke：真实 EntityMAC、ALMAAgent、QLearner 和 mixer 使用已有轨迹构成的 batch，每方法每次检查两次优化器更新；为核对新循环专属梯度，检查重复一次，总计临时 20 次 update。imagined 输出为 `[3,2,10,9]`。五个方法的 loss/梯度均有限，第二次新 `loop_*` 参数梯度绝对和依次为 `0.001270264 / 0.044881802 / 0.001506771 / 0.679230750 / 0.369145721`。online 与 target 使用同一归档深度；内存序列化恢复网络、优化器和深度随机流后，Q 最大差均为零。

策略接口检查使用原 FrozenPolicyAdapter 类定义，绕过本机不同 HAD 版本的初始化：eval pad 为 50/112，首 act 不替换模型或 args，同步重复 act 使用缓存，reset 保留 execution 和独立随机流。smoke 未生成正式模型、对局数据或保留检查代码；这些结果只证明数值执行与恢复语义。

共享调度的 CPU 多进程 smoke 已验证：两个真实进程争同一任务锁只有一个领取成功，释放后第三个进程可重新领取；两卡各四个 slot 共八个互不重合位置。并发 host 状态和实验元数据更新保留两方字段；配额不足不会提前冻结校准或选型文件。临时代码及测试目录已删除。

完整容量 smoke 在本机 RTX 5070 Ti 上，用已有状态构造 `B=32,T=101,A=10,E=23` 的 M4 batch。只逐层重算时峰值 allocated/reserved 为 `8.12/10.63 GiB`；改为按 observer chunk 重算完整循环之后，真实前反向及一次 RMSprop 更新的峰值为 `5.08/7.46 GiB`。batch、深度、参数及前向定义均保留。这是单任务实际测量；四任务的 reserved 简单合计约 `29.85 GiB`，供 48GB 卡的容量规划使用，不是远端四进程联合测量。

五个核心以相同参数和保存输入核对完整循环重算与直接计算，强制小块拼接后的读出最大差 `4.18e-7`、参数梯度最大差 `1.87e-9`，均在 `1e-6` 容差内。以上检查不改变原 HAD 环境接口。

执行状态：代码及本机临时 smoke 已完成；本轮 15 个正式任务未在本机启动。运行时用 `git log -1 --oneline` 查看拉取的具体提交。真实收益、循环价值及论文主线由云端结果判定。
