# LEAF main1009：五种循环候选执行指南

2026-10-09。本文是代码与运行指南；实验结果只更新到[main1009 唯一正式报告](../Open-SCORE/outputs/main1009/实验报告.md)。15 个 1M 训练任务已全部完成，不要重复启动训练。8,100 个 final 和 120 个 best 离线机制快照已完成。本机原生校准与机制任务均已停止，部分记录保留，不能记为完成。校准、选型、原生机制、完整 OOD、自适应和反事实的完整配额未完成。

用户最新指令是：**全部实验都在另一台已配好 HAD 和多 GPU 的机器执行，本机不再启动对局。** 本机只准备代码与指南。不安装依赖、不重训、不修改模型或 HAD 物理接口。

离线诊断证明对应反馈被模型使用；完整回报、困难状态深轮收益及按需分配须由下面冻结协议验证。M1 至 M5 对应不同论文主题，完整讨论只保留在正式报告中。

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
| 原生机制 | 六配置，各 10 场，44000–44009；五方法、三个训练种子，fixed1/2/3/4 与方法专属反馈切断，总计 4,860 场 |
| 校准 | 四个训练 ID 配置，各 40 场，42000–42039；阈值 `{0.01,0.03,0.1,0.3,1}` |
| 选型 | 四个 ID 配置，各 100 场，40000–40099；与校准独立 |
| 完整评估 | 原 24 配置，各 300 场，9000–9299；固定 1/2/3/4 轮、自适应、预算匹配随机深度 |
| OOD 随机预算匹配 | 阈值冻结后，每配置 10 场，43000–43009；只用计算计数匹配，不用回报选择规则 |
| 反事实 | 五配置各 10 场；步数 0/5/10；每快照最多两个存活智能体，各自只改变当前决策深度为 1/2/4 |
| 独立确认 | 锁定候选与旧 LEAF/REFIL/TransfQMix；24 配置各 300 场，110000–110299 |

反事实配置固定为 `(4,4,2)`、`(10,10,3)`、`(30,30,2)`、`(50,50,2)`、`(30,30,12)`。恢复环境、随机流、策略 batch/hidden/cache 与必要 grouping 状态，其他智能体和未来决策均执行四轮。R4 直接复用同场景的事实轨迹最终 D，不重复运行；全轮最多 750 场事实轨迹及 9,000 条 R1/R2 干预尾段。以物理存活数量及最近攻击者竞争程度作独立物理复杂度指标，在同一 N/K 和快照步数内分层。缺失或提前终止的快照如实记录，不补充额外场景。

以 `D=-reward`、`delta(D)=max(0.01,0.02*D)` 为预先约定的实际意义容差：

1. 循环资格：四轮相对一轮的平均伤害下降超过 `delta(D1)`，至少两个训练种子改善，任何种子的恶化不超过对应容差。
2. 自适应资格：相对四轮伤害增加必须在容差内，并且矩阵计算量降低至少 10%，或实际预算与随机深度相差不超过 5% 且收益超过容差。
3. 在合格方法中按 ID 四轮伤害选择；差异在容差内时优先低计算成本，再比较参数量，仍相同按 M1/M4/M3/M2/M5。只选一个统一方法，不按种子挑赢家。
4. 方法与阈值锁定后公开所有 OOD 结果，不通过 OOD 重排候选。没有合格方法时确认性能最佳候选，但明确标记循环动机未验证。

同 checkpoint 深度比较验证冻结策略的深度效用，不能替代重新训练单轮的贡献对照。胜出方法的训练消融、参数匹配和额外 Set Transformer 等训练基线后置。完整故事还需要反事实证明复杂决策从深度中获益更多，以及自适应比预算匹配的随机分配更有效。

## 原生机制矩阵：在另一机器完整执行

另一机器继续使用原 HAD 接口，训练对应 `CORE_VERSION=had-workbench-2.1.0`、`PHYSICS_PROTOCOL=rebuild-calibrated-v3-r7-target-initialization`。本机曾临时恢复原源码以运行已停止的任务，没有修改 SkyHAD；本机进程退出后清理临时运行源码，正式权重与已落盘结果保留。

机制配置为 `(4,4,2)`、`(10,10,3)`、`(10,10,12)`、`(30,30,2)`、`(50,50,2)`、`(30,30,12)`；每配置使用场景 44000–44009，训练种子 0/1/2，正式 `final.pt`。所有方法运行 fixed1/2/3/4，再运行下面的 fixed4 切断条件。总计 90 个配置级分片、4,860 场完整闭环对局；M1/M3/M4 各 900 场，M2/M5 各 1,080 场。

| 方法 | 专属切断 |
|---|---|
| M1 | `query_update_frozen`：计算 query 更新后覆盖回 previous query |
| M2 | `query_update_frozen`；`reverse_kv_clamp`：实体端追加的 query K/V 固定为初始 b0，reader query 正常更新 |
| M3 | `query_update_frozen` |
| M4 | `entity_gru_hidden_reset`：实体 GRU hidden 改为初始 E0，仍保留当前注意力消息 |
| M5 | `slot_gru_hidden_reset`；`slot_competition_removed`：每个 slot 独立沿实体归一化，其他更新保留 |

切断仅作用于当前评估进程，不改模型文件或参数。M4/M5 的 reset 隔离直接 GRU 状态承接，注意力消息仍可传递上轮信息，因此不能称为切断全部循环。比较正常四轮与切断的伤害差 `Dcut−D4`，再与 fixed1/2/3/4 的伤害差共同判断各故事；动作或 hidden 敏感性不能替代奖励证据。

此阶段独立于 calibrate/select，结果写入同一 `main1009/loop_records.jsonl`、`split=mechanism`、协议 `leaf1009_native_feedback_v1`，**不进入正式阈值或方法选型**。十场/配置只是机制与初步外推证据，不能称为完成正式 300 场 OOD。另一机器不加 `--max-minutes`，按完整配额执行。中断后原指令重跑，已完成记录跳过；本机已有 partial 保留，不另建实验目录。

## 另一机器：复用已有权重完成全部正式评估

发布分支为 `leaf/main1009-dual-route`。本次 Git 同时提供代码和评估输入：main1009 的 15 个 `final.pt`、15 个 `best.pt`、配置、已有训练/验证与机制记录、图表和唯一报告，以及 main0928 的旧 NoMem LEAF、REFIL、TransfQMix 共 9 个 `final.pt` 和配置。采用普通 Git，拉取即获得这些文件，无需 Git LFS 或额外下载权重。恢复训练的 replay/resume 文件不属于本次评估输入。

在另一机器已有仓库根目录执行下面命令，继续使用已经配置好的原 HAD 环境。若该机器已有同路径的自有实验结果，先保留它们；使用另一份 checkout 接收本次发布，避免覆盖已有数据。不要用 `git reset --hard` 或清理 outputs 解决拉取冲突。

```bash
git fetch origin
git switch leaf/main1009-dual-route
git pull --ff-only origin leaf/main1009-dual-route
cd Open-SCORE
python scripts/leaf1009.py run --suite loop_candidates --devices all --jobs-per-gpu 8
```

默认读取当前 `Open-SCORE/outputs/main1009` 中已上传的权重，新结果也进入这个目录；旧基线默认从 `Open-SCORE/outputs/main0928` 只读加载。无需设置路径变量，不启动训练。普通 Git 下载量约 427 MiB 原始文件，实际传输量由 Git 压缩决定。

### 推荐：一条指令完成整轮评估

上述 `run` 就是整轮执行入口，没有 105 分钟或其他自动截止。它使用全部可见 GPU，每卡八个独立评估进程。先完成 ID calibrate/select 冻结及计算预算匹配，再把原生机制、候选固定深度、旧基线、自适应/随机及反事实放入同一个多 GPU 队列；完整配额成功后执行独立 confirm，最后更新统计图与唯一报告。失败或中断后停止后继阶段，重跑同一指令恢复；不按 OOD 改选方法。

如果希望关闭终端后继续运行，可在 `Open-SCORE` 中使用：

```bash
nohup python -u scripts/leaf1009.py run --suite loop_candidates --devices all --jobs-per-gpu 8 > outputs/main1009/evaluation.log 2>&1 &
tail -f outputs/main1009/evaluation.log
```

`tail` 只查看日志，退出查看不停止后台评估。`--devices` 使用当前 `CUDA_VISIBLE_DEVICES` 下的逻辑编号；默认 `all` 使用所有可见卡。每卡八进程是评估并发设置，可按显存与 CPU 负载调整 `--jobs-per-gpu`，不改变实验配额。

以下是需要手动分阶段调度时的等价指令；已经启动 `run` 时不要另启重复主控。

### 手动方式一：所有可见卡依次完成各阶段

使用全部可见 GPU，每卡八个独立评估进程。下面变量默认就是 Git 上传后的目录；只有复用其他存储位置时才需要改动。`set -e` 使任一阶段失败或中断后停止后继阶段；成功仍需确认日志的 `coverage.global_complete=true`。

```bash
set -e
export LEAF_OUTPUT="$(pwd)/outputs/main1009"
export LEAF_OLD_OUTPUT="$(pwd)/outputs/main0928"
python scripts/leaf1009.py calibrate --suite loop_candidates --output "$LEAF_OUTPUT" --devices all --jobs-per-gpu 8
python scripts/leaf1009.py select --suite loop_candidates --output "$LEAF_OUTPUT" --devices all --jobs-per-gpu 8
python scripts/leaf1009.py mechanism --suite loop_candidates --output "$LEAF_OUTPUT" --devices all --jobs-per-gpu 8
python scripts/leaf1009.py eval --suite loop_candidates --output "$LEAF_OUTPUT" --devices all --jobs-per-gpu 8
python scripts/leaf1009.py eval --suite references --output "$LEAF_OUTPUT" --source-output "$LEAF_OLD_OUTPUT" --devices all --jobs-per-gpu 8
python scripts/leaf1009.py adaptive --suite loop_candidates --output "$LEAF_OUTPUT" --devices all --jobs-per-gpu 8
python scripts/leaf1009.py counterfactual --suite loop_candidates --output "$LEAF_OUTPUT" --devices all --jobs-per-gpu 8
python scripts/leaf1009.py confirm --suite selected --output "$LEAF_OUTPUT" --source-output "$LEAF_OLD_OUTPUT" --devices all --jobs-per-gpu 8
python scripts/leaf1009.py report --output "$LEAF_OUTPUT"
```

### 手动方式二：ID 冻结后分卡并行

先在协调终端使用全部卡完成 calibrate 和 select，两阶段都成功且全配额覆盖后，再启动下面五项。每个终端进入同一个已更新的 `Open-SCORE` 并设置上述两项绝对路径。至少五张卡时使用五个互不重叠的卡组；四张卡时先完成较小的 mechanism，再启动其余四项。更多卡优先增加候选固定深度与 adaptive 的卡。例如八卡可分为候选 `0,1,2`、旧基线 `3`、反事实 `4`、adaptive `5,6`、mechanism `7`。

```bash
# 先在一个协调终端完成；失败时不进入下面四项并行任务。
set -e
python scripts/leaf1009.py calibrate --suite loop_candidates --output "$LEAF_OUTPUT" --devices all --jobs-per-gpu 8
python scripts/leaf1009.py select --suite loop_candidates --output "$LEAF_OUTPUT" --devices all --jobs-per-gpu 8
```

```bash
# 原生机制，五卡例用卡 4；四卡时在其余四项之前以全部卡完成。
python scripts/leaf1009.py mechanism --suite loop_candidates --output "$LEAF_OUTPUT" --devices 4 --jobs-per-gpu 8
```

```bash
# 终端 A：候选固定 1/2/3/4 轮，四卡例用卡 0。
python scripts/leaf1009.py eval --suite loop_candidates --output "$LEAF_OUTPUT" --devices 0 --jobs-per-gpu 8
```

```bash
# 终端 B：旧基线重评，四卡例用卡 1。
python scripts/leaf1009.py eval --suite references --output "$LEAF_OUTPUT" --source-output "$LEAF_OLD_OUTPUT" --devices 1 --jobs-per-gpu 8
```

```bash
# 终端 C：冻结自适应与实际预算匹配随机深度，四卡例用卡 2。
python scripts/leaf1009.py adaptive --suite loop_candidates --output "$LEAF_OUTPUT" --devices 2 --jobs-per-gpu 8
```

```bash
# 终端 D：单决策原生反事实，四卡例用卡 3。
python scripts/leaf1009.py counterfactual --suite loop_candidates --output "$LEAF_OUTPUT" --devices 3 --jobs-per-gpu 8
```

**等待全部五项返回成功，并逐项确认全配额完成，才执行下方 confirm/report。** 某一项失败或中断，只恢复该项；不要提前继续确认。共享锁防止多个进程重复领取同一分片，但一个主控退出不能证明其他服务器的全局配额已经完成。

```bash
set -e
python scripts/leaf1009.py confirm --suite selected --output "$LEAF_OUTPUT" --source-output "$LEAF_OLD_OUTPUT" --devices all --jobs-per-gpu 8
python scripts/leaf1009.py report --output "$LEAF_OUTPUT"
```

`--devices` 是当前 `CUDA_VISIBLE_DEVICES` 下的逻辑编号。不同终端若设置了不同的可见卡列表，应按各自列表重新编号，避免实际抢同一张卡。每卡八进程是评估并发默认值，不是重新训练四份模型。

### 完整配额、恢复与结果边界

| 阶段 | 完整配额 |
|---|---:|
| mechanism | 4,860 场，五方法的完整闭环深度与专属反馈切断 |
| calibrate | 21,600 场：5 方法 × 3 种子 × 4 ID 配置 × 40 场 × 9 臂 |
| select | 36,000 场：5 × 3 × 4 × 100 × 6 臂 |
| eval 候选 | 432,000 场：5 × 3 × 24 配置 × 300 × 4 深度 |
| eval 旧基线 | 64,800 场：3 方法 × 3 种子 × 24 × 300 |
| adaptive | 216,000 场；另有 15,000 场 OOD 计算预算数据：5 方法 × 3 种子 × 20 配置 × 10 场 × 5 计算臂，不用于观察奖励或选型 |
| counterfactual | 最多 750 场 factual4，另最多 9,000 条 R1/R2 干预尾段 |
| confirm | 129,600 场：锁定候选 fixed4/adaptive/random 加三个旧基线，3 种子 × 24 × 300 |

全部实验均安排到另一机器，不在本机启动。已有校准和机制部分记录保留；在同一输出目录重跑即可恢复。跨服务器调度必须共享同一个绝对 `--output`，存储支持文件锁；Linux 使用 `fcntl.flock`。锁随进程退出释放，不手动删除锁文件或维护心跳。相同分片已有完整记录会跳过；反事实部分场景可能重放 factual 前缀，未落盘尾段需重算。

单独恢复机制时，在另一机器执行下面命令，不加时限，不另起实验目录：

```bash
python scripts/leaf1009.py mechanism --suite loop_candidates --output "$LEAF_OUTPUT" --devices all --jobs-per-gpu 8
```

候选配额已冻结，不传 `--episodes` 或自定义反事实步数。`confirm --suite selected` 自动运行锁定候选及三个旧基线；不按 OOD 换赢家或替换 final。按场景聚类，三个训练种子分别报告。十场机制矩阵即使完成，也不能替代完整外推、state-level 反事实或预算匹配随机对照。

反事实从已有前向缓存记录合法 Q、planar/native 动作、与四轮的动作及 hidden 差、相对进入 hidden 的更新幅度；不增加 forward 或环境步。`physical_alive_count` 与 `observed_entity_count` 分开记录，旧 `visible_count` 兼容保留，不能称为真实观测实体数。距离和最近攻击者关系统一使用 XY 平面。收益包含当前动作和提交时序 hidden 的共同效应。

`report` 保留 `<!-- leaf1009_loop_candidates_v2 -->` 之前的研究分析，重生成后面的协议记录章节。实际各深度 D 图、回报–MAC 图和原生反事实收益图只有对应记录产生后才出现；离线动作差异图不能充当这些图。所有结果归 main1009 的同一份正式报告，不另建远端总结。

## 文献依据与执行状态

- [Recurrent Relational Networks](https://arxiv.org/html/1711.08028v2)：节点状态、消息、原输入的迭代更新；也展示部分任务单轮足够。
- [Perceiver](https://arxiv.org/abs/2103.03206)：迭代注意力与潜在状态。
- [Slot Attention](https://arxiv.org/abs/2006.15055)：竞争式注意力与 slot GRU。
- [GTrXL](https://arxiv.org/abs/1910.06764)：RL 中的门控与稳定化。
- [Universal Transformers](https://arxiv.org/abs/1807.03819)、[ACT](https://arxiv.org/abs/1603.08983)、[PonderNet](https://arxiv.org/abs/2107.05407)：共享循环及计算分配。本文的停止启发式不是这些方法的原样复现。

文献支持可检验设计，不证明 HAD 中必然有效。M2 私有 query 写回为本项目的组合设计。

离线验收使用已有 main0928 保存状态：seed 0、`(10,10,2)`、场景 9000、步数 0。旧 NoMem 严格加载原权重，重算 Q 与保存值最大误差 `9.54e-7`。五个新核心的 fixed4 与逐 observer 执行四轮、退化随机四轮均对齐，override2 与 fixed2 对齐；全 key 屏蔽和混合深度反传均为有限值。反传未执行优化器更新，以上只验证实现路径。

执行矩阵离线检查为五种不同 `leaf_loop_core`、种子 0/1/2、15 个训练任务、360 个配置级正式固定深度评估分片。本机曾成功导入原 HAD，原生任务现已停止；只有另一机器生成的实际完整记录和全局覆盖才能证明矩阵完成。

按新增要求完成 CPU smoke：真实 EntityMAC、ALMAAgent、QLearner 和 mixer 使用已有轨迹构成的 batch，每方法每次检查两次优化器更新；为核对新循环专属梯度，检查重复一次，总计临时 20 次 update。imagined 输出为 `[3,2,10,9]`。五个方法的 loss/梯度均有限，第二次新 `loop_*` 参数梯度绝对和依次为 `0.001270264 / 0.044881802 / 0.001506771 / 0.679230750 / 0.369145721`。online 与 target 使用同一归档深度；内存序列化恢复网络、优化器和深度随机流后，Q 最大差均为零。

策略接口检查使用原 FrozenPolicyAdapter 类定义，绕过本机不同 HAD 版本的初始化：eval pad 为 50/112，首 act 不替换模型或 args，同步重复 act 使用缓存，reset 保留 execution 和独立随机流。smoke 未生成正式模型、对局数据或保留检查代码；这些结果只证明数值执行与恢复语义。

共享调度的 CPU 多进程 smoke 已验证：两个真实进程争同一任务锁只有一个领取成功，释放后第三个进程可重新领取；两卡各四个 slot 共八个互不重合位置。并发 host 状态和实验元数据更新保留两方字段；配额不足不会提前冻结校准或选型文件。临时代码及测试目录已删除。

完整容量 smoke 在本机 RTX 5070 Ti 上，用已有状态构造 `B=32,T=101,A=10,E=23` 的 M4 batch。只逐层重算时峰值 allocated/reserved 为 `8.12/10.63 GiB`；改为按 observer chunk 重算完整循环之后，真实前反向及一次 RMSprop 更新的峰值为 `5.08/7.46 GiB`。batch、深度、参数及前向定义均保留。这是单任务实际测量；四任务的 reserved 简单合计约 `29.85 GiB`，供 48GB 卡的容量规划使用，不是远端四进程联合测量。

五个核心以相同参数和保存输入核对完整循环重算与直接计算，强制小块拼接后的读出最大差 `4.18e-7`、参数梯度最大差 `1.87e-9`，均在 `1e-6` 容差内。以上检查不改变原 HAD 环境接口。

执行状态：15 个正式训练任务均已完成，75,000 条训练验证记录已归并；离线机制主矩阵及 best/final 比较已完成，图与统计进入唯一报告。上述 smoke 属于训练前历史验收，不是本次新增训练。本机原生校准与机制均已停止并保留 partial，完整原生机制、正式深度、OOD、自适应与反事实尚未完成，全部按本指南在另一机器执行。运行时用 `git log -1 --oneline` 查看实际代码提交。
