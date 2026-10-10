# LEAF main1009：原 NoMem 深度补评与历史循环候选指南

更新：2026-10-10。另一机器的 **1009_eval 整轮评估已完成并合并**，包括校准、选型、原生机制、正式固定深度与旧基线、自适应、反事实及独立确认。15 个 1M 训练任务、8,100 个 final 和 120 个 best 离线机制快照也已完成。唯一正式报告为 [LEAF_main1009_完整分析报告.md](LEAF_main1009_完整分析报告.md)；本文件只保留运行指南。原有整轮结果已完成，**无需重跑 `run`、`calibrate`、`select`、`train` 或其他旧阶段**。

全部新增对局继续在原 1009_eval 机器的既有 HAD、多 GPU 环境执行，本机只更新代码与文档。不安装依赖、不重训、不修改模型权重或 HAD 物理接口。本次只补原 NoMem 的固定深度评估，不扩展 M1–M5 的运行深度。

## 本次已获批：原 NoMem R1–R6 完整补评

| 项目 | 冻结范围 |
|---|---|
| 唯一方法 | 原 main0928 NoMem LEAF：`regir_nomem`，沿用三个正式 1M `final.pt` |
| 深度 | 固定 R1、R2、R3、R4、R5、R6；每场从初始状态执行同一固定深度 |
| 配置 | 原正式 eval 的全部 24 配置，沿用 `FINAL_CONFIGS`，不增减 N/K 组合 |
| 训练种子 | 0、1、2，不挑选单个种子 |
| 场景 | 每个方法×深度×配置×训练种子使用 9000–9299，各 300 场 |
| 完整矩阵 | 6 深度 × 24 配置 × 3 种子 × 300 = **129,600 场** |
| 默认复用 | 当前 main1009 正式 eval 中原 NoMem 的 R4，**21,600 场** |
| 新增对局 | R1、R2、R3、R5、R6，共 **108,000 场**；不重复运行已复用的 R4 |
| 权重来源 | 原机器现有 `Open-SCORE/outputs/main0928`，只读加载 |
| 结果位置 | 原机器现有 `Open-SCORE/outputs/main1009`，保留全部原始记录、已完成进度与权重 |

原 NoMem 的共享实体循环和 learned JK 可用同一 checkpoint 执行六轮，无需增加模型参数。R5/R6 超过原训练最大四轮，属于**推理深度外推**；learned JK 的融合归一化集合也从 R1–R4 扩展到 R1–R5/R6，所以差异同时包含实体处理与融合候选集合变化，不能只归因于最后两轮。该补评不重新选型、不重校准阈值，不改变已有 M4 故事或确认结论。

### 原机器更新代码与两份 docs，保留 outputs 和权重

在原 1009_eval checkout 的仓库根目录执行以下命令。只恢复列出的三个文件，不切换分支、不执行 `git pull` 或 `git reset --hard`，不清理或 stash outputs；即使 outputs 已被 Git 跟踪且存在本机修改，也保留原内容。这三个代码/文档文件自身的本地修改会被替换，若有独有改动先合并。

本次 Git 发布代码、文档和可读小结果，不发布约 680 MB 的原始 JSONL；下面的恢复命令只更新代码与两份 docs，不将发布结果覆盖到原机器 outputs。

```bash
git fetch origin leaf/main1009-dual-route
git restore --source=FETCH_HEAD --worktree -- \
  Open-SCORE/scripts/leaf1009.py \
  docs/LEAF_main1009_循环候选与远端指南.md \
  docs/LEAF_main1009_完整分析报告.md
cd Open-SCORE
export LEAF_OUTPUT="$PWD/outputs/main1009"
export LEAF_OLD_OUTPUT="$PWD/outputs/main0928"
```

沿用原机器已配置好的 Python/HAD 环境。`LEAF_OLD_OUTPUT` 指向原权重来源，`LEAF_OUTPUT` 指向含本机完整 1009_eval 记录的输出目录；不另建实验输出目录。

### 先查看计划，再运行完整补评

新入口是 **`depth-eval`**，不是历史 `depth`。先用 dry-run 查看 24 配置完整矩阵和 R4 复用覆盖，不启动对局：

```bash
python scripts/leaf1009.py depth-eval --methods regir_nomem --depths 1,2,3,4,5,6 --source-output "$LEAF_OLD_OUTPUT" --output "$LEAF_OUTPUT" --devices all --jobs-per-gpu 8 --dry-run
```

正式执行相同命令，去掉 `--dry-run`：

```bash
python scripts/leaf1009.py depth-eval --methods regir_nomem --depths 1,2,3,4,5,6 --source-output "$LEAF_OLD_OUTPUT" --output "$LEAF_OUTPUT" --devices all --jobs-per-gpu 8
python scripts/leaf1009.py report --section nomem-depth --output "$LEAF_OUTPUT"
```

使用所有可见 GPU，每卡八个独立评估进程。无需 `--episodes`、`--resume` 或 `--max-minutes`；24 配置、三个种子和每配置 300 场按本次协议执行。完成资格以完整矩阵覆盖为准，不能把单机队列退出或部分落盘结果描述为全部完成。

`report --section nomem-depth` 只更新唯一正式报告的 **3.10 原 NoMem 深度补评章节**及对应结果图，不重写前文比较或末尾 M4 方法故事；不调用历史无 section 的整轮报告命令。原有其他实验结果继续保留。

### 后台执行与断点继续

关闭终端后仍需运行时，在已设置上述路径变量的 `Open-SCORE` 目录执行：

```bash
nohup python -u scripts/leaf1009.py depth-eval --methods regir_nomem --depths 1,2,3,4,5,6 --source-output "$LEAF_OLD_OUTPUT" --output "$LEAF_OUTPUT" --devices all --jobs-per-gpu 8 >> "$LEAF_OUTPUT/nomem_depth_eval.log" 2>&1 &
tail -f "$LEAF_OUTPUT/nomem_depth_eval.log"
```

`tail` 只查看日志，退出查看不会停止评估。中断后在同一代码、权重来源和输出目录重跑同一正式命令，或重跑同一后台命令；已完成记录跳过，未完成进度继续，无需额外 `--resume` 或时限参数。不要同时启动重复补评主控。全配额完成后再执行上面的 `report --section nomem-depth`。

## 之前整轮指南（历史协议，当前无需重跑）

**以下所有章节保留此前整轮实施、验收和多机调度的历史说明，其中“未完成”“本机已停止”等状态是当时记录，不代表当前完成状态。旧报告路径与无 section 的 report 命令也属于历史；当前唯一报告和本次补评入口以上文为准。以下 `run`、训练、校准及整轮队列命令不属于本次补评。**

## 历史整轮固定范围

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

### 实时查看任务进度与整轮 ETA

代码放在现有 `Open-SCORE/scripts/leaf1009.py`，无需新增脚本、安装包或重启评估。在同一代码目录的另一个终端执行：

```bash
conda activate sarc
cd /inspire/hdd/project/urbanlowaltitude/fengkairui-25026/shenqili/TP/LEAF1009-lite/Open-SCORE
python -B scripts/leaf1009.py progress --output outputs/main1009 --devices all --jobs-per-gpu 16 --watch-seconds 30
```

终端每 30 秒在固定屏幕内刷新，默认只显示整体总览，不列出 PID 或逐方法/种子/场景细节。总览包括运行任务数、并发上限、含部分完成的整体任务进度、完整任务数、各阶段完成场数/百分比/完整任务数/运行数/近期速度 ETA，以及整轮剩余时间和北京时间预计结束点。整体任务进度按各分片完成比例等权平均，不等同于剩余时间比例。整个协议共有 1,809 个任务分片；选型完成前，确认阶段用待选方法占位，不预先指定赢家。反事实的进度单位是完整场景，其他阶段是 episode。界面使用终端独立屏幕缓冲区并按窗口高度、字符宽度裁剪，避免长输出向下滚动；退出后恢复原终端。

查看器只读已有记录和 Linux `/proc` 的任务日志描述符，不加载策略、不运行 HAD、不写文件。逐场记录增量读取，不反复重扫整份记录。运行主控存在时，自动读取其每卡并发与设备配置；没有运行主控时，使用命令中的设置做预算估算。`Ctrl+C` 只退出查看器。只看一次可将 `--watch-seconds` 改为 `0`。

整轮 ETA 按任务依赖图和本机 GPU slot 数估算：机制与旧基线不等待校准；选型与预算统计依赖校准；候选固定深度与反事实依赖选型；自适应和确认还依赖预算冻结。跨服务器的完成记录计入全局进度，但运行进程数、并发上限和计划 ETA 仅覆盖当前服务器；多机协作时不把该 ETA 当作整个集群的准确剩余时间。优先使用同方法、同规模、同实验臂实测 episode 时间；未测实验臂使用同规模代理，未测规模使用实体数线性到 observer×实体数平方的计算包络，未测基线使用其他模型的宽代理，未测反事实按每场 1–13 次 rollout 范围估算。范围是工程预算区间，**不是置信区间或保证的上下界**；CPU/GPU 争用、对局长度与存储开销会改变速度。界面公开各代理占比及直接实测覆盖率，低覆盖时整轮数字标为 `PROVISIONAL`。当前任务和阶段的近期 ETA 需要至少一分钟速度采样。不能把本机部分诊断时间当成远端服务器的实测时间。

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

### 共享存储、多服务器与按依赖调度

完整协议为 1,809 个分片、最多 920,610 场完整对局，另最多 9,000 段反事实尾段。分片不是单场对局。各阶段原配额保持不变。

更新后的 `run` 使用一个 GPU slot 池，立即放行校准、机制和旧基线。校准冻结后同时放行选型与预算统计；选型冻结后放行候选固定深度和反事实；选型及预算冻结后放行自适应和独立确认。方法仅按 ID 选择，OOD 不用于改选。已完整记录的分片不再启动 GPU worker。

所有服务器使用相同代码、相同 HAD 环境与同一个共享绝对输出目录；共享文件系统必须支持跨主机 `flock`，不能启用仅本机生效的文件锁。每台服务器仅运行一个 `run` 主控。任务锁跨服务器共享，主控先跳过被占用任务，worker 再持锁执行。冻结文件和运行状态的发布也有共享锁。某台主控本地队列结束时，其他服务器可能尚在执行；以全局记录覆盖判断完成。

用户当前选择每卡 32 并行。每台服务器执行以下相同命令（路径保持原共享挂载）：

```bash
conda activate sarc
cd /inspire/hdd/project/urbanlowaltitude/fengkairui-25026/shenqili/TP/LEAF1009-lite/Open-SCORE
export LEAF_OUTPUT="$PWD/outputs/main1009"
export LEAF_OLD_OUTPUT="$PWD/outputs/main0928"
nohup python -u scripts/leaf1009.py run --suite loop_candidates --output "$LEAF_OUTPUT" --source-output "$LEAF_OLD_OUTPUT" --devices all --jobs-per-gpu 32 >> "$LEAF_OUTPUT/evaluation.log" 2>&1 &
echo "本机主控 PID: $!"
```

增加服务器无需复制权重或另建结果目录。查看器读取共享记录，因此完成量是全局的；进程数是本机的。已运行的主控必须正常停止并重启，才能使用新的调度逻辑；仅覆盖 Python 文件不会替换主控内已加载的函数。使用 SIGTERM 或前台 Ctrl+C，等待当前对局和 worker 退出后重开，保留所有结果与锁文件。未落盘对局重算，完整记录续跑。
### 无需重启评估的三机统一状态面板

将最新 `scripts/leaf1009.py` 覆盖到共享代码目录即可；评估主控无需停止。旧的查看器需要退出重开。默认 progress 仍只读；`--status-only` 和 `--cluster` 将本机进程快照写入 main1009 下独立的 `.leaf1009_status_<hostname>.json`；各服务器分文件发布，临时写入路径还包含监控 PID，并通过原子替换发布，避免多机共写状态的冲突。监控不读取或修改 runner_state.json，不修改训练权重、对局记录或冻结文件。

每台服务器在 sarc 环境及 Open-SCORE 目录各启动一次轻量状态上报（只扫描本机 /proc，不读整份对局记录，不启动评估）：

```bash
nohup python -B scripts/leaf1009.py progress --output "$PWD/outputs/main1009" --devices all --jobs-per-gpu 32 --watch-seconds 10 --status-only >> outputs/main1009/evaluation.log 2>&1 &
echo "本机监控 PID：$!"
```

在任意一台服务器打开统一固定屏幕查看器：

```bash
python -B scripts/leaf1009.py progress --output "$PWD/outputs/main1009" --devices all --jobs-per-gpu 32 --watch-seconds 10 --cluster
```

面板显示所有已上报 hostname、任务数、主控数、上报年龄及 RUN/WAIT/STOP/STALE 状态；超过 max(60 秒, 三个刷新周期) 的状态标记过期，排除其任务和容量。RUN 表示有评估 worker，WAIT 表示有主控但尚无 worker，STOP 表示当前无主控和 worker，STALE 代表上报过期，不能据此断言评估已停止。各服务器必须具有不同 hostname，否则无法区分。集群任务数、阶段任务数和并发上限合并有效快照；ETA 使用有效服务器的合计 slot 和共享实测耗时估算，仍是粗略预算。未启动上报的服务器不会出现在列表中，应核对三台 hostname 都出现。
