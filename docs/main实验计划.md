# 跨规模攻防泛化：main 实验计划

本文是 **main 正式实验** 的唯一规划文档。它同时记录：已经跑过、可以迁入复用的资产；协议变更后必须补做的训练与评估；机制分析与算力实验；以及尚未编写、但执行前必须落地的代码架构。

正式结果只写在 `Open-SCORE/outputs/main/实验报告.md`，分成 **主要实验结果**（三图三表，训练或评估一有新数据就刷新）和 **详细与次要结果**。根目录 `实验记录.md` 只保留跨版本摘要和链接。

当前状态（2026-09-17）：执行已开始。论文方法名暂定 **ReGIR**，代码键 `regir`。唯一写入目录为 `outputs/main/`；v3/v4/v5 迁入后只扫描 main。

---

## 0. 如何阅读、如何执行

### 0.1 三层分工

| 文件 | 作用 | 谁写 |
|---|---|---|
| 本文 `docs/main实验计划.md` | 实验清单、协议、架构、命令、完成判定 | 计划阶段一次写清，协议变更时改这里 |
| `Open-SCORE/outputs/main/实验报告.md` | 实际数字、图、运行状态 | 评估/训练落盘后由报告器刷新 |
| `实验记录.md` | 跨版本一行摘要 + 链接 | 仅在 main 切换写入目录后改链接 |

读计划时先看 §1 方法表和 §2 协议，再看 §11 任务总表判断「已完成 / 进行中 / 未开始」。不要用 v5 报告里的 1:2 数字当作 main 主表。

### 0.2 两条独立指令

训练和评估分成两个入口，互不抢对方的并发槽：

- **训练指令**：最多 **2** 个 GPU 训练任务，按显存分三波：cycle（ReGIR/消融）1 路、ALMA 2 路、matched 1 路。只做 `1M` 物理步、池内验证、写 `resume.pt` / `best.pt`。
- **评估指令**：最多 **2** 个评估任务。终评、深度扫描、机制探测、锚点、时延都走这条。现有终评默认 CPU，可与训练并行；**时延实验必须独占 GPU**，启动时若检测到训练进程则排队。

两条指令每次启动都先 **扫描** `outputs/main/`，只调度未完成且未在跑的任务，所有任务都按已有记录断点续跑。

### 0.3 执行代理的固定动作

以后任何一次「开始跑 main」都必须按这个顺序，不能凭记忆跳过：

1. 确认代码侧协议已按本文改完（接口 50/12、方法注册、评估配置、双入口）。未改代码则只许迁数据，不许开新训。
2. 扫描 `Open-SCORE/outputs/main/`：方法目录、`best.pt` / `resume.pt`、`episodes.csv` / `progress.csv`、控制台日志。
3. 对照 §11 把每条任务标成 `completed` / `running` / `pending` / `blocked`。
4. 训练入口只取 `pending` 的 `train.*`；评估入口只取 `pending` 的 `eval.*`。
5. 中断必须留下可恢复进度：训练写 `resume.pt`，评估按 `(phase, config, episode_seed, checkpoint, cycle_depth)` 跳过已写局。

---

## 1. 方法命名与实验版图

论文主方法名称：**ReGIR**（暂定名称；实现仍为全实体共享循环）。它就是当前实现里的 `refil_cycle` / 报告中的 REFIL-B：在原 REFIL 路径旁加入全实体共享循环、数量条件注入、个体跨轮读取，残差进入 GRU。

代码方法键以本节为准；旧键只作迁入别名。

### 1.1 主对照（7 个方法，主表）

学习方法一律 **3 个训练种子** `{0,1,2}`。规则没有训练种子：每配置 300 局，环境种子 `9000–9299`。随机加速度 **不是** 主表方法，只作为 NDS 分母的锚点。

| 报告名 | 方法键 | 角色 |
|---|---|---|
| ReGIR | `regir`（迁入别名 `refil_cycle`） | 所提方法 |
| REFIL | `refil` | 直接母体 |
| QMIX | `b2_qmix_atten` | 实体注意力价值分解 |
| DCG | `dcg` | 协调图 |
| SPECTra | `spectra` | 注意力 / 价值分解基线 |
| ALMA | `alma` | 全场低层 + 目标子任务分配 |
| Rule nv1 | `rule_nv1` | 预测拦截 + 整数规划规则 |

QMIX 始终指 `b2_qmix_atten`，不是有序展平的 `b0_qmix`。`b0_qmix`、`gnn_qmix` 迁入后只留档，不进主表。

### 1.2 消融对照（6 个方法，消融表）

同一训练协议、同一终评配置。四个变体都是 ReGIR 的单因素改动；REFIL 与 ReGIR 原样进入该表，使消融表自己闭合。

| 报告名 | 方法键 | 相对 ReGIR 只改什么 | 要回答的问题 |
|---|---|---|---|
| ReGIR | `regir` | 无 | 全模型 |
| REFIL | `refil` | 去掉整条全局循环旁路 | 循环模块整体有没有用 |
| ReGIR−REFIL | `regir_norefil` | 去掉原 REFIL 实体注意力路径和想象分组辅助损失；个体表征只来自循环读取 + GRU | 循环能否脱离 REFIL 关系学习独立工作 |
| ReGIR−\(z_n\) | `regir_nocount` | 关掉循环内数量编码对 query 的注入 | 收益是不是换皮的数量条件 |
| ReGIR-\(R{=}1\) | `regir_r1` | 训练和终评都固定 \(R=1\) | 主收益是不是多轮 / 随机深度训练 |
| ReGIR-last | `regir_last` | 训练和执行都只读最后一轮，不做跨轮 softmax 加权 | 跨轮表征库有没有独立贡献 |

`regir_r1` 的终评深度是 1，不参加「固定 \(R=4\)」的主对照终评口径。深度扫描仍只对主方法 `regir` 做。

### 1.3 参数匹配（算力实验专用，3 种子）

| 报告名 | 方法键 | 定义 |
|---|---|---|
| REFIL-matched | `refil_matched` | 原 REFIL，把可训练参数补到与 ReGIR 相差不超过 5%（优先加宽 `attn_embed_dim` 或在 fc2 后加零初始化 MLP）。优化器、想象分组、预算与 REFIL 相同 |

它不进入消融表的「单因素」叙事，只进入 §8 参数与时延。若 matched 在 40v40 / 50v50 上仍明显差于 ReGIR，则不能把主结果写成容量效应。

### 1.4 迁入但不进入 main 表的历史臂

v4 的距离分组、数量 FiLM，以及 v5 的 `refil_card` / `refil_feedback` / `refil_slot` / 去 LN 的 count，全部迁入 main 目录以免丢失，**不再补种子、不进主对照、不进消融表**。`refil_slot` 若迁入时仍未跑完，允许它在训练队列里以最低优先级收尾，避免浪费已有步数；收尾后也不扩成 3 种子。

---

## 2. 训练、验证、终评协议

除规模集合外，训练预算和选模规则与 v3/v5 相同，便于复用已有权重。

### 2.1 不变项

| 项 | 取值 |
|---|---|
| 环境 | HAD Workbench `had-workbench-2.1.0` / `rebuild-calibrated-v3-r7-target-initialization`，二维 `damage` |
| 蓝方 | `reactive` + `rush`，不训练对手 |
| 红方动作 | 9 个平面加速度；每物理步决策一次 |
| 学习奖励 | \(-\Delta D_t\) + 与 v3 相同的距离势塑形，`reward_mode=damage` |
| 报告指标 | 未折扣物理 \(D\)（越低越好）与 \(\mathrm{NDS}\)；学习曲线纵轴仍画 \(G=-D\) |
| 训练池 | \(N_R=N_B\in\{4,6,8,10\}\)，\(K\in\{1,2,3\}\) 均匀采样 |
| 训练槽 | pad=`train`：10 红、10 蓝、3 目标 |
| 预算 | 100 万物理步，8 个并行采集环境 |
| 验证 | 每 2 万步；4 配置各 25 局，共 100 局；共 50 个验证点 |
| 选模 | 只看池内 4 配置平均 \(D\)，用 `best.pt` |
| 终评 | 每配置 300 局，环境种子 `9000–9299`；贪心 |
| ReGIR 训练深度 | \(R\sim U\{1,2,3,4\}\)（`regir_r1` 除外） |
| ReGIR 常规终评深度 | \(R=4\)（`regir_r1` 用 \(R=1\)） |
| 训练种子 | 学习方法 `{0,1,2}` |
| 并发 | 训练最多 2；评估最多 2 |

池内验证配置不变：`4v4 K2`、`6v6 K2`、`8v8 K2`、`10v10 K3`。

### 2.2 规模集合：相对 v3–v5 的变更

**取消** 红蓝 1:2 轴。已有 1:2 局迁入 CSV 后保留，但不计入 main 完成判定，主报告不画该轴。

**等规模 1:1、K=2** 在原 `10/15/20/25/30/40` 上增加 **5v5** 与 **50v50**：

- 5v5：训练池内未见过的人数，但夹在 4 与 6 之间，是 **插值**。
- 15–50：训练最大为 10，是 **外推**；50 超过当前代码接口 40，必须先把评估槽扩到 50。

**目标数** 保留原 `10/15/20/30v × K=4,6`，并 **只在 10v10 与 30v30 上** 增加 K=9、K=12。K=9/12 超过当前接口 6，必须先把目标槽扩到 12。

### 2.3 终评配置清单（24 个不重复配置）

| 轴 | 配置 | 相对旧 23 配置 |
|---|---|---|
| 训练池内 | 4v4 K2，6v6 K2，8v8 K2，10v10 K3 | 沿用 |
| 等规模 K2 | 5v5，10v10，15v15，20v20，25v25，30v30，40v40，50v50 | 新增 5、50；去掉无 |
| 目标外推 | 10/15/20/30v 的 K4、K6 | 沿用 |
| 目标外推（加测） | 10v10 K9，10v10 K12，30v30 K9，30v30 K12 | 新增 |
| 已删除 | 2v4、4v8、8v16、15v30、20v40 的 K2 | 不再测 |

10v10 K2 只出现在等规模轴；10v10 K3 只出现在池内。目标图里的 K2 点复用等规模，不另开配置。

每方法每种子终评 \(24\times 300=7200\) 局。规则与随机各再跑一遍新配置上缺的局。

### 2.4 评估接口上限（执行前必须改代码）

| | 现行 | main 需要 |
|---|---|---|
| 红 / 蓝上限 | 40 / 40 | **50 / 50** |
| 目标上限 | 6 | **12** |
| 评估实体槽 | 86 | **112** |
| `Scale` 断言 | \(N\le40,K\le6\) | \(N\le50,K\le12\) |

训练 pad 仍是 10/10/3，**不重训** 已有实体注意力方法就能评 50v50 和 K=12，前提是权重不把 `n_entities` 写进参数（REFIL / QMIX-Atten / ReGIR / SPECTra / DCG 满足）。ALMA 的子任务 one-hot 宽度在训练时写死：现有 `n_extra_tasks=3` 只能覆盖到 K=6。要在同一套 ALMA 权重上评 K=9/12，必须用 `n_extra_tasks=9` **重训 3 个种子**，旧 ALMA seed 0 不能直接评新目标数。

HAD Workbench 是否允许 50 机、12 个目标，是开评前的环境核对项；若原生拒绝，先改包装层或确认工厂参数，不能把失败局写成方法分数。

### 2.5 每个指标的文献出处、选择理由与可证伪假设

下面每一条都回答三件事：文献里这个量用来干什么；它为什么适合本任务；它怎样支持或推翻我们的假设。公式与实验操作仍只保留这些量，不另派生分数。

假设一览（后文用编号引用）：

| 编号 | 主张 | 用来检验的量 |
|---|---|---|
| H1 | 冻结零样本下 ReGIR 优于母体、其他学习方法和规则，且该优势在未见规模上仍成立 | \(D\)、NDS |
| H2 | ReGIR 相对 REFIL 的增益可归因于四个可独立关掉的部件 | \(D\) |
| H3 | 共享循环在逐步凝练全局关系，而不是把同一张注意力复制四遍 | 一局逐轮注意力图 |
| H4 | 不同个体会读取不同轮次，因此跨轮表征库有独立贡献 | 同一局 \(\alpha\) |
| H5 | 大规模部署仍受益于更多循环轮 | \(D(N,R)\) |
| H6 | 主结果来自循环结构而非多出来的参数；加深有可测的逐步时延代价 | 参数量、matched 的 \(D\)、ms/step |

#### \(D\)：任务原目标（主对照、消融、外推、轮数的共同读数）

- **文献。** 强化学习的优化对象是回报 \(G=\sum_t r_t\)（Sutton & Barto, 2018, Ch.3）。合作 MARL 基准把「环境成功定义」直接当作主指标：SMAC 的 test win rate 是「限时内消灭全部敌人的回合比例」（Samvelyan et al., AAMAS 2019, §3 与附录 B.1）；没有胜负的任务则报 episode return。REFIL 在多任务星际上仍报 win rate，并按任务画出从易到难的衰减（Iqbal et al., ICML 2021, Fig.4–5）。本环境 `damage` 的逐步奖励是 \(r_t=-\Delta D_t\)，未折扣回报恰为 \(-D\)，因此 \(D\) 就是这里的 \(G\)（符号取反，越低越好）。
- **为何选。** 不另造「泛化分」。方法好不好，就是漏防伤害低不低。消融只改结构、轮数扫描只改 \(R\)，比较对象不变，所以这些实验也只读 \(D\)——与 REFIL 消融只报同一 win rate 的写法相同（Iqbal et al., 2021, Fig.4）。
- **证明什么。** 假设 H1：ReGIR 在冻结零样本部署下优于 REFIL 及其他学习方法和规则。支持条件是未见规模上 \(\bar D\) 更低。假设 H2：某个部件是增益来源。支持条件是关掉该部件后 \(D\) 明显变差。假设 H5：加深循环能改善大规模部署。支持条件是同一检查点上 \(D(N,R)\) 随 \(R\) 下降。
- **何时证伪。** 未见的 15–50v 或 K=9/12 上 ReGIR 的 \(D\) 不低于 REFIL；或某个消融臂的 \(D\) 与全模型无差别；或 50v50 上 \(R=1\) 与 \(R=4\) 的 \(D\) 无差别。

#### \(\mathrm{NDS}\)：跨规模、跨难度的规则归一化（只用于主对照与外推）

- **文献。** 不同游戏、不同地图的绝对分数不可比。DQN 用人类归一化分 \(\mathrm{HNS}=(\mathrm{score}-\mathrm{random})/(\mathrm{human}-\mathrm{random})\)，使 0 对齐随机、1 对齐人类，再跨 49 个难度差很大的 Atari 游戏汇报（Mnih et al., Nature 2015, Fig.3 及 Methods）。SMAC 不把 2s3z 与 27m_vs_30m 的绝对 win rate 当成同一尺度，而是 **每个场景都报 heuristic 基线**（Samvelyan et al., 2019, Fig.3 与 Table 2）。二者解决的是同一问题：场景变难或变易时，绝对分会跟着走。
- **公式。** 本任务分数取 \(-D\)（越高越好），规则扮演 HNS 里的 human。代入即
  \[
  \mathrm{NDS}=\frac{(-D)-(-D_{\mathrm{random}})}{(-D_{\mathrm{rule}})-(-D_{\mathrm{random}})}
  =\frac{D_{\mathrm{random}}-D}{D_{\mathrm{random}}-D_{\mathrm{rule}}}.
  \]
  这是 HNS 的同构，不是自造外推指数。0 对齐随机，1 对齐规则，大于 1 优于规则。
- **为何选。** 固定地图上放大编队时，规则的绝对 \(D\) 往往下降（拦截网变密），学习方法的绝对 \(D\) 也可能下降。只看「50v50 的 \(D\) 低于 10v10」会被难度漂移解释掉。NDS 把每个配置上的随机与规则当作该配置的 0 和 1，这正是 HNS 跨游戏、SMAC 跨地图的用法。消融与轮数扫描不跨方法族、只看结构或 \(R\)，绝对 \(D\) 已够，不再报 NDS。
- **证明什么。** 假设 H1 的跨规模部分：ReGIR 不是因为大场景变简单才看起来好，而是相对 **同规模规则** 仍更好或衰退更慢。支持条件是 15–50v 与 K=9/12 上 NDS 高于 REFIL，并至少不系统性掉到规则以下却把「绝对 D 下降」写成外推。
- **何时证伪。** 大编队上 \(D\) 下降但 NDS 同时低于 REFIL（任务变易、方法相对更差）；或新配置尚无随机/规则锚点时不能用 NDS 下结论，只报 \(D\)。

#### 零样本大场景协议（外推的定义，不是第三个指标）

- **文献。** 规模泛化在文献里是 **实验协议**，不是新分数：小规模（或少实体）训练，权重冻结，在训练未出现的实体数量上直接评同一任务指标。UPDeT 在 3m 上训练后 zero-shot 评 5m/7m，主指标仍是 win rate，并强调不改结构、不微调（Hu et al., ICLR 2021, §4 transfer）。ADAPT 用同一口径的 zero-shot win rate 表比较 3m→5m、5m→7m（Zhang et al., AAAI 2026, Table 2）。SMACv2 把「评估时泛化到未见过的程序生成设置」写进基准定义（Ellis et al., NeurIPS 2023）。REFIL 则用任务从易到难的 win rate 衰减说明泛化面更宽（Iqbal et al., 2021, Fig.5）。
- **为何这样定义。** 实体注意力能吃更长输入，只说明接口兼容。外推成立与否，仍看未见 \(N\)、\(K\) 上的 \(D\) 与 NDS。
- **证明什么。** 假设 H1：仅在 \(N\le10\)、\(K\le3\) 上训练的 ReGIR，在 5v5（插值）以及 15–50v、K=9/12（外推）上仍优于母体。协议本身不产生新数字。

#### 逐轮注意力图：循环是否在凝练信息

- **文献。** 注意力权重的含义是「算下一个表征时这个对象占多少」（Bahdanau et al., ICLR 2015；Vaswani et al., NeurIPS 2017）。Clark 等据此把 **单句注意力图** 当作 BERT 在看句法关系的证据，而不是再训一个探针分数（Clark et al., BlackboxNLP 2019, Fig.5；文中写明 *an attention weight has a clear meaning*）。与本方法同构的 **循环 Transformer** 用逐步注意力证明后轮在凝练：前几步分布平坦，后几步集中到回答所需的支撑事实（Dehghani et al., ICLR 2019, Appendix F）。
- **为何选图而不是新标量。** Universal Transformer 证明「反复加工会聚焦到关键信息」的方式就是一局（一篇故事）上的注意力图。本任务对应一局攻防、同一时刻、同一观察者、四轮自注意力。若后轮从散看全场变成盯住突击蓝机或受威胁目标，机制假设成立；这比类型质量、半径、漂移更直接，也更符合上述文献。
- **证明什么。** 假设 H3：共享循环在逐步重组全局关系，而不是把同一张注意力复制四遍。支持条件是后轮关注对象相对前轮明显收窄或转向威胁。
- **何时证伪。** 四轮图几乎一样；或后轮更散、没有对准交战实体。

#### 跨轮权重 \(\alpha\)：个体是否在用不同深度

- **文献。** 同一套注意力定义用在「源=各轮读取」上：\(\alpha^{(r)}\) 就是该轮表征在个体决策里的权重。多深度融合的先例是 ELMo 对 LSTM 各层做任务相关加权（Peters et al., NAACL 2018, §3.2），以及 Adaptive Computation Time 把「用了多少计算步」本身当作分析对象（Graves, 2016）。Universal Transformer 的 ponder time 也是在问：这一位置停在第几步（Dehghani et al., 2019）。
- **为何选。** ReGIR 的跨轮读取就是对 \(\{H^{(1)},\ldots,H^{(R)}\}\) 的 softmax。\(\alpha\) 直接读出「这个智能体这一步用了哪一轮」。不必再算熵。
- **证明什么。** 假设 H4：不同个体、不同时刻会选用不同计算深度，因此保留表征库有意义。支持条件是同一局里 \(\alpha\) 并非全体塌到最后一轮。
- **何时证伪。** 几乎所有红方 \(\alpha\) 都钉在 \(R=4\)。此时 H4 不成立，且应看到消融 `regir_last` 的 \(D\) 接近全模型。

#### \(D(N,R)\)：轮数如何影响部署（只测 ReGIR）

- **文献。** 可变计算量模型的标准报告是 **任务指标随计算步数变化**。ACT 画出 accuracy 与 ponder 步数的关系（Graves, 2016）。Universal Transformer 比较固定 \(T\) 步与动态停机，并观察更难的 bAbI 子任务平均步数更高（Dehghani et al., 2019, §3 与 Appendix F：1/2/3 个支撑事实对应约 2.3 / 3.1 / 3.8 步）。它们都不另定义「饱和轮」或「深度弹性」。
- **为何选。** 部署时能改的只有 \(R\)。一张 \(R\times N\) 的 \(D\) 表就能看出：某规模再加深是否还降伤。主表仍固定 \(R=4\)，避免用事后最优轮数冒充方法。
- **证明什么。** 假设 H5：大规模需要（或至少受益于）更多循环轮。支持条件是大 \(N\) 上 \(D\) 随 \(R\) 下降的幅度不小于小 \(N\)。
- **何时证伪。** 全表对 \(R\) 平坦，或大 \(N\) 上加深反而升 \(D\)。此时仍可保留 \(R=4\) 作为约定，但不能写「规模越大越要加深」。

#### 参数量与逐步决策时间：增益是否只是容量、部署贵不贵

- **文献。** 架构论文同时报质量、参数量和速度，以免把变大的网络写成新机制。Transformer 原文 Table 3 并列 BLEU、参数量与训练代价（Vaswani et al., 2017）。Kaplan 等表明损失随参数量下降，因此 **多出来的参数本身就能抬分**，要用容量对齐的基线才能把收益归到结构（Kaplan et al., 2020）。SMAC 附录要求报告计算资源与墙钟（Samvelyan et al., 2019, Appendix B.1）。部署侧把逐步推理时延当作对照量：AVACraft 在星际场景上把 MARL 的逐步推理报成约 8–10 ms/step（Ma et al., ACL Findings 2026, Table 8）。
- **为何选这两个。** 参数量回答「是不是只是更大的 REFIL」；ms/step（实体表→全体红方动作，不含环境步进）回答「多一轮循环的部署代价」。不报 FLOPs。`refil_matched` 把宽度加到与 ReGIR 相差不超过 5%，其 \(D\) 仍用任务原指标。
- **证明什么。** 假设 H6：主结果来自循环结构而非参数量。支持条件是 matched REFIL 的 \(D\) 仍差于 ReGIR。时延不证明 H1，只给出 H5 的代价：若多一轮几乎不降 \(D\) 却明显加时，部署应停在更浅的 \(R\)。
- **何时证伪。** matched 的 \(D\) 已经接近 ReGIR（H6 不成立）；或 ReGIR 在 50v50 上一步决策不可用。

各实验对应关系：

| 实验 | 核心量（≤2） | 主要假设 |
|---|---|---|
| A 主对照、C 规模外推 | \(D\)、NDS | H1 方法更好且能外推 |
| B 消融 | \(D\) | H2 部件有贡献 |
| D 凝练 | 一局逐轮注意力、同一局 \(\alpha\) | H3 凝练；H4 分轮使用 |
| E 轮数（仅 ReGIR） | \(D(N,R)\) | H5 深度影响大规模部署 |
| F 算力 | 参数量、ms/step | H6 不是纯容量；给出深度的代价 |

参考文献（指标选择所依据的原文）：

1. Sutton, R. S., & Barto, A. G. *Reinforcement Learning: An Introduction*. 2nd ed. MIT Press, 2018.
2. Samvelyan, M., et al. The StarCraft Multi-Agent Challenge. *AAMAS*, 2019. https://arxiv.org/abs/1902.04043
3. Iqbal, S., et al. Randomized Entity-wise Factorization for Multi-Agent Reinforcement Learning. *ICML*, 2021. https://arxiv.org/abs/2006.04222
4. Hu, S., et al. UPDeT: Universal Multi-agent Reinforcement Learning via Policy Decoupling with Transformers. *ICLR*, 2021. https://arxiv.org/abs/2101.08001
5. Zhang, Z., Chen, S., Li, Y., & Wang, F. ADAPT: Adaptive Decentralized Architecture with Perception-Aligned Training for Structural Generalization in Multi-Agent RL. *AAAI*, 2026. https://doi.org/10.1609/aaai.v40i34.40096
6. Ellis, B., et al. SMACv2: An Improved Benchmark for Cooperative Multi-Agent Reinforcement Learning. *NeurIPS*, 2023. https://arxiv.org/abs/2212.07489
7. Mnih, V., et al. Human-level control through deep reinforcement learning. *Nature*, 2015. HNS 公式见 Fig.3 与 Methods。
8. Bahdanau, D., Cho, K., & Bengio, Y. Neural Machine Translation by Jointly Learning to Align and Translate. *ICLR*, 2015.
9. Vaswani, A., et al. Attention Is All You Need. *NeurIPS*, 2017.
10. Clark, K., Khandelwal, U., Levy, O., & Manning, C. D. What Does BERT Look At? An Analysis of BERT’s Attention. *BlackboxNLP*, 2019. https://aclanthology.org/W19-4828/
11. Dehghani, M., et al. Universal Transformers. *ICLR*, 2019. https://arxiv.org/abs/1807.03819
12. Graves, A. Adaptive Computation Time for Recurrent Neural Networks. 2016. https://arxiv.org/abs/1603.08983
13. Peters, M. E., et al. Deep contextualized word representations. *NAACL*, 2018.
14. Kaplan, J., et al. Scaling Laws for Neural Language Models. 2020. https://arxiv.org/abs/2001.08361
15. Ma, W., Fu, Y., Zhang, Z., Ghanem, B., & Li, G. AVA: Attentive VLM Agent for Mastering StarCraft II. *ACL Findings*, 2026. Table 8 报 MARL 约 8–10 ms/step。https://aclanthology.org/2026.findings-acl.208/

---

## 3. 实验 A：主对照

**目的。** 小规模训练、冻结零样本部署，比较 ReGIR 与六个 baseline。

**训练。** `regir` / `refil` / `b2_qmix_atten` / `dcg` / `spectra` / `alma` × 种子 0,1,2。规则不训。

**评估。** 每种子 24 配置 × 300 局贪心；规则与随机同样 24×300。ReGIR 另做 §7 深度扫描。学习曲线只画池内验证。

**核心指标（2）。** \(D\)、NDS，出处与证伪条件见 §2.5。用来检验 H1：冻结零样本下 ReGIR 的漏防更低，且相对同规模规则仍更好。图和表落点见 §12.6。

**完成判定。** 六个学习方法 3 种子均有 `best.pt` 且 24 配置满 300 局；新配置锚点满 300 局。旧 1:2 不影响判定。

---

## 4. 实验 B：消融

**目的。** 把 ReGIR 相对 REFIL 的增益拆成四个可独立关掉的部件。实现见 §12.3。

四个变体各 3 种子，终评配置与 A 相同。`regir` 与 `refil` **复用 A**，不另训。

**核心指标（1）。** \(D\)（Iqbal et al., 2021 消融只报同一任务指标）。用来检验 H2：四个部件各自关掉后损伤应上升。出处见 §2.5。

**完成判定。** 四臂 3 种子训完并完成 24×300。

---

## 5. 实验 C：规模外推怎么定义、怎么看

不新训。外推就是实验 A 在未见规模上的 \(D\) 与 NDS，不另开指标。

**定义（与 UPDeT / ADAPT 的 zero-shot 相同）。** 训练只见 \(N\in\{4,6,8,10\}\)、\(K\le 3\)；测试时参数冻结，直接跑 \(N=5\)（插值）以及 \(N=15,\ldots,50\) 和 \(K=9,12\)（外推）。网络能吃更长实体表只说明接口兼容，不说明泛化。

**核心指标（2）。** 仍是 \(D\) 与 NDS，协议对应 UPDeT / SMACv2 的 zero-shot，不另开分数（§2.5）。检验 H1 的外推部分：15–50v 与 K=9/12 上 \(D\) 低于 REFIL，且 NDS 不因任务变易而虚高。若 50v50 的绝对 \(D\) 低于 10v10、规则也更低，只说明场景变易。

目标数用同一对指标，图画 10v10 / 30v30 上 \(K=2,4,6,9,12\) 的 \(D\)。单格数字放详细部分。

---

## 6. 实验 D：循环是否在凝练信息

**目的。** 检验 H3：循环在逐步凝练全局信息；以及 H4：个体会读取不同轮次。不是第三个任务分数。指标出处见 §2.5（Bahdanau/Vaswani 的权重含义；Universal Transformer 附录 F 的逐步注意力图；ELMo / ACT 的多深度加权）。

**做法。** 冻结 ReGIR `best.pt`（优先 seed 0），跑 **一局** `30v30 K2`（环境种子 `9000`，贪心 \(R=4\)），记下每轮自注意力。挑一个交战中段的物理步画图。若 9000 几乎没有交战，改用 `9001`，报告里写明。

**核心展示（2，都是图）。**

1. **逐轮注意力。** 同一时刻、同一观察者，\(R=1,2,3,4\) 四张图：连线指向红 / 蓝 / 目标。若后轮从散看全场变成盯住突击蓝机或受威胁目标，这就是凝练。
2. **跨轮读取 \(\alpha\)。** 同一局里各红方对四轮的 softmax 权重。若所有人塌到最后一轮，消融 `regir_last` 应接近全模型。

实现：前向可选返回注意力，默认训练关闭。该局写入 `trajectories`。不扫多配置、不聚合标量。

**完成判定。** 至少一局完整注意力可画上述两张图。缺钩子则 `blocked`。

---

## 7. 实验 E：循环轮数与部署（只测 ReGIR）

**目的。** 检验 H5：同一套冻结 ReGIR 只改执行 \(R\)，大规模部署是否仍受益于更多轮。不测其他方法。指标即 \(D(N,R)\)，对应 ACT / Universal Transformer「任务指标随计算步数变化」（§2.5）。

**扫描。** 仅 `regir` 三种子；\(R=1,\ldots,6\)；等规模 K2 的 5/10/15/20/25/30/40/50；每格 300 局。\(R=4\) 已有终评则复用。`regir_r1` 不参加本扫描。seed 0 的 10–40 迁入后只补 5 与 50。

**核心指标（1）。** \(D(N,R)\)。一张表：行是 \(R\)，列是规模，格是 3 种子平均 \(D\)。主表部署仍固定 \(R=4\)。不定义 \(R^\star\)、\(\Delta(N)\)、饱和轮、相关系数。

**完成判定。** 三种子 × 8 规模 × 6 深度满 300 局（含复用）。

---

## 8. 实验 F：参数量与逐步决策时间

**目的。** 检验 H6：主结果不是「更大的 REFIL」；并给出多一轮循环的部署代价。指标为参数量与逐步 ms/step（Vaswani Table 3；Kaplan 等的容量效应；SMAC 要求报墙钟；AVA 的 ms/step，见 §2.5）。不报 FLOPs。

**核心指标（2）。**

1. **可训练参数量。** 同一评估接口上数 `state_dict`。ReGIR 相对 REFIL 的差额决定 `refil_matched` 的加宽，相差超过 5% 则先改宽度再训。
2. **逐步决策时间。** 从当前实体表到全体红方动作的 GPU 墙钟，不含 `env.step`。测 `regir`（\(R=4\)）、`refil`、`refil_matched`、`dcg`；规模 10 与 50。ReGIR 另在这两档上测 \(R=1,\ldots,6\)，得到每加一轮的增量。热身 1 局后统计 5 局，单位 ms/step。测 GPU 时若有训练进程则 `blocked`。

`refil_matched` 三种子，协议同 REFIL，终评 24×300，比较仍用 \(D\)。结果进详细部分。

**完成判定。** 参数列得出；matched 三种子终评齐；10/50 上上述方法的时延写完。

---

## 9. 已有资产与迁入后的状态

下列状态以 2026-09-17 扫描 `outputs/main_v3`、`main_v4`、`main_v5` 以及现摘录目录 `outputs/main` 为准。迁入 main 之后，执行时 **只再扫 main**。

### 9.1 可迁入并在新协议下部分复用的训练

| 方法 | 种子 | 训练 | 旧终评（23 配置含 1:2） | 迁入后还缺什么 |
|---|---|---|---|---|
| QMIX | 0,1,2 | 完成（seed 0 约 94 万步，已有 best 与终评，视为完成、不重训） | 完成 | 新配置终评：5v5、50v50、K9、K12 |
| REFIL | 0,1,2 | 完成 1M | 完成 | 同上 |
| DCG | 0,1,2 | 完成 1M | 完成 | 同上 |
| SPECTra | 0,1,2 | 完成 1M | 完成 | 同上 |
| ALMA | 0 | 完成 1M，`n_extra_tasks=3` | 完成旧 23 配置 | **不能评 K=9/12**。等规模 5/50 可先用旧权重补评；正式主表要求 K12，故 ALMA 三种子按 `n_extra_tasks=9` **重训**，旧 seed 0 降为档案 |
| ALMA | 1,2 | 未开始 | 无 | 与上：直接按新 `n_extra_tasks=9` 训 |
| ReGIR (`refil_cycle`) | 0 | 完成 1M | 完成旧 23 配置；等规模 10–40 的 \(R=1\ldots6\) 已齐 | 改键为 `regir`；补 5v5/50v50 终评与深度；补 K9/K12；一局注意力；时延 |
| ReGIR | 1,2 | 未开始 | 无 | 全新训练 + 全套评估 |
| 规则 / 随机 | — | 无 | 旧 23 配置锚点齐 | 补 6 个新配置 × 300 局 |

### 9.2 消融与 matched：全部未开始

`regir_norefil`、`regir_nocount`、`regir_r1`、`regir_last`、`refil_matched` 方法键已实现，待训。3 种子 × 5 臂 = 15 个新的 1M 训练。

### 9.3 迁入但不进入正式表

| 来源 | 内容 | 迁入后 |
|---|---|---|
| main_v4 | L0.15 / L0.30 / count（带 LN）seed 0 | 档案；CSV 保留，报告不画 |
| main_v5 | count 去 LN、card、feedback seed 0；slot 训练中（约 7 万 / 100 万步） | 档案；slot 可最低优先级 `--resume` 收尾 |
| main_v3 | `b0_qmix`、`gnn_qmix` seed 0 | 档案 |
| 现 `outputs/main` | 摘录 CSV / 图 / 报告 | 被完整 main 覆盖；摘录报告不保留为第二份正式报告 |

### 9.4 旧终评里哪些格子可以直接算「已完成」

对已经迁入的方法 × 种子，若 `episodes.csv` 中 `phase=final_eval` 且配置属于新 24 集合、局数满 300、checkpoint 仍指向该次 `best@t_env`，则该格 `completed`。因此 10–40 的等规模和原 K4/K6 **不必重评**。1:2 格子标记为 `legacy`。新格子一律 `pending`。

深度扫描同理：seed 0 的 10–40 各 R 已满则 completed；5 与 50 为 pending。

---

## 10. 数据迁入 `outputs/main/`

执行阶段的第一件事是迁入，不是开新训。目标：main 成为 **唯一完整实验目录**（权重、resume、CSV、图、一份报告）。

### 10.1 目录布局

```
Open-SCORE/outputs/main/
  实验报告.md
  figures/
  episodes.csv
  learning.csv
  progress.csv
  trajectories.csv
  timing.csv             # 新，时延
  inventory.json         # 扫描器写出的任务状态，可覆盖生成
  <method>/train/seed_<k>/
    config.json
    best.pt
    resume.pt            # 训练未完成时必须有
    resume.prev.pt
    console.log
```

方法目录名用新键。`refil_cycle` 迁入后目录改名为 `regir`，或保留物理目录并在扫描器里做别名；推荐 **目录改名为 `regir`**，CSV 里 `method` 同时从 `refil_cycle` 改写成 `regir`，避免主扫描认不出主方法。

### 10.2 拷贝顺序

1. 把当前 `outputs/main/` 中仅含摘录的报告/图视为可覆盖产物（CSV 将与 v3 合并，不要手工删逐局数据前先合并）。
2. 拷贝 `main_v3` 的方法目录：`b2_qmix_atten`、`refil`、`dcg`、`spectra`、`alma`、以及档案用的 `b0_qmix`、`gnn_qmix`。
3. 拷贝 `main_v5/refil_cycle` → `main/regir`。
4. 拷贝 v4/v5 档案臂（可选但要求完整）：count、local、card、feedback、slot。v4 与 v5 都有 `refil_count` 时：v5 去 LN 用 `refil_count`，v4 带 LN 用 `refil_count_ln`，以免覆盖。
5. 合并 CSV：以 `episodes` / `learning` / `progress` / `trajectories` 的主键去重。主键与现日志一致：`(run, method, seed, phase, eval_point, config, episode_seed, checkpoint)`；深度扫描再加解析出的 \(R\)。冲突时保留更完整的一行（字段更多或 `t_env` 更大）。
6. `version` 字段统一写成 `main`；另存 `origin`（v3/v4/v5）仅在需要追溯时使用。若现 schema 不加列，则 origin 只写在 `inventory.json`。
7. 此后 **禁止** 再向 `main_v3/v4/v5` 写入。v5 上若还有正在跑的 slot / feedback 深度扫描，必须先停到 `resume.pt` 落盘，再拷贝，再从 main 恢复。

### 10.3 迁入后立即扫描一次

生成 `inventory.json`，人工核对：REFIL/QMIX/DCG/SPECTra 三种子权重在、ReGIR seed 0 在、ALMA 旧 seed 0 在、CSV 局数与 v3/v5 报告一致。核对失败不得开新训。

---

## 11. 任务总表

标识规则：`train.<method>.s<k>`，`eval.final.<method>.s<k>`，`eval.depth.regir.s<k>`，`eval.mech.regir.s0`，`eval.timing`，`eval.anchors`。状态取值：`completed` / `running` / `pending` / `blocked` / `legacy`。

下面是迁入完成后、代码改完后的应然清单。未改代码前，所有新臂训练都是 `blocked`。

### 11.1 训练（入口 A，并发 ≤ 2，按显存装箱）

| ID | 现状 | 说明 |
|---|---|---|
| `train.b2_qmix_atten.s{0,1,2}` | completed | 迁入即可 |
| `train.refil.s{0,1,2}` | completed | 迁入即可 |
| `train.dcg.s{0,1,2}` | completed | 迁入即可 |
| `train.spectra.s{0,1,2}` | completed | 迁入即可 |
| `train.regir.s0` | completed | 由 `refil_cycle` 改键 |
| `train.regir.s{1,2}` | pending | 新训 1M，`--resume` |
| `train.alma.s{0,1,2}` | pending | 新协议 `n_extra_tasks=9` 重训；旧 seed 0 降档 |
| `train.regir_norefil.s{0,1,2}` | pending | 去掉 REFIL 局部注意力 |
| `train.regir_nocount.s{0,1,2}` | pending | 关掉数量注入 |
| `train.regir_r1.s{0,1,2}` | pending | `global_depths=[1]`，终评 R=1 |
| `train.regir_last.s{0,1,2}` | pending | 只读最后一轮 |
| `train.refil_matched.s{0,1,2}` | blocked→pending | 先数参数再定宽度 |
| `train.refil_slot.s0` | 档案，可选 running | 不进表；若迁入时未完成可收尾 |

新训共 **2（ReGIR）+ 3（ALMA）+ 12（消融）+ 3（matched）= 20** 个 1M 任务。按并发 2，墙钟大约是 10 个任务串行长度。

训练队列建议顺序（先出主结果，再拆机制）：

1. `regir` seed 1、2  
2. `alma` seed 0、1、2  
3. `regir_r1`、`regir_nocount`（先回答最可能的替代解释）  
4. `regir_last`、`regir_norefil`  
5. `refil_matched`  
6. 可选 slot 收尾  

### 11.2 评估（入口 B，并发 ≤ 2）

| ID | 现状 | 说明 |
|---|---|---|
| `eval.anchors` | 部分完成 | 补 5v5、50v50、10v10 K9/K12、30v30 K9/K12；旧配置不重跑 |
| `eval.final.{qmix,refil,dcg,spectra}.s{0,1,2}` | 部分完成 | 只补 6 个新配置 × 300 |
| `eval.final.regir.s0` | 部分完成 | 同上 |
| `eval.final.regir.s{1,2}` | pending | 训练结束后 24×300 |
| `eval.final.alma.s{0,1,2}` | pending | 新 ALMA 训完后 24×300 |
| `eval.final.regir_*.s{0,1,2}` | pending | 各消融臂训完后 24×300 |
| `eval.final.refil_matched.s{0,1,2}` | pending | 同上 |
| `eval.depth.regir.s0` | 部分完成 | 补 5v5、50v50 的 R=1..6；10–40 复用 |
| `eval.depth.regir.s{1,2}` | pending | 8 规模 × 6 深度；R=4 复用终评 |
| `eval.mech.regir.s0` | pending | 一局 30v30 注意力 |
| `eval.timing` | pending | GPU 空闲时跑；覆盖 §8.3 格子 |

评估优先级：

1. 锚点新配置（否则新格子没有 NDS）  
2. 已有权重的新配置终评（QMIX/REFIL/DCG/SPECTra/ReGIR-s0）——立刻能更新外推图  
3. 随训练完成而解锁的终评  
4. ReGIR 深度补点  
5. 一局注意力  
6. 时延（独占 GPU）  

### 11.3 断点约定

| 任务类 | 恢复依据 |
|---|---|
| 训练 | `resume.pt`（或 `resume.prev.pt`）；`config.json` 与 CLI 的 seed / env / batch_size_run / t_max / 方法覆盖项必须一致，否则拒绝 resume |
| 终评 | `episodes.csv` 中已有 `(final_eval, config, episode_seed, checkpoint)` |
| 深度 | 同上，checkpoint 带 `/R{r}`；R=4 等规模终评可视为已完成 |
| 机制 | `trajectories` 中该局注意力；有记录即完成 |
| 时延 | `timing.csv` 主键 `(method, seed, config, device, cycle_depth, repeat_id)` |
| 锚点 | `(method∈{random,rule_nv1}, config, episode_seed)` |

中断：训练入口写 `stop.request`，评估入口写 `eval.stop.request`，各自把当前 batch / 当前局写完再退。下次扫描自动跳过已写局。

---

## 12. 为完成 main 将要构建的代码架构

本节是后续改代码的蓝图。**本次不实现。** 原则：加在现有文件上，不为实验再开一套导出或网页。

### 12.1 模块划分

```
Open-SCORE/scripts/train.py      # 现有；收敛为「只训练」
Open-SCORE/scripts/eval.py       # 新建：只评估。扫描 inventory，并发 ≤2
open_score/eval/inventory.py     # 新建：扫描 main，产出任务状态
open_score/eval/protocol.py      # 改：规模集合、去掉 1:2、新终评/深度/机制配额
open_score/eval/anchors.py       # 改：新 FINAL_CONFIGS
open_score/eval/mechanism.py     # 新建：导出一局注意力与 α
open_score/eval/timing.py        # 新建：逐步决策计时
open_score/eval/report.py        # 改：主对照/消融/外推/深度/时延都进同一份报告
open_score/envs/features.py      # 改：MAX 50/50/12
open_score/envs/scales.py        # 改：Scale 断言与 SCALE_POOLS
open_score/algos/__init__.py     # 改：方法键、ReGIR 覆盖、matched 宽度
open_score/models/entity_encoder.py  # 改：norefil / nocount / last、注意力钩子
open_score/utils/logging.py      # 改：DEFAULT_OUTPUT → outputs/main，VERSION=main
```

不新增实验版本目录。不把 v4/v5 报告复制进 docs。

### 12.2 扫描器 `inventory.py`

输入：`outputs/main` 路径。输出：内存结构 + `inventory.json`。

对每个已注册方法键：

1. 看 ` <method>/train/seed_k/` 是否有 `config.json`、`best.pt`、`resume.pt`、`progress` 行。  
2. 训练完成：`t_env` 达到预算且存在 `best.pt`（或现有 `_training_finished` 逻辑）。  
3. 训练进行中：目录被占用或 progress 为 running。  
4. 终评剩余：`remaining_jobs(final_jobs(...), rows)`，配置用新 24 集合。  
5. 深度剩余：仅 `regir`。  
6. `blocked`：方法键未实现、缺 `best.pt` 却去评估、ALMA 旧权重评 K>6、时延需要 GPU 但 GPU 被训练占用。

训练入口和评估入口都调用同一扫描器，避免两边对「做完了」理解不一致。

### 12.3 四个消融臂的具体改法

全部挂在 `V5_OVERRIDES` 一类的覆盖字典里，不要复制一份 encoder。

**`regir_norefil`**

- `global_branch=cycle`，深度采样与主方法相同。  
- `agent.imagine=False`，`lmbda=0`。  
- 个体局部路径：不跑原 REFIL 注意力；`local` 用自身实体的线性 + ReLU 得到 64 维（或直接 0，再让循环读取成为唯一输入）。推荐 **自身 MLP**，否则循环读取的残差加在全零上，初始化行为与主方法差太远。  
- mixer 仍用 `flex_qmix`，无想象辅助。  
- 这回答「没有 REFIL 关系学习与想象分解，循环还能否学」。

**`regir_nocount`**

- 与主方法相同，但 `count_to_token` 不加入 query（实现为跳过加法，参数可保留但正向为 0，避免 checkpoint 形状纠纷）。  
- 不删除 count MLP 也可以，只要前向保证 \(z_n\) 不影响循环。报告中写明「前向无数量注入」。

**`regir_r1`**

- `global_depths=[1]`，`global_eval_depth=1`。采集、学习、终评都是一轮。  
- 不要用主方法冻结权重评 R=1 来冒充本臂。

**`regir_last`**

- 深度采样仍为 `{1,2,3,4}`。  
- `read_context` 不用 `_jk`，只取 `states[depth-1]` 做一次 `read_attn`。  
- 终评仍 R=4，因此与主方法的差别只在「读一层还是加权读库」。

**`refil_matched`**

- `global_branch` 关闭。  
- 在登记前用一次脚本数清 ReGIR seed 0 与 REFIL seed 0 的参数差，选定 `attn_embed_dim` 或附加 MLP 宽度，写入覆盖项并冻结，三种子共用。

**`regir` 别名**

- 加载器接受 `regir` 与 `refil_cycle`。新训只写 `regir`。报告名仍为 ReGIR。

### 12.4 协议代码

`SCALE_POOLS`：

- 删除 `extrapolation_ratio`。  
- `extrapolation_agents = (5,10,15,20,25,30,40,50)` 的同数 K2。  
- `extrapolation_targets` = 原 K4/K6 四规模 + `(10,10,9),(10,10,12),(30,30,9),(30,30,12)`。  
- `FINAL_CONFIGS` 由池内 ∪ 上述两轴去重得到 24 个。  
- `Scale.__post_init__` 上限改为 50/50/12。  
- `features.MAX_AGENTS=50`，`MAX_BLUE=50`，`MAX_TARGETS=12`。  
- ALMA 配置 `n_extra_tasks: 9`。旧 ALMA checkpoint 评 K≤6 仍可用档案脚本，但不走正式 `eval.final.alma`。

深度扫描配置改为新的 8 个等规模点。机制与时延配置写在 `protocol.py` 常量里，不要散落脚本。

### 12.5 双入口行为

**`train.py --stage train --group main --output .../main --steps 1000000 --batch-size-run 8 --resume --max-concurrent 2`**

- `--group main` 重新定义为：扫描后所有 `pending` 的 `train.*` 正式臂（含消融与 matched 与新 ALMA 与 ReGIR 1/2），**不再** 把 DCG/SPECTra 三种子当作默认组。  
- 不再在训练进程旁边自动塞 depth_eval（现在评估入口负责，且评估并发单独计数）。  
- 仍检测占用目录，跳过已在跑的种子。

**`eval.py --group main --output .../main --resume --max-concurrent 2`**

- 扫描 `eval.*`，按 §11.2 优先级取最多 2 个。  
- `--only anchors|final|depth|mech|timing` 可限制种类。  
- `--device cpu|cuda`；`timing` 默认 cuda，若忙则 blocked。  
- 每个子进程跑完一批后刷新报告。

两条命令可以同时开：训练上限 2 路、按显存装箱，评估占 2 CPU 终评。用户若只开一条，另一条就是 0。不要在一个进程里把训练和评估的并发加在同一个计数器上。

### 12.6 报告器：一份报告、两块内容、表随训练评估刷新

唯一正式报告仍是 `Open-SCORE/outputs/main/实验报告.md`。训练每写一次进度、评估每落一批局，都调用现有 `refresh_report`，**主要结果里的图和表按当时已有数据重算**。缺的格子写 —，未满 300 局的格用当前局数均值并在表注写 `n=`，三种子未齐则报已有种子的均值、不画假误差带。不得等全部完成才出表。

报告固定两大部分。

**一、主要实验结果**

只放一眼能看懂的东西：三张图、三张表。不多加诊断图。

图（数据一到就重画）：

1. 训练曲线：池内验证 \(G=-D\)，学习方法 3 种子均值，规则水平线。
2. 等规模外推：横轴 \(N=5,\ldots,50\)，纵轴 \(D\)，ReGIR 对六个 baseline。这张图就是规模外推的主证据。
3. 目标数外推：10v10 与 30v30 上 \(K=2,4,6,9,12\) 的 \(D\)。

表（最多三张，同样实时刷新）：

| 表 | 内容 | 格子 |
|---|---|---|
| 表 1 主对照 | 行是方法（ReGIR、REFIL、QMIX、DCG、SPECTra、ALMA、规则），列是池内 / 等规模 / 目标 三类平均 \(D\)，括号里 NDS | 未评估完的类写 — |
| 表 2 消融 | 行是六个消融方法，列同上三类平均 \(D\) | ReGIR / REFIL 与表 1 同源 |
| 表 3 运行与关键格 | 上半：每个方法×种子的训练步、验证 D、终评已完成配置数；下半：5v5、10v10、30v30、50v50、10v10 K12、30v30 K12 的当前 \(D\) | 用来在训练中途看到「现在打成什么样」 |

表 1、表 2 是论文主结果的浓缩；表 3 保证训练或评估一开始就能在报告顶部看到进度和关键数字。

**二、详细与次要结果**

主块看完再往下翻。全部用 \(D\) / NDS 或 §6–§8 规定的那一两样，不再发明分数。

- 24 配置逐格 \(D\) 与 NDS（主对照）
- 消融逐格 \(D\)
- ReGIR 的 \(R\times N\) 的 \(D\) 表（实验 E）
- 一局注意力图与 \(\alpha\)（实验 D）
- 参数量与 10/50 决策时间（实验 F）；matched 的等规模 \(D\) 可叠在详细图上
- 池内拦截 / 碰撞等行为数字，若需要只留一张图，不进主要部分

不得用 1:2 旧图充数。历史档案臂不进这两部分。

---

## 13. 执行命令（代码落地后使用）

解释器与现在相同。均在仓库根目录 `E:\Code\Open_Score`。

迁入（一次性，人工或专用函数，不是训练）：把 v3/v4/v5 按 §10 拷入 `Open-SCORE/outputs/main/`，改键 `refil_cycle`→`regir`，合并 CSV，写 `inventory.json`。

扫描（只读）：

```powershell
& 'D:\Software\Anaconda\envs\torch310\python.exe' -X utf8 .\Open-SCORE\scripts\eval.py --stage inventory --output .\Open-SCORE\outputs\main
```

训练（最多 2 个 1M 任务，重方法按显存只开 1 路；断点必须加 `--resume`）：

```powershell
& 'D:\Software\Anaconda\envs\torch310\python.exe' -X utf8 .\Open-SCORE\scripts\train.py --stage train --group main --steps 1000000 --batch-size-run 8 --resume --max-concurrent 2 --output .\Open-SCORE\outputs\main
```

评估（最多 2 个评估任务）：

```powershell
& 'D:\Software\Anaconda\envs\torch310\python.exe' -X utf8 .\Open-SCORE\scripts\eval.py --group main --resume --max-concurrent 2 --output .\Open-SCORE\outputs\main
```

只补锚点 / 只测时延：

```powershell
& 'D:\Software\Anaconda\envs\torch310\python.exe' -X utf8 .\Open-SCORE\scripts\eval.py --group main --only anchors --resume --max-concurrent 2 --output .\Open-SCORE\outputs\main
& 'D:\Software\Anaconda\envs\torch310\python.exe' -X utf8 .\Open-SCORE\scripts\eval.py --group main --only timing --device cuda --max-concurrent 1 --output .\Open-SCORE\outputs\main
```

停止（各一条，互不影响）：

```powershell
& 'D:\Software\Anaconda\envs\torch310\python.exe' -X utf8 .\Open-SCORE\scripts\train.py --stage stop --output .\Open-SCORE\outputs\main
& 'D:\Software\Anaconda\envs\torch310\python.exe' -X utf8 .\Open-SCORE\scripts\eval.py --stage stop --output .\Open-SCORE\outputs\main
```

训练写 `stop.request`，评估写 `eval.stop.request`。不要杀进程导致 `resume.pt` 不完整。

单任务调试：

```powershell
& 'D:\Software\Anaconda\envs\torch310\python.exe' -X utf8 .\Open-SCORE\scripts\train.py --stage single --method ReGIR --seed 1 --steps 1000000 --batch-size-run 8 --run train --resume --output .\Open-SCORE\outputs\main
```

未实现的方法键必须直接报错退出，不能静默跳过否则扫描会永远 pending。

---

## 14. 如何理解各块实验

假设编号与文献依据见 §2.5。

| 假设 | 读者会问 | 看什么 | 核心量 |
|---|---|---|---|
| H1 | 是否优于已有方法和规则？能否外推？ | 主要结果表 1 + 等规模 / 目标图 | \(D\)、NDS |
| H2 | 四个部件哪个关键？ | 主要结果表 2 | \(D\) |
| H3 | 循环有没有凝练全局信息？ | 一局逐轮注意力 | 注意力图 |
| H4 | 个体用了哪一轮？ | 同一局的 \(\alpha\) | \(\alpha\) |
| H5 | 部署选几轮？ | \(R\times N\) 的 \(D\) 表；主表仍 \(R=4\) | \(D\) |
| H6 | 是否只是多了参数？一步要多久？ | matched 的 \(D\)、参数量、10/50 ms/step | \(D\)、参数量、时延 |

1:2 从主叙事拿掉后，外推主张只剩等人数放大和目标数增加。legacy 1:2 需要时在详细部分用一句话交代来源。

---

## 15. 风险与依赖

1. **接口 50/12** 改动会碰到 DCG 配对、ALMA 分配、mixer chunk、显存。先在空网络上对 50v50 K12 做一次前向，再开评。  
2. **ALMA 重训** 与旧 seed 0 不可混在同一行均值里。主表 ALMA 只报新三种子。  
3. **50v50 任务可能变易**（固定地图更密的拦截网）。必须同时看 \(D\) 和 NDS，不能只看绝对损伤下降。  
4. **K=12 在 10v10** 上红方极度不够分目标，D 会很高；比较的是方法间差异，不是接近零伤。  
5. **GPU 时延** 与训练互斥。计划里不要假设训练同时能量时延。  
6. **注意力钩子** 改变前向返回值时必须默认关闭，否则训练速度与 checkpoint 数值会变。  
7. **matched 宽度** 若改变 `attn_embed_dim`，mixer 超网输入维可能跟着变，参数差要按整网重数。  
8. 现 v5 若仍在写 `refil_slot` / 报告刷新，迁入前必须停干净，否则 CSV 双写。

---

## 16. 本计划明确不做的事

- 不为 card / feedback / 距离分组 / 数量 FiLM 补种子或重评 50v50。  
- 不把训练分布改成含 5 或 50，不把 1:2 加回训练。  
- 不换对手、不上部分可观测、不把接口扩到 50 以外。  
- 不另建方法报告、阶段总结或第二套输出目录。  
- 不另造斜率、保持率、泄漏率、严格完成率等派生外推分数；每个实验最多两个核心量。
- 执行阶段按 §17 落地。

---

## 17. 落地顺序

1. 停 v5 写入 → 迁入 main → 扫描核对。  
2. 改接口 50/12、协议 24 配置、报告去掉 1:2。  
3. 注册 `regir` 别名与四个消融 + matched。  
4. 拆分 train / eval 入口与 inventory。  
5. 开评估入口补已有权重的新配置和锚点（此时已能更新外推图）。  
6. 开训练入口按 §11.1 队列跑 20 个新 1M。  
7. 钩子就绪后跑机制与深度补点；GPU 空闲时跑时延。  
8. 报告齐 3 种子主对照后，再强调消融与机制，避免主表还缺种子时先写机制故事。

以上完成后，`outputs/main/` 应能单独支撑论文主实验：权重、逐局数据、一份带图报告、可断点恢复的训练与评估队列。
