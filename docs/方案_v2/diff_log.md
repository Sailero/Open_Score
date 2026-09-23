# 跨规模 v2：官方实现与 HAD 迁移差异

> 历史迁移记录，仅供核对早期实现差异；当前协议、结果和论文判断分别见 [main0921 实验计划](../../Open-SCORE/outputs/main0921/实验计划.md)、[正式报告](../../Open-SCORE/outputs/main0921/实验报告.md)及[项目核心分析](../当前实验分析与论文主张讨论.md)。

本清单记录当时的方法和接口差异，不代表当前 main0921 的执行状态；现行验证、训练数据与图表见 [main0921 正式报告](../../Open-SCORE/outputs/main0921/实验报告.md)。

## 来源与版本

| 来源 | 固定提交 | 用途 |
|---|---|---|
| [ALMA](https://github.com/shariqiqbal2810/ALMA) | `81bd5c475972b982ce470b1ed78154ec39d236da` | 共享 EpisodeBatch、ReplayBuffer、并行 runner、QMIX learner、实体注意力和 FlexQMIX；FF 原生环境 |
| [REFIL](https://github.com/shariqiqbal2810/REFIL) | `ffe23a1c62fd3f7304e700e69b30dcb0c29586cc` | 原版算法与配置对照，非第二套训练主干 |
| [DCG](https://github.com/wendelinboehmer/dcg) | `4de100cddf7c3a7035cd89a47d7c1b8a878e7428` | 成对收益、Max-Sum、专用 learner 与 rel_overgen 原生环境 |
| [SPECTra](https://github.com/funny-rl/SPECTra) | `ffababf6187216c9d16b2109ee8ef6fe5fdf1172` | GRF 树的 SAQA、循环 agent、集合 mixer、NQLearner；SMACv2 树的 GNN 编码器参照 |

这些方法属于官方代码基础上的 HAD 场景迁移；本轮不声称原样复现官方基准成绩。

## 已与用户确认的范围和结构

- 完整实现六方法：B0-QMIX、B2-QMIX-Atten、REFIL、DCG、GNN-QMIX、SPECTra。B1、COPA 不额外做 HAD 适配；保留主干中已有的源码不等于完成这些可选方法。ALMA 原属同一批可选项，v3 期间按用户决定补做成正式臂，见文末[ALMA 分层臂的接入](#alma-分层臂的接入v3-追加)。
- B0/B2/REFIL 不使用 GRU；DCG/GNN/SPECTra 保留循环结构。B2 与 REFIL 除随机实体分解辅助损失外保持一致。历史 S1 实际使用 GRUCell(64,64)，本轮从零开始，不加载旧 S1 权重。
- 原 REFIL 使用 GRU 处理部分观测。ALMA 附录 A 明确采用前馈替代来加速无明显部分观测的任务，本轮三个基础方法采用该选择。ALMA 默认 PopArt 和 64 步片段采样不启用，使用完整回合；奖励仍为原始负伤害增量。
- 保留各方法官方优化语义：REFIL 的辅助权重 0.5、8 次批次更新；DCG buffer 500、epsilon 退火 50000、每采样批 1 次更新；SPECTra Adam、TD(lambda=0.6)、每批 1 次更新。DCG 从原单环境采样迁到共享并行 runner；实际 worker 数由测量后统一确认。
- B0 保留顺序敏感的展平结构，其集合置换不变性与普通 QMIX mixer 的相应对齐性质记不适用；不加排序或注意力使其通过这些检查。

## 环境及接口适配

- 外部 HAD 仓库不修改。只有 `had_wrapper.py` 直接导入 HAD；规则模块的导入和状态组装做必要调整，规则决策算法不变。原方案“移动规则文件内容不改”按该依赖隔离约束修正。
- 环境 reset 仍使用原生初始化；每物理步绕开原生逐智能体观测和嵌套全局状态组装。原方案“完全不调用原生观测”限定于逐步热路径。
- 实体槽固定为 30 红、30 蓝、6 目标，死亡保留位置。基础实体表 10 维，只存一份；相对几何在前向时构造。上一动作仅在适用的网络输入中拼一次，t=0 为零。
- 规则/模型评估状态补充初始生命值、目标累计伤害和实际上一动作，保留 `act(state, side, action_ids)` 签名。评估与训练重用相同物理、蓝方重分配和种子语义。
- DCG 添加初始节点/边 padding 掩码，死亡节点仍保留；按初始有效图归一化，排除非法动作和 padding 的负无穷项。PyTorch 原生 scatter 替代旧 torch_scatter 编译扩展。
- SPECTra 移除足球传球动作、守门员/单球固定分段，换为红蓝目标实体和统一 9 动作头；显式掩蔽 padding，保留 SAQA 与集合超网络。GNN 按有效实体计数做消息聚合，提供 FullObs/kNN-10。
- Windows 使用 spawn worker 和按需环境导入；runner 注册表也按需导入，HAD/FF 环境子进程不加载训练用 PyTorch，减少 Windows 进程内存；现代 PyTorch 的掩码运算使用 bool。注意力全空行在 softmax 前使用有限占位值、之后权重清零，避免 REFIL 空分组产生 NaN 梯度；PopArt 关闭时反归一化直接返回输入。回放使用 tuple 索引及显式复制以兼容当前 PyTorch。不会因导入 HAD/FF 而要求安装 SC2。

## 评估、日志及恢复

- 每 2% 正式预算精确评估 100 局：4v4 K2、6v6 K2、8v8 K2、10v10 K3 各 25 局，分别分配种子 9500–9599 的连续 25 种子段。允许最后一个不满 worker 数的批次，避免 ALMA 原版把 100 局舍入为 96 局。
- 最终评估覆盖训练池内 4 个配置（4v4 K2、6v6 K2、8v8 K2、10v10 K3）加三条外推线：等规模 10/15/20/25/30/40v 同数 K2，1:2 的 2v4/4v8/8v16/15v30/20v40 K2，以及 10/15/20/30v 同数上 K=2/4/6（K=2 与等规模线共用）。去重后 23 个配置各 300 局。学习方法与随机/nv1 锚点共用种子 9000–9299；已有配置不重跑，缺的配置按同一批种子补齐。B0 仍只评训练池 4 配置。每配置保留前 2 局轨迹。旧配置行留在 CSV，报告不再列出。目标点图按编队规模着色、按方法区分线型。
- 训练实体表按训练池上界 10 红 / 10 蓝 / 3 目标打包（23 行），不再为训练回放预留 40v40 的 86 槽。40/40/6 只用于训练结束后的 `load_policy` / 规则评估路径。集合方法的权重不把槽位数写进参数；已用 40 槽开过的 resume 仍按保存的 `n_agents` 续跑，避免回放形状对不上。父进程状态行每 10 秒只读 `progress.csv` 每条作业的最后一行，并打印已训练墙钟时间；正式报告刷新不再整表加载 `learning.csv`。
- 选型只用训练池内 4 个验证配置。`validation_score` 增加显式断言：一旦选型输入里出现非训练配置即报错，代码层面阻断测试集泄漏。
- 评估记录以权重版本而非文件名标识（`best@<训练步>`）。`best.pt` 会被更好的模型覆盖，仅用文件名去重会让旧评估结果顶替新模型的配额。沿用已冻结的 `checkpoint` 列，不改 CSV 结构。
- 红蓝攻击机均开火自毁，红方死因增加 `self_destruct`。死亡归因按物理处理顺序，每单位只归因一次；评估与锚点开启已有物理事件，训练采样关闭重型诊断。
- 未受伤目标的首次伤害时间、规则锚点的 Q 值等使用 null；报告显示“未受伤/不适用”。完整性检查要求适用字段有效，不用假零填充不存在的量。
- 使用方案允许的 CSV；复杂字段为可解析 JSON 单元格。不新增 Parquet 依赖。所有方法合并到同类数据表，用运行/方法/种子字段区分，只有一份正式 Markdown 报告。
- 恢复点包含在线/目标网络、优化器、回放、随机流、步数与评估进度；中断后优先完成同一策略的未完成验证点，再继续训练。先提交必要轨迹，再提交对应逐局结果。恢复时比对全部配置项，只放行 output、device/use_cuda、两个 interval、implementation_revision、concurrency、entity_pad 与 resume 本身；改动 `gamma`、探索调度、`buffer_size`、`lr` 或 `env_args` 会被拒绝并要求改用新 `--run`，避免旧回放与新配置混用。选优先提交权重再更新分数，写盘失败不会留下"新分数＋旧模型"。
- 正式运行名与非 HAD 环境互斥；`--stage single` 在原生环境下默认运行名为 `single_<env>`，不会误入需要伤害 D 字段的 HAD 评估协议。
- 修复后的一轮现记为版本 `main_v3`，输出目录 `Open-SCORE/outputs/main_v3/`。正式运行名由散落的字面量 `stage3` 改为单一常量 `open_score.utils.logging.FORMAL_RUN = "train"`，`--stage stage3` 相应改名 `--stage train`；分阶段的命名已无对应计划。旧的 `stage3` 记录留在 v2 目录不动，避免 `remaining_jobs` 把旧验证点当成新运行的已完成配额。
- 规则与随机锚点走 `rules.run_episode` 的原生 HAD 环境，不经过 `HADWrapper`，因此尾段折算对锚点 D 没有影响；v2 的 8400 局锚点已并入 v3 的 `episodes.csv`（改写 `version`/`run` 两列），不重复运行。`--stage train` 不再附带锚点任务。
- 正式训练的准备门槛收窄为 `--stage validate` 的正确性检查。E0 学习链路与吞吐基准属于 v2 分阶段计划，吞吐已测且与本轮的价值/槽位修复无关；实测同一负载在本机的重复测量为 597–873 步/秒（8v8），开关尾段折算无差异，该阈值本身落在测量噪声带内。

## 已测量的结构性限制及其处置

红方全灭而蓝方仍存活时，五个实体类方法的团队价值恒为 0 且对应梯度为 0（B2、REFIL、GNN、SPECTra 经由 `AttentionHyperNet` 先按 agent 掩码清零再只对 agent 行求均值，`w1/b1/w_final/V` 同时归零；DCG 无 mixer，效用与收益项同样清零）。实测该状态下 q_tot 精确为 0、梯度范数为 0，而 B0 因使用全局状态向量的普通 QMIX 超网络仍为 0.054、梯度范数 12.1，故这是 B0 与其余五臂之间的结构性不对称。

发生率已实测：蓝方攻击机开火即自毁，回合通常在红方全灭前自然结束。随机策略逐步检测，训练池 4/6/8/10 对称配置为 0.0%；nv1 规则锚点在训练池四配置与对称外推 12/16/20 均为 0.0%，仅 8v16 为 2.5%、8v20 为 5.0%。早期学习策略的一份 20000 步回放中为 19/20000 个转移（0.1%）。该状态只影响 TD 目标——动作选择用各智能体 Q_i，评估阶段 mixer 不参与——因此对训练分布内的三个基础实验不产生可测偏差。

不改五个网络结构以保留官方保真度；训练中按方法记录该状态的实际发生计数，若计数变得不可忽略再决定迁移方式（可选路径：红方全灭时终止回合并把后续伤害折入一次性终止奖励，方法无关且 D 定义不变，代价是尾段折扣权重略有偏差）。

**后续处置（v2 正式运行后修复）：** 1M 步的正式回放实测该状态占 B2 4.99%、REFIL 5.34% 的转移，不再可忽略，已按上述可选路径迁移。`HADWrapper.step` 在红方全灭时用空动作把剩余物理步跑完，按 $\sum_k\gamma^k r_{t+k}$ 折算成一次性奖励并置 `terminated=True`；折算按真实折扣累加，因此尾段权重没有偏差，不是"略有偏差"。红方全灭后所有单位已死、动作被强制为 no-op，尾段与红方策略无关，这对策略优化严格等价。物理、蓝方调度与 $D$ 不变，全灭中间状态不再进入回放。`train_wipeout_steps` 保留，含义改为被折算的尾段长度。

**FlexQMixer 的 $Q_{tot}\ge-1$ 下界（v2 正式运行后修复）：** 承接的 ALMA 版 `AttentionHyperNet.forward` 把 `nonneg` 提到形参默认 `th.abs`，四个调用点里 `hyper_b_1` 与 `V` 未覆盖该默认，于是 $b_1\ge0$、$V\ge0$。配合 `softmax_mixing_weights` 的 $w_{final}$（和为 1、非负）与 $\mathrm{ELU}\ge-1$，得到 $Q_{tot}=\sum_j w_j\,\mathrm{ELU}(z_j)+V\ge-1$，与 $-5$ 至 $-16$ 的真实回报不可达。REFIL 原始实现 `ffe23a1` 的 `AttentionHyperNet.forward` 没有 `nonneg` 形参，`abs`/`softmax` 只在调用点作用于 $w_1$ 与 $w_{final}$，$b_1$ 与 $V$ 是带符号的，故这是移植缺陷而非设计取舍；ALMA 默认开 PopArt，其输出仿射掩盖了该下界，本项目按方案关闭 PopArt 后才暴露。修法是给这两个调用点传恒等约束。单调性只要求权重非负，$b_1$ 与 $V$ 不乘 $q_i$，$\partial Q_{tot}/\partial q_i$ 不受影响。

**$b_1$/$V$ 随规模缩放（同批修复）：** 同一函数的 `vector`/`scalar` 模式先按 agent 掩码把死亡与 padding 行清零，再对固定的 20 个 agent 槽求均值，等于把 $b_1$ 与 $V$ 乘上存活比例 $n/20$；4v4 与 20v20 之间相差 5 倍，正好压在跨规模这条主轴上。改为对存活 query 数做掩码均值（除以 `(1 - agent_mask).sum().clamp_min(1)`）。实测 4 智能体与 10 智能体编队的 $|b_1|$ 均值由 2.5 倍差变为一致。

**B0 空槽与外推（同批修复）：** 训练池上界为 10 红、10 蓝、3 目标，B0 展平编码器与普通 QMIX 超网络却按 46 槽建参数，红槽 10–19、蓝槽 30–39、目标槽 43–45 共 23 个槽在训练中从不激活，对应 253/506 个输入列恒为零，其权重只接收 padding。B0 的顺序敏感结构本就不具备外推能力（新增编队只能落到从未训练过的列上），因此明确按训练池定义槽位：`configs/b0_qmix.yaml` 增加 `pool_slots: [10, 10, 3]`，编码器与 mixer 的槽位和 agent 混合行都压到该规模，参数量由 281,994 降到 143,562。越界编队在选定配置处直接报错（`HADWrapper.reset` 与 `FrozenPolicyAdapter`），不做静默截断；回放中未填充时间步的全零掩码看起来像满编 20v20，因此不在张量热路径上做该断言。最终评估相应把 B0 限定在训练池 4 个配置（`protocol.POOL_ONLY_METHODS`），外推行在报告中显示"—"。同时 `hypernet_layers: 2` 补齐：`hypernet_embed: 128` 一直写在配置里，但只有两层超网络分支会读它，此前走的是单层路径。

- E0 每方法 1 种子、20000 物理步；REFIL 两种 worker 数各测独占/4并发、每任务 20000 步，共 200000 步。达到配额时以采样截断结束最后收集片段并保留 bootstrap，不自动扩大步数。
- 本机未安装 gfootball，SPECTra 按方案最终执行清单使用 FF 替代原生检查。正式基础实验结束后停止，主实验预算和启动另行讨论。

## 实际结果

通过、失败、未执行、资源限制和实际吞吐均以唯一实验报告及其原始 CSV 为准；本清单不把尚未执行的检查写成通过。

## 执行状态清单

截至正式基础实验启动前。

### 已完成

- 六个方法的代码实现：B0 朴素 QMIX、B2 QMIX-Atten、REFIL、DCG、GNN-QMIX、SPECTra-QMIX⁺。超参数与官方固定提交逐项对齐，偏离项均记录在本文件上方各节。
- 环境 wrapper：`had_wrapper.py` 是唯一导入 HAD 的模块。物理推进、团队奖励、终止/截断与原生 Parallel API 数值等价。
- V1 环境语义检查 4 项：奖励等于逐步伤害增量、回合奖励和等于最终 D、掩码与死亡语义、蓝方重分配一致性。
- V2 网络性质检查：实体置换不变/等变、变实体数前向、QMIX 单调性、mixer 与官方实现数值对齐。
- V3 算法组件检查：注意力掩码正确性、死亡单位仅保留 noop、REFIL 按回合固定随机分组与辅助权重、DCG 收益对称化与 STAR 解码、GNN kNN 邻接、SPECTra SAQA 与 TD(λ=0.6)。
- V4 E0 冒烟：六方法各 1 种子 20000 物理步，在原生 FF 环境跑通完整学习链路。
- 吞吐基准：环境侧 8v8 889 步/秒、20v20 K4 283 步/秒；训练侧 8 worker 独占 144.5 步/秒、4 并发每任务 57 步/秒。
- 停止与恢复机制：`--stage stop` 受控保存，恢复点覆盖在线/目标网络、优化器、回放、随机流、步数与评估进度，并已在 3425 步和 13458 步两次实测验证。
- 本轮五项修复：评估记录绑定权重版本、选优先写盘再更新分数、恢复时全量配置比对、正式运行名与非 HAD 环境互斥、全灭状态发生计数落到训练日志。
- 本轮补齐外推评估：最终评估与锚点从 4 个训练池配置扩到 14 个配置，报告绘图与表格相应扩展，并在选型入口加入测试集泄漏断言。
- 全部 14 个评估配置已在真实训练环境路径上跑通到回合结束，实体槽在 20v20 K6 恰好用满 46 个。B0 改为训练池槽位后只覆盖其中 4 个训练池配置。

### 未开始

- 三个基础实验的正式训练（B0、B2、REFIL，各 1 种子 100 万步）及其最终评估。这是当前唯一的下一步。
- DCG、GNN-QMIX、SPECTra 的正式训练。
- 3 种子完整矩阵。
- E3 对手泛化、E4 训练分布对照、E5 消融。
- 正式锚点 300 局 × 14 配置（8400 局）尚未产生数据，随基础实验并行运行。
- 论文主表与主图。

## v3 首轮训练的诊断与第二批修改

v3 第一次正式训练在 B0 59.1 万步、B2 40.9 万步、REFIL 30.0 万步时受控停止，全部记录与权重已删除，结论保留在此。锚点不经过 `HADWrapper`，8400 局原始记录跨轮沿用。

**REFIL 确实在学，但太慢。** 验证 D 从 6.41 降到 3.99（pt13，约 26 万步），优于随机锚点 5.28 约 24%，仍远差于规则锚点 1.46；每局拦截数 1.36→3.14，`red_left` 3.44→1.97，行为向规则策略的"以红换蓝"模式移动。B2 在 4.28 与 6.58 之间震荡，greedy 策略的 noop 占比在 0.000 与 0.595 之间跳变。

**价值传播速率是主要瓶颈。** 硬目标网络每 200 回合（约 9000 物理步）刷新一次，配 1 步 TD，每次刷新只把前瞻推进一个时间步。B2 在 t_env=133312、updates=2824 时已刷新 14.1 次，按每步平均奖励 −5.0/45 预测 Q_tot=−1.56，实测 −1.54，吻合到 1%。而蓝方首次造成伤害的中位步数是 30（p25=26、p75=36），即因果动作与惩罚相隔约 30 步，需要约 30 次刷新、约 27 万步才能把信用链走通，100 万步预算内总共只有 111 次刷新。

**B0 的发散源于混合权重参数化，不是本项目的改动。** `configs/b0_qmix.yaml` 用 `softmax_mixing_weights: false`（参考 PyMARL 的 `abs()`，无上界），B2/REFIL 用 ALMA 的 `true`（softmax 后每行和为 1）。无界权重下 Q_tot 可以靠超网络权重自增来追目标，γ=0.99 在 45 步回合上几乎不构成收缩，于是自激：q_tot 从 +0.75 涨到 +622.8，loss 237309，梯度范数 2.47e6，而真实回报恒为负。ALMA 默认的 `popart: True` 是另一层保护，本项目设为 `false`，且 PopArt 分支只存在于 `FlexQMixer`，B0 用的 `QMixer` 没有。v2 同样出现该现象，与本轮 `_signed`、`_live_queries` 等修复无关。

**第二批修改（三项）。** 目标刷新间隔与 1 步 TD 目标按用户决定保持 QMIX 官方取值不动，改从奖励侧解决同一瓶颈。

- **势函数奖励塑形**：`HADWrapper` 新增 `shaping_coef`（默认 1.0）与 `shaping_range`（4000），$\Phi(s) = -c\sum_b \mathrm{clip}(1 - d_b/4000, 0, 1)$，$d_b$ 为存活蓝方到最近目标的距离，回合奖励加 $\gamma\Phi(s')-\Phi(s)$，终止状态按惯例取 $\Phi=0$。实测蓝方出生于约 3586、以约 91/步逼近，故 $\Phi(s_0)\approx 0$。折扣塑形项望远镜求和为 $-\Phi(s_0)$，只依赖初始状态，按 Ng, Harada & Russell (1999) 最优策略严格不变；`return_sum` 与回合摘要仍记录物理回报，`D` 与锚点可比性不受影响，新增 `V1-shaping-invariance` 直接校验该恒等式（3 个种子，1e-6 容差）。红方全灭折叠尾段若在截断处被强制判为终止，退还末步记入的后继势。实测非零奖励步占比由 10.6% 升到 100%，每局约 3 次均值 +0.56 的正脉冲（拦截），奖励标准差由 0.42 降到 0.23。
- **探索退火**：三个配置 `epsilon_anneal_time` 由 500000 降到 100000。ε 逐 agent 独立采样，7 个存活 agent 全部贪婪的概率是 $(1-\epsilon)^7$；原计划在 REFIL 出现改善的 12 万步处 ε 仍有 0.77，联合贪婪策略几乎从不被采样，而 bootstrap 的 `max` 正是在该处取值。
- **GRU**：三个配置 `agent.recurrent` 由 false 改为 true。参考 PyMARL QMIX 的 `rnn_agent` 是 fc-GRUCell-fc，B0 此前无记忆并不符合"经典 QMIX"；拦截需要红方咬住同一追击目标而非逐步重新决策。ALMA 自己的 `refil.yaml`/`qmix_atten.yaml` 用 `recurrent: False`，此项为对捐赠实现默认值的偏离。`TemporalHead` 已存在且为 DCG/GNN 所用，开启后 56 项模型检查全部通过，含 REFIL 想象分支与三方法的 `episode_hidden_reset`；`V3.b0_qmix.no_identity_history` 改为断言无 agent id、无上一动作、无实体上一动作，`V3.*.recurrent_state` 的不适用分支改由配置而非方法名决定。

- **B0 混合权重**（第四项）：`configs/b0_qmix.yaml` 的 `softmax_mixing_weights` 由 false 改为 true。留用 `abs()` 会让 B0 这条臂大概率重复发散而无可用结果，且读者会归因于基线未调；softmax 是 B2/REFIL 所用同一份 ALMA 代码里的现成选项，三方法统一后 B0 与 B2 之间只剩编码器/mixer 架构一个差异，对照更受控。代价是 B0 不再严格等于参考 PyMARL QMIX，记为对捐赠实现的偏离，可一行回退。改动后 64 项验证全部通过（62 passed / 2 not_applicable）。

**进阶三方法（DCG、GNN-QMIX、SPECTra）启动前的对齐。** 塑形写在 `load_config` 的共享默认里，六方法自动生效；三者均已 `agent.recurrent: true`，GNN 已是 `softmax_mixing_weights: true`，DCG 无 mixer，SPECTra 自带 mixer，架构无需改动。改动两处：

- `gnn_qmix.yaml`、`spectra.yaml` 的 `epsilon_anneal_time` 由 500000 降到 100000，与前三方法一致。DCG 官方值 50000 保留，在安全一侧。
- `dcg.yaml`、`spectra.yaml` 的 `training_iters` 由官方值 1 提到 8，DCG 的 `buffer_size` 由 500 提到 5000。理由：`training_iters` 决定每采集 8 个回合做多少次梯度更新，100 万环境步约 22200 回合、2780 次采集，官方值 1 只给约 2780 次更新，是另外四个方法的 1/8；REFIL 是在约 42 万步、约 9300 次更新时才压过规则锚点的，2780 次更新对应 REFIL 在 12.5 万步、D≈4.4 的位置。跨规模预算以环境步定义，因此每环境步的更新次数按实验控制变量统一，而 SPECTra 的 Adam、`lr: 0.001` 与 TD(λ=0.6) 属于方法定义，保持官方值不动。每回合复用次数为 `training_iters × batch_size / batch_size_run`，统一后六方法同为 32 次且与 buffer 大小无关，该取值已在本轮前三方法上跑满 100 万步验证稳定；DCG 若只改 `training_iters` 而保留 500 局缓冲，会形成官方与本项目都不存在的组合，且 12 种池内组合平均只有 42 局，故一并对齐。

`scripts/train.py --stage train` 默认 `--group main`：DCG、SPECTra × 种子 0/1/2 共 6 项，并发槽固定为 3。GNN-QMIX 只保留已完成的 seed 0，后续种子不再排队。B0 / B2 / REFIL 的种子 1/2 用 `--group baseline`，不与主实验抢槽。已完成且存在 `final.pt` 或 `best.pt` 的直接跳过。终端仍只刷新当前进行中的状态行。

**外推改为三条线。** 等规模为 10/15/20/25/30/40v 同数 K2；1:2 线为 2v4/4v8/8v16/15v30/20v40 K2；目标图为 10/15/20/30v 同数的 K=2/4/6，按规模着色、按方法分线型。评估接口上界为 40/40/6；训练回放只建训练池 10/10/3。旧配置行留在 CSV，报告不再列出。已有配置的 300 局沿用。DCG / GNN / SPECTra 共用该 `TEST_POOL`。集合方法的 checkpoint 不把槽位数写进权重，`load_policy` 按当前接口重建 pad 后再加载。

`parallel_runner` 中预算边界截断的回合此前用累计奖励覆盖摘要里的 `return`，塑形后该值不再是物理回报，改为保留环境摘要的物理值并另记 `shaped_return`。

## 本轮吞吐诊断与恢复修正

首次 REFIL 4-worker 独占尝试最后已提交 2495 物理步、56 次学习更新，训练循环约 110.641 秒。执行工具的中断直接结束进程组，未触发 Python 的受控停止处理；该尝试没有恢复点，至多一批 400 步未落盘量无法精确核实。用户已批准优化后重做首个 20000 步测量，失败片段单列，其他九格配额不变。原始记录与日志保留。

调度器新增 	rain.py --stage stop 请求入口，由调度器和训练循环读取后在完整采样与学习批次结束时保存；工具会话不再用强制中断。恢复时已完成任务保留，未启动任务正常开始。编码器仅跳过必然零输出的死亡/padding 查询，块大小由 256 调整为 1024；存活但可见 key 为空的 REFIL 查询仍保留。网络参数、算法、学习超参数、8 次更新与训练分布不变；数值等价和直接耗时证据见正式报告。

随后通过可靠停止入口保存3425步，确认网络/目标/优化器、76回合回放和全部随机状态完整，并恢复后验证步数与8次批次更新连续。真实回放诊断后冻结不参与优化的target参数，并对FlexQMixer按256个独立状态分块重算，损失与在线梯度保持一致。进一步在13458步可靠保存后规范仅filled=0的时间padding掩码，mixer按filled压缩；有效最终观测和bootstrap保留。所有分段累计步数保留，最终实现的测速起点/步数/耗时由measurement_*字段明确记录，不把不同实现区间混作单一测速。

## ALMA 分层臂的接入（v3 追加）

此前本项目只把 ALMA 当作共享主干，方法本身未适配：`METHODS` 无 `alma`、`load_config` 把 `hier_agent.task_allocation` 与 `copa` 硬置为关闭、回放 scheme 缺 `entity2task_mask`/`task_mask`/`hier_decision`、runner 的观测白名单挡掉子任务掩码、`SharedEntityMAC._build_agents` 不构造上层网络、训练循环不调用 `alloc_train_aql`、检查点不保存上层权重与其两个优化器、最终评估的 `FrozenPolicyAdapter` 不提供子任务输入。本轮按用户决定补全该链路并作为第六条正式臂并入 `main_v3`，不新建输出目录或报告。

**形态。** 低层与 B2-QMIX-Atten 同构（同一注意力编码器、FlexQMIX、GRU、`softmax_mixing_weights`、8 次批次更新），差异只有分层：`hier_agent.task_allocation: aql`，`agent.subtask_cond: mask`，`mixer_subtask_cond` 按上游 `run.py` 的关系跟随为 `mask`。选 `mask` 而非 `full_obs`：掩码在编码器之前改 `obs_mask`，本项目的共享编码器原样兼容；`full_obs` 走 `inputs['task_embeds']`，而该键只有上游 `EntityBase` 会读，本项目的编码器会静默丢弃它。其余 `hier_agent` 超参数保持 `config/default.yaml` 的官方默认（`action_length: 5`、`n_proposals: 32`、`alloc_critic: standard`、`alloc_policy: autoreg`、`pi_pointer_net: true`、`decay_old: 150000`、`entropy_loss: 0.01`、`max_bs: 400`），上游未提供 ALMA 的算法 yaml。

**子任务集合与归属（公平性）。** 一个存活目标一个子任务。目标行归属自身子任务；红方行在决策点由上层分配覆盖；**蓝方行按当前最近目标归属**，这是任何方法都能从同一份实体表算出的几何量。`HADWrapper` 内有蓝方规则分组的真实指派，但那是蓝方私有意图，用它等于把"谁打哪个目标"直接送给红方，其余五臂只能自行推断，故不使用。代价是归属随蓝方移动抖动，上层每 5 步才重决策。这一步不是可选项：默认的 pointer-net 上层把非智能体实体按子任务聚合成任务表示，若只有目标行归属自身，每个任务的表示就退化成该目标自身的嵌入、计数恒为 1，上层无从区分"被围攻的目标"和"无人问津的目标"。

**奖励分解。** `subtask_cond` 非空时上游 learner 改用 `task_rewards`/`tasks_terminated`，HAD 原先没有这两项。按目标分解：`task_rewards[j] = -Δ(目标 j 的伤害)`，势函数塑形项按蓝方所属目标一并分解，逐步满足 $\sum_j$ `task_rewards` $=$ 团队奖励（含塑形），红方全灭折叠尾段按同一折扣同步累加、截断退还也同步退还。防御子任务不会中途完成，因此活动子任务与回合同时终止，padding 槽恒为 0（上游按 `task_has_agents` 掩掉）。新增 `V1-subtask-decomposition` 校验该恒等式与"逐目标伤害之和 $=D$"（8v8 K3、3 个种子，实测最大偏差 1.1e-15）。

**跨规模的任务维。** `TaskEmbedder` 把子任务 one-hot 宽度写进权重（`nn.Linear(n_tasks + n_extra_tasks, embed_dim)`），而训练按训练池 pad 只有 3 个目标槽、评估接口有 6 个。`configs/alma.yaml` 设 `n_extra_tasks: 3`，训练期宽度 3+3=6；`load_policy` 按检查点保存的宽度反推 `n_extra_tasks = 6 - n_tasks = 0`，宽度不变。这是 ALMA 自己的机制（多学几个嵌入以泛化到更多子任务），必须在开训前设好，事后无法补救。新增 `V3.alma.subtask_width_extrapolation` 直接验证"K≤3 训练的权重在 K=6 上加载并前向"。

**保持关闭 PopArt。** 上游 ALMA 默认开 PopArt，本项目按方案全局关闭，上层 AQL 的 TD 目标因此是原始尺度（分段奖励已除以 `action_length`，量级约 −1）。这同时暴露了上游一处未被覆盖的分支：`StandardAllocCritic.denormalize` 先访问只在 PopArt 下存在的 `targ_rms`，关闭时抛 `AttributeError`，改为先判 `use_popart`。另修两处现代 PyTorch 的掩码 dtype：`AttentionHyperNet._duplicate_per_task` 与 `AutoregressiveAllocPolicy` 的两个 `masked_fill` 收到 uint8 掩码，按本文上方"掩码运算使用 bool"的同一处置改为 bool。

**其余接线。** `make_scheme` 在 `multi_task` 下补齐 5 个键；runner 的观测白名单放行子任务掩码，并按 `action_length` 写 `hier_decision`（回合首状态为决策点，终止状态不是），post-transition 带上分解奖励；检查点与 `resume.pt` 增加 `mac.alloc_policy`/`alloc_critic` 与 `alloc_pi_optimizer`/`alloc_q_optimizer`/`last_alloc_target_update_episode`；`FrozenPolicyAdapter` 用与环境同一个 `features.task_masks` 提供子任务输入与同一决策时钟，因此最终评估路径与训练时行为一致。上层的 Q/π 更新与低层同在 `training_iters` 循环内（上游位置），并按 `decay_old` 过滤过旧回合；上游该调用挂在设备判断的 `elif` 上，`buffer_cpu_only: true` 时永不执行，本项目不沿用该写法。

**新增检查（8 项，全部通过）。** 除与其余五臂同套的置换不变/等变、变编队参数量、死亡 no-op、GRU 状态重置、编码器执行等价、单调性、mixer 对齐、上一动作与注意力掩码外，另有三项针对分层：`V3.alma.allocation_gates_observation` 验证分层有因果通路（换分配会改变低层 Q，且本子任务外的实体对该智能体不可见）——这正是"只接上层、低层不条件化"会退化成空转的那一类缺陷；`V3.alma.allocation_validity` 验证每个存活智能体恰好占一个活动子任务、死亡智能体不占、决策写回回放；`V3.alma.subtask_width_extrapolation` 如上。

**吞吐。** CPU 单核对照（同为 4 worker、同一时刻的旁路短跑）：B2 约 20 秒/采集学习周期，ALMA 约 39 秒，即约 2 倍，代价来自逐任务 mixer 前向与上层 32 个提案的自回归采样。GPU 正式预算据此按 B2 的约 2 倍估计。
