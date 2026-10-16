# LEAF 实验章节（Section V）撰写方案

更新：2026-10-15。本文是论文第五章 Experiments 的设计蓝本，回答三件事：实验章节写什么，每个结论由哪份数据支撑，图表在 TASE 双栏版面中如何安排。撰写 `Papers/1015/sections/sec_5_6` 和重构实验图表时以本文为准。

数据全部来自 `Open-SCORE/outputs/main` 中已保存的结果，本次构思时逐项复核过。标注为**派生统计**的数值只读取已有逐局记录重新汇总，没有新增训练、对局或模型。叙事参考 [story.md](../Papers/story.md)，其中与数据不完全一致的表述，在第 2 节给出修正写法。图表部分只规定展示什么信息、用单栏还是双栏，具体视觉设计留给作图环节。

---

## 0. 速览

### 0.1 章节骨架（双栏约 4 页）

| 小节 | 回答的问题 | 图表 |
|---|---|---|
| V-A Experimental Setup | 在什么任务、规模协议、基线与统计口径下比较 | 表 E1 规模协议；表 E2 方法与训练设置 |
| V-B Zero-Shot Scale Generalization | 只在小规模训练一次的策略，部署到更大规模时能否更好地保护目标 | 图 E1 训练曲线与三类外推（双栏，必需）；表 E3 主结果（双栏） |
| V-C Ablation | 哪些设计带来了规模泛化 | 表 E4 单因素消融链（双栏） |
| V-D Mechanism Analysis | LEAF 为什么有效：三个核心特点 | 图 E2 结构而非容量（单栏）；图 E3 记忆后置（单栏）；图 E4 任意执行深度（双栏） |

### 0.2 V-D 展示的三个核心特点

1. **收益来自威胁情境构建的计算组织，与参数量无关。** 参数与 LEAF 相同的 REFIL-matched 损伤高 2.3–3.9 倍；参数多 60% 的 Unshared rounds 没有一致收益；情境构建还明显加快学习。
2. **记忆放在读取之后，情境解读在规模变化时更稳定。** LEAF 的 GRU 记忆随部署规模明显漂移（分类器识别规模的 AUC 为 0.81–0.89），而 query 所用的当前自身输入几乎不带规模信息（0.54–0.62）。让记忆进入 query 后，与 LEAF 的差距从 ID 的 30% 扩大到外推组的 54%–77%；在四个大规模配置的四项保护指标上，Memory query 全部介于 REFIL 与 LEAF 之间。
3. **随机深度训练把核心情境前置到前两轮，任何执行深度都是可用策略。** 前两轮约占 99% 融合权重；执行 1–6 轮的损伤几乎不变；同结构、同 query 下只按固定四轮训练的模型，执行一轮时损伤约为其四轮时的 15 倍。部署时可以用一轮换取约 29% 的决策时延节省。

### 0.3 本次数据核对得到的新事实

1. LEAF 在全部 **24 个评估配置**上的平均损伤都是 8 个方法中最低的。
2. 三个 LEAF 模型中**最差的一个**，在四个组别上都优于每个学习基线的**最好模型**，也优于 Rule。这是不依赖显著性检验的稳健性陈述。
3. 逐键比对 8 个 LEAF 模型的保存配置后发现：除 LEAF 外，7 个历史变体都使用带记忆的 query，且彼此之间只差一个开关。因此存在一条**每步只改一项**的消融链：Direct read → One round → Fixed depth → Memory query → LEAF。1014 版"不是单因素消融"的写法可以改为清晰的单因素叙事（第 6 节）。
4. `aggregates.json` 的 `comparison.history_interventions.depth` 保存了 Memory query、Fixed depth、Unshared rounds 在 **300 场正式池**上的冻结执行深度评估（主报告 §4.9 只用了其中 100 场）。Memory query 与 Fixed depth 只差训练深度一项，构成"随机深度 vs 固定深度"的干净对照。
5. **受损回合比例 P** 与**最差 10% 回合平均损伤 CVaR** 可由逐局 D 直接算出，覆盖全部 8 个方法，包括没有保存碰撞和拦截字段的 19 维 SPECTra、TransfQMix。LEAF 在四组都最低。
6. `concurrency` 是恢复训练时允许改变的调度字段，属于 `RESUME_MUTABLE`，不改变学习条件。Memory query seed1 的并发差异不构成算法混杂。
7. 机制原始记录保存了 LEAF 每个决策的完整 GRU 记忆。经用户同意做了只读的记忆漂移分析（第 7.2 节）：记忆随规模漂移，当前自身输入基本不变。这补上了 story 中"小规模学到的记忆在大规模下偏离训练分布"的直接证据。

### 0.4 已确认的决定

| 事项 | 决定 |
|---|---|
| 硬件 | 只在时延测量处说明本机 RTX 5070 Ti 配置，训练硬件不单独描述 |
| 载具模型 | 暂时模糊化，只写策略在 SkyHAD 仿真环境中调用，不写固定翼或质点，后续由用户调整 |
| 派生统计 P 与 CVaR | 进入主表，并替代 story 中的碰撞主张 |
| 可选分析 A（记忆漂移） | 已完成，结果见第 7.2 节 |
| 可选实验 B（补训） | 不做，原因见第 12 节 |

---

## 1. 证据地图

每个主张对应一个实验和一个图表，引言中的主张都要能在这里找到落点。

| 问题 | 主张 | 主要证据 | 图表 | 证据强度 |
|---|---|---|---|---|
| Q1 规模泛化 | 小规模训练一次，部署到至多 50 架/方或 12 个目标时，LEAF 的目标损伤最低 | 四组均值最低；24 配置全部最低；最差种子优于基线最好种子；N 从 15 到 50 损伤维持在 0.56–0.61，六个学习基线全部单调上升；人数与联合外推中 Rule 优于所有学习基线，LEAF 在每个 N 都低于 Rule；独立场景池排序复现 | 图 E1、表 E3 | 强 |
| Q1 保护质量 | LEAF 让目标完全不受损的回合最多，尾部损伤最低 | P 与 CVaR 四组最低（派生统计，8 方法全覆盖） | 表 E3 | 强 |
| Q2 情境构建 | 读取前先构建威胁情境是泛化的基础 | Direct read → One round（单因素）四组降低 33%–39% | 表 E4 | 强（均值），种子 2/3 |
| Q2 多轮精炼 | 一轮不够，多轮精炼在外推中收益更大 | One round → Fixed depth（单因素）降低 17%–51%，N/Joint 三种子全部改善 | 表 E4 | 强 |
| Q2 记忆位置 | 当前状态 query 在外推中优势扩大 | Memory query → LEAF（单因素）：差距 30% → 54%–77% | 表 E4、图 E3 | 中强（均值明确，ID 3/3、OOD 2/3） |
| Q3 结构而非容量 | 收益不是参数量带来的 | REFIL-matched、Unshared rounds、同参数变体、学习速度、冻结旁路 | 图 E2 | 强 |
| Q3 记忆后置 | 记忆随规模漂移而当前自身输入不漂移；记忆进入 query 会让外推更差且不稳定；记忆在下游仍有用 | 记忆漂移 AUC 0.81–0.89 vs 自身输入 0.54–0.62（每个种子一致）；保护指标 16/16 有序；GRU 重置 8/9 变差 | 图 E3 | 强 |
| Q3 任意深度 | 随机深度让每个出口可用，部署时可按算力选轮数 | 融合权重；R1–R6 平坦；固定深度浅出口崩溃（单因素）；时延；R5/R6 外推 | 图 E4 | 强 |

---

## 2. 对 story.md 的辩证核对

story 的主线与数据一致，以下逐条说明哪些可以直接写、哪些需要改写。

| story 表述 | 数据核对 | 判断 | 建议写法 |
|---|---|---|---|
| 训练至多 10 防御者、3 目标，部署到至多 50 防御者、12 目标 | 最大配置为 (50,50,2) 和 (30,30,12)，没有 (50,50,12) | 需微调 | up to 50 vehicles per side **or** 12 targets，避免暗示二者同时达到 |
| 损伤比各组最强的学习或规则基线低 43% 到 64% | ID −43.4%（REFIL）、K −44.5%（REFIL）、N −62.8%（Rule）、Joint −64.2%（Rule） | 准确 | 原样使用 |
| 友军碰撞也少于所有学习基线 | 19 维 SPECTra、TransfQMix 没有保存碰撞字段；Rule 在 N/Joint 的碰撞比例低于 LEAF | 不能原样写 | 摘要与引言改用 P 和 CVaR 主张（8 方法全覆盖）。碰撞只在机制节对 LEAF、Memory query、REFIL 报告 |
| 不做情境构建、直接读原始实体，四组都明显变差 | Direct read 每个种子都劣于 LEAF 每个种子；但 Direct read 同时使用记忆 query | 准确，归因需改 | 归因用单因素对照 Direct read → One round（−33%–39%） |
| 让记忆进入查询后，差距在外推组中明显扩大 | +30%、+54%、+74%、+77%；种子层面 ID 3/3、OOD 各 2/3，主要由 seed1 拉开 | 均值准确 | 以均值和 SD 陈述，并补充"种子间离散随之增大"。不写 in every seed |
| 规模一变，小规模学到的记忆就会偏离训练分布 | 只读分析：LEAF 的 GRU 记忆识别部署规模的 AUC 为 0.81–0.89，当前自身输入只有 0.54–0.62，三个种子一致 | 准确 | 可以直接写，并给出两组 AUC。注意分析对象是 LEAF 自己的记忆，Memory query 的记忆没有保存 |
| 专门训练的单轮模型也明显落后 | 相对 LEAF 成立但混入 query 差异；单因素对照 One round → Fixed depth 四组改善 | 准确，归因需改 | 用 One round → Fixed depth 支撑"多轮精炼"，而非与 LEAF 直接比 |
| 约 99% 融合权重落在前两轮 | 98.76%–98.91% | 准确 | 原样使用，并说明它描述读取分配，不等于因果贡献 |
| 只执行两轮时，与四轮相差约 1% | All24 R2 0.7928 vs R4 0.7845，+1.06% | 准确 | 原样使用。不要扩展成每个配置都如此 |
| 多深度训练的算子只执行一轮，联合外推损伤约为专门单轮模型的一半 | 1.097 vs 2.157，比值 0.51；跨评估池，且混入 query 差异；同 query 的单因素版本（Memory query R1 vs One round）只好约 10% | 数值准确，归因需谨慎 | 作为观察写出，不写成"因为随机深度"。随机深度的因果证据改用固定深度浅出口崩溃 |
| 超出训练范围到五轮、六轮时性能稳定 | All24 R5 0.7851、R6 0.7849，变化 <0.1% | 准确 | 原样使用 |
| 随机深度把核心信息凝练到前两轮，在性能和效率间取得平衡 | 融合权重、出口平坦、时延 R1 −29%、R2 −18% | 准确 | 补充：部署者显式选择轮数，融合权重本身不会自动提前停止 |
| 同一个策略能用于任意规模 | 只评估到 50 架/方、12 个目标 | 过强 | across all evaluated scales |
| 两个运动状态相同的攻击者，单看自身分不出来（动机） | 受控覆盖线性探针接近 50% 机会水平，没有形成表征层面的证据 | 动机可保留，实验不宣称表征解码了覆盖关系 | 实验章节只用行为与结构证据，不写 LEAF learns to encode coverage |

另有两点需要提醒写作者：

- story 批评实体 Transformer "把记忆当作 token"与"层数固定"。实验中对应的受控证据是我们自己的 Memory query 变体与固定深度变体，TransfQMix 只提供整体对比。不要把 TransfQMix 的落后直接归因于这两点。
- Stage fusion 与参数共享在正式深度 R4 下没有独立的性能收益（Last read、Unshared rounds 与 LEAF 接近）。它们的价值体现在执行深度变化与参数效率上。第四章如有"融合提升性能"的表述，建议改为"融合让同一组权重在每个出口都给出完整决策特征"。

---

## 3. 版面规划（TASE 双栏）

### 3.1 尺寸约束

- 单栏宽 `\columnwidth` ≈ 3.5 in（88.9 mm），双栏宽 `\textwidth` ≈ 7.16 in（181.9 mm）。
- 缩放后图中最小字号不低于 7 pt。双栏 1×4 横条图每个子图约 1.7 in 宽，需要共享图例和纵轴标签。
- 表格在 `\footnotesize` 下，`0.41±0.10` 形式（两位小数）的 4 列数值可以放进单栏；三位小数或超过 4 列带 ± 的数值需要 `table*`。
- IEEE 的 `figure*`/`table*` 只出现在页面顶部，且在源码位置的下一页出现，应把它们放在对应小节开头之前。

### 3.2 浮动体清单

| 编号 | 内容 | 栏宽 | 优先级 | 小节 |
|---|---|---|---|---|
| 表 E1 | 规模协议：训练分布与 24 个评估配置 | 单栏 | 必需 | V-A |
| 表 E2 | 方法与训练设置：共享设置、各方法差异、部署参数量 | 单栏（若过长改 `table*`） | 必需 | V-A |
| 图 E1 | 8 方法训练曲线 + K/N/Joint 外推，1×4 | 双栏 | 必需 | V-B |
| 表 E3 | 主结果：D±SD、P、CVaR，四组 | 双栏 | 必需 | V-B |
| 表 E4 | 单因素消融链与旁支 | 双栏 | 必需 | V-C |
| 图 E2 | 结构而非容量：参数量–损伤散点 + 学习曲线 | 单栏 | 推荐 | V-D.1 |
| 图 E3 | 记忆后置：记忆漂移 + 分组差距 | 单栏 | 推荐 | V-D.2 |
| 图 E4 | 任意执行深度：出口稳健性 + 融合权重 + 时延–损伤 | 双栏 | 必需 | V-D.3 |

正文共 4 张图（2 双栏、2 单栏）、4 张表（2 单栏、2 双栏）。若篇幅紧张，按以下顺序压缩：先把表 E3 改为单栏只列 D（P、CVaR 进正文与附录），再把图 E2 并入正文数字，最后把表 E2 的方法细节移到附录。

前面章节已有的浮动体：引言 Fig. 1（单栏，(a) 威胁情境示意，(b) 单次读取、实体 Transformer 与 LEAF 的信息流对比），第三章的场景图与设置表，第四章的框架图（双栏）与算法框。引言不再有设计对比表，也不画规模图。与本章的分工：规模协议只由表 E1 呈现；表 E2 列出各方法的协调机制、记忆形式、训练设置与部署参数量。全文双栏浮动体为框架图、图 E1、图 E4 与双栏表 E3、E4，每个都只能出现在页面顶部，排版时需要错开。

### 3.3 篇幅预算

| 小节 | 文字 | 浮动体 | 合计 |
|---|---|---|---|
| V-A | 0.6 页 | 表 E1、表 E2 约 0.35 页 | ≈1.0 页 |
| V-B | 0.5 页 | 图 E1、表 E3 约 0.5 页 | ≈1.0 页 |
| V-C | 0.4 页 | 表 E4 约 0.2 页 | ≈0.6 页 |
| V-D | 0.9 页 | 图 E2、E3、E4 约 0.65 页 | ≈1.5 页 |

### 3.4 全文视觉约定

- 方法顺序在所有图例和表格中一致：QMIX、REFIL、DCG、ALMA、TransfQMix、SPECTra、Rule、LEAF。表格中 LEAF 放最后一行并加粗。
- 每个方法在所有图中颜色和标记固定。LEAF 用最醒目的颜色和更粗的线，Rule 用虚线中性色。LEAF 变体与基线不共用颜色。
- 学习方法的误差带或误差线都是三个训练种子的样本 SD，Rule 没有训练种子，只画参考线。
- 损伤跨度大的图（外推、执行深度）建议使用对数纵轴，否则 LEAF、REFIL、Rule 在底部挤成一团。
- 训练范围在每个外推子图里用浅色底纹标出（K ≤ 3、N ≤ 10、R ≤ 4）。

---

## 4. V-A 实验设置

开头一句说明本节交代任务、规模协议、基线和统计口径，让后续结果可复现、可比较。环境细节在第三章定义，这里只给复现必需的数值。

### 4.1 环境与任务

| 项目 | 设定 |
|---|---|
| 平台与任务 | SkyHAD 二维 damage 任务：红方防御者保护 K 个静止目标，蓝方攻击者由固定的 reactive–rush 策略控制 |
| 载具与动作 | 载具在 SkyHAD 仿真环境中推进，速度 35–120 m/s，加速度上限 40 m/s²；策略每步输出 9 个离散加速度指令之一（零加速度与 8 个方向） |
| 交战规则 | 两机扫掠间距 ≤20 m 同时损毁；距离小于 200 m 自动开火，伤害在 200–400 m 线性衰减；开火后的载具自毁（攻防双方相同） |
| 对手 | 按公开覆盖与距离抽取一个目标，全体攻击者冲向该目标；每 5 步或出现伤亡时重新选择；指派对红方不可见 |
| 时长 | Δt = 1 s，最多 100 步；全部攻击者被消灭时自然结束，红方全灭后损伤继续累计 |
| 奖励 | 每步 −ΔD_t；训练时另加势函数塑形（β = 1，距离尺度 4000 m）；评估只统计未塑形的累计损伤 D |
| 观测 | 以接收者为中心的实体表：位置、速度、存活、生命值、目标损伤份额、类型独热（10 维），红方实体再拼接上一动作（9 维），死亡与填充实体被掩码 |

载具模型按用户决定暂时模糊化：正文只写策略在 SkyHAD 仿真环境中调用、输出离散加速度指令，不写固定翼、质点等模型类型，后续由用户统一调整。记录备查：正式结果使用 `spatial_dim=2` 的平面设定。

### 4.2 规模协议（表 E1，单栏）

训练时每局独立采样 N ∈ {4,6,8,10}（N_R = N_B = N）和 K ∈ {1,2,3}。评估冻结 final 权重，直接部署到以下 24 个配置，所有配置攻防人数相等。

| 组别 | 配置数 | 配置 (N, K) | 相对训练范围 |
|---|---:|---|---|
| ID | 4 | (4,2)、(6,2)、(8,2)、(10,3) | 训练范围内 |
| K-OOD | 4 | (10,4)、(10,6)、(10,9)、(10,12) | 目标数扩大到 4 倍 |
| N-OOD | 6 | (15,2)、(20,2)、(25,2)、(30,2)、(40,2)、(50,2) | 人数扩大到 5 倍 |
| Joint-OOD | 8 | (15,4)、(15,6)、(20,4)、(20,6)、(30,4)、(30,6)、(30,9)、(30,12) | 二者同时扩大 |
| Extra | 2 | (5,2)、(10,2) | 训练范围内，只进入 All24 和曲线锚点 |

表注写明：每个配置使用 300 个共同初始场景（场景种子 9000–9299），各方法共享；独立确认池为 110000–110299。

### 4.3 基线选择与适配（表 E2 下半部分）

**选择依据。** 基线按第二章相关工作的四个方法族选取，每族至少一个代表，每族回答一个对照问题。分族与第二章各小节一一对应，正文和表 E2 沿用同一分法：

| 方法族（第二章小节） | 方法 | 对照的问题 |
|---|---|---|
| 分配型协调（II-A） | Rule、ALMA | 把威胁情境压缩成离散指派能否随规模泛化。Rule 每步求解集中式整数指派，ALMA 学习目标级分配 |
| 值分解与显式协调（II-B） | QMIX（实体注意力适配）、DCG | 改进个体效用的组合方式（单调混合、成对 payoff）能否替代读取前的情境构建 |
| 跨规模的实体方法（II-C） | REFIL、SPECTra | 用自身 query 一次读取各自嵌入的实体是否足够。REFIL 是 LEAF 的宿主，是最接近的对照 |
| 实体 Transformer（II-D） | TransfQMix | 多层实体交互、记忆 token 参与每一层、深度固定，能否泛化 |

引言 Fig. 1(b) 只用信息流对比单次读取、实体 Transformer 与 LEAF 三种组织方式，没有逐个方法的设计表。表 E2 因此保留下表中各方法的协调机制与记忆形式，并列出训练设置、优化目标与部署参数量。

**共同适配。** 所有学习方法使用同一实体接口（19 维特征与相同掩码）、同一组 9 个动作、同一对手、同一训练分布与 1M 物理步预算，且都带时序记忆。每个基线保留原结构与学习规则，只替换输入接口、动作头与掩码。可变规模由各自的机制处理：FlexQMIX 实体超网络混合器、SPECTra 的集合超网络、TransfQMix 的 Transformer 混合器；DCG 在部署人数上构建完全图；ALMA 把每个受保护目标作为一个子任务。SPECTra 与 TransfQMix 按 19 维输入重训，保留各自实现的 Adam 与 TD(λ)。

各方法要点（写入表 E2 或正文一句话说明）：

| 方法 | 协同机制 | 记忆 | 优化与目标 | 部署参数量 |
|---|---|---|---|---:|
| QMIX | 实体注意力 actor + 单调实体超网络混合器（无 imagined 分组） | GRU-64 | RMSprop 5e-4，一步 Double-Q | 102,025 |
| REFIL | QMIX + 随机实体分组辅助 TD（λ = 0.5） | GRU-64 | 同上 | 102,025 |
| DCG | 完全协调图上的效用与成对 payoff，8 轮 Max-Sum | GRU-64 | 同上，ε 退火 50k 步 | 112,474 |
| ALMA | 学习的目标级任务分配（每 5 步、32 个候选）+ 任务条件 actor | GRU-64 | RMSprop 5e-4 + 分配 Q 学习 | 282,827 |
| TransfQMix | Transformer actor 与混合器（2 层、4 头、宽 32） | 记忆 token | Adam 1e-3，TD(λ = 0.6) | 50,729 |
| SPECTra | 类型化投影 + 自身查询注意力 actor，集合超网络混合器 | GRU-64 | Adam 1e-3，TD(λ = 0.6) | 47,433 |
| Rule | 每步选出离防御者最近的 min(N_R, N_B) 个攻击者，求解 0–1 整数指派：最小化防御者到攻击者一步匀速预测位置的总距离，每个防御者负责一个攻击者，每个入选攻击者至少一人负责；再选最接近拦截速度的加速度基元 | 无 | 无训练 | — |
| LEAF | REFIL 宿主上的循环威胁情境推理 | GRU-64 | RMSprop 5e-4，一步 Double-Q + imagined | 175,754 |
| REFIL-matched（仅机制节） | REFIL 在注意力后加残差 MLP 64→576→64 | GRU-64 | 同 REFIL | 176,393 |

部署参数量统计自 seed0 的 `final.pt` 中部署时使用的网络（DCG 含 utility 与 payoff 头，ALMA 含分配策略与评分网络），不计训练用混合器和注意力缩放缓冲。

### 4.4 训练设置（表 E2 上半部分）

| 设置 | 取值 |
|---|---|
| 预算 | 每种子 1M 物理环境步，训练种子 0、1、2 |
| 采样与更新 | 8 个并行环境；每收集一批更新 8 次 |
| 回放与批量 | 回放 5000 局，批量 32 局 |
| 价值学习 | γ = 0.99，梯度裁剪 10，目标网络每 200 局硬更新 |
| 探索 | ε 从 1.0 线性降到 0.05，用时 100k 步（DCG 50k） |
| LEAF 结构 | 实体编码宽 128；工作宽度 d = 64；4 头注意力；FFN 宽 128；GRU 64；W_f 零初始化 |
| LEAF 深度 | 训练时 L ~ U{1,2,3,4}：采集时每局抽一次，更新时每个 minibatch 抽一次，online、target 与 imagined 分支一致；部署 L_exec = 4 |
| LEAF 损失 | (1 − λ) L_TD + λ L_imagined，λ = 0.5 |
| 验证 | 每 20k 步一次，共 50 个点；4 个 ID 配置各 25 局，独立场景池 9500–9599；只画学习曲线，不用于选模型 |

### 4.5 评估协议、指标与统计

- **零样本部署。** 使用完成预算的 final 权重，冻结参数，贪心动作，不做任何微调。best 权重不参与正式比较。
- **共同场景。** 每个模型、每个配置运行同一组 300 个初始场景。对手的目标选择依赖防御者位置，所以同一初始场景在不同方法下会演化出不同的对手行为，这是闭环防御评估的应有之义。
- **指标。**

| 符号 | 含义 | 方向 | 覆盖 |
|---|---|---|---|
| D | 整局累计目标损伤（主指标，不按目标血量封顶） | ↓ | 全部方法 |
| P | 至少一个目标受损的回合比例 | ↓ | 全部方法（派生统计） |
| CVaR | 每个配置中损伤最高的 10% 回合（300 场中的 30 场）的平均 D，描述尾部风险 | ↓ | 全部方法（派生统计） |
| I | 被拦截攻击者占初始攻击者比例 | ↑ | 仅机制节的 LEAF、Memory query、REFIL |
| C | 因友军碰撞损失的防御者占初始防御者比例 | ↓ | 同上 |

- **汇总。** 先在每个模型、每个配置内对场景取平均，再在组内对配置等权平均，最后报告三个训练种子的均值 ± 样本 SD。Rule 没有训练种子，其 SD 为场景 SD，用 † 标注。
- **统计措辞。** 只有三个训练种子，正文不写 significantly。用"四组均值都更低"、"三个种子都改善"、"每个配置都最低"这类可直接核对的表述。
- **独立确认。** LEAF、REFIL、SPECTra、TransfQMix 另在场景 110000–110299 上评估，作为一句话的复现说明。

### 4.6 硬件与软件

按用户决定，正文不单独描述训练硬件，只在时延测量处说明本机配置。建议把这句话放在 V-D.3 报告时延的位置，V-A 不再重复：

- Decision latency is measured on a workstation with an NVIDIA GeForce RTX 5070 Ti GPU and an Intel Core i7-14700KF CPU, using PyTorch 2.13 with CUDA 13.0.

时延测量的完整口径（写入图注或附录）：

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti，16 GB，驱动 610.88 |
| CPU 与内存 | Intel Core i7-14700KF（20 核 28 线程），32 GB |
| 软件 | Windows 11；Python 3.10；PyTorch 2.13.0，CUDA 13.0 |
| 设置 | FP32，关闭 TF32，CPU 单线程，batch 为 1，调用前后同步 CUDA；计时为整支防御队伍一次决策，不含环境推进 |

---

## 5. V-B 主对比：零样本规模泛化

开头一句：本节检验只在小规模训练一次的策略，冻结后部署到更多目标、更多载具或二者同时增加时，能否比现有方法更好地保护目标。

### 5.1 图 E1：训练曲线与三类外推（双栏，必需）

推荐布局为 1×4 横条（7.16 in × 约 2.1 in），图例在上方排成一行，四个子图共享方法编码。

| 子图 | 横轴 | 纵轴 | 内容 | 关键信息 |
|---|---|---|---|---|
| (a) Learning on ID validation | 环境步 0–1M | 验证 D | 7 个学习方法的均值曲线与种子 SD 带；Rule 为 ID 水平参考虚线 | LEAF 下降最快、终点最低 |
| (b) Target-count extrapolation | K ∈ {2,3,4,6,9,12}，N = 10 | D（建议对数） | 各配置均值与种子 SD；K ≤ 3 加训练范围底纹 | 每个 K 下 LEAF 最低 |
| (c) Team-size extrapolation | N ∈ {10,15,20,25,30,40,50}，K = 2 | D（建议对数） | 同上；N ≤ 10 加底纹 | LEAF 从 15 到 50 基本持平，所有学习基线随 N 上升 |
| (d) Joint extrapolation | 8 个联合配置，按实体总数排序，刻度写 15/4 等 | D（建议对数） | 同上 | 每个联合配置 LEAF 最低 |

- (b) 中 K = 2、3 的点分别来自 (10,2)（Extra）和 (10,3)（ID）；(c) 中 N = 10 的点来自 (10,2)。这些锚点把训练边界画进图中，读者能直接看到越过边界后各方法如何变化。
- (a) 曲线默认不平滑。如果为可读性做平滑，窗口不超过 3 个验证点，并在图注写明。
- 备选布局：2×2，每个子图更大，(d) 改为方法×配置的热力图并标数值。
- 去掉 1014 版的第 (e) 子图（独立确认池），确认结果改为正文一句话。

图注要点：训练只用 N ≤ 10、K ≤ 3；(a) 为固定 ID 验证池；(b)–(d) 为冻结 final 策略在每配置 300 个共同场景上的结果；误差为三个训练种子的样本 SD；Rule 为确定性规则，画参考线。

### 5.2 表 E3：主结果（双栏）

三个指标块，每块四个组别。D 给均值 ± SD，P 与 CVaR 给均值（SD 放附录）。每列最优加粗、次优下划线，最后一行给 LEAF 相对各组最强对手的变化。

**D，均值 ± 三训练种子 SD（来源：`comparison.group_summary`）**

| 方法 | ID | K-OOD | N-OOD | Joint-OOD |
|---|---:|---:|---:|---:|
| QMIX | 2.318±1.431 | 5.009±2.497 | 5.139±3.111 | 7.415±3.970 |
| REFIL | 0.715±0.049 | 1.796±0.304 | 2.209±0.277 | 3.308±0.303 |
| DCG | 2.028±0.437 | 5.839±1.920 | 5.021±2.658 | 8.092±3.909 |
| ALMA | 1.326±0.193 | 3.022±0.344 | 3.605±0.742 | 5.375±1.036 |
| TransfQMix | 1.529±0.283 | 3.953±0.495 | 4.024±0.968 | 6.630±1.203 |
| SPECTra | 2.485±0.209 | 6.743±0.528 | 5.049±0.589 | 9.838±0.490 |
| Rule † | 1.462±0.717 | 3.212±1.126 | 1.545±0.881 | 3.060±0.976 |
| **LEAF** | **0.405±0.097** | **0.996±0.233** | **0.575±0.235** | **1.097±0.105** |
| 相对最强对手 | −43%（REFIL） | −45%（REFIL） | −63%（Rule） | −64%（Rule） |

**P（%）与 CVaR，派生统计（均值 ± 三训练种子 SD，Rule 无 SD）**

| 方法 | P: ID | K | N | Joint | CVaR: ID | K | N | Joint |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| QMIX | 82.1±19.1 | 92.9±8.3 | 92.0±11.4 | 94.4±8.0 | 5.18±2.10 | 10.94±4.14 | 11.57±5.38 | 15.98±6.20 |
| REFIL | 44.4±2.7 | 60.4±6.9 | 67.0±2.6 | 73.2±4.4 | 2.79±0.11 | 6.18±0.55 | 7.81±0.72 | 10.66±0.58 |
| DCG | 85.3±5.3 | 96.4±2.3 | 94.8±3.6 | 97.9±1.7 | 4.99±0.99 | 12.07±3.21 | 11.77±4.50 | 17.32±7.71 |
| ALMA | 67.8±7.1 | 83.0±3.6 | 90.3±5.8 | 92.6±3.6 | 3.86±0.34 | 7.78±0.50 | 9.10±1.21 | 12.97±1.68 |
| TransfQMix | 68.5±8.1 | 88.1±2.6 | 89.5±5.3 | 93.9±3.2 | 4.39±0.42 | 9.63±1.02 | 10.84±1.68 | 15.87±1.87 |
| SPECTra | 90.9±2.6 | 98.9±0.9 | 95.6±3.5 | 99.3±0.5 | 5.71±0.25 | 13.30±0.51 | 12.09±0.56 | 19.49±1.01 |
| Rule † | 80.3 | 92.6 | 75.1 | 88.9 | 3.77 | 7.15 | 4.67 | 7.70 |
| **LEAF** | **27.6±7.0** | **41.4±9.2** | **31.4±10.8** | **39.1±1.6** | **2.10±0.15** | **4.14±0.34** | **3.10±0.65** | **5.41±0.43** |
| 相对最强对手 | −38% | −31% | −53% | −47% | −25% | −33% | −34% | −30% |

表注：† Rule 无训练种子，D 的 SD 为场景 SD；QMIX 指实体注意力适配加 FlexQMIX 混合器；SPECTra 与 TransfQMix 使用 19 维输入重训；P 与 CVaR 由正式逐局记录计算。紧凑备选：单栏表只列 D，P 与 CVaR 写进正文并放附录。

### 5.3 段落安排与建议主句

每段遵循"现象、原因、结论"。以下英文主句的数值都已复核，可直接使用或改写。

**第 1 段：整体优势。**
- LEAF has the lowest mean target damage in all four groups and in each of the 24 evaluation configurations.
- Relative to the strongest competitor in each group, LEAF reduces damage by 43% in ID, 45% under target-count extrapolation, 63% under team-size extrapolation, and 64% under joint extrapolation.
- The advantage holds across training runs. The worst of the three LEAF models has lower damage than the best model of every baseline in all four groups.

**第 2 段：随规模变化的趋势（对应图 E1 (b)–(d)）。**
- As teams grow from 15 to 50 vehicles per side, LEAF's damage stays between 0.56 and 0.61, whereas every learned baseline degrades steadily. REFIL, for example, rises from 1.44 to 3.08.
- Rule is the only other method that does not degrade. Its damage falls from 1.76 at 20 vehicles per side to 1.17 at 50 because it re-solves the assignment at every scale. Under team-size and joint extrapolation it outperforms every learned baseline, and LEAF stays below it at every team size.（已核对：六个学习基线从 N = 15 到 50 都单调上升；引言用到"大规模下所有学习基线都不如 Rule"，这里是它的落点。）
- 解释：攻击者数量增加时，需要区分的攻击者–目标–队友组合迅速增多。只做一次自身查询读取的方法在读取前没有让实体彼此交互，LEAF 在读取前先做多轮实体交互。只写结构差异与行为结果，不写 LEAF 的表示编码了覆盖关系。
- 目标数增加时所有方法的损伤都会上升，因为攻击者可选的目标更多。这里只写 LEAF 在每个 K 下最低，不写增长更慢（K = 12 与 K = 4 之比，LEAF 并不比 REFIL 小）。

**第 3 段：保护质量（表 E3 的 P 与 CVaR）。**
- LEAF keeps every target undamaged in 59% to 72% of episodes. No other method exceeds 56% in any group, and Rule stays below 25%.
- The mean damage of LEAF's worst 10% of episodes is 25% to 34% lower than that of the strongest competitor in each group.
- Rule 的洞察：Rule 是大规模下平均损伤最低的对手，因为整数指派对每个规模都重新求解。但它在 75%–93% 的回合里让至少一个目标受损，说明按距离分配责任无法区分哪个攻击者真正威胁未覆盖的目标。

**第 4 段：学习效率与可复现性。**
- LEAF reaches a validation damage of 1.0 after about 0.56M steps. REFIL needs about 0.86M steps, and the other learned baselines do not reach this level within the 1M-step budget.（基于未平滑均值曲线首次越过 1.0 的验证点。）
- 独立场景池一句话：On an independent pool of 300 scenes per configuration, the All24 damage is 0.790 for LEAF, 2.086 for REFIL, 4.255 for TransfQMix, and 6.296 for SPECTra, reproducing the ordering.

### 5.4 本节不要写的内容

- 不写 LEAF 的友军碰撞低于所有学习基线（两个 19 维基线没有碰撞记录，Rule 在 N/Joint 的碰撞更低）。
- 不写 LEAF 随 K 的退化更慢，不写 LEAF 在所有指标所有设置下都最好。
- 不把验证曲线末点与正式终评混用；验证池与正式池不同。
- 不混入归档的 10 维 SPECTra、TransfQMix 结果。

---

## 6. V-C 消融：哪些设计带来规模泛化

开头一句：本节从头训练 LEAF 的变体，逐项检验第四章的设计如何影响规模泛化。所有变体与 LEAF 使用相同的训练预算、种子和正式评估。

### 6.1 组织原则：单因素消融链

逐键比对保存配置（`models/seed_0/config.json`）后，相邻模型只差一个开关：

| 对照 | 唯一不同的配置 | 检验的设计 |
|---|---|---|
| Direct read → One round | `global_read_h0`：读原始 H⁰ → 先做一轮情境更新 | 读取前是否构建威胁情境 |
| One round → Fixed depth | `global_depths`：[1] → [4]，执行深度 1 → 4 | 一轮与多轮精炼 |
| Fixed depth → Memory query | `global_depths`：[4] → [1,2,3,4] | 固定深度与随机深度训练 |
| Memory query → LEAF | query 中的历史 hidden：保留（截断梯度）→ 置零 | 记忆进入 query 与记忆后置 |
| Fixed depth → Unshared rounds | `rer_update`：tied → untied4 | 轮间共享参数 |
| Memory query → Last read | `read_last_round`：融合 → 末轮 | 中间读出融合 |
| Memory query → No host | 去掉 REFIL 注意力、imagined 分组与辅助损失 | REFIL 宿主 |

主链 Direct read → One round → Fixed depth → Memory query → LEAF 依次加入 story 的每一项设计：先构建情境，再多轮精炼，再随机深度，最后把记忆移出 query。截断梯度只阻断历史 hidden 到 query 的梯度，前向仍读取它，所以 Memory query 族与 LEAF 的 query 条件确实不同。

### 6.2 表 E4：消融链与旁支（双栏）

推荐把"相对上一行改变了什么"直接写进表中，主链在上、旁支在下。D 为均值 ± 三训练种子 SD（来源：`design_comparison.group_summary`），后四列可选给出相对父模型的变化。

| 步骤 | 模型 | 相对父模型的改变 | 执行轮数 | Query | ID | K-OOD | N-OOD | Joint-OOD |
|---|---|---|---:|---|---:|---:|---:|---:|
| 0 | Direct read | 读取原始实体 H⁰ | 0 | 含记忆 | 0.928±0.210 | 1.958±0.457 | 2.020±0.446 | 3.196±0.687 |
| 1 | One round | + 一轮情境构建 | 1 | 含记忆 | 0.609±0.272 | 1.203±0.402 | 1.355±0.884 | 2.157±1.006 |
| 2 | Fixed depth | + 四轮精炼（固定深度训练） | 4 | 含记忆 | 0.407±0.204 | 1.004±0.665 | 0.659±0.184 | 1.265±0.230 |
| 3 | Memory query | + 随机深度训练 | 4 | 含记忆 | 0.527±0.227 | 1.529±0.734 | 0.998±0.621 | 1.946±1.229 |
| 4 | **LEAF** | + 当前状态 query（记忆后置） | 4 | 当前状态 | **0.405±0.097** | **0.996±0.233** | 0.575±0.235 | **1.097±0.105** |
| 旁支 | Unshared rounds | 父：Fixed depth；轮间参数独立（+60% 参数） | 4 | 含记忆 | 0.476±0.183 | 1.127±0.157 | **0.543±0.156** | 1.232±0.253 |
| 旁支 | Last read | 父：Memory query；只用末轮读出 | 4 | 含记忆 | 0.434±0.083 | 1.020±0.100 | 0.652±0.160 | 1.269±0.202 |
| 旁支 | No host | 父：Memory query；去掉 REFIL 宿主 | 4 | 含记忆 | 0.510±0.138 | 1.768±0.621 | 1.263±0.653 | 2.319±0.889 |

相邻步骤的相对变化（负值表示损伤降低）：

| 步骤 | ID | K-OOD | N-OOD | Joint-OOD | 种子层面 |
|---|---:|---:|---:|---:|---|
| 0 → 1 情境构建 | −34% | −39% | −33% | −33% | 各组 2/3 种子改善；LEAF 每个种子都优于 Direct read 每个种子 |
| 1 → 2 多轮精炼 | −33% | −17% | −51% | −41% | ID、K 2/3；N、Joint 3/3 |
| 2 → 3 随机深度 | +29% | +52% | +51% | +54% | Memory query 在 10/12 个种子–组别对中更差（ID、K 3/3，N、Joint 2/3）；seed1 放大了均值差距，但方向基本一致 |
| 3 → 4 记忆后置 | −23% | −35% | −42% | −44% | ID 3/3，OOD 各 2/3 |

### 6.3 段落安排与建议主句

**第 1 段：情境构建是基础。**
- Reading raw entities raises damage at every scale. A single round of context construction removes 33% to 39% of the damage in each group, with the query and training otherwise unchanged.
- 原因：一轮实体间注意力让攻击者的表示可以吸收它所指向的目标及附近队友的信息，自身查询读到的是彼此交互过的实体。只写这一结构作用，不写表示已经编码了覆盖关系（受控探针没有给出这一证据）。

**第 2 段：多轮精炼在外推中收益更大。**
- Four refinement rounds lower damage relative to one round by 33% in ID and by 51% and 41% under team-size and joint extrapolation. All three seeds improve in both extrapolation groups.
- 原因：多轮精炼在训练范围内已有收益，在人数外推和联合外推中收益更大，实体越多越需要在已交互的表示上继续加工。不要写"单轮在小规模已经足够"，ID 的 −33% 不支持这一说法。

**第 2.5 段：随机深度不提升四轮性能（必须如实写出）。**
- Sampled-depth training does not lower the damage at four rounds. Within the history-query family it raises the mean damage by 29% to 54%, and Memory query is worse than Fixed depth in 10 of 12 seed-group comparisons. Its benefit appears when the execution depth changes, which Section V-D examines.
- 写法：紧接着指向 V-D.3 的出口稳健性（固定深度执行一轮时损伤约为四轮的 15 倍）。LEAF 把随机深度与当前状态 query 结合后四组都不差于 Fixed depth，可作一句补充，但这组对照同时差 query，不能当作随机深度没有代价的证据。

**第 3 段：记忆后置在外推中更重要。**
- Feeding the recurrent state into the query increases damage by 30% in ID and by 54% to 77% in the three extrapolation groups.
- 种子离散同步扩大：Joint-OOD 的 SD 从 LEAF 的 0.105 增至 Memory query 的 1.229。可写 Its seed-to-seed spread grows as well，与 V-D.2 衔接。

**第 4 段：融合与共享在正式深度下没有可见收益。**
- Reading only the last round or untying the rounds does not raise the four-round damage relative to the parent models. These choices matter when the execution depth changes, which Section V-D examines.
- 数据（与父模型比，单因素）：Last read 相对 Memory query 四组均值低 18%–35%（12 个种子–组别对中 8 个更好）；Unshared rounds 相对 Fixed depth 互有高低（ID +17%、K +12%、N −18%、Joint −3%）。所以正文不写"融合或共享提升性能"。
- No host 一句：Removing the REFIL host raises damage in all three extrapolation groups relative to Memory query, so LEAF benefits from the entity grouping of its host.

### 6.4 风险与写法边界

- **随机深度在 R4 下的代价。** 在含记忆 query 的变体族中，Memory query（随机深度）在 R4 的均值比 Fixed depth 高 29%–54%，12 个种子–组别对中 10 个更差。seed1 放大了均值差距，但只去掉 Memory query 一方的 seed1 来宣称"两者接近"属于挑选数据；两方同时去掉 seed1 后，ID 与 K 仍差 34% 与 74%。LEAF（随机深度、当前 query）在四组都不差于 Fixed depth，但同时差 query。论文中随机深度的价值以 V-D.3 的出口稳健性为证据，不写随机深度提升或不影响正式深度性能。
- **融合与共享没有 R4 独立收益。** 旁支要与父模型比：Last read 相对 Memory query 反而更低，Unshared rounds 相对 Fixed depth 互有高低。正文不写"融合/共享带来提升"。融合的作用目前只有冻结干预证据（强制末轮读出使已训练 LEAF 大幅变差），Last read 没有出口深度评估，因此也不写"融合让浅出口可用"。
- **One round 的 seed2 偏差较大。** 0 → 1 的种子层面为 2/3。均值四组一致，正文写均值即可。
- 如审稿人要求更完整的单因素对照，可补训 Fixed depth + 当前 query、Last read + 当前 query 各三种子。这超出现有数据范围，只作为备选，见第 12 节。

### 6.5 可选图：消融路线图（单栏）

横轴为步骤 0–4，纵轴为 D（对数），四条线对应四个组别，旁支用空心点挂在父步骤附近。它直观展示每一步的方向，但步骤 3 的上升会很醒目，需要在图注指向图 E4。默认只用表 E4，此图作为备选。

---

## 7. V-D 机制分析：三个核心特点

开头一句：本节解释 LEAF 为什么有效，依次检验三个性质：收益来自情境构建的计算组织而非参数量；记忆后置让情境解读在规模变化时更稳定；随机深度训练让每个执行深度都是可用策略。

### 7.1 特点一：结构而非容量（图 E2，单栏两子图）

**要回答的问题。** LEAF 的优势是不是因为参数更多。

**证据。**

| 证据 | 数据 |
|---|---|
| 参数匹配 | REFIL-matched 部署参数 176,393，LEAF 175,754（差 0.36%）。REFIL-matched 的 D 为 0.914 / 2.493 / 2.269 / 4.170，是 LEAF 的 2.3–3.9 倍，且四组都比 REFIL 本身更差 |
| 参数更多但无一致收益 | Unshared rounds 281,738（+60%），R4 与 LEAF 相当；ALMA 部署参数 282,827，Joint D 5.375 |
| 同参数不同组织 | Direct read、One round、Fixed depth 与 LEAF 保存的网络参数相同（Direct read 不使用其中的情境更新模块），Joint D 从 3.196 到 1.097 |
| 学习速度 | 50 个验证点的平均 D：LEAF 1.76±0.13，One round 1.73±0.07，Fixed depth 1.69±0.20；Direct read 2.52±0.19，REFIL 2.46±0.07，REFIL-matched 2.51±0.16 |
| 冻结旁路 | 已训练 LEAF 改为读原始 H⁰：每配置 10 场、三个模型，ΔD 为 +7.56（10v10 K2）、+18.96（50v50 K2）、+32.97（30v30 K12），9/9 个模型–配置对都变差 |

**图 E2 规格。**
- (a) 横轴为部署参数量（对数），纵轴为 Joint-OOD D（对数）。点：8 个对比方法、REFIL-matched、Unshared rounds、One round、Fixed depth。用竖向虚线连接 x ≈ 176k 处的 LEAF、REFIL-matched、One round、Fixed depth，突出同参数下损伤跨度。不画 Direct read 与 No host（它们保存了未使用的参数，横坐标会误导）。
- (b) 验证学习曲线：LEAF、One round、Direct read、REFIL、REFIL-matched 五条线。信息：凡是有至少一轮情境构建的模型都以 LEAF 的速度学习，Direct read 与 REFIL-matched 跟随 REFIL。
- 备选 (b')：冻结干预柱状图（读 H⁰、关系残差置零两项，三配置）。

**建议主句。**
- REFIL-matched has an actor of the same size as LEAF within 0.4%, yet its damage is 2.3 to 3.9 times higher. Adding these parameters to REFIL raises its damage in all four groups.
- Context construction also speeds up learning. Every variant with at least one context round learns at the pace of LEAF, whereas Direct read and REFIL-matched follow REFIL.

### 7.2 特点二：记忆放在读取之后（图 E3，单栏两子图）

**要回答的问题。** 为什么情境读取只用当前状态，而把记忆放到下游。论证顺序为：原因（记忆随规模漂移，当前状态不漂移）→ 后果（记忆进入 query 后外推更差）→ 行为（保护指标）→ 记忆仍有用（下游）。

**原因：记忆漂移的只读分析。** 数据为 LEAF 三个模型在 10v10 K2（训练范围内，作参照）、50v50 K2（人数外推）、30v30 K12（联合外推）上的标准 R4 回合，每配置每模型 10 个共同场景。对每个训练种子，用分类器区分参照配置与外推配置下的决策样本，比较两类特征：GRU 记忆 h_{t−1}（64 维），以及 query 唯一依赖的当前自身输入（自身位置、速度、存活、生命值与上一动作）。AUC 为 0.5 表示该特征不携带规模信息。

| 外推配置 | 记忆：线性 AUC | 记忆：非线性 AUC | 自身输入：线性 AUC | 自身输入：非线性 AUC |
|---|---:|---:|---:|---:|
| 50v50 K2 | 0.887±0.023 | 0.919±0.012 | 0.620±0.040 | 0.658±0.048 |
| 30v30 K12 | 0.811±0.039 | 0.836±0.042 | 0.536±0.008 | 0.537±0.038 |

（均值 ± 三训练种子 SD。逐种子线性 AUC：记忆 0.888 / 0.864 / 0.910 与 0.800 / 0.854 / 0.778；自身输入 0.593 / 0.602 / 0.666 与 0.531 / 0.545 / 0.533。）

- 三个种子、两个配置、两种分类器下，记忆的 AUC 都明显高于自身输入。非线性分类器近似给出该特征可携带规模信息的上限。query 是自身输入的确定性函数，所以它能带入的规模信息不会超过自身输入这一列。
- 三个配置的平均 ‖h‖ 都在 6.15–6.36，漂移体现为表示方向的变化，与数值放大无关。
- 维度检查：把记忆压到前 8 个主成分后，50v50 K2 上记忆仍高于自身输入（0.715±0.024 vs 0.585±0.039，三种子一致）；30v30 K12 上降维后不再一致（0.588±0.107，seed2 为 0.465），说明联合外推下的漂移主要落在方差较小的方向上。正文报告完整特征的结果即可，此项放附录。
- 分析对象是 LEAF 自己的记忆。Memory query 的记忆没有保存，所以这里证明的是"这类记忆会随规模漂移"，不是直接测量 Memory query 的 query 漂移。

**后果、行为与下游作用。**

| 证据 | 数据 |
|---|---|
| 差距随外推扩大 | Memory query 相对 LEAF：ID +30%，K +54%，N +74%，Joint +77%（表 E4） |
| 种子可靠性 | Joint-OOD 种子值：LEAF 1.134 / 0.979 / 1.178；Memory query 1.348 / 3.360 / 1.132 |
| 保护行为（正式 300 场，`mechanism.formal`） | 四个配置 × 四项指标共 16 项比较中，Memory query 全部介于 LEAF 与 REFIL 之间，见下表 |
| 记忆仍在下游起作用 | 冻结 LEAF 每步重置 GRU：8/9 个模型–配置对变差（ΔD 在 10v10 K2、50v50 K2、30v30 K12 上分别为 +1.42、+1.29、+1.66，均为三模型均值，每配置 10 场） |

| 配置 | 方法 | I ↑ (%) | P ↓ (%) | CVaR ↓ | C ↓ (%) |
|---|---|---:|---:|---:|---:|
| 10v10 K2 | LEAF / MQ / REFIL | 86.4 / 81.0 / 77.0 | 30.4 / 42.4 / 55.2 | 2.43 / 2.71 / 3.54 | 1.8 / 2.5 / 4.7 |
| 30v30 K2 | LEAF / MQ / REFIL | 83.0 / 78.8 / 75.1 | 29.6 / 47.8 / 70.9 | 3.18 / 4.35 / 7.80 | 6.3 / 9.0 / 13.4 |
| 50v50 K2 | LEAF / MQ / REFIL | 78.7 / 75.7 / 72.3 | 28.1 / 45.2 / 70.6 | 3.13 / 5.11 / 10.85 | 11.1 / 14.7 / 19.6 |
| 30v30 K12 | LEAF / MQ / REFIL | 81.5 / 77.5 / 74.5 | 48.3 / 63.8 / 78.2 | 9.42 / 11.72 / 17.37 | 7.6 / 9.4 / 12.7 |

（MQ 即 Memory query；表中 CVaR 来自 `mechanism.formal` 的 `worst10pct_D`，与表 E3 的派生 CVaR 定义一致。）

**图 E3 规格（单栏两子图，按"原因 → 后果"排列）。**
- (a) Memory drifts, the current state does not：横轴为两个外推配置（50v50 K2、30v30 K12），每个配置两组柱或点，分别是记忆与当前自身输入的 AUC，叠加三个种子的散点，0.5 处画机会水平虚线。主值用线性分类器，非线性结果可用空心标记叠加。信息：记忆能识别部署规模，query 的输入几乎不能。
- (b) The cost of memory in the query：四个组别上 LEAF 与 Memory query 的 D，柱或点加三个种子的散点，标注相对差距 +30%、+54%、+74%、+77%。信息：差距随分布偏移扩大，Memory query 的种子离散也扩大。
- 保护指标（上表 16/16 有序）放正文一句话和附录表。备选：把 (b) 换成四个大规模配置上 LEAF、Memory query、REFIL 的 I、P、CVaR、C 小图，或 50v50 K2 与 30v30 K12 的首次受损累计发生曲线（`mechanism.formal.first_damage_cdf`；首次受损的受限平均时间 LEAF 约 71–84 步、REFIL 约 50–67 步，未受损回合按 100 步计，只用于这一均值）。

**建议主句。**
- A linear classifier identifies the deployment scale from LEAF's recurrent state with an AUC of 0.89 for 50 vehicles per side and 0.81 for 12 targets. From the current self features that form the query, it reaches only 0.62 and 0.54.
- Memory learned in small engagements thus moves to new regions as the scale grows, while the current self state stays close to the training distribution.
- Feeding memory into the query carries this drift into context reading. The damage gap to LEAF grows from 30% in ID to 77% under joint extrapolation.
- Memory query lies between REFIL and LEAF on interception, damage incidence, worst-case damage, and collisions in every large configuration.
- Resetting the recurrent state at every step worsens damage in eight of nine model–configuration pairs. LEAF still relies on memory for action selection. It only keeps memory out of the step that reads the threat context.

**边界。** 漂移分析每配置只有 10 个场景、三个模型，图注写明。分析对象是 LEAF 的记忆，正文写成"the recurrent state"即可，不写成 Memory query 的 query 发生了漂移。

### 7.3 特点三：随机深度让每个执行深度都可用（图 E4，双栏 1×3）

**要回答的问题。** 随机深度训练带来了什么，部署时能否用更少轮数换取更低时延。

**证据。**

| 证据 | 数据 |
|---|---|
| 读取前置 | 正常 R4 执行中前两轮融合权重合计 98.91%、98.76%、98.84%（10v10 K2、30v30 K2、30v30 K12）；α₁ 约 74%–76%，α₂ 约 23%–25%，α₄ 约 0.03% |
| 读取因接收者和时刻而异 | 同一状态下不同防御者的 α₁ 标准差 11.4–13.9 个百分点；同一防御者相邻步变化 5.3–5.5 个百分点 |
| 全配置深度曲线 | LEAF All24：R1 0.802、R2 0.793、R3 0.789、R4 0.785、R5 0.785、R6 0.785。R2 比 R4 高 1.1%，R1 高 2.3%，R5/R6 变化 <0.1% |
| 随机深度 vs 固定深度（单因素） | 见下表：随机深度训练的模型各出口平坦，固定深度训练的模型浅出口崩溃 |
| 强制末轮读出 | 冻结 LEAF 改用第 4 轮读出：ΔD +6.73、+19.94、+25.56；均匀融合 +3.61、+8.84、+17.66。Memory query 在 30v30 K12 上强制 read4 时 D 从 3.37 升至 27.99，强制 read1、read2 只略高于学习融合（4.16、4.04） |
| 时延 | 本机 50v50 闭环每次全队决策：R1 7.47 ms、R2 8.67、R3 9.55、R4 10.59；One round 7.51；Fixed depth R4 10.52；Unshared R4 10.48。R1 比 R4 少 29%，R2 少 18% |
| 一轮出口 | LEAF 执行一轮的 Joint-OOD D 为 1.097，专门训练的 One round 为 2.157 |

**执行深度对照（3 个 K2 配置 10v10、30v30、50v50 等权，正式池 300 场 × 3 种子）：**

| R | LEAF（随机深度） | Memory query（随机深度） | Fixed depth（固定 4） | Unshared rounds（固定 4，不共享） |
|---:|---:|---:|---:|---:|
| 1 | 0.568 | 1.189 | 9.573 | 11.602 |
| 2 | 0.551 | 0.963 | 7.119 | 6.126 |
| 3 | 0.536 | 0.958 | 2.770 | 1.902 |
| 4 | 0.532 | 0.969 | 0.642 | 0.544 |
| 5 | 0.538 | 0.971 | 0.602 | 无第 5 轮参数 |
| 6 | 0.538 | 0.971 | 0.666 | 无第 6 轮参数 |

逐种子 R1/R4 比值：随机深度的 6 个模型在 0.84–1.49 之间；固定深度的 6 个模型中 5 个在 12.7–29.0 之间，Fixed depth seed1 为 1.16。Memory query 与 Fixed depth 只差训练深度，Fixed depth 与 Unshared rounds 只差参数共享。LEAF 一列来自 `loop_summary.csv`（深度评估池），其余三列来自 `history_interventions.depth`，R4 复用正式 final 评估。

**图 E4 规格（双栏 1×3）。**
- (a) Exit robustness：横轴执行深度 R = 1–6，纵轴 D（对数）。四条均值线加各种子细线，R > 4 加"超出训练深度"底纹。信息：随机深度训练的两条线平坦，固定深度训练的两条线在浅出口高一个数量级；共享参数的 Fixed depth 还能执行 5、6 轮。备选：拆成三个配置各一个小图。
- (b) Front-loaded reading：三个配置的 α₁–α₄ 堆叠条，旁边用点和误差线标出接收者间 SD 与相邻步变化。信息：每个防御者主要依赖前两轮，但依赖程度因防御者和时刻而异。
- (c) Compute–damage trade-off：横轴为 50v50 K2 闭环全队决策时延，纵轴为 50v50 K2 的 D。LEAF R1–R4 连成折线，One round、Fixed depth R4、Unshared R4 为独立标记，在图边缘用箭头标出 Fixed depth R1 的 D = 14.2（超出坐标范围，时延未单独测量）。信息：LEAF 一轮的时延与 One round 相同，损伤与 LEAF 四轮相当，约为 One round 的三分之一。备选：纵轴改为 All24 D（需从 `design_comparison.config_summary` 计算 One round、Fixed depth、Unshared 的 All24）。

**建议主句。**
- About 99% of the fusion weight falls on the first two rounds in every configuration we examined, so the policy reads its context almost entirely from the early rounds.（不写"算子在前两轮建好了核心情境"，融合权重不等于因果贡献。）
- Executing two rounds changes the damage over all 24 configurations by 1.1% relative to four rounds, and executing five or six rounds, beyond the training range, changes it by less than 0.1%.
- Random-depth training is what keeps the shallow exits usable. With the same architecture and query, a model trained only at four rounds incurs 15 times its own four-round damage when executed for one round. Five of the six fixed-depth models collapse in this way, whereas all six random-depth models stay within a factor of 1.5.
- A full-team decision for 50 defenders takes 7.5 ms with one round and 10.6 ms with four rounds on an RTX 5070 Ti, two orders of magnitude below the 1 s decision interval. At this scale, one round raises damage by 0.6% relative to four rounds, so an operator can cut latency by 29% without retraining.（0.6% 为 50v50 K2 的 R1 0.564 对 R4 0.561，与时延同一配置；全部 24 配置平均为 2.3%，引言用的是后者。）
- Even with a single round, LEAF's joint-extrapolation damage is about half that of a model trained for one round only.

**边界。**
- 最后一条主句跨评估池且混入 query 差异，只作为观察，不写成"因为随机深度"。随机深度的因果证据是 Memory query 与 Fixed depth 的单因素对照。
- 融合权重描述读取分配，不等于奖励贡献。执行时 R 由部署者设定，融合权重本身不会提前停止计算。
- 深度排序因配置而异（例如 30v30 K12 的 R1 均值低于 R4），只写聚合层面的平坦，不写越深越好或单调。
- 如果第四章保留"前缀融合稳定性"命题，可用同状态诊断验证：执行两轮替代四轮时，48,705 个决策中贪心动作改变 1.1%–1.7%；改变的决策有 78%–81% 落在 Q-margin 最低的 10%；满足命题充分条件的 38,620 个决策无一改变动作。数据在 `mechanism.frozen.margin_analysis`。

---

## 8. 附录与补充材料候选

| 内容 | 用途 | 来源 |
|---|---|---|
| 24 配置逐项 D 表（8 方法 + Rule） | 支撑"每个配置都最低" | `comparison.config_summary`、`rule_config` |
| 逐种子分组 D（8 方法 + 7 变体） | 支撑种子层面陈述 | `comparison.per_seed_groups` |
| P 与 CVaR 的 SD 及逐配置值 | 表 E3 的完整版本 | 派生统计，见第 10 节 |
| 冻结干预全表（读 H⁰、残差置零、read4、均匀融合、GRU 重置、R1、R2、固定融合） | 机制节的完整数据 | `mechanism.frozen.combined_interventions` |
| LEAF R1–R6 逐配置表（K/N/Joint） | 深度排序随配置变化的完整呈现 | `loop_summary.csv`，主报告 §4.4–4.6 |
| 典型轨迹（LEAF vs REFIL，15v15） | 定性展示：LEAF 拦截全部 15 个攻击者、目标无损；REFIL 第 25 步首次受损、第 27 步失去最后一名防御者 | `trajectories.json.gz` |
| 固定人数 100 次决策时延 | 时延的纯计算口径：50v50 R1 5.85 ms → R4 8.88 ms | `decision_timing.json` 的 `decision_timing_fixed100` |
| 独立确认池分组结果 | 复现性 | `comparison.confirmation` |
| LEAF、Memory query、REFIL 在四个大规模配置上的 I、P、CVaR、C | 支撑"Memory query 介于二者之间（16/16）" | `agg.mechanism.formal.summary`，见 7.2 节表 |
| 记忆漂移的逐种子 AUC、非线性与 PCA-8 维度检查 | 图 E3 (a) 的完整版本 | 7.2 节，口径见第 10 节 |

轨迹图如果使用，只画 LEAF 与 REFIL（或再加 Rule）。现有轨迹中的 SPECTra 是归档 10 维模型，与主表的 19 维 SPECTra 不一致，不能进论文。

---

## 9. 不进入正文的内容

| 内容 | 原因 |
|---|---|
| 受控覆盖关系线性探针 | 准确率接近 50% 机会水平，没有形成证据；写进去会削弱动机 |
| Fixed depth 100 场浅层对照（主报告 §4.9） | 已被 300 场完整数据与 Memory query 单因素对照取代 |
| 归档的 10 维 SPECTra、TransfQMix 结果 | 与当前 19 维主比较不是同一模型 |
| 重复追击比例 | LEAF 在 50v50 K2 高于 REFIL（43.9% vs 35.5%），不支持"减少重复追击"，也不是本文主张 |
| 压力与 α₁ 的相关 | 三个配置方向不一致，不支持统一规律 |
| 损伤集中度 HHI | 只在 D > 0 回合定义，解释成本高，与主线无关 |
| 1014 版第 (e) 子图的混合池比较 | LEAF 参照来自深度/确认池，与正式池并列容易误读 |

---

## 10. 数据来源与派生统计口径

所有路径相对 `Open-SCORE/outputs/main/`。`aggregates.json` 简写为 `agg`。

| 图表 | 数据 | 字段与筛选 |
|---|---|---|
| 图 E1 (a) | `had/aggregates.json` | `agg.comparison.validation_curves`：每方法 50 点，x 用 `t_env_mean`，带 `reward.per_seed`（D = −reward） |
| 图 E1 (b)–(d) | 同上 | `agg.comparison.config_summary`（`D_per_seed`）；Rule 用 `agg.comparison.rule_config` |
| 表 E3 的 D | 同上 | `agg.comparison.group_summary`、`per_seed_groups` |
| 表 E3 的 P、CVaR | `had/episodes.csv` | 学习方法：`phase=final_eval` 且 `arm ∈ {final, standard}`（SPECTra、TransfQMix 取 `standard`，`source_version=main1013_base`）；Rule：`phase=anchor`、`method=rule_nv1`。每个方法、种子、配置正好 300 局 |
| 表 E4 | `had/aggregates.json` | `agg.design_comparison.group_summary`、`per_seed_groups` |
| 部署参数量 | `had/<方法>/models/seed_0/final.pt` | `networks.agent` 与 `networks.mac` 中的部署网络（DCG：agent + utilities + payoffs；ALMA：agent + alloc_policy + alloc_critic），排除 `attn.scale_factor` 缓冲 |
| 图 E2 (b) | `had/aggregates.json` | `agg.comparison.validation_curves`（15 个方法都在） |
| 图 E3 (a) | `had/regir_nomem/mechanisms/results.json.gz` | 记忆漂移分析，口径见下方"记忆漂移分析定义" |
| 图 E3 (b) | `had/aggregates.json` | `agg.design_comparison.per_seed_groups`（`regir_nomem`、`regir_sg`） |
| 保护指标（正文与附录） | 同上 | `agg.mechanism.formal.summary[].metrics`：`blue_intercepted_fraction`、`any_damage`、`worst10pct_D`、`collision_fraction`；首次受损曲线 `first_damage_cdf` |
| 图 E4 (a) | `had/loop_summary.csv`；`had/aggregates.json` | LEAF：`split=nomem_depth_eval`、`arm=fixed1…fixed6`；其余：`agg.comparison.history_interventions.depth`（`regir_sg`、`regir_fixed4_sg`、`regir_untied4_sg`） |
| 图 E4 (b) | `had/aggregates.json` | `agg.mechanism.normal.feature_analysis.across_seeds[].mean_alpha`；接收者 SD 与相邻步变化见主报告表 5 及 `mechanism.normal` |
| 图 E4 (c) | `had/decision_timing.json`；`had/loop_summary.csv`；`agg.design_comparison.config_summary` | 闭环 20 回合有效步加权均值；D 取 50v50 K2 |
| 机制文字 | `had/aggregates.json`；`summary.json` | 冻结干预 `agg.mechanism.frozen`；All24 R1–R6 `summary.json` 的 `nomem_depth_analysis`；Memory query 读出干预 `agg.comparison.history_interventions.readout` |

**派生统计定义。**
- P：每个模型、配置内 `D > 1e−9` 的回合比例，再按 4.5 节方式汇总。
- CVaR：每个模型、配置内按 D 从高到低取前 30 局（300 局的 10%）求平均，再按同样方式汇总。
- 两者只读正式逐局记录，不改变正式 D 统计。

**记忆漂移分析定义**（只读，没有新增对局，结果只记录在本文）。
- 样本：`episodes` 中 `method=regir_nomem`、`arm=standard` 的回合，配置 10v10 K2、50v50 K2、30v30 K12，训练种子 0–2，场景 9000–9009。只有这三个配置的 LEAF 标准回合保存了 `h_prev`，REFIL 与其他臂没有。
- 每个决策步 t ≥ 1、每个存活防御者取一个样本。记忆为 `decisions[t].h_prev`（base64 float32，行与 `active_defender_ids` 对应）。自身输入取 `trajectory[t−1]`（即 t 时刻状态与 t−1 时刻动作）：位置 / 2500、速度 / 120、存活、生命值，以及 `actions[id]` 的 27 类独热编码。
- 每个种子分别比较参照配置与外推配置。在两域共有的每个决策步上取相同样本数（随机种子 20261015），按场景种子做 5 折 GroupKFold，报告各折 AUC 的均值，再给出三个训练种子的均值 ± SD。
- 分类器：StandardScaler + LogisticRegression(C = 1, max_iter = 3000)；非线性版本为 HistGradientBoostingClassifier(max_iter = 300, learning_rate = 0.05, random_state = 0)；维度检查为 StandardScaler + PCA(8) + LogisticRegression。
- 每种子每配置的平衡样本约 1,750–1,835 对。

**口径提醒。**
- 深度池的 LEAF R4（All24 0.7845）与正式比较（All24 0.7854）来自不同评估池，不在同一表中混用。图 E4 (a) 中 LEAF R4 = 0.532 与正式池同三配置均值 0.525 的细微差异也来自这一点。
- Memory query、Fixed depth、Unshared rounds 的冻结深度记录来自 main0928 原始登记，带 `legacy_had_protocol_unspecified` 标识，与表 E4 的正式数据同源。
- 冻结干预每配置只有 10 个场景，图注与正文需要写明。
- 闭环时延测量使用独立的 SkyHAD 3.0.0 运行时，只用于时延，不与正式损伤混用。

---

## 11. 写作约定

**名称。** 方法：LEAF、QMIX、REFIL、DCG、ALMA、TransfQMix、SPECTra、Rule。变体：Direct read、One round、Fixed depth、Memory query、Unshared rounds、Last read、No host、REFIL-matched。组别：ID、K-OOD、N-OOD、Joint-OOD；正文分别称 in-distribution、target-count extrapolation、team-size extrapolation、joint extrapolation。

**符号。** 第四章用 L 表示轮数：训练 L ~ U{1,…,4}，部署时的执行深度为 L_exec。实验章节统一用 L，执行深度写作 L = 1,…,6 或 one to six rounds。本文中的 R1–R6 只是数据文件里的出口记法，即 L = 1–6，论文正文与图中不出现 R1–R6。损伤统一用 D。

**小节开头。** 每个小节第一句说明它回答的问题，例如：
- V-B：We first ask whether a policy trained once on small engagements protects targets better than existing methods when deployed at larger scales.
- V-C：We next retrain variants of LEAF to identify which designs produce this generalization.
- V-D：Finally, we examine why LEAF works through three properties of its computation.

**风格**（与 `Papers/paper_style.md` 一致）。不用分号；正文少用括号；不用引号强调；不写 not A but B；不写 significantly；用主动语态；正文只引用关键数字，完整数值放表中；局限只出现在第六章。

**主张强度。** 能说"每个配置"、"每个种子"的地方就这样说，这是三种子条件下最有力且可核对的表述。均值成立而种子不全一致时，只陈述均值，不写 consistently。

---

## 12. 已确认的决定与备注

1. **硬件。** 正文只在时延测量处说明本机 RTX 5070 Ti 配置（建议句见 4.6 节），训练硬件不单独描述。备查：保存的训练配置显示训练在远程 Linux 服务器完成，所以不要写"所有实验均在该机完成"。
2. **载具模型。** 暂时模糊化，只写策略在 SkyHAD 仿真环境中调用、输出离散加速度指令。备查：正式结果为 `spatial_dim=2` 的平面设定，而 SkyHAD 的 `UAV_fixedwing` 是只支持三维的六自由度刚体模型，第三章定稿时需与此一致。**当前已出现直接冲突，需要用户决定：** 第三章 `problem.tex` 按 SkyHAD 4.2 固定翼建模（27 个加速度意图、速度 18–30 m/s、2×2×1 km 场景），第四章也写 27 个意图；全部正式数据来自平面设定（9 个动作、速度 35–120 m/s、5×5 km、开火 200 m 与伤害 200–400 m）。4.1 节的表若照写，会与第三、四章矛盾。可选方向是把第三章改为与数据一致的设定，或用固定翼重跑实验；在决定之前，V-A 只写协议层面的设置（规模、时长、对手、指标），动力学数值引用第三章。
3. **派生统计 P 与 CVaR。** 进入主表 E3，并替代 story 中无法完整支撑的碰撞主张。它们只读正式逐局记录，覆盖全部 8 个方法。
4. **可选分析 A：记忆漂移。** 已完成，结果与口径见 7.2 节和第 10 节。没有新增对局，也没有保存新文件。
5. **可选实验 B：不补训。** 现有 Fixed depth 与其他历史变体一样使用含记忆的 query，而 LEAF 使用当前状态 query。所以 Fixed depth 与 LEAF 之间同时差训练深度和 query 两项。"随机深度 vs 固定深度"的单因素证据已由 Memory query 对 Fixed depth 提供（两者 query 相同），本章随机深度的论证全部基于这一组。补训"Fixed depth + 当前 query"只用来回答一个审稿人可能追问的问题：在 LEAF 的 query 下，随机深度是否会降低四轮时的性能。这个问题并非多余：含记忆 query 的变体族中，随机深度在 12 个种子–组别对中有 10 个更差。现有数据中 LEAF 在四组都不差于 Fixed depth，可以先这样回应，确有需要再补训。
