# LEAF：可迭代正文初稿大纲

本轮目标：先确定论文的中心故事、段落次序和三张概念图的分工。英文锚句可以直接进入初稿；中文说明用于解释每段的写作意图。实验图保留明确的位置、问题和取数来源，后续围绕需要表达的结论再制作。必要的方法、设置和证据均安排在正文中。

工作标题：**LEAF: Looped Entity Attention Fusion for Scale-Generalizable Multi-Agent Defense**

## 1. 全文只围绕这一条主线

**跨规模目标保护需要同一策略理解变化的实体关系，并为每个防御者提取与自身行动有关的信息。LEAF 通过共享迭代构建实体上下文，再依据防御者自身状态与历史读取、融合中间表征；不同展开深度共同参与训练，使这条决策路径能够在变化的规模下复用。**

贯穿全文的英文中心句：

> LEAF learns shared iterative entity processing and gives each defender direct access to its intermediate states for cooperative protection across changing populations.

读者应形成的理解顺序：

1. **任务：** 数量变化会改变目标覆盖、威胁关系与防御责任。
2. **设计：** 实体需要形成上下文，每个防御者需要从中读出自己的决策信息。
3. **学习：** 同一更新与读取路径在不同展开深度下共同优化。
4. **证据：** 跨规模保护效果、相同单轮执行下的策略质量、当前策略对中间读出和历史的使用，分别支撑上述设计。

### 章节之间的接力

| 章节 | 回答的问题 | 本章结束时读者应得到什么 | 下一章接住什么 |
| --- | --- | --- | --- |
| I. Introduction | 为什么提出这条计算路径？ | 任务需求、方法想法和检验问题 | 想法与既有工作的关系 |
| II. Related Work | 这条路径承接哪些研究？ | 学习基础与组合设计的清楚定位 | 具体任务和接口 |
| III. Problem Formulation and Preliminaries | 策略看到什么、优化什么、怎样迁移？ | 同一冻结策略跨规模评价的定义 | 方法要接入的输入、记忆与价值学习接口 |
| IV. Looped Entity Attention Fusion | 信息怎样形成、读取并进入动作？ | 一次物理决策内的完整信息流 | 各模块对应的可检验问题 |
| V. Experimental Evaluation | 这条设计带来什么效果，策略实际使用了什么？ | 性能、训练与行为使用三个层次的证据 | 方法对自动化控制的启示 |
| VI. Conclusion | 全文最终建立了什么？ | 一条可复述的设计认识及其证据 | 后续工作围绕已建立的认识展开 |

## 2. 摘要与面向应用的开篇

### Abstract：可直接修改的英文初稿

> Cooperative target protection requires a shared policy to coordinate defenders as vehicle populations and protected targets change. We study frozen-policy transfer across these changes against a fixed reactive attacker controller. We present Looped Entity Attention Fusion (LEAF), which repeatedly updates receiver-relative entity representations with a shared attention block. Each defender uses its own state and temporal memory to read and fuse intermediate representations before selecting an action. Sampled-depth training couples short and long computation paths within entity-factorized value learning. In simulation, LEAF improves target protection across population shifts relative to REFIL and yields a more effective one-round policy than independent one-round training. Representation probes reveal accessible geometric threat information, while frozen-policy interventions show that early reads and temporal memory influence decisions. These findings support shared iterative processing with individual access to intermediate information for cooperative control across changing populations.

逐句目的：需求 → 研究设置 → 共享更新 → 个体读取 → 训练设计 → 主要效果 → 行为与表征证据 → 方法意义。摘要中的 probe 句随最终实验重心决定保留篇幅。

### Note to Practitioners：写作位置与内容

安排在摘要之后，先从部署需求讲起，再解释使用接口。初稿按四步展开：

1. 目标保护系统会面对不同车辆数和目标数，统一策略便于跨配置部署。
2. LEAF 接收实体状态，为每个防御者维护时间记忆，并输出离散控制动作。
3. 仿真评价展示固定策略在这些规模变化中的保护效果。
4. 实际系统映射需要确定状态获取、动作接口与计算周期；若准备强调实时部署，在实验中加入实测时间槽。

英文开篇锚句：

> Protection systems often operate with changing numbers of vehicles and assets, creating a practical need for controllers that can be reused across engagement configurations.

## 3. 三张概念图先建立共同理解

沿用原稿的实体符号和颜色：防御者为红色三角形，攻击者为蓝色三角形，目标为金色六边形；橙色表示实体更新，紫色表示读取与融合，青色表示时间决策。当前三张均为生图概念初稿。

| 图 | 只承担的核心信息 | 建议位置 | 与其他图的分工 |
| --- | --- | --- | --- |
| 图1：动机对比 | 先形成实体上下文，再让个体读取各阶段的信息 | Introduction 第2—3段附近 | 建立设计直觉 |
| 图2：防御场景与规模变化 | 同一策略面对车辆数、目标数及二者共同变化 | Problem Formulation 开头 | 定义应用与迁移轴 |
| 图3：LEAF 框架 | 共享更新 → 各轮读取 → 融合 → 一次循环决策 | Method 开头 | 解释方法组织方式 |

### 图1图注初稿

> **Entity processing for individual decisions.** A direct entity read aggregates encoded entity features for action selection. LEAF develops entity context through shared refinement and gives each defender access to intermediate representations through a self- and history-conditioned reader. The illustrated updates occur within one decision step.

图中每组实体条纹只示意表征类型。正文在此解释“上下文”与“个体相关性”，随后进入完整框架。

### 图2图注初稿

> **Cooperative target protection across population changes.** Defenders coordinate their responses to protect targets from incoming attackers. A shared policy is trained in small engagements and evaluated with frozen parameters as vehicle populations, target populations, or both increase. Icon counts illustrate the three types of population change; the evaluation configurations are defined in the experimental protocol.

正文说明公开物理状态和攻击者私有目标分配。场景中的示意排列只帮助识别任务角色。

### 图3图注初稿

> **Conceptual organization of LEAF at one physical decision.** Shared, count-conditioned attention updates a receiver-relative entity workspace. One query, formed from the initial self token and a stop-gradient copy of previous temporal memory, reads every intermediate set. Learned fusion is projected and added to the base feature before one GRU update produces the next memory and action values. Update and readout parameters are reused across rounds; the next observation initializes a fresh workspace. Entity rows and feature cells are schematic.

各个 Read 框重复标注 Self + history，表示同一决策内共享的查询条件。Base feature 来自同一实体编码接口；正文给出融合后的残差投影与循环更新公式。

## 4. Introduction：六段，每段推进一步

### I-1：把规模变化写成具体的协作决策问题

**段落目的：** 让读者先理解为什么需要这个方法。

> Cooperative target protection requires defenders to redistribute their effort as the numbers of vehicles and protected assets change.

句子路线：S1提出任务；S2用“攻击者接近目标、队友可能已覆盖”说明关系；S3说明新增目标与车辆如何改变责任分配；S4提出同一策略复用的需求。

收束句：

> A reusable controller must turn these changing relationships into actions for each member of the team.

### I-2：从任务导出两类信息处理需求

**段落目的：** 连接可变实体接口、实体上下文和个体相关性。

> Entity-based policies provide a common interface to changing scenes, while individual decisions depend on both entity context and receiver-specific relevance.

句子路线：S1承接接口；S2介绍 REFIL 的学习基础；S3解释攻击者表征如何结合目标和队友关系；S4说明同一上下文对不同防御者具有不同意义；S5引出个体读取。

收束句：

> This motivates a policy architecture that develops entity context and then exposes it to each defender through an individualized readout.

**引用与图：** REFIL；图1。

### I-3：说明共享迭代与中间状态访问为何配合

**段落目的：** 把设计理由与已有研究连接起来。

> Shared iterative attention offers a reusable operator for developing entity context, and intermediate-state access lets decisions draw on several stages of that computation.

句子路线：S1提出原则；S2承接 relational attention 与 Universal Transformer；S3说明后续轮处理前一轮已形成的上下文；S4用跨层表征组合解释保留中间状态的意义；S5落到当前防御者怎样读取。

收束句：

> The resulting design task is to connect these intermediate states to the defender's own state and history.

**引用：** Zambaldi；Universal Transformer；Jumping Knowledge。

### I-4：一段讲清 LEAF 的主要路径

**段落目的：** 让读者能够复述方法。

> LEAF connects shared entity refinement to individual action selection through history-conditioned reading and learned cross-round fusion.

句子路线：S1给方法定位；S2从相对实体视图进入共享更新；S3介绍 self 与历史构成的查询；S4说明各轮读取与融合；S5说明融合信息进入循环价值网络。

收束句：

> This architecture makes processing depth a property of both representation learning and the information available to the decision network.

**图：** 图3。

### I-5：由结构引出跨深度训练

**段落目的：** 解释实验为什么同时考察训练方式和执行轮数。

> Sampling the unfolding depth trains the shared policy through both short decision paths and longer sequences of entity updates.

句子路线：S1提出训练安排；S2说明早期状态可直接服务决策并继续参与后续更新；S3说明同一参数在不同展开长度下优化；S4引出相同单轮执行的比较。

收束句：

> We therefore examine the quality of the learned one-round policy alongside the effects of additional execution rounds and intermediate reads.

### I-6：用贡献与检验问题结束引言

**段落目的：** 明确本文交付的设计和证据。

> The evaluation connects this design to population transfer, the quality of shallow execution, and the information used by the trained policy.

三项贡献锚句：

- **Architecture:** A shared refinement and individual readout architecture that connects intermediate entity states to recurrent action values.
- **Training:** A sampled-depth learning design that couples short and long unfoldings within the entity-factorized value objective.
- **Evidence:** A population-transfer study with matched-depth comparisons and targeted interventions on intermediate reads and temporal conditioning.

此段只给效果方向、比较对象和证据类型。具体结果放在对应图表，正文解释其意义。

## 5. Related Work：三条关系，五个短段

| 段落 | 本段的问题 | 展开顺序 | 本段落点 |
| --- | --- | --- | --- |
| RW-1 | 变化的团队与任务需要什么接口？ | 可变实体表示 → 实体策略与价值学习 | 集合接口为策略复用提供基础 |
| RW-2 | 与最接近的方法怎样衔接？ | REFIL 的 imagined decomposition → 接收者读取的邻近工作 | LEAF 发展实体编码到循环 utility 的路径 |
| RW-3 | 为什么共享更新并访问中间状态？ | relational attention／Universal Transformer → Jumping Knowledge | 组合共享迭代与多阶段访问 |
| RW-4 | 为什么训练展开深度和历史条件值得讨论？ | LayerDrop 的深度训练思路 → 时间上下文 | LEAF 的共享展开与历史读取在价值学习中相接 |
| RW-5 | 这项研究服务什么自动化需求？ | 协作保护、追踪与规模变化中的控制 | 冻结策略跨车辆和目标数量复用 |

两句定位文字：

> LEAF retains REFIL's learning interface and develops the path from contextual entity states to receiver-conditioned recurrent utilities.

> LEAF combines shared refinement, intermediate-state access, and depth-varying training in a receiver-relative workspace with history-conditioned reading.

引用围绕实际关系选择。首轮优先使用现有已核对文献中的 REFIL、UPDeT、TransfQMix、SPECTra、Zambaldi、Universal Transformer、Jumping Knowledge、LayerDrop，以及与应用段直接相关的自动化工作。每篇文献承担一个明确的承接作用。

## 6. Problem Formulation and Preliminaries：先任务，再接口

### III-A. Cooperative Target Protection

**P-1：角色与动作。** 定义防御者、攻击者、目标和初始配置 \(\mathbf n=(N_R,N_B,K)\)。用图2说明协作覆盖与攻击接近。正文紧凑交代运动、交互和终止规则；具体物理常数集中在一个设置表。

> We study a cooperative target-protection task in which defenders coordinate their actions against incoming attackers across a changing set of protected targets.

**P-2：观测与历史。** 按“可见的实体物理状态 → 隐含的攻击者目标分配 → 历史对当前判断的作用”展开。

> Public entity states describe the physical scene, while the attacker's private target assignment makes recent motion informative for defensive decisions.

**P-3：保护目标。** 一次定义累计目标损伤、团队奖励及数值方向，后文沿用。

\[
\Delta D_t=\sum_{k\in\mathcal T}\Delta d_{k,t},\qquad
D=\sum_t\Delta D_t,\qquad r_t=-\Delta D_t.
\]

段末说明：相同配置、相同场景条件下比较各策略的保护效果。

### III-B. Population Generalization

**P-4：迁移的定义。** 固定物理规则与反应式攻击者，将规模配置作为变化轴；共享策略参数在测试时保持固定。

> We evaluate whether a single learned defender policy preserves effective cooperation as vehicle and target populations change.

本段依次定义训练配置集合、测试变化轴和评价量 \(\mathbb E[D\mid\mathbf n,\pi_\theta,\pi_B]\)。具体配置列表进入实验设置框，任务定义集中表达迁移含义。

### III-C. Entity-Factorized Value Learning

**P-5：执行接口。** 按“实体编码 → 基础特征 \(\ell_{i,t}\) → 记忆 \(h_{i,t}\) → 动作值 \(Q_i\)”介绍 LEAF 接入位置。

**P-6：训练接口。** 用一段说明 centralized mixer、主分支与 imagined 分支、共同 TD 目标。保留 REFIL 引用，并给出总目标的紧凑形式。

> LEAF augments the entity feature entering the recurrent utility network while retaining the entity-factorized cooperative learning interface.

这一章交付给 Method 的对象只有：实体输入与 mask、基础特征、个体记忆、团队价值学习目标。

## 7. Method：让公式服务于信息流

开篇先给图3和一段总览，再按四个小节展开。每一节先说它完成什么，再给公式，最后解释为何下一节需要其输出。

### IV-A. Receiver-Relative Entity Workspace

**M-1：初始化。** 解释相对防御者的实体表示、self 身份与 valid mask，给出 \(H^{(0)}\) 的构造。共享参数与不同接收者视角在此建立。

> Each defender constructs an entity workspace that expresses the scene relative to its own state while preserving the identities of the participating entities.

**M-2：数量条件。** 简述 typed counts 如何提供当前有效实体数量信息；精确编码与 mask 约定放在本节末的紧凑实现段。当前图稿与已有实验对应 counts-on 版本。

过渡：初始化完成后，同一实体工作区进入共享更新。

### IV-B. Shared Contextual Refinement

**M-3：核心算子。** 用一个公式组表达共享更新和中间状态保留。

\[
H^{(r)}=F_\theta(H^{(r-1)};z_i,m_i),\qquad r=1,\ldots,R.
\]

> A shared attention block repeatedly updates the entity workspace, allowing each application to operate on the context formed by the preceding one.

**M-4：轮次含义。** 解释 keys/values 来自上一轮，所有轮次作用于当前允许的实体集合，各轮状态保留给 reader。注意力残差、归一化与前馈的定义紧跟核心公式放在正文中。

过渡：得到一系列具有上下文的实体集合后，接收者从中提取决策信息。

### IV-C. History-Conditioned Reading and Fusion

**M-5：共享查询与逐轮读取。**

\[
q_{i,t}=W_q[H^{(0)}_{i,t,\mathrm{self}};\operatorname{sg}(h_{i,t-1})]+b_q,
\qquad u^{(r)}_{i,t}=\operatorname{Read}(q_{i,t},H^{(r)}_{i,t};m_{i,t}).
\]

> The defender uses its initial self representation and temporal memory to read each intermediate entity set with a common query.

本段按“查询输入 → 对应轮次的 K/V → 该轮摘要”展开；随后一句说明 query 副本的 stop-gradient 约定。

**M-6：融合与动作。**

\[
\alpha^{(r)}=\operatorname{softmax}_r(s(u^{(r)})),\qquad
u=\sum_r\alpha^{(r)}u^{(r)},
\]
\[
\ell'_{i,t}=\ell_{i,t}+W_fu_{i,t}+b_f,\qquad
h_{i,t}=\operatorname{GRU}(\ell'_{i,t},h_{i,t-1}),\qquad
Q_i=W_oh_{i,t}+b_o.
\]

> Learned fusion combines the round-specific summaries into a residual decision feature, after which the recurrent utility network performs one temporal update.

段末交代：融合投影零初始化；新的观测重新构建实体工作区，循环记忆进入下一物理时刻。上述公式给出了图3省略的残差投影。

### IV-D. Learning Across Processing Depths

**M-7：训练怎样使用结构。** 说明采样展开深度、共享参数在短路径和长路径中的共同优化，以及普通部署采用的深度。

> Sampling the unfolding depth exposes the shared policy to both direct use of early representations and longer sequences of contextual refinement.

**M-8：学习目标与执行属性。** 紧凑列出主／辅助 TD 目标、奖励 shaping 与终止处理的必要定义；最后说明参数共享、实体重排一致性和计算量随展开长度变化。将伪代码放在本节末，保持一次决策的自然顺序。

**本章收束：** 共享更新组织实体上下文，个体读取连接上下文与动作，跨深度训练共同优化这条路径。实验据此检验效果和使用方式。

## 8. Experiments：一个设置段，三个核心问题

### V-A. Setup and Evaluation Protocol

设置段承担复现与解释比较条件的任务，必要数值在此集中呈现：训练／测试配置、交互预算、最终 checkpoint、实际训练 seed 数、评价场景及统计单位。

**【设置框待排版】**

| 字段 | 初稿内容 |
| --- | --- |
| 任务与对手 | HAD；固定反应式攻击者；公开实体物理状态 |
| 训练／测试变化 | 平衡团队；车辆数量、目标数量、联合变化；测试时冻结策略 |
| LEAF 身份 | counts-on；query history stop-gradient；训练采样展开深度 |
| 主要指标 | 累计目标损伤，越低表示保护越好 |
| 汇总次序 | 每个 seed 内 scene → configuration → group；随后跨训练 seeds 汇总 |
| 不确定性 | 算术均值与样本标准差，并标出各方法实际训练 seed 数 |
| 主要对照 | REFIL、capacity control、独立单轮、ARR及已完成外部基线 |

准确范围和预算从已有已审计设置表填入一次。REFIL-matched 与 Rule 的定义补齐后进入表注。

### V-B. Can a Single Policy Transfer Across Populations?

**核心结论：** 同一冻结 LEAF 策略在所评估的规模变化中改善目标保护。

**英文初稿：**

> We first evaluate target protection as team size and the number of protected targets change. A single trained LEAF policy is deployed across the evaluation configurations with fixed parameters. LEAF achieves lower mean target damage than REFIL across the population-shift groups and the evaluated configurations. The group comparison summarizes overall protection quality, while configuration-level curves show how that advantage behaves as the deployment setting grows. Together, these results establish the cross-scale performance of the complete LEAF policy in the target-protection task.

句子目的：问题 → 比较设置 → 结果方向 → 图表如何分工 → 本节意义。

**【主表 T1 留位】** 方法 × ID／车辆变化／目标变化／联合变化；实际 seed 数与 mean ± sample SD。

**【实验图 E1 留位】** 三个对齐面板，分别呈现三类规模变化。只选对判断趋势有帮助的比较曲线，其余已完成基线进入主表。

**【可选场景展示留白】** 若要直观展示“防御责任怎样随场景变化”，选择同一初态下 LEAF 与 REFIL 的代表性轨迹，标出实际动作、目标接近与损伤时刻。需要配对轨迹或回放；选择规则在图注说明。

### V-C. What Does Training Across Depths Change?

**核心结论：** 相同单轮执行条件下，跨深度训练得到更有效的完整策略。

**英文初稿：**

> We next examine the policy learned through multi-depth training by executing both LEAF and an independently trained shallow comparator with a single entity update. LEAF yields lower damage across the matched evaluation settings, with a consistent direction across the common training seeds. The shared interface and matched execution depth make this a comparison of the learned shallow policies. The result supports training through multiple unfolding depths as a way to obtain an effective one-round decision policy.

句子目的：控制执行深度 → 观察方向 → 明确比较对象 → 解释训练意义。

**【实验图 E2a 留位】** 公共 seed 的配对点：训练方式不同，执行深度相同。图注交代共同交互预算和不同训练展开计算量。

ARR 与执行深度扫描可在这一节各用一句解释其支持的结构观察。正文的主结论围绕共同训练得到的策略质量。

### V-D. How Are Intermediate States and History Used?

**核心结论：** 早期读出与查询中的时间历史参与当前决策。

**第1段：中间读出。**

> We then examine how the deployed policy uses its intermediate reads. Paired interventions preserve the current observation and reconstructed recurrent history while changing which reads enter fusion. Keeping the earliest pair of reads preserves most sampled actions, whereas selecting only the first read or removing the second produces larger changes. Closed-loop evaluations also show scene-dependent outcomes under different readout choices. These observations identify the early reads as active inputs to action selection.

句子目的：问题 → 干预条件 → 主要模式 → 闭环行为衔接 → 本段结论。

**第2段：历史条件。**

> We next examine the temporal information supplied to the reader. A paired intervention changes the historical query component while preserving the current observation and incoming recurrent state. The resulting action changes show that the trained reader uses temporal context, including in larger-team evaluations. This connects the history-conditioned design to the decisions produced by the deployed policy.

句子目的：承接个体读取 → 明确改变项 → 观察 → 方法对应关系。

**【实验图 E2b–c 留位】** 与 E2a 合成一张有统一字号、留白和配色的证据图：b为选择性读出干预，c为查询历史干预。每个面板只回答一个问题。

**【表征探针可选位】** 若本轮需要进一步解释“实体更新形成什么信息”，加入一个短段及小面板：输入编码、首轮、后续轮的 threat-probe，并保留随机特征与打乱标签控制。需要先确定这层表征解释在全文中的篇幅。

可用锚句：

> Held-out probes recover geometric threat information more effectively after the first learned entity update, linking the refinement stage to information that is relevant to target protection.

### 现有实验内容的取舍

| 内容 | 初稿安排 |
| --- | --- |
| 完成的主比较与跨规模趋势 | T1 + E1 |
| 匹配单轮执行、读出干预、历史干预 | E2a–c |
| ARR、运行深度变化 | 与相应问题合并叙述 |
| threat probe | 一个可选段落／面板 |
| token cosine、fusion weights、norm shares、gradient routes | 保留记录，待形成独立问题后选用 |
| 大量相近读出变体、探索性探针 | 根据核心结论合并或省略 |
| 未完成实验 | 保留明确的证据槽 |

## 9. Conclusion：用一段结束完整故事

> LEAF connects shared iterative entity processing with history-conditioned access to intermediate representations for cooperative target protection. The resulting policy transfers across changes in vehicle and target populations, and training across unfolding depths also yields an effective shallow execution path. Targeted interventions link the deployed policy's decisions to early reads and temporal context. Together, these findings support organizing multi-agent decision making around contextual entity representations and individualized access to their intermediate states.

句子目的：方法路径 → 任务与训练证据 → 行为证据 → 设计认识。讨论后续方向时，选择一项与作者下一阶段目标直接相关的内容接在末尾。

## 10. 下一轮逐句展开时的编辑规则

- 每段先用一句回答本段问题，再交代证据或机制，段末把问题交给下一段。
- 同一个结论在引言概括，在实验建立，在结论提炼；三个位置各有不同作用。
- 数值集中在设置与图表。结果段只有在量级本身改变解释时引用关键数值。
- 方法段顺着对象流动写：输入是什么、模块做什么、输出被谁使用。
- 文献用于承接设计原则；贡献写成本文建立的路径、训练安排和经验认识。
- 实验段先讲比较如何对应问题，再讲观察意味着什么。
- 每张图保留一个核心信息。新增面板须能回答一个正文已提出的问题。
- 需要补足的展示以“问题＋所需记录＋预期图位”留白，便于下一轮直接处理。

**当前稿的可迭代单位：** 三张概念图、六段引言、三个实验问题。后续逐个确认并展开这些单位，即可形成连贯的正文初稿。
