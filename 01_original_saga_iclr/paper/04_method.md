<!-- 状态: 完整可用初稿 | 与 prototype/ 代码模块一一对应; 伪代码与超参表在文末 -->

# 4. Method: SAGA

SAGA (Scale-Agnostic Grouping and Adaptation) is a hierarchical team policy operating at two timescales: a **commander** invoked every $k$ environment steps produces a grouping plan and per-agent goals; a shared **micro policy** acts at every step conditioned on its goal. Both are built on a common size-agnostic representation. Figure 1 [架构图] gives an overview; we describe the three components and then the training procedure.

**Design constraints.** Every component must satisfy: (C1) *size-agnosticism* — parameters independent of $N_t$ and $M_t$; (C2) *permutation invariance/equivariance* over unit indices; (C3) *decentralized executability* — between commander broadcasts, units act on local observations only, matching a bandwidth-limited setting where goals are broadcast at period $k$; (C4) *opponent-anchoring* — macro decisions are functions of the opponent's observed structure, not of fixed conventions.

## 4.1 Entity-field encoder (module A)

All observed units — teammates, opponents, arriving reinforcements — are embedded as tokens. For observer $i$, each visible entity $e$ yields
$\tau_{i,e} = [\,\text{rel. kinematics};\ \text{health frac.};\ \text{team flag};\ x_e\,] \in \mathbb{R}^{d_\tau}$,
where $x_e$ is the type-attribute vector, so unseen unit *types* enter through the same interface as unseen unit *counts* (attribute extrapolation requires no architectural change).

Observer embeddings use **single-query cross-attention** (following the SAQA pattern [SPECTra]): a learned query token attends over the observer's entity set,
$$
h_i \;=\; \mathrm{FF}\Big(\mathrm{MHA}\big(q_0,\ \{W \tau_{i,e}\}_{e \in \mathcal{V}_i}\big)\Big) \in \mathbb{R}^{d},
$$
with cost $O(|\mathcal{V}_i|\, d)$ — linear, not quadratic, in entity count. A team-level **population summary** $\bar h$ (masked mean/max pooling over all observed entities, Deep-Sets style) is computed once per commander step and provides cheap global context without erasing entity identity. All parameters are shared across units and independent of $(N_t, M_t)$, satisfying C1–C2.

## 4.2 Opponent-anchored commander (module B)

Every $k$ steps, the commander computes the grouping plan $(\mathcal{C}, g)$ in three stages.

**(B1) Adaptive opponent clustering.** The observed opponent tokens $\{\tau^B_j\}$ are softly clustered by iterative **slot attention** with $K_{\max}$ learnable slots: for $L$ iterations, attention is normalized *across slots* (slots compete for entities), followed by a GRU slot update. The output is cluster descriptors $c_1, \dots, c_{K_{\max}}$ and an **occupancy** $o_\kappa = \sum_j A_{j\kappa}$ measuring the entity mass each slot explains. Downstream use is gated by occupancy (below), so the *effective* cluster count $K = |\{\kappa : o_\kappa > \epsilon\}|$ is data-dependent: a concentrated opponent yields one active cluster; a three-pronged advance yields three. No stage references $M_t$ or a fixed $K$ (C1, C4). Naive slot attention keeps all $K_{\max}$ slots active; we obtain genuinely adaptive cluster counts with two unsupervised auxiliary losses: a *spatial coherence* term $\mathcal{L}_{\text{coh}}$ (soft-assignment-weighted intra-cluster position variance) and a *concave sparsity* term $\mathcal{L}_{\text{sp}} = \sum_\kappa \sqrt{o_\kappa / M}$ penalizing fragmented occupancy, with the sparsity weight warmed up to avoid early single-slot collapse. On synthetic probes with 1–4 ground-truth opponent groups, this recovers the true group count within $\pm 1$ in [96]% of scenes (correlation [0.81]) with no supervision (Appendix E.1), validating that the effective cluster count is data-dependent as designed.

**(B2) In-context style embedding.** A GRU encodes the recent sequence of team-level engagement statistics $\psi_{t-Tk:t}$ — opponent count and spatial dispersion, closing speed, firing frequency, regrouping frequency — into a style vector $z \in \mathbb{R}^{d_z}$. The statistics are computed from observations only; no opponent labels exist at training or test time. $z$ is the mechanism for *behavioral* (as opposed to structural) opponent anchoring: two opponents with identical spatial structure but different aggression should induce different allocations.

**(B3) Force–task matching.** Each active cluster spawns $n_{\text{task}}$ task tokens (engage / contain / flank), computed as $\mathrm{FF}([c_\kappa; e_{\text{type}}; z])$, and $n_{\text{free}}$ cluster-independent reserve tokens. Own-unit embeddings attend over task tokens:
$$
\mathrm{assign}_{i\cdot} = \mathrm{softmax}\big( \langle W_u h_i,\ \text{tasks} \rangle / \sqrt{d} + \log o_{\cdot} \big), \qquad g_i = \textstyle\sum_{\text{tasks}} \mathrm{assign}_{i\cdot}\, \text{task}_\cdot,
$$
where the $\log o$ term gates empty clusters' tasks off smoothly. The per-unit goal $g_i$ — a continuous vector carrying *which* opponent group to affect and *how* — is broadcast and held constant until the next commander step (C3). Reserve tasks give the policy a learned "do not overcommit" option, which we find important against degenerate opponents (Section 5 ablation).

**Why anchoring yields dual generalization (mechanism summary).** Scale shift changes *how many* opponents there are; anchoring maps it to *how many/large* the inferred clusters are, to which task generation and matching respond compositionally. Opponent shift changes *where mass is and how it behaves*; anchoring maps it to cluster descriptors and $z$. In both cases the change is absorbed at the representation's input, not by the parameters — which is precisely what permits zero-shot transfer of a single $\theta$.

## 4.3 Goal-conditioned micro policy (module C)

Every unit runs a shared policy $\pi^{\text{lo}}(a_i \mid o_i, g_i)$: the entity-field embedding $h_i$ concatenated with $g_i$ feeds (i) a maneuver head (discretized turn/acceleration bins) and (ii) a target head that **scores each visible opponent token** against a query derived from $h_i$, plus a learned null-target logit — so the target action space grows and shrinks with visibility, requiring no fixed maximum (C1). Weight sharing across all units makes reinforcements trivially absorbable: a new unit is simply a new token producing a new $h_i$ and receiving a goal at the next broadcast.

## 4.4 Training: value decomposition and the SxS autocurriculum

**Objective.** SAGA trains end-to-end with MAPPO-style centralized-critic PPO. The critic is a permutation-invariant set network over all entities (size-agnostic, C1) producing a team value. To combat lazy-agent credit assignment at large $N$, the team reward is augmented with **task-aligned shaping**: unit $i$ receives a small bonus proportional to progress on *its assigned cluster* (damage dealt to it, containment maintained), computed from the commander's assignment weights — coupling the two levels' gradients without hand-designed roles. This shaping is not potential-based and can in principle bias the optimum; we keep its weight small (0.1× the team-reward scale), and ablation (d) verifies its role is to accelerate credit assignment rather than to alter final behavior qualitatively (details in Appendix C.2). The commander and micro policy are optimized jointly on $\mathcal{L}_{\text{PPO}} + \lambda_{\text{coh}} \mathcal{L}_{\text{coh}}$; goals are treated as deterministic latent features (gradient flows through $g_i$), avoiding the instability of separately learned option policies.

**SxS autocurriculum.** Each episode samples a pair $(\sigma, \mu)$:

- *Scale axis* $\sigma$: initial sizes $N_0, M_0$, force ratio, reinforcement rate and batch distribution — from a grid over the training ranges.
- *Style axis* $\mu$: from a lightweight league of three sources: (1) **parameterized scripts** (~12 families — pursuit, kiting, focus-fire, flanking, feint, swarm-split, passive — each with continuously randomized parameters, spanning incompetent to competent play); (2) **prioritized self-snapshots**: past checkpoints of $\theta$, sampled $\propto$ current loss rate against them (prioritized fictitious self-play), supplying adaptive, "intelligent" opposition and non-transitivity coverage; (3) **periodic exploiters**: every $E$ updates, a fresh opponent is trained against frozen $\theta$ for a small budget and added to the league — a minimal-viable version of league exploiters [AlphaStar] costing $\sim$5–10% of main-policy compute per exploiter.

Sampling over the $(\sigma, \mu)$ grid follows the **learning frontier**: cells are drawn $\propto \exp\{-(W_{\text{cell}} - 1/2)^2 / T\}$, concentrating on near-fair matchups where the policy gradient signal is largest (a discrete, regret-flavored simplification of UED). The two axes are sampled *jointly*, exposing the representation to e.g. "many opponents playing a style just learned at small scale" — combinations that per-axis curricula never produce; the S×S-joint ablation isolates this effect.

## 4.5 Pseudocode

```text
Algorithm 1: SAGA training with SxS autocurriculum
Input: scenario grid Σ_train, script library; K_max, commander period k
Initialize θ (encoder, commander, micro policy, critic); league L ← scripts; win-rate table W
for iteration = 1, 2, ...:
    # --- collect ---
    for env in parallel batch:
        (σ, μ) ~ frontier-sampling(W)              # joint 2-D curriculum
        rollout episode:
            every k steps:  cluster opponents → (c, o); update z; match → goals g
            every step:     each unit: a_i ~ π_lo(·| o_i, g_i)
            on reinforcement event: new tokens absorbed; goals refresh at next broadcast
    update W with episode outcomes
    # --- learn ---
    θ ← PPO(θ; team reward + task-aligned shaping + λ_coh·L_coh)
    if iteration mod snapshot_every == 0:  L ← L ∪ {frozen θ}
    if iteration mod exploiter_every == 0: train exploiter vs frozen θ (small budget); L ← L ∪ {exploiter}
Output: single parameter vector θ valid for any (N, M) and opponent
```

## 4.6 超参与实现要点（附录素材）

| 项 | 值（初设） | 备注 |
|---|---|---|
| $d$ (model dim) | 64–128 | 原型 64 已验证形状 |
| $K_{\max}$ / slot 迭代 $L$ | 6–8 / 3 | 消融 K_max 敏感性 |
| 指挥周期 $k$ | 8 | 消融预期 U 形 |
| 任务类型 $n_{\text{task}}$ / 自由 slot | 3 / 2 | engage, contain, flank |
| $\lambda_{\text{coh}}$ | 1e-2 | 聚类空间一致性正则 |
| 快照池 / exploiter 频率 | 每 50 次更新 / 每 200 次 | exploiter 预算 = 主策略 5% |
| 训练规模域 | $N_0, M_0 \in [4, 16]$ | 测试外推至 64+ |
| 框架 | JAX (env+train 全向量化) | 原型为 PyTorch 接口版 |

**实现注意（来自原型的教训，写作时删）**：
1. ~~slot occupancy 需配合正则才会退化空簇~~ → **已解决并实测**（P0 测试通过，96%/0.81）：工艺为 $\mathcal{L}_{\text{coh}} + \beta \mathcal{L}_{\text{sp}}$，$\beta$ 前 40% 训练 warmup，sqrt 参数 clamp 防 NaN——两条工艺细节都写入 §4.6 与规格文档；
2. 训练器必须按"同一 (N,M) 桶"分批做 padded batch，否则 GPU 利用率不可接受；
3. `team_stats` 统计量的最终清单（16 维 v1.0）见规格文档 §2.2，与 z_opp 的 t-SNE 可分性实验联动确定。

**理论支撑（正文引用位置备忘）**：命题 1-2（置换等变、规模无关与线性复杂度）在 §4.1-4.3 末尾一句话引用；命题 3（实体增删 $O(1/M)$ 扰动界）在 §4.2 末连接"增援平滑吸收"；命题 4（摘要空间覆盖泛化界）作为 §4.2 "mechanism summary" 段的形式化版本引用。全文见 `08_theory.md`。

---

# 中文翻译

# 4. 方法：SAGA

SAGA（规模无关分组与自适应）是一个在两个时间尺度上运行的层次化团队策略：一个**指挥器**每 $k$ 个环境步调用一次，产生分组计划和逐智能体目标；一个共享的**微观策略**在每步基于目标行动。两者都建立在一个通用的规模无关表示之上。图 1 [架构图] 给出了概览；我们描述三个组件然后介绍训练过程。

**设计约束。** 每个组件必须满足：(C1) *规模无关*——参数与 $N_t$ 和 $M_t$ 无关；(C2) *置换不变/等变*——对单位索引；(C3) *去中心化可执行性*——在指挥器广播之间，单位仅基于局部观测行动，匹配带宽受限的设定，其中目标以周期 $k$ 广播；(C4) *对手锚定*——宏观决策是对手观测到的结构的函数，而非固定约定。

## 4.1 实体场编码器（模块 A）

所有观测到的单位——队友、对手、到达的增援——都被嵌入为 token。对于观测者 $i$，每个可见实体 $e$ 生成
$\tau_{i,e} = [\,\text{相对运动学};\ \text{生命值比例};\ \text{队伍标志};\ x_e\,] \in \mathbb{R}^{d_\tau}$，
其中 $x_e$ 是类型属性向量，因此未见的单位*类型*通过与未见单位*数量*相同的接口进入（属性外推不需要架构改变）。

观测者嵌入使用**单查询交叉注意力**（遵循 SAQA 模式 [SPECTra]）：一个可学习的查询 token 在观测者的实体集合上做注意力，
$$
h_i \;=\; \mathrm{FF}\Big(\mathrm{MHA}\big(q_0,\ \{W \tau_{i,e}\}_{e \in \mathcal{V}_i}\big)\Big) \in \mathbb{R}^{d},
$$
计算成本为 $O(|\mathcal{V}_i|\, d)$——与实体数量呈线性而非二次关系。团队级的**种群摘要** $\bar h$（对所有观测实体的掩码均值/最大池化，Deep-Sets 风格）每个指挥步骤计算一次，提供廉价的全局上下文而不抹去实体身份。所有参数跨单位共享且与 $(N_t, M_t)$ 无关，满足 C1–C2。

## 4.2 对手锚定指挥器（模块 B）

每 $k$ 步，指挥器分三个阶段计算分组计划 $(\mathcal{C}, g)$。

**(B1) 自适应对手聚类。** 观测到的对手 token $\{\tau^B_j\}$ 通过迭代 **slot attention**（含 $K_{\max}$ 个可学习 slot）进行软聚类：经过 $L$ 次迭代，注意力*跨 slot* 归一化（slot 竞争实体），后接 GRU slot 更新。输出是聚类描述子 $c_1, \dots, c_{K_{\max}}$ 和**占用度** $o_\kappa = \sum_j A_{j\kappa}$，度量每个 slot 解释的实体质量。下游使用按占用度门控（见下），因此*有效*聚类数 $K = |\{\kappa : o_\kappa > \epsilon\}|$ 是数据依赖的：集中的对手产生一个激活聚类；三路推进产生三个。没有阶段引用 $M_t$ 或固定的 $K$（C1, C4）。朴素 slot attention 保持所有 $K_{\max}$ 个 slot 激活；我们通过两个无监督辅助损失获得真正自适应的聚类数：一个*空间一致性*项 $\mathcal{L}_{\text{coh}}$（软分配加权的簇内位置方差）和一个*凹稀疏性*项 $\mathcal{L}_{\text{sp}} = \sum_\kappa \sqrt{o_\kappa / M}$ 惩罚碎片化占用，稀疏权重经过预热以避免早期单 slot 坍缩。在具有 1–4 个真实对手组的合成探针上，该方法在 [96]% 的场景中以 $\pm 1$ 的精度恢复真实组数（相关性 [0.81]），无需任何监督（附录 E.1），验证了有效聚类数确实如设计的那样是数据依赖的。

**(B2) 上下文风格嵌入。** 一个 GRU 对最近的团队级交战统计序列 $\psi_{t-Tk:t}$ 进行编码——对手数量和空间散布、接近速度、射击频率、重组频率——得到风格向量 $z \in \mathbb{R}^{d_z}$。统计量仅从观测计算；训练和测试时不存在对手标签。$z$ 是*行为*（而非结构性）对手锚定的机制：两个空间结构相同但攻击性不同的对手应该引发不同的分配。

**(B3) 兵力-任务匹配。** 每个激活聚类生成 $n_{\text{task}}$ 个任务 token（交战/牵制/迂回），计算为 $\mathrm{FF}([c_\kappa; e_{\text{type}}; z])$，加上 $n_{\text{free}}$ 个与聚类无关的预备 token。己方单位嵌入对任务 token 做注意力：
$$
\mathrm{assign}_{i\cdot} = \mathrm{softmax}\big( \langle W_u h_i,\ \text{tasks} \rangle / \sqrt{d} + \log o_{\cdot} \big), \qquad g_i = \textstyle\sum_{\text{tasks}} \mathrm{assign}_{i\cdot}\, \text{task}_\cdot,
$$
其中 $\log o$ 项平滑地门控掉空聚类的任务。逐单位目标 $g_i$——一个连续向量，携带*影响哪个*对手组以及*如何*影响的信息——被广播并保持不变直到下一个指挥步骤（C3）。预备任务给策略一个学习到的"不过度投入"选项，我们发现这对退化型对手很重要（第 5 节消融）。

**为什么锚定产生双重泛化（机制总结）。** 规模偏移改变了*多少*对手；锚定将其映射为推断聚类*有多少/多大*，任务生成和匹配对此进行组合式响应。对手偏移改变了*质量在哪里以及如何行动*；锚定将其映射为聚类描述子和 $z$。在两种情况下，变化都在表示的输入端被吸收，而非由参数承担——这正是允许单一 $\theta$ 零样本迁移的原因。

## 4.3 目标条件微观策略（模块 C）

每个单位运行共享策略 $\pi^{\text{lo}}(a_i \mid o_i, g_i)$：实体场嵌入 $h_i$ 与 $g_i$ 拼接后输入 (i) 机动头（离散化转向/加速度分箱）和 (ii) 目标头，后者对每个可见对手 token 基于从 $h_i$ 衍生的查询进行**评分**，加上一个可学习的空目标 logit——因此目标动作空间随可见度增减，无需固定最大值（C1）。所有单位的权重共享使增援可平凡地被吸收：新单位只是一个产生新 $h_i$ 并在下次广播时接收目标的新 token。

## 4.4 训练：价值分解与 SxS 自动课程

**目标。** SAGA 使用 MAPPO 风格的集中式评论家 PPO 进行端到端训练。评论家是一个置换不变的集合网络，处理所有实体（规模无关，C1），产生团队价值。为了在大 $N$ 时对抗懒惰智能体的信用分配问题，团队奖励增加了**任务对齐塑形**：单位 $i$ 获得一个与*其被分配的聚类*的进展成正比的小额奖金（对其造成的伤害、维持的牵制），从指挥器的分配权重计算——耦合两个层级的梯度而无需手工设计的角色。此塑形非基于势函数，原则上可能偏移最优值；我们保持其权重较小（团队奖励尺度的 0.1 倍），消融 (d) 验证其作用是加速信用分配而非定性地改变最终行为（详见附录 C.2）。指挥器和微观策略在 $\mathcal{L}_{\text{PPO}} + \lambda_{\text{coh}} \mathcal{L}_{\text{coh}}$ 上联合优化；目标被视为确定性隐变量（梯度流过 $g_i$），避免了单独学习的选项策略的不稳定性。

**SxS 自动课程。** 每个回合采样一个配对 $(\sigma, \mu)$：

- *规模轴* $\sigma$：初始规模 $N_0, M_0$、兵力比、增援率和批量分布——从训练范围的网格上采样。
- *风格轴* $\mu$：从三个来源的轻量级联赛中抽取：(1) **参数化脚本**（约 12 个族——追击、风筝、集火、迂回、佯攻、集群分裂、被动——每个具有连续随机化参数，覆盖从退化到胜任的水平）；(2) **优先自快照**：$\theta$ 的过去检查点，按当前对其的损失率成比例采样（优先虚拟自博弈），提供自适应的"智能"对抗和非传递性覆盖；(3) **周期性 exploiter**：每 $E$ 次更新，一个新对手针对冻结的 $\theta$ 以小预算训练并加入联赛——一个最小可行版本的联赛 exploiter [AlphaStar]，每个 exploiter 成本约为主策略计算的 5–10%。

在 $(\sigma, \mu)$ 网格上的采样遵循**学习前沿**：单元格按 $\exp\{-(W_{\text{cell}} - 1/2)^2 / T\}$ 被抽取，集中在接近公平对局的位置，此处策略梯度信号最大（UED 的一种离散的、基于遗憾的简化）。两个轴*联合*采样，使表示暴露于例如"在小规模下刚学会的风格面对大量对手"的组合——单轴课程永远不会产生这种组合；S×S 联合消融隔离了这一效果。

## 4.5 伪代码

```text
算法 1：SAGA 的 SxS 自动课程训练
输入：场景网格 Σ_train，脚本库；K_max，指挥周期 k
初始化 θ（编码器、指挥器、微观策略、评论家）；联赛 L ← 脚本；胜率表 W
for iteration = 1, 2, ...:
    # --- 采集 ---
    for env in 并行批量:
        (σ, μ) ~ 前沿采样(W)              # 联合二维课程
        回合展开:
            每 k 步：聚类对手 → (c, o)；更新 z；匹配 → 目标 g
            每步：各单位：a_i ~ π_lo(·| o_i, g_i)
            增援事件：新 token 被吸收；目标在下次广播时刷新
    用回合结果更新 W
    # --- 学习 ---
    θ ← PPO(θ; 团队奖励 + 任务对齐塑形 + λ_coh·L_coh)
    if iteration mod snapshot_every == 0:  L ← L ∪ {冻结 θ}
    if iteration mod exploiter_every == 0: 训练 exploiter vs 冻结 θ（小预算）；L ← L ∪ {exploiter}
输出：适用于任意 (N, M) 和对手的单一参数向量 θ
```
