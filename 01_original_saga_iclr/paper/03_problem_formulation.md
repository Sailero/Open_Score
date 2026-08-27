<!-- 状态: 完整可用初稿 | 形式化定义完整, 记号表见文末 -->

# 3. Problem Formulation

## 3.1 Two-team stochastic games with dynamic populations

We consider a two-team zero-sum partially observable stochastic game with *dynamic populations*,

$$
\mathcal{G} \;=\; \big\langle \mathcal{S},\, \mathcal{A},\, \mathcal{X},\, P,\, \rho,\, r,\, \Omega,\, \gamma \big\rangle .
$$

At time $t$ the state $s_t \in \mathcal{S}$ contains two *variable-size* sets of units, $\mathcal{U}^A_t = \{u^A_1, \dots, u^A_{N_t}\}$ and $\mathcal{U}^B_t = \{u^B_1, \dots, u^B_{M_t}\}$. Each unit $u$ carries a physical state (position, velocity, heading, health) and a **type-attribute vector** $x_u \in \mathcal{X} \subset \mathbb{R}^{d_x}$ (maximum speed, engagement radius, maximum health, damage, cooldown, sensing radius), so heterogeneity is parameterized continuously rather than by discrete classes.

**Population dynamics.** Team sizes $N_t, M_t$ change within an episode through two mechanisms: *attrition* — a unit whose health reaches zero is removed — and *reinforcement* — governed by a marked point process $\rho$ that inserts batches of new units (with sampled attributes and entry locations) at random times. Neither team controls or observes $\rho$'s schedule in advance. This makes the *index set* of agents itself part of the stochastic dynamics, a property most MARL formalisms exclude by assumption.

**Observations and actions.** Each unit $u \in \mathcal{U}^A_t$ receives $o_u^t = \Omega(s_t, u)$: the set of entity feature vectors of all units (either team) within its sensing radius, expressed in $u$'s frame. Its action $a_u = (a^{\text{man}}_u, a^{\text{tgt}}_u)$ consists of a maneuver (turn-rate/acceleration) and a target selection over *currently visible* opposing units or a null target — hence the action space, like the observation, varies with the entity count. The team receives a shared zero-sum reward $r$ (damage differential plus terminal outcome), discounted by $\gamma$.

**Team policies.** A team policy is a mapping $\pi_\theta$ that, for any realized population, produces a joint action: $\pi_\theta : (o_{u_1}, \dots, o_{u_{N_t}}) \mapsto (a_{u_1}, \dots, a_{u_{N_t}})$. We require $\pi_\theta$ to be well-defined for **every** $N_t, M_t \in \mathbb{N}$ with one fixed parameter vector $\theta$ — the *size-agnosticism* requirement — and to be equivariant to permutations of unit indices, since indices carry no meaning.

*Remark (information structure).* We allow $\pi_\theta$ to be history-dependent with a periodic-broadcast constraint: a team-level module may compute and broadcast per-unit conditioning information every $k$ steps, while between broadcasts each unit acts on its local observation and cached conditioning only. This matches bandwidth-limited deployment and is the information structure our method (and the fixed-$K$ hierarchical baselines) instantiate; flat baselines are the special case of stateless per-step policies.

## 3.2 Scenario and opponent distributions

A **scenario** $\sigma = (N_0, M_0, \rho, \chi) \in \Sigma$ fixes the initial team sizes, the reinforcement process, and the attribute distribution $\chi$ over unit types. An **opponent** is a team policy $\mu \in \mathcal{M}$ for team $B$. Training draws from $(\sigma, \mu) \sim \mathcal{D}_{\text{train}} = \Sigma_{\text{train}} \times \mathcal{M}_{\text{train}}$. The value of policy $\theta$ under a pair $(\sigma,\mu)$ is

$$
V(\theta;\, \sigma, \mu) \;=\; \mathbb{E}\!\left[\textstyle\sum_t \gamma^t r_t \,\middle|\, \pi_\theta \text{ vs. } \mu,\ \sigma\right],
$$

and we will primarily report the induced win rate $W(\theta; \sigma, \mu)$.

## 3.3 Dual zero-shot generalization

**Definition 1 (Dual zero-shot generalization).** Given disjoint test sets $\Sigma_{\text{test}} \cap \Sigma_{\text{train}} = \emptyset$ and $\mathcal{M}_{\text{test}} \cap \mathcal{M}_{\text{train}} = \emptyset$, the *dual generalization matrix* of $\theta$ is

$$
\mathbf{G}(\theta)\big[\sigma, \mu\big] \;=\; W(\theta;\, \sigma, \mu), \qquad (\sigma, \mu) \in \big(\Sigma_{\text{train}} \cup \Sigma_{\text{test}}\big) \times \big(\mathcal{M}_{\text{train}} \cup \mathcal{M}_{\text{test}}\big),
$$

evaluated **without any test-time parameter update**. The block $(\Sigma_{\text{train}}, \mathcal{M}_{\text{train}})$ is in-distribution performance; the single-axis blocks $(\Sigma_{\text{test}}, \mathcal{M}_{\text{train}})$ and $(\Sigma_{\text{train}}, \mathcal{M}_{\text{test}})$ measure scale-only and opponent-only generalization; the joint block $(\Sigma_{\text{test}}, \mathcal{M}_{\text{test}})$ is our primary object of study.

$\Sigma_{\text{test}}$ contains three graded families: (i) *interpolation/extrapolation* — initial sizes beyond the training range (e.g., train $N_0, M_0 \in [4,16]$, test up to $64$), including asymmetric ratios up to $1{:}3$; (ii) *in-episode dynamics* — reinforcement processes with rates and batch sizes outside the training range, and adversarial attrition events (instantaneous loss of a random half of the team); (iii) *attribute shift* — opponent unit attributes extrapolated beyond the training support (faster, longer-ranged, higher-health types).

$\mathcal{M}_{\text{test}}$ contains: (i) held-out parameterized scripts, including degenerate (passive, fleeing) and irregular (feint-and-ambush, sacrificial rush, split-and-merge) styles; (ii) independently trained learning opponents (different algorithms and seeds); and (iii) **exploiters**: best responses trained from scratch against the frozen final $\theta$, whose success upper-bounds the policy's worst-case vulnerability and serves as a tractable proxy for exploitability.

**Definition 2 (Recovery time).** For a mid-episode population event at time $t_e$ (reinforcement or attrition on either side), define the instantaneous *dominance* $D_t = \big(\sum_{u \in \mathcal{U}^A_t} hp_u - \sum_{v \in \mathcal{U}^B_t} hp_v\big) / \big(\sum_u hp_u + \sum_v hp_v\big) \in [-1, 1]$ and its sliding-window trend $\dot D_t$ (window $w$, both specified in Appendix D.2). The *recovery time* is $\min\{ \Delta \ge 0 : \dot D_{t_e + \Delta} \ge \dot D_{t_e^-} - \epsilon \}$ — how quickly the team's trajectory returns to its pre-event trend. Being computed from observable material state, this metric is model-free and comparable across methods; it isolates *adaptation speed* from raw strength.

## 3.4 The grouping sub-problem

We make the macro-level decision structure explicit. A **grouping plan** at commander step $k$ is a pair $(\mathcal{C}_k, g_k)$: a soft partition $\mathcal{C}_k$ of the *observed opponent set* into $K_k$ clusters (with $K_k$ data-dependent, not fixed), and an assignment $g_k$ mapping each own unit to a tactical goal derived from a cluster (engage, contain, flank) or to a reserve goal. The hypothesis this paper tests can now be stated precisely:

> **Opponent-anchoring hypothesis.** Let the grouping plan be a function of the opponent's observed structure and recent behavior, $(\mathcal{C}_k, g_k) = f_\theta(\mathcal{U}^B_{\le t}, \mathcal{U}^A_t)$, with $f_\theta$ permutation-invariant and size-agnostic in both arguments. Then a single $\theta$ trained on $\mathcal{D}_{\text{train}}$ achieves higher dual generalization $\mathbf{G}(\theta)$ than (a) flat policies of comparable capacity and training budget without grouping structure, and (b) hierarchical policies trained identically whose grouping is anchored to the ego team with a fixed cluster count $K$ — with the largest margins in test cells where the opponent's structure differs most from training.

Claims (a) and (b) map one-to-one onto baselines and ablations in Section 5.

## 3.5 记号表（写作用，投稿前移入附录）

| 记号 | 含义 |
|---|---|
| $N_t, M_t$ | t 时刻双方存活单位数（随攻减/增援变化） |
| $x_u \in \mathcal{X}$ | 单位类型属性向量（连续参数化异构） |
| $\rho$ | 增援标记点过程 |
| $\sigma \in \Sigma$ | 场景（初始规模、增援过程、属性分布） |
| $\mu \in \mathcal{M}$ | 对手团队策略 |
| $\mathbf{G}(\theta)$ | 双重泛化矩阵（主评测对象） |
| $(\mathcal{C}_k, g_k)$ | 第 k 个指挥周期的分组计划（对手软划分 + 己方任务分配） |
| $K_k$ | 数据依赖的激活簇数 |

---

# 中文翻译

# 3. 问题形式化

## 3.1 动态种群的双队随机博弈

我们考虑一个具有*动态种群*的双队零和部分可观测随机博弈，

$$
\mathcal{G} \;=\; \big\langle \mathcal{S},\, \mathcal{A},\, \mathcal{X},\, P,\, \rho,\, r,\, \Omega,\, \gamma \big\rangle .
$$

在时刻 $t$，状态 $s_t \in \mathcal{S}$ 包含两个*变长*的单位集合，$\mathcal{U}^A_t = \{u^A_1, \dots, u^A_{N_t}\}$ 和 $\mathcal{U}^B_t = \{u^B_1, \dots, u^B_{M_t}\}$。每个单位 $u$ 携带物理状态（位置、速度、航向、生命值）和一个**类型属性向量** $x_u \in \mathcal{X} \subset \mathbb{R}^{d_x}$（最大速度、交战半径、最大生命值、伤害、冷却时间、感知半径），因此异构性通过连续参数化而非离散类别来实现。

**种群动态。** 团队规模 $N_t, M_t$ 在回合内通过两种机制变化：*减员*——生命值归零的单位被移除——和*增援*——由标记点过程 $\rho$ 控制，在随机时刻插入批量新单位（具有采样的属性和进入位置）。双方均不控制或提前观测 $\rho$ 的时间表。这使得智能体的*索引集*本身成为随机动态的一部分——大多数 MARL 形式化通过假设排除了这一性质。

**观测与动作。** 每个单位 $u \in \mathcal{U}^A_t$ 接收 $o_u^t = \Omega(s_t, u)$：其感知半径内所有单位（双方）的实体特征向量集合，以 $u$ 的坐标系表示。其动作 $a_u = (a^{\text{man}}_u, a^{\text{tgt}}_u)$ 由机动（转向率/加速度）和对*当前可见*对方单位的目标选择或空目标组成——因此动作空间和观测一样随实体数量变化。团队接收共享的零和奖励 $r$（伤害差加终局结果），以 $\gamma$ 折扣。

**团队策略。** 团队策略是一个映射 $\pi_\theta$，对任何实现的种群，产生联合动作：$\pi_\theta : (o_{u_1}, \dots, o_{u_{N_t}}) \mapsto (a_{u_1}, \dots, a_{u_{N_t}})$。我们要求 $\pi_\theta$ 对**每一个** $N_t, M_t \in \mathbb{N}$ 以单一固定参数向量 $\theta$ 良定义——即*规模无关*要求——且对单位索引的置换等变，因为索引不携带意义。

*备注（信息结构）。* 我们允许 $\pi_\theta$ 依赖历史，具有周期广播约束：团队级模块可以每 $k$ 步计算并广播逐单位的条件信息，而在广播间隔内每个单位仅基于其局部观测和缓存的条件信息行动。这匹配了带宽受限的部署场景，是我们的方法（和固定 $K$ 的层次化基线）所实例化的信息结构；扁平基线是无状态逐步策略的特殊情况。

## 3.2 场景与对手分布

一个**场景** $\sigma = (N_0, M_0, \rho, \chi) \in \Sigma$ 固定初始团队规模、增援过程和单位类型的属性分布 $\chi$。一个**对手**是团队 $B$ 的团队策略 $\mu \in \mathcal{M}$。训练从 $(\sigma, \mu) \sim \mathcal{D}_{\text{train}} = \Sigma_{\text{train}} \times \mathcal{M}_{\text{train}}$ 中抽取。策略 $\theta$ 在配对 $(\sigma,\mu)$ 下的价值为

$$
V(\theta;\, \sigma, \mu) \;=\; \mathbb{E}\!\left[\textstyle\sum_t \gamma^t r_t \,\middle|\, \pi_\theta \text{ vs. } \mu,\ \sigma\right],
$$

我们主要报告导出的胜率 $W(\theta; \sigma, \mu)$。

## 3.3 双重零样本泛化

**定义 1（双重零样本泛化）。** 给定不相交的测试集 $\Sigma_{\text{test}} \cap \Sigma_{\text{train}} = \emptyset$ 和 $\mathcal{M}_{\text{test}} \cap \mathcal{M}_{\text{train}} = \emptyset$，$\theta$ 的*双重泛化矩阵*为

$$
\mathbf{G}(\theta)\big[\sigma, \mu\big] \;=\; W(\theta;\, \sigma, \mu), \qquad (\sigma, \mu) \in \big(\Sigma_{\text{train}} \cup \Sigma_{\text{test}}\big) \times \big(\mathcal{M}_{\text{train}} \cup \mathcal{M}_{\text{test}}\big),
$$

**不进行任何测试时参数更新**进行评估。块 $(\Sigma_{\text{train}}, \mathcal{M}_{\text{train}})$ 是分布内性能；单轴块 $(\Sigma_{\text{test}}, \mathcal{M}_{\text{train}})$ 和 $(\Sigma_{\text{train}}, \mathcal{M}_{\text{test}})$ 分别度量仅规模和仅对手泛化；联合块 $(\Sigma_{\text{test}}, \mathcal{M}_{\text{test}})$ 是我们的主要研究对象。

$\Sigma_{\text{test}}$ 包含三个分级族：(i) *内插/外推*——超出训练范围的初始规模（例如，训练 $N_0, M_0 \in [4,16]$，测试至 $64$），包括最高 $1{:}3$ 的非对称比例；(ii) *回合内动态*——速率和批量规模超出训练范围的增援过程，以及对抗性减员事件（瞬间损失团队随机一半）；(iii) *属性偏移*——对手单位属性外推至训练支撑之外（更快、更远程、更高生命值的类型）。

$\mathcal{M}_{\text{test}}$ 包含：(i) 保留的参数化脚本，包括退化型（被动、逃跑）和非常规型（佯攻-伏击、牺牲突击、分合式）风格；(ii) 独立训练的学习对手（不同算法和种子）；以及 (iii) **exploiter**：针对冻结的最终 $\theta$ 从头训练的最佳响应，其成功率上界策略的最坏情况脆弱性，作为可利用性的可行代理。

**定义 2（恢复时间）。** 对于时刻 $t_e$ 的回合内种群事件（任一方的增援或减员），定义瞬时*优势度* $D_t = \big(\sum_{u \in \mathcal{U}^A_t} hp_u - \sum_{v \in \mathcal{U}^B_t} hp_v\big) / \big(\sum_u hp_u + \sum_v hp_v\big) \in [-1, 1]$ 及其滑动窗口趋势 $\dot D_t$（窗口 $w$，两者在附录 D.2 中指定）。*恢复时间*为 $\min\{ \Delta \ge 0 : \dot D_{t_e + \Delta} \ge \dot D_{t_e^-} - \epsilon \}$——团队的轨迹恢复到事件前趋势的速度。由于该指标从可观测的物质状态计算得出，它是模型无关的且跨方法可比的；它将*适应速度*与原始强度隔离开来。

## 3.4 分组子问题

我们使宏观层面的决策结构显式化。指挥步骤 $k$ 处的**分组计划**是一个对 $(\mathcal{C}_k, g_k)$：*观测到的对手集合*的软划分 $\mathcal{C}_k$ 为 $K_k$ 个聚类（$K_k$ 是数据依赖的，不是固定的），以及一个分配 $g_k$ 将每个己方单位映射到从聚类派生的战术目标（交战、牵制、迂回）或预备目标。本文测试的假设现在可以精确表述：

> **对手锚定假设。** 令分组计划为对手观测到的结构和近期行为的函数，$(\mathcal{C}_k, g_k) = f_\theta(\mathcal{U}^B_{\le t}, \mathcal{U}^A_t)$，其中 $f_\theta$ 对两个参数都是置换不变且规模无关的。则在 $\mathcal{D}_{\text{train}}$ 上训练的单一 $\theta$ 比以下方法实现更高的双重泛化 $\mathbf{G}(\theta)$：(a) 具有可比容量和训练预算但没有分组结构的扁平策略，以及 (b) 以相同方式训练但分组锚定于己方团队且组数固定为 $K$ 的层次化策略——在对手结构与训练差异最大的测试单元中，差距最大。

主张 (a) 和 (b) 与第 5 节中的基线和消融实验一一对应。
