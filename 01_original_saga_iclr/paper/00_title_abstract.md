<!-- 状态: 完整可用初稿 | 摘要中带 [数字占位] 的部分待实验数据回填 -->
<!-- 术语约定: 全文使用中性术语 (adversarial swarm games / pursuit & attrition), 环境命名 SwarmArena, 避免军事措辞 -->

# Title

**Dual Zero-Shot Generalization in Adversarial Swarm Games via Opponent-Anchored Dynamic Grouping**

备选标题（按保守程度排序）：
1. *Opponent-Anchored Dynamic Grouping for Scale- and Opponent-Agnostic Multi-Agent Competition*
2. *One Policy for Any Swarm: Joint Generalization over Team Sizes and Opponent Strategies in Adversarial Multi-Agent Games*
3. *SAGA: Scale-Agnostic Grouping and Adaptation for Adversarial Swarm Games*

---

# Abstract

Teams of learning agents deployed in adversarial swarm games face two distribution shifts that existing multi-agent reinforcement learning (MARL) methods handle poorly: the *sizes* of both teams at test time — including mid-episode reinforcements and attrition — rarely match those seen during training, and the *opponent's strategy* is never the one trained against. Prior work addresses each axis in isolation: permutation-invariant architectures generalize across population sizes but are studied almost exclusively in cooperative tasks, while population-based self-play hardens policies against unseen opponents but at fixed, small scales. We argue that these are two faces of a single representation problem, and that both generalizations emerge jointly when a team's *grouping structure* — how it partitions itself into task groups — is anchored to the opponent's real-time composition rather than to fixed agent indices, a fixed number of groups, or a fixed opponent model. We instantiate this principle in SAGA, a hierarchical policy consisting of (i) a permutation-invariant entity encoder whose parameters are independent of both team sizes, (ii) an opponent-anchored commander that softly clusters observed opponents into an adaptive number of groups and assigns per-agent tactical goals conditioned on an in-context embedding of the opponent's style, and (iii) a shared goal-conditioned low-level policy. SAGA is trained with a joint two-dimensional autocurriculum spanning team-size configurations and a lightweight opponent league of parameterized scripts, historical snapshots, and periodically trained exploiters. Trained only on engagements with 4–16 agents per team against the training league, a single SAGA policy transfers zero-shot to [64+]-agent engagements, asymmetric force ratios, mid-episode reinforcements, and held-out opponent styles including freshly trained exploiters, improving win rate over the strongest scalable baseline by [XX] percentage points on the joint scale-by-opponent generalization matrix. Analyses show that the learned grouping mirrors the opponent's spatial organization and that the opponent-style embedding separates unseen strategies without supervision.

**Keywords**: multi-agent reinforcement learning, zero-shot generalization, self-play, hierarchical policies, dynamic grouping, adversarial games

---

## 摘要逻辑链（写作自检）

1. 部署时的两大分布移：规模（含 episode 内增减员）与对手策略 —— 痛点普适且具体；
2. 现状：两轴各自有成熟解，但从未统一 —— 空白；
3. 核心主张（一句话创新）：分组结构锚定对手实时组成 → 双重泛化同时涌现 —— 可证伪；
4. 方法三件套 + 二维课程 —— 实现路径；
5. 结果量化承诺（占位）—— 训练 4–16 → 零样本 64+/增援/未见对手/exploiter；
6. 机制证据：簇-敌结构对齐 + 风格嵌入无监督可分。

---

# 中文翻译

# 标题

**通过对手锚定动态分组实现对抗性集群博弈中的双重零样本泛化**

备选标题（按保守程度排序）：
1. *面向规模与对手无关的多智能体竞争的对手锚定动态分组*
2. *一个策略适用于任何集群：对抗性多智能体博弈中团队规模与对手策略的联合泛化*
3. *SAGA：用于对抗性集群博弈的规模无关分组与自适应方法*

---

# 摘要

在对抗性集群博弈中部署的学习智能体团队面临两种分布偏移，而现有的多智能体强化学习（MARL）方法难以同时应对：测试时双方团队的*规模*——包括回合内的增援和减员——很少与训练时一致，且*对手的策略*永远不是训练时所对抗的那个。先前的工作分别在每个轴上进行处理：置换不变架构可以跨种群规模泛化，但几乎仅在合作任务中得到研究；基于种群的自博弈可以使策略对未见对手更加鲁棒，但仅限于固定的小规模场景。我们认为这是同一个表示问题的两个方面，当团队的*分组结构*——即如何将自身划分为任务组——锚定于对手的实时组成，而非固定的智能体索引、固定的组数或固定的对手模型时，两种泛化能力将同时涌现。我们将这一原则实例化为 SAGA，一个层次化策略，包含：(i) 一个置换不变的实体编码器，其参数与双方团队规模无关；(ii) 一个对手锚定的指挥器，通过迭代 slot attention 将观测到的对手软聚类为自适应数量的组，将对手的近期行为嵌入为上下文风格向量，并通过交叉注意力匹配为每个智能体分配战术目标；(iii) 一个共享的目标条件低层策略。SAGA 通过联合的二维自动课程进行训练，该课程跨越团队规模配置和一个轻量级对手联赛（包括参数化脚本、历史快照和定期训练的 exploiter）。仅在每队 4–16 个智能体的对局中训练，单个 SAGA 策略可零样本迁移至 [64+] 个智能体的对局、非对称兵力比、回合内增援，以及包括新训练的 exploiter 在内的未见对手风格，在联合的规模-对手泛化矩阵上相比最强可扩展基线提高了 [XX] 个百分点的胜率。分析表明，学到的分组结构映射了对手的空间组织，且对手风格嵌入在无监督条件下可分离未见策略。

**关键词**：多智能体强化学习、零样本泛化、自博弈、层次化策略、动态分组、对抗性博弈
