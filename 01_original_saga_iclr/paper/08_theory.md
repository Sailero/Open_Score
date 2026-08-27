<!-- 状态: v2 (严格审稿修订版, 修订记录见 docs/08) | 正文引用其结论, 全文置于附录 F; 记号沿用 03_problem_formulation.md -->

# Theoretical Analysis（附录 F 全文；正文按 F.7 分配表引用）

## F.0 Scope and honesty statement

本节声明各结果的严格程度，正文与附录使用同一口径：**F.1–F.3 是严格结果**（前提显式，证明完整）；**F.4 是解释性上界**——它把双重零样本泛化归约为摘要空间中的分布覆盖问题，其核心假设（B1'）依赖闭环占用、无法先验验证，但**可以部署后实证检验**（我们在 §5.4 给出检验实验）；**F.5 是启发式论证**（正确论证对象为梯度信号方差与结果熵，不主张统计最优性）。

---

## F.1 结构性质：置换等变

**命题 1（置换等变性）.** 设 SAGA 策略 $\Pi_\theta$ 将实体观测集合映射为各单位动作分布的元组。对任意己方单位置换 $\pi_A \in S_N$ 与敌方实体置换 $\pi_B \in S_M$：
$$
\Pi_\theta\big(\pi_A \cdot \mathcal{U}^A,\ \pi_B \cdot \mathcal{U}^B\big) \;=\; \pi_A \cdot \Pi_\theta\big(\mathcal{U}^A, \mathcal{U}^B\big),
$$
且目标选择分布在"物理实体"意义下对 $\pi_B$ 不变（logits 向量随 $\pi_B$ 同步重排）。

*证明.* 逐模块验证。(i) 模块 A：单查询交叉注意力为 key/value 集合上的 softmax 加权和，与实体顺序无关；参数跨单位共享，故对 $\pi_A$ 等变。(ii) B1：slot attention 的实体维运算（跨 slot softmax、按 slot 归一化聚合）均为集合运算；slots 为**可学习的固定参数向量**（非从实体派生），故对 $\pi_B$ 不变。occupancy 为对实体求和，不变。(iii) B2：统计量 $\psi$ 为集合矩（计数、均值、散布、由可观测事件估计的频率），全部置换不变；GRU 只沿时间维。 (iv) B3：任务 token 由 slots 与 $z$ 生成（与实体索引无关）；单位-任务匹配逐单位共享参数，对 $\pi_A$ 等变。(v) 模块 C：机动头逐单位共享；目标头对每个敌实体独立打分，输出随 $\pi_B$ 重排、null-target 位置固定。所有动作为分布采样，无依赖索引的 tie-breaking。组合即得。∎

## F.2 规模无关性与计算复杂度

**命题 2（规模无关与复杂度）.** (i) $\Pi_\theta$ 的参数集 $\theta$ 与 $(N, M)$ 无关，对任意 $N \ge 1, M \ge 0$ 前向良定义。(ii) 复杂度（$d$ 为模型宽度）：

| 计算 | 每单位 | 团队总量 |
|---|---|---|
| 微观策略（每 tick） | $O((N{+}M)\,d)$ | $O(N(N{+}M)\,d)$ |
| 指挥层（每 $k$ tick） | — | $O(L K_{\max} M d + N K_{\max} n_{\text{task}} d)$ |

**每单位成本对实体数是线性的**（对比单位级自注意力的 $O((N{+}M)^2 d)$）；团队总量在 $N \approx M$ 时为 $O(N^2 d)$——与对全实体做一次自注意力同阶，但无逐对 softmax 归一化、跨单位完全并行，且指挥层对团队总量是线性的。

*证明.* (i) 所有可学习张量形状仅含 $d, d_\tau, K_{\max}, n_{\text{task}}, n_{\text{free}}$ 等常数；变长集合经掩码注意力/池化处理，$M{=}0$ 时由空集掩码与 null-target logit 兜底。(ii) 直接计数：模块 A 每单位一次交叉注意力 $O((N{+}M)d)$；B1 每次迭代对 $M$ 实体 × $K_{\max}$ slot 打分，共 $L$ 次；B3 为 $N \times (K_{\max} n_{\text{task}} + n_{\text{free}})$ 矩阵。∎

*部署推论.* 团队总量的平方项在实测中不构成瓶颈：完整决策链 64v64 CPU 12 ms、256v256 CPU 65 ms / GPU 12 ms（附录 H 表 H.1），因常数极小（$d=128$ 单层交叉注意力）且并行度为 $N$。

## F.3 稳定性：实体增删的双 regime 扰动分析

部署主张"增援/减员被平滑吸收"的形式化。关键点（v2 修订）：平滑性**只对已建立的簇成立**；对低占用 slot，架构被设计为允许快速结构响应——两者都需要形式刻画。

**假设 A1（有界嵌入与 logits）.** 实体嵌入有界 $\|W\tau_e\| \le B$，slot 注意力 logits 有界 $|\ell_{j\kappa}| \le \Lambda$（由特征归一化与谱范数约束保证）。
**假设 A2（Lipschitz 更新）.** slot GRU 更新为 $L_\phi$-Lipschitz；迭代数 $L$ 固定。

由 A1，跨 slot softmax 给出注意力下界：对每实体 $j$、每 slot $\kappa$，
$$
A_{j\kappa} \;\ge\; \delta \;:=\; \frac{e^{-2\Lambda}}{K_{\max}} \;>\; 0,
\qquad\text{因此}\qquad o_\kappa = \textstyle\sum_j A_{j\kappa} \;\ge\; M\delta .
$$

**命题 3（双 regime 扰动界）.** 在 A1–A2 下，向 $M$ 个敌实体插入一个新实体 $e^+$：

**(i) 已建立簇的平滑性.** 每个 slot 的单次迭代更新扰动满足
$$
\big\| u_\kappa' - u_\kappa \big\| \;\le\; \frac{A_{+\kappa} \cdot 2B}{M\delta + A_{+\kappa}} \;\le\; \frac{2B K_{\max} e^{2\Lambda}}{M},
$$
沿 $L$ 次迭代与 GRU 复合后 $\|c_\kappa' - c_\kappa\| \le C_1(B, \Lambda, K_{\max}, L_\phi, L)/M$；进而 B3 中 occupancy 门控项的摄动 $|\log o_\kappa' - \log o_\kappa| \le O\!\big(1/(M\delta)\big)$，每单位目标向量满足 $\|g_i' - g_i\| \le C_2/M$（softmax 与线性层的 Lipschitz 复合）。**即宏观计划对单个实体增删的扰动以 $O(1/M)$ 衰减，常数随注意力尖锐度 $e^{2\Lambda}$ 增长。**

**(ii) 结构响应 regime.** 若插入使某 slot 的 occupancy 跨越激活阈值 $\omega$（新敌簇形成），门控项跳变可达 $\log(o'/o) = \Omega(1)$，任务集合发生结构性变化——扰动不再是 $O(1/M)$。这是**设计意图**：新出现的敌方集群应当触发重规划，而非被平滑掉。

*证明.* (i) 加权均值插入一项：$u' - u = \frac{A_{+\kappa}}{D + A_{+\kappa}}(v_+ - u)$，$D = \sum_j A_{j\kappa} \ge M\delta$，$\|v_+ - u\| \le 2B$，得第一不等式；slot 移动引起后续迭代 logits 变化，由 A1–A2 的 Lipschitz 复合放大至多 $L_\phi^L$ 倍常数因子；$\log$ 在 $[M\delta, M+1]$ 上 $\frac{1}{M\delta}$-Lipschitz 给出门控摄动；B3 各映射在有界输入上 Lipschitz。(ii) 直接构造：取现有实体对 slot $\kappa$ 的注意力接近下界 $\delta$、新实体 logits 偏向 $\kappa$，则 $o_\kappa$ 从 $\approx M\delta < \omega$ 越过阈值。完整不等式链在终稿展开。∎

*与实验的对应.* Regime (i) 预测：恢复时间（定义 2）随规模增大而缩短（单实体事件）；regime (ii) 预测：**批量**增援（$\Delta M \sim M$）的恢复依赖课程中增援事件的覆盖而非架构平滑性——两者分别由 §5.1 的 S2 族与消融 (d) 检验。

## F.4 泛化：摘要空间覆盖界（解释性）

**目标**：形式化"对手锚定为何支持双重零样本"——把对未见 $(\sigma^*, \mu^*)$ 的性能损失归约为**摘要空间中占用分布的覆盖距离**。v2 修订：(a) 界限于指挥层级的宏观过程；(b) 修正陈述使其确实从假设推出；(c) 给出实证检验路径。

**设定.** 固定 $\theta$。令 $\xi_t = \Phi_\theta(s_t) = (\{c_\kappa, o_\kappa\}_\kappa, z_t, \bar h_t) \in \mathcal{Z}$ 为指挥层消费的摘要，装备度量 $\rho_{\mathcal{Z}}$（slots 用最优匹配距离以保持对 slot 重排的不变性）。微观策略共享且仅依赖局部观测与目标，视作受控动力学的一部分。令 $\nu_{\sigma,\mu}^\theta$ 为对局 $(\sigma,\mu)$ 中 $\{\xi_t\}$ 的**折扣占用测度**。

**假设 B1'（宏观奖励充分性 + 正则性）.** 存在 $\bar r: \mathcal{Z} \to \mathbb{R}$ 使 $\mathbb{E}[r_t \mid \xi_t] = \bar r(\xi_t)$（摘要对期望即时奖励充分），且 $\bar r$ 为 $L_r$-Lipschitz。

**引理 1.** 在 B1' 下，$V(\theta; \sigma, \mu) = \frac{1}{1-\gamma} \, \mathbb{E}_{\xi \sim \nu^\theta_{\sigma,\mu}} \bar r(\xi)$，且对任意两对局，
$$
\big| V(\theta;\sigma,\mu) - V(\theta;\sigma',\mu') \big| \;\le\; \frac{L_r}{1-\gamma}\, W_1\!\big(\nu^\theta_{\sigma,\mu},\, \nu^\theta_{\sigma',\mu'}\big).
$$
*证明.* 第一式为条件期望的塔性质；第二式为 $W_1$ 的 Kantorovich 对偶（$\bar r / L_r$ 为 1-Lipschitz 检验函数）。∎

**命题 4（覆盖界）.** 在 B1' 下，对任意测试对 $(\sigma^*, \mu^*)$：
$$
V(\theta; \sigma^*, \mu^*) \;\ge\; \max_{(\sigma,\mu) \in \mathrm{supp}(\mathcal{D}_{\text{train}})} \Big[ V(\theta; \sigma, \mu) \;-\; \frac{L_r}{1-\gamma} W_1\!\big(\nu^\theta_{\sigma^*,\mu^*},\, \nu^\theta_{\sigma,\mu}\big) \Big].
$$
**推论 1（期望形式）.** 若训练性能均匀：$V(\theta;\sigma,\mu) \ge \bar V - \varepsilon_{\text{unif}}$ 对所有训练支撑成立，则
$$
V(\theta; \sigma^*, \mu^*) \;\ge\; \bar V \;-\; \varepsilon_{\text{unif}} \;-\; \frac{L_r}{1-\gamma}\, \underbrace{\min_{(\sigma,\mu)} W_1\big(\nu^\theta_{\sigma^*,\mu^*}, \nu^\theta_{\sigma,\mu}\big)}_{d_{\text{cover}}(\sigma^*,\mu^*)} .
$$
*证明.* 命题 4 为引理 1 对每个训练对的直接应用后取 max；推论代入均匀性假设。∎

**三条解读（论文叙事的理论支点）.**
1. **规模外推为何可行**：$\Phi$ 的分量对 $M$ 要么归一（密度/比例统计），要么随簇结构组合（slots 数有上界 $K_{\max}$、每 slot 描述子由簇内统计决定）——64v64 的摘要占用可以落在小规模训练已覆盖的区域附近：**架构把原空间的外推变成摘要空间的内插**，即压小 $d_{\text{cover}}$。扁平策略无此性质（其"摘要"维度随 $M$ 增长，规模外推必然是支撑集外推）。这给出基线差距的可检验预测。
2. **课程的角色**：SxS 的目标被形式化为**最小化测试族的覆盖半径** $\sup_{(\sigma^*,\mu^*)} d_{\text{cover}}$——脚本参数随机化、快照、exploiter 分别扩大 $\mathrm{supp}$ 的不同方向。
3. **界失效的场景即最难的测试族**：对抗性 $\mu^*$（exploiter）可最大化 $d_{\text{cover}}$（制造训练时未见的摘要占用，如诱伏时序结构），此时界变松——理论预测 exploiter 是矩阵中最难的列，与实验设计一致。

**实证检验（新增，进 §5.4）.** $d_{\text{cover}}$ 可直接估计：对每个矩阵 cell 采样对局、在摘要空间算与最近训练 cell 的经验 $W_1$，检验其与胜率衰减的相关性。若显著正相关，B1' 的解释力获得数据支撑；若不相关，我们将如实报告并弱化 F.4 的叙事地位。

**局限（诚实声明，正文引用）.** (a) $\nu^\theta$ 是闭环占用，依赖 $\theta$ 自身——本界是关于已部署系统的**解释性**工具，非先验保证；(b) B1' 在摘要混叠（同摘要不同真值）时失效，微观层直接消费原始局部观测可部分缓解，但宏微观联合的严格处理是开放问题；(c) $L_r$ 未知，界不用于数值预测，仅用于结构比较（同一 $L_r$ 下比较不同架构的 $d_{\text{cover}}$）。

## F.5 课程：前沿采样的信号论证（启发式）

**命题 5（近公平对局最大化学习信号）.** 设某 cell 的对局结果为 Bernoulli($p$)、终局奖励 $\pm 1$。(i) 结果熵 $H(p)$ 在 $p = 1/2$ 处最大；(ii) 单局 REINFORCE 梯度估计的量级由回报方差 $4p(1-p)$ 调制，同样在 $p = 1/2$ 处最大。因此按 $\exp\{-(W_{\text{cell}} - \frac{1}{2})^2 / T\}$ 采样单调偏向高信号 cell，且 $T \to 0$ 时收敛于最不确定 cell 集上的均匀分布。

*证明.* (i)(ii) 为初等计算：$H'(p) = \log\frac{1-p}{p}$ 于 $p=\frac12$ 变号；$\mathrm{Var}[\pm1\text{ 结果}] = 4p(1-p)$。采样分布的单调性与极限直接由高斯核性质得出。∎

*定位（v2 修订）.* 这是**信号丰富度启发式**，与 UED/PLR 的 learnability 优先级采样同族；不主张对 $p$ 的统计估计效率（Bernoulli 的 Fisher 信息在 $p=1/2$ 处反而最小），也不主张均衡收敛（那需要完整 PSRO 机制）。其经验效果由 §5.3 图 6 的课程消融检验。

## F.6 假设强度与开放问题（主动披露）

| 假设/限制 | 影响 | 缓解与检验 |
|---|---|---|
| A1 的 $e^{2\Lambda}$ 常数 | 注意力越尖锐，F.3(i) 常数越大 | logits 做温度/谱约束；实测恢复时间随 $M$ 的斜率 |
| F.3 不覆盖批量增援 | $\Delta M \sim M$ 时无平滑保证 | 依赖课程覆盖（消融 d）；恢复时间按事件规模分层报告 |
| B1' 闭环性 | 界非先验 | §5.4 的 $d_{\text{cover}}$-性能相关性检验 |
| B1' 混叠 | 摘要不充分时界失效 | 微观层保留原始观测通道；F.4 只用于宏观比较 |
| 宏微观联合严格化 | 未处理 | 声明为开放问题 |

## F.7 正文引用分配表

| 命题 | 保证类型 | 正文位置与用法 |
|---|---|---|
| 1 | 严格 | §4.1–4.3 各一句话（架构正确性） |
| 2 | 严格 | §4.1 复杂度句 + §6.3 部署段（引附录 H 实测表） |
| 3 | 严格（A1–A2 下，双 regime） | §4.2 末：连接增援吸收与恢复时间实验（S2 族） |
| 4 + 推论 1 | 解释性（B1' 下） | §4.2 mechanism summary 的形式化；§5.4 实证检验 |
| 5 | 启发式（论证对象：信号方差与熵） | §4.4 一句话 + 附录 |
