# COGAR 方法框架

COGAR 是暂定名：**Calibrated Outcome-Guided Adversarial Reallocation**。它不是一个四层大一统系统，而是一条紧凑方法链：冻结执行器产生可信干预数据，战果模型提供风险收益，双方 commander 进行滚动分兵。

## 1. 总体框架

~~~mermaid
flowchart LR
    A[阶段 1<br/>共享任务条件下层] --> B[克隆同一战场快照]
    B --> C[对多个完整双边编组<br/>做 paired intervention rollouts]
    C --> D[阶段 2 / EGOM<br/>成功-伤亡-耗时分布 + 不确定性]
    D --> E[阶段 3 / COGAR<br/>候选生成 + 风险收益矩阵 + 矩阵博弈]
    E --> F[执行 H 个低层步]
    F --> G{死亡 / 增援 / 关键点变化?}
    G -->|是或到期| B
    G -->|否| F
    F -. 可选阶段 4 .-> A
~~~

## 2. 阶段 1：冻结的通用下层

使用参数共享、实体集合编码和任务 token 的 MAPPO/IPPO：

\[
\pi_L(a_i\mid e_i,\{e_j^{\rm ally}\},\{e_k^{\rm enemy}\},x,m).
\]

训练分布随机化局部 1–4 vs 1–4、位置/血量、三类任务以及脚本与历史策略对手池。首稿的目标不是无限泛化，而是获得一个覆盖明确、可重复的执行器。所有上层 baseline 使用同一 checkpoint。

为了尽早隔离研究问题，P0 可先用 focus-fire、kite、guard 等强脚本替代学习下层；若连脚本下层上不同分兵都不改变结果，没必要先训练 MARL。

## 3. 阶段 2：Execution-Grounded Outcome Model（EGOM）

### 3.1 预测对象

EGOM 不是个体能力打分，也不是旧 commander 的 critic。给定当前快照和双方**完整候选分配**，它直接预测未来一个宏观时域 \(H\) 的结果：

\[
Y_H = \{z_j,\Delta n_j^R,\Delta n_j^B,T_j,d_j\}_{j=1}^{J},
\]

其中 \(z_j\) 是战线任务是否成功，\(\Delta n^R,\Delta n^B\) 是双方损失，\(T_j\) 是解决时间或截尾时间，\(d_j\) 是资产损伤/进度。若某些输出在 P0 不稳定，最低保留成功、双方损失和耗时。

模型条件为：

\[
p_\theta(Y_H\mid s,g^R,g^B,\nu_L,h^{-}),
\]

其中 \(\nu_L\) 是冻结下层版本，\(h^{-}\) 是可选的对手近期轨迹。若不使用 \(h^{-}\)，预测只能解释为对训练对手池的边缘结果，不能声称辨识了隐藏打法。

### 3.2 同态势干预数据

每条数据不是自然日志中的一个分配，而按以下协议生成：

1. 从真实 rollout 或覆盖采样器取得快照 \(s_q\)，保存环境与 RNG 状态；
2. 为同一快照生成 \(K\) 个可行我方分配和 \(K'\) 个敌方分配，形成候选对；
3. 对每个候选对从完全相同的 \(s_q\) 恢复仿真，保持该分配一个宏观时域；
4. 使用相同的 16–32 个随机种子集合展开各候选，即 common random numbers；
5. 聚合成功次数、伤亡与耗时样本，保留原始重复结果。

这是仿真中的显式 assignment intervention；文中不把它夸大成一般观察数据上的因果识别。

数据划分按“源快照 + 场景配置 + 对手策略”分组。同一快照的不同候选和重复种子绝不能跨 train/calibration/test，否则会造成严重泄漏。

### 3.3 网络和损失

最小网络：

1. 分别编码友军、敌军、资产、任务和分配标签；
2. 用 Deep Sets / Set Transformer 得到每条战线表示；
3. 用一个跨战线 attention block 输入完整 partition context，避免错误假设各联盟完全独立；
4. 输出成功概率、伤亡分位数/均值和截尾耗时；
5. 训练 3–5 个独立初始化的 ensemble。

损失可写为：

\[
\mathcal L = \lambda_z\mathcal L_{\rm binomial}
+\lambda_n\mathcal L_{\rm casualty}
+\lambda_T\mathcal L_{\rm time}
+\lambda_d\mathcal L_{\rm asset}.
\]

成功头用重复 rollout 的 Binomial/Beta-Binomial likelihood；连续结果首版用 Huber 或分位数损失。温度缩放只在独立 calibration split 上拟合。ensemble 方差作为认知不确定性近似，不能直接宣称 Bayesian 保证。

### 3.4 optimizer-aware 补数据

组合优化器会寻找模型系统性高估的编组。初次训练后做最多两轮数据聚合：

- 让当前 COGAR 选择高收益候选；
- 加入 ensemble 分歧最大、风险下界与均值排序差异最大的候选；
- 用真实仿真验证这些候选并回填；
- 重新训练、重新校准，并单独报告“被选中方案”的误差。

这一步是防止 optimizer's curse 的必要实验机制，不是无限在线学习。

## 4. 阶段 3：双边风险感知滚动分兵

### 4.1 候选生成

直接枚举 \(J^n\) 个个体分配不可行。三战线同质首版先枚举兵力计数向量，其数量为

\[
\binom{n+J-1}{J-1}.
\]

当 \(n\le 12,J=3\) 时最多 91 个计数向量。再用距离、血量和单位类型的 min-cost matching 把具体单位放入每条战线。异构/任务版用 beam search 保留每方 \(B=32\) 或 \(64\) 个可行完整分配，并保留 reserve 与上一次分配。

所有方法使用相同候选集，避免 proposed 因枚举更多而获益。

### 4.2 风险收益

对双方候选 \(a\in\mathcal G^R,b\in\mathcal G^B\)，EGOM 给出多维分布。红方短时域收益示意为：

\[
\hat U_\theta(s,a,b)=
w_z\sum_j \operatorname{LCB}_\alpha[Z_j]
-w_R\mathbb E[\Delta n_j^R]
+w_B\mathbb E[\Delta n_j^B]
-w_T\mathbb E[T_j]
-\lambda C_{\rm switch}(a,g_{t-1}^R)
-\beta\sigma_{\rm epi}(s,a,b).
\]

权重先由环境终局目标固定，不为 proposed 单独调参。LCB、switch cost 与 uncertainty penalty 分别控制危险高估、频繁换组和 OOD 外推。蓝方采用 \(-\hat U\) 的对称零和定义；非零和扩展不进入首稿。

### 4.3 求解上层博弈

对 \(B_R\times B_B\) 收益矩阵求一个熵正则化 maximin 分布，可用线性规划、mirror descent 或小规模 fictitious play。执行时按固定随机种子从混合策略采样一个分配，保持 \(H\) 个低层步。

若实验只控制一方，敌方候选由规则、历史 commander 和 learned commander 组成，求其最坏 \(k\) 个或 CVaR；不能假设敌方按本文收益合作。

矩阵求解不是贡献，必须与以下对照分离：手工收益 + 同一求解器、EGOM 均值 + greedy、未校准 scalar Q + 同一求解器、oracle rollout + 同一求解器。

### 4.4 事件触发滚动执行

默认每 \(H=10\) 或 \(20\) 个低层步重规划；死亡、增援和关键点状态变化提前触发。设置最短保持时间和 switch cost，避免事件连续发生时抖动。报告重分配次数和推理延迟，而不是只报胜率。

## 5. 对手未知时怎么办

首稿不另立“对手策略辨识”模块，采用两个等级：

- **基础版：**训练时对手池边缘化，EGOM 在不确定时给出更宽分布，COGAR 用风险下界；
- **增强版消融：**输入最近 \(L=8\) 个宏观/微观摘要（位置变化、集火、退避、分兵变化），比较是否改善未见对手校准。

只有增强版稳定改善最终结果，才在论文中称为 history-conditioned outcome prediction；不能用对手类型分类准确率代替任务证据。

## 6. 阶段 4：只做交替共同适应消融

普通联合训练已经被 ALMA、C2、DT-GAT 等覆盖。若资源允许，只测试：

1. 固定 EGOM 和 commander，低学习率微调下层；
2. 用 KL 锚定预训练策略并混入旧任务回放；
3. 冻结更新后的下层，重新采集/校准 EGOM；
4. 再更新 commander，一共 1–2 轮。

离散分配处停止梯度，不让上层直接利用可微代理漏洞。若 OOD 收益提升不足 5 个百分点、ID 下降超过 2 个百分点或 ECE 明显变差，从主文删除。

## 7. 训练与推理伪代码

~~~text
Pretrain and freeze task-conditioned lower policy pi_L

for snapshot s in coverage sampler:
    candidates_R, candidates_B = generate_feasible_allocations(s)
    for selected pair (g_R, g_B):
        for common seed omega in Omega:
            restore(s, omega)
            hold (g_R, g_B) for H low-level steps under pi_L
            record success, casualties, time, asset progress

train EGOM ensemble on grouped intervention data
calibrate success probabilities on a disjoint calibration split

for macro step t:
    generate the same-budget candidate sets for both sides
    evaluate all candidate pairs with EGOM
    build risk- and switch-aware payoff matrix
    solve regularized matrix game and execute one allocation for H steps
    replan early on death, reinforcement, or asset-state events

repeat at most twice with optimizer-selected high-risk candidates added to data
~~~

## 8. 复杂度与实时门槛

EGOM 对 \(B_RB_B\) 个候选对批处理。若 \(B_R=B_B=32\)，是 1,024 个轻量集合前向；若延迟过高先减少 beam 或只评估 top-k 对手响应，而不是牺牲校准。矩阵博弈本身规模很小。

验收门槛：一次上层决策的 p95 延迟低于一个低层控制周期的 10%，并给出效果—延迟—重编频率 Pareto。超过门槛时只能称离线/低频规划，不能称实时。

## 9. 真正贡献的因果消融

必须保持 encoder、参数量、候选集、下层和训练预算一致，比较：

- full COGAR；
- no outcome model / direct commander；
- shuffled EGOM；
- uncalibrated scalar-Q；
- calibrated mean without uncertainty；
- no paired interventions（只用自然日志）；
- local-only without full partition context；
- oracle Monte Carlo EGOM。

只有 full 相对容量匹配对照提高最终任务结果，且误差下降出现在上层真正选中的方案上，才能支持方法链。
