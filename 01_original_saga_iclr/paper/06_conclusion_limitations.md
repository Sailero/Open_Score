<!-- 状态: 完整可用初稿 | 结论段数字占位待回填 -->

# 6. Conclusion, Limitations, and Broader Impact

## 6.1 Conclusion

We formalized dual zero-shot generalization — joint transfer across team sizes (including in-episode population dynamics) and unseen opponent strategies — as the central deployment gap in two-team competitive MARL, and argued that the two axes collapse into one representation problem: grouping decisions must be anchored to the opponent's observed real-time structure rather than to fixed indices, fixed group counts, or fixed opponent models. SAGA instantiates this principle with a size-agnostic entity encoder, an opponent-anchored commander whose cluster structure and style conditioning are computed from observations, and a shared goal-conditioned micro policy, trained under a joint scale-by-style autocurriculum with a lightweight, academically reproducible league. Trained only at 4–16 units per team, a single SAGA policy transfers zero-shot to [64+]-unit engagements, absorbs mid-episode reinforcements and attrition, and withstands held-out styles and freshly trained exploiters, with ablations tracing each generalization axis to the mechanism designed for it. We release SwarmArena and the dual-generalization protocol as a benchmark for this joint setting.

## 6.2 Limitations

1. **Kinematic abstraction.** SwarmArena uses point-mass kinematics in the plane; high-fidelity vehicle dynamics, sensing noise, and communication dropout beyond periodic broadcast are not modeled. Our claims concern decision structure, not platform realism.
2. **Exploitability proxy.** Exploiter training provides a lower-bound probe of worst-case vulnerability, not an exploitability certificate; full game-theoretic guarantees (e.g., PSRO-style convergence) are beyond the paper's compute regime.
3. **Attribute extrapolation has limits.** Zero-shot robustness to opponent unit types degrades beyond moderate attribute shift (Section 5.5); semantic-level adaptation — e.g., recognizing that a new unit class demands a qualitatively different doctrine, possibly with foundation-model priors — is deliberately left to future work.
4. **Commander period is fixed.** The two-timescale split uses a fixed broadcast period $k$; learning adaptive invocation (event-triggered re-planning) is a natural extension.
5. **Two teams, zero-sum.** Multi-party engagements and general-sum objectives (escort, coverage under adversarial interference) are not covered.

## 6.3 Deployment Readiness

Beyond benchmark performance, the dual zero-shot property is precisely the contract that real host systems — game AI services, simulation-based training platforms, multi-robot testbeds — impose on team policies: entity counts vary per match and within matches, opponent styles span the full competence spectrum including deliberately exploitative play, and online retraining after release is typically prohibited. SAGA's runtime interface requires only an entity feature list (the native world representation of game engines and robotics middleware) and a two-rate tick loop matching standard server architectures; inference is a single [~1M]-parameter forward pass with cost linear in entity count — measured at [12] ms on CPU for 64-vs-64 and [65] ms for 256-vs-256 on commodity hardware, within the AI-tick budget of typical hosts (Appendix H). The commander's cluster and task assignments are explicit intermediate quantities, providing an auditable trace of macro decisions, and the script league doubles as a certified fallback controller. We outline a staged path — released checkpoints with a versus-API, integration into public game-AI arenas with third-party opponents, and a command-layer-only SDK for host-native execution — in Appendix H.

## 6.4 Broader Impact and Ethics Statement

This work studies coordination and robustness in abstract adversarial swarm games. The techniques are dual-use: analogous decision problems arise in civilian domains (competitive logistics, autonomous traffic negotiation, robot sports, epidemic containment as adversarial control) and in defense applications. We deliberately evaluate in abstracted point-mass environments, release no hardware-specific components, and focus scientific claims on representation and generalization principles. We believe open publication of robustness principles — particularly *why* policies overfit to opponents and scales, and how to measure it — supports defensive evaluation and red-teaming of deployed multi-agent systems, which outweighs the marginal capability uplift of an abstract benchmark. We encourage downstream users to observe applicable regulations and institutional review for any embodied deployment.

## 6.5 Reproducibility Statement

All environments (SwarmArena, adapters to MPE and MAgent2), the SAGA implementation, league training pipeline, evaluation matrices, and analysis scripts will be released under the MIT license with exact configuration files and seeds. SwarmArena is additionally exposed through the standard PettingZoo ParallelEnv interface and **passes the official PettingZoo parallel API test**, with verified deterministic seeded rollouts (identical seeds reproduce identical trajectories); researchers can therefore use it with existing community tooling independently of our training code. Every reported number specifies its $(\sigma, \mu)$ evaluation cell; the protocol in Section 3.3 is deterministic given seeds. [代码仓库链接占位]

---

# 中文翻译

# 6. 结论、局限性与社会影响

## 6.1 结论

我们将双重零样本泛化——在团队规模（包括回合内种群动态）和未见对手策略上的联合迁移——形式化为双队竞争 MARL 中的核心部署差距，并论证了两个轴坍缩为一个表示问题：分组决策必须锚定于对手观测到的实时结构，而非固定索引、固定组数或固定对手模型。SAGA 通过规模无关的实体编码器、对手锚定的指挥器（其聚类结构和风格条件从观测计算得出）以及共享的目标条件微观策略来实例化这一原则，在联合规模-风格自动课程下配合轻量级的、学术可复现的联赛进行训练。仅在每队 4–16 个单位上训练，单一 SAGA 策略零样本迁移至 [64+] 单位的对局，吸收回合内增援和减员，并抵御保留风格和新训练的 exploiter，消融实验将每个泛化轴追溯到为其设计的机制。我们发布 SwarmArena 和双重泛化协议作为该联合设定的基准。

## 6.2 局限性

1. **运动学抽象。** SwarmArena 使用平面上的质点运动学；高保真的载具动力学、感知噪声和超出周期广播的通信中断未被建模。我们的主张关注决策结构，而非平台真实性。
2. **可利用性代理。** Exploiter 训练提供了最坏情况脆弱性的下界探测，而非可利用性证书；完整的博弈论保证（例如 PSRO 式收敛）超出了本文的计算范围。
3. **属性外推有其限制。** 对对手单位类型的零样本鲁棒性在中等属性偏移之后退化（第 5.5 节）；语义级适应——例如识别一种新的单位类别需要质变的战术，可能需要基础模型先验——被有意留给未来工作。
4. **指挥周期是固定的。** 双时间尺度分解使用固定的广播周期 $k$；学习自适应调用（事件触发的重规划）是一个自然的扩展。
5. **两队，零和。** 多方对局和一般和目标（护航、对抗干扰下的覆盖）未被涵盖。

## 6.3 部署就绪度

超越基准性能，双重零样本特性恰恰是真实宿主系统——游戏 AI 服务、基于仿真的训练平台、多机器人测试台——对团队策略施加的契约：实体数量每场比赛不同且在比赛内变化，对手风格覆盖从退化到刻意利用的全部能力谱系，且发布后通常禁止在线再训练。SAGA 的运行时接口仅需一个实体特征列表（游戏引擎和机器人中间件的原生世界表示）和一个双速率 tick 循环，匹配标准服务器架构；推理是一次约 [~1M] 参数的前向传播，成本与实体数量呈线性——在普通硬件上 64v64 CPU 测得 [12] ms，256v256 测得 [65] ms，在典型宿主的 AI tick 预算内（附录 H）。指挥器的聚类和任务分配是显式的中间量，提供了宏观决策的可审计追踪，脚本联赛可兼作认证的后备控制器。我们在附录 H 中概述了一条分阶段的路径——发布带对战 API 的检查点、集成到具有第三方对手的公共游戏 AI 竞技场、以及仅指挥层的 SDK 用于宿主原生执行。

## 6.4 社会影响与伦理声明

本工作研究抽象对抗性集群博弈中的协调与鲁棒性。这些技术具有双重用途：类似的决策问题出现在民用领域（竞争性物流、自动驾驶交通协商、机器人运动、作为对抗控制的流行病遏制）和国防应用中。我们有意在抽象的质点环境中评估，不发布任何特定硬件组件，并将科学主张集中在表示和泛化原理上。我们相信公开发表鲁棒性原则——特别是*为什么*策略会对对手和规模过拟合，以及如何度量它——支持已部署多智能体系统的防御性评估和红队测试，其收益超过了一个抽象基准的边际能力提升。我们鼓励下游用户遵守适用法规和机构审查，以进行任何实体化部署。

## 6.5 可复现性声明

所有环境（SwarmArena、对 MPE 和 MAgent2 的适配器）、SAGA 实现、联赛训练流水线、评估矩阵和分析脚本将在 MIT 许可证下发布，附带精确的配置文件和种子。SwarmArena 还通过标准的 PettingZoo ParallelEnv 接口暴露，并**通过了官方 PettingZoo parallel API 测试**，经过验证的确定性种子回放（相同种子产生完全相同的轨迹）；因此研究人员可以使用现有社区工具独立于我们的训练代码来使用它。每个报告的数字都指定了其 $(\sigma, \mu)$ 评估单元格；第 3.3 节中的协议在给定种子下是确定性的。[代码仓库链接占位]
