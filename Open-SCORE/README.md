# Open-SCORE：从小型子博弈到开放人口指挥

> **我们学习可跨人数复用、对策略差异敏感的小规模单目标攻防能力，再用可校准的局部博弈模型驱动双方滚动分组，检验其能否在伤亡、增援、未知对手和未见总规模下提高最坏全局胜率。**

```text
S1 变规模实体 QMIX + 团队 PSRO   → 可复用局部策略种群
S2 策略条件结局—时间预测 + 校准    → 可比较的局部博弈证据
S3 对手信念 + 双边列生成 + maximin → 受限动态分组指挥
S4 宏观价值残差 + 种群式守卫更新   → 结果对齐的共同适应
```

## 当前研究决定

- S1 训练支持域为红方不少于蓝方、双方各 1–4 人的 10 个规模；其余 6 格只作分布外压力测试，避免把物理劣势误写成已支持的“任意规模”。
- 一套参数共享的 DeepSets–GRU–QMIX 覆盖全部已注册规模；动作始终是 27 个量化加速度，不包含追敌、守点等宏动作。
- 课程先按 1v1→2v*→3v*→4v* 解锁，再依据 TD-error 学习进展采样，并保留 20% 均匀覆盖。
- 双方 warm-start 后运行团队级 PSRO。停止要求估计 NashConv、元博弈价值和 BR 训练预算同时满足，不用规则胜率冒充收敛证明。
- S3 定义为受限开放人口随机分配博弈；LP 只是单个宏观状态的阶段求解器，组合候选由双边列生成控制，长期性由宏观转移/终端价值提供。
- 未知对手的行为上下文必须改变 S3 的 ambiguity set；开集时回退保守 maximin，不能只输出一个对手标签。
- S4 先只做宏观价值残差；只有 held-out 最坏胜率改善后才追加新的 S1 最佳响应，不把全链路反传作为必要步骤。
- HAD 是第一验证平台；SMAClite-AD 是首选第二环境。Gigastep 仅建议在 WSL2 做大规模外部压力测试。

## 阅读入口

1. [研究结论与创新边界](docs/00_研究结论与阅读路线.md)
2. [问题定义](docs/01_问题定义与创新边界.md)
3. [相关工作、环境与源码审计](docs/02_相关工作与环境审计.md)
4. [论文式方法简版](docs/03_方法简版.md)
5. [数学模型与四阶段规格](docs/04_方法实现规格与伪代码.md)
6. [实验协议](docs/05_实验设计与预期证据.md)
7. [工作量与论文边界](docs/06_工作量_档位与拆稿.md)
8. [S1 代码、运行与 pilot 记录](docs/07_代码框架与第一阶段Demo.md)
9. [研究问题讨论与 S3/S4 可行性](docs/08_研究问题讨论与S3_S4可行性.md)
10. [论文框架](paper/论文框架.md)

## 运行

```powershell
python -m pip install -e .
python -m pytest -q
python scripts/smoke_stage1.py

# RTX 3060 上的短 2v2 PSRO 管线测试
conda run -n gpu_py_310 python scripts/train_stage1_psro.py `
  --mode pilot2v2 --device cuda --iterations 1 `
  --br-episodes 8 --batch-episodes 4 --payoff-episodes 1 --max-steps 40
```

正式 1–4 课程入口使用 `--mode curriculum`，但默认参数仍是开发配置；论文实验必须按配置扩大 BR 步数、payoff 局数和随机种子。

## 已实现与未实现

已实现 S1 的随机目标 HAD、变规模观测/mask、实体 QMIX、完整 episode replay、序列 Double-Q + TD(λ)、双边 rollout、规则对手、学习进展课程、团队 PSRO、meta-Nash、估计 NashConv、CLI、checkpoint API 和回归测试。S2–S4 的核心数据结构仍在，供后续开发。

尚未得到训练收敛或论文胜率结论；短 2v2 pilot 只证明环境—replay—双边 learner—payoff—PSRO 可闭环。SMAClite-AD、策略条件化 S2 数据集、S3 对手上下文/列生成在线闭环和 S4 outer loop 尚未实现。
