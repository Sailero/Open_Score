# Open-SCORE：开放规模攻防的四阶段能力—指挥框架

> **一句话：本文研究如何在攻防双方因战损和增援而持续改变规模时，通过可变规模执行、能力评估、动态重编组和上下层协同学习，维持多目标防御的总体胜率。**

本目录同时包含研究规格和最小可运行代码骨架。四阶段作为一个统一系统贡献：

```text
S1 变规模 QMIX 执行
   → S2 双边编组结果评估
   → S3 人口事件后的稳健重编组
   → S4 带规模守卫的上下层交替优化
   └──────────────────────→ 刷新 S1 能力版本
```

每个阶段只保留一个核心机制，不再增加新聚类器、PSRO、LLM、对手辨识网络或可微优化器。QMIX、REFIL、ALMA、Set Transformer 和 HARL 都有官方源码并作为实现基础或 baseline；真正需要验证的创新是四个接口能否共同解决双方人口变化下的多目标攻防，而不是把已有组件分别宣称为新算法。

## 阅读顺序

1. [研究结论与阅读路线](docs/00_研究结论与阅读路线.md)
2. [问题定义与创新边界](docs/01_问题定义与创新边界.md)
3. [相关工作与环境审计](docs/02_相关工作与环境审计.md)
4. [四阶段方法简版](docs/03_方法简版.md)
5. [详细技术方案与核心代码](docs/04_方法实现规格与伪代码.md)
6. [实验设计与验收门槛](docs/05_实验设计与预期证据.md)
7. [工作量、档位与拆分](docs/06_工作量_档位与拆稿.md)
8. [代码框架与第一阶段 Demo](docs/07_代码框架与第一阶段Demo.md)
9. [论文框架](paper/论文框架.md)

## 代码入口

```powershell
python -m pip install -e .
python -m pytest
python scripts/smoke_stage1.py
```

当前 smoke 已在 CPU 上完成 HAD 3v4/2-target 环境步、变规模 QMIX 前向、TD loss 和 backward；5 个单测通过。它证明接口和梯度链可执行，不表示策略已经训练收敛。

## 环境结论

- **HAD/Open-HAD：第一 Demo 和主机制环境。** 当前只用打击智能体和目标；已经有 S1 adapter，尚缺环境自有 PRNG、clone/restore、任务 API 和增援。
- **ALMA `sc2multiarmy`：官方复现锚点。** AQL、heuristic 和 joint training 是必须 baseline，建议放在 WSL2 隔离环境。
- **Open-SMAX-AD：第二动力学。** 只有 Open-HAD 闭环成立后才开发；JAX GPU 不支持原生 Windows，目标 5070 Ti 上采用 WSL2。

## 当前真值

已实现：四阶段核心类、HAD S1 适配、配置、测试和 smoke。

未实现：完整 episode runner/replay/sequence learner、任何收敛结果、S2 干预数据管线、S3 完整在线 runner、S4 outer loop、ALMA 官方复现和 Open-SMAX-AD。

因此，文档中的胜率均为 go/no-go 门槛，不是已经获得的结果。
