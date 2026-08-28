# Third Paper workspace

> **当前论文一句话：本文研究如何在攻防双方因战损和增援而持续改变规模时，通过可变规模执行、能力评估、动态重编组和上下层协同学习，维持多目标防御的总体胜率。**

工作区分为三部分：

- `01_original_saga_iclr/`：历史 SAGA 大设想归档，本轮未修改；
- `HAD_Env/`：自研三维攻防环境，作为第一阶段 Demo；本轮修复独立导入和目标阵营伤害语义；
- `02_q2_capability_aware_dynamic_command/`：Open-SCORE 四阶段论文规格和最小可运行代码。

建议阅读：

1. [研究结论与阅读路线](02_q2_capability_aware_dynamic_command/docs/00_研究结论与阅读路线.md)
2. [问题定义与创新边界](02_q2_capability_aware_dynamic_command/docs/01_问题定义与创新边界.md)
3. [相关工作与环境审计](02_q2_capability_aware_dynamic_command/docs/02_相关工作与环境审计.md)
4. [四阶段方法简版](02_q2_capability_aware_dynamic_command/docs/03_方法简版.md)
5. [详细技术方案与核心代码](02_q2_capability_aware_dynamic_command/docs/04_方法实现规格与伪代码.md)
6. [实验设计与验收门槛](02_q2_capability_aware_dynamic_command/docs/05_实验设计与预期证据.md)
7. [工作量、档位与拆分](02_q2_capability_aware_dynamic_command/docs/06_工作量_档位与拆稿.md)
8. [代码框架与第一阶段 Demo](02_q2_capability_aware_dynamic_command/docs/07_代码框架与第一阶段Demo.md)
9. [论文框架](02_q2_capability_aware_dynamic_command/paper/论文框架.md)

## 当前代码状态

已经实现四阶段核心类、HAD Stage-1 适配器、变规模实体 QMIX、配置、smoke 和测试。在当前 CPU PyTorch 环境中：

```powershell
cd 02_q2_capability_aware_dynamic_command
python -m pytest
python scripts/smoke_stage1.py
```

当前结果为 5 tests passed，HAD 3v4/2-target 完成一次环境步、TD loss 和 backward。尚无策略收敛或论文实验结果。

下一目标是用 3–5 周完成 S1 HAD Demo；四阶段双环境投稿级工作当前估计约 121–204 人日，不能压缩成无条件的 1–2 个月承诺。

## Git 恢复点

- `2d7e0b0`：上一版冻结 S1、聚焦选择可靠性的 SCORE 方案；
- `d5ac696`：Claude 备用框架；
- `abadfcc`：更早的 Claude 总体分析和原 HAD 状态。
