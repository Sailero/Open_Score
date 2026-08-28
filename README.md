# Third Paper workspace

> **当前论文一句话：我们研究如何把一个开放规模的多目标攻防态势拆成若干个1–4对1–4单目标子博弈，并利用可复用的微操能力、局面结局预测和实时分组博弈，在伤亡或增援后维持全局防守胜率。**

工作区分为三部分：

- `01_original_saga_iclr/`：历史 SAGA 大设想归档，本轮未修改；
- `HAD_Env/`：自研三维攻防环境，作为第一阶段纯加速度 Demo；
- `02_q2_capability_aware_dynamic_command/`：Open-SCORE 四阶段论文规格、HAD适配和最小可运行代码。

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

当前代码覆盖HAD 3v4/1-target双边纯加速度环境步、TD loss和backward；6项接口测试已通过。尚无策略收敛或论文实验结果。

下一目标是用5–8周完成S1 HAD多规模训练与PSRO pilot；完整双环境投稿级工作估计约119–200人日。

## Git 恢复点

- `2d7e0b0`：上一版冻结 S1、聚焦选择可靠性的 SCORE 方案；
- `d5ac696`：Claude 备用框架；
- `abadfcc`：更早的 Claude 总体分析和原 HAD 状态。
