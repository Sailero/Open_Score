# Third Paper workspace

> **当前论文一句话：本文研究如何让多目标防御指挥官在战损后在线重编组时，少选“预测安全、实则失守”的分兵方案。**

本工作区把历史大设想、原始自建环境与当前单篇论文规格分开：

- `01_original_saga_iclr/`：原始 SAGA / ICLR-scale 构想、论文、理论和原型归档，本轮未改动。
- `HAD_Env/`：用户自研三维攻防环境；当前论文拟只保留打击智能体和目标点并适配为 Open-HAD，本轮只审计、未修改。
- `02_q2_capability_aware_dynamic_command/`：当前 SCORE 论文的精简研究规格、论文框架、参考文献和方法示意图。

建议阅读：

1. [研究结论与阅读路线](02_q2_capability_aware_dynamic_command/docs/00_研究结论与阅读路线.md)
2. [问题定义与创新边界](02_q2_capability_aware_dynamic_command/docs/01_问题定义与创新边界.md)
3. [相关工作与环境审计](02_q2_capability_aware_dynamic_command/docs/02_相关工作与环境审计.md)
4. [方法简版](02_q2_capability_aware_dynamic_command/docs/03_方法简版.md)
5. [方法实现规格与伪代码](02_q2_capability_aware_dynamic_command/docs/04_方法实现规格与伪代码.md)
6. [实验设计与预期证据](02_q2_capability_aware_dynamic_command/docs/05_实验设计与预期证据.md)
7. [工作量、档位与拆稿](02_q2_capability_aware_dynamic_command/docs/06_工作量_档位与拆稿.md)
8. [图表与生成记录](02_q2_capability_aware_dynamic_command/docs/07_图表与生成记录.md)
9. [论文框架](02_q2_capability_aware_dynamic_command/paper/论文框架.md)

当前交付是**经调研与审计后的研究定义**，不是已完成的算法或实验。旧 COGAR 文档、占位代码和无效配置已从当前工作树移除；它们仍可从 Git 恢复：

- `abadfcc`：备份 Claude 总体分析、原 Q2 目录与 HAD 环境状态；
- `d5ac696`：备份根目录的 Claude 备用框架草稿。

单人 5–8 周的现实目标是完成 E0–E2 并判断论文是否成立；若要形成含强基线、正式 seeds 和第二动力学关键复核的 Q2 投稿稿，当前估计为单人 17–25 周，不能把尚未测速的实验量压成无条件的 1–2 个月承诺。
