# Workspace manifest

**Last restructured:** 2026-08-28
**Workspace:** `D:\Code\Third_Paper`

## 01_original_saga_iclr

历史 SAGA 大设想的归档。本轮没有修改，也不把其中的实验结果或原型自动视为 SCORE 的证据。

## HAD_Env

用户自研三维攻防环境，是 Open-HAD 的唯一代码起点。现有资产包括运动、打击/拦截、目标耐久与死亡逻辑；已审计到缺失依赖、全局随机数、无 snapshot/restore、无上层 assignment、无增援接口、友伤/终止等语义风险。本轮只理解与记录，没有改动代码。

## 02_q2_capability_aware_dynamic_command

当前论文规格，工作题目为：

> *SCORE: Selection-Calibrated Outcome-Guided Reallocation for Dynamic-Population Multi-Target Defense*

目录只保留论文真正需要的材料：

- `docs/00–07`：结论、问题边界、相关工作/环境审计、方法简版、实现规格/伪代码、实验协议、工作量/投稿档位、五图记录；
- `docs/figures/`：按冻结协议重新生成的五张概念参考图；投稿时仍应依照同文档的 Mermaid 重绘为矢量图；
- `paper/论文框架.md`：从标题、摘要到 Method、Experiments、Limitations 的整篇骨架；
- `paper/references.bib`：已核验引用键的 BibTeX；
- `README.md`：当前目录入口。

旧 COGAR 占位源码、配置、测试、分散章节和过时图已删除，因为它们既不实现当前 SCORE，也会误导完成度。当前目录因此是**论文研究规格而非软件包**。

## 研究真值边界

- 已完成：HAD/ALMA/SMAX 适配审计、核心文献定位、创新边界、SCORE 方法协议、完整伪代码、E0–E4 实验与统计方案、工作量/拆稿判断、论文框架和五张概念图。
- 尚未完成：Open-HAD 适配、冻结 S1、双边干预数据、SCORE 实现、任何 SCORE 实验、ALMA 官方复现、Open-SMAX-AD 新任务扩展和投稿稿结果。
- 所有数值都是预算、预注册门槛或待检验假设；不能当作已经得到的性能。
- 第一篇只包含冻结 S1 下的 S2+S3；上下层共同适应 S4 属于第二篇。

## Git 可恢复点

- `abadfcc`：本轮重构前的 Claude 总体分析、Q2 旧目录与 HAD 状态；
- `d5ac696`：根目录 Claude 备用框架草稿；
- 更早的历史方案仍可由 Git log 定位，不在当前入口重复陈列。
