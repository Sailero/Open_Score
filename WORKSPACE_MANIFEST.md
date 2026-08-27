# Workspace manifest

**Last restructured:** 2026-08-27
**Workspace:** `D:\Code\Third_Paper`

## 01_original_saga_iclr

历史 SAGA 大设想的完整归档。该目录保留原论文、理论、环境、原型、审计和实验输出，本阶段不在其中继续修改。

## 02_q2_capability_aware_dynamic_command

当前独立研究目录，工作题目为：

> *COGAR: Calibrated Outcome-Guided Adversarial Reallocation for Open-Population Multi-Front Games*

目录职责：

- `docs/`：中文决策材料，包括具体场景、创新审计、调研、复现底座、方法、实验、风险、八周路线和投稿选择。
- `paper/`：英文论文第 1–5 章工作稿及 BibTeX。
- `src/swarmbattle/`：已有 NumPy 双队攻防 P0；支持死亡、增援和变长实体，但尚无三路资产与上层 assignment API。
- `src/q2marl/`：已有实体集合 / HPN 风格网络骨架；不是 COGAR 的完整实现。
- `baseline/`：统一 baseline 协议。
- `upstream/`：外部论文、代码、许可证和复现策略；不把无许可证仓库复制进本项目。
- `configs/`：P0 与计划实验配置。

## 研究真值边界

- 已完成：当前代码盘点、文献调研、创新性否证、具体问题和方法设计、实验与八周路线。
- 已有可运行资产：SwarmBattle P0、脚本对手、集合编码模型及测试。
- 尚未完成：三路关键资产场景、干预式 rollout 数据集、校准战果模型、双边滚动分兵求解器、外部环境验证。
- 因此，所有“提升”“降低”均写作验收门槛或研究假设，不能当作已取得结果。

## Git 可恢复点

- `f281296`：CDOC 研究版本。
- `8766809`：开放人口上层对抗分配版本。
- `9a9b77e`：本轮重构前备份，已推送 `origin/master`。

原目录名 `02_q2_open_population_adversarial_allocation/` 已在本轮重构中改为当前目录名；旧内容仍可从 `9a9b77e` 完整恢复。
