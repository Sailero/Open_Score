# Workspace manifest

**Updated:** 2026-08-28
**Workspace:** `D:\Code\Third_Paper`

## 01_original_saga_iclr

历史 SAGA 大构想归档。本轮未修改，也不把其中代码或结果视为 Open-SCORE 证据。

## HAD_Env

Open-HAD 的代码起点。现有红蓝打击/侦察/干扰类、目标、三维运动、攻击与观测接口。本轮只为第一阶段做两项明确修改：

1. 删除未使用且缺失的 `common.arguments` 导入，使环境可独立 import；
2. 目标只受蓝方攻击者伤害，防止红方防守者摧毁己方目标。

仍缺环境自有 PRNG、clone/restore、参数化终止、任务 assignment、增援槽位和正式回归测试。

## 02_q2_capability_aware_dynamic_command

工作题目：

> *Open-SCORE: Capability-Aware Hierarchical Learning for Open-Population Multi-Target Defense*

目录包含：

- `docs/00–07`：结论、定义、调研、四阶段方法、实验、工作量和代码路线；
- `src/open_score/`：S1–S4 核心模块和 HAD 适配器；
- `configs/`：第一阶段 YAML；
- `scripts/`：CPU/CUDA smoke；
- `tests/`：五项核心测试；
- `paper/`：论文框架与 BibTeX；

上一版窄方案的历史概念图已删除；新图应在 S1 Demo 收敛后依据真实接口重绘。

## 真值边界

- 已完成：源码调研、环境审计、四阶段数学/代码接口、可执行 smoke、核心测试、实验和工作量设计。
- 尚未完成：episode replay/runner/sequence learner、任何训练曲线或胜率、Open-HAD P0/P1、S2 数据、S3 在线闭环、S4 outer loop、ALMA 复现和 Open-SMAX-AD。
- “任意规模”只表示网络结构不绑定固定 $N$；经验结论必须限定在已测试规模区间。
- 四阶段作为一个框架贡献，但每阶段必须有消融和硬门槛。

## Git 可恢复点

- `2d7e0b0`：上一版单篇收缩方案及五张方法图；
- `d5ac696`：Claude 备用框架；
- `abadfcc`：更早的 Claude 总体分析、旧 Q2 目录和原 HAD 状态。
