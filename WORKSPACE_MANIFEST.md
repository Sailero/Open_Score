# Workspace manifest

**Updated:** 2026-08-30
**Active root:** `E:\Code\Open_Score\Open-SCORE`

主分支只保留 `Open-SCORE/`：

- `src/HAD_Env/`：已纳入项目包的自研三维 HAD 环境；
- `src/open_score/envs/`：HAD 的 S1 适配器，以及独立命名、未覆盖上游注册项的 SMAClite-AD 扩展；
- `src/open_score/stage1/`：变规模实体 QMIX/VDN/MAPPO、课程、episode replay、双边 runner 与序列 learner；PSRO 代码保留，但本轮不作为验收项；
- `src/open_score/stage2/`：HAD 前向推演数据契约、监督学习、校准和直观评估指标；`stage3..4/` 仍只保留接口；
- `docs/`、`paper/`：研究定义、调研、方法与实验协议；
- `scripts/`、`tests/`：smoke、2v2 PSRO pilot 和回归测试。

历史内容恢复点：

- 远程分支 `archive/pre-s1-full-20260828`，提交 `9271abb`：本轮整理前的完整工作区；
- 主分支不再保留 `01_original_saga_iclr/`、顶层 `HAD_Env/` 或旧 `02_q2_capability_aware_dynamic_command/` 的重复副本。

真值边界：SMAClite 原版和独立 SMAClite-AD 环境、HAD 上的 S1 三类基线，以及基于真实 HAD 前向推演数据的 S2 管线均已进入小规模验证；短训练只证明管线或学习信号时，不能表述为正式收敛。HAD 场景严格要求红方初始数量大于蓝方。S3 闭环与 S4 共同适应本轮冻结。
