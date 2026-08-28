# Workspace manifest

**Updated:** 2026-08-28
**Active root:** `D:\Code\Third_Paper\Open-SCORE`

主分支只保留 `Open-SCORE/`：

- `src/HAD_Env/`：已纳入项目包的自研三维 HAD 环境；
- `src/open_score/stage1/`：变规模实体 QMIX、课程、episode replay、双边 runner、序列 learner 与 PSRO；
- `src/open_score/stage2..4/`：后续阶段的最小接口；
- `docs/`、`paper/`：研究定义、调研、方法与实验协议；
- `scripts/`、`tests/`：smoke、2v2 PSRO pilot 和回归测试。

历史内容恢复点：

- 远程分支 `archive/pre-s1-full-20260828`，提交 `9271abb`：本轮整理前的完整工作区；
- 主分支不再保留 `01_original_saga_iclr/`、顶层 `HAD_Env/` 或旧 `02_q2_capability_aware_dynamic_command/` 的重复副本。

真值边界：S1 训练闭环已经可运行；短 2v2 pilot 只证明管线，不证明策略收敛。SMAClite-AD、完整 S2 数据、S3 闭环与 S4 共同适应尚未实现。
