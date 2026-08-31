# Workspace manifest

**Updated:** 2026-08-31
**Active project:** `E:\Code\Open_Score\Open-SCORE`

当前只保留一个项目根和一个正式执行入口：

- `Open-SCORE/README.md`：仓库入口；
- `Open-SCORE/docs/05_实验设计与预期证据.md`：S1/S2 唯一权威实验协议。

`Open-SCORE/docs/03_方法简版.md`（及 HTML 浏览版）只是长期方法背景，包含本轮不执行的 PSRO、S3 和 S4，不是实验入口。

`src/` 是 HAD、SMAClite-AD、Stage 1/2 实现，`configs/` 与 `scripts/` 是协议的机器入口，`tests/` 只产生工程证据。PSRO、S3、S4 代码可保留，但本轮不执行。

旧 pilot、smoke、重复报告和证据副本已经清理；运行期文件统一进入被 Git 忽略的 `outputs/`。当前没有按冻结协议产生的正式性能结果，任何 S1/S2 完成声明都必须以唯一实验协议的正式门控为准。

清理前恢复点：`3228da1802bfedd1cfa90b1f8ff4bc083a7c3389`。更早的完整工作区仍可从远程分支 `archive/pre-s1-full-20260828`（`9271abb`）恢复。
