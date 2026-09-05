# Open-SCORE v3.0

研究已知且完整策略固定的对手条件下，红方如何根据局内减员动态分组。红方下层冻结，每局最多 50 个物理步。v3 基于 v2 十万步对照的失败诊断，采用统一的冻结下层输入适配与结构化候选分组，并提供 PPO、教师初始化 PPO、候选 Double DQN 三条过夜训练路线。

今晚运行入口、预算、输出和验收说明见 [v3 过夜方案](固定对手下动态分组研究方案_v3_过夜训练.md)。[环境失败诊断](Open-SCORE/docs/failure_analysis_v3.md)与[PPO 审计](Open-SCORE/docs/ppo_audit_v3.md)分别记录已有证据和仍需验证的假设。代码入口为 [run_overnight_portfolio.py](Open-SCORE/scripts/run_overnight_portfolio.py)。

代码和可直接运行的续训命令见 [Open-SCORE/README.md](Open-SCORE/README.md)，实际小预算验证见 [results_v2.md](Open-SCORE/docs/results_v2.md)。研究设计见 [固定对手下动态分组研究方案_v2.md](固定对手下动态分组研究方案_v2.md)，实现说明见 [method_v2.md](Open-SCORE/docs/method_v2.md)。

先进入 `Open-SCORE` 目录，再按运行指南进行单算法最小验证。开发验证与正式多种子实验分别报告；实现完成不代表已经证明性能优势。

[另一份研究讨论稿](固定对手下动态分组研究方案.md)按用户要求保留其新增修改；v2 方案与原始结果作为历史研究版本保留，v3 与 v2 的执行器输入及动作域不同，不能混称同一协议下的算法收益。
