# Open-SCORE v4.0

研究已知且完整策略固定的对手条件下，红方如何根据局内减员动态分组。v4 的所有上层共用纯规则红方执行器，每局最多 50 个物理步，不再依赖 S1 的四人支持范围。比较三条上层学习路线与三条 Blotto 式分配路线；优先验证完整编组 S2 的续行价值与滚动重规划。

当前入口、单独运行主方法及全部路线的命令见 [运行指南](Open-SCORE/README.md)，方法与实验语义见 [v4 研究方案](固定对手下动态分组研究方案_v4.md)，验证证据见 [v4 验证报告](Open-SCORE/docs/results_v4.md)。代码入口为 [run_research_comparison.py](Open-SCORE/scripts/run_research_comparison.py)。

历史 v3 失败诊断与审计继续保留：[环境诊断](Open-SCORE/docs/failure_analysis_v3.md)、[PPO 审计](Open-SCORE/docs/ppo_audit_v3.md)。旧 S1/S2 模型和历史结果保留用于对照，不能将其预测或胜率冒充为新规则底层的结果。

先进入 `Open-SCORE` 目录，再按运行指南进行单算法最小验证。开发验证与正式多种子实验分别报告；实现完成不代表已经证明性能优势。

[另一份研究讨论稿](固定对手下动态分组研究方案.md)按用户要求保留其新增修改。v2/v3/v4 的执行器或动作域不同，历史实验分开报告，检查点不能跨版本直接恢复。
