# Open-SCORE 运行指南

当前 v5 使用训练种子 `20260907`，红方 8/12/16/24/32 人，蓝方为红方人数的 1/2、3/4 或全部，15 个配比等权混合。已知 reactive 对手、共享规则底层、双目标、原生终局奖励和最多 50 个物理步保持不变。

研究背景和历史结果见 [实验记录](../实验记录.md)，方法及参数见 [v5 执行方案](../动态分组_六任务并行复现与诊断执行方案_v5.md)。

## 启动或恢复

在 VS Code 打开 `E:\Code\Open_Score`，按 `Ctrl+Shift+B`。默认任务在本地集成终端启动或恢复已有进度，已有运行器时显示其状态。也可以在 PowerShell 执行：

```powershell
Set-Location 'E:\Code\Open_Score\Open-SCORE'
& 'D:\Software\Anaconda\envs\torch310\python.exe' -u scripts/run_research_v5.py resume --run-dir outputs/v5
```

只查看终端进度，不启动训练：

```powershell
& 'D:\Software\Anaconda\envs\torch310\python.exe' -u scripts/run_research_v5.py status --watch --run-dir outputs/v5
```

只恢复 AR-PPO 或重新生成已有数据的报告：

```powershell
& 'D:\Software\Anaconda\envs\torch310\python.exe' -u scripts/run_research_v5.py resume --method ar_ppo --run-dir outputs/v5
& 'D:\Software\Anaconda\envs\torch310\python.exe' scripts/run_research_v5.py report --run-dir outputs/v5
```

`run`、`resume`、`status`、`report` 是统一入口。方法名为 `rollout`、`mcts_dpw`、`candidate_ppo`、`ar_ppo`、`exit`、`bridge`、`paired_value` 和 `baselines`。同一任务的必要实验臂依次执行，已完成记录自动跳过。监控终端关闭不停止训练。

## 结果在哪里

每个版本只使用一个目录：`outputs/s1_s2`、`v2`、`v3`、`v4`、`v5`。每个版本的 `实验报告.md` 包含所有方法及总体比较，不再生成网页或六套重复报告。

当前结果入口：[v5 实验报告](outputs/v5/实验报告.md)。各方法的数据保存在 `data.sqlite`，共享训练数据和评估开局保存在 `shared/data.sqlite`；模型放在 `models`，图放在 `figures`，日志追加到一个 `run.log`。SQLite 中按运行、种子、方法、实验臂及训练/验证/正式评估区分记录，历史原始轨迹和快照以原内容合并保存。

权重保留最终 `final.pt` 和验证最优 `best.pt`；运行期间另保留恢复点 `resume.pt`。同一方法确有多套权重时以模型名区分。数据是分析的权威来源，报告和图片从这些数据更新；需要给其他工具分析时再按需导出。

## 当前运行范围与结果口径

2026-09-07 整理前保存 AR-PPO 到 292,606 个物理步。恢复后完成该臂约 500,000 步、剩余固定验证及 1,500 局正式评估，不重跑其他已完成实验臂。未执行的额外候选诊断、搜索预算比较和独立部署延迟测量按用户要求取消，已有诊断数据仍保留。

14 个正式实验臂共使用同一批 1,500 个开局，满额共 21,000 局执行。主表使用最终模型，验证最优模型只作补充。训练窗口胜率、验证胜率和正式胜率分开，按人数和难度报告分子分母；单个训练种子的结果不代表跨种子稳健优势。执行错误和中断不记成任务失败。

旧版环境、模型加载和规则执行器仍被当前方法使用的代码保留；已废弃训练入口、工程测试及校准审核流程不再作为启动前提。