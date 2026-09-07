# Open-SCORE

研究已知固定对手下的动态分组与资源分配。当前 v6.4 根据 [v6 执行方案](dynamic_grouping_v6_execution_plan.md) 实现，使用更新渲染后的独立 **HAD Workbench** 环境，以及共享规则底层。

v6 配置在 [research_v6.yaml](Open-SCORE/configs/research_v6.yaml)。六种方法为 `BLOTTO_Count`、`BLOTTO_Group`、`ALMA_Alloc`、`MAPPO_Intent`、`ALMA_Group`、`ALMA_S2`，后者为优先验证的主方法。五场景包含 10v10/20v20 的两目标与三目标任务，以及 10v6 两目标辅助任务；每个 RL 方法每个种子训练 5,000,000 物理步，三个种子合计四种 RL 共 60,000,000 步。

本机已将 `E:\Code\Open_Score_HAD_Workbench` 以 editable 包 `had_env` 安装到 torch310。v6 直接调用该新版的物理引擎、规则执行器及蓝方逻辑；不调用项目内旧 `HAD_Env` 作为物理环境。保留分组协议的 50 步防守成功终止，通用 Parallel API 的时间截断不替代此定义。S2 在新环境中重新采集和训练；六种方法部署决策均不调用真实模拟分支。

这套 60M 配额很重：本机同一小批真实状态上的三条 AQL 完整 CUDA 更新约 2.94/3.86/6.46 秒，按观察到的事件频率外推，整轮可能为**数周量级**，不能按 20 小时估算。此为初始化模型短样本外推，三任务并发共享 GPU，实际时长还包括采样、S2 准备与评价；保持原方案配额，尚未启动正式长训练。

启动或恢复整轮只需这一条命令：

```powershell
& 'D:\Software\Anaconda\envs\torch310\python.exe' -u -m open_score.research_v6 run-all
```

入口自动完成共享 S2 数据和模型、六方法队列、固定验证、2M/5M 正式评价与有限诊断。最多三个重任务、每个两个采样进程；配置不设墙钟停止。再次输入同一命令会跳过已完成任务并从完整检查点继续。终端每 10 秒显示阶段、真实步数、优化次数、训练胜率、最近验证及当前阶段剩余时间，细节追加到各方法 `run.log`。结果只写入 `Open-SCORE/outputs/v6`，共用 SQLite 数据和必要权重；产生真实记录后每 5 分钟更新唯一 `实验报告.md` 及原位图表，结束时立即汇总，没有网页或逐回合文件。

- [实验记录](实验记录.md)：各版本的研究问题、关键结果和报告入口。
- [已完成的 v5 唯一实验报告](Open-SCORE/outputs/v5/实验报告.md)：方法、训练/验证曲线、正式胜率图、完整结果表与结论限制。
- [项目工作约定](AGENTS.md)：文件数量、结果合并与任务执行方式。

v5 训练与正式评估已于 **2026-09-07 17:25:52** 完成：14/14 个实验臂、21,000 局。用户已关闭该轮，旧执行方案和 VS Code 启动任务已移除。最终成果合并为一份 Markdown 报告，引用保留的图片与必要数据，各方法不另建报告。

本次代码整理版本为 v5.2，物理实验协议仍是 v5.1。未执行的额外诊断、预算比较与独立部署时延测量已取消。此前旧输出的删除被自动执行审核拒绝，以下保留当时限定目录的清理命令，不表示本次清理已经完成：

```powershell
$outputRoot = 'E:\Code\Open_Score\Open-SCORE\outputs'
$retiredNames = @(
  'analysis_exports', 'historical_s1_s2', 'overnight_audit', 'overnight_ppo_audit',
  'overnight_runtime_benchmark', 'overnight_validation', 'v2_device_probe_1024',
  'v2_device_probe_20260906', 'v2_parallel_100k', 'v2_parallel_100k_w3_new', 'v2_validation',
  'v3_overnight', 'v3_validation', 'v4_comparison', 'v4_cpu_parallel_benchmark',
  'v4_cpu_parallel_benchmark_two_repeats', 'v4_integration_first', 'v4_integration_second',
  'v4_learning_validation', 'v4_question_audit', 'v4_validation', 'v5_acceptance',
  'v5_acceptance_final', 'v5_candidate_vectorization', 'v5_dashboard_qa', 'v5_parallel',
  'v5_public_integration_audit', 'v5_smoke_cli', 'v5_smoke_cli_2', 'v5_smoke_final',
  'v5_validation', 'v5_value_bench_probe', 'v5_value_tasks_integration', 'v5_value_tasks_integration_v2'
)
foreach ($name in $retiredNames) {
    $candidatePath = Join-Path $outputRoot $name
    if (Test-Path -LiteralPath $candidatePath) {
        $target = Get-Item -LiteralPath $candidatePath
        if ($target.Parent.FullName -ne $outputRoot -or $target.LinkType) { throw 'Unexpected cleanup target' }
        Remove-Item -LiteralPath $target.FullName -Recurse -Force
    }
}
$projectRoot = 'E:\Code\Open_Score\Open-SCORE'
foreach ($name in @('tests', 'monitoring', 'docs', '.pytest_cache')) {
    $candidatePath = Join-Path $projectRoot $name
    if (Test-Path -LiteralPath $candidatePath) {
        $target = Get-Item -LiteralPath $candidatePath
        if ($target.Parent.FullName -ne $projectRoot -or $target.LinkType) { throw 'Unexpected cleanup target' }
        Remove-Item -LiteralPath $target.FullName -Recurse -Force
    }
}
```

[原始修改稿](固定对手下动态分组研究方案.md)按用户要求保留，用于历史讨论。不同底层、难度和协议的历史胜率分开解释。
