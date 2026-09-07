# Open-SCORE

研究已知固定对手下的动态分组与资源分配。本轮 v5 使用共享规则底层、两个关键目标、最多 50 个物理步，在 15 个红蓝人数配比上比较六类方法。

- [实验记录](实验记录.md)：各版本的研究问题、关键结果和报告入口。
- [本轮唯一实验报告](Open-SCORE/outputs/v5/实验报告.md)：方法、训练/验证曲线、正式胜率图、完整结果表与结论限制。
- [项目工作约定](AGENTS.md)：文件数量、结果合并与任务执行方式。

本轮训练与正式评估已于 **2026-09-07 17:25:52** 完成：14/14 个实验臂、21,000 局。用户已关闭本轮，执行方案和 VS Code 启动任务已移除。最终成果合并为一份 Markdown 报告，引用保留的图片与必要数据，各方法不另建报告。

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
