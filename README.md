# Open-SCORE

研究已知固定对手下的动态分组与资源分配。当前 v5 使用共享规则底层、两个关键目标、最多 50 个物理步，在 15 个红蓝人数配比上比较六类方法。

- [实验记录](实验记录.md)：各版本的研究问题、关键结果和报告入口。
- [v5 执行方案](动态分组_六任务并行复现与诊断执行方案_v5.md)：当前方法、参数及剩余工作。
- [运行指南](Open-SCORE/README.md)：启动、恢复、终端进度和报告。
- [项目工作约定](AGENTS.md)：文件数量、结果合并与任务执行方式。

在本工作区 VS Code 按 `Ctrl+Shift+B` 启动或恢复 v5，在集成终端查看进度；已经运行时连接现有进度，不重复训练。结果统一保存在 [outputs/v5](Open-SCORE/outputs/v5)，每版本只保留一份主报告，数据与模型按方法组织。

2026-09-07 按用户要求取消网页及工程测试流程。本轮继续完成剩余训练、固定验证和正式胜率评估；已完成的科研诊断保留，未执行的额外诊断、预算比较与独立部署时延测量已取消。

本次代码整理版本为 v5.2，物理实验协议仍是 v5.1。新成果已归并为五个版本目录；旧输出的删除被自动执行审核拒绝，因此旧目录尚在。下列命令仅清理本次已归并的明确目录；请在本地 PowerShell 执行，不会触及新的五个版本目录或恢复后的训练：

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

[原始修改稿](固定对手下动态分组研究方案.md)按用户要求保留，用于历史讨论，不替代当前执行方案。不同底层、难度和协议的历史胜率分开解释。
