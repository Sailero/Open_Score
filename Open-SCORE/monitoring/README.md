# v5.1 实时实验报告

该报告服务只读取现有实验日志，写入运行目录下的 `live/`，不修改、导入或启动训练器。代码放在独立 `monitoring/` 目录，因此不改变本轮已冻结的算法源码指纹。仍以 `shared/budget_manifest.json` 中的方法版本及配额为准。

运行中的本地页面：<http://127.0.0.1:8765/live/>。页面和服务每 10 秒检查新记录；没有新验证点时保留上一点，不补造曲线。独立候选诊断来自原运行器约 5 分钟一次的汇总，页面单独标注其时间。

```powershell
Set-Location 'E:\Code\Open_Score\Open-SCORE'
& 'D:\Software\Anaconda\envs\torch310\python.exe' -X utf8 -u monitoring/live_v5.py --interval 10 --port 8765
```

已有服务时该命令只打印地址，不重复启动。VS Code 也提供“Open-SCORE v5.1: 实时图表报告服务”任务。停止报告服务不会停止训练；重新运行可以恢复报告更新。该服务仅绑定本机回环地址。

输出结构（相对于实验运行目录）：

- `live/index.html`、`live/report.md`：六任务总报告，9 个主要实验臂及 5 个公共对照分别展示。
- `live/tasks/T1…T6/index.html`、`report.md`：六个独立任务报告。
- `live/methods/<method>/index.html`、`report.md`：每个实验臂与公共对照的独立报告。
- `live/methods/<method>/curves.png`、`curves.json`：可保存的曲线及实际数据点。
- `live/snapshot.json`、`service_status.json`：机器可读快照及服务状态。

图表区分物理训练步、构造回合、ExIt 轮次、监督训练 epoch 与测试覆盖；训练损失、势差奖励、留出预测误差不能作为原生任务胜率。只有完整的固定验证集才画入验证胜率曲线，部分验证仍在表格中标明。T4 的教师/学生按同一回合族匹配；T6 所选门控阈值的验证成绩标为选参数据。末期主表只读取预先指定的最终检查点，不挑选测试最优模型。

Markdown 和 PNG 会随服务更新，但不同编辑器不一定自动重载预览；需要自动刷新时使用上面的浏览器入口。总实验完成状态需六任务、全部末期测试与独占部署测量都完成，和方法是否有效分别标记。

在没有报告服务运行的独立测试目录可以只生成一次：

```powershell
& 'D:\Software\Anaconda\envs\torch310\python.exe' -X utf8 monitoring/live_v5.py --run-dir outputs/v5_acceptance_final --once
& 'D:\Software\Anaconda\envs\torch310\python.exe' -X utf8 -m pytest monitoring/test_live_v5.py -q
```

2026-09-07 验收：12 项报告测试通过；实际浏览器验证了 10 秒自动刷新、曲线加载和 75 个本地链接。运行器及四个尚未结束的工作进程保持原 PID，冻结源码、资产及清单核验通过。首次部分测试出现零胜率时的误差线浮点边界问题已修正，失败记录与回归证据保存在 [验收目录](../outputs/v5_dashboard_qa/)。
