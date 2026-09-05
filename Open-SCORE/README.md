# Open-SCORE：多目标身份级动态分组

当前实验版本是 **ID-CS-BBG + SALDAE-DO / v6**。它连接三个阶段：冻结的 Stage1 局部协同策略、完整 1–4v1–4 的 Stage2 动态局部结局模型，以及身份级 Stage3 联盟结构 Blotto。v6 不改变博弈模型，在 v5 候选列 MILP-DO 之上增加约束 SALDAE anytime 最佳响应器。

Stage3 的双方上层动作都是带标签智能体到公开 `(目标, 对抗通道)` 的同时分区。联合动作直接得到：

```text
group i: target=t_i, channel=h_i, Red=(exact IDs), Blue=(exact IDs)
```

同一目标允许多个组；每个 Blue 身份必须恰好进入一个活跃组，每个 Red 身份必须恰好进入一个活跃组或 reserve，身份不能重复；任何非空局部组硬性不超过 4v4。Blue 无对应 Red 防守时可显式形成 `0v1`–`0v4`。物理实验覆盖 30/40/50 个总实体和每局固定的 2/4/5 个目标，不把 1000 人作为 HAD 实验。

## 文档

- [完整方法](docs/07_MVP完整方法_S1-S3.md)：任务边界、Stage2 数据语义、身份级数学模型、Red集合装填/Blue集合划分 MILP、Double Oracle、理论依据和未来问题。
- [实验协议](docs/08_S3实验执行与分析协议.md)：v5与SALDAE-DO增量实验设置、验收指标、运行命令、进度和结果位置。
- [Round-01 历史对照](outputs/round_01_mvp/round_01_report.md)：只作旧结果量级参考。

当前主实验分别假定 Blue 底层规则为已知的 `rush` 或 `split_rush`，但 Blue 上层联盟与目标分配仍由博弈玩家选择。真实 Blue 策略库的发现、PSRO 扩库和开放集检测属于后续工作。

物理对照 `revealed_blue_br` 观察从同一 Blue 均衡策略采样的当前动作，再在与主方法相同的候选域中求 Red 1–4人组及 reserve 最佳响应。它仅是当前局部价值代理与候选域内的信息上界，不是可部署策略，也不是 HAD 真实胜率的理论上界。

## 运行

快速连通性测试：

```powershell
& .\scripts\run_stage23_aligned.ps1 -Mode Smoke
```

正式后台实验：

```powershell
& .\scripts\run_stage23_aligned.ps1 -Mode Formal -RunRoot "outputs\mvp\stage23_identity_blotto_v5" -Background
```

若该目录中的 Stage2 已经完成且通过验收，只恢复 Stage3：

```powershell
& .\scripts\run_stage23_aligned.ps1 -Mode Formal -RunRoot "outputs\mvp\stage23_identity_blotto_v5" -ResumeStage3 -Background
```

恢复入口会先验证 Stage2 状态、协议配置以及数据集、Stage1、训练配置和 checkpoint 哈希；不会重新采集或训练。

SALDAE-DO 独立增量实验会直接复用上述Stage1/2，并以旧MILP均衡支持热启动：

```powershell
& .\scripts\run_stage3_saldae.ps1 -Mode Formal -Background
```

程序实时记录采集、训练、求解和 HAD 评估进度，并自动生成 Stage2、Stage3 与联合报告。

## 结果

正式根目录是 `outputs/mvp/stage23_identity_blotto_v5/`：

- `pipeline.log`：Stage2→Stage3 实时日志；
- `pipeline_status.json`：流水线状态；
- `reused_stage2_validation.json`：恢复运行时的 Stage2 哈希校验；
- `stage23_report.md`：联合结论和新旧对比；
- `stage2/stage2_report.md`：数据、训练、校准和形状诊断；
- `stage3/stage3_report.md`：求解正确性、规划规模和 HAD 胜率；
- `stage3/raw/physical_identity_episodes.csv`：逐局结果和逐次实名分组。

身份级纯博弈与 MILP 最佳响应位于 `src/open_score/stage3/identity_blotto.py`，HAD/Stage2 桥接位于 `src/open_score/stage3/identity_runtime.py`，自动评估位于 `scripts/evaluate_stage3_identity.py`。

SALDAE-DO 求解器位于 `src/open_score/stage3/saldae.py`，独立评估位于 `scripts/evaluate_stage3_saldae.py`；正式结果写入 `outputs/mvp/stage3_saldae_do_v1/`。
