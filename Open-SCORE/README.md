# Open-SCORE：未知对手意图下的长期身份分组

本轮重点研究 Stage3，也就是上层分组：**观察蓝方的运动，判断它准备怎样分配兵力，再决定哪些红方成员去保护哪个目标。** 蓝方的底层运动规则已知，但本局使用的上层分配规则和当前分组不公开。我们希望红方在合理的决策时间内，接近“提前知道蓝方规则”的参照方法。

整套框架的技术名称是 BA-DIB；当前主方法在交流中统一称为“学习推断＋多步规划”，代码名为 `qom_mcp`。名称不代表优势已经得到证明。目前只有初步成套结果，尚未证明接近知情参照，计算时间也仍需优化。

简化协议：Blue 在一局中固定四种上层策略之一，不切换策略类型；固定规则仍可依据位置/伤亡重新分配成员。本轮不运行 feint_switch、开放集检测或 novelty fallback。

Stage1 是冻结的局部控制器，Stage2 是冻结的结局/时间代理。两者为所有方法共用，不再扩展为本轮论文的独立创新点。全局物理规模为 30/40/50 人、2/4/5 个固定目标，伤亡改变有效规模，每个局部匹配最多 4v4。

当前已合并历史证据、保留旧 identity 负结果，并实现新方法和正式流水线。新方法效果以报告中完成的注册结果为准；测试通过不代表已经达到 oracle 接近性门槛。

- **建议先读：[Stage3 目前的结果、方法含义与下一步](docs/10_Stage3阶段成果与下一步.md)**
- [论文研究主线与实验论证](paper/论文框架.md)
- [MVP 完整方法：研究背景、Stage1–3 建模与贡献边界](docs/07_MVP完整方法_S1-S3.md)
- [MVP 实验协议：基线、执行设置、统计分析与现有证据](docs/08_S3实验执行与分析协议.md)
- [原计划漏洞与建模修订记录](docs/09_BA-DIB_审查与完整建模.md)
- [唯一 Stage1–3 实验报告](outputs/stage123_unknown_upper_v1/experiment_report.md)
- [固定实验配置](configs/stage123_unknown_upper.yaml)

在本目录执行唯一正式入口：

```powershell
& .\scripts\run_stage123_unknown_upper.ps1 -Mode Formal -Background -Workers 2
```

命令立即返回 PID、日志、状态和报告路径。运行顺序为冻结资产 SHA256 校验、必要的约束检查、QOM 数据采集和训练、1,920 局核心比较及 480 局同预算机制消融（共 2,400 局）、确定性合并与综合报告。扩展中间诊断默认关闭。重复执行相同命令会复用内容与输入哈希一致的单元；不要在已有协调进程运行时重复启动。Stage2 不重新采集或训练。

```text
outputs/stage123_unknown_upper_v1/pipeline.log
outputs/stage123_unknown_upper_v1/status.json
outputs/stage123_unknown_upper_v1/experiment_report.md
```

核心八方法为 balanced_identity、legacy_idb、event_risk_idb、finite_type_bbr、qom_bbr、qom_mcp、known_upper_mcp、revealed_blue_action_br；机制消融为 finite_type_mcp 和 prior_mcp。QOM-MCP 是预注册主方法，显式 Bayesian 模型是强结构化参照。若它更强，照实报告，不在正式测试后替换主方法。

正文关注胜率、oracle gap、意图推断和规划时间，并分别检查两种 Blue 底层。已知上层 oracle 看不到当前随机动作；完整动作可见参照另列。共用有限候选域不构成全局 Nash 或最优性证书。

测试：

```powershell
& D:\Software\Anaconda\envs\torch310\python.exe -m pytest -q
git diff --check
```

约 120 MB 的 Stage2 JSONL、所有模型和新轨迹仅留本地；Git 保留代码、配置、数据哈希、统一报告和核心图片。克隆仓库后需另行提供清单指定的冻结资产。历史 identity/SALDAE 实现与文档保留供复核，旧运行入口需要显式指定输出位置，不能覆盖当前冻结证据。
