# Open-SCORE：未知对手意图下的长期身份分组

本轮论文研究 **BA-DIB（Belief-Aware Dynamic Identity Blotto）**：Blue 底层 `rush` / `split_rush` 已知且固定，上层分组、攻击目标和调整时机未知；Red 仅根据公开轨迹积累信念，联合选择自身成员、保护目标、拦截意图及后备队。核心目标是使防守胜率接近同预算的已知上层策略 oracle。

简化协议：Blue 在一局中固定四种上层策略之一，不切换策略类型；固定规则仍可依据位置/伤亡重新分配成员。本轮不运行 feint_switch、开放集检测或 novelty fallback。

Stage1 是冻结的局部控制器，Stage2 是冻结的结局/时间代理。两者为所有方法共用，不再扩展为本轮论文的独立创新点。全局物理规模为 30/40/50 人、2/4/5 个固定目标，伤亡改变有效规模，每个局部匹配最多 4v4。

当前已合并历史证据、保留旧 identity 负结果，并实现新方法和正式流水线。新方法效果以报告中完成的注册结果为准；测试通过不代表已经达到 oracle 接近性门槛。

- [论文研究主线与实验论证](paper/论文框架.md)
- [完整建模、原计划漏洞、文献与协议](docs/09_BA-DIB_审查与完整建模.md)
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
