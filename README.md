# Open-SCORE

当前仓库保留 HAD Workbench 的环境、规则策略和公开接口。旧研究主线已于 **2026-09-08 16:00:19 +08:00** 确认停止；探索 final 为 Git `cc2ab86`，历史研究源码可从该提交查阅。

## 环境与安装

唯一底层是独立仓库 [HAD Workbench](../Open_Score_HAD_Workbench/README.md)，本仓库直接导出其原生实现。当前本地依赖路径为 `E:/Code/Open_Score_HAD_Workbench`，该仓库未在本次清理中修改。Open-SCORE 不再提供训练、评估或研究报告生成命令。

```powershell
& 'D:\Software\Anaconda\envs\torch310\python.exe' -m pip install -e ./Open-SCORE
```

依赖由 `had-env` 声明；渲染及 Workbench 界面的可选安装方法见其 README。此命令供后续安装使用，本次归档没有重装环境。

## 公开接口

| 导入 | 用途 |
|---|---|
| `open_score.make_env`、`open_score.parallel_env` | 原生 Parallel / MPE 环境工厂 |
| `open_score.grouping.KnownOpponentEnv`、`make_grouping_env` | 原生事件式分组环境；后者默认规则执行器 |
| `Entity`、`Group`、`Grouping`、`DecisionState`（均位于 `open_score.grouping`） | 实体、分组动作与公开状态，沿用原生序列化与合法性约束 |
| `RuleExecutor`、`RulePolicy` | 共享规则底层与上层规则策略 |
| `rule_grouping`、`grand_grouping`、`decode_counts` | 规则分组、每目标大组及人数分配解码 |

规则策略支持 `rule`、`grand`、`random`、`static_rule`；`reset()` 重置策略状态，`act(state)` 返回原生 `Grouping`。固定对手接口在 `had_env.grouping.opponents`，物理适配器和快照类型在 `had_env.grouping.adapter`。

```python
from open_score.grouping import make_grouping_env, RulePolicy

env = make_grouping_env(red=8, blue=8, targets=2, opponent="reactive", seed=0)
policy = RulePolicy("rule", seed=0)
state = env.reset()
policy.reset()
grouping = policy.act(state)
state, reward, done, info = env.step(grouping)
env.adapter.env.close()
```

示例用于说明接口，本次未运行仿真。分组环境原生时限为任务终止；Parallel/MPE 工厂采用自己的时限与截断语义，二者不混用。多目标位置、零人数、事件返回和快照格式均以 Workbench 原生 API 为准；已移除 v6 包装、奖励塑形、S2 投影及旧训练张量接口。

## 历史研究

[实验记录](实验记录.md) 维护跨版本摘要。六份唯一正式报告均在当前根目录：

- [S1/S2 与早期 Stage3](实验报告_s1_s2.md)
- [v2：选择释放与重建](实验报告_v2.md)
- [v3：较长训练与底层迁移](实验报告_v3.md)
- [v4：共享规则底层与价值搜索](实验报告_v4.md)
- [v5：混合难度六路线](实验报告_v5.md)
- [v6：新环境六方法与 S2 修订](实验报告_v6.md)

数据、日志、配置快照和已有模型原位保留于 `Open-SCORE/outputs`；历史冻结模型仍在 `Open-SCORE/assets/frozen`，当前接口不加载这些模型。多数输出资产被 Git 忽略，不属于探索 final 提交的备份范围。恢复旧研究需使用探索 final 的代码与对应历史环境，当前精简接口不兼容旧训练检查点。

[原始研究修改稿](固定对手下动态分组研究方案.md) 与 [v6 原执行方案](dynamic_grouping_v6_execution_plan.md) 仅作为历史设计材料保留。各版实际完成量、取消项和结论以根目录正式报告为准。
