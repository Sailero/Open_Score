# Open-SCORE

**LEAF**以攻击者、目标和队友构成的威胁情境支持多目标防护。当前主工作区保留LEAF、七个设计比较、七个学习基线及Rule/Random的HAD结果；同一冻结策略的实体规模外推、执行深度、机制、时延和受控探针在一份[主实验报告](docs/LEAF_主实验结果分析.md)中解释。

| 入口 | 内容 |
|---|---|
| [主实验结果分析](docs/LEAF_主实验结果分析.md) | 当前正式比较、学习验证、设计比较、正常/冻结机制、R1–R6、独立确认、时延与探针 |
| [方法与变体说明](docs/LEAF_方法与变体说明.md) | 当前前向、训练/执行深度及保存设计比较的query条件 |
| [Baseline与Rule实现说明](docs/LEAF_Baseline与Rule实现说明.md) | 七基线、Rule/Random、共同对手、优化设置与实现来源 |
| [HAD环境与问题建模](docs/HAD_环境与问题建模.md) | 当前正式数据的实体接口、物理与奖励协议 |
| [论文包](Papers/1014/README.md) | [main.tex](Papers/1014/main.tex)、[PDF](Papers/1014/main.pdf)和可编辑示意图 |
| [主数据](Open-SCORE/outputs/main) | 同类合并数据、必要final/best/config及共享实验图 |
| [归档说明](backup/README.md) | 与当前研究无关但独有的额外结构、旧协议、10维基线、SMACv2及其他结果 |

内部键`regir_nomem`用于加载LEAF已有权重；它表示当前query不读历史hidden，时序GRU与上一动作仍保留。七个历史设计比较的query条件不同，不能统称单因素消融。数据按来源、run、方法、种子、协议与实验臂区分，展示与目录不按日期版本组织。

## 当前成果

当前比较统计位于`had/aggregates.json`的`comparison`，设计比较取`design_comparison`，机制取`mechanism.normal/frozen/formal`。SPECTra和TransfQMix采用19维输入重训，各三个种子、每模型1M物理步，50个验证点及正式/确认各24配置×300场；共116,400条逐局记录已保存。正式All24 D分别为**6.2871±0.0289、4.2518±0.7733**。新终评未记录碰撞、拦截或受损事件比例，不套用原10维模型指标。

LEAF、七个设计比较及当前基线的权重位于`main/had/<方法>/models/seed_<种子>/`。保留已有`final.pt`、`best.pt`和`config.json`；best不替代正式final。仅用于旧10维SPECTra展示轨迹的seed0权重单列在`spectra/models/trajectory_entity10/seed_0/`，其真实artifact身份为`53e7f877dd9548a784e6d7376b42b7e2`。

正常与冻结机制原始记录统一在`had/regir_nomem/mechanisms/results.json.gz`，各数据池保留自身协议、配对和复用身份；展示轨迹为`had/trajectories.json.gz`。R1–R6补评有完整432单元，R4复用21,600场，新增108,000场的逐场JSONL未提供；不以现存R4逐场记录替代缺失出口数据。

八组实验图共同保存在`main/had/figures/`，每组PNG供报告阅读、PDF供论文排版。论文实验图片直接引用这一份来源。Git发布代码、论文、报告、聚合统计与图片；原始逐局数据、模型及恢复进度保留本地。

## 环境与安装

HAD物理来自独立SkyHAD/HAD依赖，项目仅由`open_score/envs/had_wrapper.py`访问原生实现。正式结果登记为`had-workbench-2.1.0 / rebuild-calibrated-v3-r7-target-initialization`；部分冻结干预和时延测量使用独立SkyHAD协议，其差异在主报告中保留，不能用当前源码反向替旧数据补写协议。

采用仓库内可编辑安装，配置位于`Open-SCORE/configs/`、命令入口位于`Open-SCORE/scripts/`：

```powershell
& 'D:\Software\Anaconda\envs\torch310\python.exe' -m pip install -e './Open-SCORE[training]'
```

`training`声明训练及绘图依赖，不负责选择CUDA构建。其他机器使用已经配置的Python/PyTorch环境并安装其原生HAD依赖。

## 训练、状态与正式评估

训练必须显式指定方法、种子和物理步预算。下面只展示入口，不表示启动新实验：

```powershell
python -X utf8 Open-SCORE/scripts/train.py --stage train --method regir_nomem,refil --seeds 0,1,2 --steps 1000000
python -X utf8 Open-SCORE/scripts/train.py --stage status
python -X utf8 Open-SCORE/scripts/train.py --stage stop
```

默认使用GPU 0、1，每张卡三项训练任务，共六槽；完成一项即补入下一项。独立评估池默认两个CPU worker，完成预算的final立即进入正式评估，不等待全部训练结束。只补齐同一协议未完成记录；恢复使用`--resume`，不抹掉已有进度。独立确认必须显式加`--confirmation`，不默认扩大评估配额。方法名称使用上述文档列出的内部注册键。

```powershell
python -X utf8 Open-SCORE/scripts/eval.py --stage final --method regir_nomem,refil --seeds 0,1,2
python -X utf8 Open-SCORE/scripts/eval.py --stage final --method regir_nomem --checkpoint Open-SCORE/outputs/main/had/regir_nomem/models/seed_0/final.pt
python -X utf8 Open-SCORE/scripts/eval.py --stage depth --method regir_nomem --seeds 0,1,2 --depths 1,2,3,4,5,6
python -X utf8 Open-SCORE/scripts/eval.py --stage anchors
python -X utf8 Open-SCORE/scripts/eval.py --stage confirmation --method regir_nomem --seeds 0,1,2
python -X utf8 Open-SCORE/scripts/eval.py --stage status
python -X utf8 Open-SCORE/scripts/eval.py --stage stop
```

`eval.py`接受单个checkpoint，或方法/种子列表；默认两个CPU worker。深度、锚点和确认按明确选择运行，不随final额外启动。状态与停止入口保留简表和可恢复进度。来源协议不同的数据分别保存和解释。

已有图表的唯一刷新入口为：

```powershell
python -X utf8 Open-SCORE/scripts/plot.py
```

它读取保存数据和聚合统计，不加载模型，不启动训练、环境回放或评估，不生成另一份报告。

## 回放与自定义比较

在仓库根目录运行`python -X utf8 viewer.py`，打开终端打印的地址。默认端口8765，已占用时顺延；`--port 8770`可指定端口。前后端协议为**API 5**，后端代码改变后重启服务。

红方可选checkpoint、端到端策略或分层规则；蓝方可选端到端或分层规则。请求采用`red_strategy/blue_strategy`对象，默认红方规则为`nv1 + predictive_intercept`，蓝方训练对手为`reactive + rush`：开局、每5步或发生伤亡时重新选择一个目标，全体存活蓝方冲向该目标。其他蓝方选项包括集火最近目标、均分攻击多目标和各自冲最近目标。nv1每物理步以一步匀速预测和0–1整数规划分配红方责任。

模型清单发现与`config.json`同目录的推理权重，`resume.pt`仅用于恢复训练。自定义checkpoint可填绝对或仓库相对路径，必须包含当前加载器使用的`config/networks/progress`与已注册`config.method`。已有七类主基线checkpoint适用于二维damage、每队最多50、目标最多12、现有episode_limit=100；具体限制以所选模型真实配置为准。

```python
red_strategy = {
    "architecture": "checkpoint",
    "checkpoint": "Open-SCORE/outputs/main/had/regir_nomem/models/seed_0/final.pt",
}
blue_strategy = {
    "architecture": "hierarchical",
    "layers": {"grouping": "reactive", "control": "rush"},
}
```

选择场景、对手、种子、统计局数与步数后，可把当前配置加入比较队列。各项按明确填写的局数依次运行；同条件比较校验场景、物理参数、对手、种子与时限。结果仅用于此次交互，保存于服务内存与浏览器sessionStorage，不进入正式数据或报告。

现有`register_end_to_end_policy(side, name, label, factory)`接受提供`reset()`和`act(state, side, action_ids)`的策略；通过`viewer.py --policy-module <模块名或文件>`导入注册代码。动作字典以全局ID映射原生动作ID，二维模型0–8索引需转换为传入的`action_ids`。

## 原生任务接口

```python
from open_score import make_env

env = make_env("defense", api="parallel", red_count=8, blue_count=8,
               target_count=2, task_mode="damage", spatial_dim=2,
               target_health=2.0, max_cycles=100, seed=0)
observations, infos = env.reset(seed=0)
```

damage每物理步红方奖励为负的新增目标伤害，整局未折扣回报为`R=-D`。目标持续存在，伤害不按初始血量封顶；蓝方全灭自然终止，红方全灭继续计入目标损伤，到时截断保留bootstrap。团队奖励不按队员人数重复求和。

survival在任一目标被毁或蓝方全灭时自然终止；空间维度默认2，旧三维接口必须显式传`spatial_dim=3`。物理参数、实体特征、死亡/padding与蓝方事件规则以[环境说明](docs/HAD_环境与问题建模.md)、[规则说明](docs/LEAF_Baseline与Rule实现说明.md#9-rule-与-random没有训练权重的两条锚点)及对应协议为准。
