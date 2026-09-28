# Open-SCORE (`remote_train` / main0928)

当前实验版本为 **main0928**：无 count 的 Looped 战役，3 种子，HAD + SMACv2。协议与命令见 [实验计划](Open-SCORE/outputs/main0928/实验计划.md)。

```bash
git clone -b remote_train git@github.com:Sailero/Open_Score.git
cd Open_Score
bash scripts/setup_farm.sh
source envs/local.env
bash Open-SCORE/outputs/main0928/run_all.sh start
```

HADE 与 SMACv2 源码在 `third_party/`。StarCraft II 4.10 由 `setup_farm.sh` 按暴雪 EULA 下载，不进 git。多机挂同一 `outputs/main0928` 后各跑同一条 `start`，任务自动抢占。


main0921 的 HAD 正式矩阵登记 15 个学习方法，SMACv2 登记 6 个方法；方法定义、权重身份与已完成结果分别见[实验计划](Open-SCORE/outputs/main0921/实验计划.md)和[正式报告](Open-SCORE/outputs/main0921/实验报告.md)。viewer 仍可发现本机保留的历史兼容权重。`best.pt` 是训练内验证选优档案，`final.pt` 是达到预算的终评权重；文件存在不代表该运行的正式评估已经结束。

旧研究主线已于 **2026-09-08 16:00:19 +08:00** 确认停止；探索 final 为 Git `cc2ab86`，历史研究源码可从该提交查阅。随后的环境标定与覆盖式规则已并入 [main v1](Open-SCORE/outputs/main_v1/实验报告.md)。该轮停止决定与“仅保留规则”的描述属于历史轮次。

## 环境与安装

HAD底层为独立包，源码随仓库 vendoring 在 [third_party/HADE](third_party/HADE/README.md)，协议 `had-workbench-2.1.0`。本仓库通过 `Open-SCORE/open_score/envs/had_wrapper.py` 访问其原生实现，不再依赖 sibling `../HADE`。历史 v7 使用 r2，不代表当前任务与规则的结果。覆盖规则评估在 `open_score.rules`；回放器为仓库根目录的 `viewer.py` 与 `viewer.html`。

跨规模实验使用扁平包 `Open-SCORE/open_score/`，通过仓库内可编辑安装运行。`open_score.envs.make_entity_env` 提供 damage 任务实体接口：当前训练池槽位上限为 10 红方、10 蓝方、3 目标，评估接口上限为 50 红方、50 蓝方、12 目标；每局训练规模独立采样，死亡和 padding 槽保留，模型动作编号为 0–8。仅逐步热路径绕开原生观测；reset 沿用 HAD 原生初始化。评估和规则锚点共享蓝方事件调度、物理过程、动作映射和诊断采集。

```powershell
& 'D:\Software\Anaconda\envs\torch310\python.exe' -m pip install -e './Open-SCORE[training]'
```

`had-env` 声明环境依赖，`training` extra 声明七方法训练与出图所需依赖。旧Windows示例解释器为上述 `torch310`；当前Linux入口采用main0921计划中的解释器。历史环境已安装 PyTorch 2.13.0+cu130、NumPy 1.26.4、SciPy 1.15.3、Matplotlib 3.10.9、PyYAML 6.0.3；extra 不负责选择 CUDA 构建。渲染及 Workbench 界面的安装方法见其 README。

本轮支持保留源码目录的可编辑安装。方法配置保留在包外的 `Open-SCORE/configs/`，命令入口保留在 `Open-SCORE/scripts/`；独立 wheel 不是当前支持的运行方式。vendored 主干及补丁内的 YAML 与已有 LICENSE 已声明为 package data，不复制包外配置来扩展发布体系。

## 公开接口

| 导入 | 用途 |
|---|---|
| `open_score.envs.make_entity_env`、`HADEntityEnv` | damage 跨规模实体环境；10 维基础特征、固定槽与原生动作映射 |
| `open_score.algos.train(name, cfg)` | 七方法共享训练入口；预算必须显式指定 |
| `open_score.algos.load_policy(name, checkpoint)` | 冻结 checkpoint 转为 `reset()`／`act(state, side, action_ids)` 策略，可注册到规则评估与回放接口 |
| `open_score.make_env`、`open_score.parallel_env` | 原生 Parallel / MPE 环境工厂 |
| `open_score.rules.KnownOpponentEnv`、`make_grouping_env` | 原生事件式分组环境；后者默认规则执行器 |
| `Entity`、`Group`、`Grouping`、`DecisionState`（均位于 `open_score.rules`） | 实体、分组动作与公开状态，沿用原生序列化与合法性约束 |
| `RuleExecutor`、`RulePolicy` | 共享规则底层与上层规则策略 |
| `CoveragePolicy`、`CoverageExecutor`、`run_episode`、`evaluate` | 当前 `rule_nv1_v1`：每步整数规划分组与匀速直线预测 |
| `rule_grouping`、`grand_grouping`、`decode_counts` | 规则分组、每目标大组及人数分配解码 |

规则策略支持 `rule`、`grand`、`random`、`static_rule`；`reset()` 重置策略状态，`act(state)` 返回原生 `Grouping`。项目代码统一通过 `open_score.envs` 访问固定对手和物理适配器，只有 `had_wrapper.py` 直接导入 HAD。

网页首页直接提供红方算法与模型选择；规则分层仍可选择。红方规则上层 `nv1` 每步用 `p + v × Δt` 预测蓝方下一步位置，通过 0–1 整数规划最小化总拦截距离；下层 `predictive_intercept` 将指派转为飞行动作。每个红方恰好负责一个蓝方、每个参与分组的蓝方至少分配一个红方。红方不足时仅纳入最近的 R 个蓝方。蓝方上层默认 `reactive`，每 5 步或伤亡时重新选目标，下层 `rush` 每步控制飞行。完整定义见 [现行规则说明](docs/three_layer_rule_mathematical_spec.md)。旧覆盖策略别名和历史策略选择入口已移除，历史结果仍保留原记录。

## 可视化回放与自定义比较

在仓库根目录运行：

```powershell
& 'D:\Software\Anaconda\envs\torch310\python.exe' -X utf8 .\viewer.py
```

打开终端实际打印的地址：默认 `http://127.0.0.1:8765/`；若已被旧服务占用，会自动改用 8766 等空闲端口。也可通过 `--port 8770` 明确指定端口。当前前后端协议为 **API 5**，修改 Python 后端后需要重新启动；旧服务不会自动更新，访问旧地址仍会得到旧模型目录。首页直接显示红方模型、蓝方对手和基本评估参数，模型筛选、任务细节、自定义路径与比较队列默认折叠。红方可选择训练权重（`checkpoint`）、端到端（`end_to_end`）或规则分层（`hierarchical`）；蓝方保留端到端与规则分层。XY/XZ 图等比例绘制，damage 显示累计伤害及红方回报，survival 显示胜负指标。

红方选择训练权重后，可按版本、方法、训练种子和 `best/final/latest` 筛选。清单自动发现输出目录中与 `config.json` 同目录的保留模型，点击刷新可读取新增权重；`resume.pt` 专用于恢复训练，不进入可选推理清单。每项显示来源与适用限制，历史不兼容资产仍可查看其清单和原因。自定义路径接受本机 `.pt` 文件的绝对路径或仓库相对路径，加入当前服务会话后即可选择；文件应包含当前加载器使用的 `config/networks/progress` 字段，`config.method` 必须对应已有算法实现。只添加可信本地权重，加载使用 PyTorch checkpoint 反序列化。

当前七种 checkpoint 都只控制红方，适用于 **二维 `damage`、红蓝各最多 50、目标最多 12、`max_steps` 不超过模型的 `episode_limit`（现有权重为 100）**。QMIX-Base 的 `pool_slots=[10,10,3]` 进一步限制为最多 10 红、10 蓝、3 目标。旧 ALMA（`n_extra_tasks=3`）不能评 K>6；新 ALMA 按 `n_extra_tasks=9` 训练。viewer 使用 CPU 顺序推理，并按每局种子重置模型状态和 PyTorch 随机流。原生 FF / `rel_overgen` 的 E0 模型不适用于 HAD 回放。

蓝方训练对手是首页默认的 **自适应集火单目标（训练）**，对应 `reactive + rush`：每 5 步或发生伤亡时，根据红方防守覆盖和蓝方距离抽取一个目标，全部存活蓝方冲向该目标，下一次决策可能换目标。**集火最近目标**（`concentrated + rush`）同样全队集中，但按蓝方平均距离选择最近目标；**均分攻击多目标**（`balanced + rush`）将人数均分给各目标。另一个动作基线 **各自冲最近目标** 则是各架飞机独立选最近目标。三个分组策略都可在首页直接选择，飞行控制细选保留在更多设置中。

设置场景、蓝方、回放种子与统计局数后，可单独运行，或将当前配置加入比较队列，再切换红方算法、训练种子或 checkpoint 添加其他项。队列保留各项配置，依次执行每项明确填写的统计局数（API 字段 `stats`），不自动扩大局数或种子范围。比较同一场景下的红方算法时，使用相同蓝方策略、物理参数、种子、统计局数与步数上限；同条件比较会校验这些条件。结果保存在服务内存与浏览器 `sessionStorage`，用于本次交互，不写入正式训练 CSV 或实验报告；服务重启后内存任务不保留。

API 5 通过 `red_strategy/blue_strategy` 传策略，切换架构后只提交该架构字段。例如，viewer 请求中的红方模型选择为：

```python
red_strategy = {
    "architecture": "checkpoint",
    "checkpoint": "Open-SCORE/outputs/main_v3/refil/train/seed_0/best.pt",
}
blue_strategy = {
    "architecture": "hierarchical",
    "layers": {"grouping": "reactive", "control": "rush"},
}
```

`checkpoint` 是 viewer 的加载入口；直接调用 `run_episode` 时，应先用 `load_policy` 加载并注册为 `end_to_end` 策略。旧 `red_policy/blue_rule/blue_opponent/blue_style` 参数不再接受。

不同网络结构或自行实现的算法可使用已有的 `register_end_to_end_policy(side, name, label, factory)` 接口。将注册代码放入自己的模块，在服务启动时通过 `--policy-module` 导入（支持模块名或本地 Python 文件，可重复指定）：

```powershell
& 'D:\Software\Anaconda\envs\torch310\python.exe' -X utf8 .\viewer.py --policy-module my_policies
```

模块导入时注册策略，例如 `register_end_to_end_policy("red", "my_policy", "我的算法", lambda seed: MyPolicy(seed))`。`factory(seed)` 返回提供 `reset()` 和 `act(state, side, action_ids)` 的独立对象。`state` 是公开物理状态，不含对手本步指令；`act` 返回 `{全局智能体ID: 原生动作ID}`，二维模型的 0–8 动作索引需通过 `action_ids` 转换。注册完成后，表单从 `STRATEGY_CATALOG` 生成对应队伍的端到端选项。自定义分层网络也可在该对象内部完成多层决策；新增规则层选项则需同时实现相应执行逻辑。

当前数值基准：双方无人机 HP=1、满伤=1、速度范围 35–120、最大加速度 40；二维满伤 / 自动开火半径 200、归零半径 400；三维对应为 300、600；碰撞距离 20。存活任务目标默认 HP=2；damage 目标不扣血、不封顶。双方性能一致用于减少比较中的隐藏数值优势；尚未通过新胜率评估证明任务平衡。二维 200–400、三维 300–600 内的范围伤害仍按原线性曲线衰减，不把每个波及目标强行算为一次满伤。

目标初始血量可逐环境传 `target_health=2.0`。公共默认值在 [HAD 配置](../Open_Score_HAD_Workbench/had_env/core/config.py)：`initial_health` 是目标初始血量，`AttackIntensity` 是伤害强度，`PlanarAltitude` 是二维平面高度，速度、加速度、攻击距离和世界边界等也在此文件。网页和训练默认 `target_initialization="random"`，每局按 `DefaultTargetRegion` 采样目标；可切换 `"fixed"`，以区域中心 x/z、内部等距 y 生成固定布局。二维统一使用平面高度；三维目标高度默认 500–1500 米。显式 `target_positions` 可指定固定坐标，默认随机训练不传它。修改源码后须在已有任务结束后重启 `viewer.py`；网页检查 API 版本以避免新表单被旧服务忽略。

## 两种任务与 MARL 接入

```python
from open_score import make_env  # 等价于 from had_env import make_env

env = make_env(
    "defense", api="parallel", red_count=8, blue_count=8, target_count=2,
    task_mode="damage", spatial_dim=2, target_health=2.0,
    max_cycles=100, seed=0,
)
observations, infos = env.reset(seed=0)
# joint_actions 包含当前 env.agents 的所有红、蓝智能体。
# observations, rewards, terminated, truncated, infos = env.step(joint_actions)
```

- `spatial_dim=2` 是默认值：所有实体在同一高度运动，垂直速度与加速度为零。Parallel/MPE 提供 9 个离散平面动作或 2 维连续动作。`spatial_dim=3` 恢复 27 个离散动作或 3 维连续动作。观测仍保留 xyz/vxyz 的 11 维实体结构；三维旧模型必须显式选择三维并核对其协议。
- `task_mode="survival"` 是工厂兼容默认值，任一目标被摧毁即旧任务结束，保留旧奖励契约。需要旧三维场景时同时传 `spatial_dim=3`。
- `task_mode="damage"` 是网页默认任务：目标保持有限的初始血量并持续存在，累计未被血量截断的实际伤害。每物理步红方奖励为 `−新增目标伤害`，蓝方为相反数；整局未折扣回报等于 `−累计伤害/+累计伤害`。累计 5.5 即红 −5.5、蓝 +5.5；终局不另加奖励。此模式的 `target_health` 不设伤害上限。
- 伤害任务仅蓝方攻击机全灭自然终止；红方全灭继续运行。`max_cycles/max_steps` 为采样上限，达到时截断，`bootstrap_mask=1`；自然终止为 0。若训练目标就是固定时长的有限和，应显式采用时间特征及相应终端价值边界，不能把采样截断误当自然终止。
- 伤害任务保留固定槽到整局结束。死亡槽零观测、强制空动作，但继续接收团队奖励。`agent_mask` 用于 actor，`bootstrap_mask` 用于团队 critic。死亡当步更新使用动作产生前的存活掩码。同队 `team_reward` 已是同一份团队奖励，不能再按队员相加。
- 累计伤害、时间和维度通过 `info` 提供。任意红蓝 MARL 策略可接 Parallel 联合动作字典，或接 MPE 固定顺序动作列表；网页通过 `red_strategy/blue_strategy` 对象传入统一入口。分组适配器保留原 27 个动作编号，二维只从其中 9 个平面方向选择，与训练接口的 0–8 编号勿混用。

此前环境接入轮次的 Parallel RNG、实体缓存和伤害距离过滤调整属于历史环境修改；当前跨规模训练与评估状态以[main0921 正式实验报告](Open-SCORE/outputs/main0921/实验报告.md)为准，研究判断见[项目核心分析与论文定位](docs/当前实验分析与论文主张讨论.md)。完整原生接口见 [HAD API](../Open_Score_HAD_Workbench/docs/API.md)。

```python
from open_score.rules import make_grouping_env, RulePolicy

env = make_grouping_env(red=8, blue=8, targets=2, opponent="reactive", seed=0)
policy = RulePolicy("rule", seed=0)
state = env.reset()
policy.reset()
grouping = policy.act(state)
state, reward, done, info = env.step(grouping)
env.adapter.env.close()
```

上述示例说明原生分组接口。分组环境 `survival` 保留原生时限终止与 0/1 奖励；传入 `task_mode="damage"` 后返回整个宏步的伤害奖励之和，并区分自然终止与截断。Parallel/MPE 工厂采用自己的时限语义。多目标位置、零人数、事件返回和快照格式均以 Workbench 原生 API 为准；v6 包装、奖励塑形、S2 投影及旧训练张量接口已归档。

## 查看运行、停止与恢复

默认输出目录为 `Open-SCORE/outputs/main/`，正式运行标识为 `train`。合并记录通过版本、方法、运行和种子字段区分；模型、配置与控制台日志按 `<方法>/train/seed_<种子>/` 保存。唯一正式 Markdown 报告为该目录的 `实验报告.md`，图表放在 `figures/`。训练终端定期显示进度、吞吐、loss、评估进度和最近完整验证 D；报告按已落盘数据原位刷新。

当前训练入口使用 `--stage train` 或 `--stage single`。`--group main` 按 inventory 扫描未完成的 ReGIR / ALMA / 消融 / matched 任务，并发上限 2，但 ReGIR 类按显存一次只跑 1 路；终评与深度扫描由 `eval.py` 负责，最多 2 路。预算必须显式传 `--steps`；现有正式权重对应 100 万物理步、8 worker。学习奖励默认 `--reward-mode damage`（仅 −ΔD）。

启动两条、停止两条，训练和评估互不影响。均在仓库根目录 `E:\Code\Open_Score` 执行：

```powershell
& 'D:\Software\Anaconda\envs\torch310\python.exe' -X utf8 .\Open-SCORE\scripts\train.py --stage train --group main --steps 1000000 --batch-size-run 8 --resume --max-concurrent 2 --output .\Open-SCORE\outputs\main
& 'D:\Software\Anaconda\envs\torch310\python.exe' -X utf8 .\Open-SCORE\scripts\eval.py --group main --resume --max-concurrent 2 --output .\Open-SCORE\outputs\main
```

```powershell
& 'D:\Software\Anaconda\envs\torch310\python.exe' -X utf8 .\Open-SCORE\scripts\train.py --stage stop --output .\Open-SCORE\outputs\main
& 'D:\Software\Anaconda\envs\torch310\python.exe' -X utf8 .\Open-SCORE\scripts\eval.py --stage stop --output .\Open-SCORE\outputs\main
```

恢复未完成项需要同目录存在可读的 `resume.pt`（或 `resume.prev.pt`），并加上 `--resume`。调度器跳过已有完成记录与保留模型的任务。只处理一个任务时使用 `--stage single`。

查看 ReGIR 种子 1 的日志：

```powershell
Get-Content -LiteralPath '.\Open-SCORE\outputs\main\regir\train\seed_1\console.log' -Encoding UTF8 -Tail 30 -Wait
```

从已有记录刷新同一份报告和图表：

```powershell
& 'D:\Software\Anaconda\envs\torch310\python.exe' -X utf8 .\Open-SCORE\scripts\plot.py --run train
```

训练每 10 分钟以及每个验证点会把完整 `resume.pt` 刷盘，并留下上一份 `resume.prev.pt`。意外断电后 `--resume` 会先读 `resume.pt`，读不了再用 pending 或上一份。计划关机用上面的两条 stop：训练写完当前 batch，评估写完当前局。等待状态显示 `stopped` 与可恢复步数后再关闭终端。

若原任务使用了 `--output`，停止与恢复也应指定相同目录。恢复时保留原方法、环境、种子、`--steps`、`--batch-size-run`、运行标识和输出目录，并增加 `--resume`。恢复点包含 replay、优化器、目标网络、随机流和进度；`best/final/latest.pt` 用于模型查看，不能替代恢复点。控制台步数可以新于可恢复步数，最多差一个保存间隔。上一轮 E0 与吞吐矩阵见 [main v2](Open-SCORE/outputs/main_v2/实验报告.md)。

## 历史研究

[实验记录](实验报告/实验记录.md) 维护跨版本摘要。当前正式对照归 [main](Open-SCORE/outputs/main/实验报告.md)；上一轮循环结构对照见 [main v5](Open-SCORE/outputs/main_v5/实验报告.md)；v4 距离分组与带 LN 的数量条件见 [main v4](Open-SCORE/outputs/main_v4/实验报告.md)；v3 完整五臂对照见 [main v3](Open-SCORE/outputs/main_v3/实验报告.md)；上一轮跨规模失败结果见 [main v2](Open-SCORE/outputs/main_v2/实验报告.md)；S1–v7 合并见 [main v1](Open-SCORE/outputs/main_v1/实验报告.md)；ALMA 适配补充见 [alma_probe_v3](Open-SCORE/outputs/alma_probe_v3/实验报告.md)。

截至目录重组后，`outputs/main_v3` 保留完整训练权重；`main_v4` 保留上一轮新臂；`main_v5` 写入四条全局臂，去 LN 的 count 仍占用同目录的另一方法名。`main_v2` 保留逐局数据与报告、无 `.pt`。`main_v1` 与 `alma_probe_v3` 只留报告、图和记录。`assets/frozen` 仍有历史 `lcl.pt`、`dlom.pt`，当前 `load_policy` 不加载旧格式。

表内路径相对 `Open-SCORE/`。历史资产仍在本地，viewer 可列出其来源与不兼容原因，但当前 `load_policy` 不加载旧 `model/actor/policy/online` 格式。部分历史 checkpoint 的反序列化还依赖已归档的 `open_score.grouping` 或 `open_score.envs.had_stage3` 模块。使用旧研究权重需同时恢复对应模型代码、配置与历史环境；`dlom.pt` 本身是结果预测器，不是完整动作策略。历史研究源码入口为探索 final `cc2ab86`，之后的修订应以相应报告和权重来源为准。多数输出资产被 Git 忽略，不属于该提交的备份范围。

原始研究修改稿与 v6 原执行方案可分别从 Git `cc2ab86:固定对手下动态分组研究方案.md`、`cc2ab86:dynamic_grouping_v6_execution_plan.md` 查阅。各版实际完成量、取消项和结论以对应 `outputs/main_v*` 报告为准。
