# Open-SCORE

主方法为 **LEAF-NoMem（`regir_nomem`）**。现有实验归并为两个结果目录：

| 结果 | 内容 | 正式分析 |
|---|---|---|
| [outputs/main](Open-SCORE/outputs/main) | NoMem、原 ReGIR 训练变体及 HAD 基线 | [主实验结果分析](docs/LEAF_主实验结果分析.md) |
| [outputs/supp](Open-SCORE/outputs/supp) | M1–M5 循环候选、SMAC 与其他探索 | [补充实验结果分析](docs/LEAF_补充实验结果分析.md) |

配套阅读：[方法与变体说明](docs/LEAF_方法与变体说明.md)讲解 NoMem 的完整计算、ReGIR 家族与五个循环候选的结构；[Baseline 与 Rule 实现说明](docs/LEAF_Baseline与Rule实现说明.md)解释七个学习基线、规则指派、随机动作及共同对手。两份说明与结果报告配套，区分实际实现、设计动机和已完成的实验证据。

数据保留来源版本、运行、环境协议、种子和实验臂；不同协议的成绩分别解释。权重位于 `<结果目录>/<环境>/<方法>/models/<训练来源>/seed_<种子>/`，保留 `final.pt`、`best.pt` 与加载配置。训练完成情况由权重 metadata 与原登记分别记录。数据、来源映射及历史表统一见各目录的 `summary.json`，实验协议见 `experiment.json`。正式归并成果以这两份报告和元数据为准。

归并前的完整本地保存点为 Git `8d8fa28`，包含当时的实验数据与权重。2026-10-11 已完成旧来源、过时文档、临时文件及24个空CSV的清理，`outputs` 仅保留 `main/supp`，合计约2.80 GB。删除前核对并补齐4张独有历史曲线及1,154条科学诊断变体，正式成绩不变；归档保留159个独有final/best及加载配置。来源映射、重复计时摘要和完整清理记录见两目录 `summary.json` 的 `archive.retention_verification` 与 `archive.cleanup`。[实验记录](实验报告/实验记录.md) 维护跨版本摘要；论文包仍位于 [Papers](Papers)。本次整理尚未Git提交或推送，本地保存点不代表远端已备份归并版本。

## 环境与安装

HAD底层为独立仓库 [HADE](../HADE/README.md)，本仓库通过 `Open-SCORE/open_score/envs/had_wrapper.py` 访问其原生实现。本机依赖路径为 `/data3/dell/Saileron/projects/HADE`，当前协议为 `had-workbench-2.1.0` / `rebuild-calibrated-v3-r7-target-initialization`。历史 v7 使用 r2，不代表当前任务与规则的结果。覆盖规则评估在 `open_score.rules`；回放器为仓库根目录的 `viewer.py` 与 `viewer.html`。

跨规模实验使用扁平包 `Open-SCORE/open_score/`，通过仓库内可编辑安装运行。`open_score.envs.make_entity_env` 提供 damage 任务实体接口：当前训练池槽位上限为 10 红方、10 蓝方、3 目标，评估接口上限为 50 红方、50 蓝方、12 目标；每局训练规模独立采样，死亡和 padding 槽保留，模型动作编号为 0–8。仅逐步热路径绕开原生观测；reset 沿用 HAD 原生初始化。评估和规则锚点共享蓝方事件调度、物理过程、动作映射和诊断采集。

```powershell
& 'D:\Software\Anaconda\envs\torch310\python.exe' -m pip install -e './Open-SCORE[training]'
```

`had-env` 声明环境依赖，`training` extra 声明七方法训练与出图所需依赖。旧Windows示例解释器为上述 `torch310`；Linux使用所在机器已配置的环境。历史环境已安装 PyTorch 2.13.0+cu130、NumPy 1.26.4、SciPy 1.15.3、Matplotlib 3.10.9、PyYAML 6.0.3；extra 不负责选择 CUDA 构建。渲染及 Workbench 界面的安装方法见其 README。

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
    "checkpoint": "Open-SCORE/outputs/main/had/regir_nomem/models/main0928/seed_0/final.pt",
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

此前环境接入轮次的 Parallel RNG、实体缓存和伤害距离过滤调整属于历史环境修改；当前主方法的结果与判断见[主实验结果分析](docs/LEAF_主实验结果分析.md)，循环候选及其他环境见[补充实验结果分析](docs/LEAF_补充实验结果分析.md)。完整原生接口见 [HAD API](../Open_Score_HAD_Workbench/docs/API.md)。

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

## 已归档实验

2026-10-13新增的基线输入对齐实验使用 `Open-SCORE/scripts/main1013_base.py`，仅重训SPECTra/TransfQMix各三个种子。actor与mixer补齐红方上一动作，训练与评估采用最终LEAF的共同HAD协议，同时保留基线自身优化器和TD目标。代码与远程入口已准备；尚未启动新训练或评估。唯一报告、数据和必要模型统一位于 [outputs/main_1013_base](Open-SCORE/outputs/main_1013_base)，详情见[该版本报告](Open-SCORE/outputs/main_1013_base/实验报告.md)。在远程 `Open-SCORE` 目录运行：

```bash
mkdir -p outputs/main_1013_base
nohup python -u scripts/main1013_base.py run --devices 0,1 --jobs-per-gpu 3 --max-jobs 6 --resume > outputs/main_1013_base/run.log 2>&1 &
python scripts/main1013_base.py progress --watch-seconds 3
```

新入口会核对HAD版本和物理协议。停止用 `python scripts/main1013_base.py stop`，恢复重新执行上述 `run --resume`；关闭只读进度面板不停止任务。已有远程main1013运行不受新版本影响。

结果目录保存已有数据与推理权重，旧任务启动、恢复和报告刷新命令已停用。历史单轮从头训练与多轮权重的单轮推理是两类不同对照，详见主报告。独有历史数据保留来源协议，未提供的原始记录如实注明。

环境建模见 [HAD 环境与问题建模](docs/HAD_环境与问题建模.md)，早期方法移植差异见 [迁移记录](docs/方案_v2/diff_log.md)。更早研究源码可从 Git `cc2ab86` 查看；本次归并不改变 HAD 物理接口或算法实现。
