# HAD-Env

独立的异构攻防博弈仿真环境。根目录 `make_env.py` 是统一入口，支持 MPE 风格的场景组织、
PettingZoo 同时动作接口，以及固定列表兼容接口。物理核心保留 HAD 的运动、碰撞、规则开火、伤害与自毁时序。

本项目不依赖 Open-SCORE、研究训练脚本或冻结模型。默认安装不需要 Torch、Pygame 或 Qt。
分组实验、完整回放和 Qt 科研工作台保留在独立子模块中。

## 安装

Python 3.10+：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
# 按需安装：Pygame 渲染，或完整科研界面/导出/测试
.\.venv\Scripts\python.exe -m pip install -e ".[render]"
.\.venv\Scripts\python.exe -m pip install -e ".[viewer,test]"
```

本机专用解释器为 `.venv\Scripts\python.exe`，与原研究环境分开。
也可以从其他项目安装本仓库：`python -m pip install -e E:\Code\Open_Score_HAD_Workbench`。

## 统一接口

在项目根目录或者安装后：

```python
from make_env import make_env  # 也支持 from had_env import make_env

env = make_env("defense", spatial_dim=2, red_count=8, blue_count=8, max_cycles=100)
obs, infos = env.reset(seed=42)
try:
    while env.agents:
        actions = {name: env.action_space(name).sample() for name in env.agents}
        obs, rewards, terminated, truncated, infos = env.step(actions)
finally:
    env.close()
```

双方按同一时刻的观测提交一个动作字典，每次 `step` 推进一个原生物理步。公开实体名为
`red_0...`、`blue_0...`，`possible_agents` 固定。默认 `survival` 中，死亡实体返回最后一次转换后移出 `agents`；
`damage` 中，双方全部槽位保留到全局终止或截断，死亡槽位仍领取后续团队奖励。
`state()` 返回集中训练可用的全局状态，动作/观测空间在一个环境实例内固定。
空间默认是二维：所有实体固定在 `z=100` 平面，z 速度和加速度为零。
三维策略和已有 27 动作模型必须显式设置 `spatial_dim=3`，不能沿用默认构造后直接加载。

| 参数/返回值 | 语义 |
|---|---|
| `red_count`、`blue_count` | 各队总人数，包含所有异构角色；默认各 4 |
| `red_scouts`、`red_disturbers` 等 | 从总人数中划分角色，剩余为 Attack；蓝方至少一名 Attack |
| `target_count` | 保护目标数，默认 2 |
| `spatial_dim=2` | 默认二维平面；`spatial_dim=3` 使用三维出生和飞行 |
| `target_initialization="random"` | 训练默认，每局重新采样目标；`"fixed"` 使用固定布局 |
| `target_positions=None` | 可显式传入每个目标的固定 `[x,y,z]` 坐标；传入后使用固定模式 |
| `task_mode="survival"` | 旧默认：目标扣血，任一目标毁伤或蓝方 Attack 全灭决定自然终局 |
| `task_mode="damage"` | 目标不扣血、不死亡；红队每步为新增目标伤害的负值，蓝队为正值 |
| `target_health=None` | 默认读取环境目标 HP；显式值必须为正有限数。在 damage 中仅保留为目标显示/观测 HP，不限制伤害得分 |
| `continuous=False` | 二维 `Discrete(9)`，三维 `Discrete(27)`；二维基元从原 27 个按 z=0 筛出，零号均为零加速度 |
| `continuous=True` | 二维为 `(2,)`、三维为 `(3,)`，每维有限且在 `[-1,1]`；二维解码补 z=0，再乘原生加速度上限 |
| `render_mode` | `None`、`human`、`rgb_array`；图形依赖按需导入 |
| `max_cycles` | 原生飞行采样上限，默认 100；到时仅截断，不额外判胜 |
| 观测 | `(总实体数-1, 11)`：相对位置/速度、health、alive、红/蓝/目标标志；死亡行置零，`entity_mask` 在 info |
| `infos[a]["observation_entities"]` | 观测每行对应的实体名，死亡槽位仍保留 |
| `rewards[a]` | 每个槽位各自收到完整团队奖励；同队奖励不能求和当作团队回报。四维 `LatentReward` 同时保存在 info |
| `reward_weights` | 仅 survival 可显式设置四维潜在奖励权重；episode 分量已含 RealReward，不应再重复相加。damage 拒绝此参数 |

全态势观测和规则开火保持原样，侦察角色不会在这个接口中自动引入新的隐藏观测。
精确的奖励、终局及物理协议说明见 [docs/API.md](docs/API.md)。

伤害任务沿用同一调用循环，只需更改构造参数：

```python
env = make_env("defense", task_mode="damage", target_health=2.0, spatial_dim=2,
               red_count=8, blue_count=8, max_cycles=100)
```

damage 的目标 HP 始终保持正有限初值。每步实际计算的原始目标伤害累加到 `target_damage`，
不按目标剩余 HP 截断，不封顶，也不做归一化；同一步多来源伤害全部计入。
`infos[a]["episode_returns"]` 为累计 `{"Red": -target_damage, "Blue": target_damage}`，
每个队员的逐步奖励累计等于自己队伍的这一值。蓝方 Attack 全灭时自然终止，
红方全灭不会提前结束目标受损过程；`max_cycles` 到时仅截断，不判胜。

训练时，`agent_mask`/`alive` 表示返回观测对应的存活状态；死亡槽位观测和 `entity_mask` 清零，
离散 `action_mask` 只开放零号动作，连续模式强制零向量。
`bootstrap_mask = 1 - global_terminated` 是团队 critic 的掩码，不能用存活掩码替代：
单个 agent 死亡或采样截断后仍应保留 bootstrap，只有自然终止清零。
死亡当步动作的 actor 更新使用前一观测的存活掩码，避免删掉造成死亡的真实动作。
每实体 11 维观测和集中 state 维度保持兼容；`cycle`、`max_cycles`、`remaining_cycles` 在 info，
没有自动加入模型输入。若训练有限时限目标，调用方需显式向模型提供时间信息。
`spatial_dim`、`plane_altitude` 同样在 info；二维的局部相对 z/垂直速度为零，集中状态保留固定平面高度。
保持 11 维形状不等于模型协议兼容，二维/三维动作空间和几何分布必须分别配置。

旧 MPE 的固定列表调用可以使用：

```python
env = make_env("defense", api="mpe", red_count=4, blue_count=4)
obs_n = env.reset(seed=42)
obs_n, reward_n, done_n, info = env.step([0] * env.n)
env.close()
```

此接口提供 `.n`、列表 `.action_space` / `.observation_space` 和展平观测；
离散动作接受整数或匹配当前动作空间的严格 one-hot：二维 9 维、三维 27 维。
survival 已死亡槽位在最后一次转换后返回零观测、零奖励和 `done=True`；
damage 死亡槽位返回零观测，持续收到团队伤害奖励，并在全局终止或截断时才置 `done=True`。
这是一层调用约定适配，不保证所有旧 MADDPG 代码无需调整即可运行。

## 科研工作台

```powershell
.\run_workbench.ps1
# 或
.\.venv\Scripts\python.exe -m had_env.workbench live
```

默认打开独立、暂停的 8v8 分组调试会话。支持播放/单物理步/决策边界/伤亡跳转、XY 战场与 XZ 高度、
缩放选择、编组和任务关系、真实事件、决策详情、两策略比较及分支。关闭窗口只停止它拥有的调试进程。

```powershell
had-workbench record --policy rule --output outputs/rule.json.gz
had-workbench record --policy grand --output outputs/grand.json.gz
had-workbench view outputs/rule.json.gz --compare outputs/grand.json.gz
had-workbench branch outputs/rule.json.gz --step 5 --policy grand --seed 99 --output outputs/branch.json.gz
had-workbench export outputs/rule.json.gz outputs/trajectory.svg --step 10
had-workbench export outputs/rule.json.gz outputs/replay.mp4
had-workbench evaluate examples/evaluation.json --output outputs/evaluation
```

决策步会受伤亡影响，用 `had-workbench inspect` 确认分支位置。新录制含真实物理帧和受约束的 JSON 快照；
旧 v4/v5 稀疏日志仍能只读打开。不同源码的录制可查看，但离线续演严格要求行为指纹匹配。
迁移前的旧录制不会被冒充为当前版本的可恢复快照。

PNG/SVG/PDF/MP4 由 CLI 从记录生成，并保存来源 sidecar；GUI 的“导出画面”保存当前战场布局。
分支和播放不会修改原轨迹。关闭实时工作进程有 2 秒保存退出期限，长策略/慢磁盘时末次 partial 可能未写完；
正式评估使用 `record` / `evaluate` 并等待正常完成。

算法通过标准 Parallel API 接入；上层分组算法也可使用
`had_env.workbench.session.SimulationSession` 和 `register_policy(name, factory)`。
`FlightSession(red_policy=..., blue_policy=...)` 可分别接入两队飞行动作策略，
也可用 `step({"red": ..., "blue": ...})` 显式提供两队固定 roster 的全部动作。
上层分组的红方接口是 `act(state) -> Grouping`；蓝方由 `opponent` 决定分组，
`KnownOpponentEnv` 使用 `rush` 底层规则，直接使用 `HADStage3Adapter` 时可选择 `blue_style`
或提供 `blue_action_ids`。
学习模型由调用方提供，不随环境捆绑。使用 PyTorch 策略时，先导入 Torch；GPU 模型和 CUDA 上下文应在
创建 `PolicyAdapter` 前准备好，避免在第一次动作内初始化新的随机源。

## 精简后的结构

```text
make_env.py             统一入口
had_env/
  factory.py            场景与 API 选择
  environment.py        Parallel / MPE 调用适配
  scenarios/defense.py   场景构建、重置、观测和奖励回调
  core/                 HAD 原生物理与可选 Pygame 渲染
  grouping/             分组状态、规则对手、控制执行及快照
  workbench/            记录、评估、分支、Qt 与导出
examples/               最小调用与评估配置
tests/                  独立测试及旧物理基准
docs/                   API 与验收说明
```

输出、虚拟环境、缓存、训练检查点均不进 Git。原研究项目的训练方案、实验记录、模型和运行器已经移出本仓库。

## 验证与来源

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

测试包括 PettingZoo 官方 `parallel_api_test` / `parallel_seed_test`、原始物理黄金轨迹、
分组规则迁移对照、随机隔离、Qt 控件、真实 spawn 进程、记录分支和导出。
具体结果与版本来源见 [docs/VALIDATION.md](docs/VALIDATION.md)。
本轮任务模式与接口修改仅作静态审查、语法检查，未运行这些测试、仿真评估或性能探测；
历史验证和吞吐结果不代表新模式已经验收。

组织方式参考 [原 MPE 的 make_env/Scenario](https://github.com/openai/multiagent-particle-envs/blob/master/make_env.py)，
同时动作接口依据 [PettingZoo Parallel API](https://pettingzoo.farama.org/api/parallel/)。
参考的是模块与接口组织，HAD 的攻防动力学仍来自本项目原实现。

## 物理协议变更

`CORE_VERSION = had-workbench-2.1.0`，`PHYSICS_PROTOCOL = rebuild-calibrated-v3-r7-target-initialization`。
本轮在 v3-r2 物理上增加显式 survival/damage 任务、二维/三维配置、目标伤害计数和任务快照约束；
不同维度、任务、HP、时限或物理协议的快照不能混用，旧黄金轨迹与历史实验结果不作为当前协议的新结果。

v3-r2 已采用的物理规则和实现（本轮继续使用）：

- 开火与命中都用移动前位置。
- 红方开火改为射程内有蓝方，蓝方改为射程内有目标；不再把自身计入密度判据。
- 合成速度近零时保持航向，不再 180° 掉头。
- 位置触界时清掉指向界外的速度分量。
- 碰撞改为扫掠检测，本轮已死者不再链杀。
- 三维初速度方向为球面均匀，速率边缘分布仍按三分量均匀模长；本轮默认二维改为平面方向，并固定高度。
- `rotate_restrict_velocity` 使用确定性正交轴；当前 `wMax=π` 下该分支仍不进入。
- 每步不再 `deepcopy` 整个世界，改为 `WorldKinematics` 数组快照。
- 观测相对量按场地对角线与 \(2v_{\max}\) 缩放。

当前默认红蓝双方均为 \(v_{min}/v_{max}/a_{max}=35/120/40\)，蓝速度与加速度系数均为 1；无人机 HP=1，`AttackDistance={2: [200,400], 3: [300,600]}`、`AttackIntensity=1`，存活任务目标 HP=2，碰撞距离 20，\(H=100\)。红方满伤拦截扣蓝方 1 点，蓝方满伤命中目标记 1 点；二维 200–400、三维 300–600 范围伤害仍线性衰减。数值一致是可解释的初始基准，本次没有运行胜率标定。此前 1.2 伤害 / 2.2 目标 HP / 蓝方系数 1.25 的记录属于旧协议。
二维红方仍从同一目标环带采样；越界候选点会被拒绝并重新采样，不会压到地图边界。每个位置最多尝试 4096 次，无法取得合法点时明确报错；三维保留原有边界裁剪行为。二维开局间距由 `PlanarSpawnSeparation=30` 约束，严格大于碰撞半径，不保证后续不会碰撞。
原生 Parallel 的 `max_cycles` 为采样截断；分组 survival 可由 `horizon_policy` 规定时限结果，默认 `red_win`。
分组 damage 忽略时限胜负奖励并报告累计伤害；`KnownOpponentEnv` 默认 100 步、支持 1–500 步，
旧 `known-rule-mixed-v5.1` 场景仍保留其明确的 50 步定义。

当前计算路径在求伤害距离前先筛掉未开火/未启用干扰的来源，damage 奖励跳过几何 shaping；
Parallel 预缓存实体名称并移除不再使用的全局 NumPy 随机状态交换。
这些是静态可确认的计算删减，本轮没有测量吞吐或宣称具体提速比例。
