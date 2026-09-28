# 接口与协议

当前核心为 `had-workbench-2.1.0`，物理协议为 `rebuild-calibrated-v3-r7-target-initialization`。
`task_mode` 在实例构造时明确指定，默认 `survival`，可选 `damage`。两种任务的计分、终止和快照不能混用。
空间默认 `spatial_dim=2`；三维必须显式设置 `spatial_dim=3`，既有三维模型也须使用此配置。

## 原生飞行控制入口

```python
from had_env import make_env

env = make_env("defense", task_mode="damage", target_health=2.0, spatial_dim=2,
               red_count=8, blue_count=8, target_count=2,
               max_cycles=100, continuous=False)
obs, infos = env.reset(seed=42)
try:
    while env.agents:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terminated, truncated, infos = env.step(actions)
    score = infos["red_0"]["episode_returns"]
finally:
    env.close()
```

这是调用示例，本轮未执行。`task_mode="survival"` 使用旧默认计分。
`target_health=None` 读取环境默认目标 HP，当前为 2.0；显式值须为正有限数。
damage 中此值是目标保持的观测/显示 HP，不是伤害预算，不改变伤害比例。
二维所有实体的高度固定为配置 `PlanarAltitude=100`，垂直速度和加速度为零。
工厂本轮只公开 `spatial_dim`，不单独公开平面高度参数。

`make_env(..., target_initialization="random")` 是训练默认：每次 reset 在 `DefaultTargetRegion` 中重新采样目标。选 `"fixed"` 后目标坐标与种子无关，x/z 取区域中点，y 在区域内部等距排列；也可传 `target_positions=[[x,y,z], ...]` 指定固定坐标，显式坐标优先。`initialization="random"/"uniform"` 则单独控制无人机开局，不控制目标模式。二维目标 z=100；三维默认目标 z 在 500–1500 米。相同配置重复传同一 seed 会复现相同布局；训练连续 reset 不要反复固定同一个 seed。`info["target_initialization"]` 返回实际使用的模式。

攻击半径在 `had_env/core/config.py` 的 `AttackDistance` 字典中按维度配置：二维 `[200, 400]`、三维 `[300, 600]`，依次为满伤半径和伤害归零半径（米）。创建环境后保存于 `env.attack_distance`，自动开火半径 `env.fire_range` 默认等于满伤半径；二维与三维实例互不改写参数。`info` 同步返回 `attack_distance`、`fire_range`。

同阵营进入攻击范围不会触发开火，也不会承受队友的爆炸伤害。红方自动打击蓝方攻击智能体，蓝方自动打击目标点；同阵营间 20 米扫掠碰撞判定保留。碰撞计算使用同一步内相同时间参数的最近距离，不能把不同时刻经过同一位置判为碰撞。

红蓝双方基于同一时刻的观测提交一个动作字典，每次 `step` 推进一个物理步。
公开名称为 `red_0...`、`blue_0...`；`possible_agents`、动作空间和观测空间在实例内固定。
`api="mpe"` 提供展平观测及固定列表接口，`info["n"]` 保存各槽位的详细终止、截断和任务信息。

| 空间配置 | 离散动作 | 连续动作 | MPE one-hot |
|---|---|---|---|
| `spatial_dim=2`（默认） | `Discrete(9)`，原 27 个基元中 z=0 的顺序子集 | 每实体 `(2,)`，范围 `[-1,1]²` | 9 维 |
| `spatial_dim=3` | `Discrete(27)`，原三维加速度基元 | 每实体 `(3,)`，范围 `[-1,1]³` | 27 维 |

两种离散空间的索引 0 都是零加速度；其余索引不应跨维度直接复用。
二维连续向量在接口内补 z=0 后送给仍使用三维坐标的物理核心，再乘原生 `aMax`。
所有连续值均须有限；MPE one-hot 必须严格匹配当前空间长度。
规则开火由引擎决定，不增加开火动作或轮流行动。
动作字典须恰好覆盖 `env.agents`；有效参与者的动作在物理更新前整体校验。
damage 死亡槽仍须提供字典键，但值被忽略并强制为零动作。

## 两种任务的伤害与回报

| 项目 | `survival`（默认） | `damage` |
|---|---|---|
| 目标 HP | 按伤害扣减，最低为 0 | 始终保持正有限初值，不扣血、不死亡 |
| 自然终止 | 任一目标毁伤，或蓝方 Attack 全灭 | 蓝方 Attack 全灭 |
| 原生胜负 `outcome_red` | 目标毁伤为负、蓝方 Attack 全灭为正 | 始终为 0，伤害任务不映射成胜负 |
| 每步红/蓝团队奖励 | 原 `RealReward` 合约 | `-step_target_damage` / `+step_target_damage` |
| 累计 `episode_returns` | `None`，沿用原奖励合约 | `{"Red": -target_damage, "Blue": target_damage}` |
| `reward_weights` | 可显式选择四维原生 latent 奖励 | 拒绝，保持原始伤害计分 |

`step_target_damage` 是本物理步所有目标收到的原始伤害之和；`target_damage` 是自 reset 起的累计值。
两者均在 HP 截断之前计量，同一步多个来源的伤害全部计入，不按目标 HP 封顶，不归一化。
例如两次分别造成 1 点满伤，红队两步奖励为 `-1, -1`，累计为 `-2`；蓝队为对应正值。
damage 目标显示 HP 为 2 时始终保持 2，累计伤害可以继续超过 2。双方无人机 HP=1、满伤=1、最高速度=120、最大加速度=40；二维 200–400、三维 300–600 距离的范围伤害保留原有线性衰减。
这里的累计回报是未折扣和；训练算法选择折扣率不改变环境记录的总伤害。

每名队员收到完整的当步团队奖励，并非按人数分摊。`rewards[a]` 与 `infos[a]["team_reward"]`
在 damage 中相等；不能把同队 `rewards` 求和或把同队多个 `episode_returns` 相加当作团队得分。
`LatentReward` 保留四维形状；damage 中四维均为零，纯伤害奖励从 `RealReward`、`rewards[a]` 或 `team_reward` 读取。
survival 显式设置 `reward_weights` 时返回 `dot(LatentReward, reward_weights)`，默认仍为 `RealReward`；
第四个 latent 分量已含 `RealReward`，不应重复相加。

## 固定槽位、终止与 bootstrap

survival 保持旧 agent 生命周期：死亡者返回最后一次转换，随后从 `agents` 移除；MPE 的对应旧槽随后为零奖励和 `done=True`。
damage 保留所有原始红蓝槽位直到全局终止或采样截断。死亡者的观测全零、动作强制为零，
但继续领取后续团队伤害奖励；其个体 `terminated` 不因死亡而变成真。红方全灭也不会提前停止环境，
避免未来目标损失从红队回报中消失。

Parallel 的 `max_cycles` 是采样截断，不是“撑到时限即胜”的奖励规则。
蓝方 Attack 全灭为自然终止；若同一步也达到采样上限，以自然终止为准。
最终转换仍返回观测与 info，`state()` 仍可取得最终全局状态；只有后续的 `env.agents` 变为空。

| `infos[a]` 字段 | 含义 |
|---|---|
| `alive`、`agent_mask` | 返回观测对应的存活状态，决定下一次是否有有效 actor 动作 |
| `action_mask` | 离散存活槽全部为 1；死亡槽仅零号动作是 1。连续模式为 `None`，死亡动作仍强制零向量 |
| `entity_mask` | 与局部观测各实体行对应；damage 死亡观察者的整个 mask 也为零 |
| `terminated`、`truncated` | 当前槽位本次转换的结束标志；survival 个体死亡可 `terminated=True` |
| `global_terminated` | 任务自然终止，不能等同于“某个 agent 已死” |
| `global_truncated` | 达到 `max_cycles` 且未自然终止 |
| `bootstrap_mask` | `float(not global_terminated)`，用于团队 critic；自然终止为 0，死亡或采样截断保持 1 |
| `spatial_dim`、`plane_altitude` | 当前二维/三维配置及平面高度；三维不以此高度约束运动 |

actor 的动作损失使用动作发生前的存活 mask；直接使用死亡后的 `agent_mask=0` 会把致死动作也删掉。
团队 critic 不应把 `bootstrap_mask` 再乘存活 mask，也不能把 `terminated or truncated` 一律当作 bootstrap 为零。
截断时应使用最后状态估值，不能用下一次 reset 的初始状态替代。
survival 仍保留历史逐 agent 终止语义；上述 `bootstrap_mask` 特指全局团队价值，不替代个体价值函数自己的终止约定。

## 观测、集中状态与原生任务信息

每行 **11 维**为相对位置 3、相对速度 3、health 1、alive 1、红/蓝/目标标志各 1。
Parallel 观测形状为 `(总实体数 - 1, 11)`，排除自身，按固定世界顺序排列。
死亡实体行全零，行顺序不随死亡改变；`observation_entities` 给出对应名称。
相对位置按场地对角线归一化，相对速度按速度尺度归一化，不强制落在 `[-1,1]`。
侦察角色不会在此接口自动引入遮挡或隐藏观测。

二维仍保留三维坐标槽：局部相对 z、z 速度为零；集中绝对 z 对应固定平面高度。
`state()` 是展平的集中绝对状态，共 `总实体数 × 11` 维，包含所有实体的归一化绝对位置/速度、
health、alive 和阵营标志；死亡实体整行置零。damage 目标保持有限 HP 和存活位，不用无穷大污染模型输入。
本轮保留既有观测/state 维度；物理时间 `sim_time`、`cycle`、`max_cycles` 和 `remaining_cycles` 只在 info。
这些时间字段不会自动进入 actor/critic；若把时限定义为有限 horizon 任务的一部分，调用方需明确将时间加入模型输入。
维持 11 维仅保证结构形状不变，不能据此推断二维和三维 checkpoint 可互换。

底层 `HADEnv` 同样接受 `task_mode`、`target_health`、`spatial_dim`。`world.task_info()` 返回：

| 字段 | 内容 |
|---|---|
| `task_mode` | `survival` 或 `damage` |
| `spatial_dim`、`plane_altitude` | 空间维度及平面高度配置 |
| `step_target_damage` | 最新物理步的总目标伤害 |
| `target_damage` | 自 reset 起的总目标伤害 |
| `target_damage_by_target` | `目标实体 Id -> 累计原始伤害`，不是分组 API 的目标局部索引 |
| `episode_returns` | damage 的红蓝累计团队回报；survival 为 `None` |
| `episode_done` | 自然终止状态，不包含调用方的采样上限 |
| `termination_reason` | `target_destroyed`、`blue_attackers_destroyed` 或 `None`；damage 不出现目标毁伤终止 |

重复读取 `task_info()`/奖励不会再次累加伤害，reset 清零逐步和累计计数。
原生 `is_episode_done()` 判断任务自然完成，`is_terminal()` 仅返回 survival 胜负符号，在 damage 中始终为 0。
原生 `get_done()` 在 survival 中保留逐 agent 死亡标志，damage 中所有槽仅随全局自然终止置真；
采样上限由调用方管理。推荐 MARL 使用统一 Parallel/MPE 接口的全局标志和 mask。

## 飞行与分组策略接入

标准 Parallel API 只接收外部给出的联合动作，不运行训练算法或捆绑模型。
`FlightSession(red_policy=..., blue_policy=...)` 可分别接两队的 `act(observation)` 策略，
每队观测包含固定 roster 的 `observation`、`agent_ids` 和 `alive_mask`。
也可 `session.step({"red": red_actions, "blue": blue_actions})` 直接给两队全部动作；
每队可按固定 roster 顺序提供数组，或按全局 agent Id 提供字典。
省略某队时调用其注册策略，未提供策略则使用零加速度。Flight 的 `continuous_native` 接口按原生向量校验，
不要与 Parallel 的 `continuous=True` 边界约束混为同一动作协议。

上层 `KnownOpponentEnv`/`SimulationSession` 的红方接口为 `act(state) -> Grouping`，
包含编组、保护目标和 reserve；共享底层执行器产生飞行动作。
蓝方 `opponent` 可选 `reactive`、`balanced`、`concentrated`，双方在同一决策前状态上独立选择。
默认每 5 个物理步及非终局伤亡边界重新决策。`KnownOpponentEnv` 默认 `max_steps=100`，
支持 1–500，`command_interval` 须在 1 到 `max_steps` 之间。
它使用蓝方 `rush` 底层规则；直接使用 `HADStage3Adapter.step(..., blue_style=...)` 可选 `rush`/`split_rush`，
或通过 `blue_action_ids` 显式覆盖蓝方动作。

分组 survival 的防守时限属于原任务定义，`horizon_policy` 可规定 `red_win`/`draw`/`blue_win`；
`KnownOpponentEnv` 返回原有终局成功 0/1 奖励。分组 damage 返回整个宏步所有物理步的负目标伤害和，
`info["delta"]` 记录推进步数，`macro_target_damage` 记录该宏步伤害；包括红方全灭后自动继续推进产生的伤害。
damage 时限为采样截断，`success=None`，不额外发放“存活/歼灭”奖励。

工作台保留历史 `known-rule-mixed-v5.1` 的特定 50 步 survival 场景定义；这不是当前环境统一的 50 步限制。
修改该场景的目标、对手、HP、时限等应使用 `had-grouping-scenario-v1`；damage 使用 `had-grouping-damage-v1`。
任务标识也不能代替物理协议标识，旧物理下的胜率、黄金轨迹和吞吐不能与当前结果混算。

## 物理时序与计算路径

本轮基于已重标定 v3-r2 物理增加任务计分，继续使用以下时序：

1. 对本步位移做扫掠碰撞，当前碰撞顺序中已死亡者不再链式杀伤。
2. 在移动前位置决定自动开火。
3. 以同一份 `WorldKinematics` 状态快照结算，开火及命中距离均基于移动前位置；智能体移动先更新位置，再更新速度。
4. 开火攻击者自毁；survival 目标扣血，damage 目标只累加伤害并保持 HP。

伤害事件的 `damage_scope=source_unclipped` 表示来源的未截断伤害贡献，
同一步多条来源事件是同时发生的，不能当作顺次扣血。`record_events=False` 只关闭详细事件记录，不关闭计分。

当前先过滤未开火/未启用干扰的来源再求伤害距离；damage 奖励跳过几何 shaping。
Parallel 每步只生成一次原生观测，预缓存观测实体名称，删除无用的全局 NumPy RNG 状态交换和重复观测复制。
核心使用本实例 `np_random`，转向限制的反向轴已为确定性实现。
本轮仅做静态审查和语法检查，未运行测试、仿真评估或性能探测，不据此给出每秒步数或具体提速比例。
历史基准仍保留其原版本和原结果。

## 随机性、分支与来源

公共环境的 reset 与各动作/观测空间使用本实例独立种子流，不改外部全局随机状态。
分组及 Flight 策略扩展由 `PolicyAdapter` 复制实例并管理 Python/NumPy 与可选 Torch 随机流。
分组 `act(state)` 得到复制的状态；`act_env(branch)` 得到独立规划环境，是可信本地扩展接口。
Flight 策略使用 `act(observation)`，不接受分组专用的 `act_env`。

Torch 是调用方可选依赖。先导入 Torch 并初始化所需 GPU 模型/上下文，再创建策略适配器。
显示插值只影响位置，HP、动作、指派、事件和物理步仍为真实离散记录。

分组完整录制保存决策边界的类型白名单 JSON 快照，不加载任意 pickle；离线续演校验源码行为身份和类型结构。
快照必须保留目标逐步/累计伤害、物理步和任务配置，恢复前检查空间维度、平面高度、任务模式、目标 HP、时限/终局策略与物理协议。
跨维度、跨任务或跨物理协议快照应拒绝恢复；旧版/稀疏轨迹仍可只读查看。
`FlightSession` 支持内存分支，CLI 离线快照入口针对上层分组。
本轮不新增训练器、模型或评估预算。
