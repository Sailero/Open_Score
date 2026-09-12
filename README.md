# Open-SCORE

当前主线为 **跨规模攻防泛化 v2**：以 HAD damage 任务为底层，接入 B0 QMIX、B2 QMIX-Atten、REFIL、DCG、GNN-QMIX、SPECTra 六种方法。代码、统一验证记录和已授权短跑结果集中在[本轮唯一实验报告](Open-SCORE/outputs/crossscale_v2/实验报告.md)。截至 2026-09-10，HAD 物理／奖励／终止验证、8 worker 混合规模掩码检查已通过；单进程 8v8 K2 为 889.29 步/秒，20v20 K4 为 282.91 步/秒。六方法 E0 各 20000 步全部完成，合计 120000 原生物理步；只确认学习与保存链路，不作收敛结论。REFIL 的 4/8 worker × 1/4 并发任务吞吐矩阵也已完成，共 10 个任务、每个累计 20000 步，合计 200000 HAD 物理步；具体实现的计时区间与资源见报告。已记录的 loss/grad 均有限，最终与恢复权重存在，阶段三准备检查通过。用户已确认阶段三每方法 100 万物理步、8 worker、种子 0；B0、B2、REFIL 三基础方法并行，第四槽执行规则锚点。正式训练尚未运行，由用户手动启动。

旧研究主线已于 **2026-09-08 16:00:19 +08:00** 确认停止；探索 final 为 Git `cc2ab86`，历史研究源码可从该提交查阅。随后 v7 重标定引擎并增加覆盖式规则与本地回放器，见 [实验报告 v7](实验报告/实验报告_v7.md)。该轮停止决定与“仅保留规则”的描述属于历史轮次。

## 环境与安装

唯一底层是独立仓库 [HAD Workbench](../Open_Score_HAD_Workbench/README.md)，本仓库通过 `Open-SCORE/open_score/envs/had_wrapper.py` 访问其原生实现，无第二份环境代码。本地依赖路径为 `E:/Code/Open_Score_HAD_Workbench`，当前协议为 `had-workbench-2.1.0` / `rebuild-calibrated-v3-r7-target-initialization`。历史 v7 使用 r2，不代表当前任务与规则的结果。覆盖规则评估在 `open_score.rules`；回放器为仓库根目录 `python viewer.py`，打开 `http://127.0.0.1:8765`。

跨规模实验使用扁平包 `Open-SCORE/open_score/`，通过仓库内可编辑安装运行。`open_score.envs.make_entity_env` 提供固定 20 红方、20 蓝方、6 目标槽的 damage 任务实体接口；每局训练规模独立采样，死亡和 padding 槽保留，模型动作编号为 0–8。仅逐步热路径绕开原生观测；reset 沿用 HAD 原生初始化。评估和规则锚点共享蓝方事件调度、物理过程、动作映射和诊断采集。

```powershell
& 'D:\Software\Anaconda\envs\torch310\python.exe' -m pip install -e './Open-SCORE[training]'
```

`had-env` 声明环境依赖，`training` extra 声明六方法训练与出图所需依赖。当前运行解释器为上述 `torch310`，已安装 PyTorch 2.13.0+cu130、NumPy 1.26.4、SciPy 1.15.3、Matplotlib 3.10.9、PyYAML 6.0.3；extra 不负责选择 CUDA 构建。渲染及 Workbench 界面的安装方法见其 README。

本轮支持保留源码目录的可编辑安装。方法配置保留在包外的 `Open-SCORE/configs/`，命令入口保留在 `Open-SCORE/scripts/`；独立 wheel 不是当前支持的运行方式。vendored 主干及补丁内的 YAML 与已有 LICENSE 已声明为 package data，不复制包外配置来扩展发布体系。

## 公开接口

| 导入 | 用途 |
|---|---|
| `open_score.envs.make_entity_env`、`HADEntityEnv` | damage 跨规模实体环境；10 维基础特征、固定槽与原生动作映射 |
| `open_score.algos.train(name, cfg)` | 六方法共享训练入口；预算必须显式指定 |
| `open_score.algos.load_policy(name, checkpoint)` | 冻结 checkpoint 转为 `reset()`／`act(state, side, action_ids)` 策略，可注册到规则评估与回放接口 |
| `open_score.make_env`、`open_score.parallel_env` | 原生 Parallel / MPE 环境工厂 |
| `open_score.rules.KnownOpponentEnv`、`make_grouping_env` | 原生事件式分组环境；后者默认规则执行器 |
| `Entity`、`Group`、`Grouping`、`DecisionState`（均位于 `open_score.rules`） | 实体、分组动作与公开状态，沿用原生序列化与合法性约束 |
| `RuleExecutor`、`RulePolicy` | 共享规则底层与上层规则策略 |
| `CoveragePolicy`、`CoverageExecutor`、`run_episode`、`evaluate` | 当前 `rule_nv1_v1`：每步整数规划分组与匀速直线预测 |
| `rule_grouping`、`grand_grouping`、`decode_counts` | 规则分组、每目标大组及人数分配解码 |

规则策略支持 `rule`、`grand`、`random`、`static_rule`；`reset()` 重置策略状态，`act(state)` 返回原生 `Grouping`。项目代码统一通过 `open_score.envs` 访问固定对手和物理适配器，只有 `had_wrapper.py` 直接导入 HAD。

当前网页默认双方使用分层策略。红方上层 `nv1` 每步用 `p + v × Δt` 预测蓝方下一步位置，通过 0–1 整数规划最小化总拦截距离；下层 `predictive_intercept` 将指派转为飞行动作。每个红方恰好负责一个蓝方、每个参与分组的蓝方至少分配一个红方。红方不足时仅纳入最近的 R 个蓝方。蓝方上层默认 `reactive`，每 5 步或伤亡时重新选目标，下层 `rush` 每步控制飞行。完整定义见 [现行规则说明](docs/three_layer_rule_mathematical_spec.md)。旧覆盖策略别名和历史策略选择入口已移除，历史结果仍保留原记录。

网页对红蓝双方采用相同结构：先选 `end_to_end` 或 `hierarchical`；端到端只有一个动作策略，分层按实际层级显示各层策略。默认端到端目录提供最近威胁直追 / 最近目标直冲的规则基线及随机基线；训练 checkpoint 在显式加载并注册后进入该目录。蓝方分层上层可选 `reactive/concentrated/balanced`，下层可选 `rush/split_rush`。XY/XZ 图等比例绘制，伤害与存活任务使用各自指标；同条件比较校验物理协议及实际环境参数。

API 4 的策略输入是两个对称对象；切换架构后只提交该架构字段，旧 `red_policy/blue_rule/blue_opponent/blue_style` 参数会被拒绝：

```python
red_strategy = {
    "architecture": "hierarchical",
    "layers": {"grouping": "nv1", "control": "predictive_intercept"},
}
blue_strategy = {"architecture": "end_to_end", "policy": "nearest_target"}
# run_episode(..., red_strategy=red_strategy, blue_strategy=blue_strategy)
```

`STRATEGY_CATALOG` 是表单与后端校验共用的策略目录。训练后的动作策略可在服务启动前调用 `register_end_to_end_policy(side, name, label, factory)` 注册：`factory(seed)` 返回提供 `reset()` 和 `act(state, side, action_ids)` 的对象。`state` 是公开物理状态，不含对手本步指令；返回 `{全局智能体ID: 原生动作ID}`。二维模型的 0–8 动作索引需通过 `action_ids` 转换。新增分层算法在对应层实现并注册实际执行逻辑后，表单按层目录生成，不能只增加一个未实现的名称。

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

此前环境接入轮次仅修改环境、接入与展示，没有进行该轮网络训练。其 Parallel RNG、实体缓存和伤害距离过滤调整属于历史环境修改；本轮跨规模实现未修改外部 HAD。当前验证与已授权短跑状态以本轮实验报告为准。完整原生接口见 [HAD API](../Open_Score_HAD_Workbench/docs/API.md)。

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

阶段三已确认每方法 100 万物理步、8 worker、种子 0；B0、B2、REFIL 同时训练，第四槽执行规则锚点。当前尚未启动，由用户在 PowerShell 手动执行以下命令：

```powershell
& 'D:\Software\Anaconda\envs\torch310\python.exe' -X utf8 'E:\Code\Open_Score\Open-SCORE\scripts\train.py' --stage stage3 --steps 1000000 --batch-size-run 8 --seed 0
```

本轮只有一个输出目录 `Open-SCORE/outputs/crossscale_v2/`。合并记录通过方法、运行和种子字段区分；模型与控制台日志按 `<方法>/<运行>/seed_<种子>/` 保存。唯一正式 Markdown 报告为该目录的 `实验报告.md`，图表放在 `figures/`。训练启动终端每 10 秒显示进度、吞吐、loss、评估进度和最近完整验证 D；报告每 5 分钟、完整验证结束及任务退出时原位刷新，编辑器中重新打开或刷新即可查看。

阶段三启动并生成日志后，可在另一终端读取 REFIL 种子 0 的训练日志；其他基础方法替换路径中的方法名。以下相对路径命令在仓库根目录 `E:\Code\Open_Score` 执行：

```powershell
Get-Content -LiteralPath '.\Open-SCORE\outputs\crossscale_v2\refil\stage3\seed_0\console.log' -Tail 30 -Wait
```

从已有记录立即重画同一份报告，不运行训练或新评估：

```powershell
& 'D:\Software\Anaconda\envs\torch310\python.exe' -X utf8 .\Open-SCORE\scripts\plot.py --run stage3
```

阶段三日志路径的运行段使用 `stage3`。刷新命令的 `--run` 只标记刷新来源，同一份报告始终保留 E0、全部吞吐任务和正式阶段三各自的数据与图表，不会切换成另一份报告。已完成的 E0 日志仍在各方法的 `e0/seed_0/console.log`。图表只显示实际已落盘数据；E0 的原生环境结果与 HAD 指标分开解释。正式阶段三、最终评价及规则锚点尚未运行，用户执行上述已确认配置的启动命令后才开始产生结果。

需要停止时，在另一个终端运行下面的命令。调度器会请求训练子任务完成当前采样和学习批次，保存 `resume.pt` 后退出；评价与规则锚点保留已落盘的逐局结果，恢复时补齐剩余局。等待状态显示 `stopped` 与可恢复步数后再关闭终端。执行工具的强制中断曾直接结束整个进程组、跳过 Python 的 Ctrl+C 保存处理，因此工具会话使用此停止入口。

```powershell
& 'D:\Software\Anaconda\envs\torch310\python.exe' -X utf8 .\Open-SCORE\scripts\train.py --stage stop
```

恢复时在原启动命令末尾增加 `--resume`，并保留原方法、环境、种子、`--steps`、`--batch-size-run`、运行标识和输出目录。恢复点包含 replay、优化器、目标网络、随机流和进度；`final.pt` 与验证选优 `best.pt` 用于冻结评估。已运行但尚未保存的日志步数不等同于可恢复步数。

吞吐矩阵已全部完成，无需再运行。以下仅保留本轮矩阵的恢复命令记录；调度器会保留已完成任务，每个任务累计配额仍为 20000 步。阶段三如需恢复，在上方正式启动命令末尾增加 `--resume`，并等待原调度器退出后再执行。

```powershell
& 'D:\Software\Anaconda\envs\torch310\python.exe' -X utf8 .\Open-SCORE\scripts\train.py --stage benchmark --run benchmark_optimized --resume
```

## 历史研究

[实验记录](实验报告/实验记录.md) 维护跨版本摘要。本轮报告归 `Open-SCORE/outputs/crossscale_v2/`，历史各版唯一正式报告在 `实验报告/`：

- [S1/S2 与早期 Stage3](实验报告/实验报告_s1_s2.md)
- [v2：选择释放与重建](实验报告/实验报告_v2.md)
- [v3：较长训练与底层迁移](实验报告/实验报告_v3.md)
- [v4：共享规则底层与价值搜索](实验报告/实验报告_v4.md)
- [v5：混合难度六路线](实验报告/实验报告_v5.md)
- [v6：新环境六方法与 S2 修订](实验报告/实验报告_v6.md)
- [v7：环境成熟化与覆盖式规则](实验报告/实验报告_v7.md)

数据、日志、配置快照和已有模型原位保留于 `Open-SCORE/outputs`；历史冻结模型仍在 `Open-SCORE/assets/frozen`，当前接口不加载这些模型。多数输出资产被 Git 忽略，不属于探索 final 提交的备份范围。恢复旧研究需使用探索 final 的代码与对应历史环境，当前精简接口不兼容旧训练检查点。

原始研究修改稿与 v6 原执行方案可分别从 Git `cc2ab86:固定对手下动态分组研究方案.md`、`cc2ab86:dynamic_grouping_v6_execution_plan.md` 查阅。各版实际完成量、取消项和结论以 `实验报告/` 中的正式报告为准。
