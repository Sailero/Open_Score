# SMAClite 原版复现与 SMAClite-AD 环境审计报告

> 审计日期：2026-08-30  
> 结论级别：环境安装、算法流水线与接口 smoke 已完成；长时训练与收敛结论尚未完成  
> 主机：Windows 11 Pro（build 26200；Python 的兼容字符串显示为 `Windows-10-10.0.26200`），Intel i7-14700KF，NVIDIA RTX 5070 Ti 16 GB，Python 3.10.20

## 1. 可直接使用的结论

本机已经形成两条严格分离的实验线：

1. **原版 SMAClite 线**：上游 v2.0.0 的 13 个 JSON 场景保持逐字节不变，使用官方 `smaclite/*-v0` Gymnasium ID。官方 `3s5z` 已完成确定性 reset 和 step smoke；官方 ePyMARL 的 QMIX、VDN、MAPPO 已在 `2s_vs_1sc` 完成极小预算训练流水线。
2. **SMAClite-AD 线**：独立协议 ID 为 `OpenSCORE/SMACliteAD-Asset-v0`，Red 是攻击方，Blue 是资产防守方。它复用上游单位、碰撞、移动与伤害动力学，但单独增加双边控制、资产目标、动态规模和 padded mask。它不是上游官方场景，论文中必须始终写成 **“SMAClite-AD（Open-SCORE 扩展）”**。

当前证据足以说明“环境和三类算法训练链可运行”，不足以说明“QMIX/VDN/MAPPO 已收敛”或“SMAClite-AD 已达到论文数值”。后两项需要正式训练预算与多随机种子结果。

## 2. 上游身份、安装与指纹

| 项目 | 锁定值 |
|---|---|
| SMAClite 仓库 | `https://github.com/uoe-agents/smaclite` |
| tag | `v2.0.0` |
| commit | `e936d9dbf4f85551d6fd445a6c1150867bc79c55` |
| Python distribution | `smaclite==0.0.1`（上游 `setup.py` 的版本号） |
| stock 场景联合 SHA-256 | `0c862da08a59410a5832bc59899e39419d9e749f3ac41b20e34c463fbe8ef4b0` |
| ePyMARL commit | `cbc38c09588064eab978501d0f12c2cf58fa7fc2` |

逐文件哈希、依赖版本和核心源文件哈希记录在 `upstream/smaclite-v2.0.0.lock.json`。每次正式实验前应先运行 smoke；联合哈希变化时必须停止实验，而不是继续混用结果。

### 2.1 Windows 安装命令

在项目目录执行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File ".\scripts\install_smaclite.ps1" `
  -PythonPath "D:\Software\Anaconda\envs\torch310\python.exe" `
  -GitPath "D:\Software\Git\cmd\git.exe"
```

`Bypass` 只作用于这一个安装子进程，不会修改机器的全局 ExecutionPolicy；本机直接执行 `.ps1` 会被现有策略拒绝，因此上面的写法是已实测命令。

脚本会使用 `D:\Software\Git\cmd\git.exe` 克隆固定 commit，检查工作树无修改，锁定下列已实测组合，然后进行 editable install：

- NumPy 1.26.4；
- Gymnasium 1.2.2；
- Rtree 1.0.0；
- pygame 2.6.1；
- scikit-learn 1.7.2；
- setuptools 80.9.0；
- PyTorch 2.13.0+cu130。

之所以不用普通 wheel，是因为上游 v2.0.0 的 `setup.py` 构建出的 wheel 没有包含 `smaclite.env.maps` 等子目录和 JSON 数据。实测普通安装导入时报：

```text
ModuleNotFoundError: No module named 'smaclite.env.maps'
```

固定 commit 的 editable install 不修改上游源码，并已通过官方 stock ID 的 reset/step。因此，这是当前 Windows 机器上的可复现兼容方案，不是对环境逻辑的补丁。

### 2.2 许可证边界

审计的 v2.0.0 commit 根目录没有 `LICENSE` 文件。报告只能陈述这个可验证事实，不能自行推定许可。当前实现没有把上游源码/JSON 复制进本仓库，而是由安装脚本获取上游并在运行期导入。公开发布容器、上游副本或包含其数据的复现包之前，仍应向上游作者确认许可；本报告不提供法律意见，也不从“用于学术实验”推导任何再分发权。

## 3. 原版 SMAClite 实测

### 3.1 官方环境 smoke

执行：

```powershell
$env:PYTHONPATH = "$PWD\src"
& "D:\Software\Anaconda\envs\torch310\python.exe" `
  "scripts\smoke_smaclite.py" --mode stock
```

`smaclite/3s5z-v0` 的实际结果：

| 检查项 | 实测值 |
|---|---:|
| allied agents / enemies | 8 / 8 |
| 单智能体 observation | `(128,)` |
| global state | `(216,)` |
| 最大动作数 | 14 |
| 固定 seed 两次 reset | 逐元素相同 |
| 一步 primitive action | 成功，reward 0.0，未终止 |
| stock JSON 联合哈希 | 与 lock 一致 |

`SMACliteStockAdapter` 没有改变官方环境：它只把每个官方 flat observation 无损放进一个 observation entity，并把官方 flat state 放进一个 state entity，使项目内统一训练器可以读取 `TeamObservation/GlobalState`：

```text
entity_obs      [8, 1, 128]
entity_mask     [8, 1]
self_obs        [8, 1]
task_obs        [8, 1]
agent_mask      [8]
avail_actions   [8, 14]
state_entities  [1, 216]
state_mask      [1]
```

这里的“单实体”只是无损兼容容器，不能在论文中描述成上游原生 entity observation。

### 3.2 ePyMARL 三算法 pipeline smoke

上游 SMAClite README 推荐 ePyMARL。Windows 原生 torch310 中已经在官方 ePyMARL commit `cbc38c0…` 上完成：

| 算法 | 场景 | `time_limit` | 请求 `t_max` | 实际最后 `t_env` | Sacred 状态 | 有效训练量证据 |
|---|---|---:|---:|---:|---|---|
| QMIX | `2s_vs_1sc` | 25 | 50 | 75 | `COMPLETED` | loss 与 grad_norm 各 2 个记录 |
| VDN | `2s_vs_1sc` | 25 | 50 | 75 | `COMPLETED` | loss 与 grad_norm 各 2 个记录 |
| MAPPO | `2s_vs_1sc` | 25 | 50 | 75 | `COMPLETED` | policy/critic loss 与两类 grad_norm 各 3 个记录 |

三个 smoke 的 `battle_won_mean` 都是 0。原因是只有三个 25-step 回合且探索率几乎仍为 1；这正是 pipeline smoke，不是失败的收敛实验，也不能与论文 win rate 对比。

pipeline smoke 的命令模板如下，`ALGORITHM` 分别替换为 `qmix`、`vdn`、`mappo`：

```powershell
$env:GIT_PYTHON_GIT_EXECUTABLE = "D:\Software\Git\cmd\git.exe"
cd upstream\external\epymarl
& "D:\Software\Anaconda\envs\torch310\python.exe" src\main.py `
  --config=ALGORITHM --env-config=smaclite with `
  env_args.time_limit=25 env_args.map_name=2s_vs_1sc `
  t_max=50 test_interval=50 test_nepisode=1 batch_size_run=1 `
  log_interval=25 runner_log_interval=25 learner_log_interval=25 `
  use_cuda=False save_model=False
```

QMIX/VDN smoke 使用 `batch_size=2 buffer_size=10`，MAPPO 使用 `batch_size=1 buffer_size=2`。Sacred 0.8.7 仍导入已弃用的 `pkg_resources`，所以需要 `setuptools==80.9.0`；当前旧 VS Code 进程还必须显式给 GitPython 指定 `git.exe`。重启 VS Code 后通常不再需要后一个环境变量。

原始日志位于忽略版本控制的：

```text
upstream/external/epymarl/results/sacred/{qmix,vdn,mappo}/2s_vs_1sc/1/
```

### 3.3 官方 horizon 与算法默认配置的 50k pilot

QA 进一步发现，3.2 节为提速显式使用的 `time_limit=25` 不是 `smaclite.yaml` 的官方默认 horizon 150；smoke 不能承担标准场景学习结论。随后在不修改 ePyMARL/SMAClite checkout 的前提下重跑：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_epymarl_stock_smoke.ps1 `
  -Algorithm all -TMax 50000 `
  -TestInterval 45000 -TestEpisodes 20 -LogInterval 5000 `
  -UseStockTrainingDefaults
```

该模式不覆盖官方 `time_limit=150`，并保留 QMIX/VDN 的 `batch_size=32, buffer_size=5000` 与 MAPPO 的 `batch_size_run=batch_size=buffer_size=10`。三算法都在 CUDA 上以 Sacred `COMPLETED` 结束，checkout commit 均为干净的 `cbc38c0…`。

| 算法 | Sacred run | 近终点 test step | first→near-final test return | near-final 回合长度 | near-final `battle_won` |
|---|---|---:|---:|---:|---:|
| QMIX | `qmix/.../8` | 45,047 | 0.0000→10.0392 | 32.0 | 0.000 |
| VDN | `vdn/.../5` | 45,042 | 0.0000→8.7843 | 32.0 | 0.000 |
| MAPPO | `mappo/.../4` | 45,799 | 1.3490→9.9451 | 32.0 | 0.000 |

MAPPO 的 first logged test 已在 `t_env=612`，不能称为纯随机初始化基线；QMIX/VDN 也只称 first logged point。`test_interval` 放在 90% 预算处，是为了避免最后一个可变长度 episode 直接跨过 `t_max` 而漏掉训练后测试。策略在测试后仍训练到 50k，所以表中是 near-final 而不是 exact-final checkpoint 结果。

三算法的 return 和回合长度都发生明显变化，但 `battle_won` 全为 0；任务胜率复现仍未建立。20 个 test episodes 位于同一个训练 seed 内，QMIX/VDN 的 return std 为 0 或数值零，不能冒充 20 个独立 seed 或据此给置信区间。50k 只占上游 QMIX/VDN `t_max=2,050,000` 的 2.44%，以及 MAPPO `t_max=20,050,000` 的 0.249%；本轮也没有保存 checkpoint。正式 stock 复现仍需相应上游预算、至少 5 个预注册训练 seed、checkpoint 和不确定性区间。

机器证据与被采纳 Sacred 文件 SHA-256 见 `docs/evidence/epymarl_stock_50k_torch310.json`。中途运行过的 25-step 扩展诊断、漏掉 near-final test 的 run，以及被主动中止的 run 均不进入该证据。

## 4. SMAClite-AD 的修改契约

### 4.1 严格隔离原则

环境分三层，任何论文结果都能判断来自哪一层：

| 层 | ID/类 | 是否改上游场景 | 用途 |
|---|---|---|---|
| 官方环境 | `smaclite/*-v0` | 否 | 与社区基线对齐 |
| 无损 tensor wrapper | `SMACliteStockAdapter` | 否 | 接入项目统一训练器 |
| AD 扩展 | `OpenSCORE/SMACliteAD-Asset-v0` / `SMACliteADEnv` | 独立内存场景 | 新攻防问题 |

AD 扩展放在 `open_score.envs.smaclite_ad`，不会注册或覆盖任何 `smaclite/*-v0` ID，不会向上游场景目录写文件。测试还会在运行 AD 前后重新计算 stock 哈希。

### 4.2 角色、目标和终止

- Red：攻击方；
- Blue：防守方；
- 资产：使用上游 `SPINE_CRAWLER` 的生命、护甲、半径、碰撞和受击逻辑，但强制 `StopCommand`，因此它是不会主动开火的目标；
- Red 胜：资产生命降为 0；
- Blue 胜：Red 全灭，或资产存活到 `episode_limit`；
- 同一物理 tick 中若资产与最后一个 Red 同时死亡，资产摧毁优先，记 Red 胜；
- `elimination` 模式也受支持，但当前主实验固定用 `asset` 模式。

场景在内存生成：Red marine 位于左侧，Blue marine 与资产位于右侧，地形复用官方 SIMPLE terrain 对象。没有复制或改写官方 JSON。

### 4.3 原子动作

双方都只使用 SMAClite 原生原子动作：

| 动作编号 | 语义 |
|---:|---|
| 0 | noop，仅允许死亡或 padding 槽位 |
| 1 | stop |
| 2–5 | 四个方向移动 |
| Red 的 `6 ... 6+B-1` | 直接攻击实际存在的 Blue 单位；其余 Blue target 槽由 mask 禁用 |
| Red 的 `6+max_blue`（当前为 11） | 直接攻击资产；跨实际配比保持固定 |
| Blue 的 `6 ... 6+R-1` | 直接攻击 Red 单位 |

不存在手写的“集火”“绕后”“拦截”等宏动作。规则策略也只能根据目标方向选择上述原子动作，因此与学习策略动作空间一致。

### 4.4 动态规模与 mask

每个 episode 的实际 roster 在环境实例创建时确定；同一套策略通过固定上界和 mask 跨实例复用：

```text
max_entities = max_red + max_blue + 1 asset
action_dim   = 6 + max(max_red, max_blue + 1 asset)
```

观测契约为：

```text
entity_obs      [max_agents(side), max_entities, 13]
entity_mask     [max_agents(side), max_entities]
self_obs        [max_agents(side), 12]
task_obs        [max_agents(side), 8]
agent_mask      [max_agents(side)]
avail_actions   [max_agents(side), action_dim]
state_entities  [max_entities, 11]
state_mask      [max_entities]
```

实体特征包含相对/绝对位置、距离、生命、护盾、冷却、存活和 self/ally/enemy/asset 关系。队友与资产位置共享，敌人遵守上游 9-unit sight range。全局 state 只供 CTDE。任务特征显式给出资产方向与生命、红蓝规模比例和角色。padding 和死亡槽只允许 noop。

这个表示满足排列共享网络和跨配比训练的必要条件，但“一个 checkpoint 在未见配比上的泛化”仍必须通过 held-out ratio 实验验证，不能仅凭 mask 设计宣称成立。

### 4.5 奖励

终局是严格零和：Red 终局收益 `+1/-1`，Blue 取相反数。小幅势函数 shaping 为：

```text
r_dense_red = shaping_scale * (
    asset_fraction_decrease
    + 0.25 * blue_health_fraction_decrease
    - 0.25 * red_health_fraction_decrease
    + approach_weight * minimum_asset_distance_decrease
)
r_blue = -r_red
```

默认 `shaping_scale=0.10`、`approach_weight=0.0`，因此基础环境在未显式开启距离项时行为不变。后续最小学习验证为了让 45-episode 预算能产生方向性信号，显式使用 `shaping_scale=0.50, approach_weight=1.0`。距离项是前后最小攻击方—资产距离之差，不是“冲锋”宏动作；策略仍只能选择原子移动。正式论文必须报告关闭 shaping 的消融，以排除奖励设计替代策略学习的质疑。

## 5. SMAClite-AD 实测证据

执行：

```powershell
$env:PYTHONPATH = "$PWD\src"
& "D:\Software\Anaconda\envs\torch310\python.exe" `
  "scripts\smoke_smaclite.py" --mode all `
  --output "docs\evidence\smaclite_validation_torch310.json"
& "D:\Software\Anaconda\envs\torch310\python.exe" `
  -m pytest tests\test_smaclite_ad.py -q
```

实测 pytest 为 **7 passed**。覆盖内容包括 stock hash、官方 flat wrapper、三种动态配比、TeamObservation/GlobalState 校验、固定 seed 转移一致、奖励零和、资产终止分支和 padding 非 noop 拒绝。

规则参考策略 smoke 的真实结果如下。Red 直接冲向资产，Blue 追击最近 Red；这只是环境动力学和终止链检查。

| Red:Blue | steps | 终止原因 | Red outcome | 最终 Red hp | 最终 Blue hp | 最终资产 hp |
|---|---:|---|---:|---:|---:|---:|
| 2:1 | 23 | attackers eliminated | -1 | 0.000 | 1.000 | 0.835 |
| 3:2 | 18 | attackers eliminated | -1 | 0.000 | 1.000 | 0.859 |
| 5:3 | 22 | attackers eliminated | -1 | 0.000 | 1.000 | 0.754 |

三个配比的 padded shapes 完全相同、两队 return 逐步相加为 0、资产确实受到伤害。结果同时揭示当前朴素冲锋规则很弱，并不说明环境失衡；需要学习策略和多种规则对手后才能讨论难度。

机器可读的完整输出在 `docs/evidence/smaclite_validation_torch310.json`。它的状态字段明确是 `smoke_and_contract_validation_not_convergence_claim`。

## 6. 如何保证修改后仍有较高认可度

认可度不能来自把 AD 扩展称为“官方 SMAClite”，而应来自可审计的实验设计：

1. **先做官方锚点**：QMIX、VDN、MAPPO 先在未经修改的 stock ID 上做多 seed 收敛复现，再进入 AD。
2. **保持物理内核不动**：单位数值、伤害、移动、RVO2 碰撞和 tick 逻辑来自固定上游 commit；新增部分只负责控制权、目标、奖励、终止和 tensor 表示。
3. **独立命名**：所有表格分列 `SMAClite-stock` 与 `SMAClite-AD`，不将扩展结果放进官方 benchmark 数值列。
4. **保存证据链**：commit、依赖、seed、配置、场景 hash、代码 hash、原始 metrics 与 checkpoint 一同归档。
5. **修改消融**：至少比较 elimination/asset、无 shaping/默认 shaping、固定规模/动态规模、资产可见性共享/局部可见。
6. **跨算法一致性**：同一 observation/action/termination contract 同时跑 VDN、QMIX、MAPPO；若只有某一算法有效，要优先排查实现偏置。
7. **跨配比泛化**：训练配比与测试配比严格分开，报告 seen ratio、interpolation ratio、extrapolation ratio，而不是把训练集随机重采样写成泛化。

推荐正式实验档位：

| 阶段 | 场景/配比 | 算法 | 最低种子数 | 判定目标 |
|---|---|---|---:|---|
| Stock-A | `2s_vs_1sc` | QMIX/VDN/MAPPO | 5 | 复现收敛趋势与 win rate |
| Stock-B | `3s5z`、`3s_vs_5z` | QMIX/VDN/MAPPO | 5 | 排除只在最小图可用 |
| AD-fixed | 2:1、3:2、5:3 | 三算法 | 5 | 每个固定配比可学习 |
| AD-dynamic | 训练 2:1、3:2、4:2、5:3 | 共享动态策略 | 5 | 同 checkpoint 覆盖 seen ratios |
| AD-held-out | 3:1、4:3、6:3 | 不再训练 | 5 | 插值/外推泛化 |

正式收敛阈值不能在看到结果后临时制定。建议预注册为：滑动窗口 win rate、AUC、达到 50%/80% win rate 的 environment steps、跨 seed IQM 与 95% bootstrap CI；同时保存 crash、NaN 和超时率。

## 7. 当前完成边界与下一接口

### 已完成

- SMAClite v2.0.0 固定 commit 安装；
- stock 场景逐文件哈希与 lock；
- stock `3s5z` reset/step 与 seed 确定性；
- 官方 ePyMARL QMIX/VDN/MAPPO 在 stock `2s_vs_1sc` 的真实训练 pipeline smoke；
- 无损 stock tensor wrapper；
- 独立 SMAClite-AD 双边环境、动态规模、资产、mask、seed、零和奖励；
- stock/AD 自动测试和机器可读 smoke 证据。

### 尚未完成，不能提前声称

- stock 三算法的正式多 seed 收敛数值；
- 三算法在 SMAClite-AD 的长训练与胜率收敛；
- 动态规模 checkpoint 的 held-out ratio 泛化；
- 与 SC2 原版 SMAC 的数值等价性；
- 上游许可确认。

主线训练器可直接使用：

```python
from open_score.envs import (
    SMACliteStockAdapter,
    SMACliteADEnv,
    tensorize_smaclite_stock_observation,
    tensorize_smaclite_ad_observation,
)
```

stock adapter 实例暴露 `ENTITY_DIM / SELF_DIM / TASK_DIM / STATE_ENTITY_DIM / ACTION_DIM`；AD 类暴露同名维度并用 `action_dim` 给出由规模上界决定的动作宽度。下一轮训练应消费这些实例维度，不应在算法中硬编码 `3s5z` 或某个红蓝配比。

## 8. QMIX、VDN、MAPPO 在 SMAClite-AD 的最小真实训练验证

> **QA 降级声明（2026-08-30）**：本节记录的是第一版 `spawn_jitter=0` 的固定布局运行。所谓 6 个 evaluation episodes 实际只有三个 ratio 各一个确定性初态，即 **6 nominal / 3 effective layouts**；因此本节所有改善只能作为 deterministic pipeline smoke，不能称为独立样本上的 paired 学习证据。该运行还只保存 final checkpoint，episode 30 的 QMIX 最佳点不可复放。修复后的随机布局、validation-best/final checkpoint 与一次性 held-out 结果以第 9 节和 schema v2 机器证据为准。

### 8.1 验证设计

本节是 2026-08-30 完成的第二层验证，比 ePyMARL 的 `t_max=50` pipeline smoke 更进一步，但仍不是正式收敛实验：

- 三算法都从随机初始化开始；
- Red 为学习的资产攻击方，Blue 为固定 `idle` 规则方；
- 一个共享 checkpoint 循环训练 2:1、3:2、5:3 三种实际配比；
- padding 上界固定为 Red 6、Blue 5、实体 12、动作 12；
- 每算法训练 45 episodes，每个 ratio 恰好 15 episodes；
- QMIX/VDN 使用 3-episode replay batch，MAPPO 使用 3-episode on-policy batch；
- paired before/after evaluation 名义上使用完全相同且顺序一致的 6 个环境 seed，但实际只有三个 ratio 各一个固定初态；
- `episode_limit=60`，`shaping_scale=0.50`，`approach_weight=1.0`；
- 只使用 seed `20260830` 和 CPU。

以下命令仅用于解释旧证据，参数接口已经被 schema v2 脚本替代，**不可直接重放**：

```powershell
$env:PYTHONPATH = "$PWD\src"
& "D:\Software\Anaconda\envs\torch310\python.exe" `
  scripts\train_smaclite_ad_baselines.py `
  --algorithms qmix vdn mappo `
  --seeds 20260830 `
  --episodes 45 --episode-limit 60 `
  --eval-episodes-per-ratio 2 --eval-every 15 `
  --batch-episodes 3 `
  --agent-hidden-dim 32 --critic-hidden-dim 32 `
  --ppo-epochs 2 --device cpu `
  --output-dir outputs\smaclite_ad_baselines `
  --evidence-dir docs\evidence
```

固定 `idle` 对手是最小“能否学”的单边控制检查，不代表有对抗性的防守策略。规则可解性上界使用同一原子动作空间的 `rush_asset` Red，对同一 `idle` Blue 在三个 ratio 上均达到 100% 胜率；平均回合长度分别为 58、42、28，说明 60-step horizon 下三个任务都可解。

### 8.2 三层判定，不混写结论

本验证把结论严格分为三层：

1. **训练 pipeline 成立**：参数哈希改变、梯度和 loss 有限、每种 ratio 至少进入过一次梯度 batch；
2. **小预算学习信号**：paired shaped return 改善，而且至少两个 ratio 改善；
3. **任务成功/收敛**：最终 win rate 改善，并在多 seed 长预算下稳定。当前三算法都没有达到第三层。

真实汇总如下：

| 算法 | env steps | 更新次数 | 参数 L2 变化 | 初始→最终 mean return | paired Δ | 最终 win rate | 最终平均资产生命 | 用时 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| QMIX | 2672 | 43 | 1.1673 | -1.0000 → -0.7772 | +0.2228 | 0.000 | 0.954 | 47.3 s |
| VDN | 2700 | 43 | 0.9012 | -1.0000 → -0.8067 | +0.1933 | 0.000 | 0.711 | 46.1 s |
| MAPPO | 2700 | 15 | 1.1945 | -1.0000 → -0.8739 | +0.1261 | 0.000 | 1.000 | 44.7 s |

三个 checkpoint 的参数均真实改变，最后一次更新指标均为有限值；三者的梯度历史都覆盖 `2:1, 3:2, 5:3`，跨 ratio 张量 shape 审计全部相同。因此“AD 环境可以被三种 learner 正常训练”已经成立。

但是 paired 改善的含义不同：

- **QMIX**：6/6 paired returns 改善；最终只有 2:1 对资产造成伤害，平均剩余生命 86.3%。episode 30 的评估曾出现 33.3% 胜率、mean return -0.0053，episode 45 又回落到 0% 胜率，说明学习信号真实但不稳定，且当前只保存了最终 checkpoint。
- **VDN**：4/6 paired returns 改善；3:2 的最终资产生命降到 13.3%，但 2:1 和 5:3 没有造成资产伤害，5:3 return 还比初始差 0.0781。它学到明显局部行为，却没有形成可靠跨规模策略。
- **MAPPO**：6/6 paired returns 改善，但资产生命仍为 100%。收益来自更靠近资产的距离 shaping；这证明 actor/critic 接收到了可优化信号，不能解释为完成攻击任务。

所以自动字段同时保存：

```text
learning_status     = small_budget_learning_signal_observed
task_success_status = win_rate_not_improved
formal_convergence_claim = false
```

这三个字段必须一起阅读。只摘取第一个字段会夸大结果。

### 8.3 证据文件与测试

tracked 证据为：

```text
docs/evidence/smaclite_ad_baselines_summary.json
docs/evidence/smaclite_ad_baselines_curve.csv
```

summary 保存命令参数、机器信息、每次更新覆盖的 ratio、tensor shape、初末模型 SHA-256、参数变化、初末 evaluation、逐 ratio paired delta、最终 learner metrics 和规则可解性参考；curve 保存 episode 0/15/30/45 的训练与评估轨迹。checkpoint 位于被 Git 忽略的 `outputs/smaclite_ad_baselines/checkpoints/`。

针对性测试命令：

```powershell
& "D:\Software\Anaconda\envs\torch310\python.exe" -m pytest `
  tests\test_smaclite_ad.py tests\test_smaclite_ad_training.py -q
```

实测为 **11 passed**。新增测试不只检查 forward：QMIX、VDN、MAPPO 都会在包含三种 ratio 的真实短回合 batch 上执行 backward/optimizer step，并断言参数发生变化。

### 8.4 下一次正式训练应改变什么

当前最重要的不是增加更多算法，而是把“shaped approach”推进到“稳定摧毁资产”：

1. 至少 5 个预注册 seed，不按单 seed 最佳 episode 选择结论；
2. 增加预算并降低后半段探索率，单独保存 validation-best 与 final checkpoint；
3. 训练后必须对 `idle`、`intercept` 两种 Blue 规则方评估；
4. 加入 `approach_weight=0` 和 `shaping_scale=0` 消融；
5. 训练 ratio 与 held-out ratio 分离，至少测试 3:1、4:3、6:3；
6. 用 win rate、资产伤害、return AUC、跨 seed IQM 和 bootstrap CI 共同判定，而不是只比较最后一次 shaped return。

截至本报告，最严谨的表述是：**SMAClite-AD 已通过三算法真实多配比训练验证，并出现可重复审计的小预算学习信号；固定预算下的最终任务成功与跨配比收敛尚未建立。**

## 9. QA 修复：随机布局、独立 held-out 与可复放 best checkpoint

### 9.1 环境协议修复

QA 发现，上游 SMAClite 的 `seed` 会改变物理 tick 内的更新顺序，但不会改变由 `MapInfo.groups` 决定的初始位置。第一版虽然传入多个 seed，每个 ratio 的初态仍完全相同。修复后新增显式配置：

```text
spawn_jitter: [0, 3] map units
default: 0.0
formal small-budget run: 2.0
generator: numpy.default_rng(seed xor 0x5A17D0A1)
constraints: map bounds + normal terrain + no pairwise overlap
rejection limit: 64 deterministic attempts
```

`spawn_jitter=0` 保持基础协议和旧 smoke 不变；训练脚本拒绝用 0 生成 schema v2 学习证据。每次 reset 记录：seed、accepted attempt、实际 group centers、每个 Red/Blue/asset 的坐标与半径、最小净空、随机化配置 SHA-256、纯几何 layout SHA-256 和包含 seed 的 sample SHA-256。同 seed 得到相同几何和 state，不同 seed 得到不同几何。

自动测试连续检查了 11 个随机 seed，所有单位均在地图边界内、位于 NORMAL terrain、两两无碰撞。针对性测试总数更新为 **15 passed**。

动态动作协议也同步修复：资产动作不再是随实际 Blue 数量漂移的 `6+B`，而是固定为：

```text
asset_action_id = 6 + max_blue_agents = 11
```

三种 ratio 的资产动作语义因此一致；不存在的 Blue target 槽由 action mask 禁用。

### 9.2 无泄漏评估与 checkpoint

修复后的单 seed 运行对每个算法使用：

- training：每 ratio 15 个唯一布局，共 45 个；
- validation：每 ratio 5 个唯一布局，共 15 个；
- held-out：每 ratio 5 个唯一布局，共 15 个；
- training/validation/held-out 的 layout hash 交集均为空；
- `training_layout_hashes_per_ratio` 持久化全部 45 个训练布局 hash，`training_layout_schedule` 逐 episode 记录 ratio、实际 reset seed、accepted attempt、layout/sample/config SHA-256，`training_layout_manifest` 还保留实际 group centers、单位/资产坐标半径与合法性结果；第三方无需重跑学习器即可用 tracked summary 重算 train-vs-evaluation 交集，并可按 seed 只 reset 复核几何；
- validation 在 episode 15/30/45 选择 `(win_rate, mean_return)` 字典序最优 checkpoint；
- held-out 在 checkpoint 身份冻结后才进行一次 sweep，同时评估 initial、validation-best 和 final；
- validation-best 与 final 分别保存模型、optimizer、metadata 和文件 SHA-256。

当前 tracked summary 的训练布局字段是在原训练结束后，按已记录参数、1-based `training_seed + episode_index * 101` 和原 ratio 轮转顺序，仅执行 reset 重建的；写回前核对了 accepted attempt、config/layout/sample hash、每 ratio 唯一数及三类交集。它能证明“当前锁定代码与配置可确定性重建这些几何并且 split 无交集”，不能反向证明旧 learner 的历史轨迹或动作序列。修改后的脚本会在每次训练 rollout 内直接从 `final_info` 捕获这些字段，后续运行证据强度更高。`layout_sha256` 只覆盖六位小数几何摘要且不含 seed；`sample_sha256` 才包含 seed/attempt；二者都不是 raw state、地形文件或完整轨迹 hash。

文件命名为：

```text
outputs/smaclite_ad_baselines_randomized/checkpoints/{algorithm}_seed20260830_validation_best.pt
outputs/smaclite_ad_baselines_randomized/checkpoints/{algorithm}_seed20260830_final.pt
```

即使 QMIX/VDN 的 validation-best 恰好发生在最后一次评估，best 与 final 也保留为两个角色明确、metadata 不同的独立文件；MAPPO 的 best 在 episode 15，与 episode 45 final 是不同模型状态。

schema v2 机器证据的精确执行命令为：

```powershell
$env:PYTHONPATH = "$PWD\src"
& "D:\Software\Anaconda\envs\torch310\python.exe" `
  scripts\train_smaclite_ad_baselines.py `
  --algorithms qmix vdn mappo `
  --seeds 20260830 `
  --episodes 45 --episode-limit 60 `
  --validation-episodes-per-ratio 5 `
  --heldout-episodes-per-ratio 5 `
  --eval-every 15 --batch-episodes 3 `
  --agent-hidden-dim 32 --critic-hidden-dim 32 `
  --ppo-epochs 2 `
  --shaping-scale 0.5 --approach-weight 1.0 `
  --spawn-jitter 2.0 --device cpu `
  --output-dir outputs\smaclite_ad_baselines_randomized `
  --evidence-dir docs\evidence
```

### 9.3 随机布局重跑结果

配置仍为 45 training episodes、60-step horizon、`shaping_scale=0.50`、`approach_weight=1.0`、seed `20260830`。下表只报告 **独立 held-out 布局上的 validation-best**：

| 算法 | updates | best episode | held-out initial→best return | paired Δ | positive ratios | held-out win | 资产生命 | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| QMIX | 43 | 45 | -1.0000 → -0.8257 | +0.1743 | 3/3 | 0.000 | 1.000 | approach signal only |
| VDN | 43 | 45 | -1.0000 → -0.8633 | +0.1367 | 3/3 | 0.000 | 1.000 | approach signal only |
| MAPPO | 15 | 15 | -1.0000 → -0.8621 | +0.1379 | 3/3 | 0.000 | 1.000 | pipeline/approach only |

三算法在三个 held-out ratio 上的 mean-return delta 都为正；规则 `rush_asset` 在另外 15 个随机布局上保持 100% 胜率，说明任务可解。然而三个学习策略都没有伤害资产、没有提高 win rate。由此只能得出：网络在独立布局上学到了由距离 shaping 支持的“接近资产”行为，**任务学习仍未建立**。证据字段必须联合读取：

```text
learning_status     = small_budget_learning_signal_observed
task_success_status = win_rate_not_improved
formal_convergence_claim = false
```

这里的 `learning_status` 是预先定义阈值下的行为信号标签，不是显著性检验；当前只有一个训练 seed，也没有置信区间，不能写成“统计显著”或“稳定泛化”。旧固定布局的资产伤害和 QMIX 瞬时胜率不再用于主结论。

### 9.4 MAPPO 已知限制

当前共享 `VariableScaleMAPPO` 的 actor 能从 `self_obs` 看到 episode progress，但其 `CentralStateValue` 是仅编码 `state_entities/state_mask` 的 feed-forward critic。SMAClite-AD 的 global state entity 特征没有 remaining horizon；同一物理状态在 episode 早期和接近 60-step 截止时可能有不同价值，因此 critic 输入对这个有限时域任务不是完整 Markov state。

本轮没有热修改共用 MAPPO 架构，以免让 HAD 与 SMAClite-AD 基线在未审计情况下分叉。因此 MAPPO 的有限梯度、参数变化和 held-out approach return 只证明 pipeline 可运行；在给 critic 加入规范化 remaining horizon 并重新做 stock/AD 对照前，不应用它支持 MAPPO 性能结论。

### 9.5 最终证据

当前有效证据为 schema v2：

```text
docs/evidence/smaclite_ad_baselines_summary.json
docs/evidence/smaclite_ad_baselines_curve.csv
```

summary 明确包含 `legacy_fixed_layout_result.status = deterministic_pipeline_smoke_only`、环境协议/上游 commit/源码 hash/hash 规范、有效唯一布局数、全部 45 个训练布局的 ordered hashes/seed/sample hashes 与完整 manifest、validation/held-out 的实际布局 manifest、可由这些 tracked hashes 独立重算的 split 泄漏审计、best/final checkpoint metadata 与 SHA-256，以及 validation/held-out 分离结果。第 8 节保留为错误发现前的审计历史，不得作为最终结果引用。
