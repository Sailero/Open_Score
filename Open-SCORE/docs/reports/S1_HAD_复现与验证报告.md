# S1 / HAD 复现、训练与证据报告

> 冻结日期：2026-08-30  
> 平台：Windows、`torch310`、Python 3.10.20、PyTorch 2.13.0+cu130、RTX 5070 Ti 16 GB  
> 结论级别：工程闭环与小预算学习检查；**不是正式收敛结论**

## 1. 先给结论

本轮已经把 HAD 的 Stage 1 做成一个可安装、可训练、可评估、可保存和可续训的完整闭环，并提供了两类可直接运行的策略：

1. 规则策略：Red `guard` / Blue `rush`，无需 checkpoint；
2. 智能策略：一套跨六个合法人数配比共享的 variable-entity QMIX，保留 best 与 final checkpoint、SHA256、重载评估和 `--resume` 接口。

VDN、QMIX、MAPPO 均在相同观测、相同 27 个加速度动作和相同 HAD 动力学上完成了真实 CUDA rollout 与参数更新。当前代码版本的 36 局/算法 smoke 中，QMIX 的 paired validation payoff 从 −0.600 提升到 −0.267；VDN 未改善；MAPPO 中途轻微改善但 final 回落。

更长的 Red-QMIX targeted 训练在 2v1 上从 −0.4 提升到 best 0.0、final −0.2；从该 best checkpoint 续训六规模后，阶段初始 −0.533、阶段 best −0.367、final −0.633。独立 held-out seed 上，规则 guard 为 +0.433 / 71.7% 胜率，动态 QMIX best 为 −0.433 / 28.3%。因此：

- “环境和三算法可以正常训练”成立；
- “同一智能策略真实覆盖六个配比”成立；
- “小预算内出现过相对初始化改善”成立；
- “智能策略已经稳定收敛、达到优秀策略或超过规则策略”**不成立**。

当前可用决策建议是：规则 `guard` 作为默认可部署策略，动态 QMIX best 作为可续训的研究策略。不能把 best-only 结果包装成 final 收敛。

## 2. HAD 语义与严格人数约束

### 2.1 红蓝角色

HAD 源码审计得到的实际语义如下。

| 对象 | 语义 |
|---|---|
| Red | 保护资产的防守方 |
| Blue | 突击资产的攻击方 |
| Entity | Red 保护的固定资产 |
| Red 胜 | 所有 Blue attack agents 被毁 |
| Blue 胜 | 资产健康值降到终止阈值 |
| 超时 | 按本 Stage-1 协议记 Red 防守成功 |

`AttackAgent.update_status` 明确规定：任意阵营的打击智能体一旦开火，随后立即把自身 `Health` 置为 0。由于这个自毁语义，本轮按用户要求把 HAD 主协议收紧为：

$$n_{Red}>n_{Blue},\qquad 1\le n_{Red},n_{Blue}\le4.$$

合法规模只有六格：

```text
2v1, 3v1, 3v2, 4v1, 4v2, 4v3
```

`HADStage1Adapter` 在构造时直接拒绝 Red≤Blue；训练 sampler、配置、smoke、PSRO 旧入口和测试都同步采用严格大于，而不是只在某一个 YAML 中写约束。

### 2.2 为什么不再保留 1v1 / 2v2 HAD 主结果

旧文档把 Red≥Blue 的十格登记为训练域，但这与本轮明确要求不一致。现在 HAD 主结果不再使用平局人数；SMAClite 原版的标准对称场景属于另一条复现轨道，不能用来绕过 HAD 的业务约束。

## 3. 环境修改边界与可认可性

### 3.1 本轮保留的物理语义

本轮没有把 `guard`、`rush` 或“朝最近敌人”塞入学习动作。学习者始终只输出：

- 一个零加速度；
- $\{-1,0,1\}^3$ 的 26 个非零归一化方向。

HAD 再乘以每个智能体自身的最大加速度，并继续使用原有位置、速度、转向、命中和规则开火逻辑。规则宏观语义只存在于独立 baseline controller 中。

### 3.2 Stage-1 adapter 的显式改动

| 改动 | 原因 | 审计措施 |
|---|---|---|
| 严格 `Red > Blue` | 自毁打击语义和用户约束 | 构造期异常 + sampler + 测试 |
| 目标位置按 seed 在注册区域随机 | 防止记住单点 | 相同 seed 重现、不同 seed 改变 |
| 变实体/变人数 mask | 一套参数覆盖六规模 | padding 不改变 Q 值测试 |
| 终局 ±1 + potential shaping | 缓解纯加速度长时域稀疏奖励 | shaping 前后值进入 `info`，Red/Blue 严格零和 |
| terminal potential 置零 | 保留终局任务语义 | 终局回归测试 |
| `state_mask` 表示真实槽位而非存活 | 全灭终局仍有真实实体槽 | all-destroyed 长跑 bug 回归测试 |
| best/final 分开保存 | 防止只报最好点 | summary 同时记录二者 |

当前 potential 为防守视角，包含目标健康、双方存活比例、攻击者到目标净空距离和防守者到攻击者拦截距离；训练 CLI 的 `shaping_scale=0.5`。死亡由 health/alive 特征表示，不能再把死亡实体错误当成 batch padding。

### 3.3 学术认可边界

HAD 是自研环境，因此即使接口严谨，也不能单独获得与公开标准 benchmark 相同的外部认可。本轮提高可信度的方式是：

- 原始 HAD 物理引擎与 Open-SCORE adapter 分层；
- 所有 adapter 改动、配置、seed、曲线和哈希可审计；
- 三算法共用同一 contract，不给某一算法额外信息；
- 明确区分 smoke、短预算趋势、held-out 和正式收敛；
- 由未修改 SMAClite 标准场景承担公开基线复现，由 SMAClite-AD 承担第二动力学外部验证。

这比声称“改后的 HAD 等同标准 benchmark”更可信。

## 4. 策略与算法实现

### 4.1 规则策略

`RuleBasedController` 提供 `guard`、`rush`、`intercept`、`split_rush` 和 `engage`。本轮冻结的默认策略对是：

- Red：`guard`，在资产与最近威胁之间构造拦截点；
- Blue：`rush`，直接趋近资产。

规则策略和学习策略使用相同的 27 个底层加速度 ID；区别只是规则控制器显式计算目的地。

### 4.2 动态规模智能策略

智能策略是共享参数的 entity–GRU–QMIX：

- masked set encoder 处理数量变化的实体集合；
- 一个 recurrent utility network 在所有 agent slot 和六个规模间共享；
- state-conditioned monotonic mixer 保持 $\partial Q_{tot}/\partial Q_i\ge0$；
- agent/entity/action mask 明确区分死亡、不可用动作和 batch padding；
- checkpoint 的网络维度不依赖实际人数。

设计依据来自 [REFIL](https://proceedings.mlr.press/v139/iqbal21a.html) 的可变实体价值分解、[EPC](https://sites.google.com/view/epciclr2020) 的小到大 population curriculum，以及 [SPMARL](https://proceedings.mlr.press/v267/zhao25o.html) 的 TD 学习进展选规模思想。

当前实现**不是** REFIL 或 SPMARL 的逐行复现。它采用三阶段规模解锁、快慢学习信号 EMA、访问奖励、20% 概率均匀覆盖，并新增“每个新解锁规模至少强制首访一次”。是否优于简单均匀采样仍需正式消融。

### 4.3 三个可运行算法

| 算法 | 本轮实现 |
|---|---|
| VDN | 共享 recurrent utility；仅对活跃 agent Q 做标准加和 |
| QMIX | 共享 utility + 变实体单调 mixer；Double-Q + TD(λ) |
| MAPPO | 共享 recurrent categorical actor + 集中式 entity-state critic；GAE + clipped PPO + value clipping；本轮仅作管线 smoke |

三者都使用全 episode padding，MAPPO 的 actor loss 按活跃 agent mask 计算，critic/GAE 按有效时间 mask 计算；MAPPO rollout batch 每次更新后清空，未把 replay 错当成 on-policy 数据。

独立 QA 发现当前 MAPPO 集中 critic 是 feed-forward entity critic，而 HAD/SMAClite-AD 的 global state 没有显式 `step_count` 或剩余 horizon。actor 的 GRU 可以从历史隐式计时，critic 却会把相同物理状态在早期/晚期视为同一输入；当 timeout 本身决定胜负时，这不是严格的有限时域 Markov value baseline。因此当前 MAPPO 数值只能证明 PPO/GAE 数据链和梯度可运行，不能作为公平性能复现。下一协议必须给 state 增加剩余时间特征，或把 critic 改成 recurrent 后重新生成 checkpoint 与证据。

## 5. 代码与入口

| 文件 | 作用 |
|---|---|
| `src/open_score/envs/had_stage1.py` | HAD 双边 adapter、严格规模、观测与 reward contract |
| `src/open_score/stage1/entity_qmix.py` | variable-entity QMIX |
| `src/open_score/stage1/baselines.py` | VDN、MAPPO、GAE/PPO learner |
| `src/open_score/stage1/curriculum.py` | 严格六规模课程和首访保证 |
| `src/open_score/stage1/runner.py` | rule/Q/PPO controllers 与双边 rollout |
| `scripts/train_stage1_baselines.py` | 三算法训练、paired validation、best/final、warm-resume、CSV/JSON |
| `scripts/evaluate_stage1_had_policy.py` | rule 或 checkpoint 的独立推理评估 |
| `configs/stage1_had_reproduction.yaml` | smoke / 小预算 / 正式实验分级配置 |
| `tests/test_stage1_had_baselines.py` | 严格规模、首访、VDN、MAPPO、终局 state/reward 回归 |

## 6. 测试记录

在 `D:\Software\Anaconda\envs\torch310\python.exe` 下完成：

- Stage1/HAD + 原 framework 共 16 项测试通过；
- CUDA 单步 HAD→QMIX→backward smoke 通过；
- VDN whole-episode update 通过；
- MAPPO 两 epoch clipped on-policy update 通过；
- checkpoint 保存、warm-resume optimizer/target/curriculum、再次评估通过；
- 长跑触发的 all-destroyed terminal state 已加入回归测试。

CUDA smoke 的关键字段为：Red=3、Blue=2、实体数=6、动作数=27、zero-sum check=0、loss/backward 有限。

`pygame` 是 HAD import 的实际运行依赖，已加入 `pyproject.toml` 和 `requirements.txt`。

## 7. 实验分级与真实结果

### 7.1 证据等级

| 等级 | 能说明什么 | 本轮状态 |
|---|---|---|
| 接口 smoke | reset/step/mask/梯度/保存可运行 | 通过 |
| 小预算学习检查 | paired validation seed 下是否出现改善信号 | 部分通过 |
| 小规模收敛验证 | 多节点平台、best≈final、独立 seed 稳定 | 未通过 |
| 正式实验 | ≥5 seeds、置信区间、预注册预算/holdout | 未执行 |

### 7.2 当前代码版三算法 smoke

协议：Red 对固定 Blue `rush`，同一 seed 20260830；六规模课程；每算法 36 局；每个评估节点每格 5 局。payoff 和胜率均为受控 Red 视角。MAPPO 还受第 4.3 节 finite-horizon critic 缺失时间输入的限制，只验训练管线。

| 算法 | 初始 payoff / 胜率 | best payoff / 胜率 | final payoff / 胜率 | 结论 |
|---|---:|---:|---:|---|
| QMIX | −0.600 / 20.0% | −0.267 / 36.7% | −0.267 / 36.7% | 观察到短预算改善 |
| VDN | −0.600 / 20.0% | −0.600 / 20.0% | −0.600 / 20.0% | 未建立学习 |
| MAPPO | −0.533 / 23.3% | −0.467 / 26.7% | −0.600 / 20.0% | 中途改善、final 回落 |

证据：[summary](../evidence/stage1_had_current_smoke_summary.json)、[curves](../evidence/stage1_had_current_smoke_training_curves.csv)。

### 7.3 Targeted Phase A：Red QMIX 固定 2v1

协议：300 局、8-episode batch、512 replay、每局 2 次 update、epsilon 在 10,000 环境步衰减、10 个固定 paired validation seed。该集合被反复用于 best checkpoint 选择，因此不是独立 test。

| 节点 | payoff | 胜率 |
|---|---:|---:|
| 初始 | −0.400 | 30.0% |
| best，episode 200 | 0.000 | 50.0% |
| final，episode 300 | −0.200 | 40.0% |

共 8,395 个环境步和 586 次 learner update。TD 绝对误差最后约 0.134。best 优于初始，但 episode 250/300 回落，不能称平台收敛。

证据：[summary](../evidence/stage1_had_targeted_2v1_summary.json)、[curves](../evidence/stage1_had_targeted_2v1_training_curves.csv)。

### 7.4 Targeted Phase B：从 Phase A best 续训六规模

Phase B 从 episode 200 的 best checkpoint 恢复 online、target、optimizer、learner step、curriculum、环境步数和 epsilon 进度；模型不是重新初始化。但 replay buffer 没有写入 checkpoint，rollout RNG 也不是逐 bit 恢复，CLI 仅校验算法、阵营和 shaping scale，未全量校验所有模型/训练超参数。因此准确名称是 **checkpoint warm-resume**，不是 exact continuation。再训练 300 局，每格用相同的 10 个 paired validation seed 评估。

| 节点 | 六规模平均 payoff | 胜率 |
|---|---:|---:|
| phase 初始 | −0.533 | 23.3% |
| phase best，累计 episode 275 | −0.367 | 31.7% |
| phase final，累计 episode 500 | −0.633 | 18.3% |

实际采样次数为：2v1=248、3v1=32、3v2=58、4v1=61、4v2=42、4v3=59。六格都进入真实训练，因而“动态规模训练”不是仅指 forward 接口可运行。

best 的逐规模 payoff 为：

```text
2v1  0.0    3v1 -0.2    3v2 -1.0
4v1 +0.2    4v2 -0.4    4v3 -0.8
```

best checkpoint 的选择标准是六规模**平均 payoff**，不是 worst-scale；best 时最坏规模仍为 −1.0。Phase B best 相对 phase 初始改善，但 final 显著回落，显示遗忘/策略不稳定；本阶段 `learning_status=learning_not_established`。

Phase A summary 的 `updates=586` 是累计 learner step。warm-resume 使用的 episode-200 checkpoint 为 learner step 386；Phase B final 为 step 972，所以 Phase B 本轮新增 586 次更新，而不是新增 972 次。

证据：[summary](../evidence/stage1_had_targeted_dynamic_summary.json)、[curves](../evidence/stage1_had_targeted_dynamic_training_curves.csv)。

### 7.5 Blue 固定 2v1 诊断

为检查失败是否只来自防守侧的间接拦截任务，保持 Red>Blue 不变，额外训练 Blue-QMIX 300 局。其 initial/best/final 均为 −1.0、0% 胜率和 50 步超时。虽然 TD 降到约 0.09，但策略结果没有改善，因此没有继续做 Blue 六规模训练。

证据：[summary](../evidence/stage1_had_targeted_blue_2v1_summary.json)、[curves](../evidence/stage1_had_targeted_blue_2v1_training_curves.csv)。

### 7.6 Checkpoint 重载与 held-out

动态 QMIX best checkpoint 从磁盘重新构造网络和 learner 后，在选模使用的同一 paired validation seed 集上得到 −0.367 / 31.7%，与保存时 summary 完全一致；这只证明 checkpoint 在当前 workspace 可真实重载运行，而不是独立泛化测试。

证据：[reload eval](../evidence/stage1_had_qmix_best_reload_eval.json)。

最终 test 使用从未参与训练或选模的 seed 组 29260830..29260889，按 scale-major 顺序每格 10 局；两种策略严格使用相同 episode seed：

| 策略 | validation mean / 胜率 | held-out test mean / 胜率 | held-out worst scale | 结论 |
|---|---:|---:|---:|---|
| Red rule `guard` | +0.167 / 58.3% | +0.433 / 71.7% | 0.0 | 当前默认可用策略 |
| validation-selected dynamic QMIX | −0.367 / 31.7% | −0.433 / 28.3% | −1.0 | 明显弱于规则，不可称优秀 |

合并证据：[held-out test](../evidence/stage1_had_heldout_test.json)。原始只读评估输出：[rule held-out](../evidence/stage1_had_rule_heldout_eval.json)、[QMIX held-out](../evidence/stage1_had_qmix_best_heldout_eval.json)。

### 7.7 性能计时限制

2026-08-30 17:35 起同机存在 `E:\Code\SARC` 的外部 GPU 训练进程。本项目没有终止或干预该进程。targeted dynamic、Blue 诊断和后续 smoke 的 wall time / steps-per-second 不是隔离性能基准，不能据此比较 QMIX、VDN、MAPPO 的速度；报告只把它们作为“任务确实执行完成”的运行元数据。

## 8. 两套策略怎样运行

### 8.1 安装与测试

```powershell
cd E:\Code\Open_Score\Open-SCORE
& D:\Software\Anaconda\envs\torch310\python.exe -m pip install -e .
& D:\Software\Anaconda\envs\torch310\python.exe -m pytest -q
& D:\Software\Anaconda\envs\torch310\python.exe scripts\smoke_stage1.py
```

### 8.2 运行规则策略

```powershell
& D:\Software\Anaconda\envs\torch310\python.exe scripts\evaluate_stage1_had_policy.py `
  --policy rule --side Red --rule-style guard --device cuda `
  --episodes-per-scale 10 --seed 29260830
```

### 8.3 运行智能策略 checkpoint

当前 workspace 的 best checkpoint：

```text
outputs/stage1_had_targeted_dynamic/checkpoints/qmix_seed20260830_best.pt
SHA256 e4cb143a647c72d11ecc6c5832bea4e6ab0316d0688369440b13150e634e174b
```

评估：

```powershell
& D:\Software\Anaconda\envs\torch310\python.exe scripts\evaluate_stage1_had_policy.py `
  --policy checkpoint --algorithm qmix `
  --checkpoint outputs\stage1_had_targeted_dynamic\checkpoints\qmix_seed20260830_best.pt `
  --side Red --device cuda --episodes-per-scale 10 --seed 28260830
```

继续训练：

```powershell
& D:\Software\Anaconda\envs\torch310\python.exe scripts\train_stage1_baselines.py `
  --algorithms qmix --device cuda --seeds 20260830 `
  --episodes 300 --batch-episodes 8 --replay-episodes 512 `
  --updates-per-episode 2 --epsilon-anneal-steps 10000 `
  --eval-every 75 --eval-episodes-per-scale 10 `
  --resume outputs\stage1_had_targeted_dynamic\checkpoints\qmix_seed20260830_best.pt `
  --output-dir outputs\stage1_had_targeted_dynamic_resume
```

`outputs/` 和 `*.pt` 被 `.gitignore` 排除。checkpoint 在当前 workspace 可直接运行，但 clean clone 不包含弱、未收敛权重；需按 Phase A→Phase B 命令重跑。仓库保留 summary、curve、checkpoint hash 和完全可再生成的入口。

当前 pilot 权重生成于完整协议元数据加入之前，包含算法、阵营、规模和 shaping，但没有保存 `max_steps` 与模型配置；评估器因此会输出 `legacy_checkpoint_incomplete_metadata`，不能从文件本身证明 CLI horizon 完全匹配训练。新训练 checkpoint 已写入 `protocol_version/environment_protocol/continuation_contract`，并强校验 horizon 与 shaping；这不会追溯性改变旧权重或其 SHA256。

## 9. 为什么现在还不能写“收敛”

本轮同时看到了三个反证：

1. Phase A best 0.0，但 final 回落到 −0.2；
2. Phase B best −0.367，但 final 回落到 −0.633；
3. held-out 上智能策略远弱于规则 guard。

loss / TD error 下降只证明 value fitting 变稳定，不等价于胜率收敛。Blue 诊断尤其说明“TD≈0.09”可以与“胜率=0%”同时出现。

正式声称收敛前至少需要：

- 5 个预注册 seed；
- 每格独立 held-out 至少 50–100 局及 Wilson / bootstrap 区间；
- 连续至少 3 个评估节点进入平台，且 best–final gap 受控；
- 与 rule、VDN、MAPPO、uniform curriculum、固定规模模型比较；
- 旧规模遗忘曲线和最坏规模性能；
- 隔离或记录并发负载，不把墙钟吞吐当算法结论。

## 10. 下一轮最小续接方案

不引入 PSRO，优先解决 S1 学习质量：

1. 将 validation seed 与最终 test seed 在训练前固定分离；
2. 为六规模课程加入最小配额，而不只是“首次必访 + 概率覆盖”；
3. 比较 uniform sampler，确认当前 learning-progress sampler 是否过度回采 2v1；
4. 做 n-step / action-repeat / reward-normalization 消融，但每项都作为 protocol 版本，不静默改变 HAD；
5. 加入 REFIL/attention-QMIX 强 baseline，判断失败来自 mixer 还是探索；
6. 如果使用 rule trajectory 辅助，只能明确写成 imitation warm start + RL，不能冒充纯 RL；
7. 在至少两个 seed 上先让 2v1 best≈final，再扩大六规模。

S2 可以先用 rule policy 和版本冻结的 pilot checkpoint 采集“多策略质量层级”数据，但必须把 policy ID / checkpoint hash 作为条件，并把当前 QMIX 标成 `pilot_non_converged`。不能把它当作已获得的优秀 S1 oracle。

## 11. 哈希与证据清单

### 11.1 当前源码哈希

```text
had_stage1.py                    0EB6C619DEAE152B71F3EC133A6D082B9B46D0275EEB0C9A12EA433B3932FFA9
baselines.py                     C1F335BB56E1FEF5538FC3A9299305A4A3B4197F49D29172B3967002D722A94B
curriculum.py                    6A727E1F7371EBE903BE18E6A259627AD3DAF0038F7C4B56087790C2F0703CC4
train_stage1_baselines.py        293D820F7D2CA6CA2CF552CC641888A032B8D31813FD2D8C047430E8A7713562
evaluate_stage1_had_policy.py    D097F9C30508C4045BACA13EDAC2499191CC52FEBCEAD7A516AEBCB4C97C403D
```

### 11.2 关键证据哈希

```text
stage1_had_current_smoke_summary.json            3EB7358408CF6F1068D316292C7BE263FF54BD3947D711DF2BC96CC6B690D183
stage1_had_targeted_2v1_summary.json              6D76C458ECF151392CBD49B60F65EB94EE180F17E98DF09DDAC3E77C154EF8EA
stage1_had_targeted_dynamic_summary.json          45639DAB25E6145C87947C2B99CB9845C1777139853C8853A55EEF4B3EE7C9F0
stage1_had_targeted_blue_2v1_summary.json         58093D0CFB718CB430B5FC6AACACF0F1F31FA5733B55F8748A6871AFE3216CDD
stage1_had_rule_heldout_eval.json                 FC57B30DA5F91A21478789BA583E243F9FC1F91D27F1EF7DB630CE5E10C13C4C
stage1_had_qmix_best_heldout_eval.json            9E4BFE86910FB103882A7DFD1D8ADE0E76A8B8BF3EC37A786B62EF49987E568B
stage1_had_qmix_best_reload_eval.json             9D7F8D77D69E51D8C9763D8BE7580ED7C9DA6BB15F193BE420EAE55A945D80DE
stage1_had_heldout_test.json                      14F1A6CC9EEAE1FE6B3A65DEEE0F7279BDD2CCEA904B4A396AD723F9FC068F85
```

源码在报告生成后若继续修改，应重新计算哈希；实验 JSON 内的 checkpoint SHA256 是判断权重是否一致的首要依据。
