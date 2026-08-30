# Open-SCORE：从小型子博弈到开放人口指挥

> **我们学习可跨人数复用、对策略差异敏感的小规模单目标攻防能力，再用可校准的局部博弈模型驱动双方滚动分组，检验其能否在伤亡、增援、未知对手和未见总规模下提高最坏全局胜率。**

```text
S1 规则策略 + 变规模 QMIX/VDN/MAPPO → 可复用局部决策策略（本轮）
S2 结局—时间监督学习 + 独立校准      → 直观、可审计的评估器（本轮）
S3 对手信念 + 双边列生成 + maximin   → 受限动态分组指挥（冻结）
S4 宏观价值残差 + 种群式守卫更新     → 结果对齐的共同适应（冻结）
```

## 当前研究决定

- 两个主环境分别是自研 HAD 和可审计的 SMAClite-AD；未经修改的 SMAClite stock 场景只作外部复现锚点。
- HAD 中 Red 是目标防守方、Blue 是攻击方；由于攻击单位开火后自毁，本轮只注册严格 `Red > Blue` 的 `2v1、3v1、3v2、4v1、4v2、4v3`。SMAClite-AD 中 Red/Blue 角色相反，结果不得混写。
- 一套共享的变实体网络覆盖已注册配比。HAD 动作为 27 个量化加速度；SMAClite-AD 保留 SMAC 式移动/指定目标攻击语义，并通过 mask 处理动态规模。
- QMIX/VDN 使用共享实体编码和价值分解，MAPPO 使用共享 actor 与集中实体 critic；学习进展课程借鉴 SPMARL，但当前实现不是完整 REFIL 或 SPMARL 复现。
- PSRO 代码保留为未来对手多样性实验，本轮不运行、不作为 S1/S2 验收门槛。
- S3 定义为受限开放人口随机分配博弈；LP 只是单个宏观状态的阶段求解器，组合候选由双边列生成控制，长期性由宏观转移/终端价值提供。
- 未知对手的行为上下文必须改变 S3 的 ambiguity set；开集时回退保守 maximin，不能只输出一个对手标签。
- S4 先只做宏观价值残差；只有 held-out 最坏胜率改善后才追加新的 S1 最佳响应，不把全链路反传作为必要步骤。
- HAD 是第一验证平台；SMAClite-AD 是首选第二环境。Gigastep 仅建议在 WSL2 做大规模外部压力测试。

## 阅读入口

1. [研究结论与创新边界](docs/00_研究结论与阅读路线.md)
2. [问题定义](docs/01_问题定义与创新边界.md)
3. [相关工作、环境与源码审计](docs/02_相关工作与环境审计.md)
4. [论文式方法简版](docs/03_方法简版.md)
5. [数学模型与四阶段规格](docs/04_方法实现规格与伪代码.md)
6. [实验协议](docs/05_实验设计与预期证据.md)
7. [工作量与论文边界](docs/06_工作量_档位与拆稿.md)
8. [S1 代码、运行与 pilot 记录](docs/07_代码框架与第一阶段Demo.md)
9. [S1/HAD 复现、训练与证据报告](docs/reports/S1_HAD_复现与验证报告.md)
10. [研究问题讨论与 S3/S4 可行性](docs/08_研究问题讨论与S3_S4可行性.md)
11. [SMAClite 原版与 AD 环境审计报告](docs/09_SMAClite原版与AD环境审计报告.md)
12. [第二阶段监督学习与评估报告](docs/10_第二阶段监督学习与评估报告.md)
13. [S1/S2 里程碑验收与续训接口](docs/11_S1_S2里程碑验收与续训接口.md)
14. [论文框架](paper/论文框架.md)

## 运行

```powershell
D:\Software\Anaconda\envs\torch310\python.exe -m pip install -e .
D:\Software\Anaconda\envs\torch310\python.exe -m pytest -q
D:\Software\Anaconda\envs\torch310\python.exe scripts\smoke_stage1.py

# 固定上游 commit 安装 SMAClite；不会覆盖官方 Gym ID 或修改上游地图
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\install_smaclite.ps1

# SMAClite stock：官方 150-step horizon + 算法默认 batch/replay 的单 seed 50k pilot
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_epymarl_stock_smoke.ps1 `
  -Algorithm all -TMax 50000 -TestInterval 45000 `
  -TestEpisodes 20 -LogInterval 5000 -UseStockTrainingDefaults

# HAD：三算法的小预算动态规模复现入口
D:\Software\Anaconda\envs\torch310\python.exe scripts\train_stage1_baselines.py `
  --algorithms qmix vdn mappo --device cuda --episodes 72

# SMAClite-AD：2:1、3:2、5:3 的三算法训练链
D:\Software\Anaconda\envs\torch310\python.exe scripts\train_smaclite_ad_baselines.py `
  --algorithms qmix vdn mappo --ratios 2:1,3:2,5:3 --device cuda

# HAD 前向推演数据 → S2 监督训练、校准与完整评估
D:\Software\Anaconda\envs\torch310\python.exe scripts\train_stage2.py `
  --config configs\stage2_had_smoke.yaml --device cuda
```

官方 ePyMARL 的 stock QMIX/VDN/MAPPO 安装和运行命令分别见 `scripts/install_epymarl_windows.ps1` 与 `scripts/run_epymarl_stock_smoke.ps1`。所有默认训练预算仍是开发/冒烟配置；论文结论必须另做预注册、多 seed、独立配对评估与置信区间。

## 已实现与未实现

已实现：HAD 严格六规模适配、规则策略、共享 QMIX/VDN/MAPPO、replay/课程/checkpoint；固定上游版本的 SMAClite stock 与独立 SMAClite-AD；官方 ePyMARL stock 三算法训练链；S2 的真实 HAD forward-rollout 数据契约、root 隔离切分、联合结局—时间 ensemble、校准、风险上界和决策指标。

当前证据边界：stock 50k pilot 的三算法 return 均明显变化，但 near-final `battle_won` 仍为 0；AD 三算法在独立随机布局上只出现 approach-shaping signal；HAD 定向训练的智能策略 held-out 弱于规则。S2 的失守排序已有信号，但时间 MAE 未优于均值基线，风险上界也过宽。尚无正式多 seed 收敛、held-out 配比/策略泛化或论文胜率结论；S3/S4 未实现且本轮冻结。
