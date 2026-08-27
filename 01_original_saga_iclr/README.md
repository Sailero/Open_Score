# SAGA：集群对抗中规模无关的动态分组分簇与双重零样本泛化

面向 **ICLR 2027** 的研究项目工作区。核心科学问题：

> 能否学习一个单一的分层集群对抗策略，使其对"双方规模（含对抗中途增减员）"与"对手策略风格"实现联合的零样本泛化（dual zero-shot generalization）？

## 目录结构

```
Code/
├── README.md                        本文件
├── docs/
│   ├── 01_文献调研与研究现状.md      五条主线的文献综述 + 空白定位（Gap Analysis）
│   ├── 02_研究方案_SAGA.md          问题聚焦、研究价值、方法设计、实验协议、时间线（主文档）
│   ├── 03_预期结果与风险.md          按 claim 组织的预期结果、负面结果、风险对冲
│   ├── 04_资深工程师审视报告.md      批判性评估：可行性/价值/ICLR就绪度打分/P0-P3工程路线图
│   ├── 05_价值论证与部署路径.md      问题价值三重映射 + 部署契约（延迟实测）+ 三级部署路线
│   ├── 06_网络与训练系统设计规格.md  token schema v1.0 / 逐层网络规格 / JAX训练系统架构 / 验收清单
│   ├── 07_技术路线图.md             9周冲刺排期：决策门G1✅/G2、预案A/B、算力预算、你的任务清单
│   ├── 08_严格审稿意见与修订记录.md  第一轮自审：2处理论实质错误+全部修复记录
│   ├── 09_第二轮自审与一致性修订.md  理论v2复核（无新错误）+ 跨文档一致性 + 主张-实验-工程映射闭合检查
│   ├── 10_论文源流与领域地图.md      思想谱系图 + 五条主线导航 + 两周入行路径 + 术语速查（入行指南）
│   ├── 11_可行性复核与执行手册.md    依赖链无断链证明 + 算力逐项核算(~25 A100·天) + 三档出口 + 操作序列
│   ├── 12_实验就绪度审计_基线与ICLR标准.md  基线设计与实现状态 + 先手bug修复 + ICLR标准核查表
│   ├── 13_开源与可复现性审计.md      开源/自研盘点 + PettingZoo合规测试✅ + 开源差距清单
│   └── figures/
│       ├── smoke_test_rollout.png   原型环境一局对抗的轨迹渲染
│       └── demo_learning_curve.png  训练demo学习曲线
├── paper/                           论文草稿（每章一个md，状态见 paper/README.md）
│   ├── 00_title_abstract.md         标题+摘要（完整初稿，数字占位）
│   ├── 01_introduction.md           引言+3条贡献与创新点（定稿级）
│   ├── 02_related_work.md           相关工作（完整初稿）
│   ├── 03_problem_formulation.md    形式化：动态种群博弈+双重泛化矩阵+可证伪假设
│   ├── 04_method.md                 方法章：三模块+SxS课程+伪代码+超参表
│   ├── 05_experiments.md            实验章大纲（"图表合同"形式）
│   ├── 06_conclusion_limitations.md 结论/局限/伦理/可复现声明
│   ├── 07_appendix_outline.md       附录大纲
│   └── references.md                引用键映射
└── prototype/                       可运行原型 + demo（正式大规模实验将迁移到 JAX/Linux）
    ├── swarmbattle/
    │   ├── env.py                   自研两队对抗环境：变规模、中途增援、部分可观测、属性化异构
    │   └── scripts.py               参数化脚本对手库（SxS 课程的风格轴，原型含 4 种风格）
    ├── envs/
    │   └── adapters.py              统一环境适配层：SwarmBattle + MPE simple_tag + MAgent2 battle
    ├── saga/
    │   ├── encoder.py               模块A：实体场编码器（SAQA 交叉注意力，置换不变）
    │   ├── commander.py             模块B：对手锚定动态分簇指挥层（软聚类 + 风格嵌入 + 兵力-任务匹配，批处理）
    │   ├── policy.py                模块C：目标条件底层策略 + 置换不变 critic
    │   └── agent.py                 A+B+C 组装为 actor-critic（三个环境共用同一网络）
    ├── train_demo.py                训练 demo：PPO 训练 + 零样本规模迁移评测 + 跨环境冒烟
    ├── test_clustering_p0.py        P0 决策门测试：动态分簇可学习性（✅已通过 96%/0.81）
    ├── test_baselines.py            基线就绪度测试（4个学习型基线接口对齐✅）
    ├── test_pz_env.py               开源合规测试（PettingZoo官方API测试✅+确定性✅）
    ├── baselines/flat_mappo.py      基线B1：MAPPO-flat（固定维度扁平策略）
    ├── swarmbattle/pz_env.py        SwarmArena 的 PettingZoo 标准API封装
    ├── bench_latency.py             推理延迟基准（部署契约实测：64v64 CPU 12ms）
    ├── viewer/
    │   ├── build_viewer.py          生成自包含 HTML 回放器
    │   └── replay.html              （生成物）浏览器回放前端，双击即开
    ├── outputs/                     （生成物）checkpoint、回放 JSON、迁移评测结果
    └── run_smoke_test.py            冒烟测试：接口联通性 + 数量无关性验证 + 轨迹渲染
```

## 阅读顺序

1. `docs/02_研究方案_SAGA.md` —— 问题聚焦（§1）、研究价值（§2）、方法（§3）、实验设计（§4）、时间线（§6）
2. `docs/01_文献调研与研究现状.md` —— 为什么这个位置是空白（§8 Gap Analysis 是核心表格）
3. `docs/03_预期结果与风险.md` —— 每个 claim 的预期结果与降级预案

## 运行 demo（使用 conda 环境 gpu_py_310）

已在本机验证：Python 3.10.14 / torch 2.4.1+cu124（RTX 3060）/ pettingzoo 1.25 / magent2 0.3.4。

```bash
# 完整训练 demo：PPO 训练 SAGA（SwarmBattle 6v6 vs 脚本对手）
#   -> 学习曲线 docs/figures/demo_learning_curve.png
#   -> 零样本规模迁移评测（6v6 训练 -> 12v12 / 24v24 / 12v24+增援 直接测试）
#   -> 同一网络在 MPE simple_tag 与 MAgent2 battle 上 rollout（架构通用性）
#   -> HTML 回放器 prototype/viewer/replay.html（双击用浏览器打开）
C:\Users\hp\.conda\envs\gpu_py_310\python.exe prototype/train_demo.py

# 快速验证（~2 分钟）
C:\Users\hp\.conda\envs\gpu_py_310\python.exe prototype/train_demo.py --episodes 20

# 消融：去掉模块B（指挥层）
C:\Users\hp\.conda\envs\gpu_py_310\python.exe prototype/train_demo.py --no-commander

# 仅接口冒烟（不训练）
C:\Users\hp\.conda\envs\gpu_py_310\python.exe prototype/run_smoke_test.py
```

## Demo 实测结果（2026-07-20，RTX 3060，120 局约 24 分钟）

诚实汇报：这是**流水线验证级**的结果，不是论文级结果（120 局 vs 论文计划的 10⁹ 环境步）。

- **训练有信号**：6v6 对削弱版 greedy 脚本，回报从 -7.9 → 约 -5.5（70-90 局段），胜率 0% → 5-10%（`docs/figures/demo_learning_curve.png`）；
- **全评测协议跑通**：零样本规模迁移（6v6→12v12/24v24/12v24+增援）自动评测并输出 `outputs/scale_transfer.json`——当前训练量下大规模迁移胜率为 0，这正是论文要解决的问题本身，评测轴已就绪；
- **架构通用性验证**：同一组网络参数（111,250 个，与数量无关）在 SwarmBattle（6-24 单位）、MPE simple_tag（3 追捕者）、MAgent2 battle（12v12）三个环境全部完成 rollout；
- **回放前端**：`prototype/viewer/replay.html` 双击即开，内含训练前/训练后 6v6/零样本 24v24 三段回放，可播放、拖动、查看双方存活数。
- 已知问题：笔记本 GPU 偶发 CUDA unknown error（疑似 Windows TDR），训练脚本已加 CPU 自动回退；正式训练在 Linux/WSL2 上不受影响。

## 公开基准环境策略

| 环境 | 角色 | 状态 |
|---|---|---|
| SwarmBattle（自研） | 主战场：规模/增援/对手脚本完全可控 | ✅ 本机跑通 |
| MPE simple_tag（PettingZoo） | 公认基准 sanity check（连续对抗） | ✅ 本机跑通 |
| MAgent2 battle_v4 | 公认大规模两队对抗基准（架构通用性证据） | ✅ 本机跑通 |
| SMAC/SMACv2 | 引用单位设定；正式实验用 SMAX 替代（快 4 个数量级） | 不部署 |
| JaxMARL/SMAX、Gigastep | 正式大规模训练（JAX 需 Linux/WSL2 GPU） | 第二阶段 |

## 关键设计决策（TL;DR）

- **场景**：二维连续平面两队对抗（SMAX 风格），单位 = 质点运动学 + 射程/血量/冷却，属性向量参数化异构类型；训练域 4–16 单位，测试外推 64+，episode 内泊松增援/减员。
- **方法** = 置换不变实体编码（模块A）+ 对手锚定动态分簇指挥（模块B，簇数由敌方实时结构决定而非超参数）+ 目标条件共享底层策略（模块C）+ 规模×风格二维联合种群课程（脚本库 + 自我快照 + 轻量 exploiter）。
- **容量控制**：不做 3D 高保真、不做通信学习；LLM 机型适应仅作 Future Work；异构属性外推作为附加实验轴。
- **降级预案**：若动态分簇不稳定，退守"数量无关分层 + 联合课程 + 双重零样本评测协议"仍构成完整贡献。
