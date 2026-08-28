# 代码框架与第一阶段 HAD Demo

## 1. 当前目录

```text
02_q2_capability_aware_dynamic_command/
├── configs/
│   └── stage1_had_demo.yaml
├── scripts/
│   └── smoke_stage1.py
├── src/open_score/
│   ├── contracts.py
│   ├── nn.py
│   ├── envs/had_stage1.py
│   ├── stage1/entity_qmix.py
│   ├── stage1/losses.py
│   ├── stage2/outcome_model.py
│   ├── stage3/commander.py
│   └── stage4/coadaptation.py
├── tests/test_framework.py
├── upstream/README.md
├── pyproject.toml
└── requirements.txt
```

代码不是从旧 PyMARL 整仓复制，而是按 QMIX/REFIL/ALMA 的公开公式建立最小接口。这样第一阶段能在 Windows 原生 HAD 上调试；官方 ALMA/REFIL 仓库后续放在隔离的 WSL2 环境中做 baseline 复现。

## 2. 当前已经能做什么

- HAD 只用红蓝打击智能体和目标点；
- 通过 9 个离散运动原语控制连续加速度；
- 一个共享 Q 网络接受不同数量实体和任务 token；
- 变数量 mixer 保持 QMIX 单调性；
- 执行一次 HAD transition、Double-Q TD loss 和 backward；
- S2、S3、S4 的核心类可以单独实例化；
- 5 个单测覆盖 padding/排列不变性、mixer 单调性、S2/S3、S4 回滚和 HAD contract。

本轮还修复了 HAD 的两个阻断项：删除未使用但缺失的 `common.arguments` 导入；目标只受到蓝方攻击者伤害，避免红方防守者开火时摧毁自家目标。旧语义可由 Git 恢复。

## 3. 本机与目标硬件

当前 shell 实测为 Python 3.9.13、PyTorch 1.13 CPU build，并识别到 RTX 3060 Laptop；这与用户计划的 i7-14700 + RTX 5070 Ti 不一致，可能不是最终训练环境。迁移到目标机器时：

1. 建议 Python 3.10 或 3.11；
2. 从 [PyTorch 官方安装选择器](https://docs.pytorch.org/get-started/locally/)安装当时稳定的 Windows CUDA wheel；
3. 不在 requirements 中写死 CUDA build；
4. 用 `torch.cuda.is_available()` 和一次 backward 验证显卡；
5. Open-SMAX-AD 使用 WSL2，因为 JAX 官方不提供原生 Windows NVIDIA GPU wheel。

首阶段 HAD 的瓶颈大概率在 Python 环境循环和 `deepcopy`，不是 5070 Ti。i7-14700 先从 8 个 rollout worker 起做实测，再决定是否增到 12/16；不要在没有 profiler 时直接开满线程。

## 4. 安装与 smoke

在当前目录执行：

```powershell
python -m pip install -e .
python -m pytest
python scripts/smoke_stage1.py
```

当前机器的验证结果是 5 tests passed，smoke 在 CPU 上完成 HAD 环境步、loss 和反向传播。该结果只验证代码链路，不表示策略已经学会。

## 5. 第一阶段尚缺的五个核心文件

### 5.1 `stage1/replay.py`

存储完整 episode，字段包含当前/下一实体集合、mask、task、action、team reward、terminated、truncated 和 scenario id。采样时按 batch 内最大 $N,E,T$ padding，保留 `filled` mask。

### 5.2 `stage1/learner.py`

实现 GRU burn-in、sequence Double-Q、TD($\lambda$)、梯度裁剪、target hard update 和 optimizer checkpoint。先用单步 loss 对齐数值，再换序列 loss。

### 5.3 `envs/had_factory.py`

按配置采样 $(N_D,N_A,M)$ 并创建/复用 HAD 实例；负责 train/held-out 场景隔离。第一版 episode 内不增援。

### 5.4 `runners/episode_runner.py`

每个 worker 持有独立环境、seed 和 hidden state；死亡个体只允许 no-op。Windows 上先用 `spawn` 多进程并测吞吐，若序列化开销过大则退回串行批量环境。

### 5.5 `scripts/train_stage1.py` 与 `evaluate_stage1.py`

训练脚本只读取 YAML 并写结构化 JSONL/TensorBoard；评估脚本冻结 checkpoint，在每个规模和固定 episode seeds 上输出 win、timeout、target health、steps 和 latency。

## 6. 实现顺序

1. 先做单规模 2v2/1 target 过拟合，确认奖励和动作可学；
2. 接 episode replay 和 sequence learner；
3. 加 2/4/6 多规模 scenario sampling；
4. 加 Fixed-Capacity QMIX baseline；
5. 加 held-out 奇数人数评估；
6. 最后移植 REFIL attention-QMIX baseline。

不要一开始同时做双方学习。红方 QMIX 对蓝方脚本通过后再换边；两方自博弈、PSRO 和对手策略库均后移。

## 7. 第一阶段完成定义

只有同时满足下列条件才进入 S2：

- 单一 checkpoint 覆盖全部训练规模；
- held-out 人数无需重新初始化网络或 mixer；
- 三 seed 的方向一致；
- 比 Fixed-Capacity QMIX 的最坏规模胜率更高；
- agent/entity padding 和排列测试通过；
- checkpoint、配置、环境版本和评估 seeds 可追溯。

## 8. 方法图安排

上一版“冻结 S1、S4 作为未来工作”的五张 PNG 已删除，因为它们与当前四阶段闭环相冲突。新总图和四张分图应在 S1 Demo 收敛、各模块接口不再变化后按真实实现重绘，当前以[完整算法伪代码](04_方法实现规格与伪代码.md#6-完整算法伪代码)作为唯一规范。
