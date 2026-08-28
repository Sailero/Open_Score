# Open-SCORE：从小型子博弈到开放人口指挥

> **我们研究如何把一个开放规模的多目标攻防态势拆成若干个1–4对1–4单目标子博弈，并利用可复用的微操能力、局面结局预测和实时分组博弈，在伤亡或增援后维持全局防守胜率。**

```text
S1 1--4 vs 1--4 local QMIX + team PSRO
     ↓ versioned local capability
S2 outcome/time MLP + empirical 95% interval
     ↓ local payoff evidence
S3 bilateral grouping maximin game
     ↓ global assignments
S4 final-outcome residual + guarded alternating updates
     └──────── refresh S1/S2/S3 ────────┘
```

## 当前决定

- HAD是第一平台：单随机目标、双方各1–4人、双方只控制27个量化加速度。
- S1使用小型DeepSets-GRU-QMIX；warm-start成功后再做3轮PSRO pilot，最多8轮。
- S2使用规范化输入的简单MLP ensemble，只预测结局—时间联合分布。
- S3根据S2区间构造双方分组收益矩阵并在线求maximin。
- S4用完整轨迹最终胜负学习全局残差，并交替更新/验收各版本。
- SMAClite-AD取代Open-SMAX-AD成为首选第二环境；原版SMAClite不能直接支持双边PSRO和保护目标。

## 阅读入口

1. [研究结论与本轮修正](docs/00_研究结论与阅读路线.md)
2. [问题定义和四阶段输入输出](docs/01_问题定义与创新边界.md)
3. [SMAClite、PSRO及源码调研](docs/02_相关工作与环境审计.md)
4. [论文式方法简版](docs/03_方法简版.md)
5. [详细数学方案与核心代码](docs/04_方法实现规格与伪代码.md)
6. [实验设计](docs/05_实验设计与预期证据.md)
7. [工作量和论文边界](docs/06_工作量_档位与拆稿.md)
8. [HAD S1代码手册](docs/07_代码框架与第一阶段Demo.md)
9. [论文框架](paper/论文框架.md)

## 运行

```powershell
conda activate gpu_py_310
python -m pip install -e .
python -m pytest
python scripts/smoke_stage1.py
```

当前 `gpu_py_310` 已实测使用RTX 3060完成CUDA前向、HAD双边环境步、QMIX loss和backward。该环境目前缺少pytest，且SciPy 1.9.3与NumPy 1.26.4有兼容警告；项目依赖已要求 `scipy>=1.11.4`。

## 真值边界

已实现：双边纯加速度HAD适配、随机单目标、简化变规模QMIX、PSRO元博弈、S2 MLP、S3 maximin、S4 residual/rollback及smoke/tests。

未实现：完整sequence learner、双方训练runner、PSRO BR训练、S2 rollout数据集、S3个体匹配/闭环、S4 outer loop和SMAClite-AD。

因此当前没有训练收敛、PSRO收敛或论文胜率结果。
