# HAD 第一阶段代码与实验手册

## 1. 已实现代码

```text
configs/stage1_had_demo.yaml
scripts/smoke_stage1.py
src/open_score/
├── contracts.py
├── nn.py                         # 简单 DeepSets
├── envs/had_stage1.py           # 双边、单随机目标、27纯加速度
├── stage1/entity_qmix.py         # 小型变人数 QMIX
├── stage1/losses.py              # 单步 Double-Q smoke
├── stage1/psro.py                # meta-Nash/NashConv骨架
├── stage2/outcome_model.py       # outcome-time MLP/ensemble
├── stage3/commander.py           # 分组收益和maximin LP
└── stage4/coadaptation.py        # 最终胜负残差/guard/rollback
```

当前代码能验证接口和梯度，但不是完整训练器。

## 2. HAD S1 的准确语义

### 场景

- 红方防守一个目标，蓝方攻击该目标；
- 目标每局在 `target_region` 内重新采样；
- 红蓝人数各为1–4；
- 防守成功：蓝方全灭或目标存活至时限；
- 攻击成功：目标被摧毁。

### 动作

动作ID只映射到

```text
(0,0,0) 和 {-1,0,1}^3 中26个非零方向的单位向量
```

HAD把向量乘以智能体自身的 `aMax`。动作中没有“朝最近敌人”“朝目标”“拦截”等规则语义。目标位置、敌我状态只通过观测进入QMIX。

### 奖励

红方奖励为终局 $\pm1$ 加小权重势函数差分，蓝方严格取负。势函数只包含目标健康和归一化敌我平均健康，不使用到目标距离等可能直接指定战术的 shaping。

## 3. 网络

首版保持简单：

- 实体：两层MLP后 masked mean/max/count；
- 个体记忆：64维GRU；
- 动作头：27维；
- mixer：32维状态编码、16维mixing；
- 防守和攻击各自一套参数，不共享权重，但结构相同。

如果1–4规模下DeepSets已经足够，就不加入attention。REFIL/Attention-QMIX作为强baseline，而不是默认结构。

## 4. `gpu_py_310` 使用

当前实测CUDA smoke已经在RTX 3060上运行。建议进入项目目录后：

```powershell
conda activate gpu_py_310
python -m pip install -e .
python -m pip install pytest
python -m pytest
python scripts/smoke_stage1.py
```

项目现在要求 `scipy>=1.11.4`，用于兼容当前NumPy 1.26.4。现有环境仍是SciPy 1.9.3，因此在未升级前会出现版本警告。Conda的 `anaconda-cloud-auth` 插件警告来自base环境，不是CUDA/PyTorch问题。

## 5. 下一步必须补的文件

### `stage1/replay.py`

存储完整双方episode：当前/下一状态、mask、双方动作、双方零和奖励、terminated、truncated、规模、目标位置和opponent policy ID。按batch最大人数/实体数padding。

### `stage1/learner.py`

实现GRU burn-in、sequence Double-Q、TD($\lambda$)、target hard update、梯度裁剪和checkpoint。防守/攻击learner分开更新。

### `envs/had_factory.py`

按插值或全覆盖协议采样 $(n_D,n_A)$，缓存16类HAD实例并分离train/eval seeds。

### `runners/competitive_runner.py`

同时加载红蓝policy，维护双方hidden state，记录policy ID。Windows先用4–8个spawn worker测吞吐，再决定是否增加。

### `stage1/psro_runner.py`

负责payoff cell评估、meta-Nash、对混合策略采样对手、训练双方近似BR、更新种群和估计NashConv。

### `scripts/train_stage1.py` / `evaluate_stage1.py`

输出JSONL/TensorBoard、16格胜率矩阵、逐策略payoff、NashConv、环境步数和wall time。

## 6. 实现顺序

1. 1v1固定目标过拟合，验证加速度能学；
2. 2v2随机目标，验证协作和奖励；
3. 防守/攻击分别对规则策略warm-start；
4. 接入1–4多规模采样和固定容量QMIX baseline；
5. 跑插值协议；
6. 同时自博弈作为PSRO前对照；
7. 跑3轮PSRO pilot；
8. 只有NashConv/最坏胜率改善且wall time可接受，才扩到8轮。

## 7. PSRO核心循环

```text
Pi_D, Pi_A = {rule policies, warm-start QMIX}
for k in 1..8:
    evaluate missing payoff cells over registered scales and seeds
    sigma_D, sigma_A = zero_sum_meta_nash(M)
    br_D = train_QMIX_BR(opponent ~ sigma_A, scales ~ P_train)
    br_A = train_QMIX_BR(opponent ~ sigma_D, scales ~ P_train)
    evaluate BRs on independent seeds
    add useful BRs to Pi_D/Pi_A
    stop if estimated NashConv <= 0.10 for 3 iterations
```

PSRO交付的是元策略混合。在线子博弈开始时，为整支队伍采样一个policy checkpoint并固定到子博弈结束；S2对冻结的元策略版本做边缘化评估。若最终部署必须只有一条策略，可在S1稳定后做population distillation，但它不是首版必需项。

## 8. 验收

- 双边纯加速度语义和零和奖励测试通过；
- 同一网络运行所有1–4规模；
- 目标位置随机且seed可重现；
- 训练/评估payoff不复用随机种子；
- 平衡规模能稳定学到非随机策略；
- 插值规模平均下降不超过0.15；
- PSRO pilot相对naive self-play降低estimated NashConv或提升最坏对手胜率；
- 记录RTX 3060总训练wall time。

这些条件通过后，S1才提供可信的“局部能力版本”给S2。
