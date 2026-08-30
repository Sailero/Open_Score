# S1 代码、运行与验证记录

## 1. 已落地的闭环

```text
src/HAD_Env/                         原 HAD，目标在最左侧区域逐局随机
src/open_score/envs/had_stage1.py    双方纯加速度、零和终局与势函数塑形
src/open_score/stage1/
├── entity_qmix.py                   共享 DeepSets–GRU utility + 变人数单调 mixer
├── baselines.py                     同一数据契约下的变规模 VDN 与 MAPPO
├── curriculum.py                    分阶段解锁 + TD 学习进展采样
├── replay.py                        变人数/变时长 episode padding
├── runner.py                        双边 rollout、QMIX/随机/规则控制器
├── learner.py                       序列 Double-Q、TD(λ)、target、checkpoint
├── training.py                      混合对手 best-response 训练与独立评估
├── psro.py                          meta-Nash、估计 NashConv、停止规则
└── psro_trainer.py                  双团队 PSRO iteration
scripts/smoke_stage1.py              单步前后向 smoke
scripts/train_stage1_baselines.py    规则/QMIX/VDN/MAPPO 复现、评估与 checkpoint
scripts/train_stage1_psro.py         后续 PSRO 研究入口（本轮不验收）
```

S2 已有独立的真实 HAD 推演—监督训练—校准—评估闭环；S3/S4 本轮冻结。

## 2. 场景与动作

红方防守一个目标，蓝方攻击；目标每局从最左侧
$x\in[-2300,-1900],y\in[-1200,1200],z\in[50,300]$ 均匀采样。相同 seed 重现同一目标与出生状态。红方从左半区、蓝方从右半区随机出生。

每个智能体动作仅为零向量或 $\{-1,0,1\}^3$ 的 26 个归一化方向，HAD 再乘单位自身最大加速度。`guard/rush/intercept` 只存在于规则对手中，不进入学习者动作空间。

防守收益为终局 $\pm1$ 加势函数差分

$$r_D=r_{terminal}+0.1[\gamma\Phi(s')-\Phi(s)],\qquad r_A=-r_D,$$

其中 $\Phi=h_{target}+0.25(\bar h_D-\bar h_A)$。目标被毁为攻击胜，攻击者全灭或存活至时限为防守胜。

## 3. 为什么这样训练人数

HAD 的两类攻击单位在开火后都会自毁，因此本轮支持域严格定义为

$$\mathcal C_{train}=\{(n_D,n_A):1\le n_A<n_D\le4\},$$

即 `2v1、3v1、3v2、4v1、4v2、4v3` 共六格。平局规模和红方更少的规模在当前物理语义下不作为压力测试，避免把自毁机制与数量劣势混为一个不可解释因素。

课程包含两层：

1. 参考 EPC 的 population curriculum，按 `2v1 → 3v1/3v2 → 4v1/4v2/4v3` 逐级解锁；
2. 参考 SPMARL/PLR 的 learning-progress 思路，在已解锁规模内用快慢 TD-error EMA 之差加访问次数奖励采样，并混入 20% 均匀分布防遗忘。

这不是逐规模训练六套模型：从第一局开始始终是同一套参数共享网络。当前“共享实体编码 + 学习进展课程”受 REFIL/SPMARL 启发，但没有实现 REFIL 的随机因子辅助目标或完整 SPMARL；正式报告必须按这个边界命名。

## 4. 网络与训练

- 实体集合编码：masked mean/max/count；
- 共享个体网络：64 维 GRU，27 维 Q 值；
- mixer：状态条件正权重、活跃人数均值缩放，保持 $\partial Q_{tot}/\partial Q_i\ge0$；
- replay：记录规模、seed、目标位置和双方 policy name，并按 batch 内最大时间、人数和实体数补零，终止处切断 bootstrap；
- learner：recurrent Double-Q、TD(λ)=0.6、Huber loss、梯度裁剪和周期 target hard update；
- 红蓝各自训练一套团队网络，回报严格相反。

最多 4v4 不足以证明 Transformer 必需，因此首版保持小网络；REFIL/Attention-QMIX 是强 baseline，而不是默认堆叠模块。

## 5. PSRO 语义与停止（后续接口）

PSRO 的一个玩家是整支协作团队，一个 checkpoint 可运行所有注册人数。每轮：评估种群 payoff matrix、解零和 meta-Nash、双方分别对对手混合策略训练 QMIX 近似 BR、独立评估后入库。

PSRO 代码保留，但用户已明确本轮先不承担这部分计算量，因此不进入当前 S1 验收。未来启用时，正式停止需连续 3 轮同时满足：

- estimated NashConv ≤ 0.10；
- meta-game value 变化 ≤ 0.03；
- 双方 BR 均达到预注册的最小训练预算。

规则策略平均胜率、archive 平均值和训练 loss 都只是诊断。由于 BR 是近似的，即使 estimated NashConv 很小也不能称为精确 Nash；正式实验还需多 seed、BR 学习曲线和独立 payoff 局。

## 6. 本轮实测

截至 2026-08-30 在 Windows 11、i7-14700KF、RTX 5070 Ti、`torch310` 完成：

- 严格六规模的环境、mask、zero-sum shaping、replay、VDN/QMIX/MAPPO 参数更新和 checkpoint resume 自动测试；
- 未参与选模的六规模各 10 局 held-out 上，规则 `Red guard vs Blue rush` 的平均防守 payoff 为 `+0.433`、胜率 `71.7%`；
- 当前代码版三算法各 36 回合 CUDA smoke 均真实更新参数：QMIX validation payoff `-0.600 → -0.267`，VDN `-0.600 → -0.600`，MAPPO 中途到 `-0.467` 但 final 回落到 `-0.600`。MAPPO 的 feed-forward critic 尚无 horizon 输入，只能算训练管线 smoke；
- QMIX 固定 `2v1` 的 300 回合定向运行：8395 环境步、586 次更新、约 181.5 秒；同一组 10 个评估 seed 上由 `-0.4` 改善到最终 `-0.2`，第 200 回合最佳 `0.0`。这是小预算改善信号，尚未胜过规则 guard 的 `0.0`。
- 从固定 `2v1` validation-best warm-resume 到六规模后，validation-best 为 `-0.367`、final 为 `-0.633`；一次性 held-out 上 validation-selected QMIX 为 `-0.433 / 28.3%`，明显弱于规则策略。

上述训练只有一个 seed，validation 还被反复用于选模；只有最后的 held-out 集合是一次性测试。结果能证明训练链和局部改善，不能称为“优秀策略”“稳定收敛”或论文结果。完整解释见 `docs/reports/S1_HAD_复现与验证报告.md`；机器证据在 `docs/evidence/stage1_had_*.json/csv`，checkpoint 在被 `.gitignore` 排除的 `outputs/` 下并由证据 JSON 保存 SHA-256。

## 7. 推荐运行顺序

```powershell
Set-Location E:\Code\Open_Score\Open-SCORE
D:\Software\Anaconda\envs\torch310\python.exe -m pytest -q
D:\Software\Anaconda\envs\torch310\python.exe scripts\smoke_stage1.py

# 三算法、严格六规模的小预算复现
D:\Software\Anaconda\envs\torch310\python.exe scripts\train_stage1_baselines.py `
  --algorithms qmix vdn mappo --device cuda --episodes 72

# 定向固定规模；正式使用前应预注册种子和评估局数
D:\Software\Anaconda\envs\torch310\python.exe scripts\train_stage1_baselines.py `
  --algorithms qmix --device cuda --fixed-scale 2 1 `
  --episodes 300 --updates-per-episode 2 --eval-episodes-per-scale 10
```

最后一条只是下一轮工程起点，不是论文最终预算。正式预算应依据每阶段学习曲线、吞吐和至少 3 个开发 seed 决定。

## 8. 进入 S2 的 go/no-go

1. 同一 checkpoint 在全部六个严格 `Red > Blue` 规模稳定运行；
2. 对规则库和独立学习对手，固定 `2v1` 与动态六规模显著优于随机；
3. 预注册留出 1–2 个支持域规模，不能用训练过的六格结果冒充零样本泛化；
4. 学习进展课程优于均匀采样或用更少样本达到同等最坏规模表现；
5. 规则策略和至少一套智能策略 checkpoint 可重复加载；PSRO 不作为本轮门槛；
6. 保存每条 rollout 的防守/攻击策略 ID、checkpoint lineage 和能力版本，保证 S2 能构造策略条件样本；
7. 至少存在可复现的策略对 payoff 差异/克制关系，否则将“策略条件非传递性”降为消融而非贡献；
8. 三个开发 seed 和完整 wall time 可接受。

若变规模执行失败，S2 仍可用规则 rollout 验证工程，但不能宣称已获得第一阶段优秀策略；S3/S4 继续冻结。PSRO 后续只能在共享策略已经稳定之后作为对手多样性 baseline。
