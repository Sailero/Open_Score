# S1 代码、运行与验证记录

## 1. 已落地的闭环

```text
src/HAD_Env/                         原 HAD，目标在最左侧区域逐局随机
src/open_score/envs/had_stage1.py    双方纯加速度、零和终局与势函数塑形
src/open_score/stage1/
├── entity_qmix.py                   共享 DeepSets–GRU utility + 变人数单调 mixer
├── curriculum.py                    分阶段解锁 + TD 学习进展采样
├── replay.py                        变人数/变时长 episode padding
├── runner.py                        双边 rollout、QMIX/随机/规则控制器
├── learner.py                       序列 Double-Q、TD(λ)、target、checkpoint
├── training.py                      混合对手 best-response 训练与独立评估
├── psro.py                          meta-Nash、估计 NashConv、停止规则
└── psro_trainer.py                  双团队 PSRO iteration
scripts/smoke_stage1.py              单步前后向 smoke
scripts/train_stage1_psro.py         2v2 pilot / 1–4 curriculum CLI
```

S2–S4 代码只保留接口，不在本阶段调用。

## 2. 场景与动作

红方防守一个目标，蓝方攻击；目标每局从最左侧
$x\in[-2300,-1900],y\in[-1200,1200],z\in[50,300]$ 均匀采样。相同 seed 重现同一目标与出生状态。红方从左半区、蓝方从右半区随机出生。

每个智能体动作仅为零向量或 $\{-1,0,1\}^3$ 的 26 个归一化方向，HAD 再乘单位自身最大加速度。`guard/rush/intercept` 只存在于规则对手中，不进入学习者动作空间。

防守收益为终局 $\pm1$ 加势函数差分

$$r_D=r_{terminal}+0.1[\gamma\Phi(s')-\Phi(s)],\qquad r_A=-r_D,$$

其中 $\Phi=h_{target}+0.25(\bar h_D-\bar h_A)$。目标被毁为攻击胜，攻击者全灭或存活至时限为防守胜。

## 3. 为什么这样训练人数

正式支持域不是随意抽 16 格，而是

$$\mathcal C_{train}=\{(n_D,n_A):1\le n_A\le n_D\le4\},$$

共 10 格。选择它是因为当前任务设定明确保证防守人数不少于攻击人数；所有 $n_D<n_A$ 的 6 格仍评估，但标为 out-of-support stress。

课程包含两层：

1. 参考 EPC 的 population curriculum，按 1v1、2v1/2v2、3v1/3v2/3v3、4v1–4v4 逐级解锁；
2. 参考 SPMARL/PLR 的 learning-progress 思路，在已解锁规模内用快慢 TD-error EMA 之差加访问次数奖励采样，并混入 20% 均匀分布防遗忘。

这不是逐规模训练十套模型：从第一局开始始终是同一套参数共享网络。与纯线性课程、全程均匀随机和固定规模逐模型训练的比较将决定该课程是否值得保留。

## 4. 网络与训练

- 实体集合编码：masked mean/max/count；
- 共享个体网络：64 维 GRU，27 维 Q 值；
- mixer：状态条件正权重、活跃人数均值缩放，保持 $\partial Q_{tot}/\partial Q_i\ge0$；
- replay：记录规模、seed、目标位置和双方 policy name，并按 batch 内最大时间、人数和实体数补零，终止处切断 bootstrap；
- learner：recurrent Double-Q、TD(λ)=0.6、Huber loss、梯度裁剪和周期 target hard update；
- 红蓝各自训练一套团队网络，回报严格相反。

最多 4v4 不足以证明 Transformer 必需，因此首版保持小网络；REFIL/Attention-QMIX 是强 baseline，而不是默认堆叠模块。

## 5. PSRO 语义与停止

PSRO 的一个玩家是整支协作团队，一个 checkpoint 可运行所有注册人数。每轮：评估种群 payoff matrix、解零和 meta-Nash、双方分别对对手混合策略训练 QMIX 近似 BR、独立评估后入库。

正式停止需连续 3 轮同时满足：

- estimated NashConv ≤ 0.10；
- meta-game value 变化 ≤ 0.03；
- 双方 BR 均达到预注册的最小训练预算。

规则策略平均胜率、archive 平均值和训练 loss 都只是诊断。由于 BR 是近似的，即使 estimated NashConv 很小也不能称为精确 Nash；正式实验还需多 seed、BR 学习曲线和独立 payoff 局。

## 6. 本轮实测

2026-08-28 在本机完成：

- `python -m pytest -q`：10 项通过；
- CPU smoke：HAD 3v4、随机目标、单步 loss/backward 通过；
- `gpu_py_310` CUDA smoke：RTX 3060 Laptop GPU 上通过；
- 2v2 PSRO pipeline：1 轮、双方各 8 个 BR episode、40 步上限，约 7.9 秒，种群从 2×2 扩到 3×3，红/蓝 learner 最后 loss 分别约 0.411/0.696。

短 pilot 的 payoff 每格只有 1 局，且 BR 预算仅 272/297 环境步，因此 `oracle_budget_met=false`；其中任何胜率、meta value 或 estimated NashConv 都没有统计意义。它证明的是数据流、双边梯度、payoff 扩表和 meta-solver 没有接口错误。

## 7. 推荐运行顺序

```powershell
cd D:\Code\Third_Paper\Open-SCORE
python -m pytest -q
python scripts/smoke_stage1.py

conda run -n gpu_py_310 python scripts/train_stage1_psro.py `
  --mode pilot2v2 --device cuda --iterations 1 `
  --br-episodes 8 --batch-episodes 4 --payoff-episodes 1 --max-steps 40

# 扩展前先把每阶段 episode 数、payoff 局数和输出路径写入实验登记表
conda run -n gpu_py_310 python scripts/train_stage1_psro.py `
  --mode curriculum --device cuda --iterations 3 `
  --br-episodes 2000 --payoff-episodes 20 --max-steps 300
```

最后一条只是下一轮工程起点，不是论文最终预算。正式预算应依据每阶段学习曲线、吞吐和至少 3 个开发 seed 决定。

## 8. 进入 S2 的 go/no-go

1. 同一 checkpoint 在全部 10 个训练规模稳定运行；
2. 对规则库和独立学习对手，平衡规模表现显著优于随机；
3. 对 6 个未支持压力格报告性能边界，不把失败隐藏在平均值中；
4. 学习进展课程优于均匀采样或用更少样本达到同等最坏规模表现；
5. PSRO 相对 simultaneous self-play 提升最坏对手胜率，且 BR oracle 展示充分训练；
6. 保存每条 rollout 的防守/攻击策略 ID、checkpoint lineage 和能力版本，保证 S2 能构造策略条件样本；
7. 至少存在可复现的策略对 payoff 差异/克制关系，否则将“策略条件非传递性”降为消融而非贡献；
8. 三个开发 seed 和完整 wall time 可接受。

若变规模执行失败，S2–S4 暂停；若 PSRO 失败但共享策略成功，可把 PSRO 降为 baseline，先以多策略自博弈种群生成 S2 数据。
