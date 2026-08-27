# 上游复现来源与状态

## 主基线：HPN / PyMARL3

- 论文：Hao et al., *Boosting Multiagent Reinforcement Learning via Permutation Invariant and Permutation Equivariant Networks*, ICLR 2023.
- OpenReview：https://openreview.net/forum?id=OxNQXyZK-K8
- 官方代码：https://github.com/tjuHaoXiaotian/pymarl3
- 作用：主复现对象；其 PI 输入层与 PE 输出层定义本项目的基线骨架。

建议在 Linux 单独获取：

```bash
git clone https://github.com/tjuHaoXiaotian/pymarl3.git pymarl3
```

## 强近邻：SPECTra

- 论文：https://arxiv.org/abs/2503.11726
- 公开代码：https://github.com/funny-rl/SPECTra
- 作用：SAQA 和可扩展 mixer 的强近邻/对比；不作为“顶会主基线”。

## 对手泛化依据：OEOM

- AAAI 2025 论文：https://doi.org/10.1609/aaai.v39i22.34488
- 作用：说明固定对手集合会造成 OOD 泛化问题，并为时序 in-context 对手表征提供直接依据。

## 重定位后必须对照的近邻

- TAO（ICLR 2024）：https://proceedings.iclr.cc/paper_files/paper/2024/hash/a1d04870cf83a0f29819d66f1dfdbfcb-Abstract-Conference.html
- OM-TCN（Information Sciences 2022）：https://doi.org/10.1016/j.ins.2022.08.101
- Bayes-OKR（Knowledge-Based Systems 2022）：https://doi.org/10.1016/j.knosys.2022.108404
- HyperJD2TSSM（Engineering Applications of Artificial Intelligence 2026）：https://doi.org/10.1016/j.engappai.2026.114402

它们分别覆盖未见对手上下文适应、动态时序对手预测、回合内策略切换，以及变化人数与动态意图的联合世界模型。当前 OC-HPN 因此只作为通用上下文基线；新主张限定为阵容变化对策略识别的混淆诊断及 CDOC 轻量修复。

## 本机获取记录

2026-08-27 尝试浅克隆 SPECTra 时，终端到 GitHub 的 HTTPS 连接被重置。为避免把网络问题误写成“已复现”，本项目采用公开论文重写架构级最小骨架；失败克隆残留仅保存在 `upstream/failed_clone_empty/`，不参与运行。正式实验前必须在 Linux 训练机重新拉取并记录 commit SHA、依赖锁和地图版本。
