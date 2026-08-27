# CDOC-HPN：对抗减员下的阵容—策略解耦对手适应

## 一句话定位

**我们提出规模—策略解耦的在线对手建模，让团队智能体从减员/增援轨迹中分清“敌人变少了”和“敌人换打法了”，从而在无需微调时减少误判并更快适应未见策略。**

工作题目：*Do Opponent Models Mistake Attrition for Strategy Change? Composition-Disentangled Online Adaptation in Team-Competitive MARL*。

推荐方法名：**Composition-Disentangled Opponent Context with HPN（CDOC-HPN）**。

## 为什么重新定位

上一版 OC-HPN 的“集合池化 + GRU + FiLM + 辅助预测”容易复现，但单独看不够新：既有工作已经研究了时序对手建模、未见对手的 in-context 适应和回合内策略切换；2026 年近邻工作还联合处理了变化的对手数量与动态意图。仍然存在且更有实际意义的问题是：团队对抗中的战损、增援和单位类型变化会改变观测集合，即使对手策略并未变化；通用对手上下文可能把这种阵容变化误判成策略切换。

因此：

- 现有 `OpponentConditionedHPN` 保留为**通用时序上下文基线**，不再作为最终创新；
- 新主线显式分离即时阵容上下文 `c_t` 与规模归一化行为上下文 `z_t`；
- 用“阵容是否变化 × 策略是否变化”的 2×2 受控干预作为主协议；
- 主要指标改为策略切换检测 AUROC、阵容变化下误报率、检测延迟、切换后遗憾和胜率，而不只看联合 OOD 平均胜率。

## 目录

```text
docs/
  00_项目定位与删减决策.md
  01_完整调研报告.md                上一版调研，已加重定位说明
  02_核心论文与复现审计.md
  03_三版技术路线.md
  04_实验规划与预期结果.md
  05_投稿与八周执行计划.md
  06_研究价值与创新性复核.md        本轮否证式调研与最终定位
paper/
  00_title_abstract.md
  01_introduction.md
  02_related_work.md
  03_preliminaries.md
  04_method_candidates.md
  05_experiments_plan.md
  references.bib
src/q2marl/
  models.py                         HPN 与通用 OC-HPN 基线骨架
src/swarmbattle/                    可控两队环境与脚本对手
tests/test_models.py                置换、padding、变规模与梯度测试
```

## 当前代码边界

当前代码已经实现可运行的 HPN 风格 PI/PE 骨架和通用 OC-HPN，并通过结构单元测试；它们是主实验所需的基线。**CDOC-HPN 的成对干预损失、条件行为编码器和策略变化检测头尚未实现，不能把现有 OC-HPN 的可运行性写成新方法结果。** 先用当前基线验证“阵容变化导致策略误报”的问题确实存在，再实现最小 CDOC，能避免为一个不存在的故障过度设计。

## 最先做的 P0

在 SwarmBattle 生成四类等长事件：无变化、仅阵容变化、仅策略变化、二者同时变化。冻结同一批 HPN/OC-HPN checkpoint，测量对手上下文变化分数：

1. 若 OC-HPN 在“仅阵容变化”上的策略切换误报率明显高于无变化，并且误报与战损幅度正相关，则问题成立；
2. CDOC-HPN 必须相对 OC-HPN 降低至少 30% 的阵容误报，同时不牺牲策略变化 AUROC 和训练域胜率；
3. 若通用 OC-HPN 本身没有明显混淆，停止该方法主张，不通过追加模块制造结果。

## 快速验证现有基线

```powershell
C:\Users\hp\.conda\envs\gpu_py_310\python.exe scripts\smoke_test.py
C:\Users\hp\.conda\envs\gpu_py_310\python.exe -m unittest discover -s tests -v
```

