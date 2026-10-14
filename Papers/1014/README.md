# LEAF 论文文件包

本目录保留当前一份LEAF论文。入口为[main.tex](main.tex)，编译结果为[main.pdf](main.pdf)。正式实验数据、统计与证据范围见唯一[主实验报告](../docs/LEAF_主实验结果分析.md)；本文件只说明论文文件与排版。

## 编译与共享图

论文使用pdfLaTeX/BibTeX或现有Tectonic编译。TeX、参考文献和第三方IEEEtran文件保存在本目录；[THIRD_PARTY.md](THIRD_PARTY.md)记录第三方来源。

```text
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

`figures/framework.tex`是可编辑TikZ结构图，`figures/fig_motivation.png`是概念示意。实验图只有一份，共同保存在[主结果figures](../Open-SCORE/outputs/main/had/figures)，论文经`../Open-SCORE/outputs/main/had/figures/<名称>.pdf`引用矢量图，报告使用同源PNG。论文引用`fig_comparison`、`fig_ablation`、`fig_mechanism`与`fig_protection`；轨迹、时延、浅层对照与探针的共享图由主报告解释。导出论文包时携带这些已引用的图，不在仓库复制第二套实验图片。

已有实验图片统一由仓库根目录的`python -X utf8 Open-SCORE/scripts/plot.py`刷新。该入口读取保存统计，不加载模型，不新增训练、评估或诊断，Papers不保留另一套分析脚本。

## 叙述与数据口径

实验正文按对比、设计比较、机制和补充控制组织；主指标为累计目标损伤D↓，奖励R=-D。当前比较包含LEAF、REFIL、ALMA、TransfQMix、DCG、SPECTra、Rule、QMIX；QMIX指现有实体注意力适配。每模型1M物理步、三个训练种子；正式评估每配置300个共同场景。学习方法误差为训练种子样本SD，Rule/Random为场景SD。

- 正式比较与学习验证读取`aggregates.json.comparison`；SPECTra和TransfQMix采用已完成的19维输入重训final，输入含上一动作并在actor/mixer一致，采集八环境、每批八次更新。其自身Adam/1e-3、TD(λ=0.6)、半MSE和结构保留，不能称全部训练超参数统一或充分调参。原10维结果单独归档；新终评没有记录碰撞、拦截或受损事件比例，不由旧模型补填。
- 设计比较读取`design_comparison`。四模块C/Q/F/H对应Direct read、Memory query、Last read和No host；这些保存变体均保留截梯度历史query，LEAF使用当前状态query。除Memory query的入口对照外，其余与LEAF比较还混入query差异，不能宣称全部为单因素消融。单轮、固定四轮、不共享参数及REFIL-matched放入补充控制。
- 正常读取、冻结干预与正式保护统计分别取`mechanism.normal/frozen/formal`。原始机制统一保存于[results.json.gz](../Open-SCORE/outputs/main/had/regir_nomem/mechanisms/results.json.gz)，各池保持自身runtime、局部标准路径与配对/复用身份。冻结干预及same-state/Q-margin边界见主报告第4.11节；它们说明已训练策略的依赖，不等于从头重训组件的独立收益。
- 冻结深度使用同三份LEAF final、24配置、R1–R6。R5/R6超出训练范围，结论限于所测深度；深度池R4不替代主比较统计。缺失的新增108,000场逐场原始日志不用于构造额外场景区间。
- 展示轨迹、20回合部署时延、固定人数100次决策、Fixed depth三配置浅层对照及受控覆盖线性探针保留各自预定范围和局限，不用局部示例替代正式排名。

理论分析给出掩码实体重排的一致性和有限深度截短的动作稳定充分条件。时间记忆隔离只针对固定当前输入的读取路径，不宣称闭环完全不受历史影响；不假设残差attention是压缩映射，不把1–6轮结果推广为任意深度收敛。

## 辅助示意图

[figures/fig_motivation.png](figures/fig_motivation.png) 使用内置 ImageGen 生成，表示相同接近状态下目标覆盖不同的概念情境。它不是实验截图或测量结果；正文图注及致谢中均已区分。原生 TikZ 架构图与实验矢量图分别承担精确结构和数据展示。

生成提示词如下（内置工具模式，未使用 API/CLI）：

```text
Use case: scientific-educational.
Asset type: conceptual motivation illustration for a peer-reviewed robotics/multi-agent learning paper, not experimental evidence.
Create a refined, minimal, wide 2.4:1 landscape scientific editorial illustration on a pure white background. Two equal side-by-side panels, clean flat/isometric aircraft and protected-asset icons, restrained dark navy outlines, teal defender aircraft, coral-orange attacker aircraft, ochre protected building. Crisp readable at half-page print size, plenty of white space.
Primary message: the same attacker approach can mean a different threat depending on teammate coverage of its target.
Left panel title exactly "Covered target". One orange attacker approaches one ochre protected asset. Two teal defending aircraft are already positioned near this asset; use very subtle dotted teal coverage arcs. A separate teal defender at the lower edge is the acting receiver, labeled exactly "Acting defender".
Right panel title exactly "Uncovered target". Repeat the SAME orange attacker and ochre asset at the SAME relative positions with the SAME approach arrow and scale. The two defenders near the asset are absent. Keep the acting defender in the same lower-edge position. A clean orange outline emphasizes the undefended asset, without flames or damage.
Use a single short orange approach arrow from each attacker toward its asset. Near these arrows put the exact same label "Same approach". Under left panel put exactly "Threat depends on existing coverage"; under right put exactly "Threat changes when coverage changes".
At bottom center a simple small legend with teal aircraft "Defender", orange aircraft "Attacker", ochre building "Protected asset".
Illustration should convey scene context and scientific comparison, not tactical instructions. No weapons, no explosions, no landscape clutter, no gradients, no photorealism, no watermark, no numerical data, no charts, no performance claims. Ensure only the specified labels, sharp coherent typography, panels visually matched.
```
