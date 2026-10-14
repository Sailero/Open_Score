# LEAF v4 论文文件包

本版以用户提供的论文想法和 `docs` 已整理结果为依据，围绕“威胁情境如何支持零样本规模泛化”组织论文。入口为 [main.tex](main.tex)，编译结果为 [main.pdf](main.pdf)。v1–v3 原稿保留。本目录是论文包说明，不另建实验报告；正式结果仍见[主实验结果分析](../../docs/LEAF_主实验结果分析.md)与[补充实验结果分析](../../docs/LEAF_补充实验结果分析.md)。

## 编译与文件

本目录包含编译所需的 TeX、参考文献、IEEEtran 排版文件和全部论文图片。Overleaf 使用 `main.tex`、pdfLaTeX 与 BibTeX；也可在本地运行：

```text
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

当前 PDF 为 **12 页（包括参考文献）**，包含 5 张图、5 张表。已使用现有 Tectonic 完成无警告编译，并用现有 PyMuPDF 逐页渲染检查；无溢出版框、未解析引用或缺失文献。不涉及策略训练、环境回放或新增科研评估。第三方排版文件见 [THIRD_PARTY.md](THIRD_PARTY.md)。

## 叙述与数据口径

实验正文按四部分组织：对比实验、历史设计比较、机制实验、补充分析。主指标统一为累计目标损伤 **D↓**，奖励为 `R=-D`。对比图与主表统一包含 LEAF、REFIL、ALMA、TransfQMix、DCG、SPECTra、Rule、QMIX；QMIX 指现有实体注意力适配。设计比较将方法分为情境构建 C、当前状态查询 Q、阶段融合 F、宿主与时序整合 H 四个模块，分别对应 Direct read、Memory query、Last read、No host。保存配置确认这四个历史变体均保留截梯度历史查询，LEAF 为当前状态查询；Direct read、Last read、No host 与 LEAF 比较时还混有查询差异，不能宣称全部是单因素消融。单轮、固定四轮、不共享参数亦保留历史查询，与 REFIL-matched 一并放入第四小节。

机制部分依次支撑三项主张：H1 已训练策略使用了构建后的当前情境；H2 中间读取随接收者与时间变化，并与下游记忆协同；H3 学到的精炼在所评估的不同深度下仍可复用。保护行为图将这些观察连接到拦截、碰撞与目标损伤。容量匹配、训练深度性质与独立场景确认归入补充分析。

- 正式比较与学习验证：以[主报告](../../docs/LEAF_主实验结果分析.md)的 `paper_current` 视图为准，聚合位于 `Open-SCORE/outputs/main/had/aggregates.json`。SPECTra 与 TransfQMix 使用新 `main1013_base` 的三种子 final；其余方法保持 `paper1007` 来源。每模型 1M 物理步，每配置 300 个共同场景；误差为三种子样本 SD，Rule/Random 单独注明场景 SD。旧 10 维基线保留在历史 `paper1007` 视图与同一主报告内，不与新模型混合平均。
- 新基线协议对齐 19 维实体输入（含上一动作，actor/mixer 一致）、任务与采样控制（8 个并行采集环境、每次采集 8 次学习更新）；保留基线自身 Adam 1e-3、TD(λ=0.6) 和半 MSE，不宣称所有优化超参数统一，也不把成绩变化仅归因于输入。新基线终评未记录碰撞、拦截与损伤事件比例；这些位置标为未记录，不能套用旧基线指标。
- 消融及已记录的保护行为统计仍使用 `paper1007`。LEAF、REFIL 的主表、机制、深度与各自确认池口径保持不变；独立确认采用新 SPECTra/TransfQMix 的确认结果。
- 冻结深度：同三份 LEAF final、24 配置、R1–R6，使用已整理深度补评。R5/R6 超出训练深度。该池的 R4 与主表分别保留，不用小数差异覆盖原主表；不从缺失的逐场记录构造额外区间。
- 正常读出：沿用 v3 的正常执行机制结果。辅助冻结干预：沿用补充报告第 7.6 节，与各自本地标准路径配对；这类干预衡量已训练策略的依赖，不等于从头重训的消融。
- 所有图表仅重排、重绘已有结果，没有新增种子、训练、评估或诊断。脚本读取原有数据位置，不复制实验数据目录。

理论分析给出掩码实体重排的一致性与有限深度读出截短的动作稳定充分条件。时间记忆隔离只针对固定当前输入的读取路径；不宣称闭环完全不受历史影响。未假设残差注意力是压缩映射，未把 1–6 轮的结果推广为任意深度收敛。

## 图表入口

- `figures/fig_motivation.png`：威胁情境动机；`figures/framework.tex`：可编辑架构图。
- `figures/fig_comparison.pdf/png`：八种方法的 ID 学习验证、K/N 外推与联合外推；比较曲线读取 `paper_current`（兼容旧归档时回退 `paper1007`）。
- `figures/fig_ablation.pdf/png`：四模块分解及历史设计相对 LEAF 的损伤变化；图注注明历史查询差异。差值在 `paper1007` 原始精度上计算后舍入，绝对均值与 SD 保留在表中。
- `figures/fig_mechanism.pdf/png`：按 H1/H2/H3 排列的局部干预、正常读出、冻结深度及正式保护行为；空心点是独立重训单轮参照。
- `sections/publication_tables.tex`：正式对比、历史设计比较、冻结干预、补充控制四张结果表；环境参数表位于 `sections/problem.tex`。

从仓库根目录执行 `python -X utf8 Papers/v4/analysis/publication_figures.py` 可重绘实验图。需要现有 Matplotlib/NumPy；只读取已保存报告与聚合统计，不加载模型、不启动评估。

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
