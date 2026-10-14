# LEAF v3 论文文件包

本目录是 LEAF v3 的论文源文件与配套分析。主文件为 [main.tex](main.tex)，编译结果为 [main.pdf](main.pdf)。v2 稿件内容保留原状。

## 编译

将 Overleaf 文件包中的内容直接上传，主文件设为 `main.tex`，编译器选 **pdfLaTeX**，参考文献使用 **BibTeX**。本地标准编译顺序为：

```text
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

本机已使用 Tectonic 成功编译为11页，正文引用35篇文献，并完成逐页图文检查。TeX 编译所需的正文、参考文献、矢量图片、IEEEtran 类和算法排版文件均位于本目录，以相对路径引用。第三方排版文件见 [THIRD_PARTY.md](THIRD_PARTY.md)。

## 本版内容

主方法 **LEAF** 对应已有 `regir_nomem` 正式 final 模型：当前自身状态构造读取查询，共享循环更新与中间读取融合保留，下游 GRU 承接时间记忆。所有任务性能统一使用 **R=-D，越高越好**。

实验正文由对比、消融、机制、典型轨迹四部分组成。比较与消融均包含训练验证曲线、目标外推、规模外推、联合外推，以及均值±样本标准差的正式结果表。机制重点是训练过的执行深度、正常读取记录及保护行为。数据来源和全部历史实验细节保留在唯一报告中。

唯一完整报告为 [LEAF主实验结果分析](../../docs/LEAF_主实验结果分析.md)，覆盖正式比较、历史来源及v3机制证据。

## 数据与分析入口

正式HAD回合与七基线验证记录统一位于 [`Open-SCORE/outputs/main/had/episodes.csv`](../../Open-SCORE/outputs/main/had/episodes.csv)，模型身份和归档路径位于 [`summary.json`](../../Open-SCORE/outputs/main/summary.json)，原补充来源登记在其 `sources.main0928_supp` 中。6,300个旧dup回合全部保留独立历史身份，其中2,398个损伤值与正式记录不同；读取正式参考时排除 `historical_outcome_variant`。原720局NoMem辅助记录位于 [`mechanisms/results.json.gz`](../../Open-SCORE/outputs/main/had/regir_nomem/mechanisms/results.json.gz)，v2/v3独有机制与轨迹数据保留在各稿件目录。

- `data/paper1007.json`：正式比较、训练验证、消融及正式行为统计。学习方法先按配置平均场景、按组等权配置，再跨三训练种子报告样本 SD。
- `data/mechanism1008.json.gz`：v3 正常深度与中间读取分析，包含模型、配置、场景、执行臂、原始状态、读取向量、融合权重、事件及汇总。
- `data/trajectories1008.json.gz`：七方法在 5v5、15v15 两个统一场景的已有真实回放。
- `analysis/publication_figures.py`：从已有紧凑统计重绘比较与消融图表。
- `analysis/mechanism_leaf1008.py`：正常深度补评与离线汇总；`--summarize-only` 仅重算已有记录及图。
- `analysis/aux_leaf1008.py --plot-only`：仅从已有轨迹记录重绘案例图。

分析入口使用本仓库的 Open-SCORE 源码、现有正式模型及 SkyHAD；它们不是 TeX 编译依赖。本机 Python 环境为 `torch310`，运行前设置 `PYTHONPATH=E:/Code/SkyHAD`。本版没有新增策略训练。

Overleaf ZIP 只打包论文编译与阅读材料，原始补评数据及分析入口保存在本地 v3 目录，完整结果由唯一 Markdown 报告链接。
