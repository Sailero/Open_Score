# LEAF TASE 稿件包（1007 稿，2026-10-08 修订）

本目录是 LEAF 1007 稿的 Overleaf 编译包，2026-10-08 修订补齐主比较的五基线原始学习曲线、核心五方法的学习与外推图、四组绝对R及C正文表，并补充上下文绕过、固定融合及动作间隔机制分析和七方法5v5/15v15轨迹。论文入口为 [main.tex](main.tex)，编译输出为 [main.pdf](main.pdf)。稿件包含正式比较与已完成的冻结辅助评估；辅助回放与历史正式记录的一致范围见末节。

## 编译

将本目录的内容上传为一个 Overleaf 项目，设置主文件为 `main.tex`，编译器为 **pdfLaTeX**，参考文献使用 **BibTeX**。保持 `sections/`、`figures/`、`data/` 的相对位置。

本地在本目录执行：

```text
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

本轮完整修订已使用 Tectonic 0.17 成功编译为 **13页**，并逐页检查图表与排版；没有Overfull、未定义引用或缺失字符，保留少量Underfull提示。尚未在Overleaf云端执行编译。当前稿件超过12页，最终投稿篇幅与超页安排参见[TASE作者指南](https://www.ieee-ras.org/publications/t-ase/information-for-authors/)。

正文、参考文献和图均通过相对路径引用。`references.bib`、`IEEEtran.cls`、`IEEEtran.bst`、算法排版文件及引用的论文图随包提供；其余宏包由标准 TeX 环境提供。编译不依赖仓库外的实验目录、Python 环境或模型权重。第三方排版文件说明见 [THIRD_PARTY.md](THIRD_PARTY.md)。

## 方法与实验身份

- 论文主方法 **LEAF** 对应保存的 `regir_nomem`（NoMem）配置，正式比较使用各训练种子的 `final` 模型。
- `global_query_no_memory=true`：关系读出查询为 `[H0_self; 0]`。下游 GRU 保留跨物理决策的时间记忆。
- `skip_count_inject=true`：关闭显式 typed-count 注入。共享关系更新与状态依赖的跨轮融合保留；训练采样深度 1–4，正式评估深度为 4。
- 原运行标识为 `main0928_looped_nocount_v1`。原战役以 `regir_sg`（Looped）登记主方法；本稿选用已有 NoMem 结果并命名为 LEAF，不改写原实验身份。
- 论文锁定已有结果的 saved config、`checkpoint_id` 和记录的 `t_env`。当前 `main` 与原 `remote_train` 的完整调度登记不同；重开原训练战役需要相应调度与 registry，不能用“当前最新版”替代已有实验协议。

## 数据与统计口径

正式 HAD 数据位于仓库的 `Open-SCORE/outputs/main/had/`。唯一完整正式报告为 [LEAF_主实验结果分析.md](../../docs/LEAF_主实验结果分析.md)，模型身份、归档路径及历史来源统一登记在 [summary.json](../../Open-SCORE/outputs/main/summary.json)。这些仓库外链用于追溯，不是 Overleaf 编译依赖。

包内紧凑统计入口为 [`data/paper1007.json`](data/paper1007.json)，由正式记录派生，用于核对本稿表格与科研图。正式评估覆盖 24 个配置，每个训练模型、每个配置使用共同的 300 个场景种子 9000–9299。主要性能指标为原始奖励 **`R=-D`，越高越好**，不包含训练 shaping 项。

学习方法先在配置内平均场景，再按组等权平均配置；表中 `mean±SD` 的 SD 是 **3 个训练种子的样本标准差**（分母 2）。Rule 没有训练种子：按共同场景种子先平均组内配置，再计算 **300 个场景的样本标准差**（分母 299）。两种 SD 表示不同的变异来源，均不是置信区间。

学习图使用原始验证记录，均为50个检查点、3个训练种子，每点四个ID配置各25局，四配置分别使用固定9500–9599池中的连续25个场景seed。按验证序号对齐，以实际环境步数均值为横轴，报告均值±训练种子样本SD，不平滑。正式终评使用9000–9299，因此学习末点与正式ID表值可不同。主比较含LEAF与五个学习基线（REFIL、QMIX-Atten、DCG、SPECTra、ALMA），Rule仅作final-ID参考。消融图含LEAF、SG、NoREFIL、R0、1-round（R1），其余三面板分别展示K、N及全部八个Joint配置的绝对奖励；核心五方法的四组绝对R、C表在正文，十变体完整R表保留于附录。三组配对解释不等同于所有方法相对LEAF的单组件消融。

2026-10-08补充的 [HAD基线验证回合](../../Open-SCORE/outputs/main/had/episodes.csv) 已归入主档案，其原 `main0928_supp` inventory保存在 [summary.json](../../Open-SCORE/outputs/main/summary.json) 的 `sources.main0928_supp` 中：历史来源为main0923，CSV保留 `source_version=main0921`。7个已有基线×3训练种子×50点×4配置×25局，共105,000个验证回合已归并；图和派生统计读取canonical正式记录，不从PDF恢复数值，也没有重新训练基线。21个末点的均值与实际 `t_env` 匹配原记录。补充包的三份QMIX-Atten配置因环境标记不属于HAD而排除，canonical正确配置保留。

补充包6,300个旧dup回合中2,398个损伤值与main0928不同（最大绝对损伤差19.68796079），全部6,300个回合均存在至少一项科学诊断差异。主档案按独立历史身份保留这些记录，并标记 `archive_record_status=historical_outcome_variant`；正式dup结论仍读取原main0928记录。与canonical逐项一致的151,200个基线final和14,400个锚点回合已去重。episodes709,500、learning157,065、progress525,319是归并前main0928来源文件的历史计数，当前归档计数与来源覆盖以 `summary.json` 为准。旧队列、调度及运行日志已清理，必要运行元数据保存在归档汇总中。

## 机制证据与轨迹

原 `regir_sg` 的深度扫描、读出干预属于历史查询分支，不能标作 LEAF 机制结果。学习设计对照与部署期冻结干预分别解释；NoMem 保留 GRU，不能称为移除了全部时间记忆。

LEAF机制数据与空间轨迹使用当前本机辅助回放，按保存的模型配置和身份加载。原标准回放覆盖LEAF与REFIL的三个训练种子，在 `(10,10,2)`、`(50,50,2)`、`(30,30,12)` 配置各使用场景9000–9009。180局中167局的11个核对字段与历史正式记录完全一致；13局差异均来自LEAF seed2，其中5局奖励不同。干预结果与配对的当前标准回放比较，不替换正式CSV的final统计，不推断历史差异根因。新收集的90局LEAF standard全部通过旧local standard的11字段核对；这说明本轮与旧本地控制一致，不意味着13个历史差异已经消失。

原辅助评估完成720局：LEAF的standard、L1、L2、read4、uniform、residual_zero、gru_reset七臂各90局，另有REFIL standard90局。90局LEAF标准轨迹同时提供同状态前向诊断，包含于720局中。2026-10-08在相同三配置、三模型种子、9000–9009场景池新增270局：standard重采集、H0直接读出（loop_bypass）、固定经验融合（fixed_fusion）各90局，没有新增训练。两批机制共990次闭环执行，其中90次是控制重采集，不是新增独立场景。另有七方法×两规模14次轨迹回放，本轮284次、累计1004次已执行；该执行数不表示1004个独立场景样本。

原720局记录入口为 [`results.json.gz`](../../Open-SCORE/outputs/main/had/regir_nomem/mechanisms/results.json.gz)，新增机制记录与每轮读出表征保留为包内 [`data/mechanism1008.json.gz`](data/mechanism1008.json.gz)，轨迹为 [`data/trajectories1008.json.gz`](data/trajectories1008.json.gz)，紧凑统计入口为 [`data/paper1007.json`](data/paper1007.json)。闭环奖励差先在同一模型和场景内相对当前standard配对，平均各种子的十场景后，再报告三训练种子的均值与样本SD。

上下文、阶段读出和时间记忆分别解释。H0直接读出保留关系投影和GRU，在三个配置的配对奖励变化为−7.556±1.365、−18.962±5.828、−32.968±8.447，九个种子—配置均值均降低。固定融合权重按每个模型在旧标准轨迹中的各配置活跃决策平均alpha，再等权平均三个配置；没有按奖励拟合，也不是held-out检验。其配对奖励变化为+0.006±0.109、+0.092±0.121、−0.612±0.471：两个K2配置方向混合，仅30v30 K12三种子均降低，因此不声称动态融合在所有配置更优。

三个配置的既有事实诊断中，前两轮平均融合权重为98.59–98.91%；两轮执行改变1.13–1.66%的存活防御者贪心动作，GRU重置改变58.37–62.90%。既有48,705个活跃决策中有753个L2动作改变，最低10%动作间隔包含这些改变的比例按三种子平均为78.630±9.011%、80.876±17.440%、78.305±13.787%。充分稳定条件 `g > 2 ||deltaQ||_inf` 覆盖的38,620个决策没有反例，但它只保证同状态贪心动作稳定，不保证闭环奖励。read4、uniform及完整残差置零降低全部九个种子—配置均值，GRU重置降低八个；L1/L2奖励方向仍混合。这些冻结干预衡量已有策略的部署期依赖，与重训练设计对照分开解释。

三张图随包提供：[冻结计算干预图](figures/fig_leaf_final1007_mechanism.pdf)、[fusion与边界决策图](figures/fig_leaf_fusion_sensitivity.pdf)和[七方法真实空间轨迹图](figures/fig_leaf_final1007_trajectory.pdf)。轨迹固定模型seed0、场景seed9000，在5v5 K2与15v15 K2展示LEAF、五个学习基线与Rule；相同规模的七方法初始状态一致，空间快照统一为实际物理步30，完整累积奖励曲线在真实终止时停止。14条当前本地回放中11条通过历史11字段核对；5v5的DCG及两个规模的ALMA存在差异。LEAF与REFIL两规模均匹配历史；5v5二者R均为−1，15v15为LEAF R=0、REFIL R=−5。固定例子不替代正式300场景统计，也不保证主方法每例最优。

`analysis/`中的两个脚本供本仓库复现新增分析；环境回放需要现有Open-SCORE源码、正式权重及HAD模块，不属于TeX编译依赖。torch310运行前设置`PYTHONPATH=E:/Code/SkyHAD`；具体命令及离线重绘选项见脚本。Overleaf包包含TeX依赖、最终PDF、必要科研图、三份数据文件和这两个复现入口，排除编译日志与页面检查缓存。

运行版本按实际元数据保留：旧720局为`had-workbench-2.1.0`，新270局为`skyhad-3.0.0`，14局轨迹为`skyhad-3.0.0rc2`；旧、新协议标签分别为`rebuild-calibrated-v3-r7-target-initialization`与`skyhad-v3-rigid-body-task-start-combat`。标准重复的初始状态及11字段一致，不构成完整运行环境等价证明。论文附录披露该差异，冻结干预继续使用本地配对控制，正式统计不变。
