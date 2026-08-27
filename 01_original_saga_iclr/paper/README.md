# 论文草稿目录（ICLR 2027 投稿）

**标题（暂定）**：*Dual Zero-Shot Generalization in Adversarial Swarm Games via Opponent-Anchored Dynamic Grouping*

每章一个独立 md 文件，便于并行修改与版本追踪；最终拼接转 LaTeX（ICLR 模板）。

## 章节状态一览

| 文件 | 章节 | 状态 | 说明 |
|---|---|---|---|
| `00_title_abstract.md` | 标题+摘要 | ✅ 完整初稿 | 摘要含 [数字占位]，实验后回填 |
| `01_introduction.md` | §1 Introduction | ✅ 完整初稿 | **贡献与创新点已定稿**（3 条贡献 + 创新点自检批注） |
| `02_related_work.md` | §2 Related Work | ✅ 完整初稿 | 五条主线 + Positioning 段 |
| `03_problem_formulation.md` | §3 问题形式化 | ✅ 完整初稿 | 动态种群博弈定义、双重泛化矩阵 **G**、对手锚定假设（可证伪表述） |
| `04_method.md` | §4 Method | ✅ 完整初稿 | 三模块 + SxS 课程 + 伪代码 + 超参表；含原型教训批注 |
| `05_experiments.md` | §5 Experiments | 📝 大纲（按用户要求） | 以"图表合同"形式写死每个实验的假设/形式/成功判据 |
| `06_conclusion_limitations.md` | §6 结论/局限/部署/伦理 | ✅ 完整初稿 | 新增 Deployment Readiness 段（延迟实测支撑） |
| `07_appendix_outline.md` | 附录 A–H | ✅ 编排完成 | 每节标注素材来源与就绪度；新增 H 部署契约节 |
| `08_theory.md` | 理论分析（附录 F 全文） | ✅ v2 审稿修订版 | 命题 1-5 含证明；v2 修复：命题5论证方向、命题4推导一致性、命题3双regime前提、命题2复杂度口径（修订记录 `../docs/08`） |
| `references.md` | 参考文献 | ✅ 引用键映射完整 | 待转 BibTeX，2 处补引已标注 |

## 写作约定

1. **术语中性化**（审视报告 §2 的要求）：全文用 adversarial swarm games / engagement / attrition，环境叫 **SwarmArena**（代码中的 SwarmBattle 在开源前统一改名），不出现军事平台措辞。
2. **占位符**：`[XX]`/`[64+]` 等方括号数字为实验回填点；`<!-- -->` 注释与"中文批注"小节为写作过程记录，投稿前删除。
3. **贡献锁定为 3 条**（问题+协议 / 方法 / 证据），新增内容只能强化这 3 条，不得增列。
4. 各章 [引用键] 与 `references.md` 一一对应。

## 与仓库其他部分的关系

- 研究设计依据：`../docs/02_研究方案_SAGA.md`（中文完整方案）
- 现状与差距：`../docs/04_资深工程师审视报告.md`（ICLR 就绪度打分 + P0-P3 工程路线图）
- 实验章的图表合同 ↔ 审视报告 §4 的工程任务一一对应，作为"写作反向锁定工程"的机制。

## 下一步（按依赖顺序）

1. P0：动态分簇可学习性单元测试（决定 §4.2 保留动态版还是降级）+ GPU 资源确认；
2. P1-P2：JAX 迁移 + 训练器重写 + league 流水线（审视报告 §4）；
3. 实验跑完后回填 §5 与摘要占位；
4. md → LaTeX（ICLR 2027 模板），Ethics/Reproducibility 移至规定位置。
