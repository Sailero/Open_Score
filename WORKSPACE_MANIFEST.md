# Workspace manifest

**Reorganized**: 2026-08-27  
**Original workspace**: `D:\Code\Third_Paper`

## 01_original_saga_iclr

原始 SAGA 项目整体移动到此目录，未拆散其 `docs/`、`paper/`、`prototype/`、`README.md` 和 `LICENSE` 的内部相对关系。整理时原项目含 56 个非缓存研究/代码/输出文件；Python 缓存文件一并保留，以避免在无版本库的情况下误删用户材料。

`_incomplete_spectra_clone_20260827/` 是整理过程中因 GitHub HTTPS 连接重置产生的空/不完整技术残留，不属于 SAGA 研究内容，也不能运行。

## 02_q2_scale_opponent_generalization

新建的独立二区研究目录。它不引用归档目录中的代码路径；所需的 SwarmBattle 环境和脚本已复制到自身 `src/`，因此后续修改不会污染原 SAGA。2026-08-27 二次复核后，当前主问题为“阵容变化是否被在线对手模型误判为策略变化”，现有 OC-HPN 降为基线，CDOC-HPN 为候选主方法；以 `docs/06_研究价值与创新性复核.md` 为最终口径。

## 恢复方式

本次整理只执行移动和复制，没有删除原研究文件。若要恢复旧布局，可把 `01_original_saga_iclr/` 下的 `README.md`、`LICENSE`、`docs/`、`paper/`、`prototype/` 移回工作区根目录；不建议这样做，因为会重新混合两个研究范围。
