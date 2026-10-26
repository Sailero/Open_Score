# Open-SCORE · LEAF

LEAF 通过当前实体上下文、接收者读取和时序记忆支持多目标防护。本仓库保留一个 HAD 主实验：LEAF、七个当前学习基线和 Rule / Random 共十种策略，以及同一 LEAF checkpoint 的 R1–R6 冻结深度与机制分析。

| 入口 | 内容 |
|---|---|
| [主实验结果分析](docs/LEAF_主实验结果分析.md) | 唯一正式报告：十策略比较、验证曲线、正常保护、冻结深度、独立确认、机制与已保存时延 |
| [方法说明](docs/LEAF_方法与变体说明.md) | LEAF 实际前向、训练深度、执行深度与解释边界 |
| [Baseline 与 Rule 实现](docs/LEAF_Baseline与Rule实现说明.md) | 七个学习基线、两条规则锚点及实际适配 |
| [HAD 环境与问题建模](docs/HAD_环境与问题建模.md) | 实体、物理、动作、奖励和终止协议 |
| [实验材料](Open-SCORE/outputs/main) | 保存配置、筛选后的聚合统计、六组 PNG/PDF 图 |

保留方法为 `regir_nomem`（LEAF）、`refil`、`refil_matched`、`b2_qmix_atten`、`dcg`、`spectra`、`alma`、`transfqmix`、`rule_nv1`、`random_accel`。`regir_nomem` 表示 query 不读取上一时刻 hidden，Temporal GRU 和上一动作仍保留；Random 在历史数据中的来源名为 `random`，公开聚合保留 `source_method` 别名。

## 结果与材料范围

正式成绩取 [aggregates.json](Open-SCORE/outputs/main/had/aggregates.json) 的 `comparison`。当前 SPECTra / TransfQMix 使用19维实体输入，各三种子、每模型1M物理步；正式 All24 目标损伤分别为 **6.2871±0.0289、4.2518±0.7733**。八个学习策略与两条锚点均进入主比较图和表；没有保存的碰撞或拦截指标保持缺失。

代码、报告、24份模型配置、聚合统计与图片随 Git 发布。**原始 CSV / JSONL / 压缩数据和模型权重保留本地，不使用 Git LFS 或 Releases 分发。** 克隆仓库可以检查代码和报告、重绘聚合图；精确复算既有成绩、重绘逐步轨迹或运行已有模型，需要对应本地原始材料和 final/best 权重。权重目录约定为 `Open-SCORE/outputs/main/had/<method>/models/seed_<seed>/`；正式成绩使用 final，best 仅作验证最优备份。

历史来源名、模型身份和协议各自保留。LEAF 与部分基线的历史训练物理登记不完整；主性能、冻结补评、SkyHAD 辅助机制与轨迹不能宣称全部同一协议。R1–R6 保留432个单元汇总，其中 R4 复用21,600场，新增108,000场逐场 JSONL 未提供。完整边界见主报告。

## 安装

在仓库根目录创建单独环境，使用 Python 3.10 或更高版本及适合本机的 PyTorch 构建：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e "./Open-SCORE[training]"
```

原生 HAD 依赖固定为 [Sailero/SkyHAD 的提交 9e25a1c](https://github.com/Sailero/SkyHAD/tree/9e25a1c3e32cba31f00c0a42bd2c1e58988afaa2)，完整安装标识为：

```text
had-env @ git+https://github.com/Sailero/SkyHAD.git@9e25a1c3e32cba31f00c0a42bd2c1e58988afaa2
```

该提交登记 `had-workbench-2.1.0 / rebuild-calibrated-v3-r7-target-initialization`；包版本号本身不足以定位物理实现。此安装入口用于当前代码，不能补全未记录的历史环境，也不能仅凭安装成功保证重现全部历史得分。

## 训练与评估入口

配置位于 `Open-SCORE/configs/`，入口位于 `Open-SCORE/scripts/`。训练必须明确方法、种子和预算；以下是命令示例，本次整理没有执行它们：

```powershell
python -X utf8 Open-SCORE/scripts/train.py --stage train --method regir_nomem,refil --seeds 0,1,2 --steps 1000000
python -X utf8 Open-SCORE/scripts/train.py --stage status
python -X utf8 Open-SCORE/scripts/train.py --stage stop
python -X utf8 Open-SCORE/scripts/eval.py --stage final --method regir_nomem,refil --seeds 0,1,2
python -X utf8 Open-SCORE/scripts/eval.py --stage depth --method regir_nomem --seeds 0,1,2 --depths 1,2,3,4,5,6
python -X utf8 Open-SCORE/scripts/eval.py --stage anchors
```

恢复使用 `--resume`；独立确认须显式选择 `--stage confirmation`。模型和原始输出属于本地实验材料，现有进度应恢复使用。任何新的实验运行都需单独约定方法、种子和预算。

## 从已有结果绘图

```powershell
python -X utf8 Open-SCORE/scripts/plot.py
```

该命令直接读取公开聚合，刷新比较、机制、保护、时延与 LEAF 探针图。它不加载模型或环境，不训练、不评估，也不生成额外报告。逐步轨迹未发布，缺少本地材料时会明确跳过轨迹图；仓库保留已生成图片。

已有本地材料可通过只读路径提供，图片仍写入当前仓库 `outputs/main/had/figures/`：

```powershell
python -X utf8 Open-SCORE/scripts/plot.py --data-root "<本地材料目录>/outputs/main"
```

轨迹先按所选方法和当前 final 身份筛选。现有轨迹仅覆盖 LEAF、REFIL、QMIX-Atten、DCG、ALMA 和 Rule；其余四策略没有对应的当前有效轨迹，不补跑或替换为其他模型。原始材料目录可含历史行，分析时须保留来源、运行、种子、协议和实验臂身份。
