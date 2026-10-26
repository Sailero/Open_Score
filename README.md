# Open-SCORE · LEAF

本分支将算法、环境和运行工具直接放在仓库根目录。八个学习方法为 LEAF、REFIL、REFIL-matched、QMIX、DCG、SPECTra、ALMA、TransfQMix，另有 Rule / Random 两条不训练锚点。仓库内包含原生 PureHAD，不再需要从另一个 Git 仓库安装 HAD。

| 入口 | 内容 |
|---|---|
| [方法、基线与环境说明](docs/方法、基线与环境说明.md) | LEAF 前向、各基线实际实现、Rule / Random、物理与任务协议 |
| [项目框架](docs/项目框架.md) | 完整文件树、文件职责、实现映射、训练 / 评估 / 绘图数据流 |
| [算法](algos/) / [环境](envs/) / [配置](configs/) | 学习实现、PureHAD 与可读运行配置 |
| [保存材料](outputs/) | 历史完整配置、聚合统计和六组 PNG / PDF 图 |

LEAF 的内部键为 `regir_nomem`：query 不读取上一时刻 hidden，Temporal GRU 和上一动作仍然保留。QMIX 的兼容键为 `b2_qmix_atten`，本地结构是 attention actor + FlexQMIX，关闭 REFIL imagined 辅助训练。Random 的执行策略为 `random_accel`，锚点记录ID为 `random`；公开聚合沿用 `random_accel` 并保留来源别名。

## 安装

使用 Python 3.10 或更高版本及适合本机的 PyTorch 构建，在仓库根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

requirements.txt 先以 editable 方式安装仓库内 PureHAD，再安装本项目及训练依赖。源码根目录通过打包映射保持 Python 包名 `open_score`，无需额外的同名嵌套目录。PureHAD 位于 `envs/PureHAD/`，来源是 SkyHAD 提交 `9536b74bdfaf75b832598e4b12d356f2e5a52ffc`；其实际核心登记为 `skyhad-3.0.0a1`，物理协议为 `rebuild-calibrated-v3-r7-target-initialization`。历史训练环境登记不完整，当前安装入口不能保证精确重现全部历史成绩。

## 训练与冻结评估

`configs/base.yaml` 保存公共控制，各学习方法 YAML 保存结构与学习差异；`configs/rule.yaml` 和 `configs/random.yaml` 直接控制锚点执行，蓝方策略从公共 env_args 读取。每次运行把合并后的完整配置保存到模型目录，历史 `config.json` 保持原样供加载与溯源。

普通训练须明确方法、种子、步预算。默认单 GPU、单训练槽，可用 --devices 与 --per-gpu 显式调整。以下完整训练示例选择两个 GPU、每 GPU 三个槽；默认不追加正式评估，只有 --eval-after-train 才启用独立评估池。实际运行范围与预算按当次实验约定：

```powershell
python -X utf8 scripts/train.py --stage train --method regir_nomem,refil --seeds 0,1,2 --steps 1000000 --devices 0,1 --per-gpu 3
python -X utf8 scripts/train.py --stage status
python -X utf8 scripts/train.py --stage stop
python -X utf8 scripts/train.py --stage train --method regir_nomem,refil --seeds 0,1,2 --steps 1000000 --devices 0,1 --per-gpu 3 --resume
python -X utf8 scripts/eval.py --stage final --method regir_nomem,refil --seeds 0,1,2
python -X utf8 scripts/eval.py --stage depth --method regir_nomem --seeds 0,1,2 --depths 1,2,3,4,5,6
python -X utf8 scripts/eval.py --stage anchors
```

默认输出为 `outputs/`，模型位置为 `outputs/<method>/models/seed_<seed>/`。`--output` 可指定同一实验的输出根目录。停止请求在完整批次或完整对局边界生效，`--resume` 继续使用已保存进度。正式评估加载 `final.pt`；`best.pt` 仅保留验证最优备份。独立确认需显式选择 `--stage confirmation`，短链路检查材料统一放在 `outputs/smoke/`。

## 已约定的短链路与历史对照入口

```powershell
python -X utf8 scripts/train.py --stage smoke --devices 0 --per-gpu 2
python -X utf8 scripts/eval.py --stage reproduce --reference-output "<旧本地材料>/outputs/main" --device cpu
```

smoke 默认运行八个学习方法、训练种子0、4v4/K2、batch_run=2、batch=2、buffer=16，每方法在两个独立进程会话中各完成至少一次真实优化更新，第二会话从完整状态恢复。4096 步是诊断上限，不表示完成了该预算训练；已有进度用同一命令加 --resume 继续。reproduce 固定使用历史 final 与训练种子0，对十策略在 4v4/K2 和 15v15/K2、场景9000–9002上执行60个案例，源档案只读。

两者默认写入同一个 outputs/smoke/，唯一紧凑汇总为 [summary.json](outputs/smoke/summary.json)；原始逐场对照、日志和恢复权重留在本地。当前保存的检查为：八个学习方法在 seed0 均通过两个独立进程会话的真实训练、保存与恢复，每方法累计2次优化更新。历史 final 对照完成60/60项，其中53项的目标损伤 D 与回合长度完全相同；3项长度相同，D 仅相差不超过1.8×10⁻¹⁵；另4项 ALMA 的 D 或长度与档案不同。对这4个已选开局使用旧工程和原9e25a1c环境重放，D 与长度均与新代码完全相同，说明这4项档案差异在重构前也存在；这不能写成 ALMA 历史成绩已完全复现，也不推定末位差的具体来源。

环境动态对照已通过：同一 seed9000 与固定二维动作序列下，4v4/K2 和15v15/K2 的原生环境各比较100步；两个规模的旧 / 新封装对照分别覆盖35步和44步完整回合。状态、动作、奖励、事件、实体 mask、势函数塑形及终止字段均逐项精确相等，未采用容差。原生 `core_version` 标签改变，r7物理协议相同；48个原生 Python 文件的导入计算 AST 也相同（排除版本元数据与未使用演示入口）。该证据限于已约定的两个开局和动作序列。证据集中在同一份[summary.json](outputs/smoke/summary.json)。

## 从保存统计绘图

```powershell
python -X utf8 scripts/plot.py
```

绘图读取 `outputs/aggregates.json` 和 `outputs/summary.json`，更新 `outputs/figures/`。它不加载策略或环境，不运行训练或评估，也不创建 Markdown 报告。保存图为[学习与外推比较](outputs/figures/fig_comparison.png)、[机制与深度](outputs/figures/fig_mechanism.png)、[群体保护](outputs/figures/fig_protection.png)、[轨迹](outputs/figures/fig_trajectory.png)、[已保存时延](outputs/figures/fig_decision_timing.png)及[覆盖关系探针](outputs/figures/coverage_binding_probe.png)。

未发布原始轨迹时，命令明确跳过轨迹重绘，并保留已有轨迹图片。已有本地材料可按只读路径提供；兼容新的平铺目录与旧 `outputs/main/had/` 布局，图片仍写入当前输出目录：

```powershell
python -X utf8 scripts/plot.py --data-root "<本地平铺 outputs 目录>"
python -X utf8 scripts/plot.py --data-root "<旧本地材料>/outputs/main"
```

轨迹按方法、训练种子和当前 final 身份筛选，当前有效轨迹仅覆盖 LEAF、REFIL、QMIX、DCG、ALMA 和 Rule。缺失策略或场景不会补跑。

## 发布材料范围

代码、说明、24 份历史模型配置、聚合统计与图片随 Git 发布。原始 CSV / JSONL / 压缩数据、数组和模型权重留在本地；不使用 Git LFS 或 Releases。克隆仓库可读代码、加载配置、重绘已有聚合图；加载历史策略、精确复算成绩或重绘轨迹需要相应的本地原始材料及 final / best 权重。

本分支只保留两份合并说明，历史正式结果报告在 `main` 分支。`outputs/` 保留的历史结果具有各自的来源、运行、种子、协议与实验臂，当前代码与 PureHAD 的可执行性不等于全部历史科学结论已经精确复现。
