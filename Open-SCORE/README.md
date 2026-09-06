# Open-SCORE v5.1：单种子混合难度六任务

当前实验采用一个训练种子 `20260907`，红方 8/12/16/24/32，蓝方为红方人数的 1/2、3/4 或全部，共 15 个场景单元等权混合。六任务包含 9 个主要实验臂，另有 5 个公共对照；正式测试每单元 100 局。

实验总账见 [实验记录](../实验记录.md)，算法语义和完整配额见 [执行方案](../动态分组_六任务并行复现与诊断执行方案_v5.md)。所有方法固定同一规则底层、已知 reactive 对手、两个目标、原生终局奖励和最多 50 个物理步。

## 当前 v5.1 运行入口

在本项目目录执行；以下 `prepare/calibrate/freeze` 仅用于启动新的正式实验。已经冻结的目录直接 `run-all` 或 `watch`。

```powershell
Set-Location 'E:\Code\Open_Score\Open-SCORE'
& 'D:\Software\Anaconda\envs\torch310\python.exe' scripts/run_research_v5.py prepare --config configs/research_v5/standard.yaml --run-dir outputs/v5_parallel/v5_20260907_main
& 'D:\Software\Anaconda\envs\torch310\python.exe' scripts/run_research_v5.py calibrate --run-dir outputs/v5_parallel/v5_20260907_main --seconds 900
# 正式启动前完成验证并提交源码，再冻结协议与配额。
& 'D:\Software\Anaconda\envs\torch310\python.exe' scripts/run_research_v5.py freeze --run-dir outputs/v5_parallel/v5_20260907_main
& 'D:\Software\Anaconda\envs\torch310\python.exe' -u scripts/run_research_v5.py run-all --run-dir outputs/v5_parallel/v5_20260907_main --parallel-tasks 6
```

`run-all` 自动跳过已完成任务，其他任务从已保存的状态继续。单独运行或恢复某个任务：

```powershell
& 'D:\Software\Anaconda\envs\torch310\python.exe' -u scripts/run_research_v5.py task --task T3 --run-dir outputs/v5_parallel/v5_20260907_main
```

T1 为 rollout，T2 为 MCTS，T3 为候选/AR 两臂 PPO，T4 为 ExIt 风格迭代，T5 为 BRIDGE 组划分，T6 为 BCE/ADV/gated 配对价值诊断。

只看实时进度，不重复启动：

```powershell
& 'D:\Software\Anaconda\envs\torch310\python.exe' -u scripts/run_research_v5.py watch --run-dir outputs/v5_parallel/v5_20260907_main
```

本工作区 VS Code 默认构建任务为“v5.1 启动或恢复六任务”，按 `Ctrl+Shift+B` 可在本地集成终端运行。启动脚本先核验运行锁及其进程：已有运行器则只连接监控；没有运行器则启动或恢复。通过 Tasks: Run Task 也可打开独立的仅监控任务。监控显示的 ETA 是当前阶段估计，正式任务没有墙钟截止时间。

输出包含 `shared/budget_manifest.json`、每任务的模型/训练记录/诊断、`comparison.csv`、`comparison_summary.json`、`final_report.md`。逐局评估保存在 `T*/evaluations/<method>/<split>/<checkpoint>/episodes.jsonl`；完整事件、候选和分支使用同目录 `families/*.jsonl.gz` 按回合原子分片，首行是回合指标，其余行带 `record_type`。

T5 在构造配额的 25%、50%、75%、100% 各用同一批 150 局验证，支持恢复；最终主表仍使用最后模型。六任务结束后，`run-all` 还会在其他实验工作进程退出的条件下，逐一测量 14 个方法在共同 30 个状态上的 CPU 单线程部署延迟，输出 `reports/serial_latency.json`。外部系统负载会记录，但不保证整机独占。主报告将这项测量与并行评估期间的事件延迟分开；只运行 `summarize` 不会补做测量。

最小端到端检查使用独立目录：

```powershell
& 'D:\Software\Anaconda\envs\torch310\python.exe' scripts/run_research_v5.py prepare --smoke --run-dir outputs/v5_validation/smoke_new
& 'D:\Software\Anaconda\envs\torch310\python.exe' scripts/run_research_v5.py run-all --run-dir outputs/v5_validation/smoke_new --parallel-tasks 6
```

较小配置只验证实现，不表示已训练充分或性能有效。正式主要结果使用固定配额末期模型；训练、验证、测试和旧协议结果分开报告。

### 回放原始评估轨迹

以下命令从指定运行目录的正式测试分片中，按路径顺序各选第一局成功和失败代表；若只存在其中一种结果，就回放已有的一种。`$replayRun` 也可以改成某个具体的 `families/*.jsonl.gz` 文件。当前可直接验证的目录是 `outputs/v5_acceptance`；正式训练产生测试分片后，将其改成 `outputs/v5_parallel/v5_20260907_main`。验收目录的回放只验证轨迹可复现，不是正式性能证据。

回放读取首行 `episode` 的 `EpisodeSpec` 字段重建开局，再执行各 `event` 保存的 `selected_plan_raw`。每次执行前核对完整状态哈希，执行后核对步数与指挥事件，最后核对原生终局、成功标志和总步数。使用生成轨迹时冻结的代码版本；出现不一致会立即报出对应事件。

```powershell
Set-Location 'E:\Code\Open_Score\Open-SCORE'
$replayRun = 'outputs/v5_acceptance'
@'
from pathlib import Path
import gzip, json, sys
from open_score.grouping.domain import Grouping
from open_score.research_v5.protocol import EpisodeSpec, digest
from open_score.research_v5.simulator import make_env

source = Path(sys.argv[1])
if source.is_file():
    selected = [source]
else:
    examples = {}
    for path in sorted(source.glob('T*/evaluations/*/test/*/families/*.jsonl.gz')):
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            header = json.loads(next(stream))
        assert header['record_type'] == 'episode'
        examples.setdefault(bool(header['success_native']), path)
        if len(examples) == 2:
            break
    selected = list(examples.values())
if not selected:
    raise SystemExit('No committed test episodes yet; select a completed run or one .jsonl.gz file.')
for path in selected:
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        episode = json.loads(next(stream))
        assert episode['record_type'] == 'episode'
        spec = EpisodeSpec(**{key: episode[key] for key in EpisodeSpec.__dataclass_fields__})
        env = make_env(spec)
        steps = events = 0
        info = {}
        try:
            for line in stream:
                event = json.loads(line)
                if event['record_type'] != 'event':
                    continue
                assert not env.done, 'Extra event after terminal'
                assert digest(env.state().to_dict()) == event['state_hash'], event['event_id']
                _, _, _, info = env.step(Grouping.from_dict(event['selected_plan_raw']))
                assert info['delta'] == event['physical_steps'], event['event_id']
                assert info['event_reason'] == event['next_event_type'], event['event_id']
                steps += info['delta']
                events += 1
            assert env.done, 'Replay did not reach native terminal'
            assert bool(info['success']) == bool(episode['success_native'])
            assert steps == env.state().step == episode['physical_steps']
            assert events == episode['command_events']
            assert info['event_reason'] == episode['termination_reason']
            print(f'REPLAY OK success={info["success"]} steps={steps} events={events}\n{path}')
        finally:
            env.close()
'@ | & 'D:\Software\Anaconda\envs\torch310\python.exe' -X utf8 - $replayRun
```

### 完成后的 Git 归档

正式 `run-all` 完成六任务、14 臂末期测试和串行部署测量后，会自动将本轮汇总报告、实验记录、配额与来源说明、任务分析及曲线提交到 Git，提交说明为 `results: v5.1 completed seed 20260907`。归档只提交明确列出的本轮文件，使用 `git commit --only`，不会夹带其他已暂存修改；大型模型检查点、原始轨迹分片和训练日志保留在本地输出目录，不加入该次结果提交。

归档状态、提交号和实际文件清单写入运行目录的 `result_archive.json`。短测不自动提交。若仅归档失败，实验产物仍保留，恢复同一 `run-all` 可重试末尾归档；不应重新训练来处理 Git 问题。

## 历史 v4 使用说明

以下保留 v4 的历史运行入口，不作为 v5.1 的启动命令。所有 v4 路线使用同一纯规则红方底层及已知固定蓝方策略；上层最多每 5 步和非终止伤亡后决策，物理任务最多 50 步。红方组规模没有四人上限。

先进入 `E:\Code\Open_Score\Open-SCORE`。下列命令使用本机已验证的 Torch Python，输出到新目录；已有兼容目录继续运行时加 `--resume`。`--workers` 可以在恢复时改变，`--steps` 可以增加；修改源码、执行器或实验协议后必须使用新目录。

### 先运行主方法 B3

```powershell
& 'D:\Software\Anaconda\envs\torch310\python.exe' scripts/run_research_comparison.py run --routes b3_global --workers 6 --device auto --output outputs/v4_main
```

该命令先收集共享规则底层的数据并训练局部/全局 S2，再测试 B3 和默认四种规则对照。B3 通过监督学习训练评估器，部署时搜索完整编组；它不额外训练 PPO。默认 S2 全局 120 个回合族、局部 160 个回合族，各训练最多 40 个 epoch。`--steps` 控制三条 RL 路线的物理交互预算，不控制 S2 的数据量。

只检查主方法端到端链路，可用更短的命令；这不代表性能实验：

```powershell
& 'D:\Software\Anaconda\envs\torch310\python.exe' scripts/run_research_comparison.py run --routes b3_global --smoke --workers 1 --device auto --output outputs/v4_main_smoke
```

### 六路线首轮比较：每路一个种子，最多六任务并行

```powershell
& 'D:\Software\Anaconda\envs\torch310\python.exe' scripts/run_research_comparison.py run --seeds 20260906 --workers 6 --steps 100000 --device auto --output outputs/v4_comparison
```

三条学习路线为 `r1_ppo`、`r2_teacher_ppo`、`r3_ddqn`；三条分配路线为 `b1_counts`、`b2_local`、`b3_global`。B1/B2/B3 共用准备阶段模型，无重复的策略训练。每路的规则下层相同，PPO actor/critic 不共享参数，教师只用于初始化，训练只使用原生终局奖励。

`--steps 100000` 为学习路线的物理步数目标；默认每路另有 6.5 小时兜底上限，先达到者停止。默认独立测试为 8/12/16/24/32 五种规模，每规模 100 局；学习路线的 best/latest/initialized 分别报告。首轮一个训练种子用于筛选方向，不能据此宣称跨训练种子的稳定优势。

复测并发吞吐：

```powershell
& 'D:\Software\Anaconda\envs\torch310\python.exe' scripts/run_research_comparison.py benchmark --benchmark-workers 3 6 --benchmark-steps 1024 --output outputs/v4_benchmark
```

先做分组有效性诊断：

```powershell
& 'D:\Software\Anaconda\envs\torch310\python.exe' scripts/run_research_comparison.py diagnose --scales 4 8 16 32 --eval-episodes 20 --device cpu --output outputs/v4_rule_gate
```

输出包括根目录 `comparison_report.md`、`comparison_results.csv`、汇总 JSON、调度状态与每路日志，路线目录中的检查点和评估原始记录，以及 `shared/seed_20260906/` 中的 S2 曲线、校准、候选排序和旧模型迁移报告。准确文件索引及本次实测见 [v4 验证报告](docs/results_v4.md)。运行中不是所有路线都需要 GPU；数据采集与物理仿真主要在 CPU，神经网络训练可使用 CUDA。多个训练种子会各自重新训练 S2，再供同种子的六路线共享。

完整参数及单独 `prepare/train/evaluate/report` 入口：

```powershell
& 'D:\Software\Anaconda\envs\torch310\python.exe' scripts/run_research_comparison.py --help
```

### 历史 v3 运行记录

v3 的入口是基于失败诊断重建的三路线过夜训练。历史预算、结果和适用范围已归并到 [实验记录](../实验记录.md)。下方原 v2 说明作为历史版本保留；旧检查点不能跨新增源码直接 `--resume`，重现原 v2 应使用对应历史提交和独立输出。

在此目录执行下面这一条命令即可同时启动三个独立候选方案。上层使用 GPU，原冻结下层使用 CPU；每路最多 7 小时训练，45 分钟评估，总上限 8 小时，`--steps 0` 不在十万步提前结束。

```powershell
& 'D:\Software\Anaconda\envs\torch310\python.exe' scripts/run_overnight_portfolio.py --device cuda --workers 3 --steps 0 --train-hours 7 --eval-minutes 45 --total-hours 8 --output outputs/v3_overnight
```

三路线是 `ppo_structured`、`ppo_teacher`、`candidate_q`，均为一个训练种子；这是算法路线筛选，不是三种子显著性研究。每路使用相同已知 `reactive` 对手、50 步任务、`count_clip3` 下层输入适配和最多 24 个候选动作。只更新上层模型参数；物理世界和所有蓝方实体不变。

汇总文件为 `outputs/v3_overnight/portfolio_report.md` 和 `portfolio_results.csv`。各种子的术语不用于这里：每个路线目录保存 `latest.pt`、验证集选优的 `best.pt`、训练曲线、原始日志、配对测试报告和分组轨迹。测试包括 8/12/16/24/32，对照包含四种无需训练的规则策略。

意外中断时用同一命令加 `--resume`，保留全部参数和输出路径。若从头重跑，改为新输出目录。只查看配置可加 `--dry-run`；短链路检查可运行 `python scripts/run_overnight_portfolio.py --smoke --device cuda --output outputs/v3_smoke_new`。当前交付的短测和诊断不等于已经完成七小时训练或证明上层优势。

## 原 v2 研究与运行记录

本项目研究：在对手完整规则已知、红方下层执行器冻结的条件下，如何利用当前物理态势与已有编组，学习成员减员后的长期分组决策。主方法 B4 依次选择需要释放的成员，再通过合法掩码重建编组，使用真实共享环境的最终成功回报进行 PPO 训练。

每局两个固定目标，最多 **50 个物理步**；每 **5 步及非终止减员事件**获得重决策机会。红方组容量至多 4，允许同目标多组及后备队。守住时限或消灭全部蓝方为成功；任一关键目标被毁为失败。终局成功奖励为 1，其余为 0，`gamma=1`。

## 先运行主算法

在本目录执行。当前机器可用解释器为 `D:\Software\Anaconda\envs\torch310\python.exe`；默认 Anaconda Python 没有 Torch。该环境已有 CUDA 版 Torch 及项目依赖。其他机器需要 Python 3.10 或更新版本，先安装适合硬件的 Torch，再执行 `python -m pip install -e ".[test]"`。

已交付验证结果及其限制见 [小预算验证报告](docs/results_v2.md)。这些结果用于检查训练和评估链路，当前不能支持算法优势。继续训练已交付的 B4 checkpoint：

```powershell
.\scripts\run_main_algorithm.ps1 -TrainMinutes 20 -TrainSteps 0 -EvalEpisodes 100 -EvalMinutes 4 -Output outputs/v2_validation/main_ready -Resume
```

该命令从已完成恢复验证的最新模型（2,204 个物理步、11 次更新）**新增 20 分钟训练**，之后自动评估；`TrainSteps 0` 表示只按时间限制，不设置额外交互步数上限。预计整条命令约 21–25 分钟，取决于实际决策和评估耗时；这是一段追加预算，不是收敛保证。任务时限始终为 50 步。

评估为训练前模型、训练后模型和“训练后相同初始编组、随后保持”的静态对照，每个策略计划 100 局。评估另有 4 分钟预算，只比较完整配对单元并公开实际完成数。旧步数的评估目录保留，训练日志追加；`latest.pt`、当前状态及最新报告索引更新。按需保存的编号快照用于回看历史模型。

要从头创建独立的小预算验证，使用新输出目录，最多训练 2,000 个物理交互步及 5 分钟，以先达到者为准：

```powershell
.\scripts\run_main_algorithm.ps1 -Python 'D:\Software\Anaconda\envs\torch310\python.exe' -TrainMinutes 5 -TrainSteps 2000 -EvalEpisodes 20 -Output outputs/v2_main
```

等价 Python 入口：

```powershell
& D:\Software\Anaconda\envs\torch310\python.exe -m open_score run --config configs/known_opponent_v2.yaml --profile briefing --method selective --steps 2000 --train-seconds 300 --eval-episodes 20 --output outputs/v2_main
```

已有输出目录继续训练时，在上述命令末尾增加 `-Resume` 或 `--resume`。恢复读取最近完成更新的模型和优化器，从新回合继续；未完成的采样批次不恢复。源代码、配置和冻结资产哈希必须一致。修改实现后使用新的输出目录，不能混用已有结果。

时间预算在安全计算边界检查，记录实际耗时和超时量，不能当作硬实时保证。后续训练规模与总预算根据诊断、吞吐和学习曲线确定；训练时间与评估时间分别记录。

`train`、`evaluate`、`diagnose`、`compare`、`report` 可单独运行，完整选项查看：

```powershell
& D:\Software\Anaconda\envs\torch310\python.exe -m open_score --help
```

## 三个种子、四个并发任务：B4 与 B6

以下专用脚本只训练主算法 `selective` 和规则释放 `rule`。三个种子共六个独立任务，最多同时运行四个；每个进程使用一个 CPU 计算线程。空槽自动启动后续任务，训练后按种子并行进行配对评估并汇总。

```powershell
& 'D:\Software\Anaconda\envs\torch310\python.exe' scripts/run_parallel_comparison.py --seeds 20260905 20260906 20260907 --workers 4 --steps 150000 --checkpoint-every 25000 --train-minutes 80 --eval-minutes 10 --eval-episodes 100 --total-minutes 180 --output outputs/v2_parallel_3seeds --resume
```

默认混合 8v8/12v12 训练，在 8v8、12v12、16v16 上每方法每种子计划评估 100 局。每局最多 50 步。B6 以规则选择释放成员，但重建器仍训练；红方下层与完整对手始终冻结。

当前 i7-14700KF、32 GB 内存机器的四进程短试跑测得每任务约 36–52 物理步/秒。15 万步是依据该吞吐和时间窗口选取的请求上限，不是收敛要求；按短跑速度外推整轮约 1.7–2.5 小时，尚未实测完整长训练。以完成步数为主，不在 120 分钟截停。每任务最多累计 80 分钟，整个调用以 180 分钟兜底，提前预留评估和收尾时间。达到调度截止时只停止该脚本启动的子进程；已完成更新保留。

每 25,000 步保存一个固定预算节点，节点之间从新回合继续。若有模型未到请求上限，所有模型统一使用共同完成的最高节点评估，例如全部采用 125,000 步节点；禁止把不同训练预算的最新模型直接混入对照。图中训练曲线也截至统一评估节点。单次宏观转移使实际步数最多多出 4 步，报告同时公开实际节点与请求上限。

`progress.json` 显示阶段、统一评估步数和完成情况；`logs/` 保存各任务日志，`scheduler_events.jsonl` 记录进程启动/结束。汇总见 `comparison_report.md`、`comparison_summary.json` 和 `paired_differences.json`，各种子的曲线及评估图自动生成。三种子的结果属于探索证据，不自动判定方法有效。`--resume` 补齐相同设置下未完成的任务，不重复训练已完成节点；如需改变预算或设置，使用新输出目录。`--dry-run` 可只显示配置。

当前 `batch_size: 16` 是一次 PPO 优化使用的上层事件数；每次采样 64 个上层事件。实现逐事件调用 `evaluate_action` 后合并损失，尚未把环境和变长解码整理为批量张量。因此增大该数值主要改变梯度统计和更新次数，不能保证 GPU 吞吐提升。128 维、2 层、4 头来自根目录 v2 方案的第一版建议；这些参数尚未调优，也不能保证大规模策略质量。

2026-09-06 在本机 RTX 5070 Ti 上对主算法进行约 1,024 步的顺序短测：batch 16 时 CPU 约 63 物理步/秒，上层 CUDA/下层 CPU 约 43，两者 CUDA 约 48；batch 64 时依次约 60、28、25。该测量是当前逐事件实现的短时吞吐，不是收敛比较或并行性能保证。复测速率可运行 `python scripts/benchmark_training_devices.py --output outputs/device_benchmark_new --steps 1024`，结果写入 `benchmark.json`，独立于正式训练输出。后续 GPU 优化应先批量化编码、采样和解码，再评估较大 rollout、minibatch 和网络的效果。

Windows 写入 checkpoint 或 JSON 时，对文件替换的 WinError 5/32/33 增加有限退避重试，始终保留上一份有效文件。仅对本次保存重试修复，`python scripts/repair_windows_checkpoint_run.py --output <旧并行输出目录>` 可迁移旧版运行后再 `--resume`；脚本限定精确的新旧源码哈希，备份原文件，并核对模型、优化器、随机状态和计数器全部不变。它不允许跨训练逻辑或配置变更恢复。

## 方法与验证范围

| 编号 | CLI 方法名 | 用途 |
| --- | --- | --- |
| B0 | `static` | 开局均衡分组，此后只删除死亡成员 |
| B1 | `dlom` | 已知蓝方分布下的 DLOM 期望代理搜索 |
| B2 | `full` | 与 B4 共用编码、修复、critic、PPO，每次全量重构 |
| B3 | `random` | 随机选择释放对象，使用共同学习型修复器 |
| B4 | `selective` | 学习选择释放成员和合法重建，主方法 |
| B5 | `alma` | ALMA 风格适配：候选生成、动作 Q 选择、重放 TD 与赢家轨迹训练 |
| B6 | `rule` | 基于受损组和空间邻近关系的规则释放，共用修复器 |

B5 是针对本项目接口的 **ALMA 风格适配**，不称为原论文的严格复现。B3 的正式归因比较需要使用训练数据估计并冻结与 B4 接近的释放人数分布；同时报告真实释放比例。B0 的静态规则比较不能代替“相同初始编组、是否允许后续调整”的配对诊断。

配置提供四种运行范围：`smoke` 检查端到端链路，`briefing` 优先主算法及有限评估，`minimal` 用于小规模对照和诊断，`formal` 用于正式协议。正式协议预留五个独立训练种子及每核心测试配置 200 回合；训练总物理步数根据最小验证确定并冻结。默认不自动启动正式全套实验。

开发阶段先使用已知 `reactive` 对手；`concentrated` 和 `balanced` 用于分别固定的复核实验。三个对手的规则、参数和 `rush` 下层均不在训练中更新。规则可以依据公开态势产生随机动作，红方不读取尚未公开的动作样本或未来随机状态。

## 结果怎样解读

输出目录保存初始化模型、最近完成更新的 checkpoint、训练/回合日志、评估与报告。训练与评估可分别执行，即使后续诊断尚未完成，也可读取已有完整回合和最新模型。

最小有效证据包括真实交互步数、完成更新数、参数变化、checkpoint 重载、实际完成回合数与成功数、决策耗时、释放比例、任务与队友关系变化、约束违规数和冻结资产哈希。有限评估遇到时间上限时，只将完整配对单元纳入比较，并公开完成数；未完成回合不当作失败或成功。

单种子最小验证只说明实现能训练和评估，不等价于收敛、不证明优于基线，也不替代 E1 组关系有效性、E2 候选与排序、E3 更新范围的诊断和正式多种子结果。实际指标以对应输出目录的运行报告为准，不借用旧实验胜率。

## 冻结资产与源码

`assets/frozen/lcl.pt` 是已有 REFIL-QMIX 下层；`assets/frozen/dlom.pt` 是已有局部结果模型。两者从备份提交对应的本地资产迁移，来源和 SHA-256 见 [资产清单](assets/frozen/manifest.json)。v2 不重新训练它们。

DLOM 的直接训练支持是 **1–4 红方对 1–4 蓝方、50 步任务时限**。它提供局部存活代理，未用重新编组反事实或全局长期结果训练；B1 的配对仅用于评分，物理执行仍是共享环境。下层支持变数量实体输入不保证在所有实际接触人数下都有良好能力，需要执行诊断。

v2 源码主要位于 `src/open_score/grouping`，保留 HAD 物理环境和冻结推理所需模块。历史研究问题及结果见 [实验记录](../实验记录.md)，旧接口与验收细节见 [method_v2.md](docs/method_v2.md)。

运行测试：

```powershell
& D:\Software\Anaconda\envs\torch310\python.exe -m pytest
```
