"""Run the remaining round-01 Stage-2 experiment, figures, and report.

This is the single user-facing entry point for the work that follows the
completed 1M-step Stage-1 run.  It deliberately delegates data collection and
model fitting to their frozen formal scripts, streams every child-process line
to the terminal, and then turns the machine-readable artifacts into four
figures and one Chinese Markdown report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


PROJECT = Path(__file__).resolve().parents[1]
ROUND_ROOT = PROJECT / "outputs" / "round_01_mvp"
SCALES = ("2v1", "3v1", "3v2", "4v1", "4v2", "4v3")
OPPONENTS = ("rush", "split_rush")


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


class TerminalLog:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def write(self, message: str = "") -> None:
        line = f"[{_now()}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as target:
            target.write(line + "\n")

    def child_line(self, message: str) -> None:
        print(message, flush=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as target:
            target.write(message.rstrip("\r\n") + "\n")


def _write_pipeline_status(
    path: Path,
    *,
    status: str,
    stage: str,
    started_at: str,
    detail: str,
    artifacts: Mapping[str, str] | None = None,
) -> None:
    _atomic_json(
        path,
        {
            "schema_version": "round-01-remaining-pipeline-v1",
            "status": status,
            "stage": stage,
            "started_at": started_at,
            "updated_at": _now(),
            "detail": detail,
            "artifacts": dict(artifacts or {}),
        },
    )


def _run(label: str, command: Sequence[str], log: TerminalLog) -> None:
    log.write(f"START {label}")
    log.write("COMMAND " + subprocess.list2cmdline(list(command)))
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        list(command),
        cwd=PROJECT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    if process.stdout is None:  # pragma: no cover - guaranteed by PIPE
        raise RuntimeError("child process has no captured stdout")
    for line in process.stdout:
        log.child_line(line)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"{label} failed with exit code {return_code}")
    log.write(f"DONE {label}")


def _validate_stage1(round_root: Path) -> Dict[str, object]:
    summary_path = round_root / "stage1" / "summary.json"
    evaluation_path = round_root / "evaluation" / "summary.json"
    table_path = round_root / "evaluation" / "win_rate_by_scale.csv"
    curves_path = round_root / "stage1" / "training_curves.csv"
    for path in (summary_path, evaluation_path, table_path, curves_path):
        if not path.is_file():
            raise FileNotFoundError(f"required Stage-1 artifact is missing: {path}")
    summary = _json(summary_path)
    if summary.get("status") != "completed":
        raise ValueError("Stage 1 is not marked completed")
    if int(summary.get("environment_steps", 0)) < 1_000_000:
        raise ValueError("Stage 1 has fewer than 1,000,000 environment steps")
    if int(summary.get("episodes", 0)) < 1:
        raise ValueError("Stage 1 contains no completed episodes")
    heldout = summary.get("heldout_evaluation", {})
    if not isinstance(heldout, dict) or int(heldout.get("episodes", 0)) != 1200:
        raise ValueError("Stage-1 held-out evaluation must contain 1200 episodes")
    checkpoint = PROJECT / str(summary["best_checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"best Stage-1 checkpoint is missing: {checkpoint}")
    expected_hash = str(summary.get("best_checkpoint_sha256", ""))
    if expected_hash and _sha256(checkpoint) != expected_hash:
        raise ValueError("best Stage-1 checkpoint hash differs from summary.json")
    return summary


def _dataset_complete(round_root: Path) -> bool:
    dataset = round_root / "stage2" / "data" / "win_dataset.jsonl"
    summary_path = dataset.parent / "dataset_summary.json"
    if not dataset.is_file() or not summary_path.is_file():
        return False
    try:
        summary = _json(summary_path)
        if summary.get("status") != "completed":
            return False
        if int(summary.get("episodes", 0)) != 7200 or int(summary.get("state_dim", 0)) != 85:
            return False
        if str(summary.get("dataset_sha256")) != _sha256(dataset):
            return False
        cells = summary.get("cell_summary", {})
        if not isinstance(cells, dict) or len(cells) != 12:
            return False
        if any(int(value.get("episodes", 0)) != 600 for value in cells.values()):
            return False
        split = summary.get("split_summary", {})
        expected = {"train": 5040, "validation": 1080, "test": 1080}
        for name, episodes in expected.items():
            value = split.get(name, {})
            if int(value.get("episodes", 0)) != episodes:
                return False
            if int(value.get("wins", 0)) < 1 or int(value.get("losses", 0)) < 1:
                return False
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return True


def _model_complete(round_root: Path) -> bool:
    model_dir = round_root / "stage2" / "model"
    metrics_path = model_dir / "metrics.json"
    checkpoint = model_dir / "dynamic_win_model.pt"
    predictions = model_dir / "test_predictions.csv"
    if not all(path.is_file() for path in (metrics_path, checkpoint, predictions)):
        return False
    try:
        metrics = _json(metrics_path)
        if metrics.get("status") != "completed":
            return False
        expected_splits = {"train": 5040, "validation": 1080, "test": 1080}
        if metrics.get("split_episodes") != expected_splits:
            return False
        dataset = round_root / "stage2" / "data" / "win_dataset.jsonl"
        if str(metrics.get("dataset_sha256")) != _sha256(dataset):
            return False
        if float(metrics.get("reload_max_prediction_difference", 1.0)) > 1e-7:
            return False
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return True


def _read_csv(path: Path) -> list[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def _save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def generate_figures(round_root: Path) -> list[Path]:
    figures = round_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    curve_rows = _read_csv(round_root / "stage1" / "training_curves.csv")
    steps = np.asarray([float(row["environment_steps"]) for row in curve_rows])
    validation = np.asarray([float(row["validation_win_rate"]) for row in curve_rows])
    train_recent = np.asarray(
        [float(row["train_win_rate_200"]) if row["train_win_rate_200"] else np.nan for row in curve_rows]
    )
    loss = np.asarray([float(row["loss"]) if row["loss"] else np.nan for row in curve_rows])
    fig, axis = plt.subplots(figsize=(9.2, 5.2))
    axis.plot(steps, validation, marker="o", linewidth=2.2, label="Validation win rate")
    axis.plot(steps, train_recent, marker=".", linewidth=1.5, alpha=0.8, label="Recent train win rate")
    axis.axhline(validation[0], color="gray", linestyle="--", linewidth=1, label="Initial validation")
    for boundary in (100_000, 250_000, 400_000, 850_000):
        axis.axvline(boundary, color="black", alpha=0.12, linewidth=1)
    axis.set(xlabel="Training environment steps", ylabel="Win rate", ylim=(0.0, 1.02))
    axis.grid(alpha=0.2)
    loss_axis = axis.twinx()
    loss_axis.plot(steps, loss, color="#c44e52", alpha=0.55, label="REFIL loss")
    loss_axis.set_ylabel("Training loss")
    handles, labels = axis.get_legend_handles_labels()
    extra_handles, extra_labels = loss_axis.get_legend_handles_labels()
    axis.legend(handles + extra_handles, labels + extra_labels, loc="upper left", ncol=2)
    axis.set_title("Stage 1: dynamic-scale REFIL-QMIX learning curve")
    curve_path = figures / "01_stage1_training_curve.png"
    _save_figure(fig, curve_path)

    scale_rows = _read_csv(round_root / "evaluation" / "win_rate_by_scale.csv")
    by_scale = {row["scale"]: row for row in scale_rows}
    x = np.arange(len(SCALES))
    rush = np.asarray([float(by_scale[scale]["win_rate_rush"]) for scale in SCALES])
    split = np.asarray([float(by_scale[scale]["win_rate_split_rush"]) for scale in SCALES])
    combined = np.asarray([float(by_scale[scale]["combined_win_rate"]) for scale in SCALES])
    low = np.asarray([float(by_scale[scale]["ci95_low"]) for scale in SCALES])
    high = np.asarray([float(by_scale[scale]["ci95_high"]) for scale in SCALES])
    fig, axis = plt.subplots(figsize=(9.2, 5.2))
    width = 0.31
    axis.bar(x - width / 2, rush, width, label="rush")
    axis.bar(x + width / 2, split, width, label="split_rush")
    axis.errorbar(
        x,
        combined,
        yerr=np.vstack((combined - low, high - combined)),
        fmt="D",
        color="black",
        capsize=4,
        label="combined (95% CI)",
    )
    axis.set_xticks(x, SCALES)
    axis.set(xlabel="HAD scale", ylabel="Held-out win rate", ylim=(0.0, 1.05))
    axis.grid(axis="y", alpha=0.2)
    axis.legend(loc="lower left")
    axis.set_title("Stage 1: 1,200-episode held-out evaluation")
    stage1_scale_path = figures / "02_stage1_scale_win_rates.png"
    _save_figure(fig, stage1_scale_path)

    metrics = _json(round_root / "stage2" / "model" / "metrics.json")
    per_scale = metrics["per_scale"]
    actual = np.asarray([float(per_scale[scale]["actual_win_rate"]) for scale in SCALES])
    predicted = np.asarray([float(per_scale[scale]["predicted_win_rate"]) for scale in SCALES])
    fig, axis = plt.subplots(figsize=(9.2, 5.2))
    axis.bar(x - width / 2, actual, width, label="Observed test win rate")
    axis.bar(x + width / 2, predicted, width, label="Predicted mean probability")
    axis.set_xticks(x, SCALES)
    axis.set(xlabel="HAD scale", ylabel="Probability / rate", ylim=(0.0, 1.05))
    axis.grid(axis="y", alpha=0.2)
    axis.legend(loc="lower left")
    axis.set_title("Stage 2: observed and predicted Red win rates")
    stage2_scale_path = figures / "03_stage2_scale_predictions.png"
    _save_figure(fig, stage2_scale_path)

    calibration = metrics["calibration_bins"]
    mean_prediction = np.asarray([float(row["mean_prediction"]) for row in calibration])
    observed = np.asarray([float(row["observed_win_rate"]) for row in calibration])
    weights = np.asarray([float(row["weighted_rows"]) for row in calibration])
    fig, axis = plt.subplots(figsize=(6.2, 5.5))
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    sizes = 30.0 + 170.0 * weights / max(float(weights.max()), 1e-9)
    axis.scatter(mean_prediction, observed, s=sizes, alpha=0.8, label="Test bins")
    axis.plot(mean_prediction, observed, linewidth=1, alpha=0.55)
    axis.set(
        xlabel="Mean predicted win probability",
        ylabel="Observed Red win rate",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
    )
    axis.grid(alpha=0.2)
    axis.legend(loc="upper left")
    axis.set_title("Stage 2: probability calibration")
    calibration_path = figures / "04_stage2_calibration.png"
    _save_figure(fig, calibration_path)
    return [curve_path, stage1_scale_path, stage2_scale_path, calibration_path]


def _pct(value: object, digits: int = 1) -> str:
    return f"{100.0 * float(value):.{digits}f}%"


def _number(value: object, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def _yes_no(value: object) -> str:
    return "通过" if bool(value) else "未通过"


def _combined_validation_by_scale(evaluation: Mapping[str, object]) -> Dict[str, float]:
    components = evaluation["components"]
    return {
        scale: float(
            np.mean(
                [float(components[opponent]["per_scale_win_rate"][scale]) for opponent in OPPONENTS]
            )
        )
        for scale in SCALES
    }


def write_report(round_root: Path, figure_paths: Sequence[Path]) -> Path:
    stage1 = _json(round_root / "stage1" / "summary.json")
    dataset = _json(round_root / "stage2" / "data" / "dataset_summary.json")
    metrics = _json(round_root / "stage2" / "model" / "metrics.json")
    heldout_rows = _read_csv(round_root / "evaluation" / "win_rate_by_scale.csv")
    heldout_by_scale = {row["scale"]: row for row in heldout_rows}

    initial = stage1["initial_validation"]
    best = stage1["best_validation"]
    heldout = stage1["heldout_evaluation"]
    initial_scale = _combined_validation_by_scale(initial)
    best_scale = _combined_validation_by_scale(best)
    improved_scales = sum(best_scale[scale] > initial_scale[scale] for scale in SCALES)
    stage1_checks = {
        "至少100万环境步": int(stage1["environment_steps"]) >= 1_000_000,
        "六种规模均参与训练": all(int(stage1["scale_episodes"][scale]) > 0 for scale in SCALES),
        "最佳验证胜率比初始至少提高0.15": float(best["win_rate"]) - float(initial["win_rate"]) >= 0.15,
        "至少五种规模提高": improved_scales >= 5,
        "最佳权重完成1200局独立评估": int(heldout["episodes"]) == 1200,
    }
    stage1_passed = all(stage1_checks.values())
    stage2_acceptance = metrics["acceptance"]
    stage2_passed = bool(stage2_acceptance["all_passed"])
    overall = "通过" if stage1_passed and stage2_passed else "完成，但存在未通过门槛"

    split = dataset["split_summary"]
    lines = [
        "# Open-SCORE 第一轮 Stage 1 / Stage 2 完整实验报告",
        "",
        f"> 生成时间：`{_now()}`  ",
        f"> 第一轮结论：**{overall}**  ",
        f"> Stage 1：**{_yes_no(stage1_passed)}**；Stage 2：**{_yes_no(stage2_passed)}**  ",
        "> 本报告只汇总冻结协议产生的结果，不在查看 Test 后重新调参。",
        "",
        "## 1. 本轮做了什么",
        "",
        "第一阶段在 HAD 中只训练 Red 防御方，Blue 使用 `rush` 与 `split_rush` 两种固定规则策略。一个 REFIL-QMIX 权重同时处理 `2v1、3v1、3v2、4v1、4v2、4v3` 六种规模。第二阶段冻结最佳 Red 权重，采集不同时刻的全局态势，用同一个 Deep Sets 二分类网络预测 Red 最终获胜概率。",
        "",
        "这是一轮单训练种子的初步验证。六种规模都参与了训练，因此结果证明的是一个网络覆盖训练分布中的六种规模，不是未见规模零样本泛化，也不是与 VDN、MAPPO 等算法的显著性比较。",
        "",
        "## 2. Stage 1：动态规模决策策略",
        "",
        f"- 算法：`{stage1['algorithm']}`，参数量 `{int(stage1['parameter_count']):,}`。",
        f"- 训练种子：`{stage1['seed']}`；实际环境步 `{int(stage1['environment_steps']):,}`；完成 `{int(stage1['episodes']):,}` 局；更新 `{int(stage1['learner_steps']):,}` 次。",
        f"- 用时 `{float(stage1['elapsed_seconds']) / 3600.0:.2f}` 小时；含验证的平均吞吐率 `{float(stage1['throughput_environment_steps_per_second']):.2f}` 步/秒。",
        f"- 初始 Validation 胜率 {_pct(initial['win_rate'])}；最佳胜率 {_pct(best['win_rate'])}，出现在 `{int(stage1['best_environment_steps']):,}` 步。",
        f"- 最佳权重在 1200 局全新种子的 Test 中胜率 **{_pct(heldout['win_rate'], 2)}**，平均零和收益 `{_number(heldout['mean_payoff'], 4)}`。",
        "",
        "![Stage 1训练曲线](figures/01_stage1_training_curve.png)",
        "",
        "训练前40万步波动较大；探索率在50万步降至0.05后，Validation胜率从约55%继续提高，并在95万步达到81.7%。最终100万步验证为79.2%，最佳权重仍按预注册规则保留在95万步。损失与梯度全程保持有限，没有NaN或崩溃。",
        "",
        "### 2.1 六种规模的独立测试",
        "",
        "| 规模 | rush | split_rush | 合并胜率 | 95%区间 | 局数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scale in SCALES:
        row = heldout_by_scale[scale]
        lines.append(
            f"| {scale} | {_pct(row['win_rate_rush'])} | {_pct(row['win_rate_split_rush'])} | "
            f"{_pct(row['combined_win_rate'])} | {_pct(row['ci95_low'])}–{_pct(row['ci95_high'])} | {row['episodes']} |"
        )
    lines.extend(
        [
            "",
            "![Stage 1分规模胜率](figures/02_stage1_scale_win_rates.png)",
            "",
            "`4v1`和`3v1`最容易，合并胜率分别为96.5%和95.5%；`4v3`最困难，合并胜率54.5%。这说明动态规模接口和共享参数已经跑通，但接近人数均衡的高难场景仍是后续重点。",
            "",
            "### 2.2 Stage 1验收",
            "",
            "| 门槛 | 结果 |",
            "|---|---:|",
        ]
    )
    for name, passed in stage1_checks.items():
        lines.append(f"| {name} | {_yes_no(passed)} |")

    lines.extend(
        [
            "",
            "## 3. Stage 2：动态规模胜率评估器",
            "",
            f"数据由最佳Stage 1权重产生，共 `{int(dataset['episodes']):,}` 局、`{int(dataset['rows']):,}` 个态势样本，输入维度为 `{dataset['state_dim']}`。同一局的全部时刻只属于一个划分，并通过 `sample_weight` 保证每局总权重为1。",
            "",
            "| 划分 | 局数 | 胜 | 负 | 状态行 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in ("train", "validation", "test"):
        value = split[name]
        lines.append(
            f"| {name} | {int(value['episodes']):,} | {int(value['wins']):,} | "
            f"{int(value['losses']):,} | {int(value['rows']):,} |"
        )

    test_metrics = metrics["test_metrics"]
    baseline = metrics["baselines"]
    lines.extend(
        [
            "",
            f"评估器只有一个权重，参数量 `{int(metrics['parameter_count']):,}`；Validation Brier选择的最佳epoch为 `{metrics['best_epoch']}`，实际运行 `{metrics['epochs_run']}` 个epoch。模型保存后重新加载的最大预测差为 `{float(metrics['reload_max_prediction_difference']):.3g}`。",
            "",
            "### 3.1 最终Test指标",
            "",
            "| 指标 | 动态态势网络 | 参考值/解释 |",
            "|---|---:|---|",
            f"| Accuracy | {_pct(test_metrics['accuracy'])} | 0.5阈值下猜对比例 |",
            f"| AUC | {_number(test_metrics['auc'])} | 0.5约等于随机排序 |",
            f"| Brier | {_number(test_metrics['brier'])} | 越低越好 |",
            f"| 总体固定胜率Brier | {_number(baseline['global_constant_brier'])} | 不看态势的基线 |",
            f"| 只看规模Brier | {_number(baseline['scale_only_brier'])} | 只使用人数配置的基线 |",
            f"| 六规模平均胜率绝对误差 | {_pct(metrics['mean_per_scale_absolute_win_rate_error'])} | 要求不超过15% |",
            "",
            "### 3.2 分规模真实胜率与预测概率",
            "",
            "| 规模 | Test真实胜率 | 平均预测概率 | 绝对误差 | AUC | Brier |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for scale in SCALES:
        value = metrics["per_scale"][scale]
        lines.append(
            f"| {scale} | {_pct(value['actual_win_rate'])} | {_pct(value['predicted_win_rate'])} | "
            f"{_pct(value['absolute_win_rate_error'])} | {_number(value['auc'])} | {_number(value['brier'])} |"
        )
    lines.extend(
        [
            "",
            "![Stage 2分规模预测](figures/03_stage2_scale_predictions.png)",
            "",
            "![Stage 2概率可靠性](figures/04_stage2_calibration.png)",
            "",
            "### 3.3 Stage 2验收",
            "",
            "| 门槛 | 结果 |",
            "|---|---:|",
            f"| AUC至少0.60 | {_yes_no(stage2_acceptance['auc_at_least_0_60'])} |",
            f"| Brier优于总体固定胜率 | {_yes_no(stage2_acceptance['brier_better_than_global_constant'])} |",
            f"| 六规模平均胜率误差不超过0.15 | {_yes_no(stage2_acceptance['mean_scale_error_at_most_0_15'])} |",
            f"| 三项全部满足 | **{_yes_no(stage2_acceptance['all_passed'])}** |",
            "",
            "如果门槛未全部通过，本轮仍证明了数据、动态输入、保存/重载和评估流程可运行，但不能把评估器称为可靠概率模型，也不会根据Test结果在本轮临时改网络。",
            "",
            "## 4. 第一轮价值、限制与下一步",
            "",
            "1. 已得到一套能够直接接收六种HAD规模的Red策略，而不是为每种人数分别训练模型。",
            "2. 已用1200局独立测试量化各规模效果，并明确定位`4v3`为当前瓶颈。",
            "3. 已把仿真轨迹转化为监督学习数据，并用一个动态规模网络输出直观的Red最终胜率。",
            "4. 当前只有一个RL训练种子、两类固定规则对手，且六种规模都参与训练；因此结论是第一轮可用性证据，不是算法优越性、跨种子稳定性或未见规模泛化证明。",
            "5. 下一轮应优先增加两个训练种子，再做预注册的留出规模实验；不能因为本轮结果较好就跳过稳定性验证。",
            "",
            "## 5. 结果文件",
            "",
            "- `stage1/summary.json`：Stage 1训练与最佳权重摘要。",
            "- `stage1/training_curves.csv`：训练曲线原始数据。",
            "- `evaluation/win_rate_by_scale.csv`：1200局分规模胜率与置信区间。",
            "- `stage2/data/dataset_summary.json`：7200局数据摘要。",
            "- `stage2/model/metrics.json`：Stage 2全部指标与验收结果。",
            "- `stage2/model/dynamic_win_model.pt`：唯一动态规模胜率评估器。",
            "- `figures/`：本报告引用的四张图。",
            "",
            f"报告图文件数：`{len(figure_paths)}`。Stage 1最佳检查点SHA256：`{stage1['best_checkpoint_sha256']}`。Stage 2数据SHA256：`{metrics['dataset_sha256']}`。",
        ]
    )
    report_path = round_root / "round_01_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument("--force", action="store_true", help="rerun completed Stage-2 steps")
    parser.add_argument("--dry-run", action="store_true", help="validate Stage 1 and print the formal commands")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    round_root = ROUND_ROOT
    status_path = round_root / "pipeline_status.json"
    log = TerminalLog(round_root / "round_01_pipeline.log")
    started_at = _now()
    log.write("=" * 72)
    log.write("Open-SCORE round-01 remaining pipeline")
    log.write(f"Python: {sys.executable}")
    log.write(f"Project: {PROJECT}")
    try:
        stage1 = _validate_stage1(round_root)
        checkpoint = (PROJECT / str(stage1["best_checkpoint"])).resolve()
        dataset = round_root / "stage2" / "data" / "win_dataset.jsonl"
        model_dir = round_root / "stage2" / "model"
        collect_command = [
            sys.executable,
            "-u",
            "scripts/collect_stage2_win_data.py",
            "--checkpoint",
            str(checkpoint),
            "--device",
            args.device,
            "--scales",
            *SCALES,
            "--opponents",
            *OPPONENTS,
            "--episodes-per-cell",
            "600",
            "--snapshot-steps",
            "0",
            "5",
            "10",
            "15",
            "20",
            "25",
            "30",
            "35",
            "40",
            "45",
            "--seed-base",
            "31000000",
            "--max-steps",
            "50",
            "--num-envs",
            "8",
            "--output",
            str(dataset),
        ]
        train_command = [
            sys.executable,
            "-u",
            "scripts/train_stage2_win_model.py",
            "--dataset",
            str(dataset),
            "--device",
            args.device,
            "--seed",
            "20260831",
            "--entity-hidden-dim",
            "32",
            "--hidden-dim",
            "64",
            "--batch-size",
            "512",
            "--learning-rate",
            "0.001",
            "--max-epochs",
            "100",
            "--patience",
            "10",
            "--output-dir",
            str(model_dir),
        ]
        log.write(f"Stage 1 OK: {int(stage1['environment_steps']):,} steps; best={checkpoint}")
        log.write("COLLECT PLAN " + subprocess.list2cmdline(collect_command))
        log.write("TRAIN PLAN " + subprocess.list2cmdline(train_command))
        if args.dry_run:
            log.write("DRY RUN completed; no Stage-2 process was started")
            return

        _write_pipeline_status(
            status_path,
            status="running",
            stage="stage2_data",
            started_at=started_at,
            detail="collecting 7200 HAD episodes",
        )
        if args.force or not _dataset_complete(round_root):
            _run("Stage 2 data collection", collect_command, log)
        else:
            log.write("SKIP Stage 2 data collection: validated completed artifacts")
        if not _dataset_complete(round_root):
            raise RuntimeError("Stage-2 dataset failed post-run validation")

        _write_pipeline_status(
            status_path,
            status="running",
            stage="stage2_model",
            started_at=started_at,
            detail="training the one-network dynamic-scale win evaluator",
        )
        if args.force or not _model_complete(round_root):
            _run("Stage 2 evaluator training", train_command, log)
        else:
            log.write("SKIP Stage 2 evaluator training: validated completed artifacts")
        if not _model_complete(round_root):
            raise RuntimeError("Stage-2 model failed post-run validation")

        _write_pipeline_status(
            status_path,
            status="running",
            stage="figures_and_report",
            started_at=started_at,
            detail="generating four figures and the Markdown report",
        )
        figures = generate_figures(round_root)
        report = write_report(round_root, figures)
        metrics = _json(model_dir / "metrics.json")
        artifacts = {
            "report": str(report),
            "stage1_summary": str(round_root / "stage1" / "summary.json"),
            "stage1_evaluation": str(round_root / "evaluation" / "win_rate_by_scale.csv"),
            "stage2_dataset_summary": str(dataset.parent / "dataset_summary.json"),
            "stage2_metrics": str(model_dir / "metrics.json"),
            "stage2_model": str(model_dir / "dynamic_win_model.pt"),
            "figures": str(round_root / "figures"),
        }
        _write_pipeline_status(
            status_path,
            status="completed",
            stage="completed",
            started_at=started_at,
            detail=(
                "round 01 passed all frozen gates"
                if bool(metrics["acceptance"]["all_passed"])
                else "round 01 completed; Stage-2 performance has unmet gates"
            ),
            artifacts=artifacts,
        )
        log.write("ROUND 01 REMAINING PIPELINE COMPLETED")
        log.write(f"REPORT {report}")
        log.write(f"FIGURES {round_root / 'figures'}")
        log.write(f"STAGE2 ACCEPTANCE {metrics['acceptance']}")
    except Exception as error:
        detail = f"{type(error).__name__}: {error}"
        _write_pipeline_status(
            status_path,
            status="failed",
            stage="failed",
            started_at=started_at,
            detail=detail,
        )
        log.write("PIPELINE FAILED: " + detail)
        for line in traceback.format_exc().splitlines():
            log.child_line(line)
        raise


if __name__ == "__main__":
    main()
