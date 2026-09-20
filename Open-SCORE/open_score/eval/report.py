"""Refresh the one illustrated report from committed experiment records."""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import threading

from open_score.utils.logging import DEFAULT_OUTPUT, FORMAL_RUN, V3_OUTPUT, V4_OUTPUT, V5_OUTPUT, read_latest, read_records, unique_episodes
from .protocol import (CYCLE_SERIES_METHODS, DEPTH_SWEEP_DEPTHS, EQUAL_SCALE_CONFIGS, FINAL_CONFIGS,
                       FINAL_EPISODES_PER_CONFIG, POOL_ONLY_METHODS, RATIO_CONFIGS,
                       TARGET_K_VALUES, TARGET_PLOT_NS, TARGET_SCALE_NS, TEST_CONFIGS, VALIDATION_CONFIGS,
                       compute_nds, config_key, config_label, depth_eval_total, final_configs,
                       OFFICIAL_CHECKPOINT, official_eval_t_env, parse_checkpoint_stem,
                       parse_checkpoint_t_env, parse_sweep_depth, run_directory,
                       training_finished, validation_score)

METHODS = ("b2_qmix_atten", "refil", "dcg", "spectra", "alma")
SIDE_METHODS = ("b0_qmix", "gnn_qmix")
ALL_METHODS = METHODS + SIDE_METHODS
LABELS = {"b0_qmix": "B0", "b2_qmix_atten": "QMIX", "refil": "REFIL",
          "dcg": "DCG", "gnn_qmix": "GNN", "spectra": "SPECTra", "alma": "ALMA",
          "refil_local_mild": "REFIL-L0.15", "refil_local_mid": "REFIL-L0.30",
          "refil_count": "REFIL-C", "refil_count_ln": "REFIL-C",
          "refil_count_noln": "REFIL-C−LN",
          "refil_cycle": "REFIL-B", "refil_card": "REFIL-A",
          "refil_feedback": "REFIL-F", "refil_slot": "REFIL-K10",
          "regir": "ReGIR", "regir_norefil": "ReGIR−REFIL", "regir_nocount": "ReGIR−zn",
          "regir_r1": "ReGIR-R1", "regir_last": "ReGIR-last", "refil_matched": "REFIL-matched",
          "random": "Random", "rule_nv1": "Rule nv1"}
COLORS = {"b0_qmix": "#5b6770", "b2_qmix_atten": "#1675b8", "refil": "#e36b32",
          "dcg": "#7955a3", "gnn_qmix": "#32865e", "spectra": "#bf4c79",
          "alma": "#b08a1e", "random": "#979fa5", "rule_nv1": "#267c4f",
          "alma_fullobs": "#3d6ea8", "alma_blue": "#c47b2b", "alma_event": "#6a4c93",
          "alma_nomask": "#1a7f7a",
          "refil_local_mild": "#2a9d8f", "refil_local_mid": "#c47b2b",
          "refil_count": "#3d6ea8", "refil_count_ln": "#3d6ea8",
          "refil_count_noln": "#6aa9e8",
          "refil_cycle": "#1b7f7a", "refil_card": "#3d6ea8",
          "refil_feedback": "#c47b2b", "refil_slot": "#6a4c93",
          "regir": "#1b7f7a", "regir_norefil": "#5b8a72", "regir_nocount": "#3d6ea8",
          "regir_r1": "#c47b2b", "regir_last": "#6a4c93", "refil_matched": "#8c4b23"}
V4_METHODS = ("refil_local_mild", "refil_local_mid", "refil_count", "refil_count_noln", "refil")
V4_TRAIN_METHODS = ("refil_local_mild", "refil_local_mid", "refil_count")
V4_COMPARE_ARMS = tuple(method for method in V4_METHODS if method != "refil")
V4_SEEDS = (0,)
V5_GLOBAL_METHODS = ("refil_cycle", "refil_card", "refil_feedback", "refil_slot")
V5_METHODS = V5_GLOBAL_METHODS + ("refil",)
V5_TRAIN_METHODS = ("refil_count",) + V5_GLOBAL_METHODS
V5_COUNT_METHODS = ("refil_count_noln", "refil_count_ln")
V5_SEEDS = (0,)
PROBE_METHODS = ("alma", "alma_fullobs", "alma_blue", "alma_event", "alma_nomask")
PROBE_LABELS = {"alma": "ALMA-掩码", "alma_fullobs": "ALMA-全场",
                "alma_blue": "ALMA-蓝方", "alma_event": "ALMA-事件",
                "alma_nomask": "ALMA-NoMask",
                "b2_qmix_atten": "v3 QMIX", "refil": "v3 REFIL",
                "random": "Random", "rule_nv1": "Rule nv1"}
PROBE_PLOT_LABELS = {"alma": "ALMA-mask", "alma_fullobs": "ALMA-fullobs",
                     "alma_blue": "ALMA-blue", "alma_event": "ALMA-event",
                     "alma_nomask": "ALMA-NoMask",
                     "b2_qmix_atten": "v3 QMIX", "refil": "v3 REFIL",
                     "random": "Random", "rule_nv1": "Rule nv1"}
PROBE_SEEDS = (0,)
# Hard-mask ALMA episode/learning CSVs were not recovered during the
# directory migration. Numbers below are copied from the pre-swap main v3 report.
ARCHIVED_MASK_ALMA = {
    "last_mean": -6.07,
    "last_std": 0.47,
    "seed0_last": -6.57,
    "intercept": 0.38,
    "seeds": (
        {"seed": 0, "status": "archived", "steps": 1000000, "points": "50/50",
         "best": -3.73, "best_step": 280358},
        {"seed": 1, "status": "archived", "steps": 1000000, "points": "50/50",
         "best": -4.75, "best_step": 580119},
        {"seed": 2, "status": "archived", "steps": 1000000, "points": "50/50",
         "best": -4.70, "best_step": 200113},
    ),
}
APPENDIX_HEADING = "## 可靠性问题与三个改进方案"
BENCHMARK_REVISION = "filled_padding_compact_mixer_frozen_target"
FORMAL_SEEDS = (0, 1, 2)
_REPORT_THREAD_LOCK = threading.RLock()


def _preserve_appendix(output):
    path = Path(output) / "实验报告.md"
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    idx = text.find(APPENDIX_HEADING)
    if idx < 0:
        return ""
    return text[idx:].rstrip() + "\n"


def _config_axis(config):
    if config in VALIDATION_CONFIGS:
        return "训练池内"
    if config in EQUAL_SCALE_CONFIGS:
        return "等规模"
    if config in RATIO_CONFIGS:
        return "1:2"
    return "目标外推"


@contextmanager
def _report_lock(output):
    """Serialize a full render, without a separate filesystem lock artifact."""
    output.mkdir(parents=True, exist_ok=True)
    with _REPORT_THREAD_LOCK:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            api = ctypes.WinDLL("kernel32", use_last_error=True)
            api.CreateMutexW.argtypes = (ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR)
            api.CreateMutexW.restype = wintypes.HANDLE
            api.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
            api.WaitForSingleObject.restype = wintypes.DWORD
            api.ReleaseMutex.argtypes = (wintypes.HANDLE,)
            api.CloseHandle.argtypes = (wintypes.HANDLE,)
            name = "Local\\OpenScoreReport_" + str(output.resolve()).casefold().replace("\\", "_").replace(":", "_").replace("/", "_")
            handle = api.CreateMutexW(None, False, name)
            if not handle:
                raise ctypes.WinError(ctypes.get_last_error())
            acquired = False
            try:
                result = api.WaitForSingleObject(handle, 120000)
                if result not in (0, 0x80):
                    raise TimeoutError("The previous experiment-report update is still running")
                acquired = True
                yield None
            finally:
                if acquired:
                    api.ReleaseMutex(handle)
                api.CloseHandle(handle)
        else:
            import fcntl
            path = output / "实验报告.md"
            path.touch(exist_ok=True)
            with path.open("r+b") as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                try:
                    yield stream
                finally:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _number(value, digits=4):
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}" if math.isfinite(float(value)) else "—"
    except (TypeError, ValueError):
        return str(value)


def _numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _cell(value):
    if value is None:
        return "—"
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    elif isinstance(value, float):
        value = f"{value:.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _table(rows, columns):
    result = ["| " + " | ".join(label for _, label in columns) + " |",
              "|" + "|".join("---" for _ in columns) + "|"]
    result.extend("| " + " | ".join(_cell(row.get(key)) for key, _ in columns) + " |" for row in rows)
    return result


def _atomic_text(path, content):
    if path.name == "实验报告.md" and "| B0 D |" in content:
        # A live job started before B0 left the main table must not put it back.
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = content.encode("utf-8")
    with path.open("wb") as stream:
        stream.write(data)
        stream.truncate()
        stream.flush()
        os.fsync(stream.fileno())


def _save_figure(fig, path):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        fig.savefig(temporary, format="png", dpi=150, bbox_inches="tight", facecolor="white")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _anchor_values(rows):
    grouped = defaultdict(dict)
    for row in rows:
        if row["phase"] == "anchor" and row["method"] in ("random", "rule_nv1"):
            grouped[(row["method"], config_key(row["config"]))][row["episode_seed"]] = row
    values = {}
    for key, records in grouped.items():
        expected = set(range(9000, 9300))
        values[key] = dict(n=len(records), complete=set(records) == expected,
                           D=statistics.mean(row["D"] for row in records.values()),
                           rho=statistics.mean(row["rho"] for row in records.values()))
    return values


def _nds(rho, config, anchors):
    random = anchors.get(("random", config), {})
    rule = anchors.get(("rule_nv1", config), {})
    if not random.get("complete") or not rule.get("complete"):
        return None
    return compute_nds(rho, random["rho"], rule["rho"])


def _validation_points(rows, anchors):
    grouped = defaultdict(list)
    for row in rows:
        if row["phase"] == "train_eval":
            grouped[(row["method"], row["seed"], row["eval_point"])].append(row)
    result = []
    for (method, seed, point), episodes in grouped.items():
        score = validation_score(episodes)
        if score is None:
            continue
        by_config = defaultdict(list)
        for row in episodes:
            by_config[config_key(row["config"])].append(row)
        per_config = {config: statistics.mean(row["rho"] for row in values) for config, values in by_config.items()}
        scores = [_nds(rho, config, anchors) for config, rho in per_config.items()]
        result.append(dict(method=method, seed=seed, eval_point=point,
                           t_env=max(row["t_env"] for row in episodes), D=score,
                           rho=statistics.mean(per_config.values()),
                           NDS=statistics.mean(scores) if all(value is not None for value in scores) else None))
    return sorted(result, key=lambda row: (row["method"], row["seed"], row["t_env"]))


def _mean_std(values):
    values = [float(value) for value in values if _numeric(value)]
    if not values:
        return None, None
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def _pool_return(anchors, policy):
    """Red return on the same 4 training-pool configs used for validation."""
    values = [anchors[(policy, config)]["D"] for config in VALIDATION_CONFIGS
              if anchors.get((policy, config), {}).get("complete")]
    if len(values) != len(VALIDATION_CONFIGS):
        return None
    return -statistics.mean(values)


def _anchor_plot_styles(include_random=True):
    styles = (("random", "--"), ("rule_nv1", "-.")) if include_random else (("rule_nv1", "-."),)
    return styles


def _plot_learning(plt, output, points, anchors, methods=None, include_random=True,
                   stem="learning"):
    """Red episode return R=-D, mean±std over training seeds."""
    methods = METHODS if methods is None else methods
    points = [row for row in points if row["method"] in methods and _numeric(row.get("D"))]
    if not points:
        return None
    fig, axis = plt.subplots(1, 1, figsize=(7.6, 4.4), constrained_layout=True)
    for method in methods:
        by_point = defaultdict(list)
        for row in points:
            if row["method"] == method:
                by_point[row["eval_point"]].append(-row["D"])
        xs, ys, err = [], [], []
        for point, rewards in sorted(by_point.items()):
            mean, std = _mean_std(rewards)
            if mean is None:
                continue
            times = [row["t_env"] for row in points if row["method"] == method and row["eval_point"] == point]
            xs.append(statistics.mean(times))
            ys.append(mean)
            err.append(std)
        if xs:
            axis.plot(xs, ys, color=COLORS[method], linewidth=2.2, label=LABELS[method])
            axis.fill_between(xs, [y - e for y, e in zip(ys, err)], [y + e for y, e in zip(ys, err)],
                              color=COLORS[method], alpha=.18, linewidth=0)
    for policy, style in _anchor_plot_styles(include_random):
        value = _pool_return(anchors, policy)
        if value is not None:
            axis.axhline(value, color=COLORS[policy], linestyle=style, linewidth=1.6, label=LABELS[policy])
    axis.set(title="In-distribution red return (higher is better)",
             xlabel="Environment steps", ylabel="Red episode return  R = −D")
    axis.grid(alpha=.2)
    axis.legend(fontsize=9)
    path = output / "figures" / f"{stem}.png"
    _save_figure(fig, path)
    plt.close(fig)
    return path


def _groups(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in keys)].append(row)
    return groups


def _final_diagnostics(rows):
    """Summarize only this run's committed final/anchor episodes, never training."""
    grouped = defaultdict(dict)
    for row in rows:
        if row.get("phase") in ("final_eval", "anchor"):
            key = (row["method"], row["seed"], config_key(row["config"]), row["phase"])
            grouped[key][row["episode_seed"]] = row
    text = ["## 最终评价行为与死亡诊断", ""]
    if not grouped:
        return text + ["最终评价与规则锚点诊断未运行，尚无可汇总的逐局记录；训练、验证或吞吐数据不代填此处。", ""]

    def mean_cell(values):
        valid = [value for value in values if _numeric(value)]
        return f"{statistics.mean(valid):.4f}（{len(valid)}）" if valid else "未记录（0）"

    red_causes = (("shot_down", "被击落"), ("friendly_collision", "同队碰撞"),
                  ("enemy_collision", "异队碰撞"), ("boundary", "边界"), ("self_destruct", "开火自毁"))
    blue_causes = (("intercepted", "被拦截"), ("self_destruct", "自毁"), ("collision", "碰撞"))
    deaths, actions, geometry, targets, q_values = [], [], [], [], []
    analysis = defaultdict(list)
    for (method, seed, config, phase), by_seed in sorted(grouped.items()):
        episodes = list(by_seed.values())
        complete = set(by_seed) == set(range(9000, 9300))
        identity = dict(method=method, seed=seed, config=config_label(config), n=f"{len(episodes)}/300",
                        status="完成" if complete else "进行中，暂定")
        death = dict(identity)
        for side, causes in (("red", red_causes), ("blue", blue_causes)):
            for cause, _ in causes:
                death[f"{side}_{cause}"] = mean_cell(
                    row[f"{side}_deaths_by_cause"].get(cause)
                    if isinstance(row.get(f"{side}_deaths_by_cause"), dict) else None for row in episodes)
        for field in ("red_left", "blue_left", "ep_len"):
            death[field] = mean_cell(row.get(field) for row in episodes)
        deaths.append(death)

        histograms = [row["action_hist"] for row in episodes
                      if isinstance(row.get("action_hist"), (list, tuple)) and len(row["action_hist"]) == 9
                      and all(_numeric(value) and value >= 0 for value in row["action_hist"])
                      and sum(row["action_hist"]) > 0]
        fractions = [statistics.mean(hist[i] / sum(hist) for hist in histograms) for i in range(9)] if histograms else []
        action = dict(identity, hist_n=len(histograms),
                      distribution=" / ".join(f"{100 * value:.2f}" for value in fractions) if fractions else "未记录",
                      noop=mean_cell(row.get("noop_frac") for row in episodes),
                      entropy=mean_cell(row.get("action_entropy") for row in episodes))
        actions.append(action)

        spatial = dict(identity)
        for field in ("mean_speed", "mean_pairwise_dist", "mean_dist_to_nearest_target", "friendly_collisions", "min_pairwise_dist_p05"):
            spatial[field] = mean_cell(row.get(field) for row in episodes)
        collision_values = [row["friendly_collisions"] for row in episodes if _numeric(row.get("friendly_collisions"))]
        collision_n = sum(value > 0 for value in collision_values)
        spatial["collision_episodes"] = f"{collision_n}/{len(collision_values)}" if collision_values else "未记录"
        geometry.append(spatial)

        target = dict(identity)
        damage_cells, uninjured_cells, first_cells = [], [], []
        for index in range(config[2]):
            damages = [row["damage_by_target"][index] for row in episodes
                       if isinstance(row.get("damage_by_target"), (list, tuple)) and len(row["damage_by_target"]) > index]
            first = [row["first_damage_step_by_target"][index] for row in episodes
                     if isinstance(row.get("first_damage_step_by_target"), (list, tuple))
                     and len(row["first_damage_step_by_target"]) > index]
            observed = [value for value in first if value is None or _numeric(value)]
            hit = [value for value in observed if _numeric(value)]
            untouched = sum(value is None for value in observed)
            damage_cells.append(f"T{index}: {mean_cell(damages)}")
            uninjured_cells.append(f"T{index}: {untouched}/{len(observed)}" if observed else f"T{index}: 未记录")
            first_cells.append(f"T{index}: " + (mean_cell(hit) if hit else "未受伤" if observed else "未记录"))
        target.update(damage="；".join(damage_cells), uninjured="；".join(uninjured_cells), first="；".join(first_cells))
        targets.append(target)

        q_row = dict(identity)
        for field in ("q_tot_mean", "q_tot_std", "q_i_mean"):
            q_row[field] = "不适用" if phase == "anchor" else mean_cell(row.get(field) for row in episodes)
        q_values.append(q_row)
        if complete:
            analysis[(method, seed, phase)].append(dict(config=config_label(config), episodes=episodes,
                collision_fraction=collision_n / len(collision_values) if collision_values else None))

    text += [f"仅汇总 {FORMAL_RUN} 的 final_eval 与 anchor，按方法、训练种子和配置分组；每组目标为固定 300 个评价种子。"
             "未满 300 局的表格是已落盘样本的暂定值，不用于下方完成组分析。各标量为有效回合等权均值，"
             "单元格括号为该字段有效回合数，缺失值不补零；未按 ep_len 再加权。死亡数为每局人数均值。", ""]
    identity_columns = [("method", "方法"), ("seed", "训练种子"), ("config", "配置"), ("n", "局数"), ("status", "状态")]
    text += ["红方死亡与回合结束状态：", ""]
    text += _table(deaths, identity_columns + [(f"red_{cause}", label) for cause, label in red_causes]
                   + [("red_left", "红方存活"), ("blue_left", "蓝方存活"), ("ep_len", "回合长度")]) + [""]
    text += ["蓝方死亡：", ""]
    text += _table(deaths, identity_columns + [(f"blue_{cause}", label) for cause, label in blue_causes]) + [""]
    text += ["动作统计仅包含每步动作前仍存活的红方。九维分布先将每局 action_hist 归一化，再对有效回合等权平均，"
             "按模型动作 0–8 顺序显示百分比，0 为 noop；不因较长回合或更多存活红方增加回合权重。"
             "noop 为已记录 noop_frac 的回合均值，动作熵为每局自然对数熵的均值，并非均值分布的熵。", ""]
    text += _table(actions, identity_columns + [("hist_n", "分布有效局"), ("distribution", "动作 0 / 1 / 2 / 3 / 4 / 5 / 6 / 7 / 8：%"),
                   ("noop", "noop 比例"), ("entropy", "动作熵（nat）")]) + [""]
    text += ["空间与拥挤诊断采用环境原生距离、速度单位。表中 p05 是每局 p05 的均值，不是跨局合并后的第 5 百分位。"
             "碰撞局数统计 friendly_collisions>0 的有效回合；同队碰撞均值按去重后的红红无序对计数。", ""]
    text += _table(geometry, identity_columns + [("mean_speed", "速度"), ("mean_pairwise_dist", "红红间距"),
                   ("mean_dist_to_nearest_target", "到最近目标距离"), ("min_pairwise_dist_p05", "每局最小红红间距 p05"),
                   ("friendly_collisions", "同队碰撞对/局"), ("collision_episodes", "有同队碰撞局/有效局")]) + [""]
    text += ["逐目标伤害含零伤害回合；T0、T1 等表示环境目标索引，不代表跨回合固定的空间位置。"
             "首次受伤步只对实际受伤回合取条件均值；null 单列为“未受伤”计数，不转换为第 0 步或最大步。", ""]
    text += _table(targets, identity_columns + [("damage", "逐目标伤害均值（有效局）"),
                   ("uninjured", "未受伤局/有记录局"), ("first", "受伤条件下首次受伤步（受伤局）")]) + [""]
    text += ["Q 统计使用实际贪心动作的在线价值，先在各回合内汇总，再对回合等权平均。"
             "q_tot_std 列是每局时间标准差的均值，不是总体 Q 标准差或跨训练种子的标准误。"
             "规则锚点没有模型价值，显示“不适用”；模型缺记录显示“未记录”，不代填为 0。", ""]
    text += _table(q_values, identity_columns + [("q_tot_mean", "Qtot 回合均值"),
                   ("q_tot_std", "每局 Qtot 时间标准差"), ("q_i_mean", "Qi 回合均值")]) + [""]
    if analysis:
        text += ["已完成组的直接观察（不作因果或策略退化判定）：", ""]
        for (method, seed, phase), groups in sorted(analysis.items()):
            observed = [row for group in groups for row in group["episodes"]]
            notes = []
            causes = [row.get("red_deaths_by_cause") for row in observed]
            if all(isinstance(value, dict) and all(_numeric(value.get(key)) for key, _ in red_causes) for value in causes):
                counts = {key: sum(value[key] for value in causes) for key, _ in red_causes}
                total = sum(counts.values())
                if total:
                    maximum = max(counts.values())
                    leading = "、".join(label for key, label in red_causes if counts[key] == maximum)
                    notes.append(f"合计 {total:g} 次红方死亡中，最多的原生归因为{leading}（各 {maximum:g} 次，{maximum/total:.1%}）")
                else:
                    notes.append("未记录红方死亡")
            collision = [group["collision_fraction"] for group in groups if group["collision_fraction"] is not None]
            if collision:
                notes.append(f"各配置有同队碰撞的回合比例为 {min(collision):.1%}–{max(collision):.1%}")
            if phase != "anchor":
                q = [(group["config"], statistics.mean(row["q_tot_mean"] for row in group["episodes"]))
                     for group in groups if all(_numeric(row.get("q_tot_mean")) for row in group["episodes"])]
                if len(q) >= 2:
                    notes.append("Qtot 均值随配置的记录为 " + "、".join(f"{config}: {value:.4f}" for config, value in q))
            if notes:
                text += [f"{method}/{seed} 的 {len(groups)} 个完整配置：" + "；".join(notes) + "。", ""]
        text += ["上述死亡次数按已完成配置的原生事件累计；其余指标保持逐配置汇总。"
                 "Q 的量级变化本身不能证明归一化失效，碰撞、动作分布或距离变化也不能单独说明防守效果；"
                 "需结合相同配置的 D/rho 与保留轨迹解释。阶段三单个训练种子的结果不作为多种子稳定性证据。", ""]
    else:
        text += ["尚无完整的 300 局诊断组，暂不形成完成组分析。", ""]
    return text


def _range(values):
    values = [value for value in values if _numeric(value)]
    return f"{min(values):.5g}–{max(values):.5g}" if values else "—"


def _e0_summary(learning, progress):
    learned = _groups(learning, ("method", "seed"))
    statuses = {(row["method"], row["seed"]): row for row in progress}
    rows = []
    for method in ALL_METHODS:
        keys = sorted({key for key in (*learned.keys(), *statuses.keys()) if key[0] == method}) or [(method, 0)]
        for key in keys:
            records = learned.get(key, [])
            last = records[-1] if records else {}
            state = statuses.get(key, {})
            measured_steps = max(int(last.get("t_env", 0)), int(state.get("t_env", 0)))
            losses = [row["loss"] for row in records if _numeric(row.get("loss"))]
            gradients = [row["grad_norm"] for row in records if _numeric(row.get("grad_norm"))]
            rows.append(dict(method=method, seed=key[1], env=state.get("phase", "rel_overgen" if method == "dcg" else "ff"),
                             steps=measured_steps, quota=20000, updates=last.get("updates", state.get("updates", 0)),
                             status=state.get("status", "有学习记录，待状态更新" if records else "未启动"),
                             loss=_number(losses[-1] if losses else None, 5), loss_range=_range(losses),
                             grad=_number(gradients[-1] if gradients else None, 5), grad_range=_range(gradients),
                             recorded_at=max(last.get("recorded_at", ""), state.get("recorded_at", "")) or "—"))
    return rows


def _plot_e0(plt, output, learning):
    from matplotlib.ticker import MaxNLocator
    if not any(_numeric(row.get("loss")) or _numeric(row.get("grad_norm")) for row in learning):
        return None
    grid = -(-len(ALL_METHODS) // 3)
    fig, axes = plt.subplots(grid, 3, figsize=(14, 4 * grid), constrained_layout=True)
    for axis in axes.flat[len(ALL_METHODS):]:
        axis.set_axis_off()
    for axis, method in zip(axes.flat, ALL_METHODS):
        records = [row for row in learning if row["method"] == method]
        gradient_axis = axis.twinx()
        for (_, seed), values in _groups(records, ("method", "seed")).items():
            values = sorted(values, key=lambda row: row["t_env"])
            for metric in (("loss", "im_loss") if method == "refil" else ("loss",)):
                points = [row for row in values if _numeric(row.get(metric))]
                if points:
                    axis.plot([row["t_env"] for row in points], [row[metric] for row in points],
                              color=COLORS[method], linestyle="--" if metric == "im_loss" else "-",
                              label=f"{metric}, seed {seed}", linewidth=1.1)
            points = [row for row in values if _numeric(row.get("grad_norm"))]
            if points:
                gradient_axis.plot([row["t_env"] for row in points], [row["grad_norm"] for row in points],
                                   color="#8a9198", alpha=.65, linestyle=":", label=f"gradient norm, seed {seed}")
        environment = "rel_overgen" if method == "dcg" else "ff"
        axis.set(title=f"{method} | {environment}", xlabel="Native environment physical steps", ylabel="Batch loss")
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        gradient_axis.set_ylabel("Gradient norm", color="#737c85")
        axis.grid(alpha=.18)
        handles, labels = axis.get_legend_handles_labels()
        gradient_handles, gradient_labels = gradient_axis.get_legend_handles_labels()
        if handles or gradient_handles:
            axis.legend(handles + gradient_handles, labels + gradient_labels, fontsize=7)
        else:
            axis.text(.5, .5, "No learner update recorded", ha="center", transform=axis.transAxes)
    path = output / "figures" / "E0.png"
    _save_figure(fig, path)
    plt.close(fig)
    return path


def _config_returns(rows, anchors, configs):
    grouped = _groups(rows, ("method", "seed"))
    per_method = defaultdict(lambda: defaultdict(list))
    for (method, seed), group in grouped.items():
        by_config = defaultdict(dict)
        for row in group:
            by_config[config_key(row["config"])][row["episode_seed"]] = row
        for config in configs:
            records = by_config[config]
            if set(records) != set(range(9000, 9300)):
                continue
            per_method[method][config].append(-statistics.mean(row["D"] for row in records.values()))
    anchors_r = {}
    for policy in ("random", "rule_nv1"):
        anchors_r[policy] = {config: -anchors[(policy, config)]["D"]
                             for config in configs if anchors.get((policy, config), {}).get("complete")}
    return per_method, anchors_r


def _plot_scale_line(plt, output, rows, anchors, configs, *, stem, title, xlabel, methods=None,
                     include_random=True):
    methods = METHODS if methods is None else methods
    per_method, anchors_r = _config_returns(rows, anchors, configs)
    if not per_method and not any(anchors_r.values()):
        return None
    fig, axis = plt.subplots(1, 1, figsize=(7.2, 3.8), constrained_layout=True)
    xs = list(range(len(configs)))
    for method in methods:
        values = per_method.get(method, {})
        points = []
        for index, config in enumerate(configs):
            mean, std = _mean_std(values.get(config, []))
            if mean is not None:
                points.append((index, mean, std))
        if points:
            axis.errorbar([p[0] for p in points], [p[1] for p in points],
                          yerr=[p[2] for p in points], marker="o", color=COLORS[method],
                          linewidth=2.0, capsize=2.5, label=LABELS[method])
    for policy, style in _anchor_plot_styles(include_random):
        measured = [(index, anchors_r[policy][config]) for index, config in enumerate(configs)
                    if config in anchors_r[policy]]
        if measured:
            axis.plot([x for x, _ in measured], [y for _, y in measured], linestyle=style,
                      marker="s", color=COLORS[policy], label=LABELS[policy])
    axis.set(title=title, xlabel=xlabel, ylabel="Red episode return R = −D",
             xticks=xs, xticklabels=[config_label(config) for config in configs])
    axis.tick_params(axis="x", labelrotation=30, labelsize=8)
    axis.grid(alpha=.2)
    if axis.lines:
        axis.legend(fontsize=8)
    path = output / "figures" / f"{stem}.png"
    _save_figure(fig, path)
    plt.close(fig)
    return path


def _plot_target_k(plt, output, rows, anchors, methods=None, include_random=True):
    methods = METHODS if methods is None else methods
    configs = tuple((n, n, k) for n in TARGET_SCALE_NS for k in TARGET_K_VALUES)
    per_method, anchors_r = _config_returns(rows, anchors, configs)
    if not per_method and not any(anchors_r.values()):
        return None
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.6), sharex=True, sharey=True,
                             constrained_layout=True)
    for axis, n in zip(axes.flat, TARGET_SCALE_NS):
        for method in methods:
            points = []
            for k in TARGET_K_VALUES:
                mean, std = _mean_std(per_method.get(method, {}).get((n, n, k), []))
                if mean is not None:
                    points.append((k, mean, std))
            if points:
                axis.errorbar([p[0] for p in points], [p[1] for p in points],
                              yerr=[p[2] for p in points], marker="o",
                              color=COLORS[method], linewidth=2.0, capsize=2.5,
                              label=LABELS[method])
        for policy, style in _anchor_plot_styles(include_random):
            measured = [(k, anchors_r[policy][(n, n, k)]) for k in TARGET_K_VALUES
                        if (n, n, k) in anchors_r[policy]]
            if measured:
                axis.plot([x for x, _ in measured], [y for _, y in measured],
                          linestyle=style, marker="s", color=COLORS[policy],
                          label=LABELS[policy])
        axis.set(title=f"{n}v{n}", xticks=list(TARGET_K_VALUES))
        axis.grid(alpha=.2)
        if axis is axes.flat[0] and axis.lines:
            axis.legend(fontsize=7, loc="lower left")
    fig.supxlabel("K")
    fig.supylabel("Red episode return R = −D")
    path = output / "figures" / "scale_targets.png"
    _save_figure(fig, path)
    plt.close(fig)
    return path


def _plot_final(plt, output, rows, anchors, methods=None, include_random=True):
    methods = METHODS if methods is None else methods
    rows = [row for row in rows if row["phase"] == "final_eval" and row["method"] in methods]
    return [
        _plot_scale_line(plt, output, rows, anchors, EQUAL_SCALE_CONFIGS,
                         stem="scale_equal", title="Equal-scale 1:1", xlabel="Roster",
                         methods=methods, include_random=include_random),
        _plot_scale_line(plt, output, rows, anchors, RATIO_CONFIGS,
                         stem="scale_ratio", title="1:2 red:blue", xlabel="Roster",
                         methods=methods, include_random=include_random),
        _plot_target_k(plt, output, rows, anchors, methods=methods,
                       include_random=include_random),
    ]


def _official_seed_t_env(output, progress, method, seed):
    directory = run_directory(output, method, seed)
    if not training_finished(directory, (progress or {}).get((method, seed))):
        return None
    return official_eval_t_env(directory)


def _depth_sweep_groups(episodes, methods=CYCLE_SERIES_METHODS, progress=None, output=None):
    """(method, seed, depth, config) -> D list.

    Only the weights now on disk count. R=4 may reuse 1:1 final_eval from that
    same ``best@t_env``, and only if the seed already has a depth sweep.
    """
    equal = set(EQUAL_SCALE_CONFIGS)
    output = Path(DEFAULT_OUTPUT if output is None else output)
    progress = progress or {}
    official = {}
    has_sweep = set()
    for row in episodes:
        if row.get("method") not in methods:
            continue
        key = (row["method"], int(row["seed"]))
        if key not in official:
            official[key] = _official_seed_t_env(output, progress, key[0], key[1])
        if row.get("phase") != "depth_eval" or official[key] is None:
            continue
        if parse_checkpoint_t_env(row.get("checkpoint")) == official[key]:
            has_sweep.add(key)
    grouped = defaultdict(list)
    for row in episodes:
        if row.get("method") not in methods or row.get("phase") != "depth_eval":
            continue
        if not _numeric(row.get("D")):
            continue
        cfg = config_key(row["config"])
        if cfg not in equal:
            continue
        key = (row["method"], int(row["seed"]))
        depth = parse_sweep_depth(row.get("checkpoint"))
        t_env = parse_checkpoint_t_env(row.get("checkpoint"))
        if depth is None or t_env is None or t_env != official.get(key):
            continue
        grouped[(row["method"], int(row["seed"]), int(depth), cfg)].append(float(row["D"]))
    finals = defaultdict(list)
    for row in episodes:
        if row.get("phase") != "final_eval" or row.get("method") not in methods:
            continue
        if not _numeric(row.get("D")):
            continue
        cfg = config_key(row["config"])
        if cfg not in equal:
            continue
        key = (row["method"], int(row["seed"]))
        if key not in has_sweep:
            continue
        if parse_checkpoint_t_env(row.get("checkpoint")) != official.get(key):
            continue
        finals[(row["method"], int(row["seed"]), cfg)].append(float(row["D"]))
    for (method, seed, cfg), values in finals.items():
        key = (method, seed, 4, cfg)
        if len(grouped[key]) < FINAL_EPISODES_PER_CONFIG:
            grouped[key] = list(values)
    return grouped


def _depth_cell_scores(episodes, methods=CYCLE_SERIES_METHODS, progress=None, output=None):
    """Mean D per method, cycle depth, and 1:1 config. A cell needs 300 episodes."""
    grouped = _depth_sweep_groups(episodes, methods, progress=progress, output=output)
    per_seed = defaultdict(list)
    for (method, _seed, depth, cfg), values in grouped.items():
        if len(values) < FINAL_EPISODES_PER_CONFIG:
            continue
        per_seed[(method, int(depth), cfg)].append(statistics.mean(values))
    cells = {}
    for (method, depth, cfg), means in per_seed.items():
        cells.setdefault(method, {}).setdefault(depth, {})[cfg] = statistics.mean(means)
    return cells


def _best_cycle_depth_table(by_depth):
    """One row per 1:1 scale: the R with the lowest mean D (ties keep the smaller R)."""
    rows = []
    for cfg in EQUAL_SCALE_CONFIGS:
        ranked = [(by_depth.get(depth, {}).get(cfg), depth) for depth in DEPTH_SWEEP_DEPTHS]
        ranked = [(value, depth) for value, depth in ranked if value is not None]
        if not ranked:
            rows.append(dict(scale=config_label(cfg), best="—", D="—", tied=""))
            continue
        best_d = min(value for value, _ in ranked)
        tied = [depth for value, depth in ranked if abs(value - best_d) <= 1e-4]
        extra = "、".join(str(depth) for depth in tied[1:])
        rows.append(dict(scale=config_label(cfg), best=str(tied[0]), D=_number(best_d, 4),
                         tied=extra or "—"))
    return rows


def _depth_sweep_scores(episodes, methods=CYCLE_SERIES_METHODS, progress=None, output=None):
    """Mean 1:1 D per method and cycle depth. A config enters once it has 300 episodes."""
    cells = _depth_cell_scores(episodes, methods, progress=progress, output=output)
    scores = {}
    for method, by_depth in cells.items():
        scores[method] = {}
        for depth, by_cfg in by_depth.items():
            config_means = [by_cfg[cfg] for cfg in EQUAL_SCALE_CONFIGS if cfg in by_cfg]
            mean, std = _mean_std(config_means)
            if mean is None:
                continue
            scores[method][depth] = dict(D=mean, std=std, n=len(config_means),
                                         return_mean=-mean)
    return scores


def _depth_progress_lines(episodes, methods=CYCLE_SERIES_METHODS, progress=None, output=None):
    grouped = _depth_sweep_groups(episodes, methods, progress=progress, output=output)
    quota = FINAL_EPISODES_PER_CONFIG
    n_cfg = len(EQUAL_SCALE_CONFIGS)
    total = depth_eval_total()
    lines = []
    for method in methods:
        by_depth = defaultdict(lambda: defaultdict(int))
        for (name, _seed, depth, cfg), values in grouped.items():
            if name != method:
                continue
            by_depth[depth][cfg] = max(by_depth[depth][cfg], len(values))
        if not by_depth:
            continue
        done = 0
        bits = []
        for depth in DEPTH_SWEEP_DEPTHS:
            complete = sum(1 for cfg in EQUAL_SCALE_CONFIGS if by_depth[depth].get(cfg, 0) >= quota)
            done += sum(min(by_depth[depth].get(cfg, 0), quota) for cfg in EQUAL_SCALE_CONFIGS)
            bits.append(f"$R={depth}$ {complete}/{n_cfg}")
        lines.append(f"{LABELS[method]} 扫描 {done}/{total} 局；已满 300 局的 1:1 配置："
                     + "，".join(bits) + "。")
    return lines


def _v5_depth_section_lines(output, episodes, figure, scale_figure=None):
    lines = ["## 循环轮数（等规模 1:1）", "",
             r"循环 Transformer 臂（REFIL-B / F / K10）在 best 上固定贪心，只跑等规模 1:1 "
             r"（10/15/20/25/30/40v K2），比较 $R=1,2,3,4,5,6$。每配置 300 局。"
             r"$R=4$ 复用终评已有的 1:1 局。下图横轴是编队规模、一条线一个循环轮数；"
             r"下表行是 $R$、列是规模。数字是 300 局平均 $D$，越低红方越好。"
             "若规模越大越需要加深循环，大 $N$ 上更深的 $R$ 应明显低于浅的 $R$。", ""]
    progress = _depth_progress_lines(episodes, CYCLE_SERIES_METHODS)
    if progress:
        lines.extend(progress + [""])
    if scale_figure:
        lines += [f"![循环轮数×规模]({scale_figure.relative_to(output).as_posix()})", ""]
    elif figure:
        lines += [f"![循环轮数]({figure.relative_to(output).as_posix()})", ""]
    cells = _depth_cell_scores(episodes, CYCLE_SERIES_METHODS)
    for method in CYCLE_SERIES_METHODS:
        if method not in cells:
            continue
        scale_table = []
        for depth in DEPTH_SWEEP_DEPTHS:
            row = dict(R=f"$R={depth}$")
            for cfg in EQUAL_SCALE_CONFIGS:
                value = cells[method].get(depth, {}).get(cfg)
                row[config_label(cfg)] = _number(value, 4) if value is not None else "—"
            scale_table.append(row)
        lines += [f"{LABELS[method]}：行是循环轮数，列是 1:1 规模，格子为平均 $D$。", ""]
        lines += _table(scale_table, [("R", "$R$"),
                                      *[(config_label(cfg), config_label(cfg))
                                        for cfg in EQUAL_SCALE_CONFIGS]]) + [""]
        best_table = _best_cycle_depth_table(cells[method])
        if best_table:
            lines += [f"{LABELS[method]} 各规模上 $D$ 最低的循环轮数；并列时取更小的 $R$。", ""]
            lines += _table(best_table, [("scale", "规模"), ("best", "最优 $R$"),
                                         ("D", "该格 $D$"), ("tied", "并列")]) + [""]
    if not cells and not progress:
        lines += ["尚无 1:1 循环轮数记录。", ""]
    return lines


def _plot_cycle_depth(plt, output, episodes, methods=CYCLE_SERIES_METHODS, progress=None):
    scores = _depth_sweep_scores(episodes, methods, progress=progress, output=output)
    series = [method for method in methods if scores.get(method)]
    if not series:
        return None
    depths = list(DEPTH_SWEEP_DEPTHS)
    fig, axis = plt.subplots(1, 1, figsize=(7.4, 3.8), constrained_layout=True)
    width = 0.8 / max(len(series), 1)
    xs = list(range(len(depths)))
    for index, method in enumerate(series):
        offset = (index - (len(series) - 1) / 2) * width
        heights, errors, positions = [], [], []
        for x, depth in zip(xs, depths):
            item = scores[method].get(depth)
            if not item:
                continue
            positions.append(x + offset)
            heights.append(item["return_mean"])
            errors.append(0.0 if item["std"] is None else item["std"])
        if positions:
            axis.bar(positions, heights, width=width * 0.92, yerr=errors, capsize=2.5,
                     color=COLORS[method], label=LABELS[method], zorder=2)
    axis.set(title="Equal-scale 1:1 vs cycle depth", xlabel="Cycle rounds R",
             ylabel="Red episode return R = −D", xticks=xs,
             xticklabels=[str(depth) for depth in depths])
    axis.axhline(0, color="#888", linewidth=0.8, zorder=1)
    axis.grid(axis="y", alpha=.2)
    if series:
        axis.legend(fontsize=8)
    path = output / "figures" / "cycle_depth.png"
    _save_figure(fig, path)
    plt.close(fig)
    return path


def _plot_cycle_depth_scale(plt, output, episodes, methods=CYCLE_SERIES_METHODS, progress=None):
    """One line per R: D versus 1:1 roster size. Lower D is better for Red."""
    cells = _depth_cell_scores(episodes, methods, progress=progress, output=output)
    series = [method for method in methods if cells.get(method)]
    if not series:
        return None
    palette = ("#4c78a8", "#f58518", "#54a24b", "#e45756", "#b279a2", "#8c564b")
    ns = [cfg[0] for cfg in EQUAL_SCALE_CONFIGS]
    fig, axes = plt.subplots(1, len(series), figsize=(7.4, 3.8), squeeze=False,
                             constrained_layout=True)
    for axis, method in zip(axes[0], series):
        for index, depth in enumerate(DEPTH_SWEEP_DEPTHS):
            xs, ys = [], []
            for cfg, size in zip(EQUAL_SCALE_CONFIGS, ns):
                value = cells[method].get(depth, {}).get(cfg)
                if value is not None:
                    xs.append(size)
                    ys.append(value)
            if xs:
                axis.plot(xs, ys, marker="o", linewidth=2.0,
                          color=palette[index % len(palette)], label=f"R={depth}")
        axis.set(title=f"{LABELS[method]}: D vs scale by R",
                 xlabel="N (NvN K2)", ylabel="D (lower is better)", xticks=ns)
        axis.grid(alpha=.2)
        if axis.lines:
            axis.legend(fontsize=8, ncol=3, loc="upper left")
    path = output / "figures" / "cycle_depth_scale.png"
    _save_figure(fig, path)
    plt.close(fig)
    return path


def _behavior_rates(episodes):
    intercept, deliver, friendly = [], [], []
    for row in episodes:
        red, blue, _ = config_key(row["config"])
        blue_deaths = row.get("blue_deaths_by_cause") if isinstance(row.get("blue_deaths_by_cause"), dict) else None
        red_deaths = row.get("red_deaths_by_cause") if isinstance(row.get("red_deaths_by_cause"), dict) else None
        if not blue_deaths or not red_deaths or blue <= 0 or red <= 0:
            continue
        intercept.append(blue_deaths.get("intercepted", 0) / blue)
        deliver.append(blue_deaths.get("self_destruct", 0) / blue)
        friendly.append(red_deaths.get("friendly_collision", 0) / red)
    return dict(intercept=_mean_std(intercept)[0], deliver=_mean_std(deliver)[0],
                friendly=_mean_std(friendly)[0])


def _id_behavior(final_rows, episodes, methods=None):
    methods = METHODS if methods is None else methods
    result = {}
    pool = [row for row in final_rows if config_key(row["config"]) in set(VALIDATION_CONFIGS)]
    for method in methods:
        result[method] = [_behavior_rates(group) for _, group in sorted(_groups(
            [row for row in pool if row["method"] == method], ("seed",)).items())]
    for method in ("random", "rule_nv1"):
        rows = [row for row in episodes if row["method"] == method and row["phase"] == "anchor"
                and config_key(row["config"]) in set(VALIDATION_CONFIGS)]
        result[method] = [_behavior_rates(rows)] if rows else []
    return {key: [row for row in rows if all(row.get(field) is not None
                  for field in ("intercept", "deliver", "friendly"))] for key, rows in result.items()}


def _plot_behavior(plt, output, behavior, methods=None, include_random=True):
    methods = METHODS if methods is None else methods
    order = (("random",) if include_random else ()) + methods + ("rule_nv1",)
    if not any(behavior.get(method) for method in order):
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), constrained_layout=True)
    for axis, field, title, ylabel in (
        (axes[0], "intercept", "Blue intercept rate", "Intercepted blues / N_B"),
        (axes[1], "friendly", "Red friendly-collision deaths", "Friendly-collision deaths / N_R"),
    ):
        xs, ys, err, colors, names = [], [], [], [], []
        for index, method in enumerate(order):
            mean, std = _mean_std([row[field] for row in behavior.get(method, [])])
            if mean is None:
                continue
            xs.append(index)
            ys.append(mean)
            err.append(0.0 if method in ("random", "rule_nv1") else std)
            colors.append(COLORS[method])
            names.append(LABELS[method])
        if not xs:
            axis.text(.5, .5, "尚无诊断记录", ha="center", va="center", transform=axis.transAxes)
            continue
        axis.bar(xs, ys, color=colors, width=.72, yerr=err, capsize=3, error_kw={"linewidth": 1})
        axis.set(title=title, ylabel=ylabel, xticks=xs, xticklabels=names)
        axis.grid(axis="y", alpha=.2)
    path = output / "figures" / "behavior.png"
    _save_figure(fig, path)
    plt.close(fig)
    return path


def _completed_benchmarks(rows):
    latest = {(row["run"], row["method"], row["seed"]): row for row in rows}
    return [row for row in latest.values() if row.get("status") in ("complete", "completed")
            and row.get("t_env") == 20000 and _numeric(row.get("steps_per_second"))
            and row["steps_per_second"] > 0]


def _benchmark_summary(rows):
    grouped = defaultdict(list)
    for row in _completed_benchmarks(rows):
        if (not _numeric(row.get("current_steps_per_second")) or row["current_steps_per_second"] <= 0
                or row.get("implementation_revision") != BENCHMARK_REVISION or row.get("measurement_steps", 0) <= 0):
            continue
        variant = row["run"].rsplit("_w", 1)[0]
        grouped[(variant, row["implementation_revision"], row["batch_size_run"], row["concurrency"])].append(row)
    summary = []
    for (variant, revision, workers, concurrency), records in sorted(grouped.items()):
        speeds = [row["current_steps_per_second"] for row in records]
        mean = statistics.mean(speeds)
        data = dict(variant=variant, revision=revision, workers=workers, concurrency=concurrency, n=len(records),
                    mean=mean, minimum=min(speeds), maximum=max(speeds), total_rate=sum(speeds),
                    measurement_start_range=_range([row["measurement_start_t_env"] for row in records]),
                    measurement_steps_range=_range([row["measurement_steps"] for row in records]),
                    rate_range=_range(speeds),
                    time_100k=f"{100000/mean/60:.2f}（{100000/max(speeds)/60:.2f}–{100000/min(speeds)/60:.2f}）",
                    time_1m=f"{1000000/mean/3600:.2f}（{1000000/max(speeds)/3600:.2f}–{1000000/min(speeds)/3600:.2f}）")
        for field, name in (("current_peak_private_gib", "private"), ("current_peak_rss_gib", "rss"), ("current_peak_cuda_gib", "cuda")):
            values = [row[field] for row in records if _numeric(row.get(field))]
            data[f"{name}_range"] = _range(values)
            data[f"{name}_sum"] = sum(values) if len(values) == len(records) else None
        overhead = [row["wall_seconds"] - row["training_seconds"] for row in records
                    if _numeric(row.get("wall_seconds")) and _numeric(row.get("training_seconds"))]
        data["other_seconds"] = _range(overhead)
        summary.append(data)
    revisions = sorted({row["revision"] for row in summary})
    for row in summary:
        row["revision_tag"] = f"R{revisions.index(row['revision'])+1}"
    return summary


def _timing_summary(rows):
    by_run = defaultdict(list)
    unique = {(row["run"], row.get("implementation_revision"), row["method"], row["seed"], row["t_env"]): row for row in rows}
    for row in unique.values():
        if row["run"].startswith("benchmark") and all(_numeric(row.get(key)) for key in
                ("collect_seconds", "learn_seconds", "collect_physical_steps")):
            by_run[(row["run"], row.get("implementation_revision") or "历史记录未标识版本", row["method"], row["seed"])].append(row)
    summary = []
    for (run, revision, method, seed), records in sorted(by_run.items()):
        collect = sum(row["collect_seconds"] for row in records)
        learn = sum(row["learn_seconds"] for row in records)
        total = collect + learn
        summary.append(dict(run=run, revision=revision, method=method, seed=seed, batches=len(records),
                            physical_steps=sum(row["collect_physical_steps"] for row in records),
                            collect_seconds=collect, learn_seconds=learn,
                            collect_fraction=f"{100*collect/total:.1f}%" if total > 0 else "—",
                            learn_fraction=f"{100*learn/total:.1f}%" if total > 0 else "—"))
    return summary


def _resource_pressure(rows):
    """Describe recorded four-REFIL pressure without collecting new samples."""
    fields = ("process_tree_private_gib", "process_tree_rss_gib", "cuda_reserved_gib",
              "cuda_peak_allocated_gib", "system_available_gib", "steps_per_second", "t_env")
    by_run = defaultdict(list)
    for row in rows:
        if row["run"].startswith("benchmark") and row["run"].endswith("_c4") and row["method"] == "refil":
            by_run[row["run"]].append(row)
    text = []
    for run, records in sorted(by_run.items()):
        latest, selected, ranking = {}, None, None
        samples = defaultdict(list)
        for row in sorted(records, key=lambda value: value["recorded_at"]):
            latest[row["seed"]] = row
            if row.get("status") == "training" and all(_numeric(row.get(field)) for field in fields):
                samples[row["seed"]].append(row)
            current = [latest.get(seed) for seed in range(4)]
            if not all(value and value.get("status") == "training" and value["t_env"] >= 10000
                       and all(_numeric(value.get(field)) for field in fields) for value in current):
                continue
            times = [datetime.fromisoformat(value["recorded_at"]) for value in current]
            if (max(times) - min(times)).total_seconds() > 60:
                continue
            reserved = sum(value["cuda_reserved_gib"] for value in current)
            candidate = (reserved, -min(value["system_available_gib"] for value in current))
            if ranking is None or candidate > ranking:
                selected, ranking = list(current), candidate
        if selected is None:
            continue
        table = [dict(seed=row["seed"], t_env=row["t_env"], recorded_at=row["recorded_at"],
                      speed=row["steps_per_second"], private=row["process_tree_private_gib"],
                      rss=row["process_tree_rss_gib"], reserved=row["cuda_reserved_gib"],
                      allocated_peak=row["cuda_peak_allocated_gib"], available=row["system_available_gib"],
                      difference=row["process_tree_private_gib"] - row["cuda_reserved_gib"]) for row in selected]
        reserved = sum(row["reserved"] for row in table)
        text += [f"{run} 的后半程资源压力窗口：仅取四任务均超过 10000 步、状态为 training、"
                 "且最后记录相距不超过 60 秒的样本；选择 CUDA reserved 合计最高的窗口，同值时取系统可用内存最低者。"
                 "这是相近时间的四条进度记录，并非硬件同步快照，也不是整个运行的时间平均。", ""]
        text += _table(table, [("seed", "任务/种子"), ("t_env", "已观测步"), ("recorded_at", "采样时间"),
                               ("speed", "记录步/秒"), ("private", "私有提交 GiB"), ("rss", "RSS GiB"),
                               ("reserved", "CUDA reserved GiB"), ("allocated_peak", "CUDA allocated 峰值 GiB"),
                               ("difference", "private−reserved GiB"), ("available", "系统可用 RAM GiB")]) + [""]
        text += [f"该窗口 CUDA reserved 合计 {reserved:.3f} GiB，本卡总显存约 15.92 GiB；"
                 f"各任务 CUDA allocated 峰值为 {_range(row['allocated_peak'] for row in table)} GiB，"
                 f"记录训练速率为 {_range(row['speed'] for row in table)} 步/秒，系统可用 RAM 最低 "
                 f"{min(row['available'] for row in table):.3f} GiB。只查看 allocated 峰值会遗漏 allocator 保留的显存。"
                 f"同一窗口每任务 private−reserved 为 {_range(row['difference'] for row in table)} GiB；"
                 "此差值仅作观测对照，不是可直接划分 CPU/GPU 物理占用的内存账目。", ""]
        stable = None
        for seed, values in samples.items():
            start = None
            for row in values:
                if start is None or row["cuda_reserved_gib"] != start["cuda_reserved_gib"] or row["t_env"] < start["t_env"]:
                    start = row
                span = row["t_env"] - start["t_env"]
                if span > 0 and (stable is None or span > stable[0]):
                    stable = (span, seed, start, row)
        if stable:
            _, seed, first, last = stable
            text += [f"另取本运行 reserved 连续保持同值、覆盖步数最长的一段：种子 {seed} 从 "
                     f"{first['t_env']:,} 到 {last['t_env']:,} 步，reserved 为 {first['cuda_reserved_gib']:.3f} GiB；"
                     f"私有提交 {first['process_tree_private_gib']:.3f}→{last['process_tree_private_gib']:.3f} GiB"
                     f"（变化 {last['process_tree_private_gib']-first['process_tree_private_gib']:+.3f}），"
                     f"RSS {first['process_tree_rss_gib']:.3f}→{last['process_tree_rss_gib']:.3f} GiB。"
                     "RSS 与私有提交的变化含义不同，RSS 增长本身不能证明新增提交量泄漏。", ""]
        text += ["结合完整回合长度随批次变化、allocator 保留量与私有提交的共同变化，"
                 "当前证据支持“动态张量形状造成 CUDA 缓存/碎片以及 Windows 驱动相关提交压力”的推断。"
                 "进度 CSV 没有驱动内部内存归因，不能据此宣称已证明 CPU 泄漏，"
                 "也不能仅凭低可用内存确定各部分对吞吐下降的贡献。未增加内存探测或改变矩阵实现。", ""]
    if text:
        text += ["回放容量的代码侧估算：当前 HAD scheme 每个存储状态的裸张量为 7508 字节；"
                 "efficient_store 按真实 filled 长度 detach 后 clone。20000 物理步加约 500 个最终观察约为 "
                 "146.8 MiB/任务，另有每回合 t_added 与 Python 对象开销，不能用这一短跑张量量解释数 GiB 提交差异。"
                 "buffer 5000 回合若全按最长 101 状态计算，裸张量上限约 3.531 GiB/任务；实际回合较短时更小。"
                 "正式长训练会继续填充回放，不能把 20000 步时的资源需求直接线性缩小或视为长期上限。", ""]
    return text


def _plot_execution(plt, output, progress, benchmarks, verification):
    if not (progress or benchmarks or verification):
        return None
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    measured = [row for row in progress if row.get("run") == FORMAL_RUN and _numeric(row.get("t_env"))]
    if measured:
        labels = [f"{row['method']}/{row['seed']}" for row in measured]
        axes[0].barh(labels, [row["t_env"] for row in measured], color="#277ca5")
        axes[0].set_xlabel("Recorded training physical steps")
    else:
        axes[0].text(.5, .5, "Formal HAD training not started", ha="center", transform=axes[0].transAxes, fontsize=9)
    summaries = _benchmark_summary(benchmarks)
    speeds = [(f"{row['variant'].removeprefix('benchmark_')} {row['revision_tag']}\nw={row['workers']}, jobs={row['concurrency']}, n={row['n']}",
               row["mean"], row["minimum"], row["maximum"]) for row in summaries]
    if speeds:
        means = [mean for _, mean, _, _ in speeds]
        spread = [[mean-low for _, mean, low, _ in speeds], [high-mean for _, mean, _, high in speeds]]
        axes[1].barh([label for label, _, _, _ in speeds], means, xerr=spread, capsize=3, color="#4b9888")
        axes[1].set_xlabel("Measured physical steps / second")
    else:
        axes[1].text(.5, .5, "No completed current-revision task yet", ha="center", transform=axes[1].transAxes, fontsize=8)
    counts = Counter("failed\n(historical interrupt)"
                     if row.get("id") == "stop.execution_tool_interrupt" and row.get("status") == "failed"
                     else str(row.get("status", "unrecorded")) for row in verification)
    if counts:
        axes[2].bar(range(len(counts)), list(counts.values()), color="#577590")
        axes[2].set_xticks(range(len(counts)), list(counts), rotation=25, ha="right")
        axes[2].set_ylabel("Recorded verification items")
    else:
        axes[2].text(.5, .5, "Verification not recorded", ha="center", transform=axes[2].transAxes)
    for axis, title in zip(axes, ("Formal HAD progress", "Throughput: mean and min-max", "Verification status")):
        axis.set_title(title)
        axis.grid(axis="x" if axis != axes[2] else "y", alpha=.15)
    path = output / "figures" / "execution.png"
    _save_figure(fig, path)
    plt.close(fig)
    return path


def _side_metric_note(episodes, anchors):
    """One-line extras for arms kept off the formal comparison."""
    descriptions = {
        "b0_qmix": "有序展平 QMIX，槽位按训练池定义，只评池内 4 配置",
        "gnn_qmix": "一层图卷积接到同一套 QMIX",
    }
    parts = []
    for method in SIDE_METHODS:
        points = [row for row in _validation_points(episodes, anchors)
                  if row["method"] == method and row["eval_point"] == 50]
        if not points:
            continue
        mean, _ = _mean_std([-row["D"] for row in points])
        pool = [row for row in episodes if row["method"] == method and row.get("phase") == "final_eval"
                and config_key(row["config"]) in set(VALIDATION_CONFIGS)]
        rates = _behavior_rates(pool) if pool else {}
        intercept = rates.get("intercept")
        extra = f"，拦截率 {intercept:.2f}" if intercept is not None else ""
        parts.append(f"{LABELS[method]}（{descriptions.get(method, method)}）"
                     f"seed 0 池内验证回报 ${mean:.2f}${extra}")
    if not parts:
        return None
    return "另有未列入对照的备选结果：" + "；".join(parts) + "。不当正式基线，原始记录仍在 CSV。"


def _plus_minus(values, digits=3):
    values = [float(value) for value in values if _numeric(value)]
    if not values:
        return "—"
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{statistics.mean(values):.{digits}f}±{statistics.stdev(values):.{digits}f}"


def _final_seed_scores(rows, anchors, methods=None):
    methods = METHODS if methods is None else methods
    scores = defaultdict(lambda: defaultdict(dict))
    for (method, seed), values in _groups(rows, ("method", "seed")).items():
        if method not in methods:
            continue
        grouped = defaultdict(dict)
        for row in values:
            grouped[config_key(row["config"])][row["episode_seed"]] = row
        for config in final_configs(method):
            records = grouped[config]
            if set(records) != set(range(9000, 9300)):
                continue
            D = statistics.mean(row["D"] for row in records.values())
            rho = statistics.mean(row["rho"] for row in records.values())
            scores[method][config][seed] = dict(D=D, rho=rho, NDS=_nds(rho, config, anchors))
    return scores


def _refresh_report(output=DEFAULT_OUTPUT, *, run=FORMAL_RUN, report_stream=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    episodes = unique_episodes(read_records(output, "episodes", run=run))
    progress = {(method, seed): row for (method, recorded_run, seed), row in
                read_latest(output, "progress", run=run).items() if method in METHODS}
    anchors = _anchor_values(episodes)
    points = [row for row in _validation_points(episodes, anchors) if row["method"] in METHODS]
    final_rows = [row for row in episodes if row["phase"] == "final_eval" and row["method"] in METHODS]
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    unfinished = [f"{LABELS[method]}/seed {seed}" for (method, seed), row in sorted(progress.items())
                  if row.get("status") in ("training", "evaluating", "running")]
    planned = [(method, seed) for method in METHODS for seed in FORMAL_SEEDS]
    finished = sum(1 for key in planned if progress.get(key, {}).get("status") in ("completed", "complete"))
    if unfinished:
        status = " 仍在运行：" + "、".join(unfinished) + "。"
    elif finished == len(planned):
        status = " 三种子均已结束。"
    elif finished:
        status = f" 已完成 {finished}/{len(planned)} 次训练。"
    else:
        status = " 正式训练尚未开始。"
    comparison = "、".join(LABELS[method] for method in METHODS)
    pool_only_main = [LABELS[method] for method in METHODS if method in POOL_ONLY_METHODS]
    text = ["<!-- report_layout: qmix_baseline -->", "# 跨规模攻防泛化 main v3", "",
            f"更新时间：{now}。比较 {comparison}，各 3 个训练种子。" + status,
            "本版本修复了 v2 的团队价值下界、$b_1$/$V$ 随规模缩放、红方全灭尾段与有序空槽四处问题，"
            "诊断见 [v2 报告](../main_v2/实验报告.md) 与 "
            "[差异清单](../../../docs/方案_v2/diff_log.md)。规则与随机锚点与算法无关；"
            "已有配置沿用 v2 的 300 局种子，新增配置按同一种子批补齐。",
            "训练的是红方，团队回报 $R=-D$（逐步奖励 $r_t=-\\Delta D_t$）。"
            "训练池 $N\\in\\{4,6,8,10\\}$、$K\\in\\{1,2,3\\}$ 均匀采样；验证只看池内 4 配置；"
            f"最终评估 {len(FINAL_CONFIGS)} 个配置各 300 局。外推分三条线："
            "等规模 10/15/20/25/30/40v 同数，1:2 的 2v4/4v8/8v16/15v30/20v40，"
            "以及 10/15/20/30v 同数上的目标数 K=2/4/6。"
            + (f"其中 {'、'.join(pool_only_main)} 只覆盖训练池 "
               f"{len(VALIDATION_CONFIGS)} 个配置。" if pool_only_main else "")
            + "逐局与 loss 仍在同目录 CSV。",
            "对照基线 QMIX 是实体注意力 mixer 的 QMIX，不是有序展平那一支。"
            "ALMA 是唯一的分层臂，低层与该 QMIX 同构，差别是多一层子任务分配："
            "上层每 5 步把红方分到目标子任务；低层看全场，分配作为任务嵌入；"
            "mixer 是单路团队 Q，低层 TD 用团队回报。硬掩码原版和其余适配臂见 "
            "[ALMA 探针](../alma_probe_v3/实验报告.md)。"
            "蓝方按“当前最近目标”归属，这是公开几何量；蓝方私有指派不进入红方输入。", ""]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    behavior = _id_behavior(final_rows, episodes)
    learning_figure = _plot_learning(plt, output, points, anchors)
    results_figures = [path for path in _plot_final(plt, output, final_rows, anchors) if path]
    behavior_figure = _plot_behavior(plt, output, behavior)
    random_r, rule_r = _pool_return(anchors, "random"), _pool_return(anchors, "rule_nv1")
    last_r = {}
    for method in METHODS:
        last = [row["D"] for row in points if row["method"] == method and row["eval_point"] == 50]
        mean, std = _mean_std([-value for value in last])
        last_r[method] = (mean, std)
    intercept = {method: _mean_std([row["intercept"] for row in behavior.get(method, [])])
                 for method in ("random", *METHODS, "rule_nv1")}
    friendly = {method: _mean_std([row["friendly"] for row in behavior.get(method, [])])
                for method in ("random", *METHODS, "rule_nv1")}
    text += ["## 结论", ""]
    ranked = [method for method in METHODS if last_r[method][0] is not None]
    if random_r is not None and rule_r is not None and ranked:
        ranking = "；".join(
            f"{LABELS[method]} ${last_r[method][0]:.2f}\\pm{last_r[method][1]:.2f}$" for method in ranked)
        versus_rule = []
        for method in ranked:
            mean = last_r[method][0]
            if mean >= rule_r - 1e-6:
                versus_rule.append(f"{LABELS[method]}达到或超过规则")
            elif mean > random_r + 0.3:
                versus_rule.append(f"{LABELS[method]}高于随机、低于规则")
            else:
                versus_rule.append(f"{LABELS[method]}仍接近随机")
        pending = [LABELS[method] for method in METHODS if method not in ranked]
        text += [
            f"红方回报越高越好。训练池内随机约 ${random_r:.2f}$，规则约 ${rule_r:.2f}$。"
            f"已有 1M 验证回报的方法：{ranking}。最终评估用途中 best，不是曲线最右端。"
            + (f"尚未写结论的方法：{'、'.join(pending)}。" if pending else ""),
            "；".join(versus_rule) + "。",
        ]
        known = [method for method in ("random", *METHODS, "rule_nv1") if intercept[method][0] is not None]
        if {"random", "rule_nv1", "b2_qmix_atten", "refil"} <= set(known):
            learned_ix = "，".join(
                f"{LABELS[method]} {intercept[method][0]:.2f}"
                for method in METHODS if intercept[method][0] is not None)
            text += [
                f"训练池内拦截率：规则 {intercept['rule_nv1'][0]:.2f}，随机 {intercept['random'][0]:.2f}，"
                f"{learned_ix}。"
                f"QMIX 同队碰撞死亡占比 {friendly['b2_qmix_atten'][0]:.2f}，随机 {friendly['random'][0]:.2f}，规则 {friendly['rule_nv1'][0]:.2f}。",
            ]
        text += [""]
    else:
        text += ["记录尚未齐，暂不写结论。", ""]
    note = _side_metric_note(episodes, anchors)
    if note:
        text += [note, ""]

    text += ["## 训练曲线", "",
             "纵轴是红方整局回报 $R=-D$，与训练目标一致。每 2 万步在池内 4 配置上贪心 100 局，"
             "实线为三种子均值，色带为标准差。随机/规则是同一 4 配置上的水平基准（各 300 局，无训练种子）。"
             "途中不评外推。", ""]
    if learning_figure:
        text += [f"![训练曲线]({learning_figure.relative_to(output).as_posix()})", ""]
    else:
        text += ["尚无足够记录绘制训练曲线。", ""]

    text += ["## 最终评估", "",
             "best 模型，每配置 300 局。三张图分别是等规模 1:1、红蓝 1:2、以及目标数扩充。"
             "目标点图按 10/15/20/30v 分成四格，颜色与另外两张图相同。"
             "图中误差线与表中 ± 都是跨训练种子的标准差。随机与规则锚点各 300 局，无训练种子。"
             + (f"{'、'.join(pool_only_main)} 的有序固定槽位按训练池规模定义，没有更大编队的参数，"
                "因此只在训练池 4 个配置上评估，外推行显示“—”。" if pool_only_main else ""), ""]
    captions = ("等规模 1:1", "1:2 配比", "目标数（一格一个编队规模）")
    for path, caption in zip(results_figures, captions):
        text += [f"![{caption}]({path.relative_to(output).as_posix()})", ""]
    scores = _final_seed_scores(final_rows, anchors)
    final_table = []
    for config in FINAL_CONFIGS:
        row = dict(config=config_label(config), axis=_config_axis(config))
        random, rule = anchors.get(("random", config), {}), anchors.get(("rule_nv1", config), {})
        row["random"] = _number(random.get("D")) if random.get("complete") else "—"
        row["rule"] = _number(rule.get("D")) if rule.get("complete") else "—"
        for method in METHODS:
            values = list(scores[method][config].values())
            row[method] = _plus_minus([item["D"] for item in values])
            row[f"{method}_n"] = len(values)
            row[f"{method}_nds"] = _plus_minus([item["NDS"] for item in values if item["NDS"] is not None])
        final_table.append(row)
    if any(scores[method] for method in METHODS) or anchors:
        text += _table(final_table, [
            ("config", "配置"), ("axis", "轴"), ("random", "随机 D"), ("rule", "规则 D"),
            *[(method, f"{LABELS[method]} D") for method in METHODS],
        ]) + [""]
        text += _table(final_table, [
            ("config", "配置"), ("axis", "轴"),
            *[(f"{method}_nds", f"{LABELS[method]} NDS") for method in METHODS],
        ]) + [""]
    else:
        text += ["最终评估尚无已完成配置。", ""]

    text += ["## 行为诊断", "",
             "与训练曲线相同的 4 个训练池配置。左图：蓝方被拦下的比例，越高说明红方越像在防守。"
             "右图：红方死于同队碰撞的人数占比，越低越好。误差线为训练种子标准差。", ""]
    if behavior_figure:
        text += [f"![行为诊断]({behavior_figure.relative_to(output).as_posix()})", ""]
    else:
        text += ["尚无足够诊断记录。", ""]

    status_rows = []
    for method in METHODS:
        for seed in (0, 1, 2):
            row = progress.get((method, seed), {})
            series = [item for item in points if item["method"] == method and item["seed"] == seed]
            best = min(series, key=lambda item: (item["D"], item["t_env"])) if series else None
            status_rows.append(dict(
                method=LABELS[method], seed=seed, status=row.get("status", "未开始"),
                steps=row.get("t_env", 0), budget=row.get("budget_steps", 1000000),
                points=f"{len(series)}/50",
                best=_number(best["D"]) if best else "—",
                best_step=best["t_env"] if best else "—"))
    text += ["## 运行状态", ""]
    text += _table(status_rows, [("method", "方法"), ("seed", "种子"), ("status", "状态"),
                                 ("steps", "已训步"), ("budget", "预算"), ("points", "验证点"),
                                 ("best", "best 验证 D"), ("best_step", "best 步数")]) + [""]

    streams = ("episodes", "learning", "progress", "verification", "benchmarks", "trajectories")
    text += ["## 原始数据", "",
             "；".join(f"[{name}.csv]({name}.csv)" for name in streams if (output / f"{name}.csv").exists()) + "。", ""]
    report_path = output / "实验报告.md"
    appendix = _preserve_appendix(output)
    content = "\n".join(text)
    if appendix:
        content = content.rstrip() + "\n\n" + appendix
    if report_stream is None:
        _atomic_text(report_path, content)
    else:
        # Keep the same inode for the POSIX interprocess lock.
        report_stream.seek(0)
        report_stream.write(content.encode("utf-8"))
        report_stream.truncate()
        report_stream.flush()
        os.fsync(report_stream.fileno())
    return report_path


def _is_probe_output(output):
    return Path(output).resolve().name in {"alma_probe_v3", "alma_probe"}


def _is_v4_output(output):
    return Path(output).resolve().name == "main_v4"


def _is_v5_output(output):
    return Path(output).resolve().name == "main_v5"


def _relabel_method(rows, src, dst):
    relabeled = []
    for row in rows:
        item = dict(row)
        if item.get("method") == src:
            item["method"] = dst
        relabeled.append(item)
    return relabeled


def _v3_comparison_episodes():
    """Reuse v3 REFIL seed 0 and policy-independent anchors; do not copy the directory."""
    if not V3_OUTPUT.exists():
        return []
    rows = unique_episodes(read_records(V3_OUTPUT, "episodes", run=FORMAL_RUN))
    kept = []
    for row in rows:
        method = row.get("method")
        if method in ("random", "rule_nv1"):
            kept.append(row)
        elif method == "refil" and int(row.get("seed", 0)) == 0:
            kept.append(row)
    return kept


def _v4_count_episodes():
    """Reuse v4 REFIL-C seed 0 (LayerNorm + φ₂); do not copy the directory."""
    if not V4_OUTPUT.exists():
        return []
    rows = unique_episodes(read_records(V4_OUTPUT, "episodes", run=FORMAL_RUN))
    kept = [row for row in rows
            if row.get("method") == "refil_count" and int(row.get("seed", 0)) == 0]
    return _relabel_method(kept, "refil_count", "refil_count_ln")


def _v5_count_noln_episodes():
    """Reuse v5 REFIL-C−LN seed 0 (φ₂ FiLM without LayerNorm); do not copy."""
    if not V5_OUTPUT.exists():
        return []
    rows = unique_episodes(read_records(V5_OUTPUT, "episodes", run=FORMAL_RUN))
    kept = [row for row in rows
            if row.get("method") == "refil_count" and int(row.get("seed", 0)) == 0]
    return _relabel_method(kept, "refil_count", "refil_count_noln")


def _v3_reference():
    """Anchors and seed-0 last validation from the formal v3 directory."""
    if not V3_OUTPUT.exists():
        return {}, {}
    episodes = unique_episodes(read_records(V3_OUTPUT, "episodes", run=FORMAL_RUN))
    anchors = _anchor_values(episodes)
    points = _validation_points(episodes, anchors)
    refs = {}
    for method in ("b2_qmix_atten", "refil"):
        last = [-row["D"] for row in points
                if row["method"] == method and row.get("eval_point") == 50 and row.get("seed") == 0
                and _numeric(row.get("D"))]
        if last:
            refs[method] = statistics.mean(last)
    return anchors, refs


def _plot_probe_learning(plt, output, points, anchors, refs):
    fig, axis = plt.subplots(1, 1, figsize=(7.6, 4.4), constrained_layout=True)
    for method in PROBE_METHODS:
        by_point = defaultdict(list)
        for row in points:
            if row["method"] == method:
                by_point[row["eval_point"]].append(-row["D"])
        xs, ys, err = [], [], []
        for point, rewards in sorted(by_point.items()):
            mean, std = _mean_std(rewards)
            if mean is None:
                continue
            times = [row["t_env"] for row in points if row["method"] == method and row["eval_point"] == point]
            xs.append(statistics.mean(times))
            ys.append(mean)
            err.append(std)
        if xs:
            axis.plot(xs, ys, color=COLORS[method], linewidth=2.2, label=PROBE_PLOT_LABELS[method])
            axis.fill_between(xs, [y - e for y, e in zip(ys, err)], [y + e for y, e in zip(ys, err)],
                              color=COLORS[method], alpha=.18, linewidth=0)
    for method, style in (("b2_qmix_atten", "--"), ("refil", "-.")):
        if method in refs:
            axis.axhline(refs[method], color=COLORS[method], linestyle=style, linewidth=1.4,
                         label=f"{PROBE_PLOT_LABELS[method]} seed0")
    for policy, style in (("random", "--"), ("rule_nv1", "-.")):
        value = _pool_return(anchors, policy)
        if value is not None:
            axis.axhline(value, color=COLORS[policy], linestyle=style, linewidth=1.2, label=LABELS[policy])
    axis.set(title="ALMA probe: in-distribution red return",
             xlabel="Environment steps", ylabel="Red episode return  R = −D")
    axis.grid(alpha=.2)
    axis.legend(fontsize=8)
    path = output / "figures" / "learning.png"
    _save_figure(fig, path)
    plt.close(fig)
    return path


def _refresh_probe_report(output, *, run=FORMAL_RUN, report_stream=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    episodes = unique_episodes(read_records(output, "episodes", run=run))
    progress = {(method, seed): row for (method, recorded_run, seed), row in
                read_latest(output, "progress", run=run).items() if method in PROBE_METHODS}
    anchors, refs = _v3_reference()
    points = [row for row in _validation_points(episodes, anchors) if row["method"] in PROBE_METHODS]
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    unfinished = [f"{PROBE_LABELS[method]}/seed {seed}" for (method, seed), row in sorted(progress.items())
                  if row.get("status") in ("training", "evaluating", "running")]
    live_methods = tuple(method for method in PROBE_METHODS if method != "alma")
    planned = [(method, seed) for method in live_methods for seed in PROBE_SEEDS]
    finished = sum(1 for key in planned if progress.get(key, {}).get("status") in ("completed", "complete", "failed"))
    if unfinished:
        status = " 仍在运行：" + "、".join(unfinished) + "。"
    elif finished == len(planned):
        status = " 五臂对照已齐：四条适配臂有本目录验证记录，硬掩码三种子为迁移前留档。"
    elif finished:
        status = f" 已完成 {finished}/{len(planned)} 次适配臂记录。"
    else:
        status = " 记录尚未开始。"
    text = [
        "<!-- report_layout: alma_probe -->",
        "# ALMA 场景适配对照（alma_probe_v3）",
        "",
        f"更新时间：{now}。这是 main v3 的补充对照，只留验证记录和本报告，不保留权重。"
        "主实验中的 ALMA 已换成全场臂，名称仍是 ALMA；硬掩码原版留在这里。"
        + status,
        "适配臂各 1 个种子、1,000,000 物理步、50 个池内验证点。"
        "硬掩码 ALMA 原为三种子正式训练，逐局 CSV 未转入本目录，数字抄自迁移前主报告。"
        "主报告结论以 [main v3](../main_v3/实验报告.md) 为准。",
        "前三刀是递进关系；**ALMA-NoMask** 从全场臂分叉。**ALMA-掩码** 是原论文默认硬掩码路径。",
        "",
        "- **ALMA-掩码**（`alma`）：目标子任务，硬掩码，每任务一个 Q，子任务奖励，每 5 步分配。",
        "- **ALMA-全场**（`alma_fullobs`）：去掉硬掩码，低层看全场；单路团队 Q，团队回报。主实验现用这一路，仍称 ALMA。",
        "- **ALMA-蓝方**（`alma_blue`）：在全场之上把子任务改成活着的蓝方槽，训练垫 10、评估宽 40。",
        "- **ALMA-事件**（`alma_event`）：在蓝方子任务之上改为伤亡 / 公开最近目标变化 / 最多 10 步再分配。",
        "- **ALMA-NoMask**（`alma_nomask`）：全场任务标签在注意力之前进入实体；每任务一个 Q，子任务奖励。",
        "",
        "对照虚线来自 main v3 的 QMIX / REFIL seed 0 第 50 个验证点，以及同一套规则 / 随机锚点。",
        "",
    ]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    learning_figure = _plot_probe_learning(plt, output, points, anchors, refs)
    last_r = {}
    for method in PROBE_METHODS:
        last = [row["D"] for row in points if row["method"] == method and row["eval_point"] == 50]
        mean, std = _mean_std([-value for value in last])
        last_r[method] = (mean, std)
    if last_r.get("alma", (None, None))[0] is None:
        last_r["alma"] = (ARCHIVED_MASK_ALMA["last_mean"], ARCHIVED_MASK_ALMA["last_std"])
    text += ["## 结论", ""]
    random_r, rule_r = _pool_return(anchors, "random"), _pool_return(anchors, "rule_nv1")
    ranked = [method for method in PROBE_METHODS if last_r[method][0] is not None]
    if ranked:
        ranking = "；".join(
            f"{PROBE_LABELS[method]} ${last_r[method][0]:.2f}\\pm{last_r[method][1]:.2f}$" for method in ranked)
        text += [
            f"红方回报越高越好。v3 池内随机约 ${random_r:.2f}$，规则约 ${rule_r:.2f}$。"
            if random_r is not None and rule_r is not None else "锚点尚未读到。",
            f"1M 末验证回报：{ranking}。全场接近规则；蓝方次之；事件高于随机、低于规则；"
            "掩码与 NoMask 后期塌缩，接近随机。主实验因此用全场臂作为 ALMA。",
        ]
        text += [""]
    else:
        text += ["记录尚未齐，暂不写结论。先看学习曲线是否离开随机带。", ""]
    text += [
        "## 硬掩码 ALMA 留档",
        "",
        "三种子正式训练曾在 main v3 完成。迁移时逐局验证记录未能写入本目录，"
        "下表抄自当时主报告，不是重新评估。seed 0 末点约 "
        f"${ARCHIVED_MASK_ALMA['seed0_last']:.2f}$，三种子末点均值 "
        f"${ARCHIVED_MASK_ALMA['last_mean']:.2f}\\pm{ARCHIVED_MASK_ALMA['last_std']:.2f}$，"
        f"池内拦截率 {ARCHIVED_MASK_ALMA['intercept']:.2f}。",
        "",
    ]
    text += _table([
        dict(seed=row["seed"], status=row["status"], steps=row["steps"],
             points=row["points"], best=_number(row["best"]), best_step=row["best_step"])
        for row in ARCHIVED_MASK_ALMA["seeds"]
    ], [("seed", "种子"), ("status", "来源"), ("steps", "已训步"), ("points", "验证点"),
        ("best", "best 验证回报"), ("best_step", "best 步数")]) + [""]
    if learning_figure:
        text += ["## 训练池内验证曲线", "",
                 f"![学习曲线]({learning_figure.relative_to(output).as_posix()})", ""]
    if points:
        rows = []
        for method in PROBE_METHODS:
            series = [row for row in points if row["method"] == method]
            if not series:
                continue
            last = [row for row in series if row["eval_point"] == max(item["eval_point"] for item in series)]
            mean, std = _mean_std([-row["D"] for row in last])
            rows.append(dict(method=PROBE_LABELS[method], point=last[0]["eval_point"],
                             steps=int(statistics.mean(row["t_env"] for row in last)),
                             ret=_number(mean), std=_number(std)))
        if not any(row["method"] == PROBE_LABELS["alma"] for row in rows):
            rows.insert(0, dict(method=PROBE_LABELS["alma"], point=50, steps=1000000,
                                ret=_number(ARCHIVED_MASK_ALMA["last_mean"]),
                                std=_number(ARCHIVED_MASK_ALMA["last_std"])))
        text += ["## 当前验证点", ""]
        text += _table(rows, [("method", "方法"), ("point", "验证点"), ("steps", "步数"),
                              ("ret", "回报"), ("std", "标准差")]) + [""]
    status_rows = []
    for method, seed in planned:
        row = progress.get((method, seed), {})
        best = None
        series = [item for item in points if item["method"] == method and item.get("seed") == seed]
        if series:
            best = min(series, key=lambda item: item["D"])
        status_rows.append(dict(
            method=PROBE_LABELS[method], seed=seed, status=row.get("status", "未开始"),
            steps=row.get("t_env", 0), budget=row.get("budget_steps", 1000000),
            points=f"{len({item['eval_point'] for item in series})}/50",
            best=_number(-best["D"]) if best else "—",
            best_step=best["t_env"] if best else "—"))
    text += ["## 运行状态", ""]
    text += _table(status_rows, [("method", "方法"), ("seed", "种子"), ("status", "状态"),
                                 ("steps", "已训步"), ("budget", "预算"), ("points", "验证点"),
                                 ("best", "best 验证回报"), ("best_step", "best 步数")]) + [""]
    streams = ("episodes", "learning", "progress")
    text += ["## 原始数据", "",
             "；".join(f"[{name}.csv]({name}.csv)" for name in streams if (output / f"{name}.csv").exists()) + "。",
             f"正式对照仍以 [main v3 主报告](../main_v3/实验报告.md) 为准。", ""]
    report_path = output / "实验报告.md"
    content = "\n".join(text)
    if report_stream is None:
        _atomic_text(report_path, content)
    else:
        report_stream.seek(0)
        report_stream.write(content.encode("utf-8"))
        report_stream.truncate()
        report_stream.flush()
        os.fsync(report_stream.fileno())
    return report_path


def _method_config_d(scores, method, config):
    values = list(scores[method][config].values())
    if not values:
        return None
    return statistics.mean(item["D"] for item in values)


def _v4_analysis_text(scores, last_r, behavior, rule_r):
    """Seed-0 comparison of the v4 arms and reused count-without-LN against REFIL."""
    complete = [config for config in FINAL_CONFIGS
                if all(_method_config_d(scores, method, config) is not None for method in V4_METHODS)]
    if len(complete) != len(FINAL_CONFIGS):
        return ["记录尚未齐，暂不写对照分析。", ""]
    wins = Counter()
    eta_order = 0
    count_beats = 0
    count_noln_beats = 0
    gaps = defaultdict(list)
    axis_values = defaultdict(lambda: defaultdict(list))
    for config in complete:
        ds = {method: _method_config_d(scores, method, config) for method in V4_METHODS}
        wins[min(ds, key=ds.get)] += 1
        if ds["refil_local_mid"] >= ds["refil_local_mild"] >= ds["refil"] - 1e-12:
            eta_order += 1
        if ds["refil_count"] + 1e-12 < ds["refil"]:
            count_beats += 1
        if ds["refil_count_noln"] + 1e-12 < ds["refil"]:
            count_noln_beats += 1
        axis = _config_axis(config)
        for method in V4_COMPARE_ARMS:
            gaps[axis, method].append(ds[method] - ds["refil"])
        for method, value in ds.items():
            axis_values[axis][method].append(value)
    axis_rank = []
    for axis in ("训练池内", "等规模", "1:2", "目标外推"):
        means = {method: statistics.mean(values) for method, values in axis_values.get(axis, {}).items()}
        if means:
            best = min(means, key=means.get)
            axis_rank.append(f"{axis}最低平均 $D$ 是 {LABELS[best]}")
    gap_bits = []
    for axis in ("训练池内", "等规模", "1:2", "目标外推"):
        parts = []
        for method in V4_COMPARE_ARMS:
            values = gaps.get((axis, method), [])
            if values:
                parts.append(f"{LABELS[method]} {statistics.mean(values):+.2f}")
        if parts:
            gap_bits.append(f"{axis}相对 REFIL：{'，'.join(parts)}")
    friendly = {method: _mean_std([row["friendly"] for row in behavior.get(method, [])])[0]
                for method in ("rule_nv1", *V4_METHODS)}
    friendly_bits = "，".join(
        f"{LABELS[method]} {friendly[method]:.3f}"
        for method in V4_METHODS if friendly.get(method) is not None)
    last_bits = "；".join(
        f"{LABELS[method]} ${last_r[method][0]:.2f}$" for method in V4_METHODS if last_r[method][0] is not None)
    win_bits = "，".join(f"{LABELS[method]} {wins[method]}" for method in V4_METHODS)
    paragraphs = [
        "三臂配置与落盘一致：局部两臂 `imagine_group=mixed_distance`、$r=0.4$，"
        r"$\eta=0.15$ / $0.30$；数量臂 `count_cond=phi2`、分组仍是原版。"
        "1M、50/50 验证、终评 23 配置×300 局均写完。评估时 pad 随 `n_agents`/`n_entities`/`n_tasks` 更新，"
        "数量切片红 / 蓝 / 目标槽不错位。"
        r"去 LN 的数量臂复用 [main v5](../main_v5/实验报告.md) seed 0 的同一套 $\phi_2$ FiLM（`count_ln=False`），不拷贝目录。",
        r"距离分组按约定是独立 Bernoulli：$s_e=(1-\eta)+\eta\exp(-d^2/(2r^2))$，$\pi_e=p\,s_e$，中心实体用未缩放的 $p$。"
        r"给定 $p=0.5$ 时，任意实体与中心同组的概率恒为 $0.5$，距离并不制造“靠近就编进同一组”。"
        r"它主要把组 A 的期望规模从 $p$ 压到 $p\,\bar s<p$。$\eta$ 越大，$\bar s$ 越小，组越瘦。"
        "终评里 L0.30 几乎处处差于 L0.15，两者都差于原版 REFIL，与这个收缩方向一致。"
        r"带 LN 的数量臂在注意力输出上始终做 LayerNorm 再 FiLM；即使 $\alpha=\beta=0$，前向也与原版 REFIL 不完全相同。"
        r"去掉 LN 之后，$\alpha=\beta=0$ 时前向与原版一致，学到的计数条件仍是另一条臂。",
        f"池内末点验证回报（seed 0，$R=-D$，越高越好）：{last_bits}。"
        + (f"规则约 ${rule_r:.2f}$。" if rule_r is not None else ""),
        f"终评 23 个配置中，最低 $D$ 次数：{win_bits}。"
        f"L0.30 的 $D$ ≥ L0.15 ≥ REFIL 的配置有 {eta_order}/23。"
        f"带 LN 的数量臂 $D$ 低于原版 REFIL 的配置有 {count_beats}/23；"
        f"去 LN 的有 {count_noln_beats}/23。"
        + (("；".join(axis_rank) + "。") if axis_rank else ""),
        ("。".join(gap_bits) + "。") if gap_bits else "",
        "等规模与目标外推上，原版 REFIL 的优势随编队变大而拉开；L0.30 在 30/40v 和 $K=6$ 掉得最明显。"
        "1:2 线上各学习方法彼此接近，大编队都略差于规则，说明劣势兵力下分组或计数都没有改掉原版已经碰到的墙。",
        (f"池内同队碰撞死亡占比：规则 {friendly['rule_nv1']:.3f}，{friendly_bits}。"
         if friendly.get("rule_nv1") is not None and friendly_bits else ""),
        "以上都是单一种子。不能把 L0.15 与数量臂相对原版的差距读成显著变差，方向却一致：这几处改动都没有超过已修正的原版 REFIL。",
    ]
    return [paragraph for paragraph in paragraphs if paragraph] + [""]


def _refresh_v4_report(output, *, run=FORMAL_RUN, report_stream=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    episodes = unique_episodes(
        read_records(output, "episodes", run=run)
        + _v3_comparison_episodes()
        + _v5_count_noln_episodes())
    progress = {(method, seed): row for (method, recorded_run, seed), row in
                read_latest(output, "progress", run=run).items() if method in V4_TRAIN_METHODS}
    anchors = _anchor_values(episodes)
    points = [row for row in _validation_points(episodes, anchors) if row["method"] in V4_METHODS]
    final_rows = [row for row in episodes if row["phase"] == "final_eval" and row["method"] in V4_METHODS]
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    unfinished = [f"{LABELS[method]}/seed {seed}" for (method, seed), row in sorted(progress.items())
                  if row.get("status") in ("training", "evaluating", "running")]
    planned = [(method, seed) for method in V4_TRAIN_METHODS for seed in V4_SEEDS]
    finished = sum(1 for key in planned if progress.get(key, {}).get("status") in ("completed", "complete"))
    if unfinished:
        status = " 仍在运行：" + "、".join(unfinished) + "。"
    elif finished == len(planned):
        status = " 三个新臂的 seed 0 均已结束。"
    elif finished:
        status = f" 已完成 {finished}/{len(planned)} 次新臂训练。"
    else:
        status = " 正式训练尚未开始。"
    comparison = "、".join(LABELS[method] for method in V4_METHODS)
    text = ["# 跨规模攻防泛化 main v4", "",
            f"更新时间：{now}。比较 {comparison}，以及规则锚点。" + status,
            "本批先训两个算法、三个任务，全部 seed 0。"
            "局部协作两臂只改 REFIL 辅助分组：先抽原版共享 $p$，再按与随机中心的距离缩放，"
            r"$\pi_e=p\,s_e$，$s_e=(1-\eta)+\eta\exp(-d_{ie}^{2}/(2r^{2}))$。"
            r"两臂 $r=0.4$（约 1000 m），$\eta=0.15$（更贴近原版）与 $\eta=0.30$（仍温和）。"
            r"数量条件一臂只加 $\phi_2$ 有界计数，FiLM 调节注意力输出；分组仍是原版 REFIL。"
            "原版 REFIL、规则和随机复用 [main v3](../main_v3/实验报告.md) 的 seed 0 / 锚点，不重跑。"
            r"去 LN 的数量臂 REFIL-C−LN 复用 [main v5](../main_v5/实验报告.md) seed 0，不重跑、不拷贝目录。",
            "评估协议与 v3 相同：1M 物理步、50 次池内验证（4 配置×25 局）、"
            f"终评 {len(FINAL_CONFIGS)} 个配置各 300 局，指标 $D$ 与 NDS。"
            "图上学习方法一律用 seed 0，不把 v3 三种子均值画成对照。"
            "三种子接口已留，本批不排队。不把 C 与 G 叠在同一臂，也不做多尺度数量编码。",
            "训练的是红方，团队回报 $R=-D$。训练池 $N\\in\\{4,6,8,10\\}$、$K\\in\\{1,2,3\\}$ 均匀采样；"
            "验证只看池内 4 配置。外推分三条线：等规模 10/15/20/25/30/40v 同数，"
            "1:2 的 2v4/4v8/8v16/15v30/20v40，以及 10/15/20/30v 同数上的目标数 K=2/4/6。"
            "逐局与 loss 仍在同目录 CSV。", ""]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    behavior = _id_behavior(final_rows, episodes, methods=V4_METHODS)
    learning_figure = _plot_learning(plt, output, points, anchors, methods=V4_METHODS,
                                    include_random=False)
    results_figures = [path for path in _plot_final(
        plt, output, final_rows, anchors, methods=V4_METHODS, include_random=False) if path]
    behavior_figure = _plot_behavior(plt, output, behavior, methods=V4_METHODS,
                                    include_random=False)
    random_r, rule_r = _pool_return(anchors, "random"), _pool_return(anchors, "rule_nv1")
    last_r = {}
    for method in V4_METHODS:
        last = [row["D"] for row in points if row["method"] == method and row["eval_point"] == 50
                and row.get("seed") == 0]
        mean, std = _mean_std([-value for value in last])
        last_r[method] = (mean, std)
    intercept = {method: _mean_std([row["intercept"] for row in behavior.get(method, [])])
                 for method in ("random", *V4_METHODS, "rule_nv1")}
    scores = _final_seed_scores(final_rows, anchors, methods=V4_METHODS)
    text += ["## 结论", ""]
    ranked = [method for method in V4_METHODS if last_r[method][0] is not None]
    if random_r is not None and rule_r is not None and ranked:
        ranking = "；".join(
            f"{LABELS[method]} ${last_r[method][0]:.2f}$" for method in ranked)
        versus_rule = []
        for method in ranked:
            mean = last_r[method][0]
            if mean >= rule_r - 1e-6:
                versus_rule.append(f"{LABELS[method]}达到或超过规则")
            elif mean > random_r + 0.3:
                versus_rule.append(f"{LABELS[method]}高于随机、低于规则")
            else:
                versus_rule.append(f"{LABELS[method]}仍接近随机")
        pending = [LABELS[method] for method in V4_TRAIN_METHODS if method not in ranked]
        text += [
            f"红方回报越高越好。训练池内随机约 ${random_r:.2f}$，规则约 ${rule_r:.2f}$。"
            f"已有 1M 验证回报的方法（seed 0）：{ranking}。最终评估用途中 best，不是曲线最右端。"
            + (f"尚未写结论的新臂：{'、'.join(pending)}。" if pending else ""),
            "；".join(versus_rule) + "。",
        ]
        known = [method for method in ("random", *V4_METHODS, "rule_nv1") if intercept[method][0] is not None]
        if {"random", "rule_nv1"} <= set(known) and any(intercept[method][0] is not None for method in V4_METHODS):
            learned_ix = "，".join(
                f"{LABELS[method]} {intercept[method][0]:.2f}"
                for method in V4_METHODS if intercept[method][0] is not None)
            text += [
                f"训练池内拦截率：规则 {intercept['rule_nv1'][0]:.2f}，随机 {intercept['random'][0]:.2f}，"
                f"{learned_ix}。",
            ]
        text += [""]
    else:
        text += ["记录尚未齐，暂不写结论。", ""]

    text += ["## 分析", ""] + _v4_analysis_text(scores, last_r, behavior, rule_r)

    text += ["## 训练曲线", "",
             "纵轴是红方整局回报 $R=-D$，与训练目标一致。每 2 万步在池内 4 配置上贪心 100 局。"
             "当前各学习方法只有 seed 0，没有跨训练种子色带。"
             "REFIL 是 v3 seed 0 的复用曲线；REFIL-C−LN 是 v5 seed 0 的复用曲线。"
             "点划线是同一 4 配置上的规则基准（300 局）。"
             "图上不画随机，以免纵轴被拉开；随机数字仍在终评表里，并用于 NDS。"
             "途中不评外推。", ""]
    if learning_figure:
        text += [f"![训练曲线]({learning_figure.relative_to(output).as_posix()})", ""]
    else:
        text += ["尚无足够记录绘制训练曲线。", ""]

    text += ["## 最终评估", "",
             "best 模型，每配置 300 局。三张图分别是等规模 1:1、红蓝 1:2、以及目标数扩充。"
             "目标点图按 10/15/20/30v 分成四格，颜色与另外两张图相同。"
             "当前为 seed 0，表中不写跨训练种子标准差。图上只留规则锚点；随机仍在表中，复用 v3 各 300 局。", ""]
    captions = ("等规模 1:1", "1:2 配比", "目标数（一格一个编队规模）")
    for path, caption in zip(results_figures, captions):
        text += [f"![{caption}]({path.relative_to(output).as_posix()})", ""]
    final_table = []
    for config in FINAL_CONFIGS:
        row = dict(config=config_label(config), axis=_config_axis(config))
        random, rule = anchors.get(("random", config), {}), anchors.get(("rule_nv1", config), {})
        row["random"] = _number(random.get("D")) if random.get("complete") else "—"
        row["rule"] = _number(rule.get("D")) if rule.get("complete") else "—"
        for method in V4_METHODS:
            values = list(scores[method][config].values())
            row[method] = _plus_minus([item["D"] for item in values])
            row[f"{method}_nds"] = _plus_minus([item["NDS"] for item in values if item["NDS"] is not None])
        final_table.append(row)
    if any(scores[method] for method in V4_METHODS) or anchors:
        text += _table(final_table, [
            ("config", "配置"), ("axis", "轴"), ("random", "随机 D"), ("rule", "规则 D"),
            *[(method, f"{LABELS[method]} D") for method in V4_METHODS],
        ]) + [""]
        text += _table(final_table, [
            ("config", "配置"), ("axis", "轴"),
            *[(f"{method}_nds", f"{LABELS[method]} NDS") for method in V4_METHODS],
        ]) + [""]
    else:
        text += ["最终评估尚无已完成配置。", ""]

    text += ["## 行为诊断", "",
             "与训练曲线相同的 4 个训练池配置。左图：蓝方被拦下的比例，越高说明红方越像在防守。"
             "右图：红方死于同队碰撞的人数占比，越低越好。当前为 seed 0。"
             "图上不画随机。", ""]
    if behavior_figure:
        text += [f"![行为诊断]({behavior_figure.relative_to(output).as_posix()})", ""]
    else:
        text += ["尚无足够诊断记录。", ""]

    status_rows = []
    for method in V4_TRAIN_METHODS:
        for seed in V4_SEEDS:
            row = progress.get((method, seed), {})
            series = [item for item in points if item["method"] == method and item["seed"] == seed]
            best = min(series, key=lambda item: (item["D"], item["t_env"])) if series else None
            status_rows.append(dict(
                method=LABELS[method], seed=seed, status=row.get("status", "未开始"),
                steps=row.get("t_env", 0), budget=row.get("budget_steps", 1000000),
                points=f"{len(series)}/50",
                best=_number(best["D"]) if best else "—",
                best_step=best["t_env"] if best else "—"))
    refil_series = [item for item in points if item["method"] == "refil" and item["seed"] == 0]
    refil_best = min(refil_series, key=lambda item: (item["D"], item["t_env"])) if refil_series else None
    noln_series = [item for item in points if item["method"] == "refil_count_noln" and item["seed"] == 0]
    noln_best = min(noln_series, key=lambda item: (item["D"], item["t_env"])) if noln_series else None
    status_rows.append(dict(
        method=LABELS["refil_count_noln"], seed=0, status="复用 v5",
        steps=noln_best["t_env"] if noln_best else 0,
        budget=1000000, points=f"{len(noln_series)}/50",
        best=_number(noln_best["D"]) if noln_best else "—",
        best_step=noln_best["t_env"] if noln_best else "—"))
    status_rows.append(dict(
        method=LABELS["refil"], seed=0, status="复用 v3",
        steps=refil_best["t_env"] if refil_best else 0,
        budget=1000000, points=f"{len(refil_series)}/50",
        best=_number(refil_best["D"]) if refil_best else "—",
        best_step=refil_best["t_env"] if refil_best else "—"))
    text += ["## 运行状态", ""]
    text += _table(status_rows, [("method", "方法"), ("seed", "种子"), ("status", "状态"),
                                 ("steps", "已训步"), ("budget", "预算"), ("points", "验证点"),
                                 ("best", "best 验证 D"), ("best_step", "best 步数")]) + [""]

    streams = ("episodes", "learning", "progress", "verification", "benchmarks", "trajectories")
    local = "；".join(f"[{name}.csv]({name}.csv)" for name in streams if (output / f"{name}.csv").exists())
    text += ["## 原始数据", "",
             (local + "。" if local else "")
             + "对照用的原版 REFIL 与锚点见 [main v3](../main_v3/)。"
             "去 LN 的 REFIL-C−LN 逐局见 [main v5](../main_v5/)。", ""]
    report_path = output / "实验报告.md"
    content = "\n".join(text)
    if report_stream is None:
        _atomic_text(report_path, content)
    else:
        report_stream.seek(0)
        report_stream.write(content.encode("utf-8"))
        report_stream.truncate()
        report_stream.flush()
        os.fsync(report_stream.fileno())
    return report_path


def _v5_analysis_text(scores, last_r, behavior, rule_r):
    complete = [config for config in FINAL_CONFIGS
                if all(_method_config_d(scores, method, config) is not None for method in V5_METHODS)]
    if len(complete) != len(FINAL_CONFIGS):
        return ["记录尚未齐。主对照是全实体共享循环（REFIL-B）、数量校正注意力（REFIL-A）、"
                "决策反馈（REFIL-F）和 10 槽压缩（REFIL-K10），相对原版 REFIL。"
                "固定槽不是方案二的定义，只回答要不要潜在槽。"
                "去 LN 的数量 FiLM 训练在本目录，对照图在 [main v4](../main_v4/实验报告.md)。", ""]
    wins = Counter()
    gaps = defaultdict(list)
    for config in complete:
        ds = {method: _method_config_d(scores, method, config) for method in V5_METHODS}
        wins[min(ds, key=ds.get)] += 1
        for method in V5_GLOBAL_METHODS:
            gaps[method].append(ds[method] - ds["refil"])
    last_bits = "；".join(
        f"{LABELS[method]} ${last_r[method][0]:.2f}$" for method in V5_METHODS if last_r[method][0] is not None)
    friendly = {method: _mean_std([row["friendly"] for row in behavior.get(method, [])])[0]
                for method in ("rule_nv1", *V5_METHODS)}
    friendly_bits = "，".join(
        f"{LABELS[method]} {friendly[method]:.3f}"
        for method in V5_METHODS if friendly.get(method) is not None)
    win_bits = "，".join(f"{LABELS[method]} {wins[method]}" for method in V5_METHODS)
    gap_bits = "，".join(
        f"{LABELS[method]} {statistics.mean(gaps[method]):+.2f}" for method in V5_GLOBAL_METHODS)
    return [
        r"四臂都保留原 REFIL 路径，只并行加入全局分支并残差进 GRU。"
        r"REFIL-B 让全部允许实体走共享 SelfAttn 循环，个体跨轮读取；$R$ 从 $\{1,2,3,4\}$ 采样，评估固定 4。"
        r"REFIL-K10 只用 10 个潜在槽做同样的循环与读取。"
        r"REFIL-A 是单次按红/蓝/目标校正的全局注意力。"
        r"REFIL-F 在全实体循环上写入 9 维暂定动作偏好，执行时整队 MAC 可见这些偏好。",
        f"池内末点验证回报（seed 0，$R=-D$）：{last_bits}。"
        + (f"规则约 ${rule_r:.2f}$。" if rule_r is not None else ""),
        f"终评 23 个配置最低 $D$ 次数：{win_bits}。相对原版 REFIL 的平均 $D$ 差：{gap_bits}。",
        (f"池内同队碰撞死亡占比：规则 {friendly['rule_nv1']:.3f}，{friendly_bits}。"
         if friendly.get("rule_nv1") is not None and friendly_bits else ""),
        "单一种子。B 相对原版有外推收益，才支持循环加工；K10 接近 B 说明槽压缩够用，明显差于 B 则不应把固定 $K$ 当成核心假设。",
        "",
    ]


def _refresh_v5_report(output, *, run=FORMAL_RUN, report_stream=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    local = _relabel_method(unique_episodes(read_records(output, "episodes", run=run)),
                            "refil_count", "refil_count_noln")
    episodes = unique_episodes(local + _v4_count_episodes() + _v3_comparison_episodes())
    progress = {(method, seed): row for (method, recorded_run, seed), row in
                read_latest(output, "progress", run=run).items() if method in V5_TRAIN_METHODS}
    anchors = _anchor_values(episodes)
    all_points = _validation_points(episodes, anchors)
    points = [row for row in all_points if row["method"] in V5_METHODS]
    final_rows = [row for row in episodes if row["phase"] == "final_eval" and row["method"] in V5_METHODS]
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    unfinished = [f"{LABELS.get('refil_count_noln' if method == 'refil_count' else method, method)}/seed {seed}"
                  for (method, seed), row in sorted(progress.items())
                  if row.get("status") in ("training", "evaluating", "running")
                  or (row.get("phase") in ("final_eval", "depth_eval") and row.get("status") in ("running",))]
    planned = [(method, seed) for method in V5_TRAIN_METHODS for seed in V5_SEEDS]
    finished = sum(1 for key in planned if progress.get(key, {}).get("status") in ("completed", "complete"))
    global_planned = [(method, seed) for method in V5_GLOBAL_METHODS for seed in V5_SEEDS]
    global_finished = sum(1 for key in global_planned
                          if progress.get(key, {}).get("status") in ("completed", "complete"))
    if unfinished:
        status = " 仍在运行：" + "、".join(unfinished) + "。"
    elif global_finished == len(global_planned):
        status = " 四条全局臂 seed 0 均已结束。"
    elif global_finished:
        status = f" 全局臂已完成 {global_finished}/{len(global_planned)}。"
    elif finished:
        status = f" 已完成 {finished}/{len(planned)} 次本目录训练。"
    else:
        status = " 正式训练尚未开始。"
    comparison = "、".join(LABELS[method] for method in V5_METHODS)
    text = ["# 跨规模攻防泛化 main v5", "",
            f"更新时间：{now}。比较 {comparison}，以及规则锚点。" + status,
            "本批 seed 0 共五个训练任务：去 LN 的 `refil_count`，以及四个全局臂 "
            "`refil_cycle`（全实体共享循环）、`refil_card`（数量校正注意力）、"
            "`refil_feedback`（决策反馈）、`refil_slot`（10 槽压缩对照）。"
            "`--group v5` 一条命令排队，训练并发上限由 `--max-concurrent` 控制。"
            r"循环臂训练时 $R$ 从 $\{1,2,3,4\}$ 均匀采样，验证和终评固定 $R=4$。"
            r"B/F/K10 训练完成后另做等规模 1:1 上 $R=1,\ldots,6$ 的循环轮数对照，与后续训练并行，不占训练槽。"
            "原版 REFIL 复用 [main v3](../main_v3/实验报告.md) seed 0；规则与随机锚点仍复用 v3。"
            "去 LN 的数量臂曲线与终评画在 [main v4](../main_v4/实验报告.md)，本目录主图仍是四条全局臂对照原版 REFIL。",
            "评估协议与 v3/v4 相同：1M 物理步、50 次池内验证（4 配置×25 局）、"
            f"终评 {len(FINAL_CONFIGS)} 个配置各 300 局，指标 $D$ 与 NDS。"
            "图上学习方法一律用 seed 0。奖励仍是 v3 默认的 damage 加距离塑形。",
            "训练的是红方，团队回报 $R=-D$。训练池 $N\\in\\{4,6,8,10\\}$、$K\\in\\{1,2,3\\}$ 均匀采样；"
            "验证只看池内 4 配置。外推分三条线：等规模 10/15/20/25/30/40v 同数，"
            "1:2 的 2v4/4v8/8v16/15v30/20v40，以及 10/15/20/30v 同数上的目标数 K=2/4/6。"
            "逐局与 loss 仍在同目录 CSV。图上不画随机，以免纵轴被拉开。", ""]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    behavior = _id_behavior(final_rows, episodes, methods=V5_METHODS)
    learning_figure = _plot_learning(plt, output, points, anchors, methods=V5_METHODS,
                                    include_random=False)
    results_figures = [path for path in _plot_final(
        plt, output, final_rows, anchors, methods=V5_METHODS, include_random=False) if path]
    depth_figure = _plot_cycle_depth(plt, output, episodes, methods=CYCLE_SERIES_METHODS)
    depth_scale_figure = _plot_cycle_depth_scale(plt, output, episodes, methods=CYCLE_SERIES_METHODS)
    behavior_figure = _plot_behavior(plt, output, behavior, methods=V5_METHODS,
                                    include_random=False)
    random_r, rule_r = _pool_return(anchors, "random"), _pool_return(anchors, "rule_nv1")
    last_r = {}
    for method in V5_METHODS:
        last = [row["D"] for row in points if row["method"] == method and row["eval_point"] == 50
                and row.get("seed") == 0]
        mean, std = _mean_std([-value for value in last])
        last_r[method] = (mean, std)
    intercept = {method: _mean_std([row["intercept"] for row in behavior.get(method, [])])
                 for method in ("random", *V5_METHODS, "rule_nv1")}
    scores = _final_seed_scores(final_rows, anchors, methods=V5_METHODS)
    text += ["## 结论", ""]
    ranked = [method for method in V5_METHODS if last_r[method][0] is not None]
    if random_r is not None and rule_r is not None and ranked:
        ranking = "；".join(
            f"{LABELS[method]} ${last_r[method][0]:.2f}$" for method in ranked)
        versus_rule = []
        for method in ranked:
            mean = last_r[method][0]
            if mean >= rule_r - 1e-6:
                versus_rule.append(f"{LABELS[method]}达到或超过规则")
            elif mean > random_r + 0.3:
                versus_rule.append(f"{LABELS[method]}高于随机、低于规则")
            else:
                versus_rule.append(f"{LABELS[method]}仍接近随机")
        pending = [LABELS[method] for method in V5_GLOBAL_METHODS
                   if last_r.get(method, (None,))[0] is None]
        text += [
            f"红方回报越高越好。训练池内随机约 ${random_r:.2f}$，规则约 ${rule_r:.2f}$。"
            f"已有 1M 验证回报的方法（seed 0）：{ranking}。最终评估用途中 best，不是曲线最右端。"
            + (f"尚未写结论的新臂：{'、'.join(pending)}。" if pending else ""),
            "；".join(versus_rule) + "。",
        ]
        known = [method for method in ("random", *V5_METHODS, "rule_nv1") if intercept[method][0] is not None]
        if {"random", "rule_nv1"} <= set(known) and any(intercept[method][0] is not None for method in V5_METHODS):
            learned_ix = "，".join(
                f"{LABELS[method]} {intercept[method][0]:.2f}"
                for method in V5_METHODS if intercept[method][0] is not None)
            text += [
                f"训练池内拦截率：规则 {intercept['rule_nv1'][0]:.2f}，随机 {intercept['random'][0]:.2f}，"
                f"{learned_ix}。",
            ]
        text += [""]
    else:
        text += ["记录尚未齐，暂不写结论。", ""]

    text += ["## 分析", ""] + _v5_analysis_text(scores, last_r, behavior, rule_r)

    text += ["## 训练曲线", "",
             "纵轴是红方整局回报 $R=-D$，与训练目标一致。每 2 万步在池内 4 配置上贪心 100 局。"
             "当前各学习方法只有 seed 0。REFIL 复用 v3。循环臂的曲线是评估 $R=4$ 的池内验证。"
             "点划线是规则基准。图上不画随机；随机数字仍在终评表里，并用于 NDS。", ""]
    if learning_figure:
        text += [f"![训练曲线]({learning_figure.relative_to(output).as_posix()})", ""]
    else:
        text += ["尚无足够记录绘制训练曲线。", ""]

    text += ["## 最终评估", "",
             "best 模型，每配置 300 局。三张图分别是等规模 1:1、红蓝 1:2、以及目标数扩充。"
             "当前为 seed 0。图上只留规则锚点；随机仍在表中。", ""]
    captions = ("等规模 1:1", "1:2 配比", "目标数（一格一个编队规模）")
    for path, caption in zip(results_figures, captions):
        text += [f"![{caption}]({path.relative_to(output).as_posix()})", ""]
    final_table = []
    for config in FINAL_CONFIGS:
        row = dict(config=config_label(config), axis=_config_axis(config))
        random, rule = anchors.get(("random", config), {}), anchors.get(("rule_nv1", config), {})
        row["random"] = _number(random.get("D")) if random.get("complete") else "—"
        row["rule"] = _number(rule.get("D")) if rule.get("complete") else "—"
        for method in V5_METHODS:
            values = list(scores[method][config].values())
            row[method] = _plus_minus([item["D"] for item in values])
            row[f"{method}_nds"] = _plus_minus([item["NDS"] for item in values if item["NDS"] is not None])
        final_table.append(row)
    if any(scores[method] for method in V5_METHODS) or anchors:
        text += _table(final_table, [
            ("config", "配置"), ("axis", "轴"), ("random", "随机 D"), ("rule", "规则 D"),
            *[(method, f"{LABELS[method]} D") for method in V5_METHODS],
        ]) + [""]
        text += _table(final_table, [
            ("config", "配置"), ("axis", "轴"),
            *[(f"{method}_nds", f"{LABELS[method]} NDS") for method in V5_METHODS],
        ]) + [""]
    else:
        text += ["最终评估尚无已完成配置。", ""]

    text += _v5_depth_section_lines(output, episodes, depth_figure, depth_scale_figure)

    text += ["## 行为诊断", "",
             "与训练曲线相同的 4 个训练池配置。左图：蓝方被拦下的比例，越高说明红方越像在防守。"
             "右图：红方死于同队碰撞的人数占比，越低越好。当前为 seed 0。图上不画随机。", ""]
    if behavior_figure:
        text += [f"![行为诊断]({behavior_figure.relative_to(output).as_posix()})", ""]
    else:
        text += ["尚无足够诊断记录。", ""]

    status_rows = []
    for method, seed in planned:
        row = progress.get((method, seed), {})
        plot_method = "refil_count_noln" if method == "refil_count" else method
        series = [item for item in all_points if item["method"] == plot_method and item["seed"] == seed]
        best = min(series, key=lambda item: (item["D"], item["t_env"])) if series else None
        status_label = row.get("status", "未开始")
        if row.get("phase") == "depth_eval" and row.get("total"):
            status_label = (f"depth_eval {int(row.get('completed') or 0)}/{int(row['total'])}"
                            + (f" R={int(row['cycle_depth'])}" if row.get("cycle_depth") is not None else ""))
        elif row.get("phase") == "final_eval" and row.get("total"):
            status_label = f"final_eval {int(row.get('completed') or 0)}/{int(row['total'])}"
        status_rows.append(dict(
            method=LABELS.get(plot_method, method), seed=seed, status=status_label,
            steps=row.get("t_env", 0), budget=row.get("budget_steps", 1000000),
            points=f"{len(series)}/50",
            best=_number(best["D"]) if best else "—",
            best_step=best["t_env"] if best else "—"))
    for method, label, note in (
        ("refil_count_ln", LABELS["refil_count_ln"], "复用 v4"),
        ("refil", LABELS["refil"], "复用 v3"),
    ):
        series = [item for item in all_points if item["method"] == method and item["seed"] == 0]
        best = min(series, key=lambda item: (item["D"], item["t_env"])) if series else None
        status_rows.append(dict(
            method=label, seed=0, status=note,
            steps=best["t_env"] if best else 0,
            budget=1000000, points=f"{len(series)}/50",
            best=_number(best["D"]) if best else "—",
            best_step=best["t_env"] if best else "—"))
    text += ["## 运行状态", ""]
    text += _table(status_rows, [("method", "方法"), ("seed", "种子"), ("status", "状态"),
                                 ("steps", "已训步"), ("budget", "预算"), ("points", "验证点"),
                                 ("best", "best 验证 D"), ("best_step", "best 步数")]) + [""]

    streams = ("episodes", "learning", "progress", "verification", "benchmarks", "trajectories")
    local_files = "；".join(f"[{name}.csv]({name}.csv)" for name in streams if (output / f"{name}.csv").exists())
    text += ["## 原始数据", "",
             (local_files + "。" if local_files else "")
             + "对照用的原版 REFIL 与锚点见 [main v3](../main_v3/)。"
             "去 LN 的 count 臂训练记录在本目录；带 LN 的对照图见 [main v4](../main_v4/)。", ""]
    report_path = output / "实验报告.md"
    content = "\n".join(text)
    if report_stream is None:
        _atomic_text(report_path, content)
    else:
        report_stream.seek(0)
        report_stream.write(content.encode("utf-8"))
        report_stream.truncate()
        report_stream.flush()
        os.fsync(report_stream.fileno())
    return report_path


MAIN_COMPARE = ("regir", "refil", "b2_qmix_atten", "dcg", "spectra", "alma")
MAIN_ABLATION = ("regir", "refil", "regir_norefil", "regir_nocount", "regir_r1", "regir_last")
MAIN_KEY_CONFIGS = ((5, 5, 2), (10, 10, 2), (30, 30, 2), (50, 50, 2), (10, 10, 12), (30, 30, 12))
POOL_CONFIGS = VALIDATION_CONFIGS
TARGET_DETAIL_CONFIGS = tuple((n, n, k) for n in (10, 15, 20, 30) for k in (4, 6)) + (
    (10, 10, 9), (10, 10, 12), (30, 30, 9), (30, 30, 12))


def _is_main_output(output):
    return Path(output).resolve().name == "main"


def _axis_sets():
    return (("池内", POOL_CONFIGS), ("等规模", EQUAL_SCALE_CONFIGS), ("目标", TARGET_DETAIL_CONFIGS))


def _usable_final_eval(rows, progress=None, output=None):
    """Keep final_eval rows that match the weights now on disk.

    Official identity is ``best.pt``'s ``t_env``, not the newest CSV tag. Mid-
    training leftovers such as ``best@280193`` stay on disk but never enter
    means, R=4 reuse, or figures.
    """
    progress = progress or {}
    output = Path(DEFAULT_OUTPUT if output is None else output)
    official = {}
    finished = {}
    for row in rows:
        if row.get("phase") != "final_eval" or row.get("method") is None:
            continue
        key = (row["method"], int(row["seed"]))
        if key in official:
            continue
        directory = run_directory(output, key[0], key[1])
        finished[key] = training_finished(directory, progress.get(key))
        official[key] = official_best_t_env(directory)
    kept = []
    for row in rows:
        if row.get("phase") != "final_eval":
            continue
        key = (row["method"], int(row["seed"]))
        if not finished.get(key):
            continue
        t_env = parse_checkpoint_t_env(row.get("checkpoint"))
        wanted = official.get(key)
        if wanted is None or t_env != wanted:
            continue
        kept.append(row)
    return kept


def _stale_final_eval_note(episodes, progress, methods, output=None):
    stale = []
    latest = {}
    output = Path(DEFAULT_OUTPUT if output is None else output)
    for row in episodes:
        if row.get("phase") != "final_eval" or row.get("method") not in methods:
            continue
        t_env = parse_checkpoint_t_env(row.get("checkpoint"))
        if t_env is None:
            continue
        key = (row["method"], int(row["seed"]))
        latest[key] = max(latest.get(key, -1), t_env)
    for key, t_env in sorted(latest.items()):
        method, seed = key
        official = _official_seed_t_env(output, progress, method, seed)
        if official is not None and t_env != official:
            stale.append(f"{LABELS.get(method, method)} seed {seed}（终评 `best@{t_env}`，磁盘 `best.pt` 是 {official}）")
    if not stale:
        return None
    return "下列种子的终评对不上磁盘上的最终权重，主表和深度图都不计入，正在按 `best.pt` 重评：" + "、".join(stale) + "。"


def _d_by_config(rows, methods):
    grouped = defaultdict(lambda: defaultdict(list))
    n_by = defaultdict(lambda: defaultdict(int))
    for row in rows:
        if row.get("phase") != "final_eval" or row.get("method") not in methods:
            continue
        if not _numeric(row.get("D")):
            continue
        cfg = config_key(row["config"])
        grouped[row["method"]][cfg].append(float(row["D"]))
        n_by[row["method"]][cfg] += 1
    return grouped, n_by


def _nds_d(d, config, anchors):
    random = anchors.get(("random", config), {})
    rule = anchors.get(("rule_nv1", config), {})
    if not random.get("complete") or not rule.get("complete"):
        return None
    return compute_nds(d, random["D"], rule["D"])


def _axis_mean_d(grouped, method, configs, anchors):
    values, nds_values, counts = [], [], []
    for cfg in configs:
        series = grouped.get(method, {}).get(cfg)
        if not series:
            continue
        d = statistics.mean(series)
        values.append(d)
        counts.append(len(series))
        nds = _nds_d(d, cfg, anchors)
        if nds is not None:
            nds_values.append(nds)
    if not values:
        return None, None, 0
    return statistics.mean(values), (statistics.mean(nds_values) if nds_values else None), min(counts)


def _plot_equal_d(plt, output, grouped, anchors, methods, stem="scale_equal",
                  include_random=True):
    fig, axis = plt.subplots(1, 1, figsize=(7.4, 3.9), constrained_layout=True)
    xs = list(range(len(EQUAL_SCALE_CONFIGS)))
    drawn = False
    for method in methods:
        points = []
        for index, cfg in enumerate(EQUAL_SCALE_CONFIGS):
            series = grouped.get(method, {}).get(cfg)
            if series:
                points.append((index, statistics.mean(series)))
        if points:
            axis.plot([p[0] for p in points], [p[1] for p in points], marker="o",
                      color=COLORS.get(method, "#444"), linewidth=2.0, label=LABELS.get(method, method))
            drawn = True
    for policy, style in _anchor_plot_styles(include_random):
        ys = []
        for cfg in EQUAL_SCALE_CONFIGS:
            row = anchors.get((policy, cfg), {})
            ys.append(row["D"] if row.get("n") else None)
        xs_ok = [i for i, y in enumerate(ys) if y is not None]
        if xs_ok:
            axis.plot(xs_ok, [ys[i] for i in xs_ok], linestyle=style, marker="s",
                      color=COLORS[policy], label=LABELS[policy])
            drawn = True
    axis.set(title="Equal-scale damage (lower is better)", xlabel="Roster", ylabel="D",
             xticks=xs, xticklabels=[config_label(cfg) for cfg in EQUAL_SCALE_CONFIGS])
    axis.tick_params(axis="x", labelrotation=30, labelsize=8)
    axis.grid(alpha=.2)
    if drawn:
        axis.legend(fontsize=8)
    path = output / "figures" / f"{stem}.png"
    _save_figure(fig, path)
    plt.close(fig)
    return path


def _plot_target_d(plt, output, grouped, anchors, methods, stem="scale_targets",
                   include_random=True):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6), sharey=True, constrained_layout=True)
    ks = (2, 4, 6, 9, 12)
    drawn = False
    for axis, n in zip(axes, TARGET_PLOT_NS):
        for method in methods:
            points = []
            for k in ks:
                series = grouped.get(method, {}).get((n, n, k))
                if series:
                    points.append((k, statistics.mean(series)))
            if points:
                axis.plot([p[0] for p in points], [p[1] for p in points], marker="o",
                          color=COLORS.get(method, "#444"), linewidth=2.0, label=LABELS.get(method, method))
                drawn = True
        for policy, style in _anchor_plot_styles(include_random):
            points = []
            for k in ks:
                row = anchors.get((policy, (n, n, k)), {})
                if row.get("n"):
                    points.append((k, row["D"]))
            if points:
                axis.plot([p[0] for p in points], [p[1] for p in points], linestyle=style,
                          marker="s", color=COLORS[policy], label=LABELS[policy])
                drawn = True
        axis.set(title=f"{n}v{n}", xlabel="K", xticks=list(ks))
        axis.grid(alpha=.2)
        if axis is axes[0] and drawn:
            axis.legend(fontsize=7)
    fig.supylabel("D")
    path = output / "figures" / f"{stem}.png"
    _save_figure(fig, path)
    plt.close(fig)
    return path


ABLATION_WHY = {
    "regir": ("全模型", "其余消融的参照"),
    "refil": ("去掉整条全局循环旁路", "循环模块整体有没有用"),
    "regir_norefil": ("去掉原 REFIL 实体注意力和想象分组，个体表征只来自循环读取 + GRU",
                      "循环能否脱离 REFIL 独立工作"),
    "regir_nocount": ("关掉循环内数量编码对 query 的注入", "收益是不是换皮的数量条件"),
    "regir_r1": ("训练和终评都固定 $R=1$", "主收益是不是多轮 / 随机深度训练"),
    "regir_last": ("训练和执行都只读最后一轮，不做跨轮加权", "跨轮表征库有没有独立贡献"),
}
STATUS_METHODS = tuple(dict.fromkeys((*MAIN_COMPARE, *MAIN_ABLATION, "refil_matched")))


def _axis_d(grouped, method, configs, anchors):
    return _axis_mean_d(grouped, method, configs, anchors)[0]


def _rank_methods(grouped, configs, methods, anchors):
    ranked = []
    for method in methods:
        mean, nds, nmin = _axis_mean_d(grouped, method, configs, anchors)
        if mean is not None:
            ranked.append((mean, method, nds, nmin))
    ranked.sort()
    return ranked


def _seed_phrase(seeds_by_method, method):
    n = len(seeds_by_method.get(method, ()))
    if not n:
        return "尚无可用终评种子"
    if n < len(FORMAL_SEEDS):
        return f"仅 {n} 个可用终评种子（表中 `s={n}`）"
    return f"{n} 个终评种子"


def _load_mechanism(output, run=FORMAL_RUN):
    """Stream trajectories.csv and keep the first ReGIR mechanism episode."""
    path = Path(output) / "trajectories.csv"
    if not path.exists():
        return None
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    from open_score.utils.logging import _decode_row
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if "mechanism" not in (row.get("phase") or ""):
                continue
            try:
                decoded = _decode_row(row, path)
            except (ValueError, json.JSONDecodeError):
                continue
            if decoded.get("phase") != "mechanism":
                continue
            if run is not None and decoded.get("run") != run:
                continue
            if decoded.get("method") != "regir":
                continue
            payload = decoded.get("trajectory")
            if not isinstance(payload, dict):
                continue
            attention = payload.get("attention") or {}
            if not attention.get("round_self_attn"):
                continue
            return dict(decoded, attention=attention, frames=payload.get("frames") or [])
    return None


def _mean_square(batch):
    if not batch or not isinstance(batch[0], list):
        return None
    n_batch = len(batch)
    rows, cols = len(batch[0]), len(batch[0][0])
    acc = [[0.0] * cols for _ in range(rows)]
    for item in batch:
        for i, row in enumerate(item):
            for j, value in enumerate(row):
                acc[i][j] += float(value)
    return [[value / n_batch for value in row] for row in acc]


def _plot_mechanism(plt, output, payload):
    if not payload:
        return None, None
    attention = payload["attention"]
    frames = payload.get("frames") or []
    step = int(attention.get("step") or max(len(frames) // 2, 0))
    frame = frames[min(step, len(frames) - 1)] if frames else {}
    reds = frame.get("red") or []
    blues = frame.get("blue") or []
    targets = frame.get("targets") or []
    n_red, n_blue, n_tgt = len(reds), len(blues), len(targets)
    live = n_red + n_blue + n_tgt
    rounds = [_mean_square(item) for item in attention.get("round_self_attn") or []]
    rounds = [item for item in rounds if item]
    if not rounds or live <= 0 or not reds:
        return None, None
    observer = 0
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.4), constrained_layout=True)
    for axis, weights, depth in zip(axes.ravel(), rounds, range(1, 5)):
        crop = [row[:live] for row in weights[:live]]
        row = crop[observer] if observer < len(crop) else crop[0]
        ranked = sorted(enumerate(row), key=lambda item: item[1], reverse=True)[:8]
        for unit, color, marker in (
            (reds, "#1b7f7a", "o"), (blues, "#c4473a", "s"), (targets, "#c47b2b", "*"),
        ):
            xs = [item["position"][0] for item in unit]
            ys = [item["position"][1] for item in unit]
            if xs:
                axis.scatter(xs, ys, c=color, s=22 if marker != "*" else 70, marker=marker, zorder=2)
        ox, oy = reds[observer]["position"][:2]
        axis.scatter([ox], [oy], c="#111", s=46, marker="o", zorder=3)
        for index, weight in ranked:
            if index == observer:
                continue
            if index < n_red:
                dest = reds[index]["position"]
            elif index < n_red + n_blue:
                dest = blues[index - n_red]["position"]
            else:
                dest = targets[index - n_red - n_blue]["position"]
            axis.annotate("", xy=(dest[0], dest[1]), xytext=(ox, oy),
                          arrowprops=dict(arrowstyle="->", color="#333",
                                          alpha=min(1.0, 0.25 + 2.5 * weight), lw=1.1))
        axis.set(title=f"R={depth}  observer=red0", xlabel="x", ylabel="y")
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=.15)
    path = output / "figures" / "mechanism_attention.png"
    _save_figure(fig, path)
    plt.close(fig)

    alpha = attention.get("alpha") or []
    alpha_path = None
    if alpha and isinstance(alpha[0], list):
        active = [row for row in alpha if sum(float(value) for value in row) > 0.05]
        shown = active or alpha
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), constrained_layout=True)
        axes[0].imshow(shown, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        axes[0].set(title="Cross-round α (active observers × R)", xlabel="Round",
                    ylabel="Observer", xticks=list(range(len(shown[0]))))
        means = [statistics.mean(row[j] for row in alpha) for j in range(len(alpha[0]))]
        axes[1].bar(list(range(1, len(means) + 1)), means, color="#1b7f7a")
        axes[1].set(title="Mean α per round", xlabel="Round", ylabel="α",
                    xticks=list(range(1, len(means) + 1)), ylim=(0, 1))
        axes[1].grid(axis="y", alpha=.2)
        alpha_path = output / "figures" / "mechanism_alpha.png"
        _save_figure(fig, alpha_path)
        plt.close(fig)
    return path, alpha_path


def _mechanism_alpha_means(payload):
    alpha = (payload or {}).get("attention", {}).get("alpha") or []
    if not alpha or not isinstance(alpha[0], list):
        return []
    return [statistics.mean(row[j] for row in alpha) for j in range(len(alpha[0]))]


def _compare_reading(grouped, anchors, seeds_by_method):
    lines = [
        "对照回答假设 H1：只在 $N\\in\\{4,6,8,10\\}$、$K\\le 3$ 上训练，冻结后零样本部署，"
        "ReGIR 的漏防 $D$ 是否低于母体 REFIL、其他学习方法和规则。",
        "怎么看：纵轴 $D$ 越低红方越好。NDS 把同一配置上的随机当作 0、规则当作 1，"
        "用来避免「大编队绝对 $D$ 下降其实是场景变易」。括号里是 NDS；`s=` 表示计入均值的终评种子数，"
        "少于 3 表示还不能当多种子结论。",
    ]
    equal = _rank_methods(grouped, EQUAL_SCALE_CONFIGS, MAIN_COMPARE, anchors)
    if not equal:
        lines.append("等规模终评尚未齐，下面先看已有训练曲线和已完成方法。")
        return lines
    best_d, best, best_nds, _ = equal[0]
    names = "，".join(
        f"{LABELS[method]} {value:.3f}" + (f"（NDS {nds:.2f}）" if nds is not None else "")
        for value, method, nds, _ in equal)
    lines.append(f"等规模 1:1 轴当前排序（平均 $D$）：{names}。最低是 {LABELS[best]}。")
    regir = next((item for item in equal if item[1] == "regir"), None)
    refil = next((item for item in equal if item[1] == "refil"), None)
    rule_vals = [anchors[("rule_nv1", cfg)]["D"] for cfg in EQUAL_SCALE_CONFIGS
                 if anchors.get(("rule_nv1", cfg), {}).get("complete")]
    rule = statistics.mean(rule_vals) if rule_vals else None
    if regir and refil:
        gap = refil[0] - regir[0]
        lines.append(
            f"ReGIR 等规模 $D={regir[0]:.3f}$（{_seed_phrase(seeds_by_method, 'regir')}），"
            f"REFIL 为 ${refil[0]:.3f}$，相差 {gap:+.3f}。"
            + (f"规则同轴约 ${rule:.3f}$。" if rule is not None else "")
            + (" 若 ReGIR 只有 seed 0，这是单种子快照，seed 1/2 在 1M 权重上重评之前不要写成多种子结论。"
               if len(seeds_by_method.get("regir", ())) < 3 else ""))
    pool = _rank_methods(grouped, POOL_CONFIGS, MAIN_COMPARE, anchors)
    target = _rank_methods(grouped, TARGET_DETAIL_CONFIGS, MAIN_COMPARE, anchors)
    if pool:
        lines.append("池内（训练见过的 4 配置）最低是 "
                     f"{LABELS[pool[0][1]]} $D={pool[0][0]:.3f}$。")
    if target:
        lines.append("目标数外推（更高 $K$）最低是 "
                     f"{LABELS[target[0][1]]} $D={target[0][0]:.3f}$。")
    return lines


def _latest_train_eval(episodes):
    last = {}
    for row in episodes:
        if row.get("phase") != "train_eval" or not _numeric(row.get("D")):
            continue
        key = (row["method"], int(row["seed"]))
        point = int(row.get("eval_point") or 0)
        t_env = int(row.get("t_env") or 0)
        prev = last.get(key)
        if prev is None or point > prev["eval_point"] or (
                point == prev["eval_point"] and t_env >= prev["t_env"]):
            last[key] = dict(eval_point=point, t_env=t_env, D=float(row["D"]))
    return last


def _ablation_eval_status(episodes, progress):
    """One row per ablation seed: training vs the newest final_eval tag."""
    latest = {}
    n_ep = defaultdict(int)
    tags = {}
    for row in episodes:
        if row.get("phase") != "final_eval" or row.get("method") not in MAIN_ABLATION:
            continue
        t_env = parse_checkpoint_t_env(row.get("checkpoint"))
        if t_env is None:
            continue
        key = (row["method"], int(row["seed"]))
        prev = latest.get(key, -1)
        if t_env > prev:
            latest[key] = t_env
            n_ep[key] = 0
            tags[key] = row.get("checkpoint")
        if t_env == latest.get(key):
            n_ep[key] += 1
            tags[key] = row.get("checkpoint")
    val = _latest_train_eval(episodes)
    rows = []
    for method in MAIN_ABLATION:
        if method in ("regir", "refil"):
            continue
        for seed in FORMAL_SEEDS:
            key = (method, seed)
            trained = progress.get(key) or {}
            trained_t = trained.get("t_env")
            status = str(trained.get("status") or "—")
            t_eval = latest.get(key)
            finished = status in ("complete", "completed") or (
                trained_t is not None and int(trained_t) >= 980000)
            stale = (trained_t is not None and t_eval is not None
                     and t_eval + 20000 < int(trained_t))
            if t_eval is None:
                verdict = "尚无终评"
            elif not finished:
                verdict = "训练未完，不进表"
            elif stale:
                verdict = "过期，不进表"
            else:
                verdict = "计入"
            pool = val.get(key)
            rows.append(dict(
                method=LABELS.get(method, method), seed=seed,
                train=("—" if trained_t is None else
                       f"{int(trained_t)}" + ("" if finished else f" {status}")),
                checkpoint=tags.get(key) or "—",
                n=n_ep.get(key) or 0,
                pool=("—" if pool is None else f"{pool['D']:.3f} @{pool['t_env']}"),
                verdict=verdict,
            ))
    return rows


def _ablation_reading(grouped, anchors, seeds_by_method, progress):
    lines = [
        "消融回答假设 H2：把 ReGIR 相对 REFIL 的增益拆成四个可独立关掉的部件。"
        "ReGIR 与 REFIL 复用对照实验，不另训。消融表只看 $D$，不再报 NDS。",
        "各臂相对全模型只改一件事：",
    ]
    for method in MAIN_ABLATION:
        change, question = ABLATION_WHY[method]
        lines.append(f"- **{LABELS[method]}**（`{method}`）：{change}。要问的是：{question}。")
    ref = _axis_d(grouped, "regir", EQUAL_SCALE_CONFIGS, anchors)
    bits = []
    missing = []
    for method in MAIN_ABLATION:
        value = _axis_d(grouped, method, EQUAL_SCALE_CONFIGS, anchors)
        if value is None:
            states = []
            for seed in FORMAL_SEEDS:
                row = progress.get((method, seed), {})
                if row.get("status"):
                    states.append(f"s{seed} {row.get('status')}@{row.get('t_env') or '—'}")
            missing.append(f"{LABELS[method]}" + (f"（{ '，'.join(states) }）" if states else "（尚无可用终评）"))
            continue
        extra = _seed_phrase(seeds_by_method, method)
        if ref is None or method == "regir":
            bits.append(f"{LABELS[method]} 等规模 $D={value:.3f}$（{extra}）")
        else:
            bits.append(f"{LABELS[method]} 等规模 $D={value:.3f}$，相对全模型 {value - ref:+.3f}（{extra}）")
    if bits:
        lines.append("当前可用终评：" + "；".join(bits) + "。关掉某一部件后 $D$ 明显升高，说明那一部件有贡献。")
    if missing:
        lines.append("还不能进消融均值的臂：" + "、".join(missing) + "。")
        lines.append("这不是「没做评估」。四个变体大多已经跑过 24×300 终评，但 `best.pt` 一出现就开跑，"
                     "checkpoint 停在十几万到四十万步；和 ReGIR seed 1/2 同类。那些局的 $D$ 会到 5–16，"
                     "一旦进均值，消融会看起来全面崩盘。等规模/目标图因此只画已对齐的 ReGIR / REFIL。"
                     "下面「终评对齐」表列出每条种子的终评标签；池内验证 $D$ 来自训练曲线最右端，可以看，不能当外推。")
    if ref is not None:
        last = _axis_d(grouped, "regir_last", EQUAL_SCALE_CONFIGS, anchors)
        r1 = _axis_d(grouped, "regir_r1", EQUAL_SCALE_CONFIGS, anchors)
        nocount = _axis_d(grouped, "regir_nocount", EQUAL_SCALE_CONFIGS, anchors)
        norefil = _axis_d(grouped, "regir_norefil", EQUAL_SCALE_CONFIGS, anchors)
        if last is not None and last <= ref + 0.05:
            lines.append("ReGIR-last 已接近全模型，跨轮读取的独立贡献目前偏弱。")
        if r1 is not None and r1 <= ref + 0.05:
            lines.append("ReGIR-R1 已接近全模型，还不能把主收益写成多轮循环。")
        if nocount is not None and nocount <= ref + 0.05:
            lines.append("关掉 $z_n$ 几乎不伤，数量注入不是当前增益的主要来源。")
        if norefil is not None and norefil <= ref + 0.05:
            lines.append("去掉 REFIL 路径后仍接近全模型，循环旁路可以单独工作。")
    return lines


def _depth_reading(cells):
    by_depth = (cells or {}).get("regir") or {}
    if not by_depth:
        return ["循环轮数扫描还没有满 300 局的 1:1 格子。"]
    means = []
    for depth in DEPTH_SWEEP_DEPTHS:
        values = [by_depth.get(depth, {}).get(cfg) for cfg in EQUAL_SCALE_CONFIGS]
        values = [value for value in values if value is not None]
        if values:
            means.append((depth, statistics.mean(values)))
    if not means:
        return ["循环轮数扫描还没有可平均的格子。"]
    best_depth, best_d = min(means, key=lambda item: item[1])
    r4 = next((item[1] for item in means if item[0] == 4), None)
    spread = max(item[1] for item in means) - min(item[1] for item in means)
    large = []
    for cfg in EQUAL_SCALE_CONFIGS[-3:]:
        series = [(depth, by_depth.get(depth, {}).get(cfg)) for depth in DEPTH_SWEEP_DEPTHS]
        series = [(depth, value) for depth, value in series if value is not None]
        if len(series) >= 2:
            shallow_vals = [value for depth, value in series if depth <= 2]
            deep_vals = [value for depth, value in series if depth >= 4]
            if shallow_vals and deep_vals:
                large.append((cfg, min(shallow_vals), min(deep_vals)))
    lines = [
        "轮数实验只改冻结 ReGIR 的执行 $R$，检验假设 H5：大规模部署是否仍受益于更多轮。"
        "主对照终评固定 $R=4$，不用事后挑最好的 $R$。",
        f"当前 1:1 八个规模上，平均 $D$ 最低的是 $R={best_depth}$（{best_d:.3f}）。"
        + (f"约定部署的 $R=4$ 为 {r4:.3f}。" if r4 is not None else "")
        + f"各 $R$ 均值相差 {spread:.3f}。",
    ]
    if spread < 0.08:
        lines.append("各 $R$ 几乎重叠，还不能写成「规模越大越要加深」。$R=4$ 只是协议约定，不是这条曲线上的异常点。")
    if large:
        worse = sum(1 for _, shallow, deep in large if deep + 0.02 < shallow)
        if worse == 0:
            lines.append("40–50 人上加深相对 $R\\le 2$ 没有稳定降伤，H5 的「大规模需要更深」目前不成立。")
        else:
            lines.append("大编队上 $R\\ge 4$ 相对浅轮有降伤，和 H5 同向。")
    lines.append("只计入已经做过深度扫描、且终评 `best@t_env` 与该扫描相同的种子；"
                 "别的种子的中途终评不再画进 $R=4$。")
    return lines


def _mechanism_reading(payload):
    if not payload:
        return ["机制探测（实验 D）还没有可画的注意力局。冻结 ReGIR seed 0 的 `best.pt` 后，"
                "应在 30v30 K2 上记下一局逐轮自注意力和跨轮 $\\alpha$。"]
    means = _mechanism_alpha_means(payload)
    step = (payload.get("attention") or {}).get("step")
    seed = (payload.get("attention") or {}).get("episode_seed") or payload.get("episode_seed")
    lines = [
        "机制图来自冻结 ReGIR seed 0、30v30 K2 的一局（环境种子 "
        f"{seed}，取第 {step} 步）。检验 H3：后轮是否从散看全场变成盯住突击蓝机或受威胁目标；"
        "以及 H4：个体是否读取不同轮次，而不是全体塌到最后一轮。",
    ]
    if means:
        last = means[-1]
        top = max(range(len(means)), key=lambda i: means[i]) + 1
        pretty = "，".join(f"$R={i+1}$ {value:.2f}" for i, value in enumerate(means))
        lines.append(f"这一步全体观察者的平均 $\\alpha$：{pretty}。质量最大的是 $R={top}$。")
        if last >= 0.75:
            lines.append("平均 $\\alpha$ 大部分钉在最后一轮，H4 偏弱；若消融 ReGIR-last 也接近全模型，跨轮库的独立贡献就更弱。")
        elif max(means) <= 0.45:
            lines.append("各轮 $\\alpha$ 比较分散，个体确实在用不同深度，和 H4 同向。")
        else:
            lines.append("最后一轮有权重，但没有独占，跨轮读取不是摆设。")
    lines.append("注意力图里青点是红方、红方块是蓝方、星号是目标；箭头从同一名红方观察者指向该轮权重最高的实体。"
                 "若四张图几乎一样，H3（逐步凝练）不成立。")
    return lines


def _refresh_main_report(output=DEFAULT_OUTPUT, *, run=FORMAL_RUN, report_stream=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    episodes = unique_episodes(read_records(output, "episodes", run=run))
    progress = {(method, seed): row for (method, recorded_run, seed), row in
                read_latest(output, "progress", run=run).items()}
    anchors = _anchor_values(episodes)
    points = [row for row in _validation_points(episodes, anchors) if row["method"] in MAIN_COMPARE + MAIN_ABLATION]
    final_rows = _usable_final_eval(episodes, progress, output=output)
    stale_note = _stale_final_eval_note(episodes, progress, set(MAIN_COMPARE + MAIN_ABLATION + ("refil_matched",)), output=output)
    grouped, n_by = _d_by_config(final_rows, set(MAIN_COMPARE + MAIN_ABLATION + ("refil_matched",)))
    seeds_by_method = defaultdict(set)
    for row in final_rows:
        if row.get("method"):
            seeds_by_method[row["method"]].add(int(row["seed"]))
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    learning = _plot_learning(plt, output, points, anchors, methods=MAIN_COMPARE, include_random=True)
    equal_fig = _plot_equal_d(plt, output, grouped, anchors, MAIN_COMPARE)
    target_fig = _plot_target_d(plt, output, grouped, anchors, MAIN_COMPARE)
    ablation_learning = _plot_learning(plt, output, points, anchors, methods=MAIN_ABLATION,
                                      include_random=False, stem="ablation_learning")
    ablation_equal = _plot_equal_d(plt, output, grouped, anchors, MAIN_ABLATION,
                                  stem="ablation_equal", include_random=False)
    ablation_target = _plot_target_d(plt, output, grouped, anchors, MAIN_ABLATION,
                                    stem="ablation_targets", include_random=False)
    depth_fig = _plot_cycle_depth(plt, output, episodes, methods=("regir",), progress=progress)
    depth_scale_fig = _plot_cycle_depth_scale(plt, output, episodes, methods=("regir",), progress=progress)
    mechanism = _load_mechanism(output, run)
    mech_attn_fig, mech_alpha_fig = _plot_mechanism(plt, output, mechanism)

    def _seed_note(method):
        nseeds = len(seeds_by_method.get(method, ()))
        return f" s={nseeds}" if nseeds and nseeds < len(FORMAL_SEEDS) else ""

    def cell_d(method, cfg):
        series = grouped.get(method, {}).get(cfg)
        if not series:
            return "—"
        d = statistics.mean(series)
        n = len(series)
        nds = _nds_d(d, cfg, anchors)
        note = f" n={n}" if n < FINAL_EPISODES_PER_CONFIG else ""
        note += _seed_note(method)
        if nds is None:
            return f"{d:.3f}{note}"
        return f"{d:.3f} ({nds:.2f}){note}"

    def axis_cell(method, configs):
        mean, nds, nmin = _axis_mean_d(grouped, method, configs, anchors)
        if mean is None:
            return "—"
        extra = f" n≥{nmin}" if nmin < FINAL_EPISODES_PER_CONFIG else ""
        extra += _seed_note(method)
        if nds is None:
            return f"{mean:.3f}{extra}"
        return f"{mean:.3f} ({nds:.2f}){extra}"

    table1 = []
    for method in (*MAIN_COMPARE, "rule_nv1"):
        if method == "rule_nv1":
            row = {"method": LABELS[method]}
            for name, configs in _axis_sets():
                values = [anchors[(method, cfg)]["D"] for cfg in configs if anchors.get((method, cfg), {}).get("complete")]
                row[name] = f"{statistics.mean(values):.3f}" if values else "—"
            table1.append(row)
            continue
        table1.append({"method": LABELS.get(method, method),
                       "池内": axis_cell(method, POOL_CONFIGS),
                       "等规模": axis_cell(method, EQUAL_SCALE_CONFIGS),
                       "目标": axis_cell(method, TARGET_DETAIL_CONFIGS)})
    table2 = []
    for method in MAIN_ABLATION:
        table2.append({"method": LABELS.get(method, method),
                       "池内": axis_cell(method, POOL_CONFIGS),
                       "等规模": axis_cell(method, EQUAL_SCALE_CONFIGS),
                       "目标": axis_cell(method, TARGET_DETAIL_CONFIGS)})
    report_seeds = list(FORMAL_SEEDS)
    extra_started = any(progress.get((method, seed))
                        for method in STATUS_METHODS
                        for seed in (3, 4))
    if extra_started:
        report_seeds.extend((3, 4))
    table3_train = []
    for method in STATUS_METHODS:
        for seed in report_seeds:
            row = progress.get((method, seed), {})
            table3_train.append(dict(
                method=LABELS.get(method, method), seed=seed,
                status=row.get("status", "—"),
                steps=row.get("t_env"),
                val=row.get("latest_validation_D"),
                final=sum(1 for cfg in FINAL_CONFIGS if n_by.get(method, {}).get(cfg, 0) >= FINAL_EPISODES_PER_CONFIG)
                if seed == 0 else "—"))
    table3_key = []
    for method in (*MAIN_COMPARE, "rule_nv1"):
        row = {"method": LABELS.get(method, method)}
        for cfg in MAIN_KEY_CONFIGS:
            row[config_label(cfg)] = cell_d(method, cfg) if method != "rule_nv1" else (
                f"{anchors[(method, cfg)]['D']:.3f}" if anchors.get((method, cfg), {}).get("n") else "—")
        table3_key.append(row)
    cells = _depth_cell_scores(episodes, ("regir",), progress=progress)
    compare_lines = _compare_reading(grouped, anchors, seeds_by_method)
    ablation_lines = _ablation_reading(grouped, anchors, seeds_by_method, progress)
    ablation_status = _ablation_eval_status(episodes, progress)
    depth_lines = _depth_reading(cells)
    mechanism_lines = _mechanism_reading(mechanism)

    def _figure(path, caption):
        return [f"![{caption}](figures/{path.name})", ""] if path else []

    text = ["# 跨规模攻防泛化 main", "",
            f"更新时间：{now}。主方法 **ReGIR**（代码键 `regir`）。"
            "训练只见 $N\\in\\{4,6,8,10\\}$、$K\\le 3$；终评 24 配置各 300 局贪心。"
            "主指标是漏防伤害 $D$（越低越好）；对照和外推另报规则归一化 NDS（0=随机，1=规则，大于 1 优于规则）。",
            "报告按实验分章：对照、消融、机制。训练或评估一有新数据就重算图、表和下面这些说明。"
            "未满 300 局的格带 `n=`；终评种子不足 3 个带 `s=`。档案臂、1:2 旧局、未完成训练和过期中途终评不进均值。", ""]

    text += ["## 一、对照实验", ""]
    text += [line for line in compare_lines] + [""]
    text += ["### 训练曲线", "",
             "池内 4 配置、每 2 万步贪心 100 局，纵轴是红方回报 $R=-D$（越高越好）。"
             "阴影是已有训练种子的标准差。点划线是规则，虚线是随机。曲线看的是训练过程，终评用的是 `best.pt`，不是曲线最右端。", ""]
    text += _figure(learning, "对照：池内训练曲线")
    text += ["### 等规模外推", "",
             "横轴是未见或插值的 1:1 编队（5 到 50 人，$K=2$），纵轴 $D$。"
             "5v5 夹在训练池 4 与 6 之间，是插值；15–50 人是外推。这张图是 H1 跨规模部分的主证据。", ""]
    text += _figure(equal_fig, "对照：等规模外推")
    text += ["### 目标数外推", "",
             "左 10v10、右 30v30，横轴 $K=2,4,6,9,12$。训练最多见过 $K=3$，所以 $K\\ge 4$ 都是外推。"
             "10v10 K12 上红方极度不够分目标，$D$ 会很高；比较的是方法之间的差距，不是接近零伤。", ""]
    text += _figure(target_fig, "对照：目标数外推")
    text += ["### 三类平均", "",
             "行是方法，列是池内 / 等规模 / 目标三类配置的平均 $D$，括号是 NDS。", ""]
    text += _table(table1, (("method", "方法"), ("池内", "池内"), ("等规模", "等规模"), ("目标", "目标"))) + [""]
    text += ["关键配置（一眼看插值 5v5、训练附近 10v10、外推 30/50 以及高目标数）：", ""]
    key_cols = [("method", "方法")] + [(config_label(cfg), config_label(cfg)) for cfg in MAIN_KEY_CONFIGS]
    text += _table(table3_key, key_cols) + [""]
    text += ["### 逐格明细", "", "池内：", ""]
    detail = []
    for method in MAIN_COMPARE:
        row = {"method": LABELS.get(method, method)}
        for cfg in FINAL_CONFIGS:
            row[config_label(cfg)] = cell_d(method, cfg)
        detail.append(row)
    text += _table([{"method": row["method"], **{config_label(cfg): row[config_label(cfg)] for cfg in POOL_CONFIGS}}
                    for row in detail],
                   (("method", "方法"),) + tuple((config_label(cfg), config_label(cfg)) for cfg in POOL_CONFIGS)) + [""]
    text += ["等规模：", ""]
    text += _table([{"method": row["method"], **{config_label(cfg): row[config_label(cfg)] for cfg in EQUAL_SCALE_CONFIGS}}
                    for row in detail],
                   (("method", "方法"),) + tuple((config_label(cfg), config_label(cfg)) for cfg in EQUAL_SCALE_CONFIGS)) + [""]

    text += ["## 二、消融实验", ""]
    text += ablation_lines + [""]
    text += ["### 消融训练曲线", "",
             "与对照同一套池内验证。若某条臂的终评还不能进表，仍可以从这条曲线看它在训练池里有没有学会。", ""]
    text += _figure(ablation_learning, "消融：池内训练曲线")
    text += ["### 消融等规模与目标数", "",
             "读法与对照章相同。某条臂若整段高于全模型，关掉的那一部件就有贡献。", ""]
    text += _figure(ablation_equal, "消融：等规模外推")
    text += _figure(ablation_target, "消融：目标数外推")
    text += ["### 终评对齐", "",
             "四个变体的终评局已经在磁盘上，但标签对不上最终权重，所以官方三类平均是 —。"
             "「池内验证 $D$」是 `train_eval` 最后一点，不是 300 局终评。", ""]
    text += _table(ablation_status, (
        ("method", "方法"), ("seed", "seed"), ("train", "训练步数"),
        ("checkpoint", "已有终评"), ("n", "终评局"), ("pool", "池内验证 D"),
        ("verdict", "进均值"),
    )) + [""]
    text += ["### 消融三类平均", "",
             "ReGIR / REFIL 与对照章同源。格子为 — 表示还没有可用的最终权重终评。", ""]
    text += _table(table2, (("method", "方法"), ("池内", "池内"), ("等规模", "等规模"), ("目标", "目标"))) + [""]

    text += ["## 三、机制评估", ""]
    text += ["### 循环轮数", ""]
    text += depth_lines + [""]
    text += _figure(depth_fig, "机制：等规模平均回报随 R")
    text += _figure(depth_scale_fig, "机制：D 随规模和 R")
    if "regir" in cells:
        scale_table = []
        for depth in DEPTH_SWEEP_DEPTHS:
            row = dict(R=f"$R={depth}$")
            for cfg in EQUAL_SCALE_CONFIGS:
                value = cells["regir"].get(depth, {}).get(cfg)
                row[config_label(cfg)] = _number(value, 4) if value is not None else "—"
            scale_table.append(row)
        text += ["行是执行轮数，列是 1:1 规模，格子是平均 $D$：", ""]
        text += _table(scale_table, [("R", "$R$"),
                                     *[(config_label(cfg), config_label(cfg))
                                       for cfg in EQUAL_SCALE_CONFIGS]]) + [""]
    text += ["### 一局注意力与跨轮读取", ""]
    text += mechanism_lines + [""]
    text += _figure(mech_attn_fig, "机制：同一时刻四轮自注意力")
    text += _figure(mech_alpha_fig, "机制：跨轮读取 α")

    text += ["## 四、运行状态", "",
             "每个方法×种子的训练步和池内验证 $D$。终评满格只在 seed 0 行统计该方法已满 300 局的配置数。", ""]
    text += _table(table3_train, (("method", "方法"), ("seed", "种子"), ("status", "状态"),
                                  ("steps", "步数"), ("val", "验证 D"), ("final", "终评满格"))) + [""]
    if stale_note:
        text += [stale_note, ""]
    text += ["档案臂、1:2 旧局和未完成训练不进入上表均值。", ""]
    content = "\n".join(text) + "\n"
    report_path = output / "实验报告.md"
    if report_stream is None:
        _atomic_text(report_path, content)
    else:
        report_stream.seek(0)
        report_stream.write(content.encode("utf-8"))
        report_stream.truncate()
        report_stream.flush()
        os.fsync(report_stream.fileno())
    return report_path


def refresh_report(output=DEFAULT_OUTPUT, *, run=FORMAL_RUN):
    """Render the one report of this version; a side run never replaces it."""
    output = Path(output)
    try:
        with _report_lock(output) as stream:
            formal = run if run == FORMAL_RUN else FORMAL_RUN
            if _is_probe_output(output):
                return _refresh_probe_report(output, run=formal, report_stream=stream)
            if _is_main_output(output):
                return _refresh_main_report(output, run=formal, report_stream=stream)
            if _is_v5_output(output):
                return _refresh_v5_report(output, run=formal, report_stream=stream)
            if _is_v4_output(output):
                return _refresh_v4_report(output, run=formal, report_stream=stream)
            return _refresh_report(output, run=formal, report_stream=stream)
    except TimeoutError:
        print("report refresh skipped: previous update still running", flush=True)
        return None


_ASYNC_REPORT = {}


def refresh_report_async(output=DEFAULT_OUTPUT, *, run=FORMAL_RUN):
    """Start a report render if one is not already running. Never blocks the caller."""
    output = Path(output)
    key = str(output.resolve())
    thread = _ASYNC_REPORT.get(key)
    if thread is not None and thread.is_alive():
        return thread

    def worker():
        try:
            refresh_report(output, run=run)
        except (MemoryError, TimeoutError, OSError) as error:
            print(f"report refresh skipped: {type(error).__name__}", flush=True)

    thread = threading.Thread(target=worker, daemon=True, name="refresh_report")
    _ASYNC_REPORT[key] = thread
    thread.start()
    return thread
