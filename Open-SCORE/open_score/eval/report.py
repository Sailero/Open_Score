"""Refresh the one illustrated report from committed experiment records."""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import statistics
import threading

from open_score.utils.logging import DEFAULT_OUTPUT, FORMAL_RUN, read_latest, read_records, unique_episodes
from .protocol import (EQUAL_SCALE_CONFIGS, FINAL_CONFIGS, POOL_ONLY_METHODS, RATIO_CONFIGS,
                       TARGET_K_VALUES, TARGET_SCALE_NS, TEST_CONFIGS, VALIDATION_CONFIGS,
                       compute_nds, config_key, config_label, final_configs, validation_score)

METHODS = ("b2_qmix_atten", "refil", "dcg", "spectra", "alma")
SIDE_METHODS = ("b0_qmix", "gnn_qmix")
ALL_METHODS = METHODS + SIDE_METHODS
LABELS = {"b0_qmix": "B0", "b2_qmix_atten": "QMIX", "refil": "REFIL",
          "dcg": "DCG", "gnn_qmix": "GNN", "spectra": "SPECTra", "alma": "ALMA",
          "random": "Random", "rule_nv1": "Rule nv1"}
COLORS = {"b0_qmix": "#5b6770", "b2_qmix_atten": "#1675b8", "refil": "#e36b32",
          "dcg": "#7955a3", "gnn_qmix": "#32865e", "spectra": "#bf4c79",
          "alma": "#b08a1e", "random": "#979fa5", "rule_nv1": "#267c4f"}
BENCHMARK_REVISION = "filled_padding_compact_mixer_frozen_target"
FORMAL_SEEDS = (0, 1, 2)
_REPORT_THREAD_LOCK = threading.RLock()


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
                result = api.WaitForSingleObject(handle, 30000)
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
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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


def _plot_learning(plt, output, points, anchors):
    """Red episode return R=-D, mean±std over training seeds."""
    points = [row for row in points if row["method"] in METHODS and _numeric(row.get("D"))]
    if not points:
        return None
    fig, axis = plt.subplots(1, 1, figsize=(7.6, 4.4), constrained_layout=True)
    for method in METHODS:
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
    for policy, style in (("random", "--"), ("rule_nv1", "-.")):
        value = _pool_return(anchors, policy)
        if value is not None:
            axis.axhline(value, color=COLORS[policy], linestyle=style, linewidth=1.6, label=LABELS[policy])
    axis.set(title="In-distribution red return (higher is better)",
             xlabel="Environment steps", ylabel="Red episode return  R = −D")
    axis.grid(alpha=.2)
    axis.legend(fontsize=9)
    path = output / "figures" / "learning.png"
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


def _plot_scale_line(plt, output, rows, anchors, configs, *, stem, title, xlabel):
    per_method, anchors_r = _config_returns(rows, anchors, configs)
    if not per_method and not any(anchors_r.values()):
        return None
    fig, axis = plt.subplots(1, 1, figsize=(7.2, 3.8), constrained_layout=True)
    xs = list(range(len(configs)))
    for method in METHODS:
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
    for policy, style in (("random", "--"), ("rule_nv1", "-.")):
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


def _plot_target_k(plt, output, rows, anchors):
    configs = tuple((n, n, k) for n in TARGET_SCALE_NS for k in TARGET_K_VALUES)
    per_method, anchors_r = _config_returns(rows, anchors, configs)
    if not per_method and not any(anchors_r.values()):
        return None
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.6), sharex=True, sharey=True,
                             constrained_layout=True)
    for axis, n in zip(axes.flat, TARGET_SCALE_NS):
        for method in METHODS:
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
        for policy, style in (("random", "--"), ("rule_nv1", "-.")):
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


def _plot_final(plt, output, rows, anchors):
    rows = [row for row in rows if row["phase"] == "final_eval" and row["method"] in METHODS]
    return [
        _plot_scale_line(plt, output, rows, anchors, EQUAL_SCALE_CONFIGS,
                         stem="scale_equal", title="Equal-scale 1:1", xlabel="Roster"),
        _plot_scale_line(plt, output, rows, anchors, RATIO_CONFIGS,
                         stem="scale_ratio", title="1:2 red:blue", xlabel="Roster"),
        _plot_target_k(plt, output, rows, anchors),
    ]


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


def _id_behavior(final_rows, episodes):
    result = {}
    pool = [row for row in final_rows if config_key(row["config"]) in set(VALIDATION_CONFIGS)]
    for method in METHODS:
        result[method] = [_behavior_rates(group) for _, group in sorted(_groups(
            [row for row in pool if row["method"] == method], ("seed",)).items())]
    for method in ("random", "rule_nv1"):
        rows = [row for row in episodes if row["method"] == method and row["phase"] == "anchor"
                and config_key(row["config"]) in set(VALIDATION_CONFIGS)]
        result[method] = [_behavior_rates(rows)] if rows else []
    return {key: [row for row in rows if all(row.get(field) is not None
                  for field in ("intercept", "deliver", "friendly"))] for key, rows in result.items()}


def _plot_behavior(plt, output, behavior):
    order = ("random",) + METHODS + ("rule_nv1",)
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


def _final_seed_scores(rows, anchors):
    scores = defaultdict(lambda: defaultdict(dict))
    for (method, seed), values in _groups(rows, ("method", "seed")).items():
        if method not in METHODS:
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
    text = ["<!-- report_layout: qmix_baseline -->", "# 跨规模攻防泛化 v3", "",
            f"更新时间：{now}。比较 {comparison}，各 3 个训练种子。" + status,
            "本版本修复了 v2 的团队价值下界、$b_1$/$V$ 随规模缩放、红方全灭尾段与有序空槽四处问题，"
            "诊断见 [v2 报告](../crossscale_v2/实验报告.md) 与 "
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
            "ALMA 是唯一的分层臂，低层与该 QMIX 同构，差别只在多一层子任务分配："
            "上层每 5 步把红方分到子任务，低层只观测本子任务内的实体，奖励按目标分解、逐步求和等于团队回报。"
            "子任务集合就是目标集合，蓝方按“当前最近目标”归属——这个量任何方法都能从同一份实体表算出，"
            "蓝方自己的分组指派属于私有信息，不进入红方输入。", ""]

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
    content = "\n".join(text)
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


def refresh_report(output=DEFAULT_OUTPUT, *, run=FORMAL_RUN):
    """Render the one report of this version; a side run never replaces it."""
    output = Path(output)
    with _report_lock(output) as stream:
        return _refresh_report(output, run=run if run == FORMAL_RUN else FORMAL_RUN,
                               report_stream=stream)
