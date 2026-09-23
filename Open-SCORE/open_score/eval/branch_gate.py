"""Pre-registered branch gate (section 4.3): ladder C -> A -> B on the HAD trial.

Reads only retained records (episodes.csv, learning.csv, final.pt) and writes
decision/分支判定.md, decision/decision.json and, for an automatic decision,
decision/branch.json. A user selection (branch.json with mode="user") is never
overwritten.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import time

import numpy as np

from .experiment import (CANDIDATES, LADDER, FOUNDATION, GATE, FAILURE, TRIAL_SEEDS, SEEDS,
                         method_seeds, run_directory, atomic_json, read_json, PROFILE, episodes,
                         validation_jobs)

FINAL_EPISODES, AUX_EPISODES = episodes(300), episodes(GATE["aux_episodes"])

GROUPS = {
    "ID": ((4, 4, 2), (6, 6, 2), (8, 8, 2), (10, 10, 3)),
    "N-OOD": tuple((n, n, 2) for n in (15, 20, 25, 30, 40, 50)),
    "K-OOD": tuple((10, 10, k) for k in (4, 6, 9, 12)),
    "联合": tuple((n, n, k) for n in (15, 20, 30) for k in (4, 6)) + ((30, 30, 9), (30, 30, 12)),
}
OOD = ("N-OOD", "K-OOD", "联合")
NAMES = {"C": "C 1-round", "A": "A ARR", "B": "B IAR", "R0": "R0"}
INTENT_CONFIGS = ((10, 10, 2), (30, 30, 2), (50, 50, 2))
DEPTH_CONFIGS = ((10, 10, 2), (50, 50, 2))


def iqm(values):
    """Interquartile mean (Agarwal et al. 2021): trim floor(25%) of the runs at each end."""
    values = np.sort(np.asarray(values, dtype=float), axis=-1)
    n = values.shape[-1]
    k = int(math.floor(0.25 * n))
    return values[..., k:n - k].mean(-1)


def _records(output, methods):
    """One pass over episodes.csv for the gate's methods."""
    path = Path(output) / "episodes.csv"
    final, r1, validation, intent = {}, {}, {}, {}
    wanted = {json.dumps(m) for m in methods}
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        at = {k: header.index(k) for k in ("method", "seed", "phase", "arm", "config", "episode_seed", "D",
                                             "eval_point")}
        intent_at = header.index("intent_stats") if "intent_stats" in header else None
        env_at = header.index("env") if "env" in header else None
        for cells in reader:
            if len(cells) != len(header) or cells[at["method"]] not in wanted:
                continue
            if env_at is not None and cells[env_at] not in ('"had"', "", "null"):
                continue
            method, seed = json.loads(cells[at["method"]]), json.loads(cells[at["seed"]])
            phase, arm = json.loads(cells[at["phase"]]), json.loads(cells[at["arm"]])
            config = json.loads(cells[at["config"]])
            key = (config["N_R"], config["N_B"], config["K"])
            episode, damage = json.loads(cells[at["episode_seed"]]), float(json.loads(cells[at["D"]]))
            if phase == "train_eval":
                validation.setdefault((method, seed, json.loads(cells[at["eval_point"]])), {})[(key, episode)] = damage
            elif arm == "final":
                final.setdefault((method, seed), {}).setdefault(key, {})[episode] = damage
                stats = json.loads(cells[intent_at]) if intent_at is not None and cells[intent_at] else None
                if stats and key in INTENT_CONFIGS and episode < 9000 + AUX_EPISODES:
                    intent.setdefault((method, seed, key), []).append(stats)
            elif arm in ("gate:depth:R1", "depth:R1", "deploy:R1") and episode < 9000 + AUX_EPISODES:
                r1.setdefault((method, seed), {}).setdefault(key, {})[episode] = damage
    return final, r1, validation, intent


def _learning(output, methods):
    path = Path(output) / "learning.csv"
    spikes, peaks, intent_loss = {}, {}, {}
    if not path.exists():
        return spikes, peaks, intent_loss
    wanted = {json.dumps(m) for m in methods}
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        mi, si, ti = header.index("method"), header.index("seed"), header.index("metrics")
        env_at = header.index("env") if "env" in header else None
        for cells in reader:
            if len(cells) != len(header) or cells[mi] not in wanted:
                continue
            if env_at is not None and cells[env_at] not in ('"had"', "", "null"):
                continue
            key = (json.loads(cells[mi]), json.loads(cells[si]))
            metrics = json.loads(cells[ti])
            if "grad_spikes" in metrics:
                spikes[key] = max(spikes.get(key, 0), int(metrics["grad_spikes"]))
            if "grad_norm_max" in metrics:
                peaks[key] = max(peaks.get(key, 0.0), float(metrics["grad_norm_max"]))
            if "intent_loss" in metrics:
                intent_loss[key] = float(metrics["intent_loss"])
    return spikes, peaks, intent_loss


def _parameters(output, method, seed):
    import torch
    path = run_directory(output, method, seed) / "final.pt"
    saved = torch.load(path, map_location="cpu", weights_only=False)
    agent = saved["networks"]["agent"]
    return int(sum(v.numel() for v in agent.values()))


def _config_means(rows):
    """{config: mean D over the complete 300 frozen episodes} or None if incomplete."""
    out = {}
    for config in sum(GROUPS.values(), ()):
        episodes = rows.get(config, {})
        if len(episodes) < FINAL_EPISODES:
            return None
        out[config] = float(np.mean([episodes[9000 + i] for i in range(FINAL_EPISODES)]))
    return out


def _failure(validation, method, seed):
    def score(point):
        rows = validation.get((method, seed, point), {})
        return float(np.mean(list(rows.values()))) if len(rows) == len(validation_jobs("had", 1, 0)) else None
    final, middle = score(50), score(25)
    failed = ((final is not None and final > FAILURE["final_validation_D"])
              or (middle is not None and middle > FAILURE["midpoint_validation_D"]))
    return dict(final_validation_D=final, validation_D_500k=middle, failed=bool(failed))


def evaluate(output, *, include_b=True, preview=False, seed=0):
    """Compute all gate statistics. With preview=True incomplete seeds are skipped."""
    output = Path(output)
    levels = [b for b in LADDER if include_b or b != "B"]
    methods = {b: CANDIDATES[b] for b in levels}
    methods["R0"] = FOUNDATION
    final, r1, validation, intent = _records(output, list(methods.values()))
    spikes, peaks, intent_loss = _learning(output, list(methods.values()))
    configs = sum(GROUPS.values(), ())
    table, missing = {}, []
    for name, method in methods.items():
        seeds_ok, matrix, per_seed = [], [], []
        for s in method_seeds("had", method):
            means = _config_means(final.get((method, s), {}))
            if means is None:
                missing.append(f"{method} s{s}")
                continue
            seeds_ok.append(s)
            matrix.append([means[c] for c in configs])
            groups = {g: float(np.mean([means[c] for c in GROUPS[g]])) for g in GROUPS}
            groups["O"] = float(np.mean([groups[g] for g in OOD]))
            fail = _failure(validation, method, s)
            depth = {}
            for c in DEPTH_CONFIGS:
                one = r1.get((method, s), {}).get(c, {})
                four = final.get((method, s), {}).get(c, {})
                if len(one) >= AUX_EPISODES:
                    ids = [9000 + i for i in range(AUX_EPISODES)]
                    depth[f"{c[0]}v{c[1]} K{c[2]}"] = dict(R1=float(np.mean([one[i] for i in ids])),
                                                         R4=float(np.mean([four[i] for i in ids])))
            per_seed.append(dict(seed=s, **groups, **fail, spikes=spikes.get((method, s)),
                                 grad_norm_max=peaks.get((method, s)), intent_loss=intent_loss.get((method, s)),
                                 depth=depth, parameters=_parameters(output, method, s)))
        table[name] = dict(method=method, seeds=seeds_ok, matrix=np.asarray(matrix), per_seed=per_seed)
    if missing and not preview:
        raise ValueError("Gate inputs incomplete: " + ", ".join(missing))
    return table, _intent_accuracy(intent, methods.get("B")), configs


def _intent_accuracy(intent, method):
    if method is None:
        return {}
    out = {}
    for config in INTENT_CONFIGS:
        correct = count = 0
        hist = np.zeros(9)
        by_round = {}
        for (m, s, c), rows in intent.items():
            if m != method or c != config:
                continue
            for stats in rows:
                last = str(max(int(k) for k in stats["count"]))
                count += stats["count"][last]
                correct += stats["correct"][last]
                hist += np.asarray(stats["label_hist"])
                for k, n in stats["count"].items():
                    item = by_round.setdefault(k, [0, 0])
                    item[0] += stats["correct"][k]; item[1] += n
        if count:
            out[f"{config[0]}v{config[1]} K{config[2]}"] = dict(
                top1=correct / count, majority=float(hist.max() / hist.sum()), pairs=int(count),
                by_round={k: v[0] / v[1] for k, v in sorted(by_round.items())})
    return out


def _bootstrap(table, configs, names, reps, rng):
    """P[IQM_O(X) < IQM_O(Y)]: seeds resampled per candidate, configs resampled within groups (shared)."""
    group_index = {g: [configs.index(c) for c in GROUPS[g]] for g in OOD}
    draws = {}
    config_draws = {g: rng.integers(0, len(idx), size=(reps, len(idx))) for g, idx in group_index.items()}
    for name in names:
        matrix = table[name]["matrix"]
        n = matrix.shape[0]
        seeds = rng.integers(0, n, size=(reps, n))
        sampled = matrix[seeds]                                   # reps x n x configs
        per_group = []
        for g, idx in group_index.items():
            cols = np.asarray(idx)[config_draws[g]]               # reps x |g|
            values = np.take_along_axis(sampled, cols[:, None, :].repeat(n, 1), axis=2)
            per_group.append(values.mean(-1))                     # reps x n
        o = np.mean(per_group, axis=0)                            # reps x n
        draws[name] = iqm(o)
    return {(x, y): float((draws[x] < draws[y]).mean()) for x in names for y in names if x != y}


def decide(output, *, include_b=True, preview=False, write=True, now=None):
    output = Path(output)
    table, intent, configs = evaluate(output, include_b=include_b, preview=preview)
    names = [n for n in (list(LADDER) + ["R0"]) if n in table and len(table[n]["seeds"]) >= 2]
    ladder = [b for b in LADDER if b in names]
    stats = {}
    for name in names:
        rows = table[name]["per_seed"]
        o = [r["O"] for r in rows]
        by_validation = sorted((r for r in rows if r["final_validation_D"] is not None),
                               key=lambda r: r["final_validation_D"])[:3]
        stats[name] = dict(
            O=float(iqm(o)), groups={g: float(iqm([r[g] for r in rows])) for g in GROUPS},
            mean=float(np.mean(o)), median=float(np.median(o)), per_seed=o,
            best3_by_validation=float(np.mean([r["O"] for r in by_validation])) if len(by_validation) == 3 else None,
            failures=int(sum(r["failed"] for r in rows)), spikes=[r["spikes"] for r in rows],
            parameters=rows[0]["parameters"] if rows else None, n=len(rows))
    prob = _bootstrap(table, configs, names, GATE["bootstrap"], np.random.default_rng(20260924))

    def better(x, y):
        improvement = (stats[y]["O"] - stats[x]["O"]) / stats[y]["O"]
        return dict(improvement=improvement, probability=prob[(x, y)],
                    significant=bool(improvement >= GATE["margin"] and prob[(x, y)] >= GATE["prob"]))

    comparisons = {f"{x}>{y}": better(x, y) for x in names for y in names if x != y}
    foundation = comparisons.get("C>R0", dict(significant=False))
    reasons, choice, mode = [], None, "manual"
    if not foundation["significant"]:
        reasons.append("地基检查未通过：C 没有显著优于 R0（二阶关系是关键的主线不成立），必须人工决策")
    else:
        eligible = []
        for level, name in enumerate(ladder):
            lower, higher = ladder[:level], ladder[level + 1:]
            a = all(comparisons[f"{name}>{low}"]["significant"] for low in lower)
            b = not any(stats[h]["O"] < stats[name]["O"] for h in higher)
            c = all(stats[name]["best3_by_validation"] is not None and stats[low]["best3_by_validation"] is not None
                    and stats[name]["best3_by_validation"] < stats[low]["best3_by_validation"] for low in lower)
            d = stats[name]["failures"] <= GATE["max_failures"]
            stats[name]["conditions"] = dict(a=a, b=b, c=c, d=d)
            if a and b and c and d:
                eligible.append(name)
        if "C" in stats and all(stats[h]["O"] >= stats["C"]["O"] for h in ladder if h != "C"):
            choice, mode = "C", "auto"
            reasons.append("A、B 的 O-IQM 都不低于 C：自动选 C")
        elif eligible:
            choice, mode = eligible[-1], "auto"
            reasons.append(f"{NAMES[choice]} 是满足 (a)–(d) 的最高一级")
        else:
            reasons.append("没有同时满足 (a)–(d) 的级别（典型情况：更高一级数值更好但不显著），转人工决策")
    for name in names:
        if stats[name]["failures"] >= 2:
            reasons.append(f"{NAMES[name]} 失败 {stats[name]['failures']}/{stats[name]['n']}：已标出")
            if mode == "auto":
                mode, choice = "manual", None
    result = dict(profile=PROFILE, version=GATE["version"], computed_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  preview=preview, include_b=include_b, ladder=ladder, foundation_ok=foundation["significant"],
                  mode=mode, choice=choice, reasons=reasons, stats=stats, comparisons=comparisons,
                  intent_accuracy=intent,
                  per_seed={n: table[n]["per_seed"] for n in names})
    if write:
        directory = output / "decision"
        directory.mkdir(parents=True, exist_ok=True)
        stem = "preview" if preview else "decision"
        atomic_json(directory / f"{stem}.json", _jsonable(result))
        (directory / ("分支判定_预览.md" if preview else "分支判定.md")).write_text(_render(result), encoding="utf-8")
        if not preview:
            existing = read_json(directory / "branch.json")
            if mode == "auto" and not existing:
                atomic_json(directory / "branch.json", dict(branch=choice, mode="auto", decided_at=result["computed_at"],
                                                            gate_version=GATE["version"], reasons=reasons))
            elif mode == "manual" and not existing:
                (directory / "DECISION_REQUIRED").write_text(
                    "等待选择主方法：bash run_all.sh select A|B|C\n" + "\n".join(reasons) + "\n", encoding="utf-8")
    return result


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _render(result):
    lines = [f"# 分支判定{'（预览，不触发分支）' if result['preview'] else ''}", ""]
    if not result["foundation_ok"]:
        lines += ['<p style="color:red"><b>地基检查未通过：C 没有显著优于 R0，必须人工决策。</b></p>', ""]
    decision = (f"**自动选定：{NAMES[result['choice']]}**" if result["mode"] == "auto"
                else "**需要人工决策**：`bash run_all.sh select A|B|C`")
    lines += [decision, "", *[f"- {r}" for r in result["reasons"]], "",
              f"规则版本 `{result['version']}`；计算时间 {result['computed_at']}；阶梯 {' → '.join(result['ladder'])}"
              f"{'' if result['include_b'] else '（IAR 未在等待期限内就绪，只在 C、A 之间判定）'}。", "",
              "## 主统计量（O = N-OOD、K-OOD、联合三组的平均；越低越好）", "",
              "| 级别 | n | O IQM | ID | N-OOD | K-OOD | 联合 | 均值 | 中位数 | 最好3种子(按验证) | 失败 | 尖峰 | 参数量 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in result["stats"].items():
        g = s["groups"]
        best = "—" if s["best3_by_validation"] is None else f"{s['best3_by_validation']:.3f}"
        lines.append(f"| {NAMES[name]} | {s['n']} | **{s['O']:.3f}** | {g['ID']:.3f} | {g['N-OOD']:.3f} | "
                     f"{g['K-OOD']:.3f} | {g['联合']:.3f} | {s['mean']:.3f} | {s['median']:.3f} | {best} | "
                     f"{s['failures']} | {s['spikes']} | {s['parameters']:,} |")
    lines += ["", "## 两两比较（X 优于 Y：相对改善 ≥10% 且 bootstrap 概率 ≥0.90）", "",
              "| 比较 | 相对改善 | P(X 优于 Y) | 显著 |", "|---|---|---|---|"]
    for key, c in result["comparisons"].items():
        x, y = key.split(">")
        lines.append(f"| {NAMES[x]} 优于 {NAMES[y]} | {100 * c['improvement']:.1f}% | {c['probability']:.3f} | "
                     f"{'是' if c['significant'] else '否'} |")
    lines += ["", "## 条件 (a)–(d)", ""]
    for name, s in result["stats"].items():
        if "conditions" in s:
            lines.append(f"- {NAMES[name]}：{s['conditions']}")
    lines += ["", "## 逐种子", "", "| 级别 | 种子 | O | ID | 最终验证 D | 50万步验证 D | 失败 | 尖峰 | R1/R4（10v10、50v50 K2） |",
              "|---|---|---|---|---|---|---|---|---|"]
    for name, rows in result["per_seed"].items():
        for r in rows:
            depth = "；".join(f"{k}: {v['R1']:.3f}/{v['R4']:.3f}" for k, v in r["depth"].items()) or "—"
            fv = "—" if r["final_validation_D"] is None else f"{r['final_validation_D']:.3f}"
            mv = "—" if r["validation_D_500k"] is None else f"{r['validation_D_500k']:.3f}"
            lines.append(f"| {NAMES[name]} | {r['seed']} | {r['O']:.3f} | {r['ID']:.3f} | {fv} | {mv} | "
                         f"{'是' if r['failed'] else ''} | {r['spikes']} | {depth} |")
    if result["intent_accuracy"]:
        lines += ["", "## IAR 意图预测准确率（最后一轮，top-1；对照为“总是预测最常见动作”）", "",
                  "| 配置 | top-1 | 多数类基线 | 各轮 | 样本对 |", "|---|---|---|---|---|"]
        for config, a in result["intent_accuracy"].items():
            rounds = " / ".join(f"{v:.3f}" for v in a["by_round"].values())
            lines.append(f"| {config} | {a['top1']:.3f} | {a['majority']:.3f} | {rounds} | {a['pairs']:,} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--without-b", action="store_true")
    options = parser.parse_args()
    result = decide(options.output, include_b=not options.without_b, preview=options.preview)
    print(result["mode"], result["choice"], result["reasons"])
