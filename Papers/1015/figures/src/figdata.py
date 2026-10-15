"""Statistics behind the Section V figures and tables of the 1015 draft.

Everything here is read from saved LEAF records under Open-SCORE/outputs/main.
No policy, simulator, or training code runs. ``load()`` returns the cached
statistics in ``figure_data.json`` and ``python figdata.py`` rebuilds them and
prints the numbers quoted in the text.
"""
from __future__ import annotations

import argparse
import base64
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
MAIN = ROOT / "Open-SCORE" / "outputs" / "main"
HAD = MAIN / "had"
CACHE = Path(__file__).with_name("figure_data.json")

LEAF = "regir_nomem"
BASELINES = ("b2_qmix_atten", "refil", "dcg", "alma", "transfqmix", "spectra")
VARIANTS = ("regir_r0_sg", "regir_r1_sg", "regir_fixed4_sg", "regir_sg",
            "regir_untied4_sg", "regir_last_sg", "regir_norefil_sg", "refil_matched")
GROUPS = ("ID", "K-OOD", "N-OOD", "Joint-OOD")
EXTRA = ("5-2", "10-2")
DRIFT_SEED = 20261015
DRIFT_REFERENCE = "10-2"
DRIFT_SHIFTED = ("50-2", "30-12")


def key(config):
    return f"{config[0]}-{config[-1]}"


def read_json(path):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def decode(array):
    raw = base64.b64decode(array["base64"])
    return np.frombuffer(raw, dtype=np.dtype(array["dtype"])).reshape(array["shape"])


def seed_stats(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
            "seeds": [float(v) for v in values]}


# ---------------------------------------------------------------- formal comparison
def comparison(agg):
    comp = agg["comparison"]
    groups = {g: [key(c) for c in comp["groups"][g]] for g in GROUPS}
    configs = defaultdict(dict)
    for row in comp["config_summary"]:
        configs[row["method"]][key(row["config"])] = {
            **seed_stats(row["D_per_seed"]),
            "collision": row["collision_fraction"]["mean"] if row.get("collision_fraction") else None}
    for row in comp["rule_config"]:
        configs["rule"][key(row["config"])] = {"mean": -row["reward"]["mean"],
                                               "sd": row["reward"]["sample_sd"], "seeds": None}
    group_rows = defaultdict(dict)
    for row in comp["group_summary"]:
        if not row["reward"].get("per_seed"):
            continue
        stats = seed_stats([-v for v in row["reward"]["per_seed"]])
        coll = row.get("collision_fraction")
        stats["collision"] = seed_stats(coll["per_seed"]) if coll and coll.get("per_seed") else None
        group_rows[row["method"]][row["group"]] = stats
    for g in GROUPS:
        group_rows["rule"][g] = {"mean": float(np.mean([configs["rule"][c]["mean"] for c in groups[g]])),
                                 "sd": None, "seeds": None}
    curves = defaultdict(list)
    for row in sorted(comp["validation_curves"], key=lambda r: r["eval_point"]):
        curves[row["method"]].append(row)
    validation = {}
    for method, rows in curves.items():
        seeds = np.array([[-v for v in r["reward"]["per_seed"]] for r in rows])
        validation[method] = {"t": [float(r["t_env_mean"]) for r in rows],
                              "mean": [float(v) for v in seeds.mean(1)],
                              "sd": [float(v) for v in seeds.std(1, ddof=1)],
                              "seeds": seeds.T.tolist()}
    rule_id = comp["rule_ID_reference"]["reward"]
    confirm = defaultdict(lambda: defaultdict(dict))
    for row in comp["confirmation"]["config_summary"]:
        for s, v in enumerate(row["D_per_seed"]):
            confirm[row["method"]][s][key(row["config"])] = v
    return {"groups": groups, "configs": dict(configs), "group": dict(group_rows),
            "validation": validation,
            "rule_id_reference": {"mean": -rule_id["mean"], "scene_sd": rule_id["sample_sd"]},
            "confirmation_cells": {m: {str(s): c for s, c in v.items()} for m, v in confirm.items()}}


def loop_rows():
    with (HAD / "loop_summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def confirmation_all24(comp, rows):
    """All24 damage on the independent scene pool 110000-110299."""
    cells = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if (row["split"] == "confirm" and row["arm"] == "original" and int(row["episodes"]) == 300
                and row["method"] in (LEAF, "refil")
                and row.get("archive_source", row.get("source_version")) == "main1009"):
            cells[row["method"]][int(row["seed"])][key(json.loads(row["config"]))] = float(row["D"])
    for method, seeds in comp["confirmation_cells"].items():
        for s, cfgs in seeds.items():
            cells[method][int(s)].update(cfgs)
    out = {}
    for method, seeds in cells.items():
        out[method] = seed_stats([np.mean(list(seeds[s].values())) for s in sorted(seeds)])
        out[method]["n_configs"] = [len(seeds[s]) for s in sorted(seeds)]
    return out


# ---------------------------------------------------------------- per-episode metrics
def episode_metrics(groups):
    """Damage incidence P and the mean of the worst 10% episodes (CVaR) per model and configuration."""
    import pandas as pd
    cols = ["method", "seed", "phase", "config", "episode_seed", "D", "arm", "source_version"]
    df = pd.read_csv(HAD / "episodes.csv", usecols=cols, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    for c in cols:
        df[c] = df[c].str.strip('"')
    learned = df[(df.phase == "final_eval") & df.arm.isin(["final", "standard"])
                 & df.method.isin((LEAF,) + BASELINES)]
    learned = learned[~learned.method.isin(["spectra", "transfqmix"]) | (learned.source_version == "main1013_base")]
    rule = df[(df.phase == "anchor") & (df.method == "rule_nv1")]
    frames = {"learned": learned, "rule": rule}
    cells = defaultdict(lambda: defaultdict(dict))
    rule_scene = {}
    for name, frame in frames.items():
        frame = frame.assign(D=frame.D.astype(float), cfg=frame.config.map(lambda s: key(list(json.loads(s).values()))))
        for (method, seed, cfg), part in frame.groupby(["method", "seed", "cfg"]):
            d = np.sort(part.D.to_numpy())[::-1]
            if len(d) != 300:
                raise ValueError(f"{method} seed {seed} {cfg}: {len(d)} episodes")
            cells["rule" if name == "rule" else method][int(seed)][cfg] = {
                "P": float((d > 1e-9).mean()), "CVaR": float(d[:30].mean())}
            if name == "rule":
                rule_scene[cfg] = part.set_index("episode_seed").D.astype(float)
    out = {}
    for method, seeds in cells.items():
        out[method] = {}
        for g, cfgs in groups.items():
            per_seed = {m: [np.mean([seeds[s][c][m] for c in cfgs]) for s in sorted(seeds)] for m in ("P", "CVaR")}
            out[method][g] = {m: seed_stats(v) for m, v in per_seed.items()}
    rule_sd = {}
    for g, cfgs in groups.items():
        scene_means = pd.concat([rule_scene[c] for c in cfgs], axis=1).mean(axis=1)
        rule_sd[g] = float(scene_means.std(ddof=1))
    return out, rule_sd


# ---------------------------------------------------------------- parameters
def deployed_parameters():
    """Parameters of the networks used at deployment, read from seed-0 final checkpoints."""
    import torch

    def count(tree):
        if hasattr(tree, "numel"):
            return int(tree.numel())
        if isinstance(tree, dict):
            return sum(count(v) for k, v in tree.items() if "scale_factor" not in str(k))
        return 0

    out = {}
    for method in (LEAF,) + BASELINES + VARIANTS:
        ckpt = torch.load(HAD / method / "models" / "seed_0" / "final.pt", map_location="cpu", weights_only=False)
        nets = ckpt["networks"]
        # The controller state already contains the agent together with any deployed heads.
        out[method] = count(nets.get("mac", {})) or count(nets["agent"])
    return out


# ---------------------------------------------------------------- execution depth
def depth_profiles(agg, rows):
    """Closed-loop damage at frozen execution depths 1-6."""
    profiles = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for row in rows:
        if row["method"] == LEAF and row["split"] == "nomem_depth_eval" and int(row["episodes"]) == 300:
            depth = int(row["arm"].replace("fixed", ""))
            profiles[LEAF][key(json.loads(row["config"]))][depth][int(row["seed"])] = float(row["D"])
    for row in agg["comparison"]["history_interventions"]["depth"]:
        if row["readout"] != "learned":
            continue
        for s, v in enumerate(row["reward"]["per_seed"]):
            profiles[row["method"]][key(row["config"])][row["cycle_depth"]][s] = -v
    out = {}
    for method, cfgs in profiles.items():
        out[method] = {c: {str(d): [seeds[s] for s in sorted(seeds)] for d, seeds in sorted(depths.items())}
                       for c, depths in cfgs.items()}
    return out


def readout_interventions(agg):
    out = defaultdict(dict)
    for row in agg["comparison"]["history_interventions"]["readout"]:
        out[key(row["config"])][row["readout"]] = seed_stats([-v for v in row["reward"]["per_seed"]])
    return dict(out)


def frozen_interventions(agg):
    out = defaultdict(dict)
    for row in agg["mechanism"]["frozen"]["combined_interventions"]["across_seeds"]:
        out[key(row["config"])][row["arm"]] = {
            "dD": seed_stats([-v for v in row["paired_reward"]["per_seed"]]),
            "action_change": seed_stats(row["action_change_fraction"]["per_seed"])}
    return dict(out)


def fusion_weights(agg):
    out = {}
    for row in agg["mechanism"]["normal"]["feature_analysis"]["across_seeds"]:
        out[key(row["config"])] = {
            "alpha": row["mean_alpha"]["mean"],
            "alpha_seeds": row["mean_alpha"]["per_seed"],
            "receiver_sd": row["mean_within_step_defender_alpha_sd"]["mean"][0],
            "receiver_sd_seeds": [s[0] for s in row["mean_within_step_defender_alpha_sd"]["per_seed"]],
            "step_change": row["mean_same_defender_adjacent_step_alpha_abs_change"]["mean"][0],
            "step_change_seeds": [s[0] for s in row["mean_same_defender_adjacent_step_alpha_abs_change"]["per_seed"]]}
    return out


def timing(summary):
    out = defaultdict(dict)
    for row in summary["decision_timing20"]["summary"]:
        out[key(row["config"])][row["arm"]] = {"ms": row["active_ms_per_step"], "episode_sd": row["episode_sd_ms"],
                                               "steps": row["active_decision_steps"]}
    fixed = defaultdict(dict)
    for row in summary["decision_timing_fixed100"]["cells"]:
        fixed[key(row["config"])][row["arm"]] = {"ms": row["mean_ms"], "sd": row["call_sd_ms"]}
    return {"closed_loop": dict(out), "fixed_count": dict(fixed),
            "hardware": summary["decision_timing20"]["hardware"]}


def protection(agg):
    metrics = ("blue_intercepted_fraction", "any_damage", "worst10pct_D", "collision_fraction",
               "first_damage_restricted_time")
    out = defaultdict(dict)
    for row in agg["mechanism"]["formal"]["summary"]:
        out[row["method"]][key(row["config"])] = {m: seed_stats(row["metrics"][m]["per_seed"]) for m in metrics}
    cdf = defaultdict(dict)
    for row in agg["mechanism"]["formal"]["first_damage_cdf"]:
        cdf[row["method"]][key(row["config"])] = {"t": row["t"], "mean": row["mean"], "sd": row["sample_sd"]}
    return dict(out), dict(cdf)


# ---------------------------------------------------------------- raw mechanism records
def mechanism_records():
    return read_json(HAD / LEAF / "mechanisms" / "results.json.gz")["episodes"]


def margin_analysis(episodes):
    """Same-state comparison of two-round and four-round greedy intents."""
    rows = defaultdict(lambda: {"margin": [], "disp": [], "changed": []})
    for ep in episodes:
        if ep["method"] != LEAF or ep["arm"] != "standard":
            continue
        cell = rows[(key(ep["config"]), ep["training_seed"])]
        for d in ep["decisions"]:
            q = decode(d["q_standard"])
            alt = d["alternatives"]["L2"]
            top = np.sort(q, axis=1)
            cell["margin"].append(top[:, -1] - top[:, -2])
            cell["disp"].append(2 * np.abs(decode(alt["q_delta"])).max(axis=1))
            cell["changed"].append(np.asarray(alt["greedy_action"]) != np.asarray(d["greedy_standard"]))
    deciles = defaultdict(list)
    cells, scatter = {}, {"margin": [], "disp": [], "changed": [], "config": []}
    for (cfg, seed), cell in sorted(rows.items()):
        margin, disp, changed = (np.concatenate(cell[k]) for k in ("margin", "disp", "changed"))
        rank = np.argsort(np.argsort(margin, kind="stable"), kind="stable") / len(margin)
        bins = np.minimum((rank * 10).astype(int), 9)
        deciles[cfg].append([float(changed[bins == b].mean()) for b in range(10)])
        low = margin <= np.quantile(margin, 0.1)
        cells[f"{cfg}/{seed}"] = {"decisions": int(len(margin)), "changed": int(changed.sum()),
                                  "certified": int((margin > disp).sum()),
                                  "certified_changed": int(((margin > disp) & changed).sum()),
                                  "low10_share_of_changes": float(changed[low].sum() / max(changed.sum(), 1))}
        scatter["margin"] += np.round(margin, 6).tolist()
        scatter["disp"] += np.round(disp, 6).tolist()
        scatter["changed"] += changed.astype(int).tolist()
        scatter["config"] += [cfg] * len(margin)
    return {"cells": cells, "decile_change_rate": {c: v for c, v in deciles.items()}, "scatter": scatter}


def fusion_distribution(episodes):
    """Per-decision first-round fusion weight in normal four-round execution."""
    out = defaultdict(list)
    for ep in episodes:
        if ep["method"] != LEAF:
            continue
        cfg = key(ep["config"])
        normal = (ep["arm"] == "standard" and cfg in ("10-2", "30-12")) or (ep["arm"] == "L4" and cfg == "30-2")
        if not normal:
            continue
        for d in ep["decisions"]:
            out[cfg] += np.round(decode(d["alpha"])[:, 0], 5).tolist()
    return dict(out)


def drift_samples(episodes):
    samples = defaultdict(lambda: {"step": [], "scene": [], "memory": [], "self": []})
    for ep in episodes:
        if ep["method"] != LEAF or ep["arm"] != "standard":
            continue
        cell = samples[(ep["training_seed"], key(ep["config"]))]
        trajectory = ep["trajectory"]
        for d in ep["decisions"]:
            t = d["step"]
            if t < 1:
                continue
            memory = decode(d["h_prev"])
            frame = trajectory[t - 1]
            red = {unit["id"]: unit for unit in frame["red"]}
            for row, i in enumerate(d["active_defender_ids"]):
                unit = red[i]
                action = np.zeros(27)
                action[frame["actions"][i]] = 1.0
                cell["self"].append(np.concatenate([np.asarray(unit["position"]) / 2500,
                                                    np.asarray(unit["velocity"]) / 120,
                                                    [float(unit["alive"]), float(unit["health"])], action]))
                cell["memory"].append(memory[row])
                cell["step"].append(t)
                cell["scene"].append(ep["episode_seed"])
    return {k: {f: np.asarray(v) for f, v in cell.items()} for k, cell in samples.items()}


def memory_drift(episodes):
    """How well a classifier tells the training-range scale from a shifted scale.

    Each training seed compares 10v10 K2 with one shifted configuration on step-matched
    balanced samples, using 5-fold cross-validation grouped by scene.
    """
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    samples = drift_samples(episodes)
    models = {
        "linear": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)),
        "nonlinear": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0),
        "pca8": lambda: make_pipeline(StandardScaler(), PCA(8), LogisticRegression(C=1.0, max_iter=3000)),
    }
    grid = np.linspace(0, 1, 101)
    out = {"auc": defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
           "roc": defaultdict(lambda: defaultdict(list)), "pairs": defaultdict(list), "memory_norm": {}}
    for (seed, cfg), cell in samples.items():
        out["memory_norm"][f"{cfg}/{seed}"] = float(np.linalg.norm(cell["memory"], axis=1).mean())
    for shifted in DRIFT_SHIFTED:
        for seed in (0, 1, 2):
            ref, ext = samples[(seed, DRIFT_REFERENCE)], samples[(seed, shifted)]
            rng = np.random.default_rng(DRIFT_SEED)
            pick_ref, pick_ext = [], []
            for t in sorted(set(ref["step"]) & set(ext["step"])):
                a, b = np.flatnonzero(ref["step"] == t), np.flatnonzero(ext["step"] == t)
                n = min(len(a), len(b))
                pick_ref += rng.choice(a, n, replace=False).tolist()
                pick_ext += rng.choice(b, n, replace=False).tolist()
            out["pairs"][shifted].append(len(pick_ref))
            y = np.r_[np.zeros(len(pick_ref)), np.ones(len(pick_ext))]
            groups = np.r_[ref["scene"][pick_ref], ext["scene"][pick_ext]]
            for feature in ("memory", "self"):
                x = np.vstack([ref[feature][pick_ref], ext[feature][pick_ext]])
                for name, make in models.items():
                    scores, folds = np.zeros(len(y)), []
                    for train, test in GroupKFold(n_splits=5).split(x, y, groups):
                        model = make().fit(x[train], y[train])
                        scores[test] = model.predict_proba(x[test])[:, 1]
                        folds.append(roc_auc_score(y[test], scores[test]))
                    out["auc"][shifted][feature][name].append(float(np.mean(folds)))
                    if name == "linear":
                        fpr, tpr, _ = roc_curve(y, scores)
                        out["roc"][shifted][feature].append(np.interp(grid, fpr, tpr).round(5).tolist())
    return json.loads(json.dumps(out))


# ---------------------------------------------------------------- build and digest
def build():
    agg = read_json(HAD / "aggregates.json")
    summary = read_json(MAIN / "summary.json")
    comp = comparison(agg)
    rows = loop_rows()
    p_cvar, rule_scene_sd = episode_metrics(comp["groups"])
    episodes = mechanism_records()
    protection_metrics, first_damage = protection(agg)
    data = {
        **comp,
        "confirmation_all24": confirmation_all24(comp, rows),
        "p_cvar": p_cvar,
        "rule_group_scene_sd": rule_scene_sd,
        "parameters": deployed_parameters(),
        "depth": depth_profiles(agg, rows),
        "nomem_depth_all24": summary["nomem_depth_analysis"]["groups"],
        "readout_interventions": readout_interventions(agg),
        "frozen_interventions": frozen_interventions(agg),
        "fusion": fusion_weights(agg),
        "fusion_alpha1": fusion_distribution(episodes),
        "timing": timing(summary),
        "protection": protection_metrics,
        "first_damage": first_damage,
        "margin": margin_analysis(episodes),
        "memory_drift": memory_drift(episodes),
    }
    CACHE.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    return data


def load(rebuild=False):
    if rebuild or not CACHE.is_file():
        return build()
    return json.loads(CACHE.read_text(encoding="utf-8"))


def all24(data, method, seeds=False):
    cfgs = [c for g in GROUPS for c in data["groups"][g]] + list(EXTRA)
    per_seed = [np.mean([data["configs"][method][c]["seeds"][s] for c in cfgs]) for s in range(3)]
    return per_seed if seeds else float(np.mean(per_seed))


def digest(data):
    """Print every number the text quotes, next to the value it is checked against."""
    g = data["group"]
    methods = BASELINES + ("rule",)
    print("== Group damage D (mean +- seed SD)")
    for m in BASELINES + ("rule", LEAF):
        print(f"  {m:14s}", "  ".join(f"{g[m][k]['mean']:.3f}" + (f"+-{g[m][k]['sd']:.3f}" if g[m][k]['sd'] else "")
                                       for k in GROUPS))
    print("  rule scene SD:", {k: round(v, 3) for k, v in data["rule_group_scene_sd"].items()})
    for k in GROUPS:
        best = min(methods, key=lambda m: g[m][k]["mean"])
        print(f"  {k}: LEAF vs strongest ({best}) {100 * (g[LEAF][k]['mean'] / g[best][k]['mean'] - 1):+.1f}%")
        worst_leaf = max(g[LEAF][k]["seeds"])
        best_base = min(min(g[m][k]["seeds"]) for m in BASELINES)
        print(f"     worst LEAF seed {worst_leaf:.3f} < best baseline seed {best_base:.3f} and Rule "
              f"{g['rule'][k]['mean']:.3f}: {worst_leaf < best_base and worst_leaf < g['rule'][k]['mean']}")
    cfgs = [c for k in GROUPS for c in data["groups"][k]] + list(EXTRA)
    lowest = sum(all(data["configs"][LEAF][c]["mean"] < data["configs"][m][c]["mean"] for m in methods) for c in cfgs)
    print(f"  LEAF lowest in {lowest}/{len(cfgs)} configurations")
    n_curve = [data["configs"][LEAF][f"{n}-2"]["mean"] for n in (15, 20, 25, 30, 40, 50)]
    r_curve = [data["configs"]["refil"][f"{n}-2"]["mean"] for n in (15, 20, 25, 30, 40, 50)]
    print(f"  LEAF N=15..50 range {min(n_curve):.3f}-{max(n_curve):.3f}; REFIL {r_curve[0]:.2f} -> {r_curve[-1]:.2f}")
    print("== Collision fraction by group, LEAF vs REFIL (%)")
    for k in GROUPS:
        a, b = g[LEAF][k]["collision"]["mean"] * 100, g["refil"][k]["collision"]["mean"] * 100
        print(f"  {k}: {a:.2f} vs {b:.2f} ratio {a / b:.2f}")
    print("== P (%) and CVaR")
    pc = data["p_cvar"]
    for m in BASELINES + ("rule", LEAF):
        print(f"  {m:14s} P", "  ".join(f"{100 * pc[m][k]['P']['mean']:.1f}" + (f"+-{100 * pc[m][k]['P']['sd']:.1f}"
                                         if pc[m][k]['P']['sd'] else "") for k in GROUPS),
              "| CVaR", "  ".join(f"{pc[m][k]['CVaR']['mean']:.2f}" + (f"+-{pc[m][k]['CVaR']['sd']:.2f}"
                                    if pc[m][k]['CVaR']['sd'] else "") for k in GROUPS))
    for k in GROUPS:
        for metric in ("P", "CVaR"):
            best = min(methods, key=lambda m: pc[m][k][metric]["mean"])
            print(f"  {k} {metric}: LEAF vs strongest ({best}) "
                  f"{100 * (pc[LEAF][k][metric]['mean'] / pc[best][k][metric]['mean'] - 1):+.1f}%")
    print("== Ablation chain D")
    for m in ("regir_r0_sg", "regir_r1_sg", "regir_fixed4_sg", "regir_sg", LEAF, "regir_untied4_sg",
              "regir_last_sg", "regir_norefil_sg", "refil_matched"):
        print(f"  {m:18s}", "  ".join(f"{g[m][k]['mean']:.3f}+-{g[m][k]['sd']:.3f}" for k in GROUPS))
    chain = ("regir_r0_sg", "regir_r1_sg", "regir_fixed4_sg", "regir_sg", LEAF)
    for a, b in zip(chain, chain[1:]):
        rel = [100 * (g[b][k]["mean"] / g[a][k]["mean"] - 1) for k in GROUPS]
        better = [sum(y < x for x, y in zip(g[a][k]["seeds"], g[b][k]["seeds"])) for k in GROUPS]
        print(f"  {a} -> {b}:", " ".join(f"{r:+.0f}%" for r in rel), "seeds improved", better)
    print("  REFIL-matched / LEAF:", " ".join(f"{g['refil_matched'][k]['mean'] / g[LEAF][k]['mean']:.2f}" for k in GROUPS))
    print("== Parameters", data["parameters"])
    print("== Validation")
    val = data["validation"]
    for m in (LEAF, "regir_r1_sg", "regir_fixed4_sg", "regir_r0_sg", "refil", "refil_matched") + BASELINES:
        seeds = np.asarray(val[m]["seeds"]).mean(1)
        hit = next((t for t, y in zip(val[m]["t"], val[m]["mean"]) if y <= 1.0), None)
        print(f"  {m:18s} mean over 50 points {seeds.mean():.2f}+-{seeds.std(ddof=1):.2f}  first D<=1.0 at {hit}")
    print("== Execution depth, three K=2 configurations (10, 30, 50)")
    for m in (LEAF, "regir_sg", "regir_fixed4_sg", "regir_untied4_sg"):
        prof = data["depth"][m]
        line = []
        ratios = []
        for d in map(str, range(1, 7)):
            if all(d in prof[c] for c in ("10-2", "30-2", "50-2")):
                line.append(f"L{d} {np.mean([np.mean(prof[c][d]) for c in ('10-2', '30-2', '50-2')]):.3f}")
        for s in range(3):
            r1 = np.mean([prof[c]["1"][s] for c in ("10-2", "30-2", "50-2")])
            r4 = np.mean([prof[c]["4"][s] for c in ("10-2", "30-2", "50-2")])
            ratios.append(round(r1 / r4, 2))
        print(f"  {m:18s}", " ".join(line), "| L1/L4 per seed", ratios)
    a24 = data["nomem_depth_all24"]["All24"]["depths"]
    print("  LEAF All24:", {d: round(v["mean"], 4) for d, v in a24.items()})
    print("== Fusion", {c: [round(100 * a, 2) for a in v["alpha"]] + [round(100 * (v["alpha"][0] + v["alpha"][1]), 2)]
                        for c, v in data["fusion"].items()})
    print("== Timing 50-2", {a: round(v["ms"], 2) for a, v in data["timing"]["closed_loop"]["50-2"].items()})
    print("== Margin", {c: v for c, v in list(data["margin"]["cells"].items())})
    md = data["memory_drift"]
    print("== Memory drift AUC")
    for shifted in DRIFT_SHIFTED:
        for feature in ("memory", "self"):
            for name in ("linear", "nonlinear", "pca8"):
                v = md["auc"][shifted][feature][name]
                print(f"  {shifted} {feature:6s} {name:9s} {np.mean(v):.3f}+-{np.std(v, ddof=1):.3f}", [round(x, 3) for x in v])
        print("  balanced pairs", md["pairs"][shifted])
    print("  memory norm", {k: round(v, 2) for k, v in md["memory_norm"].items()})
    print("== Confirmation All24", {m: round(v["mean"], 3) for m, v in data["confirmation_all24"].items()})
    print("== Frozen interventions dD")
    for cfg, arms in data["frozen_interventions"].items():
        print(f"  {cfg}", {a: round(v["dD"]["mean"], 2) for a, v in arms.items()})
    print("== Memory query readout", {c: {r: round(v["mean"], 2) for r, v in rr.items()}
                                      for c, rr in data["readout_interventions"].items()})


LABELS = {"b2_qmix_atten": "QMIX", "refil": "REFIL", "dcg": "DCG", "alma": "ALMA", "transfqmix": "TransfQMix",
          "spectra": "SPECTra", "rule": r"Rule$^\dagger$", LEAF: "LEAF"}


def marked(values, fmt, lower=True):
    """Format a column with the best entry in bold and the second best underlined."""
    order = np.argsort(values if lower else [-v for v in values])
    out = [fmt(v) for v in values]
    out[order[0]] = rf"\textbf{{{out[order[0]]}}}"
    out[order[1]] = rf"\underline{{{out[order[1]]}}}"
    return out


def tables(data):
    """Print the LaTeX rows of the main-result and ablation tables."""
    methods = BASELINES + ("rule", LEAF)
    g, pc = data["group"], data["p_cvar"]
    cols = []
    for k in GROUPS:
        means = [g[m][k]["mean"] for m in methods]
        best = marked(means, lambda v: f"{v:.2f}")
        cols.append([f"{b}$\\pm${(g[m][k]['sd'] or data['rule_group_scene_sd'][k]):.2f}"
                     for b, m in zip(best, methods)])
    for k in GROUPS:
        cols.append(marked([100 * pc[m][k]["P"]["mean"] for m in methods], lambda v: f"{v:.1f}"))
    for k in GROUPS:
        cols.append(marked([pc[m][k]["CVaR"]["mean"] for m in methods], lambda v: f"{v:.2f}"))
    for i, m in enumerate(methods):
        cells = [c[i] for c in cols]
        print(f"{LABELS[m]} & " + " & ".join(cells[:4]) + " && " + " & ".join(cells[4:8]) + " && "
              + " & ".join(cells[8:]) + r"\\")
    rel = []
    for k in GROUPS:
        rel.append(g[LEAF][k]["mean"] / min(g[m][k]["mean"] for m in methods[:-1]) - 1)
    for metric in ("P", "CVaR"):
        for k in GROUPS:
            rel.append(pc[LEAF][k][metric]["mean"] / min(pc[m][k][metric]["mean"] for m in methods[:-1]) - 1)
    fmt = lambda r: f"$-${abs(100 * r):.0f}\\%"
    print(r"LEAF vs.\ best & " + " & ".join(map(fmt, rel[:4])) + " && " + " & ".join(map(fmt, rel[4:8])) + " && "
          + " & ".join(map(fmt, rel[8:])) + r"\\")
    print()
    chain = ("regir_r0_sg", "regir_r1_sg", "regir_fixed4_sg", "regir_sg", LEAF)
    for i, m in enumerate(chain):
        cells = []
        for k in GROUPS:
            cell = f"{g[m][k]['mean']:.2f}$\\pm${g[m][k]['sd']:.2f}"
            if i:
                r = 100 * (g[m][k]["mean"] / g[chain[i - 1]][k]["mean"] - 1)
                cell += f" {{\\scriptsize({'+' if r > 0 else '$-$'}{abs(r):.0f}\\%)}}"
            cells.append(cell)
        print(m, "& " + " & ".join(cells) + r"\\")
    for m in ("regir_untied4_sg", "regir_last_sg", "regir_norefil_sg", "refil_matched"):
        print(m, "& " + " & ".join(f"{g[m][k]['mean']:.2f}$\\pm${g[m][k]['sd']:.2f}" for k in GROUPS) + r"\\")
    print()
    print("mean validation D over 50 points:",
          {m: round(float(np.asarray(v["seeds"]).mean()), 2) for m, v in sorted(data["validation"].items())})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="recompute from the saved records")
    parser.add_argument("--tables", action="store_true", help="print the LaTeX rows of the Section V tables")
    args = parser.parse_args()
    data = load(args.rebuild)
    tables(data) if args.tables else digest(data)
