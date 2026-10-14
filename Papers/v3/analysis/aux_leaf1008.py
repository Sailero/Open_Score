"""Fourteen preselected frozen-final HAD trajectory replays for paper v3.

Only Papers/v3 is written. Reuse the canonical loader, runner, and auxiliary
serialization/replay comparison helpers; do not train or change formal data.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import csv
import gzip
import json
from pathlib import Path
import platform
import random
import time

PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parents[1]
SOURCE = REPO / "Open-SCORE"
OUTPUT = SOURCE / "outputs/main"
sys.path.insert(0, str(SOURCE / "scripts"))
sys.path.insert(0, str(SOURCE))

import numpy as np
import torch
from aux_leaf1007 import physical_state, replay_check, save, REPLAY_FIELDS
from open_score.algos import load_policy
from open_score.eval.anchors import ANCHOR_STRATEGIES, BLUE_STRATEGY
from open_score.rules import register_end_to_end_policy, run_episode
from had_env.core.version import CORE_VERSION, PHYSICS_PROTOCOL

CONFIGS = ((5, 5, 2), (15, 15, 2))
METHODS = ("regir_nomem", "refil", "b2_qmix_atten", "dcg", "spectra", "alma", "rule_nv1")
LABELS = dict(zip(METHODS, ("LEAF", "REFIL", "QMIX-Atten", "DCG", "SPECTra", "ALMA", "Rule")))
SCENE, SEED, SNAPSHOT = 9000, 0, 30
SCHEMA = "had-seven-method-fixed-trajectories1008-v1"
DATA = PAPER / "data/trajectories1008.json.gz"


def key(e):
    return e["method"], tuple(e["config"])


def references(checkpoint_ids):
    wanted = {}
    with (OUTPUT / "had/episodes.csv").open(encoding="utf-8-sig", newline="") as f:
        for raw in csv.DictReader(f):
            if raw.get("archive_record_status", "").strip('"') == "historical_outcome_variant":
                continue
            method = json.loads(raw["method"])
            if method not in METHODS or int(json.loads(raw["episode_seed"])) != SCENE:
                continue
            phase = json.loads(raw["phase"])
            if phase != ("anchor" if method == "rule_nv1" else "final_eval"):
                continue
            if method != "rule_nv1" and (int(json.loads(raw["seed"])) != SEED or json.loads(raw["arm"]) != "final"):
                continue
            cfg = json.loads(raw["config"])
            cfg = tuple(int(cfg[k]) for k in ("N_R", "N_B", "K"))
            if cfg not in CONFIGS:
                continue
            fields = (*REPLAY_FIELDS, "checkpoint_id", "recorded_at")
            row = {k: json.loads(raw[k]) if raw[k] else None for k in fields}
            if method != "rule_nv1" and row["checkpoint_id"] != checkpoint_ids[method]:
                raise RuntimeError(f"Formal checkpoint mismatch: {method}/{cfg}")
            if (method, cfg) in wanted:
                raise RuntimeError(f"Duplicate formal reference: {method}/{cfg}")
            wanted[(method, cfg)] = row
    if len(wanted) != 14:
        raise RuntimeError(f"Expected 14 formal/anchor references, found {len(wanted)}")
    return wanted


def summarize(data):
    groups = {}
    rows = []
    for ep in data["episodes"]:
        cfg = tuple(ep["config"])
        initial = ep["initial_state"]
        if cfg in groups and initial != groups[cfg]:
            raise RuntimeError(f"Methods do not share the physical initial state: {cfg}")
        groups[cfg] = initial
        s = ep["episode_summary"]
        frame = next((f for f in ep["trajectory"] if f["step"] == min(SNAPSHOT, s["ep_len"])), None)
        if frame is None:
            raise RuntimeError("Missing actual snapshot frame")
        ep["snapshot"] = {"planned_step": SNAPSHOT, "actual_step": frame["step"],
            "early_terminal": s["ep_len"] < SNAPSHOT,
            "R_at_actual_step": -sum(t["cumulative_damage"] for t in frame["targets"])}
        rows.append({"method": ep["method"], "config": ep["config"], "R": -s["D"],
            "collision_percent": 100*s["red_deaths_by_cause"]["friendly_collision"]/cfg[0],
            "intercepted_fraction": s["blue_deaths_by_cause"]["intercepted"]/cfg[1],
            "red_left": s["red_left"], "blue_left": s["blue_left"], "end_step": s["ep_len"],
            "damage_by_target": s["damage_by_target"], "first_damage_step_by_target": s["first_damage_step_by_target"],
            "formal_11_fields_matched": ep["replay_validation"]["matched"],
            "different_fields": list(ep["replay_validation"]["differences"]), "checkpoint_id": ep["checkpoint_id"]})
    data["summary"] = {"cases": rows, "completed_episodes": len(rows),
        "matched_formal_or_anchor": sum(r["formal_11_fields_matched"] for r in rows),
        "same_initial_state_checked_configs": [list(c) for c in groups],
        "scope": "Fixed illustrative scene, not a new population estimate or checkpoint selection"}
    data["execution_status"] = "complete" if len(rows) == 14 else "partial"


def plot(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 8,
        "axes.titlesize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "DejaVu Sans"})
    colors = dict(zip(METHODS, ("#087f8c", "#ed8b2d", "#4778b6", "#9660b7", "#ca566f", "#83a64b", "#6b737f")))
    fig, axes = plt.subplots(4, 4, figsize=(7.16, 6.6))
    spatial_handles = [Line2D([], [], color="#17669b", label="Defenders"),
        Line2D([], [], color="#cf5f2a", label="Attackers"),
        Line2D([], [], color="#252525", marker="*", linestyle="none", label="Targets"),
        Line2D([], [], color="#666666", marker="o", markerfacecolor="none", linestyle="none", label="Initial"),
        Line2D([], [], color="#666666", marker="x", linestyle="none", label="Destroyed")]
    for block, cfg in enumerate(CONFIGS):
        eps = [next(e for e in data["episodes"] if key(e) == (method, cfg)) for method in METHODS]
        histories, all_xy = [], []
        for ep in eps:
            fs = [ep["initial_state"], *[f for f in ep["trajectory"] if f["step"] <= SNAPSHOT]]
            histories.append(fs)
            all_xy.extend([np.asarray(e["position"][:2])/1000 for f in fs for side in ("red", "blue", "targets") for e in f[side]])
        extent = np.asarray(all_xy)
        lo, hi = extent.min(0)-.5, extent.max(0)+.5
        for index, (ep, fs) in enumerate(zip(eps, histories)):
            ax = axes[2*block+index//4, index%4]
            for side, color in (("red", "#17669b"), ("blue", "#cf5f2a")):
                for i in range(len(fs[0][side])):
                    xy = np.asarray([f[side][i]["position"][:2] for f in fs])/1000
                    ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=.65, alpha=.7 if block == 0 else .55)
                    ax.scatter(*xy[0], s=7, facecolors="none", edgecolors=color, linewidths=.5)
                    ax.scatter(*xy[-1], s=10, color=color,
                        marker="o" if fs[-1][side][i]["alive"] else "x", linewidths=.65, zorder=4)
            for i, t in enumerate(fs[0]["targets"]):
                xy = np.asarray(t["position"][:2])/1000
                ax.scatter(*xy, marker="*", s=33, color="#252525", zorder=5)
                ax.annotate(f"T{i+1}", xy, xytext=(3, 4), textcoords="offset points", fontsize=8)
            s = ep["snapshot"]
            stamp = f"end T={s['actual_step']} < 30" if s["early_terminal"] else "t=30"
            shown_reward = 0. if abs(s["R_at_actual_step"]) < 1e-9 else s["R_at_actual_step"]
            ax.set_title(f"{LABELS[ep['method']]} ({cfg[0]}v{cfg[1]})\n{stamp}, R={shown_reward:.2g}", pad=3)
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
            ax.set_aspect("equal", adjustable="box")
            ax.xaxis.set_major_locator(MaxNLocator(3)); ax.yaxis.set_major_locator(MaxNLocator(3))
            ax.tick_params(pad=1, length=2)
            if index % 4 == 0: ax.set_ylabel("y (km)", labelpad=1)
            if index >= 4: ax.set_xlabel("x (km)", labelpad=1)
            if cfg == (15, 15, 2) and ep["method"] in ("regir_nomem", "refil"):
                frame = fs[-1]
                alive_defenders = sum(bool(a["alive"]) for a in frame["red"])
                snapshot_damage = sum(t["cumulative_damage"] for t in frame["targets"])
                if snapshot_damage <= 1e-12:
                    protection = "No target damage"
                else:
                    first_damage = next(f["step"] for f in ep["trajectory"]
                        if f["step"] <= frame["step"]
                        and sum(t["cumulative_damage"] for t in f["targets"]) > 1e-12)
                    protection = f"First damage: t={first_damage}"
                # Use the lower outer corner so the box covers no recorded path.
                ax.text(.97, -.22, f"{alive_defenders} defenders alive\n{protection}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    linespacing=1.05, clip_on=False, zorder=8,
                    bbox={"boxstyle": "round,pad=.22", "facecolor": "white",
                          "edgecolor": "#888888", "linewidth": .5})
        ax = axes[2*block+1, 3]
        for ep in eps:
            times = [0]+[f["step"] for f in ep["trajectory"]]
            rewards = [0.]+[-sum(t["cumulative_damage"] for t in f["targets"]) for f in ep["trajectory"]]
            ax.step(times, rewards, where="post", color=colors[ep["method"]], linewidth=1.2, alpha=.85)
            ax.scatter(times[-1], rewards[-1], s=13, color=colors[ep["method"]], zorder=4)
        ax.axvline(SNAPSHOT, color="#777777", linestyle="--", linewidth=.7)
        ax.set_title(f"{cfg[0]}v{cfg[1]}, K=2\nFull replay reward", pad=3)
        ax.set_xlim(0, 100); ax.set_xticks((0, 30, 60, 100))
        ax.set_xlabel("Physical step (1 s)", labelpad=1); ax.set_ylabel("Cumulative R = −D", labelpad=1)
        ax.yaxis.set_major_locator(MaxNLocator(4)); ax.grid(alpha=.18)
        ax.tick_params(pad=1, length=2)
    fig.legend(handles=spatial_handles, loc="upper center", ncol=5, frameon=False,
        bbox_to_anchor=(.5, 1.0), handlelength=1.2, columnspacing=1)
    fig.legend(handles=[Line2D([], [], color=colors[m], label=LABELS[m]) for m in METHODS],
        loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, -.002),
        handlelength=1.3, columnspacing=1.2)
    fig.subplots_adjust(left=.07, right=.985, bottom=.10, top=.912, hspace=.98, wspace=.56)
    figures = PAPER/"figures"
    figures.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(figures/f"fig_leaf_final1007_trajectory.{ext}", dpi=220)
    plt.close(fig)
    data["figures"] = {"trajectory": "figures/fig_leaf_final1007_trajectory.pdf",
        "width_inches": 7.16, "height_inches": 6.6, "smallest_font_pt": 8,
        "spatial_snapshot": "min(preselected step30, actual termination), identical coordinates within each scale",
        "reward_curves": "actual full episode; terminal dot, no extrapolation after termination"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--plot-only", action="store_true")
    opts = parser.parse_args()
    if opts.plot_only:
        data = json.loads(gzip.decompress(DATA.read_bytes()))
        if data["execution_status"] != "complete": raise RuntimeError("All14 replays required before plotting")
        plot(data); return
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    np.random.seed(0); random.seed(0); torch.manual_seed(0)
    if opts.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    summary = json.loads((OUTPUT/"summary.json").read_text(encoding="utf-8"))
    registered = {c["method"]: c for c in summary["checkpoints"]
        if c.get("env") == "had" and c["method"] in METHODS and int(c["seed"]) == SEED
        and c["kind"] == "final"}
    ids = {method: c["artifact_id"] for method, c in registered.items()}
    refs = references(ids)
    checkpoints, policies = [], {}
    for method in METHODS[:-1]:
        path = REPO / registered[method]["archived_path"]
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["artifact_id"] != ids[method]: raise RuntimeError(f"Registered final mismatch: {method}")
        checkpoints.append({"method": method, "training_seed": SEED, "checkpoint_kind": "final",
            "checkpoint_id": ids[method], "checkpoint_path": str(path.relative_to(REPO)).replace("\\", "/"),
            "t_env": saved["progress"]["t_env"], "strict_agent_mac_mixer_load": True})
        policy = load_policy(method, path); policy.set_device(opts.device)
        if method == "regir_nomem": policy.set_eval_depth(4)
        policies[method] = policy
        register_end_to_end_policy("red", f"paper1008_{method}", "frozen final fixed trajectory", lambda _, p=policy: p)
    if DATA.exists():
        data = json.loads(gzip.decompress(DATA.read_bytes()))
        if data["schema"] != SCHEMA or data["runtime"]["device"] != opts.device or data["checkpoints"] != checkpoints:
            raise RuntimeError("Resume identity changed")
    else:
        data = {"schema": SCHEMA, "checkpoints": checkpoints, "episodes": [], "execution_status": "partial",
            "evaluation": {"configs": [list(c) for c in CONFIGS], "methods": list(METHODS),
                "learning_model_seed": SEED, "Rule_training_seed": None, "episode_seed": SCENE,
                "max_steps": 100, "task_mode": "damage", "spatial_dim": 2, "target_initialization": "random",
                "blue_strategy": BLUE_STRATEGY, "Rule_strategy": ANCHOR_STRATEGIES["rule_nv1"],
                "LEAF_depth": 4, "snapshot_step": SNAPSHOT, "planned_episodes": 14, "training_updates": 0,
                "policy_seed": "torch.manual_seed(SeedSequence([training_seed,N_R,N_B,K,episode_seed]).generate_state(1)[0])",
                "selection": "Prespecified two scales, scene9000, modelseed0 final; no result-based selection"},
            "runtime": {"device": opts.device, "python": sys.version, "torch": torch.__version__,
                "numpy": np.__version__, "platform": platform.platform(), "threads": torch.get_num_threads(),
                "tf32_matmul": False, "tf32_cudnn": False, "had_core_version": CORE_VERSION,
                "had_physics_protocol": PHYSICS_PROTOCOL},
            "source": {"formal_csv": "Open-SCORE/outputs/main/had/episodes.csv",
                "runner": "Papers/v3/analysis/aux_leaf1008.py", "reuse": "Open-SCORE/scripts/aux_leaf1007.py helpers and canonical load_policy/run_episode",
                "formal_data_modified": False, "historical_differences": "Retained; local illustrative replay does not replace formal population statistics"}}
    existing = {key(e) for e in data["episodes"]}
    for cfg in CONFIGS:
        for method in METHODS:
            if (method, cfg) in existing: continue
            initial, actions = [], []
            def record(step, state, native_actions, reward, info):
                if not initial: initial.append(physical_state(state))
                actions.append({"decision_step": int(state.step), "result_step": int(step),
                    "actions": {str(k): int(v) for k, v in native_actions.items()}, "native_reward": float(reward)})
            strategy = ANCHOR_STRATEGIES["rule_nv1"] if method == "rule_nv1" else {
                "architecture": "end_to_end", "policy": f"paper1008_{method}"}
            torch.manual_seed(int(np.random.SeedSequence([SEED, *cfg, SCENE]).generate_state(1)[0]))
            started = time.perf_counter()
            with torch.inference_mode():
                result = run_episode(red=cfg[0], blue=cfg[1], targets=cfg[2], seed=SCENE,
                    red_strategy=strategy, blue_strategy=BLUE_STRATEGY, max_steps=100,
                    task_mode="damage", spatial_dim=2, target_initialization="random",
                    diagnostics=True, record_events=True, retain_trajectory=True, record=False, on_step=record)
            s = dict(result["episode_summary"])
            for f in result["trajectory"]:
                f["physical_reward"] = -float(f["step_damage"])
                f["target_step_rewards"] = [-float(t["step_damage"]) for t in f["targets"]]
            if abs(sum(f["physical_reward"] for f in result["trajectory"])+s["D"]) > 1e-7:
                raise RuntimeError("Trajectory reward is inconsistent with native -D")
            ep = {"method": method, "training_seed": None if method == "rule_nv1" else SEED,
                "config": list(cfg), "episode_seed": SCENE, "arm": "standard", "checkpoint_kind": "rule" if method == "rule_nv1" else "final",
                "checkpoint_id": None if method == "rule_nv1" else ids[method], "episode_summary": s,
                "initial_state": initial[0], "trajectory": result["trajectory"], "events": result["events"],
                "actions": actions, "bounds": result["bounds"], "terminated": result["terminated"], "truncated": result["truncated"],
                "elapsed_seconds": time.perf_counter()-started, "replay_validation": replay_check(s, refs[(method, cfg)])}
            data["episodes"].append(ep); existing.add(key(ep)); summarize(data); save(data, DATA)
            print(json.dumps({"completed": len(data["episodes"]), "method": method, "config": cfg,
                "R": -s["D"], "T": s["ep_len"], "matched": ep["replay_validation"]["matched"],
                "different_fields": list(ep["replay_validation"]["differences"]), "seconds": round(ep["elapsed_seconds"], 2)}), flush=True)
    summarize(data)
    if data["execution_status"] == "complete": plot(data)
    save(data, DATA)
    print("DATA", DATA, "STATUS", data["execution_status"], flush=True)


if __name__ == "__main__":
    main()
