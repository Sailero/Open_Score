"""Frozen loop/fusion controls for LEAF v2; all writes stay under Papers/v2.

Reuse 90 prespecified factual scenes (3 models x 3 configurations x 10 scenes).
Recapture standard execution and evaluate H0 reading and fixed empirical fusion.
No optimization, new training seeds, or changes to formal evaluation records.

PowerShell launch from the repository root (HAD is an external source dependency):
    $env:PYTHONPATH='E:/Code/SkyHAD'
    & 'D:/Software/Anaconda/envs/torch310/python.exe' Papers/v2/analysis/mechanism_leaf1008.py
After all270 episodes are saved, append --summarize-only to compute offline
statistics and publication figures without running a policy or environment.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
import base64
import gzip
import json
from pathlib import Path
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
from aux_leaf1007 import (save, packed, replay_check, rng_snapshot, rng_restore,
    CONFIGS, SCENES, physical_state)
from open_score.algos import load_policy
from open_score.eval.anchors import BLUE_STRATEGY
from open_score.rules import register_end_to_end_policy, run_episode
from had_env.core.version import CORE_VERSION, PHYSICS_PROTOCOL

OLD = OUTPUT / "had/regir_nomem/mechanisms/results.json.gz"
DATA = PAPER / "data/mechanism1008.json.gz"
SCHEMA = "leaf-loop-fusion1008-v1"
ARMS = ("standard", "loop_bypass", "fixed_fusion")


def unpack(x):
    return np.frombuffer(base64.b64decode(x["base64"]), dtype=x["dtype"]).reshape(x["shape"])


def ek(e):
    return int(e["training_seed"]), tuple(e["config"]), int(e["episode_seed"]), e["arm"]


def fixed_weights(old):
    weights, per_cfg = {}, []
    for seed in range(3):
        means = []
        for cfg in CONFIGS:
            ds = [d for e in old["episodes"] if e["method"] == "regir_nomem" and
                e["arm"] == "standard" and e["training_seed"] == seed and tuple(e["config"]) == cfg
                for d in e["decisions"]]
            n = sum(d["n_active"] for d in ds)
            a = np.sum([d["alpha_sum"] for d in ds], axis=0) / n
            means.append(a)
            per_cfg.append({"training_seed": seed, "config": list(cfg), "decisions": n, "alpha": a.tolist()})
        w = np.mean(means, axis=0)
        weights[str(seed)] = (w / w.sum()).tolist()
    return weights, per_cfg


class Control:
    def __init__(self, policy, weights):
        self.policy, self.branch = policy, policy.mac.agent.global_net
        self.weights = weights
        self.arm, self.capture = "standard", False
        self.handle = self.branch.read_score.register_forward_hook(self._score)

    def _score(self, module, inputs, output):
        index = self.index
        self.index += 1
        if self.capture:
            self.reads.append(inputs[0].detach().clone())
            self.scores.append(output.detach().clone())
        if self.mode == "fixed_fusion":
            return torch.full_like(output, float(np.log(self.weights[index % 4])))

    def configure(self, arm):
        self.mode, self.index = arm, 0
        self.policy.set_eval_depth(4)
        self.branch.readout_override = None
        self.branch.args.global_read_h0 = arm == "loop_bypass"

    def reset(self):
        self.policy.reset()
        self.configure(self.arm)
        self.decisions, self.initial = [], None

    @torch.no_grad()
    def act(self, state, side, action_ids):
        p, mac = self.policy, self.policy.mac
        if self.initial is None: self.initial = physical_state(state)
        self.configure(self.arm)
        before = mac.hidden_states.clone()
        self.capture = self.arm == "standard"
        self.reads, self.scores = [], []
        actions = p.act(state, side, action_ids)
        self.capture = False
        if self.arm != "standard": return actions
        after, cache = mac.hidden_states.clone(), mac._decision_q.clone()
        alive = [i for i, e in enumerate(state.red) if e.alive]
        if not alive: return actions
        q0 = cache.reshape(p.args.n_agents, -1)[alive]
        legal = p.batch["avail_actions"][:, int(state.step)].reshape(p.args.n_agents, -1).bool()[alive]
        a0 = q0.masked_fill(~legal, -torch.inf).argmax(-1)
        ids = [int(state.red[i].id) for i in alive]
        assert all(actions[e] == int(action_ids[int(a)]) for e, a in zip(ids, a0))
        assert len(self.reads) == len(self.scores) == 4
        reads = torch.stack(self.reads, dim=1)[alive]
        alpha = torch.softmax(torch.cat(self.scores, dim=-1), dim=-1)[alive]
        early = (reads[:, :2] * alpha[:, :2, None]).sum(1) / alpha[:, :2].sum(1, keepdim=True)
        full = (reads * alpha[:, :, None]).sum(1)
        projection_delta = torch.nn.functional.linear(full - early, self.branch.fuse.weight)
        norm_reads = torch.nn.functional.normalize(reads, dim=-1)
        cosines = (norm_reads[:, 1:] * norm_reads[:, :-1]).sum(-1)
        alternatives = {}
        main_rng = rng_snapshot()
        try:
            for arm in ARMS[1:]:
                mac.hidden_states = before.clone()
                self.configure(arm)
                rng_restore(main_rng)
                mac.forward(p.batch, t=int(state.step), test_mode=True)
                q = mac._decision_q.reshape(p.args.n_agents, -1)[alive].clone()
                aa = q.masked_fill(~legal, -torch.inf).argmax(-1)
                alternatives[arm] = {"q_delta": packed(q-q0), "greedy_action": aa.cpu().tolist(),
                    "changed_defender_ids": [ids[i] for i in (aa != a0).nonzero().flatten().cpu().tolist()]}
        finally:
            mac.hidden_states, mac._decision_q = after, cache
            self.configure("standard")
            rng_restore(main_rng)
        self.decisions.append({"step": int(state.step), "active_defender_ids": ids, "n_active": len(alive),
            "q_standard": packed(q0), "legal_actions": legal.cpu().tolist(), "greedy_standard": a0.cpu().tolist(),
            "alpha": packed(alpha), "read_norms": packed(reads.norm(dim=-1)),
            "adjacent_read_cosines": packed(cosines), "projected_L4_minus_L2_norm": packed(projection_delta.norm(dim=-1)),
            "alternatives": alternatives})
        return actions


def stats(values):
    x = np.asarray(values, dtype=float)
    return {"mean": float(x.mean()), "sample_sd": float(x.std(ddof=1)), "per_seed": x.tolist()}


def summarize(data):
    eps = {ek(e): e for e in data["episodes"]}
    rows = []
    for cfg in CONFIGS:
        for arm in ARMS[1:]:
            reward, changes = [], []
            for seed in range(3):
                keys = [(seed, cfg, scene, arm) for scene in SCENES]
                if not all(k in eps and (*k[:3], "standard") in eps for k in keys): break
                reward.append(float(np.mean([-eps[k]["episode_summary"]["D"] +
                    eps[(*k[:3], "standard")]["episode_summary"]["D"] for k in keys])))
                ds = [d for scene in SCENES for d in eps[seed, cfg, scene, "standard"]["decisions"]]
                changes.append(sum(len(d["alternatives"][arm]["changed_defender_ids"]) for d in ds) /
                    sum(d["n_active"] for d in ds))
            if len(reward) == 3:
                rows.append({"config": list(cfg), "arm": arm, "paired_reward": stats(reward), "action_change_fraction": stats(changes)})
    recaptured = [e for e in eps.values() if e["arm"] == "standard"]
    data["summary"] = {"interventions": rows, "completed_episodes": len(eps),
        "standard_recaptures": len(recaptured),
        "standard_matches_old_local": sum(e["old_local_replay_validation"]["matched"] for e in recaptured)}
    data["execution_status"] = "complete" if len(eps) == 270 else "partial"


def offline_analysis(data, old):
    """Describe saved factual decisions; never call a policy or environment."""
    margin_rows, margin_groups = [], []
    checks = {"standard_argmax_mismatch": 0, "changed_ids_vs_greedy_mismatch": 0,
        "reconstructed_L2_argmax_mismatch": 0, "nonfinite_rows": 0}
    for cfg in CONFIGS:
        rows = []
        for seed in range(3):
            eps = [e for e in old["episodes"] if e["method"] == "regir_nomem" and
                e["arm"] == "standard" and e["training_seed"] == seed and tuple(e["config"]) == cfg]
            if sorted(e["episode_seed"] for e in eps) != list(SCENES):
                raise ValueError(f"Incomplete old factual bank: {cfg}, seed {seed}")
            margins, deltas, alphas, changes = [], [], [], []
            for e in eps:
                for d in e["decisions"]:
                    q = unpack(d["q_standard"]).astype(np.float64)
                    delta = unpack(d["alternatives"]["L2"]["q_delta"]).astype(np.float64)
                    alpha = unpack(d["alpha"]).astype(np.float64)
                    changed = np.isin(d["active_defender_ids"],
                        d["alternatives"]["L2"]["changed_defender_ids"])
                    a0 = np.asarray(d["greedy_standard"])
                    a2 = np.asarray(d["alternatives"]["L2"]["greedy_action"])
                    checks["standard_argmax_mismatch"] += int(np.sum(q.argmax(1) != a0))
                    checks["changed_ids_vs_greedy_mismatch"] += int(np.sum(changed != (a2 != a0)))
                    checks["reconstructed_L2_argmax_mismatch"] += int(np.sum((q + delta).argmax(1) != a2))
                    checks["nonfinite_rows"] += int(np.sum(~np.isfinite(q).all(1) |
                        ~np.isfinite(delta).all(1) | ~np.isfinite(alpha).all(1)))
                    sorted_q = np.sort(q, axis=1)
                    margins.append(sorted_q[:, -1] - sorted_q[:, -2])
                    deltas.append(np.abs(delta).max(1)); alphas.append(alpha); changes.append(changed)
            margin, delta, alpha, changed = map(np.concatenate, (margins, deltas, alphas, changes))
            late = alpha[:, 2:].sum(1)
            certificate = margin > 2 * delta
            threshold = float(np.quantile(margin, .1))
            low = margin <= threshold
            n, nc = len(margin), int(changed.sum())
            if not 0 < nc < n:
                raise ValueError(f"Conditional margin means undefined: {cfg}, seed {seed}")
            row = {"config": list(cfg), "training_seed": seed, "episodes": len(eps),
                "active_defender_decisions": n, "changed_actions": nc,
                "action_change_fraction": nc / n,
                "mean_margin_changed": float(margin[changed].mean()),
                "mean_margin_unchanged": float(margin[~changed].mean()),
                "median_margin_changed": float(np.median(margin[changed])),
                "median_margin_unchanged": float(np.median(margin[~changed])),
                "mean_late_alpha_changed": float(late[changed].mean()),
                "mean_late_alpha_unchanged": float(late[~changed].mean()),
                "mean_alpha": alpha.mean(0).tolist(),
                "stability_certificate_count": int(certificate.sum()),
                "stability_certificate_fraction": float(certificate.mean()),
                "stability_certificate_counterexamples": int(np.sum(certificate & changed)),
                "q10_margin": threshold, "low10_decision_count": int(low.sum()),
                "low10_actual_fraction": float(low.mean()),
                "changes_in_low10": int(np.sum(low & changed)),
                "low10_share_of_changes": float(np.sum(low & changed)) / nc,
                "low10_change_fraction": float(changed[low].mean()),
                "rest90_change_fraction": float(changed[~low].mean())}
            rows.append(row); margin_rows.append(row)
        fields = ("action_change_fraction", "mean_margin_changed", "mean_margin_unchanged",
            "median_margin_changed", "median_margin_unchanged", "mean_late_alpha_changed",
            "mean_late_alpha_unchanged", "stability_certificate_fraction", "q10_margin",
            "low10_actual_fraction", "low10_share_of_changes", "low10_change_fraction", "rest90_change_fraction")
        group = {"config": list(cfg), "active_defender_decisions": sum(r["active_defender_decisions"] for r in rows),
            "changed_actions": sum(r["changed_actions"] for r in rows),
            "stability_certificate_counterexamples": sum(r["stability_certificate_counterexamples"] for r in rows)}
        group.update({field: stats([r[field] for r in rows]) for field in fields})
        group["alpha"] = vector_stats([r["mean_alpha"] for r in rows])
        margin_groups.append(group)
    data["margin_analysis"] = {"source": str(OLD.relative_to(REPO)), "factual_episodes": 90,
        "active_actions": 9, "legal_action_basis": "HAD features.available_actions: every live entity has all nine planar actions; deceased rows excluded. Saved standard and L2 greedy decisions also checked.",
        "margin": "largest minus second-largest standard Q among nine legal actions",
        "stability_certificate": "margin > 2 * max(abs(Q_L2 - Q_L4)); sufficient for same-state greedy stability, not necessary and not a closed-loop guarantee",
        "lowest_decile": "within configuration/model seed, margin <= np.quantile(margin, 0.1), including threshold ties",
        "statistics": "pool active decisions within each model/configuration's ten scenes; across three model seeds report equal-weight mean and sample SD, not a pooled-decision uncertainty estimate",
        "integrity": checks, "per_seed": margin_rows, "across_seeds": margin_groups}

    read_rows, read_groups = [], []
    for cfg in CONFIGS:
        rows = []
        for seed in range(3):
            eps = [e for e in data["episodes"] if e["arm"] == "standard" and
                e["training_seed"] == seed and tuple(e["config"]) == cfg]
            if sorted(e["episode_seed"] for e in eps) != list(SCENES):
                raise ValueError(f"Incomplete recaptured bank: {cfg}, seed {seed}")
            ds = [d for e in eps for d in e["decisions"]]
            arrays = {key: np.concatenate([unpack(d[key]).astype(np.float64) for d in ds])
                for key in ("read_norms", "adjacent_read_cosines", "projected_L4_minus_L2_norm", "alpha")}
            n = sum(d["n_active"] for d in ds)
            row = {"config": list(cfg), "training_seed": seed, "episodes": len(eps),
                "active_defender_decisions": n,
                "mean_read_norms": arrays["read_norms"].mean(0).tolist(),
                "mean_adjacent_read_cosines": arrays["adjacent_read_cosines"].mean(0).tolist(),
                "mean_projected_L4_minus_L2_norm": float(arrays["projected_L4_minus_L2_norm"].mean()),
                "mean_alpha": arrays["alpha"].mean(0).tolist()}
            rows.append(row); read_rows.append(row)
        read_groups.append({"config": list(cfg), "active_defender_decisions": sum(r["active_defender_decisions"] for r in rows),
            "read_norms": vector_stats([r["mean_read_norms"] for r in rows]),
            "adjacent_read_cosines": vector_stats([r["mean_adjacent_read_cosines"] for r in rows]),
            "projected_L4_minus_L2_norm": stats([r["mean_projected_L4_minus_L2_norm"] for r in rows]),
            "alpha": vector_stats([r["mean_alpha"] for r in rows])})
    data["read_diagnostics"] = {"factual_episodes": 90,
        "statistics": "live-decision averages within configuration/model seed; equal mean and sample SD across three seeds",
        "adjacent_pairs": [[1, 2], [2, 3], [3, 4]],
        "projection_difference": "norm of W_f times (learned L4 fusion minus normalized first-two fusion); common bias cancels",
        "per_seed": read_rows, "across_seeds": read_groups}

    # Each intervention uses its own run's local standard, even when recapture matches.
    old_eps = {ek(e): e for e in old["episodes"] if e["method"] == "regir_nomem"}
    new_eps = {ek(e): e for e in data["episodes"]}
    combined_rows, combined_groups = [], []
    plot_arms = ("L1", "L2", "loop_bypass", "read4", "uniform", "fixed_fusion", "residual_zero", "gru_reset")
    for cfg in CONFIGS:
        for arm in plot_arms:
            source, bank = ("mechanism1008", new_eps) if arm in ARMS[1:] else ("auxiliary1007", old_eps)
            rows = []
            for seed in range(3):
                pairs = [(bank[seed, cfg, scene, arm], bank[seed, cfg, scene, "standard"]) for scene in SCENES]
                reward = [-e["episode_summary"]["D"] + s["episode_summary"]["D"] for e, s in pairs]
                ds = [d for _, s in pairs for d in s["decisions"]]
                n = sum(d["n_active"] for d in ds)
                changed = sum(len(d["alternatives"][arm]["changed_defender_ids"]) for d in ds)
                row = {"config": list(cfg), "training_seed": seed, "arm": arm, "source": source,
                    "paired_scenes": len(pairs), "mean_paired_reward": float(np.mean(reward)),
                    "paired_reward_by_scene": [{"episode_seed": scene, "delta_R": float(dr)} for scene, dr in zip(SCENES, reward)],
                    "active_defender_decisions": n, "changed_actions": changed, "action_change_fraction": changed / n}
                rows.append(row); combined_rows.append(row)
            combined_groups.append({"config": list(cfg), "arm": arm, "source": source,
                "paired_reward": stats([r["mean_paired_reward"] for r in rows]),
                "action_change_fraction": stats([r["action_change_fraction"] for r in rows]),
                "active_defender_decisions": sum(r["active_defender_decisions"] for r in rows),
                "changed_actions": sum(r["changed_actions"] for r in rows)})
    data["combined_interventions"] = {"arms": list(plot_arms), "per_seed": combined_rows,
        "across_seeds": combined_groups,
        "statistics": "closed-loop differences paired by scene within model seed; same-state rates pool live factual decisions within model seed; equal mean and sample SD across three model seeds",
        "control": "each arm uses the local standard from its own source run; formal performance tables unchanged"}


def vector_stats(values):
    x = np.asarray(values, dtype=float)
    return {"mean": x.mean(0).tolist(), "sample_sd": x.std(0, ddof=1).tolist(), "per_seed": x.tolist()}


def plot_offline(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans"})
    folder = PAPER / "figures"
    arms = data["combined_interventions"]["arms"]
    labels = {"L1": "L1", "L2": "L2", "loop_bypass": "Read H0 (no loop)", "read4": "Last read only",
        "uniform": "Uniform fusion", "fixed_fusion": "Fixed mean fusion", "residual_zero": "No relation residual", "gru_reset": "GRU reset"}
    means = {(tuple(r["config"]), r["arm"]): r for r in data["combined_interventions"]["across_seeds"]}
    fig, ax = plt.subplots(2, 3, figsize=(7.16, 5.5), sharey=True)
    y = np.arange(len(arms))
    for col, cfg in enumerate(CONFIGS):
        for row in range(2):
            panel = ax[row, col]
            for index, arm in enumerate(arms):
                if arm in ("loop_bypass", "fixed_fusion"):
                    panel.axhspan(index - .46, index + .46, color="#eee9f5", zorder=0)
                metric = means[cfg, arm]["paired_reward" if row == 0 else "action_change_fraction"]
                factor = 1 if row == 0 else 100
                color = "#76539b" if arm in ("loop_bypass", "fixed_fusion") else "#17669b"
                panel.errorbar(factor * metric["mean"], index, xerr=factor * metric["sample_sd"],
                    fmt="o", color=color, capsize=2, markersize=4, linewidth=1.1, zorder=3)
                panel.scatter(factor * np.asarray(metric["per_seed"]), index + np.array([-.12, 0, .12]),
                    color="#555555", s=9, alpha=.7, zorder=4)
            panel.set_ylim(len(arms) - .55, -.55)
            panel.set_yticks(y, [labels[a] for a in arms])
            panel.grid(axis="x", color="#dddddd", linewidth=.5, zorder=0)
            panel.set_axisbelow(True)
            if row == 0:
                panel.axvline(0, color="#888888", linewidth=.7, linestyle="--")
                panel.set_title(f"({chr(97 + col)}) {cfg[0]}v{cfg[1]}, K{cfg[2]}")
                panel.set_xlabel("Paired reward difference\nΔR vs learned L4")
            else:
                panel.set_xlim(0, 100)
                panel.set_xticks([0, 25, 50, 75, 100])
                panel.set_title(f"({chr(100 + col)}) {cfg[0]}v{cfg[1]}, K{cfg[2]}")
                panel.set_xlabel("Same-state action\nchanges (%)")
    fig.legend(handles=[Line2D([], [], marker="o", color="#17669b", markersize=4, label="Mean ± sample SD"),
        Line2D([], [], marker="o", linestyle="", color="#555555", markersize=3, label="Three model seeds")],
        loc="upper center", bbox_to_anchor=(.6, .997), ncol=2, frameon=False)
    fig.subplots_adjust(left=.195, right=.985, bottom=.115, top=.925, wspace=.24, hspace=.38)
    base = folder / "fig_leaf_final1007_mechanism"
    fig.savefig(base.with_suffix(".pdf")); fig.savefig(base.with_suffix(".png"), dpi=300)
    plt.close(fig)

    groups = {tuple(r["config"]): r for r in data["margin_analysis"]["across_seeds"]}
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.85))
    colors = ["#17669b", "#d46a35", "#76539b"]
    short = [f"{cfg[0]}v{cfg[1]}\nK{cfg[2]}" for cfg in CONFIGS]
    for cfg, color in zip(CONFIGS, colors):
        metric = groups[cfg]["alpha"]
        ax[0].errorbar(np.arange(1, 5), 100 * np.asarray(metric["mean"]),
            yerr=100 * np.asarray(metric["sample_sd"]), color=color, marker="o", markersize=3,
            capsize=2, linewidth=1, label=f"{cfg[0]}v{cfg[1]}, K{cfg[2]}")
    ax[0].set(xlabel="Relation round", ylabel="Fusion weight (%)", xticks=[1, 2, 3, 4], ylim=(-3, 100))
    ax[0].legend(loc="upper right", frameon=False, handlelength=1.5, borderpad=.1)
    ax[0].set_title("(a) Learned fusion weights")
    for index, cfg in enumerate(CONFIGS):
        unchanged = groups[cfg]["mean_late_alpha_unchanged"]
        changed = groups[cfg]["mean_late_alpha_changed"]
        for seed in range(3):
            ax[1].plot([index-.16, index+.16], [100*unchanged["per_seed"][seed], 100*changed["per_seed"][seed]],
                color="#999999", linewidth=.6, alpha=.6, zorder=1)
        for metric, offset, color in ((unchanged, -.16, "#17669b"), (changed, .16, "#d46a35")):
            ax[1].scatter(np.full(3, index+offset), 100*np.asarray(metric["per_seed"]), color=color, s=12, alpha=.55, zorder=2)
            ax[1].scatter(index+offset, 100*metric["mean"], color=color, marker="D", s=25, zorder=3)
        metric = groups[cfg]["low10_share_of_changes"]
        ax[2].errorbar(index, 100*metric["mean"], yerr=100*metric["sample_sd"], fmt="o",
            color="#17669b", capsize=3, markersize=4, linewidth=1.1)
        ax[2].scatter(index+np.array([-.08, 0, .08]), 100*np.asarray(metric["per_seed"]), color="#555555", s=10, zorder=3)
    ax[1].set(xticks=range(3), xticklabels=short, ylabel="Rounds 3–4 weight (%)", ylim=(0, 21))
    ax[1].set_title("(b) Late weights at\nL2-changed decisions")
    ax[1].legend(handles=[Line2D([], [], marker="D", linestyle="", color="#17669b", markersize=4, label="Unchanged"),
        Line2D([], [], marker="D", linestyle="", color="#d46a35", markersize=4, label="L2 changed")],
        loc="upper left", frameon=False, handletextpad=.3, borderpad=.1)
    ax[2].axhline(10, linestyle="--", color="#888888", linewidth=.7)
    ax[2].set(xticks=range(3), xticklabels=short, ylabel="Share of L2 action changes (%)", ylim=(0, 103))
    ax[2].set_title("(c) Lowest-margin decile")
    for panel in ax:
        panel.grid(axis="y", color="#dddddd", linewidth=.5)
        panel.set_axisbelow(True)
    fig.subplots_adjust(left=.085, right=.99, bottom=.17, top=.815, wspace=.46)
    base = folder / "fig_leaf_fusion_sensitivity"
    fig.savefig(base.with_suffix(".pdf")); fig.savefig(base.with_suffix(".png"), dpi=300)
    plt.close(fig)
    data["figures"] = [f"figures/{name}.{ext}" for name in
        ("fig_leaf_final1007_mechanism", "fig_leaf_fusion_sensitivity") for ext in ("pdf", "png")]
    data["figure_semantics"] = {"width_inches": 7.16, "minimum_font_points": 8,
        "mechanism": "eight frozen controls; horizontal mean ± sample SD across three model seeds and each seed point; shaded rows are H0 reading and fixed mean fusion",
        "fusion_sensitivity": "all three panels use the original90 factual episodes. (a) mean ± model-seed SD; (b) small points are model seeds, diamonds their mean, lines pair within model, no SD errorbars; (c) mean ± model-seed SD and seed points, dashed10% reference. Neither sensitivity nor Q-margin is a reward guarantee."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summarize-only", action="store_true")
    opts = parser.parse_args()
    if opts.summarize_only:
        import had_env
        data = json.loads(gzip.decompress(DATA.read_bytes()))
        summarize(data)
        if data["execution_status"] != "complete":
            raise RuntimeError("Offline publication summaries require all270 registered episodes")
        old = json.loads(gzip.decompress(OLD.read_bytes()))
        data["runtime"]["had_module"] = str(Path(had_env.__file__).resolve())
        offline_analysis(data, old); plot_offline(data); save(data, DATA)
        print(json.dumps({"completed_episodes": data["summary"]["completed_episodes"],
            "old_factual_episodes": data["margin_analysis"]["factual_episodes"],
            "margin_integrity": data["margin_analysis"]["integrity"],
            "new_factual_episodes": data["read_diagnostics"]["factual_episodes"],
            "combined_intervention_groups": len(data["combined_interventions"]["across_seeds"]),
            "had_module": data["runtime"]["had_module"],
            "figures": data["figures"]}, indent=2)); return
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    np.random.seed(0); random.seed(0); torch.manual_seed(0)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    old = json.loads(gzip.decompress(OLD.read_bytes()))
    weights, per_cfg = fixed_weights(old)
    refs = {ek(e): e for e in old["episodes"] if e["method"] == "regir_nomem" and e["arm"] == "standard"}
    summary = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
    registered = {int(c["seed"]): c for c in summary["checkpoints"]
        if c.get("env") == "had" and c["method"] == "regir_nomem" and c["kind"] == "final"}
    ids = {seed: c["artifact_id"] for seed, c in registered.items()}
    if DATA.exists():
        data = json.loads(gzip.decompress(DATA.read_bytes()))
        if data["schema"] != SCHEMA or data["fixed_fusion_weights"] != weights:
            raise RuntimeError("Resume specification changed")
    else:
        data = {"schema": SCHEMA, "episodes": [], "fixed_fusion_weights": weights,
            "fixed_fusion_configuration_means": per_cfg, "training_updates": 0,
            "evaluation": {"configs": [list(c) for c in CONFIGS], "training_seeds": [0, 1, 2],
                "episode_seeds": list(SCENES), "arms": list(ARMS), "planned_closed_loop_episodes": 270,
                "fixed_fusion": "Within each model, average factual alpha over live decisions within configuration, then weight three configurations equally. Same existing diagnostic pool; no reward fitting or held-out claim.",
                "loop_bypass": "Read original H0 tokens, retain self query, learned read attention, projection including bias, local host, and downstream GRU; four contextual updates removed.",
                "same_state": "Reuse standard inputs and incoming hidden state; restore outgoing hidden/cached Q and RNG after two counterfactual forwards.",
                "max_steps": 100, "task_mode": "damage", "spatial_dim": 2, "target_initialization": "random",
                "blue_strategy": BLUE_STRATEGY, "policy_seed": "SeedSequence([training_seed,N_R,N_B,K,episode_seed])"},
            "runtime": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "device": "cuda",
                "gpu": torch.cuda.get_device_name(0), "threads": 1, "tf32": False,
                "had_core_version": CORE_VERSION, "had_physics_protocol": PHYSICS_PROTOCOL},
            "source": {"old_local": str(OLD.relative_to(REPO)), "formal_data_modified": False,
                "runner": "Papers/v2/analysis/mechanism_leaf1008.py"}, "checkpoints": [], "execution_status": "partial"}
    existing = {ek(e) for e in data["episodes"]}
    for seed in range(3):
        path = REPO / registered[seed]["archived_path"]
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["artifact_id"] != ids[seed]: raise RuntimeError("Registered checkpoint mismatch")
        ck = {"training_seed": seed, "checkpoint_id": ids[seed], "kind": "final", "path": str(path.relative_to(REPO))}
        if ck not in data["checkpoints"]: data["checkpoints"].append(ck)
        policy = load_policy("regir_nomem", path); policy.set_device("cuda")
        control = Control(policy, weights[str(seed)])
        name = f"leaf1008_mechanism_seed{seed}"
        register_end_to_end_policy("red", name, "frozen loop/fusion controls", lambda _, p=control: p)
        for cfg in CONFIGS:
            for scene in SCENES:
                for arm in ARMS:
                    key = (seed, cfg, scene, arm)
                    if key in existing: continue
                    control.arm = arm
                    torch.manual_seed(int(np.random.SeedSequence([seed, *cfg, scene]).generate_state(1)[0]))
                    started = time.perf_counter()
                    with torch.inference_mode():
                        result = run_episode(red=cfg[0], blue=cfg[1], targets=cfg[2], seed=scene,
                            red_strategy={"architecture": "end_to_end", "policy": name}, blue_strategy=BLUE_STRATEGY,
                            max_steps=100, task_mode="damage", spatial_dim=2, target_initialization="random",
                            diagnostics=True, record_events=True, retain_trajectory=False, record=False)
                    s = dict(result["episode_summary"])
                    ep = {"method": "regir_nomem", "training_seed": seed, "config": list(cfg), "episode_seed": scene,
                        "arm": arm, "checkpoint_id": ids[seed], "episode_summary": s,
                        "initial_state": control.initial, "events": result["events"],
                        "elapsed_seconds": time.perf_counter()-started}
                    if arm == "standard":
                        ref = refs[key]
                        ep["decisions"] = control.decisions
                        ep["old_local_replay_validation"] = replay_check(s, {**ref["episode_summary"],
                            "checkpoint_id": ref["checkpoint_id"], "recorded_at": old.get("updated_at")})
                    data["episodes"].append(ep); existing.add(key)
                    summarize(data); save(data, DATA)
                    print(json.dumps({"completed": len(existing), "seed": seed, "config": cfg, "scene": scene,
                        "arm": arm, "R": -s["D"], "T": s["ep_len"], "seconds": round(ep["elapsed_seconds"], 2),
                        "old_local_match": ep.get("old_local_replay_validation", {}).get("matched")}), flush=True)
        control.handle.remove()
    print(json.dumps(data["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
