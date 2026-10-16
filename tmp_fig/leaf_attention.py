"""Temporary LEAF visualization: five games, one random decision per game.

Run from the repository root with the configured torch310 Python. If had_env
is unavailable, pass --had-root pointing to a retained compatible source root.
The single data file preserves completed games; --replot never runs games.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(name, "1")
sys.path.insert(0, str(ROOT / "Open-SCORE"))


def save_data(path, data):
    pending = path.with_suffix(".json.pending")
    pending.write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                  allow_nan=False), encoding="utf-8")
    pending.replace(path)


class CapturePolicy:
    """Reservoir sample uniformly from decisions with live Red and Blue."""

    def __init__(self, policy, seed):
        import numpy as np
        self.policy = policy
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.policy.reset()
        self.sample = None
        self.eligible = 0

    def act(self, state, side, action_ids):
        import numpy as np
        import torch
        agent = self.policy.mac.agent
        branch = agent.global_net
        eligible = any(e.alive for e in state.red) and any(e.alive for e in state.blue)
        if eligible:
            self.eligible += 1
        capture = eligible and int(self.rng.integers(self.eligible)) == 0
        if capture:
            agent.enable_capture(max_decisions=1, max_observers=self.policy.args.n_agents)
        else:
            agent.disable_capture()
        branch.capture_attention = capture
        branch.read_attn.capture_attention = capture
        branch.self_attn.capture_attention = capture
        actions = self.policy.act(state, side, action_ids)
        if not capture:
            return actions
        if len(agent.last_capture) != 1:
            raise RuntimeError("Expected one capture from this real policy decision")
        record = agent.last_capture[0]
        weights = record["read_attn"].cpu().numpy()
        origins = record["origin"].numpy()
        # The model puts each observer first; undo that swap before labelling.
        inverse = np.argsort(origins, axis=-1)
        weights = np.take_along_axis(weights, inverse[:, None, :], axis=-1)
        mask = np.take_along_axis(record["key_mask"].numpy(), inverse, axis=-1)
        na = int(self.policy.args.n_agents)
        nb = int(self.policy.args.n_entities) - na - int(self.policy.args.n_tasks)
        slots = np.r_[np.arange(len(state.red)), na + np.arange(len(state.blue)),
                      na + nb + np.arange(len(state.targets))]
        weights, mask = weights[..., slots], mask[:, slots]
        alpha = record["alpha"].numpy()
        with torch.no_grad():
            reads = record["u"]
            if getattr(self.policy.args, "rer_score_normalize", False):
                reads = torch.nn.functional.normalize(reads, dim=-1) * reads.shape[-1] ** .5
            scores = branch.read_score(reads).squeeze(-1).cpu().numpy()
        # These checks validate only the requested sampled tensors, not a suite.
        if weights.shape[1] != 4 or alpha.shape[1] != 4:
            raise RuntimeError("The visualization requires the normal four-round LEAF")
        if not (np.isfinite(weights).all() and np.isfinite(alpha).all()
                and np.allclose(weights.sum(-1), 1, atol=2e-5)
                and np.allclose(alpha.sum(-1), 1, atol=2e-5)
                and np.allclose(torch.softmax(torch.as_tensor(scores), -1).numpy(),
                                alpha, atol=2e-5)
                and np.all(weights[np.broadcast_to(mask[:, None, :], weights.shape)] == 0)):
            raise RuntimeError("Captured attention/fusion normalization or mask is inconsistent")
        labels = ([f"R{i+1}" for i in range(len(state.red))]
                  + [f"B{i+1}" for i in range(len(state.blue))]
                  + [f"T{i+1}" for i in range(len(state.targets))])
        self.sample = dict(step=int(state.step), eligible_decisions=None,
                           observer_ids=record["observer_ids"].tolist(),
                           entity_slots=slots.tolist(), labels=labels,
                           attention=weights.tolist(), masked=mask.tolist(),
                           fusion=alpha.tolist(), fusion_logits=scores.tolist(),
                           alive_red=sum(e.alive for e in state.red),
                           alive_blue=sum(e.alive for e in state.blue))
        return actions


def plot_sample(sample, metadata, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm
    from matplotlib.patches import Rectangle
    import numpy as np

    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False,
                         "font.size": 10, "savefig.facecolor": "white"})
    attention, alpha = np.asarray(sample["attention"]), np.asarray(sample["fusion"])
    mask = np.asarray(sample["masked"], dtype=bool)
    observers = sample["observer_ids"]
    labels = sample["labels"]
    nrows, ncols = len(observers), len(labels)
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#eeeeee")
    # One common nonlinear colour scale reveals small weights. Printed numbers
    # are the original percentages; no per-panel rescaling or renormalization.
    norm = PowerNorm(gamma=.5, vmin=0, vmax=1)
    fig, axes = plt.subplots(2, 2, figsize=(21, 12.5))
    fig.subplots_adjust(left=.055, right=.94, bottom=.20, top=.83,
                        wspace=.16, hspace=.43)
    for depth, ax in enumerate(axes.flat):
        values = np.full((nrows, ncols + 3), np.nan)
        values[:, :ncols] = np.where(mask, np.nan, attention[:, depth])
        values[:, -2:] = alpha[:, depth, None]
        im = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(list(range(ncols)) + [ncols, ncols+1.5], labels + ["", "融合 α"])
        ax.set_yticks(range(nrows), [f"R{i+1}" for i in observers])
        ax.set_xlabel("被关注实体（R 防御者 / B 攻击者 / T 目标）", labelpad=9)
        ax.set_ylabel("接收者 / 防御者")
        ax.set_title(f"第 {depth+1} 层读取注意力  |  本层平均融合比例 {100*alpha[:, depth].mean():.3g}%",
                     fontsize=13, fontweight="bold", pad=14)
        ax.tick_params(axis="both", length=0, labelsize=9)
        for tick, label in zip(ax.get_xticklabels(), labels + ["", "融合 α"]):
            tick.set_color("#a43435" if label.startswith("R") else
                           "#245b9c" if label.startswith("B") else
                           "#946d14" if label.startswith("T") else "#222222")
        for row in range(nrows):
            for col in range(ncols + 2):
                value = values[row, col]
                if np.isfinite(value):
                    number = f"{100*value:.3g}" if col == ncols+1 else f"{100*value:.1f}"
                    ax.text(col+.5 if col == ncols+1 else col, row, number, ha="center", va="center",
                            fontsize=8.4 if col == ncols+1 else 7.7,
                            color="white" if norm(value) > .60 else "#152735",
                            fontweight="bold" if col == ncols+1 else "normal")
                elif col < ncols:
                    ax.text(col, row, "×", ha="center", va="center", color="#999999", fontsize=9)
            own = sample["entity_slots"].index(observers[row])
            ax.add_patch(Rectangle((own-.48, row-.48), .96, .96, fill=False,
                                   edgecolor="#dc6946", linewidth=1.5))
        # Team boundaries and a separate frame distinguish fusion from entity weights.
        for boundary in (metadata["red"]-.5, metadata["red"]+metadata["blue"]-.5):
            ax.axvline(boundary, color="white", lw=2)
        ax.add_patch(Rectangle((ncols+.5, -.5), 2, nrows, fill=False,
                               edgecolor="#343434", linewidth=1.2))
        for spine in ax.spines.values():
            spine.set_visible(False)
    color_ax = fig.add_axes([.956, .235, .012, .48])
    bar = fig.colorbar(im, cax=color_ax, ticks=[0, .01, .05, .1, .25, .5, 1])
    bar.ax.set_yticklabels(["0", "1", "5", "10", "25", "50", "100"])
    bar.set_label("权重（%，平方根色阶）", labelpad=12)
    fig.suptitle(f"LEAF 逐层注意力与融合比例 · 随机对抗 {sample['game']}/5",
                 fontsize=23, fontweight="bold", y=.965)
    fig.text(.5, .908,
             f"{metadata['red']} vs {metadata['blue']} · {metadata['targets']} 个目标 · "
             f"episode seed = {sample['episode_seed']} · 决策前 t = {sample['step']} "
             f"/ {sample['episode_length']} 步 · 存活 R/B = {sample['alive_red']}/{sample['alive_blue']}",
             ha="center", fontsize=13)
    fig.text(.5, .87,
             f"同一 LEAF final（训练 seed={metadata['model_seed']}），正常 R=4；"
             "每场均匀随机抽取 1 个双方仍有存活智能体的决策时刻", ha="center", fontsize=11)
    fig.text(.055, .083,
             "读图：每行是一个存活防御者；单元格为本层 reader 注意力（4 头平均，单位 %），实体列每行合计 100%。\n"
             "最右列 α 是该防御者在本层的融合比例；同一防御者的 4 个 α 合计 100%。橙框标出自身，× 表示死亡 / 被屏蔽实体。",
             fontsize=11, linespacing=1.8)
    fig.text(.055, .025,
             f"α = softmax(s1, s2, s3, s4)，s_r = read_score(u_r)  |  蓝方 reactive–rush  |  "
             f"运行时 {metadata['had_core_version']} / {metadata['physics_protocol']}",
             fontsize=9, color="#666666")
    path = output / f"leaf_attention_{sample['game']:02d}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Saved {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "Open-SCORE/outputs/main/had/regir_nomem/models/seed_0/final.pt")
    parser.add_argument("--had-root", type=Path, help="Root containing a compatible retained had_env package")
    parser.add_argument("--seed", type=int, default=20261016, help="Random sampling seed (not training seed)")
    parser.add_argument("--replot", action="store_true", help="Only redraw the saved samples, without environment execution")
    options = parser.parse_args()
    output = Path(__file__).resolve().parent
    data_path = output / "leaf_attention_data.json"
    data = json.loads(data_path.read_text(encoding="utf-8")) if data_path.exists() else None
    if options.replot:
        if data is None:
            parser.error("No saved samples available")
    else:
        if options.had_root:
            sys.path.insert(0, str(options.had_root.resolve()))
        import numpy as np
        import torch
        from had_env.core.version import CORE_VERSION, PHYSICS_PROTOCOL
        from open_score.algos import load_policy
        from open_score.rules import register_end_to_end_policy, run_episode
        from open_score.eval.anchors import BLUE_STRATEGY
        torch.set_num_threads(1)
        torch.manual_seed(options.seed)
        checkpoint = options.checkpoint.resolve()
        policy = load_policy("regir_nomem", checkpoint)
        policy.set_device("cpu")
        policy.set_eval_depth(4)
        cfg = policy.args
        if (cfg.method != "regir_nomem" or cfg.global_branch != "cycle"
                or getattr(cfg, "rer_readout", "learned") != "learned"
                or getattr(cfg, "read_last_round", False)
                or getattr(cfg, "leaf_loop_core", None)):
            raise RuntimeError("Expected the current LEAF learned-fusion checkpoint")
        seeds = np.random.default_rng(options.seed).choice(1_000_000, 5, replace=False) + 1_000_000
        metadata = dict(checkpoint=str(checkpoint), model_seed=int(cfg.seed), sample_seed=options.seed,
                        red=8, blue=8, targets=2, depth=4, max_steps=int(cfg.episode_limit),
                        had_root=str(options.had_root.resolve()) if options.had_root else None,
                        had_core_version=CORE_VERSION, physics_protocol=PHYSICS_PROTOCOL,
                        episode_seeds=seeds.tolist())
        if data is not None and data["metadata"] != metadata:
            raise RuntimeError("Existing samples belong to different settings; preserve them before changing this run")
        if data is None:
            data = dict(metadata=metadata, samples=[])
            save_data(data_path, data)
        for index in range(len(data["samples"]), 5):
            episode_seed = int(seeds[index])
            capture = CapturePolicy(policy, np.random.SeedSequence([options.seed, episode_seed]))
            name = f"tmp_leaf_attention_{index}"
            register_end_to_end_policy("red", name, "LEAF temporary attention", lambda _, p=capture: p)
            result = run_episode(red=8, blue=8, targets=2, seed=episode_seed,
                                 red_strategy={"architecture": "end_to_end", "policy": name},
                                 blue_strategy=BLUE_STRATEGY, max_steps=int(cfg.episode_limit),
                                 record=False, task_mode="damage", spatial_dim=2,
                                 target_initialization="random", diagnostics=True,
                                 record_events=False, retain_trajectory=False)
            if capture.sample is None:
                raise RuntimeError("This game had no decision with live Red and Blue")
            sample = dict(capture.sample, game=index+1, episode_seed=episode_seed,
                          episode_length=int(result["episode_summary"]["ep_len"]))
            sample["eligible_decisions"] = capture.eligible
            data["samples"].append(sample)
            save_data(data_path, data)
            print(f"Game {index+1}/5 complete: seed={episode_seed}, t={sample['step']}, "
                  f"eligible={capture.eligible}, fusion mean={np.asarray(sample['fusion']).mean(0).round(5)}", flush=True)
    for sample in data["samples"]:
        plot_sample(sample, data["metadata"], output)


if __name__ == "__main__":
    main()
