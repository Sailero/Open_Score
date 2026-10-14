"""Approved 720 frozen-final HAD episodes; no training or registry changes.

Run with torch310, --output outputs/main0928. Each batch atomically saves each
standard replay and every ten interventions, and lasts at most 30 minutes.
Every batch also saves its final progress. Standard replays are compared with the
formal CSV; explicitly retained discrepancies use local paired controls.
"""
from __future__ import annotations
import argparse
import base64
from collections import defaultdict
import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from open_score.algos import load_policy
from open_score.eval.anchors import BLUE_STRATEGY
from open_score.rules import register_end_to_end_policy, run_episode
import had_env
from had_env.core.version import CORE_VERSION, PHYSICS_PROTOCOL

CONFIGS = ((10, 10, 2), (50, 50, 2), (30, 30, 12))
SCENES = tuple(range(9000, 9010))
ARMS = ("standard", "L1", "L2", "read4", "uniform", "residual_zero", "gru_reset")
ALTERNATIVES = ARMS[1:]
SCHEMA = "leaf-final-mechanism1007-v2"
REPLAY_FIELDS = ("D", "ep_len", "terminated_naturally", "damage_by_target",
    "first_damage_step_by_target", "red_left", "blue_left", "red_deaths_by_cause",
    "blue_deaths_by_cause", "friendly_collisions", "action_hist")


def cfg_key(cfg):
    return tuple(int(cfg[k]) for k in ("N_R", "N_B", "K")) if isinstance(cfg, dict) else tuple(cfg)


def physical_state(state):
    def rows(items):
        return [{"id": int(e.id), "position": list(e.position), "velocity": list(e.velocity),
            "health": float(e.health), "alive": bool(e.alive),
            "cumulative_damage": float(e.cumulative_damage)} for e in items]
    return {"step": int(state.step), "red": rows(state.red), "blue": rows(state.blue), "targets": rows(state.targets)}


def packed(tensor):
    arr = tensor.detach().cpu().numpy().astype("<f4", copy=False)
    return {"shape": list(arr.shape), "dtype": "<f4", "base64": base64.b64encode(arr.tobytes()).decode("ascii")}


def rng_snapshot():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def rng_restore(value):
    random.setstate(value[0]); np.random.set_state(value[1]); torch.set_rng_state(value[2])
    if value[3]: torch.cuda.set_rng_state_all(value[3])


class FrozenIntervention:
    def __init__(self, policy):
        self.policy = policy
        self.branch = getattr(policy.mac.agent, "global_net", None)
        self.arm, self.diagnose, self.zero_residual, self.capture = "standard", False, False, False
        self.handles = []
        if self.branch is not None:
            self.handles = [self.branch.fuse.register_forward_hook(self._fuse),
                self.branch.read_score.register_forward_hook(self._score),
                policy.mac.agent._head.rnn.register_forward_pre_hook(self._gru)]

    def _fuse(self, module, inputs, output):
        if self.capture: self.residual = output.detach().clone()
        if self.zero_residual: return torch.zeros_like(output)  # zero both W_f u and b_f, retain weights

    def _score(self, module, inputs, output):
        if self.capture: self.scores.append(output.detach().clone())

    def _gru(self, module, inputs):
        if self.capture: self.fused_input = inputs[0].detach().clone()

    def configure(self, arm):
        if self.branch is not None:
            self.policy.set_eval_depth(int(arm[1]) if arm in ("L1", "L2") else 4)
            self.branch.readout_override = arm if arm in ("read1", "read4", "uniform") else None
            self.zero_residual = arm == "residual_zero"

    def reset(self):
        self.policy.reset(); self.configure(self.arm)
        self.decisions, self.initial = [], None

    @torch.no_grad()
    def act(self, state, side, action_ids):
        p, mac = self.policy, self.policy.mac
        if self.initial is None: self.initial = physical_state(state)
        if self.arm == "gru_reset": mac.hidden_states.zero_()  # keep last-action features and batch
        before = mac.hidden_states.clone()
        self.capture = self.diagnose
        self.scores, self.residual, self.fused_input = [], None, None
        actions = p.act(state, side, action_ids)
        self.capture = False
        if not self.diagnose: return actions
        after, cache = mac.hidden_states.clone(), mac._decision_q.clone()
        alive = [i for i, e in enumerate(state.red) if e.alive]
        if not alive: return actions
        q0 = cache.reshape(p.args.n_agents, -1)
        legal = p.batch["avail_actions"][:, int(state.step)].reshape(p.args.n_agents, -1).bool()
        a0 = q0.masked_fill(~legal, -torch.inf).argmax(-1)[alive]
        ids = [int(state.red[i].id) for i in alive]
        assert all(actions[e] == int(action_ids[int(a)]) for e, a in zip(ids, a0))
        ts = slice(int(state.step), int(state.step) + 1)
        inputs, _ = mac._build_inputs(p.batch, ts)
        fixed = {k: v.clone() for k, v in inputs.items() if torch.is_tensor(v)}
        main_rng = rng_snapshot()
        assert len(self.scores) == 4 and self.residual is not None
        alpha = torch.softmax(torch.cat(self.scores, dim=-1), dim=-1)[alive]
        rn = self.residual[alive].norm(dim=-1)
        ln = (self.fused_input - self.residual)[alive].norm(dim=-1)
        alternatives, q1, a1 = {}, None, None
        try:
            for arm in (*ALTERNATIVES, "read1"):
                mac.hidden_states = torch.zeros_like(before) if arm == "gru_reset" else before.clone()
                self.configure(arm); rng_restore(main_rng)
                original, _ = mac._build_inputs(p.batch, ts)
                assert all(torch.equal(original[k], v) for k, v in fixed.items() if k != "hidden_state")
                mac.forward(p.batch, t=int(state.step), test_mode=True)
                q = mac._decision_q.reshape(p.args.n_agents, -1)[alive].clone()
                aa = q.masked_fill(~legal[alive], -torch.inf).argmax(-1)
                delta = q - q0[alive]
                if arm == "L1": q1, a1 = q, aa
                if arm == "read1":
                    err = float((q - q1).abs().max().item())
                    if not torch.allclose(q, q1, atol=1e-6, rtol=1e-5) or not torch.equal(aa, a1):
                        raise RuntimeError(f"L1/read1 equivalence failed at step {state.step}: {err}")
                    equivalence = {"max_absolute_q_difference": err, "greedy_actions_identical": True}
                    continue
                alternatives[arm] = {"greedy_action": aa.cpu().tolist(),
                    "changed_defender_ids": [ids[i] for i in (aa != a0).nonzero().flatten().cpu().tolist()],
                    "q_delta": packed(delta), "squared_q_difference_sum": float(delta.square().sum()),
                    "absolute_q_difference_sum": float(delta.abs().sum())}
        finally:
            mac.hidden_states, mac._decision_q = after, cache
            self.configure("standard"); rng_restore(main_rng)
        self.decisions.append({"step": int(state.step), "active_defender_ids": ids, "n_active": len(alive),
            "h_prev": packed(before.reshape(p.args.n_agents, -1)[alive]), "q_standard": packed(q0[alive]),
            "greedy_standard": a0.cpu().tolist(), "native_actions": {str(k): int(v) for k, v in actions.items()},
            "alpha": packed(alpha), "residual_l2": packed(rn), "local_l2": packed(ln),
            "residual_to_local_l2": packed(rn / ln.clamp_min(1e-12)), "alpha_sum": alpha.sum(0).cpu().tolist(),
            "residual_l2_sum": float(rn.sum()), "residual_to_local_l2_sum": float((rn / ln.clamp_min(1e-12)).sum()),
            "L1_read1_equivalence": equivalence, "alternatives": alternatives})
        return actions


def diagnostic_summary(ds):
    n = sum(d["n_active"] for d in ds)
    result = {"active_defender_decisions": n, "decision_steps": len(ds), "alternatives": {}}
    for arm in ALTERNATIVES:
        changed = sum(len(d["alternatives"][arm]["changed_defender_ids"]) for d in ds)
        ss = sum(d["alternatives"][arm]["squared_q_difference_sum"] for d in ds)
        sa = sum(d["alternatives"][arm]["absolute_q_difference_sum"] for d in ds)
        result["alternatives"][arm] = {"active_defender_decisions": n, "changed_actions": changed,
            "action_change_fraction": changed/n if n else None, "q_elements": n*9,
            "q_squared_difference_sum": ss, "q_absolute_difference_sum": sa,
            "q_rmse": (ss/(n*9))**.5 if n else None, "q_mean_absolute_difference": sa/(n*9) if n else None}
    result["mean_alpha"] = (np.sum([d["alpha_sum"] for d in ds], axis=0)/n).tolist() if n else None
    result["mean_residual_l2"] = sum(d["residual_l2_sum"] for d in ds)/n if n else None
    result["mean_residual_to_local_l2"] = sum(d["residual_to_local_l2_sum"] for d in ds)/n if n else None
    result["L1_read1_equivalence"] = {"verified_decision_steps": len(ds), "greedy_actions_identical": True,
        "max_absolute_q_difference": max((d["L1_read1_equivalence"]["max_absolute_q_difference"] for d in ds), default=0.)}
    return result


def formal_records(output, ids):
    path = output/"had/episodes.csv"
    wanted = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for raw in csv.DictReader(f):
            method = json.loads(raw["method"])
            if method not in ("regir_nomem", "refil") or json.loads(raw["phase"]) != "final_eval": continue
            row = {k: json.loads(v) if v else None for k, v in raw.items()}
            key = (method, int(row["seed"]), cfg_key(row["config"]), int(row["episode_seed"]))
            if key[2] not in CONFIGS or key[3] not in SCENES or row["arm"] != "final": continue
            if row["checkpoint_id"] != ids[key[:2]]: raise RuntimeError(f"Formal checkpoint mismatch: {key}")
            if key in wanted and any(wanted[key][k] != row[k] for k in REPLAY_FIELDS):
                raise RuntimeError(f"Conflicting formal originals: {key}")
            wanted[key] = row
    if len(wanted) != 180: raise RuntimeError(f"Expected 180 formal reference rows, got {len(wanted)}")
    return path, wanted


def replay_check(summary, formal):
    different = {}
    for key in REPLAY_FIELDS:
        a, b = summary.get(key), formal.get(key)
        same = bool(np.allclose(a, b, atol=1e-7, rtol=0)) if key in ("D", "damage_by_target") else a == b
        if not same: different[key] = {"actual": a, "formal": b}
    return {"matched": not different, "checked_fields": list(REPLAY_FIELDS), "differences": different,
        "formal_checkpoint_id": formal["checkpoint_id"], "formal_recorded_at": formal["recorded_at"]}


def save(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".gz.pending")
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    with gzip.open(pending, "wt", encoding="utf-8", compresslevel=1) as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    # Windows readers can briefly hold the destination while inspecting a
    # completed snapshot. Preserve the pending episode and retry the rename.
    for attempt in range(50):
        try:
            pending.replace(path)
            break
        except PermissionError:
            if attempt == 49:
                raise
            time.sleep(.1)


def metadata(output, device, checkpoints, formal_path):
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, text=True).strip()
        remote = subprocess.check_output(["git", "rev-parse", "origin/remote_train"], cwd=ROOT.parent, text=True).strip()
    except subprocess.SubprocessError: commit, remote = None, None
    return {"schema": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(), "checkpoint_kind": "final",
        "training_updates": 0, "formal_final_reproduction": "checked per factual episode", "output_version": str(output),
        "formal_csv": str(formal_path), "checkpoints": checkpoints,
        "evaluation": {"configs": [list(c) for c in CONFIGS], "training_seeds": [0,1,2], "episode_seeds": list(SCENES),
            "max_steps": 100, "task_mode": "damage", "spatial_dim": 2, "target_initialization": "random",
            "blue_strategy": BLUE_STRATEGY, "reported_reward": "R=-D", "LEAF_arms": list(ARMS), "REFIL_arms": ["standard"],
            "planned_closed_loop_episodes": 720,
            "policy_seed": "np.random.SeedSequence([training_seed,N_R,N_B,K,episode_seed]).generate_state(1)[0]; torch.manual_seed",
            "same_state_semantics": "Factual L4 entity inputs and previous actions, factual incoming GRU state except explicit reset; restore hidden/cached Q/RNG after diagnostics; interventions do not affect factual actions",
            "residual_zero_semantics": "fuse forward hook zeros W_f u+b_f output; weights unchanged, computation retained",
            "gru_reset_semantics": "Zero mac.hidden_states per physical decision; retain previous-action features and episode batch",
            "read1_reuse": "L4-read1 Q/action equivalence to L1 checked at every factual live decision, no redundant closed-loop arm",
            "denominator": "alive defender decisions on factual L4 trajectories, 9 acceleration actions",
            "raw_float_arrays": "base64 little-endian float32 C-order with shape/dtype",
            "trajectories": "all90 factual LEAF episodes plus2 preselected REFIL seed0 scene9000 10K2/50K2 episodes",
            "case_rule": "trainingseed0 scene9000 10K2/50K2; shared snapshot at REFIL first target damage, else first interception; explicitly label early terminal"},
        "local_source": {"git_commit_reference": commit, "remote_train_reference": remote, "working_copy_used": True,
            "archived_remote_execution_reproduction_claimed": False, "runner": str(Path(__file__).resolve()),
            "load_policy": str(ROOT/"open_score/algos/__init__.py"), "entity_encoder": str(ROOT/"open_score/models/entity_encoder.py"),
            "had_module": str(Path(had_env.__file__).resolve()), "had_core_version": CORE_VERSION, "had_physics_protocol": PHYSICS_PROTOCOL},
        "runtime": {"python": sys.version, "python_executable": sys.executable, "torch": torch.__version__, "numpy": np.__version__,
            "torch_cuda": torch.version.cuda, "platform": platform.platform(), "device": device,
            "gpu_model": torch.cuda.get_device_name(0) if device == "cuda" else None, "torch_cpu_threads": torch.get_num_threads(),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32},
        "episodes": [], "execution_status": "partial"}


def episode_key(ep):
    return (ep["method"], ep["training_seed"], tuple(ep["config"]), ep["episode_seed"], ep["arm"])


def summarize(data):
    es = data["episodes"]
    factual = {(e["training_seed"], tuple(e["config"]), e["episode_seed"]):e for e in es if e["method"]=="regir_nomem" and e["arm"]=="standard"}
    groups = defaultdict(list)
    for e in es: groups[(e["method"],e["training_seed"],tuple(e["config"]),e["arm"])].append(e)
    per_seed = []
    for (method,seed,cfg,arm), episodes in groups.items():
        delta = [-e["episode_summary"]["D"]+factual[(seed,cfg,e["episode_seed"])]["episode_summary"]["D"]
            for e in episodes if (seed,cfg,e["episode_seed"]) in factual]
        r = {"method":method,"training_seed":seed,"config":list(cfg),"arm":arm,"n_episodes":len(episodes),
            "mean_R":float(np.mean([-e["episode_summary"]["D"] for e in episodes])),"paired_n":len(delta),
            "mean_delta_R_vs_LEAF_standard":float(np.mean(delta)) if delta else None,
            "mean_friendly_collisions":float(np.mean([e["episode_summary"]["friendly_collisions"] for e in episodes])),
            "mean_collision_fraction":float(np.mean([e["episode_summary"]["red_deaths_by_cause"]["friendly_collision"]/cfg[0] for e in episodes])),
            "mean_intercepted_fraction":float(np.mean([e["episode_summary"]["blue_deaths_by_cause"]["intercepted"]/cfg[1] for e in episodes]))}
        if method=="regir_nomem" and arm=="standard": r["same_state"]=diagnostic_summary([d for e in episodes for d in e["decisions"]])
        per_seed.append(r)
    across = defaultdict(list)
    for r in per_seed: across[(r["method"],tuple(r["config"]),r["arm"])].append(r)
    agg = []
    for (method,cfg,arm), rows in across.items():
        r={"method":method,"config":list(cfg),"arm":arm,"n_training_seeds":len(rows),"n_episodes":sum(v["n_episodes"] for v in rows)}
        for field in ("mean_R","mean_delta_R_vs_LEAF_standard","mean_friendly_collisions","mean_collision_fraction","mean_intercepted_fraction"):
            values=[v[field] for v in rows if v[field] is not None]
            r[field]=float(np.mean(values)) if values else None
            r[field+"_sample_SD"]=float(np.std(values,ddof=1)) if len(values)>1 else None
        if method=="regir_nomem" and arm=="standard":
            diag={"active_defender_decisions":sum(v["same_state"]["active_defender_decisions"] for v in rows),"alternatives":{},
                "L1_read1_max_absolute_q_difference":max(v["same_state"]["L1_read1_equivalence"]["max_absolute_q_difference"] for v in rows)}
            for a in ALTERNATIVES:
                ds=[v["same_state"]["alternatives"][a] for v in rows]
                vals=[v["action_change_fraction"] for v in ds]
                diag["alternatives"][a]={"changed_actions":sum(v["changed_actions"] for v in ds),
                    "active_defender_decisions":sum(v["active_defender_decisions"] for v in ds),
                    "mean_action_change_fraction":float(np.mean(vals)),"action_change_fraction_sample_SD":float(np.std(vals,ddof=1)) if len(vals)>1 else None,
                    "mean_q_rmse":float(np.mean([v["q_rmse"] for v in ds]))}
            diag["mean_alpha"]=np.mean([v["same_state"]["mean_alpha"] for v in rows],axis=0).tolist()
            diag["alpha_sample_SD"]=np.std([v["same_state"]["mean_alpha"] for v in rows],axis=0,ddof=1).tolist() if len(rows)>1 else None
            r["same_state"]=diag
        agg.append(r)
    data["summary"]={"per_seed":per_seed,"across_seeds":agg,
        "SD_semantics":"sample SD across training-seed means; paired scene differences computed within seed",
        "completed_closed_loop_episodes":len(es),"training_episodes":0,"factual_replays_checked":sum(e["arm"]=="standard" for e in es),
        "factual_replays_matched":sum(e.get("replay_validation",{}).get("matched",False) for e in es)}
    data["execution_status"]="complete" if len(es)==720 else "partial"


def case_snapshot(ep):
    damaged=[t for t in ep["episode_summary"]["first_damage_step_by_target"] if t is not None]
    if damaged: return min(damaged),"First target damage (REFIL)"
    before={e["id"]:e["alive"] for e in ep["initial_state"]["blue"]}
    for f in ep["trajectory"]:
        for e in f["blue"]:
            if before[e["id"]] and not e["alive"]:
                hit=[v for v in ep["events"] if v.get("step")==f["step"] and v.get("target_id")==e["id"]]
                excluded=any(v.get("kind")=="collision" or (v.get("kind")=="self_destruct" and v.get("health_before",0)>0) for v in hit)
                if not excluded: return f["step"],"First REFIL interception"
            before[e["id"]]=e["alive"]
    return ep["episode_summary"]["ep_len"],"REFIL end: no earlier event"


def plot(data,figures):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":8.5,"axes.labelsize":8.5,"axes.titlesize":8.5,"xtick.labelsize":8,
        "ytick.labelsize":8,"legend.fontsize":8,"pdf.fonttype":42,"ps.fonttype":42,"font.family":"DejaVu Sans"})
    figures.mkdir(parents=True,exist_ok=True)
    colors={"LEAF":"#17669b","REFIL":"#cf5f2a"}
    fig,ax=plt.subplots(2,3,figsize=(7.16,5.0),constrained_layout=True)
    fig.suptitle("Spatial panels: blue defenders; orange attackers; stars mark targets",fontsize=8)
    cases=[]
    for row,cfg in enumerate(CONFIGS[:2]):
        eps=[next(e for e in data["episodes"] if e["method"]==m and e["training_seed"]==0 and tuple(e["config"])==cfg and e["episode_seed"]==9000 and e["arm"]=="standard") for m in ("regir_nomem","refil")]
        cut,reason=case_snapshot(eps[1]); points=[]
        cases.append({"config":list(cfg),"training_seed":0,"episode_seed":9000,"shared_snapshot_step":cut,"snapshot_rule":reason,
            "LEAF_end_step":eps[0]["episode_summary"]["ep_len"],"REFIL_end_step":eps[1]["episode_summary"]["ep_len"]})
        for col,(ep,label) in enumerate(((eps[1],"REFIL"),(eps[0],"LEAF"))):
            frames=[ep["initial_state"],*[f for f in ep["trajectory"] if f["step"]<=cut]]
            for side,color,prefix in (("red","#17669b","Defenders"),("blue","#cf5f2a","Attackers")):
                for i in range(len(frames[0][side])):
                    xy=np.asarray([f[side][i]["position"][:2] for f in frames])/1000; points.extend(xy.tolist())
                    ax[row,col].plot(xy[:,0],xy[:,1],color=color,linewidth=.55,alpha=.45 if row else .65,label=prefix if i==0 else None)
                    ax[row,col].scatter(*xy[0],s=9,facecolors="none",edgecolors=color,linewidths=.5,alpha=.8)
                    ax[row,col].scatter(*xy[-1],s=10,color=color,marker="o" if frames[-1][side][i]["alive"] else "x",linewidths=.55)
            for i,target in enumerate(ep["initial_state"]["targets"]):
                xy=np.asarray(target["position"][:2])/1000; points.append(xy.tolist())
                ax[row,col].scatter(*xy,s=36,marker="*",color="#252525",zorder=5)
                ax[row,col].annotate(f"T{i+1}",xy,xytext=(3,5) if i==0 else (-5,5),ha="left" if i==0 else "right",textcoords="offset points",fontsize=8)
            end=ep["episode_summary"]["ep_len"]
            qualifier=f"early terminal {end} < {cut}" if end<cut else f"step {cut}"
            ax[row,col].set_title(f"{label}, {cfg[0]}v{cfg[1]}, K{cfg[2]}\n{qualifier}")
            ax[row,col].set_xlabel("x (km)"); ax[row,col].set_ylabel("y (km)"); ax[row,col].set_aspect("equal",adjustable="box")
        xy=np.asarray(points)
        for col in (0,1):
            ax[row,col].set_xlim(xy[:,0].min()-.5,xy[:,0].max()+.5); ax[row,col].set_ylim(xy[:,1].min()-.5,xy[:,1].max()+.5)
        for ep,label in zip(eps,("LEAF","REFIL")):
            t=[0]+[f["step"] for f in ep["trajectory"]]
            r=[0.]+[-sum(v["cumulative_damage"] for v in f["targets"]) for f in ep["trajectory"]]
            ax[row,2].step(t,r,where="post",color=colors[label],label=label,linewidth=1.25)
            ax[row,2].scatter(t[-1],r[-1],s=14,color=colors[label],zorder=4)
            ax[row,2].annotate(f"end {t[-1]}",(t[-1],r[-1]),xytext=(3,-12 if label=="LEAF" else 8),textcoords="offset points",fontsize=8,color=colors[label])
        ax[row,2].axvline(cut,color="#666666",linestyle="--",linewidth=.8)
        ax[row,2].set_title(f"Shared snapshot at step {cut}\n{reason}")
        ax[row,2].set_xlabel("Physical step (1 s)"); ax[row,2].set_ylabel("Cumulative R = −D")
        ax[row,2].set_xlim(0,100); ax[row,2].grid(alpha=.2); ax[row,2].legend(frameon=False)
    for ext in ("pdf","png"): fig.savefig(figures/f"fig_leaf_final1007_trajectory.{ext}",dpi=220)
    plt.close(fig); data["selected_cases"]=cases
    labels=("L1","L2","Read4","Uniform","No\nresidual","GRU\nreset")
    fig,ax=plt.subplots(2,3,figsize=(7.16,4.3),constrained_layout=True)
    for col,cfg in enumerate(CONFIGS):
        for j,arm in enumerate(ALTERNATIVES):
            rows=[r for r in data["summary"]["per_seed"] if r["method"]=="regir_nomem" and tuple(r["config"])==cfg and r["arm"]==arm]
            vals=[r["mean_delta_R_vs_LEAF_standard"] for r in rows]
            ax[0,col].errorbar(j,np.mean(vals),yerr=np.std(vals,ddof=1),fmt="o",color="#17669b",capsize=2,markersize=4)
            ax[0,col].scatter(j+np.linspace(-.12,.12,3),vals,color="#333333",s=8,alpha=.65)
            facts=[r for r in data["summary"]["per_seed"] if r["method"]=="regir_nomem" and tuple(r["config"])==cfg and r["arm"]=="standard"]
            vals=[100*r["same_state"]["alternatives"][arm]["action_change_fraction"] for r in facts]
            ax[1,col].errorbar(j,np.mean(vals),yerr=np.std(vals,ddof=1),fmt="o",color="#cf5f2a",capsize=2,markersize=4)
            ax[1,col].scatter(j+np.linspace(-.12,.12,3),vals,color="#333333",s=8,alpha=.65)
        ax[0,col].set_title(f"{cfg[0]}v{cfg[1]}, K{cfg[2]}"); ax[0,col].axhline(0,color="#777777",linewidth=.7,linestyle="--")
        ax[0,col].set_ylabel("Paired ΔR vs. standard"); ax[1,col].set_ylabel("Same-state action changes (%)")
        for row in (0,1):
            ax[row,col].set_xticks(range(6),labels,rotation=40,ha="right")
            ax[row,col].grid(axis="y",alpha=.2)
        ax[1,col].set_ylim(bottom=0)
    for ext in ("pdf","png"): fig.savefig(figures/f"fig_leaf_final1007_mechanism.{ext}",dpi=220)
    plt.close(fig)
    data["figures"]={"trajectory":"fig_leaf_final1007_trajectory.pdf","mechanism":"fig_leaf_final1007_mechanism.pdf","width_inches":7.16,"smallest_font_pt":8}


def debug_replay(output, path):
    """Only the actual failed cell: fresh CUDA plain/diagnosed, then CPU plain."""
    with gzip.open(path,"rt",encoding="utf-8") as f: data=json.load(f)
    failed=next(e for e in reversed(data["episodes"]) if not e.get("replay_validation",{}).get("matched",True))
    method,seed,cfg,scene=failed["method"],failed["training_seed"],tuple(failed["config"]),failed["episode_seed"]
    ids={(c["method"],c["training_seed"]):c["checkpoint_id"] for c in data["checkpoints"]}
    _,refs=formal_records(output,ids)
    outcomes=[]
    for device,diagnose in (("cuda",False),("cuda",True),("cpu",False)):
        if device=="cpu" and any(e["episode_summary"]["D"]!=failed["episode_summary"]["D"] for e in outcomes): break
        checkpoint=output/"had"/method/"train"/f"seed_{seed}"/"final.pt"
        p=load_policy(method,checkpoint); p.set_device(device); p.set_eval_depth(4)
        initial=[]; decisions=[]
        if diagnose:
            w=FrozenIntervention(p); w.diagnose=True
        else: w=p
        def record(step,state,actions,reward,info):
            if not initial: initial.append(physical_state(state))
            alive=[i for i,e in enumerate(state.red) if e.alive]
            decisions.append({"step":state.step,"native_actions":{str(k):int(v) for k,v in actions.items()},
                "active_defender_ids":[int(state.red[i].id) for i in alive],
                "q_standard":packed(p.mac._decision_q.reshape(p.args.n_agents,-1)[alive]),
                "h_after":packed(p.mac.hidden_states.reshape(p.args.n_agents,-1)[alive])})
        name=f"leaf_debug_{device}_{int(diagnose)}"
        register_end_to_end_policy("red",name,"actual replay mismatch debugging",lambda _,obj=w:obj)
        torch.manual_seed(int(np.random.SeedSequence([seed,*cfg,scene]).generate_state(1)[0])); t0=time.perf_counter()
        with torch.inference_mode():
            r=run_episode(red=cfg[0],blue=cfg[1],targets=cfg[2],seed=scene,
                red_strategy={"architecture":"end_to_end","policy":name},blue_strategy=BLUE_STRATEGY,
                max_steps=100,task_mode="damage",spatial_dim=2,target_initialization="random",diagnostics=True,
                record_events=True,retain_trajectory=True,record=False,on_step=record)
        ep={"method":method,"training_seed":seed,"config":list(cfg),"episode_seed":scene,
            "attempt_kind":"actual_replay_mismatch_debug","device":device,"with_same_state_diagnostics":diagnose,
            "checkpoint_id":ids[(method,seed)],"elapsed_seconds":time.perf_counter()-t0,
            "episode_summary":r["episode_summary"],"replay_validation":replay_check(r["episode_summary"],refs[(method,seed,cfg,scene)]),
            "initial_state":initial[0],"trajectory":r["trajectory"],"decisions":decisions,"events":r["events"]}
        data.setdefault("debug_attempts",[]).append(ep); save(data,path); outcomes.append(ep)
        print(json.dumps({"DEBUG_DEVICE":device,"DIAGNOSE":diagnose,"R":-r["episode_summary"]["D"],
            "end":r["episode_summary"]["ep_len"],"matched":ep["replay_validation"]["matched"]}),flush=True)
    print("DEBUG_ATTEMPTS",len(outcomes),"scientific_cells",len(data["episodes"]),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/main0928")
    parser.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    parser.add_argument("--max-new-episodes",type=int,default=720)
    parser.add_argument("--batch-minutes",type=float,default=28)
    parser.add_argument("--factual-only",action="store_true",help="Standard replays only, included in720 budget")
    parser.add_argument("--plot-only",action="store_true")
    parser.add_argument("--debug-replay",action="store_true",help="Minimal direct debugging of the already saved replay mismatch")
    parser.add_argument("--record-replay-differences",action="store_true",
        help="Retain historical replay discrepancies and use current local standard episodes as paired controls")
    opts=parser.parse_args()
    if not 0<opts.batch_minutes<=30: raise ValueError("Batch must be <=30 minutes")
    output=opts.output.resolve(); path=output/"had/regir_nomem/auxiliary1007/results.json.gz"; figures=output/"had/figures"
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    if opts.device=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    if opts.debug_replay:
        debug_replay(output,path); return
    if opts.plot_only:
        with gzip.open(path,"rt",encoding="utf-8") as f: data=json.load(f)
        if data["execution_status"]!="complete": raise RuntimeError("Plots require completed approved batch")
        summarize(data); plot(data,figures); save(data,path); print("PLOTS",figures,flush=True); return
    inventory=json.loads((output/"inventory.json").read_text(encoding="utf-8"))
    ids={(c["method"],int(c["seed"])):c["checkpoint_id"] for c in inventory["checkpoints"] if c.get("env")=="had" and c["method"] in ("regir_nomem","refil")}
    checkpoints=[]; policies={}
    for method in ("regir_nomem","refil"):
        for seed in range(3):
            checkpoint=output/"had"/method/"train"/f"seed_{seed}"/"final.pt"
            saved=torch.load(checkpoint,map_location="cpu",weights_only=False)
            if saved["artifact_id"]!=ids[(method,seed)]: raise RuntimeError(f"Registered final mismatch: {method}/{seed}")
            checkpoints.append({"method":method,"training_seed":seed,"checkpoint_kind":"final","checkpoint_path":str(checkpoint),
                "checkpoint_id":saved["artifact_id"],"progress":{k:v for k,v in saved["progress"].items() if k!="grad_norm_history"},"config":saved["config"],"strict_agent_and_mixer_load":True})
            p=load_policy(method,checkpoint); p.set_device(opts.device); policies[(method,seed)]=FrozenIntervention(p)
            register_end_to_end_policy("red",f"leaf_final_aux1007_{method}_s{seed}","registered frozen final auxiliary",lambda _,w=policies[(method,seed)]:w)
    formal_path,references=formal_records(output,ids)
    if path.exists():
        with gzip.open(path,"rt",encoding="utf-8") as f: data=json.load(f)
        if data["schema"]!=SCHEMA or data["runtime"]["device"]!=opts.device: raise RuntimeError("Resume schema/device mismatch")
        if [(c["method"],c["training_seed"],c["checkpoint_id"]) for c in data["checkpoints"]]!=[(c["method"],c["training_seed"],c["checkpoint_id"]) for c in checkpoints]: raise RuntimeError("Resume checkpoint set changed")
    else: data=metadata(output,opts.device,checkpoints,formal_path)
    if opts.record_replay_differences:
        data["analysis_scope"] = {
            "kind": "current-local frozen-final auxiliary reevaluation",
            "paired_control": "same local standard execution on the same scene and model seed",
            "formal_csv_modified": False,
            "historical_replay_discrepancies": "retained per episode; historical cause unresolved",
            "decision": "Continue the approved 720 unique cells after direct CUDA plain/diagnostic and CPU plain checks excluded diagnostic contamination in the first divergent scene.",
        }
    elif any(not e.get("replay_validation",{}).get("matched",True) for e in data["episodes"]):
        raise RuntimeError("Resolve previous replay mismatch or explicitly record differences before interventions")
    existing={episode_key(e) for e in data["episodes"]}
    jobs=[(m,s,c,scene,"standard") for s in range(3) for c in CONFIGS for scene in SCENES for m in ("regir_nomem","refil")]
    if not opts.factual_only: jobs += [("regir_nomem",s,c,scene,a) for s in range(3) for c in CONFIGS for a in ALTERNATIVES for scene in SCENES]
    first=[(m,0,c,9000,"standard") for c in CONFIGS[:2] for m in ("regir_nomem","refil")]
    jobs=first+[j for j in jobs if j not in first]; started=time.monotonic(); added=0
    for method,seed,cfg,scene,arm in jobs:
        key=(method,seed,cfg,scene,arm)
        if key in existing: continue
        if added>=opts.max_new_episodes or time.monotonic()-started>=opts.batch_minutes*60: break
        w=policies[(method,seed)]; w.arm=arm; w.diagnose=method=="regir_nomem" and arm=="standard"
        keep=w.diagnose or (method=="refil" and seed==0 and scene==9000 and cfg in CONFIGS[:2])
        torch.manual_seed(int(np.random.SeedSequence([seed,*cfg,scene]).generate_state(1)[0])); t0=time.perf_counter()
        with torch.inference_mode():
            result=run_episode(red=cfg[0],blue=cfg[1],targets=cfg[2],seed=scene,
                red_strategy={"architecture":"end_to_end","policy":f"leaf_final_aux1007_{method}_s{seed}"},
                blue_strategy=BLUE_STRATEGY,max_steps=100,task_mode="damage",spatial_dim=2,target_initialization="random",
                diagnostics=True,record_events=True,retain_trajectory=keep,record=False)
        summary=dict(result["episode_summary"]); elapsed=time.perf_counter()-t0
        ep={"method":method,"training_seed":seed,"config":list(cfg),"episode_seed":scene,"arm":arm,
            "checkpoint_kind":"final","checkpoint_id":ids[(method,seed)],"elapsed_seconds":elapsed,"episode_summary":summary,
            "terminated":result["terminated"],"truncated":result["truncated"],"events":result["events"]}
        if arm=="standard": ep["replay_validation"]=replay_check(summary,references[(method,seed,cfg,scene)])
        if keep:
            ep["initial_state"]=w.initial; ep["trajectory"]=result["trajectory"]; ep["bounds"]=result["bounds"]
            for f in ep["trajectory"]:
                f["physical_reward"]=-float(f["step_damage"]); f["target_step_rewards"]=[-float(t["step_damage"]) for t in f["targets"]]
            if abs(sum(f["physical_reward"] for f in ep["trajectory"])+summary["D"])>1e-7: raise RuntimeError("Trajectory reward != native -D")
        if w.diagnose: ep["decisions"]=w.decisions; ep["same_state_summary"]=diagnostic_summary(w.decisions)
        data["episodes"].append(ep); existing.add(key); added+=1
        # Factual probes dominate the saved payload. Retain every standard
        # immediately, then checkpoint short intervention groups without
        # repeatedly rewriting all factual arrays after every single rollout.
        if arm=="standard" or added%10==0:
            summarize(data); save(data,path)
        print(json.dumps({"completed":len(data["episodes"]),"method":method,"seed":seed,"config":cfg,"scene":scene,"arm":arm,
            "seconds":round(elapsed,3),"R":-summary["D"],"steps":summary["ep_len"],"replay_matched":ep.get("replay_validation",{}).get("matched")}),flush=True)
        if arm=="standard" and not ep["replay_validation"]["matched"]:
            print(json.dumps(ep["replay_validation"],ensure_ascii=False),flush=True)
            if not opts.record_replay_differences:
                raise RuntimeError("Factual replay mismatch: interventions stopped")
    summarize(data)
    if data["execution_status"]=="complete": plot(data,figures)
    save(data,path)
    print("DATA",path,"STATUS",data["execution_status"],"BATCH_NEW",added,flush=True)


if __name__=="__main__": main()
