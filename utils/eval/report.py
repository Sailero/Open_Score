"""Render the selected HAD figures from saved statistics.

No policy, simulator, training, evaluation or scientific-statistics refresh is
imported or executed. Optional local trajectory data are read without modification.
"""
from __future__ import annotations
import gzip
import json
from pathlib import Path
import threading
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

FIGURE_NAMES = ("fig_comparison", "fig_mechanism", "fig_protection", "fig_trajectory", "fig_decision_timing", "coverage_binding_probe")
TEAL, ORANGE, PURPLE = "#007F7A", "#D68B3C", "#8270A8"
BLUE, RED, INK, GRAY, LIGHT = "#507AA4", "#B9636E", "#22343D", "#77838B", "#E5EAED"
METHODS = ("regir_nomem", "refil", "refil_matched", "b2_qmix_atten", "dcg", "spectra", "alma", "transfqmix", "rule_nv1", "random_accel")
ANCHORS = ("rule_nv1", "random_accel")
NAMES = dict(zip(METHODS, ("LEAF", "REFIL", "REFIL-matched", "QMIX", "DCG", "SPECTra", "ALMA", "TransfQMix", "Rule", "Random")))
COLORS = dict(zip(METHODS, (TEAL, ORANGE, "#A46A2F", "#BB7A9B", RED, "#87933D", BLUE, PURPLE, GRAY, "#3E4D55")))
MARKERS = dict(zip(METHODS, ("o", "s", "X", "h", "v", "P", "^", "D", "x", "+")))
LINESTYLES = dict(zip(METHODS, ("-", "-", "--", "-", "-.", ":", "-", "--", "--", ":")))
TRAJECTORY_METHODS = ("regir_nomem", "refil", "b2_qmix_atten", "dcg", "alma", "rule_nv1")
TRAJECTORY_CONFIGS = ((5, 5, 2), (15, 15, 2))
TRAJECTORY_LABELS = {m:NAMES[m] for m in TRAJECTORY_METHODS}
SNAPSHOT = 30
_RENDER_LOCK = threading.RLock()


def read_json(path):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(fig, basename):
    paths=[]
    for ext in ("pdf", "png"):
        path=FIGURES / f"{basename}.{ext}"
        fig.savefig(path, facecolor="white", dpi=260)
        paths.append(path)
    plt.close(fig)
    return paths


def damage(row):
    if isinstance(row.get("damage"), dict):
        return row["damage"]["mean"], row["damage"]["sample_sd"]
    return -row["reward"]["mean"], row["reward"]["sample_sd"]

def style(ax, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#A9B3B9")
    ax.tick_params(length=2.4, color="#A9B3B9", pad=2)
    ax.set_axisbelow(True)
    ax.grid(axis=grid, color=LIGHT, lw=.55)

def title(ax, letter, text):
    ax.set_title(f"({letter})  {text}", loc="left", pad=7, fontweight="bold", color=INK)

def comparison(p, methods=None, name="fig_comparison", confirm=True):
    methods = METHODS if methods is None else methods
    config = {(r["method"], tuple(r["config"])): r for r in p["config_summary"] + p.get("rule_config", []) + p.get("random_config", []) if r["method"] in methods}
    fig = plt.figure(figsize=(8.4, 7.6 if confirm else 5.8))
    gs = fig.add_gridspec(3 if confirm else 2, 2, left=.075, right=.945, top=.95, bottom=.16, hspace=.65, wspace=.50, height_ratios=(1, 1.35, 1.02) if confirm else (1, 1.35))
    ax = fig.add_subplot(gs[0, 0])
    for method in methods[::-1]:
        if method in ANCHORS:
            row = p["rule_ID_reference"] if method == "rule_nv1" else next(r for r in p["random_group_summary"] if r["group"] == "ID")
            ax.axhline(-row["reward"]["mean"], color=COLORS[method], ls=LINESTYLES[method], lw=.85)
            continue
        rows = sorted((r for r in p["validation_curves"] if r["method"] == method), key=lambda r: r["eval_point"])
        x = np.array([r["t_env_mean"] for r in rows]) / 1e6
        y = np.array([-r["reward"]["mean"] for r in rows])
        sd = np.array([r["reward"]["sample_sd"] for r in rows])
        ax.fill_between(x, y-sd, y+sd, color=COLORS[method], alpha=.055, lw=0)
        ax.plot(x, y, color=COLORS[method], ls=LINESTYLES[method], marker=MARKERS[method], markevery=7, ms=2.2, lw=1.55 if method == "regir_nomem" else .85, zorder=6 if method == "regir_nomem" else 3)
    title(ax, "a", "Learning on ID validation")
    ax.set(xlim=(0, 1.01), ylim=(0, 8.1), xticks=(0, .25, .5, .75, 1), xlabel="Environment steps (millions)", ylabel=r"Target damage $D$ $\downarrow$")
    style(ax)

    def curve(ax, configs, xs):
        for method in methods[::-1]:
            rows = [config[method, cfg]["reward"] for cfg in configs]
            y = np.array([-r["mean"] for r in rows])
            sd = np.array([r["sample_sd"] for r in rows])
            if method not in ANCHORS:
                ax.fill_between(xs, y-sd, y+sd, color=COLORS[method], alpha=.06, lw=0)
            ax.plot(xs, y, marker=MARKERS[method], ms=2.8, color=COLORS[method], ls=LINESTYLES[method], lw=1.6 if method == "regir_nomem" else .9, zorder=6 if method == "regir_nomem" else 3)
        ax.set_ylim(bottom=0)
        style(ax)

    ax = fig.add_subplot(gs[0, 1])
    ks = (4, 6, 9, 12)
    curve(ax, [(10, 10, k) for k in ks], ks)
    title(ax, "b", "Target-count extrapolation")
    ax.set(xticks=ks, xlabel=r"Targets $K$ (10 vehicles per team)", ylabel=r"Target damage $D$ $\downarrow$")
    ax.text(.025, .96, r"Training: $K\leq 3$", va="top", transform=ax.transAxes, fontsize=6.5, color=GRAY)

    ax = fig.add_subplot(gs[1, 0])
    ns = (15, 20, 25, 30, 40, 50)
    curve(ax, [(n, n, 2) for n in ns], ns)
    title(ax, "c", "Team-size extrapolation")
    ax.set(xticks=(15, 25, 40, 50), xlabel=r"Vehicles per team $N$ ($K=2$)", ylabel=r"Target damage $D$ $\downarrow$")
    ax.text(.025, .96, r"Training: $N\leq 10$", va="top", transform=ax.transAxes, fontsize=6.5, color=GRAY)

    ax = fig.add_subplot(gs[1, 1])
    joint = [tuple(c) for c in p["groups"]["Joint-OOD"]]
    values = np.array([[-config[m, c]["reward"]["mean"] for c in joint] for m in methods])
    vmax = max(22, float(np.ceil(values.max())))
    im = ax.imshow(values, aspect="auto", cmap="YlOrBr", vmin=0, vmax=vmax)
    for i in range(len(methods)):
        for j in range(len(joint)):
            ax.text(j, i, f"{values[i, j]:.1f}", ha="center", va="center", fontsize=5.8, color="white" if values[i, j] >= vmax*.55 else INK, fontweight="bold" if i == 0 else "normal")
    ax.set_xticks(range(len(joint)), [f"{c[0]}\n{c[2]}" for c in joint])
    ax.set_yticks(range(len(methods)), [NAMES[m] for m in methods], fontsize=6.2)
    ax.tick_params(length=0, pad=2)
    ax.set_xlabel(r"Joint configurations: $N$ (top), $K$ (bottom)", labelpad=3)
    for spine in ax.spines.values(): spine.set_visible(False)
    title(ax, "d", "Joint extrapolation")
    cax = ax.inset_axes([1.025, .0, .027, 1.0])
    cb = fig.colorbar(im, cax=cax, ticks=np.linspace(0, vmax, 3))
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=5.8, length=2, pad=2)
    cb.ax.set_title(r"$D$", fontsize=7, pad=4)
    handles = [Line2D([], [], color=COLORS[m], lw=1.3, marker=MARKERS[m], ms=3, ls=LINESTYLES[m], label=NAMES[m]) for m in methods]
    fig.legend(handles=handles, ncol=5, frameon=False, loc="lower center", bbox_to_anchor=(.51, .035), handlelength=2, columnspacing=1.5, labelspacing=.8)
    if confirm:
        confirmation_panel(fig, gs[2, :], p)
    fig.text(.5, .012, "Eight trained policies; Rule / Random horizontal lines are ID scene references, not training curves.", ha="center", fontsize=6, color=GRAY)
    return save(fig, name)


def confirmation_panel(fig, spec, p):
    groups = ("ID", "K-OOD", "N-OOD", "Joint-OOD", "All24")
    refs={(row["split"],row["group"]):damage(row) for row in p["confirmation_reference"]["group_summary"]}
    formal={(r["method"],r["group"]):damage(r) for r in p["group_summary"]}
    confirm={(r["method"],r["group"]):damage(r) for r in p["confirmation"]["group_summary"]}
    ax=fig.add_subplot(spec)
    for method,offset in (("regir_nomem",-.23),("transfqmix",0),("spectra",.23)):
        for j,group in enumerate(groups):
            a=refs["eval",group] if method=="regir_nomem" else formal[method,group]
            b=refs["confirm",group] if method=="regir_nomem" else confirm[method,group]
            x=j+offset
            ax.plot((x-.035,x+.035),(a[0],b[0]),color=COLORS[method],lw=.65,alpha=.6)
            ax.errorbar(x-.035,a[0],yerr=a[1],fmt=MARKERS[method],ms=3.1,color=COLORS[method],capsize=1.8,elinewidth=.65)
            ax.errorbar(x+.035,b[0],yerr=b[1],fmt=MARKERS[method],ms=3.1,mfc="white",color=COLORS[method],capsize=1.8,elinewidth=.65)
    title(ax,"e","Formal and independent confirmation pools")
    ax.set(xticks=range(5),xticklabels=("ID",r"$K$-OOD",r"$N$-OOD","Joint OOD","All24"),
           ylabel=r"Target damage $D$ $\downarrow$",ylim=(0,11.8),xlim=(-.5,4.5))
    style(ax)
    ax.text(.012,.97,"LEAF: main1009 reference (separate from panels a-d)",transform=ax.transAxes,va="top",color=TEAL,fontsize=6)
    ax.legend(handles=[Line2D([],[],marker="o",color=INK,ls="",ms=3,label="Formal: 9000-9299"),
                       Line2D([],[],marker="o",mfc="white",color=INK,ls="",ms=3,label="Confirm: 110000-110299")],
              loc="upper right",frameon=False,fontsize=5.8,ncol=2,handletextpad=.3,columnspacing=1)


def mechanism_figure(mechanism, summary):
    interventions=mechanism["frozen"]["combined_interventions"]["across_seeds"]
    alpha=mechanism["normal"]["feature_analysis"]["across_seeds"]
    depth=summary["nomem_depth_analysis"]["groups"]
    arms=("loop_bypass","residual_zero","gru_reset","read4","uniform","fixed_fusion","L1","L2")
    labels=("Bypass context","Zero context residual","Reset temporal GRU","Final read only","Uniform fusion","Fixed mean fusion","Execute L1","Execute L2")
    cfgs=((10,10,2),(50,50,2),(30,30,12))
    lookup={(tuple(r["config"]),r["arm"]):r for r in interventions}
    fig=plt.figure(figsize=(7.16,5.25))
    gs=fig.add_gridspec(2,2,left=.17,right=.979,bottom=.13,top=.94,wspace=.46,hspace=.65,height_ratios=(1,1.15))
    ax=fig.add_subplot(gs[0,0]);yy=np.arange(8)[::-1]
    for c,(cfg,color,marker) in enumerate(zip(cfgs,(TEAL,BLUE,PURPLE),("o","s","D"))):
        rows=[lookup[cfg,arm]["paired_reward"] for arm in arms]
        ax.errorbar([-r["mean"] for r in rows],yy+(1-c)*.18,xerr=[r["sample_sd"] for r in rows],fmt=marker,color=color,
                    ms=2.5,capsize=1.5,elinewidth=.65,label=f"{cfg[0]}/{cfg[2]}")
    ax.axvline(0,color=GRAY,lw=.7,ls="--")
    ax.set(yticks=yy,yticklabels=labels,ylim=(-.6,7.7),xlabel=r"Paired damage increase $\Delta D$")
    title(ax,"a","H1: Frozen policy dependence")
    style(ax,"x");ax.grid(axis="y",visible=False);ax.spines["left"].set_visible(False);ax.tick_params(axis="y",length=0,labelsize=5.9)
    ax.legend(title=r"$N/K$",loc="lower right",bbox_to_anchor=(.99,.02),ncol=3,frameon=False,fontsize=5.5,title_fontsize=5.5,handletextpad=.2,columnspacing=.5)
    inner=gs[0,1].subgridspec(1,2,width_ratios=(1.2,1),wspace=.54)
    ax=fig.add_subplot(inner[0]);yy=np.arange(3)[::-1];left=np.zeros(3)
    for r,color in enumerate((TEAL,"#67B6AA",ORANGE,PURPLE)):
        vals=np.array([row["mean_alpha"]["mean"][r]*100 for row in alpha])
        ax.barh(yy,vals,left=left,height=.5,color=color,edgecolor="white",linewidth=.4,label=fr"$\alpha_{r+1}$")
        if r<2:
            for y,base,value in zip(yy,left,vals):ax.text(base+value/2,y,f"{value:.0f}",ha="center",va="center",color="white" if r==0 else INK,fontsize=6)
        left+=vals
    ax.set(xlim=(0,102),ylim=(-.6,3.1),yticks=yy,yticklabels=[f"{r['config'][0]}/{r['config'][2]}" for r in alpha],
           xticks=(0,50,100),xlabel="Fusion weight (%)",ylabel=r"$N/K$")
    title(ax,"b","H2: Normal stage use")
    ax.legend(loc="upper center",bbox_to_anchor=(.6,.98),ncol=4,frameon=False,fontsize=5.1,handlelength=.6,handletextpad=.15,columnspacing=.35)
    style(ax,"x");ax.grid(axis="y",visible=False)
    ax.text(.5,-.36,r"$\alpha_1+\alpha_2$: 98.8-98.9%",ha="center",transform=ax.transAxes,fontsize=5.9,color=TEAL)
    other=fig.add_subplot(inner[1],sharey=ax)
    for field,offset,marker,color,label in (("mean_within_step_defender_alpha_sd",.11,"o",BLUE,"Receivers"),
            ("mean_same_defender_adjacent_step_alpha_abs_change",-.11,"s",ORANGE,"Time")):
        vals=np.array([[row[field][k][0]*100 for k in ("mean","sample_sd")] for row in alpha])
        other.errorbar(vals[:,0],yy+offset,xerr=vals[:,1],fmt=marker,color=color,ms=2.8,capsize=1.6,elinewidth=.65,label=label)
    other.set(xlim=(0,19),xticks=(0,8,16),xlabel=r"$\alpha_1$ variation (pp)");other.tick_params(labelleft=False)
    other.legend(loc="upper center",bbox_to_anchor=(.4,.99),frameon=False,fontsize=5.1,handletextpad=.2)
    style(other,"x");other.grid(axis="y",visible=False);other.spines["left"].set_visible(False)
    ax=fig.add_subplot(gs[1,0]);rounds=np.arange(1,7)
    ax.axvspan(4.5,6.25,color="#EBEFF6",zorder=0);ax.axvline(4.5,color="#8899AD",lw=.8,ls=(0,(2,3)))
    for group,name,color in (("All24","All24",INK),("ID","ID",TEAL),("K-OOD",r"$K$-OOD",ORANGE),("N-OOD",r"$N$-OOD",BLUE),("Joint-OOD","Joint OOD",PURPLE)):
        rows=[depth[group]["depths"][str(r)] for r in rounds]
        y=np.array([r["mean"] for r in rows]);sd=np.array([r["seed_sd"] for r in rows])
        ax.fill_between(rounds,y-sd,y+sd,color=color,alpha=.06,lw=0)
        ax.plot(rounds,y,color=color,marker="o",ms=2.4,lw=1.5 if group=="All24" else .9,label=name)
    ax.set(xlim=(.8,6.2),ylim=(.2,2.65),xticks=rounds,xlabel=r"Frozen execution depth $L$",ylabel=r"Target damage $D$ $\downarrow$")
    title(ax,"c","H3: Reusable refinement");style(ax)
    ax.legend(loc="upper right",bbox_to_anchor=(.99,.86),ncol=2,frameon=False,fontsize=5.5,handlelength=1.1,handletextpad=.25,columnspacing=.6)
    ax.text(5.3,2.49,"Unseen",ha="center",fontsize=5.8,color=BLUE)
    ax=fig.add_subplot(gs[1,1]);yy=np.arange(8)[::-1]
    for c,(cfg,color,marker) in enumerate(zip(cfgs,(TEAL,BLUE,PURPLE),("o","s","D"))):
        rows=[lookup[cfg,arm]["action_change_fraction"] for arm in arms]
        ax.errorbar([r["mean"]*100 for r in rows],yy+(1-c)*.18,xerr=[r["sample_sd"]*100 for r in rows],fmt=marker,color=color,ms=2.5,capsize=1.5,elinewidth=.65)
    ax.set(yticks=yy,yticklabels=labels,ylim=(-.6,7.7),xlim=(0,100),xticks=(0,25,50,75,100),xlabel="Same-state action changes (%)")
    title(ax,"d","Local action dependence")
    style(ax,"x");ax.grid(axis="y",visible=False);ax.spines["left"].set_visible(False);ax.tick_params(axis="y",length=0,labelsize=5.9)
    fig.text(.53,.026,"Separate pools: frozen interventions / normal L4 reads / full 24-config depth evaluation.\nError bars: three-model sample SD; same-state diagnostic actions are not executed.",ha="center",fontsize=5.8,color=GRAY,linespacing=1.4)
    return save(fig,"fig_mechanism")


def protection_figure(formal):
    lookup={(r["method"],tuple(r["config"])):r["metrics"] for r in formal["summary"]}
    methods=("regir_nomem","refil");cfgs=((10,10,2),(30,30,2),(30,30,12),(50,50,2))
    fig=plt.figure(figsize=(7.16,4.55));gs=fig.add_gridspec(2,1,left=.08,right=.985,bottom=.15,top=.94,hspace=.57)
    upper=gs[0].subgridspec(1,3,wspace=.42)
    for i,(field,name) in enumerate((("collision_fraction","Collision mortality"),("blue_intercepted_fraction","Attacker interception"),("any_damage","Damage incidence"))):
        ax=fig.add_subplot(upper[i])
        for method in methods:
            stats=[lookup[method,cfg][field] for cfg in cfgs]
            ax.errorbar(range(4),[r["mean"]*100 for r in stats],yerr=[r["sample_sd"]*100 for r in stats],color=COLORS[method],marker=MARKERS[method],ms=2.5,capsize=1.6,lw=1.2 if method=="regir_nomem" else .8)
        title(ax,chr(97+i),name);ax.set(xticks=range(4),xticklabels=("10/2","30/2","30/12","50/2"),xlabel=r"$N/K$",ylabel="%");style(ax)
    lower=gs[1].subgridspec(1,2,wspace=.35)
    for i,cfg in enumerate(((50,50,2),(30,30,12))):
        ax=fig.add_subplot(lower[i])
        for method in methods:
            row=next(r for r in formal["first_damage_cdf"] if r["method"]==method and tuple(r["config"])==cfg)
            mean,sd=np.array(row["mean"]),np.array(row["sample_sd"])
            ax.fill_between(row["t"],np.maximum(0,mean-sd),np.minimum(1,mean+sd),color=COLORS[method],alpha=.08,lw=0)
            ax.step(row["t"],mean,where="post",color=COLORS[method],lw=1.3 if method=="regir_nomem" else .85)
        title(ax,chr(100+i),f"First damage: N={cfg[0]}, K={cfg[2]}")
        ax.set(xlim=(0,100),ylim=(0,1),xticks=(0,25,50,75,100),xlabel="Physical step",ylabel=r"$\Pr(T_{\rm first}\leq t)$");style(ax)
    handles=[Line2D([],[],color=COLORS[m],lw=1.3,marker=MARKERS[m],ms=3,label=NAMES[m]) for m in methods]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.52,.025),ncol=3,frameon=False)
    return save(fig,"fig_protection")

def trajectory_figure(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    colors = {m:COLORS[m] for m in TRAJECTORY_METHODS}
    fig, axes = plt.subplots(4, 4, figsize=(7.16, 7.0))
    spatial_handles = [Line2D([], [], color="#17669b", label="Defenders"),
        Line2D([], [], color="#cf5f2a", label="Attackers"),
        Line2D([], [], color="#252525", marker="*", linestyle="none", label="Targets"),
        Line2D([], [], color="#666666", marker="o", markerfacecolor="none", linestyle="none", label="Initial"),
        Line2D([], [], color="#666666", marker="x", linestyle="none", label="Destroyed")]
    for block, cfg in enumerate(TRAJECTORY_CONFIGS):
        eps = [next(e for e in data["episodes"] if (e["method"], tuple(e["config"])) == (method, cfg)) for method in TRAJECTORY_METHODS]
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
            shown_damage = 0. if abs(s["R_at_actual_step"]) < 1e-9 else -s["R_at_actual_step"]
            ax.set_title(f"{TRAJECTORY_LABELS[ep['method']]} ({cfg[0]}v{cfg[1]})\n{stamp}, D={shown_damage:.2g}", pad=3)
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
        axes[2*block+1, 2].set_visible(False)
        ax = axes[2*block+1, 3]
        for ep in eps:
            times = [0]+[f["step"] for f in ep["trajectory"]]
            damages = [0.]+[sum(t["cumulative_damage"] for t in f["targets"]) for f in ep["trajectory"]]
            ax.step(times, damages, where="post", color=colors[ep["method"]], linewidth=1.2, alpha=.85)
            ax.scatter(times[-1], damages[-1], s=13, color=colors[ep["method"]], zorder=4)
        ax.axvline(SNAPSHOT, color="#777777", linestyle="--", linewidth=.7)
        ax.set_title(f"{cfg[0]}v{cfg[1]}, K=2\nFull replay damage", pad=3)
        ax.set_xlim(0, 100); ax.set_xticks((0, 30, 60, 100))
        ax.set_xlabel("Physical step (1 s)", labelpad=1); ax.set_ylabel(r"Target damage $D$", labelpad=1)
        ax.yaxis.set_major_locator(MaxNLocator(4)); ax.grid(alpha=.18)
        ax.tick_params(pad=1, length=2)
    fig.legend(handles=spatial_handles, loc="upper center", ncol=5, frameon=False,
        bbox_to_anchor=(.5, 1.0), handlelength=1.2, columnspacing=1)
    fig.legend(handles=[Line2D([], [], color=colors[m], label=TRAJECTORY_LABELS[m]) for m in TRAJECTORY_METHODS],
        loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, .04),
        handlelength=1.3, columnspacing=1.2)
    fig.subplots_adjust(left=.07, right=.985, bottom=.15, top=.912, hspace=.98, wspace=.56)
    fig.text(.5, .012, "Saved skyhad-3.0.0rc2 examples; policy identities match selected final checkpoints.\nDCG 5v5 and ALMA examples differ from historical event records. Examples do not replace formal statistics.", ha="center", fontsize=5.8, color=GRAY)
    return save(fig, "fig_trajectory")


def decision_timing_figure(summary):
    closed=[dict(row) for row in summary["decision_timing20"]["summary"]]
    detail_path=DATA_HAD/"decision_timing.json"
    if any(not row.get("per_episode_mean_ms") for row in closed) and detail_path.is_file():
        details={(row["arm"],tuple(row["config"])):row
                 for row in read_json(detail_path).get("summary",[]) if row.get("method")=="regir_nomem"}
        for row in closed:
            detail=details.get((row["arm"],tuple(row["config"])),{})
            values=detail.get("per_episode_mean_ms")
            if not row.get("per_episode_mean_ms") and values and len(values)==row["episodes"]:
                row["per_episode_mean_ms"]=values
    fixed=summary["decision_timing_fixed100"]["cells"]
    labels=("LEAF R1","LEAF R2","LEAF R3","LEAF R4")
    colors=("#6495CD","#5285BD","#3D73AF","#2A5F99")
    fig,axes=plt.subplots(2,2,figsize=(7.16,5.0))
    fig.subplots_adjust(left=.15,right=.99,top=.94,bottom=.13,hspace=.48,wspace=.62)
    max_closed=max(max(r["per_episode_mean_ms"]) if r.get("per_episode_mean_ms") else
                   max(r["episode_mean_ms"]+r["episode_sd_ms"],r["active_ms_per_step"]) for r in closed)
    max_fixed=max(r["mean_ms"] for r in fixed)
    for col,cfg in enumerate(((30,30,2),(50,50,2))):
        yy=np.arange(len(labels))[::-1];ax=axes[0,col]
        for y,arm,color in zip(yy,labels,colors):
            row=next(r for r in closed if r["arm"]==arm and tuple(r["config"])==cfg)
            vals=row.get("per_episode_mean_ms")
            if vals:
                ax.scatter(vals,y+np.linspace(-.13,.13,len(vals)),s=6,color=color,alpha=.4,lw=0)
            else:
                ax.errorbar(row["episode_mean_ms"],y+.12,xerr=row["episode_sd_ms"],
                            fmt="|",ms=4,color=color,capsize=2,elinewidth=.8)
            value=row["active_ms_per_step"]
            ax.scatter(value,y,s=23,color=color,edgecolor="white",lw=.4,zorder=5)
            ax.text(value+.18,y+.14,f"{value:.3f}",fontsize=5.8,va="bottom")
        ax.set(yticks=yy,yticklabels=labels,xlim=(0,max_closed*1.06),xlabel="Full-team decision latency (ms)")
        title(ax,chr(97+col),f"Closed loop: N={cfg[0]}, 20 scenes")
        style(ax,"x");ax.tick_params(axis="y",length=0,labelsize=6.3)
        ax=axes[1,col]
        for y,arm,color in zip(yy,labels,colors):
            row=next(r for r in fixed if r["arm"]==arm and tuple(r["config"])==cfg)
            value=row["mean_ms"];ax.barh(y,value,height=.55,color=color)
            ax.text(value+.12,y,f"{value:.3f}",fontsize=5.8,va="center")
        ax.set(yticks=yy,yticklabels=labels,xlim=(0,max_fixed*1.15),xlabel="Full-team decision latency (ms)")
        title(ax,chr(99+col),f"Fixed count: N={cfg[0]}, 100 calls")
        style(ax,"x");ax.tick_params(axis="y",length=0,labelsize=6.3)
    upper_note="Upper: episode means (small points), active-step weighted mean (large point)."
    if any(not row.get("per_episode_mean_ms") for row in closed):
        upper_note="Upper: available episode means (small points), otherwise saved episode mean +/- SD (error bar).\nActive-step weighted mean: large point."
    fig.text(.52,.02,upper_note+"\nLower: arithmetic mean of 100 saved direct decisions; each row shares its x scale.",ha="center",fontsize=5.8,color=GRAY)
    return save(fig,"fig_decision_timing")


def coverage_probe_figure(summary):
    groups=summary["coverage_binding_probe"]["groups"]
    arms=("LEAF-H0","LEAF-H1","LEAF-H2");colors=("#97A3AD",BLUE,TEAL)
    fig,axes=plt.subplots(2,2,figsize=(7.16,4.75),sharey=True)
    fig.subplots_adjust(left=.08,right=.985,top=.83,bottom=.16,hspace=.40,wspace=.16)
    for ax,group in zip(axes.flat,groups):
        for x,arm,color in zip(range(3),arms,colors):
            row=group["arms"][arm]["test_accuracy"];mean,sd=row["mean"]*100,row["seed_sd"]*100
            ax.bar(x,mean,color=color,width=.6,alpha=.85);ax.errorbar(x,mean,yerr=sd,color=INK,capsize=3,lw=.8)
            for seed,marker in ((0,"o"),(1,"^"),(2,"s")):
                ax.scatter(x+(seed-1)*.1,row["seed_values"][seed]*100,marker=marker,s=17,color=INK,zorder=5)
            ax.text(x,63.3,f"{mean:.2f}%",ha="center",fontsize=6)
        ax.axhline(50,color=INK,ls="--",lw=.9)
        label=group["label"] if len(group["configs"])==1 else "Equal-weight three configurations"
        ax.set(title=label,xticks=range(3),xticklabels=("LEAF H0","LEAF H1","LEAF H2"),ylim=(40,65),yticks=(40,45,50,55,60,65))
        ax.tick_params(axis="x",labelsize=6);style(ax)
    axes[0,0].set_ylabel("Held-out accuracy (%)");axes[1,0].set_ylabel("Held-out accuracy (%)")
    handles=[Line2D([],[],color=INK,marker=m,ls="",label=f"Model seed {s}") for s,m in ((0,"o"),(1,"^"),(2,"s"))]
    handles.append(Line2D([],[],color=INK,ls="--",label="50% chance level"))
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.52,.98),ncol=4,frameon=False,fontsize=6.3)
    fig.text(.52,.035,"Frozen LEAF features; saved balanced split: 70 fitting / 30 held-out templates per configuration.\nBars: three-model means; error bars: sample SD; all four states share each template's split.",ha="center",fontsize=5.8,color=GRAY)
    return save(fig,"coverage_binding_probe")


def render_figures(output=None, *, data_root=None):
    """Render saved summaries and trajectories; never run or recompute an experiment."""
    global HAD, DATA_HAD, FIGURES
    output=Path(output) if output is not None else Path(__file__).resolve().parents[2]/"outputs"
    with _RENDER_LOCK:
        HAD=output
        saved_root=Path(data_root) if data_root is not None else output
        DATA_HAD=saved_root/"had" if (saved_root/"had").is_dir() else saved_root
        FIGURES=HAD/"figures"
        aggregates=read_json(HAD/"aggregates.json");summary=read_json(output/"summary.json")
        current=aggregates["comparison"]
        mechanism=aggregates.get("mechanism")
        if mechanism is None:
            raise ValueError("Saved neutral mechanism summaries are required; rendering never recomputes them from raw episodes")
        FIGURES.mkdir(parents=True,exist_ok=True)
        with plt.rc_context({"font.family":"DejaVu Sans","font.size":7,"axes.titlesize":8,"axes.labelsize":7,
                "xtick.labelsize":6.5,"ytick.labelsize":6.5,"legend.fontsize":6.7,"pdf.fonttype":42,"axes.linewidth":.6,
                "text.color":INK,"axes.labelcolor":INK,"xtick.color":INK,"ytick.color":INK,"lines.solid_capstyle":"round"}):
            paths=[]
            paths += comparison(current)
            paths += mechanism_figure(mechanism,summary)
            paths += protection_figure(mechanism["formal"])
            trajectory_path=DATA_HAD/"trajectories.json.gz"
            if trajectory_path.is_file():
                trajectories=read_json(trajectory_path)
                finals={(row["method"],row["seed"]):row["artifact_id"] for row in summary["checkpoints"] if row["kind"]=="final"}
                trajectories["episodes"]=[ep for ep in trajectories["episodes"]
                    if ep["method"] in TRAJECTORY_METHODS and (ep["method"]=="rule_nv1"
                    or (ep["method"],ep.get("training_seed")) in finals
                    and ep.get("checkpoint_id")==finals[ep["method"],ep.get("training_seed")])]
                available={(ep["method"],tuple(ep["config"])) for ep in trajectories["episodes"]}
                if all((method,cfg) in available for method in TRAJECTORY_METHODS for cfg in TRAJECTORY_CONFIGS):
                    paths += trajectory_figure(trajectories)
                else:
                    print("Skipped fig_trajectory: selected saved checkpoint/configuration coverage is incomplete.")
            else:
                print("Skipped fig_trajectory: local trajectories.json.gz unavailable; use --data-root for saved data.")
            paths += decision_timing_figure(summary)
            paths += coverage_probe_figure(summary)
        return paths


def refresh_report(output=None, *, run=None):
    """Compatibility entry: render saved figures without creating a Markdown report."""
    return render_figures(output)
