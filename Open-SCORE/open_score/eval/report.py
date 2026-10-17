"""Render exactly eight shared HAD figures from saved statistics.

No policy, simulator, training, evaluation or scientific-statistics refresh is
imported or executed. Reports and the current paper share the same PNG/PDF pair.
"""
from __future__ import annotations
import csv
import gzip
import json
import math
import os
import shlex
from pathlib import Path
import statistics
import sys
import threading
from datetime import datetime, timedelta, timezone
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

FIGURE_NAMES = ("fig_comparison", "fig_ablation", "fig_mechanism", "fig_protection", "fig_trajectory", "fig_decision_timing", "fixed_depth_pilot100", "coverage_binding_probe")
TEAL, ORANGE, PURPLE = "#007F7A", "#D68B3C", "#8270A8"
BLUE, RED, INK, GRAY, LIGHT = "#507AA4", "#B9636E", "#22343D", "#77838B", "#E5EAED"
METHODS = ("regir_nomem", "refil", "alma", "transfqmix", "dcg", "spectra", "rule_nv1", "b2_qmix_atten")
DESIGNS = ("regir_nomem", "regir_sg", "regir_r0_sg", "regir_r1_sg", "regir_norefil_sg", "regir_last_sg", "regir_fixed4_sg", "regir_untied4_sg")
NAMES = dict(zip(METHODS, ("LEAF", "REFIL", "ALMA", "TransfQMix", "DCG", "SPECTra", "Rule", "QMIX")))
NAMES.update(dict(zip(DESIGNS[1:], ("Memory query", "Direct read", "One round", "No host", "Last read", "Fixed depth", "Unshared rounds"))))
COLORS = dict(zip(METHODS, (TEAL, ORANGE, BLUE, PURPLE, RED, "#87933D", GRAY, "#BB7A9B")))
COLORS.update(dict(zip(DESIGNS[1:], ("#7388b7", "#ba5671", "#745399", "#c79532", "#67A7A1", "#9A8669", "#7C8B59"))))
MARKERS = dict(zip(METHODS, ("o", "s", "^", "D", "v", "P", "x", "h")))
MARKERS.update(dict(zip(DESIGNS[1:], ("s", "v", "D", "^", "P", "h", "X"))))
LINESTYLES = dict(zip(METHODS, ("-", "-", "-", "--", "-.", ":", "--", "-")))
LINESTYLES.update(dict(zip(DESIGNS[1:], ("-", "--", "-.", ":", "--", "-.", ":"))))
TRAJECTORY_METHODS = ("regir_nomem", "refil", "b2_qmix_atten", "dcg", "spectra", "alma", "rule_nv1")
TRAJECTORY_CONFIGS = ((5, 5, 2), (15, 15, 2))
TRAJECTORY_LABELS = {m:NAMES[m] for m in TRAJECTORY_METHODS}
TRAJECTORY_LABELS["spectra"] = "SPECTra (10D)"
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
    config = {(r["method"], tuple(r["config"])): r for r in p["config_summary"] + p.get("rule_config", [])}
    fig = plt.figure(figsize=(7.16, 6.52 if confirm else 5.35))
    gs = fig.add_gridspec(3 if confirm else 2, 2, left=.075, right=.947, top=.95 if confirm else .89, bottom=.14 if confirm else .19, hspace=.65 if confirm else .58, wspace=.50, height_ratios=(1, 1.18, 1.02) if confirm else (1, 1.18))
    ax = fig.add_subplot(gs[0, 0])
    for method in methods[::-1]:
        if method == "rule_nv1":
            ax.axhline(-p["rule_ID_reference"]["reward"]["mean"], color=COLORS[method], ls=LINESTYLES[method], lw=.85)
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
            if method != "rule_nv1":
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
    vmax = 22 if confirm else max(6, float(np.ceil(values.max())))
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
    cb = fig.colorbar(im, cax=cax, ticks=(0, 10, 20))
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=5.8, length=2, pad=2)
    cb.ax.set_title(r"$D$", fontsize=7, pad=4)
    handles = [Line2D([], [], color=COLORS[m], lw=1.3, marker=MARKERS[m], ms=3, ls=LINESTYLES[m], label=NAMES[m]) for m in methods]
    fig.legend(handles=handles, ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(.51, .025), handlelength=2, columnspacing=2.4 if confirm else 1.5, labelspacing=.8)
    if confirm:
        confirmation_panel(fig, gs[2, :], p)
    else:
        fig.text(.5, .985, "Historical saved designs: query conditions differ", ha="center", va="top", fontsize=7, color=GRAY)
    return save(fig, name)


def confirmation_panel(fig, spec, p):
    groups = {key:p["groups"][key] for key in ("ID", "K-OOD", "N-OOD", "Joint-OOD")}
    groups["All24"] = list(dict.fromkeys(tuple(c) for cs in groups.values() for c in cs)) + [(5,5,2),(10,10,2)]
    cells = {}
    with (HAD/"loop_summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if (row["method"]=="regir_nomem" and row["split"] in ("eval","confirm") and row["arm"]=="original"
                    and row.get("archive_source",row.get("source_version"))=="main1009" and int(row["episodes"])==300):
                key=(row["split"],int(row["seed"]),tuple(json.loads(row["config"])))
                if key in cells: raise ValueError("Duplicate saved LEAF reference cell")
                cells[key]=float(row["D"])
    refs={}
    for split in ("eval","confirm"):
        for group, configs in groups.items():
            means=[statistics.mean(cells[split,seed,tuple(c)] for c in configs) for seed in (0,1,2)]
            refs[split,group]=(statistics.mean(means),statistics.stdev(means))
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


def mechanism_figure(mechanism, summary, design):
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
    design_groups={(r["method"],r["group"]):damage(r)[0] for r in design["group_summary"]}
    for group,color,marker in (("N-OOD",BLUE,"^"),("Joint-OOD",PURPLE,"D")):
        ax.plot(1,design_groups["regir_r1_sg",group],marker=marker,ms=4.5,mfc="white",mec=color,mew=1,ls="")
    ax.set(xlim=(.8,6.2),ylim=(.2,2.65),xticks=rounds,xlabel=r"Frozen execution depth $L$",ylabel=r"Target damage $D$ $\downarrow$")
    title(ax,"c","H3: Reusable refinement");style(ax)
    ax.legend(loc="upper right",bbox_to_anchor=(.99,.86),ncol=2,frameon=False,fontsize=5.5,handlelength=1.1,handletextpad=.25,columnspacing=.6)
    ax.text(.02,.96,"Open markers: separately trained one-round",transform=ax.transAxes,va="top",color=GRAY,fontsize=5.5)
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
    methods=("regir_nomem","regir_sg","refil");cfgs=((10,10,2),(30,30,2),(30,30,12),(50,50,2))
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
    fig.text(.5, .012, "SPECTra: historical 10D seed0, artifact 53e7f877dd9548a784e6d7376b42b7e2.\nFixed examples stop at actual termination; they do not replace formal 300-scene statistics.", ha="center", fontsize=5.8, color=GRAY)
    return save(fig, "fig_trajectory")


def decision_timing_figure(summary):
    closed=[dict(row) for row in summary["decision_timing20"]["summary"]]
    detail_path=HAD/"decision_timing.json"
    if any(not row.get("per_episode_mean_ms") for row in closed) and detail_path.is_file():
        details={(row["arm"],tuple(row["config"])):row
                 for row in read_json(detail_path).get("summary",[])}
        for row in closed:
            detail=details.get((row["arm"],tuple(row["config"])),{})
            values=detail.get("per_episode_mean_ms")
            if not row.get("per_episode_mean_ms") and values and len(values)==row["episodes"]:
                row["per_episode_mean_ms"]=values
    fixed=summary["decision_timing_fixed100"]["cells"]
    labels=("LEAF R1","LEAF R2","LEAF R3","LEAF R4","One round","Fixed depth","Unshared rounds")
    colors=("#6495CD","#5285BD","#3D73AF","#2A5F99",GRAY,ORANGE,PURPLE)
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


def fixed_depth_figure(summary):
    from matplotlib.ticker import ScalarFormatter
    groups=summary["fixed_depth_pilot100"]["groups"]
    fig,axes=plt.subplots(2,2,figsize=(7.16,4.75),sharey=True)
    fig.subplots_adjust(left=.08,right=.985,top=.80,bottom=.15,hspace=.38,wspace=.18)
    xs=(1,2,3,4)
    for ax,group in zip(axes.flat,groups):
        fixed=[group["fixed_depth"][str(r)] for r in xs]
        leaf=[group["LEAF_reference"][str(r)] for r in xs]
        for seed,marker,ls in ((0,"^","--"),(1,"s",":"),(2,"v","-.")):
            ax.plot(xs,[r["seed_values"][seed] for r in fixed],color=GRAY,marker=marker,ms=3,ls=ls,lw=.8,alpha=.65)
        ax.plot(xs,[r["mean"] for r in fixed],color=ORANGE,marker="o",ms=3.5,lw=1.7)
        ax.plot(xs,[r["mean"] for r in leaf],color=TEAL,marker="D",ms=3,lw=1.3,ls="--")
        ratios=group["paired_fixed_depth_contrasts"]
        ax.text(.97,.96,f"Fixed-depth mean damage ratio\nL1/L4: {ratios['R1_vs_R4']['damage_ratio_of_means']:.2f}\nL2/L4: {ratios['R2_vs_R4']['damage_ratio_of_means']:.2f}",
                transform=ax.transAxes,ha="right",va="top",fontsize=5.6,bbox=dict(boxstyle="round,pad=.25",facecolor="white",edgecolor=LIGHT))
        label=group["label"] if len(group["configs"])==1 else "Equal-weight three configurations"
        ax.set(title=label,xticks=xs,yscale="log",ylim=(.2,40),xlabel="Frozen execution depth L")
        style(ax);ax.set_yticks((.2,.5,1,2,5,10,20,40));ax.yaxis.set_major_formatter(ScalarFormatter())
    axes[0,0].set_ylabel(r"Target damage $D$ (log scale)")
    axes[1,0].set_ylabel(r"Target damage $D$ (log scale)")
    handles=[Line2D([],[],color=ORANGE,marker="o",label="Fixed depth mean (100 scenes/cell)"),
             Line2D([],[],color=TEAL,marker="D",ls="--",label="LEAF reference (300 scenes/cell)")]
    handles += [Line2D([],[],color=GRAY,marker=m,ls=ls,label=f"Fixed depth: seed {s}") for s,m,ls in ((0,"^","--"),(1,"s",":"),(2,"v","-."))]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.52,.99),ncol=2,frameon=False,fontsize=6.2)
    fig.text(.52,.025,"Saved three-config comparison. Means first average configurations within each model.\nScene counts and query settings differ; no cross-method scene-paired inference is made.",ha="center",fontsize=5.8,color=GRAY)
    return save(fig,"fixed_depth_pilot100")


def coverage_probe_figure(summary):
    groups=summary["coverage_binding_probe"]["groups"]
    arms=("LEAF-H0","LEAF-H1","LEAF-H2","Direct read-H0");colors=("#97A3AD",BLUE,TEAL,ORANGE)
    fig,axes=plt.subplots(2,2,figsize=(7.16,4.75),sharey=True)
    fig.subplots_adjust(left=.08,right=.985,top=.83,bottom=.16,hspace=.40,wspace=.16)
    for ax,group in zip(axes.flat,groups):
        for x,arm,color in zip(range(4),arms,colors):
            row=group["arms"][arm]["test_accuracy"];mean,sd=row["mean"]*100,row["seed_sd"]*100
            ax.bar(x,mean,color=color,width=.6,alpha=.85);ax.errorbar(x,mean,yerr=sd,color=INK,capsize=3,lw=.8)
            for seed,marker in ((0,"o"),(1,"^"),(2,"s")):
                ax.scatter(x+(seed-1)*.1,row["seed_values"][seed]*100,marker=marker,s=17,color=INK,zorder=5)
            ax.text(x,63.3,f"{mean:.2f}%",ha="center",fontsize=6)
        ax.axhline(50,color=INK,ls="--",lw=.9)
        label=group["label"] if len(group["configs"])==1 else "Equal-weight three configurations"
        ax.set(title=label,xticks=range(4),xticklabels=("LEAF H0","LEAF H1","LEAF H2","Direct read H0"),ylim=(40,65),yticks=(40,45,50,55,60,65))
        ax.tick_params(axis="x",labelsize=6);style(ax)
    axes[0,0].set_ylabel("Held-out accuracy (%)");axes[1,0].set_ylabel("Held-out accuracy (%)")
    handles=[Line2D([],[],color=INK,marker=m,ls="",label=f"Model seed {s}") for s,m in ((0,"o"),(1,"^"),(2,"s"))]
    handles.append(Line2D([],[],color=INK,ls="--",label="50% chance level"))
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.52,.98),ncol=4,frameon=False,fontsize=6.3)
    fig.text(.52,.035,"Frozen policy features; saved balanced split: 70 fitting / 30 held-out templates per configuration.\nBars: three-model means; error bars: sample SD; all four states share each template's split.",ha="center",fontsize=5.8,color=GRAY)
    return save(fig,"coverage_binding_probe")


def render_figures(output=None):
    """Render saved summaries and trajectories; never run or recompute an experiment."""
    global HAD, FIGURES
    output=Path(output) if output is not None else Path(__file__).resolve().parents[2]/"outputs/main"
    with _RENDER_LOCK:
        HAD=output/"had";FIGURES=HAD/"figures"
        aggregates=read_json(HAD/"aggregates.json");summary=read_json(output/"summary.json")
        current=aggregates["comparison"]
        design=aggregates["design_comparison"]
        mechanism=aggregates.get("mechanism")
        if mechanism is None:
            raise ValueError("Saved neutral mechanism summaries are required; rendering never recomputes them from raw episodes")
        trajectories=read_json(HAD/"trajectories.json.gz")
        FIGURES.mkdir(parents=True,exist_ok=True)
        with plt.rc_context({"font.family":"DejaVu Sans","font.size":7,"axes.titlesize":8,"axes.labelsize":7,
                "xtick.labelsize":6.5,"ytick.labelsize":6.5,"legend.fontsize":6.7,"pdf.fonttype":42,"axes.linewidth":.6,
                "text.color":INK,"axes.labelcolor":INK,"xtick.color":INK,"ytick.color":INK,"lines.solid_capstyle":"round"}):
            paths=[]
            paths += comparison(current)
            paths += comparison(design,DESIGNS,"fig_ablation",False)
            paths += mechanism_figure(mechanism,summary,design)
            paths += protection_figure(mechanism["formal"])
            paths += trajectory_figure(trajectories)
            paths += decision_timing_figure(summary)
            paths += fixed_depth_figure(summary)
            paths += coverage_probe_figure(summary)
        return paths


def refresh_report(output=None, *, run=None):
    """Compatibility entry: render saved figures; maintain the one Markdown report in docs."""
    return render_figures(output)


def command_text(argv, platform=None):
    """Return an executable command for the machine recorded with the run."""
    platform = os.name if platform is None else platform
    if platform == "nt":
        return "powershell", "& " + " ".join("'" + str(value).replace("'", "''") + "'" for value in argv)
    return "bash", shlex.join([str(value) for value in argv])


def _render_friendly_training(root, *, config=None, state=None, run="train", experiment_arm=None):
    from open_score.utils.logging import read_records
    from open_score.eval.experiment import read_json as optional_json, run_directory, process_lock
    metadata = optional_json(root / "experiment.json", {})
    scheduler = optional_json(root / "scheduler.train.json", {})
    planned = {job["experiment_arm"]: job.get("config", {}) for job in scheduler.get("planned", [])}
    active = {job["experiment_arm"]: job for job in scheduler.get("active", []) if job.get("experiment_arm")}
    queued = {job["experiment_arm"] for job in scheduler.get("waiting_train", []) if job.get("experiment_arm")}
    finished = {job["experiment_arm"]: job for job in scheduler.get("finished", []) if job.get("experiment_arm")}
    figure_path, report_path = root / "had/figures/training.png", root / "报告.md"
    configs, progress, learning, episodes, validations = {}, {}, {}, {}, {}

    def finite_points(rows, field):
        return [(int(row["t_env"]), float(row[field])) for row in rows
                if row.get(field) is not None and math.isfinite(float(row[field]))]

    def number(value):
        return "waiting" if value is None else f"{float(value):.5g}"

    def elapsed(value):
        seconds = int(value or 0)
        return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"

    for arm in ("A", "B", "C"):
        retained = metadata.get("arms", {}).get(arm, {})
        directory = run_directory(root, "regir_nomem", 0, experiment_arm=arm)
        cfg = {**retained.get("config", {}), **planned.get(arm, {}), **optional_json(directory / "config.json", {})}
        if config is not None and config.get("experiment_arm") == arm:
            cfg.update(config)
        configs[arm] = cfg
        row = optional_json(directory / "resource.json", {})
        if state is not None and config is not None and config.get("experiment_arm") == arm:
            row.update(state)
        if arm in active:
            row["status"] = "running"
        elif arm in queued:
            row["status"] = "queued" if scheduler.get("status") in ("running", "stopping") else "pending"
        elif arm in finished:
            row["status"] = finished[arm]["status"]
        row.setdefault("status", "not_started")
        progress[arm] = row
        arm_run = cfg.get("run", run)
        rows = read_records(root, "learning", run=arm_run, method="regir_nomem", seed=0,
                            env="had", experiment_arm=arm)
        learning[arm] = sorted({int(item["t_env"]): item for item in rows}.values(), key=lambda item: item["t_env"])
        episodes[arm] = read_records(root, "episodes", run=arm_run, method="regir_nomem", seed=0,
                                     env="had", experiment_arm=arm)
        grouped = {}
        for item in episodes[arm]:
            if item.get("phase") == "train_eval":
                key = (json.dumps(item["config"], sort_keys=True), int(item["episode_seed"]))
                grouped.setdefault(int(item["eval_point"]), {})[key] = item
        points = []
        for point, by_key in sorted(grouped.items()):
            items = list(by_key.values())
            counts = {}
            for item in items:
                key = json.dumps(item["config"], sort_keys=True)
                counts[key] = counts.get(key, 0) + 1
            if len(items) == 100 and len(counts) == 4 and set(counts.values()) == {25}:
                points.append(dict(point=point, t_env=max(int(item["t_env"]) for item in items),
                    D=statistics.mean(float(item["D"]) for item in items), rows=items))
        validations[arm] = points

    with process_lock(root / ".report.lock") as claimed:
        if not claimed:
            return figure_path, report_path
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        visible_steps = max(20000,
            *(int(row.get("observed_t_env", row.get("t_env", 0))) for row in progress.values()),
            *(int(item["t_env"]) for rows in learning.values() for item in rows),
            *(int(item["t_env"]) for rows in validations.values() for item in rows))
        visible_steps = min(1000000, visible_steps * 1.05)
        with _RENDER_LOCK:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
            colors = {"A": TEAL, "B": BLUE, "C": ORANGE}
            for arm in ("A", "B", "C"):
                series = (finite_points(learning[arm], "train_D"), finite_points(validations[arm], "D"),
                          finite_points(learning[arm], "loss"))
                for index, (ax, data) in enumerate(zip(axes, series)):
                    if data:
                        xs, ys = zip(*data)
                        ax.plot(np.asarray(xs) / 1e6, ys, color=colors[arm], label=arm,
                                marker="o" if index == 1 else None, markersize=2.5, linewidth=1.1)
            for ax, caption, ylabel in zip(axes,
                    ("Training damage (batch mean)", "Common validation damage", "Training loss"),
                    ("Target damage D", "Target damage D", "Loss")):
                if not ax.lines:
                    ax.text(.5, .5, "waiting for data", ha="center", va="center", transform=ax.transAxes, color=GRAY)
                else:
                    ax.legend(title="Arm", fontsize=8)
                ax.set(title=caption, xlabel="Physical steps (millions)", ylabel=ylabel, xlim=(0, visible_steps / 1e6))
                style(ax)
            transitions = progress["C"].get("curriculum_transitions", [])
            if not transitions and learning["C"]:
                transitions = learning["C"][-1].get("curriculum_transitions", [])
            for transition in transitions:
                step = int(transition.get("t_env", 0))
                if step > 0:
                    for ax in axes:
                        ax.axvline(step / 1e6, color=colors["C"], linestyle=":", linewidth=.7, alpha=.6)
            fig.suptitle("LEAF | sky_friendly_v1 | seed 0 | A: random depth, B: fixed depth 4, C: birth curriculum", fontsize=10)
            temporary = figure_path.with_name(f".training.{os.getpid()}.png.pending")
            try:
                fig.savefig(temporary, format="png", dpi=160, facecolor="white")
                os.replace(temporary, figure_path)
            finally:
                plt.close(fig)
                temporary.unlink(missing_ok=True)

        lines = ["# LEAF 统一友好任务实验报告", "",
            f"更新时间：{datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')}（Asia/Shanghai，UTC+8）。",
            "", "同一输出包含 A/B/C 三个实验臂，每臂 seed0、1,000,000 物理步。未启动的实验臂保持未运行，不补造数据。",
            "", "## 真实运行状态", "",
            "| 实验臂 | 状态 / 阶段 | 已采集 / 完整批次 / 恢复点 | 完整验证点 | C课程阶段 | 训练 / 验证耗时 |",
            "|---|---|---|---|---|---|"]
        for arm in ("A", "B", "C"):
            row = progress[arm]
            observed = row.get("observed_t_env", row.get("t_env", 0))
            stage = row.get("curriculum_stage", "waiting") if arm == "C" else "—"
            label = {"not_started": "未运行", "queued": "排队", "pending": "待恢复", "running": "运行中",
                     "completed": "已完成", "complete": "已完成", "stopped": "已停止",
                     "stopped_before_start": "未运行（已停止）", "failed": "失败"}.get(row["status"], row["status"])
            lines.append(f"| {arm} | {label} (`{row['status']}`) / `{row.get('current_phase', 'waiting')}` | "
                f"{observed:,} / {int(row.get('t_env', 0)):,} / {int(row.get('safe_checkpoint_t_env', 0)):,} | "
                f"{len(validations[arm])}/50 | {stage} | {elapsed(row.get('training_seconds'))} / {elapsed(row.get('validation_seconds'))} |")
        lines += ["", "采集中的步数仅用于进度展示。断点恢复使用包含 replay、runner、优化器和随机状态的完整批次 checkpoint；预算与全部 50 个验证点完成后才标记完成。终评由独立命令显式启动。",
            "", "## 共同任务与唯一差异", "",
            "A 使用 R1–R4 随机训练深度、R4验证；B 仅将训练深度固定为 R4；C 保留 A 的深度设置，仅加入出生课程。三臂使用相同最终友好任务、相同验证场景、相同验证种子与预算。全体飞机的四元数与角速度公开，观测布局由实际配置登记。",
            "", "| 实验臂 | lr / gamma / imagined权重 | epsilon退火 / 下限 | 训练深度 / 验证深度 | 实体维度 / 动作数 |",
            "|---|---|---|---|---|"]
        for arm, cfg in configs.items():
            lines.append(f"| {arm} | {number(cfg.get('lr'))} / {number(cfg.get('gamma'))} / {number(cfg.get('lmbda'))} | "
                f"{cfg.get('epsilon_anneal_time', 'waiting')} / {number(cfg.get('epsilon_finish'))} | "
                f"`{cfg.get('global_depths', 'waiting')}` / {cfg.get('global_eval_depth', 'waiting')} | "
                f"{cfg.get('entity_shape', 'waiting')} / {cfg.get('n_actions', 'waiting')} |")
        lines += ["", "完整生效配置、物理参数与观测布局见 [experiment.json](experiment.json) 及各臂模型目录的 `config.json`。原始伤害指标满足 R=-D，D越低越好；loss和训练回报可含配置中的potential shaping。",
            "", "### 最近训练批次的平均回报分量", "", "| 实验臂 | 原始回报 | shaping回报 | 逐物理步塑形后合计 |", "|---|---|---|---|"]
        for arm in ("A", "B", "C"):
            latest = learning[arm][-1] if learning[arm] else {}
            lines.append(f"| {arm} | {number(latest.get('reward_raw_sum'))} | {number(latest.get('reward_shaping_sum'))} | {number(latest.get('reward_shaped_sum'))} |")
        lines += [
            "", "这些分量是逐物理步未折扣统计的批次均值，原始回报为 -D；它们不是 replay 中经过尾段折叠、折扣及可能的退款修正后的奖励总和。",
            "", "## 训练、共同验证与 loss", "", "![三臂训练损伤、共同验证损伤与loss](had/figures/training.png)", "",
            "曲线为实际完整训练批次及完整100局验证点，不平滑、不插值。橙色竖虚线为C课程实际切换位置。缺失面板显示waiting。",
            "", "## 最近五个完整验证点", "",
            "| 实验臂 | 实际点数 | 平均 D | 零受损比例 | 蓝方拦截比例 | 首次受损受限均值（步） |",
            "|---|---|---|---|---|---|"]
        for arm in ("A", "B", "C"):
            recent = validations[arm][-5:]
            items = [item for point in recent for item in point["rows"]]
            average = statistics.mean(float(item["D"]) for item in items) if items else None
            zero = statistics.mean(float(item["D"]) <= 1e-12 for item in items) if items else None
            intercepted = []
            first_damage = []
            for item in items:
                causes = item.get("blue_deaths_by_cause")
                if isinstance(causes, dict) and "intercepted" in causes:
                    intercepted.append(float(causes["intercepted"]) / int(item["config"]["N_B"]))
                times = item.get("first_damage_step_by_target")
                if isinstance(times, list):
                    known = [float(value) for value in times if value is not None]
                    first_damage.append(min(known) if known else float(configs[arm].get("episode_limit", 100)))
            lines.append(f"| {arm} | {len(recent)} | {number(average)} | {number(zero)} | "
                f"{number(statistics.mean(intercepted) if intercepted else None)} | {number(statistics.mean(first_damage) if first_damage else None)} |")
        lines += ["", "使用最多最近五个已完成验证点（每点100局）；不足五点时只汇总实际数据。拦截比例为事件归因为intercepted的蓝方数量除以初始蓝方数量。首次受损时间取目标中最早受损步，无受损回合按100步计入受限均值；未记录的诊断显示waiting。",
            "", "### 最近五点按配置损伤", "", "| 实验臂 | 配置 | 实际局数 | 平均 D |", "|---|---|---|---|"]
        for arm in ("A", "B", "C"):
            grouped = {}
            for point in validations[arm][-5:]:
                for item in point["rows"]:
                    cfg = item["config"]
                    key = (int(cfg["N_R"]), int(cfg["N_B"]), int(cfg["K"]))
                    grouped.setdefault(key, []).append(float(item["D"]))
            if not grouped:
                lines.append(f"| {arm} | waiting | 0 | waiting |")
            for key, values in sorted(grouped.items()):
                lines.append(f"| {arm} | {key[0]}v{key[1]} K{key[2]} | {len(values)} | {number(statistics.mean(values))} |")
        lines += ["", "## C出生课程", "",
                  f"当前阶段：`{progress['C'].get('curriculum_stage', '未运行')}`。实际阶段应用记录：`{transitions}`。",
                  "训练在完整批次边界切换阶段；验证始终回到公共最终任务，随后训练重置重新应用当前课程。完整阶段表保存于C配置，恢复沿用原计划与实际阶段。",
                  "", "## 正式终评", "", "| 实验臂 | 已记录 / 7200 | 状态 | 样本平均 D | 零损伤比例 |", "|---|---|---|---|---|"]
        for arm in ("A", "B", "C"):
            final = [item for item in episodes[arm] if item.get("phase") == "final_eval" and item.get("arm") != "confirmation"]
            evaluation = optional_json(run_directory(root, "regir_nomem", 0, experiment_arm=arm) / "evaluation.json", {})
            if evaluation.get("checkpoint_id"):
                final = [item for item in final if item.get("checkpoint_id") == evaluation["checkpoint_id"]]
            final = list({(json.dumps(item["config"], sort_keys=True), int(item["episode_seed"])): item
                          for item in final}.values())
            count = len(final)
            final_status = "未运行 / waiting" if not count else "完整" if count == 7200 else "进行中 / 不排行"
            lines.append(f"| {arm} | {count}/7200 | {final_status} | "
                f"{number(statistics.mean(float(item['D']) for item in final) if final else None)} | "
                f"{number(statistics.mean(float(item['D']) <= 1e-12 for item in final) if final else None)} |")
        lines += ["", "标准终评为24种配置×300局；上表只汇总已记录的标准终评样本，不混入确认集。未收齐7200局时只报告当前样本均值与比例，不作完整性能排行；实际运行或停止状态见各臂 `evaluation.json`。"]
        exemplar = next((cfg for cfg in configs.values() if cfg.get("launch_argv")), None)
        exemplar = exemplar or next((cfg for cfg in configs.values() if cfg), {})
        platform = exemplar.get("launch_platform", os.name)
        executable = exemplar.get("python_executable", sys.executable)
        shell, train_command = command_text(exemplar.get("launch_argv") or [executable, "-u", "Open-SCORE/scripts/train.py",
            "--stage", "train", "--preset", "sky_friendly_v1", "--arm", "all", "--seed", "0", "--steps", "1000000",
            "--output", str(root), "--devices", "0,1", "--per-gpu", "1", "--progress", "tqdm", "--live-curves"], platform)
        lines += ["", "## 启动、恢复与查看", "", f"```{shell}", train_command, "```", "",
                  "同一命令追加 `--resume` 恢复；已完成臂跳过、已中断臂恢复、未运行臂从零开始。A/B先占两张卡，C在任一槽空闲后接续，不自动执行终评。", ""]
        commands = [
            [executable, "Open-SCORE/scripts/train.py", "--stage", "status", "--output", str(root)],
            [executable, "Open-SCORE/scripts/train.py", "--stage", "stop", "--arm", "A", "--output", str(root)],
            [executable, "Open-SCORE/scripts/train.py", "--stage", "stop", "--arm", "all", "--output", str(root)],
            [executable, "Open-SCORE/scripts/eval.py", "--stage", "final", "--arm", "all", "--method", "regir_nomem",
             "--seed", "0", "--output", str(root), "--workers", "2", "--device", "cpu"],
        ]
        lines += [f"```{shell}", *[command_text(command, platform)[1] for command in commands], "```", "",
                  "共享数据位于 `had/episodes.csv`、`had/learning.csv`、`had/progress.csv`，由 `experiment_arm` 区分A/B/C；原 `arm` 字段仍表示标准/确认等评估分支。所有正式结果合并在本报告，不生成每臂报告。"]
        temporary_report = report_path.with_name(f".报告.{os.getpid()}.md.pending")
        try:
            temporary_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
            os.replace(temporary_report, report_path)
        finally:
            temporary_report.unlink(missing_ok=True)
    return figure_path, report_path


def render_training(output, *, method="regir_nomem", seed=0, run="train", config=None, state=None, experiment_arm=None):
    """Refresh one training PNG and its one report from actual saved records."""
    from open_score.utils.logging import read_records
    root = Path(output)
    root = root.parent if root.name == "had" else root
    from open_score.eval.experiment import read_json as optional_json
    metadata = optional_json(root / "experiment.json", {})
    scheduler = optional_json(root / "scheduler.train.json", {})
    if ((config or {}).get("preset") == "sky_friendly_v1" or metadata.get("preset") == "sky_friendly_v1"
            or scheduler.get("preset") == "sky_friendly_v1"):
        return _render_friendly_training(root, config=config, state=state, run=run, experiment_arm=experiment_arm)
    directory = root / "had" / method / "models" / f"seed_{int(seed)}"

    def optional_json(path):
        return read_json(path) if path.exists() else {}

    config = optional_json(directory / "config.json") if config is None else dict(config)
    progress = optional_json(directory / "resource.json")
    progress.update(state or {})
    training = read_records(root, "learning", run=run, method=method, seed=int(seed), env="had")
    training = sorted({int(row["t_env"]): row for row in training}.values(), key=lambda row: row["t_env"])
    episodes = read_records(root, "episodes", run=run, method=method, seed=int(seed), env="had")
    grouped = {}
    for row in episodes:
        if row.get("phase") == "train_eval":
            key = (json.dumps(row["config"], sort_keys=True), int(row["episode_seed"]))
            grouped.setdefault(int(row["eval_point"]), {})[key] = row
    validations = []
    for point, rows_by_key in sorted(grouped.items()):
        rows = list(rows_by_key.values())
        counts = {}
        for row in rows:
            key = json.dumps(row["config"], sort_keys=True)
            counts[key] = counts.get(key, 0) + 1
        if len(rows) == 100 and len(counts) == 4 and set(counts.values()) == {25}:
            validations.append(dict(point=point, t_env=max(int(row["t_env"]) for row in rows),
                                    D=statistics.mean(float(row["D"]) for row in rows)))

    def finite_points(rows, field):
        return [(int(row["t_env"]), float(row[field])) for row in rows
                if row.get(field) is not None and math.isfinite(float(row[field]))]

    budget = int(config.get("t_max", progress.get("budget_steps", 0)))
    completed_steps = int(progress.get("t_env", 0))
    observed_steps = int(progress.get("observed_t_env", completed_steps))
    safe_steps = int(progress.get("safe_checkpoint_t_env", 0))
    status = progress.get("status", "waiting")
    phase = progress.get("current_phase", "waiting")
    points = (finite_points(training, "train_D"), finite_points(validations, "D"), finite_points(training, "loss"))
    visible_steps = max(20000, observed_steps, *(x for data in points for x, _ in data))
    visible_steps = min(budget, visible_steps * 1.05) if budget else visible_steps * 1.05
    figure_path = root / "had" / "figures" / "training.png"
    report_path = root / "报告.md"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    with _RENDER_LOCK:
        fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.8), constrained_layout=True)
        for ax, data, caption, ylabel, color in zip(axes, points,
                ("Training damage (batch mean)", "Validation damage (100 episodes)", "Training loss"),
                ("Target damage D", "Target damage D", "Loss"), (TEAL, BLUE, ORANGE)):
            if data:
                xs, ys = zip(*data)
                ax.plot(np.asarray(xs) / 1e6, ys, color=color, linewidth=1.15,
                        marker="o" if caption.startswith("Validation") else None, markersize=2.5)
            else:
                ax.text(.5, .5, "waiting for data", ha="center", va="center", transform=ax.transAxes, color=GRAY)
            ax.set(title=caption, xlabel="Physical steps (millions)", ylabel=ylabel)
            ax.set_xlim(0, visible_steps / 1e6)
            style(ax)
        fig.suptitle(f"LEAF seed {seed} | {status}: {phase} | {observed_steps:,}/{budget:,} steps | "
                     f"validation {len(validations)}/50", fontsize=10, color=INK)
        temporary = figure_path.with_name(".training.png.pending")
        try:
            fig.savefig(temporary, format="png", dpi=160, facecolor="white")
            os.replace(temporary, figure_path)
        finally:
            plt.close(fig)
            temporary.unlink(missing_ok=True)

        latest_train = points[0][-1][1] if points[0] else None
        latest_validation = validations[-1]["D"] if validations else None
        best_validation = min((row["D"] for row in validations), default=None)
        latest_loss = points[2][-1][1] if points[2] else None
        number = lambda value: "waiting" if value is None else f"{value:.6g}"
        identity = config.get("env_identity", {})
        environment = config.get("env_args", {})
        final_rows = sum(row.get("phase") == "final_eval" for row in episodes)
        final_status = f"已记录 {final_rows} 局，属于单独执行的评估。" if final_rows else "未运行；本次命令仅训练及训练内验证。"
        platform = config.get("launch_platform", "nt" if ":\\" in config.get("python_executable", "") else os.name)
        shell, evaluation_command = command_text([config.get("python_executable", sys.executable),
            "-X", "utf8", "Open-SCORE/scripts/eval.py", "--stage", "final", "--method", method,
            "--seed", str(seed), "--output", str(root), "--workers", "2", "--device", "cpu"], platform)
        if config.get("launch_argv"):
            _, launch_command = command_text(config["launch_argv"], platform)
        else:
            launch_command = config.get("launch_command", "waiting for launch command")
            if platform == "nt" and not launch_command.lstrip().startswith("& "):
                launch_command = "& " + launch_command
        report = f"""# LEAF SkyHAD 固定翼 seed{seed} 训练报告

更新时间：{datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')}（Asia/Shanghai，UTC+8）。本报告与下图按实际日志原地更新。

## 当前状态

- 状态：`{status}`；阶段：`{phase}`。
- 物理步：已采集 {observed_steps:,} / {budget:,}；完整采集批次 {completed_steps:,}；可恢复检查点 {safe_steps:,}。
- 验证：完成 {len(validations)} / 50 个验证点（每点 4 种配置 × 25 局）；已写入验证数据 {sum(len(rows) for rows in grouped.values())} 局。
- 最新训练批次 D：{number(latest_train)}；最新验证 D：{number(latest_validation)}；已完成验证点最低 D：{number(best_validation)}；最新 loss：{number(latest_loss)}。
- 正式终评：{final_status}

采集中的物理步仅用于展示进度；恢复训练只使用完整采集与参数更新批次的检查点。缺少数据的面板显示 waiting，不补造训练或验证结果。

## 实验配置

| 项目 | 实际配置 |
|---|---|
| 算法 / 内部键 | LEAF / `{method}` |
| 训练种子 / run | {seed} / `{run}` |
| 载具 / 动作 | `{environment.get('env_agent_type', 'waiting')}` / `{environment.get('env_agent_action_type', 'acceleration')}` |
| 任务 / 空间维度 | `{environment.get('task_mode', 'damage')}` / {config.get('spatial_dim', environment.get('spatial_dim', 'waiting'))} |
| 物理版本 / 协议 | `{identity.get('had_core_version', 'waiting')}` / `{identity.get('physics_protocol', 'waiting')}` |
| 训练分布 / 每局上限 | `{environment.get('train_dist', 'mixed_le10')}` / {environment.get('max_steps', 100)} 物理步 |
| 蓝方规则 | `{environment.get('blue_upper', 'reactive')} + {environment.get('blue_lower', 'rush')}` |
| 实体特征 / 动作数 | {config.get('entity_shape', 'waiting')} 维 / {config.get('n_actions', 'waiting')} |
| 训练深度 / 验证深度 | `{config.get('global_depths', [])}` / {config.get('global_eval_depth', 'waiting')} |
| 并行环境 / 每批学习更新 | {config.get('batch_size_run', 'waiting')} / {config.get('training_iters', 'waiting')} |
| 实现版本 | `{config.get('implementation_revision', 'waiting')}` |

完整生效物理参数与特征布局见 [experiment.json](experiment.json)；模型配置见 [config.json](had/{method}/models/seed_{seed}/config.json)。伤害任务的原始整局回报为 R=-D，D 越低越好；训练目标保留配置中的 potential shaping。

## 训练与验证曲线

![训练批次损伤、验证损伤与训练 loss](had/figures/training.png)

训练 D 与 loss 为实际完整采集/学习批次记录；验证 D 只显示已收齐 100 局的验证点。日志使用 `had/learning.csv`、`had/episodes.csv` 与 `had/progress.csv`；权重保留 `final.pt`、验证最优 `best.pt` 及训练恢复文件。仅在物理预算与全部 50 个验证点均完成后标记 completed。

## 运行命令

```{shell}
{launch_command}
```

中断后在同一命令末尾添加 `--resume` 继续。状态与停止分别使用原训练脚本的 `--stage status` / `--stage stop`，并传入相同 `--output`。

后续正式终评命令如下，当前训练任务不会自动执行：

```{shell}
{evaluation_command}
```
"""
        temporary_report = report_path.with_name(".报告.md.pending")
        try:
            temporary_report.write_text(report, encoding="utf-8")
            os.replace(temporary_report, report_path)
        finally:
            temporary_report.unlink(missing_ok=True)
    return figure_path, report_path
