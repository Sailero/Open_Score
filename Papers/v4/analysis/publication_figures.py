"""Render v4 figures using saved results; no training/evaluation/model loading.

Source precedence: docs/LEAF_主实验结果分析.md and supplementary section 7.6 are
authoritative. Published means and sample SDs are parsed directly. Validation
and comparison curves use main/had/aggregates.json['paper_current'], falling
back to 'paper1007' for older archives. Ablations retain 'paper1007'; nothing
is copied from raw data. Only R -> D and paired delta-R -> delta-D are transformed.
Formal groups average configurations within each seed, then summarize three
seed means. Rule SD is across scenes. Frozen-depth, readout, intervention, and
formal300 sources remain separate; plot captions give their sampling units.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parents[1]
FIGURES = PAPER / "figures"
TEAL, ORANGE, PURPLE = "#007F7A", "#D68B3C", "#8270A8"
BLUE, RED, INK, GRAY, LIGHT = "#507AA4", "#B9636E", "#22343D", "#77838B", "#E5EAED"
METHODS = ("regir_nomem", "refil", "alma", "transfqmix", "dcg", "spectra", "rule_nv1", "b2_qmix_atten")
NAMES = dict(zip(METHODS, ("LEAF", "REFIL", "ALMA", "TransfQMix", "DCG", "SPECTra", "Rule", "QMIX")))
COLORS = dict(zip(METHODS, (TEAL, ORANGE, BLUE, PURPLE, RED, "#87933D", GRAY, "#BB7A9B")))
MARKERS = dict(zip(METHODS, ("o", "s", "^", "D", "v", "P", "x", "h")))
LINESTYLES = dict(zip(METHODS, ("-", "-", "-", "--", "-.", ":", "--", "-")))

def section(text, heading):
    return text.split(heading, 1)[1].split("\n### ", 1)[0]

def md_table(text, prefix):
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(prefix))
    rows = []
    for line in lines[start + 2:]:
        if not line.startswith("|"): break
        rows.append([s.strip().replace("**", "") for s in line.strip("|").split("|")])
    return rows

def number(cell):
    vals = cell.replace("−", "-").replace("**", "").split("±")
    return float(vals[0]), float(vals[1]) if len(vals) > 1 else 0.

def style(ax, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#A9B3B9")
    ax.tick_params(length=2.4, color="#A9B3B9", pad=2)
    ax.set_axisbelow(True)
    ax.grid(axis=grid, color=LIGHT, lw=.55)

def title(ax, letter, text):
    ax.set_title(f"({letter})  {text}", loc="left", pad=7, fontweight="bold", color=INK)

def save(fig, basename):
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{basename}.{ext}", facecolor="white", dpi=260)
    plt.close(fig)


def comparison(p):
    config = {(r["method"], tuple(r["config"])): r for r in p["config_summary"] + p["rule_config"]}
    fig = plt.figure(figsize=(7.16, 4.72))
    gs = fig.add_gridspec(2, 2, left=.075, right=.947, top=.94, bottom=.19, hspace=.58, wspace=.40, height_ratios=(1, 1.18))
    ax = fig.add_subplot(gs[0, 0])
    for method in METHODS[::-1]:
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
        for method in METHODS[::-1]:
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
    values = np.array([[-config[m, c]["reward"]["mean"] for c in joint] for m in METHODS])
    vmax = 22
    im = ax.imshow(values, aspect="auto", cmap="YlOrBr", vmin=0, vmax=vmax)
    for i in range(len(METHODS)):
        for j in range(len(joint)):
            ax.text(j, i, f"{values[i, j]:.1f}", ha="center", va="center", fontsize=5.8, color="white" if values[i, j] >= 12 else INK, fontweight="bold" if i == 0 else "normal")
    ax.set_xticks(range(len(joint)), [f"{c[0]}\n{c[2]}" for c in joint])
    ax.set_yticks(range(len(METHODS)), [NAMES[m] for m in METHODS], fontsize=6.2)
    ax.tick_params(length=0, pad=2)
    ax.set_xlabel(r"Joint configurations: $N$ (top), $K$ (bottom)", labelpad=3)
    for spine in ax.spines.values(): spine.set_visible(False)
    title(ax, "d", "Joint extrapolation")
    cax = ax.inset_axes([1.025, .0, .027, 1.0])
    cb = fig.colorbar(im, cax=cax, ticks=(0, 10, 20))
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=5.8, length=2, pad=2)
    cb.ax.set_title(r"$D$", fontsize=7, pad=4)
    handles = [Line2D([], [], color=COLORS[m], lw=1.3, marker=MARKERS[m], ms=3, ls=LINESTYLES[m], label=NAMES[m]) for m in METHODS]
    fig.legend(handles=handles, ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(.51, .02), handlelength=2, columnspacing=2.8, labelspacing=.8)
    save(fig, "fig_comparison")

def ablation(main_text, p):
    rows = md_table(section(main_text, "### 3.2 "), "| \u65b9\u6cd5 |")
    rows = {r[0].replace("LEAF\u2013", ""): r[1:] for r in rows}
    variants = ("Direct read", "Memory query", "Last read", "No host")
    labels = ("C: bypass context", "Q: add history to query", "F: use final read only", "H: remove host package")
    # Compute differences before rounding, from the same formal aggregation.
    # The authoritative report rounds the absolute means to three decimals.
    group_damage = {(r["method"], r["group"]): -r["reward"]["mean"] for r in p["group_summary"]}
    keys = ("regir_r0_sg", "regir_sg", "regir_last_sg", "regir_norefil_sg")
    groups = ("ID", "K-OOD", "N-OOD", "Joint-OOD")
    values = np.array([[group_damage[key, group]-group_damage["regir_nomem", group] for group in groups] for key in keys])
    fig = plt.figure(figsize=(7.16, 3.20))
    modules = (("C", "Context construction", r"shared $F_\theta$"), ("Q", "Current-self reader", r"fixed $q_i$"), ("F", "Stage fusion", r"$\sum_r\alpha_r u^{(r)}$"), ("H", "Host integration", "REFIL + temporal GRU"))
    shades = ("#E6F3F1", "#FFF1DF", "#ECE7F4", "#E8EFF7")
    box_x = (.035, .275, .515, .755)
    for i, ((code, name, formula), x, shade) in enumerate(zip(modules, box_x, shades)):
        fig.text(x+.1, .915, f"{code}. {name}\n{formula}", ha="center", va="center", fontsize=7, linespacing=1.5, bbox=dict(boxstyle="round,pad=.45", facecolor=shade, edgecolor="#A6B4BE", linewidth=.65))
        if i<3:
            fig.text(x+.221, .915, r"$\longrightarrow$", ha="center", va="center", fontsize=10, color=GRAY)
    fig.text(.50, .79, "Historical variants: stopped-gradient history queries; LEAF: current-self query", ha="center", color=GRAY, fontsize=6.5)
    ax = fig.add_axes([.285, .16, .59, .56])
    im = ax.imshow(values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=2.2)
    for i in range(4):
        for j in range(4):
            ax.text(j,i,f"+{values[i,j]:.3f}",ha="center",va="center",fontsize=8,color="white" if values[i,j]>1.35 else INK,fontweight="bold" if j==3 and i==0 else "normal")
    ax.set_xticks(range(4),("ID",r"$K$-OOD",r"$N$-OOD","Joint OOD"))
    ax.set_yticks(range(4),labels,fontsize=7)
    ax.tick_params(length=0,pad=5)
    for spine in ax.spines.values():spine.set_visible(False)
    ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,4,1),minor=True)
    ax.grid(which="minor",color="white",linewidth=2);ax.tick_params(which="minor",length=0)
    cax=fig.add_axes([.90,.16,.018,.56]);cb=fig.colorbar(im,cax=cax,ticks=(0,1,2));cb.outline.set_visible(False);cb.ax.tick_params(labelsize=6,length=2,pad=2)
    cb.ax.set_title(r"$\Delta D$",fontsize=7,pad=5)
    fig.text(.56,.025,r"Damage increase versus LEAF: $\Delta D = D_{variant}-D_{LEAF}$",ha="center",fontsize=6.8,color=GRAY)
    save(fig,"fig_ablation")

def mechanism(main_text, supplement):
    depth = md_table(section(main_text, "### 4.3 "), "| \u6df1\u5ea6 |")
    alpha = md_table(section(main_text, "### 4.1 "), "| \u914d\u7f6e | \u03b11")
    behavior = md_table(section(main_text, "### 4.2 "), "| \u914d\u7f6e | \u65b9\u6cd5 |")
    interventions = {r[0]: r[1:] for r in md_table(section(supplement, "### 7.6 "), "| \u5e72\u9884 |")}
    variants={r[0].replace("LEAF\u2013", ""):r[1:] for r in md_table(section(main_text,"### 3.2 "),"| \u65b9\u6cd5 |")}
    fig=plt.figure(figsize=(7.16,4.88))
    gs=fig.add_gridspec(2,2,left=.08,right=.979,bottom=.145,top=.94,wspace=.39,hspace=.63,height_ratios=(1,1.18))

    ax=fig.add_subplot(gs[0,0])
    pos=ax.get_position();ax.set_position([pos.x0+.072,pos.y0,pos.width-.072,pos.height])
    arms=(("H0\u76f4\u63a5\u8bfb\u53d6\u65c1\u8def",r"Bypass context\nread $H^{(0)}$"),("\u5b8c\u6574\u5173\u7cfb\u6b8b\u5dee\u7f6e\u96f6","Zero context\nresidual"))
    for c,(color,marker,label) in enumerate(zip((TEAL,BLUE,PURPLE),("o","s","D"),("10/2","50/2","30/12"))):
        vals=np.array([number(interventions[key][c]) for key,_ in arms])
        ax.errorbar(-vals[:,0],np.arange(2)[::-1]+(1-c)*.19,xerr=vals[:,1],fmt=marker,color=color,ms=3.2,capsize=2,elinewidth=.75,label=label)
    ax.set(yticks=(1,0),yticklabels=[label.replace(r"\n","\n") for _,label in arms],ylim=(-.7,1.8),xlim=(0,45),xticks=(0,15,30,45),xlabel=r"Paired damage increase $\Delta D$ $\uparrow$")
    title(ax,"a","H1: Use of constructed context")
    style(ax,"x");ax.grid(axis="y",visible=False);ax.spines["left"].set_visible(False);ax.tick_params(axis="y",length=0,labelsize=6.1)
    ax.legend(title=r"$N/K$",loc="upper center",bbox_to_anchor=(.60,1.0),ncol=3,frameon=False,fontsize=5.9,title_fontsize=5.9,columnspacing=.7,handletextpad=.3)

    inner=gs[0,1].subgridspec(1,2,width_ratios=(1.25,1),wspace=.54)
    ax=fig.add_subplot(inner[0]);yy=np.arange(3)[::-1];left=np.zeros(3)
    for r,color in enumerate((TEAL,"#67B6AA",ORANGE,PURPLE)):
        vals=np.array([number(row[r+1])[0] for row in alpha])
        ax.barh(yy,vals,left=left,height=.5,color=color,edgecolor="white",linewidth=.4,label=fr"$\alpha_{r+1}$")
        if r<2:
            for y,base,value in zip(yy,left,vals):ax.text(base+value/2,y,f"{value:.0f}",ha="center",va="center",color="white" if r==0 else INK,fontsize=6.5)
        left+=vals
    ax.set(xlim=(0,102),ylim=(-.6,3.1),yticks=yy,yticklabels=("10/2","30/2","30/12"),xticks=(0,50,100),xlabel="Fusion weight (%)",ylabel=r"$N/K$")
    title(ax,"b","H2: Receiver-specific stage use")
    ax.legend(loc="upper center",bbox_to_anchor=(.65,.97),ncol=4,frameon=False,fontsize=5.4,handlelength=.6,handletextpad=.15,columnspacing=.4)
    style(ax,"x");ax.grid(axis="y",visible=False)
    ax.text(.5,-.40,r"$\alpha_1+\alpha_2$: 98.8-98.9%",ha="center",transform=ax.transAxes,fontsize=6.5,color=TEAL)
    other=fig.add_subplot(inner[1],sharey=ax)
    for col,offset,marker,color,label in ((5,.11,"o",BLUE,"Receivers"),(6,-.11,"s",ORANGE,"Time")):
        vals=np.array([number(row[col]) for row in alpha]);other.errorbar(vals[:,0],yy+offset,xerr=vals[:,1],fmt=marker,color=color,ms=3,capsize=1.8,elinewidth=.7,label=label)
    other.set(xlim=(0,19),xticks=(0,8,16),xlabel=r"$\alpha_1$ variation (pp)");other.tick_params(labelleft=False)
    other.legend(loc="upper center",bbox_to_anchor=(.38,.99),frameon=False,fontsize=5.4,handletextpad=.2)
    style(other,"x");other.grid(axis="y",visible=False);other.spines["left"].set_visible(False)

    ax=fig.add_subplot(gs[1,0]);rounds=np.arange(1,7)
    ax.axvspan(4.5,6.25,color="#EBEFF6",zorder=0);ax.axvline(4.5,color="#8899AD",lw=.8,ls=(0,(2,3)))
    group_specs=((1,"All24",INK),(2,"ID",TEAL),(3,r"$K$-OOD",ORANGE),(4,r"$N$-OOD",BLUE),(5,"Joint OOD",PURPLE))
    for col,name,color in group_specs:
        vals=np.array([number(row[col]) for row in depth]);ax.fill_between(rounds,vals[:,0]-vals[:,1],vals[:,0]+vals[:,1],color=color,alpha=.06,lw=0);ax.plot(rounds,vals[:,0],color=color,marker="o",ms=2.5,lw=1.5 if col==1 else .95,label=name)
    for g,color,marker,label,offset in ((2,BLUE,"^","One-round: N-OOD",.08),(3,PURPLE,"D","One-round: Joint",.07)):
        mean=number(variants["One round"][g])[0];ax.plot(1,mean,marker=marker,ms=5,mfc="white",mec=color,mew=1,ls="",zorder=5);ax.text(1.22,mean+offset,label,color=color,fontsize=5.8,va="center")
    ax.set(xlim=(.8,6.2),ylim=(.2,2.62),xticks=rounds,xlabel=r"Frozen execution depth $L$",ylabel=r"Target damage $D$ $\downarrow$")
    ax.text(2.7,2.57,"Training depths",ha="center",va="top",fontsize=6.2,color=GRAY);ax.text(5.4,2.57,"Unseen",ha="center",va="top",fontsize=6.2,color=BLUE)
    title(ax,"c","H3: Reusable refinement")
    style(ax);ax.legend(loc="upper right",bbox_to_anchor=(.99,.86),ncol=2,frameon=False,fontsize=5.6,handlelength=1.1,handletextpad=.25,columnspacing=.6)

    inner=gs[1,1].subgridspec(1,3,wspace=.36);cfgs=("10v10 K2","30v30 K2","30v30 K12","50v50 K2")
    table={(r[0],r[1]):r[2:] for r in behavior};axes=[]
    for metric,(label,lim,ticks) in enumerate(((r"Collision $C\downarrow$",(0,24),(0,10,20)),(r"Intercept $I\uparrow$",(68,91),(70,80,90)),(r"Damage events $P\downarrow$",(0,100),(0,50,100))),start=1):
        ax=fig.add_subplot(inner[metric-1]);axes.append(ax);yy=np.arange(4)[::-1]
        for y,cfg in zip(yy,cfgs):
            a,b=number(table[cfg,"LEAF"][metric]),number(table[cfg,"REFIL"][metric]);ax.plot((a[0],b[0]),(y,y),color="#C9D2D6",lw=1.6,zorder=1)
            for value,col,mark in ((a,TEAL,"o"),(b,ORANGE,"s")):ax.errorbar(value[0],y,xerr=value[1],fmt=mark,ms=3,color=col,elinewidth=.75,capsize=1.7,zorder=3)
        ax.set(xlim=lim,ylim=(-.65,3.9),yticks=yy,yticklabels=("10/2","30/2","30/12","50/2") if metric==1 else [""]*4,xticks=ticks,xlabel="%");ax.set_title(label,fontsize=6.3,pad=2,y=.87)
        style(ax,"x");ax.grid(axis="y",visible=False);ax.tick_params(axis="y",length=0,labelsize=6);ax.spines["left"].set_visible(False)
    axes[0].text(0,1.065,"(d)  Protection outcomes",transform=axes[0].transAxes,fontsize=8,fontweight="bold",color=INK);axes[0].set_ylabel(r"$N/K$",labelpad=3)
    axes[1].legend(handles=[Line2D([],[],marker="o",ls="",color=TEAL,ms=3,label="LEAF"),Line2D([],[],marker="s",ls="",color=ORANGE,ms=3,label="REFIL")],loc="upper center",bbox_to_anchor=(.52,-.23),ncol=2,frameon=False,fontsize=6.1,handletextpad=.1,columnspacing=.7)
    save(fig,"fig_mechanism")

def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.7, "pdf.fonttype": 42, "axes.linewidth": .6, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "lines.solid_capstyle": "round"})
    main_text = (REPO / "docs/LEAF_主实验结果分析.md").read_text(encoding="utf-8")
    supplement = (REPO / "docs/LEAF_补充实验结果分析.md").read_text(encoding="utf-8")
    aggregates = json.loads((REPO / "Open-SCORE/outputs/main/had/aggregates.json").read_text(encoding="utf-8"))
    comparison(aggregates.get("paper_current", aggregates["paper1007"]))
    ablation(main_text, aggregates["paper1007"])
    mechanism(main_text, supplement)
    print("Rendered fig_comparison, fig_ablation, fig_mechanism (PDF and PNG) from saved results.")

if __name__ == "__main__":
    main()
