# 3 Preliminaries

## 3.1 Open-population competitive Markov game

We consider two teams \(k\in\{R,B\}\). At low-level time \(u\), team \(k\) has an active agent set \(N_u^k\). Death, withdrawal, or reinforcement events may change \(N_u^k\) within an episode. A maximum slot set is used for implementation, but only active slots participate in observations, actions, rewards, or allocations.

The environment state \(s_u\) contains active-agent attributes, positions, health and cooldown variables, three asset states, the remaining horizon, and event context. Low-level agents have local observations \(o_u^i\). A team shares a return based on asset survival or breach, casualties, mission progress, and time. The primary task is symmetric and zero-sum after swapping attack and defense roles.

## 3.2 Macro tasks and coalitions

At macro step \(t\), corresponding to low-level time \(u=tH\), each commander assigns every active agent to a front–task token:

\[
g_t^k:N_t^k\rightarrow \mathcal X\times\mathcal M_k\cup\{\mathrm{reserve}\},
\]

where \(\mathcal X=\{x_1,x_2,x_3\}\) is the set of critical assets and \(\mathcal M_k\) contains role-appropriate mission tokens such as assault, fix, hold, and intercept. Agents sharing a front token form a coalition. The allocation therefore specifies both coalition structure and mission; no independent clustering label is optimized.

A shared lower-level policy executes the allocation for \(H\) steps:

\[
a_u^i\sim \pi_L(\cdot\mid o_u^i,g_t^k(i)),\qquad u=tH,\ldots,(t+1)H-1.
\]

The frozen executor and low-level environment induce a macro transition

\[
s_{t+1}\sim P_H(\cdot\mid s_t,g_t^R,g_t^B,\pi_L).
\]

Death, reinforcement, and asset-state events may terminate the macro interval early and trigger replanning.

## 3.3 Outcome distribution versus value function

A policy value is

\[
V^{\pi_L,\pi_E}(s,g)
=\mathbb E\left[\sum_{\tau=0}^{\infty}\gamma^\tau r_{\tau}\mid s,g,\pi_L,\pi_E\right].
\]

It depends on reward shaping, discounting, and the current continuation policies. A calibrated task-success probability instead estimates

\[
p_\theta(Z=1\mid s,g^R,g^B,\pi_L,H).
\]

If the only undiscounted reward were the terminal binary event, the true value and success probability would coincide mathematically; a renamed value head would then provide no novelty. Our outcome variable is explicitly multi-dimensional:

\[
Y_H=\{Z_j,\Delta n_j^R,\Delta n_j^B,T_j,D_j\}_{j=1}^J,
\]

where \(Z_j\) denotes local task success, \(\Delta n_j^k\) losses, \(T_j\) resolution time, and \(D_j\) asset damage or progress. The model is trained with repeated rollout outcomes and evaluated for calibration, not merely return prediction.

## 3.4 Assignment interventions

For a source snapshot \(S=s\), we clone the simulator and explicitly set a feasible bilateral macro allocation \(G=(g^R,g^B)\). Holding the executor and macro horizon fixed yields the simulator intervention target

\[
m(s,g)=\mathbb E[Y_H\mid S=s,\operatorname{do}(G=g),\pi_L].
\]

This notation describes a controlled simulator intervention, not causal identification from observational deployment logs. Multiple candidate allocations are evaluated from the same state with common random numbers to reduce paired-comparison variance.

## 3.5 Calibration, false-safe rate, and allocation regret

For a binary outcome, perfect calibration requires

\[
\Pr(Z=1\mid \hat p=p)=p.
\]

We evaluate Brier score, negative log-likelihood, reliability diagrams, and expected calibration error. A decision-relevant error is the false-safe rate at threshold \(\tau\):

\[
\operatorname{FS}_\tau
=\Pr(Z=0\mid \operatorname{LCB}(\hat p)\ge\tau).
\]

Let \(U_{\rm MC}(g)\) denote a high-budget Monte Carlo estimate of a candidate's true short-horizon utility. For selected allocation \(\hat g\) and oracle allocation \(g^\star\), allocation regret is

\[
\mathcal R=U_{\rm MC}(g^\star)-U_{\rm MC}(\hat g).
\]

Both calibration and regret are measured on randomly sampled candidates and separately on candidates selected by the commander.

## 3.6 Objective

The objective is not to recover a unique ground-truth partition. We seek a commander that maximizes final team mission return while controlling dangerous model overestimation, reassignment cost, and online latency under held-out population, event, geometry, composition, and opponent-policy distributions.
