# 3 Preliminaries

## 3.1 Open-population team-competitive Markov game

We consider two teams (k\in\{R,B\}). At time (t), the active members are (N_t^k\subseteq\bar N^k), and the active local objectives are (X_t^k\subseteq\bar X^k). An event process (E) may add or remove agents and objectives during an episode. The underlying state, action, transition, and reward components are consequently time-indexed through the active sets.

The environment is team competitive: members of each team share a team reward, and the main benchmark uses (r_t^R=-r_t^B). This two-team formulation differs from cooperative open ad hoc teamwork even though both permit dynamic participation.

## 3.2 Hierarchical policies

Team (k)'s upper policy is

\[
z_t^k\sim\mu_k(\cdot\mid s_t,N_t^R,N_t^B,X_t^k),
\]

where (z_t^k(i)\in X_t^k\cup\{\varnothing\}) assigns active agent (i) to one objective. The assignment persists for (K) primitive steps unless an allowed event-triggered update occurs. A shared lower policy executes

\[
a_{t,i}\sim\pi_{\mathrm{low}}(\cdot\mid o_{t,i},z_t^k(i)).
\]

The primary study freezes (pi_{\mathrm{low}}) and changes only (mu_k).

## 3.3 Open events and test shifts

We distinguish:

- attrition: removal caused by environment interaction;
- reinforcement: exogenous activation of a new agent during the episode;
- task openness: objective appearance or disappearance;
- population shift: a train-test change in active-set size or team ratio;
- opponent shift: a train-test change in the opposing upper-level policy.

The principal test distribution combines population and opponent shift. Padding used by an implementation is not treated as a theoretical fixed population: inactive slots are excluded from the policy support and loss.

## 3.4 Empirical robustness

For candidate allocator (mu) and held-out opponent set (mathcal O), we report

\[
V_{\min}(\mu)=\min_{\nu\in\mathcal O}V(\mu,\nu).
\]

We also train an empirical best response (widehat{BR}^H(\mu)) under fixed interaction budget (H). The resulting best-response gain is an empirical exploitability proxy, not exact NashConv. Every compared policy uses the same response class, initializations, seeds, and budget.

## 3.5 Common-executor principle

Let (Pi_{mathrm{low}}) be one pretrained executor checkpoint. All primary upper-level methods use exactly (Pi_{mathrm{low}}). This blocks the confound

\[
\text{better outcome}=\text{better allocation}+\text{better primitive control}.
\]

A secondary equal-budget joint-finetuning experiment tests whether the ordering survives when this control is relaxed.
