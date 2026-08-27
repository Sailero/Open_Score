# 4 Method

## 4.1 Overview

COGAR contains three primary components: a frozen task-conditioned executor, an intervention-trained Execution-Grounded Outcome Model (EGOM), and a receding-horizon bilateral allocation solver. A fourth alternating adaptation stage is optional and is not required for the main method.

## 4.2 Task-conditioned executor

The executor shares parameters across agents and represents allies, opponents, and task entities as variable-length sets. Each local agent receives its own attributes, relative ally and opponent tokens, an asset token, and its assigned macro-task token. A Deep Sets, entity-attention, or UPDeT-style encoder feeds an IPPO/MAPPO actor and critic.

Training randomizes local force size, unit state, geometry, mission token, and opponent policy. The primary upper-level study freezes a single executor checkpoint and reuses it for every allocation method. This control prevents lower-level policy quality from being misattributed to the commander. A policy-version identifier \(\nu_L\) is stored with every outcome record.

## 4.3 Same-snapshot intervention dataset

We collect source snapshots from a coverage mixture of scripted allocators, exploratory random assignments, learned commanders, population events, and near-boundary local encounters. For source snapshot \(s_q\), a candidate generator creates feasible red and blue allocations. We select candidate pairs that cover force-ratio boundaries, diverse geometries, high predicted value, and model disagreement.

For every selected \((s_q,g^R,g^B)\), the simulator is restored from the identical serialized state and rolled forward for one macro horizon under the frozen executor. Each candidate from the same source snapshot is evaluated with the same random seed set \(\Omega_q\), which implements common random numbers. We initially use 16–32 rollouts per candidate pair, retaining individual outcomes rather than only sample means.

Dataset partitions are grouped by source snapshot, scenario generator, and opponent-policy lineage. All candidates and repeats from a source snapshot remain in one partition. Separate training, validation, calibration, ID test, and OOD test sets prevent candidate-level and opponent-lineage leakage.

## 4.4 Execution-Grounded Outcome Model

### Entity and partition representation

EGOM encodes five token families: red agents, blue agents, assets, task tokens, and allocation labels. The model forms a permutation-invariant embedding for each front and applies a cross-front block over all front embeddings. This full-partition context allows a coalition's value to depend on both teams' complete allocations and shared spatial context rather than assuming a characteristic function \(v(C)\) with no externalities.

### Distribution heads

For each front \(j\), EGOM predicts:

\[
p_\theta(Z_j=1),\quad
p_\theta(\Delta n_j^R),\quad
p_\theta(\Delta n_j^B),\quad
p_\theta(T_j),\quad
p_\theta(D_j).
\]

The binary success head uses a Binomial or Beta-Binomial likelihood over repeated rollouts. Casualties and asset progress use categorical, ordinal, or quantile heads; resolution time uses a censored or quantile loss when the macro horizon truncates an encounter. A weighted multi-task objective is

\[
\mathcal L_{\rm EGOM}
=\lambda_Z\mathcal L_Z+\lambda_R\mathcal L_R
+\lambda_B\mathcal L_B+\lambda_T\mathcal L_T+\lambda_D\mathcal L_D.
\]

We train \(M=3\)–5 independent models. Temperature scaling is fitted only on the calibration split for the binary heads. Ensemble disagreement provides an empirical epistemic-uncertainty feature. No formal Bayesian or OOD coverage guarantee is claimed.

### Optimizer-aware aggregation

After initial training, the current commander is run on training-distribution and designated adaptation states. We add candidate pairs that are selected by the solver, have high ensemble disagreement, or exhibit a large rank difference between predictive mean and lower confidence bound. These candidates are evaluated in the true simulator and added to the training set. We cap this process at two rounds and re-fit calibration after each round.

## 4.5 Candidate bilateral allocations

For three homogeneous fronts, we enumerate count vectors whose entries sum to the number of active agents. There are

\[
\binom{n+J-1}{J-1}
\]

vectors for \(n\) agents and \(J\) fronts. Specific identities are assigned through minimum-cost matching using distance, health, unit type, and the previous assignment. For task-conditioned or heterogeneous settings, beam search constructs complete allocations and retains at most \(B\in\{32,64\}\) candidates per side. The previous allocation and a reserve option are always included.

Every compared upper-level method receives the same candidate sets. Candidate-generation quality is evaluated independently against exhaustive enumeration on small instances.

## 4.6 Risk-aware payoff matrix

EGOM evaluates all candidate pairs \(a\in\mathcal G^R(s)\), \(b\in\mathcal G^B(s)\) in one batch. A red-team short-horizon payoff is

\[
\begin{aligned}
\widehat U_\theta(s,a,b)
=&\;w_Z\sum_j \operatorname{LCB}_\alpha[Z_j]
-w_R\sum_j\mathbb E[\Delta n_j^R]
+w_B\sum_j\mathbb E[\Delta n_j^B]\\
&-w_T\sum_j\mathbb E[T_j]
-w_D\sum_j\mathbb E[D_j^R]
-\lambda C_{\rm switch}(a,g_{t-1}^R)
-\beta\sigma_{\rm epi}(s,a,b).
\end{aligned}
\]

Weights are fixed from the environment objective and tuned on validation data under the same budget as baseline utility weights. Switching cost counts reassigned agents and optional travel distance. The uncertainty penalty is ablated separately from the calibrated lower confidence bound.

The primary environment is symmetric zero-sum, so the blue payoff is \(-\widehat U_\theta\). Non-zero-sum objectives are outside the initial scope.

## 4.7 Regularized macro-allocation game

The two candidate sets define a small matrix game. We compute an entropy-regularized maximin strategy by linear programming, mirror descent, or reproducible fictitious play:

\[
\max_{x\in\Delta(\mathcal G^R)}
\min_{y\in\Delta(\mathcal G^B)}
x^\top\widehat U_\theta y
+\eta H(x)-\eta H(y).
\]

The executed assignment is sampled from the resulting mixture with a recorded seed. A deterministic maximum-probability variant is included as an ablation. When only one side is controlled, the opponent candidate set contains heuristic, historical, and learned response policies, and the commander optimizes a worst-\(k\) or CVaR objective.

The game solver is not itself a claimed contribution. Hand-crafted utility, uncalibrated scalar value, EGOM mean, EGOM risk, and Monte Carlo oracle all use the same solver to isolate payoff-interface quality.

## 4.8 Receding-horizon and event-triggered execution

An allocation is normally held for \(H=10\) or \(20\) low-level steps. Death, reinforcement arrival, or asset-state change triggers early replanning after a minimum dwell time. Switching cost and dwell time prevent oscillation. The algorithm records each trigger, assignment change, travel distance, inference time, and outcome.

## 4.9 Unknown opponents

The base model marginalizes over a training opponent pool. If opponent identity is hidden, its prediction is explicitly interpreted under that pool. An optional history encoder consumes a short summary of recent movement, focus-fire, retreat, and allocation changes. This addition is retained only if it improves held-out-opponent calibration and final mission outcomes; opponent-type classification is not an objective.

## 4.10 Optional alternating adaptation

Joint hierarchy training is established in prior work and is not a primary contribution. A limited extension alternates:

1. train EGOM and the commander with \(\pi_L\) fixed;
2. fine-tune \(\pi_L\) on commander-induced tasks with a KL anchor and old-task replay;
3. recollect interventions under the new policy version and recalibrate EGOM;
4. update the commander for at most two cycles.

Gradients stop at discrete assignments. The extension is rejected if it harms ID performance, calibration, or reproducibility, or fails a predeclared OOD improvement threshold.

## 4.11 Computational cost

With 32 candidates per side, EGOM scores 1,024 bilateral pairs in a batch, followed by a small matrix-game solve. We report mean and p95 latency over hardware and candidate budgets, as well as a performance–latency Pareto. A method exceeding 10% of the low-level control period is described as low-frequency planning rather than real-time control.
