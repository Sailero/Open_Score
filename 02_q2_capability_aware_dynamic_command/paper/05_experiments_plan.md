# 5 Experiments

## 5.1 Research questions

The experiments test:

1. whether EGOM is calibrated on random and commander-selected allocations;
2. whether same-snapshot intervention data improves counterfactual allocation ranking over natural logs;
3. whether calibrated risk improves final mission outcomes beyond an uncalibrated critic or capacity-matched commander;
4. whether gains persist under held-out population, event, type, geometry, and opponent-policy shifts;
5. whether bilateral game solving improves worst-opponent robustness beyond robust greedy assignment;
6. whether any gain is compatible with low allocation churn and online latency.

## 5.2 Environments

### MultiFront-AD

The primary environment contains three critical assets, symmetric attack/defense role swaps, within-episode casualties, and zero to two reinforcement waves per side. The initial protocol uses six or eight active agents, local 1–4 vs 1–4 executor training, and maximum active slots of twelve per team.

### External validation

We will use at least one of:

- ALMA StarCraft multi-army, if its Docker/SC2 stack passes a three-day smoke test;
- a modified SMAX scenario with multiple target regions and active-slot reinforcements;
- MAgent2 Battle augmented with objectives and reinforcement events.

The external environment need not support every feature, but must test whether the payoff interface transfers beyond the custom NumPy environment.

## 5.3 Baselines

Non-learning baselines are static/even split, event-even split, nearest/threat-first, force-ratio allocation, and capacity-constrained min-cost matching. Learned-score baselines are XGBoost, a Fu-style linear capability model, Carion-style unary/pairwise scores with structured inference, REDA-style additive values, and an uncalibrated MAPPO/allocator critic. Hierarchical baselines are ALMA-style AQL, an HHMARL-style frozen commander, a pointer/MAPPO commander without EGOM, hierarchical self-play with the same upper-level capacity, and flat MAPPO/QMIX. Double-IC with hand-crafted combat utility is the primary competitive-coalition baseline. A high-budget simulator rollout planner is an oracle upper bound on small instances.

All upper-level comparisons share the same executor checkpoint, observations, candidate sets, event sequences, and compute/search budget.

## 5.4 Distribution shifts

We predefine:

- ID states with new seeds and positions;
- unseen total populations and force ratios;
- unseen reinforcement timing, burst size, and casualty rate;
- held-out unit-type combinations;
- held-out asset spacing, entrances, and occlusion;
- opponent algorithms and training lineages excluded from all training data;
- a joint shift combining population, event, and opponent changes.

Candidates and repeats from the same source snapshot cannot cross data partitions.

## 5.5 Metrics

Outcome-model metrics include Brier score, NLL, reliability plots, adaptive ECE, calibration slope/intercept, casualty and time error, predictive-interval coverage, false-safe rate, ranking correlation, top-\(k\) regret, and Monte Carlo selection regret. Each is measured on random candidates and separately after commander selection.

System metrics include asset-level mission success, team return, worst-opponent return, casualties, asset damage, completion time, recovery time and post-event control AUC, empirical fixed-budget best-response gain, assignment churn, travel distance, communication tokens, and mean/p95 latency.

## 5.6 Ablations

The causal ablation set includes:

- natural logged assignments versus paired interventions;
- single rollout versus repeated outcome distributions;
- uncalibrated versus temperature-scaled versus ensemble risk;
- local coalition only versus full-partition context;
- predicted mean versus lower confidence bound;
- no switch cost and alternative replanning frequencies;
- one fixed opponent, opponent pool, and bilateral game;
- shuffled EGOM outputs and same-capacity no-EGOM commander;
- frozen executor, naive joint fine-tuning, and alternating refresh;
- Monte Carlo oracle EGOM.

## 5.7 Statistical protocol

Development uses three seeds; final learned results use at least five independent training seeds. Each fixed matchup/event suite uses at least 500 paired evaluation episodes or an equivalent sample size justified by power analysis. We report paired bootstrap 95% confidence intervals, effect sizes, all failed runs, total environment interactions, wall-clock time, hardware, and hyperparameter-search budgets.

## 5.8 Predeclared continuation thresholds

The project continues as the full COGAR route only if:

- calibration reduces ID or OOD ECE below 0.05 or by at least 20% relative to the uncalibrated model;
- false-safe rate and oracle selection regret improve on selected candidates;
- worst-opponent success improves by roughly five percentage points or normalized return by 8% on at least two held-out force ratios;
- empirical best-response gain decreases by at least 15% relative to the capacity-matched no-EGOM method;
- ID success decreases by no more than two percentage points;
- p95 upper-level latency remains below 10% of a low-level control period.

These values are go/no-go criteria, not reported results.

## 5.9 Planned result tables

The main table reports ID success, scale-OOD success, worst held-out-opponent return, false-safe rate, allocation regret, event recovery, churn, and p95 latency. A second table reports outcome-model metrics before and after selection. A cross-play payoff matrix and reliability diagrams provide robustness and calibration evidence. Performance–latency and performance–churn curves establish operational cost.

## 5.10 Threats to validity

The key threats are a custom environment tailored to allocation, weak lower-level execution, insufficient boundary-state coverage, leakage across cloned candidates, non-additive cross-front interactions, candidate-generation bottlenecks, model exploitation by the solver, and opponent pools too similar to held-out policies. The external environment, full-partition model, oracle candidate study, lineage-level split, selected-candidate evaluation, and optimizer-aware data collection are designed to expose rather than conceal these failures.
