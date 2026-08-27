# 5 Experiments

## 5.1 Questions

1. Do intelligent counter-allocators expose a failure hidden by same-opponent evaluation?
2. Does opponent-pool training improve worst-opponent end-to-end outcomes under a common executor?
3. Does the improvement transfer to unseen team sizes, force ratios, and event schedules?
4. How much training, inference, and reallocation cost does robustness require?

## 5.2 Environments

- OpenSwarmBattle: controlled two-team attrition/reinforcement, protected assets, and attack/intercept/defend objectives; used for causal tests and full cross-play.
- ALMA-derived composite task: public-code anchor for hierarchical allocation; selected only after the author configuration passes the three-day reproduction gate.
- JaxMARL/SMAX or MAgent2 Battle: optional scale-validation environment, with active masks and a documented reinforcement wrapper.

## 5.3 Baselines

Nearest/type matching, constrained Hungarian heuristic, ALMA-style allocation, PLATO-style pointer MAPPO, DT-GAT-style dual-frequency allocation, latest-opponent self-play, and the proposed history-pool method. COPA, RAC, REDA, or flat MARL are conditional baselines depending on implementation reliability and the selected main route.

## 5.4 Evaluation splits

Train on 4v4 and 6v6 with a restricted event distribution. Evaluate on unseen force ratios, two-wave reinforcement, changed event times, held-out scripted counter-allocators, independent self-play seeds, and a fixed-budget learned best response. The headline table is the combined population-and-opponent shift, not the in-distribution setting.

## 5.5 Metrics

Primary: win rate, normalized team return, interception or asset survival, worst-held-out-opponent return, and empirical best-response gain. Secondary: post-event return AUC, recovery time, unassigned critical objectives, over-concentration, and allocation oscillation. Cost: upper inference latency, parameters, FLOPs, environment steps, and pool storage.

## 5.6 Statistical protocol

Formal experiments use at least five training seeds. Each checkpoint-matchup cell uses at least 200 paired episodes with common environment seeds. We report bootstrap 95% confidence intervals and perform prespecified OPAA comparisons with Holm correction. Hyperparameters and checkpoint selection use validation opponents only.

## 5.7 Required ablations

- no history pool;
- uniform versus hard-opponent pool sampling;
- no population randomization;
- matched fixed upper-decision frequency;
- frozen versus equal-budget joint lower-level fine-tuning.

## 5.8 Go/no-go criteria

Continue the main claim only if in-distribution win rate drops no more than two percentage points, held-out worst win improves at least five points or normalized return improves eight percent, empirical exploitability falls at least fifteen percent relative to latest-opponent self-play, and the direction is consistent across at least two unseen force ratios. These thresholds are design gates, not reported findings.
