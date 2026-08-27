# Title and Abstract

## Working title

**From Local Capability to Global Command: Calibrated Outcome-Guided Reallocation in Open-Population Multi-Front Games**

Alternative title:

**COGAR: Execution-Grounded and Risk-Aware Reallocation for Dynamic Multi-Front Team Games**

## Status note

This is a preregistration-style paper draft. The method and experiments described below are planned; no numerical improvement is claimed until the corresponding artifact and statistical result exist. COGAR is a provisional acronym pending a final name-collision search.

## Abstract

Large competitive teams cannot rely on independent action selection alone: a macro-level commander must continually decide which agents should address which local objectives. This decision becomes difficult when casualties and reinforcements change both teams within an episode, because a favorable force ratio does not imply that a temporary subteam can execute a task from its current geometry, health, time budget, and opponent behavior. Existing hierarchical multi-agent reinforcement learning, structured allocation, and coalition-formation methods already provide commanders, dynamic assignments, and joint training, but they commonly rely on implicit value functions or hand-designed utilities rather than an auditable estimate of what a fixed executor can actually accomplish.

We study a symmetric three-front attack–defense game in which both teams reallocate active agents after periodic or population-changing events. We propose COGAR, a receding-horizon allocation framework built around an Execution-Grounded Outcome Model (EGOM). From cloned battlefield snapshots, we intervene on multiple unexecuted bilateral allocations and roll each candidate out under the same frozen low-level policies and common random numbers. A permutation-invariant ensemble predicts calibrated distributions over local task success, casualties, resolution time, and asset damage. COGAR converts their lower confidence bounds, epistemic uncertainty, and reassignment costs into a bilateral payoff matrix and solves a regularized macro-allocation game before executing one low-level horizon.

The planned evaluation separates prediction quality from decision value. It measures post-selection calibration, false-safe allocations, Monte Carlo allocation regret, asset-level mission success, recovery after casualties or reinforcements, cross-play robustness, empirical best-response gain, and online latency under held-out population ratios, event schedules, unit compositions, geometries, and opponent policies. Comparisons share an identical low-level executor and candidate budget, and include heuristics, structured learned scores, ALMA-style allocation, Double-IC-style competitive coalition formation, capacity-matched learned commanders, and a rollout oracle. The central hypothesis is that execution-grounded calibration matters only if it reduces dangerous force misallocation and improves final mission outcomes under distribution shift.
