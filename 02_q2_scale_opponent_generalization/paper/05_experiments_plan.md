# 5. Experiments (locked plan)

## 5.1 Questions

1. Does a generic recurrent opponent context produce false strategy-change signals after composition-only events?
2. Does CDOC reduce those false alarms while preserving policy-switch detection?
3. Does identification quality translate into lower post-switch regret and stronger held-out-opponent decisions?
4. Are gains retained at unseen population scales with acceptable overhead?

## 5.2 Controlled event protocol

We cross a composition intervention `C` with an opponent-policy intervention `S` at a randomized event time, producing `C0S0`, `C1S0`, `C0S1`, and `C1S1`. Composition interventions remove, reinforce, or substitute opponent units while keeping their controller fixed. Strategy interventions switch controller implementations while keeping composition fixed when possible. Matched manifests share initial seeds, event times, and pre-event states.

SwarmBattle is the controlled diagnostic environment. MAgent2 Battle is the public primary environment and additionally supplies natural attrition. Training and held-out opponent controllers are separated by implementation, not random seed. The scale-ID/OOD by opponent-ID/OOD matrix is reported as a secondary generalization result.

## 5.3 Methods

Main comparisons are padded IPPO, DeepSets-IPPO, HPN, generic OC-HPN, and CDOC-HPN. Ablations remove cross-composition invariance, matched-composition separation, or conditional behavior prediction; a shuffled-pair control tests whether arbitrary regularization explains results. A capacity-matched OC-HPN controls for parameter count. An oracle with true event labels gives an adaptation upper bound but is not ranked as a deployable baseline.

## 5.4 Metrics

Primary endpoints are:

- composition-only false-positive rate `FPR_comp`;
- switch-detection `AUROC_switch`;
- post-switch regret over a frozen horizon `PostSwitchRegret@K`.

Secondary outcomes include base false-positive rate, detection delay, win rate, worst held-out-opponent win rate, return, and recovery time. Mechanistic probes report how accurately composition and strategy can be decoded from the behavior code but cannot establish success without decision improvements. Cost measures include parameters, latency, memory, and scaling with entity count.

## 5.5 Statistics and leakage control

The pilot uses three training seeds; final tables use five independent training seeds and at least 100 evaluation episodes per seed, event cell, intensity, and held-out opponent. Confidence intervals are bootstrapped over training seeds. CDOC and OC-HPN use paired seeds and identical frozen event manifests. Thresholds and hyperparameters are selected only with training opponents and validation compositions; held-out opponent returns are evaluated once after freezing.

## 5.6 Continuation gates

Before implementing the full method, generic OC-HPN must show a material problem: at least a 15-percentage-point increase from `C0S0` to `C1S0` false alarms, or an absolute Spearman correlation of at least 0.3 between latent change and composition intervention magnitude.

For full experiments, CDOC should reduce `FPR_comp` by at least 30% relative to OC-HPN, lose no more than 0.02 switch AUROC, reduce `C1S1` post-switch regret by at least 15%, lose no more than three win-rate points in distribution, and add less than 20% inference latency. These are project decision thresholds, not reported findings.

## 5.7 Planned figures and tables

- Figure 1: composition-only versus strategy-only ambiguity;
- Figure 2: HPN, OC-HPN, and CDOC information paths;
- Table 1: false positives, AUROC, delay, and regret in all four event cells;
- Figure 3: change-score trajectories around event time;
- Table 2: invariance, separation, shuffled-pair, and capacity ablations;
- Figure 4: intervention magnitude versus false alarms/latent drift;
- Table 3: public MAgent2 held-out-opponent results;
- Figure 5: latency and memory versus population size.

No numerical performance claim is written before the frozen five-seed runs.

