# Working title

**Do Opponent Models Mistake Attrition for Strategy Change? Composition-Disentangled Online Adaptation in Team-Competitive MARL**

Alternative: *Composition-Disentangled Opponent Context for Variable-Population Adversarial Games*

# Abstract (pre-results draft)

Online opponent models infer behavioral changes from interaction histories, but team-competitive games contain a second source of non-stationarity: the observed opponent population changes after eliminations, reinforcements, and visibility events. A generic sequence encoder can therefore change its opponent representation even when the surviving opponents continue to execute the same policy, potentially triggering an unnecessary response. We study this **composition--strategy confounding** through a controlled protocol that independently intervenes on opponent composition and policy within an episode. We propose Composition-Disentangled Opponent Context with Hyper Policy Networks (CDOC-HPN), a lightweight extension of a permutation-aware team policy. CDOC-HPN separates an instantaneous composition code from a recurrent, population-normalized conditional-behavior code. During training, paired trajectories encourage the behavior code to remain stable for the same opponent policy under different compositions and to remain discriminative for different policies under matched compositions. The resulting change score is computed only from the behavior code, while both codes condition decentralized decisions. We evaluate false strategy-change alarms under composition-only events, detection delay and post-change regret under policy switches, zero-shot performance against held-out opponents, and inference overhead in MAgent2 Battle and a controlled two-team environment. **[After frozen five-seed experiments, insert measured false-positive reduction, switch AUROC/delay, regret, win-rate, confidence intervals, and latency. Do not claim effectiveness before these results exist.]**

