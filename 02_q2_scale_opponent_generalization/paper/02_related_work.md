# 2. Related Work

## 2.1 Permutation-aware policies and variable populations

Fixed concatenations make a policy depend on arbitrary entity order and maximum population size. Deep Sets gives a general construction for permutation-invariant functions [@zaheer2017deepsets]. In MARL, UPDeT represents entities as tokens and decouples entity-related actions with a Transformer [@hu2021updet]. HPN explicitly separates a permutation-invariant input from a permutation-equivariant entity-action output and uses hypernetworks to process variable entity sets [@hao2023hpn]. We use this PI/PE contract because it is directly testable and compatible with target-selection actions.

Variable-input validity is now a mature direction rather than our contribution. SPECTra reduces entity-processing cost with single-agent query attention [@park2025spectra]. SHPPO studies zero-shot scalable collaboration, while GIRA aligns experience under joins and exits and adapts roles across dynamic populations [@guo2025shppo; @li2026gira]. Evolutionary MARL for multi-UAV combat combines an attention architecture and death masking with diverse opponent populations, explicitly addressing crashes and support events [@wang2024evolutionary]. These methods motivate our scalable backbone, but none of these architectural properties alone determines whether a temporal latent encodes composition or strategy.

## 2.2 Opponent representation and unseen opponents

Opponent modeling conditions decisions on inferred properties of other agents. DRON learns opponent features and mixture-of-experts strategy patterns jointly with a value function [@he2016opponent]. Policy representation methods learn embeddings whose geometry reflects similarities between policies [@grover2018policyrep; @jiang2021metric]. Such representations can generalize to unseen agents, but trajectory distributions depend on both policies and the states in which those policies act.

Recent sequence models strengthen online and offline adaptation. OM-TCN serializes observed opponent behavior and predicts dynamic opponents in competitive MARL [@liu2022omtcn]. TAO uses offline trajectories and in-context learning to infer unseen opponent policies [@jing2024tao]. OEOM generates an open-ended, trajectory-diverse opponent population and trains an in-context model to recognize and respond to opponents [@jing2025oeom]. These works establish that historical context and opponent diversity matter. Our question is orthogonal: whether the inferred context remains a strategy signal when the number and composition of the opponents producing that trajectory changes.

## 2.3 Non-stationary strategies and change detection

Opponent strategy switches are not new. Earlier work embedded change detection in learning and planning against non-stationary repeated-game opponents [@hernandezleal2014switching]. Bayes-OKR maintains intra-episode beliefs, tracks opponents that switch among known policies, detects unknown policies, and reuses response knowledge [@chen2022bayesokr]. Other approaches apply temporal prediction or online error statistics to changing opponents [@liu2022omtcn; @mridul2024opsdemo].

These approaches typically define a change through an altered action distribution or model error. In variable-population team games, the sample size, state distribution, and composition of the agents producing those actions can change simultaneously. We focus on the resulting false-positive problem rather than proposing change detection as a new capability [@mridul2024opsdemo].

## 2.4 Joint world--opponent modeling

The most direct recent neighbor is HyperJD2TSSM, a hierarchical world--opponent model for partially observable multi-UAV cooperative--competitive environments [@cheng2026hyperjd2tssm]. It supports fluctuating numbers with hypernetwork-generated transition models and infers high-level intentions and low-level strategies while predicting both teams' trajectories. Its goal is unified recursive prediction and sample efficiency. Our work differs in scope and evidence: we isolate composition-only and strategy-only interventions, measure false strategy alarms and post-change decision cost, and test a lightweight separation that does not learn a full world model. We therefore do not claim to be the first to combine varying populations with dynamic opponent reasoning.

## 2.5 Evaluation environments

MAgent/MAgent2 Battle provides a many-agent, two-team grid-world with natural attrition and entity-targeted actions [@zheng2018magent]. It is our public primary environment. A controlled SwarmBattle environment supplies matched interventions, because natural game trajectories alone do not reveal the counterfactual of changing composition while holding the opponent policy fixed. The earlier scale-ID/OOD by opponent-ID/OOD matrix remains a secondary evaluation, while the composition-change by strategy-change event matrix is the primary diagnostic.
