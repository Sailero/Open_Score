# 2 Related Work

## 2.1 Variable-composition execution policies

Variable team composition has been addressed through entity-centric and graph-based representations. UPDeT decouples policies through transformer entity processing for changing observation and action configurations [@hu2021updet]. GPL models open ad hoc teams in which fixed-policy teammates enter and leave [@rahman2021gpl]. COPA uses a global coach and attention-based players to generalize to new team compositions [@liu2021copa], while REFIL improves entity-based value learning through random subgroups [@iqbal2021refil]. These works make a task-conditioned, set-based executor a natural implementation choice. They also mean that variable-length inputs, deaths, reinforcements, or attention alone cannot support a novelty claim. We treat the lower-level executor as shared infrastructure and freeze it in primary upper-level comparisons.

## 2.2 Hierarchical multi-agent allocation

Carion et al. formulate cooperative assignment as structured prediction: a learned scoring model parameterizes a centralized inference problem and transfers from small instances to much larger agent–task configurations [@carion2019structured]. ALMA decomposes composite tasks into high-level subtask allocation and low-level execution, supports separately trained components, and obtains its strongest reported performance through joint training [@iqbal2022alma]. HHMARL uses pretrained fight/escape policies under a high-level air-combat commander [@selmonaj2023hierarchical]. More recent command-and-control work alternates upper task allocation and lower execution while modeling other commanders [@c2opponent2026], and DT-GAT-MARL combines graph-based interception allocation with MAPPO maneuver control [@jia2026dtgat].

Our hierarchy is not proposed as a new abstraction. We instead ask whether an independently calibrated outcome interface provides decision value beyond a learned scalar score or implicit critic. Freezing the executor is a primary experimental control rather than a claim that separate training is generally superior.

## 2.3 Learned capability and outcome models

Dantas et al. train XGBoost on thousands of constructive simulations to predict a domain-specific engagement quality index for beyond-visual-range air combat [@dantas2021bvr]. Fu et al. learn linear task requirements and agent capabilities from team configurations and performance pairs, then embed the learned constraints in task-allocation optimization [@fu2022capabilities]. Lin and Tron estimate task-completion probability and expected reward from execution data and use confidence bounds for bi-level multi-robot allocation under unknown dynamics [@lin2025adaptive]. These studies directly preclude treating supervised combat-outcome prediction or capability-aware allocation as new by itself.

We focus on three different properties: interventions on multiple unexecuted assignments cloned from the same source state; results conditioned on complete allocations of both competitive teams; and post-selection reliability when an optimizer actively searches for overestimated candidates. The model predicts success, casualties, time, and asset progress rather than an unconstrained scalar score.

## 2.4 Competitive coalition formation

Coalition formation addresses how agents group to accomplish tasks, but much work assumes cooperative objectives or hand-designed characteristic functions. Duan et al. explicitly couple intra-team cooperation and inter-team competition for UAV swarms, and solve the resulting two-level game with iterated best response nested in fictitious play [@duan2025twolevel]. Recent adaptive alliance work also performs dynamic group formation for UAV combat [@dynamicalliance2026]. Thus, bilateral coalition formation and dynamic reassignment are established.

COGAR retains the game-theoretic allocation view but replaces an analytic coalition utility with a distribution learned from interventions on the actual fixed executor. Its empirical question is whether calibration and uncertainty reduce misallocation under population and opponent-policy shift, not whether a coalition equilibrium can be computed from a known payoff.

## 2.5 Calibration and decision-aware model error

Modern neural classifiers can be miscalibrated, and temperature scaling is a strong post-hoc baseline [@guo2017calibration]. Deep ensembles provide a practical approximation to epistemic uncertainty [@lakshminarayanan2017ensembles]. Set Transformer supplies a permutation-invariant architecture for interacting sets [@lee2019settransformer]. We use these established components without claiming them as contributions.

The relevant difficulty is selection: a combinatorial optimizer preferentially queries high predicted values, so evaluation on random candidate allocations can conceal severe error on selected allocations. We therefore measure reliability and false-safe rates after selection, compare natural-log and intervention data, and add a limited number of optimizer-aware data-collection rounds.

## 2.6 Positioning

No individual component of COGAR is unprecedented. The testable gap is narrower: whether a calibrated, intervention-trained distribution over the fixed executor's bilateral coalition outcomes can serve as a useful and auditable payoff interface for receding-horizon allocation when both teams change population within an episode. Our literature search did not identify a single prior work that simultaneously evaluates this interface, two strategic allocators, bilateral within-episode casualties and reinforcements, and post-selection reliability under held-out scale and opponent policies. This search observation is not used as a universal first claim.
