# 2 Related Work

## Open multi-agent systems

Open agent systems model environments in which agents, tasks, or agent types may enter, leave, or change during operation. Open ad hoc teamwork focuses on cooperation with dynamically participating and often unknown teammates. GPL represents such teammates through a graph-based joint value model, while CIAO gives a cooperative-game interpretation. COPA uses a coach-player hierarchy to adapt strategy and communication to changing team composition. MOHITO formalizes task-open Markov games, and PLATO jointly handles agent and task openness with a pointer actor and a variable agent-task graph critic. These studies establish variable-set processing and openness generalization as important baselines. Our setting differs by allowing two intelligent teams to choose competing upper-level allocations.

## Hierarchical multi-agent allocation

ALMA decomposes composite cooperative tasks into high-level agent-to-subtask allocation and low-level task execution, improving reuse and generalization in SaveTheCity and multi-army StarCraft. MAAAL and DyHG-3A extend allocation-action learning with behavior-tree structure and dynamic heterogeneous graphs. A separate assignment literature learns long-horizon assignment values and delegates feasibility to combinatorial solvers, as in REDA. Our work retains ALMA's decomposition but freezes a common executor to isolate strategic allocation and adds an opposing allocator rather than a passive task process.

## Dynamic grouping and population scalability

Role-based and grouping-based MARL methods, including ROMA, RODE, hierarchical mean-field methods, hypergraph grouping, and GIRA, learn interaction structures that adapt to large or changing populations. This literature makes it difficult to justify novelty from a new grouping encoder alone. We therefore use a standard set-conditioned architecture and locate the contribution in adversarial training and evaluation across opponent policies.

## Dynamic interception and reallocation

DT-GAT-MARL combines a dynamic-topology graph attention allocator with MAPPO maneuver policies and a dual-frequency hierarchy for dynamic counter-drone interception. It handles targets that appear or disappear and reports reallocation effectiveness and oscillation. Conditional reallocation and switching costs also appear in adaptive task-allocation and distributed-allocation work. These mechanisms motivate our baselines; our central departure is that both sides possess intelligent upper-level policies and both active populations can change.

## Team competition and opponent robustness

Actor-critic methods such as MADDPG and MAAC support mixed cooperative-competitive games. RAC augments actor-critic learning with own-role encoding and opponent-role prediction in team-competitive Markov games. In contrast, we do not create an opponent-identification objective. We train the allocation policy directly against a distribution of opposing allocators and evaluate it through cross-play and budgeted best responses. This tests strategic robustness at the decision layer where grouping occurs.
