# 1 Introduction

Multi-agent teams deployed in defense, interception, and search-and-rescue rarely preserve a fixed composition. Agents may be destroyed, damaged, temporarily unavailable, or reinforced, while objectives may emerge or disappear. A useful team must therefore revise not only how each agent acts but also which teammates should cooperate on which local objective.

Hierarchical coordination offers a natural abstraction. An upper-level controller allocates active agents to local engagements or objectives, and lower-level policies execute attack, interception, or defense skills inside each assigned subproblem. ALMA demonstrated the value of jointly learning these two levels in composite cooperative tasks, and subsequent work expanded allocation-action learning with dynamic graphs and richer task structures. Open-agent methods such as GPL, MOHITO, and PLATO separately established that policies should remain well defined when agents or tasks enter and leave.

Yet dynamic allocation in an attack-defense system is not only an open coordination problem. It is also a team-competitive game. A defending allocator that concentrates resources on one threat exposes another objective; an attacking allocator may deliberately split, feint, or delay reinforcements to exploit that response. Recent DT-GAT-MARL work studies hierarchical dynamic interception, but its intruders follow constant-velocity trajectories and do not learn a counter-allocation strategy. Team-competitive role-learning methods model intelligent opponents, but typically retain fixed populations and do not expose an explicit upper-level allocation game.

This gap matters because evaluation against a single scripted or latest self-play opponent can reward brittle allocation rules. Average in-distribution return may remain high even when a held-out opponent discovers a force split that the allocator consistently mishandles. Agent-count generalization alone does not address the failure: a policy can accept a variable-sized set and still be strategically exploitable.

We study upper-level allocation in an open-population team-competitive Markov game. Both teams' active sets can change within an episode through attrition and reinforcement, both teams choose assignments at the strategic timescale, and a common pretrained low-level executor carries out the assigned objectives. Sharing and freezing the executor in the main study gives a controlled answer to a narrow question: does the upper-level training rule itself produce more robust grouping decisions?

Our working method trains a set-conditioned allocator against a pool of historical opposing allocators, prioritizing matchups in which the current policy performs poorly. Unlike opponent-identification approaches, it introduces no auxiliary opponent label; adversarial information affects the same end-to-end team objective used for evaluation. We assess robustness through held-out opponent payoff matrices, worst-case return, zero-shot force-ratio tests, and a budget-controlled empirical best response, while treating regrouping statistics only as diagnostic evidence.

The intended contributions are:

1. an operational formulation of hierarchical allocation when both competing teams are open within an episode;
2. an empirical diagnosis of the cross-opponent weakness of cooperative or latest-opponent allocation training under a shared executor;
3. a lightweight opponent-pool allocation training scheme evaluated by end-to-end worst-opponent performance and empirical exploitability rather than clustering quality.

These claims are provisional until the preregistered go/no-go tests succeed. In particular, we do not claim to compute a Nash equilibrium or to introduce the first variable-population neural architecture.
