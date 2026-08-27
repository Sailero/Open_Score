# Working title and abstract

## Title

**Who Should Fight Whom When Teams Change? Robust Upper-Level Allocation in Open-Population Team Games**

Alternative:

**Robust Hierarchical Coordination for Open-Population Team Competition**

## Abstract draft

Hierarchical multi-agent reinforcement learning can coordinate composite tasks by assigning agents to local objectives and delegating execution to low-level policies. Existing allocation methods, however, are predominantly evaluated in cooperative settings or against passive and fixed-behavior targets, even when agents or tasks may enter and leave. This leaves an important failure mode untested: an intelligent opposing team can adapt its own grouping while combat losses and reinforcements alter both teams within an episode. We formulate this setting as an open-population team-competitive Markov game and isolate the upper-level allocation problem by sharing frozen low-level skills across all methods. We then introduce an adversarial allocation training scheme that samples opposing allocators from a history pool, emphasizing matchups under which the current allocator performs poorly. The allocator itself is set-conditioned and therefore remains applicable as the active ally, opponent, and objective sets change; architectural size invariance is treated as a prerequisite rather than the main contribution. We will evaluate the method against heuristic, ALMA-style, PLATO-style, dynamic-graph, and latest-opponent self-play allocators under held-out force ratios, reinforcement schedules, and counter-allocation strategies. The central claim will be supported only if the proposed training improves end-to-end worst-opponent performance and reduces empirical best-response gain without materially degrading in-distribution performance.

> Status: research-plan abstract. It deliberately contains no fabricated numerical result and must be rewritten after the main table is frozen.

## One-sentence contribution

We turn dynamic grouping from a cooperative intermediate decision into an adversarial upper-level game and show—subject to the planned experiments—that training against a history of counter-allocators improves end-to-end robustness to unseen population changes and opponents under a controlled common executor.
