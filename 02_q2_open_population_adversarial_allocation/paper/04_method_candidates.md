# 4 Method

## 4.1 Recommended method: open-population adversarial allocation

The allocator factorizes observations into active ally, opponent, and objective sets. Shared encoders produce permutation-equivariant tokens, and a pointer-style head scores every feasible agent-objective pair. This variable-set interface follows established open-system practice and is not claimed as the primary novelty.

The learning contribution is a historical opponent pool. Let (mathcal P_t) contain prior opposing allocator snapshots and (ar R_j) denote the current allocator's recent return against snapshot (j). We sample opponents according to

\[
p_t(j)=(1-\eta)|\mathcal P_t|^{-1}+\eta\frac{\exp(-\bar R_j/\tau)}{\sum_l\exp(-\bar R_l/\tau)}.
\]

The first term preserves coverage and the second revisits opponents that currently exploit the allocator. The upper policy is optimized only with team return and standard entropy/legality terms. No opponent class, strategy-change label, or role-identification target is introduced.

Snapshots are added at a fixed environment-step interval after a warm-up. Pool size is capped by retaining the newest policies plus policies that add a distinct payoff row in cross-play. The first implementation may use a fixed FIFO pool; diversity pruning is optional and cannot become necessary for the main claim.

## 4.2 Training algorithm sketch

1. Pretrain and freeze the common lower executor.
2. Initialize symmetric red/blue upper allocators and empty history pools.
3. For each rollout batch, sample an opponent from the latest policy or its history pool.
4. Randomize legal population events and collect hierarchical trajectories.
5. Update only the current upper allocator with PPO/AQL under the final team objective.
6. Update matchup returns and periodically checkpoint the upper allocator.
7. Alternate sides or exploit parameter sharing when scenarios are symmetric.

## 4.3 Candidate B: value-triggered minimal reallocation

If the adversarial gap is absent, retain the same allocator but learn a binary gate. Let (Q_{keep}) evaluate continuing the current allocation and (Q_{new}) evaluate the proposed allocation. Reallocation occurs only when

\[
Q_{new}-Q_{keep}>c_{switch}+h,
\]

where (c_{switch}) captures travel, execution interruption, and communication, and (h) is a hysteresis margin. Its claim is a better outcome-cost Pareto frontier, not higher grouping accuracy.

## 4.4 Candidate C: marginal coalition value with constrained matching

If independent pointers produce systematic conflicts, estimate the marginal value (Delta V(i,x\mid C_x)) of adding agent (i) to the current coalition on objective (x). A capacity-constrained assignment solver converts these values into a feasible allocation. This route becomes the main method only if hand-crafted utilities under the same solver are clearly inadequate.

## 4.5 Complexity and scope

With (n) active agents and (m) active objectives/opposing groups, dense pair scoring costs (O(nm d)). History-pool training changes sampling cost but not execution-time architecture. The deployed allocator contains one policy, not the pool. Sparse candidate filtering is deferred unless measured latency violates the preregistered budget.
