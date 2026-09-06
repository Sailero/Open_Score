"""Exact-simulator rollout and stochastic double progressive widening.

Only planning randomness is consumed. The original environment (including its
unobserved future random state) is restored even when a simulation raises.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import time
from typing import Any

import numpy as np

from open_score.grouping.domain import Group, Grouping
from open_score.research_v4.actions import (
    candidate_pool, current_grouping, grand_grouping, partition_key, rule_grouping,
)


def stable_seed(*parts):
    encoded = json.dumps(parts, sort_keys=True, separators=(',', ':'), default=str).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], 'little') % (2**32 - 1)


def state_key(state):
    value = state.to_dict() if hasattr(state, 'to_dict') else state
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def singleton_grouping(state):
    base = rule_grouping(state)
    return Grouping(tuple(Group(g.target, (i,)) for g in base.groups for i in g.members), base.reserve)


def paired_grouping(state):
    """Same rule identity/target assignment, spatial nearest-neighbour pairs."""
    base = grand_grouping(state)
    positions = {e.id: np.asarray(e.position) for e in state.alive('red')}
    groups = []
    for original in base.groups:
        remaining = set(original.members)
        while remaining:
            first = min(remaining)
            remaining.remove(first)
            members = [first]
            if remaining:
                second = min(remaining, key=lambda i: (np.linalg.norm(positions[first] - positions[i]), i))
                remaining.remove(second)
                members.append(second)
            groups.append(Group(original.target, tuple(members)))
    return Grouping(tuple(groups), base.reserve)


def propose_plans(state, budget=8, seed=0, b1=None):
    """Public deterministic pool, including historical B1 when available."""
    if int(budget) < 1:
        raise ValueError('candidate budget must be positive')
    roots = [current_grouping(state), rule_grouping(state), grand_grouping(state)]
    if b1 is not None:
        roots.append(b1.act(state) if hasattr(b1, 'act') else b1(state))
    pool, seen = [], set()
    rng = np.random.default_rng(stable_seed('proposal', seed, state_key(state)))
    for plan in roots + candidate_pool(state, max(int(budget) * 2, 8), rng):
        key = partition_key(plan)
        if key not in seen:
            plan.validate(state.ids('red'), state.ids('targets'), max_members=None)
            seen.add(key)
            pool.append(plan)
        if len(pool) == int(budget):
            break
    return pool


def choose_with_rule_ties(state, plans, values):
    if not plans or len(plans) != len(values) or not np.isfinite(values).all():
        raise ValueError('one finite value per nonempty candidate is required')
    rule, current = partition_key(rule_grouping(state)), partition_key(current_grouping(state))
    def key(index):
        plan = partition_key(plans[index])
        priority = 2 if plan == rule else 1 if plan == current else 0
        return float(values[index]), priority, -index
    return max(range(len(plans)), key=key)


def terminal_rollouts(env, plans, branch_seeds, continuation=None):
    """Execute each plan for one event, then frozen continuation to terminal."""
    if env.done or not plans or not branch_seeds:
        raise ValueError('nonterminal environment, candidates and branches required')
    seeds = [int(x) for x in branch_seeds]
    if len(set(seeds)) != len(seeds):
        raise ValueError('branch seeds must be distinct')
    continuation = continuation or rule_grouping
    initial = env.snapshot()
    start = env.state().step
    rows = []
    try:
        for candidate_index, plan in enumerate(plans):
            outcomes, branches = [], []
            for seed in seeds:
                env.restore(initial)
                env.set_rng(seed)
                state, _, done, info = env.step(plan)
                physical_steps = int(info['delta'])
                events = 1
                while not done:
                    following = continuation.act(state) if hasattr(continuation, 'act') else continuation(state)
                    state, _, done, info = env.step(following)
                    physical_steps += int(info['delta'])
                    events += 1
                success = int(info['success'])
                outcomes.append(success)
                branches.append(dict(seed=seed, success=success, physical_steps=physical_steps,
                                     remaining_steps=int(state.step-start), events=events,
                                     terminal_cause=info['event_reason']))
            rows.append(dict(candidate_index=candidate_index, action=plan.to_dict(),
                             value=float(np.mean(outcomes)), outcomes=outcomes, branches=branches))
    finally:
        env.restore(initial)
    return rows


class RolloutPolicy:
    def __init__(self, seed=0, candidates=8, branches=8, b1=None, continuation=None):
        self.seed, self.candidates, self.branches = int(seed), int(candidates), int(branches)
        if min(self.candidates, self.branches) < 1:
            raise ValueError('positive rollout budgets required')
        self.b1, self.continuation = b1, continuation
        self.last_trace = {}

    def reset(self):
        self.last_trace = {}

    def act(self, state):
        raise TypeError('RolloutPolicy requires act_env(env) and a real simulator snapshot')

    def act_env(self, env):
        before = time.perf_counter()
        state = env.state()
        plans = propose_plans(state, self.candidates, self.seed, self.b1)
        seeds = [stable_seed('rollout-selection', self.seed, state_key(state), b) for b in range(self.branches)]
        rows = terminal_rollouts(env, plans, seeds, self.continuation)
        selected = choose_with_rule_ties(state, plans, [row['value'] for row in rows])
        work = sum(branch['physical_steps'] for row in rows for branch in row['branches'])
        self.last_trace = dict(algorithm='rollout', candidates=rows, selected_index=selected,
                               evaluations=len(plans), branch_count=self.branches,
                               online_planner_steps=work, simulation_physical_steps=work,
                               decision_seconds=time.perf_counter()-before,
                               all_tie=len({row['value'] for row in rows}) == 1,
                               continuation_version='rule_grouping_v1' if self.continuation is None else 'custom_frozen')
        return plans[selected]


@dataclass
class ActionNode:
    action: Any
    visits: int = 0
    value: float = 0.
    # One entry per actual transition draw; repeated identical states retain
    # multiplicity. Tree selection visits never change these sampling weights.
    transitions: list = field(default_factory=list)
    unique: dict = field(default_factory=dict)


@dataclass
class StateNode:
    state: Any
    terminal: bool = False
    visits: int = 0
    actions: list = field(default_factory=list)
    action_keys: set = field(default_factory=set)


class DPWSearch:
    """Finite-budget MCTS-DPW against a generative stochastic transition model.

    model.transition(state, action, seed) -> (state, reward, terminal, steps)
    model.rollout(state, seed) -> (undiscounted future reward, physical_steps)
    model.propose(state, index, rng) -> action; index zero MUST be the baseline.
    model.state_key / action_key / tie_rank define exact identity and root ties.
    """
    def __init__(self, model, *, iterations=64, depth=3, seed=0,
                 k_action=1.5, alpha_action=.5, k_state=1., alpha_state=.5, exploration=1.,
                 physical_budget=None):
        if int(iterations) < 1 or int(depth) < 1:
            raise ValueError('positive MCTS iterations and depth required')
        if k_action <= 0 or k_state <= 0 or not 0 <= alpha_action <= 1 or not 0 <= alpha_state <= 1:
            raise ValueError('invalid progressive widening parameters')
        self.model, self.iterations, self.depth = model, int(iterations), int(depth)
        self.physical_budget = None if physical_budget is None else int(physical_budget)
        if self.physical_budget is not None and self.physical_budget < 1:
            raise ValueError('physical work budget must be positive')
        self.k_action, self.alpha_action = float(k_action), float(alpha_action)
        self.k_state, self.alpha_state, self.exploration = float(k_state), float(alpha_state), float(exploration)
        self.rng = np.random.default_rng(seed)
        self.physical_steps, self.transitions_generated, self.leaf_rollouts = 0, 0, 0
        self.depths, self.draws = [], []

    def _seed(self):
        return int(self.rng.integers(0, 2**32-1))

    def _leaf(self, node, depth):
        self.depths.append(depth)
        if node.terminal:
            return 0.
        seed = self._seed()
        value, steps = self.model.rollout(node.state, seed)
        self.physical_steps += int(steps)
        self.leaf_rollouts += 1
        self.draws.append(dict(kind='leaf', seed=seed, depth=depth, physical_steps=int(steps), value=float(value)))
        return float(value)

    def _simulate(self, node, remaining):
        if node.terminal:
            return 0.
        if remaining == 0:
            return self._leaf(node, self.depth)
        if len(node.actions) <= self.k_action * node.visits**self.alpha_action:
            # Finite/degenerate action spaces can propose duplicates; bound the
            # proposal attempts without silently adding duplicate action nodes.
            for attempt in range(32):
                action = self.model.propose(node.state, len(node.actions)+attempt, self.rng)
                key = self.model.action_key(action)
                if key not in node.action_keys:
                    node.action_keys.add(key)
                    node.actions.append(ActionNode(action))
                    break
        if not node.actions:
            raise ValueError('proposal generator returned no legal action')
        def ucb(pair):
            i, action = pair
            score = math.inf if action.visits == 0 else action.value + self.exploration * math.sqrt(math.log(max(1, node.visits))/action.visits)
            return score, -i
        selected_index, edge = max(enumerate(node.actions), key=ucb)
        new = False
        if not edge.transitions or len(edge.unique) <= self.k_state * edge.visits**self.alpha_state:
            seed = self._seed()
            following, reward, terminal, steps = self.model.transition(node.state, edge.action, seed)
            self.physical_steps += int(steps)
            self.transitions_generated += 1
            key = self.model.state_key(following)
            if key not in edge.unique:
                edge.unique[key] = StateNode(following, bool(terminal))
                new = True
            successor = edge.unique[key]
            edge.transitions.append((successor, float(reward)))
            self.draws.append(dict(kind='transition', seed=seed, depth=self.depth-remaining+1,
                                  physical_steps=int(steps), reward=float(reward), terminal=bool(terminal),
                                  action_index=selected_index, successor_key=str(key)))
        else:
            successor, reward = edge.transitions[int(self.rng.integers(len(edge.transitions)))]
        reached_depth = self.depth - remaining + 1
        future = self._leaf(successor, reached_depth) if new else self._simulate(successor, remaining-1)
        value = float(reward) + future
        node.visits += 1
        edge.visits += 1
        edge.value += (value-edge.value)/edge.visits
        return value

    def search(self, state):
        root = StateNode(state)
        iterations_done = 0
        for _ in range(self.iterations):
            self._simulate(root, self.depth)
            iterations_done += 1
            # Cost comparison stops between full simulations. A leaf always
            # reaches a native terminal; never truncate its label to fit cost.
            if self.physical_budget is not None and self.physical_steps >= self.physical_budget:
                break
        index, best = max(enumerate(root.actions), key=lambda p: (p[1].value, self.model.tie_rank(state, p[1].action), -p[0]))
        return best.action, dict(iterations=iterations_done, iteration_limit=self.iterations, depth_limit=self.depth,
            physical_budget=self.physical_budget,
            physical_budget_excess=max(0, self.physical_steps-self.physical_budget) if self.physical_budget is not None else 0,
            actual_max_depth=max(self.depths, default=0), actual_mean_depth=float(np.mean(self.depths)) if self.depths else 0.,
            root_actions=[dict(action=self.model.serialize_action(edge.action), visits=edge.visits, value=edge.value,
                               successors=len(edge.unique), transition_draws=len(edge.transitions)) for edge in root.actions],
            selected_index=index, simulation_physical_steps=self.physical_steps,
            online_planner_steps=self.physical_steps, transitions_generated=self.transitions_generated,
            leaf_rollouts=self.leaf_rollouts, simulation_draws=self.draws,
            all_tie=len({edge.value for edge in root.actions}) == 1)


@dataclass
class PhysicalState:
    snapshot: dict
    observation: Any
    terminal: bool = False


class PhysicalPlanningModel:
    def __init__(self, env, proposal_seed=0, b1=None):
        self.env, self.proposal_seed, self.b1 = env, int(proposal_seed), b1
        self._pools = {}

    @staticmethod
    def state_key(state):
        return state_key(state.observation)

    @staticmethod
    def action_key(action):
        return partition_key(action)

    @staticmethod
    def serialize_action(action):
        return action.to_dict()

    @staticmethod
    def tie_rank(state, action):
        return 2 if action == rule_grouping(state.observation) else 1 if action == current_grouping(state.observation) else 0

    def propose(self, state, index, rng):
        key = self.state_key(state)
        if key not in self._pools:
            proposals = propose_plans(state.observation, 128, self.proposal_seed, self.b1)
            rule = rule_grouping(state.observation)
            self._pools[key] = [rule] + [p for p in proposals if p != rule]
        pool = self._pools[key]
        return pool[index] if index < len(pool) else pool[int(rng.integers(len(pool)))]

    def transition(self, state, action, seed):
        self.env.restore(state.snapshot)
        # Each unobserved transition draws a new independent planning stream.
        # Reusing a successor does not reuse that snapshot's random future.
        self.env.set_rng(seed)
        following, reward, terminal, info = self.env.step(action)
        return PhysicalState(self.env.snapshot(), following, terminal), reward, terminal, int(info['delta'])

    def rollout(self, state, seed):
        if state.terminal:
            return 0., 0
        self.env.restore(state.snapshot)
        self.env.set_rng(seed)
        observation, done, steps, result = state.observation, False, 0, 0.
        while not done:
            observation, reward, done, info = self.env.step(rule_grouping(observation))
            steps += int(info['delta'])
            result += float(reward)
        return result, steps


class MCTSPolicy:
    def __init__(self, seed=0, iterations=64, depth=3, b1=None, **parameters):
        self.seed, self.iterations, self.depth = int(seed), int(iterations), int(depth)
        self.b1, self.parameters, self.last_trace = b1, parameters, {}

    def reset(self):
        self.last_trace = {}

    def act(self, state):
        raise TypeError('MCTSPolicy requires act_env(env)')

    def act_env(self, env):
        initial = env.snapshot()
        state = env.state()
        before = time.perf_counter()
        try:
            model = PhysicalPlanningModel(env, self.seed, self.b1)
            planner = DPWSearch(model, iterations=self.iterations, depth=self.depth,
                                seed=stable_seed('mcts', self.seed, state_key(state)), **self.parameters)
            action, self.last_trace = planner.search(PhysicalState(initial, state))
            self.last_trace.update(algorithm='mcts_dpw', decision_seconds=time.perf_counter()-before,
                                   continuation_version='rule_grouping_v1')
            return action
        finally:
            env.restore(initial)
