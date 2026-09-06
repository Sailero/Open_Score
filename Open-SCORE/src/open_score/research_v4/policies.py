"""Blotto-style controls and full-partition continuation planning."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from open_score.grouping.domain import Group, Grouping
from .actions import (candidate_pool, count_allocations, current_grouping, decode_counts,
                      grand_grouping, rule_grouping, search)

CONTROLS = ('rule', 'grand', 'static_rule', 'random', 'b2_count', 'b2_min',
            'b3_rebuild', 'group_only', 'target_only')


class RulePolicy:
    def __init__(self, name='rule', seed=0):
        self.name, self.seed = name, int(seed)
        self.reset()

    def reset(self):
        self.initial = None
        self.rng = np.random.default_rng(self.seed)

    def act(self, state):
        if self.name == 'grand':
            return grand_grouping(state)
        if self.name == 'random':
            pool = candidate_pool(state, 32, self.rng)
            return pool[int(self.rng.integers(len(pool)))]
        if self.name == 'static_rule':
            if self.initial is None or state.step == 0:
                self.initial = rule_grouping(state)
            return self.initial.prune(state.ids('red'))
        return rule_grouping(state)


class CountPolicy:
    """Exact enumeration of the small count domain, fixed identity decoding.

    Scores need not be exact physical payoffs. 'Exact' describes enumeration,
    never equilibrium or an optimality guarantee for the original simulator.
    """
    def __init__(self, scorer, *, batch_size=128):
        self.scorer, self.batch_size = scorer, int(batch_size)
        self.last_trace = {}

    def act(self, state):
        actions = [decode_counts(state, counts) for counts in
                   count_allocations(len(state.ids('red')), len(state.ids('targets')))]
        values = []
        for start in range(0, len(actions), self.batch_size):
            values.extend(self.scorer(state, actions[start:start+self.batch_size]))
        values = np.asarray(values, float)
        if values.shape != (len(actions),) or not np.isfinite(values).all():
            raise ValueError('count scorer returned invalid values')
        best = int(np.argmax(values))
        self.last_trace = dict(evaluations=len(actions), best_value=float(values[best]))
        return actions[best]


class SearchPolicy:
    def __init__(self, scorer, budget=64, *, rebuild=False, mode='full'):
        self.scorer, self.budget, self.rebuild, self.mode = scorer, int(budget), bool(rebuild), mode
        self.last_trace = {}

    def act(self, state):
        start = current_grouping(state)
        if self.rebuild:
            # Each reconstruction starts from singleton rule assignments,
            # independent of old group structure, with identical search budget.
            assignment = grand_grouping(state).assignment()
            start = Grouping(tuple(Group(t, (i,)) for i, t in assignment.items() if t is not None),
                             tuple(i for i, t in assignment.items() if t is None))
        elif self.mode != 'full' and state.step == 0:
            start = rule_grouping(state)
        action, self.last_trace = search(state, self.scorer, self.budget, start=start,
                                         mode=self.mode, return_trace=True)
        return action


def _checkpoint(assets, kind):
    if isinstance(assets, dict):
        for key in (f'{kind}_checkpoint', kind):
            if key in assets:
                return Path(assets[key])
        raise ValueError(f'Missing {kind}_checkpoint in evaluator assets')
    root = Path(assets)
    if root.is_file():
        return root
    candidates = (root/f'{kind}.pt', root/kind/'best.pt', root/kind/'latest.pt',
                  root/f'{kind}_model'/'best.pt', root/f'{kind}_model'/'latest.pt',
                  root/'models'/f'{kind}.pt')
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f'No {kind} evaluator under {root}')


def make_policy(route, assets=None, device='cpu', search_budget=64, seed=0, **opts):
    if route in ('rule', 'grand', 'static_rule', 'random'):
        return RulePolicy(route, seed)
    if route in ('r1_ppo', 'r2_teacher_ppo', 'r3_ddqn'):
        from .training import load_policy
        path = Path(assets)
        return load_policy(path if path.is_file() else path/'best.pt', device=device)
    from .outcomes import load_evaluator
    kind = ('count' if route == 'b1_counts' else 'local' if route in
            ('b2_local', 'b2_count', 'b2_min') else 'global')
    scorer = load_evaluator(_checkpoint(assets, kind), device=device)
    if route == 'b2_min':
        scorer.aggregation = 'min'
    if route in ('b1_counts', 'b2_count'):
        return CountPolicy(scorer)
    if route in ('b2_local', 'b2_min', 'b3_global', 'b3_rebuild', 'group_only', 'target_only'):
        mode = route if route in ('group_only', 'target_only') else 'full'
        return SearchPolicy(scorer, search_budget, rebuild=route == 'b3_rebuild', mode=mode)
    raise ValueError(f'Unknown v4 policy: {route}')
