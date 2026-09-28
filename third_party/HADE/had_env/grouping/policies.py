"""Four small baseline policies; user algorithms plug into the public API."""
import numpy as np

from .actions import candidate_pool, grand_grouping, rule_grouping


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
