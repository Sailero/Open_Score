"""Reuse exact public geometry; no hidden assignments or simulator state."""
from functools import lru_cache
import numpy as np
from .domain import IdentityUpperAction, eta


class PublicGeometry:
    def __init__(self, state):
        self.state = state
        self.ids = {side: state.ids(side) for side in ('red', 'blue')}
        self.targets = tuple(t.id for t in state.targets)
        self.lookup = {side: {e.id: e for e in getattr(state, side)} for side in self.ids}
        self.eta = {(side, e.id, t.id): eta(e, t)
                    for side, entities in self.lookup.items()
                    for e in entities.values() for t in state.targets}
        self.groups, self.blue_actions, self.pairings = {}, {}, {}
        self.validated_red = set()

    def group(self, side, target, ids):
        key = (side, target, ids)
        if key not in self.groups:
            arrivals = [self.eta[side, i, target] for i in ids]
            center = np.mean([self.lookup[side][i].position for i in ids], axis=0)
            self.groups[key] = (sum(arrivals)/len(ids), float(np.mean(arrivals)), center)
        return self.groups[key]

    def canonical(self, action):
        if action not in self.blue_actions:
            action.validate(self.ids['blue'], self.targets, 'Blue')
            result = action
            if self.state.lower == 'rush':
                assignment = action.assignment()
                groups = []
                for target in self.targets:
                    ids = sorted((i for i, t in assignment.items() if t == target),
                                 key=lambda i: (self.eta['blue', i, target], i))
                    groups.extend((target, tuple(ids[k:k+4])) for k in range(0, len(ids), 4))
                result = IdentityUpperAction(tuple(groups))
            self.blue_actions[action] = result
        return self.blue_actions[action]

    def match(self, red, blue):
        if red not in self.validated_red:
            red.validate(self.ids['red'], self.targets, blue_ids=self.ids['blue'])
            self.validated_red.add(red)
        blue = self.canonical(blue)
        key = (red, blue)
        if key in self.pairings:
            return self.pairings[key]
        pairs = []
        intents = {r: set(b) for r, b in red.intercepts}
        for target in self.targets:
            def order(action, side):
                def group_key(ids):
                    arrival, _, center = self.group(side, target, ids)
                    return arrival, tuple(center), ids
                return sorted((ids for t, ids in action.groups if t == target), key=group_key)
            reds, blues = order(red, 'red'), order(blue, 'blue')
            edges = []
            for ri, r in enumerate(reds):
                _, ar, cr = self.group('red', target, r)
                for bi, b in enumerate(blues):
                    _, ab, cb = self.group('blue', target, b)
                    overlap = len(intents.get(r, set()).intersection(b))
                    edges.append((-overlap, abs(ar-ab), float(np.linalg.norm(cr-cb)), r, b, ri, bi))
            used_r, used_b = set(), set()
            for *_, ri, bi in sorted(edges):
                if ri not in used_r and bi not in used_b:
                    pairs.append((target, reds[ri], blues[bi]))
                    used_r.add(ri); used_b.add(bi)
            pairs.extend((target, r, ()) for i, r in enumerate(reds) if i not in used_r)
            pairs.extend((target, (), b) for i, b in enumerate(blues) if i not in used_b)
        self.pairings[key] = tuple(pairs)
        return self.pairings[key]


@lru_cache(maxsize=12)
def geometry_for(state):
    return PublicGeometry(state)
