"""Persistent policy library and frozen Stage2 candidate screening."""
from __future__ import annotations
import numpy as np
from open_score.stage2 import HADCanonicalizer
from open_score.stage3.blotto import solve_restricted_matrix_game
from .domain import IdentityUpperAction, action_from_counts, canonical_blue, eta, match, quotas, red_candidates
from .world import from_public

CLOSED_TYPES = ("balanced", "concentrated_nearest", "two_front", "strategic_equilibrium")
UPPER_TYPES = CLOSED_TYPES + ("feint_switch",)


class RiskProxy:
    """Candidate heuristic only: independent local risks are not global truth."""
    def __init__(self, state, predictor):
        self.state, self.predictor = state, predictor
        self.adapter = from_public(state, 0)
        self.canonicalizer = HADCanonicalizer()
        self.cache = {}
        if predictor.steps_per_bin != 5 or predictor.model.horizon_bins != 10:
            raise ValueError("Registered frozen Stage2 requires ten five-step bins")

    def values(self, reds, blues, *, final=False):
        pairings = [[match(self.state, r, b) for b in blues] for r in reds]
        missing = sorted({pair for row in pairings for profile in row for pair in profile
                          if pair[1] and pair[2] and pair not in self.cache})
        if missing:
            states = [self.canonicalizer.to_entity_set(self.adapter.local_state_entities(t, r, b, local_step=self.state.step)) for t, r, b in missing]
            p = self.predictor.predict_joint_outcome_time(states, styles=[self.state.lower]*len(states), rosters=[(len(r), len(b)) for _, r, b in missing])
            for pair, probability in zip(missing, p):
                self.cache[pair] = (float(probability[1, 0]), float(probability[1].sum()))
        matrix = np.zeros((len(reds), len(blues)))
        blue_entities = {x.id: x for x in self.state.blue}
        targets = {x.id: x for x in self.state.targets}
        for i, row in enumerate(pairings):
            for j, profile in enumerate(row):
                survival = {x.id: 1.0 for x in self.state.targets}
                for t, r, b in profile:
                    if not b:
                        continue
                    if r:
                        risk = self.cache[(t, r, b)][int(final)]
                    else:
                        # Conservative physical reachability envelope, not an
                        # exact breach probability. No penalty outside horizon.
                        horizon = self.state.max_steps-self.state.step if final else min(5, self.state.max_steps-self.state.step)
                        reachable = any(np.linalg.norm(np.asarray(blue_entities[k].position)-targets[t].position) <= 500.0 + 300.0*horizon for k in b)
                        risk = float(reachable)
                    survival[t] *= 1-np.clip(risk, 0, 1)
                matrix[i, j] = sum(np.log(max(v, 1e-6)) for v in survival.values())/len(survival)
        return matrix


def simple_blue_actions(state, theta):
    n, m = len(state.ids("blue")), len(state.targets)
    blue = state.alive("blue")
    scores = [np.mean([eta(x,t) for x in blue]) if blue else 0 for t in state.targets]
    priority = sorted(range(m), key=lambda k: (scores[k], state.targets[k].id))
    if theta == "feint_switch" and state.step >= 10:
        priority = sorted(range(m), key=lambda k: (state.red_history_counts[k], scores[k], k))
    actions = []
    for variant in range(4):
        weights = np.ones(m)
        if theta in {"concentrated_nearest", "feint_switch"}:
            weights[:] = 0
            # Adjacent alternative has a small, registered stochastic mass.
            weights[priority[0] if variant < 3 else priority[min(1,m-1)]] = 1
        elif theta == "two_front":
            weights[:] = 0
            weights[priority[0]] = .7
            weights[priority[min(1,m-1)]] += .3
        elif theta != "balanced":
            raise ValueError(theta)
        counts = quotas(n, weights)
        action = action_from_counts(state, "blue", counts, group_size=4 if variant < 2 else 3,
                                    rotation=variant % m)
        actions.append(action)
    return tuple(actions), np.full(4, .25)


def blue_distribution(state, theta, proxy):
    if theta != "strategic_equilibrium":
        return simple_blue_actions(state, theta)
    # A restricted identity Blotto equilibrium, computed from public state.
    # It does not see the current Red simultaneous action or its method name.
    blue = tuple(dict.fromkeys(a for t in CLOSED_TYPES[:3] for a in simple_blue_actions(state,t)[0]))
    reds = red_candidates(state, limit=16)
    game = solve_restricted_matrix_game(proxy.values(reds, blue))
    return blue, np.asarray(game.attacker_mixture)


def sample_blue(state, theta, rng, proxy):
    actions, p = blue_distribution(state, theta, proxy)
    return actions[int(rng.choice(len(actions), p=p))]


def shared_candidates(state, proxy, previous=None, limit=8):
    """Same public, belief-independent candidates for all new methods/oracles."""
    base_pool = red_candidates(state, previous, limit=32)
    blue = tuple(dict.fromkeys(a for t in CLOSED_TYPES[:3] for a in simple_blue_actions(state,t)[0]))
    pool=list(base_pool)
    # Public hypothetical Blue partitions generate explicit interception intents.
    # No actual current action, true type, or secret grouping is consulted.
    for index,red in enumerate(base_pool):
        hypothesis=blue[index%len(blue)]
        intent=tuple((r,b) for _,r,b in match(state,red,hypothesis) if r and b)
        candidate=IdentityUpperAction(red.groups,red.reserve_ids,intent)
        if candidate not in pool:
            pool.append(candidate)
    values = proxy.values(pool, blue)
    # Keep balanced, hold/reserve and diverse robust top actions. Eventual S2
    # risk is a tie-breaker only; five-step physical branches decide deployment.
    robust = values.min(axis=1)
    tail = proxy.values(pool, blue, final=True).mean(axis=1)
    ranked = sorted(range(len(pool)), key=lambda i: (-robust[i], -tail[i], i))
    selected = list(dict.fromkeys([0, 1] + ranked))[:limit]
    return tuple(pool[i] for i in selected), blue, values[selected]


def decode_action(state, target_probabilities, affinity, rng):
    """Masked constrained decoding: exactly one target and group per Blue ID."""
    ids = state.ids("blue")
    p = np.asarray(target_probabilities, float)
    if p.shape != (len(ids), len(state.targets)):
        raise ValueError("Target predictions must align with live entities")
    choices = [int(rng.choice(len(state.targets), p=row/row.sum())) for row in p]
    groups = []
    for k, target in enumerate(state.targets):
        left = [i for i in range(len(ids)) if choices[i] == k]
        while left:
            anchor = left.pop(0)
            near = sorted(left, key=lambda j: (-float(affinity[anchor,j]), ids[j]))
            members = [j for j in near if affinity[anchor,j] >= .5][:3]
            groups.append((target.id, tuple(ids[j] for j in [anchor]+members)))
            left = [j for j in left if j not in members]
    action = IdentityUpperAction(tuple(groups))
    return canonical_blue(state, action)
