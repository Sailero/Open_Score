"""Two bounded-cost public-belief planners; no QOM training or hidden-action access."""
import time
import numpy as np
from .belief import OpponentBelief
from .domain import IdentityUpperAction, action_from_counts, match, prune, quotas, red_candidates, seed_for
from .geometry import geometry_for
from .policies import RiskProxy, CLOSED_TYPES, simple_blue_actions, sample_blue
from .world import branch
from .planner import history_state, public_potential

METHOD_NAMES = {
    "balanced": "均分防守",
    "legacy": "旧版分组（快速预算）",
    "belief_short": "方案一：轨迹推断＋短期验证",
    "belief_recourse": "方案二：轨迹推断＋二次调整",
    "known_short": "知情参照一：已知规则＋短期验证",
    "known_recourse": "知情参照二：已知规则＋二次调整",
}
METHODS = tuple(METHOD_NAMES)


class TemporalRisk:
    """Discounted Blue outcome-time mass, used only as a candidate/leaf heuristic."""
    def __init__(self, state, predictor):
        self.state, self.predictor = state, predictor
        self.base = RiskProxy(state, predictor)
        self.cache = {}
        # Continuous preference over all bins avoids all-zero five-step rankings.
        self.discount = np.exp(-np.arange(1, 11, dtype=float)*5/15)

    def values(self, reds, blues):
        pairings = [[match(self.state, r, b) for b in blues] for r in reds]
        missing = sorted({p for row in pairings for group in row for p in group
                          if p[1] and p[2] and p not in self.cache})
        if missing:
            states = [self.base.canonicalizer.to_entity_set(
                self.base.adapter.local_state_entities(t, r, b, local_step=self.state.step))
                for t, r, b in missing]
            probabilities = self.predictor.predict_joint_outcome_time(
                states, styles=[self.state.lower]*len(states),
                rosters=[(len(r), len(b)) for _, r, b in missing])
            self.cache.update((p, float(q[1] @ self.discount)) for p, q in zip(missing, probabilities))
        geo = geometry_for(self.state)
        values = np.zeros((len(reds), len(blues)))
        for i, row in enumerate(pairings):
            for j, pairs in enumerate(row):
                target_log = {t: 0.0 for t in geo.targets}
                for target, red, blue in pairs:
                    if not blue:
                        continue
                    if red:
                        risk = self.cache[target, red, blue]
                    else:
                        arrival = min(geo.eta["blue", k, target] for k in blue)
                        risk = .98*np.exp(-arrival/15) if arrival <= self.state.max_steps-self.state.step else 0.0
                    target_log[target] += float(np.log(max(1-float(risk), 1e-6)))
                values[i, j] = sum(target_log.values())/len(target_log)
        return values


def candidates(state, previous=None):
    """Shared public domain: generic deployments plus coherent attack responses."""
    pool = list(red_candidates(state, previous, limit=24))
    geo = geometry_for(state)
    hypotheses = tuple(dict.fromkeys(geo.canonical(a) for theta in CLOSED_TYPES[:3]
                                     for a in simple_blue_actions(state, theta)[0]))
    for blue in hypotheses:
        counts = blue.counts(geo.targets)
        weights = np.asarray(counts, float)+.05
        for size in (2, 3, 4):
            red = action_from_counts(state, "red", quotas(len(state.ids("red")), weights), group_size=size)
            intents = tuple((r, b) for _, r, b in match(state, red, blue) if r and b)
            for candidate in (red, IdentityUpperAction(red.groups, red.reserve_ids, intents)):
                if candidate not in pool:
                    pool.append(candidate)
    # Distinct member allocations are retained; no hidden current action enters.
    return tuple(pool)


def posterior_scores(risk, actions, belief):
    matrix = risk.values(actions, belief.actions)
    expected = matrix @ belief.weights
    # A small ambiguity penalty vanishes for a known type.
    ambiguity = 1-float(belief.posterior.max())
    return expected+.05*ambiguity*(matrix.min(axis=1)-expected)


def continuation_candidates(state, previous, belief):
    """Small public-belief recourse domain, shared by the second-step oracle."""
    pool = [prune(previous, state.ids("red"), state.ids("blue"))]
    counts = belief.current_targets(state).sum(axis=0)+.05
    for size in (4, 3, 2):
        pool.append(action_from_counts(state, "red", quotas(len(state.ids("red")), counts), group_size=size))
    for i in np.argsort(-belief.weights)[:2]:
        counts = np.asarray(belief.actions[i].counts([t.id for t in state.targets]), float)+.05
        for size in (4, 2):
            pool.append(action_from_counts(state, "red", quotas(len(state.ids("red")), counts), group_size=size))
    return tuple(dict.fromkeys(pool))


def surviving_belief(belief, state):
    result = belief.copy()
    result.actions = [prune(a, state.ids("blue")) for a in result.actions]
    return result


def leaf_score(state, outcome, risk, action, belief):
    if outcome:
        return float(outcome)
    score = float(posterior_scores(risk, [action], belief)[0])
    # This remains an explicit proxy; it is not a trained terminal win predictor.
    return .6*public_potential(state, 0)+.4*float(np.exp(np.clip(score, -20, 0)))


class FastCommander:
    def __init__(self, predictor, stage1, device, config):
        self.predictor, self.stage1, self.device, self.config = predictor, stage1, device, config

    def new_belief(self, method, known_type=None):
        return OpponentBelief(known_type=known_type if method.startswith("known_") else None)

    def legacy(self, state, rng):
        from open_score.stage3.identity_runtime import build_identity_event_game
        from open_score.stage3.identity_blotto import solve_identity_double_oracle
        from .world import from_public
        built = build_identity_event_game(
            from_public(state), self.predictor, blue_style=state.lower, max_group_size=4,
            full_domain_agent_threshold=8, utility_mode="joint_survival_log_probability", risk_epsilon=1e-6)
        result = solve_identity_double_oracle(
            built.game, built.initial_red, built.initial_blue, tolerance=.01,
            max_iterations=self.config["legacy_iterations"],
            oracle_time_limit_seconds=self.config["legacy_oracle_seconds"],
            oracle_mip_relative_gap=.01)
        action, _ = result.sample_profile(rng=rng)
        return IdentityUpperAction(tuple((slot.target_id, ids) for slot, ids in
                                   zip(built.slots, action.coalitions) if ids), action.reserve_ids)

    def plan(self, state, method, belief, rng, previous=None):
        if method not in METHODS:
            raise ValueError(method)
        started = time.perf_counter()
        deadline = started+self.config["decision_seconds"]
        if not state.ids("red"):
            if method not in {"balanced", "legacy"}:
                belief.prepare(state, RiskProxy(state, self.predictor), rng)
            return IdentityUpperAction(()), {"planning_seconds":time.perf_counter()-started,
                                            "branch_steps":0, "completed_scenarios":0}
        if method == "balanced":
            chosen = action_from_counts(state, "red", quotas(len(state.ids("red")), np.ones(len(state.targets))))
            return chosen, {"planning_seconds":time.perf_counter()-started,"branch_steps":0,"completed_scenarios":0}
        if method == "legacy":
            chosen = self.legacy(state, rng)
            return chosen, {"planning_seconds":time.perf_counter()-started,"branch_steps":0,"completed_scenarios":0}
        risk = TemporalRisk(state, self.predictor)
        belief.prepare(state, risk.base, rng)
        pool = candidates(state, previous)
        prior_scores = posterior_scores(risk, pool, belief)
        indices = sorted(range(len(pool)), key=lambda k:(-prior_scores[k], k))[:self.config["candidates"]]
        actions = tuple(pool[k] for k in indices)
        incumbent = actions[0]
        totals = np.zeros(len(actions))
        completed, steps, second_decisions = 0, 0, 0
        recourse = method.endswith("recourse")
        scenarios = [belief.sample(rng, state) for _ in range(self.config["opponent_samples"])]
        seeds = rng.integers(0, 2**32, len(scenarios), dtype=np.uint64)
        # A complete common-randomness scenario is the atomic scoring unit.
        # Partial candidate evaluations cannot bias selection toward early entries.
        for j, (theta, blue) in enumerate(scenarios):
            trial = []
            for red in actions:
                if time.perf_counter() >= deadline:
                    break
                local = belief.copy()
                nxt, outcome, trace = branch(state, red, blue, self.stage1, self.device, int(seeds[j]))
                steps += len(trace)-1
                for before, after in zip(trace, trace[1:]):
                    local.update(before, after)
                nxt = history_state(nxt, red, nxt.step-state.step)
                if outcome:
                    trial.append(float(outcome))
                    continue
                if recourse:
                    if time.perf_counter() >= deadline:
                        break
                    next_risk = TemporalRisk(nxt, self.predictor)
                    local.prepare(nxt, next_risk.base, np.random.default_rng(seed_for(int(seeds[j]), "future-support")))
                    future_actions = continuation_candidates(nxt, red, local)
                    future_values = posterior_scores(next_risk, future_actions, local)
                    future_red = future_actions[int(np.argmax(future_values))]
                    # theta drives only the simulated Blue; the Red selector above
                    # receives public state and posterior, never this latent truth.
                    future_blue = sample_blue(nxt, theta,
                        np.random.default_rng(seed_for(int(seeds[j]), "future-blue")), next_risk.base)
                    end, outcome, trace = branch(nxt, future_red, future_blue, self.stage1, self.device,
                                                 seed_for(int(seeds[j]), "second-world"))
                    steps += len(trace)-1
                    second_decisions += 1
                    for before, after in zip(trace, trace[1:]):
                        local.update(before, after)
                    end = history_state(end, future_red, end.step-nxt.step)
                    if outcome:
                        trial.append(float(outcome))
                    else:
                        end_risk = TemporalRisk(end, self.predictor)
                        # Leaf keeps the executed command; long-term local risk
                        # complements target health, without 50-step tail rollouts.
                        leaf_action = prune(future_red, end.ids("red"), end.ids("blue"))
                        trial.append(leaf_score(end, 0, end_risk, leaf_action, surviving_belief(local, end)))
                else:
                    next_risk = TemporalRisk(nxt, self.predictor)
                    leaf_action = prune(red, nxt.ids("red"), nxt.ids("blue"))
                    trial.append(leaf_score(nxt, 0, next_risk, leaf_action, surviving_belief(local, nxt)))
            if len(trial) != len(actions):
                break
            totals += trial
            completed += 1
            best = max(range(len(actions)), key=lambda k:(totals[k]/completed, prior_scores[indices[k]], -k))
            incumbent = actions[best]
        incumbent.validate(state.ids("red"), [t.id for t in state.targets], blue_ids=state.ids("blue"))
        elapsed = time.perf_counter()-started
        return incumbent, {
            "planning_seconds":elapsed, "branch_steps":steps, "candidate_pool":len(pool),
            "candidate_count":len(actions), "completed_scenarios":completed,
            "second_decisions":second_decisions, "budget_exceeded":elapsed>self.config["decision_seconds"],
            "analytic_fallback":completed==0, "posterior":belief.posterior.tolist(),
            "budget_semantics":"soft deadline checked at simulation boundaries; elapsed overruns logged",
        }
