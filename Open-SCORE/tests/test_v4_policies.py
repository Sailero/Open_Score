"""Solver-level fairness and ablation invariants across decisions/episodes."""
from dataclasses import replace
import math

import numpy as np

from open_score.grouping.domain import DecisionState, Entity, Group, Grouping
from open_score.research_v4.actions import partition_key, rule_grouping, search
from open_score.research_v4.environment import make_env
from open_score.research_v4.policies import CountPolicy, SearchPolicy, RulePolicy


def make_state(size=6, offset=0, step=5):
    red = tuple(Entity(offset+i, (-500.+i*170, (-1)**i*600., 100.), (20., 0., 0.), 1.) for i in range(size))
    blue = tuple(Entity(100+i, (1500.-i*90, (-1)**i*700., 100.), (-150., 0., 0.), 1.) for i in range(size))
    targets = (Entity(0, (-2100., -650., 100.), (0., 0., 0.), 1.2),
               Entity(1, (-2100., 650., 100.), (0., 0., 0.), 1.2))
    previous = Grouping((Group(0, tuple(offset+i for i in range(0, size, 2))),
                         Group(1, tuple(offset+i for i in range(1, size, 2)))))
    return DecisionState(step, 50, 'reactive', red, blue, targets, previous)


def test_count_policy_exhaustively_finds_unique_count_optimum_using_fixed_decoder():
    state = make_state(6)
    evaluated = []
    def scorer(current, pool):
        assert current is state
        evaluated.extend(pool)
        values = []
        for action in pool:
            counts = [sum(len(g.members) for g in action.groups if g.target == target) for target in (0, 1)]
            values.append(-((counts[0]-1)**2+(counts[1]-3)**2+(len(action.reserve)-2)**2))
        return values
    first = CountPolicy(scorer, batch_size=3)
    selected = first.act(state)
    assert sum(len(g.members) for g in selected.groups if g.target == 0) == 1
    assert sum(len(g.members) for g in selected.groups if g.target == 1) == 3
    assert len(selected.reserve) == 2
    assert len(evaluated) == math.comb(8, 2)
    assert len({partition_key(g) for g in evaluated}) == len(evaluated)
    assert first.last_trace['evaluations'] == len(evaluated)
    # Scorer batch boundaries must not change identity decoding or the argmax.
    assert CountPolicy(scorer, batch_size=128).act(state) == selected


def test_b2_b3_share_exact_search_mechanics_for_equivalent_rankings():
    state = make_state()
    queries = [[], []]
    def scorer(which):
        def call(current, pool):
            queries[which].extend(partition_key(p) for p in pool)
            scores = [2*len(p.reserve)+sum(g.target*len(g.members) for g in p.groups) for p in pool]
            return scores if which == 0 else [3*x+10 for x in scores]
        return call
    first, trace1 = search(state, scorer(0), budget=24, return_trace=True)
    second, trace2 = search(state, scorer(1), budget=24, return_trace=True)
    assert first == second and queries[0] == queries[1]
    assert trace1['evaluations'] == trace2['evaluations'] == 24


def test_group_only_policy_preserves_each_survivors_original_target_after_casualties():
    state = make_state()
    initial = state.previous.assignment()
    def adversarial_scorer(current, pool):
        # A large temptation to retarget; the ablation must still forbid it.
        return [1000*sum(g.target == 1 for g in p.groups)+len(p.groups) for p in pool]
    policy = SearchPolicy(adversarial_scorer, budget=24, mode='group_only')
    for step in range(5, 30, 5):
        if step == 15:
            state = replace(state, red=tuple(replace(e, health=0.) if e.id == 2 else e for e in state.red))
        action = policy.act(state)
        assert action.assignment() == {i: initial[i] for i in state.ids('red')}
        state = replace(state, step=step+5, previous=action)


def test_target_only_policy_never_reassembles_members_and_reinitializes_next_episode():
    policy = SearchPolicy(lambda current, pool: [sum(g.target*len(g.members) for g in p.groups)
                                               for p in pool], budget=24, mode='target_only')
    for offset in (0, 50):
        state = make_state(offset=offset, step=0)
        initial = rule_grouping(state)
        original_members = [set(g.members) for g in initial.groups]
        for step in range(0, 25, 5):
            if step == 10:
                state = replace(state, red=tuple(replace(e, health=0.) if e.id == offset+2 else e for e in state.red))
            action = policy.act(state)
            expected = sorted(tuple(sorted(ids & set(state.ids('red')))) for ids in original_members
                              if ids & set(state.ids('red')))
            assert sorted(g.members for g in action.groups) == expected
            state = replace(state, step=step+5, previous=action)


def test_static_rule_refreshes_initial_groups_for_a_new_episode_without_reset_call():
    policy = RulePolicy('static_rule')
    first = make_state(offset=0, step=0)
    second = make_state(offset=50, step=0)
    assert policy.act(first) == rule_grouping(first)
    selected = policy.act(second)
    assert selected == rule_grouping(second)
    assert set(selected.assignment()) == set(second.ids('red'))


def test_rule_lower_does_not_read_private_committed_blue_target_assignments():
    env = make_env(8)
    try:
        state = env.reset(817)
        grouping = rule_grouping(state)
        original = env.executor.act(env.adapter, grouping)
        snapshot = env.snapshot()
        for target in (0, 1):
            env.adapter.set_joint_assignments(grouping.assignment(), {i: target for i in env.adapter.blue_ids})
            actual = env.executor.act(env.adapter, grouping)
            assert actual == original
            assert env.previous == snapshot['previous']
    finally:
        env.close()
