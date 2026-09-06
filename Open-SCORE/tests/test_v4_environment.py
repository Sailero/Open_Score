"""Physical contracts for the common rule lower and unbounded partition space."""
import copy
import itertools
from dataclasses import replace
import numpy as np
import pytest

from open_score.grouping.domain import Group, Grouping
from open_score.research_v4.environment import make_env, responsibilities, RuleExecutor
from open_score.research_v4.actions import (count_allocations, decode_counts, grand_grouping,
    rule_grouping, candidate_pool, neighbors, partition_key, search)


def test_rule_has_no_model_and_unbounded_group_executes_reproducibly():
    env = make_env(8, opponent='balanced')
    state = env.reset(8123)
    action = Grouping((Group(state.targets[0].id, state.ids('red')),))
    assert not hasattr(env.executor, 'model')
    assert env.executor.memory() == {}
    with pytest.raises(ValueError):
        action.validate(state.ids('red'), state.ids('targets'))
    action.validate(state.ids('red'), state.ids('targets'), max_members=None)
    before = env.snapshot()
    first = env.step(action)
    env.restore(before)
    second = env.step(action)
    assert first[0].to_dict() == second[0].to_dict()
    assert first[1:] == second[1:]
    assert 1 <= first[3]['delta'] <= 5


def test_responsibilities_are_disjoint_and_respect_commanded_targets():
    env = make_env(8)
    state = env.reset(9123)
    grouping = rule_grouping(state)
    pairs = responsibilities(state, grouping)
    blue_ids = [i for _, ids in pairs for i in ids]
    assert len(blue_ids) == len(set(blue_ids))
    assert [g for g, _ in pairs] == list(grouping.groups)
    before = copy.deepcopy(grouping.to_dict())
    actions = env.executor.act(env.adapter, grouping)
    assert grouping.to_dict() == before
    assert set(actions) == set(env.adapter.red_ids)
    assert all(0 <= a < 27 for a in actions.values())


def test_partition_edits_connect_small_complete_domain():
    state = make_env(3).reset(124)
    start = Grouping((), state.ids('red'))
    seen, frontier = {partition_key(start)}, [start]
    while frontier:
        for action in neighbors(state, frontier.pop()):
            action.validate(state.ids('red'), state.ids('targets'), max_members=None)
            key = partition_key(action)
            if key not in seen:
                seen.add(key)
                frontier.append(action)
    # For each subset placed in reserve, remaining are partitioned and each
    # block receives one of two target labels: 1+6+18+22=47.
    assert len(seen) == 47


def test_counts_decode_exactly_and_size32_no_capacity_clipping():
    assert len(list(count_allocations(32, reserve=False))) == 33
    assert len(list(count_allocations(32))) == 561
    state = make_env(32).reset(42)
    action = decode_counts(state, (32, 0, 0))
    assert len(action.groups[0].members) == 32
    assert action.reserve == ()


def test_search_budget_keeps_incumbent_and_reaches_combined_changes():
    state = make_env(4).reset(11)
    initial = rule_grouping(state)
    goal = Grouping((), state.ids('red'))
    scored = []
    def scorer(s, pool):
        scored.extend(pool)
        return [len(g.reserve) for g in pool]
    result, trace = search(state, scorer, budget=100, start=initial, return_trace=True)
    assert result == goal
    assert len(scored) <= 100
    assert trace['evaluations'] == len(scored)
    assert scored[0] == initial
    assert len({partition_key(g) for g in scored}) == len(scored)


def test_group_only_and_target_only_preserve_their_invariants():
    state = make_env(6).reset(22)
    grouping = grand_grouping(state)
    for action in neighbors(state, grouping, mode='group_only'):
        assert action.assignment() == grouping.assignment()
    rosters = sorted(g.members for g in grouping.groups)
    for action in neighbors(state, grouping, mode='target_only'):
        assert sorted(g.members for g in action.groups) == rosters


def test_candidate_pool_is_deterministic_valid_and_not_mutating_rng():
    state = make_env(8).reset(712)
    first, second = candidate_pool(state, 24), candidate_pool(state, 24)
    assert first == second
    assert len(first) == 24
    assert len({partition_key(g) for g in first}) == 24
    for grouping in first:
        grouping.validate(state.ids('red'), state.ids('targets'), max_members=None)


def test_one_target_local_world_and_full_native_terminal():
    env = make_env(4, targets=1)
    state = env.reset(1203)
    while not env.done:
        state, reward, done, info = env.step(grand_grouping(state))
    assert len(state.targets) == 1
    assert 0 < state.step <= 50
    assert reward in (0., 1.)
    assert info['terminated'] and not info['truncated']
