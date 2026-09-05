"""Physical and information-contract tests for the known-opponent SMDP."""
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from open_score.grouping.domain import DecisionState, Entity, Group, Grouping
from open_score.grouping.environment import KnownOpponentEnv
from open_score.grouping.opponents import OPPONENTS, distribution


@pytest.fixture
def env():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    instance = KnownOpponentEnv(red=4, blue=4, max_steps=50, seed=19)
    instance.reset()
    yield instance
    instance.close()
    torch.set_num_threads(old_threads)


def balanced(state):
    ids = state.ids("red")
    return Grouping((Group(0, ids[:2]), Group(1, ids[2:])))


def test_grouping_canonical_cover_prune_and_serialization():
    grouping = Grouping((Group(1, (4, 2)), Group(0, (3, 1))), (6, 5))
    assert grouping == Grouping((Group(0, (1, 3)), Group(1, (2, 4))), (5, 6))
    assert grouping.validate(range(1, 7), [0, 1]) is grouping
    assert Grouping.from_dict(json.loads(json.dumps(grouping.to_dict()))) == grouping
    pruned = grouping.prune([1, 4, 5])
    assert pruned == Grouping((Group(0, (1,)), Group(1, (4,))), (5,))
    pruned.validate([1, 4, 5], [0, 1])
    for invalid in [Grouping((Group(0, (1, 1)),)), Grouping((Group(0, (1,)),), (1,)),
                    Grouping((Group(0, ()),)), Grouping((Group(9, (1,)),)),
                    Grouping((Group(0, (1, 2, 3, 4, 5)),))]:
        with pytest.raises(ValueError):
            invalid.validate([1], [0, 1])


def test_initial_state_complete_public_roster_and_roundtrip(env):
    state = env.state()
    assert state.step == 0 and state.previous == Grouping((), state.ids("red"))
    assert len(state.red) == len(state.blue) == 4 and len(state.targets) == 2
    assert set(state.memory) == set(state.last_actions) == set(state.ids("red"))
    assert all(len(row) == 64 and not any(row) for row in state.memory.values())
    assert set(state.last_actions.values()) == {-1}
    assert DecisionState.from_dict(json.loads(json.dumps(state.to_dict()))) == state
    assert {"blue_grouping", "opponent_rng", "physical"}.isdisjoint(state.to_dict())


def test_known_opponents_share_current_physical_information(env):
    state = env.state()
    changed_previous = replace(state, previous=balanced(state))
    for name in OPPONENTS:
        actions, probabilities = distribution(state, name)
        assert np.all(probabilities >= 0) and probabilities.sum() == pytest.approx(1)
        changed, changed_probabilities = distribution(changed_previous, name)
        assert actions == changed
        np.testing.assert_array_equal(probabilities, changed_probabilities)
        for action in actions:
            assert not action.reserve
            action.validate(state.ids("blue"), state.ids("targets"))
    near_first = tuple(replace(entity, position=state.targets[0].position) for entity in state.red)
    near_second = tuple(replace(entity, position=state.targets[1].position) for entity in state.red)
    _, first = distribution(replace(state, red=near_first), "reactive")
    _, second = distribution(replace(state, red=near_second), "reactive")
    assert first[1] > first[0] and second[0] > second[1]


def test_simultaneous_blue_action_is_independent_of_current_red_choice(env):
    state = env.state()
    snapshot = env.snapshot()
    env.step(balanced(state))
    first = env.blue_grouping
    env.restore(snapshot)
    env.step(Grouping((Group(1, state.ids("red")),)))
    assert env.blue_grouping == first


def test_exact_snapshot_replays_physics_lcl_memory_and_rng(env):
    first_state = env.state()
    action = balanced(first_state)
    env.step(action)
    snapshot = env.snapshot()
    first = env.step(action.prune(env.state().ids("red")))
    restored = env.restore(snapshot)
    assert restored.previous == snapshot["previous"]
    second = env.step(action.prune(restored.ids("red")))
    assert first == second


def test_independent_continuation_rng_preserves_state_and_external_rng(env):
    state = env.state()
    np.random.seed(431)
    expected = np.random.random(5)
    np.random.seed(431)
    env.set_rng(8732)
    assert env.state() == state and env.opponent == state.opponent
    env.step(balanced(state))
    np.testing.assert_equal(np.random.random(5), expected)


def test_frozen_lcl_keeps_memory_by_id_and_observes_all_blue(env, monkeypatch):
    state = env.state()
    env.executor.act(env.adapter, balanced(state))
    old_memory = {i: row.copy() for i, row in env.executor.hidden.items()}
    old_last = dict(env.executor.last_actions)
    ids = state.ids("red")
    regrouped = Grouping((Group(0, (ids[0], ids[2])), Group(1, (ids[1], ids[3]))))
    original = env.executor.model.act
    captured = []

    def inspect(observation, hidden, epsilon, last_action):
        captured.append(True)
        for batch, group in enumerate(regrouped.groups):
            for index, i in enumerate(group.members):
                np.testing.assert_array_equal(hidden[batch, index].cpu().numpy(), old_memory[i])
                assert int(last_action[batch, index].argmax()) == old_last[i]
                # Entity relation slot 9 is the opponent flag; every Blue
                # entity is visible regardless of its current target choice.
                mask = observation.entity_mask[batch, index]
                rows = observation.entity_obs[batch, index][mask]
                assert int(rows[:, 9].sum()) == len(state.ids("blue"))
        return original(observation, hidden, epsilon=epsilon, last_action=last_action)

    monkeypatch.setattr(env.executor.model, "act", inspect)
    before = {key: value.clone() for key, value in env.executor.model.state_dict().items()}
    env.executor.act(env.adapter, regrouped)
    assert captured and all(not p.requires_grad for p in env.executor.model.parameters())
    for key, value in env.executor.model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_native_horizon_is_terminal_success_not_batch_truncation():
    env = KnownOpponentEnv(red=1, blue=1, max_steps=1, seed=901)
    state = env.reset()
    after, reward, done, info = env.step(Grouping((Group(0, state.ids("red")),)))
    assert after.step == info["physical_steps"] == info["delta"] == 1
    assert done and reward == 1 and info["success"]
    assert info["terminated"] and not info["truncated"] and info["event_reason"] == "horizon"
    with pytest.raises(RuntimeError):
        env.step(after.previous)


def test_casualties_merge_and_periodic_clock_remains_absolute(env, monkeypatch):
    original = env.adapter.env.step_physics
    calls = [0]

    def physical(actions):
        original(actions)
        calls[0] += 1
        if calls[0] == 2:
            for agent in env.adapter.env.red_agents[:2]:
                agent.Health = 0

    monkeypatch.setattr(env.adapter.env, "step_physics", physical)
    state, reward, done, info = env.step(balanced(env.state()))
    assert not done and reward == 0 and info["delta"] == state.step == 2
    assert info["event_reason"] == "casualty" and len(state.ids("red")) == 2
    casualty_events = [event for event in info["events"] if event["kind"] == "agents_destroyed"]
    assert len(casualty_events) == 1 and len(casualty_events[0]["agent_ids"]) == 2
    assert set(state.memory) == set(state.last_actions) == set(state.ids("red"))
    state.previous.validate(state.ids("red"), state.ids("targets"))
    next_state, _, _, info = env.step(state.previous)
    assert next_state.step == 5 and info["delta"] == 3


def test_reserve_remains_unassigned_and_preserves_recurrent_state(env):
    state = env.state()
    env.executor.act(env.adapter, balanced(state))
    old = env.executor.memory()
    state, _, _, _ = env.step(Grouping((), state.ids("red")))
    assert all(target is None for target in env.adapter.red_assignment.values())
    assert state.memory == old
    assert all(0 <= action < 27 for action in state.last_actions.values())


def test_no_red_survivors_continue_real_blue_clock_until_native_terminal(env, monkeypatch):
    from open_score.grouping import environment as module
    original_physics = env.adapter.env.step_physics
    original_sample = module.sample
    command_steps, physical_calls = [], [0]

    def physical(actions):
        original_physics(actions)
        physical_calls[0] += 1
        if physical_calls[0] == 2:
            for agent in env.adapter.env.red_agents:
                agent.Health = 0

    def sampling(state, opponent, rng):
        command_steps.append(state.step)
        return original_sample(state, opponent, rng)

    monkeypatch.setattr(env.adapter.env, "step_physics", physical)
    monkeypatch.setattr(module, "sample", sampling)
    state, reward, done, info = env.step(balanced(env.state()))
    assert done and info["terminated"] and not info["truncated"]
    assert not state.ids("red") and not state.memory and not state.last_actions
    assert info["delta"] == state.step > 2 and physical_calls[0] == state.step
    assert info["no_red_continuation"] and info["automatic_blue_events"] >= 1
    assert command_steps[:2] == [0, 2]
    assert all(step % 5 == 0 for step in command_steps[2:])
    assert reward == float(info["success"])
