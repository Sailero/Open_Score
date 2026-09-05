"""The overnight observation adaptation cannot alter the physical task."""
import copy

import numpy as np
import pytest
import torch

from open_score.grouping.baselines import balanced_initial
from open_score.grouping.domain import Group, Grouping
from open_score.grouping.environment import KnownOpponentEnv
from open_score.overnight.environment import FocusedExecutor, make_env


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("scope,expected", [("full", [8, 8, 8]),
                                           ("nearest3", [3, 3, 3]),
                                           ("nearest_support", [1, 2, 2]),
                                           ("count_clip3", [8, 8, 8])])
def test_focused_input_retains_world_and_frozen_weights(scope, expected, monkeypatch):
    env = make_env(8, executor_scope=scope, seed=901)
    state = env.reset()
    ids = state.ids("red")
    action = Grouping((Group(0, ids[:2]), Group(0, ids[2:5]), Group(1, ids[5:])))
    original = env.executor.model.act
    observed = []

    def inspect(observation, hidden, epsilon, last_action):
        for batch, group in enumerate(action.groups):
            rows = observation.entity_obs[batch, 0][observation.entity_mask[batch, 0]]
            observed.append(int(rows[:, 9].sum()))
            assert int(rows[:, 8].sum()) == len(group.members)
            expected_count = 3 if scope == "count_clip3" else expected[batch]
            assert float(observation.self_obs[batch, 0, 8]) == pytest.approx(np.log1p(expected_count))
        return original(observation, hidden, epsilon=epsilon, last_action=last_action)

    monkeypatch.setattr(env.executor.model, "act", inspect)
    weights = {key: value.clone() for key, value in env.executor.model.state_dict().items()}
    opponent_rng = copy.deepcopy(env.opponent_rng.bit_generator.state)
    actions = env.executor.act(env.adapter, action)
    assert observed == expected
    assert env.opponent_rng.bit_generator.state == opponent_rng
    assert len(env.adapter.env.blue_agents) == len(env.state().ids("blue")) == 8
    assert env.state().blue == state.blue  # upper policy retains the true roster and positions
    assert set(actions) == set(env.adapter.red_ids)
    assert all(not parameter.requires_grad for parameter in env.executor.model.parameters())
    for key, value in env.executor.model.state_dict().items():
        torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
    # The actual adapter was never patched: independent callers still see all Blue.
    row = env.adapter.local_observation("Red", 0, ids[:2], state.ids("blue"))
    assert int(row["entity_obs"][0, :, 9].sum()) == 8
    assert float(row["self_obs"][0, 8]) == pytest.approx(np.log1p(8))


def test_full_scope_reproduces_original_physical_transition():
    adapted = make_env(8, executor_scope="full", seed=812)
    original = KnownOpponentEnv(8, 8, seed=812)
    state = adapted.reset()
    assert state == original.reset()
    action = balanced_initial(state)
    assert adapted.step(action) == original.step(action)


@pytest.mark.parametrize("scale", [8, 12, 16])
def test_focused_real_transition_preserves_all_blue_and_snapshot(scale):
    env = make_env(scale, executor_scope="nearest3", seed=scale)
    state = env.reset()
    snapshot = env.snapshot()
    action = balanced_initial(state)
    first = env.step(action)
    assert len(first[0].blue) == scale
    assert len(env.adapter.env.blue_agents) == scale
    assert first[3]["delta"] >= 1
    env.restore(snapshot)
    assert env.step(action) == first


def test_filtered_out_blue_still_moves_in_real_world():
    env = make_env(4, blue=12, executor_scope="nearest3", seed=772)
    state = env.reset()
    old = {row.id: np.array(row.position) for row in state.blue}
    after, _, _, _ = env.step(Grouping((Group(0, state.ids("red")),)))
    # Only 3 entities enter the sole group's actor observation; all 12 physical
    # opponents are retained and move under the complete known opponent rule.
    assert len(after.blue) == 12
    assert all(np.linalg.norm(np.array(row.position) - old[row.id]) > 0 for row in after.blue)


def test_filtered_out_blue_can_destroy_target():
    env = make_env(4, blue=4, executor_scope="nearest3", seed=881)
    env.reset()
    for index, red in enumerate(env.adapter.env.red_agents):
        red.position = [800.0, index * 100.0, 100.0]
        red.velocity = [0.0, 0.0, 0.0]
    for index, blue in enumerate(env.adapter.env.blue_agents[:3]):
        blue.position = [1000.0, index * 100.0, 100.0]
        blue.velocity = [0.0, 0.0, 0.0]
    remote = env.adapter.env.blue_agents[3]
    remote.position = [-2090.0, -650.0, 100.0]
    remote.velocity = [0.0, 0.0, 0.0]
    state = env.state()
    # The dangerous Blue is much farther from the Red centroid than the
    # other three, so it is omitted from lower perception only.
    center = np.mean([row.position for row in state.red], axis=0)
    distances = {row.id: np.linalg.norm(np.array(row.position) - center) for row in state.blue}
    assert max(distances, key=distances.get) == remote.Id
    after, reward, done, info = env.step(Grouping((Group(0, state.ids("red")),)))
    assert done and not info["success"] and reward == 0
    assert after.targets[0].health == 0
    assert any(event["kind"] == "targets_breached" for event in info["events"])


def test_invalid_scope_is_rejected():
    with pytest.raises(ValueError, match="scope"):
        make_env(4, executor_scope="hidden_opponent")
    with pytest.raises(ValueError, match="scope"):
        FocusedExecutor(scope="hidden_opponent")
