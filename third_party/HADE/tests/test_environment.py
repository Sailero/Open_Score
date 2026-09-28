"""Public protocol checks plus trajectories captured before package extraction."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from pettingzoo.test import parallel_api_test, parallel_seed_test

from make_env import make_env


REFERENCE = json.loads((Path(__file__).parent / "data" / "physics_reference.json").read_text(encoding="utf-8"))


def _world_rows(env):
    return [[list(map(float, x.position)), list(map(float, x.velocity)), float(x.Health),
             bool(getattr(x, "IsFire", False)), bool(getattr(x, "IsDisturb", False))]
            for x in env.world.world]


def _place(env, positions):
    for e, (p, v) in zip(env.world.world, positions):
        e.reset(list(p), list(v))
    env.world.reset_episode_state()


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_physics_matches_original_recorded_trajectory(case):
    env = make_env(**case["kwargs"], max_cycles=50)
    obs, _ = env.reset(seed=case["seed"])
    if case["preset"]:
        _place(env, case["preset"])
    assert _world_rows(env) == case["initial_world"]
    np.testing.assert_allclose(env.scenario.observation(env.world), case["initial_observation"], rtol=0, atol=0)
    for reference in case["frames"]:
        acting = env.agents.copy()
        obs, rewards, terms, truncs, infos = env.step({a: reference["actions"][env.agent_name_mapping[a]] for a in acting})
        for actual, expected in zip(_world_rows(env), reference["world"]):
            np.testing.assert_allclose(actual[0], expected[0], rtol=0, atol=1e-11)
            np.testing.assert_allclose(actual[1], expected[1], rtol=0, atol=1e-11)
            assert actual[2:] == expected[2:]
        np.testing.assert_allclose(env.state(), reference["state"], rtol=1e-7, atol=1e-7)
        for a in acting:
            i = env.agent_name_mapping[a]
            np.testing.assert_allclose(obs[a], reference["observations"][i], rtol=1e-7, atol=1e-7)
            side = "Red" if a.startswith("red") else "Blue"
            side_index = i - (case["kwargs"]["red_count"] if side == "Blue" else 0)
            assert rewards[a] == reference["rewards"]["RealReward"][side]
            np.testing.assert_allclose(infos[a]["LatentReward"], reference["rewards"]["LatentReward"][side][side_index], rtol=0, atol=1e-12)
            assert terms[a] == reference["done"][i]
            assert not truncs[a]
        assert env.outcome_red == reference["outcome"]
    env.close()


@pytest.mark.parametrize("continuous", [False, True])
def test_official_parallel_api_and_seed_checks(continuous):
    factory = lambda: make_env(red_count=3, blue_count=3, continuous=continuous)
    env = factory()
    parallel_api_test(env, num_cycles=30)
    parallel_seed_test(factory, num_cycles=20)
    env.close()


def test_stable_spaces_order_shapes_and_reward_components():
    env = make_env(red_count=3, blue_count=3, red_scouts=1, blue_disturbers=1, reward_weights=[1, .3, .7, 1])
    obs, infos = env.reset(seed=77)
    names = env.possible_agents.copy()
    action_spaces = [env.action_space(a) for a in names]
    observation_spaces = [env.observation_space(a) for a in names]
    assert infos["red_0"]["role"] == "scout"
    assert infos["blue_0"]["role"] == "disturb"
    assert all(x.shape == (7, 7) for x in obs.values())
    for a, row in obs.items():
        assert env.observation_space(a).contains(row)
        assert len(infos[a]["observation_entities"]) == 7
        assert a not in infos[a]["observation_entities"]
    _, rewards, _, _, infos = env.step({a: 0 for a in env.agents})
    for a, value in rewards.items():
        assert value == pytest.approx(np.dot(infos[a]["LatentReward"], [1, .3, .7, 1]))
    env.reset(seed=88)
    assert env.possible_agents == names
    assert all(env.action_space(a) is s for a, s in zip(names, action_spaces))
    assert all(env.observation_space(a) is s for a, s in zip(names, observation_spaces))
    assert env.state_space.contains(env.state())
    env.close()


@pytest.mark.parametrize("continuous,bad", [(False, 27), (False, -1), (False, True), (False, 2.0),
                                          (True, [0, float("nan"), 0]), (True, [0, 2, 0]),
                                          (True, [1, 0]), (True, ["0", "1", "0"])])
def test_invalid_joint_action_has_no_side_effects(continuous, bad):
    env = make_env(red_count=1, blue_count=1, continuous=continuous)
    env.reset(seed=12)
    before = _world_rows(env)
    rng = copy.deepcopy(env._legacy_random_state)
    good = [0, 0, 0] if continuous else 0
    for actions in ({"red_0": good, "blue_0": bad}, {"red_0": good}, {"unknown": good}):
        with pytest.raises(ValueError):
            env.step(actions)
        assert _world_rows(env) == before
        assert env.num_cycles == env.world.physics_step_count == 0
        np.testing.assert_array_equal(env._legacy_random_state[1], rng[1])
    env.close()


def test_death_final_transition_then_removal_without_slot_shifts():
    env = make_env(red_count=3, blue_count=1, target_count=1, record_events=True)
    env.reset(seed=3)
    _place(env, [([-500, -500, 100], [20, 0, 0]), ([-500, -500, 100], [20, 0, 0]),
                 ([-1000, 1000, 100], [20, 0, 0]), ([2000, 0, 100], [-20, 0, 0]),
                 ([-2200, 0, 100], [0, 0, 0])])
    obs, rewards, terms, truncs, infos = env.step({a: 0 for a in env.agents})
    assert terms["red_0"] and terms["red_1"]
    assert not terms["red_2"] and not terms["blue_0"]
    assert env.agents == ["red_2", "blue_0"]
    assert set(obs) == set(rewards) == set(terms) == set(truncs) == set(infos) == set(env.possible_agents)
    assert any(e["kind"] == "collision" for e in infos["red_0"]["events"])
    ids = infos["red_2"]["observation_entities"]
    assert obs["red_2"][ids.index("red_0"), -1] == 0
    obs2, rewards2, *_ = env.step({a: 0 for a in env.agents})
    assert set(obs2) == set(rewards2) == {"red_2", "blue_0"}
    assert obs2["red_2"].shape == obs["red_2"].shape
    assert obs2["red_2"][ids.index("red_0"), -1] == 0
    env.close()


def test_horizon_does_not_invent_win_and_terminal_reward_paid_once():
    env = make_env(red_count=1, blue_count=1, max_cycles=1, initialization="uniform")
    env.reset(seed=22)
    _, rewards, terms, truncs, infos = env.step({a: 0 for a in env.agents})
    assert not any(terms.values()) and all(truncs.values())
    assert all(r == 0 for r in rewards.values()) and env.outcome_red == 0
    assert env.agents == [] and env.step({}) == ({}, {}, {}, {}, {})
    env.reset(seed=22)
    env.world.blue_agents[0].Health = 0
    _, rewards, terms, truncs, _ = env.step({a: 0 for a in env.agents})
    assert rewards == {"red_0": 10., "blue_0": -10.}
    assert all(terms.values()) and not any(truncs.values())
    assert env.step({})[1] == {}
    env.close()


def test_private_rng_does_not_change_ambient_numpy_and_is_repeatable(monkeypatch):
    import importlib
    # The shipped pi limit normally leaves this fallback dormant. Lower it
    # only inside this test to exercise the preserved native random path.
    monkeypatch.setattr(importlib.import_module("had_env.core.function.Function"), "wMax", np.pi / 2)
    first, second = make_env(red_count=1, blue_count=1), make_env(red_count=1, blue_count=1)
    np.random.seed(143)
    before = copy.deepcopy(np.random.get_state())
    obs1, _ = first.reset(seed=91)
    obs2, _ = second.reset(seed=91)
    for a in obs1:
        np.testing.assert_array_equal(obs1[a], obs2[a])
    for e in (first, second):
        _place(e, [([-1000, 0, 100], [20, 0, 0]), ([2000, 0, 100], [-20, 0, 0]),
                   ([-2200, -700, 100], [0, 0, 0]), ([-2200, 700, 100], [0, 0, 0])])
        e.world.agents[0].velocity = [20., 0., 0.]
    action = {a: 0 for a in first.agents}
    action["red_0"] = 5  # a discrete acceleration containing negative x
    first.step(action)
    assert first._legacy_random_state[2] != 624
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    np.random.random(100)
    second.step(action)
    np.testing.assert_array_equal(first.state(), second.state())
    first.close()
    second.close()


def test_mpe_fixed_lists_flattened_obs_one_hot_and_done_semantics():
    env = make_env(api="mpe", red_count=1, blue_count=1, max_cycles=1, initialization="uniform")
    obs = env.reset(seed=13)
    assert env.n == 2 and len(obs) == 2
    assert all(space.contains(row) for space, row in zip(env.observation_space, obs))
    actions = [np.eye(27, dtype=np.float32)[0], 0]
    obs, rewards, dones, info = env.step(actions)
    assert dones == [True, True] and rewards == [0., 0.]
    assert all(x["truncated"] and not x["terminated"] for x in info["n"])
    obs, rewards, dones, info = env.step(actions)
    assert all(not np.any(row) for row in obs) and rewards == [0., 0.]
    assert all(x["inactive"] for x in info["n"])
    env.close()


def test_render_and_reset_lifecycle(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    env = make_env(red_count=1, blue_count=1, render_mode="rgb_array")
    with pytest.raises(RuntimeError):
        env.step({})
    env.reset(seed=1)
    before = env.state()
    frame = env.render()
    assert frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape[-1] == 3
    np.testing.assert_array_equal(env.state(), before)
    env.close()
    with pytest.raises(RuntimeError):
        env.step({})
    env.reset(seed=1)
    assert env.render().shape == frame.shape
    env.close()


def test_public_import_is_headless_without_research_dependencies():
    code = "from make_env import make_env; import sys; e=make_env(); e.reset(seed=1); e.step({a:0 for a in e.agents}); e.close(); assert not any(k in sys.modules for k in ['torch','pygame','PySide6','open_score'])"
    subprocess.run([sys.executable, "-B", "-c", code], cwd=Path(__file__).parents[1], check=True)


@pytest.mark.parametrize("kwargs", [dict(red_count=0), dict(blue_count=True), dict(target_count=0),
    dict(red_count=1, red_scouts=2), dict(blue_count=1, blue_scouts=1),
    dict(max_cycles=0), dict(reward_weights=[1, 2]), dict(reward_weights=[1, 2, 3, float("nan")]),
    dict(target_region=[[0, float("nan")]] * 3), dict(initialization="missing"),
    dict(api="missing"), dict(scenario_name="missing"), dict(continuous="yes")])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        make_env(**kwargs)
