"""Contract tests that keep official SMAClite and SMAClite-AD distinct."""

import numpy as np
import pytest

pytest.importorskip("smaclite")

from open_score.envs.smaclite_ad import (  # noqa: E402
    PROTOCOL_ID,
    SMACliteADEnv,
    SMACliteStockAdapter,
    stock_scenario_fingerprint,
    tensorize_smaclite_ad_observation,
    tensorize_smaclite_stock_observation,
)


EXPECTED_STOCK_MAPS_SHA256 = "0c862da08a59410a5832bc59899e39419d9e749f3ac41b20e34c463fbe8ef4b0"


def _first_actions(observation, count):
    return [
        int(np.flatnonzero(row)[0])
        for row in observation["avail_actions"][:count]
    ]


def test_official_stock_scenario_and_flat_tensor_adapter():
    provenance = stock_scenario_fingerprint()
    assert provenance["combined_sha256"] == EXPECTED_STOCK_MAPS_SHA256
    assert len(provenance["files"]) == 13

    env = SMACliteStockAdapter("3s5z", episode_limit=2, seed=11)
    observation, info = env.reset(seed=73)
    assert info["environment_id"] == "smaclite/3s5z-v0"
    assert observation["entity_obs"].shape == (8, 1, 128)
    assert observation["state_entities"].shape == (1, 216)
    assert observation["avail_actions"].shape == (8, 14)
    team, state = tensorize_smaclite_stock_observation(observation)
    team.validate()
    state.validate()
    actions = _first_actions(observation, env.n_agents)
    _, reward, terminated, truncated, _ = env.step(actions)
    assert isinstance(reward, float)
    assert not terminated and not truncated
    env.close()


@pytest.mark.parametrize("red_count,blue_count", [(2, 1), (3, 2), (5, 3)])
def test_ad_dynamic_rosters_share_shapes_and_masks(red_count, blue_count):
    env = SMACliteADEnv(
        red_count,
        blue_count,
        max_red_agents=6,
        max_blue_agents=5,
        episode_limit=3,
        seed=1,
    )
    observations, info = env.reset(seed=97)
    assert info["protocol_id"] == PROTOCOL_ID
    assert observations["Red"]["entity_obs"].shape == (6, 12, 13)
    assert observations["Blue"]["entity_obs"].shape == (5, 12, 13)
    assert observations["Red"]["avail_actions"].shape == (6, 12)
    assert observations["Blue"]["avail_actions"].shape == (5, 12)
    assert observations["Red"]["agent_mask"].sum() == red_count
    assert observations["Blue"]["agent_mask"].sum() == blue_count
    for side in ("Red", "Blue"):
        team, state = tensorize_smaclite_ad_observation(observations[side])
        team.validate()
        state.validate()
    env.close()


def test_ad_seeded_transition_is_deterministic_and_zero_sum():
    environments = [
        SMACliteADEnv(3, 2, max_red_agents=4, max_blue_agents=4, seed=5)
        for _ in range(2)
    ]
    transitions = []
    for env in environments:
        observations, _ = env.reset(seed=2026)
        actions = {
            side: _first_actions(observations[side], env.team_sizes[side])
            for side in ("Red", "Blue")
        }
        transition = env.step(actions)
        transitions.append(transition)
    for side in ("Red", "Blue"):
        for key in transitions[0][0][side]:
            assert np.array_equal(transitions[0][0][side][key], transitions[1][0][side][key])
    assert transitions[0][1] == transitions[1][1]
    assert transitions[0][4] == transitions[1][4]
    assert transitions[0][1]["Red"] == pytest.approx(-transitions[0][1]["Blue"])
    for env in environments:
        env.close()


def test_randomized_layout_is_seed_deterministic_distinct_and_legal():
    env = SMACliteADEnv(
        5,
        3,
        max_red_agents=6,
        max_blue_agents=5,
        spawn_jitter=2.0,
    )
    observation_a, info_a = env.reset(seed=1234)
    observation_b, info_b = env.reset(seed=1234)
    assert info_a["layout_hash"] == info_b["layout_hash"]
    assert info_a["layout"] == info_b["layout"]
    for side in ("Red", "Blue"):
        assert np.array_equal(
            observation_a[side]["state_entities"],
            observation_b[side]["state_entities"],
        )
    hashes = {info_a["layout_hash"]}
    for seed in range(1235, 1245):
        _, info = env.reset(seed=seed)
        hashes.add(info["layout_hash"])
        validation = env.validate_layout()
        assert validation["valid"]
        assert validation["boundary_violations"] == []
        assert validation["terrain_violations"] == []
        assert validation["overlaps"] == []
        assert validation["minimum_pairwise_clearance"] >= -1e-5
    assert len(hashes) == 11
    env.close()


def test_asset_action_id_is_stable_across_realised_blue_rosters():
    environments = [
        SMACliteADEnv(red, blue, max_red_agents=6, max_blue_agents=5)
        for red, blue in ((2, 1), (3, 2), (5, 3))
    ]
    assert {env.asset_action_id for env in environments} == {11}
    assert {env.ACTION_DIM for env in environments} == {12}
    for env in environments:
        env.close()


def test_asset_objective_and_stock_fingerprint_are_isolated():
    before = stock_scenario_fingerprint()["combined_sha256"]
    env = SMACliteADEnv(2, 1, max_red_agents=3, max_blue_agents=2, seed=2)
    observations, _ = env.reset(seed=3)
    env._asset.hp = 0  # force only the terminal branch under test
    actions = {
        side: _first_actions(observations[side], env.team_sizes[side])
        for side in ("Red", "Blue")
    }
    _, rewards, terminated, truncated, info = env.step(actions)
    assert terminated and not truncated
    assert info["termination_reason"] == "asset_destroyed"
    assert info["outcome_red"] == 1
    assert rewards["Red"] == pytest.approx(-rewards["Blue"])
    env.close()
    assert stock_scenario_fingerprint()["combined_sha256"] == before


def test_padded_non_noop_action_is_rejected():
    env = SMACliteADEnv(2, 1, max_red_agents=4, max_blue_agents=3, seed=4)
    env.reset(seed=5)
    with pytest.raises(ValueError, match="padding slots"):
        env.step({"Red": [1, 1, 1, 0], "Blue": [1]})
    env.close()
