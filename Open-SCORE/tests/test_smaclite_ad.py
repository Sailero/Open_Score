"""Contract tests that keep official SMAClite and SMAClite-AD distinct."""

import numpy as np
import pytest
import json
from pathlib import Path

pytest.importorskip("smaclite")

from open_score.envs.smaclite_ad import (  # noqa: E402
    PROTOCOL_ID,
    SMACliteADEnv,
    SMACliteStockAdapter,
    stock_scenario_fingerprint,
    tensorize_smaclite_ad_observation,
    tensorize_smaclite_stock_observation,
)
from smaclite.env.units.combat_type import CombatType  # noqa: E402
from open_score.stage1.smaclite_ad_training import (  # noqa: E402
    SMACliteADEpisodeRunner,
    SMACliteADFactory,
    SMACliteADRuleController,
    evaluate_smaclite_ad,
)


EXPECTED_STOCK_MAPS_SHA256 = "77508021d516a9ed8a8e520a32570bea68803c3a4ffdbf57266b05f1be65d310"


def _first_actions(observation, count):
    return [
        int(np.flatnonzero(row)[0])
        for row in observation["avail_actions"][:count]
    ]


def test_official_stock_scenario_and_entity_target_adapter():
    provenance = stock_scenario_fingerprint()
    assert provenance["combined_sha256"] == EXPECTED_STOCK_MAPS_SHA256
    assert len(provenance["files"]) == 13

    env = SMACliteStockAdapter("3s5z", episode_limit=2, seed=11)
    observation, info = env.reset(seed=73)
    assert info["environment_id"] == "smaclite/3s5z-v0"
    assert observation["entity_obs"].shape == (8, 16, 12)
    assert observation["self_obs"].shape == (8, 8)
    assert observation["state_entities"].shape == (1, 217)
    assert observation["state_entities"][0, -1] == pytest.approx(1.0)
    assert observation["avail_actions"].shape == (8, 14)
    assert np.all(observation["action_entity_index"][:, 6:] == np.arange(8, 16))
    assert np.all(observation["action_target_type"][:, 6:] == 1)
    raw = np.asarray(env.unwrapped.get_obs(), dtype=np.float32)
    assert np.array_equal(observation["self_obs"][:, :4], raw[:, :4])
    # Enemy id_in_faction=0 occupies the first target entity and retains its
    # complete upstream observation block without changing action 6 semantics.
    enemy_dim = env.unwrapped.enemy_feat_size
    assert np.array_equal(
        observation["entity_obs"][:, 8, :enemy_dim], raw[:, 4 : 4 + enemy_dim]
    )
    team, state = tensorize_smaclite_stock_observation(observation)
    team.validate()
    state.validate()
    actions = _first_actions(observation, env.n_agents)
    _, reward, terminated, truncated, _ = env.step(actions)
    assert isinstance(reward, float)
    assert not terminated and not truncated
    env.close()


def test_ad_actor_and_central_state_include_remaining_horizon():
    env = SMACliteADEnv(
        3, 2, max_red_agents=4, max_blue_agents=3, episode_limit=4, seed=13
    )
    observations, _ = env.reset(seed=79)
    initial = observations["Red"]
    assert np.allclose(initial["task_obs"][initial["agent_mask"], -1], 1.0)
    assert np.allclose(initial["state_entities"][initial["state_mask"], -1], 1.0)
    env.episode_steps = 3
    late = env.observe("Red")
    assert np.allclose(late["task_obs"][late["agent_mask"], -1], 0.25)
    assert np.allclose(late["state_entities"][late["state_mask"], -1], 0.25)
    env.close()


def test_stock_heal_and_damage_slots_have_explicit_distinct_target_types():
    env = SMACliteStockAdapter("MMM", episode_limit=2, seed=29)
    observation, _ = env.reset(seed=31)
    for agent_index, unit in env.unwrapped.agents.items():
        indices = observation["action_entity_index"][agent_index, 6:]
        types = observation["action_target_type"][agent_index, 6:]
        if unit.combat_type == CombatType.HEALING:
            assert np.array_equal(indices[: env.n_agents], np.arange(env.n_agents))
            assert np.all(types[: env.n_agents] == 2)
        else:
            enemy_count = env.unwrapped.n_enemies
            assert np.array_equal(
                indices[:enemy_count], env.n_agents + np.arange(enemy_count)
            )
            assert np.all(types[:enemy_count] == 1)
    env.close()


def test_all_13_stock_map_files_have_bijective_target_action_entity_slots():
    map_root = (
        Path(__import__("smaclite").__file__).resolve().parent
        / "env"
        / "maps"
        / "smaclite_maps"
    )
    map_names = [json.loads(path.read_text(encoding="utf-8"))["name"] for path in map_root.glob("*.json")]
    assert len(map_names) == 13
    for map_name in map_names:
        env = SMACliteStockAdapter(map_name, episode_limit=1, seed=37)
        observation, _ = env.reset(seed=41)
        for agent_index, unit in env.unwrapped.agents.items():
            mapping = observation["action_entity_index"][agent_index]
            target_type = observation["action_target_type"][agent_index]
            mapped_actions = np.flatnonzero(mapping >= 0)
            mapped_rows = mapping[mapped_actions]
            assert len(mapped_rows) == len(set(map(int, mapped_rows)))
            inverse = {int(row): int(action) for action, row in zip(mapped_actions, mapped_rows)}
            for action, entity_row in zip(mapped_actions, mapped_rows):
                assert action >= 6
                assert inverse[int(entity_row)] == int(action)
                if unit.combat_type == CombatType.HEALING:
                    assert entity_row == action - 6
                    assert target_type[action] == 2
                    assert observation["entity_obs"][agent_index, entity_row, -3] == 1.0
                else:
                    assert entity_row == env.n_agents + action - 6
                    assert target_type[action] == 1
                    assert observation["entity_obs"][agent_index, entity_row, -2] == 1.0
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
        observations, _ = env.reset(seed=17)
        red = observations["Red"]
        blue = observations["Blue"]
        assert np.all(
            red["action_entity_index"][: env.team_sizes["Red"], 6:11]
            == np.arange(6, 11)
        )
        assert np.all(
            blue["action_entity_index"][: env.team_sizes["Blue"], 6:12]
            == np.arange(6)
        )
        active_red = red["agent_mask"]
        assert np.all(red["action_entity_index"][active_red, 11] == 11)
        assert np.all(red["action_target_type"][active_red, 11] == 3)
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


def test_strict_potential_discounted_return_telescopes_to_terminal_payoff():
    gamma = 0.91
    env = SMACliteADEnv(
        2,
        1,
        max_red_agents=2,
        max_blue_agents=1,
        episode_limit=3,
        reward_mode="strict_potential",
        discount_gamma=gamma,
        shaping_scale=0.5,
        approach_weight=1.0,
        seed=41,
    )
    observations, info = env.reset(seed=43)
    initial = (*env._fractions(), env._approach_fraction())
    initial_potential = env._shaping_potential(*initial)
    rewards = []
    terminated = truncated = False
    while not (terminated or truncated):
        observations, step_rewards, terminated, truncated, info = env.step(
            {"Red": [1, 1], "Blue": [1]}
        )
        rewards.append(step_rewards["Red"])
    discounted = sum((gamma**index) * value for index, value in enumerate(rewards))
    discounted_terminal_only = gamma ** (len(rewards) - 1) * -1.0
    assert discounted == pytest.approx(
        discounted_terminal_only - initial_potential, abs=1e-6
    )
    assert info["reward_mode"] == "strict_potential"
    assert info["reward_components"]["potential_after"] == pytest.approx(0.0)
    assert info["reward_components"]["terminal_red"] == pytest.approx(-1.0)
    env.close()


def test_terminal_only_mode_has_no_dense_reward():
    env = SMACliteADEnv(
        2,
        1,
        max_red_agents=2,
        max_blue_agents=1,
        episode_limit=1,
        reward_mode="terminal_only",
        shaping_scale=10.0,
        approach_weight=10.0,
    )
    _, _ = env.reset(seed=47)
    _, rewards, terminated, truncated, info = env.step(
        {"Red": [1, 1], "Blue": [1]}
    )
    assert not terminated and truncated
    assert rewards == {"Red": -1.0, "Blue": 1.0}
    assert info["reward_components"]["shaping_red"] == pytest.approx(0.0)
    env.close()


def test_clear_then_asset_is_the_solvable_rule_and_scope_is_disclosed():
    factory = SMACliteADFactory(
        max_red_agents=6,
        max_blue_agents=5,
        episode_limit=150,
        reward_mode="terminal_only",
        spawn_jitter=0.0,
    )
    runner = SMACliteADEpisodeRunner(factory)
    clear = SMACliteADRuleController("clear_then_asset")
    intercept = SMACliteADRuleController("intercept")
    result = evaluate_smaclite_ad(
        runner,
        clear,
        intercept,
        "Red",
        [(2, 1), (3, 2), (5, 3)],
        episodes_per_ratio=1,
        seed=53,
    )
    assert result.win_rate == pytest.approx(1.0)
    assert all(cell["win_rate"] == 1.0 for cell in result.per_ratio.values())
    assert intercept.information_scope == "privileged_full_environment_state"
    factory.close()
