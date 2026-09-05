from __future__ import annotations

import numpy as np
import pytest

from open_score.envs import HADStage3Adapter
from open_score.stage3 import (
    IdentityAction,
    IdentityBlottoGame,
    apply_identity_joint_plan,
    build_identity_event_game,
    enumerate_identity_actions,
    make_engagement_slots,
    round_robin_identity_action,
    solve_identity_double_oracle,
    solve_restricted_matrix_game,
    spatial_candidate_coalitions,
)


class IdentityToyOracle:
    def evaluate(self, requests):
        return np.asarray(
            [
                0.07 * sum(red)
                - 0.013 * sum(blue)
                + 0.4 * len(red)
                - 0.55 * len(blue)
                + 0.03 * slot
                for slot, red, blue in requests
            ],
            dtype=np.float64,
        )


def _full_game(red_ids=(0, 1, 2), blue_ids=(10, 11, 12)):
    slots = make_engagement_slots((0, 1), max(len(red_ids), len(blue_ids)))
    targets = {0: (0.0, 0.0, 0.0), 1: (5.0, 0.0, 0.0)}
    red_positions = {value: (float(value), 0.0, 0.0) for value in red_ids}
    blue_positions = {value: (float(value - 10), 0.0, 0.0) for value in blue_ids}
    red_candidates = spatial_candidate_coalitions(
        red_ids, slots, red_positions, targets, full_domain=True
    )
    blue_candidates = spatial_candidate_coalitions(
        blue_ids, slots, blue_positions, targets, full_domain=True
    )
    return IdentityBlottoGame(
        red_ids,
        blue_ids,
        slots,
        red_candidates,
        blue_candidates,
        IdentityToyOracle(),
        full_coalition_domain=True,
    )


def test_identity_actions_partition_labels_and_never_exceed_four():
    game = _full_game()
    actions = enumerate_identity_actions(game.red_ids, game.slots)
    assert actions
    for action in actions:
        assert game.validate_action(action, "Red") == action
        flattened = [value for coalition in action.coalitions for value in coalition]
        assert sorted(flattened) == list(game.red_ids)
        assert max(map(len, action.coalitions)) <= 4
    with pytest.raises(ValueError, match="partition"):
        game.validate_action(
            IdentityAction(((0, 1), (1, 2)) + ((),) * (len(game.slots) - 2)),
            "Red",
        )
    gap = [()] * len(game.slots)
    gap[1] = (0, 1, 2)
    with pytest.raises(ValueError, match="non-increasing group sizes"):
        game.validate_action(IdentityAction(tuple(gap)), "Red")


def test_set_partitioning_best_responses_equal_brute_force():
    game = _full_game()
    red_actions = enumerate_identity_actions(
        game.red_ids, game.slots, allow_reserve=True
    )
    blue_actions = enumerate_identity_actions(game.blue_ids, game.slots)
    blue_support = blue_actions[:5]
    blue_mixture = np.asarray([1, 2, 3, 4, 5], dtype=np.float64)
    blue_mixture /= blue_mixture.sum()
    red_response = game.red_best_response(blue_support, blue_mixture)
    brute_red = game.payoff_matrix(red_actions, blue_support) @ blue_mixture
    assert red_response.optimal
    assert red_response.value == pytest.approx(float(brute_red.max()))

    red_support = red_actions[::3]
    red_mixture = np.arange(1, len(red_support) + 1, dtype=np.float64)
    red_mixture /= red_mixture.sum()
    blue_response = game.blue_best_response(red_support, red_mixture)
    brute_blue = red_mixture @ game.payoff_matrix(red_support, blue_actions)
    assert blue_response.optimal
    assert blue_response.value == pytest.approx(float(brute_blue.min()))


def test_identity_double_oracle_matches_complete_small_matrix():
    game = _full_game()
    red_actions = enumerate_identity_actions(
        game.red_ids, game.slots, allow_reserve=True
    )
    blue_actions = enumerate_identity_actions(game.blue_ids, game.slots)
    exact = solve_restricted_matrix_game(game.payoff_matrix(red_actions, blue_actions))
    result = solve_identity_double_oracle(
        game,
        [round_robin_identity_action(game.red_ids, game.slots)],
        [round_robin_identity_action(game.blue_ids, game.slots)],
        tolerance=1e-9,
        max_iterations=60,
    )
    assert result.converged
    assert result.full_game_certified
    assert result.value == pytest.approx(exact.value, abs=1e-8)
    assert result.exploitability <= 1e-9


def test_had_runtime_queries_and_executes_exact_id_coalitions():
    class ConstantPredictor:
        def predict_red_win(self, states, *, rosters, **_kwargs):
            assert all(1 <= red <= 4 and 1 <= blue <= 4 for red, blue in rosters)
            return np.full(len(states), 0.75, dtype=np.float32)

    adapter = HADStage3Adapter(
        4,
        3,
        2,
        max_steps=5,
        target_positions=[[-2100.0, -300.0, 100.0], [-2100.0, 300.0, 100.0]],
    )
    adapter.reset(seed=31)
    built = build_identity_event_game(
        adapter, ConstantPredictor(), full_domain_agent_threshold=8
    )
    red = built.initial_red[0]
    blue = built.initial_blue[0]
    assert np.isfinite(built.game.payoff(red, blue))
    plan = apply_identity_joint_plan(
        adapter, built.game, red, blue, blue_type_name="rush"
    )
    red_used = [value for item in plan.local_subgames for value in item.red_ids]
    blue_used = [value for item in plan.local_subgames for value in item.blue_ids]
    assert sorted(red_used) == list(adapter.red_ids)
    assert sorted(blue_used) == list(adapter.blue_ids)
    assert all(len(item.red_ids) <= 4 and len(item.blue_ids) <= 4 for item in plan.local_subgames)
    assert dict(plan.red_assignment) == adapter.red_assignment
    assert dict(plan.blue_assignment) == adapter.blue_assignment
