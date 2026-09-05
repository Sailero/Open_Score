from __future__ import annotations

import numpy as np
import pytest
import itertools

from open_score.envs import HADStage3Adapter
from open_score.stage3 import (
    GroupedBlottoGame,
    GroupedBayesianBlottoGame,
    GroupedTypedBluePolicy,
    apply_joint_grouped_plan,
    build_grouped_event_game,
    enumerate_grouped_allocations,
    group_histograms_for_total,
    grouped_allocation_from_target_counts,
    histogram_from_group_sizes,
    make_grouped_blue_type,
    pair_group_histograms,
    solve_grouped_double_oracle,
    solve_grouped_bayesian_double_oracle,
    solve_restricted_matrix_game,
)


def _smooth_payoff(targets: int, red_cap: int, blue_cap: int) -> np.ndarray:
    red = np.arange(red_cap + 1, dtype=np.float64)[None, :, None]
    blue = np.arange(blue_cap + 1, dtype=np.float64)[None, None, :]
    value = np.tanh((red - 1.2 * blue) / 2.0)
    return np.broadcast_to(value, (targets, red_cap + 1, blue_cap + 1)).copy()


def test_group_histograms_are_canonical_and_allow_repeated_targets():
    patterns = group_histograms_for_total(6, 4)
    assert histogram_from_group_sizes([3, 2, 1], 4) in patterns
    assert histogram_from_group_sizes([1, 3, 2], 4) == histogram_from_group_sizes(
        [3, 2, 1], 4
    )

    action = grouped_allocation_from_target_counts([9, 9], 4)
    game = GroupedBlottoGame(_smooth_payoff(2, 4, 4), 18, 12)
    assert game.target_counts(action, "Red") == (9, 9)
    assert game.groups(action, "Red") == ((4, 4, 1), (4, 4, 1))


def test_public_pairing_pads_only_the_shorter_group_pattern():
    red = histogram_from_group_sizes([4, 2], 4)
    blue = histogram_from_group_sizes([3], 4)
    assert pair_group_histograms(red, blue) == ((4, 3), (2, 0))

    red = histogram_from_group_sizes([2], 4)
    blue = histogram_from_group_sizes([4, 3, 1], 4)
    assert pair_group_histograms(red, blue) == ((2, 4), (0, 3), (0, 1))


def test_group_pattern_best_responses_match_full_enumeration():
    rng = np.random.default_rng(20260904)
    payoff = rng.normal(size=(2, 4, 4))
    game = GroupedBlottoGame(payoff, 4, 3)
    full_red = enumerate_grouped_allocations(game, "Red")
    full_blue = enumerate_grouped_allocations(game, "Blue")
    blue_support = full_blue[:: max(1, len(full_blue) // 5)][:5]
    mixture = np.arange(1, len(blue_support) + 1, dtype=np.float64)
    mixture /= mixture.sum()
    response = game.defender_best_response(blue_support, mixture)
    brute = game.payoff_matrix(full_red, blue_support) @ mixture
    assert response.value == pytest.approx(float(brute.max()))
    assert game.payoff_matrix([response.allocation], blue_support)[0] @ mixture == pytest.approx(
        float(brute.max())
    )

    red_support = full_red[:: max(1, len(full_red) // 6)][:6]
    mixture = np.arange(len(red_support), 0, -1, dtype=np.float64)
    mixture /= mixture.sum()
    response = game.attacker_best_response(red_support, mixture)
    brute = mixture @ game.payoff_matrix(red_support, full_blue)
    assert response.value == pytest.approx(float(brute.min()))


def test_grouped_double_oracle_matches_toy_full_matrix_value():
    rng = np.random.default_rng(71)
    game = GroupedBlottoGame(rng.normal(size=(2, 4, 4)), 3, 3)
    full_red = enumerate_grouped_allocations(game, "Red")
    full_blue = enumerate_grouped_allocations(game, "Blue")
    exact = solve_restricted_matrix_game(game.payoff_matrix(full_red, full_blue))
    result = solve_grouped_double_oracle(
        game, max_iterations=80, tolerance=1e-9, seed=4
    )
    assert not result.used_matrix_game_fallback
    assert result.converged
    assert result.value == pytest.approx(exact.value, abs=1e-8)
    assert result.exploitability <= 1e-9


def test_grouped_bayesian_double_oracle_matches_toy_harsanyi_matrix():
    rng = np.random.default_rng(88)
    games = [
        GroupedBlottoGame(rng.normal(size=(2, 3, 3)), 2, 2),
        GroupedBlottoGame(rng.normal(size=(2, 3, 3)), 2, 2),
    ]
    types = tuple(
        make_grouped_blue_type(
            game,
            name=style,
            prior=1.0,
            lower_style=style,
            upper_family="strategic",
        )
        for game, style in zip(games, ("rush", "split_rush"))
    )
    bayesian = GroupedBayesianBlottoGame(types)
    red = enumerate_grouped_allocations(games[0], "Red")
    blue_by_type = [enumerate_grouped_allocations(game, "Blue") for game in games]
    policies = tuple(
        GroupedTypedBluePolicy(tuple(values))
        for values in itertools.product(*blue_by_type)
    )
    exact = solve_restricted_matrix_game(bayesian.payoff_matrix(red, policies))
    result = solve_grouped_bayesian_double_oracle(
        bayesian, max_iterations=100, tolerance=1e-9, seed=10
    )
    assert result.converged
    assert not result.used_matrix_game_fallback
    assert result.value == pytest.approx(exact.value, abs=1e-8)
    assert result.exploitability <= 1e-9


def test_group_support_is_not_hard_capped_at_four():
    game = GroupedBlottoGame(_smooth_payoff(1, 8, 8), 5, 5)
    five = grouped_allocation_from_target_counts([5], 8, preferred_size=5)
    assert game.validate_defender_allocation(five) == five
    assert game.groups(five, "Red") == ((5,),)


def test_grouped_runtime_uses_a_budget_safe_structural_breach_penalty():
    class ConstantPredictor:
        def predict_roster_surface(
            self, states, *, target_count, red_cap, blue_cap, **_kwargs
        ):
            assert len(states) == target_count * red_cap * blue_cap
            return np.full((target_count, red_cap, blue_cap), 0.75, np.float32)

    adapter = HADStage3Adapter(
        6,
        4,
        2,
        max_steps=5,
        target_positions=[[-2100.0, -300.0, 100.0], [-2100.0, 300.0, 100.0]],
    )
    adapter.reset(seed=17)
    built = build_grouped_event_game(
        adapter,
        ConstantPredictor(),
        group_size_cap=8,
        utility_mode="joint_survival_log_probability",
        risk_epsilon=1e-6,
    )
    assert np.all(built.local_group_payoff[:, 0, 1:] == -(6 + 4 + 1))


def test_grouped_grounding_returns_explicit_disjoint_subgames_at_same_target():
    adapter = HADStage3Adapter(
        6,
        5,
        2,
        max_steps=5,
        target_positions=[[-2100.0, -400.0, 100.0], [-2100.0, 400.0, 100.0]],
    )
    adapter.reset(seed=93)
    game = GroupedBlottoGame(_smooth_payoff(2, 4, 4), 6, 5)
    red = tuple(
        value
        for pattern in (
            histogram_from_group_sizes([2, 1], 4),
            histogram_from_group_sizes([2, 1], 4),
        )
        for value in pattern
    )
    blue = tuple(
        value
        for pattern in (
            histogram_from_group_sizes([2, 1], 4),
            histogram_from_group_sizes([2], 4),
        )
        for value in pattern
    )
    grounded = apply_joint_grouped_plan(
        adapter, game, red, blue, blue_type_name="rule"
    )
    assert len([item for item in grounded.local_subgames if item.target_id == 0]) == 2
    assert len([item for item in grounded.local_subgames if item.target_id == 1]) == 2
    red_ids = [value for item in grounded.local_subgames for value in item.red_ids]
    blue_ids = [value for item in grounded.local_subgames for value in item.blue_ids]
    assert len(red_ids) == len(set(red_ids)) == 6
    assert len(blue_ids) == len(set(blue_ids)) == 5
    assert grounded.red_allocation == (3, 3)
    assert grounded.blue_allocation == (3, 2)


def test_legacy_chunker_keeps_unmatched_blue_as_zero_v_blue_groups():
    from open_score.stage3 import FrozenStage1GroupExecutor

    adapter = HADStage3Adapter(
        1,
        9,
        1,
        max_steps=5,
        target_positions=[[-2100.0, 0.0, 100.0]],
    )
    adapter.reset(seed=19)
    groups = FrozenStage1GroupExecutor(None, "cpu").micro_rosters(adapter)
    pairs = [(len(red), len(blue)) for red, blue in groups.values()]
    assert sum(red for red, _ in pairs) == 1
    assert sum(blue for _, blue in pairs) == 9
    assert max(blue for _, blue in pairs) <= 4
    assert pairs.count((0, 3)) == 2
