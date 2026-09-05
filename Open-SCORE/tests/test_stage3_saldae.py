import math

import numpy as np
import pytest

from open_score.stage3 import (
    IdentityAction,
    IdentityBlottoGame,
    SALDAEConfig,
    enumerate_coalitions,
    make_engagement_slots,
    round_robin_identity_action,
    saldae_best_response,
    solve_saldae_double_oracle,
)


class _Oracle:
    def evaluate(self, requests):
        values = []
        for slot, red, blue in requests:
            red_skill = sum(0.15 + 0.03 * (value + 1) for value in red)
            blue_skill = sum(0.17 + 0.02 * (value - 99) for value in blue)
            pair_bonus = 0.25 if {0, 1}.issubset(red) else 0.0
            values.append(math.tanh(red_skill - blue_skill + pair_bonus - 0.01 * slot))
        return np.asarray(values, dtype=np.float64)


def _game(red_count=4, blue_count=3, full_candidates=False):
    red_ids = tuple(range(red_count))
    blue_ids = tuple(range(100, 100 + blue_count))
    slots = make_engagement_slots(
        (0, 1), max(red_count, blue_count), channels_per_target=max(red_count, blue_count)
    )
    if full_candidates:
        red_groups = enumerate_coalitions(red_ids, 4)
        blue_groups = enumerate_coalitions(blue_ids, 4)
    else:
        red_groups = tuple((value,) for value in red_ids)
        blue_groups = tuple((value,) for value in blue_ids)
    return IdentityBlottoGame(
        red_ids,
        blue_ids,
        slots,
        tuple(red_groups for _ in slots),
        tuple(blue_groups for _ in slots),
        _Oracle(),
        full_coalition_domain=full_candidates,
        allow_red_reserve=True,
        allow_blue_reserve=False,
        allow_unregistered_actions=True,
    )


def _assert_partition(game, action, side):
    checked = game.validate_action(action, side)
    ids = game.red_ids if side == "Red" else game.blue_ids
    active = [value for group in checked.coalitions for value in group]
    assert len(active) == len(set(active))
    assert set(active).isdisjoint(checked.reserve_ids)
    assert set(active).union(checked.reserve_ids) == set(ids)
    assert all(len(group) <= 4 for group in checked.coalitions)


def test_saldae_prices_the_same_payoff_and_preserves_identity_constraints():
    game = _game()
    blue = round_robin_identity_action(game.blue_ids, game.slots)
    red_start = IdentityAction(tuple(() for _ in game.slots), game.red_ids)
    response, diagnostics = saldae_best_response(
        game,
        [blue],
        [1.0],
        red_player=True,
        initial_actions=[red_start],
        config=SALDAEConfig(
            search_agents=3,
            time_limit_seconds=0.3,
            max_expansions=50,
            random_seed=7,
        ),
    )
    _assert_partition(game, response.action, "Red")
    expected = float(game.payoff_matrix([response.action], [blue])[0, 0])
    assert response.value == pytest.approx(expected)
    assert response.value >= game.payoff(red_start, blue) - 1e-12
    assert diagnostics.evaluated_nodes >= 1
    assert diagnostics.search_agents == 3
    assert not response.optimal


def test_unregistered_coalitions_are_explicitly_enabled_only_for_graph_search():
    game = _game()
    red = IdentityAction(
        ((0, 1), (2, 3), *(() for _ in range(len(game.slots) - 2))),
        (),
    )
    _assert_partition(game, red, "Red")
    assert any(
        group and group not in game.red_candidates[index]
        for index, group in enumerate(red.coalitions)
    )


def test_saldae_do_retains_the_zero_sum_outer_game_without_false_certificate():
    game = _game(red_count=3, blue_count=3, full_candidates=True)
    red = round_robin_identity_action(game.red_ids, game.slots)
    blue = round_robin_identity_action(game.blue_ids, game.slots)
    result, diagnostics = solve_saldae_double_oracle(
        game,
        [red],
        [blue],
        tolerance=1e-4,
        max_iterations=3,
        saldae_config=SALDAEConfig(
            search_agents=2,
            time_limit_seconds=0.08,
            max_expansions=12,
            random_seed=11,
        ),
    )
    assert np.isclose(result.red_mixture.sum(), 1.0)
    assert np.isclose(result.blue_mixture.sum(), 1.0)
    assert not result.full_game_certified
    assert not result.candidate_domain_exact
    assert len(diagnostics.red_searches) == len(result.history)
    assert len(diagnostics.blue_searches) == len(result.history)
    for action in result.red_strategies:
        _assert_partition(game, action, "Red")
    for action in result.blue_strategies:
        _assert_partition(game, action, "Blue")
