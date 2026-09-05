from __future__ import annotations

import numpy as np
import pytest

from open_score.stage3.identity_blotto import (
    EngagementSlot,
    IdentityAction,
    IdentityBlottoGame,
    enumerate_identity_actions,
)


class _ConstantOracle:
    def __init__(self, value: float = 0.0) -> None:
        self.value = float(value)

    def evaluate(self, requests):
        return np.full(len(requests), self.value, dtype=np.float64)


def _game(
    red_ids,
    blue_ids,
    slots,
    red_candidates,
    blue_candidates,
    *,
    oracle_value: float = 0.0,
) -> IdentityBlottoGame:
    return IdentityBlottoGame(
        red_ids,
        blue_ids,
        slots,
        red_candidates,
        blue_candidates,
        _ConstantOracle(oracle_value),
        allow_red_reserve=True,
        allow_blue_reserve=False,
    )


def test_red_enumeration_contains_reserve_and_blue_cannot_use_it():
    slots = (EngagementSlot(0, 0), EngagementSlot(0, 1))
    game = _game(
        (0, 1),
        (10, 11),
        slots,
        (((0,), (1,), (0, 1)),) * 2,
        (((10,), (11,), (10, 11)),) * 2,
    )
    actions = enumerate_identity_actions((0, 1), slots, allow_reserve=True)
    assert IdentityAction(((), ()), (0, 1)) in actions
    for action in actions:
        assert game.validate_action(action, "Red") == action
        active = {agent for group in action.coalitions for agent in group}
        assert active.isdisjoint(action.reserve_ids)
        assert active.union(action.reserve_ids) == {0, 1}
    with pytest.raises(ValueError, match="reserve is disabled"):
        game.validate_action(IdentityAction(((10,), ()), (11,)), "Blue")


def test_fourteen_red_can_form_three_four_vs_one_groups_plus_reserve():
    red_ids = tuple(range(14))
    blue_ids = (100, 101, 102)
    slots = tuple(EngagementSlot(0, channel) for channel in range(3))
    red_groups = (tuple(range(0, 4)), tuple(range(4, 8)), tuple(range(8, 12)))
    blue_groups = ((100,), (101,), (102,))
    game = _game(
        red_ids,
        blue_ids,
        slots,
        tuple((group,) for group in red_groups),
        tuple((group,) for group in blue_groups),
    )
    red = game.validate_action(IdentityAction(red_groups, (12, 13)), "Red")
    blue = game.validate_action(IdentityAction(blue_groups), "Blue")
    assert tuple(map(len, red.coalitions)) == (4, 4, 4)
    assert red.reserve_ids == (12, 13)
    assert not blue.reserve_ids
    response = game.red_best_response((blue,), (1.0,))
    assert response.optimal
    assert response.action == red


def test_more_blue_agents_are_represented_by_unopposed_blue_groups():
    red_ids = tuple(range(4))
    blue_ids = tuple(range(100, 110))
    slots = tuple(EngagementSlot(0, channel) for channel in range(3))
    red_groups = (red_ids, (), ())
    blue_groups = (
        tuple(range(100, 104)),
        tuple(range(104, 108)),
        tuple(range(108, 110)),
    )
    game = _game(
        red_ids,
        blue_ids,
        slots,
        ((red_ids,), ((0,),), ((1,),)),
        tuple((group,) for group in blue_groups),
    )
    red = game.validate_action(IdentityAction(red_groups), "Red")
    blue = game.validate_action(IdentityAction(blue_groups), "Blue")
    local_sizes = tuple(
        (len(red.coalitions[index]), len(blue.coalitions[index]))
        for index in range(len(slots))
    )
    assert local_sizes == ((4, 4), (0, 4), (0, 2))
    assert np.isfinite(game.payoff(red, blue))
    response = game.blue_best_response((red,), (1.0,))
    assert response.optimal
    assert response.action == blue


def test_red_best_response_can_select_all_reserve():
    slots = (EngagementSlot(0, 0),)
    game = _game(
        (0,),
        (10,),
        slots,
        (((0,),),),
        (((10,),),),
        oracle_value=-100.0,
    )
    blue = IdentityAction(((10,),))
    response = game.red_best_response((blue,), (1.0,))
    assert response.optimal
    assert response.action == IdentityAction(((),), (0,))
    assert response.value == pytest.approx(-3.0)


def test_positive_payoff_normalisation_preserves_best_response_action():
    slots = (EngagementSlot(0, 0),)
    common = dict(
        red_ids=(0,),
        blue_ids=(10,),
        slots=slots,
        red_candidates=(((0,),),),
        blue_candidates=(((10,),),),
        payoff_oracle=_ConstantOracle(-100.0),
        structural_breach_penalty=3.0,
        allow_red_reserve=True,
        allow_blue_reserve=False,
    )
    raw = IdentityBlottoGame(**common, payoff_scale=1.0)
    normalised = IdentityBlottoGame(**common, payoff_scale=3.0)
    blue = IdentityAction(((10,),))
    raw_response = raw.red_best_response((blue,), (1.0,))
    normalised_response = normalised.red_best_response((blue,), (1.0,))
    assert raw_response.action == normalised_response.action
    assert normalised_response.value == pytest.approx(raw_response.value / 3.0)


def test_infeasible_blue_oracle_reports_candidate_coverage():
    slots = (EngagementSlot(0, 0),)
    game = _game(
        (0,),
        (10, 11),
        slots,
        (((0,),),),
        (((10,),),),
    )
    red = IdentityAction(((0,),))
    with pytest.raises(RuntimeError, match=r"side=Blue.*uncovered_identities=\(11,\)"):
        game.blue_best_response((red,), (1.0,))
