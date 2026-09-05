from __future__ import annotations

import numpy as np

from open_score.envs import HADStage3Adapter
from open_score.stage3 import (
    IdentityAction,
    IdentityBlottoGame,
    apply_identity_joint_plan,
    build_identity_event_game,
    enumerate_coalitions,
    make_engagement_slots,
)
from open_score.stage3.runtime import (
    FrozenStage1GroupExecutor,
    build_observable_threat_patrol,
    observable_threat_patrol,
)
from scripts.evaluate_stage3_identity import _revealed_blue_best_response


class _ConstantPredictor:
    def predict_red_win(self, states, *, rosters, **_kwargs):
        return np.asarray(
            [0.55 + 0.05 * (red - blue) for red, blue in rosters],
            dtype=np.float32,
        ).clip(0.05, 0.95)


class _ToyOracle:
    def evaluate(self, requests):
        return np.asarray(
            [0.4 * len(red) - 0.6 * len(blue) for _, red, blue in requests],
            dtype=np.float64,
        )


def test_observable_patrol_normalises_each_blue_threat_and_spreads_waypoints():
    patrol = observable_threat_patrol(
        reserve_ids=(0, 1, 2),
        red_positions={
            0: (1000.0, 0.0, 0.0),
            1: (1000.0, 0.0, 0.0),
            2: (1000.0, 0.0, 0.0),
        },
        blue_positions={10: (100.0, 900.0, 0.0), 11: (100.0, 800.0, 0.0)},
        target_positions={0: (0.0, -1000.0, 0.0), 1: (0.0, 1000.0, 0.0)},
        active_defenders_by_target={0: 0, 1: 0},
    )
    threat = dict(patrol.target_threat)
    assert np.isclose(sum(threat.values()), 2.0)
    assert threat[1] > threat[0]
    assert set(patrol.target_mapping()) == {0, 1, 2}
    waypoints = list(patrol.waypoint_mapping().values())
    assert all(value.shape == (3,) and np.all(np.isfinite(value)) for value in waypoints)
    # Identical reserve positions must not collapse onto one waypoint when the
    # load-aware allocator sends more than one agent to the same target.
    assert len({tuple(value) for value in waypoints}) == len(waypoints)


def test_reserve_patrol_does_not_read_blue_assignment_or_style():
    class SensitiveState(dict):
        def __getitem__(self, key):
            if key in {"assigned_target", "style", "channel", "subgroup"}:
                raise AssertionError(f"forbidden hidden Blue field read: {key}")
            return super().__getitem__(key)

    class PublicAdapter:
        def agent_states(self, side):
            if side == "Red":
                return {
                    0: {"alive": True, "position": np.asarray([500.0, 0.0, 0.0])},
                    1: {"alive": True, "position": np.asarray([400.0, 0.0, 0.0])},
                }
            return {
                10: SensitiveState(
                    alive=True,
                    position=np.asarray([100.0, 300.0, 0.0]),
                    assigned_target=999,
                    style="secret",
                    channel=123,
                )
            }

        def target_states(self):
            return {
                0: {"alive": True, "position": np.asarray([0.0, -200.0, 0.0])},
                1: {"alive": True, "position": np.asarray([0.0, 200.0, 0.0])},
            }

    patrol = build_observable_threat_patrol(
        PublicAdapter(),
        reserve_ids=(1,),
        active_red_assignment={0: 0, 1: None},
    )
    assert patrol.reserve_ids == (1,)
    assert set(patrol.waypoint_mapping()) == {1}


def test_grounded_all_reserve_keeps_none_assignment_and_executor_patrols():
    adapter = HADStage3Adapter(
        4,
        3,
        2,
        max_steps=5,
        target_positions=[[-2100.0, -300.0, 100.0], [-2100.0, 300.0, 100.0]],
    )
    adapter.reset(seed=91)
    built = build_identity_event_game(
        adapter, _ConstantPredictor(), full_domain_agent_threshold=8
    )
    red = IdentityAction(
        tuple(() for _ in built.slots), reserve_ids=built.red_alive_ids
    )
    plan = apply_identity_joint_plan(
        adapter,
        built.game,
        red,
        built.initial_blue[0],
        blue_type_name="rush",
    )
    assert plan.red_reserve_ids == built.red_alive_ids
    assert all(value is None for value in dict(plan.red_assignment).values())
    assert all(value is None for value in adapter.red_assignment.values())

    patrol = build_observable_threat_patrol(
        adapter, plan.red_reserve_ids, dict(plan.red_assignment)
    )
    executor = FrozenStage1GroupExecutor(None, "cpu", micro_grouping="planned_groups")
    actions = executor.act(
        adapter,
        roster_override={},
        reserve_waypoints=patrol.waypoint_mapping(),
    )
    assert set(actions) == set(adapter.red_ids)
    for agent_id, action in actions.items():
        direction = adapter.action_vectors[action]
        destination = patrol.waypoint_mapping()[agent_id]
        position = adapter.agent_states("Red")[agent_id]["position"]
        assert float(np.dot(direction, destination - position)) >= 0.0
    assert all(value is None for value in adapter.red_assignment.values())


def test_revealed_blue_response_reuses_same_game_and_full_one_to_four_domain():
    red_ids = (0, 1, 2, 3, 4)
    blue_ids = (10, 11, 12, 13)
    slots = make_engagement_slots((0,), 5)
    red_groups = enumerate_coalitions(red_ids, 4)
    blue_groups = enumerate_coalitions(blue_ids, 4)
    game = IdentityBlottoGame(
        red_ids,
        blue_ids,
        slots,
        tuple(red_groups for _ in slots),
        tuple(blue_groups for _ in slots),
        _ToyOracle(),
        allow_red_reserve=True,
        allow_blue_reserve=False,
        full_coalition_domain=True,
    )
    blue = game.validate_action(
        IdentityAction(((10, 11, 12, 13),) + ((),) * (len(slots) - 1)),
        "Blue",
    )
    response = _revealed_blue_best_response(game, blue)
    assert response.optimal
    assert game.validate_action(response.action, "Red") == response.action
    active = [value for group in response.action.coalitions for value in group]
    assert set(active).union(response.action.reserve_ids) == set(red_ids)
    assert not set(active).intersection(response.action.reserve_ids)
    assert max(map(len, response.action.coalitions)) <= 4


def test_former_failure_seed_revealed_response_is_feasible_without_domain_rebuild():
    adapter = HADStage3Adapter(
        18,
        12,
        2,
        max_steps=5,
        target_positions=[[-2100.0, -300.0, 100.0], [-2100.0, 300.0, 100.0]],
    )
    adapter.reset(seed=20_360_907)
    built = build_identity_event_game(
        adapter,
        _ConstantPredictor(),
        full_domain_agent_threshold=8,
        neighborhood_size=6,
        peer_count=3,
    )
    assert built.game.payoff_scale == (18 + 12 + 1) * 12
    blue = built.initial_blue[0]
    response = _revealed_blue_best_response(built.game, blue)
    assert response.optimal
    assert built.game.validate_action(response.action, "Red") == response.action
    assert built.game.validate_action(blue, "Blue") == blue
    assert max(map(len, response.action.coalitions)) <= 4
