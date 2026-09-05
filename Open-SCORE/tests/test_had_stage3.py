import copy

import numpy as np
import pytest

from open_score.envs import (
    HADCommandedRuleController,
    HADStage1Adapter,
    HADStage3Adapter,
)
from open_score.stage1.runner import RuleBasedController


TARGETS = (
    (-2100.0, -600.0, 150.0),
    (-2100.0, 600.0, 150.0),
)


def make_adapter(*, max_steps=20, style="rush"):
    return HADStage3Adapter(
        3,
        2,
        2,
        max_steps=max_steps,
        target_positions=TARGETS,
        blue_rule_style=style,
    )


def numeric_signature(adapter):
    return np.asarray(
        [
            list(np.asarray(entity.position, dtype=np.float64))
            + list(np.asarray(entity.velocity, dtype=np.float64))
            + [float(entity.Health), float(getattr(entity, "IsFire", False))]
            for entity in adapter.env.world
        ],
        dtype=np.float64,
    )


def test_ids_assignments_and_exported_command_state_are_stable():
    adapter = make_adapter()
    state = adapter.reset(seed=11)

    assert adapter.red_ids == (0, 1, 2)
    assert adapter.blue_ids == (3, 4)
    assert adapter.target_ids == (0, 1)
    assert adapter.red_assignment == {0: 0, 1: 1, 2: 0}
    assert adapter.blue_assignment == {3: 0, 4: 1}
    assert set(state["red"]) == {0, 1, 2}
    assert set(state["blue"]) == {3, 4}
    assert set(state["targets"]) == {0, 1}
    assert state["targets"][0]["entity_id"] == 5
    np.testing.assert_allclose(state["targets"][0]["position"], TARGETS[0])

    events = adapter.set_joint_assignments(
        {0: 1, 1: 1, 2: None},
        {3: 1, 4: 1},
    )
    assert {event.side for event in events} == {"Red", "Blue"}
    assert adapter.assigned_ids("Red", 1) == (0, 1)
    assert adapter.red_assignment[2] is None
    assert adapter.global_state()["assignments"] == adapter.assignments


def test_assignment_validation_rejects_wrong_side_and_dead_target():
    adapter = make_adapter()
    adapter.reset(seed=12)
    with pytest.raises(ValueError, match="non-Red"):
        adapter.set_assignments("Red", {3: 0})
    with pytest.raises(ValueError, match="2 entries"):
        adapter.set_assignments("Blue", [0])
    with pytest.raises(ValueError, match="target_id"):
        adapter.set_assignments("Blue", [0, 7])
    adapter.env.targets[1].Health = 0.0
    with pytest.raises(ValueError, match="destroyed target"):
        adapter.set_assignments("Blue", [1, 0])


def test_commanded_blue_rule_uses_upper_level_target_assignment():
    adapter = make_adapter()
    adapter.reset(seed=13)
    for blue in adapter.env.blue_agents:
        blue.position = [-1000.0, 0.0, 150.0]
        blue.velocity = [-20.0, 0.0, 0.0]
    adapter.set_assignments("Blue", {3: 0, 4: 1})

    actions = adapter.commanded_rule_actions("Blue", style="rush")
    directions = adapter.action_vectors[actions]
    desired_0 = np.asarray(TARGETS[0]) - np.asarray([-1000.0, 0.0, 150.0])
    desired_1 = np.asarray(TARGETS[1]) - np.asarray([-1000.0, 0.0, 150.0])
    assert float(np.dot(directions[0], desired_0)) > 0.0
    assert float(np.dot(directions[1], desired_1)) > 0.0
    assert directions[0, 1] < 0.0
    assert directions[1, 1] > 0.0

    controller = HADCommandedRuleController("rush")
    np.testing.assert_array_equal(
        controller.act(adapter, "Blue"),
        actions,
    )


@pytest.mark.parametrize("style", ["rush", "split_rush"])
def test_single_target_blue_rule_is_exactly_round01_compatible(style):
    reference = HADStage1Adapter(4, 3, max_steps=50)
    reference.reset(seed=131)
    target_position = np.asarray(reference.env.targets[0].position).copy()
    adapter = HADStage3Adapter(
        4,
        3,
        1,
        max_steps=50,
        target_positions=[target_position],
        blue_rule_style=style,
    )
    adapter.reset(seed=999)
    for source, destination in zip(reference.env.world, adapter.env.world):
        destination.__dict__.clear()
        destination.__dict__.update(copy.deepcopy(source.__dict__))
    adapter.set_assignments("Blue", {agent_id: 0 for agent_id in adapter.blue_ids})

    expected = RuleBasedController(style).act(
        reference,
        "Blue",
        reference.observe("Blue"),
        np.random.default_rng(7),
    )
    actual = adapter.commanded_rule_actions("Blue", style=style)
    np.testing.assert_array_equal(actual, expected)


def test_large_split_rush_group_keeps_every_lane_inside_attack_footprint():
    adapter = make_adapter(style="split_rush")
    # The largest directly collected S2 roster remains bit-for-bit compatible.
    np.testing.assert_allclose(
        [adapter._split_lateral_offset(index, 6) for index in range(6)],
        [(index - 2.5) * 180.0 for index in range(6)],
    )
    large = [adapter._split_lateral_offset(index, 20) for index in range(20)]
    assert max(abs(value) for value in large) == pytest.approx(450.0)
    assert all(left < right for left, right in zip(large, large[1:]))


@pytest.mark.parametrize("style", ["rush", "split_rush"])
def test_single_target_stage3_physics_matches_round01_rule_rollout(style):
    reference = HADStage1Adapter(3, 2, max_steps=50)
    reference.reset(seed=141)
    target_position = np.asarray(reference.env.targets[0].position).copy()
    adapter = HADStage3Adapter(
        3,
        2,
        1,
        max_steps=50,
        target_positions=[target_position],
        blue_rule_style=style,
    )
    adapter.reset(seed=777)
    for source, destination in zip(reference.env.world, adapter.env.world):
        destination.__dict__.clear()
        destination.__dict__.update(copy.deepcopy(source.__dict__))
    adapter.set_joint_assignments(
        {agent_id: 0 for agent_id in adapter.red_ids},
        {agent_id: 0 for agent_id in adapter.blue_ids},
    )
    rule = RuleBasedController(style)
    rule.reset()
    rng = np.random.default_rng(9)

    for _ in range(8):
        expected_blue = rule.act(
            reference, "Blue", reference.observe("Blue"), rng
        )
        actual_blue = adapter.commanded_rule_actions("Blue", style=style)
        np.testing.assert_array_equal(actual_blue, expected_blue)
        red_actions = np.asarray([0, 1, 2], dtype=np.int64)
        _, _, reference_done, _ = reference.step(red_actions, expected_blue)
        _, _, adapter_done, _ = adapter.step(red_actions, blue_style=style)
        np.testing.assert_allclose(
            numeric_signature(adapter),
            np.asarray(
                [
                    list(np.asarray(entity.position, dtype=np.float64))
                    + list(np.asarray(entity.velocity, dtype=np.float64))
                    + [float(entity.Health), float(getattr(entity, "IsFire", False))]
                    for entity in reference.env.world
                ],
                dtype=np.float64,
            ),
            rtol=0.0,
            atol=0.0,
        )
        assert adapter_done == reference_done
        if reference_done:
            break


def test_local_observation_is_exactly_stage1_compatible_for_selected_world():
    reference = HADStage1Adapter(3, 2, max_steps=20)
    reference.reset(seed=17)
    adapter = make_adapter(max_steps=20)
    adapter.reset(seed=99)

    # Install the same physical single-target micro-world in the multi-target
    # adapter.  The unrelated second target must not leak into policy input.
    for source, destination in zip(reference.env.red_agents, adapter.env.red_agents):
        destination.__dict__.clear()
        destination.__dict__.update(copy.deepcopy(source.__dict__))
    for source, destination in zip(reference.env.blue_agents, adapter.env.blue_agents):
        destination.__dict__.clear()
        destination.__dict__.update(copy.deepcopy(source.__dict__))
    adapter.env.targets[0].__dict__.clear()
    adapter.env.targets[0].__dict__.update(copy.deepcopy(reference.env.targets[0].__dict__))
    adapter.env.update_alive_agents()

    expected = reference.observe("Red")
    actual = adapter.local_observation(
        "Red",
        target_id=0,
        red_ids=adapter.red_ids,
        blue_ids=adapter.blue_ids,
        local_step=0,
    )
    for key in (
        "entity_obs",
        "entity_mask",
        "self_obs",
        "task_obs",
        "agent_mask",
        "avail_actions",
        "state_entities",
        "state_mask",
    ):
        np.testing.assert_allclose(actual[key], expected[key], rtol=0.0, atol=0.0)
    assert actual["entity_obs"].shape == (3, 6, 12)
    assert actual["state_entities"].shape == (6, 12)
    np.testing.assert_array_equal(actual["agent_ids"], [0, 1, 2])
    np.testing.assert_array_equal(actual["entity_ids"], [0, 1, 2, 3, 4, 5])


def test_zero_red_allocation_has_well_formed_model_view():
    adapter = make_adapter()
    adapter.reset(seed=18)
    observation = adapter.local_observation(
        "Red", target_id=0, red_ids=[], blue_ids=[3], local_step=4
    )
    assert observation["entity_obs"].shape == (0, 2, 12)
    assert observation["self_obs"].shape == (0, 10)
    assert observation["task_obs"].shape == (0, 7)
    assert observation["avail_actions"].shape == (0, 27)
    assert observation["state_entities"].shape == (2, 12)


def test_native_had_firing_is_not_silently_gated_by_upper_assignment():
    adapter = make_adapter()
    adapter.reset(seed=19)
    target = adapter.env.targets[0]
    # Blue 3 is commanded toward target 1 but is physically next to target 0.
    # Native HAD firing sees the physical world, so upper assignment must not
    # rewrite the simulator's fire/damage equation.
    adapter.set_assignments("Blue", {3: 1, 4: 1})
    attacker = adapter.env.blue_agents[0]
    attacker.position = (
        np.asarray(target.position) + np.asarray([1.0, 0.0, 0.0])
    ).tolist()
    attacker.velocity = [20.0, 0.0, 0.0]
    adapter.env.blue_agents[1].position = [2000.0, 1000.0, 500.0]
    for red in adapter.env.red_agents:
        red.position = [1500.0, -1000.0, 500.0]

    _, rewards, done, info = adapter.step([0, 0, 0])
    assert done
    assert info["outcome_red"] == -1.0
    assert rewards == {"Red": -1.0, "Blue": 1.0}
    assert target.Health == 0.0
    assert attacker.IsFire


def test_native_had_fire_can_damage_colocated_objectives():
    adapter = make_adapter()
    adapter.reset(seed=20)
    first, second = adapter.env.targets
    # Put both objectives in the same firing footprint. Native HAD applies the
    # firing attack to both; registered v2 layouts prevent such overlap through
    # geometry rather than by changing the physics engine.
    second.position = list(first.position)
    adapter.set_assignments("Blue", {3: 0, 4: 1})
    attacker = adapter.env.blue_agents[0]
    attacker.position = (
        np.asarray(first.position) + np.asarray([1.0, 0.0, 0.0])
    ).tolist()
    attacker.velocity = [20.0, 0.0, 0.0]
    adapter.env.blue_agents[1].position = [2000.0, 1000.0, 500.0]
    for red in adapter.env.red_agents:
        red.position = [1500.0, -1000.0, 500.0]

    adapter.step([0, 0, 0])
    assert first.Health == 0.0
    assert second.Health == 0.0


def test_snapshot_restore_replays_physics_assignments_and_events_exactly():
    adapter = make_adapter(max_steps=10, style="split_rush")
    adapter.reset(seed=23)
    adapter.set_joint_assignments([0, 0, 1], [1, 0])
    adapter.step([0, 1, 2])
    snapshot = adapter.snapshot()

    _, rewards_a, done_a, info_a = adapter.step([3, 4, 5])
    signature_a = numeric_signature(adapter)
    adapter.set_assignments("Blue", [0, 0])

    restored = adapter.restore(snapshot)
    assert restored["assignments"]["Blue"] == {3: 1, 4: 0}
    _, rewards_b, done_b, info_b = adapter.step([3, 4, 5])
    signature_b = numeric_signature(adapter)

    np.testing.assert_allclose(signature_a, signature_b, rtol=0.0, atol=0.0)
    assert rewards_a == rewards_b
    assert done_a == done_b
    assert info_a["events"] == info_b["events"]
    assert info_a["blue_actions"] == info_b["blue_actions"]


def test_horizon_is_an_explicit_event_and_defender_success():
    adapter = make_adapter(max_steps=1)
    adapter.reset(seed=29)
    # Keep all teams well separated for this administrative one-step horizon.
    for index, red in enumerate(adapter.env.red_agents):
        red.position = [-1000.0, -1000.0 + index * 100.0, 500.0]
    for index, blue in enumerate(adapter.env.blue_agents):
        blue.position = [2000.0, 800.0 + index * 100.0, 500.0]

    _, rewards, done, info = adapter.step([0, 0, 0], [0, 0])
    assert done
    assert info["truncated"] and not info["terminated"]
    assert rewards == {"Red": 1.0, "Blue": -1.0}
    assert [event["kind"] for event in info["events"]] == ["horizon_survived"]


def test_target_membership_is_fixed_for_the_complete_episode():
    positions = [
        [-2100.0, -800.0, 100.0],
        [-2100.0, 0.0, 100.0],
        [-2100.0, 800.0, 100.0],
    ]
    adapter = HADStage3Adapter(4, 4, 3, max_steps=20, target_positions=positions)
    adapter.reset(seed=91)
    assert adapter.target_ids == (0, 1, 2)
    assert adapter.all_target_ids == (0, 1, 2)

    assert adapter.set_active_targets([0, 1, 2]) == ()
    with pytest.raises(ValueError, match="fixed"):
        adapter.set_active_targets([0, 1])
    with pytest.raises(ValueError, match="fixes every configured target"):
        adapter.reset(seed=92, active_target_ids=[0])
    assert adapter.target_ids == (0, 1, 2)
    assert set(adapter.target_states()) == {0, 1, 2}
