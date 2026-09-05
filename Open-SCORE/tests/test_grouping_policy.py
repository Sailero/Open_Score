"""Legality and probability contracts of the executable macro-action policy."""
import copy
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from open_score.grouping.domain import DecisionState, Entity, Group, Grouping
from open_score.grouping.policy import GroupingPolicy


@pytest.fixture(autouse=True)
def small_torch_runtime():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(41)
    yield
    torch.set_num_threads(threads)


def make_state(size=6, step=5):
    red = tuple(Entity(i, (-900.0 + i * 300, -500.0 + i * 170, 100.0),
                       (-100.0 + i * 8, i * 3.0, 0.0), 1.0) for i in range(size))
    blue = tuple(Entity(100 + i, (1200.0, i * 200.0, 100.0), (-150.0, 0.0, 0.0), 1.0)
                 for i in range(3))
    targets = (Entity(0, (-2100.0, -600.0, 100.0), (0.0, 0.0, 0.0), 1.2),
               Entity(1, (-2100.0, 600.0, 100.0), (0.0, 0.0, 0.0), 1.2))
    previous = (Grouping((), tuple(range(size))) if step == 0 else
                Grouping(tuple(Group((index // 4) % 2, tuple(range(index, min(index + 4, size))))
                               for index in range(0, size, 4))))
    return DecisionState(step, 50, "reactive", red, blue, targets, previous,
                         {i: (0.0,) * 64 for i in range(size)}, {i: -1 for i in range(size)})


def policy(mode="selective"):
    return GroupingPolicy(mode, hidden_dim=32, heads=4, layers=2)


@pytest.mark.parametrize("mode", ["selective", "full", "random", "rule"])
@pytest.mark.parametrize("step", [0, 5])
def test_live_actions_are_legal_and_rollout_probabilities_replay_exactly(mode, step):
    state, model = make_state(step=step), policy(mode)
    model.eval()
    with torch.no_grad():
        decision = model.act(state, rng=np.random.default_rng(12), release_count=2)
    assert decision.action.validate(state.ids("red"), state.ids("targets")) == decision.action
    assert len(decision.released_ids) == (6 if step == 0 else 2)
    assert len(set(decision.released_ids)) == len(decision.released_ids)
    trace = json.loads(json.dumps(decision.trace))
    model.train()
    log_prob, entropy, value = model.evaluate_action(state, trace)
    torch.testing.assert_close(log_prob, decision.log_prob, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(entropy, decision.entropy, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(value, decision.value, atol=1e-6, rtol=1e-6)
    (-log_prob + value.square()).backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
    assert model.repair_query[0].weight.grad is not None
    if step == 0 or mode != "selective":
        assert model.selection_query[0].weight.grad is None
    else:
        assert model.selection_query[0].weight.grad is not None


def test_stop_is_a_probability_factor_and_leaves_pruned_groups_intact():
    state, model = make_state(), policy()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.stop_head.bias.fill_(1.0)
    state = replace(state, red=(replace(state.red[0], health=0.0),) + state.red[1:])
    decision = model.act(state, deterministic=True)
    assert decision.released_ids == ()
    assert decision.trace["selection"] == [None]
    assert decision.action == state.previous.prune(state.ids("red"))
    expected = torch.log_softmax(torch.tensor([0.0] * 5 + [1.0]), dim=0)[-1]
    torch.testing.assert_close(decision.log_prob, expected)
    torch.testing.assert_close(model.evaluate_action(state, decision.trace)[0], expected)


def test_initial_complete_construction_log_probability_is_a_sum_not_mean():
    state, model = make_state(size=2, step=0), policy("full")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    decision = model.act(state, deterministic=True, rng=np.random.default_rng(10))
    # First member: new target 0, new target 1, reserve. Second: join plus those three.
    torch.testing.assert_close(decision.log_prob, torch.tensor(-np.log(3.0) - np.log(4.0), dtype=torch.float32))
    assert decision.trace["selection"] == []
    assert len(decision.action.groups) == 1
    assert set(decision.action.groups[0].members) == {0, 1}


def test_capacity_mask_allows_multiple_groups_at_one_target():
    state, model = make_state(size=10, step=0), policy("full")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    decision = model.act(state, deterministic=True, rng=np.random.default_rng(1))
    assert [len(group.members) for group in decision.action.groups] == [4, 4, 2]
    assert {group.target for group in decision.action.groups} == {0}
    assert decision.action.reserve == ()
    model.evaluate_action(state, decision.trace)


def test_random_distribution_is_frozen_and_round_trips_configuration():
    state, model = make_state(), policy("random")
    model.set_release_distribution({"3": 20})
    decision = model.act(state, rng=np.random.default_rng(4))
    assert len(decision.released_ids) == 3
    reloaded = GroupingPolicy(**json.loads(json.dumps(model.config)))
    reloaded.load_state_dict(model.state_dict())
    torch.testing.assert_close(reloaded.evaluate_action(state, decision.trace)[0], decision.log_prob)
    assert all(isinstance(value, torch.Tensor) for value in model.state_dict().values())
    model.set_release_distribution({99: 1.0})
    assert len(model.act(state).released_ids) == len(state.red)
    with pytest.raises(ValueError):
        model.set_release_distribution({0: 0.0})


def test_rule_chooses_underfilled_group_then_nearby_reserve():
    state, model = make_state(size=8), policy("rule")
    state = replace(state, previous=Grouping((Group(0, (0, 1)), Group(1, (2, 3, 4, 5))), (6, 7)))
    decision = model.act(state)
    assert set(decision.released_ids) == {0, 1, 6, 7}
    assert set(decision.released_ids[:2]) == {0, 1}
    paired = model.act(state, release_count=3)
    assert len(paired.released_ids) == 3
    assert set(paired.released_ids[:2]) == {0, 1}


def test_zero_count_is_an_exogenous_noop_even_for_selective():
    state, model = make_state(), policy()
    decision = model.act(state, release_count=0)
    assert decision.action == state.previous
    assert decision.log_prob.item() == 0.0
    assert decision.entropy.item() == 0.0
    assert decision.trace["selection"] == []
    model.evaluate_action(state, decision.trace)


def test_identity_numbers_and_entity_order_are_not_network_features():
    state, model = make_state(), policy("full")
    with torch.no_grad():
        decision = model.act(state, rng=np.random.default_rng(7))
    permuted = replace(state, red=tuple(reversed(state.red)), blue=tuple(reversed(state.blue)),
                       targets=tuple(reversed(state.targets)))
    torch.testing.assert_close(model.value(state), model.value(permuted), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(model.evaluate_action(permuted, decision.trace)[0], decision.log_prob, atol=3e-6, rtol=3e-6)
    mapping = {i: 1000 + 7 * (len(state.red) - i) for i in state.ids("red")}
    renamed = replace(state,
        red=tuple(replace(entity, id=mapping[entity.id]) for entity in state.red),
        previous=Grouping(tuple(Group(group.target, tuple(mapping[i] for i in group.members)) for group in state.previous.groups),
                          tuple(mapping[i] for i in state.previous.reserve)),
        memory={mapping[i]: row for i, row in state.memory.items()},
        last_actions={mapping[i]: value for i, value in state.last_actions.items()})
    torch.testing.assert_close(model.value(state), model.value(renamed), atol=2e-6, rtol=2e-6)


def test_old_teammate_relation_and_lower_memory_affect_encoding():
    state, model = make_state(size=4), policy()
    first = replace(state, previous=Grouping((Group(0, (0, 1)), Group(0, (2, 3)))))
    second = replace(state, previous=Grouping((Group(0, (0, 2)), Group(0, (1, 3)))))
    assert not torch.allclose(model._encode(first).red[0], model._encode(second).red[0], atol=1e-7, rtol=1e-7)
    changed_memory = replace(first, memory={**first.memory, 0: (0.5,) * 64}, last_actions={**first.last_actions, 0: 26})
    assert not torch.allclose(model.value(first), model.value(changed_memory), atol=1e-7, rtol=1e-7)


def test_malformed_trace_is_rejected_before_ppo_update():
    state, model = make_state(), policy()
    decision = model.act(state, release_count=2)
    duplicate = copy.deepcopy(decision.trace)
    duplicate["released_ids"][1] = duplicate["released_ids"][0]
    with pytest.raises(ValueError, match="duplicate"):
        model.evaluate_action(state, duplicate)
    missing = copy.deepcopy(decision.trace)
    missing["repairs"].pop()
    with pytest.raises(ValueError, match="every released"):
        model.evaluate_action(state, missing)
    illegal = copy.deepcopy(decision.trace)
    illegal["repairs"][0] = {"agent": illegal["released_ids"][0], "kind": "new", "target": 999}
    with pytest.raises(ValueError, match="legal actions"):
        model.evaluate_action(state, illegal)


def test_policy_gradient_updates_selection_and_repair_on_fixed_macro_trace():
    state, model = make_state(), policy()
    with torch.no_grad():
        decision = model.act(state, release_count=3)
    before = model.evaluate_action(state, decision.trace)[0].item()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for _ in range(3):
        optimizer.zero_grad()
        log_prob, _, _ = model.evaluate_action(state, decision.trace)
        (-log_prob).backward()
        optimizer.step()
    assert model.evaluate_action(state, decision.trace)[0].item() > before


def test_terminal_value_is_defined_but_terminal_action_is_rejected():
    state, model = make_state(), policy()
    terminal = replace(state, red=tuple(replace(entity, health=0.0) for entity in state.red),
                       blue=(), targets=(), step=state.max_steps)
    assert torch.isfinite(model.value(terminal))
    with pytest.raises(ValueError, match="terminal"):
        model.act(terminal)


@pytest.mark.parametrize("mode", ["selective", "full", "random", "rule"])
def test_all_red_dead_is_a_noop_until_native_task_termination(mode):
    state, model = make_state(), policy(mode)
    state = replace(state, red=tuple(replace(entity, health=0.0) for entity in state.red))
    decision = model.act(state)
    assert decision.action == Grouping(())
    assert decision.released_ids == ()
    assert decision.log_prob.item() == decision.entropy.item() == 0.0
    log_prob, entropy, value = model.evaluate_action(state, decision.trace)
    assert log_prob.requires_grad and entropy.requires_grad and torch.isfinite(value)
    torch.testing.assert_close(value, decision.value)


def test_default_architecture_replays_an_unseen_larger_roster():
    state, model = make_state(size=16, step=0), GroupingPolicy()
    assert model.hidden_dim == 128 and len(model.encoder.layers) == 2
    assert all(layer.self_attn.num_heads == 4 and layer.dropout.p == 0.0 for layer in model.encoder.layers)
    model.eval()
    with torch.no_grad():
        decision = model.act(state, rng=np.random.default_rng(11))
    model.train()
    replay_log_prob, _, replay_value = model.evaluate_action(state, decision.trace)
    torch.testing.assert_close(replay_log_prob, decision.log_prob, atol=1e-5, rtol=1e-6)
    torch.testing.assert_close(replay_value, decision.value, atol=1e-6, rtol=1e-6)
    decision.action.validate(state.ids("red"), state.ids("targets"))


@pytest.mark.parametrize("count", [-1, 7, 1.2])
def test_release_count_rejects_invalid_diagnostic_override(count):
    with pytest.raises(ValueError, match="release_count"):
        policy().act(make_state(), release_count=count)
