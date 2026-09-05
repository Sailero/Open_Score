"""Behavioral contracts for bounded proposals and genuinely batched scoring."""
import copy
from dataclasses import replace

import pytest
import torch

from open_score.grouping.domain import DecisionState, Entity, Group, Grouping
from open_score.overnight.actions import candidate_actions, compact_action, rule_action
from open_score.overnight.policy import CandidateNetwork


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(51)
    yield
    torch.set_num_threads(previous)


def state(size=8):
    red = tuple(Entity(i, (-800 + 97 * i, -600 + 73 * i, 100), (30 + i, 0, 0), 1) for i in range(size))
    blue = tuple(Entity(100 + i, (1800 - 71 * i, 500 - 87 * i, 100), (-100, 0, 0), 1) for i in range(size))
    targets = (Entity(0, (-2100, -600, 100), (0, 0, 0), 1.2), Entity(1, (-2100, 600, 100), (0, 0, 0), 1.2))
    groups = tuple(Group((i // 4) % 2, tuple(range(i, min(size, i + 4)))) for i in range(0, size, 4))
    return DecisionState(5, 50, 'reactive', red, blue, targets, Grouping(groups),
                         {i: tuple(.001 * (i + j) for j in range(64)) for i in range(size)},
                         {i: i % 27 for i in range(size)})


@pytest.mark.parametrize('size', [1, 4, 8, 16, 32, 64])
def test_proposals_and_batched_forward_at_variable_scales(size):
    current = state(size)
    proposals = candidate_actions(current)
    assert len(proposals) <= 24 and len(proposals) == len(set(proposals))
    assert proposals[0] == current.previous
    assert proposals == candidate_actions(current)
    assert rule_action(current) in proposals
    assert compact_action(current) in proposals[:3]
    for proposal in proposals:
        proposal.validate(current.ids('red'), current.ids('targets'))
    model = CandidateNetwork(hidden_dim=32, heads=4, layers=1)
    scores, values = model([current], [proposals])
    assert scores.shape == (1, len(proposals)) and values.shape == (1,)
    assert torch.isfinite(scores).all() and torch.isfinite(values).all()


def test_degenerate_keep_retained_but_new_rules_deploy_compact_groups():
    current = state(12)
    current = replace(current, previous=Grouping((), current.ids('red')))
    proposals = candidate_actions(current)
    assert proposals[0] == current.previous
    assert any(proposal.reserve and proposal.groups for proposal in proposals)
    for proposal in proposals[1:]:
        assert proposal.groups
        assert len(proposal.reserve) <= len(current.red) // 8
        assert all(2 <= len(group.members) <= 4 for group in proposal.groups)


def test_initial_placeholder_cannot_be_selected_as_all_reserve():
    current = state(16)
    current = replace(current, step=0, previous=Grouping((), current.ids('red')))
    proposals = candidate_actions(current)
    assert current.previous not in proposals
    assert proposals[0] == rule_action(current)
    assert all(sum(len(group.members) for group in proposal.groups) >= 14 for proposal in proposals)


def test_compact_rule_keeps_audited_per_target_tail_singletons():
    current = state(10)
    compact = compact_action(current)
    assert sorted(len(group.members) for group in compact.groups) == [1, 1, 4, 4]
    compact.validate(current.ids('red'), current.ids('targets'))
    assert compact in candidate_actions(current, 8)[:3]


def test_padding_batch_and_single_scores_and_gradients_agree():
    states = [state(4), state(12)]
    pools = [candidate_actions(states[0], 3), candidate_actions(states[1], 9)]
    batched = CandidateNetwork(hidden_dim=32, heads=4, layers=2)
    single = copy.deepcopy(batched)
    scores, values = batched(states, pools)
    loss = sum(scores[b, :len(pool)].square().sum() for b, pool in enumerate(pools)) + values.square().sum()
    loss.backward()
    for b, (current, pool) in enumerate(zip(states, pools)):
        one_scores, one_value = single([current], [pool])
        torch.testing.assert_close(scores[b, :len(pool)], one_scores[0], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(values[b], one_value[0], atol=2e-6, rtol=2e-5)
        (one_scores.square().sum() + one_value.square().sum()).backward()
    assert torch.all(scores[0, len(pools[0]):] == -1e9)
    for left, right in zip(batched.parameters(), single.parameters()):
        if left.grad is not None:
            assert torch.isfinite(left.grad).all()
            torch.testing.assert_close(left.grad, right.grad, atol=1e-5, rtol=2e-4)
    assert batched.encoder.layers[0].self_attn.in_proj_weight.grad.abs().sum() > 0


def test_scores_invariant_to_identity_renaming_entity_and_candidate_order():
    current = state(8)
    pools = candidate_actions(current, 10)
    model = CandidateNetwork(hidden_dim=32, heads=4, layers=2).eval()
    rename_red = {entity.id: 1000 - 3 * entity.id for entity in current.red}
    rename_target = {entity.id: 70 - entity.id for entity in current.targets}
    def rename_action(action):
        return Grouping(tuple(Group(rename_target[g.target], tuple(rename_red[i] for i in g.members)) for g in action.groups),
                        tuple(rename_red[i] for i in action.reserve))
    changed = replace(current,
                      red=tuple(replace(entity, id=rename_red[entity.id]) for entity in reversed(current.red)),
                      blue=tuple(replace(entity, id=9000 + entity.id) for entity in reversed(current.blue)),
                      targets=tuple(replace(entity, id=rename_target[entity.id]) for entity in reversed(current.targets)),
                      previous=rename_action(current.previous),
                      memory={rename_red[i]: values for i, values in current.memory.items()},
                      last_actions={rename_red[i]: value for i, value in current.last_actions.items()})
    with torch.no_grad():
        scores, values = model([current, changed], [pools, [rename_action(action) for action in reversed(pools)]])
    torch.testing.assert_close(scores[0], scores[1].flip(0), atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(values[0], values[1], atol=2e-6, rtol=2e-5)


def test_network_uses_membership_and_frozen_controller_state():
    current = state(8)
    pool = candidate_actions(current, 8)
    changed = replace(current, memory={i: tuple(1.0 for _ in range(64)) for i in current.ids('red')},
                      last_actions={i: 26 for i in current.ids('red')})
    model = CandidateNetwork(hidden_dim=32, heads=4, layers=1).eval()
    scores, values = model([current, changed], [pool, pool])
    assert (scores[0] - scores[1]).abs().max() > 1e-5
    assert (values[0] - values[1]).abs().max() > 1e-5
    different = Grouping((Group(0, (0, 2, 4, 6)), Group(1, (1, 3, 5, 7))))
    scores, _ = model([current], [[current.previous, different]])
    assert (scores[0, 0] - scores[0, 1]).abs() > 1e-6


def test_full_network_is_batched_and_has_expected_capacity():
    model = CandidateNetwork()
    count = sum(parameter.numel() for parameter in model.parameters())
    assert 2_000_000 < count < 4_000_000
    current = state(16)
    scores, values = model([current, current], [candidate_actions(current, 8)] * 2)
    assert scores.shape == (2, 8)
    (scores.sum() + values.sum()).backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
