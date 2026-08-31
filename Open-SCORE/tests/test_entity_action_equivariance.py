"""Strong contracts for entity-bound Stage-1 action scores."""

import pytest
import torch

from open_score.contracts import TeamObservation
from open_score.stage1.baselines import VariableEntityActor
from open_score.stage1.entity_qmix import VariableEntityAgent


def _observation(entity_obs, mapping, available=None, entity_mask=None):
    batch, agents, entities, _ = entity_obs.shape
    action_dim = mapping.shape[-1]
    if available is None:
        available = torch.ones(batch, agents, action_dim, dtype=torch.bool)
    if entity_mask is None:
        entity_mask = torch.ones(batch, agents, entities, dtype=torch.bool)
    target_type = torch.zeros_like(mapping)
    target_type[mapping >= 0] = 1
    target_type[..., -1] = torch.where(
        mapping[..., -1] >= 0,
        torch.tensor(3, device=mapping.device),
        target_type[..., -1],
    )
    return TeamObservation(
        entity_obs=entity_obs,
        entity_mask=entity_mask,
        self_obs=torch.randn(batch, agents, 3),
        task_obs=torch.randn(batch, agents, 2),
        agent_mask=torch.ones(batch, agents, dtype=torch.bool),
        avail_actions=available,
        action_entity_index=mapping,
        action_target_type=target_type,
    )


@pytest.mark.parametrize("kind", ["q", "actor"])
@pytest.mark.parametrize("encoder_kind", ["deepset", "saqa"])
def test_target_slots_are_strongly_permutation_equivariant(kind, encoder_kind):
    torch.manual_seed(101)
    entities = torch.randn(1, 1, 4, 5)
    mapping = torch.tensor([[[-1, -1, -1, -1, -1, -1, 1, 2, 3]]])
    observation = _observation(entities, mapping)
    model = (
        VariableEntityAgent(
            5, 3, 2, 9, hidden_dim=16, encoder_kind=encoder_kind, attention_heads=4
        )
        if kind == "q"
        else VariableEntityActor(
            5, 3, 2, 9, hidden_dim=16, encoder_kind=encoder_kind, attention_heads=4
        )
    )
    model.eval()
    original, _ = model(observation)

    # Merely permuting storage and updating the explicit lookup leaves every
    # semantic action unchanged.
    permutation = torch.tensor([2, 0, 3, 1])
    inverse = torch.argsort(permutation)
    remapped = torch.where(mapping >= 0, inverse[mapping.clamp_min(0)], mapping)
    same_semantics = _observation(entities[:, :, permutation], remapped)
    same_semantics.self_obs = observation.self_obs
    same_semantics.task_obs = observation.task_obs
    invariant, _ = model(same_semantics)
    assert torch.allclose(original, invariant, atol=1e-6, rtol=1e-6)

    # Keeping row-based slot lookup fixed while swapping target rows swaps the
    # corresponding action scores; fixed primitive actions remain invariant.
    swapped_rows = entities[:, :, [0, 2, 1, 3]]
    row_bound = _observation(swapped_rows, mapping)
    row_bound.self_obs = observation.self_obs
    row_bound.task_obs = observation.task_obs
    swapped, _ = model(row_bound)
    assert torch.allclose(original[..., :6], swapped[..., :6], atol=1e-6, rtol=1e-6)
    assert torch.allclose(original[..., 6], swapped[..., 7], atol=1e-6, rtol=1e-6)
    assert torch.allclose(original[..., 7], swapped[..., 6], atol=1e-6, rtol=1e-6)
    assert torch.allclose(original[..., 8], swapped[..., 8], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("kind", ["q", "actor"])
@pytest.mark.parametrize("encoder_kind", ["deepset", "saqa"])
def test_target_mask_and_target_scorer_receive_gradients(kind, encoder_kind):
    torch.manual_seed(211)
    entities = torch.randn(1, 1, 3, 5, requires_grad=True)
    mapping = torch.tensor([[[-1, -1, -1, -1, -1, -1, 1, 2]]])
    available = torch.ones(1, 1, 8, dtype=torch.bool)
    available[..., 7] = False
    observation = _observation(entities, mapping, available)
    model = (
        VariableEntityAgent(
            5, 3, 2, 8, hidden_dim=16, encoder_kind=encoder_kind, attention_heads=4
        )
        if kind == "q"
        else VariableEntityActor(
            5, 3, 2, 8, hidden_dim=16, encoder_kind=encoder_kind, attention_heads=4
        )
    )
    scores, _ = model(observation)
    assert scores[..., 7].item() == pytest.approx(-1e9)
    scores[..., 6].sum().backward()
    head = model.q_head if kind == "q" else model.policy_head
    scorer_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in head.target_scorer.parameters()
        if parameter.grad is not None
    )
    assert scorer_grad > 0.0
    assert entities.grad is not None
    assert float(entities.grad[..., 1, :].abs().sum()) > 0.0


@pytest.mark.parametrize("kind", ["q", "actor"])
def test_saqa_padding_is_strictly_ignored(kind):
    torch.manual_seed(307)
    entities = torch.randn(1, 1, 3, 5)
    mapping = torch.tensor([[[-1, -1, -1, -1, -1, -1, 1, 2]]])
    base = _observation(entities, mapping)
    model = (
        VariableEntityAgent(5, 3, 2, 8, hidden_dim=16, encoder_kind="saqa")
        if kind == "q"
        else VariableEntityActor(5, 3, 2, 8, hidden_dim=16, encoder_kind="saqa")
    ).eval()
    reference, _ = model(base)
    padded_entities = torch.cat([entities, torch.randn(1, 1, 4, 5) * 1000], dim=2)
    padded_mask = torch.tensor([[[True, True, True, False, False, False, False]]])
    padded = _observation(padded_entities, mapping, entity_mask=padded_mask)
    padded.self_obs = base.self_obs
    padded.task_obs = base.task_obs
    actual, _ = model(padded)
    assert torch.allclose(reference, actual, atol=1e-6, rtol=1e-6)
