import copy

import numpy as np
import torch

from open_score.contracts import GlobalState, TeamObservation
from open_score.envs import HADStage1Adapter, tensorize_had_observation
from open_score.stage1 import VariableScaleQMIX
from open_score.stage2 import BilateralOutcomeModel, selection_focused_loss
from open_score.stage3 import enumerate_count_allocations, solve_defender_minimax
from open_score.stage4 import guarded_lower_update, stability_regularized_loss


def random_contract(batch=2, agents=4, entities=7):
    observation = TeamObservation(
        entity_obs=torch.randn(batch, agents, entities, 12),
        entity_mask=torch.ones(batch, agents, entities, dtype=torch.bool),
        self_obs=torch.randn(batch, agents, 10),
        task_obs=torch.randn(batch, agents, 6),
        agent_mask=torch.ones(batch, agents, dtype=torch.bool),
        avail_actions=torch.ones(batch, agents, 9, dtype=torch.bool),
    )
    state = GlobalState(
        entities=torch.randn(batch, entities, 11),
        entity_mask=torch.ones(batch, entities, dtype=torch.bool),
    )
    return observation, state


def make_model():
    return VariableScaleQMIX(12, 10, 6, 11, 9, agent_hidden_dim=32, mixer_hidden_dim=24, mixing_dim=12)


def test_stage1_ignores_padded_entities():
    torch.manual_seed(1)
    model = make_model().eval()
    observation, _ = random_contract(batch=1, agents=3, entities=5)
    base_q, _ = model.agent_q(observation)
    padded = copy.deepcopy(observation)
    padded.entity_obs = torch.cat([observation.entity_obs, torch.randn(1, 3, 4, 12)], dim=2)
    padded.entity_mask = torch.cat(
        [observation.entity_mask, torch.zeros(1, 3, 4, dtype=torch.bool)], dim=2
    )
    padded_q, _ = model.agent_q(padded)
    assert torch.allclose(base_q, padded_q, atol=1e-6)


def test_stage1_mixer_is_monotonic_for_active_agents():
    torch.manual_seed(2)
    model = make_model().eval()
    observation, state = random_contract(batch=2, agents=5, entities=8)
    observation.agent_mask[:, -1] = False
    chosen_q = torch.randn(2, 5, requires_grad=True)
    total = model.mix(chosen_q, observation, state).sum()
    gradient = torch.autograd.grad(total, chosen_q)[0]
    assert torch.all(gradient[:, :-1] >= 0.0)
    assert torch.allclose(gradient[:, -1], torch.zeros_like(gradient[:, -1]))


def test_stage2_and_stage3_core_shapes():
    torch.manual_seed(4)
    model = BilateralOutcomeModel(11, 15, 15, hidden_dim=32)
    state = torch.randn(6, 9, 11)
    state_mask = torch.ones(6, 9, dtype=torch.bool)
    defenders = torch.randn(6, 5, 15)
    defender_mask = torch.ones(6, 5, dtype=torch.bool)
    attackers = torch.randn(6, 7, 15)
    attacker_mask = torch.ones(6, 7, dtype=torch.bool)
    logits = model(state, state_mask, defenders, defender_mask, attackers, attacker_mask)
    loss = selection_focused_loss(
        logits,
        torch.tensor([0, 1, 0, 1, 0, 1]),
        torch.ones(6),
        ranking_pairs=(torch.tensor([0, 2]), torch.tensor([1, 3])),
    )
    loss.backward()
    assert logits.shape == (6,)
    model.eval()
    permutation = torch.tensor([3, 0, 6, 1, 5, 2, 4])
    permuted_logits = model(
        state,
        state_mask,
        defenders,
        defender_mask,
        attackers[:, permutation],
        attacker_mask[:, permutation],
    )
    assert torch.allclose(logits.detach(), permuted_logits.detach(), atol=1e-5)
    assert len(enumerate_count_allocations(5, 3)) == 21
    decision = solve_defender_minimax(np.array([[0.2, 0.8], [0.5, 0.4]]))
    assert np.isclose(decision.defender_mixture.sum(), 1.0)
    assert 0.0 <= decision.worst_case_breach <= 1.0
    assert decision.sample_index(np.random.default_rng(0)) in {0, 1}


def test_had_adapter_emits_variable_scale_contract():
    adapter = HADStage1Adapter(2, 3, 2, controlled_side="Red", max_steps=5)
    raw = adapter.reset(seed=3)
    observation, state = tensorize_had_observation(raw)
    observation.validate()
    state.validate()
    assert observation.entity_obs.shape == (1, 2, 7, adapter.ENTITY_DIM)
    assert state.entities.shape == (1, 7, adapter.STATE_ENTITY_DIM)
    next_raw, reward, done, info = adapter.step([7, 8])
    assert next_raw["entity_obs"].shape == raw["entity_obs"].shape
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert {"terminated", "truncated"}.issubset(info)


def test_stage4_scale_guard_rolls_back_rejected_update():
    policy = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    before = policy.weight.detach().clone()
    loss = policy(torch.ones(1, 2)).sum()
    result = guarded_lower_update(
        policy,
        optimizer,
        loss,
        old_scale_success={"small": 0.8, "large": 0.7},
        evaluate_scale_success=lambda: {"small": 0.6, "large": 0.5},
    )
    assert not result.accepted
    assert torch.allclose(policy.weight, before)

    distilled = stability_regularized_loss(
        torch.tensor(0.0),
        torch.ones(1, 2, 3),
        torch.zeros(1, 2, 3),
        torch.tensor([[True, False]]),
        stability_weight=1.0,
    )
    assert torch.isclose(distilled, torch.tensor(1.0))
