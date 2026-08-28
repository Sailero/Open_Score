import copy

import numpy as np
import torch

from open_score.contracts import GlobalState, TeamObservation
from open_score.envs import HADStage1Adapter, tensorize_had_observation
from open_score.stage1 import VariableScaleQMIX, estimated_nash_conv, solve_zero_sum_meta_game
from open_score.stage2 import OutcomeTimeMLP, outcome_time_loss
from open_score.stage3 import build_defender_payoff, enumerate_count_allocations, solve_defender_maximin
from open_score.stage4 import (
    FinalOutcomeResidual,
    final_outcome_residual_loss,
    guarded_lower_update,
    stability_regularized_loss,
)


def random_contract(batch=2, agents=4, entities=7):
    observation = TeamObservation(
        entity_obs=torch.randn(batch, agents, entities, 12),
        entity_mask=torch.ones(batch, agents, entities, dtype=torch.bool),
        self_obs=torch.randn(batch, agents, 10),
        task_obs=torch.randn(batch, agents, 6),
        agent_mask=torch.ones(batch, agents, dtype=torch.bool),
        avail_actions=torch.ones(batch, agents, 27, dtype=torch.bool),
    )
    state = GlobalState(
        entities=torch.randn(batch, entities, 11),
        entity_mask=torch.ones(batch, entities, dtype=torch.bool),
    )
    return observation, state


def make_model():
    return VariableScaleQMIX(12, 10, 6, 11, 27, agent_hidden_dim=32, mixer_hidden_dim=24, mixing_dim=12)


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
    model = OutcomeTimeMLP(input_dim=48, horizon_bins=5, hidden_dim=32)
    state = torch.randn(6, 48)
    logits = model(state)
    loss = outcome_time_loss(logits, torch.tensor([0, 2, 4, 5, 7, 10]))
    loss.backward()
    summary = model.summarize(state, steps_per_bin=10, command_bins=2)
    assert logits.shape == (6, 11)
    assert summary["expected_remaining_steps"].shape == (6,)
    assert len(enumerate_count_allocations(5, 3)) == 53
    breach_upper = np.array([[[0.10, 0.20], [0.25, 0.30]], [[0.15, 0.10], [0.20, 0.20]]])
    payoff = build_defender_payoff(breach_upper)
    decision = solve_defender_maximin(payoff)
    assert np.isclose(decision.defender_mixture.sum(), 1.0)
    assert -1.0 <= decision.worst_case_value <= 1.0
    assert decision.sample_index(np.random.default_rng(0)) in {0, 1}


def test_had_adapter_emits_variable_scale_contract():
    adapter = HADStage1Adapter(2, 3, max_steps=5)
    raw = adapter.reset(seed=3)
    target_seed3 = np.asarray(adapter.env.targets[0].position)
    adapter.reset(seed=3)
    assert np.allclose(adapter.env.targets[0].position, target_seed3)
    adapter.reset(seed=4)
    target_seed4 = np.asarray(adapter.env.targets[0].position)
    assert not np.allclose(target_seed4, target_seed3)
    assert np.all(target_seed4 >= adapter.target_region[:, 0])
    assert np.all(target_seed4 <= adapter.target_region[:, 1])
    raw = adapter.reset(seed=3)
    observation, state = tensorize_had_observation(raw["Red"])
    observation.validate()
    state.validate()
    assert observation.entity_obs.shape == (1, 2, 6, adapter.ENTITY_DIM)
    assert state.entities.shape == (1, 6, adapter.STATE_ENTITY_DIM)
    assert adapter.ACTION_DIM == 27
    assert np.all(np.linalg.norm(adapter.action_vectors, axis=1) <= 1.0 + 1e-6)
    next_raw, rewards, done, info = adapter.step([1, 2], [3, 4, 5])
    assert next_raw["Red"]["entity_obs"].shape == raw["Red"]["entity_obs"].shape
    assert np.isclose(rewards["Red"] + rewards["Blue"], 0.0)
    assert isinstance(done, bool)
    assert {"terminated", "truncated", "target_position"}.issubset(info)


def test_stage1_psro_meta_game_and_gap():
    payoff = np.array([[0.0, 1.0], [-1.0, 0.0]])
    meta = solve_zero_sum_meta_game(payoff)
    assert np.isclose(meta.defender_mixture.sum(), 1.0)
    assert np.isclose(meta.attacker_mixture.sum(), 1.0)
    assert estimated_nash_conv(meta.value, meta.value + 0.1, meta.value - 0.2) > 0.0


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

    residual = FinalOutcomeResidual(8, hidden_dim=16)
    prediction = residual(torch.randn(4, 8))
    joint_loss = final_outcome_residual_loss(
        torch.zeros(4), prediction, torch.tensor([1.0, -1.0, 1.0, -1.0])
    )
    joint_loss.backward()
    assert prediction.shape == (4,)
