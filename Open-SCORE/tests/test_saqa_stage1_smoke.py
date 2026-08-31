"""Small executable checks for the optional SAQA Stage-1 encoder."""

import pytest
import torch

pytest.importorskip("smaclite")

from open_score.envs.smaclite_ad import (  # noqa: E402
    SMACliteStockAdapter,
    tensorize_smaclite_stock_observation,
)
from open_score.stage1.baselines import (  # noqa: E402
    VariableScaleMAPPO,
    VariableScaleVDN,
)
from open_score.stage1.entity_qmix import VariableScaleQMIX  # noqa: E402


@pytest.mark.parametrize("algorithm", ["qmix", "vdn", "mappo"])
def test_three_algorithms_saqa_stock_forward_backward_and_step(algorithm):
    torch.manual_seed(401)
    env = SMACliteStockAdapter("2s_vs_1sc", episode_limit=2, seed=17)
    raw, _ = env.reset(seed=19)
    observation, state = tensorize_smaclite_stock_observation(raw)
    dimensions = (
        env.ENTITY_DIM,
        env.SELF_DIM,
        env.TASK_DIM,
        env.STATE_ENTITY_DIM,
        env.ACTION_DIM,
    )
    if algorithm == "qmix":
        model = VariableScaleQMIX(
            *dimensions,
            agent_hidden_dim=16,
            mixer_hidden_dim=16,
            mixing_dim=8,
            encoder_kind="saqa",
            attention_heads=4,
        )
        values, _ = model.agent_q(observation)
        chosen = values.max(dim=-1).values
        loss = model.mix(chosen, observation, state).sum()
        encoder = model.agent.entity_encoder
    elif algorithm == "vdn":
        model = VariableScaleVDN(
            *dimensions,
            agent_hidden_dim=16,
            encoder_kind="saqa",
            attention_heads=4,
        )
        values, _ = model.agent_q(observation)
        chosen = values.max(dim=-1).values
        loss = model.mix(chosen, observation, state).sum()
        encoder = model.agent.entity_encoder
    else:
        model = VariableScaleMAPPO(
            *dimensions,
            actor_hidden_dim=16,
            critic_hidden_dim=16,
            encoder_kind="saqa",
            attention_heads=4,
        )
        logits, _ = model.actor_logits(observation)
        finite_logits = logits.masked_fill(~observation.avail_actions, 0.0)
        loss = finite_logits.sum() + model.value(state).sum()
        encoder = model.actor.entity_encoder
    loss.backward()
    attention_gradient = sum(
        float(parameter.grad.abs().sum())
        for parameter in encoder.cross_attention.parameters()
        if parameter.grad is not None
    )
    assert attention_gradient > 0.0

    actions = [int(torch.nonzero(row)[0]) for row in observation.avail_actions[0]]
    _, reward, terminated, truncated, _ = env.step(actions)
    assert isinstance(reward, float)
    assert not terminated and not truncated
    env.close()
