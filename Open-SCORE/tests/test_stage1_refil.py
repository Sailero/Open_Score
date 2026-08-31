"""Focused checks for the round-01 REFIL-QMIX HAD implementation."""

import numpy as np
import torch

from open_score.stage1 import (
    BatchedHADRedRunner,
    CompetitiveEpisodeRunner,
    HADStage1Factory,
    QMixController,
    RuleBasedController,
    SequenceQMIXLearner,
    StepLearningProgressCurriculum,
    collate_episodes,
    make_had_qmix,
    supported_scales,
)


def _model(device=torch.device("cpu")):
    return make_had_qmix(
        device,
        agent_hidden_dim=16,
        mixer_hidden_dim=16,
        mixing_dim=8,
        encoder_kind="refil",
        attention_heads=4,
        attention_embed_dim=32,
        hypernet_hidden_dim=32,
    )


def _episodes(max_steps=3):
    runner = CompetitiveEpisodeRunner(HADStage1Factory(max_steps=max_steps))
    return [
        runner.run(
            scale,
            RuleBasedController("guard"),
            RuleBasedController("rush"),
            900 + index,
        ).red
        for index, scale in enumerate(((2, 1), (3, 2), (4, 3)))
    ]


def test_refil_base_and_imagined_losses_update_attention_and_mixer():
    torch.manual_seed(11)
    model = _model()
    learner = SequenceQMIXLearner(
        model,
        td_lambda=0.0,
        optimizer="rmsprop",
        imagine_weight=0.5,
    )
    batch = collate_episodes(_episodes())
    before = [parameter.detach().clone() for parameter in model.parameters()]
    metrics = learner.train_batch(batch)
    assert np.isfinite(
        [
            metrics.loss,
            metrics.base_loss,
            metrics.imagine_loss,
            metrics.grad_norm,
        ]
    ).all()
    assert metrics.imagine_loss is not None
    assert set(metrics.td_samples_by_scale) == {(2, 1), (3, 2), (4, 3)}
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, model.parameters())
    )


def test_refil_flex_mixer_is_monotonic_for_active_agents():
    torch.manual_seed(12)
    model = _model()
    batch = collate_episodes(_episodes())
    observation = batch.observation_at(0)
    state = batch.state_at(0)
    chosen = torch.randn(
        batch.batch_size,
        observation.agent_mask.shape[1],
        requires_grad=True,
    )
    total = model.mix(chosen, observation, state).sum()
    gradient = torch.autograd.grad(total, chosen)[0]
    assert torch.all(gradient[observation.agent_mask] >= 0.0)


def test_step_curriculum_boundaries_cap_and_roundtrip():
    curriculum = StepLearningProgressCurriculum()
    assert curriculum.active_scales(0) == ((2, 1),)
    assert curriculum.active_scales(100_000) == ((2, 1), (3, 1), (3, 2))
    assert curriculum.active_scales(250_000) == tuple(supported_scales(4, 1))
    for scale_index, scale in enumerate(curriculum.scales):
        curriculum.record_td_samples(
            {
                scale: [
                    2.0 + scale_index * 0.1
                    for _ in range(50)
                ]
                + [
                    0.2 + scale_index * 0.02
                    for _ in range(50)
                ]
            }
        )
    probabilities = curriculum.probabilities(400_000)
    assert np.isclose(sum(probabilities.values()), 1.0)
    assert min(probabilities.values()) >= 0.30 / 6 - 1e-12
    assert max(probabilities.values()) <= 0.40 + 1e-12
    state = curriculum.state_dict()
    restored = StepLearningProgressCurriculum()
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    final = restored.probabilities(850_000)
    assert set(np.round(list(final.values()), 10)) == {round(1 / 6, 10)}


def test_batched_runner_preserves_episode_contract_across_scales():
    torch.manual_seed(13)
    device = torch.device("cpu")
    model = _model(device)
    controller = QMixController(model, device, epsilon=0.0, name="refil_test")
    runner = BatchedHADRedRunner(
        HADStage1Factory(max_steps=2, shaping_scale=0.5)
    )
    scales = ((2, 1), (3, 2), (4, 3))
    episodes = runner.run_batch(
        scales,
        controller,
        [RuleBasedController("rush") for _ in scales],
        (1001, 1002, 1003),
    )
    assert [episode.red.scale for episode in episodes] == list(scales)
    assert all(episode.length == 2 for episode in episodes)
    assert all(np.isclose(episode.red.rewards + episode.blue.rewards, 0.0).all() for episode in episodes)
