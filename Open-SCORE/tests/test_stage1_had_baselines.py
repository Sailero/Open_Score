import numpy as np
import pytest
import torch

from open_score.envs import HADStage1Adapter
from open_score.stage1 import (
    CompetitiveEpisodeRunner,
    HADStage1Factory,
    LearningProgressCurriculum,
    RandomController,
    SequenceMAPPOLearner,
    SequenceQMIXLearner,
    VariableScaleMAPPO,
    VariableScaleVDN,
    collate_episodes,
    supported_scales,
)


def make_vdn() -> VariableScaleVDN:
    return VariableScaleVDN(12, 10, 6, 11, 27, agent_hidden_dim=32)


def make_mappo() -> VariableScaleMAPPO:
    return VariableScaleMAPPO(
        12, 10, 6, 11, 27, actor_hidden_dim=32, critic_hidden_dim=32
    )


def small_red_episodes():
    runner = CompetitiveEpisodeRunner(HADStage1Factory(max_steps=3))
    red = RandomController("red_random")
    blue = RandomController("blue_random")
    return [
        runner.run((2, 1), red, blue, seed=31).red,
        runner.run((3, 1), red, blue, seed=32).red,
        runner.run((3, 2), red, blue, seed=33).red,
    ]


def test_had_rejects_balanced_and_red_outnumbered_rosters():
    for scale in ((1, 1), (2, 2), (2, 3)):
        with pytest.raises(ValueError, match="strict Red numerical superiority"):
            HADStage1Adapter(*scale, max_steps=2)
    assert supported_scales(4) == [
        (2, 1),
        (3, 1),
        (3, 2),
        (4, 1),
        (4, 2),
        (4, 3),
    ]


def test_curriculum_forces_one_visit_to_each_newly_unlocked_had_scale():
    curriculum = LearningProgressCurriculum(episodes_per_stage=1)
    rng = np.random.default_rng(7)
    assert curriculum.sample(rng) == (2, 1)
    curriculum.record_episode()
    assert curriculum.sample(rng) == (3, 1)
    assert curriculum.sample(rng) == (3, 2)
    assert set(curriculum.sample_visits) == {(2, 1), (3, 1), (3, 2)}


def test_had_potential_shaping_is_zero_sum_and_zeros_terminal_potential():
    adapter = HADStage1Adapter(2, 1, max_steps=1, shaping_scale=0.5)
    adapter.reset(seed=41)
    _, rewards, done, info = adapter.step([0, 0], [0])
    assert done
    assert info["truncated"]
    assert info["defender_potential_after"] == 0.0
    assert np.isclose(rewards["Red"] + rewards["Blue"], 0.0)
    assert adapter.observe("Red")["state_mask"].all()


def test_vdn_uses_additive_active_agent_values():
    model = make_vdn()
    episode = small_red_episodes()[0]
    batch = collate_episodes([episode])
    observation = batch.observation_at(0)
    state = batch.state_at(0)
    chosen = torch.tensor([[1.5, -0.25]])
    assert torch.allclose(model.mix(chosen, observation, state), torch.tensor([1.25]))
    observation.agent_mask[:, 1] = False
    assert torch.allclose(model.mix(chosen, observation, state), torch.tensor([1.5]))


def test_vdn_sequence_learner_updates_on_multiple_had_scales():
    batch = collate_episodes(small_red_episodes())
    learner = SequenceQMIXLearner(make_vdn(), target_update_interval=1)
    before = [parameter.detach().clone() for parameter in learner.online.parameters()]
    metrics = learner.train_batch(batch)
    assert np.isfinite(metrics.loss)
    assert set(metrics.td_by_scale) == {(2, 1), (3, 1), (3, 2)}
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, learner.online.parameters())
    )


def test_mappo_sequence_learner_runs_clipped_on_policy_update():
    batch = collate_episodes(small_red_episodes())
    model = make_mappo()
    learner = SequenceMAPPOLearner(model, epochs=2)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    metrics = learner.train_batch(batch)
    values = [
        metrics.loss,
        metrics.policy_loss,
        metrics.value_loss,
        metrics.entropy,
        metrics.approximate_kl,
        metrics.clip_fraction,
        metrics.grad_norm,
    ]
    assert np.all(np.isfinite(values))
    assert metrics.entropy > 0.0
    assert 0.0 <= metrics.clip_fraction <= 1.0
    assert any(
        not torch.equal(old, new) for old, new in zip(before, model.parameters())
    )
