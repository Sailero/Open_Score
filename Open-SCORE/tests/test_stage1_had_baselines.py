import numpy as np
import pytest
import torch
from pathlib import Path

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
    EpisodeReplayBuffer,
    supported_scales,
)


def make_vdn() -> VariableScaleVDN:
    return VariableScaleVDN(
        HADStage1Adapter.ENTITY_DIM,
        HADStage1Adapter.SELF_DIM,
        HADStage1Adapter.TASK_DIM,
        HADStage1Adapter.STATE_ENTITY_DIM,
        HADStage1Adapter.ACTION_DIM,
        agent_hidden_dim=32,
    )


def make_mappo() -> VariableScaleMAPPO:
    return VariableScaleMAPPO(
        HADStage1Adapter.ENTITY_DIM,
        HADStage1Adapter.SELF_DIM,
        HADStage1Adapter.TASK_DIM,
        HADStage1Adapter.STATE_ENTITY_DIM,
        HADStage1Adapter.ACTION_DIM,
        actor_hidden_dim=32,
        critic_hidden_dim=32,
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


def test_had_explicitly_allows_unregistered_rosters_for_ood_evaluation():
    balanced = HADStage1Adapter(
        2, 2, max_steps=2, allow_unregistered_roster=True
    )
    balanced_observation = balanced.reset(seed=17)["Red"]
    assert balanced_observation["entity_obs"].shape[:2] == (2, 5)

    large = HADStage1Factory(
        max_steps=2, allow_unregistered_roster=True
    ).create((6, 3))
    large_observation = large.reset(seed=19)["Red"]
    assert large_observation["entity_obs"].shape[:2] == (6, 10)


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


def test_had_actor_and_central_state_include_remaining_horizon():
    adapter = HADStage1Adapter(2, 1, max_steps=4)
    initial = adapter.reset(seed=43)["Red"]
    assert np.allclose(initial["task_obs"][initial["agent_mask"], -1], 1.0)
    assert np.allclose(initial["state_entities"][:, -1], 1.0)
    adapter.step_count = 3
    late = adapter.observe("Red")
    assert np.allclose(late["task_obs"][late["agent_mask"], -1], 0.25)
    assert np.allclose(late["state_entities"][:, -1], 0.25)


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


def test_scale_balanced_replay_covers_each_seen_scale_when_batch_allows():
    episodes = small_red_episodes()
    replay = EpisodeReplayBuffer(capacity=12, seed=101)
    for episode in [episodes[0]] * 5 + [episodes[1]] * 2 + [episodes[2]]:
        replay.add(episode)
    batch = replay.sample_scale_balanced(3, torch.device("cpu"))
    assert {tuple(scale.tolist()) for scale in batch.scales} == {
        (2, 1),
        (3, 1),
        (3, 2),
    }


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
        metrics.actor_grad_norm,
        metrics.critic_grad_norm,
    ]
    assert np.all(np.isfinite(values))
    assert metrics.entropy > 0.0
    assert 0.0 <= metrics.clip_fraction <= 1.0
    assert any(
        not torch.equal(old, new) for old, new in zip(before, model.parameters())
    )


def test_mappo_uses_disjoint_optimizers_and_restores_value_norm(tmp_path: Path):
    batch = collate_episodes(small_red_episodes())
    learner = SequenceMAPPOLearner(make_mappo(), epochs=1)
    actor_ids = {
        id(parameter)
        for group in learner.actor_optimizer.param_groups
        for parameter in group["params"]
    }
    critic_ids = {
        id(parameter)
        for group in learner.critic_optimizer.param_groups
        for parameter in group["params"]
    }
    assert actor_ids
    assert critic_ids
    assert actor_ids.isdisjoint(critic_ids)
    assert torch.isclose(learner.value_normalizer.variance.cpu(), torch.tensor(1.0, dtype=torch.float64))
    learner.train_batch(batch)
    count = float(learner.value_normalizer.count)
    assert count > float(batch.filled.sum())
    path = tmp_path / "mappo-v2.pt"
    learner.save(path, {"purpose": "roundtrip"})
    restored = SequenceMAPPOLearner(make_mappo(), epochs=1)
    assert restored.load(path) == {"purpose": "roundtrip"}
    assert restored.learner_step == learner.learner_step
    assert torch.equal(
        restored.value_normalizer.mean.cpu(), learner.value_normalizer.mean.cpu()
    )
    assert len(restored.actor_optimizer.state) > 0
    assert len(restored.critic_optimizer.state) > 0


def test_mappo_rejects_legacy_combined_optimizer_checkpoint(tmp_path: Path):
    model = make_mappo()
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": torch.optim.Adam(model.parameters()).state_dict(),
            "learner_step": 0,
        },
        path,
    )
    with pytest.raises(ValueError, match="incompatible MAPPO checkpoint contract"):
        SequenceMAPPOLearner(make_mappo()).load(path)


def test_mappo_value_clip_center_is_rebased_after_large_statistic_shift():
    learner = SequenceMAPPOLearner(make_mappo(), epochs=1)
    old_values_on_old_scale = torch.tensor([[0.5, -0.25]], dtype=torch.float64)
    old_values_raw = learner.value_normalizer.denormalize(old_values_on_old_scale)
    returns = torch.tensor([[1_000.0]], dtype=torch.float64)
    mask = torch.ones_like(returns)

    rebased_old_values, normalized_returns = learner._update_value_scale(
        old_values_raw, returns, mask
    )

    # A large return shift makes the old and new normalized coordinates very
    # different.  The refreshed center must nevertheless represent exactly the
    # same raw predictions under the updated statistics.
    assert not torch.allclose(rebased_old_values, old_values_on_old_scale)
    assert torch.allclose(
        learner.value_normalizer.denormalize(rebased_old_values),
        old_values_raw,
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.isfinite(normalized_returns).all()
