"""Focused leakage, masking, and optimizer tests for HAD BC warm starts."""

from types import SimpleNamespace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest
import torch

from open_score.stage1 import VariableScaleMAPPO, VariableScaleVDN
from open_score.stage1.behavior_cloning import (
    audit_demonstration_seed_isolation,
    collect_guard_demonstrations,
    deterministic_demonstration_schedule,
    evaluation_seed_set,
    recurrent_imitation_objective,
    synchronize_q_target_after_behavior_cloning,
    train_recurrent_behavior_clone,
)
from open_score.stage1.curriculum import supported_scales
from open_score.stage1.learner import SequenceQMIXLearner
from open_score.stage1.replay import TeamEpisode, collate_episodes
from open_score.stage1.transfer import model_state_sha256


ROOT = Path(__file__).resolve().parents[1]


def _training_script():
    spec = spec_from_file_location(
        "train_stage1_baselines_bc_test",
        ROOT / "scripts" / "train_stage1_baselines.py",
    )
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _observation(agents=2, action_dim=3, active=(True, False)):
    entity_mask = np.zeros((agents, 2), dtype=bool)
    entity_mask[:, 0] = True
    avail = np.ones((agents, action_dim), dtype=bool)
    agent_mask = np.asarray(active, dtype=bool)
    avail[~agent_mask] = False
    avail[~agent_mask, 0] = True
    return {
        "entity_obs": np.zeros((agents, 2, 4), dtype=np.float32),
        "entity_mask": entity_mask,
        "self_obs": np.zeros((agents, 3), dtype=np.float32),
        "task_obs": np.zeros((agents, 2), dtype=np.float32),
        "agent_mask": agent_mask,
        "avail_actions": avail,
        "state_entities": np.zeros((2, 5), dtype=np.float32),
        "state_mask": np.ones(2, dtype=bool),
    }


def _episode(seed=1, length=2, inactive_action=2):
    observations = tuple(_observation() for _ in range(length + 1))
    actions = np.asarray(
        [[1, inactive_action] for _ in range(length)], dtype=np.int64
    )
    return TeamEpisode(
        observations,
        actions,
        np.zeros(length, dtype=np.float32),
        np.asarray([0.0] * (length - 1) + [1.0], dtype=np.float32),
        (2, 1),
        "Red",
        seed,
    )


def _vdn():
    return VariableScaleVDN(4, 3, 2, 5, 3, agent_hidden_dim=16)


def _mappo():
    return VariableScaleMAPPO(
        4, 3, 2, 5, 3, actor_hidden_dim=16, critic_hidden_dim=16
    )


def test_demo_seed_grid_is_balanced_and_disjoint_from_validation_and_heldout():
    scales = supported_scales(4, 1)
    styles = ("rush", "split_rush")
    schedule = deterministic_demonstration_schedule(73, scales, styles, 3)
    next_training_seed_schedule = deterministic_demonstration_schedule(
        74, scales, styles, 3
    )
    assert len(schedule) == 6 * 2 * 3
    assert len({record.seed for record in schedule}) == len(schedule)
    assert not (
        {record.seed for record in schedule}
        & {record.seed for record in next_training_seed_schedule}
    )
    validation = evaluation_seed_set(73 + 8_000_000, scales, styles, 4)
    heldout = evaluation_seed_set(73 + 18_000_000, scales, styles, 8)
    audit = audit_demonstration_seed_isolation(
        schedule, {"validation": validation, "heldout": heldout}
    )
    assert audit["passed"]
    assert not ({record.seed for record in schedule} & validation)
    assert not ({record.seed for record in schedule} & heldout)


def test_demo_seed_audit_fails_closed_on_leakage():
    schedule = deterministic_demonstration_schedule(
        11, [(2, 1)], ["rush"], 1
    )
    with pytest.raises(ValueError, match="leakage"):
        audit_demonstration_seed_isolation(
            schedule, {"heldout": [schedule[0].seed]}
        )


def test_recurrent_ce_masks_inactive_agent_actions_and_padding():
    torch.manual_seed(9)
    model = _vdn()
    first = collate_episodes([_episode(length=2, inactive_action=2)], torch.device("cpu"))
    second = collate_episodes(
        [_episode(length=2, inactive_action=1)], torch.device("cpu")
    )
    first_loss, first_correct, first_count = recurrent_imitation_objective(
        model, "vdn", first
    )
    second_loss, second_correct, second_count = recurrent_imitation_objective(
        model, "vdn", second
    )
    assert first_count == second_count == 2
    assert torch.equal(first_loss, second_loss)
    assert first_correct == second_correct


def test_q_behavior_cloning_changes_policy_and_syncs_target_without_rl_step():
    torch.manual_seed(17)
    model = _vdn()
    # SequenceQMIXLearner is interface-compatible with VDN: agent_q + mix.
    learner = SequenceQMIXLearner(model, target_update_interval=7)
    rl_optimizer_state_before = len(learner.optimizer.state)
    report = train_recurrent_behavior_clone(
        model,
        "vdn",
        [_episode(seed=index) for index in range(6)],
        epochs=8,
        batch_size=3,
        learning_rate=1e-2,
        shuffle_seed=901,
        device=torch.device("cpu"),
    )
    assert report["parameters_changed"]
    assert report["posttraining_measurement"]["cross_entropy"] < report[
        "pretraining_measurement"
    ]["cross_entropy"]
    # The dedicated BC optimizer cannot contaminate the registered RL optimizer.
    assert len(learner.optimizer.state) == rl_optimizer_state_before == 0
    assert learner.learner_step == 0
    sync = synchronize_q_target_after_behavior_cloning(learner)
    assert sync["online_target_equal_after"]
    assert learner.learner_step == 0


def test_mappo_behavior_cloning_updates_actor_only():
    torch.manual_seed(23)
    model = _mappo()
    critic_before = model_state_sha256(model.critic.state_dict())
    report = train_recurrent_behavior_clone(
        model,
        "mappo",
        [_episode(seed=index) for index in range(4)],
        epochs=2,
        batch_size=2,
        learning_rate=5e-3,
        shuffle_seed=902,
        device=torch.device("cpu"),
    )
    assert report["parameters_changed"]
    assert model_state_sha256(model.critic.state_dict()) == critic_before
    assert report["optimizer"]["parameter_scope"] == "actor_only"


def test_guard_collection_covers_every_scale_opponent_cell_and_records_steps():
    class FakeRunner:
        def run(self, scale, red_controller, blue_controller, seed):
            assert red_controller.style == "guard"
            assert blue_controller.style in {"rush", "split_rush"}
            episode = _episode(seed=seed, length=1)
            episode = TeamEpisode(
                episode.observations,
                episode.actions,
                episode.rewards,
                episode.done,
                tuple(scale),
                "Red",
                seed,
            )
            return SimpleNamespace(red=episode, outcome_red=1.0)

    scales = supported_scales(4, 1)
    dataset = collect_guard_demonstrations(
        FakeRunner(), scales, ("rush", "split_rush"), 2, 31
    )
    assert len(dataset.episodes) == 24
    assert dataset.metadata["covers_all_six_registered_scales"]
    assert set(dataset.metadata["cell_episode_counts"].values()) == {2}
    assert dataset.metadata["environment_step_count"] == 24
    assert dataset.metadata["teacher_is_performance_upper_bound"] is False


def test_aggregate_excludes_hybrid_primary_from_pure_rl_replication_gate():
    module = _training_script()
    results = []
    rule_by_seed = {}
    for seed in range(5):
        evaluation = {
            "controlled_mean_payoff": 0.5,
            "controlled_win_rate": 0.75,
            "per_scale_controlled_payoff": {"2v1": 0.5},
            "components": {
                "rush": {"per_scale_controlled_payoff": {"2v1": 0.5}}
            },
        }
        rule_by_seed[seed] = {
            "controlled_mean_payoff": 0.0,
            "per_scale_controlled_payoff": {"2v1": 0.0},
        }
        for algorithm, regime in (
            ("qmix", "rule_demonstration_bc_then_rl"),
            ("vdn", "pure_rl"),
        ):
            results.append(
                {
                    "algorithm": algorithm,
                    "seed": seed,
                    "policy_training_regime": regime,
                    "heldout_best_evaluation": evaluation,
                    "best_evaluation": evaluation,
                    "initial_evaluation": {"controlled_mean_payoff": 0.0},
                    "selected_parameters_changed": True,
                    "best_checkpoint": f"{algorithm}-{seed}.pt",
                    "best_checkpoint_sha256": f"sha-{algorithm}-{seed}",
                    "selection_protocol": {
                        "selected_checkpoint_minimum_updates_met": True
                    },
                }
            )
    aggregate = module.aggregate_multi_seed_results(
        results, rule_by_seed, [(2, 1)]
    )["algorithms"]
    assert aggregate["qmix"]["claim_role"] == "hybrid_primary_intelligent_strategy"
    assert not aggregate["qmix"]["baseline_replication_gate"]["eligible"]
    assert aggregate["qmix"]["rl_finetuning_gate"][
        "pre_stability_requirements_met"
    ]
    assert aggregate["vdn"]["baseline_replication_gate"]["eligible"]
    assert aggregate["vdn"]["claim_role"] == "pure_rl_replication_baseline"


def test_hybrid_preservation_uses_separate_epsilon_and_selection_floor():
    module = _training_script()
    args = SimpleNamespace(
        hybrid_epsilon_start=0.20,
        hybrid_epsilon_finish=0.02,
        epsilon_anneal_steps=100,
    )
    hybrid = {"policy_training_regime": "rule_demonstration_bc_then_rl"}
    pure = {"policy_training_regime": "pure_rl"}
    hybrid_epsilon, hybrid_audit = module.q_epsilon_schedule(
        hybrid, 0, args
    )
    pure_epsilon, pure_audit = module.q_epsilon_schedule(pure, 0, args)
    assert hybrid_epsilon == pytest.approx(0.20)
    assert hybrid_audit["finish"] == pytest.approx(0.02)
    assert pure_epsilon == pytest.approx(1.0)
    assert pure_audit["finish"] == pytest.approx(0.05)

    rejected = module.hybrid_checkpoint_selection_eligibility(hybrid, 99, 100)
    accepted = module.hybrid_checkpoint_selection_eligibility(hybrid, 100, 100)
    assert not rejected["eligible"]
    assert accepted["eligible"]
    assert rejected["bc_only_or_trivial_update_checkpoint_rejected"]
    assert not module.hybrid_checkpoint_selection_eligibility(
        pure, 0, 100
    )["eligible"]
    assert module.hybrid_checkpoint_selection_eligibility(
        pure, 1, 100
    )["eligible"]
