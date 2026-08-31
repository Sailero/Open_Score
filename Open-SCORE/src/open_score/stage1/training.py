"""Best-response training and evaluation utilities for Stage-1 PSRO."""

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from open_score.envs import HADStage1Adapter
from open_score.stage1.curriculum import LearningProgressCurriculum, Scale
from open_score.stage1.entity_qmix import VariableScaleQMIX
from open_score.stage1.learner import LearnerMetrics, SequenceQMIXLearner, linear_epsilon
from open_score.stage1.replay import EpisodeReplayBuffer
from open_score.stage1.runner import (
    CompetitiveEpisodeRunner,
    HADStage1Factory,
    QMixController,
    TeamController,
    frozen_qmix_controller,
)


def make_had_qmix(
    device: torch.device,
    agent_hidden_dim: int = 64,
    mixer_hidden_dim: int = 32,
    mixing_dim: int = 16,
    encoder_kind: str = "deepset",
    attention_heads: int = 4,
) -> VariableScaleQMIX:
    return VariableScaleQMIX(
        entity_dim=HADStage1Adapter.ENTITY_DIM,
        self_dim=HADStage1Adapter.SELF_DIM,
        task_dim=HADStage1Adapter.TASK_DIM,
        state_entity_dim=HADStage1Adapter.STATE_ENTITY_DIM,
        action_dim=HADStage1Adapter.ACTION_DIM,
        agent_hidden_dim=agent_hidden_dim,
        mixer_hidden_dim=mixer_hidden_dim,
        mixing_dim=mixing_dim,
        encoder_kind=encoder_kind,
        attention_heads=attention_heads,
    ).to(device)


@dataclass(frozen=True)
class EvaluationSummary:
    mean_defender_payoff: float
    defender_win_rate: float
    attacker_win_rate: float
    mean_episode_length: float
    episodes: int
    per_scale_payoff: Mapping[Scale, float]


@dataclass
class BestResponseResult:
    controller: QMixController
    learner: SequenceQMIXLearner
    episodes: int
    environment_steps: int
    mean_training_payoff: float
    latest_metrics: Optional[LearnerMetrics]


def evaluate_pair(
    runner: CompetitiveEpisodeRunner,
    red_controller: TeamController,
    blue_controller: TeamController,
    scales: Sequence[Scale],
    episodes_per_scale: int,
    seed: int,
) -> EvaluationSummary:
    if episodes_per_scale < 1 or not scales:
        raise ValueError("evaluation needs scales and at least one episode per scale")
    outcomes: List[float] = []
    lengths: List[int] = []
    by_scale: Dict[Scale, List[float]] = {scale: [] for scale in scales}
    episode_index = 0
    for scale in scales:
        for _ in range(episodes_per_scale):
            episode = runner.run(scale, red_controller, blue_controller, seed + episode_index)
            outcomes.append(episode.outcome_red)
            lengths.append(episode.length)
            by_scale[scale].append(episode.outcome_red)
            episode_index += 1
    values = np.asarray(outcomes, dtype=np.float64)
    return EvaluationSummary(
        mean_defender_payoff=float(values.mean()),
        defender_win_rate=float(np.mean(values > 0.0)),
        attacker_win_rate=float(np.mean(values < 0.0)),
        mean_episode_length=float(np.mean(lengths)),
        episodes=len(outcomes),
        per_scale_payoff={scale: float(np.mean(result)) for scale, result in by_scale.items()},
    )


def weighted_response_value(
    runner: CompetitiveEpisodeRunner,
    response: TeamController,
    opponent_population: Sequence[TeamController],
    opponent_mixture: np.ndarray,
    response_side: str,
    scales: Sequence[Scale],
    episodes_per_pair_scale: int,
    seed: int,
) -> float:
    values = []
    for index, opponent in enumerate(opponent_population):
        if response_side == "Red":
            summary = evaluate_pair(
                runner, response, opponent, scales, episodes_per_pair_scale, seed + index * 10_000
            )
        elif response_side == "Blue":
            summary = evaluate_pair(
                runner, opponent, response, scales, episodes_per_pair_scale, seed + index * 10_000
            )
        else:
            raise ValueError("response_side must be Red or Blue")
        values.append(summary.mean_defender_payoff)
    return float(np.asarray(opponent_mixture, dtype=np.float64) @ np.asarray(values))


def train_best_response(
    side: str,
    runner: CompetitiveEpisodeRunner,
    opponent_population: Sequence[TeamController],
    opponent_mixture: np.ndarray,
    device: torch.device,
    episodes: int,
    seed: int,
    curriculum: Optional[LearningProgressCurriculum] = None,
    fixed_scale: Optional[Scale] = None,
    batch_episodes: int = 8,
    replay_episodes: int = 512,
    updates_per_episode: int = 1,
    learning_rate: float = 5e-4,
    td_lambda: float = 0.6,
    epsilon_anneal_steps: int = 50_000,
    initial_model: Optional[VariableScaleQMIX] = None,
    name: str = "best_response",
) -> BestResponseResult:
    if side not in {"Red", "Blue"}:
        raise ValueError("side must be Red or Blue")
    if episodes < 1 or len(opponent_population) != len(opponent_mixture):
        raise ValueError("invalid best-response training request")
    mixture = np.asarray(opponent_mixture, dtype=np.float64)
    mixture = mixture / mixture.sum()
    rng = np.random.default_rng(seed)
    model = (
        initial_model.to(device)
        if initial_model is not None
        else make_had_qmix(device)
    )
    learner = SequenceQMIXLearner(
        model,
        learning_rate=learning_rate,
        td_lambda=td_lambda,
        target_update_interval=100,
    )
    replay = EpisodeReplayBuffer(replay_episodes, seed=seed + 101)
    controlled = QMixController(model, device, epsilon=1.0, name=name)
    environment_steps = 0
    training_payoffs: List[float] = []
    latest: Optional[LearnerMetrics] = None
    for episode_index in range(episodes):
        if fixed_scale is not None:
            scale = fixed_scale
        elif curriculum is not None:
            scale = curriculum.sample(rng)
        else:
            raise ValueError("provide fixed_scale or curriculum")
        opponent_index = int(rng.choice(len(opponent_population), p=mixture))
        opponent = opponent_population[opponent_index]
        controlled.epsilon = linear_epsilon(environment_steps, anneal_steps=epsilon_anneal_steps)
        rollout_seed = seed + episode_index * 17
        if side == "Red":
            rollout = runner.run(scale, controlled, opponent, rollout_seed)
            replay.add(rollout.red)
        else:
            rollout = runner.run(scale, opponent, controlled, rollout_seed)
            replay.add(rollout.blue)
        environment_steps += rollout.length
        training_payoffs.append(rollout.outcome_red)
        if curriculum is not None:
            curriculum.record_episode()
        if len(replay) >= batch_episodes:
            for _ in range(updates_per_episode):
                latest = learner.train_batch(replay.sample(batch_episodes, device))
                if curriculum is not None:
                    for updated_scale, td_error in latest.td_by_scale.items():
                        curriculum.update(updated_scale, td_error)
    frozen = frozen_qmix_controller(model, device, name)
    return BestResponseResult(
        frozen,
        learner,
        episodes,
        environment_steps,
        float(np.mean(training_payoffs)),
        latest,
    )
