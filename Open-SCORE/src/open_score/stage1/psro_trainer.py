"""A small, executable two-team PSRO loop for HAD Stage 1."""

import copy
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch

from open_score.stage1.curriculum import LearningProgressCurriculum, Scale
from open_score.stage1.psro import (
    PSROIterationMetrics,
    estimated_nash_conv,
    solve_zero_sum_meta_game,
)
from open_score.stage1.runner import CompetitiveEpisodeRunner, TeamController
from open_score.stage1.training import (
    BestResponseResult,
    evaluate_pair,
    train_best_response,
    weighted_response_value,
)


@dataclass(frozen=True)
class PSROIterationResult:
    metrics: PSROIterationMetrics
    payoff_before: np.ndarray
    payoff_after: np.ndarray
    defender_mixture_before: np.ndarray
    attacker_mixture_before: np.ndarray
    defender_mixture_after: np.ndarray
    attacker_mixture_after: np.ndarray
    red_training: BestResponseResult
    blue_training: BestResponseResult


class HADPSROTrainer:
    def __init__(
        self,
        runner: CompetitiveEpisodeRunner,
        red_population: Sequence[TeamController],
        blue_population: Sequence[TeamController],
        train_scales: Sequence[Scale],
        evaluation_scales: Sequence[Scale],
        device: torch.device,
        seed: int = 0,
        curriculum: Optional[LearningProgressCurriculum] = None,
        minimum_br_environment_steps_for_stopping: int = 100_000,
    ):
        if not red_population or not blue_population:
            raise ValueError("PSRO needs at least one policy for each team")
        self.runner = runner
        self.red_population: List[TeamController] = list(red_population)
        self.blue_population: List[TeamController] = list(blue_population)
        self.train_scales = tuple(train_scales)
        self.evaluation_scales = tuple(evaluation_scales)
        if not self.train_scales or not self.evaluation_scales:
            raise ValueError("PSRO needs non-empty train and evaluation scale sets")
        self.device = device
        self.seed = seed
        # Defender and attacker have different TD-error learning progress.
        # Sharing one mutable curriculum would let the first-trained side
        # unlock stages for the second side and corrupt its sampling signal.
        self.red_curriculum = copy.deepcopy(curriculum)
        self.blue_curriculum = copy.deepcopy(curriculum)
        self.minimum_br_environment_steps_for_stopping = int(
            minimum_br_environment_steps_for_stopping
        )
        self.iteration = 0
        self.previous_meta_value: Optional[float] = None

    def evaluate_payoff_matrix(self, episodes_per_scale: int, seed: int) -> np.ndarray:
        matrix = np.zeros((len(self.red_population), len(self.blue_population)), np.float64)
        for red_index, red in enumerate(self.red_population):
            for blue_index, blue in enumerate(self.blue_population):
                summary = evaluate_pair(
                    self.runner,
                    red,
                    blue,
                    self.evaluation_scales,
                    episodes_per_scale,
                    seed + red_index * 100_000 + blue_index * 10_000,
                )
                matrix[red_index, blue_index] = summary.mean_defender_payoff
        return matrix

    def run_iteration(
        self,
        br_episodes: int,
        payoff_episodes_per_scale: int,
        batch_episodes: int = 8,
        fixed_scale: Optional[Scale] = None,
    ) -> PSROIterationResult:
        if fixed_scale is None and (
            self.red_curriculum is None or self.blue_curriculum is None
        ):
            raise ValueError("provide fixed_scale or a population curriculum")
        if fixed_scale is not None and fixed_scale not in self.train_scales:
            raise ValueError("fixed_scale must belong to train_scales")
        self.iteration += 1
        base_seed = self.seed + self.iteration * 1_000_000
        payoff_before = self.evaluate_payoff_matrix(payoff_episodes_per_scale, base_seed)
        meta_before = solve_zero_sum_meta_game(payoff_before)
        old_red = tuple(self.red_population)
        old_blue = tuple(self.blue_population)

        red_training = train_best_response(
            "Red",
            self.runner,
            old_blue,
            meta_before.attacker_mixture,
            self.device,
            episodes=br_episodes,
            seed=base_seed + 200_000,
            curriculum=self.red_curriculum,
            fixed_scale=fixed_scale,
            batch_episodes=batch_episodes,
            name=f"red_br_{self.iteration}",
        )
        blue_training = train_best_response(
            "Blue",
            self.runner,
            old_red,
            meta_before.defender_mixture,
            self.device,
            episodes=br_episodes,
            seed=base_seed + 400_000,
            curriculum=self.blue_curriculum,
            fixed_scale=fixed_scale,
            batch_episodes=batch_episodes,
            name=f"blue_br_{self.iteration}",
        )
        red_response_value = weighted_response_value(
            self.runner,
            red_training.controller,
            old_blue,
            meta_before.attacker_mixture,
            "Red",
            self.evaluation_scales,
            payoff_episodes_per_scale,
            base_seed + 600_000,
        )
        blue_response_value = weighted_response_value(
            self.runner,
            blue_training.controller,
            old_red,
            meta_before.defender_mixture,
            "Blue",
            self.evaluation_scales,
            payoff_episodes_per_scale,
            base_seed + 700_000,
        )
        nash_conv = estimated_nash_conv(
            meta_before.value, red_response_value, blue_response_value
        )
        self.red_population.append(red_training.controller)
        self.blue_population.append(blue_training.controller)
        payoff_after = self.evaluate_payoff_matrix(payoff_episodes_per_scale, base_seed + 800_000)
        meta_after = solve_zero_sum_meta_game(payoff_after)
        value_change = (
            float("inf")
            if self.previous_meta_value is None
            else abs(meta_after.value - self.previous_meta_value)
        )
        self.previous_meta_value = meta_after.value
        metrics = PSROIterationMetrics(
            iteration=self.iteration,
            population_shape=payoff_after.shape,
            meta_value=meta_after.value,
            defender_response_value=red_response_value,
            attacker_response_value=blue_response_value,
            estimated_nash_conv=nash_conv,
            archive_mean_payoff=float(payoff_after.mean()),
            meta_value_change=value_change,
            oracle_budget_met=(
                red_training.environment_steps >= self.minimum_br_environment_steps_for_stopping
                and blue_training.environment_steps >= self.minimum_br_environment_steps_for_stopping
            ),
        )
        return PSROIterationResult(
            metrics,
            payoff_before,
            payoff_after,
            meta_before.defender_mixture,
            meta_before.attacker_mixture,
            meta_after.defender_mixture,
            meta_after.attacker_mixture,
            red_training,
            blue_training,
        )
