"""Stage 1: variable-scale, task-conditioned QMIX executor."""

from .entity_qmix import EntityMonotonicMixer, VariableEntityAgent, VariableScaleQMIX
from .curriculum import (
    LearningProgressCurriculum,
    all_evaluation_scales,
    supported_scales,
)
from .learner import LearnerMetrics, SequenceQMIXLearner, linear_epsilon
from .losses import QMixTransition, one_step_qmix_td_loss
from .psro import (
    MetaNash,
    PSROIterationMetrics,
    estimated_nash_conv,
    psro_has_stabilised,
    solve_zero_sum_meta_game,
)
from .psro_trainer import HADPSROTrainer, PSROIterationResult
from .replay import (
    CompetitiveEpisode,
    EpisodeReplayBuffer,
    PaddedEpisodeBatch,
    TeamEpisode,
    collate_episodes,
)
from .runner import (
    CompetitiveEpisodeRunner,
    HADStage1Factory,
    QMixController,
    RandomController,
    RuleBasedController,
    frozen_qmix_controller,
)
from .training import BestResponseResult, EvaluationSummary, evaluate_pair, make_had_qmix

__all__ = [
    "EntityMonotonicMixer",
    "LearningProgressCurriculum",
    "LearnerMetrics",
    "SequenceQMIXLearner",
    "VariableEntityAgent",
    "VariableScaleQMIX",
    "QMixTransition",
    "MetaNash",
    "PSROIterationMetrics",
    "PSROIterationResult",
    "HADPSROTrainer",
    "TeamEpisode",
    "CompetitiveEpisode",
    "PaddedEpisodeBatch",
    "EpisodeReplayBuffer",
    "HADStage1Factory",
    "CompetitiveEpisodeRunner",
    "RandomController",
    "RuleBasedController",
    "QMixController",
    "BestResponseResult",
    "EvaluationSummary",
    "all_evaluation_scales",
    "supported_scales",
    "linear_epsilon",
    "collate_episodes",
    "evaluate_pair",
    "frozen_qmix_controller",
    "make_had_qmix",
    "estimated_nash_conv",
    "one_step_qmix_td_loss",
    "psro_has_stabilised",
    "solve_zero_sum_meta_game",
]
