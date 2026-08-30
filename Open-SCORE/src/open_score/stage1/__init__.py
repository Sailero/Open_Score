"""Stage 1: variable-scale, task-conditioned QMIX executor."""

from .entity_qmix import EntityMonotonicMixer, VariableEntityAgent, VariableScaleQMIX
from .baselines import (
    CentralStateValue,
    MAPPOMetrics,
    SequenceMAPPOLearner,
    VariableEntityActor,
    VariableScaleMAPPO,
    VariableScaleVDN,
)
from .curriculum import (
    LearningProgressCurriculum,
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
    MAPPOController,
    RandomController,
    RuleBasedController,
    frozen_qmix_controller,
    frozen_mappo_controller,
)
from .training import BestResponseResult, EvaluationSummary, evaluate_pair, make_had_qmix

__all__ = [
    "EntityMonotonicMixer",
    "LearningProgressCurriculum",
    "LearnerMetrics",
    "SequenceQMIXLearner",
    "VariableEntityAgent",
    "VariableScaleQMIX",
    "VariableScaleVDN",
    "VariableScaleMAPPO",
    "VariableEntityActor",
    "CentralStateValue",
    "SequenceMAPPOLearner",
    "MAPPOMetrics",
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
    "MAPPOController",
    "BestResponseResult",
    "EvaluationSummary",
    "supported_scales",
    "linear_epsilon",
    "collate_episodes",
    "evaluate_pair",
    "frozen_qmix_controller",
    "frozen_mappo_controller",
    "make_had_qmix",
    "estimated_nash_conv",
    "one_step_qmix_td_loss",
    "psro_has_stabilised",
    "solve_zero_sum_meta_game",
]
