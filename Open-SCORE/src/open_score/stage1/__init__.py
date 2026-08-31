"""Stage 1: variable-scale, task-conditioned QMIX executor."""

from .entity_qmix import (
    EntityMonotonicMixer,
    SingleAgentQueryAttentionEncoder,
    VariableEntityAgent,
    VariableScaleQMIX,
)
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
from .behavior_cloning import (
    DEMONSTRATION_SEED_OFFSET,
    DEMONSTRATION_TRAINING_SEED_STRIDE,
    RuleDemonstrationDataset,
    audit_demonstration_seed_isolation,
    collect_guard_demonstrations,
    deterministic_demonstration_schedule,
    evaluation_seed_set,
    synchronize_q_target_after_behavior_cloning,
    train_recurrent_behavior_clone,
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
from .transfer import (
    ACTION_TARGET_CONTRACT,
    TRANSFER_SCHEMA_VERSION,
    model_state_sha256,
    transfer_stock_checkpoint,
)
from .smaclite_stock_training import (
    SMACliteStockEpisodeRunner,
    StockEpisode,
    StockEvaluation,
    evaluate_stock,
)

__all__ = [
    "EntityMonotonicMixer",
    "SingleAgentQueryAttentionEncoder",
    "LearningProgressCurriculum",
    "DEMONSTRATION_SEED_OFFSET",
    "DEMONSTRATION_TRAINING_SEED_STRIDE",
    "RuleDemonstrationDataset",
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
    "audit_demonstration_seed_isolation",
    "collect_guard_demonstrations",
    "deterministic_demonstration_schedule",
    "evaluation_seed_set",
    "synchronize_q_target_after_behavior_cloning",
    "train_recurrent_behavior_clone",
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
    "TRANSFER_SCHEMA_VERSION",
    "ACTION_TARGET_CONTRACT",
    "model_state_sha256",
    "transfer_stock_checkpoint",
    "SMACliteStockEpisodeRunner",
    "StockEpisode",
    "StockEvaluation",
    "evaluate_stock",
]
