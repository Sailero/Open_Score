"""Stage 2: strategy-conditioned local outcome-and-time prediction."""

from .calibration import CalibratedRiskBound, TemperatureScaler
from .canonical import HADCanonicalizer
from .collectors import collect_had_records, record_from_had_episode
from .data import (
    FeatureNormalizer,
    Stage2FeatureEncoder,
    Stage2Query,
    Stage2Record,
    Stage2Split,
    read_records_jsonl,
    record_from_rollout,
    split_records_by_lineage_group,
    split_records_by_root,
    validate_rollout_lineage,
    write_records_csv,
    write_records_jsonl,
)
from .metrics import evaluate_stage2_predictions, metric_definitions_zh
from .outcome_model import BootstrapOutcomeEnsemble, OutcomeTimeMLP, outcome_time_loss
from .synthetic import generate_synthetic_records
from .training import (
    Stage2FitResult,
    Stage2Prediction,
    Stage2System,
    Stage2TrainingConfig,
    fit_stage2_system,
)

__all__ = [
    "BootstrapOutcomeEnsemble",
    "CalibratedRiskBound",
    "FeatureNormalizer",
    "HADCanonicalizer",
    "OutcomeTimeMLP",
    "Stage2FeatureEncoder",
    "Stage2FitResult",
    "Stage2Prediction",
    "Stage2Query",
    "Stage2Record",
    "Stage2Split",
    "Stage2System",
    "Stage2TrainingConfig",
    "TemperatureScaler",
    "collect_had_records",
    "evaluate_stage2_predictions",
    "fit_stage2_system",
    "generate_synthetic_records",
    "metric_definitions_zh",
    "outcome_time_loss",
    "read_records_jsonl",
    "record_from_had_episode",
    "record_from_rollout",
    "split_records_by_lineage_group",
    "split_records_by_root",
    "validate_rollout_lineage",
    "write_records_csv",
    "write_records_jsonl",
]
