"""Stage 2: strategy-conditioned local outcome-and-time prediction."""

from .calibration import CalibratedRiskBound, TemperatureScaler
from .canonical import HADCanonicalizer, HADVariableSetState
from .collectors import collect_had_records, iter_had_records, record_from_had_episode
from .data import (
    FeatureNormalizer,
    PHYSICAL_REDUNDANT_RELATIONS,
    PHYSICAL_TARGET_FIELDS,
    Stage2FeatureEncoder,
    Stage2Query,
    Stage2Record,
    Stage2Split,
    read_records_jsonl,
    record_from_rollout,
    split_records_by_lineage_group,
    split_records_by_root,
    validate_formal_dataset_contract,
    validate_rollout_lineage,
    validate_counterfactual_design,
    write_records_csv,
    write_records_jsonl,
)
from .metrics import evaluate_stage2_predictions, metric_definitions_zh
from .win_model import DynamicHADWinNet
from .outcome_model import (
    BootstrapOutcomeEnsemble,
    DynamicHADOutcomeNet,
    HADDeepSetOutcomeNet,
    OutcomeTimeMLP,
    competing_risk_nll,
    outcome_time_loss,
)
from .synthetic import generate_synthetic_records
from .tabular import TabularBaselineResult, fit_hist_gradient_boosting_baseline
from .training import (
    PhysicalTargetScaler,
    Stage2FitResult,
    Stage2Prediction,
    Stage2System,
    Stage2TrainingConfig,
    fit_stage2_system,
)

__all__ = [
    "BootstrapOutcomeEnsemble",
    "CalibratedRiskBound",
    "DynamicHADWinNet",
    "DynamicHADOutcomeNet",
    "FeatureNormalizer",
    "HADCanonicalizer",
    "HADVariableSetState",
    "HADDeepSetOutcomeNet",
    "OutcomeTimeMLP",
    "PHYSICAL_REDUNDANT_RELATIONS",
    "PHYSICAL_TARGET_FIELDS",
    "PhysicalTargetScaler",
    "Stage2FeatureEncoder",
    "Stage2FitResult",
    "Stage2Prediction",
    "Stage2Query",
    "Stage2Record",
    "Stage2Split",
    "Stage2System",
    "Stage2TrainingConfig",
    "TemperatureScaler",
    "TabularBaselineResult",
    "collect_had_records",
    "iter_had_records",
    "competing_risk_nll",
    "evaluate_stage2_predictions",
    "fit_stage2_system",
    "fit_hist_gradient_boosting_baseline",
    "generate_synthetic_records",
    "metric_definitions_zh",
    "outcome_time_loss",
    "read_records_jsonl",
    "record_from_had_episode",
    "record_from_rollout",
    "split_records_by_lineage_group",
    "split_records_by_root",
    "validate_formal_dataset_contract",
    "validate_rollout_lineage",
    "validate_counterfactual_design",
    "write_records_csv",
    "write_records_jsonl",
]
