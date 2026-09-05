"""Environment adapters for a common four-stage tensor contract."""

from .had_stage1 import HADSnapshot, HADStage1Adapter, tensorize_had_observation
from .had_stage3 import (
    HADCommandedRuleController,
    HADStage3Adapter,
    HADStage3Event,
    HADStage3Snapshot,
)
from .smaclite_ad import (
    PROTOCOL_ID as SMACLITE_AD_PROTOCOL_ID,
    SMACliteADConfig,
    SMACliteADEnv,
    SMACliteStockAdapter,
    stock_scenario_fingerprint,
    tensorize_smaclite_ad_observation,
    tensorize_smaclite_stock_observation,
)

__all__ = [
    "HADStage1Adapter",
    "HADSnapshot",
    "HADCommandedRuleController",
    "HADStage3Adapter",
    "HADStage3Event",
    "HADStage3Snapshot",
    "SMACLITE_AD_PROTOCOL_ID",
    "SMACliteADConfig",
    "SMACliteADEnv",
    "SMACliteStockAdapter",
    "stock_scenario_fingerprint",
    "tensorize_had_observation",
    "tensorize_smaclite_ad_observation",
    "tensorize_smaclite_stock_observation",
]
