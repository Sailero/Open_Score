"""Environment adapters for a common four-stage tensor contract."""

from .had_stage1 import HADStage1Adapter, tensorize_had_observation
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
    "SMACLITE_AD_PROTOCOL_ID",
    "SMACliteADConfig",
    "SMACliteADEnv",
    "SMACliteStockAdapter",
    "stock_scenario_fingerprint",
    "tensorize_had_observation",
    "tensorize_smaclite_ad_observation",
    "tensorize_smaclite_stock_observation",
]
