"""FastAPI backend — async APIs for FIIntell (ingestion, inference, health)."""

from backend.data_provider import MultiAssetDataProvider, default_provider, ohlcv_bars_to_dataframe
from backend.module_b_feature_engineering import FeatureEngineer
from backend.module_c_decision_engine import DecisionEngine, DecisionEngineWeights
from backend.schemas.features import FeatureEngineeringResult
from backend.schemas.decision import DecisionLabel, DecisionResult, TradeAction

__all__ = [
    "MultiAssetDataProvider",
    "default_provider",
    "ohlcv_bars_to_dataframe",
    "FeatureEngineer",
    "FeatureEngineeringResult",
    "DecisionEngine",
    "DecisionEngineWeights",
    "DecisionLabel",
    "DecisionResult",
    "TradeAction",
]
