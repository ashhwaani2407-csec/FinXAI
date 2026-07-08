"""Pydantic schemas for API and ingestion boundaries."""

from .ingestion import (
    AssetClass,
    AssetIngestionResult,
    HistorySource,
    NewsHeadline,
    NewsSource,
    OHLCVBar,
)

from .features import FeatureEngineeringResult
from .decision import DecisionLabel, DecisionResult, TradeAction

from .recommendation import (
    HealthResponse,
    RecommendBatchItem,
    RecommendBatchRequest,
    RecommendBatchResponse,
    RecommendRequest,
    RecommendResponse,
)
from .symbols import SymbolSearchItem, SymbolSearchResponse

__all__ = [
    "AssetClass",
    "AssetIngestionResult",
    "HistorySource",
    "NewsHeadline",
    "NewsSource",
    "OHLCVBar",
    "FeatureEngineeringResult",
    "DecisionLabel",
    "DecisionResult",
    "TradeAction",
    "RecommendRequest",
    "RecommendResponse",
    "HealthResponse",
    "RecommendBatchRequest",
    "RecommendBatchResponse",
    "RecommendBatchItem",
    "SymbolSearchItem",
    "SymbolSearchResponse",
]
