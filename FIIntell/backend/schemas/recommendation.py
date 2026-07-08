"""Module D API schemas (recommendation endpoint)."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from backend.schemas.decision import DecisionResult
from backend.schemas.features import FeatureEngineeringResult
from backend.schemas.ingestion import AssetIngestionResult


class RecommendRequest(BaseModel):
    ticker: str = Field(min_length=1)
    enable_finbert: bool = True


class RecommendResponse(BaseModel):
    model_config = ConfigDict(frozen=False)

    ingestion: AssetIngestionResult
    features: FeatureEngineeringResult
    decision: DecisionResult


class RecommendBatchRequest(BaseModel):
    tickers: list[str] = Field(min_length=1, max_length=50)
    enable_finbert: bool = True


class RecommendBatchItem(BaseModel):
    model_config = ConfigDict(frozen=False)

    ticker: str
    ok: bool
    ingestion: Optional[AssetIngestionResult] = None
    features: Optional[FeatureEngineeringResult] = None
    decision: Optional[DecisionResult] = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RecommendBatchResponse(BaseModel):
    model_config = ConfigDict(frozen=False)
    items: list[RecommendBatchItem]


class HealthResponse(BaseModel):
    ok: bool = True
    service: str = "fiintell"
