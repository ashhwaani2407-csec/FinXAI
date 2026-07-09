"""Module B — Feature Engineering schemas.

This module defines the stable feature contract between:
Module A (data ingestion) -> Module B (feature engineering) -> Module C (decision engine).
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from backend.schemas.sentiment import SentimentBreakdown
from backend.schemas.ingestion import AssetClass

class FeatureEngineeringResult(BaseModel):
    model_config = ConfigDict(frozen=False)

    # Group scores used by Module C weighting.
    technical_score: float
    sentiment_score: float
    fundamentals_score: float
    geopolitics_score: float

    # Raw-ish geopolitics fields (deterministic mock).
    gpr_index: float
    gpr_score: float

    # The numeric features to feed into XGBoost (later).
    # Keep flat keys to simplify serialization and model pipelines.
    ml_vector: dict[str, float] = Field(default_factory=dict)

    # Short signals to support UI "Reasoning Summary".
    signals: list[str] = Field(default_factory=list)

    # Sentiment diagnostics for UI (FinBERT method + per-headline scores when available).
    sentiment_method: str | None = None
    sentiment_per_headline_scores: list[float] = Field(default_factory=list)

    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    as_of_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def is_success(self) -> bool:
        return not self.errors and bool(self.ml_vector)


def decimal_to_float(x: Any) -> float:
    if x is None:
        return 0.0
    if isinstance(x, float):
        return float(x)
    if isinstance(x, Decimal):
        return float(x)
    try:
        return float(x)
    except Exception:
        return 0.0

