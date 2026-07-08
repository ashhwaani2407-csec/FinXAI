"""Module C — Decision / Recommendation schemas."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class TradeAction(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class DecisionLabel(StrEnum):
    FRUITFUL_TRADE = "Fruitful/Trade"
    RISKY_AVOID = "Risky/Avoid"


class DecisionResult(BaseModel):
    model_config = ConfigDict(frozen=False)

    label: DecisionLabel
    action: TradeAction
    confidence_pct: float = Field(ge=0.0, le=100.0)

    # Weighted score in [-1..1] (fallback) or calibrated model score (if using XGBoost).
    score: float

    # High-level rationale bullets (for UI).
    reasoning: list[str] = Field(default_factory=list)

    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    as_of_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def is_success(self) -> bool:
        return not self.errors

