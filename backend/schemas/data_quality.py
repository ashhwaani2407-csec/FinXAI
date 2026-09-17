"""Data quality report for ingestion reliability gating."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class DataQualityGrade(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNUSABLE = "unusable"


class DataQualityReport(BaseModel):
    model_config = ConfigDict(frozen=False)

    score: float = Field(ge=0.0, le=100.0, description="Composite reliability score (0–100).")
    grade: DataQualityGrade
    is_tradeable: bool = Field(
        description="When false, DecisionEngine must return HOLD only (insufficient data).",
    )

    bar_count: int = Field(ge=0)
    expected_bars: int = Field(ge=1)
    coverage_pct: float = Field(ge=0.0, le=200.0)

    stale_days: int = Field(ge=0, description="Calendar days since the last OHLCV bar.")
    avg_volume_20d: float = Field(ge=0.0)
    zero_volume_days_20d: int = Field(ge=0, le=20)
    flat_price_days: int = Field(ge=0, description="Trailing days with identical closes (delisted hint).")

    flags: list[str] = Field(default_factory=list)
    summary: str = ""
