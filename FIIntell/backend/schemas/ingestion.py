"""Structured ingestion outputs (Module A) — stable contract for API + dashboard."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AssetClass(StrEnum):
    EQUITY_GLOBAL = "equity_global"
    EQUITY_INDIA = "equity_india"
    COMMODITY = "commodity"
    CRYPTO = "crypto"


class HistorySource(StrEnum):
    YFINANCE = "yfinance"
    NSELIB = "nselib"


class NewsSource(StrEnum):
    YFINANCE = "yfinance"
    GDELT = "gdelt"


class OHLCVBar(BaseModel):
    model_config = ConfigDict(frozen=True)

    date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int = Field(ge=0)

    @field_validator("open", "high", "low", "close", mode="before")
    @classmethod
    def _coerce_price(cls, v: Any) -> Decimal:
        if isinstance(v, Decimal):
            return v
        from utils.financial_decimal import to_decimal_price

        d = to_decimal_price(v)
        if d is None:
            raise ValueError("invalid OHLC price")
        return d

    @field_validator("volume", mode="before")
    @classmethod
    def _coerce_volume(cls, v: Any) -> int:
        if v is None:
            raise ValueError("volume required")
        from utils.financial_decimal import to_volume_int

        vi = to_volume_int(v)
        if vi is None:
            raise ValueError("invalid volume")
        return max(vi, 0)


class NewsHeadline(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str
    url: str | None = None
    publisher: str | None = None
    published_at_utc: datetime | None = None
    source: NewsSource


class AssetIngestionResult(BaseModel):
    model_config = ConfigDict(frozen=False)

    ticker_requested: str
    ticker_resolved_yfinance: str
    asset_class: AssetClass
    history_source: HistorySource | None
    bars: list[OHLCVBar] = Field(default_factory=list)
    headlines: list[NewsHeadline] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    fetched_at_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def is_success(self) -> bool:
        return not self.errors and bool(self.bars)
