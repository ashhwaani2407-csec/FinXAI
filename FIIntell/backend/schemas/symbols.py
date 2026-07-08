"""Schemas for natural-language symbol search."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SymbolSearchItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    display_name: str
    asset_class: str
    market: str
    exchange: str | None = None
    score: float = 0.0


class SymbolSearchResponse(BaseModel):
    model_config = ConfigDict(frozen=False)
    query: str
    items: list[SymbolSearchItem] = Field(default_factory=list)

