"""Sentiment analysis schemas (entity-linked, weighted aggregation)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class HeadlineSentimentDetail(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str
    publisher: str | None = None
    source: str | None = None
    label: str  # positive | negative | neutral
    score: float  # expected score in [-1, 1]
    positive_prob: float = 0.0
    negative_prob: float = 0.0
    neutral_prob: float = 0.0
    entity_relevance: float = Field(ge=0.0, le=1.0)
    recency_weight: float = Field(ge=0.0, le=1.0)
    source_quality_weight: float = Field(ge=0.0)
    combined_weight: float = Field(ge=0.0)
    age_hours: float | None = None


class SentimentBreakdown(BaseModel):
    model_config = ConfigDict(frozen=False)

    method: str
    score: float
    positive_pct: float = Field(ge=0.0, le=100.0)
    negative_pct: float = Field(ge=0.0, le=100.0)
    neutral_pct: float = Field(ge=0.0, le=100.0)
    headlines_in: int = 0
    headlines_used: int = 0
    headlines_filtered_out: int = 0
    entity_match_rate: float = Field(ge=0.0, le=1.0)
    avg_source_quality: float = 1.0
    company_names: list[str] = Field(default_factory=list)
    headline_details: list[HeadlineSentimentDetail] = Field(default_factory=list)
