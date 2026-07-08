"""Application settings — injectable for tests and deployment (12-factor)."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class IngestionSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FIINTELL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    http_timeout_seconds: float = Field(default=45.0, ge=1.0)
    gdelt_base_url: HttpUrl = Field(default="https://api.gdeltproject.org/api/v2/doc/doc")
    gdelt_mode: Literal["ArtList"] = "ArtList"
    news_headline_limit: int = Field(default=10, ge=1, le=50)
    history_period: Literal["1y", "2y", "6mo", "3mo", "1mo", "5d"] = "1y"
    history_interval: Literal["1d", "1wk", "1mo"] = "1d"

    yfinance_max_attempts: int = Field(default=5, ge=1, le=12)
    yfinance_min_wait_seconds: float = Field(default=1.0, ge=0.0)
    yfinance_max_wait_seconds: float = Field(default=30.0, ge=1.0)
    yfinance_throttle_seconds: float = Field(
        default=0.35,
        ge=0.0,
        description="Light pacing between Yahoo calls to reduce 429 risk in batch jobs.",
    )

    nselib_max_attempts: int = Field(default=4, ge=1, le=10)
    nselib_min_wait_seconds: float = Field(default=1.5, ge=0.0)
    nselib_max_wait_seconds: float = Field(default=45.0, ge=1.0)

    gdelt_max_attempts: int = Field(default=4, ge=1, le=10)
    gdelt_min_wait_seconds: float = Field(default=1.0, ge=0.0)
    gdelt_max_wait_seconds: float = Field(default=30.0, ge=1.0)

    httpx_max_connections: int = Field(default=20, ge=1, le=200)
    user_agent: str = Field(
        default="FIIntell/1.0 (+https://example.invalid/contact)",
        description="Identify this app to upstream HTTP endpoints.",
    )

    batch_max_concurrency: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Max concurrent ticker jobs for /recommend/batch.",
    )
    symbol_search_limit: int = Field(
        default=25,
        ge=5,
        le=100,
        description="Maximum number of symbol matches returned per search query.",
    )

    # --- Module B: sentiment (FinBERT) ---
    enable_finbert: bool = Field(
        default=True,
        description="If false, use a lightweight headline keyword heuristic instead of downloading FinBERT.",
    )
    finbert_model: str = Field(default="ProsusAI/finbert")
    finbert_device: int = Field(
        default=-1,
        description="Use -1 for CPU. Set to an integer CUDA device id if available.",
    )
    finbert_max_headlines: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Max number of headlines to score per asset.",
    )

    # --- Module B: feature postprocessing ---
    sentiment_keyword_fallback: bool = Field(
        default=True,
        description="If FinBERT fails to load or score, use keyword heuristics instead of returning 0 silently.",
    )


@lru_cache(maxsize=1)
def get_ingestion_settings() -> IngestionSettings:
    return IngestionSettings()
