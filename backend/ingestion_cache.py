"""Thread-safe per-ticker ingestion cache (rate-limit aware, no cross-ticker bleed)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from backend.schemas.ingestion import AssetIngestionResult
from backend.settings import IngestionSettings

_CACHE_SENTINEL = object()


@dataclass(frozen=True)
class IngestionCacheKey:
    """Strict cache key — one entry per resolved ticker + history window."""

    ticker: str
    history_period: str
    history_interval: str

    @classmethod
    def from_settings(cls, resolved_ticker: str, settings: IngestionSettings) -> "IngestionCacheKey":
        return cls(
            ticker=resolved_ticker.strip().upper(),
            history_period=settings.history_period,
            history_interval=settings.history_interval,
        )


@dataclass
class _CacheEntry:
    result: AssetIngestionResult
    expires_at: float


class IngestionCache:
    """TTL cache keyed by (resolved_ticker, history_period, history_interval)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[IngestionCacheKey, _CacheEntry] = {}

    def get(self, key: IngestionCacheKey) -> AssetIngestionResult | None:
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expires_at <= now:
                del self._entries[key]
                return None
            return entry.result.model_copy(deep=True)

    def set(self, key: IngestionCacheKey, result: AssetIngestionResult, ttl_seconds: int) -> None:
        expires_at = time.monotonic() + max(1, ttl_seconds)
        with self._lock:
            self._entries[key] = _CacheEntry(
                result=result.model_copy(deep=True),
                expires_at=expires_at,
            )

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


_global_cache: IngestionCache | object = _CACHE_SENTINEL


def get_ingestion_cache() -> IngestionCache:
    global _global_cache
    if _global_cache is _CACHE_SENTINEL:
        _global_cache = IngestionCache()
    return _global_cache  # type: ignore[return-value]


def reset_ingestion_cache() -> None:
    """Test helper — drop all cached ingestion payloads."""
    global _global_cache
    if _global_cache is not _CACHE_SENTINEL:
        get_ingestion_cache().clear()
