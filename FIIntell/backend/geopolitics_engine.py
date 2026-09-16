"""Geopolitical Risk Engine — real-time macro risk scoring.

Replaces the deterministic mock GPR with real market-based risk proxies:
- VIX (CBOE Volatility Index) — global fear gauge
- India VIX (^INDIAVIX) — India-specific fear gauge (for EQUITY_INDIA)
- Oil shock flag — sharp moves in crude oil (CL=F)
- USD/INR stress — currency depreciation signal (for EQUITY_INDIA)

All data sourced from yfinance. Graceful degradation: if any source fails,
its contribution defaults to neutral (0).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import yfinance as yf

from backend.schemas.ingestion import AssetClass

logger = logging.getLogger(__name__)

# TTL-based caching: risk data is expensive to fetch and doesn't change intraday.
_CACHE: dict[str, tuple[float, Any]] = {}
_CACHE_TTL_SECONDS = 900  # 15 minutes


def _cached_history(ticker: str, period: str = "30d") -> Any:
    """Fetch yfinance history with a simple TTL cache to avoid redundant API calls."""
    key = f"{ticker}:{period}"
    now = time.time()
    if key in _CACHE:
        ts, df = _CACHE[key]
        if now - ts < _CACHE_TTL_SECONDS:
            return df

    try:
        df = yf.Ticker(ticker).history(period=period, interval="1d")
        _CACHE[key] = (now, df)
        return df
    except Exception as e:
        logger.warning("geopolitics: failed to fetch %s: %s", ticker, e)
        return None


@dataclass(frozen=True)
class GeopoliticsResult:
    """Result of geopolitical risk assessment."""
    risk_index: float       # 0..100 (0 = low risk, 100 = extreme risk)
    risk_score: float       # -1..1 (negative = high risk / bearish, positive = low risk / bullish)
    risk_bucket: str        # "low", "medium", "high", "extreme"
    vix_level: float        # Current VIX value (0 if unavailable)
    vix_percentile: float   # VIX percentile vs 30-day range (0..1)
    india_vix_level: float  # Current India VIX (0 if N/A)
    oil_shock: bool         # True if oil moved > 5% in 5 days
    oil_change_5d_pct: float  # Oil 5-day % change
    usdinr_stress: bool     # True if INR depreciated > 1% in 5 days
    usdinr_change_5d_pct: float  # USD/INR 5-day % change
    warnings: list[str]


def _compute_vix_score() -> tuple[float, float, float, list[str]]:
    """Fetch VIX and compute a risk score.
    
    Returns: (vix_level, vix_percentile, vix_score, warnings)
    VIX thresholds:
        < 15: Low fear → score +0.5
        15-20: Normal → score 0
        20-30: Elevated → score -0.3
        30-40: High fear → score -0.7
        > 40: Extreme panic → score -1.0
    """
    warnings: list[str] = []
    df = _cached_history("^VIX", "30d")
    if df is None or df.empty:
        warnings.append("VIX data unavailable — using neutral.")
        return 0.0, 0.0, 0.0, warnings

    closes = df["Close"].dropna()
    if closes.empty:
        return 0.0, 0.0, 0.0, warnings

    vix = float(closes.iloc[-1])
    vix_min = float(closes.min())
    vix_max = float(closes.max())
    vix_range = max(vix_max - vix_min, 0.01)
    percentile = (vix - vix_min) / vix_range  # 0..1

    if vix < 15:
        score = 0.5
    elif vix < 20:
        score = 0.0
    elif vix < 30:
        score = -0.3 - 0.04 * (vix - 20)  # -0.3 to -0.7
    elif vix < 40:
        score = -0.7 - 0.03 * (vix - 30)  # -0.7 to -1.0
    else:
        score = -1.0

    return vix, percentile, score, warnings


def _compute_india_vix_score() -> tuple[float, float, list[str]]:
    """Fetch India VIX and compute a risk score.
    
    Returns: (india_vix_level, india_vix_score, warnings)
    """
    warnings: list[str] = []
    df = _cached_history("^INDIAVIX", "30d")
    if df is None or df.empty:
        warnings.append("India VIX unavailable — using neutral.")
        return 0.0, 0.0, warnings

    closes = df["Close"].dropna()
    if closes.empty:
        return 0.0, 0.0, warnings

    india_vix = float(closes.iloc[-1])

    # India VIX thresholds (typically lower than CBOE VIX):
    # < 12: Calm → +0.3
    # 12-18: Normal → 0
    # 18-25: Elevated → -0.3
    # > 25: High → -0.7
    if india_vix < 12:
        score = 0.3
    elif india_vix < 18:
        score = 0.0
    elif india_vix < 25:
        score = -0.3 - 0.06 * (india_vix - 18)  # -0.3 to -0.7
    else:
        score = -0.7

    return india_vix, score, warnings


def _compute_oil_shock() -> tuple[bool, float, float, list[str]]:
    """Check if oil price has made a sharp move (> 5% in 5 days).
    
    Returns: (is_shock, change_5d_pct, oil_score, warnings)
    """
    warnings: list[str] = []
    df = _cached_history("CL=F", "30d")
    if df is None or df.empty:
        warnings.append("Oil (CL=F) data unavailable — using neutral.")
        return False, 0.0, 0.0, warnings

    closes = df["Close"].dropna()
    if len(closes) < 6:
        return False, 0.0, 0.0, warnings

    current = float(closes.iloc[-1])
    five_days_ago = float(closes.iloc[-6])
    if five_days_ago <= 0:
        return False, 0.0, 0.0, warnings

    change_pct = ((current / five_days_ago) - 1.0) * 100.0
    is_shock = abs(change_pct) > 5.0

    # Oil spike = geopolitical stress (negative), oil crash also negative (demand fear)
    if is_shock:
        score = -0.4 if change_pct > 0 else -0.2  # spike is worse than crash
    else:
        score = 0.0

    return is_shock, change_pct, score, warnings


def _compute_usdinr_stress() -> tuple[bool, float, float, list[str]]:
    """Check if USD/INR has moved sharply (INR depreciation > 1% in 5 days).
    
    Returns: (is_stress, change_5d_pct, stress_score, warnings)
    """
    warnings: list[str] = []
    df = _cached_history("USDINR=X", "30d")
    if df is None or df.empty:
        warnings.append("USD/INR data unavailable — using neutral.")
        return False, 0.0, 0.0, warnings

    closes = df["Close"].dropna()
    if len(closes) < 6:
        return False, 0.0, 0.0, warnings

    current = float(closes.iloc[-1])
    five_days_ago = float(closes.iloc[-6])
    if five_days_ago <= 0:
        return False, 0.0, 0.0, warnings

    # Rising USD/INR = INR depreciation = stress
    change_pct = ((current / five_days_ago) - 1.0) * 100.0
    is_stress = change_pct > 1.0  # INR lost > 1% in 5 days

    if is_stress:
        score = -0.3
    elif change_pct > 0.5:
        score = -0.1
    elif change_pct < -0.5:
        score = 0.1  # INR strengthening = positive
    else:
        score = 0.0

    return is_stress, change_pct, score, warnings


def compute_geopolitics(asset_class: AssetClass) -> GeopoliticsResult:
    """Compute a real geopolitical risk score from market-based proxies.
    
    Weighting:
    - Global tickers: VIX 60%, Oil 40%
    - India tickers:  VIX 30%, India VIX 30%, Oil 20%, USD/INR 20%
    - Crypto:         VIX 70%, Oil 30% (crypto is VIX-sensitive)
    """
    all_warnings: list[str] = []

    # Always compute VIX and oil
    vix_level, vix_percentile, vix_score, vw = _compute_vix_score()
    all_warnings.extend(vw)

    oil_shock, oil_change, oil_score, ow = _compute_oil_shock()
    all_warnings.extend(ow)

    india_vix_level = 0.0
    india_vix_score = 0.0
    usdinr_stress = False
    usdinr_change = 0.0
    usdinr_score = 0.0

    # India-specific proxies
    if asset_class == AssetClass.EQUITY_INDIA:
        india_vix_level, india_vix_score, iw = _compute_india_vix_score()
        all_warnings.extend(iw)
        usdinr_stress, usdinr_change, usdinr_score, uw = _compute_usdinr_stress()
        all_warnings.extend(uw)

    # Weighted composite score
    if asset_class == AssetClass.EQUITY_INDIA:
        composite = (
            0.30 * vix_score
            + 0.30 * india_vix_score
            + 0.20 * oil_score
            + 0.20 * usdinr_score
        )
    elif asset_class == AssetClass.CRYPTO:
        composite = 0.70 * vix_score + 0.30 * oil_score
    else:
        # Global equities, commodities
        composite = 0.60 * vix_score + 0.40 * oil_score

    # Clamp to [-1, 1]
    composite = max(-1.0, min(1.0, composite))

    # Convert to risk index (0..100) — higher = more risk
    # Score of +1 → index 0 (no risk); score of -1 → index 100 (max risk)
    risk_index = (1.0 - composite) / 2.0 * 100.0

    # Bucket
    if risk_index < 25:
        bucket = "low"
    elif risk_index < 50:
        bucket = "medium"
    elif risk_index < 75:
        bucket = "high"
    else:
        bucket = "extreme"

    return GeopoliticsResult(
        risk_index=round(risk_index, 1),
        risk_score=round(composite, 4),
        risk_bucket=bucket,
        vix_level=round(vix_level, 2),
        vix_percentile=round(vix_percentile, 3),
        india_vix_level=round(india_vix_level, 2),
        oil_shock=oil_shock,
        oil_change_5d_pct=round(oil_change, 2),
        usdinr_stress=usdinr_stress,
        usdinr_change_5d_pct=round(usdinr_change, 2),
        warnings=all_warnings,
    )
