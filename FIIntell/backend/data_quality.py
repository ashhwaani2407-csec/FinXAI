"""Assess OHLCV reliability: missing bars, thin volume, stale/delisted symbols."""

from __future__ import annotations

from datetime import date

from backend.schemas.data_quality import DataQualityGrade, DataQualityReport
from backend.schemas.ingestion import AssetClass, AssetIngestionResult, OHLCVBar
from backend.settings import IngestionSettings

_PERIOD_EXPECTED_BARS: dict[str, int] = {
    "5d": 5,
    "1mo": 22,
    "3mo": 63,
    "6mo": 126,
    "1y": 252,
    "2y": 504,
    "3y": 756,
    "5y": 1260,
    "max": 252,
}

_MIN_AVG_VOLUME: dict[AssetClass, float] = {
    AssetClass.EQUITY_GLOBAL: 25_000.0,
    AssetClass.EQUITY_INDIA: 10_000.0,
    AssetClass.CRYPTO: 50.0,
    AssetClass.COMMODITY: 500.0,
}

_MAX_STALE_DAYS: dict[AssetClass, int] = {
    AssetClass.EQUITY_GLOBAL: 7,
    AssetClass.EQUITY_INDIA: 7,
    AssetClass.CRYPTO: 3,
    AssetClass.COMMODITY: 7,
}


def expected_bars_for_period(period: str) -> int:
    return _PERIOD_EXPECTED_BARS.get(period, 252)


def _trailing_flat_close_days(bars: list[OHLCVBar], window: int = 10) -> int:
    if not bars:
        return 0
    tail = bars[-min(window, len(bars)) :]
    closes = [float(b.close) for b in tail]
    if not closes:
        return 0
    ref = closes[-1]
    if ref == 0:
        return len(closes)
    streak = 0
    for c in reversed(closes):
        if abs(c - ref) <= max(abs(ref) * 1e-4, 1e-6):
            streak += 1
        else:
            break
    return streak


def _volume_stats(bars: list[OHLCVBar], window: int = 20) -> tuple[float, int]:
    tail = bars[-min(window, len(bars)) :]
    if not tail:
        return 0.0, 0
    vols = [int(b.volume) for b in tail]
    avg = sum(vols) / len(vols)
    zeros = sum(1 for v in vols if v <= 0)
    return avg, zeros


def _max_gap_days(bars: list[OHLCVBar], lookback: int = 90) -> int:
    tail = bars[-min(lookback, len(bars)) :]
    if len(tail) < 2:
        return 0
    max_gap = 0
    prev = tail[0].date
    for bar in tail[1:]:
        gap = (bar.date - prev).days
        max_gap = max(max_gap, gap)
        prev = bar.date
    return max_gap


def assess_data_quality(
    ingestion: AssetIngestionResult,
    settings: IngestionSettings | None = None,
) -> DataQualityReport:
    """Score ingestion reliability and decide if directional recommendations are allowed."""
    s = settings or IngestionSettings()
    min_score = float(s.data_quality_min_score)
    asset_class = ingestion.asset_class
    bars = ingestion.bars
    bar_count = len(bars)
    expected = expected_bars_for_period(s.history_period)

    flags: list[str] = []
    component_scores: list[tuple[float, float]] = []

    if ingestion.errors and not bars:
        flags.extend(ingestion.errors[:3])
        return DataQualityReport(
            score=0.0,
            grade=DataQualityGrade.UNUSABLE,
            is_tradeable=False,
            bar_count=0,
            expected_bars=expected,
            coverage_pct=0.0,
            stale_days=999,
            avg_volume_20d=0.0,
            zero_volume_days_20d=20,
            flat_price_days=0,
            flags=flags or ["no market history returned"],
            summary="Insufficient data — HOLD only.",
        )

    coverage_pct = (bar_count / expected) * 100.0 if expected > 0 else 0.0
    coverage_score = min(100.0, coverage_pct / 0.85)  # 85%+ coverage -> ~100
    if bar_count < 30:
        flags.append(f"very short history ({bar_count} bars)")
        coverage_score = min(coverage_score, 35.0)
    elif coverage_pct < 50.0:
        flags.append(f"low bar coverage ({coverage_pct:.0f}% of expected {expected})")
        coverage_score = min(coverage_score, 25.0)
    elif coverage_pct < 70.0:
        flags.append(f"partial bar coverage ({coverage_pct:.0f}%)")
        coverage_score = min(coverage_score, 55.0)
    component_scores.append((0.40, coverage_score))

    today = date.today()
    last_bar_date = bars[-1].date if bars else today
    stale_days = max(0, (today - last_bar_date).days)
    max_stale = _MAX_STALE_DAYS.get(asset_class, 7)
    if stale_days > max_stale:
        flags.append(f"stale last bar ({stale_days}d old; likely delisted/suspended)")
        freshness_score = max(0.0, 20.0 - (stale_days - max_stale) * 5.0)
    elif stale_days > max_stale - 2:
        flags.append(f"recent data lag ({stale_days}d since last bar)")
        freshness_score = 60.0
    else:
        freshness_score = 100.0
    component_scores.append((0.30, freshness_score))

    avg_vol, zero_vol_days = _volume_stats(bars)
    min_vol = _MIN_AVG_VOLUME.get(asset_class, 10_000.0)
    if avg_vol < min_vol:
        flags.append(f"thin average volume ({avg_vol:,.0f} vs min {min_vol:,.0f})")
        vol_score = max(0.0, (avg_vol / min_vol) * 70.0) if min_vol > 0 else 0.0
    else:
        vol_score = 100.0
    if zero_vol_days >= 10:
        flags.append(f"{zero_vol_days}/20 recent days have zero volume")
        vol_score = min(vol_score, 20.0)
    elif zero_vol_days >= 5:
        flags.append(f"{zero_vol_days}/20 recent days have zero volume")
        vol_score = min(vol_score, 50.0)
    component_scores.append((0.20, vol_score))

    flat_days = _trailing_flat_close_days(bars)
    gap_days = _max_gap_days(bars)
    continuity_score = 100.0
    if flat_days >= 8:
        flags.append(f"flat price for {flat_days} sessions (delisted/suspended hint)")
        continuity_score = min(continuity_score, 15.0)
    elif flat_days >= 5:
        flags.append(f"flat price for {flat_days} sessions")
        continuity_score = min(continuity_score, 45.0)
    if gap_days > 12:
        flags.append(f"large calendar gap in history ({gap_days}d between bars)")
        continuity_score = min(continuity_score, 40.0)
    component_scores.append((0.10, continuity_score))

    score = sum(w * sc for w, sc in component_scores)
    score = max(0.0, min(100.0, score))

    if score >= 75.0:
        grade = DataQualityGrade.HIGH
    elif score >= 55.0:
        grade = DataQualityGrade.MEDIUM
    elif score >= min_score:
        grade = DataQualityGrade.LOW
    else:
        grade = DataQualityGrade.UNUSABLE

    is_tradeable = score >= min_score and grade != DataQualityGrade.UNUSABLE
    if not is_tradeable:
        summary = "Insufficient data — HOLD only."
    elif grade == DataQualityGrade.LOW:
        summary = f"Marginal data quality ({score:.0f}/100); recommendation confidence reduced."
    else:
        summary = f"Data quality {grade.value} ({score:.0f}/100)."

    return DataQualityReport(
        score=round(score, 1),
        grade=grade,
        is_tradeable=is_tradeable,
        bar_count=bar_count,
        expected_bars=expected,
        coverage_pct=round(coverage_pct, 1),
        stale_days=stale_days,
        avg_volume_20d=round(avg_vol, 1),
        zero_volume_days_20d=zero_vol_days,
        flat_price_days=flat_days,
        flags=flags,
        summary=summary,
    )
