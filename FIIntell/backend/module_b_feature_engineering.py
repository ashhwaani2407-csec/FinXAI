"""Module B — Feature Engineering.

Transforms Module A output into:
1) group scores: technical/sentiment/fundamentals/geopolitics (each -1..1)
2) a flat `ml_vector` for XGBoost input (numeric floats)

Production notes:
- Keep the feature contract stable (do not reorder/remove keys without versioning).
- All external calls are best-effort; failures degrade with warnings/errors.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf

from backend.data_provider import ohlcv_bars_to_dataframe
from backend.schemas.features import FeatureEngineeringResult
from backend.schemas.ingestion import AssetClass, AssetIngestionResult, NewsHeadline
from backend.schemas.sentiment import SentimentBreakdown
from backend.sentiment_engine import PredictiveSentimentEngine
from backend.settings import IngestionSettings, get_ingestion_settings

logger = logging.getLogger(__name__)


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    if x is None or isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return 0.0
    return max(lo, min(hi, x))


def _score_from_inverse_ratio(value: float | None, ref: float, scale: float) -> float:
    """Generic -1..1 score where smaller is better (inverse ratio style)."""
    if value is None or value <= 0:
        return 0.0
    # If value == ref -> 0; if value < ref -> positive; if value > ref -> negative.
    score = (ref - value) / scale
    return _clamp(score)


def _score_from_ratio(value: float | None, ref_low: float, ref_high: float) -> float:
    """Map value between [ref_low, ref_high] into a -1..1 score (low positive, high negative)."""
    if value is None:
        return 0.0
    if ref_high <= ref_low:
        return 0.0
    # Normalize to [-1, 1]
    t = (value - ref_low) / (ref_high - ref_low)  # 0..1
    return _clamp(1.0 - 2.0 * t)


def _gpr_mock(ticker: str) -> tuple[float, float]:
    """Deterministic mock GPR index (0..100) based on ticker string."""
    s = (ticker or "").upper().strip().encode("utf-8", errors="ignore")
    digest = hashlib.sha256(s).hexdigest()
    n = int(digest[:8], 16)  # 0..2^32-1
    idx = (n % 101)  # 0..100
    score = (idx - 50.0) / 50.0  # -1..1
    return float(idx), float(score)



@dataclass(frozen=True)
class _FundamentalSnapshot:
    pe_ratio: float | None
    debt_to_equity: float | None
    market_cap: float | None


def _extract_fundamentals_yfinance(ticker: str) -> tuple[_FundamentalSnapshot, list[str]]:
    warnings: list[str] = []
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as e:
        return _FundamentalSnapshot(None, None, None), [f"fundamentals yfinance failed: {e!s}"]

    def _to_float(x: Any) -> float | None:
        try:
            if x is None:
                return None
            # avoid float drift from ints/str; float conversion still ok for ratios
            return float(Decimal(str(x)))
        except Exception:
            return None

    pe = _to_float(info.get("trailingPE") or info.get("forwardPE"))
    dte = _to_float(info.get("debtToEquity"))
    mc = _to_float(info.get("marketCap"))

    return _FundamentalSnapshot(pe, dte, mc), warnings


class FeatureEngineer:
    def __init__(self, settings: IngestionSettings | None = None) -> None:
        self._s = settings or get_ingestion_settings()
        self._sentiment = PredictiveSentimentEngine(settings=self._s)

    def build_features(self, ingestion: AssetIngestionResult) -> FeatureEngineeringResult:
        warnings: list[str] = []
        errors: list[str] = []

        try:
            bars_df = ohlcv_bars_to_dataframe(ingestion.bars)
        except Exception as e:
            return FeatureEngineeringResult(
                technical_score=0.0,
                sentiment_score=0.0,
                fundamentals_score=0.0,
                geopolitics_score=0.0,
                gpr_index=0.0,
                gpr_score=0.0,
                ml_vector={},
                signals=[],
                warnings=[],
                errors=[f"failed to convert bars to DataFrame: {e!s}"],
                as_of_utc=datetime.now(timezone.utc),
            )

        if bars_df.empty or len(bars_df) < 30:
            warnings.append("insufficient history for technical indicators; using neutral technical features.")

        technical = self._compute_technical_scores(bars_df, ingestion.asset_class)
        sentiment_score, sentiment_breakdown, sentiment_warnings = self._sentiment.analyze(
            ticker=ingestion.ticker_resolved_yfinance,
            asset_class=ingestion.asset_class,
            headlines=ingestion.headlines,
        )
        warnings.extend(sentiment_warnings)
        fundamentals, fw = self._compute_fundamentals_score(ingestion)
        warnings.extend(fw)

        gpr_index, gpr_score = _gpr_mock(ingestion.ticker_resolved_yfinance)
        geopolitics_score = float(gpr_score)

        # DecisionEngine weights come later (Module C). Here we produce group scores.
        signals: list[str] = []
        signals.extend(self._technical_signals(technical, bars_df))
        signals.extend(self._sentiment_signals(sentiment_score, sentiment_breakdown))
        signals.extend(self._fundamental_signals(fundamentals))
        signals.append(f"GPR(mock) index={gpr_index:.0f}/100")

        ml_vector = {}
        ml_vector.update(technical["ml_features"])
        ml_vector["sentiment_score"] = float(sentiment_score)
        if sentiment_breakdown is not None:
            ml_vector["sentiment_pos_pct"] = float(sentiment_breakdown.positive_pct)
            ml_vector["sentiment_neg_pct"] = float(sentiment_breakdown.negative_pct)
            ml_vector["sentiment_neu_pct"] = float(sentiment_breakdown.neutral_pct)
            ml_vector["sentiment_entity_match_rate"] = float(sentiment_breakdown.entity_match_rate)
            ml_vector["sentiment_avg_source_quality"] = float(sentiment_breakdown.avg_source_quality)
        ml_vector["gpr_score"] = geopolitics_score
        ml_vector.update(fundamentals["ml_features"])

        return FeatureEngineeringResult(
            asset_class=ingestion.asset_class,
            technical_score=float(technical["score"]),
            sentiment_score=float(sentiment_score),
            fundamentals_score=float(fundamentals["score"]),
            geopolitics_score=geopolitics_score,
            gpr_index=float(gpr_index),
            gpr_score=geopolitics_score,
            ml_vector=ml_vector,
            signals=signals,
            sentiment_method=sentiment_breakdown.method if sentiment_breakdown else None,
            sentiment_per_headline_scores=[d.score for d in (sentiment_breakdown.headline_details if sentiment_breakdown else [])],
            sentiment_breakdown=sentiment_breakdown,
            warnings=warnings,
            errors=errors,
            as_of_utc=datetime.now(timezone.utc),
        )

    async def build_features_async(self, ingestion: AssetIngestionResult) -> FeatureEngineeringResult:
        # Avoid blocking event loop: run in a thread.
        import asyncio

        return await asyncio.to_thread(self.build_features, ingestion)

    def _compute_technical_scores(self, bars_df: pd.DataFrame, asset_class: AssetClass) -> dict[str, Any]:
        close = bars_df["close"] if "close" in bars_df else pd.Series(dtype=float)
        last_close = float(close.iloc[-1]) if len(close) else 0.0

        if close.empty or len(close) < 50:
            # Neutral
            return {
                "score": 0.0,
                "ml_features": {
                    "last_close": last_close,
                    "rsi14": 0.0,
                    "macd": 0.0,
                    "macd_signal": 0.0,
                    "macd_hist": 0.0,
                    "bb_lower": 0.0,
                    "bb_mid": 0.0,
                    "bb_upper": 0.0,
                    "bb_width": 0.0,
                },
            }

        rsi_series = ta.rsi(close, length=14)
        rsi = float(rsi_series.dropna().iloc[-1]) if len(rsi_series.dropna()) else float("nan")

        macd_df = ta.macd(close, fast=12, slow=26, signal=9)
        # Column names vary slightly by pandas-ta versions; pick by substring.
        macd_col = next((c for c in macd_df.columns if "MACD_" in c or c.lower().startswith("macd_")), None)
        sig_col = next((c for c in macd_df.columns if "SIGNAL_" in c or c.lower().startswith("macds_")), None)
        hist_col = next((c for c in macd_df.columns if "HIST_" in c or c.lower().startswith("macdh_")), None)
        if macd_col is None:
            # fallback for naming like MACD_12_26_9
            macd_col = [c for c in macd_df.columns if "macd" in c.lower()][0]
        if sig_col is None:
            sig_col = [c for c in macd_df.columns if "signal" in c.lower()][0]
        if hist_col is None:
            hist_col = [c for c in macd_df.columns if "hist" in c.lower()][0]

        macd = float(macd_df[macd_col].dropna().iloc[-1]) if len(macd_df[macd_col].dropna()) else float("nan")
        macd_signal = (
            float(macd_df[sig_col].dropna().iloc[-1]) if len(macd_df[sig_col].dropna()) else float("nan")
        )
        macd_hist = float(macd_df[hist_col].dropna().iloc[-1]) if len(macd_df[hist_col].dropna()) else 0.0

        bb_df = ta.bbands(close, length=20, std=2)
        # Expected columns: BBL_20_2.0, BBM_20_2.0, BBU_20_2.0; handle variants.
        lower_col = next((c for c in bb_df.columns if str(c).upper().startswith("BBL")), None)
        mid_col = next((c for c in bb_df.columns if str(c).upper().startswith("BBM")), None)
        upper_col = next((c for c in bb_df.columns if str(c).upper().startswith("BBU")), None)
        if lower_col is None or mid_col is None or upper_col is None:
            # If schema mismatch, neutralize Bollinger.
            bb_lower = bb_mid = bb_upper = bb_width = 0.0
        else:
            bb_lower = float(bb_df[lower_col].dropna().iloc[-1]) if len(bb_df[lower_col].dropna()) else float("nan")
            bb_mid = float(bb_df[mid_col].dropna().iloc[-1]) if len(bb_df[mid_col].dropna()) else float("nan")
            bb_upper = float(bb_df[upper_col].dropna().iloc[-1]) if len(bb_df[upper_col].dropna()) else float("nan")
            width = (bb_upper - bb_lower) if (not math.isnan(bb_upper) and not math.isnan(bb_lower)) else 0.0
            bb_width = float(width / last_close) if last_close != 0 else 0.0

        # Group score heuristics (each -1..1; positive => buy/fruitful bias)
        rsi_score = _clamp((50.0 - rsi) / 20.0)
        # MACD histogram score: normalize by ~1% of price for scale stability
        scale = max(abs(last_close) * 0.01, 1e-9)
        macd_score = _clamp(macd_hist / scale)
        # Bollinger score: position relative to mid band. At lower -> +1 (buy), at upper -> -1 (sell).
        if bb_upper and bb_lower and not math.isnan(bb_upper) and not math.isnan(bb_lower):
            width = max((bb_upper - bb_lower), 1e-12)
            bb_pos = (last_close - bb_mid) / (width / 2.0)  # -1..1-ish
            bb_score = _clamp(-bb_pos)
        else:
            bb_score = 0.0

        technical_score = float(0.4 * rsi_score + 0.3 * macd_score + 0.3 * bb_score)

        ml_features = {
            "last_close": last_close,
            "rsi14": float(rsi) if not math.isnan(rsi) else 0.0,
            "macd": float(macd) if not math.isnan(macd) else 0.0,
            "macd_signal": float(macd_signal) if not math.isnan(macd_signal) else 0.0,
            "macd_hist": float(macd_hist),
            "bb_lower": float(bb_lower) if not math.isnan(bb_lower) else 0.0,
            "bb_mid": float(bb_mid) if not math.isnan(bb_mid) else 0.0,
            "bb_upper": float(bb_upper) if not math.isnan(bb_upper) else 0.0,
            "bb_width": float(bb_width),
            "technical_score": float(technical_score),  # useful as direct feature
        }

        return {"score": technical_score, "ml_features": ml_features}


    def _compute_fundamentals_score(
        self, ingestion: AssetIngestionResult
    ) -> tuple[dict[str, Any], list[str]]:
        # For commodities/crypto we may not have meaningful fundamentals; keep it neutral.
        if ingestion.asset_class in {AssetClass.COMMODITY, AssetClass.CRYPTO}:
            ml = {
                "pe_ratio": 0.0,
                "debt_to_equity": 0.0,
                "market_cap": 0.0,
            }
            return {"score": 0.0, "ml_features": ml}, [f"fundamentals neutral for asset_class={ingestion.asset_class}"]

        snapshot, warnings = _extract_fundamentals_yfinance(ingestion.ticker_resolved_yfinance)
        pe = snapshot.pe_ratio
        dte = snapshot.debt_to_equity
        mc = snapshot.market_cap

        pe_score = _score_from_inverse_ratio(pe, ref=20.0, scale=20.0) if pe is not None else 0.0
        dte_score = _score_from_ratio(dte, ref_low=0.0, ref_high=1.0) if dte is not None else 0.0

        # Market cap is generally not directional; keep near-neutral contribution
        mc_score = 0.0
        if mc is not None and mc > 0:
            # Larger caps slightly dampen risk (very small effect)
            mc_score = _clamp(math.log10(mc) / 15.0 - 0.5) * 0.2

        fundamentals_score = float(np.mean([pe_score, dte_score, mc_score]))

        ml_features = {
            "pe_ratio": float(pe) if pe is not None else 0.0,
            "debt_to_equity": float(dte) if dte is not None else 0.0,
            "market_cap": float(mc) if mc is not None else 0.0,
            "fundamentals_score": float(fundamentals_score),
        }
        return {"score": fundamentals_score, "ml_features": ml_features}, warnings

    def _technical_signals(self, technical: dict[str, Any], bars_df: pd.DataFrame) -> list[str]:
        rsi = technical["ml_features"].get("rsi14", 0.0)
        macd_hist = technical["ml_features"].get("macd_hist", 0.0)
        bb_lower = technical["ml_features"].get("bb_lower", 0.0)
        bb_upper = technical["ml_features"].get("bb_upper", 0.0)
        last_close = technical["ml_features"].get("last_close", 0.0)

        out: list[str] = []
        if rsi <= 30:
            out.append("RSI is oversold (potential BUY pressure).")
        elif rsi >= 70:
            out.append("RSI is overbought (potential SELL pressure).")

        if macd_hist > 0:
            out.append("MACD histogram is positive (momentum supportive).")
        elif macd_hist < 0:
            out.append("MACD histogram is negative (momentum bearish).")

        if bb_lower and bb_upper:
            if last_close <= bb_lower:
                out.append("Price near/below lower Bollinger band (oversold bias).")
            elif last_close >= bb_upper:
                out.append("Price near/above upper Bollinger band (overbought bias).")
        return out

    def _sentiment_signals(self, score: float, breakdown: SentimentBreakdown | None = None) -> list[str]:
        out: list[str] = []
        if score >= 0.2:
            out.append(f"Sentiment is bullish (score={score:.2f}).")
        elif score <= -0.2:
            out.append(f"Sentiment is bearish (score={score:.2f}).")
        else:
            out.append("News sentiment is neutral to mixed.")
        if breakdown and breakdown.headlines_used > 0:
            out.append(
                f"Based on {breakdown.headlines_used} headlines "
                f"(+{breakdown.positive_pct:.0f}%/-{breakdown.negative_pct:.0f}%/~{breakdown.neutral_pct:.0f}%), "
                f"entity match={breakdown.entity_match_rate:.0%}, method={breakdown.method}."
            )
        return out

    def _fundamental_signals(self, fundamentals: dict[str, Any]) -> list[str]:
        score = float(fundamentals.get("score") or 0.0)
        if score >= 0.2:
            return [f"Fundamentals look relatively stronger (score={score:.2f})."]
        if score <= -0.2:
            return [f"Fundamentals indicate elevated risk (score={score:.2f})."]
        return ["Fundamentals are neutral or missing."]
