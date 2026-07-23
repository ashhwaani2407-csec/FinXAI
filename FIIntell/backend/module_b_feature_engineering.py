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


def _safe_last(series: pd.Series) -> float:
    """Return last non-NaN value from a pandas Series, or NaN if empty."""
    cleaned = series.dropna()
    return float(cleaned.iloc[-1]) if len(cleaned) > 0 else float("nan")


def _score_from_inverse_ratio(value: float | None, ref: float, scale: float) -> float:
    """Generic -1..1 score where smaller is better (inverse ratio style)."""
    if value is None or value <= 0:
        return 0.0
    score = (ref - value) / scale
    return _clamp(score)


def _score_from_ratio(value: float | None, ref_low: float, ref_high: float) -> float:
    """Map value between [ref_low, ref_high] into a -1..1 score (low positive, high negative)."""
    if value is None:
        return 0.0
    if ref_high <= ref_low:
        return 0.0
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
    
# ---------------------------------------------------------------------------
# Sector P/E reference table (fallback when yfinance doesn't provide sectorPE)
# ---------------------------------------------------------------------------

_SECTOR_PE_DEFAULTS: dict[str, float] = {
    "technology": 30.0,
    "communication services": 22.0,
    "consumer cyclical": 22.0,
    "consumer defensive": 20.0,
    "financial services": 15.0,
    "healthcare": 22.0,
    "industrials": 20.0,
    "basic materials": 15.0,
    "energy": 12.0,
    "utilities": 18.0,
    "real estate": 25.0,
}
_DEFAULT_SECTOR_PE = 20.0
@dataclass(frozen=True)
class _FundamentalSnapshot:
    pe_ratio: float | None
    forward_pe: float | None
    price_to_book: float | None
    ev_to_ebitda: float | None
    revenue_growth: float | None
    profit_margins: float | None
    operating_margins: float | None
    debt_to_equity: float | None
    current_ratio: float | None
    market_cap: float | None
    sector: str | None
    sector_pe: float | None


def _extract_fundamentals_yfinance(ticker: str) -> tuple[_FundamentalSnapshot, list[str]]:
    warnings: list[str] = []
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as e:
        return _FundamentalSnapshot(
            None, None, None, None, None, None, None, None, None, None, None, None
        ), [f"fundamentals yfinance failed: {e!s}"]
    
    def _to_float(x: Any) -> float | None:
        try:
            if x is None:
                return None
            return float(Decimal(str(x)))
        except Exception:
            return None
    
    pe = _to_float(info.get("trailingPE"))
    fwd_pe = _to_float(info.get("forwardPE"))
    pb = _to_float(info.get("priceToBook"))
    ev_ebitda = _to_float(info.get("enterpriseToEbitda"))
    rev_growth = _to_float(info.get("revenueGrowth"))
    profit_margins = _to_float(info.get("profitMargins"))
    operating_margins = _to_float(info.get("operatingMargins"))
    dte = _to_float(info.get("debtToEquity"))
    current_ratio = _to_float(info.get("currentRatio"))
    mc = _to_float(info.get("marketCap"))
    sector = info.get("sector")
    sector_pe_raw = _to_float(info.get("sectorPE"))
    return _FundamentalSnapshot(
        pe_ratio=pe or fwd_pe,
        forward_pe=fwd_pe,
        price_to_book=pb,
        ev_to_ebitda=ev_ebitda,
        revenue_growth=rev_growth,
        profit_margins=profit_margins,
        operating_margins=operating_margins,
        debt_to_equity=dte,
        current_ratio=current_ratio,
        market_cap=mc,
        sector=sector,
        sector_pe=sector_pe_raw,
    ), warnings


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
        
    # ------------------------------------------------------------------
    # Regime Detection
    # ------------------------------------------------------------------
   
    @staticmethod
    def _detect_regime(
        close: pd.Series, sma50: float, sma200: float, adx: float
    ) -> str:
        """Classify market regime as 'bull', 'bear', or 'sideways'."""
        last_close = float(close.iloc[-1]) if len(close) > 0 else 0.0
        if adx < 20:
            return "sideways"
        if last_close > sma200 and sma50 > sma200:
            return "bull"
        if last_close < sma200 and sma50 < sma200:
            return "bear"
        return "sideways"
    
    # ------------------------------------------------------------------
    # Technical Indicators (expanded)
    # ------------------------------------------------------------------
   
    def _compute_technical_scores(self, bars_df: pd.DataFrame, asset_class: AssetClass) -> dict[str, Any]:
        close = bars_df["close"] if "close" in bars_df else pd.Series(dtype=float)
        high = bars_df["high"] if "high" in bars_df else pd.Series(dtype=float)
        low = bars_df["low"] if "low" in bars_df else pd.Series(dtype=float)
        volume = bars_df["volume"] if "volume" in bars_df else pd.Series(dtype=float)
        last_close = float(close.iloc[-1]) if len(close) else 0.0
        # Neutral defaults for all features
        neutral_ml = {
            "last_close": last_close,
            "rsi14": 0.0, "macd": 0.0, "macd_signal": 0.0, "macd_hist": 0.0,
            "bb_lower": 0.0, "bb_mid": 0.0, "bb_upper": 0.0, "bb_width": 0.0,
            "sma50": 0.0, "sma200": 0.0, "adx": 0.0,
            "obv": 0.0, "obv_slope": 0.0, "relative_volume": 0.0,
            "atr14": 0.0, "atr_pct": 0.0, "hist_volatility_20d": 0.0,
            "high_52w": 0.0, "low_52w": 0.0, "dist_to_high_pct": 0.0, "dist_to_low_pct": 0.0,
            "regime": "sideways",
        }
        
        if close.empty or len(close) < 50:
            return {"score": 0.0, "ml_features": neutral_ml}
        # --- Existing: RSI, MACD, Bollinger ---
        rsi_series = ta.rsi(close, length=14)
        rsi = _safe_last(rsi_series)
        macd_df = ta.macd(close, fast=12, slow=26, signal=9)
        macd_col = next((c for c in macd_df.columns if "MACD_" in c or c.lower().startswith("macd_")), None)
        sig_col = next((c for c in macd_df.columns if "SIGNAL_" in c or c.lower().startswith("macds_")), None)
        hist_col = next((c for c in macd_df.columns if "HIST_" in c or c.lower().startswith("macdh_")), None)
        if macd_col is None:
            macd_col = [c for c in macd_df.columns if "macd" in c.lower()][0]
        if sig_col is None:
            sig_col = [c for c in macd_df.columns if "signal" in c.lower()][0]
        if hist_col is None:
            hist_col = [c for c in macd_df.columns if "hist" in c.lower()][0]
        macd_val = _safe_last(macd_df[macd_col])
        macd_signal = _safe_last(macd_df[sig_col])
        macd_hist = _safe_last(macd_df[hist_col]) if hist_col else 0.0
        
        if math.isnan(macd_hist):
            macd_hist = 0.0
        bb_df = ta.bbands(close, length=20, std=2)
        lower_col = next((c for c in bb_df.columns if str(c).upper().startswith("BBL")), None)
        mid_col = next((c for c in bb_df.columns if str(c).upper().startswith("BBM")), None)
        upper_col = next((c for c in bb_df.columns if str(c).upper().startswith("BBU")), None)
        
        if lower_col is None or mid_col is None or upper_col is None:
            bb_lower = bb_mid = bb_upper = bb_width = 0.0
        else:
            bb_lower = _safe_last(bb_df[lower_col])
            bb_mid = _safe_last(bb_df[mid_col])
            bb_upper = _safe_last(bb_df[upper_col])
            if math.isnan(bb_lower):
                bb_lower = 0.0
            if math.isnan(bb_mid):
                bb_mid = 0.0
            if math.isnan(bb_upper):
                bb_upper = 0.0
            width = (bb_upper - bb_lower)
            bb_width = float(width / last_close) if last_close != 0 else 0.0
            
        # --- NEW: Trend (SMA50, SMA200, ADX) ---
        
        sma50_series = ta.sma(close, length=50)
        sma50 = _safe_last(sma50_series)
        if math.isnan(sma50):
            sma50 = last_close
        sma200_val = last_close  # fallback if not enough data
        if len(close) >= 200:
            sma200_series = ta.sma(close, length=200)
            sma200_val = _safe_last(sma200_series)
            if math.isnan(sma200_val):
                sma200_val = last_close
        adx_df = ta.adx(high, low, close, length=14)
        adx_val = 0.0
        if adx_df is not None and not adx_df.empty:
            adx_col = next((c for c in adx_df.columns if "ADX_" in c.upper()), None)
            if adx_col:
                adx_val = _safe_last(adx_df[adx_col])
                if math.isnan(adx_val):
                    adx_val = 0.0
                    
        # --- NEW: Volume (OBV, Relative Volume) ---
        
        obv_series = ta.obv(close, volume)
        obv_val = _safe_last(obv_series) if obv_series is not None else 0.0
        if math.isnan(obv_val):
            obv_val = 0.0
            
        # OBV slope: difference over last 5 days, normalized
        
        obv_slope = 0.0
        if obv_series is not None and len(obv_series.dropna()) >= 6:
            obv_clean = obv_series.dropna()
            obv_diff = float(obv_clean.iloc[-1] - obv_clean.iloc[-6])
            obv_denom = max(abs(float(obv_clean.iloc[-6])), 1.0)
            obv_slope = obv_diff / obv_denom
            
        # Relative volume: today's volume / 20-day average
       
        rel_vol = 1.0
        if len(volume) >= 20:
            avg_vol = float(volume.iloc[-20:].mean())
            if avg_vol > 0:
                rel_vol = float(volume.iloc[-1]) / avg_vol
                
        # --- NEW: Volatility (ATR, Historical Vol) ---
        
        atr_series = ta.atr(high, low, close, length=14)
        atr_val = _safe_last(atr_series) if atr_series is not None else 0.0
        if math.isnan(atr_val):
            atr_val = 0.0
        atr_pct = (atr_val / last_close * 100.0) if last_close > 0 else 0.0
        
        # Historical volatility: annualized std of log returns over 20 days
        
        hist_vol = 0.0
        if len(close) >= 21:
            log_returns = np.log(close.iloc[-21:] / close.iloc[-21:].shift(1)).dropna()
            if len(log_returns) >= 5:
                hist_vol = float(log_returns.std() * np.sqrt(252) * 100)  # annualized %
        
        # --- NEW: Support/Resistance (52w high/low distance) ---
        
        lookback_52w = min(len(close), 252)
        high_52w = float(close.iloc[-lookback_52w:].max())
        low_52w = float(close.iloc[-lookback_52w:].min())
        dist_to_high_pct = ((last_close - high_52w) / high_52w * 100.0) if high_52w > 0 else 0.0
        dist_to_low_pct = ((last_close - low_52w) / low_52w * 100.0) if low_52w > 0 else 0.0
        
        # --- Regime Detection ---
        
        regime = self._detect_regime(close, sma50, sma200_val, adx_val)
        
        # ==========================================
        # Score Computation
        # ==========================================
        # 1. RSI Score (regime-adjusted)
        
        if math.isnan(rsi):
            rsi_score = 0.0
            rsi = 50.0  # neutral default for ml_features
        else:
            if regime == "bull":
        
                # In bull: RSI < 40 = oversold, RSI > 80 = overbought
                rsi_score = _clamp((60.0 - rsi) / 25.0)
            elif regime == "bear":
                
                # In bear: RSI < 20 = oversold, RSI > 60 = overbought
                rsi_score = _clamp((40.0 - rsi) / 25.0)
            else:
                
                # Sideways: standard 30/70
                rsi_score = _clamp((50.0 - rsi) / 20.0)
        
        # 2. MACD Score
        
        scale = max(abs(last_close) * 0.01, 1e-9)
        macd_score = _clamp(macd_hist / scale)
        
        # 3. Bollinger Score
        if bb_upper and bb_lower and bb_upper != bb_lower:
            bb_range = max(bb_upper - bb_lower, 1e-12)
            bb_pos = (last_close - bb_mid) / (bb_range / 2.0)
            bb_score = _clamp(-bb_pos)
        else:
            bb_score = 0.0
        
        # 4. Trend Score (SMA cross + ADX strength)
        # Golden cross bias: SMA50 > SMA200 = bullish
        sma_cross_score = 0.0
        if sma200_val > 0:
            sma_ratio = (sma50 - sma200_val) / sma200_val
            sma_cross_score = _clamp(sma_ratio * 10.0)  # scaled
        
        # ADX amplifies the trend signal; weak ADX dampens it
        adx_factor = min(adx_val / 25.0, 1.5) if adx_val > 0 else 0.5
        trend_score = _clamp(sma_cross_score * adx_factor)
        
        # 5. Volume Score (OBV direction + volume confirmation)
        obv_direction_score = _clamp(obv_slope * 5.0)  # OBV rising = bullish
        vol_confirm = 0.0
        if rel_vol > 1.5:
        
            # High volume confirms the current price direction
            vol_confirm = 0.3 if (close.iloc[-1] > close.iloc[-2] if len(close) >= 2 else True) else -0.3
        volume_score = _clamp(0.6 * obv_direction_score + 0.4 * vol_confirm)
        
        # 6. Volatility Score (high vol = caution → slightly negative)
        volatility_score = 0.0
        if atr_pct > 0:
        
            # ATR% > 3% is high for equities, > 5% is very high
            vol_threshold = 5.0 if asset_class in {AssetClass.CRYPTO, AssetClass.COMMODITY} else 3.0
            volatility_score = _clamp(-(atr_pct - vol_threshold * 0.5) / vol_threshold)
        
        # 7. Support/Resistance Score
        # Near 52w low (within 10%) = buy bias; near 52w high (within 5%) = sell bias
        sr_score = 0.0
        if dist_to_high_pct > -5.0:
        
            # Near high → mildly bearish (resistance)
            sr_score = _clamp(-0.5 * (1.0 + dist_to_high_pct / 5.0))
        elif dist_to_low_pct < 10.0:
            
            # Near low → mildly bullish (support)
            sr_score = _clamp(0.5 * (1.0 - dist_to_low_pct / 10.0))
        
        # ==========================================
        # Composite Technical Score
        # ==========================================
        technical_score = float(
            0.20 * rsi_score
            + 0.15 * macd_score
            + 0.10 * bb_score
            + 0.20 * trend_score
            + 0.15 * volume_score
            + 0.10 * volatility_score
            + 0.10 * sr_score
        )
        ml_features = {
            "last_close": last_close,
        
            # Momentum (existing)
            "rsi14": float(rsi),
            "macd": float(macd_val) if not math.isnan(macd_val) else 0.0,
            "macd_signal": float(macd_signal) if not math.isnan(macd_signal) else 0.0,
            "macd_hist": float(macd_hist),
            "bb_lower": float(bb_lower),
            "bb_mid": float(bb_mid),
            "bb_upper": float(bb_upper),
            "bb_width": float(bb_width),
            
            # Trend (new)
            "sma50": float(sma50),
            "sma200": float(sma200_val),
            "adx": float(adx_val),
            
            # Volume (new)
            "obv": float(obv_val),
            "obv_slope": float(obv_slope),
            "relative_volume": float(rel_vol),
            
            # Volatility (new)
            "atr14": float(atr_val),
            "atr_pct": float(atr_pct),
            "hist_volatility_20d": float(hist_vol),
            
            # Support/Resistance (new)
            "high_52w": float(high_52w),
            "low_52w": float(low_52w),
            "dist_to_high_pct": float(dist_to_high_pct),
            "dist_to_low_pct": float(dist_to_low_pct),
            
            # Regime (new)
            "regime": regime,
            
            # Composite
            "technical_score": float(technical_score),
        }
        return {"score": technical_score, "ml_features": ml_features}
    
    
    # ------------------------------------------------------------------
    # Fundamentals (expanded)
    # ------------------------------------------------------------------
    def _compute_fundamentals_score(
        self, ingestion: AssetIngestionResult
    ) -> tuple[dict[str, Any], list[str]]:
        # For commodities/crypto we may not have meaningful fundamentals; keep it neutral.
        if ingestion.asset_class in {AssetClass.COMMODITY, AssetClass.CRYPTO}:
            ml = {
                "pe_ratio": 0.0, "price_to_book": 0.0, "ev_to_ebitda": 0.0,
                "revenue_growth": 0.0, "profit_margins": 0.0, "operating_margins": 0.0,
                "debt_to_equity": 0.0, "current_ratio": 0.0, "market_cap": 0.0,
                "fundamentals_score": 0.0,
            }
            return {"score": 0.0, "ml_features": ml}, [f"fundamentals neutral for asset_class={ingestion.asset_class}"]
        snapshot, warnings = _extract_fundamentals_yfinance(ingestion.ticker_resolved_yfinance)
        
        # --- Sector-relative P/E ---
        sector_pe_ref = _DEFAULT_SECTOR_PE
        if snapshot.sector_pe and snapshot.sector_pe > 0:
            sector_pe_ref = snapshot.sector_pe
        elif snapshot.sector:
            sector_pe_ref = _SECTOR_PE_DEFAULTS.get(
                snapshot.sector.lower(), _DEFAULT_SECTOR_PE
            )
        pe = snapshot.pe_ratio
        pe_relative_score = 0.0
        if pe is not None and pe > 0 and sector_pe_ref > 0:
        
            # pe < sector_pe → undervalued (+), pe > sector_pe → expensive (−)
            pe_relative_score = _clamp((sector_pe_ref - pe) / sector_pe_ref)
        
        # --- P/B Score ---
        pb = snapshot.price_to_book
        pb_score = 0.0
        if pb is not None and pb > 0:
        
            # < 1.0 → strong value (+1), 1-3 → neutral, > 3 → expensive (−1)
            pb_score = _clamp((2.0 - pb) / 2.0)
        
        # --- EV/EBITDA Score ---
        ev_ebitda = snapshot.ev_to_ebitda
        ev_ebitda_score = 0.0
        if ev_ebitda is not None and ev_ebitda > 0:
        
            # < 10 → cheap (+), 10-15 → fair, > 20 → expensive (−)
            ev_ebitda_score = _clamp((12.0 - ev_ebitda) / 10.0)
        
        # --- Revenue Growth Score ---
        rev_growth = snapshot.revenue_growth
        growth_score = 0.0
        if rev_growth is not None:
        
            # +20% growth → +1; −20% → −1; 0% → 0
            growth_score = _clamp(rev_growth / 0.20)
        
        # --- Margin Score (profit + operating margins) ---
        margin_score = 0.0
        margin_count = 0
        if snapshot.profit_margins is not None:
        
            # > 0.15 healthy (+), < 0.05 weak (−)
            pm_score = _clamp((snapshot.profit_margins - 0.10) / 0.10)
            margin_score += pm_score
            margin_count += 1
        if snapshot.operating_margins is not None:
            om_score = _clamp((snapshot.operating_margins - 0.10) / 0.10)
            margin_score += om_score
            margin_count += 1
        if margin_count > 0:
            margin_score /= margin_count
        
        # --- Debt Score ---
        dte = snapshot.debt_to_equity
        debt_score = 0.0
        if dte is not None:
        
            # D/E: < 50 good (+), > 200 risky (−). yfinance reports as percentage (e.g. 120 = 1.2x)
            debt_score = _clamp((100.0 - dte) / 100.0)
        
        # --- Liquidity Score ---
        cr = snapshot.current_ratio
        liquidity_score = 0.0
        if cr is not None and cr > 0:
        
            # > 1.5 liquid (+), < 1.0 risky (−)
            liquidity_score = _clamp((cr - 1.0) / 0.5)
        
        # --- India-Specific Enrichment ---
        india_bonus = 0.0
        if ingestion.asset_class == AssetClass.EQUITY_INDIA:
        
            # Delivery % score: > 50% institutional conviction (+), < 30% speculative (−)
            if ingestion.nse_delivery_pct is not None:
                india_bonus += _clamp((ingestion.nse_delivery_pct - 40.0) / 20.0) * 0.05
            
            # FII flow score: net positive = bullish
            if ingestion.nse_fii_net_buy_cr is not None:
                fii_score = _clamp(ingestion.nse_fii_net_buy_cr / 2000.0)  # ±2000cr scale
                india_bonus += fii_score * 0.05
        
        # --- Composite Fundamentals Score ---
        fundamentals_score = float(
            0.20 * pe_relative_score
            + 0.10 * pb_score
            + 0.15 * ev_ebitda_score
            + 0.20 * growth_score
            + 0.15 * margin_score
            + 0.10 * debt_score
            + 0.10 * liquidity_score
            + india_bonus  # ±0.10 max for India
        )
        ml_features = {
            "pe_ratio": float(pe) if pe is not None else 0.0,
            "price_to_book": float(pb) if pb is not None else 0.0,
            "ev_to_ebitda": float(ev_ebitda) if ev_ebitda is not None else 0.0,
            "revenue_growth": float(rev_growth) if rev_growth is not None else 0.0,
            "profit_margins": float(snapshot.profit_margins) if snapshot.profit_margins is not None else 0.0,
            "operating_margins": float(snapshot.operating_margins) if snapshot.operating_margins is not None else 0.0,
            "debt_to_equity": float(dte) if dte is not None else 0.0,
            "current_ratio": float(cr) if cr is not None else 0.0,
            "market_cap": float(snapshot.market_cap) if snapshot.market_cap is not None else 0.0,
            "fundamentals_score": float(fundamentals_score),
        }
        
        # Add India-specific ML features
        if ingestion.asset_class == AssetClass.EQUITY_INDIA:
            ml_features["nse_delivery_pct"] = float(ingestion.nse_delivery_pct or 0.0)
            ml_features["nse_fii_net_buy_cr"] = float(ingestion.nse_fii_net_buy_cr or 0.0)
            ml_features["nse_dii_net_buy_cr"] = float(ingestion.nse_dii_net_buy_cr or 0.0)
        return {"score": fundamentals_score, "ml_features": ml_features}, warnings
    
    # ------------------------------------------------------------------
    # Signal Generation (expanded)
    # ------------------------------------------------------------------
    def _technical_signals(self, technical: dict[str, Any], bars_df: pd.DataFrame) -> list[str]:
        ml = technical["ml_features"]
        rsi = ml.get("rsi14", 0.0)
        macd_hist = ml.get("macd_hist", 0.0)
        bb_lower = ml.get("bb_lower", 0.0)
        bb_upper = ml.get("bb_upper", 0.0)
        last_close = ml.get("last_close", 0.0)
        regime = ml.get("regime", "sideways")
        sma50 = ml.get("sma50", 0.0)
        sma200 = ml.get("sma200", 0.0)
        adx = ml.get("adx", 0.0)
        rel_vol = ml.get("relative_volume", 1.0)
        atr_pct = ml.get("atr_pct", 0.0)
        dist_high = ml.get("dist_to_high_pct", 0.0)
        dist_low = ml.get("dist_to_low_pct", 0.0)
        out: list[str] = []
    
        # Regime
        out.append(f"Market regime: {regime.upper()} (ADX={adx:.1f})")
        
        # RSI
        if rsi <= 30:
            out.append("RSI is oversold (potential BUY pressure).")
        elif rsi >= 70:
            out.append("RSI is overbought (potential SELL pressure).")
        
        # MACD
        if macd_hist > 0:
            out.append("MACD histogram is positive (momentum supportive).")
        elif macd_hist < 0:
            out.append("MACD histogram is negative (momentum bearish).")
        
        # Bollinger
        if bb_lower and bb_upper:
            if last_close <= bb_lower:
                out.append("Price near/below lower Bollinger band (oversold bias).")
            elif last_close >= bb_upper:
                out.append("Price near/above upper Bollinger band (overbought bias).")
        
        # Trend (SMA cross)
        if sma50 > 0 and sma200 > 0:
            if sma50 > sma200 * 1.02:
                out.append(f"Golden cross: SMA50 ({sma50:.2f}) > SMA200 ({sma200:.2f}) — bullish trend.")
            elif sma50 < sma200 * 0.98:
                out.append(f"Death cross: SMA50 ({sma50:.2f}) < SMA200 ({sma200:.2f}) — bearish trend.")
        
        # Volume
        if rel_vol > 2.0:
            out.append(f"Volume surge: {rel_vol:.1f}× the 20-day average — high conviction move.")
        elif rel_vol > 1.5:
            out.append(f"Above-average volume ({rel_vol:.1f}×) — confirms recent price action.")
        
        # Volatility
        if atr_pct > 4.0:
            out.append(f"High volatility: ATR = {atr_pct:.1f}% of price — caution advised.")
        
        # Support/Resistance
        if dist_high > -3.0:
            out.append(f"Price near 52-week high ({dist_high:+.1f}%) — resistance zone.")
        elif dist_low < 5.0:
            out.append(f"Price near 52-week low ({dist_low:+.1f}%) — potential support zone.")
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
        ml = fundamentals.get("ml_features", {})
        out: list[str] = []
        if score >= 0.2:
            out.append(f"Fundamentals look relatively stronger (score={score:.2f}).")
        elif score <= -0.2:
            out.append(f"Fundamentals indicate elevated risk (score={score:.2f}).")
        else:
            out.append("Fundamentals are neutral or missing.")
    
        # Add specific callouts for notable metrics
        rev_g = ml.get("revenue_growth", 0.0)
        if rev_g > 0.15:
            out.append(f"Revenue growing at {rev_g:.0%} — strong growth signal.")
        elif rev_g < -0.05:
            out.append(f"Revenue declining at {rev_g:.0%} — contraction warning.")
        pe = ml.get("pe_ratio", 0.0)
        if pe > 40:
            out.append(f"P/E ratio is elevated at {pe:.1f}× — priced for high growth.")
        elif 0 < pe < 10:
            out.append(f"P/E ratio is low at {pe:.1f}× — potential value opportunity.")
        dte = ml.get("debt_to_equity", 0.0)
        if dte > 200:
            out.append(f"High leverage: D/E = {dte:.0f}% — balance sheet risk.")
        
        # India-specific
        nse_del = ml.get("nse_delivery_pct", 0.0)
        if nse_del > 0:
            if nse_del > 60:
                out.append(f"NSE delivery at {nse_del:.0f}% — strong institutional conviction.")
            elif nse_del < 30:
                out.append(f"NSE delivery at {nse_del:.0f}% — speculative activity dominant.")
        nse_fii = ml.get("nse_fii_net_buy_cr", 0.0)
        if nse_fii != 0.0:
            direction = "buying" if nse_fii > 0 else "selling"
            out.append(f"FII net {direction}: ₹{abs(nse_fii):.0f} Cr.")
        return out
        
