# Walkthrough: Technical + Fundamental Features Upgrade

## Summary

Upgraded the feature engineering pipeline from 3 basic indicators to a comprehensive 22-feature system with regime detection, expanded fundamentals, and India-specific NSE enrichment.

## Changes Made

### 1. Technical Indicators (3 → 15 features)

**Before**: RSI(14), MACD(12,26,9), Bollinger Bands — score = 40% RSI + 30% MACD + 30% BB

**After**: 7 indicator groups with weighted scoring:

| Weight | Indicator | What it measures |
|--------|-----------|-----------------|
| 20% | RSI (regime-adjusted) | Momentum — RSI thresholds shift based on bull/bear/sideways regime |
| 15% | MACD histogram | Momentum direction |
| 10% | Bollinger Band position | Mean-reversion signal |
| 20% | **SMA50/200 cross + ADX** | Trend strength (golden/death cross, amplified by ADX) |
| 15% | **OBV slope + relative volume** | Volume confirmation of price moves |
| 10% | **ATR + historical volatility** | Risk/volatility caution |
| 10% | **52w high/low distance** | Support/resistance proximity |

### 2. Regime Detection

New `_detect_regime()` method classifies the market as:
- **Bull**: Price > SMA200, SMA50 > SMA200, ADX > 20
- **Bear**: Price < SMA200, SMA50 < SMA200, ADX > 20
- **Sideways**: ADX < 20 (no clear trend)

RSI interpretation adjusts per regime:
- **Bull**: Oversold at RSI < 40 (not 30), overbought at RSI > 80 (not 70)
- **Bear**: Oversold at RSI < 20, overbought at RSI > 60
- **Sideways**: Standard 30/70

### 3. Fundamentals (3 → 9 metrics)

**Before**: P/E, Debt/Equity, Market Cap — simple average

**After**: Weighted composite from 7 scored dimensions:

| Weight | Metric | Scoring Logic |
|--------|--------|--------------|
| 20% | Sector-relative P/E | Compared to sector median (hardcoded fallback table) |
| 10% | Price-to-Book | < 1 = undervalued (+), > 3 = expensive (−) |
| 15% | EV/EBITDA | < 10 = cheap (+), > 20 = expensive (−) |
| 20% | Revenue Growth | Normalized by ±20% scale |
| 15% | Profit + Operating Margins | > 15% = healthy (+), < 5% = weak (−) |
| 10% | Debt-to-Equity | < 50% good (+), > 200% risky (−) |
| 10% | Current Ratio | > 1.5 = liquid (+), < 1.0 = risky (−) |

### 4. India-Specific NSE Enrichment

For `EQUITY_INDIA` tickers:
- **Delivery %**: Pulled from `nselib.price_volume_and_deliverable_position_data` — high delivery (>50%) signals institutional conviction
- **FII/DII Flows**: Pulled from `nselib.fii_dii_trading_activity` — net FII buying is bullish
- Both contribute up to ±0.10 bonus to the fundamentals score
- Graceful degradation: if NSE data is unavailable, scores default to 0

### 5. Bug Fixes
- Fixed `SentimentBreakdown` floating-point overflow: `neutral_pct` could be `100.00000000000001` due to float math, violating pydantic `le=100` — now clamped with `min(..., 100.0)`
- Removed 260-bar cap from data provider so 3-year history flows through

## Files Modified

| File | Changes |
|------|---------|
| [module_b_feature_engineering.py](file:///c:/Algorithm/FIIntell/backend/module_b_feature_engineering.py) | Complete rewrite — 12 new indicators, regime detection, expanded fundamentals |
| [data_provider.py](file:///c:/Algorithm/FIIntell/backend/data_provider.py) | Added `_fetch_nse_enrichment()`, removed 260 bar cap |
| [schemas/ingestion.py](file:///c:/Algorithm/FIIntell/backend/schemas/ingestion.py) | Added `nse_delivery_pct`, `nse_fii_net_buy_cr`, `nse_dii_net_buy_cr` |
| [schemas/features.py](file:///c:/Algorithm/FIIntell/backend/schemas/features.py) | Added `market_regime` field |
| [sentiment_engine.py](file:///c:/Algorithm/FIIntell/backend/sentiment_engine.py) | Fixed pct overflow bug |

## Live Test Results

| Ticker | Regime | Decision | Key Signals |
|--------|--------|----------|-------------|
| **AAPL** | BULL | HOLD (0.07) | Golden cross, near 52w high resistance, rev +17% |
| **BTC-USD** | SIDEWAYS | HOLD (0.05) | Death cross, fundamentals neutral (crypto), ADX=17.7 |
| **RELIANCE.NS** | SIDEWAYS | HOLD (0.12) | Near 52w low support, death cross, rev +30% |
