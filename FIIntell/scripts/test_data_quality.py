"""Unit tests for data quality scoring and per-ticker ingestion cache."""

from __future__ import annotations

import unittest
from datetime import date, timedelta
from decimal import Decimal

from backend.data_quality import assess_data_quality, expected_bars_for_period
from backend.ingestion_cache import IngestionCache, IngestionCacheKey, reset_ingestion_cache
from backend.module_c_decision_engine import DecisionEngine
from backend.schemas.data_quality import DataQualityGrade
from backend.schemas.decision import TradeAction
from backend.schemas.features import FeatureEngineeringResult
from backend.schemas.ingestion import AssetClass, AssetIngestionResult, OHLCVBar
from backend.settings import IngestionSettings


def _bar(d: date, close: float = 100.0, volume: int = 500_000) -> OHLCVBar:
    return OHLCVBar(
        date=d,
        open=Decimal(str(close)),
        high=Decimal(str(close + 1)),
        low=Decimal(str(close - 1)),
        close=Decimal(str(close)),
        volume=volume,
    )


def _healthy_bars(count: int = 252) -> list[OHLCVBar]:
    start = date.today() - timedelta(days=count + 30)
    bars: list[OHLCVBar] = []
    d = start
    while len(bars) < count:
        if d.weekday() < 5:
            bars.append(_bar(d, close=100.0 + len(bars) * 0.05))
        d += timedelta(days=1)
    return bars[-count:]


class DataQualityScoringTest(unittest.TestCase):
    def test_healthy_history_is_tradeable(self):
        ingestion = AssetIngestionResult(
            ticker_requested="AAPL",
            ticker_resolved_yfinance="AAPL",
            asset_class=AssetClass.EQUITY_GLOBAL,
            history_source=None,
            bars=_healthy_bars(),
        )
        report = assess_data_quality(ingestion, IngestionSettings(history_period="1y"))
        self.assertGreaterEqual(report.score, 75.0)
        self.assertTrue(report.is_tradeable)
        self.assertEqual(report.grade, DataQualityGrade.HIGH)

    def test_stale_delisted_symbol_blocks_trade(self):
        bars = _healthy_bars(120)
        for i in range(-10, 0):
            bars[i] = _bar(bars[i].date, close=float(bars[-11].close), volume=0)
        bars[-1] = _bar(date.today() - timedelta(days=30), close=float(bars[-2].close), volume=0)

        ingestion = AssetIngestionResult(
            ticker_requested="DEAD",
            ticker_resolved_yfinance="DEAD",
            asset_class=AssetClass.EQUITY_GLOBAL,
            history_source=None,
            bars=bars,
        )
        report = assess_data_quality(ingestion, IngestionSettings(history_period="1y"))
        self.assertFalse(report.is_tradeable)
        self.assertIn("Insufficient data — HOLD only.", report.summary)

    def test_no_bars_is_unusable(self):
        ingestion = AssetIngestionResult(
            ticker_requested="BAD",
            ticker_resolved_yfinance="BAD",
            asset_class=AssetClass.EQUITY_GLOBAL,
            history_source=None,
            errors=["No daily bars returned for the requested window (all providers)."],
        )
        report = assess_data_quality(ingestion)
        self.assertEqual(report.score, 0.0)
        self.assertEqual(report.grade, DataQualityGrade.UNUSABLE)
        self.assertFalse(report.is_tradeable)


class DecisionEngineDataQualityGateTest(unittest.TestCase):
    def test_low_quality_forces_hold_only(self):
        engine = DecisionEngine()
        features = FeatureEngineeringResult(
            asset_class=AssetClass.EQUITY_GLOBAL,
            technical_score=0.8,
            sentiment_score=0.5,
            fundamentals_score=0.3,
            geopolitics_score=0.0,
            gpr_index=0.0,
            gpr_score=0.0,
            ml_vector={"x": 1.0},
            data_quality=assess_data_quality(
                AssetIngestionResult(
                    ticker_requested="BAD",
                    ticker_resolved_yfinance="BAD",
                    asset_class=AssetClass.EQUITY_GLOBAL,
                    history_source=None,
                    bars=_healthy_bars(10),
                )
            ),
        )
        decision = engine.decide(features)
        self.assertEqual(decision.action, TradeAction.HOLD)
        self.assertEqual(decision.confidence_pct, 0.0)
        self.assertTrue(any("Insufficient data" in r for r in decision.reasoning))


class IngestionCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        reset_ingestion_cache()

    def tearDown(self) -> None:
        reset_ingestion_cache()

    def test_cache_isolated_per_ticker(self):
        cache = IngestionCache()
        settings = IngestionSettings(history_period="1y", history_interval="1d")

        aapl = AssetIngestionResult(
            ticker_requested="AAPL",
            ticker_resolved_yfinance="AAPL",
            asset_class=AssetClass.EQUITY_GLOBAL,
            history_source=None,
            bars=_healthy_bars(5),
        )
        msft = AssetIngestionResult(
            ticker_requested="MSFT",
            ticker_resolved_yfinance="MSFT",
            asset_class=AssetClass.EQUITY_GLOBAL,
            history_source=None,
            bars=_healthy_bars(5),
        )

        key_aapl = IngestionCacheKey.from_settings("AAPL", settings)
        key_msft = IngestionCacheKey.from_settings("MSFT", settings)
        cache.set(key_aapl, aapl, ttl_seconds=600)
        cache.set(key_msft, msft, ttl_seconds=600)

        hit_a = cache.get(key_aapl)
        hit_m = cache.get(key_msft)
        self.assertIsNotNone(hit_a)
        self.assertIsNotNone(hit_m)
        assert hit_a is not None and hit_m is not None
        self.assertEqual(hit_a.ticker_resolved_yfinance, "AAPL")
        self.assertEqual(hit_m.ticker_resolved_yfinance, "MSFT")
        self.assertNotEqual(hit_a.ticker_resolved_yfinance, hit_m.ticker_resolved_yfinance)


class ExpectedBarsTest(unittest.TestCase):
    def test_period_mapping(self):
        self.assertEqual(expected_bars_for_period("1y"), 252)
        self.assertEqual(expected_bars_for_period("6mo"), 126)


if __name__ == "__main__":
    unittest.main()
