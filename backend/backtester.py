from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import pandas as pd

from backend.audit_logger import get_audit_stats
from backend.data_provider import MultiAssetDataProvider, ohlcv_bars_to_dataframe
from backend.module_b_feature_engineering import FeatureEngineer
from backend.module_c_decision_engine import DecisionEngine
from backend.schemas.decision import TradeAction
from backend.schemas.ingestion import AssetClass, AssetIngestionResult
from backend.settings import IngestionSettings


def _signal_is_correct(action: str, return_5d_pct: float | None) -> bool | None:
    if return_5d_pct is None:
        return None
    if action == TradeAction.BUY.value:
        return return_5d_pct > 0.0
    if action == TradeAction.SELL.value:
        return return_5d_pct < 0.0
    return abs(return_5d_pct) <= 0.25


def _return_for_horizon(df: pd.DataFrame, idx: int, horizon: int) -> float | None:
    if idx + horizon >= len(df):
        return None
    close_now = float(df.iloc[idx]["close"])
    close_future = float(df.iloc[idx + horizon]["close"])
    if close_now <= 0:
        return None
    return (close_future / close_now - 1.0) * 100.0


def _summarize_hit_rates(signal_rows: list[dict[str, Any]]) -> dict[str, Any]:
    actions: dict[str, dict[str, int]] = defaultdict(lambda: {"support": 0, "correct": 0})
    per_asset: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"support": 0, "correct": 0}))

    for row in signal_rows:
        action = row["action"]
        asset_class = row.get("asset_class", "unknown")
        correct = row.get("correct")
        if correct is None:
            continue
        actions[action]["support"] += 1
        actions[action]["correct"] += int(correct)
        per_asset[asset_class][action]["support"] += 1
        per_asset[asset_class][action]["correct"] += int(correct)

    summary = {
        "overall": {},
        "by_asset_class": {},
    }

    for action, data in actions.items():
        support = data["support"]
        correct = data["correct"]
        summary["overall"][action] = {
            "support": support,
            "hit_rate": round(correct / support, 4) if support else 0.0,
            "precision": round(correct / support, 4) if support else 0.0,
            "recall": round(correct / support, 4) if support else 0.0,
        }

    for asset_class, action_map in per_asset.items():
        summary["by_asset_class"][asset_class] = {}
        for action, data in action_map.items():
            support = data["support"]
            correct = data["correct"]
            summary["by_asset_class"][asset_class][action] = {
                "support": support,
                "hit_rate": round(correct / support, 4) if support else 0.0,
                "precision": round(correct / support, 4) if support else 0.0,
                "recall": round(correct / support, 4) if support else 0.0,
            }

    return summary


def run_backtest(ticker: str, lookback_days: int = 90, enable_finbert: bool = True) -> dict[str, Any]:
    settings = IngestionSettings(enable_finbert=enable_finbert)
    provider = MultiAssetDataProvider(settings=settings)
    ingestion = provider.ingest(ticker)

    if ingestion.errors and not ingestion.bars:
        raise ValueError("No market history could be fetched for the requested ticker.")

    bars_df = ohlcv_bars_to_dataframe(ingestion.bars)
    if bars_df.empty:
        raise ValueError("No historical OHLCV bars available for backtest.")

    bars_df = bars_df.reset_index().rename(columns={"index": "date"})
    start_idx = max(0, len(bars_df) - max(lookback_days, 30))
    signal_rows: list[dict[str, Any]] = []
    pnl_curve = [10000.0]
    equity = 10000.0

    for idx in range(start_idx, len(bars_df) - 1):
        window_bars = ingestion.bars[: idx + 1]
        partial_ingestion = ingestion.model_copy(deep=True)
        partial_ingestion.bars = window_bars

        features = FeatureEngineer(settings=settings).build_features(partial_ingestion)
        decision = DecisionEngine().decide(features)
        action = decision.action.value
        forward_1d = _return_for_horizon(bars_df, idx, 1)
        forward_5d = _return_for_horizon(bars_df, idx, 5)
        forward_20d = _return_for_horizon(bars_df, idx, 20)

        correct = _signal_is_correct(action, forward_5d)
        row = {
            "date": bars_df.iloc[idx]["date"].isoformat(),
            "ticker": ingestion.ticker_resolved_yfinance,
            "asset_class": ingestion.asset_class.value,
            "action": action,
            "score": float(decision.score),
            "confidence_pct": float(decision.confidence_pct),
            "return_1d_pct": forward_1d,
            "return_5d_pct": forward_5d,
            "return_20d_pct": forward_20d,
            "correct": correct,
        }
        signal_rows.append(row)

        if action == TradeAction.BUY.value and forward_5d is not None:
            equity = equity * (1.0 + (forward_5d / 100.0))
        elif action == TradeAction.SELL.value and forward_5d is not None:
            equity = equity * (1.0 - (forward_5d / 100.0))
        pnl_curve.append(equity)

    metrics = _summarize_hit_rates(signal_rows)
    stats = get_audit_stats()

    return {
        "ticker": ingestion.ticker_resolved_yfinance,
        "asset_class": ingestion.asset_class.value,
        "lookback_days": lookback_days,
        "signal_count": len(signal_rows),
        "signals": signal_rows,
        "pnl_curve": pnl_curve,
        "hit_rates": metrics,
        "audit_stats": stats,
        "final_equity": round(equity, 2),
        "total_return_pct": round(((equity / 10000.0) - 1.0) * 100.0, 2),
    }
