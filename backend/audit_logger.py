from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from backend.schemas.decision import DecisionResult, TradeAction
from backend.schemas.features import FeatureEngineeringResult
from backend.schemas.ingestion import AssetClass, AssetIngestionResult

_DB_DIR = Path(__file__).resolve().parent.parent / "data"
_DB_PATH = _DB_DIR / "audit.db"


def _ensure_db() -> None:
    _DB_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                ticker TEXT NOT NULL,
                asset_class TEXT NOT NULL,
                action TEXT NOT NULL,
                score REAL NOT NULL,
                confidence_pct REAL NOT NULL,
                technical_score REAL NOT NULL,
                sentiment_score REAL NOT NULL,
                fundamentals_score REAL NOT NULL,
                geopolitics_score REAL NOT NULL,
                market_regime TEXT,
                close_at_signal REAL,
                close_1d REAL,
                close_5d REAL,
                close_20d REAL,
                return_1d_pct REAL,
                return_5d_pct REAL,
                return_20d_pct REAL,
                feature_snapshot TEXT,
                decision_snapshot TEXT
            )
            """
        )
        conn.commit()


def _safe_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _asset_class_value(asset_class: AssetClass | str | None) -> str:
    if isinstance(asset_class, str):
        return asset_class
    return getattr(asset_class, "value", str(asset_class))


def _compute_forward_returns(
    ticker: str,
    signal_bar_date: date | None,
    close_at_signal: float | None,
) -> tuple[float | None, float | None, float | None, float | None, float | None, float | None]:
    if close_at_signal is None or signal_bar_date is None:
        return None, None, None, None, None, None

    try:
        hist = yf.Ticker(ticker).history(period="365d", interval="1d")
        if hist.empty:
            return None, None, None, None, None, None

        hist = hist.reset_index(names="date")
        hist["date"] = pd.to_datetime(hist["date"], utc=True)

        signal_day = pd.Timestamp(signal_bar_date, tz="UTC")
        future = hist.loc[hist["date"] > signal_day].copy()
        if future.empty:
            return None, None, None, None, None, None

        future = future.sort_values("date").reset_index(drop=True)
        close = float(close_at_signal)

        def _lookup(offset: int) -> float | None:
            idx = offset
            if idx < 0 or idx >= len(future):
                return None
            close_value = pd.to_numeric(future.iloc[idx]["Close"], errors="coerce")
            if pd.isna(close_value):
                return None
            return float(close_value)

        c1 = _lookup(0)
        c5 = _lookup(4)
        c20 = _lookup(19)
        r1 = (c1 / close - 1.0) * 100.0 if c1 is not None else None
        r5 = (c5 / close - 1.0) * 100.0 if c5 is not None else None
        r20 = (c20 / close - 1.0) * 100.0 if c20 is not None else None
        return c1, c5, c20, r1, r5, r20
    except Exception:
        return None, None, None, None, None, None


def log_recommendation(
    ingestion: AssetIngestionResult,
    features: FeatureEngineeringResult,
    decision: DecisionResult,
    timestamp_utc: datetime | None = None,
) -> int:
    _ensure_db()
    signal_ts = timestamp_utc or datetime.now(timezone.utc)

    signal_index = -2 if len(ingestion.bars) >= 2 else -1
    signal_bar = ingestion.bars[signal_index] if ingestion.bars else None
    signal_bar_date = signal_bar.date if signal_bar is not None else signal_ts.date()
    close_at_signal = _safe_float((signal_bar.close if signal_bar is not None else None))
    c1, c5, c20, r1, r5, r20 = _compute_forward_returns(
        ingestion.ticker_resolved_yfinance,
        signal_bar_date,
        close_at_signal,
    )

    conn = sqlite3.connect(_DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO recommendations (
                timestamp_utc, ticker, asset_class, action, score, confidence_pct,
                technical_score, sentiment_score, fundamentals_score, geopolitics_score,
                market_regime, close_at_signal, close_1d, close_5d, close_20d,
                return_1d_pct, return_5d_pct, return_20d_pct,
                feature_snapshot, decision_snapshot
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                signal_ts.isoformat(),
                ingestion.ticker_resolved_yfinance,
                _asset_class_value(ingestion.asset_class),
                decision.action.value,
                float(decision.score),
                float(decision.confidence_pct),
                float(features.technical_score),
                float(features.sentiment_score),
                float(features.fundamentals_score),
                float(features.geopolitics_score),
                features.market_regime,
                close_at_signal,
                c1,
                c5,
                c20,
                r1,
                r5,
                r20,
                json.dumps(features.model_dump(), default=str),
                json.dumps(decision.model_dump(), default=str),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def get_recent_audit(limit: int = 20) -> list[dict[str, Any]]:
    _ensure_db()
    conn = sqlite3.connect(_DB_PATH)
    try:
        rows = conn.execute(
            """
            SELECT * FROM recommendations
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    finally:
        conn.close()

    columns = [
        "id", "timestamp_utc", "ticker", "asset_class", "action", "score", "confidence_pct",
        "technical_score", "sentiment_score", "fundamentals_score", "geopolitics_score",
        "market_regime", "close_at_signal", "close_1d", "close_5d", "close_20d",
        "return_1d_pct", "return_5d_pct", "return_20d_pct", "feature_snapshot", "decision_snapshot",
    ]
    return [dict(zip(columns, row)) for row in rows]


def get_audit_stats() -> dict[str, Any]:
    _ensure_db()
    conn = sqlite3.connect(_DB_PATH)
    try:
        rows = conn.execute(
            """
            SELECT action, asset_class, return_1d_pct, return_5d_pct, return_20d_pct
            FROM recommendations
            """
        ).fetchall()
    finally:
        conn.close()

    def _classify_return(ret: float | None) -> str | None:
        if ret is None:
            return None
        if ret > 0.25:
            return TradeAction.BUY.value
        if ret < -0.25:
            return TradeAction.SELL.value
        return TradeAction.HOLD.value

    def _outcome_for_action(action: str, ret: float | None) -> bool | None:
        if ret is None:
            return None
        if action == TradeAction.BUY.value:
            return ret > 0.0
        if action == TradeAction.SELL.value:
            return ret < 0.0
        return abs(ret) <= 0.25

    per_action: dict[str, dict[str, int]] = {}
    per_asset: dict[str, dict[str, dict[str, int]]] = {}
    horizon_results: dict[str, list[bool]] = {"1d": [], "5d": [], "20d": []}
    actual_action_counts: dict[str, int] = {}
    predicted_action_counts: dict[str, int] = {}

    for action, asset_class, ret1, ret5, ret20 in rows:
        for horizon, ret in (("1d", ret1), ("5d", ret5), ("20d", ret20)):
            outcome = _outcome_for_action(action, ret)
            if outcome is None:
                continue
            horizon_results[horizon].append(outcome)

        if ret5 is None:
            continue

        true_label = _classify_return(ret5)
        if true_label is None:
            continue

        actual_action_counts[true_label] = actual_action_counts.get(true_label, 0) + 1
        predicted_action_counts[action] = predicted_action_counts.get(action, 0) + 1

        action_bucket = per_action.setdefault(action, {"support": 0, "correct": 0, "tp": 0, "fp": 0, "fn": 0})
        asset_bucket = per_asset.setdefault(asset_class, {}).setdefault(action, {"support": 0, "correct": 0, "tp": 0, "fp": 0, "fn": 0})

        action_bucket["support"] += 1
        asset_bucket["support"] += 1

        if action == true_label:
            action_bucket["correct"] += 1
            asset_bucket["correct"] += 1
            action_bucket["tp"] += 1
            asset_bucket["tp"] += 1
        else:
            action_bucket["fp"] += 1
            asset_bucket["fp"] += 1

        for other_action, other_count in actual_action_counts.items():
            if other_action == true_label:
                continue
            if other_action == action:
                continue
            if other_action not in per_action:
                per_action[other_action] = {"support": 0, "correct": 0, "tp": 0, "fp": 0, "fn": 0}
            if other_action not in per_asset.setdefault(asset_class, {}):
                per_asset[asset_class][other_action] = {"support": 0, "correct": 0, "tp": 0, "fp": 0, "fn": 0}

    for action, data in per_action.items():
        data["fn"] = max(0, actual_action_counts.get(action, 0) - data["tp"])

    for asset_class, action_map in per_asset.items():
        for action, data in action_map.items():
            data["fn"] = max(0, actual_action_counts.get(action, 0) - data["tp"])

    stats = {
        "overall": {},
        "by_asset_class": {},
        "horizon": {},
    }

    for action, data in per_action.items():
        support = int(data["support"])
        correct = int(data["correct"])
        true_positive = int(data["tp"])
        false_positive = int(data["fp"])
        false_negative = int(data["fn"])
        precision = round(true_positive / (true_positive + false_positive), 4) if (true_positive + false_positive) else 0.0
        recall = round(true_positive / (true_positive + false_negative), 4) if (true_positive + false_negative) else 0.0
        stats["overall"][action] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "hit_rate": round(correct / support, 4) if support else 0.0,
        }

    for asset_class, action_map in per_asset.items():
        stats["by_asset_class"][asset_class] = {}
        for action, data in action_map.items():
            support = int(data["support"])
            correct = int(data["correct"])
            true_positive = int(data["tp"])
            false_positive = int(data["fp"])
            false_negative = int(data["fn"])
            precision = round(true_positive / (true_positive + false_positive), 4) if (true_positive + false_positive) else 0.0
            recall = round(true_positive / (true_positive + false_negative), 4) if (true_positive + false_negative) else 0.0
            stats["by_asset_class"][asset_class][action] = {
                "support": support,
                "precision": precision,
                "recall": recall,
                "hit_rate": round(correct / support, 4) if support else 0.0,
            }

    for horizon, outcomes in horizon_results.items():
        stats["horizon"][horizon] = {
            "support": len(outcomes),
            "hit_rate": round(sum(outcomes) / len(outcomes), 4) if outcomes else 0.0,
        }

    return stats


def backfill_forward_returns() -> int:
    _ensure_db()
    conn = sqlite3.connect(_DB_PATH)
    try:
        rows = conn.execute(
            "SELECT id, ticker, timestamp_utc, close_at_signal FROM recommendations WHERE return_1d_pct IS NULL OR return_5d_pct IS NULL OR return_20d_pct IS NULL"
        ).fetchall()
        updated = 0
        for row in rows:
            rid, ticker, timestamp_utc, close_at_signal = row
            signal_ts = datetime.fromisoformat(timestamp_utc)
            signal_bar_date = signal_ts.date()
            c1, c5, c20, r1, r5, r20 = _compute_forward_returns(ticker, signal_bar_date, close_at_signal)
            conn.execute(
                "UPDATE recommendations SET close_1d=?, close_5d=?, close_20d=?, return_1d_pct=?, return_5d_pct=?, return_20d_pct=? WHERE id=?",
                (c1, c5, c20, r1, r5, r20, rid),
            )
            updated += 1
        conn.commit()
        return updated
    finally:
        conn.close()
