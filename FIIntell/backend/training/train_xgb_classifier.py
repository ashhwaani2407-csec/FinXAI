"""Train an XGBoost classifier and serialize it to `models/xgb_classifier.pkl`.

Labeling strategy (simple + deployable):
- For each ticker, use the last ~1y daily history (Module A).
- For multiple days t, compute technical score at day t (RSI/MACD/Bollinger) from
  the rolling OHLCV window.
- Use one snapshot of sentiment + fundamentals at the time of ingestion (not historical),
  and geopolitics via deterministic mock (GPR).
- Label y = 1 if close(t+1) > close(t) else 0.

This creates a practical model to unlock Module C's model-mode. For production-grade
multimodal training with full historical sentiment/geopolitics you would replace this with
time-aware feature sources and an appropriate labeling horizon.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from backend.data_provider import MultiAssetDataProvider, ohlcv_bars_to_dataframe
from backend.module_b_feature_engineering import _gpr_mock, FeatureEngineer
from backend.settings import IngestionSettings

logger = logging.getLogger(__name__)


def _safe_float(x: Any) -> float:
    try:
        if x is None:
            return 0.0
        return float(x)
    except Exception:
        return 0.0


def build_dataset_for_ticker(
    fe: FeatureEngineer,
    ingestion: Any,
    max_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not ingestion.bars:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    df = ohlcv_bars_to_dataframe(ingestion.bars)
    if df.empty or "close" not in df.columns:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    closes = df["close"].astype(float).values
    n = len(closes)
    if n < 60:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    # Compute constant group scores for this ticker snapshot.
    # - sentiment/fundamentals are computed once (latest headlines/info)
    # - geopolitics mock is deterministic
    sentiment = fe._compute_sentiment_score(ingestion.headlines)  # noqa: SLF001
    fundamentals, _fw = fe._compute_fundamentals_score(ingestion)  # noqa: SLF001
    _gpr_index, gpr_score = _gpr_mock(ingestion.ticker_resolved_yfinance)

    s = float(sentiment.get("score") or 0.0)
    f = float(fundamentals.get("score") or 0.0)
    g = float(gpr_score)

    # Build rolling technical score samples.
    # We need close(t+1), so t must be up to n-2.
    # Only use the most recent `max_days` samples.
    last_t = n - 2
    first_t = max(50, last_t - max_days)  # 50 to satisfy indicator warmups

    X_rows: list[list[float]] = []
    y_rows: list[int] = []
    for t in range(first_t, last_t + 1):
        hist_slice = df.iloc[: t + 1]
        technical = fe._compute_technical_scores(hist_slice, ingestion.asset_class)  # noqa: SLF001
        tscore = float(technical.get("score") or 0.0)

        label = 1 if closes[t + 1] > closes[t] else 0
        X_rows.append([tscore, s, f, g])
        y_rows.append(label)

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int64)
    return X, y


def train_and_serialize(
    tickers: list[str],
    enable_finbert: bool,
    max_days_per_ticker: int,
    model_out_path: Path,
) -> dict[str, Any]:
    ingestion_settings = IngestionSettings(enable_finbert=enable_finbert)
    fe = FeatureEngineer(settings=ingestion_settings)

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    provider = MultiAssetDataProvider(settings=ingestion_settings)

    for ticker in tickers:
        logger.info("Ingesting for training: %s", ticker)
        ingestion = provider.ingest(ticker)
        if ingestion.errors and not ingestion.bars:
            logger.warning("Skipping %s: no bars (%s)", ticker, ingestion.errors[:1])
            continue

        X, y = build_dataset_for_ticker(fe, ingestion, max_days=max_days_per_ticker)
        if len(y) == 0:
            logger.warning("Skipping %s: insufficient history for training samples", ticker)
            continue

        X_parts.append(X)
        y_parts.append(y)

    if not X_parts:
        raise RuntimeError("No training samples could be built. Try different tickers or enable_finbert=false.")

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    # Basic train/test split (you can later upgrade to proper time-series CV).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=max(1, os.cpu_count() or 2),
        random_state=42,
    )

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_test)

    pred = (proba >= 0.5).astype(int)
    acc = float(accuracy_score(y_test, pred))
    auc = None
    try:
        auc = float(roc_auc_score(y_test, proba))
    except Exception:
        auc = None

    model_out_path.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(model, model_out_path)

    meta = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "enable_finbert": enable_finbert,
        "tickers_used": tickers,
        "samples": {"X_shape": list(X.shape), "y_pos_rate": float(np.mean(y))},
        "metrics": {"accuracy": acc, "roc_auc": auc},
        "feature_order": ["technical_score", "sentiment_score", "fundamentals_score", "geopolitics_score"],
        "label_definition": "y=1 if close(t+1)>close(t) else 0",
    }

    meta_path = model_out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Train FIIntell XGBoost classifier (Module C model mode).")
    parser.add_argument("--tickers", nargs="*", default=["TSLA", "AAPL", "BTC-USD", "GC=F", "RELIANCE.NS"])
    parser.add_argument("--enable-finbert", action="store_true", help="Use FinBERT scoring (slower).")
    parser.add_argument("--max-days-per-ticker", type=int, default=60)
    parser.add_argument("--model-out", type=str, default="models/xgb_classifier.pkl")
    args = parser.parse_args()

    meta = train_and_serialize(
        tickers=args.tickers,
        enable_finbert=args.enable_finbert,
        max_days_per_ticker=args.max_days_per_ticker,
        model_out_path=Path(args.model_out),
    )
    logger.info("Training complete. Metrics: %s", meta["metrics"])


if __name__ == "__main__":
    main()

