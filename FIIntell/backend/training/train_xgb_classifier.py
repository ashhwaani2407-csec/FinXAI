"""Train an XGBoost classifier and serialize it to `models/xgb_classifier.pkl`.

Rebuilt pipeline featuring:
- Multi-horizon labeling (5-day smoothed return).
- Chronological split (preventing data leakage).
- Calibrated probability outputs (Platt scaling via CalibratedClassifierCV).
- Asset-class diverse universe (Equities, Crypto, Commodities).
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
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from backend.data_provider import MultiAssetDataProvider, ohlcv_bars_to_dataframe
from backend.module_b_feature_engineering import _gpr_mock, FeatureEngineer
from backend.settings import IngestionSettings

logger = logging.getLogger(__name__)


def build_dataset_for_ticker(
    fe: FeatureEngineer,
    ingestion: Any,
    max_days: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build dataset returning (X, y, dates) for chronological sorting."""
    empty_X = np.zeros((0, 4), dtype=np.float32)
    empty_y = np.zeros((0,), dtype=np.int64)
    empty_dates = np.zeros((0,), dtype="datetime64[ns]")
    
    if not ingestion.bars:
        return empty_X, empty_y, empty_dates

    df = ohlcv_bars_to_dataframe(ingestion.bars)
    if df.empty or "close" not in df.columns:
        return empty_X, empty_y, empty_dates

    closes = df["close"].astype(float).values
    dates_s = pd.to_datetime(df.index)
    n = len(closes)
    
    # We need t+5 for the 5-day return label
    if n < 65:
        return empty_X, empty_y, empty_dates

    # Constant group scores for this ticker snapshot.
    s, _breakdown, _sw = fe._sentiment.analyze(
        ticker=ingestion.ticker_resolved_yfinance,
        asset_class=ingestion.asset_class,
        headlines=ingestion.headlines,
    )
    s = float(s)
    fundamentals, _fw = fe._compute_fundamentals_score(ingestion)  # noqa: SLF001
    _gpr_index, gpr_score = _gpr_mock(ingestion.ticker_resolved_yfinance)

    f = float(fundamentals.get("score") or 0.0)
    g = float(gpr_score)

    # We need close(t+5), so t must be up to n-6.
    last_t = n - 6
    first_t = max(50, last_t - max_days + 1)  # 50 to satisfy indicator warmups

    if first_t > last_t:
        return empty_X, empty_y, empty_dates

    X_rows: list[list[float]] = []
    y_rows: list[int] = []
    dates_list: list[Any] = []
    
    for t in range(first_t, last_t + 1):
        hist_slice = df.iloc[: t + 1]
        technical = fe._compute_technical_scores(hist_slice, ingestion.asset_class)  # noqa: SLF001
        tscore = float(technical.get("score") or 0.0)

        # 5-day smoothed direction label (t+5 vs t)
        label = 1 if closes[t + 5] > closes[t] else 0
        
        X_rows.append([tscore, s, f, g])
        y_rows.append(label)
        dates_list.append(dates_s[t])

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int64)
    dates_arr = np.array(dates_list, dtype="datetime64[ns]")
    
    return X, y, dates_arr


def train_and_serialize(
    tickers: list[str],
    enable_finbert: bool,
    max_days_per_ticker: int,
    history_period: str,
    model_out_path: Path,
) -> dict[str, Any]:
    ingestion_settings = IngestionSettings(
        enable_finbert=enable_finbert,
        history_period=history_period,  # Usually "3y"
    )
    fe = FeatureEngineer(settings=ingestion_settings)
    provider = MultiAssetDataProvider(settings=ingestion_settings)

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    dates_parts: list[np.ndarray] = []

    for ticker in tickers:
        logger.info("Ingesting for training: %s", ticker)
        ingestion = provider.ingest(ticker)
        if ingestion.errors and not ingestion.bars:
            logger.warning("Skipping %s: no bars (%s)", ticker, ingestion.errors[:1])
            continue

        X, y, dates_arr = build_dataset_for_ticker(fe, ingestion, max_days=max_days_per_ticker)
        if len(y) == 0:
            logger.warning("Skipping %s: insufficient history for training samples", ticker)
            continue

        X_parts.append(X)
        y_parts.append(y)
        dates_parts.append(dates_arr)

    if not X_parts:
        raise RuntimeError("No training samples could be built. Try different tickers or enable_finbert=false.")

    # Combine across all tickers
    X_full = np.vstack(X_parts)
    y_full = np.concatenate(y_parts)
    dates_full = np.concatenate(dates_parts)

    # Chronological sort across all tickers to prevent look-ahead bias
    sort_idx = np.argsort(dates_full)
    X_sorted = X_full[sort_idx]
    y_sorted = y_full[sort_idx]
    dates_sorted = dates_full[sort_idx]
    
    total_samples = len(y_sorted)
    logger.info(f"Total samples across {len(tickers)} tickers: {total_samples}")

    # Chronological Split (80/20)
    split_idx = int(total_samples * 0.8)
    X_train, X_test = X_sorted[:split_idx], X_sorted[split_idx:]
    y_train, y_test = y_sorted[:split_idx], y_sorted[split_idx:]
    
    logger.info(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")

    base_model = XGBClassifier(
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

    # Use TimeSeriesSplit for internal calibration cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Wrap in calibration for true probabilities
    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",  # Platt scaling
        cv=tscv,
        n_jobs=1
    )

    calibrated_model.fit(X_train, y_train)
    proba = calibrated_model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    acc = float(accuracy_score(y_test, pred))
    try:
        auc = float(roc_auc_score(y_test, proba))
    except Exception:
        auc = None
        
    try:
        brier = float(brier_score_loss(y_test, proba))
    except Exception:
        brier = None

    model_out_path.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(calibrated_model, model_out_path)

    meta = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "enable_finbert": enable_finbert,
        "tickers_used": tickers,
        "history_period": history_period,
        "samples": {
            "total_shape": list(X_full.shape),
            "train_shape": list(X_train.shape),
            "test_shape": list(X_test.shape),
            "y_pos_rate_train": float(np.mean(y_train)),
            "y_pos_rate_test": float(np.mean(y_test)),
        },
        "metrics": {
            "accuracy": acc, 
            "roc_auc": auc,
            "brier_score": brier
        },
        "feature_order": ["technical_score", "sentiment_score", "fundamentals_score", "geopolitics_score"],
        "label_definition": "y=1 if close(t+5)>close(t) else 0",
        "split_strategy": "Chronological (80/20)",
        "calibration": "CalibratedClassifierCV (sigmoid) with TimeSeriesSplit",
    }

    meta_path = model_out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Train FIIntell XGBoost classifier (Module C model mode).")
    
    # 50 Diverse tickers (Equities, Tech, Crypto, Commodities, ETFs)
    default_tickers = [
        # Tech / Mega Cap
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX", "AMD", "INTC",
        # Broad Market & Sectors
        "SPY", "QQQ", "IWM", "XLF", "XLV", "XLE", "XLI", "XLK", "XLP", "XLU",
        # Cryptocurrencies
        "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
        # Commodities
        "GC=F", "CL=F", "SI=F", "HG=F", "NG=F", "ZC=F",
        # Indian/Global Equities
        "RELIANCE.NS", "TCS.NS", "INFY", "HDB", "IBN",
        # Financials / Consumer
        "JPM", "V", "WMT", "JNJ", "PG", "DIS", "HD", "UNH", "KO",
        # Industrials / Energy
        "XOM", "CVX", "CAT", "BA", "GE"
    ]
    
    parser.add_argument("--tickers", nargs="*", default=default_tickers)
    parser.add_argument("--enable-finbert", action="store_true", help="Use FinBERT scoring (slower).")
    parser.add_argument("--max-days-per-ticker", type=int, default=1000)
    parser.add_argument("--history-period", type=str, default="3y")
    parser.add_argument("--model-out", type=str, default="models/xgb_classifier.pkl")
    args = parser.parse_args()

    meta = train_and_serialize(
        tickers=args.tickers,
        enable_finbert=args.enable_finbert,
        max_days_per_ticker=args.max_days_per_ticker,
        history_period=args.history_period,
        model_out_path=Path(args.model_out),
    )
    logger.info("Training complete. Metrics: %s", meta["metrics"])


if __name__ == "__main__":
    main()
