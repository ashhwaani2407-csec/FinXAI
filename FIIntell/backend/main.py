"""FastAPI app for FIIntell.

Integration point (Modules A + B + C):
- Module A: `MultiAssetDataProvider` (history + headlines)
- Module B: `FeatureEngineer` (group scores)
- Module C: `DecisionEngine` (BUY/SELL/HOLD + confidence)
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI, HTTPException

from backend.data_provider import MultiAssetDataProvider
from backend.module_b_feature_engineering import FeatureEngineer
from backend.module_c_decision_engine import DecisionEngine
from backend.schemas.recommendation import (
    HealthResponse,
    RecommendBatchRequest,
    RecommendBatchResponse,
    RecommendBatchItem,
    RecommendRequest,
    RecommendResponse,
)
from backend.schemas.symbols import SymbolSearchResponse
from backend.settings import IngestionSettings
from backend.ticker_resolver import search_symbols

logger = logging.getLogger(__name__)

app = FastAPI(title="FIIntell", version="0.1.0")


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    return HealthResponse()


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    try:
        ingestion_settings = IngestionSettings(enable_finbert=req.enable_finbert)
        provider = MultiAssetDataProvider(settings=ingestion_settings)
        ingestion = provider.ingest(req.ticker)

        if ingestion.errors and not ingestion.bars:
            # Hard fail if there is no usable market data.
            raise HTTPException(status_code=502, detail="No market history could be fetched.")

        features = FeatureEngineer(settings=ingestion_settings).build_features(ingestion)
        decision = DecisionEngine().decide(features)

        return RecommendResponse(
            ingestion=ingestion,
            features=features,
            decision=decision,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("recommend failed for ticker=%s", req.ticker)
        raise HTTPException(status_code=500, detail=str(e))


def _run_pipeline_for_ticker(ticker: str, enable_finbert: bool) -> RecommendBatchItem:
    try:
        ingestion_settings = IngestionSettings(enable_finbert=enable_finbert)
        provider = MultiAssetDataProvider(settings=ingestion_settings)

        ingestion = provider.ingest(ticker)
        if ingestion.errors and not ingestion.bars:
            return RecommendBatchItem(
                ticker=ticker,
                ok=False,
                ingestion=ingestion,
                errors=ingestion.errors,
                warnings=ingestion.warnings,
            )

        features = FeatureEngineer(settings=ingestion_settings).build_features(ingestion)
        decision = DecisionEngine().decide(features)

        return RecommendBatchItem(
            ticker=ticker,
            ok=True,
            ingestion=ingestion,
            features=features,
            decision=decision,
            errors=[],
            warnings=ingestion.warnings,
        )
    except Exception as e:
        logger.exception("recommend batch failed for ticker=%s", ticker)
        return RecommendBatchItem(
            ticker=ticker,
            ok=False,
            ingestion=None,
            features=None,
            decision=None,
            errors=[str(e)],
        )


@app.post("/recommend/batch", response_model=RecommendBatchResponse)
async def recommend_batch(req: RecommendBatchRequest) -> RecommendBatchResponse:
    # Simple controlled concurrency so we don't trigger provider rate limits in batch usage.
    import asyncio

    max_concurrency = IngestionSettings().batch_max_concurrency
    sem = asyncio.Semaphore(max_concurrency)

    async def _guarded(t: str) -> RecommendBatchItem:
        async with sem:
            # Module A+B+C are mostly CPU/bound + sync I/O; offload to thread.
            return await asyncio.to_thread(_run_pipeline_for_ticker, t, req.enable_finbert)

    tasks = [_guarded(t) for t in req.tickers]
    items = await asyncio.gather(*tasks)
    return RecommendBatchResponse(items=items)


@app.get("/symbols/search", response_model=SymbolSearchResponse)
def symbols_search(q: str = "", limit: int = 25, asset_class: str | None = None) -> SymbolSearchResponse:
    settings = IngestionSettings()
    eff_limit = min(limit, settings.symbol_search_limit)
    items = search_symbols(q, eff_limit, asset_class=asset_class)
    return SymbolSearchResponse(query=q, items=items)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

