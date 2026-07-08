"""
Module A — multi-asset data ingestion (prices + headlines).

Production-oriented: retries, pacing, structured errors, injectable settings,
sync + async entrypoints for FastAPI / workers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx
import pandas as pd
import requests
import yfinance as yf
from tenacity import (
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from backend.schemas.ingestion import (
    AssetClass,
    AssetIngestionResult,
    HistorySource,
    NewsHeadline,
    NewsSource,
    OHLCVBar,
)
from backend.settings import IngestionSettings, get_ingestion_settings
logger = logging.getLogger(__name__)


def _gdelt_transient(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, (httpx.TimeoutException, httpx.TransportError, OSError))


_REQUEST_RETRY_EX = (
    requests.exceptions.RequestException,
    TimeoutError,
    OSError,
    ConnectionError,
)


@dataclass(frozen=True)
class _TickerContext:
    requested: str
    asset_class: AssetClass
    yfinance_ticker: str
    nse_symbol: str | None


def ohlcv_bars_to_dataframe(bars: list[OHLCVBar]) -> pd.DataFrame:
    """Build a DataFrame indexed by date for pandas-ta / research workflows."""
    if not bars:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    rows = []
    for b in bars:
        rows.append(
            {
                "date": b.date,
                "open": float(b.open),
                "high": float(b.high),
                "low": float(b.low),
                "close": float(b.close),
                "volume": b.volume,
            }
        )
    df = pd.DataFrame(rows).set_index("date").sort_index()
    return df


class MultiAssetDataProvider:
    """Fetches ~1y daily OHLCV (yfinance primary; nselib fallback for India) and up to N headlines (yfinance + GDELT)."""

    def __init__(self, settings: IngestionSettings | None = None) -> None:
        self._s = settings or get_ingestion_settings()
        # yfinance 0.2.5x may require its default (curl_cffi) session; do not inject requests.Session.
        self._http_user_agent = self._s.user_agent

    def ingest(self, ticker: str) -> AssetIngestionResult:
        warnings: list[str] = []
        errors: list[str] = []
        try:
            ctx = self._classify_ticker(ticker)
        except ValueError as e:
            return AssetIngestionResult(
                ticker_requested=ticker.strip(),
                ticker_resolved_yfinance=ticker.strip(),
                asset_class=AssetClass.EQUITY_GLOBAL,
                history_source=None,
                errors=[str(e)],
            )

        bars, hsrc, hw, he = self._fetch_history(ctx)
        warnings.extend(hw)
        errors.extend(he)

        headlines, nw = self._fetch_headlines(ctx)
        warnings.extend(nw)

        return AssetIngestionResult(
            ticker_requested=ctx.requested,
            ticker_resolved_yfinance=ctx.yfinance_ticker,
            asset_class=ctx.asset_class,
            history_source=hsrc,
            bars=bars,
            headlines=headlines,
            warnings=warnings,
            errors=errors,
        )

    async def ingest_async(self, ticker: str) -> AssetIngestionResult:
        """Run ingestion without blocking the event loop (Parallel history + news)."""
        try:
            ctx = self._classify_ticker(ticker)
        except ValueError as e:
            return AssetIngestionResult(
                ticker_requested=ticker.strip(),
                ticker_resolved_yfinance=ticker.strip(),
                asset_class=AssetClass.EQUITY_GLOBAL,
                history_source=None,
                errors=[str(e)],
            )

        loop = asyncio.get_running_loop()
        hist_fut = loop.run_in_executor(None, lambda: self._fetch_history(ctx))
        news_fut = loop.run_in_executor(None, lambda: self._fetch_headlines(ctx))

        (bars, hsrc, hw, he) = await hist_fut
        (headlines, nw) = await news_fut

        return AssetIngestionResult(
            ticker_requested=ctx.requested,
            ticker_resolved_yfinance=ctx.yfinance_ticker,
            asset_class=ctx.asset_class,
            history_source=hsrc,
            bars=bars,
            headlines=headlines,
            warnings=list(hw) + list(nw),
            errors=list(he),
        )

    def _classify_ticker(self, raw: str) -> _TickerContext:
        t = raw.strip()
        if not t:
            raise ValueError("ticker must be non-empty")

        up = t.upper()
        if self._looks_commodity(up):
            return _TickerContext(t, AssetClass.COMMODITY, up, None)

        if self._looks_crypto(up):
            yf_t = self._normalize_crypto_yf(up)
            return _TickerContext(t, AssetClass.CRYPTO, yf_t, None)

        if up.endswith(".NS") or up.endswith(".BO"):
            sym = up.split(".")[0]
            return _TickerContext(t, AssetClass.EQUITY_INDIA, up, sym)

        return _TickerContext(t, AssetClass.EQUITY_GLOBAL, up, None)

    @staticmethod
    def _looks_commodity(ticker: str) -> bool:
        return "=F" in ticker or ticker.endswith("=F")

    @staticmethod
    def _looks_crypto(ticker: str) -> bool:
        if "-USD" in ticker or "-USDT" in ticker:
            return True
        base = ticker.split("-")[0]
        return base in {"BTC", "BITCOIN", "ETH", "ETHEREUM", "SOL", "XRP"}

    @staticmethod
    def _normalize_crypto_yf(ticker: str) -> str:
        up = ticker.upper()
        if "BITCOIN" in up:
            return "BTC-USD"
        if "ETHEREUM" in up:
            return "ETH-USD"
        if up in {"BTC", "ETH"}:
            return f"{up}-USD"
        return up

    def _fetch_history(
        self, ctx: _TickerContext
    ) -> tuple[list[OHLCVBar], HistorySource | None, list[str], list[str]]:
        warnings: list[str] = []
        errors: list[str] = []

        try:
            yf_df = self._download_yfinance_history(ctx.yfinance_ticker)
        except Exception as e:
            logger.warning("yfinance history failed for %s: %s", ctx.yfinance_ticker, e)
            yf_df = pd.DataFrame()
            warnings.append(f"yfinance history error (will try fallbacks if applicable): {e!s}")

        bars: list[OHLCVBar] = []
        hsrc: HistorySource | None = None

        if yf_df is not None and not yf_df.empty:
            bars = self._bars_from_yfinance_df(yf_df)
            hsrc = HistorySource.YFINANCE

        if not bars and ctx.asset_class == AssetClass.EQUITY_INDIA and ctx.nse_symbol:
            try:
                n_df = self._download_nselib_history(ctx.nse_symbol)
                if n_df is not None and not n_df.empty:
                    bars = self._bars_from_nselib_df(n_df)
                    if bars:
                        hsrc = HistorySource.NSELIB
                        warnings.append("Used NSELib fallback because Yahoo Finance returned no rows.")
            except Exception as e:
                logger.warning("nselib history failed for %s: %s", ctx.nse_symbol, e)
                warnings.append(f"nselib fallback failed: {e!s}")

        if not bars:
            errors.append("No daily bars returned for the requested window (all providers).")

        return bars, hsrc, warnings, errors

    def _download_yfinance_history(self, ticker: str) -> pd.DataFrame:
        @retry(
            reraise=True,
            stop=stop_after_attempt(self._s.yfinance_max_attempts),
            wait=wait_exponential(
                multiplier=1,
                min=self._s.yfinance_min_wait_seconds,
                max=self._s.yfinance_max_wait_seconds,
            ),
            retry=retry_if_exception_type(_REQUEST_RETRY_EX),
        )
        def _call() -> pd.DataFrame:
            if self._s.yfinance_throttle_seconds > 0:
                time.sleep(self._s.yfinance_throttle_seconds)
            t = yf.Ticker(ticker)
            df = t.history(
                period=self._s.history_period,
                interval=self._s.history_interval,
                auto_adjust=True,
            )
            return df

        return _call()

    def _download_nselib_history(self, nse_symbol: str) -> pd.DataFrame:
        from nselib import capital_market

        end = date.today()
        start = end - timedelta(days=370)

        @retry(
            reraise=True,
            stop=stop_after_attempt(self._s.nselib_max_attempts),
            wait=wait_exponential(
                multiplier=1,
                min=self._s.nselib_min_wait_seconds,
                max=self._s.nselib_max_wait_seconds,
            ),
            retry=retry_if_exception_type(_REQUEST_RETRY_EX),
        )
        def _call() -> pd.DataFrame:
            return capital_market.price_volume_and_deliverable_position_data(
                symbol=nse_symbol,
                from_date=start.strftime("%d-%m-%Y"),
                to_date=end.strftime("%d-%m-%Y"),
            )

        return _call()

    @staticmethod
    def _scrub_nselib_columns(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.columns = [re.sub(r'^[\ufeff"]+|["]+$', "", str(c).strip()) for c in out.columns]
        out.columns = [str(c).strip() for c in out.columns]
        return out

    def _bars_from_yfinance_df(self, df: pd.DataFrame) -> list[OHLCVBar]:
        need = {"Open", "High", "Low", "Close", "Volume"}
        if not need.issubset(set(df.columns)):
            raise ValueError("unexpected yfinance OHLCV schema")

        rows: list[OHLCVBar] = []
        for idx, row in df.iterrows():
            try:
                d = idx.date() if hasattr(idx, "date") else pd.Timestamp(idx).date()
            except Exception:
                continue

            ohlc = {
                "open": row["Open"],
                "high": row["High"],
                "low": row["Low"],
                "close": row["Close"],
                "volume": row["Volume"],
            }
            if any(pd.isna(v) for v in ohlc.values() if v is not None):
                continue
            try:
                rows.append(
                    OHLCVBar(
                        date=d,
                        open=ohlc["open"],
                        high=ohlc["high"],
                        low=ohlc["low"],
                        close=ohlc["close"],
                        volume=ohlc["volume"],
                    )
                )
            except Exception:
                continue

        rows.sort(key=lambda b: b.date)
        return rows[-260:] if len(rows) > 260 else rows

    def _bars_from_nselib_df(self, df: pd.DataFrame) -> list[OHLCVBar]:
        dfn = self._scrub_nselib_columns(df)
        colmap = {
            "open": self._first_present(dfn, ["OpenPrice", "Open", "OPEN"]),
            "high": self._first_present(dfn, ["HighPrice", "High", "HIGH"]),
            "low": self._first_present(dfn, ["LowPrice", "Low", "LOW"]),
            "close": self._first_present(dfn, ["ClosePrice", "Close", "CLOSE", "LastPrice"]),
            "volume": self._first_present(
                dfn, ["TotalTradedQuantity", "Volume", "VOLUME", "Share Turnover"]
            ),
            "day": self._first_present(dfn, ["Date", "DATE", "TradeDate"]),
        }
        if not all(colmap[k] for k in ("open", "high", "low", "close", "volume", "day")):
            raise ValueError("unexpected nselib price/volume schema")

        tmp = dfn.assign(
            _day=pd.to_datetime(dfn[colmap["day"]], dayfirst=True, errors="coerce")
        ).dropna(subset=["_day"])
        tmp = tmp.sort_values("_day")

        rows: list[OHLCVBar] = []
        for _, row in tmp.iterrows():
            d = row["_day"].date()
            try:
                rows.append(
                    OHLCVBar(
                        date=d,
                        open=row[colmap["open"]],
                        high=row[colmap["high"]],
                        low=row[colmap["low"]],
                        close=row[colmap["close"]],
                        volume=row[colmap["volume"]],
                    )
                )
            except Exception:
                continue

        return rows[-260:] if len(rows) > 260 else rows

    @staticmethod
    def _first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
        cols = {c: c for c in df.columns}
        for c in candidates:
            if c in cols:
                return c
        lower_map = {str(c).lower(): c for c in df.columns}
        for c in candidates:
            lc = c.lower()
            if lc in lower_map:
                return lower_map[lc]
        return None

    def _fetch_headlines(self, ctx: _TickerContext) -> tuple[list[NewsHeadline], list[str]]:
        warnings: list[str] = []
        cap = self._s.news_headline_limit
        y_items: list[NewsHeadline] = []

        try:
            if self._s.yfinance_throttle_seconds > 0:
                time.sleep(self._s.yfinance_throttle_seconds)
            t = yf.Ticker(ctx.yfinance_ticker)
            news = getattr(t, "news", None) or []
            for n in news[:cap]:
                title = (n.get("title") or "").strip()
                if not title:
                    continue
                pub = n.get("providerPublishTime")
                dt_utc: datetime | None = None
                if isinstance(pub, (int, float)):
                    dt_utc = datetime.fromtimestamp(int(pub), tz=timezone.utc)
                y_items.append(
                    NewsHeadline(
                        title=title[:500],
                        url=n.get("link"),
                        publisher=n.get("publisher"),
                        published_at_utc=dt_utc,
                        source=NewsSource.YFINANCE,
                    )
                )
        except Exception as e:
            logger.warning("yfinance news failed for %s: %s", ctx.yfinance_ticker, e)
            warnings.append(f"yfinance news error: {e!s}")

        merged: list[NewsHeadline] = list(y_items)
        need = max(0, cap - len(merged))
        if need > 0:
            q = self._gdelt_query(ctx)
            try:
                g_items = self._download_gdelt_headlines(q, need)
                merged.extend(g_items)
            except Exception as e:
                logger.warning("GDELT headlines failed (%s): %s", q, e)
                warnings.append(f"GDELT news error (continuing with yfinance-only): {e!s}")

        merged = self._dedupe_headlines(merged)[:cap]
        return merged, warnings

    def _gdelt_query(self, ctx: _TickerContext) -> str:
        if ctx.asset_class == AssetClass.COMMODITY:
            return ctx.yfinance_ticker
        if ctx.asset_class == AssetClass.CRYPTO:
            base = ctx.yfinance_ticker.split("-")[0]
            return f"{base} cryptocurrency"
        if ctx.nse_symbol:
            return ctx.nse_symbol
        return ctx.yfinance_ticker.split(".")[0]

    def _download_gdelt_headlines(self, query: str, limit: int) -> list[NewsHeadline]:
        @retry(
            reraise=True,
            stop=stop_after_attempt(self._s.gdelt_max_attempts),
            wait=wait_exponential(
                multiplier=1,
                min=self._s.gdelt_min_wait_seconds,
                max=self._s.gdelt_max_wait_seconds,
            ),
            retry=retry_if_exception(_gdelt_transient),
        )
        def _call() -> list[NewsHeadline]:
            limits = httpx.Limits(max_connections=self._s.httpx_max_connections)
            params = {
                "query": query,
                "mode": self._s.gdelt_mode,
                "maxrecords": str(limit),
                "format": "json",
            }
            with httpx.Client(
                timeout=httpx.Timeout(self._s.http_timeout_seconds),
                limits=limits,
                headers={"User-Agent": self._s.user_agent},
            ) as client:
                r = client.get(str(self._s.gdelt_base_url), params=params)
                r.raise_for_status()
                text = (r.text or "").strip()
                if not text:
                    return []
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    logger.warning("GDELT returned non-JSON body (query=%s)", query)
                    return []

            articles = _extract_gdelt_articles(payload)
            out: list[NewsHeadline] = []
            for a in articles:
                title = (
                    str(a.get("title") or a.get("Title") or a.get("seotitle") or "")
                    .strip()
                )
                if not title:
                    continue
                url_u = a.get("url") or a.get("URL") or a.get("link")
                if isinstance(url_u, str):
                    url_u = url_u.strip() or None
                else:
                    url_u = None

                seen = a.get("seendate") or a.get("seen") or a.get("datetime")
                dt_utc = _parse_gdelt_time(str(seen)) if seen else None

                out.append(
                    NewsHeadline(
                        title=title[:500],
                        url=url_u,
                        publisher=None,
                        published_at_utc=dt_utc,
                        source=NewsSource.GDELT,
                    )
                )
                if len(out) >= limit:
                    break
            return out

        return _call()

    @staticmethod
    def _dedupe_headlines(items: list[NewsHeadline]) -> list[NewsHeadline]:
        seen: set[str] = set()
        out: list[NewsHeadline] = []
        for h in items:
            key = re.sub(r"\s+", " ", h.title.lower().strip())
            if key in seen:
                continue
            seen.add(key)
            out.append(h)
        return out


def _extract_gdelt_articles(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        for k in ("articles", "article_list", "docs", "data"):
            v = payload.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
        art = payload.get("articles")
        if isinstance(art, dict) and "results" in art and isinstance(art["results"], list):
            return [x for x in art["results"] if isinstance(x, dict)]
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    return []


def _parse_gdelt_time(raw: str) -> datetime | None:
    raw = raw.strip()
    if len(raw) >= 14 and raw[:14].isdigit():
        try:
            return datetime.strptime(raw[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def default_provider() -> MultiAssetDataProvider:
    return MultiAssetDataProvider()
