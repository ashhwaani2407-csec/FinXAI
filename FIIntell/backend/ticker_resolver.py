"""Natural-language ticker resolver for US + Indian equities."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd
import yfinance as yf
from nselib import capital_market

from backend.schemas.symbols import SymbolSearchItem


@lru_cache(maxsize=1)
def _load_nse_equity_master() -> pd.DataFrame:
    df = capital_market.equity_list()
    if df is None or df.empty:
        return pd.DataFrame(columns=["SYMBOL", "NAME OF COMPANY"])
    cols = {str(c).strip().upper(): c for c in df.columns}
    sym_col = cols.get("SYMBOL")
    name_col = cols.get("NAME OF COMPANY")
    if sym_col is None or name_col is None:
        return pd.DataFrame(columns=["SYMBOL", "NAME OF COMPANY"])
    out = df[[sym_col, name_col]].copy()
    out.columns = ["SYMBOL", "NAME"]
    out["SYMBOL"] = out["SYMBOL"].astype(str).str.strip()
    out["NAME"] = out["NAME"].astype(str).str.strip()
    return out


def _search_nse(query: str, limit: int) -> list[SymbolSearchItem]:
    q = query.strip().lower()
    if not q:
        return []
    df = _load_nse_equity_master()
    if df.empty:
        return []

    sym_hit = df["SYMBOL"].str.lower().str.contains(q, regex=False)
    name_hit = df["NAME"].str.lower().str.contains(q, regex=False)
    hits = df[sym_hit | name_hit].head(limit)
    out: list[SymbolSearchItem] = []
    for _, row in hits.iterrows():
        symbol = str(row["SYMBOL"]).strip()
        name = str(row["NAME"]).strip()
        out.append(
            SymbolSearchItem(
                symbol=f"{symbol}.NS",
                display_name=name,
                asset_class="Stocks",
                market="India",
                exchange="NSE",
                score=0.7,
            )
        )
    return out


def _infer_asset_class(symbol: str, quote_type: str, name: str) -> str:
    s = (symbol or "").upper()
    qt = (quote_type or "").upper()
    nm = (name or "").lower()
    if qt in {"CRYPTOCURRENCY"} or "-USD" in s and s.split("-")[0] in {"BTC", "ETH", "SOL", "XRP", "BNB"}:
        return "Crypto"
    if "=F" in s or qt in {"FUTURE"}:
        return "Commodities"
    if qt in {"ETF", "FUND", "MUTUALFUND"} and any(k in nm for k in ["bond", "treasury", "gilt", "fixed income"]):
        return "Bonds"
    return "Stocks"


def _search_yf(query: str, limit: int) -> list[SymbolSearchItem]:
    q = query.strip()
    if not q:
        return []
    try:
        search = yf.Search(q, max_results=max(limit, 10))
        quotes: list[dict[str, Any]] = list(getattr(search, "quotes", []) or [])
    except Exception:
        return []

    out: list[SymbolSearchItem] = []
    for item in quotes:
        qtype = str(item.get("quoteType") or "").upper()
        symbol = str(item.get("symbol") or "").strip()
        if not symbol:
            continue
        name = str(item.get("longname") or item.get("shortname") or symbol).strip()
        exch = str(item.get("exchDisp") or item.get("exchange") or "").strip() or None
        market = "India" if symbol.endswith(".NS") or symbol.endswith(".BO") else "Global/US"
        score = float(item.get("score") or 0.0)
        asset_class = _infer_asset_class(symbol, qtype, name)
        out.append(
            SymbolSearchItem(
                symbol=symbol,
                display_name=name,
                asset_class=asset_class,
                market=market,
                exchange=exch,
                score=score,
            )
        )
        if len(out) >= limit:
            break
    return out


def _default_by_asset_class(asset_class: str | None) -> list[SymbolSearchItem]:
    ac = (asset_class or "").strip().lower()
    defaults = {
        "stocks": [
            ("TSLA", "Tesla, Inc.", "Global/US", "NASDAQ"),
            ("AAPL", "Apple Inc.", "Global/US", "NASDAQ"),
            ("RELIANCE.NS", "Reliance Industries Limited", "India", "NSE"),
            ("TCS.NS", "Tata Consultancy Services Limited", "India", "NSE"),
        ],
        "bonds": [
            ("TLT", "iShares 20+ Year Treasury Bond ETF", "Global/US", "NASDAQ"),
            ("IEF", "iShares 7-10 Year Treasury Bond ETF", "Global/US", "NASDAQ"),
            ("BND", "Vanguard Total Bond Market ETF", "Global/US", "NASDAQ"),
            ("GOVT", "iShares U.S. Treasury Bond ETF", "Global/US", "NYSEARCA"),
        ],
        "crypto": [
            ("BTC-USD", "Bitcoin USD", "Global", "CCC"),
            ("ETH-USD", "Ethereum USD", "Global", "CCC"),
            ("SOL-USD", "Solana USD", "Global", "CCC"),
            ("XRP-USD", "XRP USD", "Global", "CCC"),
        ],
        "commodities": [
            ("GC=F", "Gold Futures", "Global", "COMEX"),
            ("CL=F", "Crude Oil Futures", "Global", "NYMEX"),
            ("SI=F", "Silver Futures", "Global", "COMEX"),
            ("NG=F", "Natural Gas Futures", "Global", "NYMEX"),
        ],
    }
    rows = defaults.get(ac, [])
    return [
        SymbolSearchItem(
            symbol=s,
            display_name=n,
            asset_class=asset_class.title() if asset_class else "Stocks",
            market=m,
            exchange=e,
            score=1.0,
        )
        for (s, n, m, e) in rows
    ]


def search_symbols(query: str, limit: int = 25, asset_class: str | None = None) -> list[SymbolSearchItem]:
    limit = max(5, min(int(limit), 100))
    q = (query or "").strip()

    if not q:
        return _default_by_asset_class(asset_class)[:limit]

    nse = _search_nse(q, limit=limit)
    yf_hits = _search_yf(q, limit=limit)

    merged: list[SymbolSearchItem] = []
    seen: set[str] = set()
    for item in yf_hits + nse:
        if asset_class and item.asset_class.lower() != asset_class.lower():
            continue
        key = item.symbol.upper()
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
        if len(merged) >= limit:
            break
    return merged

