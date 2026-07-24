import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import httpx
import yfinance as yf

from backend.data_provider import MultiAssetDataProvider, ohlcv_bars_to_dataframe
from backend.module_b_feature_engineering import FeatureEngineer
from backend.module_c_decision_engine import DecisionEngine
from backend.schemas.decision import TradeAction
from backend.schemas.recommendation import RecommendBatchResponse, RecommendResponse
from backend.schemas.symbols import SymbolSearchResponse
from backend.settings import IngestionSettings
from backend.ticker_resolver import search_symbols as local_symbol_search


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _action_color(action: TradeAction) -> str:
    if action == TradeAction.BUY:
        return "#22c55e"
    if action == TradeAction.SELL:
        return "#ef4444"
    return "#94a3b8"


def _bars_to_plotly_df(bars):
    df = ohlcv_bars_to_dataframe(bars)
    if df.empty:
        return df
    # Plotly is happier with datetime-like values.
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    return df


def run_pipeline(ticker: str, use_finbert: bool):
    """No @st.cache_data: cached responses caused identical scores when switching tickers."""
    backend_url = os.getenv("FIINTELL_BACKEND_URL", "").strip().rstrip("/")
    if backend_url:
        # If backend is running, use it like a real deployable service.
        try:
            # Keep backend timeout small so the UI doesn't appear stuck
            # when FIINTELL_BACKEND_URL is misconfigured/unreachable.
            with httpx.Client(timeout=httpx.Timeout(10.0)) as client:
                r = client.post(
                    f"{backend_url}/recommend",
                    json={"ticker": ticker, "enable_finbert": use_finbert},
                )
                r.raise_for_status()
                resp = RecommendResponse.model_validate(r.json())
                return resp.ingestion, resp.features, resp.decision
        except Exception:
            # Fallback to direct pipeline when backend is unavailable.
            pass

    ingestion_settings = IngestionSettings(enable_finbert=use_finbert)
    provider = MultiAssetDataProvider(settings=ingestion_settings)
    ingestion = provider.ingest(ticker)
    fe = FeatureEngineer(settings=ingestion_settings)
    features = fe.build_features(ingestion)
    de = DecisionEngine()
    decision = de.decide(features)
    return ingestion, features, decision


def run_batch_pipeline(tickers: list[str], use_finbert: bool):
    backend_url = os.getenv("FIINTELL_BACKEND_URL", "").strip().rstrip("/")
    if backend_url:
        try:
            with httpx.Client(timeout=httpx.Timeout(30.0)) as client:
                r = client.post(
                    f"{backend_url}/recommend/batch",
                    json={"tickers": tickers, "enable_finbert": use_finbert},
                )
                r.raise_for_status()
                resp = RecommendBatchResponse.model_validate(r.json())
                return resp
        except Exception:
            pass

    # Local fallback when backend API is unavailable.
    ingestion_settings = IngestionSettings(enable_finbert=use_finbert)
    provider = MultiAssetDataProvider(settings=ingestion_settings)
    fe = FeatureEngineer(settings=ingestion_settings)
    de = DecisionEngine()
    items = []
    for t in tickers:
        ing = provider.ingest(t)
        feat = fe.build_features(ing)
        dec = de.decide(feat)
        items.append(
            {
                "ticker": t,
                "ok": True if not dec.errors else False,
                "ingestion": ing.model_dump(),
                "features": feat.model_dump(),
                "decision": dec.model_dump(),
                "errors": dec.errors,
                "warnings": dec.warnings,
            }
        )
    return {"items": items}


def run_backtest_pipeline(ticker: str, lookback_days: int, use_finbert: bool):
    backend_url = os.getenv("FIINTELL_BACKEND_URL", "").strip().rstrip("/")
    if backend_url:
        try:
            with httpx.Client(timeout=httpx.Timeout(90.0)) as client:
                r = client.post(
                    f"{backend_url}/backtest",
                    json={"ticker": ticker, "lookback_days": lookback_days, "enable_finbert": use_finbert},
                )
                r.raise_for_status()
                return r.json()
        except Exception:
            pass

    return {"error": "Backend backtest endpoint unavailable."}


def get_recent_audit(limit: int = 20):
    backend_url = os.getenv("FIINTELL_BACKEND_URL", "").strip().rstrip("/")
    if backend_url:
        try:
            with httpx.Client(timeout=httpx.Timeout(12.0)) as client:
                r = client.get(f"{backend_url}/audit/recent", params={"limit": limit})
                r.raise_for_status()
                return r.json().get("items", [])
        except Exception:
            pass
    return []


@st.cache_data(ttl=10 * 60, show_spinner=False)
def search_symbols(query: str, limit: int = 25, asset_class: str | None = None):
    backend_url = os.getenv("FIINTELL_BACKEND_URL", "").strip().rstrip("/")
    q = (query or "").strip()

    if backend_url:
        try:
            with httpx.Client(timeout=httpx.Timeout(12.0)) as client:
                # include filter so users compare within same asset class
                params = {"q": q, "limit": limit}
                if asset_class:
                    params["asset_class"] = asset_class
                r = client.get(f"{backend_url}/symbols/search", params=params)
                r.raise_for_status()
                resp = SymbolSearchResponse.model_validate(r.json())
                return resp.items
        except Exception:
            pass
    return local_symbol_search(q, limit, asset_class=asset_class)


def _symbol_label(item) -> str:
    exch = f" · {item.exchange}" if getattr(item, "exchange", None) else ""
    return f"{item.display_name} ({item.symbol}) [{item.market}{exch}]"


@st.cache_data(ttl=30 * 60, show_spinner=False)
def get_ticker_profile(ticker: str) -> dict:
    """Basic info card data for selected symbol."""
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        return {}

    def _fmt_market_cap(v):
        try:
            n = float(v)
            if n >= 1e12:
                return f"{n/1e12:.2f}T"
            if n >= 1e9:
                return f"{n/1e9:.2f}B"
            if n >= 1e6:
                return f"{n/1e6:.2f}M"
            return f"{n:,.0f}"
        except Exception:
            return None

    return {
        "longName": info.get("longName") or info.get("shortName"),
        "symbol": info.get("symbol") or ticker,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "marketCap": _fmt_market_cap(info.get("marketCap")),
        "currency": info.get("currency"),
        "trailingPE": info.get("trailingPE"),
        "forwardPE": info.get("forwardPE"),
        "priceToBook": info.get("priceToBook"),
        "beta": info.get("beta"),
        "dividendYield": info.get("dividendYield"),
        "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
        "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
        "regularMarketPrice": info.get("regularMarketPrice"),
    }


def main():
    st.set_page_config(page_title="FIIntell", layout="wide")

    # Sidebar
    st.sidebar.header("Asset Selector")
    mode = st.sidebar.radio("Analysis Mode", ["Single", "Batch", "Backtest"], horizontal=True)
    asset_class_filter = st.sidebar.selectbox("Asset Class", ["Stocks", "Bonds", "Crypto", "Commodities"])
    query = st.sidebar.text_input("Search by company/common name", value="")
    found = search_symbols(query, limit=25, asset_class=asset_class_filter)
    if found:
        options_map = {_symbol_label(i): i.symbol for i in found}
        labels = list(options_map.keys())
    else:
        options_map = {}
        labels = []

    use_finbert_default = _env_bool("FIINTELL_ENABLE_FINBERT", True)
    use_finbert = st.sidebar.checkbox("Use FinBERT sentiment (slower)", value=use_finbert_default)
    refresh = st.sidebar.button("Refresh", type="primary")
    if refresh:
        search_symbols.clear()
        get_ticker_profile.clear()

    # UI: dark-mode professional styling
    st.markdown(
        """
        <style>
        .fi-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            padding: 16px;
            border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if mode == "Single":
        selected_label = st.sidebar.selectbox(
            "Select Ticker",
            labels if labels else ["No matches. Try another company name."],
            key=f"fiintell_pick_{asset_class_filter}",
        )
        if not labels:
            st.warning("No symbols found. Try examples like Tesla, Reliance, Infosys, Tata, Apple.")
            return
        ticker = options_map[selected_label]

        col_left, col_right = st.columns([2.1, 1.2])

        with col_left:
            st.subheader("Price (Daily Candlestick)")
            ingestion, features, decision = run_pipeline(ticker, use_finbert)

            if ingestion.errors:
                st.error(" ".join(ingestion.errors[:3]))
                return

            df = _bars_to_plotly_df(ingestion.bars)
            if df.empty:
                st.warning("No market history returned for this ticker.")
            else:
                fig = go.Figure(
                    data=[
                        go.Candlestick(
                            x=df.index,
                            open=df["open"].astype(float),
                            high=df["high"].astype(float),
                            low=df["low"].astype(float),
                            close=df["close"].astype(float),
                            name="OHLC",
                        )
                    ]
                )
                fig.update_layout(
                    height=420,
                    margin=dict(l=10, r=10, t=30, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#E3E3E3"),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                )
                st.plotly_chart(fig, use_container_width=True)
                profile = get_ticker_profile(ticker)
                if profile:
                    st.subheader("Ticker Snapshot")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Symbol", profile.get("symbol") or "-")
                    c2.metric("Market Cap", profile.get("marketCap") or "N/A")
                    pe_disp = profile.get("trailingPE") or profile.get("forwardPE")
                    c3.metric("P/E", f"{pe_disp:.2f}" if isinstance(pe_disp, (float, int)) else "N/A")
                    pb = profile.get("priceToBook")
                    c4.metric("P/B", f"{pb:.2f}" if isinstance(pb, (float, int)) else "N/A")

                    st.write(
                        {
                            "Name": profile.get("longName"),
                            "Sector": profile.get("sector"),
                            "Industry": profile.get("industry"),
                            "Currency": profile.get("currency"),
                            "Beta": profile.get("beta"),
                            "Dividend Yield": profile.get("dividendYield"),
                            "52W High": profile.get("fiftyTwoWeekHigh"),
                            "52W Low": profile.get("fiftyTwoWeekLow"),
                            "Market Price": profile.get("regularMarketPrice"),
                        }
                    )

        with col_right:
            st.subheader("AI Recommendation Card")
            color = _action_color(decision.action)
            st.caption(f"Resolved symbol: **{ingestion.ticker_resolved_yfinance}**")
            st.markdown(
                f"""
                <div class="fi-card">
                  <div style="font-size:38px; font-weight:800; color:{color}; line-height:1.1;">
                    {decision.action.value}
                  </div>
                  <div style="margin-top:6px; font-size:14px; color:#cbd5e1;">
                    {decision.label.value} · Confidence: {decision.confidence_pct:.1f}%
                  </div>
                  <div style="margin-top:10px; font-size:12px; color:#94a3b8;">
                    Composite score: {decision.score:.3f}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if ingestion.warnings:
                st.warning(" ".join(ingestion.warnings[:2]))

            st.subheader("Reasoning Summary")
            if decision.reasoning:
                bullets = decision.reasoning[:10]
                st.markdown("\n".join([f"- {b}" for b in bullets]))
            else:
                st.caption("No reasoning available.")

            st.subheader("Feature Group Scores (Module B)")
            st.write(
                {
                    "Technical": round(features.technical_score, 3),
                    "Sentiment": round(features.sentiment_score, 3),
                    "Fundamentals": round(features.fundamentals_score, 3),
                    "Geopolitics (mock GPR)": round(features.geopolitics_score, 3),
                }
            )

            st.subheader("News Headlines & Sentiment (Module B)")
            sentiment_method = features.sentiment_method or "unknown"
            st.caption(f"Sentiment method: {sentiment_method}")
            with st.expander("Show headlines", expanded=False):
                per_headline = features.sentiment_per_headline_scores or []
                cap = min(len(ingestion.headlines), 10)
                if cap == 0:
                    st.caption("No headlines available for this ticker.")
                else:
                    for idx in range(cap):
                        h = ingestion.headlines[idx]
                        ts = ""
                        if getattr(h, "published_at_utc", None):
                            ts_dt = h.published_at_utc
                            ts = f" · {ts_dt.strftime('%Y-%m-%d %H:%M UTC')}"
                        s = ""
                        if idx < len(per_headline):
                            s = f" · sentiment={per_headline[idx]:+.2f}"
                        st.markdown(f"- **{h.source.value}**: {h.title[:180]}{ts}{s}")
        return

    if mode == "Backtest":
        selected_label = st.sidebar.selectbox(
            "Select Backtest Ticker",
            labels if labels else ["No matches. Try another company name."],
            key=f"fiintell_backtest_pick_{asset_class_filter}",
        )
        if not labels:
            st.warning("No symbols found. Try examples like Tesla, Reliance, Infosys, Tata, Apple.")
            return
        ticker = options_map[selected_label]
        lookback_days = st.sidebar.slider("Lookback (days)", min_value=30, max_value=180, value=90, step=30)

        with st.spinner("Running backtest..."):
            result = run_backtest_pipeline(ticker, lookback_days, use_finbert)
        if result.get("error"):
            st.error(result["error"])
            return

        pnl_curve = result.get("pnl_curve") or []
        if pnl_curve:
            st.subheader("Equity Curve")
            pnl_df = pd.DataFrame({"step": list(range(len(pnl_curve))), "equity": pnl_curve})
            fig = go.Figure(data=[go.Scatter(x=pnl_df["step"], y=pnl_df["equity"], mode="lines")])
            fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Backtest Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Final Equity", f"${result.get('final_equity', 0.0):,.0f}")
        c2.metric("Total Return", f"{result.get('total_return_pct', 0.0):.2f}%")
        c3.metric("Signals", result.get("signal_count", 0))

        hit_rates = result.get("hit_rates", {})
        overall = hit_rates.get("overall", {})
        if overall:
            st.subheader("Hit Rate Table")
            st.dataframe(pd.DataFrame.from_dict(overall, orient="index"), use_container_width=True)

        audit_rows = get_recent_audit(limit=20)
        if audit_rows:
            st.subheader("Recent Audit Log")
            audit_df = pd.DataFrame(audit_rows)
            st.dataframe(audit_df, use_container_width=True)
        return

    # Batch mode
    picked = st.sidebar.multiselect("Select multiple tickers", labels, default=labels[: min(3, len(labels))])
    if not picked:
        st.info("Pick one or more tickers in sidebar to run batch analysis.")
        return
    batch_tickers = [options_map[x] for x in picked]
    batch = run_batch_pipeline(batch_tickers, use_finbert)
    items = batch.items if hasattr(batch, "items") else batch.get("items", [])
    if not items:
        st.warning("No batch results returned.")
        return

    rows = []
    for item in items:
        it = item if isinstance(item, dict) else item.model_dump()
        decision = it.get("decision") or {}
        rows.append(
            {
                "Ticker": it.get("ticker"),
                "OK": it.get("ok"),
                "Action": decision.get("action"),
                "Label": decision.get("label"),
                "Confidence %": round(float(decision.get("confidence_pct") or 0.0), 2),
                "Score": round(float(decision.get("score") or 0.0), 4),
            }
        )
    st.subheader("Batch Comparison")
    out_df = pd.DataFrame(rows)
    st.dataframe(out_df, use_container_width=True)
    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Batch Results CSV",
        data=csv_bytes,
        file_name="fiintell_batch_results.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()

