import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import httpx
import yfinance as yf

from backend.audit_logger import get_recent_audit as _audit_recent_local, get_audit_stats as _audit_stats_local
from backend.backtester import run_backtest
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
            with httpx.Client(timeout=httpx.Timeout(180.0)) as client:
                r = client.post(
                    f"{backend_url}/backtest",
                    json={"ticker": ticker, "lookback_days": lookback_days, "enable_finbert": use_finbert},
                )
                r.raise_for_status()
                return r.json()
        except Exception:
            pass

    # Local fallback — run backtest directly without the backend API.
    try:
        return run_backtest(ticker, lookback_days=lookback_days, enable_finbert=use_finbert)
    except Exception as e:
        return {"error": f"Backtest failed: {e!s}"}


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
    # Local fallback.
    try:
        return _audit_recent_local(limit=limit)
    except Exception:
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
                    "Geopolitics (VIX/Oil/Macro)": round(features.geopolitics_score, 3),
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
        run_bt = st.sidebar.button("Run Backtest", type="primary")

        if not run_bt:
            st.info("Select a ticker and lookback period, then click **Run Backtest** to simulate paper trading.")
            return

        with st.spinner(f"Running {lookback_days}-day backtest on {ticker}... (this may take a minute)"):
            result = run_backtest_pipeline(ticker, lookback_days, use_finbert)
        if result.get("error"):
            st.error(result["error"])
            return

        # --- Summary Metrics ---
        st.subheader(f"📊 Backtest Results — {result.get('ticker', ticker)}")
        final_eq = result.get("final_equity", 10000.0)
        total_ret = result.get("total_return_pct", 0.0)
        signal_count = result.get("signal_count", 0)
        ret_color = "#22c55e" if total_ret >= 0 else "#ef4444"

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Starting Capital", "$10,000")
        c2.metric("Final Equity", f"${final_eq:,.0f}")
        c3.metric("Total Return", f"{total_ret:+.2f}%")
        c4.metric("Total Signals", signal_count)

        # --- Equity Curve ---
        pnl_curve = result.get("pnl_curve") or []
        signals = result.get("signals") or []

        if pnl_curve:
            st.subheader("💰 Equity Curve")
            pnl_df = pd.DataFrame({"step": list(range(len(pnl_curve))), "equity": pnl_curve})
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                x=pnl_df["step"], y=pnl_df["equity"],
                mode="lines", name="Portfolio Equity",
                line=dict(color=ret_color, width=2.5),
                fill="tozeroy", fillcolor="rgba(34,197,94,0.08)" if total_ret >= 0 else "rgba(239,68,68,0.08)",
            ))
            fig_eq.add_hline(y=10000, line_dash="dash", line_color="rgba(255,255,255,0.3)", annotation_text="Start $10k")
            fig_eq.update_layout(
                height=320, margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E3E3E3"),
                xaxis=dict(title="Trading Day", gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(title="Equity ($)", gridcolor="rgba(255,255,255,0.06)"),
            )
            st.plotly_chart(fig_eq, use_container_width=True)

        # --- Price Chart with Signal Markers ---
        if signals:
            st.subheader("📈 Price Chart with Signals")
            sig_df = pd.DataFrame(signals)
            sig_df["date"] = pd.to_datetime(sig_df["date"])

            # Fetch price data to overlay on candlestick
            try:
                ingestion_settings = IngestionSettings(enable_finbert=False)
                provider = MultiAssetDataProvider(settings=ingestion_settings)
                ing = provider.ingest(ticker)
                price_df = _bars_to_plotly_df(ing.bars)

                if not price_df.empty:
                    # Filter to backtest period
                    min_date = sig_df["date"].min()
                    price_df = price_df.loc[price_df.index >= min_date]

                    fig_price = go.Figure()
                    fig_price.add_trace(go.Candlestick(
                        x=price_df.index,
                        open=price_df["open"].astype(float),
                        high=price_df["high"].astype(float),
                        low=price_df["low"].astype(float),
                        close=price_df["close"].astype(float),
                        name="OHLC",
                    ))

                    # BUY markers
                    buys = sig_df[sig_df["action"] == "BUY"]
                    if not buys.empty:
                        # Get close prices for buy dates from price_df
                        buy_prices = []
                        for d in buys["date"]:
                            match = price_df.index[price_df.index <= d]
                            buy_prices.append(float(price_df.loc[match[-1], "low"]) * 0.98 if len(match) > 0 else None)
                        fig_price.add_trace(go.Scatter(
                            x=buys["date"], y=buy_prices,
                            mode="markers", name="BUY",
                            marker=dict(symbol="triangle-up", size=12, color="#22c55e"),
                        ))

                    # SELL markers
                    sells = sig_df[sig_df["action"] == "SELL"]
                    if not sells.empty:
                        sell_prices = []
                        for d in sells["date"]:
                            match = price_df.index[price_df.index <= d]
                            sell_prices.append(float(price_df.loc[match[-1], "high"]) * 1.02 if len(match) > 0 else None)
                        fig_price.add_trace(go.Scatter(
                            x=sells["date"], y=sell_prices,
                            mode="markers", name="SELL",
                            marker=dict(symbol="triangle-down", size=12, color="#ef4444"),
                        ))

                    fig_price.update_layout(
                        height=420, margin=dict(l=10, r=10, t=30, b=10),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#E3E3E3"),
                        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                        yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    )
                    st.plotly_chart(fig_price, use_container_width=True)
            except Exception:
                pass  # If price fetch fails, just skip the chart

        # --- Hit Rate Table ---
        hit_rates = result.get("hit_rates", {})
        overall = hit_rates.get("overall", {})
        by_asset = hit_rates.get("by_asset_class", {})

        if overall:
            st.subheader("🎯 Hit Rate — Precision & Recall by Action")
            hr_rows = []
            for action, data in overall.items():
                hr_rows.append({
                    "Action": action,
                    "Signals": data.get("support", 0),
                    "Hit Rate": f"{data.get('hit_rate', 0) * 100:.1f}%",
                    "Precision": f"{data.get('precision', 0) * 100:.1f}%",
                    "Recall": f"{data.get('recall', 0) * 100:.1f}%",
                })
            st.dataframe(pd.DataFrame(hr_rows), use_container_width=True, hide_index=True)

        if by_asset:
            with st.expander("Hit Rate by Asset Class", expanded=False):
                for asset_cls, action_map in by_asset.items():
                    st.caption(f"**{asset_cls}**")
                    ac_rows = []
                    for action, data in action_map.items():
                        ac_rows.append({
                            "Action": action,
                            "Signals": data.get("support", 0),
                            "Hit Rate": f"{data.get('hit_rate', 0) * 100:.1f}%",
                            "Precision": f"{data.get('precision', 0) * 100:.1f}%",
                        })
                    st.dataframe(pd.DataFrame(ac_rows), use_container_width=True, hide_index=True)

        # --- Detailed Signals Table ---
        if signals:
            st.subheader("📋 Signal Details")
            sig_display = []
            for s in signals:
                correct = s.get("correct")
                if correct is True:
                    outcome = "✅ Correct"
                elif correct is False:
                    outcome = "❌ Wrong"
                else:
                    outcome = "⏳ Pending"
                sig_display.append({
                    "Date": s.get("date", "")[:10],
                    "Action": s.get("action", ""),
                    "Score": round(s.get("score", 0), 3),
                    "Confidence": f"{s.get('confidence_pct', 0):.1f}%",
                    "Return 1d": f"{s.get('return_1d_pct', 0):.2f}%" if s.get("return_1d_pct") is not None else "—",
                    "Return 5d": f"{s.get('return_5d_pct', 0):.2f}%" if s.get("return_5d_pct") is not None else "—",
                    "Return 20d": f"{s.get('return_20d_pct', 0):.2f}%" if s.get("return_20d_pct") is not None else "—",
                    "Outcome": outcome,
                })
            st.dataframe(pd.DataFrame(sig_display), use_container_width=True, hide_index=True)

        # --- Recent Audit Log ---
        with st.expander("Recent Audit Log (last 20 recommendations)", expanded=False):
            audit_rows = get_recent_audit(limit=20)
            if audit_rows:
                audit_display = []
                for row in audit_rows:
                    audit_display.append({
                        "Time": str(row.get("timestamp_utc", ""))[:19],
                        "Ticker": row.get("ticker", ""),
                        "Action": row.get("action", ""),
                        "Confidence": f"{row.get('confidence_pct', 0):.1f}%",
                        "Score": round(row.get("score", 0), 3),
                        "Return 1d": f"{row.get('return_1d_pct', 0):.2f}%" if row.get("return_1d_pct") is not None else "—",
                        "Return 5d": f"{row.get('return_5d_pct', 0):.2f}%" if row.get("return_5d_pct") is not None else "—",
                    })
                st.dataframe(pd.DataFrame(audit_display), use_container_width=True, hide_index=True)
            else:
                st.caption("No audit logs yet. Make some recommendations first.")
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

