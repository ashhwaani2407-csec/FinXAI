"""Predictive sentiment: entity linking, recency, source quality, weighted aggregation."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Iterable
from urllib.parse import urlparse

import yfinance as yf

from backend.schemas.ingestion import AssetClass, NewsHeadline, NewsSource
from backend.schemas.sentiment import HeadlineSentimentDetail, SentimentBreakdown
from backend.settings import IngestionSettings

logger = logging.getLogger(__name__)

_STOPWORDS = frozenset(
    {
        "inc",
        "inc.",
        "corp",
        "corporation",
        "company",
        "co",
        "co.",
        "ltd",
        "ltd.",
        "limited",
        "plc",
        "the",
        "and",
        "of",
        "sa",
        "nv",
        "ag",
        "group",
        "holdings",
    }
)

# Tier-1 / tier-2 publishers (substring match on publisher or URL host).
_TIER1_SOURCES = (
    "reuters",
    "bloomberg",
    "wsj",
    "wall street journal",
    "financial times",
    "ft.com",
    "cnbc",
    "marketwatch",
    "nseindia",
    "bseindia",
    "moneycontrol",
    "economictimes",
    "livemint",
    "mint",
    "business-standard",
    "thehindubusinessline",
    "investing.com",
)

_TIER2_SOURCES = (
    "yahoo",
    "seekingalpha",
    "benzinga",
    "fool.com",
    "zacks",
    "tipranks",
    "businessinsider",
    "forbes",
    "ndtv",
    "financialexpress",
)

_CRYPTO_ALIASES: dict[str, list[str]] = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "eth", "ether"],
    "SOL": ["solana", "sol"],
    "XRP": ["ripple", "xrp"],
    "BNB": ["binance coin", "bnb"],
}

_FINBERT_KEYWORDS = {
    "bull": (
        "surge",
        "soars",
        "rally",
        "beat",
        "beats",
        "profit",
        "profits",
        "growth",
        "upgrade",
        "record",
        "strong",
        "optimistic",
        "win",
        "bullish",
    ),
    "bear": (
        "falls",
        "plunge",
        "drop",
        "miss",
        "misses",
        "loss",
        "losses",
        "downgrade",
        "lawsuit",
        "bankrupt",
        "weak",
        "pessimistic",
        "decline",
        "bearish",
    ),
}


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    if math.isnan(x) or math.isinf(x):
        return 0.0
    return max(lo, min(hi, x))


@dataclass
class EntityProfile:
    ticker: str
    asset_class: AssetClass
    symbol_tokens: list[str] = field(default_factory=list)
    name_tokens: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    display_names: list[str] = field(default_factory=list)


def _tokenize_name(text: str) -> list[str]:
    raw = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [t for t in raw if len(t) >= 3 and t not in _STOPWORDS]


def _base_symbol(ticker: str) -> str:
    t = (ticker or "").upper().strip()
    if "-USD" in t or "-USDT" in t:
        return t.split("-")[0]
    if "." in t:
        return t.split(".")[0]
    return t


@lru_cache(maxsize=256)
def build_entity_profile(ticker: str, asset_class_value: str) -> EntityProfile:
    asset_class = AssetClass(asset_class_value)
    base = _base_symbol(ticker)
    profile = EntityProfile(
        ticker=ticker,
        asset_class=asset_class,
        symbol_tokens=[base.lower()] if base else [],
        aliases=[base.lower()] if base else [],
    )

    if asset_class == AssetClass.CRYPTO:
        for alias in _CRYPTO_ALIASES.get(base, [base.lower()]):
            profile.aliases.append(alias.lower())
        profile.display_names = [f"{base} cryptocurrency"]
        return profile

    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as e:
        logger.debug("entity profile yfinance info failed for %s: %s", ticker, e)
        info = {}

    long_name = str(info.get("longName") or info.get("shortName") or "").strip()
    short_name = str(info.get("shortName") or "").strip()
    isin = str(info.get("isin") or "").strip()

    names: list[str] = []
    if long_name:
        names.append(long_name)
    if short_name and short_name not in names:
        names.append(short_name)
    profile.display_names = names

    tokens: set[str] = set()
    for nm in names:
        tokens.update(_tokenize_name(nm))
    if base:
        tokens.add(base.lower())

    profile.name_tokens = sorted(tokens)
    profile.aliases = sorted(set(profile.aliases + profile.name_tokens))

    if isin:
        profile.aliases.append(isin.lower())

    return profile


def _headline_text(h: NewsHeadline) -> str:
    parts = [h.title or ""]
    if h.url:
        parts.append(h.url)
    if h.publisher:
        parts.append(h.publisher)
    return " ".join(parts).lower()


def entity_relevance(headline: NewsHeadline, profile: EntityProfile) -> float:
    text = _headline_text(headline)
    if not text.strip():
        return 0.0

    score = 0.0
    base = _base_symbol(profile.ticker).lower()

    # Strong match: exact symbol token as word (RELIANCE, TSLA, BTC).
    if base and re.search(rf"\b{re.escape(base)}\b", text):
        score += 0.55

    # Crypto pair in headline (BTC-USD).
    if profile.asset_class == AssetClass.CRYPTO and base:
        if f"{base}-usd" in text.replace(" ", "") or f"{base} usd" in text:
            score += 0.35

    alias_hits = 0
    for alias in profile.aliases:
        if len(alias) < 3:
            continue
        if alias in text or re.search(rf"\b{re.escape(alias)}\b", text):
            alias_hits += 1
    if alias_hits:
        score += min(0.45, 0.12 * alias_hits)

    # Company name tokens (e.g. "industries" + "reliance").
    name_hits = sum(1 for tok in profile.name_tokens if tok in text)
    if name_hits >= 2:
        score += 0.25
    elif name_hits == 1 and base in text:
        score += 0.15

    return _clamp(score, 0.0, 1.0)


def recency_weight(
    published_at: datetime | None,
    *,
    now: datetime,
    half_life_hours: float,
    max_age_hours: float,
) -> float:
    if published_at is None:
        return 0.45
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)
    age_h = max(0.0, (now - published_at.astimezone(timezone.utc)).total_seconds() / 3600.0)
    if age_h > max_age_hours:
        return 0.0
    if half_life_hours <= 0:
        return 1.0
    return math.exp(-math.log(2) * age_h / half_life_hours)


def source_quality_weight(headline: NewsHeadline) -> float:
    blob = " ".join(
        [
            (headline.publisher or "").lower(),
            (headline.url or "").lower(),
            headline.source.value if headline.source else "",
        ]
    )
    host = ""
    if headline.url:
        try:
            host = urlparse(headline.url).netloc.lower()
        except Exception:
            host = ""

    for needle in _TIER1_SOURCES:
        if needle in blob or needle in host:
            return 1.35
    for needle in _TIER2_SOURCES:
        if needle in blob or needle in host:
            return 1.15

    if headline.source == NewsSource.YFINANCE:
        return 1.05
    if headline.source == NewsSource.GDELT:
        return 0.92
    return 1.0


def _heuristic_probs(title: str) -> tuple[str, float, float, float, float]:
    text = (title or "").lower()
    bull = any(k in text for k in _FINBERT_KEYWORDS["bull"])
    bear = any(k in text for k in _FINBERT_KEYWORDS["bear"])
    if bull and not bear:
        return "positive", 0.75, 0.15, 0.10, 0.75
    if bear and not bull:
        return "negative", 0.15, 0.75, 0.10, -0.75
    if bull and bear:
        return "neutral", 0.25, 0.25, 0.50, 0.0
    return "neutral", 0.20, 0.20, 0.60, 0.0


@lru_cache(maxsize=1)
def _load_finbert_pipeline(model_name: str, device: int) -> Any:
    from transformers import pipeline

    return pipeline(
        task="sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
        return_all_scores=True,
        device=device,
    )


def _finbert_probs(pipeline_obj: Any, text: str) -> tuple[str, float, float, float, float]:
    out = pipeline_obj([text])[0]
    if not isinstance(out, list):
        return "neutral", 0.33, 0.33, 0.34, 0.0

    probs = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    for lab in out:
        label = str(lab.get("label") or "").lower()
        if label in probs:
            probs[label] = float(lab.get("score") or 0.0)

    p_pos, p_neg, p_neu = probs["positive"], probs["negative"], probs["neutral"]
    expected = _clamp(p_pos - p_neg)
    label = max(probs, key=probs.get)
    return label, p_pos, p_neg, p_neu, expected


def _aggregate_weighted(details: list[HeadlineSentimentDetail]) -> SentimentBreakdown:
    if not details:
        return SentimentBreakdown(
            method="none",
            score=0.0,
            positive_pct=0.0,
            negative_pct=0.0,
            neutral_pct=0.0,
            headlines_in=0,
            headlines_used=0,
            headlines_filtered_out=0,
            entity_match_rate=0.0,
        )

    total_w = sum(d.combined_weight for d in details)
    if total_w <= 0:
        return SentimentBreakdown(
            method="weighted",
            score=0.0,
            positive_pct=0.0,
            negative_pct=0.0,
            neutral_pct=100.0,
            headlines_used=len(details),
            entity_match_rate=0.0,
        )

    pos_w = neg_w = neu_w = 0.0
    score_sum = 0.0
    entity_sum = 0.0
    source_sum = 0.0

    for d in details:
        w = d.combined_weight
        score_sum += w * d.score
        entity_sum += w * d.entity_relevance
        source_sum += w * d.source_quality_weight
        if d.label == "positive":
            pos_w += w
        elif d.label == "negative":
            neg_w += w
        else:
            neu_w += w

    score = _clamp(score_sum / total_w)
    pos_pct = 100.0 * pos_w / total_w
    neg_pct = 100.0 * neg_w / total_w
    neu_pct = 100.0 * neu_w / total_w

    return SentimentBreakdown(
        method="weighted",
        score=float(score),
        positive_pct=float(pos_pct),
        negative_pct=float(neg_pct),
        neutral_pct=float(neu_pct),
        headlines_used=len(details),
        entity_match_rate=float(entity_sum / total_w),
        avg_source_quality=float(source_sum / total_w),
        headline_details=details,
    )


class PredictiveSentimentEngine:
    def __init__(self, settings: IngestionSettings | None = None) -> None:
        self._s = settings or IngestionSettings()

    def analyze(
        self,
        *,
        ticker: str,
        asset_class: AssetClass,
        headlines: Iterable[NewsHeadline],
    ) -> tuple[float, SentimentBreakdown, list[str]]:
        warnings: list[str] = []
        items = list(headlines or [])
        cap = self._s.finbert_max_headlines
        now = datetime.now(timezone.utc)

        profile = build_entity_profile(ticker, asset_class.value)

        if not items:
            empty = SentimentBreakdown(
                method="none",
                score=0.0,
                positive_pct=0.0,
                negative_pct=0.0,
                neutral_pct=0.0,
                headlines_in=0,
                headlines_used=0,
                entity_match_rate=0.0,
                company_names=profile.display_names[:3],
            )
            return 0.0, empty, warnings

        scored_rows: list[tuple[float, NewsHeadline, float, float, float]] = []
        for h in items[: cap * 3]:
            ent = entity_relevance(h, profile)
            rec = recency_weight(
                h.published_at_utc,
                now=now,
                half_life_hours=self._s.sentiment_recency_half_life_hours,
                max_age_hours=self._s.sentiment_max_headline_age_hours,
            )
            src = source_quality_weight(h)
            if rec <= 0.0:
                continue
            scored_rows.append((ent, h, rec, src, ent * rec * src))

        if not scored_rows:
            warnings.append("all headlines were older than max age window; sentiment neutral.")
            empty = SentimentBreakdown(
                method="none",
                score=0.0,
                positive_pct=0.0,
                negative_pct=0.0,
                neutral_pct=0.0,
                headlines_in=len(items),
                headlines_filtered_out=len(items),
                entity_match_rate=0.0,
                company_names=profile.display_names[:3],
            )
            return 0.0, empty, warnings

        scored_rows.sort(key=lambda x: x[4], reverse=True)

        if self._s.sentiment_enable_entity_filter:
            min_rel = self._s.sentiment_entity_min_relevance
            linked = [row for row in scored_rows if row[0] >= min_rel]
            if linked:
                selected = linked[:cap]
                filtered_out = len(scored_rows) - len(selected)
            else:
                warnings.append(
                    f"no headlines passed entity filter (min={min_rel:.2f}); using top {cap} by combined weight."
                )
                selected = scored_rows[:cap]
                filtered_out = max(0, len(scored_rows) - len(selected))
        else:
            selected = scored_rows[:cap]
            filtered_out = max(0, len(scored_rows) - len(selected))

        pipe = None
        method = "heuristic"
        if self._s.enable_finbert:
            try:
                pipe = _load_finbert_pipeline(self._s.finbert_model, self._s.finbert_device)
                method = "finbert_weighted"
            except Exception as e:
                logger.warning("FinBERT load failed: %s", e)
                if self._s.sentiment_keyword_fallback:
                    warnings.append(f"FinBERT unavailable; using heuristic sentiment: {e!s}")
                else:
                    warnings.append(f"FinBERT unavailable: {e!s}")

        details: list[HeadlineSentimentDetail] = []
        for ent, h, rec, src, _combo in selected:
            age_h = None
            if h.published_at_utc:
                pub = h.published_at_utc
                if pub.tzinfo is None:
                    pub = pub.replace(tzinfo=timezone.utc)
                age_h = (now - pub.astimezone(timezone.utc)).total_seconds() / 3600.0

            if pipe is not None:
                try:
                    label, p_pos, p_neg, p_neu, expected = _finbert_probs(pipe, h.title)
                except Exception:
                    label, p_pos, p_neg, p_neu, expected = _heuristic_probs(h.title)
                    method = "finbert_error_heuristic"
            else:
                label, p_pos, p_neg, p_neu, expected = _heuristic_probs(h.title)

            combined = ent * rec * src
            details.append(
                HeadlineSentimentDetail(
                    title=h.title[:500],
                    publisher=h.publisher,
                    source=h.source.value if h.source else None,
                    label=label,
                    score=float(expected),
                    positive_prob=float(p_pos),
                    negative_prob=float(p_neg),
                    neutral_prob=float(p_neu),
                    entity_relevance=float(ent),
                    recency_weight=float(rec),
                    source_quality_weight=float(src),
                    combined_weight=float(combined),
                    age_hours=float(age_h) if age_h is not None else None,
                )
            )

        breakdown = _aggregate_weighted(details)
        breakdown.method = method
        breakdown.headlines_in = len(items)
        breakdown.headlines_filtered_out = filtered_out
        breakdown.company_names = profile.display_names[:3]

        return float(breakdown.score), breakdown, warnings
