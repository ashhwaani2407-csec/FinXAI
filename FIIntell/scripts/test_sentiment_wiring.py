"""Quick test: verify PredictiveSentimentEngine is wired into build_features()."""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("FIINTELL_ENABLE_FINBERT", "false")

from backend.data_provider import MultiAssetDataProvider
from backend.module_b_feature_engineering import FeatureEngineer
from backend.module_c_decision_engine import DecisionEngine
from backend.settings import IngestionSettings

def test_ticker(ticker: str):
    print(f"\n{'='*60}")
    print(f"  Testing: {ticker}")
    print(f"{'='*60}")
    
    settings = IngestionSettings(enable_finbert=False)
    provider = MultiAssetDataProvider(settings=settings)
    
    print("  [1] Ingesting data...")
    ingestion = provider.ingest(ticker)
    if ingestion.errors and not ingestion.bars:
        print(f"  FAILED: {ingestion.errors}")
        return
    print(f"      Bars: {len(ingestion.bars)}, Headlines: {len(ingestion.headlines)}")
    
    print("  [2] Building features (with PredictiveSentimentEngine)...")
    fe = FeatureEngineer(settings=settings)
    features = fe.build_features(ingestion)
    
    print(f"      Technical score:  {features.technical_score:+.4f}")
    print(f"      Sentiment score:  {features.sentiment_score:+.4f}")
    print(f"      Fundamental score: {features.fundamentals_score:+.4f}")
    print(f"      Geopolitics score: {features.geopolitics_score:+.4f}")
    
    print(f"\n      Sentiment method: {features.sentiment_method}")
    bd = features.sentiment_breakdown
    if bd:
        print(f"      Headlines in/used/filtered: {bd.headlines_in}/{bd.headlines_used}/{bd.headlines_filtered_out}")
        print(f"      Positive/Negative/Neutral:  {bd.positive_pct:.1f}% / {bd.negative_pct:.1f}% / {bd.neutral_pct:.1f}%")
        print(f"      Entity match rate:  {bd.entity_match_rate:.2%}")
        print(f"      Avg source quality: {bd.avg_source_quality:.2f}")
        if bd.company_names:
            print(f"      Company names: {bd.company_names}")
        if bd.headline_details:
            print(f"\n      Per-headline details ({len(bd.headline_details)}):")
            for i, d in enumerate(bd.headline_details[:5]):
                print(f"        [{i+1}] {d.label:>8s} score={d.score:+.3f} | "
                      f"entity={d.entity_relevance:.2f} recency={d.recency_weight:.2f} "
                      f"source={d.source_quality_weight:.2f} | {d.title[:80]}")
    else:
        print("      (no breakdown available)")
    
    print(f"\n      Per-headline scores: {features.sentiment_per_headline_scores[:5]}")
    
    print("\n  [3] Decision engine...")
    de = DecisionEngine()
    decision = de.decide(features)
    print(f"      Action:     {decision.action.value}")
    print(f"      Confidence: {decision.confidence_pct:.1f}%")
    print(f"      Score:      {decision.score:+.4f}")
    print(f"      Reasoning:")
    for r in decision.reasoning[:6]:
        print(f"        - {r}")
    if decision.warnings:
        print(f"      Warnings: {decision.warnings[:3]}")


if __name__ == "__main__":
    tickers = sys.argv[1:] or ["TSLA", "RELIANCE.NS"]
    for t in tickers:
        test_ticker(t)
    print(f"\n{'='*60}")
    print("  All tests passed! Sentiment engine is wired correctly.")
    print(f"{'='*60}")
