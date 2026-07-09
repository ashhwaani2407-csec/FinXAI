"""Live test: hit the running backend API and display sentiment results."""
import json
import sys
import httpx

def test_via_api(ticker: str):
    print(f"\n{'='*65}")
    print(f"  LIVE API TEST: {ticker}")
    print(f"{'='*65}")
    
    with httpx.Client(timeout=httpx.Timeout(90.0)) as client:
        r = client.post(
            "http://127.0.0.1:8000/recommend",
            json={"ticker": ticker, "enable_finbert": False},
        )
        r.raise_for_status()
        data = r.json()
    
    f = data["features"]
    dec = data["decision"]
    
    print(f"  Technical:     {f['technical_score']:+.4f}")
    print(f"  Sentiment:     {f['sentiment_score']:+.4f}")
    print(f"  Fundamentals:  {f['fundamentals_score']:+.4f}")
    print(f"  Geopolitics:   {f['geopolitics_score']:+.4f}")
    
    bd = f.get("sentiment_breakdown")
    if bd:
        print(f"\n  Sentiment Method:  {bd['method']}")
        print(f"  Headlines in/used: {bd['headlines_in']}/{bd['headlines_used']}")
        print(f"  Pos/Neg/Neu:       {bd['positive_pct']:.1f}% / {bd['negative_pct']:.1f}% / {bd['neutral_pct']:.1f}%")
        print(f"  Entity Match:      {bd['entity_match_rate']:.2%}")
        print(f"  Avg Source Qual:   {bd['avg_source_quality']:.2f}")
        if bd.get("company_names"):
            print(f"  Company Names:     {bd['company_names']}")
        details = bd.get("headline_details", [])
        if details:
            print(f"\n  Per-headline ({len(details)}):")
            for i, d in enumerate(details[:6]):
                age = f" age={d['age_hours']:.0f}h" if d.get("age_hours") else ""
                print(f"    [{i+1}] {d['label']:>8s} score={d['score']:+.3f} "
                      f"ent={d['entity_relevance']:.2f} rec={d['recency_weight']:.2f} "
                      f"src={d['source_quality_weight']:.2f}{age}")
                print(f"         {d['title'][:90]}")
    else:
        print("\n  WARNING: No sentiment breakdown returned!")
    
    print(f"\n  DECISION: {dec['action']} | Confidence: {dec['confidence_pct']:.1f}% | Score: {dec['score']:+.4f}")
    print(f"  Reasoning:")
    for r in dec.get("reasoning", [])[:8]:
        safe = r.encode("ascii", "replace").decode()
        print(f"    - {safe}")

if __name__ == "__main__":
    tickers = sys.argv[1:] or ["TSLA"]
    for t in tickers:
        try:
            test_via_api(t)
        except Exception as e:
            print(f"\n  ERROR for {t}: {e}")
    print(f"\n{'='*65}")
    print("  Done!")
    print(f"{'='*65}")
