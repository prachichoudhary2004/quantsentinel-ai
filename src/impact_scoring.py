import os
import sys
import json
import pandas as pd
import numpy as np

# Adjust path robustly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.observability import PipelineLogger

# Credibility mapping per source
SOURCE_CREDIBILITY = {
    'Binance News': 95.0,
    'CoinDesk': 90.0,
    'Yahoo Finance Crypto': 85.0,
    'CoinTelegraph': 80.0
}

# Critical keywords that trigger outsized market movements
CRITICAL_KEYWORDS = {
    'etf': 1.0,
    'sec': 1.0,
    'approved': 1.0,
    'hack': 1.0,
    'exploit': 1.0,
    'liquidation': 0.8,
    'liquidations': 0.8,
    'cftc': 0.8,
    'blackrock': 0.9,
    'fidelity': 0.8,
    'court': 0.7,
    'treasury': 0.7,
    'insolvency': 1.0,
    'outflow': 0.5
}

def calculate_impact_score(sentiment_score: float, source: str, tokens: list, hour: int = 12) -> tuple:
    """
    Computes a multi-factor Market Impact Score between 0 and 100.
    Factors:
    1. Sentiment Strength (35%): Absolute deviation of sentiment.
    2. Source Credibility (25%): Inherent authority of publishers.
    3. Keyword Relevance (20%): Proximity density of market-moving terms.
    4. Engagement Proxy (20%): Time of day liquidity multiplier.
    """
    # Factor 1: Sentiment Strength
    sentiment_strength = abs(sentiment_score) * 100.0
    
    # Factor 2: Source Credibility
    credibility = SOURCE_CREDIBILITY.get(source, 60.0)
    
    # Factor 3: Keyword Relevance
    keyword_weight = 0.0
    found_keywords = 0
    for t in tokens:
        if t in CRITICAL_KEYWORDS:
            keyword_weight += CRITICAL_KEYWORDS[t]
            found_keywords += 1
            
    relevance = float(np.clip(keyword_weight * 20.0, 0.0, 100.0))
    
    # Factor 4: Engagement Proxy (higher during key trading hours e.g. 13-18 UTC/IST)
    if 13 <= hour <= 19:
        engagement = 95.0
    elif 8 <= hour <= 12:
        engagement = 75.0
    else:
        engagement = 50.0
        
    # Weighted calculation
    score = (0.35 * sentiment_strength) + (0.25 * credibility) + (0.20 * relevance) + (0.20 * engagement)
    impact_score = float(np.clip(score, 0.0, 100.0))
    
    # Classification
    if impact_score >= 75:
        category = "Critical Impact"
    elif impact_score >= 50:
        category = "High Impact"
    elif impact_score >= 25:
        category = "Medium Impact"
    else:
        category = "Low Impact"
        
    return impact_score, category

def run_impact_scoring_pipeline():
    """
    Ingests scored Gold articles, computes dynamic market impact parameters,
    and updates Gold storage files.
    """
    logger = PipelineLogger()
    logger.start_timer('impact_scoring')
    
    gold_dir = "data/gold/sentiment"
    if not os.path.exists(gold_dir):
        print("Gold sentiment folder missing. Run Sentiment Engine first.")
        return
        
    scored_files = [f for f in os.listdir(gold_dir) if f.endswith('.json') and f != 'market_impact_scores.json']
    impact_records = []
    
    for f_name in scored_files:
        with open(os.path.join(gold_dir, f_name), "r") as f:
            art = json.load(f)
            
        sentiment_score = art.get('sentiment_score', 0.0)
        source = art.get('source', 'Unknown')
        tokens = art.get('tokens', [])
        
        # Get hour of article
        hour = 12
        try:
            pub_date = pd.to_datetime(art['published_date'])
            hour = pub_date.hour
        except Exception:
            pass
            
        impact_score, category = calculate_impact_score(sentiment_score, source, tokens, hour)
        
        # Save metrics back to JSON
        art['market_impact_score'] = impact_score
        art['market_impact_class'] = category
        
        with open(os.path.join(gold_dir, f_name), "w") as f_out:
            json.dump(art, f_out, indent=4)
            
        impact_records.append({
            'article_id': art['article_id'],
            'title': art['title'],
            'sentiment_score': sentiment_score,
            'market_impact_score': impact_score,
            'market_impact_class': category,
            'source': source,
            'published_date': art['published_date']
        })
        
    df_impact = pd.DataFrame(impact_records)
    csv_path = f"{gold_dir}/market_impact_scores.csv"
    df_impact.to_csv(csv_path, index=False)
    
    logger.stop_timer('impact_scoring', len(impact_records))
    print(f"Market impact scoring successfully completed! Ledger saved to {csv_path}")

if __name__ == "__main__":
    run_impact_scoring_pipeline()
