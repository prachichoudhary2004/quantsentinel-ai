import os
import sys
import json
import pandas as pd
import numpy as np

# Adjust path robustly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.observability import PipelineLogger

# Retrieve configurable NLP execution mode from environment variables (Default: auto)
nlp_mode = os.environ.get('NLP_MODE', 'auto').lower()
print(f"Operational Sentiment NLP Mode configured to: {nlp_mode.upper()}")

HAS_FINBERT = False
if nlp_mode in ['auto', 'transformer']:
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        HAS_FINBERT = True
        print("Successfully imported HuggingFace 'transformers' and 'torch' for FinBERT.")
    except ImportError:
        if nlp_mode == 'transformer':
            print("CRITICAL ERROR: NLP_MODE set to 'transformer' but 'transformers' or 'torch' packages are not installed on the system.")
            sys.exit(1)
        HAS_FINBERT = False
        print("FinBERT libraries unavailable. Falling back to Lexicon Sentiment Engine.")
else:
    # Explicit 'fallback' mode: bypass heavy libraries
    HAS_FINBERT = False
    print("NLP_MODE explicitly configured to 'fallback'. Bypassing FinBERT neural classification.")

# High-fidelity Crypto Financial Sentiment Lexicon
LEXICON = {
    # Extremely Bullish (+1.0)
    'approval': 1.0,
    'approved': 1.0,
    'bullish': 1.0,
    'inflows': 1.0,
    'surged': 1.0,
    'squeeze': 1.0,
    'breakout': 1.0,
    'growth': 1.0,
    'adoption': 1.0,
    'parabolic': 1.0,
    
    # Bullish Swing (+0.5)
    'buy': 0.5,
    'bull': 0.5,
    'rally': 0.5,
    'gain': 0.5,
    'profit': 0.5,
    'support': 0.5,
    'inflow': 0.5,
    'positive': 0.5,
    
    # Bearish Swing (-0.5)
    'sell': -0.5,
    'bear': -0.5,
    'decline': -0.5,
    'loss': -0.5,
    'resistance': -0.5,
    'outflow': -0.5,
    'negative': -0.5,
    'audit': -0.5,
    
    # Extremely Bearish (-1.0)
    'bearish': -1.0,
    'liquidation': -1.0,
    'liquidations': -1.0,
    'outflows': -1.0,
    'crunch': -1.0,
    'exploit': -1.0,
    'hack': -1.0,
    'hacked': -1.0,
    'panic': -1.0,
    'exploding': -1.0,
    'crackdown': -1.0,
    'lawsuit': -1.0,
    'fraud': -1.0,
    'scam': -1.0
}

def analyze_sentiment_finbert(text: str) -> tuple:
    """
    Classifies sentiment using ProsusAI/finbert model.
    """
    if not HAS_FINBERT:
        return None, None
        
    try:
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        
        inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).detach().numpy()[0]
        
        # Finbert labels: 0 -> positive, 1 -> negative, 2 -> neutral
        pos, neg, neu = probs[0], probs[1], probs[2]
        
        sentiment_score = float(pos - neg)
        confidence_score = float(max(pos, neg, neu))
        
        return sentiment_score, confidence_score
    except Exception as e:
        print(f"FinBERT inference failed: {e}.")
        if nlp_mode == 'transformer':
            print("CRITICAL: FinBERT execution failed in strict 'transformer' mode. Halting.")
            sys.exit(1)
        return None, None

def analyze_sentiment_lexicon(tokens: list) -> tuple:
    """
    Highly accurate Cryptocurrency Financial Lexicon Engine.
    Maps token deviations to bullish/bearish scales with confidence bounds.
    """
    score_sum = 0.0
    term_count = 0
    
    for t in tokens:
        if t in LEXICON:
            score_sum += LEXICON[t]
            term_count += 1
            
    if term_count == 0:
        return 0.0, 1.0
        
    raw_avg = score_sum / term_count
    sentiment_score = float(np.clip(raw_avg, -1.0, 1.0))
    confidence_score = float(np.clip(term_count * 0.15 + 0.5, 0.5, 1.0))
    
    return sentiment_score, confidence_score

def execute_sentiment_pipeline():
    """
    Ingests preprocessed Silver articles, scores their sentiment using premium
    FinBERT or lexicon fallback, and stores Gold results.
    """
    logger = PipelineLogger()
    logger.start_timer('sentiment_engine')
    
    silver_dir = "data/silver/news_clean"
    gold_dir = "data/gold/sentiment"
    os.makedirs(gold_dir, exist_ok=True)
    
    if not os.path.exists(silver_dir):
        print("Silver preprocessed directory missing. Run Text Preprocessing first.")
        return
        
    cleaned_files = [f for f in os.listdir(silver_dir) if f.endswith('.json')]
    scored_records = []
    
    for f_name in cleaned_files:
        with open(os.path.join(silver_dir, f_name), "r") as f:
            art = json.load(f)
            
        score, confidence = None, None
        if HAS_FINBERT:
            score, confidence = analyze_sentiment_finbert(art['content_clean'])
            
        if score is None:
            score, confidence = analyze_sentiment_lexicon(art['tokens'])
            
        if score > 0.15:
            classification = "Bullish"
        elif score < -0.15:
            classification = "Bearish"
        else:
            classification = "Neutral"
            
        art['sentiment_score'] = score
        art['confidence_score'] = confidence
        art['sentiment_class'] = classification
        
        with open(os.path.join(gold_dir, f_name), "w") as f_out:
            json.dump(art, f_out, indent=4)
            
        scored_records.append({
            'article_id': art['article_id'],
            'title': art['title'],
            'sentiment_score': score,
            'confidence_score': confidence,
            'sentiment_class': classification,
            'source': art['source'],
            'published_date': art['published_date']
        })
        
    df_scores = pd.DataFrame(scored_records)
    csv_path = f"{gold_dir}/sentiment_scores.csv"
    df_scores.to_csv(csv_path, index=False)
    
    logger.stop_timer('sentiment_engine', len(scored_records))
    print(f"Sentiment scoring completed! Aggregated ledger saved to: {csv_path}")

if __name__ == "__main__":
    execute_sentiment_pipeline()
