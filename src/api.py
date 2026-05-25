from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import os
import json

# Adjust sys.path robustly
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(
    title="QuantSentinel AI – Sentiment Trading Intelligence API Service",
    description="Microservice backend exposing quantitative fills risk metrics and financial NLP sentiment signals.",
    version="3.0.0"
)

# Global variables for model caching
MODEL_PATH = "data/results/profitable_trade_model.pkl"
clf_model = None
feature_cols = []

def load_prediction_model():
    global clf_model, feature_cols
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                clf_model, feature_cols = pickle.load(f)
            print("FastAPI successfully cached the best selected ML estimator.")
        except Exception as e:
            print(f"Error loading prediction model: {e}")

# Load model on startup
@app.on_event("startup")
def startup_event():
    load_prediction_model()

class TradeInputs(BaseModel):
    Coin: str
    trade_direction: str
    Size_USD: float
    leverage: float
    sentiment_score: float
    hour: int
    day_of_week: str

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "QuantSentinel AI Microservice Engine Core",
        "quant_endpoints": ["/regimes", "/risk-report", "/portfolio-sim", "/trader-segments", "/predict"],
        "nlp_endpoints": ["/news/latest", "/sentiment/overview", "/sentiment/trends", "/topics", "/entities", "/impact-report"],
        "behavioral_endpoints": ["/behavioral-insights", "/trader-personas", "/risk-score", "/market-regimes"]
    }

# -------------------------------------------------------------
# Quant Endpoints (Existing - Backward Compatible)
# -------------------------------------------------------------
@app.get("/regimes")
def get_market_regimes_legacy():
    regime_path = "data/gold/market_regimes.csv"
    if not os.path.exists(regime_path):
        regime_path = "data/results/market_regime_states.csv"
        if not os.path.exists(regime_path):
            raise HTTPException(status_code=404, detail="Regime detection data not generated yet.")
    df = pd.read_csv(regime_path)
    return df.to_dict(orient="records")

@app.get("/risk-report")
def get_risk_report():
    risk_path = "data/gold/risk_analysis.csv"
    if not os.path.exists(risk_path):
        risk_path = "data/results/risk_analysis.csv"
        if not os.path.exists(risk_path):
            raise HTTPException(status_code=404, detail="Risk metrics data not generated yet.")
    df = pd.read_csv(risk_path)
    return df.to_dict(orient="records")

@app.get("/portfolio-sim")
def get_portfolio_simulations():
    portfolio_path = "data/gold/portfolio_simulations.csv"
    if not os.path.exists(portfolio_path):
        raise HTTPException(status_code=404, detail="Portfolio simulations not found.")
    df = pd.read_csv(portfolio_path)
    return df.tail(100).to_dict(orient="records")

@app.get("/trader-segments")
def get_trader_segments():
    segments_path = "data/results/trader_segments.csv"
    if not os.path.exists(segments_path):
        raise HTTPException(status_code=404, detail="Trader segment clustering not executed yet.")
    df = pd.read_csv(segments_path)
    return df.to_dict(orient="records")

@app.post("/predict")
def predict_trade_win(inputs: TradeInputs):
    global clf_model, feature_cols
    
    if clf_model is None or not feature_cols:
        load_prediction_model()
        if clf_model is None:
            raise HTTPException(status_code=503, detail="Prediction service unavailable. Model not serialized.")
            
    try:
        input_data = pd.DataFrame([{
            'leverage': float(inputs.leverage),
            'Size USD': float(inputs.Size_USD),
            'sentiment_score': float(inputs.sentiment_score),
            'trade_direction': inputs.trade_direction,
            'hour': int(inputs.hour),
            'day_of_week': inputs.day_of_week,
            'Coin': inputs.Coin
        }])
        
        input_encoded = pd.get_dummies(input_data, columns=['trade_direction', 'day_of_week', 'Coin'])
        
        aligned_data = {}
        for col in feature_cols:
            if col in input_encoded.columns:
                aligned_data[col] = float(input_encoded[col].iloc[0])
            else:
                aligned_data[col] = 0.0
                
        X_pred = pd.DataFrame([aligned_data])[feature_cols]
        
        prob_win = float(clf_model.predict_proba(X_pred)[0, 1])
        prediction = int(clf_model.predict(X_pred)[0])
        
        return {
            "success": True,
            "win_probability": prob_win,
            "prediction": prediction,
            "action": "APPROVED" if prob_win >= 0.55 else "SIZE DOWN / REJECT"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

# -------------------------------------------------------------
# NLP & Financial Sentiment Endpoints
# -------------------------------------------------------------
@app.get("/news/latest")
def get_latest_news():
    news_dir = "data/silver/news_clean"
    if not os.path.exists(news_dir):
        raise HTTPException(status_code=404, detail="News preprocessing Silver layer empty.")
        
    files = [f for f in os.listdir(news_dir) if f.endswith('.json')]
    articles = []
    for f_name in files[:20]:
        with open(os.path.join(news_dir, f_name), "r") as f:
            articles.append(json.load(f))
    return articles

@app.get("/sentiment/overview")
def get_sentiment_overview():
    scores_path = "data/gold/sentiment/sentiment_scores.csv"
    if not os.path.exists(scores_path):
        raise HTTPException(status_code=404, detail="Sentiment scoring gold assets not found.")
        
    df = pd.read_csv(scores_path)
    total_articles = len(df)
    bullish_count = int((df['sentiment_class'] == 'Bullish').sum())
    bearish_count = int((df['sentiment_class'] == 'Bearish').sum())
    neutral_count = int((df['sentiment_class'] == 'Neutral').sum())
    avg_score = float(df['sentiment_score'].mean())
    
    return {
        "total_articles": total_articles,
        "bullish_ratio": bullish_count / total_articles if total_articles > 0 else 0,
        "bearish_ratio": bearish_count / total_articles if total_articles > 0 else 0,
        "neutral_ratio": neutral_count / total_articles if total_articles > 0 else 0,
        "average_sentiment_score": avg_score,
        "market_sentiment_bias": "Bullish" if avg_score > 0.15 else ("Bearish" if avg_score < -0.15 else "Neutral")
    }

@app.get("/sentiment/trends")
def get_sentiment_trends():
    trends_path = "data/gold/sentiment/sentiment_trends.csv"
    if not os.path.exists(trends_path):
        raise HTTPException(status_code=404, detail="Sentiment trends analytics data not found.")
    df = pd.read_csv(trends_path)
    return df.to_dict(orient="records")

@app.get("/topics")
def get_topic_clusters():
    topics_path = "data/gold/sentiment/topic_clusters.csv"
    if not os.path.exists(topics_path):
        raise HTTPException(status_code=404, detail="Unsupervised topic modeling Gold data not found.")
    df = pd.read_csv(topics_path)
    return df.to_dict(orient="records")

@app.get("/entities")
def get_extracted_entities():
    entities_path = "data/gold/sentiment/extracted_entities.csv"
    if not os.path.exists(entities_path):
        raise HTTPException(status_code=404, detail="Extracted entities ledger not found.")
    df = pd.read_csv(entities_path)
    return df.to_dict(orient="records")

@app.get("/impact-report")
def get_impact_report():
    impact_path = "data/gold/sentiment/market_impact_scores.csv"
    if not os.path.exists(impact_path):
        raise HTTPException(status_code=404, detail="Market impact scoring Gold ledger not found.")
    df = pd.read_csv(impact_path)
    return df.to_dict(orient="records")

# -------------------------------------------------------------
# Behavioral Finance Endpoints (NEW - Phase 4)
# -------------------------------------------------------------
@app.get("/behavioral-insights")
def get_behavioral_insights():
    personas_path = "data/results/trader_behavioral_personas.csv"
    if not os.path.exists(personas_path):
        raise HTTPException(status_code=404, detail="Behavioral finance analytics not compiled yet.")
    df = pd.read_csv(personas_path)
    # Filter for standard insights
    df_insights = df[['Account', 'fomo_bias', 'panic_bias', 'overconfidence_bias', 'loss_chasing_bias', 'behavioral_risk_score', 'trader_archetype']]
    return df_insights.to_dict(orient="records")

@app.get("/trader-personas")
def get_trader_personas():
    personas_path = "data/results/trader_behavioral_personas.csv"
    if not os.path.exists(personas_path):
        raise HTTPException(status_code=404, detail="Behavioral personas data not compiled yet.")
    df = pd.read_csv(personas_path)
    
    # Calculate group aggregates
    persona_counts = df['trader_archetype'].value_counts().to_dict()
    persona_returns = df.groupby('trader_archetype')['total_realized_pnl'].mean().to_dict()
    
    return {
        "archetype_distribution": persona_counts,
        "average_realized_pnl_per_archetype": persona_returns
    }

@app.get("/risk-score")
def get_behavioral_risk_scores():
    personas_path = "data/results/trader_behavioral_personas.csv"
    if not os.path.exists(personas_path):
        raise HTTPException(status_code=404, detail="Trader risk score ledger missing.")
    df = pd.read_csv(personas_path)
    df_ranked = df[['Account', 'behavioral_risk_score', 'trader_archetype', 'win_rate']].sort_values('behavioral_risk_score', ascending=False)
    return df_ranked.to_dict(orient="records")

@app.get("/market-regimes")
def get_market_regimes_endpoint():
    regime_path = "data/results/market_regime_states.csv"
    if not os.path.exists(regime_path):
        raise HTTPException(status_code=404, detail="Regime states analytics missing.")
    df = pd.read_csv(regime_path)
    return df.to_dict(orient="records")
