import os
import sys
import json
import pandas as pd
import numpy as np

# Adjust path robustly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.observability import PipelineLogger

def compute_sentiment_trends():
    """
    Computes daily momentum, volatility, acceleration, and divergence
    ratios across historical sentiment scores and correlates them with
    trader returns (price surrogate).
    """
    logger = PipelineLogger()
    logger.start_timer('sentiment_trends')
    
    gold_dir = "data/gold/sentiment"
    scores_path = f"{gold_dir}/sentiment_scores.csv"
    
    if not os.path.exists(scores_path):
        print("Gold sentiment scores missing. Skipping trends.")
        return
        
    df_scores = pd.read_csv(scores_path)
    df_scores['date'] = pd.to_datetime(df_scores['published_date']).dt.strftime('%Y-%m-%d')
    
    # Calculate daily averages
    df_daily = df_scores.groupby('date').agg(
        avg_sentiment=('sentiment_score', 'mean'),
        article_count=('article_id', 'count')
    ).reset_index().sort_values('date').reset_index(drop=True)
    
    # 1. Sentiment Momentum (7-day simple moving average)
    df_daily['sentiment_momentum'] = df_daily['avg_sentiment'].rolling(window=7, min_periods=1).mean()
    
    # 2. Sentiment Volatility (7-day rolling standard deviation)
    df_daily['sentiment_volatility'] = df_daily['avg_sentiment'].rolling(window=7, min_periods=1).std().fillna(0.0)
    
    # 3. Sentiment Acceleration (Rate of change of momentum: momentum_today - momentum_yesterday)
    df_daily['sentiment_acceleration'] = df_daily['sentiment_momentum'].diff().fillna(0.0)
    
    # 4. Sentiment Price Divergence
    # We load trader performance (ROE) from Silver processed files to act as price returns proxy
    df_daily['divergence'] = 0.0
    silver_path = "data/silver/merged_trader_data.csv"
    if os.path.exists(silver_path):
        try:
            df_silver = pd.read_csv(silver_path)
            df_returns = df_silver.groupby('date')['roe'].mean().reset_index()
            df_returns['date'] = pd.to_datetime(df_returns['date']).dt.strftime('%Y-%m-%d')
            
            # Merge returns
            df_daily = pd.merge(df_daily, df_returns, on='date', how='left')
            df_daily['roe'] = df_daily['roe'].ffill().bfill().fillna(0.0)
            
            # Compute 7-day ROE momentum
            df_daily['price_momentum'] = df_daily['roe'].rolling(window=7, min_periods=1).mean()
            
            # Divergence = Price Momentum - Sentiment Momentum
            df_daily['divergence'] = df_daily['price_momentum'] - df_daily['sentiment_momentum']
        except Exception as e:
            print(f"Divergence calculation warning: {e}")
            
    trends_path = f"{gold_dir}/sentiment_trends.csv"
    df_daily.to_csv(trends_path, index=False)
    print(f"Sentiment trends calculated and saved to: {trends_path}")
    
    # Generate daily market briefing
    generate_daily_market_brief(df_daily, df_scores)
    
    logger.stop_timer('sentiment_trends', len(df_daily))

def generate_daily_market_brief(df_daily: pd.DataFrame, df_scores: pd.DataFrame):
    """
    Compiles a comprehensive quantitative daily brief markdown file under reports/
    Specifies encoding='utf-8' to prevent Windows UnicodeEncodeError.
    """
    os.makedirs("reports", exist_ok=True)
    report_path = "reports/daily_market_brief.md"
    
    # Calculate key indicators for today
    latest_row = df_daily.iloc[-1]
    date = latest_row['date']
    momentum = latest_row['sentiment_momentum']
    volatility = latest_row['sentiment_volatility']
    acceleration = latest_row['sentiment_acceleration']
    divergence = latest_row.get('divergence', 0.0)
    
    # Sentiment status classification
    status = "Highly Bullish" if momentum > 0.4 else ("Highly Bearish" if momentum < -0.4 else "Neutral Consolidation")
    
    # Fetch active regime
    current_regime = "Sideways Market"
    if os.path.exists("data/results/market_regime_states.csv"):
        try:
            df_reg = pd.read_csv("data/results/market_regime_states.csv")
            current_regime = df_reg.iloc[-1]['market_regime']
        except Exception:
            pass
            
    # Ingest critical impact events (Impact score > 50)
    critical_alerts = []
    impact_path = "data/gold/sentiment/market_impact_scores.csv"
    if os.path.exists(impact_path):
        try:
            df_imp = pd.read_csv(impact_path)
            high_impacts = df_imp[df_imp['market_impact_score'] >= 50.0].head(5)
            for idx, row in high_impacts.iterrows():
                critical_alerts.append(f"- **[{row['market_impact_class']}]** {row['title']} (Source: *{row['source']}*, Impact Score: **{row['market_impact_score']:.1f}/100**)")
        except Exception:
            pass
            
    # Fetch top topics
    topics_list = []
    topics_path = "data/gold/sentiment/topic_clusters.csv"
    if os.path.exists(topics_path):
        try:
            df_top = pd.read_csv(topics_path)
            top_counts = df_top['topic_theme'].value_counts().head(3)
            for topic, count in top_counts.items():
                topics_list.append(f"- **{topic}** ({count} articles matched)")
        except Exception:
            pass
            
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# QuantSentinel AI - Daily Market Briefing\n")
        f.write(f"**Date:** {date}  \n")
        f.write(f"**Document Reference:** QS-DB-{date}-NLP  \n")
        f.write(f"**Sentiment Regime Status:** `{status}`  \n")
        f.write(f"**Quant Market Alignment:** `{current_regime}`  \n\n")
        
        f.write("## 📋 EXECUTIVE SUMMARY\n")
        f.write("The Sentiment Intelligence Core has completed its continuous scraping and NLP semantic audits. ")
        f.write(f"Today's active market sentiment momentum sits at **{momentum:.4f}** with rolling volatility bounded at **{volatility:.4f}**. ")
        f.write(f"Macroprice divergence indicators highlight a delta of **{divergence:.4f}**, indicating potential risk factors.")
        f.write("\n\n")
        
        f.write("## 📊 NLP TREND MOMENTUM LEDGER\n")
        f.write(f"- **Sentiment Momentum (7d SMA):** {momentum:.4f}\n")
        f.write(f"- **Sentiment Volatility (7d Std Dev):** {volatility:.4f}\n")
        f.write(f"- **Sentiment Acceleration:** {acceleration:.4f}\n")
        f.write(f"- **Sentiment Price Divergence:** {divergence:.4f}\n\n")
        
        f.write("## 💬 DOMINANT SECTOR TOPICS\n")
        if topics_list:
            f.write("The top theme clusters identified in active financial news flows are:\n\n")
            for t in topics_list:
                f.write(t + "\n")
        else:
            f.write("- General / Neutral market consolidation events.\n")
        f.write("\n")
        
        f.write("## 🚨 CRITICAL MARKET IMPACT ALERTS\n")
        if critical_alerts:
            f.write("The following articles were classified as high-impact catalysts for risk-off/on moves:\n\n")
            for a in critical_alerts:
                f.write(a + "\n")
        else:
            f.write("No critical high-impact sentiment events detected in the active pipeline.\n")
        f.write("\n")
        
        f.write("## 🛡️ DIVERGENT RISK ASSESSMENT\n")
        if abs(divergence) > 0.15:
            f.write("> [!WARNING]\n")
            f.write(f"> **High Divergence Warning! (Value: {divergence:.4f})**\n")
            f.write("> Prices returns momentum and sentiment vectors are moving in divergent directions. ")
            f.write("> This represents a classic leading indicator of retail overconfidence or institutional accumulation phase flushes. ")
            f.write("> Restrict max accounts leverage thresholds.\n")
        else:
            f.write("> [!NOTE]\n")
            f.write("> **System Status: Normal**\n")
            f.write("> Macroprice returns and behavioral sentiment channels are structurally aligned. Standard leverage limits remain active.\n")
            
    print(f"Daily Market Brief exported successfully to: {report_path}")

if __name__ == "__main__":
    compute_sentiment_trends()
