import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

def detect_market_regimes(df_trades: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies market regimes across all dates in the dataset using a hybrid
    unsupervised clustering (K-Means) and statistical threshold engine.
    
    Features constructed:
    1. Daily Return Direction / Average ROE
    2. Rolling Volatility (30-day) of trade returns
    3. Daily Fear & Greed index score
    4. Sizing / Volume intensity
    
    Regimes classified:
    - Bull Market
    - Bear Market
    - Sideways Market
    - High Volatility / Turbulence
    """
    # 1. Group transaction fills to construct a Daily Macro Metric dataset
    print("Constructing daily macro metrics for regime detection...")
    
    # Check key columns
    roe_col = 'roe' if 'roe' in df_trades.columns else 'Closed PnL'
    pnl_col = 'Closed PnL'
    vol_col = 'Size USD'
    
    df_daily = df_trades.groupby('date').agg(
        avg_roe=(roe_col, 'mean'),
        total_pnl=(pnl_col, 'sum'),
        trade_count=(pnl_col, 'count'),
        total_volume=(vol_col, 'sum'),
        sentiment_score=('sentiment_score', 'first')
    ).reset_index()
    
    # Sort chronologically
    df_daily = df_daily.sort_values('date').reset_index(drop=True)
    
    # Fill any sentiment gaps
    df_daily['sentiment_score'] = df_daily['sentiment_score'].ffill().bfill().fillna(50)
    
    # Calculate rolling volatility of returns (10-day window to support shorter datasets)
    df_daily['rolling_volatility'] = df_daily['avg_roe'].rolling(window=10, min_periods=1).std().fillna(0.0)
    
    # Base features for clustering
    features = ['avg_roe', 'rolling_volatility', 'sentiment_score', 'total_volume']
    
    # Run K-Means if we have at least 15 days of data; else use deterministic rule-based classifications
    if len(df_daily) >= 15:
        try:
            print("Clustering daily regimes using K-Means (K=4)...")
            X = df_daily[features].copy()
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
            df_daily['cluster_id'] = kmeans.fit_predict(X_scaled)
            
            # Map Cluster IDs to semantic regimes by ranking centroids
            centroids = scaler.inverse_transform(kmeans.cluster_centers_)
            df_centroids = pd.DataFrame(centroids, columns=features)
            
            # Label mapping algorithm based on statistics
            # Bull: Highest FGI sentiment and positive roe
            # Bear: Lowest FGI sentiment and negative pnl
            # High Volatility: Highest rolling volatility
            # Sideways: Lowest rolling volatility
            
            regime_map = {}
            remaining_clusters = list(range(4))
            
            # A. High Volatility (max rolling volatility)
            vol_idx = df_centroids['rolling_volatility'].idxmax()
            regime_map[vol_idx] = "High Volatility / Turbulence"
            if vol_idx in remaining_clusters: remaining_clusters.remove(vol_idx)
                
            # B. Sideways (min rolling volatility from remaining)
            side_idx = df_centroids.loc[remaining_clusters, 'rolling_volatility'].idxmin()
            regime_map[side_idx] = "Sideways Market"
            if side_idx in remaining_clusters: remaining_clusters.remove(side_idx)
                
            # C. Bull (highest avg sentiment from remaining)
            bull_idx = df_centroids.loc[remaining_clusters, 'sentiment_score'].idxmax()
            regime_map[bull_idx] = "Bull Market"
            if bull_idx in remaining_clusters: remaining_clusters.remove(bull_idx)
                
            # D. Bear (the last remaining cluster)
            if remaining_clusters:
                bear_idx = remaining_clusters[0]
                regime_map[bear_idx] = "Bear Market"
                
            df_daily['market_regime'] = df_daily['cluster_id'].map(regime_map)
            
        except Exception as e:
            print(f"Clustering failed: {e}. Falling back to rule-based regime engine...")
            df_daily['market_regime'] = df_daily.apply(classify_regime_rule_based, axis=1)
    else:
        print("Dataset size insufficient for unsupervised clustering. Utilizing rule-based regime engine...")
        df_daily['market_regime'] = df_daily.apply(classify_regime_rule_based, axis=1)
        
    print(f"Regime classification completed successfully!")
    print(df_daily['market_regime'].value_counts())
    
    # Save results
    os.makedirs("data/results", exist_ok=True)
    df_daily.to_csv("data/results/market_regime_states.csv", index=False)
    
    return df_daily

def classify_regime_rule_based(row) -> str:
    """
    Deterministic fallback regime classifier based on technical indicators.
    """
    sent = row['sentiment_score']
    vol = row['rolling_volatility'] if 'rolling_volatility' in row else 0.05
    roe = row['avg_roe']
    
    # High Volatility rule
    if vol > 0.15 or (sent < 20 and roe < -0.05) or (sent > 80 and roe > 0.05):
        return "High Volatility / Turbulence"
    # Bull Market rule
    elif sent >= 55 and roe >= -0.01:
        return "Bull Market"
    # Bear Market rule
    elif sent <= 40 and roe < 0.01:
        return "Bear Market"
    # Sideways Market rule
    else:
        return "Sideways Market"
