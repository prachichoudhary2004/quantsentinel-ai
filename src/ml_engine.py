import os
import json
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

def run_trader_segmentation():
    print("Starting trader segmentation clustering...")
    df_realized = pd.read_csv("data/results/trader_overall.csv")
    
    # We will also add trade direction bias (Long Ratio) to the trader profiles
    df_merged = pd.read_csv("data/processed/merged_trader_data.csv")
    df_closing = df_merged[df_merged['is_closing_trade'] == True]
    
    long_counts = df_closing[df_closing['trade_direction'] == 'Long'].groupby('Account').size()
    total_counts = df_closing.groupby('Account').size()
    long_ratio = (long_counts / total_counts).fillna(0.5)
    
    df_realized['long_ratio'] = df_realized['Account'].map(long_ratio)
    
    # Columns to use for clustering (fill any NaNs in profit_factor with a neutral value like 1.0)
    df_realized['profit_factor'] = df_realized['profit_factor'].replace([np.inf, -np.inf], 100.0).fillna(1.0)
    
    features = ['total_pnl', 'win_rate', 'avg_leverage', 'avg_trade_size', 'trade_count', 'profit_factor', 'long_ratio']
    X = df_realized[features].copy()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Run KMeans with K=4 (logical number of profiles for 32 accounts)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_realized['segment_id'] = kmeans.fit_predict(X_scaled)
    
    # Map Segment IDs to professional quant profiles based on their centroids
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    df_centroids = pd.DataFrame(centroids, columns=features)
    print("\nCluster Centroids:")
    print(df_centroids)
    
    # Let's map cluster IDs to human-readable personas dynamically based on stats
    # Persona 1: High Total PnL & High Avg Trade Size -> Institutional Whales
    # Persona 2: High Avg Leverage & Low Win Rate/PnL -> High-Leverage Gamblers
    # Persona 3: High Trade Count & Moderate PnL -> Systematic Scalpers
    # Persona 4: Low Trade Count & Moderate PnL -> Passive / Low-Activity Traders
    
    # We will do a simple manual mapping based on ranking
    # Cluster with highest total PnL is "Whale Momentum Traders"
    # Cluster with highest average leverage is "High-Leverage Speculators"
    # Cluster with highest trade count is "High-Frequency Scalpers"
    # Cluster with lowest trade count is "Conservative/Tactical Retail"
    
    cluster_pnl = df_realized.groupby('segment_id')['total_pnl'].mean()
    cluster_lev = df_realized.groupby('segment_id')['avg_leverage'].mean()
    cluster_count = df_realized.groupby('segment_id')['trade_count'].mean()
    
    persona_mapping = {}
    remaining_ids = list(range(4))
    
    # Find High-Frequency Scalpers (highest trade count)
    hft_id = cluster_count.idxmax()
    persona_mapping[hft_id] = "Systematic Scalpers"
    if hft_id in remaining_ids: remaining_ids.remove(hft_id)
        
    # Find Institutional Whales (highest PnL from remaining)
    whale_id = cluster_pnl.loc[remaining_ids].idxmax()
    persona_mapping[whale_id] = "Institutional Whales"
    if whale_id in remaining_ids: remaining_ids.remove(whale_id)
        
    # Find High-Leverage Speculators (highest leverage from remaining)
    speculator_id = cluster_lev.loc[remaining_ids].idxmax()
    persona_mapping[speculator_id] = "High-Leverage Speculators"
    if speculator_id in remaining_ids: remaining_ids.remove(speculator_id)
        
    # Last one is Conservative/Tactical Retail
    if remaining_ids:
        retail_id = remaining_ids[0]
        persona_mapping[retail_id] = "Conservative / Tactical Retail"
        
    df_realized['trader_segment'] = df_realized['segment_id'].map(persona_mapping)
    print("\nTrader Personas Distribution:")
    print(df_realized['trader_segment'].value_counts())
    
    df_realized.to_csv("data/results/trader_segments.csv", index=False)
    print("Trader segmentation complete! Saved to data/results/trader_segments.csv")

def train_predictive_model():
    print("\nTraining predictive model for profitable trades...")
    df_merged = pd.read_csv("data/processed/merged_trader_data.csv")
    
    # Focus only on realized closing trades
    df_model = df_merged[df_merged['is_closing_trade'] == True].copy()
    
    # Features for classification
    # Encode categorical features: Coin, trade_direction, sentiment_class
    # We will use one-hot encoding for simplicity and robustness
    
    # Keep key continuous features
    # Select important features
    df_features = df_model[['leverage', 'Size USD', 'sentiment_score', 'trade_direction', 'hour', 'day_of_week', 'Coin']].copy()
    
    # Target
    y = df_model['win_flag'].values
    
    # One-hot encode categorical features
    X_encoded = pd.get_dummies(df_features, columns=['trade_direction', 'day_of_week', 'Coin'], drop_first=True)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42, stratify=y)
    
    print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    # Train a Random Forest Classifier
    # Limit max_depth to prevent overfitting and make it run fast
    rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    
    # Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    metrics = {
        "accuracy": accuracy,
        "roc_auc": auc,
        "classification_report": report
    }
    
    with open("data/results/model_evaluation.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Model Performance: Accuracy = {accuracy:.4f}, ROC-AUC = {auc:.4f}")
    
    # Feature Importances
    importances = rf.feature_importances_
    df_importances = pd.DataFrame({
        'feature': X_encoded.columns,
        'importance': importances
    }).sort_values('importance', ascending=False).head(20)
    
    df_importances.to_csv("data/results/feature_importance.csv", index=False)
    print("Top 10 most predictive features for trade profitability:")
    print(df_importances.head(10))
    
    # Save the trained model and feature columns list using pickle
    import pickle
    with open("data/results/profitable_trade_model.pkl", "wb") as f:
        pickle.dump((rf, X_encoded.columns.tolist()), f)
    print("Trained model saved successfully as data/results/profitable_trade_model.pkl")
    print("Predictive modeling complete! Results saved in data/results/")

if __name__ == "__main__":
    run_trader_segmentation()
    train_predictive_model()
