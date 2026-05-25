import os
import json
import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix
)

# Dynamic imports for advanced gradient boosting libraries
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

def run_trader_segmentation():
    print("Starting trader segmentation clustering...")
    df_realized = pd.read_csv("data/results/trader_overall.csv")
    
    # Add trade direction bias (Long Ratio) to the trader profiles
    df_merged = pd.read_csv("data/processed/merged_trader_data.csv")
    df_closing = df_merged[df_merged['is_closing_trade'] == True]
    
    long_counts = df_closing[df_closing['trade_direction'] == 'Long'].groupby('Account').size()
    total_counts = df_closing.groupby('Account').size()
    long_ratio = (long_counts / total_counts).fillna(0.5)
    
    df_realized['long_ratio'] = df_realized['Account'].map(long_ratio)
    df_realized['profit_factor'] = df_realized['profit_factor'].replace([np.inf, -np.inf], 100.0).fillna(1.0)
    
    features = ['total_pnl', 'win_rate', 'avg_leverage', 'avg_trade_size', 'trade_count', 'profit_factor', 'long_ratio']
    X = df_realized[features].copy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_realized['segment_id'] = kmeans.fit_predict(X_scaled)
    
    # Centroids mapping
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    df_centroids = pd.DataFrame(centroids, columns=features)
    
    cluster_pnl = df_realized.groupby('segment_id')['total_pnl'].mean()
    cluster_lev = df_realized.groupby('segment_id')['avg_leverage'].mean()
    cluster_count = df_realized.groupby('segment_id')['trade_count'].mean()
    
    persona_mapping = {}
    remaining_ids = list(range(4))
    
    hft_id = cluster_count.idxmax()
    persona_mapping[hft_id] = "Systematic Scalpers"
    if hft_id in remaining_ids: remaining_ids.remove(hft_id)
        
    whale_id = cluster_pnl.loc[remaining_ids].idxmax()
    persona_mapping[whale_id] = "Institutional Whales"
    if whale_id in remaining_ids: remaining_ids.remove(whale_id)
        
    speculator_id = cluster_lev.loc[remaining_ids].idxmax()
    persona_mapping[speculator_id] = "High-Leverage Speculators"
    if speculator_id in remaining_ids: remaining_ids.remove(speculator_id)
        
    if remaining_ids:
        retail_id = remaining_ids[0]
        persona_mapping[retail_id] = "Conservative / Tactical Retail"
        
    df_realized['trader_segment'] = df_realized['segment_id'].map(persona_mapping)
    print("\nTrader Personas Distribution:")
    print(df_realized['trader_segment'].value_counts())
    
    df_realized.to_csv("data/results/trader_segments.csv", index=False)
    print("Trader segmentation complete! Saved to data/results/trader_segments.csv")

def train_predictive_model():
    print("\nTraining predictive models comparison framework...")
    df_merged = pd.read_csv("data/processed/merged_trader_data.csv")
    
    df_model = df_merged[df_merged['is_closing_trade'] == True].copy()
    
    df_features = df_model[['leverage', 'Size USD', 'sentiment_score', 'trade_direction', 'hour', 'day_of_week', 'Coin']].copy()
    y = df_model['win_flag'].values
    
    # Categorical One-Hot Encoding
    X_encoded = pd.get_dummies(df_features, columns=['trade_direction', 'day_of_week', 'Coin'])
    
    # Align boolean columns to standard numerical format for robust training across classifiers
    for col in X_encoded.columns:
        if X_encoded[col].dtype == bool:
            X_encoded[col] = X_encoded[col].astype(int)
            
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42, stratify=y)
    print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    # Setup model comparison dictionary
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    # Dynamically append XGBoost if package exists
    if HAS_XGB:
        print("XGBoost is available. Adding to comparison suite...")
        models['XGBoost'] = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss', n_jobs=-1)
    else:
        print("XGBoost package missing. Skipping in comparison framework.")
        
    # Dynamically append LightGBM if package exists
    if HAS_LGBM:
        print("LightGBM is available. Adding to comparison suite...")
        models['LightGBM'] = LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbose=-1, n_jobs=-1)
    else:
        print("LightGBM package missing. Skipping in comparison framework.")
        
    # Training results tracking
    comparison_results = {}
    trained_estimators = {}
    best_model_name = None
    best_f1 = -1.0
    
    for name, clf in models.items():
        print(f"Training {name} Classifier...")
        clf.fit(X_train, y_train)
        trained_estimators[name] = clf
        
        # Predictions
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else [0.5] * len(y_pred)
        
        # Calculate standard metric outputs
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        auc = float(roc_auc_score(y_test, y_prob))
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        comparison_results[name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'roc_auc': auc,
            'confusion_matrix': cm
        }
        print(f"-> {name} | F1: {f1:.4f} | Accuracy: {acc:.4f} | ROC-AUC: {auc:.4f}")
        
        # Track optimal estimator maximizing F1-Score
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            
    print(f"\nBest Selected Model: {best_model_name} (F1 Score: {best_f1:.4f})")
    
    best_clf = trained_estimators[best_model_name]
    
    # Save the evaluation logs
    os.makedirs("data/results", exist_ok=True)
    with open("data/results/model_evaluation.json", "w") as f:
        json.dump({
            'best_model': best_model_name,
            'comparison': comparison_results
        }, f, indent=4)
        
    # Standardize feature importances for selected model
    if hasattr(best_clf, 'feature_importances_'):
        importances = best_clf.feature_importances_
    elif hasattr(best_clf, 'coef_'):
        importances = np.abs(best_clf.coef_[0])
    else:
        importances = np.zeros(X_encoded.shape[1])
        
    df_importances = pd.DataFrame({
        'feature': X_encoded.columns,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    df_importances.to_csv("data/results/feature_importance.csv", index=False)
    
    # Pickling the best estimator directly to preserve 100% backward compatibility
    with open("data/results/profitable_trade_model.pkl", "wb") as f:
        pickle.dump((best_clf, X_encoded.columns.tolist()), f)
        
    print(f"Trained best model ({best_model_name}) serialized successfully as profitable_trade_model.pkl!")
    
if __name__ == "__main__":
    run_trader_segmentation()
    train_predictive_model()
