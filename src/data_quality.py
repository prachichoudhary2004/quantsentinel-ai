import numpy as np
import pandas as pd
import os
import json

def run_null_checks(df: pd.DataFrame, key_cols: list) -> dict:
    """
    Validates that key columns contain zero missing/null values.
    """
    null_counts = {}
    passed = True
    for col in key_cols:
        if col in df.columns:
            count = int(df[col].isnull().sum())
            null_counts[col] = count
            if count > 0:
                passed = False
        else:
            null_counts[col] = -1 # Column missing
            passed = False
    return {'passed': passed, 'null_counts': null_counts}

def run_duplicate_checks(df: pd.DataFrame, key_col: str = 'Trade ID') -> dict:
    """
    Validates that key identifier column is free of duplicates.
    """
    if key_col not in df.columns:
        return {'passed': False, 'message': f"Key column '{key_col}' missing"}
    
    dup_count = int(df.duplicated(subset=[key_col]).sum())
    passed = dup_count == 0
    return {
        'passed': passed,
        'duplicate_count': dup_count,
        'message': "No duplicate records found" if passed else f"Found {dup_count} duplicate Trade IDs"
    }

def run_schema_checks(df: pd.DataFrame, expected_schema: dict) -> dict:
    """
    Checks that column names exist and type-check correctly.
    """
    mismatches = []
    passed = True
    
    for col, dtype in expected_schema.items():
        if col not in df.columns:
            mismatches.append(f"Column '{col}' is missing")
            passed = False
        else:
            actual_type = str(df[col].dtype)
            # Soft matching types
            if 'int' in dtype and 'int' not in actual_type:
                mismatches.append(f"Column '{col}': Expected {dtype}, got {actual_type}")
                passed = False
            elif 'float' in dtype and 'float' not in actual_type and 'double' not in actual_type:
                mismatches.append(f"Column '{col}': Expected {dtype}, got {actual_type}")
                passed = False
                
    return {
        'passed': passed,
        'mismatches': mismatches,
        'message': "Schema validated successfully" if passed else "Schema mismatch detected"
    }

def detect_outliers_zscore(df: pd.DataFrame, target_col: str, threshold: float = 3.0) -> dict:
    """
    Flags outliers using the standard Z-score method.
    """
    if target_col not in df.columns:
        return {'outlier_count': 0, 'indices': []}
        
    s = df[target_col].dropna()
    if len(s) < 2:
        return {'outlier_count': 0, 'indices': []}
        
    mean = s.mean()
    std = s.std()
    if std == 0:
        return {'outlier_count': 0, 'indices': []}
        
    z_scores = (s - mean) / std
    outliers = z_scores[np.abs(z_scores) > threshold]
    
    return {
        'outlier_count': len(outliers),
        'indices': outliers.index.tolist(),
        'mean': float(mean),
        'std': float(std)
    }

def monitor_sentiment_drift(df_silver: pd.DataFrame) -> dict:
    """
    Monitors data drift on FGI sentiment score.
    Compares the distribution of the recent 30% of trades against the historical 70% baseline.
    Computes mean drift and checks significance.
    """
    if 'sentiment_score' not in df_silver.columns:
        return {'drift_detected': False, 'message': "sentiment_score column missing"}
        
    s = df_silver['sentiment_score'].dropna().values
    if len(s) < 100:
        return {'drift_detected': False, 'message': "Dataset too small for drift analysis"}
        
    split = int(len(s) * 0.7)
    baseline = s[:split]
    recent = s[split:]
    
    mean_base = float(np.mean(baseline))
    mean_recent = float(np.mean(recent))
    
    mean_diff = abs(mean_recent - mean_base)
    
    # If the mean sentiment shifts by more than 15 points, flag a warning
    drift_detected = mean_diff > 15.0
    
    return {
        'drift_detected': drift_detected,
        'baseline_mean': mean_base,
        'recent_mean': mean_recent,
        'mean_difference': mean_diff,
        'message': "Data drift warning: sentiment score mean has shifted significantly!" if drift_detected else "No significant sentiment drift detected"
    }

def run_data_quality_suite():
    """
    Executes all validations on the silver layer and exports a comprehensive validation report.
    """
    print("Running Data Quality & Schema Integrity validations...")
    silver_path = "data/silver/merged_trader_data.csv"
    
    if not os.path.exists(silver_path):
        print(f"Silver dataset missing at {silver_path}. Skipping validation.")
        return
        
    df = pd.read_csv(silver_path)
    
    # Define expected schema
    expected = {
        'Trade ID': 'int',
        'Account': 'string',
        'Size USD': 'float',
        'Closed PnL': 'float',
        'sentiment_score': 'float',
        'leverage': 'int',
        'win_flag': 'int'
    }
    
    null_results = run_null_checks(df, ['Account', 'Trade ID', 'Size USD', 'Closed PnL'])
    dup_results = run_duplicate_checks(df, 'Trade ID')
    schema_results = run_schema_checks(df, expected)
    outliers_pnl = detect_outliers_zscore(df, 'Closed PnL')
    outliers_size = detect_outliers_zscore(df, 'Size USD')
    drift_results = monitor_sentiment_drift(df)
    
    overall_passed = null_results['passed'] and dup_results['passed'] and schema_results['passed']
    
    report = {
        'overall_passed': overall_passed,
        'null_checks': null_results,
        'duplicate_checks': dup_results,
        'schema_checks': schema_results,
        'outlier_checks': {
            'closed_pnl_outliers': outliers_pnl['outlier_count'],
            'size_usd_outliers': outliers_size['outlier_count']
        },
        'sentiment_drift': drift_results
    }
    
    # Save quality report
    os.makedirs("data/results", exist_ok=True)
    report_path = "data/results/data_quality_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"Data validation suite finished successfully! Report saved to {report_path}")
    print(f"Overall Quality Passed: {overall_passed}")

if __name__ == "__main__":
    run_data_quality_suite()
