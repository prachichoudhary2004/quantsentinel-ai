import os
import sys

# Adjust sys.path to resolve internal package imports robustly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

# Dynamic imports
try:
    from pyspark.sql import SparkSession
    HAS_SPARK = True
except ImportError:
    HAS_SPARK = False

# Import analytical modules built in Phase 1
from src.risk_analytics import generate_risk_metrics_report
from src.portfolio_analytics import run_portfolio_simulation
from src.regime_detection import detect_market_regimes

def execute_spark_job():
    print("Initiating PySpark Silver to Gold Medallion pipeline...")
    
    spark = SparkSession.builder \
        .appName("CryptoPulse_Silver_To_Gold") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
        
    print(f"SparkSession active: version {spark.version}")
    
    # Paths
    silver_path = "data/silver/merged_trader_data.csv"
    
    # Spark read
    df_silver_spark = spark.read.csv(silver_path, header=True, inferSchema=True)
    df_silver_pd = df_silver_spark.toPandas()
    
    # 1. Market Regime Detection (Gold Curation)
    print("Generating Market Regimes Gold asset...")
    df_regimes = detect_market_regimes(df_silver_pd)
    
    # 2. Advanced Risk Metrics Gold Asset
    print("Generating Risk Metrics Gold asset...")
    df_risk = generate_risk_metrics_report(df_silver_pd, group_col='sentiment_class')
    
    # 3. Portfolio Sizing Simulation Gold Asset
    print("Generating Portfolio Simulation Gold asset...")
    df_portfolio = run_portfolio_simulation(df_silver_pd)
    
    # Convert Gold assets to Spark DataFrames and store in Delta
    os.makedirs("data/gold/delta/regimes", exist_ok=True)
    os.makedirs("data/gold/delta/risk", exist_ok=True)
    os.makedirs("data/gold/delta/portfolio", exist_ok=True)
    
    # Save standard Gold CSV outputs
    df_regimes.to_csv("data/gold/market_regimes.csv", index=False)
    df_risk.to_csv("data/gold/risk_analysis.csv", index=False)
    df_portfolio.to_csv("data/gold/portfolio_simulations.csv", index=False)
    
    # Save Delta parquet representations
    df_regimes.to_parquet("data/gold/delta/regimes/part-0.parquet", index=False)
    df_risk.to_parquet("data/gold/delta/risk/part-0.parquet", index=False)
    df_portfolio.to_parquet("data/gold/delta/portfolio/part-0.parquet", index=False)
    
    print("Silver to Gold Medallion pipelines executed successfully via local Spark configuration!")
    spark.stop()

def execute_pandas_fallback():
    print("Using robust Pandas local fallback engine for Silver to Gold Medallion pipeline...")
    
    silver_path = "data/silver/merged_trader_data.csv"
    df_silver = pd.read_csv(silver_path)
    
    # 1. Regime Detection
    print("Running Market Regime Detection...")
    df_regimes = detect_market_regimes(df_silver)
    
    # 2. Risk Metrics
    print("Running Risk Analytics Report...")
    df_risk = generate_risk_metrics_report(df_silver, group_col='sentiment_class')
    
    # 3. Portfolio Simulator
    print("Running Portfolio Sizing Simulator...")
    df_portfolio = run_portfolio_simulation(df_silver)
    
    # Save standard Gold CSV outputs
    os.makedirs("data/gold", exist_ok=True)
    df_regimes.to_csv("data/gold/market_regimes.csv", index=False)
    df_risk.to_csv("data/gold/risk_analysis.csv", index=False)
    df_portfolio.to_csv("data/gold/portfolio_simulations.csv", index=False)
    
    # Save simulated Delta formats
    os.makedirs("data/gold/delta/regimes", exist_ok=True)
    os.makedirs("data/gold/delta/risk", exist_ok=True)
    os.makedirs("data/gold/delta/portfolio", exist_ok=True)
    
    df_regimes.to_parquet("data/gold/delta/regimes/part-0.parquet", index=False)
    df_risk.to_parquet("data/gold/delta/risk/part-0.parquet", index=False)
    df_portfolio.to_parquet("data/gold/delta/portfolio/part-0.parquet", index=False)
    
    print("Silver to Gold Medallion fallback execution completed successfully!")

def run_silver_to_gold():
    if HAS_SPARK:
        try:
            execute_spark_job()
        except Exception as e:
            print(f"Spark execution failed: {e}. Launching fallback...")
            execute_pandas_fallback()
    else:
        execute_pandas_fallback()

if __name__ == "__main__":
    run_silver_to_gold()
