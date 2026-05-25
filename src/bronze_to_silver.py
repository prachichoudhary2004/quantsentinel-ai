import os
import sys
import pandas as pd
import numpy as np
import hashlib

# Adjust sys.path to resolve internal package imports robustly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Dynamic PySpark imports
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, udf, when, to_date, hour, dayofweek, date_format
    from pyspark.sql.types import IntegerType, DoubleType, StringType
    HAS_SPARK = True
except ImportError:
    HAS_SPARK = False

def get_deterministic_leverage_py(trade_id, size_usd, timestamp):
    """
    Python helper for deterministic leverage simulation.
    """
    h_str = str(trade_id) if pd.notnull(trade_id) else str(timestamp)
    h = int(hashlib.md5(h_str.encode()).hexdigest(), 16)
    np.random.seed(h % (2**32))
    
    if size_usd > 50000:
        choices = [1, 2, 3, 5]
        probs = [0.45, 0.40, 0.12, 0.03]
    elif size_usd > 10000:
        choices = [1, 2, 3, 5, 10]
        probs = [0.20, 0.35, 0.25, 0.15, 0.05]
    elif size_usd > 1000:
        choices = [2, 3, 5, 10, 20]
        probs = [0.10, 0.25, 0.35, 0.20, 0.10]
    else:
        choices = [3, 5, 10, 20, 50]
        probs = [0.10, 0.20, 0.40, 0.20, 0.10]
        
    return int(np.random.choice(choices, p=probs))

def get_leverage_bucket_py(lev):
    if lev <= 3:
        return 'Low Leverage (1x-3x)'
    elif lev <= 10:
        return 'Medium Leverage (4x-10x)'
    elif lev <= 20:
        return 'High Leverage (11x-20x)'
    else:
        return 'Extreme Leverage (21x-50x)'

def execute_spark_job():
    """
    Executes the Bronze-to-Silver data pipeline using PySpark and Delta tables.
    """
    print("Initiating PySpark Bronze to Silver Medallion pipeline...")
    
    # Configure optimized Spark session
    spark = SparkSession.builder \
        .appName("CryptoPulse_Bronze_To_Silver") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.driver.memory", "2g") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()
        
    print(f"SparkSession active: version {spark.version}")
    
    # Paths
    raw_hl_path = "data/bronze/hyperliquid_trader_data.csv"
    raw_fg_path = "data/bronze/bitcoin_fear_greed_index.csv"
    
    # Read Bronze layers
    df_hl = spark.read.csv(raw_hl_path, header=True, inferSchema=True)
    df_fg = spark.read.csv(raw_fg_path, header=True, inferSchema=True)
    
    # UDF registrations for leverage calculations
    udf_leverage = udf(get_deterministic_leverage_py, IntegerType())
    udf_bucket = udf(get_leverage_bucket_py, StringType())
    
    # Preprocess Fear & Greed index
    df_fg_clean = df_fg.select(
        date_format(to_date(col("date")), "yyyy-MM-dd").alias("date"),
        col("value").alias("sentiment_score"),
        col("classification").alias("sentiment_class")
    )
    
    # Preprocess Hyperliquid dates, day_of_week and timestamps
    df_hl_clean = df_hl \
        .withColumn("datetime_ist", col("Timestamp IST").cast("string")) \
        .withColumn("date", date_format(to_date(col("Timestamp IST"), "dd-MM-yyyy HH:mm"), "yyyy-MM-dd")) \
        .withColumn("hour", hour(col("Timestamp IST").cast("timestamp"))) \
        .withColumn("day_of_week", date_format(to_date(col("Timestamp IST"), "dd-MM-yyyy HH:mm"), "EEEE")) \
        .withColumn("trade_direction", 
                    when(col("Direction").isin(['Buy', 'Open Long', 'Close Long', 'Short > Long']), 'Long')
                    .when(col("Direction").isin(['Sell', 'Open Short', 'Close Short', 'Long > Short', 'Liquidated Isolated Short']), 'Short')
                    .otherwise('Other'))
                    
    # Joint merge (Silver curation)
    df_silver = df_hl_clean.join(df_fg_clean, on="date", how="left")
    
    # Compute leverage metrics
    df_silver = df_silver \
        .withColumn("leverage", udf_leverage(col("Trade ID"), col("Size USD"), col("Timestamp IST"))) \
        .withColumn("leverage_bucket", udf_bucket(col("leverage"))) \
        .withColumn("pnl_pct", when(col("Size USD") > 0, col("Closed PnL") / col("Size USD")).otherwise(0.0)) \
        .withColumn("roe", col("pnl_pct") * col("leverage")) \
        .withColumn("is_closing_trade", when(col("Closed PnL") != 0, True).otherwise(False)) \
        .withColumn("win_flag", when(col("Closed PnL") > 0, 1).otherwise(0))
        
    # Write to Silver Delta Lake
    silver_delta_path = "data/silver/delta/"
    os.makedirs(silver_delta_path, exist_ok=True)
    df_silver.write.format("delta").mode("overwrite").save(silver_delta_path)
    
    # Write standard CSV to maintain backward compatibility
    silver_csv_path = "data/silver/merged_trader_data.csv"
    df_silver_pd = df_silver.toPandas()
    df_silver_pd.to_csv(silver_csv_path, index=False)
    os.makedirs("data/processed", exist_ok=True)
    df_silver_pd.to_csv("data/processed/merged_trader_data.csv", index=False)
    
    print(f"Bronze to Silver PySpark job completed! Data successfully written.")
    spark.stop()

def execute_pandas_fallback():
    """
    Pandas local fallback engine mimicking the PySpark pipeline.
    Ensures complete operational resilience on Windows / Non-Java architectures.
    """
    print("Using robust Pandas local fallback engine for Bronze to Silver Medallion pipeline...")
    
    # Paths
    raw_hl_path = "data/bronze/hyperliquid_trader_data.csv"
    raw_fg_path = "data/bronze/bitcoin_fear_greed_index.csv"
    
    df_hl = pd.read_csv(raw_hl_path)
    df_fg = pd.read_csv(raw_fg_path)
    
    # Timestamp parsing and day of week computation
    df_hl['datetime_ist'] = pd.to_datetime(df_hl['Timestamp IST'], format='%d-%m-%Y %H:%M')
    df_hl['date'] = df_hl['datetime_ist'].dt.strftime('%Y-%m-%d')
    df_hl['hour'] = df_hl['datetime_ist'].dt.hour
    df_hl['day_of_week'] = df_hl['datetime_ist'].dt.day_name()
    
    df_fg['date'] = pd.to_datetime(df_fg['date']).dt.strftime('%Y-%m-%d')
    df_fg_clean = df_fg[['date', 'value', 'classification']].rename(
        columns={'value': 'sentiment_score', 'classification': 'sentiment_class'}
    )
    
    # Curation join
    df_silver = pd.merge(df_hl, df_fg_clean, on='date', how='left')
    df_silver['sentiment_score'] = df_silver['sentiment_score'].ffill().bfill().fillna(50)
    df_silver['sentiment_class'] = df_silver['sentiment_class'].ffill().bfill().fillna('Neutral')
    
    # Metrics computation
    long_directions = ['Buy', 'Open Long', 'Close Long', 'Short > Long']
    short_directions = ['Sell', 'Open Short', 'Close Short', 'Long > Short', 'Liquidated Isolated Short']
    
    def map_direction(direction):
        if direction in long_directions:
            return 'Long'
        elif direction in short_directions:
            return 'Short'
        else:
            return 'Other'
            
    df_silver['trade_direction'] = df_silver['Direction'].apply(map_direction)
    df_silver['leverage'] = df_silver.apply(lambda r: get_deterministic_leverage_py(r['Trade ID'], r['Size USD'], r['Timestamp IST']), axis=1)
    df_silver['leverage_bucket'] = df_silver['leverage'].apply(get_leverage_bucket_py)
    
    df_silver['pnl_pct'] = np.where(df_silver['Size USD'] > 0, df_silver['Closed PnL'] / df_silver['Size USD'], 0.0)
    df_silver['roe'] = df_silver['pnl_pct'] * df_silver['leverage']
    df_silver['is_closing_trade'] = df_silver['Closed PnL'] != 0
    df_silver['win_flag'] = np.where(df_silver['Closed PnL'] > 0, 1, 0)
    
    # Save outputs to both Medallion and legacy directories for absolute backward compatibility
    os.makedirs("data/silver", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    df_silver.to_csv("data/silver/merged_trader_data.csv", index=False)
    df_silver.to_csv("data/processed/merged_trader_data.csv", index=False)
    
    # Simulate a Delta Lake structure on local disk using Parquet
    os.makedirs("data/silver/delta", exist_ok=True)
    df_silver.to_parquet("data/silver/delta/part-0.parquet", index=False)
    
    print(f"Bronze to Silver Pandas fallback pipeline completed! Output saved.")

def run_bronze_to_silver():
    if HAS_SPARK:
        try:
            execute_spark_job()
        except Exception as e:
            print(f"Spark execution failed: {e}. Launching fallback...")
            execute_pandas_fallback()
    else:
        execute_pandas_fallback()

if __name__ == "__main__":
    os.makedirs("data/bronze", exist_ok=True)
    if os.path.exists("data/raw/hyperliquid_trader_data.csv"):
        import shutil
        shutil.copy("data/raw/hyperliquid_trader_data.csv", "data/bronze/hyperliquid_trader_data.csv")
        shutil.copy("data/raw/bitcoin_fear_greed_index.csv", "data/bronze/bitcoin_fear_greed_index.csv")
    run_bronze_to_silver()
