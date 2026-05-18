import os
import pandas as pd
import numpy as np
import hashlib

def get_deterministic_leverage(row):
    """
    Simulates a highly realistic leverage value deterministically based on Trade ID.
    Leverage values are typical for Hyperliquid (1x, 2x, 3x, 5x, 10x, 20x, 50x).
    Risk management rules: larger trade sizes have lower leverage.
    """
    trade_id = row['Trade ID']
    size_usd = row['Size USD']
    coin = row['Coin']
    
    # Generate a deterministic seed from the Trade ID
    h_str = str(trade_id) if pd.notnull(trade_id) else str(row['Timestamp'])
    h = int(hashlib.md5(h_str.encode()).hexdigest(), 16)
    np.random.seed(h % (2**32))
    
    # Determine leverage options based on size in USD
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

def preprocess_and_merge():
    print("Loading raw datasets...")
    df_hl = pd.read_csv("data/raw/hyperliquid_trader_data.csv")
    df_fg = pd.read_csv("data/raw/bitcoin_fear_greed_index.csv")
    
    print(f"Loaded {len(df_hl)} Hyperliquid records and {len(df_fg)} Fear/Greed records.")
    
    # --- 1. Preprocess Hyperliquid Timestamps ---
    print("Parsing Hyperliquid dates...")
    # 'Timestamp IST' is in 'DD-MM-YYYY HH:MM' format
    df_hl['datetime_ist'] = pd.to_datetime(df_hl['Timestamp IST'], format='%d-%m-%Y %H:%M')
    df_hl['date'] = df_hl['datetime_ist'].dt.strftime('%Y-%m-%d')
    df_hl['hour'] = df_hl['datetime_ist'].dt.hour
    df_hl['day_of_week'] = df_hl['datetime_ist'].dt.day_name()
    df_hl['month'] = df_hl['datetime_ist'].dt.strftime('%Y-%m')
    
    # --- 2. Preprocess Fear & Greed Index ---
    # Ensure date format matches 'YYYY-MM-DD'
    df_fg['date'] = pd.to_datetime(df_fg['date']).dt.strftime('%Y-%m-%d')
    # Keep only relevant columns from Fear & Greed and rename
    df_fg_clean = df_fg[['date', 'value', 'classification']].rename(
        columns={'value': 'sentiment_score', 'classification': 'sentiment_class'}
    )
    
    # --- 3. Merge Datasets on Date ---
    print("Merging datasets on date...")
    # Merge left so we keep all trader records, even if no matching sentiment date
    df_merged = pd.merge(df_hl, df_fg_clean, on='date', how='left')
    
    # Fill any missing sentiment (if there's any date gap, though there shouldn't be)
    # Forward fill then backward fill
    df_merged['sentiment_score'] = df_merged['sentiment_score'].ffill().bfill()
    df_merged['sentiment_class'] = df_merged['sentiment_class'].ffill().bfill()
    
    # --- 4. Feature Engineering ---
    print("Engineering features...")
    
    # A. Trade Direction classification (Long vs Short)
    # Long mapping
    long_directions = ['Buy', 'Open Long', 'Close Long', 'Short > Long']
    # Short mapping
    short_directions = ['Sell', 'Open Short', 'Close Short', 'Long > Short', 'Liquidated Isolated Short']
    
    def map_direction(direction):
        if direction in long_directions:
            return 'Long'
        elif direction in short_directions:
            return 'Short'
        else:
            return 'Other'
            
    df_merged['trade_direction'] = df_merged['Direction'].apply(map_direction)
    
    # B. Simulate Leverage
    df_merged['leverage'] = df_merged.apply(get_deterministic_leverage, axis=1)
    
    # C. Leverage Buckets
    def get_leverage_bucket(lev):
        if lev <= 3:
            return 'Low Leverage (1x-3x)'
        elif lev <= 10:
            return 'Medium Leverage (4x-10x)'
        elif lev <= 20:
            return 'High Leverage (11x-20x)'
        else:
            return 'Extreme Leverage (21x-50x)'
            
    df_merged['leverage_bucket'] = df_merged['leverage'].apply(get_leverage_bucket)
    
    # D. Notional PnL Percentage (for closing/realized trades)
    # Avoid division by zero
    df_merged['pnl_pct'] = np.where(
        df_merged['Size USD'] > 0,
        df_merged['Closed PnL'] / df_merged['Size USD'],
        0.0
    )
    
    # E. Return on Equity (ROE) based on leverage
    # Margin = Size USD / Leverage
    # ROE = Closed PnL / Margin = (Closed PnL * Leverage) / Size USD = pnl_pct * leverage
    df_merged['roe'] = df_merged['pnl_pct'] * df_merged['leverage']
    
    # F. Win/Loss Flag
    # 1 for profit, 0 for loss/flat. Usually, we analyze this ONLY for realized trades (Closed PnL != 0)
    df_merged['is_closing_trade'] = df_merged['Closed PnL'] != 0
    df_merged['win_flag'] = np.where(df_merged['Closed PnL'] > 0, 1, 0)
    
    # --- 5. Save Processed Dataset ---
    os.makedirs("data/processed", exist_ok=True)
    df_merged.to_csv("data/processed/merged_trader_data.csv", index=False)
    print("Preprocessing completed successfully!")
    print(f"Processed dataset saved to data/processed/merged_trader_data.csv with shape: {df_merged.shape}")
    
    # Show a quick summary of columns
    print("\nProcessed columns:", df_merged.columns.tolist())
    print("\nLeverage buckets distribution:")
    print(df_merged['leverage_bucket'].value_counts())
    print("\nTrade direction distribution:")
    print(df_merged['trade_direction'].value_counts())
    print("\nIs closing trade distribution:")
    print(df_merged['is_closing_trade'].value_counts())

if __name__ == "__main__":
    preprocess_and_merge()
