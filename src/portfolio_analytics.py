import numpy as np
import pandas as pd

def calculate_kelly_fraction(win_rate: float, avg_win_usd: float, avg_loss_usd: float) -> float:
    """
    Calculates the Kelly Criterion fraction: f* = p - (1-p) / b
    Where p is win probability, and b is the win-to-loss ratio (avg win / avg loss).
    """
    if avg_loss_usd == 0:
        return 0.0
    b = abs(avg_win_usd / avg_loss_usd)
    if b == 0:
        return 0.0
    f_star = win_rate - (1 - win_rate) / b
    # Apply a standard fractional Kelly cap (e.g., Half-Kelly) to prevent aggressive leverage
    return float(np.clip(f_star * 0.5, 0.0, 0.5))

def position_size_fixed_fractional(capital: float, risk_percentage: float = 0.02) -> float:
    """
    Determines max capital allocation based on a fixed risk percentage (e.g. 2% of capital).
    """
    return capital * risk_percentage

def position_size_vol_adjusted(capital: float, vol: float, target_risk_usd: float = 1000.0) -> float:
    """
    Determines sizing adjusted by asset volatility.
    Sizing = target_risk_usd / vol
    """
    if vol <= 0:
        return target_risk_usd
    raw_size = target_risk_usd / vol
    # Cap size at 50% of available capital to prevent over-allocation on low volatility assets
    return float(np.clip(raw_size, 0.0, capital * 0.5))

def position_size_sentiment_adjusted(base_size: float, sentiment_score: float) -> float:
    """
    Adjusts standard allocation sizes dynamically based on daily Fear & Greed index.
    In Extreme Fear (score < 25), double the base size (Panic Accumulation).
    In Extreme Greed (score > 75), slash the base size to 25% (FOMO Risk Prevention).
    """
    if sentiment_score < 25:
        # Scale up
        multiplier = 2.0
    elif sentiment_score < 45:
        # Scale up slightly
        multiplier = 1.25
    elif sentiment_score > 75:
        # Heavy risk scaling down
        multiplier = 0.25
    elif sentiment_score > 55:
        # Scale down slightly
        multiplier = 0.75
    else:
        multiplier = 1.0
        
    return base_size * multiplier

def run_portfolio_simulation(df_trades: pd.DataFrame, initial_capital: float = 100000.0) -> pd.DataFrame:
    """
    Simulates equity curves across three major position sizing regimes:
    1. Fixed Dollar Size (Static baseline)
    2. Fixed Fractional (2% capital allocation)
    3. Sentiment-Adjusted Dynamic Kelly allocation
    
    Sorts trades by execution timestamp to realistically model bankroll growth.
    """
    df_closing = df_trades[df_trades['is_closing_trade'] == True].copy()
    if df_closing.empty:
        return pd.DataFrame()
        
    # Sort chronologically to preserve feedback loops in portfolio simulation
    if 'datetime_ist' in df_closing.columns:
        df_closing['sort_time'] = pd.to_datetime(df_closing['datetime_ist'])
    else:
        df_closing['sort_time'] = pd.to_datetime(df_closing['Timestamp IST'], format='%d-%m-%Y %H:%M', errors='coerce')
        
    df_closing = df_closing.dropna(subset=['sort_time']).sort_values('sort_time').reset_index(drop=True)
    
    # Calculate historical stats for Kelly calculation
    wins = df_closing[df_closing['win_flag'] == 1]
    losses = df_closing[df_closing['win_flag'] == 0]
    
    win_rate = len(wins) / len(df_closing) if len(df_closing) > 0 else 0.5
    avg_win = wins['Closed PnL'].mean() if len(wins) > 0 else 100.0
    avg_loss = abs(losses['Closed PnL'].mean()) if len(losses) > 0 else 100.0
    
    kelly_fraction = calculate_kelly_fraction(win_rate, avg_win, avg_loss)
    if kelly_fraction <= 0:
        kelly_fraction = 0.05 # Default fallback
        
    # Capital tracking variables
    cap_baseline = initial_capital
    cap_fractional = initial_capital
    cap_dynamic = initial_capital
    
    baseline_equity = []
    fractional_equity = []
    dynamic_equity = []
    timestamps = []
    
    # Single run simulation loop
    for i, row in df_closing.iterrows():
        pnl_pct = row['pnl_pct']
        sentiment = row['sentiment_score'] if 'sentiment_score' in row else 50
        leverage = row['leverage'] if 'leverage' in row else 1
        
        # 1. Baseline: Fixed $5000 notional size
        notional_base = 5000.0
        trade_pnl_base = notional_base * pnl_pct * leverage
        cap_baseline += trade_pnl_base
        
        # 2. Fixed Fractional: 2% of current balance
        notional_frac = cap_fractional * 0.02
        trade_pnl_frac = notional_frac * pnl_pct * leverage
        cap_fractional += trade_pnl_frac
        
        # 3. Dynamic Sentiment + Kelly Allocation
        # Base fractional allocation is dictated by Kelly fraction
        frac = position_size_sentiment_adjusted(kelly_fraction, sentiment)
        notional_dyn = cap_dynamic * frac
        trade_pnl_dyn = notional_dyn * pnl_pct * leverage
        cap_dynamic += trade_pnl_dyn
        
        # Record states
        baseline_equity.append(cap_baseline)
        fractional_equity.append(cap_fractional)
        dynamic_equity.append(cap_dynamic)
        timestamps.append(row['sort_time'])
        
    res_df = pd.DataFrame({
        'timestamp': timestamps,
        'Baseline_Fixed': baseline_equity,
        'Fixed_Fractional_2pct': fractional_equity,
        'Dynamic_Sentiment_Kelly': dynamic_equity
    })
    
    return res_df
