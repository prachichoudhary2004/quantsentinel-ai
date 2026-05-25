import os
import sys
import pandas as pd
import numpy as np

# Adjust sys.path robustly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def analyze_trader_behavior(df_trades: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes historical realized trades chronologically per wallet Account
    and flags standard behavioral biases:
    - FOMO (Greed-chasing sizing/leverage)
    - Panic Selling (Fear-driven panic exits)
    - Overconfidence (Sizing escalations post winning streaks)
    - Loss Chasing (Martingale doubling post-loss exits)
    
    Generates a Combined Behavioral Risk Score (0-100) and maps accounts
    to 6 distinct professional trader archetypes.
    """
    df_closing = df_trades[df_trades['is_closing_trade'] == True].copy()
    if df_closing.empty:
        return pd.DataFrame()
        
    # Sort chronologically to preserve trading sequences
    if 'datetime_ist' in df_closing.columns:
        df_closing['sort_time'] = pd.to_datetime(df_closing['datetime_ist'])
    else:
        df_closing['sort_time'] = pd.to_datetime(df_closing['Timestamp IST'], format='%d-%m-%Y %H:%M', errors='coerce')
        
    df_closing = df_closing.dropna(subset=['sort_time']).sort_values('sort_time').reset_index(drop=True)
    
    # Track metrics per account
    unique_accounts = df_closing['Account'].unique()
    behavioral_records = []
    
    for account in unique_accounts:
        df_acc = df_closing[df_closing['Account'] == account].copy()
        if len(df_acc) < 5:
            continue # Skip low-activity wallets for statistical relevance
            
        # Baseline averages
        avg_leverage_base = df_acc['leverage'].mean()
        avg_size_base = df_acc['Size USD'].mean()
        avg_loss_base = abs(df_acc[df_acc['Closed PnL'] < 0]['Closed PnL'].mean()) if len(df_acc[df_acc['Closed PnL'] < 0]) > 0 else 10.0
        
        # 1. FOMO Detection
        # Sizing/leverage are significantly larger when sentiment index is Greed/Extreme Greed (>55)
        df_greed = df_acc[df_acc['sentiment_score'] > 55]
        fomo_flag = 0
        if len(df_greed) > 2:
            avg_leverage_greed = df_greed['leverage'].mean()
            avg_size_greed = df_greed['Size USD'].mean()
            if (avg_leverage_greed / avg_leverage_base > 1.2) or (avg_size_greed / avg_size_base > 1.2):
                fomo_flag = 1
                
        # 2. Panic Selling Detection
        # Win rate is sub-optimal and average loss is escalated during Fear/Extreme Fear (<40)
        df_fear = df_acc[df_acc['sentiment_score'] < 40]
        panic_flag = 0
        if len(df_fear) > 2:
            win_rate_fear = df_fear['win_flag'].mean()
            avg_loss_fear = abs(df_fear[df_fear['Closed PnL'] < 0]['Closed PnL'].mean()) if len(df_fear[df_fear['Closed PnL'] < 0]) > 0 else 0.0
            if (win_rate_fear < 0.35) and (avg_loss_fear / avg_loss_base > 1.2):
                panic_flag = 1
                
        # 3. Overconfidence Detection
        # Sizing or leverage is scaled up immediately following a winning streak (3+ wins)
        overconfidence_flag = 0
        consecutive_wins = 0
        for idx in range(len(df_acc)):
            row = df_acc.iloc[idx]
            if row['win_flag'] == 1:
                consecutive_wins += 1
            else:
                consecutive_wins = 0
                
            # If we just reached a streak, inspect next trade if available
            if consecutive_wins >= 3 and idx < len(df_acc) - 1:
                next_trade = df_acc.iloc[idx + 1]
                if (next_trade['Size USD'] / avg_size_base > 1.25) or (next_trade['leverage'] / avg_leverage_base > 1.25):
                    overconfidence_flag = 1
                    break
                    
        # 4. Loss Chasing Detection
        # Size/leverage increases immediately after a realized loss within a tight spacing
        loss_chasing_flag = 0
        for idx in range(len(df_acc) - 1):
            row = df_acc.iloc[idx]
            if row['win_flag'] == 0: # realized loss
                next_trade = df_acc.iloc[idx + 1]
                time_diff = (next_trade['sort_time'] - row['sort_time']).total_seconds() / 3600.0 # spacing hours
                
                # Double down or increase leverage within 24 hours
                if time_diff <= 24.0:
                    if (next_trade['Size USD'] > row['Size USD'] * 1.2) or (next_trade['leverage'] > row['leverage']):
                        loss_chasing_flag = 1
                        break
                        
        # 5. Combined Behavioral Risk Score (0-100)
        # Linear weighting: 15% FOMO, 25% Panic, 30% Overconfidence, 30% Loss Chasing
        # Normal baseline is 10 (no bias), scales up to 95+ (all biases flagged)
        score = 10.0 + (15.0 * fomo_flag) + (25.0 * panic_flag) + (30.0 * overconfidence_flag) + (30.0 * loss_chasing_flag)
        risk_score = float(np.clip(score, 0.0, 100.0))
        
        # 6. Trader Archetype Mapping
        # Map accounts based on risk profiles, leverage baseline, and total returns
        total_pnl = df_acc['Closed PnL'].sum()
        win_rate = df_acc['win_flag'].mean()
        
        # Calculate profit factor
        pos_p = df_acc[df_acc['Closed PnL'] > 0]['Closed PnL'].sum()
        neg_p = abs(df_acc[df_acc['Closed PnL'] < 0]['Closed PnL'].sum())
        profit_factor = pos_p / neg_p if neg_p > 0 else 1.0
        
        if avg_leverage_base <= 3.0 and profit_factor >= 1.8:
            archetype = "Institutional Trader"
        elif avg_leverage_base >= 15.0 and total_pnl < 0:
            archetype = "High-Leverage Speculator"
        elif fomo_flag == 1 and total_pnl > 0:
            archetype = "Momentum Chaser"
        elif panic_flag == 1 and win_rate < 0.40:
            archetype = "Fear-Driven Trader"
        elif len(df_acc[df_acc['sentiment_score'] < 40]) > len(df_acc) * 0.4 and profit_factor >= 1.5:
            archetype = "Contrarian Trader"
        else:
            archetype = "Systematic Trader"
            
        behavioral_records.append({
            'Account': account,
            'fomo_bias': fomo_flag,
            'panic_bias': panic_flag,
            'overconfidence_bias': overconfidence_flag,
            'loss_chasing_bias': loss_chasing_flag,
            'behavioral_risk_score': risk_score,
            'trader_archetype': archetype,
            'total_realized_pnl': total_pnl,
            'win_rate': win_rate,
            'avg_leverage': avg_leverage_base,
            'avg_trade_size': avg_size_base,
            'trade_count': len(df_acc)
        })
        
    df_behavior = pd.DataFrame(behavioral_records)
    
    # Save Gold output
    os.makedirs("data/gold/sentiment", exist_ok=True)
    os.makedirs("data/results", exist_ok=True)
    df_behavior.to_csv("data/results/trader_behavioral_personas.csv", index=False)
    
    print(f"Behavioral finance analysis completed! Personas database successfully saved.")
    print(df_behavior['trader_archetype'].value_counts())
    
    return df_behavior

if __name__ == "__main__":
    silver_path = "data/silver/merged_trader_data.csv"
    if os.path.exists(silver_path):
        df = pd.read_csv(silver_path)
        analyze_trader_behavior(df)
