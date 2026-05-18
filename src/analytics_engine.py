import os
import pandas as pd
import numpy as np

def run_quantitative_analysis():
    print("Loading processed dataset...")
    df = pd.read_csv("data/processed/merged_trader_data.csv")
    print(f"Loaded {len(df)} records. Starting quant analysis...")
    
    # Filter for realized/closing trades for PnL & Win Rate calculations
    df_realized = df[df['is_closing_trade'] == True].copy()
    print(f"Analyzing {len(df_realized)} closing/realized trades...")
    
    os.makedirs("data/results", exist_ok=True)
    
    # -------------------------------------------------------------
    # 1. Profitability & Win Rate Analysis by Sentiment Class
    # -------------------------------------------------------------
    print("Analyzing Profitability & Win Rates by Sentiment...")
    
    def profit_factor(x):
        pos_pnl = x[x > 0].sum()
        neg_pnl = x[x < 0].sum()
        if neg_pnl == 0:
            return np.nan if pos_pnl == 0 else np.inf
        return pos_pnl / abs(neg_pnl)

    sentiment_perf = df_realized.groupby('sentiment_class').agg(
        total_pnl=('Closed PnL', 'sum'),
        avg_pnl=('Closed PnL', 'mean'),
        median_pnl=('Closed PnL', 'median'),
        std_pnl=('Closed PnL', 'std'),
        win_rate=('win_flag', 'mean'),
        total_volume=('Size USD', 'sum'),
        avg_volume=('Size USD', 'mean'),
        avg_roe=('roe', 'mean'),
        trade_count=('Closed PnL', 'count'),
        profit_factor=('Closed PnL', profit_factor)
    ).reset_index()
    
    # Reorder sentiment classes for logical plotting later
    sentiment_order = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
    sentiment_perf['sentiment_class'] = pd.Categorical(
        sentiment_perf['sentiment_class'], categories=sentiment_order, ordered=True
    )
    sentiment_perf = sentiment_perf.sort_values('sentiment_class').reset_index(drop=True)
    sentiment_perf.to_csv("data/results/sentiment_performance.csv", index=False)
    
    # -------------------------------------------------------------
    # 2. Leverage Behavior by Sentiment
    # -------------------------------------------------------------
    print("Analyzing Leverage Behavior by Sentiment...")
    # Leverage can be analyzed on ALL trades (opening and closing) to see behavior
    leverage_by_sentiment = df.groupby('sentiment_class').agg(
        avg_leverage=('leverage', 'mean'),
        median_leverage=('leverage', 'median'),
        max_leverage=('leverage', 'max'),
        total_trades=('leverage', 'count')
    ).reset_index()
    
    leverage_by_sentiment['sentiment_class'] = pd.Categorical(
        leverage_by_sentiment['sentiment_class'], categories=sentiment_order, ordered=True
    )
    leverage_by_sentiment = leverage_by_sentiment.sort_values('sentiment_class').reset_index(drop=True)
    leverage_by_sentiment.to_csv("data/results/leverage_by_sentiment.csv", index=False)
    
    # Leverage Bucket Distribution by Sentiment
    lev_dist = df.groupby(['sentiment_class', 'leverage_bucket']).size().unstack(fill_value=0)
    lev_dist_pct = lev_dist.div(lev_dist.sum(axis=1), axis=0) * 100
    lev_dist_pct = lev_dist_pct.reindex(sentiment_order).reset_index()
    lev_dist_pct.to_csv("data/results/leverage_distribution.csv", index=False)
    
    # -------------------------------------------------------------
    # 3. Trading Frequency & Position Sizing
    # -------------------------------------------------------------
    print("Analyzing Trading Frequency & Sizing...")
    # Unique days of each sentiment class in the entire dataset
    days_per_sentiment = df.groupby('sentiment_class')['date'].nunique()
    
    freq_sizing = df.groupby('sentiment_class').agg(
        total_trades=('Trade ID', 'count'),
        total_volume=('Size USD', 'sum'),
        avg_trade_size=('Size USD', 'mean'),
        median_trade_size=('Size USD', 'median')
    ).reset_index()
    
    # Map the unique days to compute average trades per day
    freq_sizing['unique_days'] = freq_sizing['sentiment_class'].map(days_per_sentiment)
    freq_sizing['avg_trades_per_day'] = freq_sizing['total_trades'] / freq_sizing['unique_days']
    freq_sizing['avg_volume_per_day'] = freq_sizing['total_volume'] / freq_sizing['unique_days']
    
    freq_sizing['sentiment_class'] = pd.Categorical(
        freq_sizing['sentiment_class'], categories=sentiment_order, ordered=True
    )
    freq_sizing = freq_sizing.sort_values('sentiment_class').reset_index(drop=True)
    freq_sizing.to_csv("data/results/frequency_sizing.csv", index=False)
    
    # -------------------------------------------------------------
    # 4. Symbol-Wise Performance
    # -------------------------------------------------------------
    print("Analyzing Symbol-Wise Performance...")
    # Group by Coin and Sentiment to see where the alpha is
    coin_perf = df_realized.groupby(['Coin', 'sentiment_class']).agg(
        total_pnl=('Closed PnL', 'sum'),
        avg_pnl=('Closed PnL', 'mean'),
        win_rate=('win_flag', 'mean'),
        trade_count=('Closed PnL', 'count'),
        volume=('Size USD', 'sum')
    ).reset_index()
    coin_perf.to_csv("data/results/coin_performance_by_sentiment.csv", index=False)
    
    # Overall Coin Performance
    coin_overall = df_realized.groupby('Coin').agg(
        total_pnl=('Closed PnL', 'sum'),
        avg_pnl=('Closed PnL', 'mean'),
        win_rate=('win_flag', 'mean'),
        trade_count=('Closed PnL', 'count'),
        volume=('Size USD', 'sum'),
        profit_factor=('Closed PnL', profit_factor)
    ).reset_index().sort_values('total_pnl', ascending=False)
    coin_overall.to_csv("data/results/coin_performance_overall.csv", index=False)
    
    # -------------------------------------------------------------
    # 5. Risk Analysis & Extreme Loss Patterns
    # -------------------------------------------------------------
    print("Analyzing Risk & Extreme Losses...")
    # Value at Risk (VaR) on realized PnL
    risk_stats = []
    for sent in sentiment_order:
        df_sent = df_realized[df_realized['sentiment_class'] == sent]
        if len(df_sent) > 0:
            var_95 = np.percentile(df_sent['Closed PnL'], 5) # 5th percentile is VaR 95%
            var_99 = np.percentile(df_sent['Closed PnL'], 1) # 1st percentile is VaR 99%
            worst_loss = df_sent['Closed PnL'].min()
            avg_loss = df_sent[df_sent['Closed PnL'] < 0]['Closed PnL'].mean()
            risk_stats.append({
                'sentiment_class': sent,
                'var_95': var_95,
                'var_99': var_99,
                'worst_loss': worst_loss,
                'avg_loss': avg_loss,
                'num_losses': (df_sent['Closed PnL'] < 0).sum(),
                'loss_ratio': (df_sent['Closed PnL'] < 0).sum() / len(df_sent)
            })
            
    df_risk = pd.DataFrame(risk_stats)
    df_risk.to_csv("data/results/risk_analysis.csv", index=False)
    
    # Top 20 Extreme Losses
    extreme_losses = df_realized.sort_values('Closed PnL', ascending=True).head(20)[
        ['Account', 'Coin', 'Execution Price', 'Size USD', 'Timestamp IST', 'Direction', 'Closed PnL', 'leverage', 'leverage_bucket', 'sentiment_class', 'sentiment_score']
    ]
    extreme_losses.to_csv("data/results/extreme_losses.csv", index=False)
    
    # -------------------------------------------------------------
    # 6. Correlation Analysis: Leverage vs. PnL
    # -------------------------------------------------------------
    print("Analyzing Leverage vs. PnL Correlation...")
    
    corr_data = []
    # Overall correlation
    corr_pnl = df_realized['leverage'].corr(df_realized['Closed PnL'])
    corr_roe = df_realized['leverage'].corr(df_realized['roe'])
    corr_pnl_pct = df_realized['leverage'].corr(df_realized['pnl_pct'])
    
    corr_data.append({
        'scope': 'Overall',
        'corr_leverage_pnl': corr_pnl,
        'corr_leverage_roe': corr_roe,
        'corr_leverage_pnl_pct': corr_pnl_pct,
        'sample_size': len(df_realized)
    })
    
    # Correlation by Sentiment
    for sent in sentiment_order:
        df_sent = df_realized[df_realized['sentiment_class'] == sent]
        if len(df_sent) > 5:
            corr_data.append({
                'scope': sent,
                'corr_leverage_pnl': df_sent['leverage'].corr(df_sent['Closed PnL']),
                'corr_leverage_roe': df_sent['leverage'].corr(df_sent['roe']),
                'corr_leverage_pnl_pct': df_sent['leverage'].corr(df_sent['pnl_pct']),
                'sample_size': len(df_sent)
            })
            
    df_corr = pd.DataFrame(corr_data)
    df_corr.to_csv("data/results/leverage_correlation.csv", index=False)
    
    # Leverage Binned Performance
    leverage_bins = df_realized.groupby('leverage_bucket').agg(
        avg_pnl=('Closed PnL', 'mean'),
        median_pnl=('Closed PnL', 'median'),
        avg_roe=('roe', 'mean'),
        win_rate=('win_flag', 'mean'),
        total_pnl=('Closed PnL', 'sum'),
        trade_count=('Closed PnL', 'count')
    ).reset_index()
    leverage_bins.to_csv("data/results/leverage_binned_performance.csv", index=False)
    
    # -------------------------------------------------------------
    # 7. Trader Behavioral & Psychology Insights
    # -------------------------------------------------------------
    print("Analyzing Trader Behavioral & Psychology Insights...")
    
    # Group by Account and Sentiment to find biases
    trader_behavior = df_realized.groupby(['Account', 'sentiment_class']).agg(
        total_pnl=('Closed PnL', 'sum'),
        avg_pnl=('Closed PnL', 'mean'),
        win_rate=('win_flag', 'mean'),
        avg_leverage=('leverage', 'mean'),
        avg_trade_size=('Size USD', 'mean'),
        trade_count=('Closed PnL', 'count'),
        profit_factor=('Closed PnL', profit_factor)
    ).reset_index()
    trader_behavior.to_csv("data/results/trader_behavior_by_sentiment.csv", index=False)
    
    # Overall Trader Performance
    trader_overall = df_realized.groupby('Account').agg(
        total_pnl=('Closed PnL', 'sum'),
        avg_pnl=('Closed PnL', 'mean'),
        win_rate=('win_flag', 'mean'),
        avg_leverage=('leverage', 'mean'),
        avg_trade_size=('Size USD', 'mean'),
        trade_count=('Closed PnL', 'count'),
        profit_factor=('Closed PnL', profit_factor)
    ).reset_index().sort_values('total_pnl', ascending=False)
    trader_overall.to_csv("data/results/trader_overall.csv", index=False)
    
    print("Quantitative analysis complete! All summary data saved in data/results/")

if __name__ == "__main__":
    run_quantitative_analysis()
