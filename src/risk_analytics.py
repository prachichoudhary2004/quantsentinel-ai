import numpy as np
import pandas as pd

def calculate_volatility(returns: pd.Series) -> float:
    """
    Calculates standard deviation of returns (volatility).
    """
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1))

def calculate_downside_deviation(returns: pd.Series, target_return: float = 0.0) -> float:
    """
    Calculates downside deviation (volatility of negative returns only)
    for the Sortino ratio.
    """
    if len(returns) < 2:
        return 0.0
    excess_returns = returns - target_return
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) < 2:
        return 0.0
    return float(np.sqrt(np.sum(downside_returns**2) / (len(returns) - 1)))

def calculate_max_drawdown(cumulative_returns: pd.Series) -> float:
    """
    Calculates maximum drawdown of a cumulative return series or equity curve.
    Formula: Max peak-to-trough drop.
    """
    if len(cumulative_returns) == 0:
        return 0.0
    peaks = cumulative_returns.cummax()
    drawdowns = (cumulative_returns - peaks)
    # Return absolute drawdown or negative value
    return float(drawdowns.min())

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate_daily: float = 0.0, annualization_factor: float = 365) -> float:
    """
    Calculates annualized Sharpe Ratio.
    """
    if len(returns) < 2:
        return 0.0
    mean_excess_return = returns.mean() - risk_free_rate_daily
    vol = calculate_volatility(returns)
    if vol == 0:
        return 0.0
    return float((mean_excess_return / vol) * np.sqrt(annualization_factor))

def calculate_sortino_ratio(returns: pd.Series, target_return_daily: float = 0.0, risk_free_rate_daily: float = 0.0, annualization_factor: float = 365) -> float:
    """
    Calculates annualized Sortino Ratio.
    """
    if len(returns) < 2:
        return 0.0
    mean_excess_return = returns.mean() - risk_free_rate_daily
    downside_dev = calculate_downside_deviation(returns, target_return_daily)
    if downside_dev == 0:
        return 0.0
    return float((mean_excess_return / downside_dev) * np.sqrt(annualization_factor))

def calculate_calmar_ratio(annualized_return: float, max_drawdown: float) -> float:
    """
    Calculates Calmar Ratio.
    Calmar = Annualized Return / Max Drawdown (absolute).
    """
    abs_dd = abs(max_drawdown)
    if abs_dd == 0:
        return 0.0
    return float(annualized_return / abs_dd)

def calculate_var(pnl: pd.Series, alpha: float = 0.95) -> float:
    """
    Calculates Value at Risk (VaR) using historical simulation method.
    Returns the absolute dollar downside boundary.
    e.g. if alpha=0.95, returns the 5th percentile.
    """
    if len(pnl) == 0:
        return 0.0
    percentile = (1 - alpha) * 100
    return float(np.percentile(pnl, percentile))

def calculate_cvar(pnl: pd.Series, alpha: float = 0.95) -> float:
    """
    Calculates Conditional Value at Risk (CVaR) - Expected Shortfall.
    Returns average of losses exceeding the VaR boundary.
    """
    if len(pnl) == 0:
        return 0.0
    var_val = calculate_var(pnl, alpha)
    losses_beyond_var = pnl[pnl <= var_val]
    if len(losses_beyond_var) == 0:
        return var_val
    return float(losses_beyond_var.mean())

def calculate_rolling_var(pnl: pd.Series, window: int = 30, alpha: float = 0.95) -> pd.Series:
    """
    Calculates rolling Value at Risk (VaR) historical simulation.
    """
    if len(pnl) < window:
        return pd.Series([calculate_var(pnl, alpha)] * len(pnl), index=pnl.index)
    return pnl.rolling(window=window).apply(lambda x: calculate_var(x, alpha=alpha))

def generate_risk_metrics_report(df_trades: pd.DataFrame, group_col: str = 'sentiment_class') -> pd.DataFrame:
    """
    Generates a full risk report grouped by a column (e.g. market sentiment state).
    """
    # Standardize trade data
    df_realized = df_trades[df_trades['is_closing_trade'] == True].copy()
    if df_realized.empty:
        return pd.DataFrame()
        
    unique_days = df_realized['date'].nunique() if 'date' in df_realized.columns else 1
    annualization_factor = (365 / unique_days) if unique_days > 0 else 365
    
    risk_report = []
    
    # Calculate group overall performance
    groups = df_realized[group_col].unique() if group_col in df_realized.columns else ['Overall']
    
    for g in groups:
        if group_col in df_realized.columns:
            df_g = df_realized[df_realized[group_col] == g]
        else:
            df_g = df_realized
            
        pnl = df_g['Closed PnL']
        roe = df_g['roe'] if 'roe' in df_g.columns else pnl
        
        # Cumulative returns
        cum_pnl = pnl.cumsum()
        cum_roe = roe.cumsum()
        
        # Risk factors
        vol = calculate_volatility(roe)
        downside_dev = calculate_downside_deviation(roe)
        max_dd_pnl = calculate_max_drawdown(cum_pnl)
        max_dd_roe = calculate_max_drawdown(cum_roe)
        
        # Sharpe, Sortino
        sharpe = calculate_sharpe_ratio(roe, annualization_factor=annualization_factor)
        sortino = calculate_sortino_ratio(roe, annualization_factor=annualization_factor)
        
        # Annualized return (simple approximation based on trading period)
        total_roe = roe.sum()
        annualized_roe = total_roe * (annualization_factor / len(df_realized)) # scaled relative to overall length
        calmar = calculate_calmar_ratio(annualized_roe, max_dd_roe)
        
        # VaR and CVaR (95% & 99%)
        var_95 = calculate_var(pnl, 0.95)
        var_99 = calculate_var(pnl, 0.99)
        cvar_95 = calculate_cvar(pnl, 0.95)
        cvar_99 = calculate_cvar(pnl, 0.99)
        
        risk_report.append({
            group_col: g,
            'volatility': vol,
            'downside_dev': downside_dev,
            'max_drawdown_usd': max_dd_pnl,
            'max_drawdown_roe': max_dd_roe,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'trade_count': len(df_g)
        })
        
    return pd.DataFrame(risk_report)
