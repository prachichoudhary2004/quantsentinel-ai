import os
import sys
import unittest
import pandas as pd
import numpy as np

# Adjust path robustly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.risk_analytics import (
    calculate_volatility,
    calculate_downside_deviation,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_var,
    calculate_cvar,
    calculate_rolling_var,
    generate_risk_metrics_report
)

from src.portfolio_analytics import (
    calculate_kelly_fraction,
    position_size_fixed_fractional,
    position_size_vol_adjusted,
    position_size_sentiment_adjusted,
    run_portfolio_simulation
)

from src.regime_detection import (
    detect_market_regimes,
    classify_regime_rule_based
)

from src.behavioral_intelligence import (
    analyze_trader_behavior
)

from src.data_quality import (
    run_null_checks,
    run_duplicate_checks,
    run_schema_checks,
    detect_outliers_zscore,
    monitor_sentiment_drift
)

from src.shap_explainer import (
    CryptoSHAPExplainer
)

class TestPlatformAnalytics(unittest.TestCase):
    
    def setUp(self):
        """
        Setup synthetic trade data mimicking Hyperliquid processed silver schema
        """
        # Create a series of returns
        self.returns = pd.Series([0.01, 0.02, -0.015, 0.03, -0.005, 0.01, 0.02, -0.01, 0.015, 0.025])
        
        # Create a df of synthetic trades
        self.df_trades = pd.DataFrame({
            'Trade ID': range(100, 110),
            'Account': ['Account_A'] * 5 + ['Account_B'] * 5,
            'Coin': ['SOL', 'SOL', 'SOL', 'BTC', 'BTC'] * 2,
            'Closed PnL': [100.0, 200.0, -150.0, 300.0, -50.0, 100.0, 200.0, -100.0, 150.0, 250.0],
            'Size USD': [5000.0, 10000.0, 10000.0, 10000.0, 5000.0, 5000.0, 10000.0, 10000.0, 10000.0, 10000.0],
            'leverage': [5, 5, 5, 10, 10, 5, 5, 5, 10, 10],
            'sentiment_score': [45, 60, 65, 30, 20, 45, 60, 65, 30, 20],
            'sentiment_class': ['Neutral', 'Greed', 'Greed', 'Fear', 'Extreme Fear'] * 2,
            'trade_direction': ['Long', 'Long', 'Short', 'Short', 'Long'] * 2,
            'Timestamp IST': ['25-05-2026 10:00'] * 10,
            'is_closing_trade': [True] * 10,
            'win_flag': [1, 1, 0, 1, 0, 1, 1, 0, 1, 1],
            'roe': [0.1, 0.1, -0.075, 0.3, -0.1, 0.1, 0.1, -0.05, 0.15, 0.25],
            'pnl_pct': [0.02, 0.02, -0.015, 0.03, -0.01, 0.02, 0.02, -0.01, 0.015, 0.025],
            'date': ['2026-05-25'] * 10
        })

    # -------------------------------------------------------------
    # 1. Test Advanced Risk Engine (risk_analytics.py)
    # -------------------------------------------------------------
    def test_risk_metrics_math(self):
        # Volatility
        vol = calculate_volatility(self.returns)
        self.assertGreater(vol, 0.0)
        
        # Downside Deviation
        dd = calculate_downside_deviation(self.returns)
        self.assertGreater(dd, 0.0)
        
        # Max Drawdown
        cum_ret = self.returns.cumsum()
        mdd = calculate_max_drawdown(cum_ret)
        self.assertLessEqual(mdd, 0.0)
        
        # Sharpe & Sortino
        sharpe = calculate_sharpe_ratio(self.returns, annualization_factor=1)
        sortino = calculate_sortino_ratio(self.returns, annualization_factor=1)
        self.assertNotEqual(sharpe, 0.0)
        self.assertNotEqual(sortino, 0.0)
        
        # Calmar
        calmar = calculate_calmar_ratio(0.15, -0.05)
        self.assertAlmostEqual(calmar, 3.0)
        
        # VaR and CVaR
        pnl_series = self.df_trades['Closed PnL']
        var_val = calculate_var(pnl_series, 0.95)
        cvar_val = calculate_cvar(pnl_series, 0.95)
        self.assertGreaterEqual(var_val, cvar_val)
        
        # Rolling VaR
        rolling = calculate_rolling_var(pnl_series, window=5)
        self.assertEqual(len(rolling), len(pnl_series))
        
        # Risk report generation
        df_risk = generate_risk_metrics_report(self.df_trades, group_col='sentiment_class')
        self.assertFalse(df_risk.empty)
        self.assertIn('sharpe_ratio', df_risk.columns)

    # -------------------------------------------------------------
    # 2. Test Portfolio Analytics Engine (portfolio_analytics.py)
    # -------------------------------------------------------------
    def test_portfolio_sizing(self):
        # Kelly Criterion
        f_star = calculate_kelly_fraction(0.6, 200.0, 100.0)
        self.assertGreater(f_star, 0.0)
        self.assertLessEqual(f_star, 0.5)
        
        # Fixed Fractional
        size_frac = position_size_fixed_fractional(100000.0, 0.02)
        self.assertEqual(size_frac, 2000.0)
        
        # Vol Adjusted Sizing
        size_vol = position_size_vol_adjusted(100000.0, 0.15, 1000.0)
        self.assertLessEqual(size_vol, 50000.0)
        
        # Sentiment Adjusted Sizing
        size_sent_fear = position_size_sentiment_adjusted(5000.0, 20) # Fear
        size_sent_greed = position_size_sentiment_adjusted(5000.0, 80) # Greed
        self.assertGreater(size_sent_fear, 5000.0)
        self.assertLess(size_sent_greed, 5000.0)
        
        # Portfolio simulator run
        df_sim = run_portfolio_simulation(self.df_trades)
        self.assertFalse(df_sim.empty)
        self.assertIn('Dynamic_Sentiment_Kelly', df_sim.columns)

    # -------------------------------------------------------------
    # 3. Test Market Regime Detection (regime_detection.py)
    # -------------------------------------------------------------
    def test_regime_classification(self):
        # Regime rule based mapping
        row = {'sentiment_score': 85, 'rolling_volatility': 0.05, 'avg_roe': 0.02}
        regime = classify_regime_rule_based(row)
        self.assertEqual(regime, "Bull Market")
        
        # End-to-end regime execution
        df_reg = detect_market_regimes(self.df_trades)
        self.assertFalse(df_reg.empty)
        self.assertIn('market_regime', df_reg.columns)

    # -------------------------------------------------------------
    # 4. Test Behavioral Finance Engine (behavioral_intelligence.py)
    # -------------------------------------------------------------
    def test_behavioral_intelligence(self):
        df_behavioral = analyze_trader_behavior(self.df_trades)
        self.assertFalse(df_behavioral.empty)
        self.assertIn('behavioral_risk_score', df_behavioral.columns)
        self.assertIn('trader_archetype', df_behavioral.columns)

    # -------------------------------------------------------------
    # 5. Test Data Quality & Schema Integrity (data_quality.py)
    # -------------------------------------------------------------
    def test_data_quality_suite(self):
        # Null check
        nulls = run_null_checks(self.df_trades, ['Account', 'Closed PnL'])
        self.assertTrue(nulls['passed'])
        
        # Duplicate check
        dups = run_duplicate_checks(self.df_trades, 'Trade ID')
        self.assertTrue(dups['passed'])
        
        # Schema check
        expected = {'Account': 'string', 'Closed PnL': 'float', 'leverage': 'int'}
        schema = run_schema_checks(self.df_trades, expected)
        self.assertTrue(schema['passed'])
        
        # Outlier check
        outliers = detect_outliers_zscore(self.df_trades, 'Closed PnL')
        self.assertEqual(outliers['outlier_count'], 0)
        
        # Drift monitor
        df_drift = pd.DataFrame({'sentiment_score': [50] * 100 + [85] * 50})
        drift = monitor_sentiment_drift(df_drift)
        self.assertTrue(drift['drift_detected'])

    # -------------------------------------------------------------
    # 6. Test Explainable AI local explainer (shap_explainer.py)
    # -------------------------------------------------------------
    def test_shap_explainer_fallback(self):
        explainer = CryptoSHAPExplainer(model_path="data/results/profitable_trade_model.pkl")
        df_global = explainer.get_global_importance()
        self.assertFalse(df_global.empty)
        
        # Local explainer fallback checks
        dummy_row = pd.DataFrame([[5, 5000.0, 50.0, 1, 0, 0, 0, 0, 0, 0]], columns=df_global['feature'].tolist()[:10])
        df_local = explainer.explain_local_prediction(dummy_row)
        self.assertFalse(df_local.empty)
        self.assertIn('shap_value', df_local.columns)

if __name__ == "__main__":
    unittest.main()
