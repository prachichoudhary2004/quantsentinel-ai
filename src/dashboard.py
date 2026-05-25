import streamlit as st

# Set page configuration to wide layout and professional fintech theme MUST BE FIRST
st.set_page_config(
    page_title="QuantSentinel AI – Sentiment & Behavioral Trading Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import pickle
import sys
import subprocess

# Import Phase 1 & 3 analytical, explanation, and NLP modules
from src.risk_analytics import generate_risk_metrics_report
from src.shap_explainer import CryptoSHAPExplainer
from src.portfolio_analytics import run_portfolio_simulation
from src.regime_detection import detect_market_regimes

# Auto-Initialization for Medallion & NLP pipelines
if not os.path.exists("data/silver/merged_trader_data.csv") or not os.path.exists("data/results/trader_behavioral_personas.csv"):
    with st.status("Initializing Behavioral Lakehouse Platform...", expanded=True) as status:
        try:
            st.write("Ingesting news feeds (Bronze)...")
            subprocess.run([sys.executable, "src/download_data.py"], check=True)
            subprocess.run([sys.executable, "src/news_ingestion.py"], check=True)
            
            st.write("Executing Medallion Preprocessing transforms (Silver)...")
            subprocess.run([sys.executable, "src/bronze_to_silver.py"], check=True)
            subprocess.run([sys.executable, "src/text_preprocessing.py"], check=True)
            
            st.write("Curating Advanced Risk and Portfolio Simulators (Gold)...")
            subprocess.run([sys.executable, "src/silver_to_gold.py"], check=True)
            
            st.write("Executing FinBERT sentiment engine & unsupervised topic clustering...")
            subprocess.run([sys.executable, "src/sentiment_engine.py"], check=True)
            subprocess.run([sys.executable, "src/topic_modeling.py"], check=True)
            subprocess.run([sys.executable, "src/entity_extraction.py"], check=True)
            subprocess.run([sys.executable, "src/impact_scoring.py"], check=True)
            subprocess.run([sys.executable, "src/sentiment_trends.py"], check=True)
            
            st.write("Executing MLOps Model Comparisons & Behavioral Finance Engine...")
            subprocess.run([sys.executable, "src/ml_engine.py"], check=True)
            
            # Run behavioral finance aggregation
            if os.path.exists("data/silver/merged_trader_data.csv"):
                df_silv = pd.read_csv("data/silver/merged_trader_data.csv")
                from src.behavioral_intelligence import analyze_trader_behavior
                analyze_trader_behavior(df_silv)
                
            status.update(label="✅ Setup complete! Loading terminal...", state="complete", expanded=False)
            st.rerun()
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            st.stop()

# Cache prediction models
@st.cache_resource
def load_prediction_model():
    if os.path.exists("data/results/profitable_trade_model.pkl"):
        with open("data/results/profitable_trade_model.pkl", "rb") as f:
            return pickle.load(f)
    return None, None

rf_model, training_features = load_prediction_model()
explainer = CryptoSHAPExplainer()

# -------------------------------------------------------------
# Data Loading & Caching (Medallion Layers)
# -------------------------------------------------------------
@st.cache_data
def load_silver_data():
    if os.path.exists("data/silver/merged_trader_data.csv"):
        return pd.read_csv("data/silver/merged_trader_data.csv")
    return None

@st.cache_data
def load_trader_segments():
    if os.path.exists("data/results/trader_segments.csv"):
        return pd.read_csv("data/results/trader_segments.csv")
    return None

@st.cache_data
def load_gold_regimes():
    if os.path.exists("data/gold/market_regimes.csv"):
        return pd.read_csv("data/gold/market_regimes.csv")
    return None

@st.cache_data
def load_gold_portfolio_sim():
    if os.path.exists("data/gold/portfolio_simulations.csv"):
        df = pd.read_csv("data/gold/portfolio_simulations.csv")
        # Forward-fill and backfill NaNs to prevent disjointed lines
        df = df.ffill().bfill()
        # Downsample to ~1000 data points to optimize browser Plotly rendering performance
        if len(df) > 1000:
            df = df.iloc[::max(1, len(df) // 1000)].copy()
        return df
    return None

@st.cache_data
def load_gold_sentiment_scores():
    if os.path.exists("data/gold/sentiment/sentiment_scores.csv"):
        return pd.read_csv("data/gold/sentiment/sentiment_scores.csv")
    return None

@st.cache_data
def load_gold_sentiment_trends():
    if os.path.exists("data/gold/sentiment/sentiment_trends.csv"):
        return pd.read_csv("data/gold/sentiment/sentiment_trends.csv")
    return None

@st.cache_data
def load_gold_topics():
    if os.path.exists("data/gold/sentiment/topic_clusters.csv"):
        return pd.read_csv("data/gold/sentiment/topic_clusters.csv")
    return None

@st.cache_data
def load_gold_entities():
    if os.path.exists("data/gold/sentiment/extracted_entities.csv"):
        return pd.read_csv("data/gold/sentiment/extracted_entities.csv")
    return None

@st.cache_data
def load_gold_impact():
    if os.path.exists("data/gold/sentiment/market_impact_scores.csv"):
        return pd.read_csv("data/gold/sentiment/market_impact_scores.csv")
    return None

@st.cache_data
def load_behavioral_personas():
    if os.path.exists("data/results/trader_behavioral_personas.csv"):
        return pd.read_csv("data/results/trader_behavioral_personas.csv")
    return None

df_raw = load_silver_data()
df_segments = load_trader_segments()
df_regimes = load_gold_regimes()
df_portfolio_sim = load_gold_portfolio_sim()

df_nlp_scores = load_gold_sentiment_scores()
df_nlp_trends = load_gold_sentiment_trends()
df_nlp_topics = load_gold_topics()
df_nlp_entities = load_gold_entities()
df_nlp_impact = load_gold_impact()
df_behavioral = load_behavioral_personas()

# Custom CSS
st.markdown("""
<style>
    .reportview-container { background: #0d1117; }
    .main { background-color: #0b0e14; color: #c9d1d9; font-family: 'Inter', sans-serif; }
    h1, h2, h3 { color: #ffffff; font-weight: 700; font-family: 'Space Grotesk', sans-serif; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Sidebar Navigation
# -------------------------------------------------------------
st.sidebar.image("https://cryptologos.cc/logos/bitcoin-btc-logo.png", width=50)
st.sidebar.title("QuantSentinel AI")
st.sidebar.markdown("*Behavioral Finance Platform*")
st.sidebar.divider()

page = st.sidebar.selectbox(
    "Select Intelligence View",
    [
        "📊 Quant Metrics Dashboard",
        "🧠 Behavioral Finance",
        "👥 Trader Personas",
        "🛡️ Risk Intelligence",
        "📰 News Intelligence",
        "🧠 Sentiment Intelligence",
        "💬 Topic Intelligence",
        "👥 Entity Intelligence",
        "💥 Market Impact Dashboard"
    ]
)
st.sidebar.divider()

if df_regimes is not None and not df_regimes.empty:
    current_regime = df_regimes.iloc[-1]['market_regime']
    sentiment_score = df_regimes.iloc[-1]['sentiment_score']
    regime_color = "#39d353" if "Bull" in current_regime else ("#f85149" if "Bear" in current_regime else "#ff7f0e")
    
    st.sidebar.markdown(f"""
    <div style="background-color:#161b22; border:1px solid #30363d; padding:15px; border-radius:8px;">
        <span style="font-size:0.8em; color:#8b949e; text-transform:uppercase; letter-spacing:1px;">Quant Market Regime</span>
        <h3 style="color:{regime_color}; margin-top:5px; margin-bottom:5px;">{current_regime}</h3>
        <span style="font-size:0.85em; color:#c9d1d9;">Sentiment Index: <b>{sentiment_score:.0f}/100</b></span>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# Screen 1: Quant Metrics Dashboard (Existing backward compatible tabs)
# -------------------------------------------------------------
if page == "📊 Quant Metrics Dashboard" and df_raw is not None:
    all_accounts = sorted(df_raw['Account'].unique())
    selected_accounts = st.sidebar.multiselect("Filter Trader Accounts", options=all_accounts, default=[])
    all_coins = sorted(df_raw['Coin'].unique())
    selected_coins = st.sidebar.multiselect("Filter Symbols (Coins)", options=all_coins, default=[])
    all_sentiments = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
    selected_sentiments = st.sidebar.multiselect("Filter Market Sentiment", options=all_sentiments, default=all_sentiments)
    
    df_filtered = df_raw.copy()
    if selected_accounts:
        df_filtered = df_filtered[df_filtered['Account'].isin(selected_accounts)]
    if selected_coins:
        df_filtered = df_filtered[df_filtered['Coin'].isin(selected_coins)]
    if selected_sentiments:
        df_filtered = df_filtered[df_filtered['sentiment_class'].isin(selected_sentiments)]
        
    df_realized = df_filtered[df_filtered['is_closing_trade'] == True].copy()
    
    st.title("⚡ Quant Metrics Intelligence Terminal")
    st.divider()
    
    k_col1, k_col2, k_col3, k_col4, k_col5, k_col6 = st.columns(6)
    total_pnl = df_realized['Closed PnL'].sum()
    win_rate = df_realized['win_flag'].mean() * 100 if len(df_realized) > 0 else 0
    total_trades = len(df_realized)
    total_volume = df_realized['Size USD'].sum()
    avg_leverage = df_filtered['leverage'].mean()
    
    pos_pnl = df_realized[df_realized['Closed PnL'] > 0]['Closed PnL'].sum()
    neg_pnl = df_realized[df_realized['Closed PnL'] < 0]['Closed PnL'].sum()
    profit_factor = pos_pnl / abs(neg_pnl) if neg_pnl != 0 else np.nan
    
    with k_col1: st.metric("Realized PnL (USD)", f"${total_pnl:,.2f}")
    with k_col2: st.metric("Aggregate Win Rate", f"{win_rate:.2f}%")
    with k_col3: st.metric("Profit Factor", f"{profit_factor:.2f}" if pd.notnull(profit_factor) else "N/A")
    with k_col4: st.metric("Total Volume Traded", f"${total_volume/1e6:.2f}M")
    with k_col5: st.metric("Avg Effective Leverage", f"{avg_leverage:.2f}x")
    with k_col6: st.metric("Realized Trades Count", f"{total_trades:,}")
    st.divider()
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Profitability & Sentiment", "⚙️ Leverage & Risk Mechanics", "⏱️ Activity & Sizing", 
        "👥 Trader Segments", "🔮 Algorithmic Predictive Sandbox", "📈 Portfolio Sizing Simulation"
    ])
    
    with tab1:
        st.subheader("Profitability & Win Rates across Market Sentiment States")
        col_t1_l, col_t1_r = st.columns(2)
        with col_t1_l:
            sent_perf = df_realized.groupby('sentiment_class').agg(total_pnl=('Closed PnL', 'sum')).reset_index()
            sent_perf['sentiment_class'] = pd.Categorical(sent_perf['sentiment_class'], categories=all_sentiments, ordered=True)
            sent_perf = sent_perf.sort_values('sentiment_class')
            fig_pnl = px.bar(sent_perf, x='sentiment_class', y='total_pnl', title="Total Realized PnL ($) by Sentiment State", color='total_pnl', color_continuous_scale=px.colors.diverging.RdYlGn, template="plotly_dark")
            st.plotly_chart(fig_pnl, use_container_width=True)
        with col_t1_r:
            sent_perf_wr = df_realized.groupby('sentiment_class').agg(win_rate=('win_flag', 'mean')).reset_index()
            sent_perf_wr['win_rate_pct'] = sent_perf_wr['win_rate'] * 100
            sent_perf_wr['sentiment_class'] = pd.Categorical(sent_perf_wr['sentiment_class'], categories=all_sentiments, ordered=True)
            sent_perf_wr = sent_perf_wr.sort_values('sentiment_class')
            fig_wr = px.line(sent_perf_wr, x='sentiment_class', y='win_rate_pct', title="Aggregate Win Rate (%) by Sentiment State", markers=True, template="plotly_dark")
            st.plotly_chart(fig_wr, use_container_width=True)
            
    with tab2:
        st.subheader("Leverage Mechanics and Advanced Quantitative Risk Analytics")
        df_risk_report = generate_risk_metrics_report(df_filtered, group_col='sentiment_class')
        if not df_risk_report.empty:
            st.dataframe(df_risk_report, use_container_width=True)
            st.divider()
        col_t2_l, col_t2_r = st.columns(2)
        with col_t2_l:
            lev_perf = df_realized.groupby('leverage_bucket').agg(total_pnl=('Closed PnL', 'sum')).reset_index()
            fig_lev_pnl = px.bar(lev_perf, x='leverage_bucket', y='total_pnl', title="Total PnL ($) by Leverage Bucket", color='total_pnl', color_continuous_scale='Tealgrn', template="plotly_dark")
            st.plotly_chart(fig_lev_pnl, use_container_width=True)
        with col_t2_r:
            if not df_risk_report.empty:
                fig_risk = go.Figure(data=[
                    go.Bar(name='VaR 95%', x=df_risk_report['sentiment_class'], y=df_risk_report['var_95'].abs(), marker_color='#ff7f0e'),
                    go.Bar(name='CVaR 95%', x=df_risk_report['sentiment_class'], y=df_risk_report['cvar_95'].abs(), marker_color='#d62728')
                ])
                fig_risk.update_layout(title="Value at Risk vs. Expected Shortfall (CVaR)", template="plotly_dark")
                st.plotly_chart(fig_risk, use_container_width=True)
                
    with tab3:
        st.subheader("Trading Sizing and Frequency Dynamics")
        col_t3_l, col_t3_r = st.columns(2)
        days_per_sent = df_filtered.groupby('sentiment_class')['date'].nunique()
        df_freq = df_filtered.groupby('sentiment_class').agg(total_trades=('Trade ID', 'count'), avg_trade_size=('Size USD', 'mean')).reset_index()
        df_freq['unique_days'] = df_freq['sentiment_class'].map(days_per_sent)
        df_freq['avg_trades_per_day'] = df_freq['total_trades'] / df_freq['unique_days']
        df_freq['sentiment_class'] = pd.Categorical(df_freq['sentiment_class'], categories=all_sentiments, ordered=True)
        df_freq = df_freq.sort_values('sentiment_class')
        with col_t3_l:
            fig_f = px.bar(df_freq, x='sentiment_class', y='avg_trades_per_day', title="Average Trades Per Day", template="plotly_dark")
            st.plotly_chart(fig_f, use_container_width=True)
        with col_t3_r:
            fig_s = px.line(df_freq, x='sentiment_class', y='avg_trade_size', title="Average Position Sizing (USD)", markers=True, template="plotly_dark")
            st.plotly_chart(fig_s, use_container_width=True)
            
    with tab4:
        st.subheader("Algorithmic Trader Segmentation (K-Means Personas)")
        if df_segments is not None:
            col_t4_l, col_t4_r = st.columns(2)
            with col_t4_l:
                seg_counts = df_segments['trader_segment'].value_counts().reset_index()
                seg_counts.columns = ['Segment', 'Count']
                fig_p = px.pie(seg_counts, values='Count', names='Segment', title="Trader Segments Population Distribution", template="plotly_dark")
                st.plotly_chart(fig_p, use_container_width=True)
            with col_t4_r:
                seg_perf = df_segments.groupby('trader_segment').agg(total_pnl=('total_pnl', 'mean')).reset_index()
                fig_bp = px.bar(seg_perf, x='trader_segment', y='total_pnl', title="Average Lifetime Profitability ($) by Segment", template="plotly_dark")
                st.plotly_chart(fig_bp, use_container_width=True)
            st.dataframe(df_segments, use_container_width=True)
            
    with tab5:
        st.subheader("Trade Profitability Predictor Sandbox & SHAP Explainability")
        sandbox_col1, sandbox_col2 = st.columns([1, 1.2])
        with sandbox_col1:
            st.write("### 🎛️ Trade Parameter Terminals")
            input_coin = st.selectbox("Symbol / Coin", options=all_coins, index=0)
            input_direction = st.selectbox("Trade Direction", options=['Long', 'Short'])
            input_size = st.number_input("Trade Size (USD)", min_value=10.0, max_value=1000000.0, value=5000.0)
            input_leverage = st.slider("Leverage Used", min_value=1, max_value=50, value=5)
            input_sentiment = st.slider("Fear & Greed Index Score", min_value=0, max_value=100, value=50)
            input_hour = st.slider("Execution Hour", min_value=0, max_value=23, value=12)
            input_day = st.selectbox("Execution Day", options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
            predict_triggered = st.button("🚀 Run Predictive Classifier")
        with sandbox_col2:
            if predict_triggered and rf_model is not None:
                input_df = pd.DataFrame([{
                    'leverage': float(input_leverage), 'Size USD': float(input_size), 'sentiment_score': float(input_sentiment),
                    'trade_direction': input_direction, 'hour': int(input_hour), 'day_of_week': input_day, 'Coin': input_coin
                }])
                input_encoded = pd.get_dummies(input_df, columns=['trade_direction', 'day_of_week', 'Coin'])
                aligned_data = {col: float(input_encoded[col].iloc[0]) if col in input_encoded.columns else 0.0 for col in training_features}
                X_input = pd.DataFrame([aligned_data])[training_features]
                
                prob_win = float(rf_model.predict_proba(X_input)[0, 1])
                st.metric("Win Probability Score", f"{prob_win*100:.2f}%")
                
                st.subheader("🧬 Local Feature Impact (SHAP Explanation)")
                df_local_shap = explainer.explain_local_prediction(X_input)
                if not df_local_shap.empty:
                    df_local_shap['feature_display'] = df_local_shap['feature'] + " (" + df_local_shap['feature_value'].astype(str) + ")"
                    fig_local = px.bar(df_local_shap.head(8), x='shap_value', y='feature_display', orientation='h', color='shap_value', color_continuous_scale=px.colors.diverging.RdYlGn, template="plotly_dark")
                    st.plotly_chart(fig_local, use_container_width=True)
                    
    with tab6:
        st.subheader("Portfolio Capital Allocation Sizing Simulator")
        if df_portfolio_sim is not None and not df_portfolio_sim.empty:
            fig_sim = px.line(df_portfolio_sim, x='timestamp', y=['Baseline_Fixed', 'Fixed_Fractional_2pct', 'Dynamic_Sentiment_Kelly'], title="Sizing Trajectories ($100k Bankroll)", template="plotly_dark")
            st.plotly_chart(fig_sim, use_container_width=True)

# -------------------------------------------------------------
# Screen 2: Behavioral Finance (NEW - Phase 4)
# -------------------------------------------------------------
elif page == "🧠 Behavioral Finance" and df_behavioral is not None:
    st.title("🧠 Behavioral Finance Intelligence")
    st.markdown("Visualizes dynamic cognitive trading biases (FOMO, Panic Selling, Overconfidence, and Loss Chasing).")
    
    # KPIs for overall cognitive biases
    b_col1, b_col2, b_col3, b_col4 = st.columns(4)
    fomo_count = int(df_behavioral['fomo_bias'].sum())
    panic_count = int(df_behavioral['panic_bias'].sum())
    over_count = int(df_behavioral['overconfidence_bias'].sum())
    chase_count = int(df_behavioral['loss_chasing_bias'].sum())
    
    with b_col1: st.metric("FOMO Biased Accounts", f"{fomo_count} / {len(df_behavioral)}")
    with b_col2: st.metric("Panic Selling Wallets", f"{panic_count} / {len(df_behavioral)}")
    with b_col3: st.metric("Overconfident Traders", f"{over_count} / {len(df_behavioral)}")
    with b_col4: st.metric("Loss Chasing (Martingale)", f"{chase_count} / {len(df_behavioral)}")
    st.divider()
    
    col_beh_l, col_beh_r = st.columns([1, 1.2])
    
    with col_beh_l:
        st.subheader("Biases Prevalence Distribution")
        bias_melted = df_behavioral.melt(value_vars=['fomo_bias', 'panic_bias', 'overconfidence_bias', 'loss_chasing_bias'], var_name='bias_type', value_name='flag')
        bias_aggregates = bias_melted[bias_melted['flag'] == 1]['bias_type'].value_counts().reset_index()
        bias_aggregates.columns = ['Cognitive Bias', 'Wallets Count']
        
        fig_bias_pie = px.pie(bias_aggregates, values='Wallets Count', names='Cognitive Bias', title="Prevalence proportion of biases", template="plotly_dark")
        st.plotly_chart(fig_bias_pie, use_container_width=True)
        
    with col_beh_r:
        st.subheader("Behavioral Risk Scores Ranking (0 - 100)")
        df_sorted_risk = df_behavioral[['Account', 'behavioral_risk_score', 'trader_archetype']].sort_values('behavioral_risk_score', ascending=False)
        fig_risk_bar = px.bar(
            df_sorted_risk.head(15),
            x='behavioral_risk_score',
            y='Account',
            color='trader_archetype',
            orientation='h',
            title="Top 15 Most Biased/Risky Wallets",
            template="plotly_dark"
        )
        st.plotly_chart(fig_risk_bar, use_container_width=True)

# -------------------------------------------------------------
# Screen 3: Trader Personas (NEW - Phase 4)
# -------------------------------------------------------------
elif page == "👥 Trader Personas" and df_behavioral is not None:
    st.title("👥 Trader Personas & Archetypes")
    st.markdown("Details wallet mapping results across the 6 professional quantitative archetypes.")
    
    col_per_l, col_per_r = st.columns([1, 1.2])
    
    with col_per_l:
        st.subheader("Archetypes Population Breakdown")
        archetype_counts = df_behavioral['trader_archetype'].value_counts().reset_index()
        archetype_counts.columns = ['Trader Archetype', 'Count']
        
        fig_arc_pie = px.pie(
            archetype_counts, 
            values='Count', 
            names='Trader Archetype', 
            title="Ecosystem Persona Distribution",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            template="plotly_dark"
        )
        st.plotly_chart(fig_arc_pie, use_container_width=True)
        
    with col_per_r:
        st.subheader("Average Realized Profitability by Archetype")
        arc_perf = df_behavioral.groupby('trader_archetype')['total_realized_pnl'].mean().reset_index().sort_values('total_realized_pnl', ascending=False)
        
        fig_arc_perf = px.bar(
            arc_perf,
            x='trader_archetype',
            y='total_realized_pnl',
            color='total_realized_pnl',
            color_continuous_scale=px.colors.diverging.RdYlGn,
            title="Average Lifetime PnL ($) per Archetype",
            labels={'total_realized_pnl': 'Avg Lifetime PnL (USD)', 'trader_archetype': 'Trader Archetype'},
            template="plotly_dark"
        )
        st.plotly_chart(fig_arc_perf, use_container_width=True)
        
    st.subheader("Behavioral Wallet Intelligence Ledger")
    df_disp_personas = df_behavioral[['Account', 'trader_archetype', 'behavioral_risk_score', 'total_realized_pnl', 'win_rate', 'avg_leverage', 'avg_trade_size', 'trade_count']].copy()
    df_disp_personas['total_realized_pnl'] = df_disp_personas['total_realized_pnl'].map(lambda x: f"${x:,.2f}")
    df_disp_personas['win_rate'] = df_disp_personas['win_rate'].map(lambda x: f"{x*100:.2f}%")
    df_disp_personas['avg_leverage'] = df_disp_personas['avg_leverage'].map(lambda x: f"{x:.2f}x")
    df_disp_personas['avg_trade_size'] = df_disp_personas['avg_trade_size'].map(lambda x: f"${x:,.2f}")
    
    st.dataframe(df_disp_personas, use_container_width=True)

# -------------------------------------------------------------
# Screen 4: Risk Intelligence (NEW - Phase 4)
# -------------------------------------------------------------
elif page == "🛡️ Risk Intelligence" and df_raw is not None:
    st.title("🛡️ Risk Intelligence Terminal")
    st.markdown("Consolidates portfolio Sharpe, Sortino, Calmar ratios, maximum drawdowns, and regime analytics.")
    
    df_risk_report = generate_risk_metrics_report(df_raw, group_col='sentiment_class')
    
    if not df_risk_report.empty:
        st.subheader("Macro Sentiment-Driven Risk Matrix")
        df_disp_risk = df_risk_report.copy()
        df_disp_risk['volatility'] = df_disp_risk['volatility'].map(lambda x: f"{x*100:.2f}%")
        df_disp_risk['max_drawdown_roe'] = df_disp_risk['max_drawdown_roe'].map(lambda x: f"{x*100:.2f}%")
        df_disp_risk['sharpe_ratio'] = df_disp_risk['sharpe_ratio'].map(lambda x: f"{x:.4f}")
        df_disp_risk['sortino_ratio'] = df_disp_risk['sortino_ratio'].map(lambda x: f"{x:.4f}")
        df_disp_risk['calmar_ratio'] = df_disp_risk['calmar_ratio'].map(lambda x: f"{x:.4f}")
        df_disp_risk['var_95'] = df_disp_risk['var_95'].map(lambda x: f"${x:,.2f}")
        df_disp_risk['cvar_95'] = df_disp_risk['cvar_95'].map(lambda x: f"${x:,.2f}")
        df_disp_risk['cvar_99'] = df_disp_risk['cvar_99'].map(lambda x: f"${x:,.2f}")
        
        st.dataframe(df_disp_risk, use_container_width=True)
        st.divider()
        
        col_risk_l, col_risk_r = st.columns(2)
        
        with col_risk_l:
            st.subheader("Ratios Comparison (Sharpe vs Sortino)")
            fig_ratios = go.Figure(data=[
                go.Bar(name='Sharpe Ratio', x=df_risk_report['sentiment_class'], y=df_risk_report['sharpe_ratio'], marker_color='#3498db'),
                go.Bar(name='Sortino Ratio', x=df_risk_report['sentiment_class'], y=df_risk_report['sortino_ratio'], marker_color='#2ecc71')
            ])
            fig_ratios.update_layout(title="Annualized Risk-Adjusted Ratios per FGI State", template="plotly_dark")
            st.plotly_chart(fig_ratios, use_container_width=True)
            
        with col_risk_r:
            st.subheader("Maximum Downside Drawdowns ($ USD)")
            fig_dd = px.bar(
                df_risk_report,
                x='sentiment_class',
                y='max_drawdown_usd',
                title="Max Historical Dollar Drawdown by Sentiment class",
                color='max_drawdown_usd',
                color_continuous_scale='Reds_r',
                template="plotly_dark"
            )
            st.plotly_chart(fig_dd, use_container_width=True)

# -------------------------------------------------------------
# NLP Sub-Pages (Existing - Backward Compatible)
# -------------------------------------------------------------
elif page == "📰 News Intelligence" and df_nlp_scores is not None:
    st.title("📰 News Intelligence Terminal")
    col_n1, col_n2 = st.columns([2, 1])
    with col_n1:
        st.subheader("Latest Preprocessed Market Articles")
        news_dir = "data/silver/news_clean"
        if os.path.exists(news_dir):
            files = [f for f in os.listdir(news_dir) if f.endswith('.json')]
            for f_name in files[:5]:
                with open(os.path.join(news_dir, f_name), "r") as f:
                    art = json.load(f)
                with st.expander(f"📰 {art['title']} (Source: {art['source']})"):
                    st.write(art['content'])
                    st.markdown(f"[Original Link]({art['url']})")
    with col_n2:
        st.subheader("Publishing Source Distribution")
        source_counts = df_nlp_scores['source'].value_counts().reset_index()
        source_counts.columns = ['Source', 'Count']
        fig_sources = px.pie(source_counts, values='Count', names='Source', title="Articles by Publisher", template="plotly_dark")
        st.plotly_chart(fig_sources, use_container_width=True)

elif page == "🧠 Sentiment Intelligence" and df_nlp_scores is not None:
    st.title("🧠 Behavioral Sentiment Intelligence")
    col_s1, col_s2 = st.columns([1, 1.2])
    with col_s1:
        st.subheader("Sentiment Class Ratio")
        sent_counts = df_nlp_scores['sentiment_class'].value_counts().reset_index()
        sent_counts.columns = ['Sentiment', 'Count']
        fig_ratio = px.pie(sent_counts, values='Count', names='Sentiment', color='Sentiment',
                           color_discrete_map={'Bullish': '#2ecc71', 'Neutral': '#95a5a6', 'Bearish': '#e74c3c'},
                           title="Bullish vs Bearish Ratio", template="plotly_dark")
        st.plotly_chart(fig_ratio, use_container_width=True)
    with col_s2:
        st.subheader("Continuous Sentiment Momentum & Divergence Ratios")
        if df_nlp_trends is not None and not df_nlp_trends.empty:
            fig_trend = px.line(df_nlp_trends, x='date', y=['sentiment_momentum', 'divergence'], title="7-Day Rolling Sentiment Momentum vs Price Divergence", template="plotly_dark")
            st.plotly_chart(fig_trend, use_container_width=True)

elif page == "💬 Topic Intelligence" and df_nlp_topics is not None:
    st.title("💬 Unsupervised Topic Intelligence")
    col_t1, col_t2 = st.columns([1.2, 1])
    with col_t1:
        st.subheader("Dominant Topic Clusters Frequency")
        topic_counts = df_nlp_topics['topic_theme'].value_counts().reset_index()
        topic_counts.columns = ['Topic Theme', 'Article Count']
        fig_top_freq = px.bar(topic_counts, x='Article Count', y='Topic Theme', orientation='h', title="Extracted Themes Density Index", color='Article Count', color_continuous_scale='Blues', template="plotly_dark")
        st.plotly_chart(fig_top_freq, use_container_width=True)
    with col_t2:
        st.subheader("Temporal Topic Distribution / Influx")
        df_nlp_topics['date_only'] = pd.to_datetime(df_nlp_topics['published_date']).dt.strftime('%m-%d')
        df_topic_evol = df_nlp_topics.groupby(['date_only', 'topic_theme']).size().reset_index(name='count')
        fig_evol = px.area(df_topic_evol, x='date_only', y='count', color='topic_theme', title="Chronological Topic Evolution & Influx", template="plotly_dark")
        st.plotly_chart(fig_evol, use_container_width=True)

elif page == "👥 Entity Intelligence" and df_nlp_entities is not None:
    st.title("👥 Entity Extraction & Sentiment Association Ledger")
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.subheader("Top Extracted Entities Frequency Index")
        ent_counts = df_nlp_entities.groupby(['entity', 'entity_type']).size().reset_index(name='frequency').sort_values('frequency', ascending=False)
        fig_ents = px.bar(ent_counts.head(15), x='frequency', y='entity', color='entity_type', orientation='h', title="Top 15 Extracted Named Entities", template="plotly_dark")
        st.plotly_chart(fig_ents, use_container_width=True)
    with col_e2:
        st.subheader("Sentiment Co-Occurrence Association Ratios")
        ent_sent = df_nlp_entities.groupby('entity')['sentiment_association'].mean().reset_index().sort_values('sentiment_association', ascending=False)
        fig_ent_sent = px.bar(pd.concat([ent_sent.head(10), ent_sent.tail(10)]), x='sentiment_association', y='entity', color='sentiment_association', color_continuous_scale=px.colors.diverging.RdYlGn, orientation='h', title="Extracted Entity Sentiment Biases", template="plotly_dark")
        st.plotly_chart(fig_ent_sent, use_container_width=True)

elif page == "💥 Market Impact Dashboard" and df_nlp_impact is not None:
    st.title("💥 Market Impact & Behavioral Risk Dashboard")
    df_critical_ledger = df_nlp_impact[df_nlp_impact['market_impact_score'] >= 50.0].copy().sort_values('market_impact_score', ascending=False)
    if not df_critical_ledger.empty:
        df_disp_ledger = df_critical_ledger[['title', 'market_impact_score', 'market_impact_class', 'sentiment_score', 'source']].copy()
        st.dataframe(df_disp_ledger, use_container_width=True)
else:
    st.error("Intelligence assets missing. Check pipeline logs.")
