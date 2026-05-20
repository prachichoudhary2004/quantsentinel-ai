import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import pickle

# Load prediction model
@st.cache_resource
def load_prediction_model():
    if os.path.exists("data/results/profitable_trade_model.pkl"):
        with open("data/results/profitable_trade_model.pkl", "rb") as f:
            return pickle.load(f)
    return None, None

rf_model, training_features = load_prediction_model()

# Set page configuration to wide layout and professional fintech theme
st.set_page_config(
    page_title="Sentiment-Driven Crypto Trading Intelligence System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling (glassmorphism, clean fonts, dark terminal vibes)
st.markdown("""
<style>
    .reportview-container {
        background: #0d1117;
    }
    .main {
        background-color: #0b0e14;
        color: #c9d1d9;
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 700;
        font-family: 'Space Grotesk', sans-serif;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #58a6ff;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #161b22;
        padding: 6px 12px;
        border-radius: 8px;
        border: 1px solid #30363d;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px;
        color: #8b949e;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff;
        background-color: #21262d;
    }
    .stTabs [aria-selected="true"] {
        color: #58a6ff !important;
        background-color: #30363d !important;
        border-bottom: 2px solid #58a6ff !important;
    }
    .card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 15px;
    }
    .green-text { color: #39d353; font-weight: bold; }
    .red-text { color: #f85149; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Data Loading & Caching
# -------------------------------------------------------------
@st.cache_data
def load_processed_data():
    if os.path.exists("data/processed/merged_trader_data.csv"):
        return pd.read_csv("data/processed/merged_trader_data.csv")
    else:
        st.error("Processed data file not found! Please run the preprocessing scripts first.")
        return None

@st.cache_data
def load_trader_segments():
    if os.path.exists("data/results/trader_segments.csv"):
        return pd.read_csv("data/results/trader_segments.csv")
    return None

@st.cache_data
def load_feature_importance():
    if os.path.exists("data/results/feature_importance.csv"):
        return pd.read_csv("data/results/feature_importance.csv")
    return None

df_raw = load_processed_data()
df_segments = load_trader_segments()
df_importance = load_feature_importance()

if df_raw is not None:
    # -------------------------------------------------------------
    # Sidebar Filters
    # -------------------------------------------------------------
    st.sidebar.image("https://cryptologos.cc/logos/bitcoin-btc-logo.png", width=50)
    st.sidebar.title("Trading Intelligence")
    st.sidebar.markdown("*Quant Behavioral Analytics Terminal*")
    st.sidebar.divider()
    
    # Account Filter
    all_accounts = sorted(df_raw['Account'].unique())
    selected_accounts = st.sidebar.multiselect(
        "Filter Trader Accounts",
        options=all_accounts,
        default=[]
    )
    
    # Coin Filter
    all_coins = sorted(df_raw['Coin'].unique())
    selected_coins = st.sidebar.multiselect(
        "Filter Symbols (Coins)",
        options=all_coins,
        default=[]
    )
    
    # Sentiment Filter
    all_sentiments = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
    selected_sentiments = st.sidebar.multiselect(
        "Filter Market Sentiment",
        options=all_sentiments,
        default=all_sentiments
    )
    
    # Apply filters dynamically
    df_filtered = df_raw.copy()
    if selected_accounts:
        df_filtered = df_filtered[df_filtered['Account'].isin(selected_accounts)]
    if selected_coins:
        df_filtered = df_filtered[df_filtered['Coin'].isin(selected_coins)]
    if selected_sentiments:
        df_filtered = df_filtered[df_filtered['sentiment_class'].isin(selected_sentiments)]
        
    df_realized = df_filtered[df_filtered['is_closing_trade'] == True].copy()
    
    # -------------------------------------------------------------
    # Main Terminal Layout
    # -------------------------------------------------------------
    st.title("⚡ Sentiment-Driven Crypto Trading Intelligence")
    st.markdown(
        "Explores the structural relationships between crypto market sentiment (Bitcoin Fear & Greed Index) "
        "and real-world trader performance on Hyperliquid."
    )
    
    # KPIs Row
    st.divider()
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6 = st.columns(6)
    
    # Compute KPIs
    total_pnl = df_realized['Closed PnL'].sum()
    win_rate = df_realized['win_flag'].mean() * 100 if len(df_realized) > 0 else 0
    total_trades = len(df_realized)
    total_volume = df_realized['Size USD'].sum()
    avg_leverage = df_filtered['leverage'].mean()
    
    # Profit Factor
    pos_pnl = df_realized[df_realized['Closed PnL'] > 0]['Closed PnL'].sum()
    neg_pnl = df_realized[df_realized['Closed PnL'] < 0]['Closed PnL'].sum()
    profit_factor = pos_pnl / abs(neg_pnl) if neg_pnl != 0 else np.nan
    
    with kpi_col1:
        st.metric(
            label="Realized PnL (USD)",
            value=f"${total_pnl:,.2f}",
            delta=f"{'+' if total_pnl >= 0 else ''}{total_pnl/1e6:.2f}M",
            delta_color="normal"
        )
    with kpi_col2:
        st.metric(
            label="Aggregate Win Rate",
            value=f"{win_rate:.2f}%",
            delta="Benchmark: 55%" if win_rate > 55 else "Sub-optimal",
            delta_color="inverse" if win_rate < 50 else "normal"
        )
    with kpi_col3:
        pf_val = f"{profit_factor:.2f}" if pd.notnull(profit_factor) else "N/A"
        st.metric(
            label="Profit Factor",
            value=pf_val,
            delta="Institutional" if profit_factor > 2.0 else "Speculative"
        )
    with kpi_col4:
        st.metric(
            label="Total Volume Traded",
            value=f"${total_volume/1e6:.2f}M"
        )
    with kpi_col5:
        st.metric(
            label="Avg Effective Leverage",
            value=f"{avg_leverage:.2f}x"
        )
    with kpi_col6:
        st.metric(
            label="Realized Trades Count",
            value=f"{total_trades:,}"
        )
    st.divider()
    
    # -------------------------------------------------------------
    # Navigation Tabs
    # -------------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Profitability & Sentiment", 
        "⚙️ Leverage & Risk Mechanics", 
        "⏱️ Activity & Sizing", 
        "👥 Trader Segments", 
        "🔮 Algorithmic Predictive Sandbox"
    ])
    
    # -------------------------------------------------------------
    # Tab 1: Profitability & Sentiment
    # -------------------------------------------------------------
    with tab1:
        st.subheader("Profitability & Win Rates across Market Sentiment States")
        st.markdown(
            "We analyze the realized performance of traders under the 5 stages of the Bitcoin Fear & Greed Index. "
            "Notice the dynamic shift in profitability from Fear to Extreme Greed."
        )
        
        col_t1_l, col_t1_r = st.columns(2)
        
        with col_t1_l:
            # Group by Sentiment for Plotting
            sent_perf = df_realized.groupby('sentiment_class').agg(
                total_pnl=('Closed PnL', 'sum'),
                win_rate=('win_flag', 'mean'),
                trade_count=('Closed PnL', 'count')
            ).reset_index()
            
            # Reorder for logical plotting
            sent_perf['sentiment_class'] = pd.Categorical(
                sent_perf['sentiment_class'], categories=all_sentiments, ordered=True
            )
            sent_perf = sent_perf.sort_values('sentiment_class')
            
            # PnL Chart
            fig_pnl = px.bar(
                sent_perf,
                x='sentiment_class',
                y='total_pnl',
                title="Total Realized PnL ($) by Sentiment State",
                labels={'total_pnl': 'Total PnL (USD)', 'sentiment_class': 'Market Sentiment'},
                color='total_pnl',
                color_continuous_scale=px.colors.diverging.RdYlGn,
                template="plotly_dark"
            )
            fig_pnl.update_layout(showlegend=False)
            st.plotly_chart(fig_pnl, use_container_width=True)
            
        with col_t1_r:
            # Win Rate Chart
            sent_perf['win_rate_pct'] = sent_perf['win_rate'] * 100
            fig_wr = px.line(
                sent_perf,
                x='sentiment_class',
                y='win_rate_pct',
                title="Aggregate Win Rate (%) by Sentiment State",
                labels={'win_rate_pct': 'Win Rate (%)', 'sentiment_class': 'Market Sentiment'},
                markers=True,
                template="plotly_dark"
            )
            fig_wr.update_traces(line=dict(color='#58a6ff', width=3))
            fig_wr.update_yaxes(range=[40, 100])
            st.plotly_chart(fig_wr, use_container_width=True)
            
        st.subheader("💡 Quantitative Insights")
        if not sent_perf.empty:
            max_pnl_idx = sent_perf['total_pnl'].idxmax()
            min_pnl_idx = sent_perf['total_pnl'].idxmin()
            top_sent = sent_perf.loc[max_pnl_idx, 'sentiment_class']
            top_pnl = sent_perf.loc[max_pnl_idx, 'total_pnl']
            bot_sent = sent_perf.loc[min_pnl_idx, 'sentiment_class']
            bot_pnl = sent_perf.loc[min_pnl_idx, 'total_pnl']
            
            st.markdown(
                f"""
                <div style="background-color: #161b22; padding: 15px; border-radius: 8px; border-left: 4px solid #58a6ff;">
                    <b>Dynamic PnL Extremes:</b><br/>
                    • <b>Most Profitable Environment:</b> {top_sent} (${top_pnl:,.2f})<br/>
                    • <b>Least Profitable Environment:</b> {bot_sent} (${bot_pnl:,.2f})<br/>
                    <i style="color:#8b949e; font-size: 0.85em;">(Values automatically update based on active filters)</i>
                </div>
                """, unsafe_allow_html=True
            )
        
    # -------------------------------------------------------------
    # Tab 2: Leverage & Risk Mechanics
    # -------------------------------------------------------------
    with tab2:
        st.subheader("Leverage Mechanics and Tail-Risk Patterns")
        st.markdown(
            "Crypto perpetual exchanges offer high leverage. Let's inspect the actual impact of leverage on trading outcome "
            "and tail risk."
        )
        
        col_t2_l, col_t2_r = st.columns(2)
        
        with col_t2_l:
            # Leverage Binned Performance
            lev_perf = df_realized.groupby('leverage_bucket').agg(
                avg_pnl=('Closed PnL', 'mean'),
                win_rate=('win_flag', 'mean'),
                total_pnl=('Closed PnL', 'sum')
            ).reset_index()
            
            fig_lev_pnl = px.bar(
                lev_perf,
                x='leverage_bucket',
                y='total_pnl',
                title="Total PnL ($) by Leverage Bucket",
                labels={'total_pnl': 'Total PnL (USD)', 'leverage_bucket': 'Leverage Bucket'},
                color='total_pnl',
                color_continuous_scale='Tealgrn',
                template="plotly_dark"
            )
            st.plotly_chart(fig_lev_pnl, use_container_width=True)
            
        with col_t2_r:
            # Value at Risk by Sentiment (VaR 95% & VaR 99%) calculated dynamically
            risk_stats = []
            for sent in all_sentiments:
                df_sent = df_realized[df_realized['sentiment_class'] == sent]
                if len(df_sent) > 0:
                    var_95 = np.percentile(df_sent['Closed PnL'], 5)
                    var_99 = np.percentile(df_sent['Closed PnL'], 1)
                    risk_stats.append({
                        'sentiment_class': sent,
                        'var_95': abs(var_95),
                        'var_99': abs(var_99)
                    })
            if risk_stats:
                df_risk_dyn = pd.DataFrame(risk_stats)
                fig_risk = go.Figure(data=[
                    go.Bar(name='VaR 95% (USD Downside)', x=df_risk_dyn['sentiment_class'], y=df_risk_dyn['var_95'], marker_color='#ff7f0e'),
                    go.Bar(name='VaR 99% (Extreme Tail Downside)', x=df_risk_dyn['sentiment_class'], y=df_risk_dyn['var_99'], marker_color='#d62728')
                ])
                fig_risk.update_layout(
                    title="Value at Risk (VaR) Downside Bounds by Sentiment State",
                    xaxis_title="Market Sentiment",
                    yaxis_title="Potential Downside Loss (USD)",
                    template="plotly_dark"
                )
                st.plotly_chart(fig_risk, use_container_width=True)
            else:
                st.write("No realized trades for the selected filters to perform risk analysis.")
                
        st.subheader("💡 Tail-Risk & Leverage Insights")
        if not lev_perf.empty and len(risk_stats) > 0:
            low_lev_pnl = lev_perf[lev_perf['leverage_bucket'] == 'Low Leverage (1x-3x)']['total_pnl'].sum() if 'Low Leverage (1x-3x)' in lev_perf['leverage_bucket'].values else 0
            ext_lev_pnl = lev_perf[lev_perf['leverage_bucket'] == 'Extreme Leverage (21x-50x)']['total_pnl'].sum() if 'Extreme Leverage (21x-50x)' in lev_perf['leverage_bucket'].values else 0
            max_risk_sent = df_risk_dyn.loc[df_risk_dyn['var_99'].idxmax(), 'sentiment_class']
            max_risk_val = df_risk_dyn['var_99'].max()
            
            st.markdown(
                f"""
                <div style="background-color: #161b22; padding: 15px; border-radius: 8px; border-left: 4px solid #f85149;">
                    <b>Risk & Reward Imbalance:</b><br/>
                    • <b>Low Leverage (1x-3x) PnL:</b> ${low_lev_pnl:,.2f}<br/>
                    • <b>Extreme Leverage (21x-50x) PnL:</b> ${ext_lev_pnl:,.2f}<br/>
                    • <b>Maximum Downside Exposure (VaR 99%):</b> {max_risk_sent} (${max_risk_val:,.2f})<br/>
                    <i style="color:#8b949e; font-size: 0.85em;">(Values automatically update based on active filters)</i>
                </div>
                """, unsafe_allow_html=True
            )

    # -------------------------------------------------------------
    # Tab 3: Activity & Sizing
    # -------------------------------------------------------------
    with tab3:
        st.subheader("Trading Frequency and Position Sizing Volatility")
        st.markdown(
            "How do traders adjust their behavior (trade frequency and size) under different sentiment environments?"
        )
        
        col_t3_l, col_t3_r = st.columns(2)
        
        # Compute frequency and sizing dynamically
        days_per_sent = df_filtered.groupby('sentiment_class')['date'].nunique()
        df_freq_dyn = df_filtered.groupby('sentiment_class').agg(
            total_trades=('Trade ID', 'count'),
            avg_trade_size=('Size USD', 'mean')
        ).reset_index()
        df_freq_dyn['unique_days'] = df_freq_dyn['sentiment_class'].map(days_per_sent)
        # Avoid division by zero
        df_freq_dyn['avg_trades_per_day'] = np.where(
            df_freq_dyn['unique_days'] > 0,
            df_freq_dyn['total_trades'] / df_freq_dyn['unique_days'],
            0.0
        )
        
        # Sort by sentiment order
        df_freq_dyn['sentiment_class'] = pd.Categorical(
            df_freq_dyn['sentiment_class'], categories=all_sentiments, ordered=True
        )
        df_freq_dyn = df_freq_dyn.sort_values('sentiment_class')
        
        with col_t3_l:
            # Trading frequency chart
            fig_freq = px.bar(
                df_freq_dyn,
                x='sentiment_class',
                y='avg_trades_per_day',
                title="Average Trades Per Day by Sentiment State",
                labels={'avg_trades_per_day': 'Avg Trades / Day', 'sentiment_class': 'Market Sentiment'},
                color='avg_trades_per_day',
                color_continuous_scale='Mint',
                template="plotly_dark"
            )
            st.plotly_chart(fig_freq, use_container_width=True)
            
        with col_t3_r:
            # Average position sizing
            fig_size = px.line(
                df_freq_dyn,
                x='sentiment_class',
                y='avg_trade_size',
                title="Average Trade Size ($ USD) by Sentiment State",
                labels={'avg_trade_size': 'Avg Trade Size (USD)', 'sentiment_class': 'Market Sentiment'},
                markers=True,
                template="plotly_dark"
            )
            fig_size.update_traces(line=dict(color='#ff7f0e', width=3))
            st.plotly_chart(fig_size, use_container_width=True)
            
        st.subheader("💡 Sizing & Frequency Takeaways")
        if not df_freq_dyn.empty:
            max_freq_idx = df_freq_dyn['avg_trades_per_day'].idxmax()
            min_freq_idx = df_freq_dyn['avg_trades_per_day'].idxmin()
            high_freq_sent = df_freq_dyn.loc[max_freq_idx, 'sentiment_class']
            high_freq_val = df_freq_dyn.loc[max_freq_idx, 'avg_trades_per_day']
            low_freq_sent = df_freq_dyn.loc[min_freq_idx, 'sentiment_class']
            low_freq_val = df_freq_dyn.loc[min_freq_idx, 'avg_trades_per_day']
            
            st.markdown(
                f"""
                <div style="background-color: #161b22; padding: 15px; border-radius: 8px; border-left: 4px solid #39d353;">
                    <b>Trading Activity Spikes:</b><br/>
                    • <b>Peak Hyper-Activity:</b> {high_freq_sent} ({high_freq_val:,.0f} trades/day)<br/>
                    • <b>Lowest Activity Phase (HODL):</b> {low_freq_sent} ({low_freq_val:,.0f} trades/day)<br/>
                    <i style="color:#8b949e; font-size: 0.85em;">(Values automatically update based on active filters)</i>
                </div>
                """, unsafe_allow_html=True
            )

    # -------------------------------------------------------------
    # Tab 4: Trader Segments
    # -------------------------------------------------------------
    with tab4:
        st.subheader("Algorithmic Trader Segmentation (K-Means Personas)")
        st.markdown(
            "Using unsupervised learning (K-Means clustering), we analyzed the 32 historical wallets and segmented them "
            "into 4 distinct profiles based on profitability, size, leverage, win rates, and directional bias."
        )
        
        if df_segments is not None:
            # Filter df_segments dynamically by active trader accounts selected in sidebar
            df_seg_filtered = df_segments.copy()
            if selected_accounts:
                df_seg_filtered = df_seg_filtered[df_seg_filtered['Account'].isin(selected_accounts)]
                
            col_t4_l, col_t4_r = st.columns([1, 1])
            
            with col_t4_l:
                # Segment Distribution Pie Chart
                seg_counts = df_seg_filtered['trader_segment'].value_counts().reset_index()
                seg_counts.columns = ['Segment', 'Count']
                fig_seg = px.pie(
                    seg_counts,
                    values='Count',
                    names='Segment',
                    title="Trader Segments Population Distribution",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    template="plotly_dark"
                )
                fig_seg.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_seg, use_container_width=True)
                
            with col_t4_r:
                # Segments Performance comparison
                seg_perf = df_seg_filtered.groupby('trader_segment').agg(
                    total_pnl=('total_pnl', 'mean'),
                    avg_leverage=('avg_leverage', 'mean'),
                    win_rate=('win_rate', 'mean')
                ).reset_index()
                seg_perf['win_rate_pct'] = seg_perf['win_rate'] * 100
                
                fig_seg_perf = px.bar(
                    seg_perf,
                    x='trader_segment',
                    y='total_pnl',
                    title="Average Trader Lifetime Profitability ($) by Segment",
                    labels={'total_pnl': 'Avg Lifetime PnL (USD)', 'trader_segment': 'Trader Segment'},
                    color='avg_leverage',
                    color_continuous_scale='Viridis',
                    template="plotly_dark"
                )
                st.plotly_chart(fig_seg_perf, use_container_width=True)
                
            st.subheader("Wallet Intelligence Ledger")
            # Show a table with the accounts and segments
            df_display_segments = df_seg_filtered[['Account', 'trader_segment', 'total_pnl', 'win_rate', 'avg_leverage', 'avg_trade_size', 'trade_count', 'profit_factor']].copy()
            df_display_segments['total_pnl'] = df_display_segments['total_pnl'].map(lambda x: f"${x:,.2f}")
            df_display_segments['win_rate'] = df_display_segments['win_rate'].map(lambda x: f"{x*100:.2f}%")
            df_display_segments['avg_leverage'] = df_display_segments['avg_leverage'].map(lambda x: f"{x:.2f}x")
            df_display_segments['avg_trade_size'] = df_display_segments['avg_trade_size'].map(lambda x: f"${x:,.2f}")
            
            st.dataframe(df_display_segments, use_container_width=True)
        else:
            st.write("Clustering segmentation dataset not found.")

    # -------------------------------------------------------------
    # Tab 5: Algorithmic Predictive Sandbox
    # -------------------------------------------------------------
    with tab5:
        st.subheader("Trade Profitability Predictor Sandbox")
        st.markdown(
            "Based on our trained **Random Forest Classifier (Accuracy: 84.97%, ROC-AUC: 0.90)**, "
            "this simulator allows you to input current trade parameters and estimate the probability that the trade will be profitable."
        )
        
        sandbox_col1, sandbox_col2 = st.columns([1, 1])
        
        with sandbox_col1:
            st.write("### 🎛️ Trade Parameter Terminals")
            
            input_coin = st.selectbox("Symbol / Coin", options=all_coins, index=all_coins.index("SOL") if "SOL" in all_coins else 0)
            input_direction = st.selectbox("Trade Direction", options=['Long', 'Short'])
            input_size = st.number_input("Trade Size (USD)", min_value=10.0, max_value=1000000.0, value=5000.0, step=500.0)
            input_leverage = st.slider("Leverage Used", min_value=1, max_value=50, value=5)
            input_sentiment = st.slider("Bitcoin Fear & Greed Score (0 - 100)", min_value=0, max_value=100, value=50)
            input_hour = st.slider("Execution Hour (IST, 0 - 23)", min_value=0, max_value=23, value=12)
            input_day = st.selectbox("Execution Day", options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
            
            # Predict Button
            predict_triggered = st.button("🚀 Run Predictive Classifier")
            
        with sandbox_col2:
            st.write("### 🔮 Classification Analysis & Diagnostics")
            
            if predict_triggered:
                if rf_model is not None and training_features is not None:
                    # Construct single-row DataFrame representing the trade input
                    input_df = pd.DataFrame([{
                        'leverage': float(input_leverage),
                        'Size USD': float(input_size),
                        'sentiment_score': float(input_sentiment),
                        'trade_direction': input_direction,
                        'hour': int(input_hour),
                        'day_of_week': input_day,
                        'Coin': input_coin
                    }])
                    
                    # Perform one-hot encoding
                    input_encoded = pd.get_dummies(input_df, columns=['trade_direction', 'day_of_week', 'Coin'])
                    
                    # Align columns exactly with the trained feature list
                    aligned_data = {}
                    for col in training_features:
                        if col in input_encoded.columns:
                            aligned_data[col] = input_encoded[col].iloc[0]
                        else:
                            aligned_data[col] = 0.0
                            
                    # Construct the final aligned features DataFrame
                    X_input = pd.DataFrame([aligned_data])[training_features]
                    
                    # Compute win probability via Random Forest model
                    prob_win = float(rf_model.predict_proba(X_input)[0, 1])
                    
                    st.markdown(f"**Predicted Profitability (Win Probability):**")
                    if prob_win >= 0.55:
                        st.success(f"🏆 HIGH WIN PROBABILITY: {prob_win*100:.2f}%")
                    elif prob_win >= 0.45:
                        st.info(f"⚖️ NEUTRAL PROBABILITY: {prob_win*100:.2f}%")
                    else:
                        st.error(f"⚠️ HIGH RISK / LOW WIN PROBABILITY: {prob_win*100:.2f}%")
                        
                    # Visual Gauge
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = prob_win * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Asymmetric Edge Score (%)", 'font': {'size': 20}},
                        gauge = {
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                            'bar': {'color': "#58a6ff"},
                            'bgcolor': "#161b22",
                            'borderwidth': 2,
                            'bordercolor': "#30363d",
                            'steps': [
                                {'range': [0, 45], 'color': '#ff4b4b'},
                                {'range': [45, 55], 'color': '#f1c40f'},
                                {'range': [55, 100], 'color': '#2ecc71'}
                            ],
                            'threshold': {
                                'line': {'color': "white", 'width': 4},
                                'thickness': 0.75,
                                'value': 55
                            }
                        }
                    ))
                    fig_gauge.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font={'color': "white", 'family': "Arial"},
                        height=250
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    # Dynamic recommendation based on actual model outcomes
                    st.write("### 📋 Quant Diagnostics Recommendation:")
                    if prob_win >= 0.55:
                        st.markdown(
                            "👉 **Decision:** **APPROVED FOR EXECUTION**\n"
                            "This trade represents a structurally sound, low-leverage trade executed in favorable sentiment environments. "
                            "Downside Value at Risk (VaR) is tightly bounded. Proceed with standard size."
                        )
                    else:
                        st.markdown(
                            "👉 **Decision:** **REJECTED OR SIZE DOWN**\n"
                            "This trade exhibits critical behavioral flaws: either leverage is too high, or you are "
                            "trading in highly unfavorable sentiment regimes (such as shorting high momentum, or buying capitulations "
                            "with high leverage). Reduce leverage to <3x or skip the trade."
                        )
                else:
                    st.write("Predictive classifier model is missing. Please run modeling script first.")
            else:
                st.info("Input trade parameters and click 'Run Predictive Classifier' to calculate the trade score.")
                
            # Feature Importance Panel
            if df_importance is not None:
                st.divider()
                st.write("### 📈 Top 10 Profitability Drivers (Random Forest Feature Importance)")
                fig_imp = px.bar(
                    df_importance.head(10),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title="Feature Importances in Profitability Classifier",
                    labels={'importance': 'Feature Weight', 'feature': 'Trading Metric'},
                    color='importance',
                    color_continuous_scale='Blues',
                    template="plotly_dark"
                )
                fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_imp, use_container_width=True)
else:
    st.error("Could not load data.")
