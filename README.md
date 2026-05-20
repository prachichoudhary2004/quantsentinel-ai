<h1 align="center">⚡ Sentiment-Driven Crypto Intelligence System</h1>

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version"/>
  <img src="https://img.shields.io/badge/Machine%20Learning-scikit--learn-orange.svg" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/Dashboard-Streamlit-red.svg" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/Data%20Science-Pandas%20%7C%20NumPy-yellow.svg" alt="Data Science"/>
  <img src="https://img.shields.io/badge/Status-Completed-success.svg" alt="Status"/>
</div>

<br/>

 **An end-to-end quantitative and behavioral analytics platform** built to uncover how market psychology influences trader profitability, leverage behavior, and execution risk. This project merges the **Bitcoin Fear & Greed Index** with real-world trading activity from the **Hyperliquid Perpetual DEX**.

---

## 📖 Project Overview

Understanding the cryptocurrency market requires more than just price analysis—it requires understanding *human psychology*. This project investigates the profound relationship between market sentiment (Fear vs. Greed) and trader behavior. 

By analyzing real historical trades against daily sentiment data, this intelligence system uncovers hidden patterns, such as when traders take on the most risk, when they are most likely to face liquidations, and how institutional trading strategies differ from emotionally-driven retail trading.

---

## ✨ Comprehensive Feature Set

### 1. 🧠 Sentiment-Driven Behavioral Analytics
- **Sentiment & PnL Correlation**: Analyzes win rates, total realized PnL, and Return on Equity (ROE) across 5 distinct sentiment regimes (Extreme Fear, Fear, Neutral, Greed, Extreme Greed).
- **Leverage Fallacy Detection**: Identifies how leverage behavior shifts during volatile market conditions and quantifies the amplified downside risk of extreme leverage (21x–50x).

### 2. 📈 Advanced Quantitative Risk Modeling
- **Value-at-Risk (VaR)**: Calculates 95% and 99% VaR for each sentiment class to mathematically quantify tail-risk during market panics versus euphoric rallies.
- **Profit Factors & Return Profiles**: Evaluates gross profits relative to gross losses, isolating which market environments generate the most asymmetric, positive expected value (+EV) opportunities.

### 3. 🤖 Machine Learning Implementation
- **Trader Persona Segmentation (K-Means Clustering)**: Unsupervised learning groups individual accounts into professional archetypes based on their trading statistics:
  - 🐋 *Institutional Whales* (High volume, high PnL)
  - 🎰 *High-Leverage Speculators* (Extreme risk, high volatility)
  - ⚡ *Systematic Scalpers* (High frequency, lower sizing)
  - 🛡️ *Conservative/Tactical Retail* (Low frequency, tactical entries)
- **Trade Profitability Prediction (Random Forest Classifier)**: A supervised predictive model trained to forecast whether an individual trade will be profitable based on leverage, trade size, direction, and prevailing market sentiment.

### 4. 📊 Interactive Analytics Dashboard
- A fully responsive **Streamlit UI** allowing users to interactively slice and dice data. It features:
  - Visual distribution of leverage across different sentiment states.
  - Heatmaps for extreme loss events and risk indicators.
  - Deep-dives into individual trader performance and segmented personas.

### 5. 🚀 Automated ETL Pipeline
- **Zero-Touch Ingestion**: Automatically downloads raw CSV datasets directly from Google Drive.
- **Robust Preprocessing**: Harmonizes timestamps, engineers deterministic leverage metrics, calculates ROE, assigns win/loss flags, and handles missing data.

---

## 🗄️ Datasets

This project relies on two primary data sources, provided as part of the **Primetrade.ai** Data Science Assignment:

1. **Historical Trader Data from Hyperliquid**
   - **Description**: Contains thousands of individual trades executed on the Hyperliquid Perpetual DEX.
   - **Key Features**: `Account`, `Coin`, `Execution Price`, `Size USD`, `Direction`, `Closed PnL`, `Timestamp`.
   - 🔗 [Download Raw Dataset (Google Drive)](https://drive.google.com/file/d/1IAfLZwu6rJzyWKgBToqwSmmVYU6VbjVs/view?usp=sharing)

2. **Bitcoin Market Sentiment Dataset**
   - **Description**: Daily index measuring the emotional state of the crypto market.
   - **Key Features**: `Date`, `Value (0-100)`, `Classification (Fear/Greed)`.
   - 🔗 [Download Raw Dataset (Google Drive)](https://drive.google.com/file/d/1PgQC0tO8XN-wqkNyghWc_-mnrYv_nhSf/view?usp=sharing)

---

## 🚀 Quick Start Guide

Run the complete pipeline from raw data extraction to dashboard deployment in **four simple steps**.

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Raw Datasets
This script will automatically pull the datasets from the Google Drive links above.
```bash
python src/download_data.py
```

### 3. Process Data & Train Models
Execute the core analytical engine. This cleans the data, runs the quantitative analyses, and trains the Random Forest and K-Means models. Outputs will be saved to the `data/results/` directory.
```bash
python src/data_preprocessing.py
python src/analytics_engine.py
python src/ml_engine.py
```

### 4. Launch Interactive Dashboard
Fire up the local Streamlit server to visualize the findings.
```bash
streamlit run src/dashboard.py
```
*The dashboard will automatically open in your default browser at `http://localhost:8501`.*

---

## 📂 Project Architecture

```text
📦 sentiment-driven-crypto-intelligence
├── 📁 data/
│   ├── raw/                  # Downloaded raw datasets
│   ├── processed/            # Cleaned & feature-engineered datasets
│   └── results/              # Exported analytical CSVs and serialized ML models
├── 📁 notebooks/             # Jupyter notebooks for EDA
├── 📁 reports/               # Executive summaries
├── 📁 src/
│   ├── 📥 download_data.py      # Automated dataset extraction
│   ├── 🧹 data_preprocessing.py # Data harmonization & deterministic leverage
│   ├── 🧮 analytics_engine.py   # Quant metrics: VaR, Profit Factors, correlations
│   ├── 🤖 ml_engine.py          # ML pipelines: K-Means & Random Forest
│   └── 🖥️ dashboard.py          # Streamlit interactive dashboard
├── 📄 requirements.txt       # Project dependencies
└── 📄 README.md              # You are here!
```

---

## 📊 Key Findings & Insights

1. **The Overconfidence Trap**: Peak trader profitability actually occurs during periods of **Fear** and **Extreme Greed**. Standard **Greed** phases often act as *overconfidence traps*—emotionally-driven FOMO entries that lead to elevated retail losses.
2. **The Leverage Fallacy**: Low-leverage traders (**1x–3x**) consistently capture the vast majority of aggregate profits. High-leverage trades (**21x–50x**) experience amplified downside risk due to extreme volatility exposure and sudden liquidation cascades.
3. **Asymmetric Tail-Risk**: Positions entered during systematic market panic (**Fear**) demonstrate significantly lower downside **Value-at-Risk (VaR)**, while simultaneously maintaining much stronger risk-adjusted returns and superior profit factors.

---
