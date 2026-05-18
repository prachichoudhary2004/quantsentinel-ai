# Sentiment-Driven Crypto Trading Intelligence System ⚡

Welcome to the **Sentiment-Driven Crypto Trading Intelligence System**—an end-to-end quantitative data science and behavioral finance project built to explore the structural relationships between market sentiment (Bitcoin Fear & Greed Index) and active trader execution patterns on the **Hyperliquid Perpetual DEX**.

**Objective:** The goal of this system is to identify how trader profitability, leverage usage, and behavioral risk change under different Bitcoin market sentiment regimes. By understanding these dynamics, we construct risk-mitigation policies and predictive execution signals.

This repository is designed with a **clean, modular structure**. It analyzes over 211,000 Hyperliquid transaction records across multiple active trader wallets, aligning them with daily market sentiment scores to engineer leverage metrics, run trader profile segmentations, train profitability predictive classifiers, and deploy an interactive dashboard terminal.

---

## 📂 Project Architecture

```
├── data/
│   ├── raw/                 # Raw datasets downloaded from Google Drive
│   ├── processed/           # Aligned, preprocessed, and feature-engineered CSVs
│   └── results/             # Aggregated quant outputs, clustering segments, and ML metrics
├── notebooks/
│   └── sentiment_trading_analysis.ipynb  # Quantitative research Jupyter Notebook
├── reports/
│   └── executive_summary.md # Quantitative research report & strategic recommendations
├── src/
│   ├── download_data.py     # ETL: Automated Google Drive data downloader
│   ├── data_preprocessing.py# ETL: Date harmonization, alignment, and leverage engineering
│   ├── analytics_engine.py  # Quant: Computes Profit Factors, VaR, correlations, and frequency sizing
│   ├── ml_engine.py         # ML: Runs K-Means trader clustering & Random Forest profitability modeling
│   └── dashboard.py         # App: Interactive dark-mode Streamlit Dashboard
├── requirements.txt         # Project dependency manifest
└── README.md                # System documentation (This file)
```

---

## 🛠️ Tech Stack & Requirements

The system is built entirely in **Python** using standard high-performance data libraries:
- **Core Analytics:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`, `plotly`
- **Machine Learning:** `scikit-learn`
- **Dashboard Application:** `streamlit`
- **Networking:** `requests`

---

## 🚀 Execution & Setup Guide

Follow these sequential steps to run the entire data pipeline, machine learning models, and interactive dashboard:

### 1. Clone the Workspace & Install Dependencies
First, install all necessary libraries. Run the following command from the root of the repository:
```bash
pip install -r requirements.txt
```
*(Note: If `requirements.txt` is missing, you can run `pip install pandas numpy matplotlib seaborn plotly scikit-learn streamlit requests`)*

### 2. Download Raw Datasets
Execute the ETL downloader script to fetch both datasets (Hyperliquid fills and FGI sentiment index) directly from Google Drive:
```bash
python src/download_data.py
```
This downloads and saves the datasets inside the `data/raw/` folder:
*   `data/raw/hyperliquid_trader_data.csv` (~47.5 MB)
*   `data/raw/bitcoin_fear_greed_index.csv` (~90 KB)

### 3. Harmonize and Preprocess Datasets
Clean the data, parse day-first IST timestamps, align dates, and engineer synthetic leverage metrics:
```bash
python src/data_preprocessing.py
```
This merges both datasets on date, computes notional PnL%, leverage ROE, win flags, and trade directions, and saves the result as `data/processed/merged_trader_data.csv` (211,224 rows, 30 columns).

### 4. Execute Quantitative Analytics Engine
Run the statistical computations across the 8 key quantitative focus areas (tail-risk VaR, profit factors, correlation coefficients, sizing, and asset performance):
```bash
python src/analytics_engine.py
```
This generates aggregated summaries inside the `data/results/` directory, saving them as lightweight CSVs (decoupling heavy data loading from visual frontends).

### 5. Run Behavioral Clustering & ML Models
Train the K-Means unsupervised model to segment wallets into personas, and train the RandomForest trade profitability classifier:
```bash
python src/ml_engine.py
```
This script yields:
*   `data/results/trader_segments.csv` (unsupervised personas)
*   `data/results/feature_importance.csv` (Random Forest model weights)
*   `data/results/model_evaluation.json` (**85.0% Accuracy** and **0.90 ROC-AUC** metrics)

### 6. Launch the Interactive Dashboard
Boot up the high-fidelity Streamlit app to explore the dynamic visualizations, wallet audits, and predictive trade sandbox in real-time:
```bash
streamlit run src/dashboard.py
```
Open the local URL (usually `http://localhost:8501`) in your browser to interact with the system!

---

## 📈 Core Analytical Insights Summary

### 1. The 'Smile Curve' Inefficiency
Realized trader returns exhibit a U-shaped distribution. Peak profits are captured during **Fear** (Total realized PnL: **$3.35M**, Profit Factor: **6.65**) and **Extreme Greed** (Total realized PnL: **$2.71M**, Profit Factor: **11.02**). In contrast, **Greed** represents an overconfidence trap characterized by FOMO entries, yielding higher average losses (-$181.97) and high trade-level loss ratios (23.1%).

### 2. The High Leverage Trap
Low-leverage (1x-3x) executions generate the bulk of aggregate profits (**$6.38M** realized PnL), whereas Extreme-leverage (21x-50x) trades yield a meager **$69.7K** due to tighter liquidation boundaries, force-closes, and aggressive transaction fee bleed. The correlation between leverage and dollar returns is consistently negative across every market regime.

### 3. Tail-Risk Asymmetry
Buying during **Fear** is structurally the safest asymmetric risk regime. The downside 99% Value at Risk (VaR) is minimized to only **-$236.60**, while average profit factor climbs to **6.65**, proving that entering positions during systematic market panic provides maximum risk insulation.

---

## 🎓 Executive Recommendations

1.  **Sentiment-Based Dynamic Leverage Caps:** Cap wallet leverage at **2.0x** when FGI > 75 (Greed/Extreme Greed) to protect users from sudden deleveraging liquidations. Expand leverage limits up to **5.0x** during FGI < 30 (Fear) to capitalize on high-asymmetric margin setups.
2.  **Risk Bounded Position Sizing:** Set dynamic trade limits linked directly to FGI regimes, sizing down significantly in Greed and scaling up in Fear.
3.  **Auditing PMs & Copy Trading:** Utilize our K-Means framework to audit portfolio managers, reallocating capital away from 'High-Leverage Speculators' and towards 'Institutional Whales' to stabilize long-term ecosystem TVL.
