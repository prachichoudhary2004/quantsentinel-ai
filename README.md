# ⚡ Sentiment-Driven Crypto Intelligence System

An end-to-end quantitative and behavioral analytics platform that aligns market sentiment (Bitcoin Fear & Greed Index) with active trader execution patterns on the **Hyperliquid Perpetual DEX**.

---

## 📂 Project Architecture

```text
├── data/                    # Data storage (raw, processed, and results)
├── notebooks/               # Research notebooks
├── reports/                 # Executive reports & summaries
├── src/
│   ├── download_data.py     # ETL: Automated Google Drive data downloader
│   ├── data_preprocessing.py# ETL: Date harmonization & leverage engineering
│   ├── analytics_engine.py  # Quant: Profit factors, VaR, & correlation metrics
│   ├── ml_engine.py         # ML: K-Means clustering & RandomForest classifier
│   └── dashboard.py         # App: Streamlit dark-mode interactive terminal
└── requirements.txt         # Project dependencies
```

---

## 🚀 Quick Start Guide

Run the full pipeline from raw data extraction to dashboard deployment in **four simple steps**:

### 1. Installation
Install the required quantitative analysis and dashboard dependencies:
```bash
pip install -r requirements.txt
```

### 2. Download Data
Extract the raw Hyperliquid transaction logs and Bitcoin Fear & Greed index datasets:
```bash
python src/download_data.py
```

### 3. Preprocess & Analyze
Harmonize the dates, engineer leverage metrics, and run both the Quantitative and Machine Learning engines:
```bash
# Clean & merge datasets
python src/data_preprocessing.py

# Compute quant metrics (VaR, Profit Factors)
python src/analytics_engine.py

# Train ML models (K-Means segments & Random Forest profitability)
python src/ml_engine.py
```

### 4. Launch the Dashboard
Deploy the interactive dark-mode terminal to explore data visually:
```bash
streamlit run src/dashboard.py
```

---

## 📊 Core Analytical Insights

* **The Overconfidence Trap:** Peak trader profits occur during periods of **Fear** and **Extreme Greed**. Standard **Greed** represents an overconfidence trap where FOMO entries lead to high average losses.
* **The Leverage Fallacy:** Low-leverage (1x–3x) traders capture the bulk of aggregate profits ($6.38M), while extreme leverage (21x–50x) trades bleed capital through liquidations and trading fees.
* **Asymmetric Tail-Risk:** Entering positions during systematic market panic (**Fear**) minimizes the downside 99% Value-at-Risk (VaR) to just -$236.60 while maintaining high profit factors.

---

## 🛠️ Tech Stack

- **Analytics & ML:** `pandas`, `numpy`, `scikit-learn`
- **Visualization:** `matplotlib`, `seaborn`, `plotly`
- **Interactive UI:** `streamlit`
