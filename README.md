<h1 align="center">🚀 QuantSentinel AI</h1>

<h3 align="center">Behavioral Trading Intelligence Platform</h3>

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![PySpark](https://img.shields.io/badge/PySpark-Big%20Data-orange?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green?style=flat-square)
![Delta Lake](https://img.shields.io/badge/Delta-Lakehouse-purple?style=flat-square)
![FinBERT](https://img.shields.io/badge/NLP-FinBERT-red?style=flat-square)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-yellow?style=flat-square)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b?style=flat-square)
![Docker](https://img.shields.io/badge/DevOps-Docker-2496ED?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-brightgreen?style=flat-square)

</div>

<br/>

QuantSentinel AI is an end-to-end **Behavioral Trading Intelligence Platform** that combines quantitative finance, machine learning, explainable AI, NLP-driven sentiment analysis, and modern data engineering to uncover how trader psychology impacts profitability, leverage behavior, portfolio risk, and market dynamics.

The platform integrates **Hyperliquid perpetual futures trading data**, **Bitcoin Fear & Greed sentiment**, **historical cryptocurrency market data**, and **financial news intelligence** to generate actionable behavioral, risk, and sentiment-driven insights.

---

## Table of Contents

- [Key Findings](#-key-findings)
- [System Architecture](#-system-architecture)
- [Core Capabilities](#-core-capabilities)
- [Platform Modules](#-platform-modules)
- [Data Sources](#-data-sources)
- [Dashboard](#-dashboard)
- [Quick Start](#-quick-start)
- [Technology Stack](#-technology-stack)
- [License](#-license)

---

## 💡 Key Findings

> These are the primary behavioral insights the platform surfaces across trader segments.

**The Overconfidence Trap** — Peak profitability often occurs during Fear regimes, while Greed environments frequently become overconfidence traps that encourage poor risk management and excessive leverage usage.

**The Leverage Fallacy** — Low-leverage traders consistently outperform highly leveraged participants on a risk-adjusted basis, despite generating lower gross returns.

**Asymmetric Tail Risk** — Fear regimes demonstrate superior risk-adjusted returns and lower downside exposure compared with euphoric market environments.

**Behavioral Bias Impact** — FOMO, overconfidence, panic selling, and loss chasing significantly influence profitability outcomes and risk exposure across all trader segments.

---

## 🏗 System Architecture

<img width="1440" height="1440" alt="image" src="https://github.com/user-attachments/assets/8b7a4324-f6db-4cd7-bdb9-9edcabf17942" />

---

## 🎯 Core Capabilities

### 🧠 Behavioral Finance Intelligence

Detects and quantifies trader behavioral biases in real time:

- **Trader Persona Discovery** — unsupervised segmentation of trader archetypes
- **Behavioral Risk Scoring** — composite score per trader based on bias exposure
- **FOMO Detection** — identifies momentum-chasing entries under sentiment pressure
- **Overconfidence Detection** — flags disproportionate position sizing relative to edge
- **Loss Chasing Identification** — detects revenge trading patterns after drawdowns
- **Panic Selling Analysis** — identifies capitulation exits at unfavorable prices
- **Risk Profiling & Trader Classification** — maps traders to behavioral risk tiers

---

### 📈 Quantitative Risk Analytics

Full suite of institutional-grade risk metrics computed per trader and per portfolio:

| Metric | Description |
|--------|-------------|
| VaR | Value-at-Risk at configurable confidence levels |
| CVaR | Conditional Value-at-Risk (Expected Shortfall) |
| Sharpe Ratio | Return per unit of total volatility |
| Sortino Ratio | Return per unit of downside deviation |
| Calmar Ratio | Return relative to maximum drawdown |
| Maximum Drawdown | Peak-to-trough equity decline |
| Downside Deviation | Volatility of negative returns only |
| Rolling Risk Metrics | Time-windowed versions of all metrics above |
| Kelly Criterion | Optimal position sizing given historical edge |
| Risk Budgeting | Portfolio-level allocation by risk contribution |

---

### 🤖 Machine Learning

**Supervised Learning**

- Random Forest, XGBoost, LightGBM
- Automated model benchmarking with cross-validation
- Evaluation: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix

**Unsupervised Learning**

- K-Means trader segmentation
- Market regime detection (Bull / Bear / Sideways / High Volatility)

---

### 🔍 Explainable AI (XAI)

- SHAP global feature importance — which features drive predictions across the dataset
- SHAP local explanations — why the model made a specific prediction for a specific trader
- Feature contribution analysis and interactive explainability dashboard

---

### 📰 NLP & Market Sentiment Intelligence

**News Sources**

- CoinDesk RSS
- CoinTelegraph RSS
- Yahoo Finance Crypto
- CryptoPanic API

**NLP Pipeline**

- FinBERT sentiment analysis (finance-domain BERT)
- Lexicon-based fallback engine for low-latency inference
- Topic modeling (BERTopic / LDA)
- Named Entity Recognition (coins, exchanges, institutions, regulators)
- Market impact scoring
- Sentiment momentum, volatility, and acceleration tracking

**Intelligence Outputs**

- Bullish / Bearish signal classification
- Topic evolution and emerging trend detection
- Entity-level sentiment mapping
- Daily AI-generated market brief

---

### ⚙️ Data Engineering Platform

**Medallion Architecture**

| Layer | Contents |
|-------|----------|
| Bronze | Raw ingested data — trades, prices, sentiment, news |
| Silver | Validated, cleaned, feature-engineered data |
| Gold | Analytics-ready tables: risk metrics, sentiment aggregates, behavioral features |

**Processing Stack**

- PySpark ETL pipelines with incremental processing
- Delta Lake storage with ACID guarantees
- Schema enforcement and partition optimization
- dbt data models: fact tables, dimension tables, data lineage, automated tests

---

### 🚀 Backend & Platform Engineering

**FastAPI Services**

- `/predict` — ML prediction endpoint
- `/risk` — trader and portfolio risk metrics
- `/regime` — current and historical market regime
- `/sentiment` — sentiment scores and momentum
- `/topics` — topic model outputs
- `/entities` — named entity sentiment mapping

**Production Engineering**

- Dockerized deployment with `docker-compose`
- CI/CD via GitHub Actions
- Pipeline monitoring, data quality validation, and observability metrics
- Structured execution logging

---

## 📊 Platform Modules

| Module | Functionality |
|--------|---------------|
| Behavioral Intelligence | Trader profiling, FOMO, overconfidence, loss chasing, panic selling |
| Risk Intelligence | VaR, CVaR, Sharpe, Sortino, Calmar, Drawdown analytics |
| Portfolio Simulator | Kelly sizing, risk budgeting, equity curve simulation |
| Market Regimes | Bull, Bear, Sideways, High Volatility classification |
| NLP Intelligence | FinBERT sentiment, topic modeling, NER, impact scoring |
| Explainable AI | SHAP global and local prediction explanations |
| Data Quality Center | Missing values, duplicates, outlier detection, drift monitoring |
| Observability Center | Pipeline health, runtime metrics, throughput, API health |
| Developer Portal | Interactive FastAPI documentation and endpoint explorer |

---

## 🗄 Data Sources

### Hyperliquid Trading Dataset

Historical perpetual futures trading activity with the following fields:

```
Account | Coin | Direction | Execution Price | Size USD | Closed PnL | Timestamp
```

### Bitcoin Fear & Greed Index

```
Source: https://api.alternative.me/fng/
Fields: Sentiment Score (0–100), Classification, Historical States
```

### Cryptocurrency Market Data

```
Source: CoinGecko API
Assets: Bitcoin (BTC), Ethereum (ETH), Solana (SOL)
Fields: OHLCV, Returns, Volatility, Trend Indicators
```

### Financial News

```
Sources: CoinDesk RSS, CoinTelegraph RSS, Yahoo Finance Crypto, CryptoPanic API
Used for: Sentiment Analysis, Topic Modeling, NER, Market Impact Detection
```

---

## 📊 Dashboard

The Streamlit Intelligence Portal exposes the following modules:

| Dashboard Module | Description |
|------------------|-------------|
| Quant Metrics | Risk analytics, portfolio simulator, regime detection, SHAP explanations, model benchmarks |
| Trader Persona Intelligence | Persona rankings, behavioral risk scores, profitability breakdowns |
| Risk Intelligence Center | VaR, CVaR, drawdowns, Sharpe analytics, tail risk monitoring |
| News Intelligence | Latest articles with sentiment labels, topic clusters, entity highlights |
| Sentiment Intelligence | Momentum tracking, bull/bear ratios, sentiment volatility and acceleration |
| Topic Intelligence | Topic popularity, emerging trends, topic evolution over time |
| Entity Intelligence | Coins, exchanges, institutions, and regulators with mapped sentiment |
| Market Impact Dashboard | Critical events, impact scores, risk alerts |
| Data Quality Center | Missing values, duplicate records, outlier detection, drift monitoring |
| Pipeline Observability | Pipeline health, runtime metrics, throughput monitoring, API health |
| Developer Portal | FastAPI endpoint explorer with request/response examples |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Core Pipelines

```bash
# Data ingestion and NLP pipeline
python src/news_ingestion.py
python src/text_preprocessing.py
python src/sentiment_engine.py
python src/topic_modeling.py
python src/entity_extraction.py
python src/impact_scoring.py
python src/sentiment_trends.py

# Machine learning pipeline
python src/ml_engine.py
```

### 3. Start FastAPI Backend

```bash
uvicorn src.api:app --reload
```

API docs available at: `http://localhost:8000/docs`

### 4. Launch Dashboard

```bash
streamlit run src/dashboard.py
```

Dashboard available at: `http://localhost:8501`

### 5. Docker Deployment (Optional)

```bash
docker-compose up --build
```

---

## 🛠 Technology Stack

| Category | Technologies |
|----------|-------------|
| Language | Python 3.10+ |
| Data Engineering | PySpark, Delta Lake, dbt, Pandas, NumPy |
| Machine Learning | Scikit-Learn, XGBoost, LightGBM |
| NLP | FinBERT, BERTopic, spaCy |
| Explainability | SHAP |
| Backend | FastAPI, Uvicorn |
| Frontend | Streamlit, Plotly |
| DevOps | Docker, GitHub Actions |

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

<h3 align="center">
Built for Quantitative Finance · Behavioral Analytics · NLP Intelligence · Modern Data Engineering
</h3>
