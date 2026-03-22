# 🛢️ AI-Powered Well Production Optimizer

[![Tests](https://github.com/jacopodellamico03-source/ai-well-production-optimizer/actions/workflows/tests.yml/badge.svg)](https://github.com/jacopodellamico03-source/ai-well-production-optimizer/actions/workflows/tests.yml)

> Machine learning pipeline for production forecasting, anomaly detection, and choke optimization on Equinor's Volve open dataset.

**[🚀 Live Demo](https://ai-well-appuction-optimizer-5njgmn2az67lsxtfxe7g8y.streamlit.app)**

---

## Overview

This project applies AI and machine learning to real oil & gas production data from the **Equinor Volve Field** (North Sea, 2007–2016), an openly released dataset containing 9 years of production records from 3 wells.

The goal is to demonstrate how ML can drive operational decisions in the oil & gas industry — from predicting production decline to detecting equipment anomalies and optimizing well choke settings for maximum revenue.

---

## Modules

### 📈 Production Forecast
- **Arps Decline Curve Analysis** — Exponential and hyperbolic models fitted to production data
- **XGBoost Regressor** — Trained on 13 engineered features (rolling averages, lag features, GOR, watercut)
- **LSTM Neural Network** — Sequence model with 30-day lookback window
- Side-by-side comparison of all models with MAPE and R² metrics

### 🚨 Anomaly Monitor
- **Isolation Forest** — Unsupervised anomaly detection on 7 operational features
- Detects abnormal production days based on oil rate, GOR, watercut, downhole pressure, and on-stream hours
- Interactive controls for contamination threshold and number of estimators
- Anomaly score timeline with configurable detection threshold

### ⚙️ Well Optimizer
- **Gradient Boosting Regressor** — Production simulator calibrated on historical choke-production relationships
- **Bayesian Optimization (Optuna)** — Maximizes net daily revenue over choke opening range
- **Real Brent crude prices** — Integrated via EIA/FRED historical dataset (2007–2016 average per well)
- Business impact quantified: daily, monthly, and annual revenue uplift in USD

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data | Equinor Volve Field Dataset + EIA Brent Crude Prices |
| ML Models | XGBoost, LSTM (TensorFlow/Keras), Isolation Forest, GBR |
| Optimization | Optuna (Bayesian Optimization) |
| Dashboard | Streamlit |
| Visualization | Plotly |
| Language | Python 3.11 |

---

## Project Structure

```
progetto-oil-gas/
├── dashboard/
│   └── app.py              # Streamlit dashboard
├── data/
│   ├── Volve production data.xlsx
│   └── brent_prices.csv    # EIA Brent crude historical prices
├── models/
│   ├── xgboost_F14H.pkl
│   ├── lstm_F14H.keras
│   └── scaler_F14H.pkl
├── notebooks/
│   ├── 01_EDA_Volve.ipynb
│   ├── 02_Decline_Curve_Analysis.ipynb
│   ├── 03_Anomaly_Detection.ipynb
│   └── 04_Choke_Optimizer.ipynb
└── requirements.txt
```

---

## Results

| Model | MAPE | R² |
|-------|------|----|
| Arps Exponential | ~120% | ~0.3 |
| Arps Hyperbolic | ~90% | ~0.5 |
| XGBoost (test set) | ~9.7% | ~0.95 |
| LSTM (test set) | ~14.7% | ~0.91 |

**Well Optimizer (F-14 H):**
- Baseline production: ~102 Sm³/day @ $85.28/bbl (real Brent average)
- Optimal choke identified via 200 Bayesian trials
- Business impact calculated in real USD based on historical oil prices

---

## Dataset

**Equinor Volve Field** — openly released in 2018.
Wells analyzed: `NO 15/9-F-14 H`, `NO 15/9-F-12 H`, `NO 15/9-F-11 H`

**EIA Brent Crude Oil Prices** — U.S. Energy Information Administration via FRED (Federal Reserve Bank of St. Louis). Daily prices 1987–present.

---

## Local Setup

```bash
git clone https://github.com/jacopodellamico03-source/ai-well-production-optimizer.git
cd ai-well-production-optimizer
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
streamlit run dashboard/app.py
```

---

## Setup

The dataset and trained models are **not included in this repository** (excluded via `.gitignore`). Follow the steps below to get the project running locally.

### 1. Download the Volve dataset

1. Go to the Equinor Volve Data Village page:
   `https://www.equinor.com/energy/volve-data-sharing`
2. Register and download the **Volve production data** package
3. Locate the file `Volve production data.xlsx` and place it in the `data/` folder:
   ```
   data/Volve production data.xlsx
   ```

The Brent crude price file is already tracked in the repository (`data/brent_prices.csv`).

### 2. Train the XGBoost models

Run the training script to regenerate all three well models:

```bash
python scripts/train_xgboost_F12H_F11H.py
```

This script trains `XGBRegressor` (n_estimators=500, max_depth=4, lr=0.05) on wells
`NO 15/9-F-14 H`, `NO 15/9-F-12 H`, and `NO 15/9-F-11 H` and saves:

```
models/xgboost_F14H.pkl   models/scaler_F14H.pkl
models/xgboost_F12H.pkl   models/scaler_F12H.pkl
models/xgboost_F11H.pkl   models/scaler_F11H.pkl
```

> **Note:** The LSTM model (`lstm_F14H.keras`) is not re-trained by this script.
> To regenerate it, run the full notebook `notebooks/02_Decline_Curve_Analysis.ipynb`.

---

## Author

**Jacopo Dell'Amico**
Extracurricular project — Oil & Gas AI/ML  
[GitHub](https://github.com/jacopodellamico03-source)
