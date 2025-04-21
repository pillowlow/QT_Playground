# 🧠 Weekly Swing Transformer

A research-driven backtesting system designed to **predict weekly ETF returns** using Transformer-based models, and simulate a simple **weekly buy-sell trading strategy**. It supports evaluation against a benchmark ETF (e.g., SPY) and logs trades, predictions, and performance metrics.

The win rate over past 10 years are around 80% to 90% 

---

## 📌 Features

- Dual-head Transformer model for weekly return prediction
- Per-ETF model training and top-weight selection (top-5 models)
- Strategy simulator with weekly rebalance and sell logic
- Predictive logs, value tracking, and benchmark comparisons
- Visual backtest results with matplotlib

---

## 🚀 Getting Started

### 1. Clone this repo

```bash
git clone <your-repo-url>
cd WEEKLY_SWING_TRANSFORMER
```

## Setup Python Environment
Make sure you have conda installed, then run:
```bash
conda env create -f environment.yml
conda activate stock_pred
python -m ipykernel install --user --name stock_pred --display-name "Python (stock_pred)"
```

## 3. Launch Notebook
Start Jupyter in VSCode or use:

```bash
jupyter notebook
```
Open agent/etf_weekly_swing.ipynb to train, predict, and backtest.

## Project Structure

.
├── agent/
│   └── etf_weekly_swing_test2.ipynb       # Main notebook
├── dataset/
│   ├── normalized_matrix/                 # Combined feature and mask matrices
│   ├── etf_prices_weekly.csv              # Raw weekly prices from Yahoo Finance
│   ├── backtest_trade_log.csv             # Log of buy/sell trades
│   └── backtest_value_log.csv             # Weekly portfolio valuation
├── model_weights/                         # Saved top-k models per ETF
├── environment.yml                        # Conda environment spec
└── README.md                              # ← You are here!

## 📊 Strategy
Each week, predict returns using saved top-5 models

Select top 2 ETFs to buy with equal capital

Sell at end of week regardless of performance

Compare with SPY as a benchmark