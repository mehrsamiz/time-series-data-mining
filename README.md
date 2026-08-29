# Time-Series Data Mining

Three linked coursework projects on the same NASDAQ-100 dataset, moving from raw feature
engineering through sequence forecasting to unsupervised regime discovery and rigorous model
validation. Built as part of a university Data Mining course; restructured and documented here
as a self-contained, reproducible portfolio.

Each project stands on its own — own notebook, own README, own data — but they share a single
engineered feature pipeline (base price/volume features plus RSI, MACD, ATR, and Bollinger
Bands), so results are directly comparable across the series.

## Contents

| # | Project | Focus | Techniques |
|---|---|---|---|
| 1 | [`01_data_engineering_foundations`](./01_data_engineering_foundations) | Feature engineering foundations, on categorical and time-series data | Missing-value imputation, binary/ordinal/one-hot encoding, decomposition, stationarity (ADF), ACF-driven lag selection, rolling-window volatility, cyclical calendar encoding |
| 2 | [`02_sequence_forecasting_models`](./02_sequence_forecasting_models) | Forecasting NASDAQ-100 closing price, 1-step and recursive multi-step | Classical ML baselines (Linear Regression, Random Forest, XGBoost, SVR) vs. a PyTorch LSTM; leakage-safe windowing and scaling; recursive multi-step forecasting |
| 3 | [`03_regime_clustering_and_validation`](./03_regime_clustering_and_validation) | Unsupervised market-regime discovery and validation-protocol rigor | K-means / Hierarchical / DBSCAN clustering, Purity & Entropy, shuffled-CV vs. nested `TimeSeriesSplit`, data-leakage quantification, threshold-based decision metrics (F-beta, MCC, ROC/PR-AUC) |

## What this series demonstrates

Beyond the individual results, the three projects build toward a few points that matter more to
a research-oriented reader than any single accuracy number:

- **Leakage-safe methodology, applied consistently.** Every split is chronological, every
  scaler is fit on training data only, and every rolling-window feature that spans a split
  boundary borrows only already-known trailing values — the same discipline is carried from
  the first feature-engineering notebook through to the final validation experiments.
- **Diagnosing counterintuitive results instead of reporting them at face value.** Tree
  ensembles post negative R² in Project 2 (a structural extrapolation-limit failure on a
  trending series, not a tuning issue) and get *more* competitive-looking, not less, once you
  validate them wrong in Project 3 (shuffled K-Fold inflates Random Forest's apparent accuracy
  by ~6 points versus its real test performance). Both are explained from first principles, not
  just measured.
- **Being honest about a negative result.** Project 3's best model tops out around 54% test
  accuracy with an MCC near zero — clean market regimes emerge from volatility/momentum
  features, but they carry almost no directional signal. That's reported and analyzed as the
  actual finding, rather than downplayed.

## Repository structure

```
.
├── 01_data_engineering_foundations/
│   ├── README.md
│   ├── 01-general-feature-engineering.ipynb
│   ├── 02-time-series-feature-engineering.ipynb
│   └── data/
├── 02_sequence_forecasting_models/
│   ├── README.md
│   ├── notebook.ipynb
│   ├── requirements.txt
│   ├── data/NASDAQ100_data.csv
│   └── models/best_lstm_model.pth
├── 03_regime_clustering_and_validation/
│   ├── README.md
│   ├── notebook.ipynb
│   ├── requirements.txt
│   └── data/NASDAQ100_data.csv
└── README.md
```

## Getting started

Each project has its own `requirements.txt`, since dependencies differ slightly (Project 2
needs `torch` and `xgboost`; Project 3 needs `xgboost` and `scipy`; Project 1 needs neither).
To run a specific project:

```bash
git clone <repo-url>
cd time-series-data-mining/<project-folder>
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
jupyter lab notebook.ipynb
```

Every notebook reads its data from a local `data/` folder next to it, so nothing needs to be
reconfigured — clone and run.

## Tech stack

`pandas` · `numpy` · `scikit-learn` · `statsmodels` · `xgboost` · `PyTorch` · `matplotlib` /
`seaborn` · [`FLAML`](https://github.com/microsoft/FLAML) (AutoML, Project 1)

## License

MIT. Datasets retain their original licenses — see each project's README for source and
attribution.
