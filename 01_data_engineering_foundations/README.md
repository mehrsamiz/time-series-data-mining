# Module 1: Tabular & Time-Series Feature Engineering

This module covers end-to-end feature engineering pipelines across both cross-sectional tabular data and temporal time series. The focus is on proper preprocessing without data leakage and quantifying feature impact on downstream model performance.

This is the first of a series of Data Mining coursework projects; the others
are linked from my [GitHub profile](https://github.com/mehrsamiz/time-series-data-mining).

---

## Module Breakdown

### Part 1: Categorical Feature Engineering (Bank Marketing)
- **Dataset:**UCI Bank Marketing dataset, 41,188 rows × 16 columns. Missing values are encoded as the string `"unknown"`.
- **Problem:** predict which customers are likely to subscribe when contacted again.
- **What's in the notebook:**
  1. **Missing-value imputation:** — categorical `unknown`s replaced with each
     column's mode.
  2. **Column typing:** — splitting columns into continuous, binary, nominal,
     and ordinal groups, since each needs a different encoding strategy.
  3. **Binary encoding:** — `yes`/`no` → `1`/`0`.
  4. **One-hot encoding:** — for unordered categorical features (`job`,
     `marital`), with clean `column_value` naming.
  5. **Ordinal encoding:** — for features with a natural order (`education`,
     `month`, `day_of_week`, `poutcome`), mapped to explicit numeric codes
     rather than an arbitrary label encoder.
  6. **Before/after benchmarking:** — a [FLAML](https://github.com/microsoft/FLAML)
    AutoML search (LightGBM + Random Forest) is run both before and after
    the feature engineering, to quantify the impact.
- **Impact:** Structured feature engineering improved the FLAML AutoML baseline $F_1$-score to **65.4%**.


---

### Part 2: Time-Series Feature Engineering (Multi-Domain)
A from-scratch tour of the feature-engineering techniques that turn a raw
time series into model-ready inputs — applied across three real datasets
so the same techniques can be compared across domains.

- **Datasets:** Daily Minimum Temperatures (Weather), Energy Consumption (Demand), and NASDAQ-100 (Financial Markets).
- **What's covered:**
  1. **Cross-sectional vs. time-series vs. panel data** — what makes
    time-series data structurally different, and why that matters for
    feature engineering.
  2. **Exploratory analysis** — resampling, seasonal plots, rolling
    mean/median, z-score & IQR-based outlier detection, missingness
    patterns.
  3. **Decomposition** — additive vs. multiplicative decomposition, and how
    to pick between them from the residuals.
  4. **Stationarity** — the ADF test, and the transforms (log, differencing,
    seasonal differencing) that restore stationarity, applied to three
    synthetic series each representing a different structural cause of
    non-stationarity (trend, seasonality, random walk).
  5. **Lag features** — building lag features and choosing lag horizons
    systematically from the ACF plot instead of guessing.
  6. **Rolling-window features** — multi-window rolling volatility (7/14/30/90
    days) and what each window size is actually good for.
  7. **Differencing features** — an automatic multi-step difference-feature
    generator, with a comparison of trend removal vs. volatility vs.
    forecasting usefulness across steps.
  8. **Calendar / date-time features** — weekday effects, and cyclical
    (sin/cos) encoding for periodic variables (hour, day-of-week, month)
    so models don't treat "23:00" and "00:00" as far apart.
  9. **Capstone exercise** — a complete feature-engineering pipeline on 5
    years of NASDAQ-100 data: returns, rolling stats, calendar features,
    and z-score / IQR-based anomaly detection.

Every exercise includes short written analysis of *why* a given technique
or transform behaves the way it does on that dataset, not just the code
that produces it.

---

## Getting started
Run the notebooks directly from this directory:

```bash
cd 01_data_engineering_foundations
pip install -r ../requirements.txt
jupyter lab
```

## Tech stack

`pandas` · `numpy` · `scikit-learn` · `statsmodels` · `matplotlib` /
`seaborn` · [`FLAML`](https://github.com/microsoft/FLAML) (AutoML)

## Data Attribution & Licensing

The datasets included in this repository are used strictly for non-commercial educational and research evaluation purposes:

- **Bank Marketing Dataset:** Sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing) (Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
- **Daily Minimum Temperatures & Energy Consumption:** Public domain educational datasets.
- **NASDAQ-100 Market Data:** Historical market data sourced via public financial records (Yahoo Finance / Stooq) for research benchmarking.

All datasets remain the property of their respective original creators and providers.
