# Sequence Forecasting Models — Classical ML vs. LSTM

Single-step and recursive multi-step forecasting of the NASDAQ-100 closing price, comparing
classical machine learning baselines against a PyTorch LSTM — with a controlled ablation of
hand-engineered technical indicators, and an explicit diagnosis of *why* certain model
families fail on trending financial data rather than just reporting that they do.

## Why this notebook is here

Beyond "which model has the lowest error," this project is really about three things a
research-oriented reader will care about more than the leaderboard:

1. **Correct temporal methodology** — chronological train/val/test splits, scalers and
   rolling-window features fit only on training data, and leakage-safe re-derivation of
   longer-window indicators across split boundaries.
2. **Explaining counterintuitive results from first principles**, not just reporting them.
   Tree ensembles and RBF-SVR post strongly *negative* R² here — that's diagnosed as a
   structural extrapolation-limit failure (bounded leaf outputs / local kernels vs. a
   persistently trending series), not dismissed as "the model needs tuning."
3. **Reporting a negative result honestly.** Adding four standard technical indicators
   (RSI, MACD, ATR, Bollinger Bands) doesn't help any model here, and actively triples the
   LSTM's error — documented and analyzed rather than left out of the writeup.

## Dataset

~26 years of daily NASDAQ-100 OHLCV data (2000–2026), included at
[`data/NASDAQ100_data.csv`](./data/NASDAQ100_data.csv). Chronological 70/15/15
train/validation/test split; 30-day input windows.

## What's in the notebook

| Part | Content |
|---|---|
| **0** | Data loading, leakage-safe feature engineering, scaling, and 3-D windowing |
| **1** | Classical baselines: Linear Regression, Random Forest, XGBoost, SVR (1-step) |
| **2** | A PyTorch LSTM (64 hidden units, dropout, early stopping) for 1-step forecasting |
| **3** | Adding RSI / MACD / ATR / Bollinger Bands, and re-evaluating every model |
| **4** | Recursive 5-day-ahead forecasting, with per-step error growth analysis |

## Key results (1-step forecast, test set)

| Model | RMSE | R² |
|---|---|---|
| Linear Regression | 0.043 | 0.996 |
| Random Forest | 1.770 | −5.65 |
| XGBoost | 1.773 | −5.67 |
| SVR | 2.257 | −9.82 |
| **LSTM** | **0.388** | **0.681** |

Linear regression's near-perfect R² reflects the series' near-random-walk structure (today's
price is the strongest predictor of tomorrow's) more than genuine forecasting skill — this is
confirmed directly in Part 4, where its error roughly **doubles** over a 5-day recursive
horizon, while the LSTM's grows by under **10%**. The full reasoning is in the notebook.

## Model checkpoint

The best LSTM checkpoint (by validation loss, with early stopping) is saved at
[`models/best_lstm_model.pth`](./models/best_lstm_model.pth) (19,521 parameters). Load it with:

```python
model = NASDAQ_LSTM(input_size=10, hidden_size=64, dropout_rate=0.2)
model.load_state_dict(torch.load("models/best_lstm_model.pth", weights_only=True))
model.eval()
```

## Run it

```bash
pip install -r requirements.txt
jupyter lab sequence_forecasting_models.ipynb
```

The notebook reads `data/NASDAQ100_data.csv` and writes/reads `models/best_lstm_model.pth`
relative to its own location, so it runs as-is from this folder.
