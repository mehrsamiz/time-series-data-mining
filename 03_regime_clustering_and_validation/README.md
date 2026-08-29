# Market-Regime Clustering & Rigorous Model Validation

Unsupervised regime discovery on NASDAQ-100, a controlled demonstration of how the *validation
protocol itself* can flip which model looks best on time-series data, and a decision-focused
evaluation of the winning model that goes well beyond accuracy.

## Why this notebook is here

Three things a research-oriented reader will care about more than a leaderboard:

1. **Distinguishing structural clustering from predictive clustering.** The market genuinely
   separates into three well-defined volatility/momentum regimes — but demonstrating (not just
   asserting) that clean geometric clusters don't automatically carry directional signal is the
   actual point of Part 1.
2. **Quantifying validation-protocol bias, not just naming it.** A shuffled K-Fold on
   autocorrelated time-series data makes Random Forest look competitive (CV accuracy 0.5107) with
   real test performance nearly 6 points lower (0.4502) — measured side-by-side against a
   correct nested `TimeSeriesSplit` protocol that never overstates it in the first place.
3. **Evaluating a weak classifier honestly.** The best model here tops out at ~54% test
   accuracy with an MCC near zero — rather than dressing that up, Part 3 is built around what a
   weak-but-not-useless model like this is actually good for (a small number of
   high-confidence, high-threshold signals) and what it isn't.

## Dataset

Daily NASDAQ-100 OHLCV data, included at [`data/NASDAQ100_data.csv`](./data/NASDAQ100_data.csv).
Chronological 70/15/15 train/validation/test split, held fixed across all three parts — no
shuffling anywhere in the pipeline.

## What's in the notebook

| Part | Content |
|---|---|
| **1** | K-means, hierarchical (4 linkages), and DBSCAN regime discovery; Purity/Entropy against next-day direction as an external validity check; financial interpretation of the 3-regime solution |
| **2** | Same 3 classifiers (Logistic Regression, Random Forest, XGBoost) under 3 validation protocols — fixed split, shuffled K-Fold, nested `TimeSeriesSplit` — plus two independent feature-scaling leakage experiments |
| **3** | Full metric suite (confusion matrix, F0.5/F2, ROC/PR-AUC, MCC, Gini, KS) for the Part-2 winner, a threshold-optimization sweep for capital preservation, and a per-regime accuracy breakdown |

## Key results

**Part 1.** Three regimes emerge cleanly from volatility/momentum features (calm bullish,
choppy/corrective, high-volatility panic), but Purity stays in a narrow 0.53–0.55 band and
Entropy ~0.99 across *every* method and parameter choice tried — structural regime and
next-day direction are essentially independent in this feature space.

**Part 2.** The validation protocol changes which models *look* viable, not which one actually
wins:

| Protocol | Random Forest: Val/CV → Test |
|---|---|
| Fixed split | 0.4424 → 0.4502 |
| Shuffled K-Fold (leaky) | **0.5107 → 0.4502** (6-pt overstatement) |
| Nested `TimeSeriesSplit` (correct) | 0.4914 → 0.4543 |

Logistic Regression wins under every protocol and posts the best real test accuracy overall
(0.5396, Protocol 3) — the only model whose validation score doesn't quietly lie to you.

**Part 3.** ROC-AUC (0.5330) and MCC (0.0071) agree: this is a weak classifier with almost no
overall class-separation. But its precision improves from 0.55 to ~0.60 at stricter decision
thresholds (0.55–0.60) — the practically useful takeaway for a real capital-preservation
strategy, and a concrete illustration of why AUC alone doesn't tell you which threshold to
trade at.

## Run it

```bash
pip install -r requirements.txt
jupyter lab notebook.ipynb
```

The notebook reads `data/NASDAQ100_data.csv` relative to its own location, so it runs as-is
from this folder.
