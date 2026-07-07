# Market Anomaly Detection via Autoencoder

Detects anomalous market conditions with a fully-connected autoencoder trained
on rolling windows of normalized OHLCV and order flow features. Observations
whose reconstruction error exceeds a threshold derived from the empirical
training-loss distribution are flagged as anomalies, and detection quality is
benchmarked against an Isolation Forest baseline on labeled events — earnings
shocks, flash crashes, and liquidity gaps.

**Stack:** Python, PyTorch, Pandas, scikit-learn

## Method

1. **Data** — `src/simulate.py` generates 120 trading days of minute bars
   (OHLCV, order imbalance, bid-ask spread) under a stochastic-volatility
   regime and injects ground-truth-labeled anomalies: earnings shocks
   (overnight gaps with elevated volatility/volume), flash crashes (sharp
   multi-bar drops with partial recovery and one-sided order flow), and
   liquidity gaps (evaporated volume, blown-out spreads, jumpy prices).
2. **Features** — per-bar log returns, high-low range, log volume, order
   imbalance, log spread, and signed volume, assembled into flattened rolling
   windows (30 bars) for the fully-connected autoencoder.
3. **Autoencoder** — symmetric FC encoder/decoder (input → 128 → 64 → 16 → 64
   → 128 → input), trained with MSE loss. The anomaly threshold is the 98th
   percentile of the *training* reconstruction-error distribution, so no test
   labels or scores inform it.
4. **Walk-forward retraining** — the timeline is split into an initial
   training block and six consecutive test folds. For each fold the z-score
   scaler, the autoencoder, and the Isolation Forest are refit on all bars
   strictly before the fold (expanding window), preventing any normalization
   or model leakage from the test period.
5. **Evaluation** — per-bar precision/recall/F1 plus event-level recall (an
   event counts as detected if any bar in its span is flagged), against the
   Isolation Forest baseline thresholded the same way.

## Results (seed 7, default config)

Point-level metrics over the walk-forward test region:

| model | precision | recall | F1 |
|---|---|---|---|
| autoencoder | 0.523 | 0.398 | 0.452 |
| isolation forest | 0.339 | 0.231 | 0.275 |

Event-level recall (16 labeled events in the test region):

| event type | autoencoder | isolation forest |
|---|---|---|
| earnings shock | 7/7 (100%) | 1/7 (14%) |
| flash crash | 3/3 (100%) | 3/3 (100%) |
| liquidity gap | 5/6 (83%) | 3/6 (50%) |
| **all events** | **15/16 (94%)** | **7/16 (44%)** |

Figures written to `results/`: reconstruction-error timeline with labeled
event spans (`error_timeline.png`), price with flagged bars
(`price_flags.png`), and the model comparison (`model_comparison.png`).

## Usage

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/main.py                 # full pipeline
.venv/bin/python src/main.py --help          # all knobs (days, window, quantile, folds, ...)
```

Outputs: `data/market.csv`, `data/events.csv`, `results/metrics.json`, and
the three figures.

## Layout

```
src/
  simulate.py     synthetic market data with labeled anomaly events
  features.py     bar features, train-only z-score scaling, rolling windows
  autoencoder.py  FC autoencoder, reconstruction error, empirical threshold
  baseline.py     Isolation Forest baseline
  walkforward.py  expanding-window walk-forward retraining
  evaluate.py     point- and event-level metrics
  plots.py        result figures
  main.py         end-to-end pipeline
```
