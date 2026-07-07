"""Feature engineering: bar-level features, z-score scaling, rolling windows.

The scaler is always fit on training bars only and applied out-of-sample,
so no test-period statistics ever leak into normalization.
"""

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "log_return",
    "hl_range",
    "log_volume",
    "order_imbalance",
    "log_spread",
    "signed_volume",
]


def compute_features(bars: pd.DataFrame) -> np.ndarray:
    """Per-bar feature matrix of shape (T, F) from OHLCV + order flow."""
    close = bars["close"].to_numpy()
    log_return = np.zeros(len(bars))
    log_return[1:] = np.diff(np.log(close))
    hl_range = (bars["high"] - bars["low"]).to_numpy() / close
    log_volume = np.log(bars["volume"].to_numpy())
    imbalance = bars["order_imbalance"].to_numpy()
    log_spread = np.log(bars["spread_bps"].to_numpy())
    signed_volume = imbalance * log_volume
    return np.column_stack(
        [log_return, hl_range, log_volume, imbalance, log_spread, signed_volume]
    )


class ZScoreScaler:
    """Column-wise z-score normalization fit on a training slice only."""

    def fit(self, X: np.ndarray) -> "ZScoreScaler":
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-12
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_


def make_windows(X: np.ndarray, window: int) -> np.ndarray:
    """Flatten rolling windows: row i covers bars [i, i+window) and is
    attributed to its last bar, i + window - 1."""
    T, F = X.shape
    view = np.lib.stride_tricks.sliding_window_view(X, window, axis=0)
    return view.transpose(0, 2, 1).reshape(T - window + 1, window * F)


def window_end_bars(n_bars: int, window: int) -> np.ndarray:
    """Bar index each window row is attributed to."""
    return np.arange(window - 1, n_bars)
