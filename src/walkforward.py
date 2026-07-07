"""Walk-forward retraining with per-fold z-score normalization.

The timeline is split into an initial training block followed by
consecutive test folds. For each fold, both models are retrained on all
bars strictly before the fold, the z-score scaler is refit on those same
bars, and only then is the fold scored — so neither the scaler nor the
models ever see test-period data (no leakage).
"""

from dataclasses import dataclass, field

import numpy as np

from autoencoder import empirical_threshold, reconstruction_errors, train_autoencoder
from baseline import anomaly_scores, fit_isolation_forest
from features import ZScoreScaler, make_windows, window_end_bars


@dataclass
class WalkForwardResult:
    eval_bars: np.ndarray                      # bar indices that were scored
    ae_errors: np.ndarray
    ae_flags: np.ndarray
    if_scores: np.ndarray
    if_flags: np.ndarray
    fold_bounds: list[tuple[int, int]] = field(default_factory=list)
    ae_thresholds: list[float] = field(default_factory=list)
    if_thresholds: list[float] = field(default_factory=list)


def run_walk_forward(
    features: np.ndarray,
    window: int = 30,
    initial_train_frac: float = 0.4,
    n_folds: int = 6,
    quantile: float = 0.99,
    epochs: int = 12,
    seed: int = 0,
    verbose: bool = True,
) -> WalkForwardResult:
    n_bars = features.shape[0]
    end_bars = window_end_bars(n_bars, window)

    first_test_bar = int(n_bars * initial_train_frac)
    fold_edges = np.linspace(first_test_bar, n_bars, n_folds + 1).astype(int)

    eval_bars, ae_errors, ae_flags, if_scores, if_flags = [], [], [], [], []
    result = WalkForwardResult(None, None, None, None, None)

    for k in range(n_folds):
        lo, hi = fold_edges[k], fold_edges[k + 1]
        if verbose:
            print(f"  fold {k + 1}/{n_folds}: train bars [0, {lo}), test bars [{lo}, {hi})")

        # Normalization stats from training bars only (expanding window)
        scaler = ZScoreScaler().fit(features[:lo])
        X = make_windows(scaler.transform(features), window)

        train_mask = end_bars < lo
        test_mask = (end_bars >= lo) & (end_bars < hi)
        X_train, X_test = X[train_mask], X[test_mask]

        ae = train_autoencoder(X_train, epochs=epochs, seed=seed + k)
        train_err = reconstruction_errors(ae, X_train)
        thr_ae = empirical_threshold(train_err, quantile)
        test_err = reconstruction_errors(ae, X_test)

        iforest = fit_isolation_forest(X_train, seed=seed + k)
        thr_if = empirical_threshold(anomaly_scores(iforest, X_train), quantile)
        test_scores = anomaly_scores(iforest, X_test)

        eval_bars.append(end_bars[test_mask])
        ae_errors.append(test_err)
        ae_flags.append(test_err > thr_ae)
        if_scores.append(test_scores)
        if_flags.append(test_scores > thr_if)
        result.fold_bounds.append((lo, hi))
        result.ae_thresholds.append(thr_ae)
        result.if_thresholds.append(thr_if)

    result.eval_bars = np.concatenate(eval_bars)
    result.ae_errors = np.concatenate(ae_errors)
    result.ae_flags = np.concatenate(ae_flags)
    result.if_scores = np.concatenate(if_scores)
    result.if_flags = np.concatenate(if_flags)
    return result
