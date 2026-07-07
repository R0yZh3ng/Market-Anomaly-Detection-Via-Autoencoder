"""Isolation Forest baseline, thresholded the same way as the autoencoder:
anomaly score quantile taken from the training distribution."""

import numpy as np
from sklearn.ensemble import IsolationForest


def fit_isolation_forest(X_train: np.ndarray, seed: int = 0) -> IsolationForest:
    model = IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1)
    model.fit(X_train)
    return model


def anomaly_scores(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Higher = more anomalous (negated sklearn score_samples)."""
    return -model.score_samples(X)
