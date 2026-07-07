"""Detection metrics against labeled events.

Two granularities:
  - point level: per-bar precision / recall / F1
  - event level: an event counts as detected if any bar inside its span
    is flagged; recall is reported per event type
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def point_metrics(y_true: np.ndarray, flags: np.ndarray) -> dict:
    return {
        "precision": float(precision_score(y_true, flags, zero_division=0)),
        "recall": float(recall_score(y_true, flags, zero_division=0)),
        "f1": float(f1_score(y_true, flags, zero_division=0)),
        "flagged_bars": int(flags.sum()),
        "anomalous_bars": int(y_true.sum()),
    }


def event_recall(
    events: pd.DataFrame, eval_bars: np.ndarray, flags: np.ndarray
) -> dict:
    """Fraction of labeled events with at least one flagged bar, by type."""
    flagged = set(eval_bars[flags].tolist())
    eval_lo, eval_hi = int(eval_bars.min()), int(eval_bars.max())

    per_type: dict[str, list[int]] = {}
    for ev in events.itertuples():
        if ev.end < eval_lo or ev.start > eval_hi:
            continue  # event outside the walk-forward evaluation region
        hit = any(b in flagged for b in range(ev.start, ev.end + 1))
        per_type.setdefault(ev.event_type, []).append(int(hit))

    out = {
        t: {"detected": sum(hits), "total": len(hits), "recall": sum(hits) / len(hits)}
        for t, hits in sorted(per_type.items())
    }
    all_hits = [h for hits in per_type.values() for h in hits]
    out["all_events"] = {
        "detected": sum(all_hits),
        "total": len(all_hits),
        "recall": sum(all_hits) / len(all_hits) if all_hits else 0.0,
    }
    return out


def evaluate_model(
    name: str,
    y_bar: np.ndarray,
    events: pd.DataFrame,
    eval_bars: np.ndarray,
    flags: np.ndarray,
) -> dict:
    y_eval = y_bar[eval_bars]
    return {
        "model": name,
        "point": point_metrics(y_eval, flags),
        "events": event_recall(events, eval_bars, flags),
    }
