"""Synthetic market data generator with labeled anomaly events.

Simulates minute bars (OHLCV + order flow) under a stochastic-volatility
regime, then injects three types of labeled anomalies:

  - earnings_shock: overnight price gap with elevated volatility/volume
  - flash_crash:    rapid multi-bar drop with partial recovery, one-sided
                    order flow, and a volume spike
  - liquidity_gap:  widened spreads, evaporated volume, jumpy prices

Ground-truth labels make precision/recall evaluation possible.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

BARS_PER_DAY = 390  # 09:30-16:00 minute bars


@dataclass
class Event:
    event_type: str
    start: int  # inclusive bar index
    end: int    # inclusive bar index


def _base_market(n_days: int, rng: np.random.Generator):
    """Normal-regime returns, volume, imbalance, and spread."""
    n = n_days * BARS_PER_DAY

    # AR(1) log-volatility (per-bar return vol)
    mean_lv = np.log(4e-4)
    phi, vol_of_vol = 0.995, 0.05
    shocks = rng.standard_normal(n)
    log_vol = np.empty(n)
    log_vol[0] = mean_lv
    for t in range(1, n):
        log_vol[t] = mean_lv + phi * (log_vol[t - 1] - mean_lv) + vol_of_vol * shocks[t]
    vol = np.exp(log_vol)

    ret = vol * rng.standard_normal(n)

    # U-shaped intraday volume profile with noise, amplified by |return|
    minute = np.tile(np.arange(BARS_PER_DAY), n_days)
    x = minute / (BARS_PER_DAY - 1)
    profile = 0.6 + 1.6 * (2 * x - 1) ** 2
    volume = 1e4 * profile * rng.lognormal(0.0, 0.35, n) * (1 + 3 * np.abs(ret) / vol)

    # Order imbalance correlated with returns; spread rises with volatility
    imbalance = np.clip(0.6 * ret / vol + 0.5 * rng.standard_normal(n), -1, 1)
    spread_bps = np.maximum(
        0.5, 2.0 * (vol / np.exp(mean_lv)) * rng.lognormal(0.0, 0.25, n)
    )
    return ret, vol, volume, imbalance, spread_bps


def _inject_events(ret, vol, volume, imbalance, spread_bps, n_days, rng):
    """Overwrite slices of the base series with anomalous dynamics."""
    n = len(ret)
    events: list[Event] = []
    taken = np.zeros(n, dtype=bool)

    def reserve(start, end):
        pad_lo, pad_hi = max(0, start - 60), min(n, end + 61)
        if taken[pad_lo:pad_hi].any():
            return False
        taken[pad_lo:pad_hi] = True
        return True

    def scale(count):
        return max(1, round(count * n_days / 120))

    # Earnings shocks: gap at the open, elevated vol/volume afterwards
    for _ in range(scale(8)):
        for _attempt in range(50):
            day = rng.integers(2, n_days)
            i = day * BARS_PER_DAY
            dur = int(rng.integers(40, 80))
            if reserve(i, i + dur):
                break
        else:
            continue
        gap = rng.choice([-1, 1]) * rng.uniform(0.03, 0.08)
        ret[i] += gap
        decay = np.exp(-np.arange(dur) / (dur / 3))
        ret[i : i + dur] += vol[i : i + dur] * 2.5 * decay * rng.standard_normal(dur)
        volume[i : i + dur] *= 1 + 4 * decay
        imbalance[i : i + dur] = np.clip(
            imbalance[i : i + dur] + np.sign(gap) * 0.5 * decay, -1, 1
        )
        events.append(Event("earnings_shock", i, i + dur - 1))

    # Flash crashes: sharp drop, partial recovery, one-sided flow
    for _ in range(scale(5)):
        for _attempt in range(50):
            day = rng.integers(1, n_days)
            i = day * BARS_PER_DAY + int(rng.integers(30, BARS_PER_DAY - 60))
            c, r = int(rng.integers(5, 12)), int(rng.integers(15, 30))
            if reserve(i, i + c + r):
                break
        else:
            continue
        drop = rng.uniform(0.03, 0.06)
        ret[i : i + c] -= drop / c
        ret[i + c : i + c + r] += 0.7 * drop / r
        volume[i : i + c + r] *= rng.uniform(4, 8)
        imbalance[i : i + c] = np.clip(imbalance[i : i + c] - 0.8, -1, 1)
        imbalance[i + c : i + c + r] = np.clip(imbalance[i + c : i + c + r] + 0.4, -1, 1)
        spread_bps[i : i + c + r] *= rng.uniform(3, 6)
        events.append(Event("flash_crash", i, i + c + r - 1))

    # Liquidity gaps: volume evaporates, spreads blow out, prices jump
    for _ in range(scale(7)):
        for _attempt in range(50):
            day = rng.integers(1, n_days)
            dur = int(rng.integers(40, 90))
            i = day * BARS_PER_DAY + int(rng.integers(0, BARS_PER_DAY - dur))
            if reserve(i, i + dur):
                break
        else:
            continue
        volume[i : i + dur] *= rng.uniform(0.1, 0.25)
        spread_bps[i : i + dur] *= rng.uniform(4, 8)
        jumps = rng.random(dur) < 0.15
        ret[i : i + dur] += jumps * vol[i : i + dur] * 4 * rng.standard_normal(dur)
        events.append(Event("liquidity_gap", i, i + dur - 1))

    events.sort(key=lambda e: e.start)
    return events


def simulate_market(n_days: int = 120, seed: int = 7, start_price: float = 100.0):
    """Return (bars_df, events_df).

    bars_df columns: open, high, low, close, volume, order_imbalance,
    spread_bps, is_anomaly, event_type — indexed by a synthetic minute
    timestamp over consecutive business days.
    """
    rng = np.random.default_rng(seed)
    ret, vol, volume, imbalance, spread_bps = _base_market(n_days, rng)
    events = _inject_events(ret, vol, volume, imbalance, spread_bps, n_days, rng)
    n = len(ret)

    close = start_price * np.exp(np.cumsum(ret))
    open_ = np.empty(n)
    open_[0] = start_price
    open_[1:] = close[:-1]
    wick = np.abs(rng.standard_normal(n)) * vol * close
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick

    is_anomaly = np.zeros(n, dtype=int)
    event_type = np.array([""] * n, dtype=object)
    for ev in events:
        is_anomaly[ev.start : ev.end + 1] = 1
        event_type[ev.start : ev.end + 1] = ev.event_type

    days = pd.bdate_range("2025-01-02", periods=n_days)
    idx = pd.DatetimeIndex(
        [d + pd.Timedelta(minutes=570 + m) for d in days for m in range(BARS_PER_DAY)],
        name="timestamp",
    )
    bars = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "order_imbalance": imbalance,
            "spread_bps": spread_bps,
            "is_anomaly": is_anomaly,
            "event_type": event_type,
        },
        index=idx,
    )
    events_df = pd.DataFrame(
        [{"event_type": e.event_type, "start": e.start, "end": e.end} for e in events]
    )
    return bars, events_df
