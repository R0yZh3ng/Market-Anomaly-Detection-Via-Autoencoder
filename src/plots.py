"""Result figures: error timeline, price with flags, model comparison."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

AE_COLOR = "#2a78d6"      # blue — autoencoder
IF_COLOR = "#1baf7a"      # aqua — isolation forest
EVENT_COLORS = {
    "earnings_shock": "#eda100",
    "flash_crash": "#e34948",
    "liquidity_gap": "#4a3aa7",
}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "text.color": INK,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.titlecolor": INK,
        "legend.frameon": False,
        "figure.dpi": 130,
    }
)


def _shade_events(ax, events, lo, hi, labeled=None):
    labeled = set() if labeled is None else labeled
    for ev in events.itertuples():
        if ev.end < lo or ev.start > hi:
            continue
        label = ev.event_type.replace("_", " ") if ev.event_type not in labeled else None
        labeled.add(ev.event_type)
        ax.axvspan(ev.start, ev.end, color=EVENT_COLORS[ev.event_type], alpha=0.18,
                   linewidth=0, label=label)
    return labeled


def plot_error_timeline(result, events, path, quantile=0.98):
    """AE reconstruction error over the walk-forward test region, with
    per-fold thresholds and labeled event spans."""
    fig, ax = plt.subplots(figsize=(12, 4.2))
    lo, hi = int(result.eval_bars.min()), int(result.eval_bars.max())

    _shade_events(ax, events, lo, hi)
    ax.plot(result.eval_bars, result.ae_errors, color=AE_COLOR, linewidth=0.7,
            label="reconstruction error")
    for (flo, fhi), thr in zip(result.fold_bounds, result.ae_thresholds):
        ax.hlines(thr, flo, fhi, color=INK_2, linestyle="--", linewidth=1.2)
    ax.hlines([], [], [], color=INK_2, linestyle="--", linewidth=1.2,
              label=f"fold threshold (train {100 * quantile:g}th pct)")

    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_xlabel("bar index (minutes)")
    ax.set_ylabel("mean squared reconstruction error")
    ax.set_title("Autoencoder reconstruction error — walk-forward test folds")
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_price_flags(bars, result, path):
    """Price over the evaluated region with AE-flagged bars marked."""
    fig, ax = plt.subplots(figsize=(12, 4.2))
    lo, hi = int(result.eval_bars.min()), int(result.eval_bars.max())
    close = bars["close"].to_numpy()
    x = np.arange(lo, hi + 1)

    ax.plot(x, close[lo : hi + 1], color=INK_2, linewidth=0.7, label="close")
    flagged = result.eval_bars[result.ae_flags]
    ax.scatter(flagged, close[flagged], s=9, color=AE_COLOR, zorder=3,
               label="autoencoder flag")

    ax.set_xlim(lo, hi)
    ax.set_xlabel("bar index (minutes)")
    ax.set_ylabel("price")
    ax.set_title("Price with autoencoder anomaly flags")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_metric_comparison(report_ae, report_if, path):
    """Point precision/recall/F1 plus event recall, AE vs Isolation Forest."""
    metrics = ["precision", "recall", "f1"]
    ae_vals = [report_ae["point"][m] for m in metrics]
    if_vals = [report_if["point"][m] for m in metrics]
    ae_vals.append(report_ae["events"]["all_events"]["recall"])
    if_vals.append(report_if["events"]["all_events"]["recall"])
    labels = ["precision\n(point)", "recall\n(point)", "F1\n(point)", "recall\n(event)"]

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7, 4))
    for dx, vals, color, name in [
        (-w / 2, ae_vals, AE_COLOR, "Autoencoder"),
        (w / 2, if_vals, IF_COLOR, "Isolation Forest"),
    ]:
        bars_ = ax.bar(x + dx, vals, width=w - 0.04, color=color, label=name)
        for rect, v in zip(bars_, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.015, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8, color=INK_2)

    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("score")
    ax.set_title("Detection quality — autoencoder vs Isolation Forest")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
