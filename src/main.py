"""End-to-end pipeline: simulate -> features -> walk-forward -> evaluate.

Usage: python src/main.py [--days 120] [--window 30] [--quantile 0.99] ...
Writes the dataset to data/, metrics and figures to results/.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate import evaluate_model
from features import compute_features
from plots import plot_error_timeline, plot_metric_comparison, plot_price_flags
from simulate import simulate_market
from walkforward import run_walk_forward

ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", type=int, default=120, help="trading days to simulate")
    p.add_argument("--window", type=int, default=30, help="rolling window length (bars)")
    p.add_argument("--quantile", type=float, default=0.98,
                   help="training-error quantile used as the anomaly threshold")
    p.add_argument("--folds", type=int, default=6, help="walk-forward test folds")
    p.add_argument("--initial-train", type=float, default=0.4,
                   help="fraction of bars in the initial training block")
    p.add_argument("--epochs", type=int, default=12, help="autoencoder epochs per fold")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    data_dir, results_dir = ROOT / "data", ROOT / "results"
    data_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    print(f"Simulating {args.days} trading days of minute bars (seed {args.seed})...")
    bars, events = simulate_market(n_days=args.days, seed=args.seed)
    bars.to_csv(data_dir / "market.csv")
    events.to_csv(data_dir / "events.csv", index=False)
    print(f"  {len(bars)} bars, {len(events)} labeled events "
          f"({bars['is_anomaly'].mean():.1%} anomalous bars)")

    features = compute_features(bars)

    print("Walk-forward retraining...")
    result = run_walk_forward(
        features,
        window=args.window,
        initial_train_frac=args.initial_train,
        n_folds=args.folds,
        quantile=args.quantile,
        epochs=args.epochs,
        seed=args.seed,
    )

    y_bar = bars["is_anomaly"].to_numpy()
    report_ae = evaluate_model("autoencoder", y_bar, events, result.eval_bars, result.ae_flags)
    report_if = evaluate_model("isolation_forest", y_bar, events, result.eval_bars, result.if_flags)

    report = {
        "config": vars(args),
        "n_bars": len(bars),
        "n_events": len(events),
        "models": [report_ae, report_if],
    }
    (results_dir / "metrics.json").write_text(json.dumps(report, indent=2))

    print("\n=== Point-level metrics (per-bar, walk-forward test region) ===")
    print(f"{'model':<18}{'precision':>10}{'recall':>10}{'f1':>10}")
    for r in (report_ae, report_if):
        pt = r["point"]
        print(f"{r['model']:<18}{pt['precision']:>10.3f}{pt['recall']:>10.3f}{pt['f1']:>10.3f}")

    print("\n=== Event-level recall (event detected if any bar flagged) ===")
    types = [t for t in report_ae["events"] if t != "all_events"] + ["all_events"]
    print(f"{'event type':<18}{'autoencoder':>14}{'iso. forest':>14}")
    for t in types:
        a, b = report_ae["events"][t], report_if["events"][t]
        print(f"{t:<18}{a['detected']:>7}/{a['total']:<3} {a['recall']:>4.0%}"
              f"{b['detected']:>8}/{b['total']:<3} {b['recall']:>4.0%}")

    plot_error_timeline(result, events, results_dir / "error_timeline.png",
                        quantile=args.quantile)
    plot_price_flags(bars, result, results_dir / "price_flags.png")
    plot_metric_comparison(report_ae, report_if, results_dir / "model_comparison.png")
    print(f"\nWrote metrics.json and figures to {results_dir}/")


if __name__ == "__main__":
    np.seterr(over="raise")
    main()
