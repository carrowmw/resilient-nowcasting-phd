#!/usr/bin/env python3
"""
Desk check for the quality-track forecasters (Phase C), including the ARIMA
fit/predict COST BENCHMARK that decides whether a seasonal order is feasible
for the full 81-sensor matrix.

Run INSIDE the container:
    python scripts/desk_checks/desk_check_forecasters.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

import numpy as np
import pandas as pd

from quality.contract import PerSensorSplit
from quality.evaluate import evaluate_split
from quality.forecasters import (
    ARIMAForecaster,
    LSTMSensorForecaster,
    PersistenceForecaster,
    SeasonalNaiveForecaster,
)
from quality.per_sensor_dataset import build_sensor_splits


def hr(title):
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def make_sensor(days=60, seed=0):
    rng = np.random.default_rng(seed)
    n = 96 * days
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    hours = np.asarray(idx.hour + idx.minute / 60.0, dtype=float)
    dow = np.asarray(idx.dayofweek, dtype=float)
    values = np.clip(
        60
        + 45 * np.sin(2 * np.pi * (hours - 6) / 24.0)
        + 10 * (dow >= 5)
        + rng.normal(0, 4, n),
        0, None,
    )
    values[3000:3200] = np.nan  # one 50 h outage
    series = pd.Series(values, index=idx, name="SYN__desk")
    mask = pd.Series(False, index=idx)
    return build_sensor_splits(series, mask, window=24, horizon=8)


def main():
    ctx, splits = make_sensor()
    print(f"windows: train={len(splits['train'])}, val={len(splits['val'])}, "
          f"test={len(splits['test'])}")

    hr("PART 1 — baseline sanity ladder (test RMSE, raw counts)")
    scores = {}
    for f in (PersistenceForecaster(),
              SeasonalNaiveForecaster(seasonality_days=1),
              SeasonalNaiveForecaster(seasonality_days=7)):
        f.fit(splits["train"], splits["val"])
        m = evaluate_split(splits["test"], f.predict(splits["test"]))
        scores[f.name] = m["rmse_raw"]
        diag = getattr(f, "diagnostics", {})
        print(f"  {f.name:14s} RMSE={m['rmse_raw']:7.2f}  MAE={m['mae_raw']:7.2f}  {diag}")
    assert scores["snaive_1d"] < scores["persistence"], \
        "daily seasonal naive must beat persistence on a daily-cycle signal"

    hr("PART 2 — LSTM (uv, 3 epochs, cpu)")
    lstm = LSTMSensorForecaster(feature_groups=("count",), hidden_dim=32, num_layers=1,
                                dropout=0.0, lr=0.003, batch_size=256, epochs=3,
                                patience=3, seed=0, device="cpu")
    t0 = time.time()
    report = lstm.fit(splits["train"], splits["val"])
    m = evaluate_split(splits["test"], lstm.predict(splits["test"]))
    print(f"  epochs={report.epochs_run}  best_val={report.best_val_loss:.4f}  "
          f"RMSE={m['rmse_raw']:.2f}  wall={time.time() - t0:.1f}s")

    hr("PART 3 — ARIMA cost benchmark (drives the seasonal-order decision)")
    variants = [
        ("ARIMA(1,0,1)", (1, 0, 1), (0, 0, 0, 0)),
        ("ARIMA(2,0,2)", (2, 0, 2), (0, 0, 0, 0)),
        ("SARIMA(1,0,1)(1,0,1,96)", (1, 0, 1), (1, 0, 1, 96)),
    ]
    n_probe = 200  # forecast origins measured per variant
    thin = PerSensorSplit(
        context=ctx, split="test",
        window_start_idx=splits["test"].window_start_idx[:n_probe],
        seq_id=splits["test"].seq_id[:n_probe],
    )
    for label, order, seasonal in variants:
        f = ARIMAForecaster(order=order, seasonal_order=seasonal, max_eval_history=2688)
        t0 = time.time()
        try:
            f.fit(splits["train"], splits["val"])
            fit_s = time.time() - t0
            t1 = time.time()
            preds = f.predict(thin)
            pred_s = time.time() - t1
            m = evaluate_split(thin, preds)
            per_1k = pred_s / n_probe * 1000
            full_est = fit_s + per_1k * len(splits["test"]) / 1000
            print(f"  {label:26s} fit={fit_s:6.1f}s  predict={per_1k:6.2f}s/1k-origins  "
                  f"RMSE={m['rmse_raw']:7.2f}  est. full-test wall={full_est:6.1f}s/sensor")
        except Exception as exc:
            print(f"  {label:26s} FAILED after {time.time() - t0:.1f}s: "
                  f"{type(exc).__name__}: {exc}")
    print("\n  Decision input: multiply est. wall by 81 sensors × 2 horizons × "
          "variants.\n  If SARIMA(s=96) exceeds the budget, the config keeps "
          "seasonal_order=(0,0,0,0)\n  and seasonality is covered by the seasonal-naive "
          "baselines (documented in the paper).")

    hr("DESK CHECK PASSED")


if __name__ == "__main__":
    main()
