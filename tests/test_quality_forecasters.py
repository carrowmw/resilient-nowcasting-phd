# Tests for src/quality/forecasters.py + interpolation + periodogram.
import numpy as np
import pandas as pd
import pytest

from quality.forecasters import (
    ARIMAForecaster,
    LSTMSensorForecaster,
    PersistenceForecaster,
    SeasonalNaiveForecaster,
)
from quality.interpolation import apply_interpolation
from quality.per_sensor_dataset import build_sensor_splits
from quality.periodogram import compute_lomb_scargle, dominant_periods
from quality.registry import build_forecaster
from quality.evaluate import evaluate_split, predictions_frame


@pytest.fixture(scope="module")
def sensor():
    rng = np.random.default_rng(7)
    n = 96 * 40  # 40 days
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    hours = np.asarray(idx.hour + idx.minute / 60.0, dtype=float)
    values = 50 + 40 * np.sin(2 * np.pi * (hours - 6) / 24.0) + rng.normal(0, 2, n)
    values = np.clip(values, 0, None)
    values[2000:2100] = np.nan
    series = pd.Series(values, index=idx, name="SYN__daily")
    mask = pd.Series(False, index=idx)
    ctx, splits = build_sensor_splits(series, mask, window=24, horizon=8)
    return ctx, splits


def test_persistence_exact(sensor):
    ctx, splits = sensor
    f = PersistenceForecaster()
    f.fit(splits["train"], splits["val"])
    preds = f.predict(splits["test"])
    np.testing.assert_allclose(preds, ctx.values_scaled[splits["test"].origin_idx])


def test_seasonal_naive_daily_beats_persistence_on_daily_signal(sensor):
    """On a clean daily sinusoid, snaive(1d) must dominate persistence at 2 h."""
    ctx, splits = sensor
    sn = SeasonalNaiveForecaster(seasonality_days=1)
    sn.fit(splits["train"], splits["val"])
    pe = PersistenceForecaster()
    pe.fit(splits["train"], splits["val"])
    m_sn = evaluate_split(splits["test"], sn.predict(splits["test"]))
    m_pe = evaluate_split(splits["test"], pe.predict(splits["test"]))
    assert m_sn["rmse_raw"] < m_pe["rmse_raw"]
    assert sn.diagnostics["test_fallback_frac"] < 0.2


def test_seasonal_naive_no_future_reference(sensor):
    ctx, splits = sensor
    sn = SeasonalNaiveForecaster(seasonality_days=1)
    s = int(round(1 * ctx.steps_per_day()))
    lag_idx = splits["test"].target_idx - s
    assert (lag_idx <= splits["test"].origin_idx).all()


def test_seasonal_naive_rejects_seasonality_below_horizon(sensor):
    ctx, splits = sensor
    sn = SeasonalNaiveForecaster(seasonality_days=0.01)  # ~1 step < horizon 8
    sn.fit(splits["train"], splits["val"])
    with pytest.raises(AssertionError, match="seasonal period"):
        sn.predict(splits["test"])


def test_arima_fits_and_predicts(sensor):
    ctx, splits = sensor
    f = ARIMAForecaster(order=(1, 0, 1), min_fit_steps=96, max_eval_history=672)
    report = f.fit(splits["train"], splits["val"])
    assert report.fitted and report.notes["fit_steps"] >= 96
    # Predict a subset for speed: build a thinned view of the test split.
    from quality.contract import PerSensorSplit
    thin = PerSensorSplit(
        context=ctx, split="test",
        window_start_idx=splits["test"].window_start_idx[::50],
        seq_id=splits["test"].seq_id[::50],
    )
    preds = f.predict(thin)
    assert len(preds) == len(thin)
    assert np.isfinite(preds).all()
    # ARIMA on a strong sinusoid must at least beat predicting the train mean (~0 scaled).
    rmse_arima = np.sqrt(np.mean((preds - thin.y_scaled) ** 2))
    rmse_mean = np.sqrt(np.mean(thin.y_scaled ** 2))
    assert rmse_arima < rmse_mean


def test_lstm_trains_and_beats_mean(sensor):
    ctx, splits = sensor
    f = LSTMSensorForecaster(
        feature_groups=("count",), hidden_dim=16, num_layers=1, dropout=0.0,
        lr=0.005, batch_size=256, epochs=3, patience=3, seed=1, device="cpu",
    )
    report = f.fit(splits["train"], splits["val"])
    assert report.fitted and report.epochs_run >= 1
    preds = f.predict(splits["test"])
    assert len(preds) == len(splits["test"])
    rmse = np.sqrt(np.mean((preds - splits["test"].y_scaled) ** 2))
    rmse_mean = np.sqrt(np.mean(splits["test"].y_scaled ** 2))
    assert rmse < rmse_mean


def test_registry_builds_all_types():
    specs = {
        "persistence": {"type": "persistence"},
        "snaive_daily": {"type": "seasonal_naive", "seasonality_days": 1},
        "arima": {"type": "arima", "order": [1, 0, 1]},
        "lstm_uv": {"type": "lstm", "feature_groups": ["count"], "hidden_dim": 8,
                    "num_layers": 1, "dropout": 0.0, "lr": 0.01, "batch_size": 64,
                    "epochs": 1, "patience": 1},
    }
    for key, spec in specs.items():
        f = build_forecaster(key, spec, device="cpu")
        assert f.name == key


def test_registry_rejects_unknown_type():
    with pytest.raises(AssertionError, match="unknown forecaster type"):
        build_forecaster("bad", {"type": "prophet"})


def test_predictions_frame_schema(sensor):
    ctx, splits = sensor
    f = PersistenceForecaster()
    f.fit(splits["train"], splits["val"])
    preds = f.predict(splits["test"])
    frame = predictions_frame(splits["test"], preds, "persistence", 8, "none")
    assert len(frame) == len(splits["test"])
    assert (frame["y_pred_raw"] >= 0).all()
    assert frame["sensor_id"].nunique() == 1


def test_evaluate_split_real_only_metrics(sensor):
    ctx, splits = sensor
    # Mark half the targets as interpolated and check metric split.
    test = splits["test"]
    ctx.interpolated_mask[test.target_idx[: len(test) // 2]] = True
    try:
        f = PersistenceForecaster()
        preds = f.predict(test)
        m = evaluate_split(test, preds)
        assert m["n_real_targets"] < m["n_windows"]
        assert np.isfinite(m["rmse_raw_real_only"])
    finally:
        ctx.interpolated_mask[:] = False


# ---------------------------------------------------------------- interpolation

DAY_GAP = slice(328, 332)  # 10:00–11:00 on day 4 — a short daytime gap


def make_gappy_series():
    idx = pd.date_range("2024-01-01", periods=96 * 5, freq="15min")
    values = np.full(len(idx), 10.0)
    night = np.asarray((idx.hour >= 0) & (idx.hour < 6))
    values[night] = np.nan
    values[DAY_GAP] = np.nan
    return pd.Series(values, index=idx, name="GAPPY__x")


def test_night_zero_fills_only_night():
    s = make_gappy_series()
    filled, mask = apply_interpolation(s, "night_zero")
    assert (filled[mask.to_numpy() & np.asarray(filled.index.hour < 6)] == 0).all()
    # Daytime gap untouched.
    assert filled.iloc[DAY_GAP].isna().all()


def test_linear_respects_max_gap():
    s = make_gappy_series()
    filled, mask = apply_interpolation(s, "linear", max_gap_steps=8)
    assert not filled.iloc[DAY_GAP].isna().any()      # 4-step gap filled
    assert filled.isna().sum() > 0                    # 24-step night gaps not filled


def test_interpolation_never_alters_observed():
    s = make_gappy_series()
    for method in ("night_zero", "linear", "spline", "knn"):
        filled, _ = apply_interpolation(s, method)
        observed = s.notna()
        np.testing.assert_allclose(filled[observed], s[observed])


def test_spline_sparse_series_graceful_fallback():
    idx = pd.date_range("2024-01-01", periods=20, freq="15min")
    s = pd.Series(np.nan, index=idx, name="sparse__x")
    s.iloc[0] = 5.0
    s.iloc[5] = 10.0
    s.iloc[10] = 7.0  # exactly 3 points (<= spline_order 3)
    filled, mask = apply_interpolation(s, "spline", spline_order=3)
    assert mask.sum() == 0
    assert filled.notna().sum() == 3


def test_knn_fills_everything_nonnegative():
    s = make_gappy_series()
    filled, mask = apply_interpolation(s, "knn", knn_neighbors=5)
    assert filled.notna().all()
    assert (filled >= 0).all()
    assert mask.sum() == s.isna().sum()


def test_knn_sparse_series_adaptive_k():
    idx = pd.date_range("2024-01-01", periods=20, freq="15min")
    s = pd.Series(np.nan, index=idx, name="sparse__x")
    s.iloc[0] = 5.0
    s.iloc[5] = 10.0
    s.iloc[10] = 7.0  # 3 points < k=5
    filled, mask = apply_interpolation(s, "knn", knn_neighbors=5)
    assert filled.notna().all()
    assert (filled >= 0).all()
    assert mask.sum() == 17


# ----------------------------------------------------------------- periodogram

def test_lomb_scargle_finds_24h_peak_despite_gaps():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2024-01-01", periods=96 * 30, freq="15min")
    hours = np.asarray((idx - idx[0]) / pd.Timedelta(hours=1), dtype=float)
    values = 10 + 5 * np.sin(2 * np.pi * hours / 24.0) + rng.normal(0, 0.5, len(idx))
    values[np.flatnonzero(rng.random(len(idx)) < 0.3)] = np.nan  # 30% missing
    s = pd.Series(values, index=idx, name="P__x")
    pg = compute_lomb_scargle(s, max_period_hours=100.0)
    peaks = dominant_periods(pg["period_hours"], pg["power"], top_k=1)
    assert peaks[0]["period_hours"] == pytest.approx(24.0, rel=0.05)
