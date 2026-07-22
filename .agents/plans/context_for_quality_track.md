# Context for Architect: Quality-Aware Per-Sensor Track (MDPI Sensors rebuild)

**Generated:** 2026-07-22
**Branch:** `feat/quality-track`
**Purpose:** Rebuild of the MDPI Sensors paper's research (data quality vs
forecasting performance, Newcastle UO pedestrian flows) on the RIOT codebase,
addressing peer review. Companion to `context_for_architect.md` (Ch5 negative
baseline) — this track is deliberately decoupled from the GNN harness.

Source artifacts: `temp/mdpi_sensors.pdf` (submitted paper),
`temp/old_codebase_digest.txt` (legacy phd-project code).
Plan of record: `/root/.claude/plans/sequential-popping-wind.md` (approved 2026-07-22).

---

## Architecture

Research unit = one **directional series** `{location}__{dir}` (matches the
paper's sensor naming), NOT the `[B, N=81, T, D]` node batch of `src/harness`.
The track has its own lightweight contract that mirrors the harness philosophy
(capability gating, manifests, offensive asserts, config-driven):

```
src/quality/
  contract.py     SensorForecaster ABC, QualityCapabilities, SensorContext,
                  PerSensorSplit (index-based windows — never materialized)
  loader.py       SensorSeriesStore: person-class frame → per-series 15-min
                  NaN-gapped grid over the series' own active span
  quality_metrics.py  completeness, consecutive runs, window arithmetic,
                  assign_split_windows (SHARED by viability rule + datasets),
                  sensor-selection manifest
  per_sensor_dataset.py  build_sensor_splits (train-only scaler), feature
                  groups (count/half_day/day/week/quarter/year/timestamp),
                  WindowDataset (lazy torch gather)
  forecasters.py  Persistence, SeasonalNaive (fallback fraction reported),
                  ARIMA (SARIMAX fit-once + Kalman extend walk-forward),
                  PerSensorLSTM (bidirectional, paper architecture)
  registry.py     build_forecaster(model_key, spec) factory
  interpolation.py  none / night_zero / linear / spline / knn (+ mask)
  periodogram.py  Lomb–Scargle on raw gapped series AND extracted sequences
  evaluate.py     raw + scaled + real-targets-only metrics; per-window
                  predictions frame (DM-test substrate)
  stats.py        Spearman+FDR, partial Spearman, bootstrap CIs, segmented
                  threshold regression, Diebold–Mariano (HAC+HLN), RF (exploratory)
  plotting.py     paper rcParams profile (figure-readability fix) + figures
scripts/run_quality_experiments.py     resumable job runner (registry.json)
scripts/aggregate_quality_results.py   master table, stats, tables, figures
configs/quality/{quality.yaml, quality_smoke.yaml}
```

### Window/split contract
- Window i: input steps `[start, start+window)`, scalar target at
  `start + window + horizon - 1`. Defaults: window=24 (6 h), horizons {8, 24}.
- Per-run chronological split 70/15/15 with boundary purge (window+horizon−1
  windows dropped at val/test starts). Runs too short for a purged internal
  split are assigned **whole-run** to one split by deficit (see Issues #3).
- All models predict the identical window set per (sensor, horizon, variant);
  per-window forecasts persisted → paired DM tests.
- Scaler (count + none of the cyclic features) fit on TRAIN steps only.

### Documented deviations from the original paper (report in rewrite)
1. Train-only scaler (legacy fit on full series = leakage).
2. All viable sensors via explicit manifest rule (legacy: 12 hand-picked).
3. Boundary purging + whole-run assignment (legacy stride-1 sequential split
   leaked overlapping windows across split boundaries).
4. Fractional-hour cyclic encodings (legacy used integer hour; flag
   `features.fractional_time: false` restores legacy behavior).
5. Metrics reported on raw counts (primary) + scaled (continuity).
6. Lomb–Scargle actually computed (legacy: motivation only), on both raw
   gapped series and extracted sequences.

---

## Issues Encountered (running log)

1. **ParquetConnector rejects NaN values by contract** — correct for the GNN
   track, wrong for a missingness study. `SensorSeriesStore` reads the master
   parquet directly and treats NaN-value rows as "no observation" (they are
   resample_traffic grid artifacts), then re-grids. DB path unaffected.
2. **`src/analysis.py` contains no Lomb–Scargle** despite earlier exploration
   claims — it is Optuna study tooling. LS implemented fresh in
   `quality/periodogram.py` via `scipy.signal.lombscargle`.
3. **Boundary purging annihilated val/test windows for short-run sensors**
   (desk check finding): a purge of window+horizon−1=31 windows exceeds the
   val quota of any run with <~210 windows — which would have silently
   excluded exactly the below-threshold (low-quality) sensors the study is
   about (selection bias on the headline correlation). Fixed with whole-run
   assignment for short runs (zero cross-split leakage by construction);
   `assign_split_windows` is now shared by the viability rule and dataset
   construction so they can never disagree.
4. **sklearn KNNImputer is O(n²)** over ~1e5 grid points — replaced with
   KNeighborsRegressor (KD-tree) fit on observed points in a temporal feature
   space; identical semantics for one target column.
5. **SARIMA with s=96 is ~90 s/sensor vs ~1 s/sensor non-seasonal**
   (desk_check_forecasters benchmark: fit 81 s + 8.1 s/1k origins vs 0.5 s +
   0.9 s/1k; RMSE 4.89 vs 5.38 on synthetic). Config default keeps
   `seasonal_order: [0,0,0,0]`; seasonality is covered by snaive_daily/weekly
   baselines. Document in the paper; revisit only if reviewers insist.
6. **Pre-existing test failures on this branch** (not caused by this track):
   9 failures in `tests/test_harness.py` / `tests/test_pipeline.py` fail
   identically with the quality track stashed (WIP from
   `feature/ch5-observability-baseline`). All 48 quality-track tests pass.
7. **statsmodels + matplotlib added to requirements.txt** — Docker images
   must be rebuilt before container runs ("Sync or Sink").

## Data facts
- `data/processed/master_dataset_15T.parquet`: person-class rows are already
  a continuous 15-min grid per (location, dir) with NaN gaps; 81 locations,
  ~199 directional series, spans up to 2023-07-01 → 2026-06-08 (per-series
  active spans vary; NaN fraction ~9–35%).
- The runner's sensor set is defined ONCE by the `none`-variant viability
  manifest and reused across interpolation variants (comparability).

## Verification status (2026-07-22)
- 48/48 quality tests pass (`tests/test_quality_*.py`); pre-existing suite
  unchanged (see Issue #6).
- 3 desk checks pass (`scripts/desk_checks/desk_check_quality_{metrics,forecasters,stats}.py`).
- Smoke run end-to-end (`configs/quality/quality_smoke.yaml`: 3 sensors,
  5 models, 2 variants → 10 jobs, 0 failures, ~1 min CPU) + aggregation:
  MV-LSTM best, snaive competitive, DM tests significant where expected,
  real-only vs all-target metrics diverge under night_zero as designed.

## Next steps (exp/ branch)
1. Optuna tuning: one study per {UV, MV} × {8, 24} on a stratified sensor
   sample (mirror paper Table 2's single global config; reuse src/tune.py wiring).
2. Full matrix on GPU runner: `run.sensors: all`, all interpolation variants,
   `run_ablations: true`, `wandb.enabled: true`.
3. `scripts/aggregate_quality_results.py` → paper tables/figures under
   `artifacts/quality/analysis/`.
