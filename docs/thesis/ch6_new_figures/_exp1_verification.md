# Chapter 5 (Experiment 1): fact verification against the code and artefacts

Generated on 2026-10-13. This was a read-only investigation; the scratch computations are in `/tmp/e1/*.py`. The line numbers refer to
`docs/thesis/full/chapters/chapter5_experiment1_single_node.tex`.

## 0. Source map (read this first)

There are two Experiment 1 run sets, and Chapter 5 mixes them:

| Set | Path | Date | Viable sensors | Models | Quality table |
|---|---|---|---|---|---|
| **v1** | `artifacts/quality_full` | 2026-08-03 (git a8ac741) | 114 | LSTM-MV/UV + baselines | its own (e.g. Bedson completeness 0.81) |
| **v2** | `artifacts/quality_full_v2` | 2026-08-20 | 110 | all 12 | its own (Bedson completeness 0.63) |

No `src/quality/quality_metrics.py` changes were committed between the two runs, so the quality differences presumably come from a reprocessed `master_dataset`. The exact cause cannot be determined.

The "mixed" source below means v1 `analysis/results.parquet` errors combined with v2 `quality_table_none.parquet` features. This is what
`scripts/experiment1/plot_threshold_clean.py` and `plot_ch5_audit_cluster_and_corr.py` read.

| Thesis item | Source it reproduces from |
|---|---|
| Table 5.2/5.3 headline (LSTM-MV h8 median RMSE 6.16 / CV 56.0%; h24 4.69 / 62.1%) | **v2** (on the 110 sensors, v1 gives 6.03 / 57.6 and 5.06 / 61.5) |
| Table 5.6, matched pairs, extreme events, cluster medians 47.7 / 59.3, Heber | **v2** |
| Table 5.4 (correlations) | **v1 errors + v1's own quality features**, restricted to v2's 110 viable sensors |
| Fig 5.6 threshold, line 290 (2.90 h), line 294 outlier values | **mixed** |
| RF R² 0.25 / 0.37 | **v1, all 114 sensors** |

**Recommendation:** regenerate every Chapter 5 statistic from v2 (the canonical 12-model run). Alternatively, state explicitly which run each statistic comes from. As it stands, the chapter is not internally consistent.

---

## Q1: Outliers and threshold (line 290–294, Fig 5.6, Table 5.6)

**The worst five streams in v2 (canonical, LSTM-MV, h8, 110 sensors):**

| Stream | CV-RMSE | RMSE | avg seq len | completeness |
|---|---|---|---|---|
| NclUrbanScienceBuildingSquareMid__west_to_east | 224.62% | 42.12 | 2.26 h | 69.50% |
| Heber St | 187.47% | 52.61 | | |
| Bedson s→n | 163.30% | | | |
| Percy s→n | 157.10% | | | |
| Bedson n→s | 107.36% | | | |

- **The line 294 values (225.40 / 219.69 / 191.41%) do not match Table 5.6.** They are v1 CV-RMSE values paired with v2 sequence lengths (1.67 / 2.26 / 1.79 h), i.e. the mixed source.
- In v1 (114 sensors), the worst stream is QuaysideTrinityChare w→e at 339.2%. That stream is not in the v2 viable set.

**What drives the UrbanScience w→e error:**
- 744 test windows; mean y 18.75, median 12, max 1018.
- Active span 14 Nov 2023 – 27 Jun 2024; seq_count 1669; longest sequence 19.75 h.
- A handful of spikes dominate the error:
  - 1018 on Sat 27 Apr 2024 16:00 (prediction 28);
  - 450 on 13 Apr 13:30;
  - 274 on 11 May 16:15.
- The top 5 residuals account for 94.1% of the SSE.
- Dropping the top residual gives a CV of 122.85%; dropping the top 3 gives 67.21%.
- The processed data contain no duplicate timestamps. The other raw highs (576, 450, 424, 419) all fall on Saturday afternoons, which suggests events rather than faults.
- The opposite direction (e→w) has a CV of 87.6% and a maximum of 236.

**Breakpoint fits** use `src/quality/stats.py:147` `segmented_threshold_fit`: a 200-point grid over the 10–90% quantiles, AIC with k=4 vs k=2, and 1000 bootstrap samples with seed 42.

| Data | n | Breakpoint | 95% CI | Slope below | Slope above | ΔAIC |
|---|---|---|---|---|---|---|
| Mixed (reproduces thesis exactly) | 110 | 2.90 | [1.87, 3.23] | −27.37 | −0.44 | 9.94 |
| Mixed, excluding 3 outliers (reproduces thesis) | 107 | 2.96 | [1.94, 3.39] | −14.69 | −0.50 | 7.78 |
| v2 canonical | 110 | 2.95 | [1.90, 3.42] | −22.26 | −0.48 | 7.17 |
| v2, excluding 3 outliers | 107 | 2.94 | [1.94, 4.91] | −14.12 | −0.53 | 4.50 |
| v2, h=24 | 110 | 2.57 | [1.93, 4.92] | | | 0.39 (no threshold) |

- 8 of the 110 streams have an average sequence length above 6 h (confirmed). 61 are below 2.9 h.
- **Conclusion:** the threshold of about 2.9–3.0 h is robust across sources at h8. If the text is moved to v2, use 2.95 h [1.90, 3.42].
- The h24 threshold is not supported (ΔAIC 0.39); add this as a limitation.

## Q2: Table 5.4 (bivariate and partial Spearman)

`partial_spearman` is at `src/quality/stats.py:60`. It uses rank residualisation with dof = n−2−k.

**Reproduced exactly** by v1 errors combined with v1's own quality features on the 110 v2-viable sensors:

| Horizon | Predictor pairs (bivariate / partial \| volume) | Volume | ρ(seq, vol) |
|---|---|---|---|
| h8 | −0.618/−0.515, −0.567/−0.489, −0.539/−0.402 | −0.403 | 0.881 |
| h24 | −0.407/−0.369, −0.341/−0.379, −0.335/−0.336 | −0.192 | 0.889 |

**v2 canonical values** (bivariate / partial | volume):

| Horizon | seq len | completeness | diurnal | volume | ρ(seq, vol) |
|---|---|---|---|---|---|
| h8 | −0.577/−0.438 | −0.543/−0.390 | −0.343/**−0.106 (n.s.)** | −0.423 | 0.863 |
| h24 | −0.405/−0.447 | −0.282/−0.422 | **−0.146 (p=0.13)/−0.108 (n.s.)** | **−0.099 (p=0.30, n.s.)** | 0.887 |

- Other v2 statistics: ρ(seq, comp) = 0.779; partial seq|comp = −0.182 (p=0.058); comp|seq = −0.293 (p=0.002).
- Mixed source: diurnal bivariate −0.327, partial −0.097; partial seq|comp = −0.128 (p=0.185) at h8.
- **Discrepancy:** under v2, the caption's "all significant" is false.
  - The diurnal partial correlation given volume is about −0.1 (n.s.).
  - At h24, diurnal bivariate (p=0.13) and volume (p=0.30) are n.s.
  - The sequence-length effect independent of completeness is marginal (p=0.058).
- I found no script in the repo that produces the partial|volume columns, or the log-volume regression the text quotes (t=0.31).
  - My OLS of `cv_rmse ~ seq + comp + log(vol)` gives t(log vol) = 0.37 (v1own h8), 0.83 (mixed h8), 0.24 (v2 h8) and 1.85 (v2 h24, p=0.07).
  - The qualitative claim ("volume adds nothing") holds at h8. 0.31 itself is not reproduced exactly.

## Q3: Random-forest R²

The code is `src/quality/stats.py:326` `rf_quality_to_error`:
- `RandomForestRegressor(n_estimators=300, random_state=42, oob_score=True)`;
- `RepeatedKFold(5,10, random_state=42)`;
- permutation importance with 30 repeats.

The predictors are `QUALITY_PREDICTORS` in `scripts/experiment1/aggregate_quality_results.py`: completeness_active, avg_seq_len_hours, std_seq_len_hours and seq_count (no volume). The target is cv_rmse_raw.

- **The thesis values 0.25 / 0.37 are reproduced exactly** as OOB R² on v1 data, using **all 114 sensors**.
- **The OOB estimate is optimistic.** The bootstrap is at row level, so the two directions of one location can fall on opposite sides of the in-bag/OOB split.

| Validation scheme | h8 | h24 |
|---|---|---|
| Location-grouped GroupKFold(5), 54 locations, pooled R² | **−0.08** | **−0.74** |
| Same, plus volume | +0.14 | −0.02 |
| Plain RepeatedKFold, mean R² | −0.04 (median 0.125) | |
| v2 (110 sensors), OOB | 0.041 | −0.006 |
| v2 (110 sensors), grouped pooled | 0.040 | −0.143 |
| v2 (110 sensors), OOB with volume | 0.077 | 0.114 |

- Fold-mean R² values under grouped CV are unstable, with very large negatives at h24.
- **R² 0.81 for RMSE ("preliminary study") is not reproduced** by this pipeline spec. The same 114-sensor spec with an RMSE target gives OOB 0.686 (h8) and 0.902 (h24). Values of about 0.81 appear only when volume is added: v1own 110 with 4q+vol gives 0.818, and v2 4q+vol gives 0.792.
- `artifacts/quality_full/analysis/rf_exploratory_lstm_mv_h{8,24}.json` shows OOB 0.846 / 0.914 (n=114). These values are inconsistent with a CV-RMSE target and were presumably produced with the earlier RMSE target. The `rf_importance_*.png` figures share that timestamp (2026-08-03), so the target behind them cannot be confirmed.
- **Recommendation (correction):** report the RF as non-predictive under location-grouped CV, or drop the R² claim. At minimum, state that 0.25/0.37 is ungrouped OOB on 114 sensors.

## Q4: Matched-pairs analysis (KNN vs none)

The code is `scripts/experiment1/quality_temporal_scale_analysis.py:294` `experiment_l_matched_pairs`. It inner-merges on (sensor_id, target_dt), and the per-sensor CV uses the none-arm mean y as a common denominator.

**Matched set (all confirmed from v2):**
- 50,051 matched steps across **108** sensors, 19 Jan 2024 – 7 Jun 2026.
- y_true is identical across arms, and every KNN target in the matched set is a real observation.

**Results:**

| Measure | none | KNN | Difference |
|---|---|---|---|
| Pooled RMSE | 14.67 | 17.62 | +20.1% relative |
| Pooled CV-RMSE (pooled RMSE / pooled mean) | 69.74% | 83.77% | **+14.03 pp** (write "pp", not "%") |
| Sensor-mean CV | 58.27 | 62.61 | 4.34 ± 8.49 pp |
| Sensor-median CV | 55.19 | 59.30 | **+4.11 is a difference of medians**; the median of the paired differences is **2.11 pp** |

- Paired tests on the sensor-mean CV: t(107) = 5.31, p = 6×10⁻⁷; Wilcoxon p = 4.6×10⁻⁸.
- none is better on 78/108 sensors.
- Matched steps per sensor: minimum 51, median 376.5.

**Other checks:**
- **The two dropped sensors** are both directions of NclQuaysideWesleySquare. Their none-arm test windows run Sep 2023 – Mar 2025, while the KNN test windows run 19 Jun – 13 Oct 2025, so there is no overlap.
- **Window counts:** none 333,917 vs KNN 1,221,570 (×3.658). However, **349,673 (28.6%) of the KNN windows have imputed targets**. Windows with real targets number 871,897, which is ×2.61. If the text says "3.7× more data", qualify it.
- **Hour profile of the matched steps:**
  - 90.6% fall between 13:00 and 22:59; 13–21 h holds 85.0%; 00–08 h holds 4.2%.
  - The hourly bins at 07, 08, 09 and 10 h have n = 8, 8, 8 and 11; 11 h has n = 97.
  - The line 361 claims for 08:00 (RMSE 38.8 none vs 26.9 KNN) and for 09:00–13:00 therefore rest on very small samples. Drop them, or flag them as n ≤ 11.
- Timestamps are tz-naive in the parquet, so "UTC" cannot be verified from the data.

**Split scheme:** both arms use the same algorithm (Q8a). Because KNN fills the gaps, its runs are long and receive an internal chronological 70/15/15 split, so the test set is the last 15% of the lifespan. This is why the test periods differ between arms, and why only matched steps are comparable.

## Q5: KNN imputation (what the code actually does)

The code is `src/quality/interpolation.py` `_knn_impute`.

- **Model:** `KNeighborsRegressor(n_neighbors=k, weights="distance")`, with k=5 from the config (`knn_neighbors: 5`) and the default Euclidean metric.
- **Features:** sin/cos of hour-of-day (fractional, 15-min resolution), sin/cos of day-of-week, and normalised time t∈[0,1].
- **Donors are observed steps of the same sensor only.** There are no cross-sensor donors. In practice the neighbours are the same 15-min slot and weekday in nearby weeks.
- **It is fitted on all observed points, including the validation and test periods.** This leaks future-period profile information into imputed training inputs. List it as a limitation, or as a correction if the text claims there is no leakage.
- **No maximum gap length.** `max_gap_steps: 8` applies only to linear and spline interpolation, so KNN fills gaps of any length. The output is clipped at ≥ 0.

## Q6: Extreme events

The code is `quality_temporal_scale_analysis.py:228` `experiment_k_extreme_events`, with the band CV computed in `plot_ch5_audit_temporal_scale.py:207-244`.

- σ is the per-sensor SD of y_true over the **test predictions**.
- The normal band is μ±1σ, the 1–3σ band is excluded, and the extreme band is > μ+3σ.
- Sensors with SD < 1 are skipped.
- Band CV = band RMSE / band mean y.

**v2, none arm, 110 sensors:**

| Measure | Normal | Extreme |
|---|---|---|
| Median RMSE | 4.87 | 24.21 (×4.97) |
| Median CV | 46.6 | 51.2 (confirmed) |
| MAPE, mean across sensors | 58.9 | 45.1 |
| MAPE, median across sensors | 50.4 | 43.7 |

- **MAPE 58.9 → 45.1 are means**, whereas the other figures are medians. Use the medians (50.4 → 43.7) or label the means as such.
- **"Peak residuals reaching 690"** is in fact the maximum per-sensor extreme-band **RMSE** (Heber). The largest single residual is 2169.
- Median share of steps per sensor: normal 77.8%, excluded band 20.7%, extreme 1.4%.
- Extreme-band y_true, pooled: median 64, mean 102.5; 30% of steps are ≥ 100 and 0.17% are ≥ 1000. The per-sensor extreme-band mean has a median of 57.7 (IQR 32.9–129.7).
- **Line 497 ("denominator in the hundreds or thousands") is wrong for a typical sensor.** The median is about 60 counts. Correct it.
- For reference, the KNN arm gives RMSE 4.26 → 23.08, CV 67.8 → 53.1 and median MAPE 68.8 → 43.2.

## Q7: Spatial clusters

**Thesis-figure code (`scripts/experiment1/plot_ch5_audit_cluster_and_corr.py`):**
- KMeans with k=2 (hard-coded), random_state=42, n_init=10, on StandardScaler features.
- Features: avg_seq_len, completeness, diurnal_capture_ratio, mean_count_raw, total_poi_count, transit_count and dist_to_city_centre. The POI data come from `data/external/places_cache.json`.
- Clusters are labelled core/suburban by their mean distance to the city centre.
- Other scripts use different feature sets: `quality_deep_analysis.py:159-240` (silhouette sweep, includes food_drink) and `quality_extended_analysis.py` (8 features plus Kruskal–Wallis).

**Results:**
- Silhouette scores for k=2..6 (mixed source) are 0.460, 0.414, 0.369, 0.432 and 0.456. k=2 is best, but k=6 is nearly as good, so say that k=2 was chosen by silhouette with a narrow margin.
- Cluster sizes are 28 vs 82 streams (14 vs 39 locations). No location is split across clusters.
- Mann–Whitney U on CV-RMSE, h8:
  - **v2:** U=881, **p=0.067**. The medians (core 47.7, suburban 59.3) match the thesis.
  - Mixed: U=873, p=0.060 (52.9 vs 58.9).
  - At h24: v2 p=0.51; mixed p=0.997.
- **Correction:** "significantly lower" is false. Use "lower but not significantly (p=0.067)", and note that there is no difference at h24.
- ρ(distance, CV) on the mixed source is −0.037 (the thesis says −0.01).

## Q8: Split procedure, masked transformer and seeds

**(a) Split.** The code is `src/quality/quality_metrics.py:134-205` `assign_split_windows`.
- Each contiguous run is split chronologically 70/15/15, with a purge of W+H−1 steps.
- Runs too short for an internal split are assigned wholly to a single split by a running deficit counter. Ties are broken in the order train > val > test.
- The same function is used for every interpolation variant.
- **There is no "alternating daily" or day-by-day logic.** If the thesis describes an alternating daily split, that is a correction.
- Valid windows come from `find_consecutive_sequences(series, min_length=window+horizon)` (`per_sensor_dataset.py:56`). The inputs are [s, s+W) and the scalar target is at s+W+H−1. A tripwire asserts that no window contains a NaN.

**(b) Masked transformer.** The code is `src/quality/masked_dataset.py` and `transformer_forecaster.py:116-195`. **The masking is functionally inert.**
- The mask is derived from NaN in `values_raw`, but the windows come from the same contiguous `build_sensor_splits`.
- The mask is therefore all-False, and the missingness and time-delta channels are all zero.
- n_windows are identical to transformer_mv (333,917 predictions each).
- `run_quality_experiments.py:317` runs masked models only for 'none'.
- Median RMSE is 6.18 (masked) vs 6.13 (standard); the near-identity is expected given the above.
- **Correction or limitation:** the masked model never sees a gap, so it cannot test mask-aware learning. Either reword the claim, or state this explicitly as a limitation.

**(c) Seeds.** There is a **single seed, 42** (`configs/quality/quality_full_v2.yaml` `run.seed: 42`; `run_quality_experiments.py:271-272`). There are no repeated seeds and no seed variance for Experiment 1, so list this as a limitation.

## Q9: Heber St case study

No script in the repo produces these values, but they reproduce from the v2 predictions:

| Treatment | RMSE | CV |
|---|---|---|
| Halve 2242 → 1121 | 41.96 | 151.23% |
| Omit the 2242 step | 38.14 | 138.98% |
| Restrict to y ≤ 260 (10 surge steps removed) | 17.17 | 66.51% |

- **The "duplicate record" claim cannot be verified, and 2242 is plausibly genuine.**
  - The processed data have no duplicate timestamps.
  - Other Saturday 16:00 matchday readings reach 1966 (25 Oct 2025), 1916 (26 Apr 2025), 1880 (18 Apr 2026) and 1810 (2 May 2026), with others at 1398 and 1304.
  - The author must check the raw database before calling it a duplicate. Otherwise reword it as "an extreme matchday surge".
- **The test maximum excluding 2242 is 952, not 1,154.** The other test surges are 263, 480, 918, 757, 952, 653, 708, 574 and 744. 1,154 occurs on 28 Sep 2025, which is outside the test split.
- **The Fig 5.12 caption is doubtful.**
  - Tuesday 21 April 2026 peaks at 150 (14:15) and Tuesday 28 April 2026 at 206 (16:00).
  - The true matchday peaks around those dates were 1880 (Sat 18 Apr) and 1810 (Sat 2 May).
  - Labelling 21 and 28 April as matchday surges looks wrong. Re-check the plotted window and the labels.

---

## Correction vs limitation summary

| Item | Type |
|---|---|
| Mixed v1/v2 sources (line 294 vs Table 5.6, Table 5.4, RF, Fig 5.6) | Correction (harmonise to v2 or label the source) |
| Table 5.4 "all significant" | Correction (false under v2) |
| RF R² 0.25/0.37 (ungrouped, 114 sensors), 0.81 not reproducible | Correction |
| "Alternating daily" split | Correction (code splits contiguous runs chronologically) |
| Masked transformer inert | Correction to the claim, plus a limitation |
| Cluster difference "significant" | Correction (p=0.067) |
| Extreme-band denominator "hundreds or thousands" | Correction (median about 60) |
| MAPE means vs other medians; "residuals reaching 690" | Correction |
| Pooled CV difference "%" → "pp"; +4.11 is a difference of medians | Correction |
| Heber "duplicate" and 1,154; Fig 5.12 dates | Correction (verify against the raw DB) |
| 3.7× data includes 28.6% imputed targets | Correction/qualification |
| Hourly claims at 08:00–10:00 (n ≤ 11) | Correction (drop or flag) |
| KNN fitted on test-period observations, no max gap | Limitation |
| Single seed (42) | Limitation |
| h24 threshold unsupported (ΔAIC 0.39) | Limitation |
