# Zero-Collapse in RIoT v4: Diagnosis, Architectural Remedies and Experiment Plan

**Branch:** `feat/riot-v4` · **Date:** 2026-10-01 · **Scope:** `src/dataset.py`, `src/pipeline.py`, `src/decoders.py`, `src/encoder.py`, `src/train.py`, `src/harness/evaluate.py`, `src/export_ui_payload.py`, `configs/train/v4/*`

**Problem statement.** Under RIoT v4 the model predicts zero too often. This is worst on some sensors and during missing-data periods, where the predictions collapse to ~0. Zero is the cheap answer for the loss. The asymmetry that matters operationally is that under-predicting a large count (truth 400, prediction 0) is very costly, while confusing 0 with 1 is irrelevant.

**TL;DR.**

1. **This is mostly not a head-vs-head tension.** It is a target-contamination problem: the forecasting, imputation and kriging losses *all* reward zeros at missing slots, and so do the validation metric and the checkpoint selection. Missing values are filled with `0.0` (`src/dataset.py:147`), and that filled zero becomes the ground truth for every head.
2. **The data contains no real zeros.** In the v4 window, 52.8% of the 81-sensor grid is missing and **0.000% of observed readings are 0**. The raw API never emits a zero row (0 of 7.9M person rows). Every zero the model is trained to predict is synthetic.
3. **The model can see when a zero is coming.** The structural mask tells it the input is missing, and P(target missing | last input step missing) = **0.92**. A ZINB head can push `pi → 1` on those slots and drive their NLL to ~0. This shortcut is cheaper than anything else in the objective. The bf16 `pi` saturation crash on 2026-09-29 (EXPERIMENTS.md, "ZeroAcc 63% on 363k nocturnal zeros") was a symptom of it: those "zeros" were missing slots.
4. **What hides behind the gaps is not small.** Using each sensor's slot-of-week median as a rough prior, the expected count inside the missing slots is about **1.1× the total observed count**. Only ~13% of missing slots are plausibly true zeros, and 21% sit where the sensor normally reads more than 20.
5. **Recommended fix, in order:** (a) mask every loss and metric to observed targets; (b) train with *structured* artificial gaps (blocks and transplanted real outage patterns) on observed data, encoded exactly like real gaps; (c) give the model the information it needs to predict a level during an outage (calendar features, sensor identity, a seasonal-profile prior, longer context); (d) replace ZINB with an **outage-aware censored likelihood** that explains missingness with a separate outage head instead of a zero count; (e) add magnitude-aware loss weighting and upper-quantile heads.
6. **The existing v4 ablation results are confounded.** Every configuration trained and was selected on synthetic zeros. Re-run the ablation suite on top of the corrected training (Phase 7 below).

---

## 1. Evidence

### 1.1 Data statistics (computed 2026-10-01 from `data/processed/master_dataset_15T.parquet`, person class, `dir=all`, 2023-04-02 → 2026-04-02, matching the v4 config)

| Statistic | Value |
|---|---|
| Grid | 96,576 timesteps × 81 sensors |
| Missing fraction (whole grid) | **52.8%** |
| Missing fraction, daytime 08–20h | **42.0%** |
| Missing fraction by hour | 0–4h: 72–82% · 08–17h: ~41% · 23h: 63% |
| Observed readings equal to 0 | **0.000%** (raw source: 0 / 7,922,488 rows) |
| Share of zeros in the filled grid that are synthetic | **100%** |
| Observed count quantiles (p50 / p90 / p99 / p99.9) | 10 / 58 / 246 / 851 (mean 26.3, std 63.0) |
| Share of Σy² contributed by the top 10% of observations | **93%** |
| Per-sensor missingness | min 5%, median 56%, max 100%; 45/81 sensors > 50%, 15/81 > 80% |
| Gap lengths (steps of 15 min) | median 1, p90 8, p99 34; but **83%** of missing slots lie in gaps > 12 steps (> 3 h) and **72.5%** in gaps > 96 steps (> 24 h) |
| P(target missing \| last input step missing) | **0.923** |
| P(target missing \| last input step observed) | 0.086 |

**Zero-plausibility of missing slots.** For each sensor × slot-of-week I took the median of *observed* counts as a rough level λ:

| Condition among missing slots | Share |
|---|---|
| λ > 1 | 73.5% |
| λ > 5 | 41.0% |
| λ > 20 | 20.8% |
| λ > 50 | 12.1% |
| Mean Poisson P(Y = 0 \| λ) | 12.7% |
| Σλ over missing slots / Σ observed counts | **1.11** |

*Caveat:* the profile is fitted on observed slots only, so it is biased upward wherever missingness is MNAR (quiet intervals go unreported). Treat it as an upper-leaning estimate. Even if it is off by a factor of 2–3, the conclusion holds: a large fraction of the missing mass is not zero.

**Heartbeat cross-check** (`data/heartbeat/sensor_heartbeat-1.csv`: 2026-03-10 → 2026-04-07, 57 overlapping sensors). 92% of missing slots in that window have a live heartbeat, including **76.5% of slots inside gaps longer than 24 h**. A sensor that reports zero pedestrians for days, including daytime, is implausible, so heartbeat means *device online*, not *counting correctly*. It cannot be used directly as an outage label. It is still useful as a weak feature, and as a sanity check for short gaps: a live heartbeat at night in a 1–4 step gap is the strongest evidence of a true zero we have.

**Interpretation.** "Missing" is a mixture of two very different things:

- **Quiet slots.** Short gaps, mostly at night, at low-volume sensors. The true value is 0 or very small.
- **Outages.** Long gaps that cover whole days, including peaks. The true value is unknown and often large.

Today the model treats both as observed zeros.

### 1.2 Code paths that reward zero

| # | Location | What happens | Effect |
|---|---|---|---|
| C1 | `src/dataset.py:147` | `val_df.fillna(0.0)` before scaling | Missing slots become raw 0 → scaled ≈ −0.42 in inputs **and** in `target_impute` |
| C2 | `src/dataset.py:251` | `target_forecast = unscaled_traffic[idx + seq_len]` with no observed mask | ~53% of forecast targets are synthetic zeros |
| C3 | `src/decoders.py:363`, `:273` | ZINB NLL over all targets; `y < eps` takes the `ll_zero` branch | `pi → 1` gives near-zero loss on synthetic zeros. Strongest shortcut in the objective |
| C4 | `src/decoders.py:339-352` | Traffic MSLE over **all** positions, not only `node_mask` positions | ~85% of the imputation loss is reconstructing *visible* inputs, half of which are synthetic zeros: a copy task |
| C5 | `src/decoders.py:374` → `compute_kriging_loss` | Kriging MSLE over all held-out slots | Kriging is trained to produce zeros wherever the held-out sensor was offline |
| C6 | `src/train.py:665-668` | Artificial mask is iid per element (15%), NaN → 0 scaled; **no indicator** in the mask channels | Artificial gaps look different from real ones: an artificial gap reads 0.0 with mask = 0, a real gap reads −0.42 with mask = 1. The model can tell them apart: real gap → output zero, artificial gap → interpolate. iid single-step masks never teach outage-length recovery (83% of missing mass is in gaps > 3 h) |
| C7 | `src/pipeline.py:149`, `src/dataset.py:314` | Three different "missing" encodings: artificial (scaled 0 = mean count), real (scaled −0.42), spatial holdout (scaled 0, mask forced to 1) | Inconsistent signal. The model must learn three conventions |
| C8 | `src/harness/evaluate.py:60-77` | RMSE / MAE / ZeroAcc computed over all targets | ZeroAcc measures only how well the model detects *missingness*; RMSE/MAE reward predicting 0 in gaps |
| C9 | `src/train.py:749-751, 801` | Checkpoint selection on `val_loss_total`, which includes the contaminated terms | Selection actively prefers the collapsed model |
| C10 | `src/dataset.py:206-210` | Dynamic channels are traffic, humidity and solar radiation only. No hour, weekday or holiday features, and no per-sensor identity in the dynamic path | With `seq_len = 12` (3 h), most outage targets have a fully blank input window. The model has *no information* about what level to predict, so the loss-minimising answer is the conditional mean of the targets, which is zero (C2) |
| C11 | `src/export_ui_payload.py:1068` | UI shows `E[Y] = mu·(1 − pi)` | Even when `mu` is reasonable, saturated `pi` drives the displayed value to 0. Part of the visible collapse may be `pi`, not `mu` (check in Phase 0) |
| C12 | `src/dataset.py:142-144` | `ablate_masks` zeroes `structural_masks` in place | Once losses are masked (Phase 1), the loss mask must come from an **un-ablated** copy, or the `no_masks` ablation silently trains on synthetic zeros again |

### 1.3 Why it presents as a "tension"

The imputation head appears to fight the forecast head, but both are pulled the same way. Every head scores best by recognising "the mask is on" and emitting zero. The ZINB head is most efficient at it: one logit (`pi`) absorbs the entire gap population. The imputation and kriging heads must emit −0.42 in scaled space, which is easy too. Whatever residual gradient conflict exists sits on top of a shared, dominant incentive to detect missingness. Phase 6 measures the real conflict *after* the targets are fixed. I expect it to be small.

### 1.4 Why MSLE / log-space losses make high values worse

Once the targets are clean, the second issue is how errors on large counts are weighted:

- MSLE (`scale_invariant_mse`) and the log-link NB both compress large values. Under uncertainty, the MSLE-optimal prediction is the *geometric* mean exp(E[log(1+y)]) − 1, which sits systematically below the arithmetic mean. That biases peaks downward.
- Under MSLE, predicting 0 when the truth is 400 costs (log 401)² ≈ 36. Predicting 40 when the truth is 400 costs ≈ 5.2. Predicting 0 when the truth is 1 costs ≈ 0.48. The ratios look sensible, but the *frequency* of low counts (median 10) means peaks contribute little gradient in aggregate, while the top 10% of observations carry 93% of the squared-error mass the user actually cares about.
- What you want is the opposite emphasis: tolerant at the bottom, strict about *under*-prediction at the top. Section 2.5 adds that explicitly.

---

## 2. Remedies

Remedies are ordered by expected impact per unit of effort. R1–R2 are corrections, not design choices, and should land before anything else is judged.

### R1. Target hygiene: train and evaluate only where truth exists (fixes C2–C5, C8, C9, C12)

- `RealIoTDataset.__getitem__` returns two extra fields from an **un-ablated** copy of `total_mnar_mask`:
  - `target_forecast_observed` `[N]` = `structural_masks_raw[idx + seq_len, :, 0] == 0`
  - `input_observed` `[N, T]` for the input window
- `compute_zinb_nll` (or its replacement) takes an `observed` mask and averages over observed targets only.
- Imputation loss on the traffic channel is computed on **artificially masked AND originally observed** positions only (`node_mask & input_observed`). The current "reconstruct everything" term can be kept as a small auxiliary weight if wanted, but restricted to observed slots.
- Kriging loss restricted to observed slots of the held-out nodes.
- Metrics (`evaluate_batch`, `metrics.py`): RMSE / MAE / impute / kriging on observed targets only. Retire `zero_acc` (Section 3 replaces it).
- Checkpoint selection uses the observed-only loss (or the composite selection score in Section 3).
- Empty-mask guards in the same style as the existing kriging empty-holdout guard.

Expected effect: the shortcut disappears from the objective. On its own this does not teach the model what to predict in a gap. With iid masks the model rarely sees a long gap with a scored target. R2 and R3 cover that.

### R2. Gap-realistic self-supervision: make artificial gaps indistinguishable from real ones (fixes C6, C7)

- **One encoding for "missing".** Real gaps, artificial gaps and spatial holdout all become value = 0 (scaled) plus missing indicator = 1. The `node_mask` positions set `structural_masks[..., 0] = 1` (and the night channel as appropriate). The model can then no longer branch on "real gap vs artificial gap", and anything it learns from artificial gaps transfers to real ones.
- **Structured masking strategies** applied to *observed* data:
  - `iid`: current behaviour, as a baseline.
  - `block`: contiguous runs per sensor, length ~ geometric or empirical (matching the gap-length distribution in 1.1, including blocks covering the whole window).
  - `transplant`: copy a real missingness pattern from another sensor or another day onto an observed window. This is the strongest option because it matches the real MNAR structure (time-of-day dependence, correlated outages), as in CSDI / SAITS-style training.
  - `tail`: mask the last *k* steps of the input window, so the forecast is made from inside a simulated outage. This teaches exactly the deployment scenario: forecasting while the sensor is down.
  - `sensor_dropout`: blank the whole window for a node. This is already present as the spatial holdout and should be unified with the above.
- Artificial masks are drawn only over observed positions, so every scored target is real.

### R3. Give the model the information needed to predict a level during an outage (fixes C10)

A window that is blank for 3 h contains no information about the level. The model must get it elsewhere:

- **Calendar features:** sin/cos hour-of-day, sin/cos day-of-week, weekend flag, UK bank holiday / university term flag (Newcastle and Durham sensors are strongly term-driven). Add as dynamic channels. They are deterministic, so they are always observed, including inside gaps and for the forecast horizon.
- **Seasonal-profile prior channel:** the train-only sensor × slot-of-week median (log1p), computed in `RealIoTDataset` from the training slice only (no leakage; reuse the F-C02 pattern). This gives the model a strong default for "what is normal here now" during a gap, so it only has to learn deviations. The prior must also be available inductively for unseen nodes. For inductive holdout nodes, either derive it from neighbours or from static features via a small regressor, or zero it and set a flag. Test both (E3.3).
- **Learned sensor embedding** for transductive nodes (`nn.Embedding(num_nodes, d)` added to the GAT input), with embedding dropout so the model does not overfit to identity. Inductive holdout nodes get the mean or an embedding predicted from static features.
- **Longer context:** `seq_len` 12 → 48 → 96 (12 h / 24 h). 83% of missing mass is in gaps > 3 h. Attention cost is quadratic in T but small at T = 96; the GAT cost is linear in T. Alternatively, add a lag-96 / lag-672 (same slot yesterday / last week) summary channel instead of a long window.
- **Log1p input transform** for the traffic channel before standardisation. With std 63 and p99.9 = 851, standardised raw counts let a few peaks dominate the input scale.
- **Gap-run-length feature:** steps since last observation, capped and log-scaled. It is causal, distinguishes "1-step blip at 3am" from "day 4 of an outage", and is the main input to the outage head in R4.

### R4. Replace ZINB with an outage-aware censored likelihood (core architectural change)

ZINB assumes structural zeros exist in the data. This data has *none*: observations are left-truncated at 1 and every zero is a missing slot. The likelihood should model what is actually observed:

```
Let  q = P(outage | context)              — outage head (sigmoid)
     NB(y; mu, theta)                     — count head ("what the sensor would see if working")

Observed slot with count y ≥ 1:   P = (1 − q) · NB(y)
Missing slot:                     P = q + (1 − q) · NB(0)       (outage OR truly zero, unreported)
```

Properties:

- **It separates the two causes.** If context implies a high count, NB(0) ≈ 0, so the only way to explain a missing slot is to raise `q`. The gradient flows into the outage head, *not* into `mu`. That is exactly the "zero with no ground truth vs predict the correct value" distinction requested. For quiet night-time short gaps, NB(0) is genuinely large and the model can explain the gap as a true zero.
- `mu` becomes the **counterfactual expected count with the sensor working**, which is what the UI should display for missing periods (fixes C11). `q` is a separately useful product: an outage probability per sensor.
- Observed slots use the zero-truncation correction implicitly: the (1 − q)·NB(y) term together with the missing-slot term is a proper distribution over {missing, 1, 2, …}.
- **Identifiability.** `q` must not be able to explain everything. Mitigations, each tested in E4.x:
  - feed the outage head only missingness-related inputs (gap run-length, recent mask history, sensor reliability statistic, heartbeat where available, hour), *not* the full latent;
  - stop-gradient from `q` into the shared encoder;
  - a weak prior: a BCE term pulling `q` toward a gap-length heuristic, e.g. q̂ = 0.05 for gaps ≤ 4 steps at night and 0.95 for gaps > 96 steps;
  - combine with R2: on artificially masked slots the target is *known* to be an outage with a known count, so add a supervised BCE(q, 1) and the NB(y) term on those slots. This anchors both heads.
- Missing-slot NLL can be down-weighted (`w_missing_lik`) as a knob, since it carries less information about `mu` than observed slots.

Simpler alternatives to compare against:

- **(a) NB on observed only**: R1 with ZINB → NB. Drops `pi`, since there are no structural zeros among observed slots.
- **(b) Zero-truncated NB on observed only**: P(y \| y ≥ 1) = NB(y) / (1 − NB(0)). This is the exact likelihood if the API drops zeros. Beware: it needs care when `mu` is very small (1 − NB(0) → 0); clamp.
- **(c) Keep ZINB but mask to observed**: the minimal change, which is R1 alone.

### R5. Magnitude-aware loss: penalise missing a high count

All of these are applied on observed (or artificially masked, known-truth) slots only:

- **Magnitude-weighted NLL:** w(y) = 1 + (y / y₀)^α with y₀ ≈ p75 (27) and α ∈ {0.5, 1}. Normalise weights per batch so the loss scale stays comparable.
- **Asymmetric under-prediction penalty** on the point prediction ŷ (= `mu` for the count head): L_under = mean( relu(y − ŷ) · g(y) ), e.g. a pinball loss at τ = 0.7–0.8 on raw or sqrt scale. This penalises predicting 0 for 400 linearly in the shortfall and ignores 1-vs-0 confusions (shortfall ≤ 1).
- **Quantile heads** (q50, q90) on the forecast and imputation paths, trained with pinball loss. q90 directly answers "how high could it be during this outage" and makes under-prediction visible in evaluation.
- **Imputation loss mixing:** α·MSLE + (1 − α)·Huber(raw counts, δ ≈ 20). Pure MSLE under-weights peaks (Section 1.4).
- **Tolerance band at the low end:** no loss for |y − ŷ| ≤ 1 when y ≤ 2 (epsilon-insensitive). Optional; this encodes the stated "0 vs 1 is irrelevant" directly.

### R6. Multi-task balancing (only if a conflict is still measurable after R1–R2)

- Diagnostic: cosine similarity between the gradients of `loss_forecast` and `loss_impute` w.r.t. the shared encoder parameters, logged every N steps. Persistently negative values indicate a real conflict.
- Options: homoscedastic uncertainty weighting (Kendall et al. 2018), GradNorm, PCGrad. Alternatively, a staged schedule: pre-train with imputation (masked-autoencoder style, R2), then fine-tune with forecast.
- Stop-gradient variant: imputation head on a detached latent, or a forecast head that reads a detached imputed sequence. This measures how much each task helps the other.

### R7. Smaller items

- **Missing-aware attention:** use the missing indicator to build a key-padding mask in the temporal Transformer (or attention bias −∞ for missing keys). Missing steps then contribute nothing to other steps' representations, but can still be *queried* for imputation.
- **Graph propagation into outages:** in the GAT, down-weight messages *from* nodes whose current step is missing. This extends the existing holdout edge-dropping (`src/pipeline.py:173-185`) to per-timestep real missingness.
- **Export:** show `mu` (counterfactual) plus a q50–q90 band for slots in a gap, and `q` as an outage overlay. Stop showing `mu·(1 − pi)` for missing slots (C11).

---

## 3. Evaluation protocol (needed before any experiment can be judged)

All metrics are computed on the fixed inductive holdout manifest and the chronological test split, for both observed nodes (transductive) and inductive nodes. **No metric ever scores a synthetic zero.**

| Metric | Definition | Why |
|---|---|---|
| `obs_rmse`, `obs_mae` | Forecast error on observed targets only | Replaces the contaminated RMSE/MAE |
| `peak_rmse`, `peak_mae` | Error on observed targets ≥ train p90 (58) | What you actually care about |
| `peak_underpred_rate` | P(ŷ < 0.5·y \| y ≥ p90) | Direct collapse-on-peaks measure |
| `bias_by_decile` | mean(ŷ − y) per target decile | Shows the systematic low-bias shape |
| `low_tol_acc` | P(\|ŷ − y\| ≤ 1 \| y ≤ 2) | Confirms low-end tolerance isn't destroyed (secondary) |
| `so_rmse@L` | **Synthetic-outage RMSE**: blank observed test windows using real gap patterns of length L ∈ {1, 4, 12, 48, 96}, then score imputation and forecast against the hidden truth | The only way to score "predicting inside a gap" with real ground truth |
| `so_mass_ratio@L` | Σŷ / Σy over synthetic-outage slots | 1.0 = level recovered; ~0 = collapse |
| `collapse_rate` | P(ŷ < 1 \| real missing slot, profile λ > 10) | Real-gap behaviour without ground truth: predicting ~0 where the sensor normally sees > 10 is almost certainly wrong |
| `real_gap_mass_ratio` | Σŷ / Σλ over real missing slots | Plausibility check against the seasonal profile (not ground truth; sanity only) |
| `night_short_gap_zero_rate` | P(ŷ < 1 \| real gap ≤ 4 steps, 00–05h, heartbeat alive) | The one regime where predicting ~0 *is* desirable |
| `crps`, `pit_hist`, `q90_coverage` | Probabilistic calibration on observed targets | Ensures R5 weighting doesn't destroy calibration |
| `outage_auc` | AUC of `q` vs synthetic-outage labels, plus long-gap heuristic labels | Only for R4 models |

**Selection score for checkpointing:** `S = obs_nll + λ·so_rmse@12_scaled`, i.e. the observed-only likelihood plus a synthetic-outage term computed on a fixed seeded validation outage mask, so it stays deterministic across epochs. Report every metric in the table regardless of which score selected the checkpoint.

**Statistical protocol:** 3 seeds (42, 43, 44) per configuration; report mean ± std; paired comparisons on the identical holdout manifest (`holdout_manifest_sha` must match). Call a difference real if it exceeds 2× the pooled seed std and points the same way on val and test.

---

## 4. Experiment plan

This plan extends the existing v4 suite (`configs/train/v4/`: `full_inductive`, `ablation_no_impute`, `ablation_no_kriging`, `ablation_no_masks`, `ablation_no_pe`, `baseline_lstm`, `spatial_high_dropout`, `spatial_small_batch`). Phases are gated: each phase's winner becomes the base for the next, which keeps the run count manageable.

New config keys (read with `cfg["key"]`, per the project's fail-fast doctrine; add to every v4 YAML):

```yaml
data:
  mask_strategy: iid          # iid | block | transplant | tail | mixed
  mask_block_len_dist: empirical   # geometric | empirical
  unify_missing_encoding: false
  calendar_features: false
  profile_prior: false        # none | transductive | inductive_neighbour | inductive_static
  log1p_inputs: false
  gap_runlength_feature: false
  sensor_embedding_dim: 0
loss:
  target_mask: all            # all (current, contaminated) | observed
  impute_positions: all       # all | masked_observed | masked_observed_plus_aux
  forecast_likelihood: zinb   # zinb | nb | ztnb | outage_mixture
  w_missing_lik: 1.0
  outage_prior_weight: 0.0
  magnitude_weight_alpha: 0.0
  magnitude_weight_y0: 27.0
  underpred_tau: null         # e.g. 0.75
  w_underpred: 0.0
  quantile_heads: []          # e.g. [0.5, 0.9]
  impute_huber_mix: 0.0
  mtl_balancing: none         # none | uncertainty | gradnorm | pcgrad
selection:
  metric: val_loss_total      # val_loss_total | obs_nll_plus_so
```

### Phase 0: Measure the collapse on existing checkpoints (no retraining)

| ID | Description | Output |
|---|---|---|
| E0.1 | Implement the Section 3 metrics as a standalone evaluation script over a saved checkpoint and holdout manifest | Metric table for current `riot-v4-full-inductive` and LSTM baseline |
| E0.2 | Decompose collapse: on real gap slots, plot distributions of `mu`, `pi`, `mu·(1 − pi)` vs gap length and hour | Shows whether collapse is `pi` saturation (C11), `mu` collapse, or both |
| E0.3 | Same metrics for trivial baselines: seasonal-profile median, last observed value, linear interpolation | Floors any model must beat. The profile median is a strong outage baseline; if RIoT doesn't beat it on `so_rmse@96`, that is the headline finding |
| E0.4 | Evaluate all existing v4 ablation checkpoints with the new metrics | Confirms the confound: rankings on contaminated vs clean metrics |

**Gate:** proceed regardless; Phase 0 is the baseline for everything else.

### Phase 1: Target hygiene (R1)

Base: `train_v4_full_inductive.yaml`.

| ID | `target_mask` | `impute_positions` | kriging masked | selection | Hypothesis |
|---|---|---|---|---|---|
| E1.0 | all | all | no | val_loss_total | Control (re-run of current v4 with new metrics) |
| E1.1 | observed (forecast only) | all | no | val_loss_total | Removes the ZINB shortcut; impute/kriging still leak |
| E1.2 | observed | masked_observed | no | val_loss_total | Adds impute hygiene |
| E1.3 | observed | masked_observed | yes | val_loss_total | Full hygiene |
| E1.4 | observed | masked_observed | yes | obs_nll_plus_so | Full hygiene plus correct selection |
| E1.5 | E1.4 with `forecast_likelihood: nb` | | | | `pi` is unnecessary once there are no zeros |

Runs: 6 × 3 seeds = 18. **Expected:** `collapse_rate` drops sharply from E1.0 to E1.3, `obs_rmse` improves, and `so_rmse@96` improves only modestly (needs R2/R3). **Gate:** adopt E1.4 or E1.5 as the base.

### Phase 2: Gap-realistic masking (R2)

Base: Phase 1 winner.

| ID | `mask_strategy` | `unify_missing_encoding` | `mask_prob` / coverage | Hypothesis |
|---|---|---|---|---|
| E2.0 | iid | false | 0.15 | Control |
| E2.1 | iid | true | 0.15 | Encoding unification alone |
| E2.2 | block (empirical lengths) | true | 0.15 | Long-gap recovery |
| E2.3 | transplant | true | 0.15 | Realistic MNAR structure |
| E2.4 | tail (k ∈ [1, T]) | true | 0.15 | Forecast-from-inside-outage |
| E2.5 | mixed (iid 0.25 / block 0.25 / transplant 0.3 / tail 0.2) | true | 0.15 | Combined |
| E2.6–E2.8 | best of above | true | 0.15 / 0.30 / 0.50 | Mask-ratio sweep (MAE literature suggests high ratios help) |

Runs: 9 × 3 = 27. **Key metrics:** `so_rmse@{12, 48, 96}`, `so_mass_ratio`, `collapse_rate`. **Gate:** adopt the best strategy and ratio.

### Phase 3: Information for outage prediction (R3)

Base: Phase 2 winner. Additive design, then leave-one-out on the final stack.

| ID | Change | Hypothesis |
|---|---|---|
| E3.1 | + calendar features | Largest single gain for long gaps |
| E3.2 | + log1p inputs | Stabilises scale; helps peaks in input |
| E3.3a/b/c | + profile prior (transductive / inductive-neighbour / inductive-static) | Strong default level in gaps; check inductive-node gains separately |
| E3.4 | + gap run-length feature | Quiet vs outage separation |
| E3.5 | + sensor embedding (d = 8, embedding dropout 0.2) | Transductive level; measure the inductive penalty |
| E3.6 | `seq_len` 48 | Longer context |
| E3.7 | `seq_len` 96 (or 12 + lag-96/lag-672 channels) | Daily context vs cost |
| E3.8 | Full stack minus each component (leave-one-out) | Attribution for the thesis |

Runs: about 12 × 3 = 36. **Gate:** adopt the full stack minus any component whose removal doesn't hurt (parsimony).

### Phase 4: Likelihood (R4)

Base: Phase 3 winner.

| ID | `forecast_likelihood` | Extras | Hypothesis |
|---|---|---|---|
| E4.0 | nb (or Phase 1 winner) | | Control |
| E4.1 | ztnb | | Exact truncation likelihood on observed |
| E4.2 | outage_mixture | outage head reads full latent | Baseline mixture; check identifiability |
| E4.3 | outage_mixture | outage head reads missingness features only, plus stop-gradient | Prevents `q` from absorbing count signal |
| E4.4 | outage_mixture | E4.3 + supervised BCE on artificial outages | Anchored outage head |
| E4.5 | outage_mixture | E4.4 + gap-length prior (`outage_prior_weight` 0.1) | Heuristic regularisation |
| E4.6 | outage_mixture | E4.4 with `w_missing_lik` ∈ {0.25, 1.0} | How much real-gap likelihood helps or hurts `mu` |
| E4.7 | E4.4 applied to imputation and kriging heads as well | | Consistent treatment across heads |

Runs: 8 × 3 = 24. **Key metrics:** `collapse_rate`, `real_gap_mass_ratio`, `night_short_gap_zero_rate` (should stay high, since true zeros remain zeros), `outage_auc`, `crps`. **Failure signature to watch for:** `q → 1` everywhere, with `mu` drifting upward without bound on missing slots. Detect it via `real_gap_mass_ratio ≫ 1`.

### Phase 5: Magnitude emphasis (R5)

Base: Phase 4 winner.

| ID | Change |
|---|---|
| E5.1 | `magnitude_weight_alpha` 0.5 |
| E5.2 | `magnitude_weight_alpha` 1.0 |
| E5.3 | Under-prediction pinball τ = 0.75, `w_underpred` ∈ {0.1, 0.5} |
| E5.4 | Quantile heads [0.5, 0.9] |
| E5.5 | Imputation Huber mix 0.5 |
| E5.6 | Low-end tolerance band |
| E5.7 | Best combination |

Runs: about 8 × 3 = 24. **Trade-off to report:** a Pareto plot of `peak_rmse` / `peak_underpred_rate` against `crps` and `low_tol_acc`. Pick the knee, not the extreme.

### Phase 6: Multi-task conflict (R6)

| ID | Change |
|---|---|
| E6.0 | Log gradient cosine (forecast vs impute vs kriging) on the Phase 5 winner *and* on E1.0 |
| E6.1 | Uncertainty weighting |
| E6.2 | PCGrad |
| E6.3 | Impute-pretrain → forecast fine-tune |
| E6.4 | Detached-latent imputation head |

Runs: about 5 × 3 = 15. **Hypothesis:** the cosine is strongly negative in E1.0 and close to 0 after R1–R2. If so, skip balancing and report the diagnostic as evidence that the "tension" was target contamination.

### Phase 7: Re-run the thesis ablation suite on the corrected base

Repeat the existing v4 suite on the final configuration ("v4.1"), with the new metrics, 3 seeds each:

| Existing ablation | v4.1 equivalent | Note |
|---|---|---|
| `full_inductive` | v4.1 full | Ceiling |
| `ablation_no_impute` | `w_impute = 0` | With R2 the impute head is the main gap-learning signal. Expect a much bigger effect than before |
| `ablation_no_kriging` | `w_kriging = 0` | |
| `ablation_no_masks` | `ablate_masks = true`, **loss mask still from un-ablated copy** (C12) | Under v4 this ablation may have *looked* worse on contaminated metrics while collapsing less. Worth a sentence in the thesis |
| `ablation_no_pe` | `use_positional_encoding = false` | Calendar features may make PE redundant; check |
| `baseline_lstm` | LSTM with the same target hygiene and calendar features | The baseline adapter also needs R1 (`src/harness/adapters.py`, which uses MSE on all targets) |
| `spatial_high_dropout`, `spatial_small_batch` | Unchanged settings on v4.1 | |
| **new:** `ablation_no_outage_head` | v4.1 with NB instead of the outage mixture | Isolates R4 |
| **new:** `ablation_iid_masks` | v4.1 with iid masks | Isolates R2 |
| **new:** `ablation_no_calendar_profile` | v4.1 without calendar or profile | Isolates R3 |
| **new:** `baseline_profile_median` | Non-learned seasonal-profile baseline | Floor |

Runs: about 12 × 3 = 36.

**Total budget:** about 180 training runs across Phases 1–7 if every phase runs fully. You can cut it to about 90 by running single-seed screens in Phases 2, 3 and 5 and confirming only the top 2 candidates per phase with 3 seeds.

---

## 5. Implementation order (suggested PR breakdown)

1. **PR-A: Evaluation harness (Phase 0).** Section 3 metrics, synthetic-outage evaluator with a fixed seeded outage manifest saved next to `holdout_manifest.json`, and baseline predictors. No training-behaviour change.
2. **PR-B: Target hygiene (R1).** Dataset returns un-ablated observed masks; `compute_loss` / `compute_kriging_loss` / ZINB / baseline adapter accept them; config keys `loss.target_mask`, `loss.impute_positions`, `selection.metric`. Default `all` keeps the current behaviour reproducible for E1.0.
3. **PR-C: Masking strategies and encoding unification (R2).** GPU-side mask generators in `train.py` (keep CUDA RNG, as in the existing §14 remediation), transplant pattern bank built once from the training split.
4. **PR-D: Inputs (R3).** Calendar, log1p, profile prior (train-slice only), gap run-length, sensor embedding. `time_series_dim` grows, so checkpoints are invalidated. Note it in EXPERIMENTS.md, as the MNAR mask change did.
5. **PR-E: Outage-aware likelihood (R4)** in `decoders.py`, with desk checks in `scripts/desk_checks/` mirroring `desk_check_unified_loss.py`: hand-computed NLL for (observed y = 5), (missing, mu = 0.1), (missing, mu = 300), plus the bf16 path.
6. **PR-F: Magnitude losses and quantile heads (R5).**
7. **PR-G: Suite configs** under `configs/train/v4_1/` and registration in `scripts/riotv4/run_v4_server_suite.py`.

Tests to add (`tests/`): loss ignores synthetic-zero targets (changing the value at a missing slot must not change the loss); `ablate_masks` does not change the loss mask; artificial and real gaps produce identical encoder inputs when `unify_missing_encoding` is true; outage-mixture NLL matches hand-computed values; the synthetic-outage evaluator never masks already-missing slots.

---

## 6. Risks and open questions

- **True zeros are unknowable from this source.** The API's omission of zero rows makes "quiet" and "outage" observationally identical slot by slot. R4 separates them statistically, using context, but cannot do so perfectly. The thesis should state this limitation explicitly. If the data provider can confirm whether zero-count intervals are dropped, or supply device-level counting-status logs, that would be the best possible label source.
- **Heartbeat semantics.** A live heartbeat during multi-day gaps suggests the counting pipeline can fail while the device is up. Ask the provider. If heartbeat history extends beyond the one month available locally, it is still a valuable weak feature for short gaps.
- **Profile prior leakage.** It must be fitted on the training slice only, and inductive nodes must not see their own profile.
- **Sensors with ~100% missingness** (15 sensors > 80%): consider excluding them from loss participation entirely while keeping them as graph nodes. With R1 they contribute almost no loss anyway, but they still inflate the denominators of per-node metrics.
- **Distribution shift.** Missingness changes over three years (sensor roll-outs and failures). Check whether the test period's gap structure matches training; transplant masks should be drawn per split.
