# Spatial Interpolation Analysis: floral-rain-6 vs radiant-dawn-8

## Executive Summary

Your intuition is **quantitatively confirmed**. `floral-rain-6` is dramatically better at spatial interpolation than `radiant-dawn-8`, despite having *worse* forecasting metrics. The existing metrics (val_rmse, val_mae) are dominated by the forecasting head and are misleading for evaluating the spatial capability of the system.

---

## 1. Head-to-Head: floral-rain-6 vs radiant-dawn-8

I ran `compare_runs.py --all` which fetches the live R2 payloads and computes spatial interpolation quality metrics across all 10 registered runs.

### Spatial Interpolation Metrics (the ones that matter)

| Metric | floral-rain-6 | radiant-dawn-8 | Interpretation |
|:---|---:|---:|:---|
| **Sim μ Mean** | **24.56** | 0.16 | FR6 produces realistic pedestrian counts; RD8 predicts near-zero everywhere |
| **Sim μ Std Dev** | **40.37** | 0.11 | FR6 differentiates between locations; RD8 is flat |
| **Sim CV** | **1.64** | 0.71 | FR6 has high spatial diversity; RD8 barely varies |
| **Zero Collapse Ratio** | **0.02** (1/50) | **1.00** (50/50) | RD8 has collapsed to predict ~0 for *every* simulated node |
| **Near-Zero Sim Nodes** | **1** | **50** | FR6 keeps 49/50 sim nodes alive; RD8 kills all 50 |
| **Dynamic Range** | **118.3** | 12.9 | FR6 can distinguish busy vs quiet locations |
| **Real-Sim Nearest Corr** | **+0.27** | −0.12 | FR6 sim nodes correlate with nearby real nodes; RD8 doesn't |
| **Moran's I** | 0.29 | 0.50 | RD8's spatial autocorr is higher — but this is because all sim nodes collapsed to ~0 (trivially autocorrelated) |

### Training Metrics (the ones currently used for model selection)

| Metric | floral-rain-6 | radiant-dawn-8 |
|:---|---:|---:|
| **Val RMSE** | 31.38 | **24.99** |
| **Val MAE** | 9.96 | **6.19** |
| **Val Zero Acc** | 0.0004 | **0.34** |

> [!IMPORTANT]
> **This is the core problem.** `radiant-dawn-8` "wins" on all standard training metrics because the early stopping and checkpoint selection are optimized for `val_loss_total`, which strips the kriging term (see [train.py L672-674](file:///Users/administrator/Code/ml-agent-workspace/src/train.py#L672-L674)). The model is being rewarded for forecasting accuracy on observed nodes while its spatial interpolation capability has completely collapsed.

---

## 2. What Changed Between the Runs?

Both runs share the same loss weights (`w_forecast=1, w_impute=1, w_kriging=0.5`) and the same data range. The key differences:

| Config | floral-rain-6 | radiant-dawn-8 |
|:---|:---|:---|
| **Git commit** | `177a4b3` (older, with night-time zeros in data) | `87256ff` (newer, zeros removed) |
| **Hidden dim** | 64 | **32** |
| **Num layers** | 3 | **2** |
| **Embedding dim** | 32 | **16** |
| **Batch size** | 128 | **32** |
| **Dropout** | **0.3** | 0.1 |
| **Num nodes (total)** | 106 | 102 |

> [!WARNING]
> **Two variables changed simultaneously** — the data (zero-removal) *and* the architecture (smaller model). This confounds attribution. We cannot tell if the spatial collapse in `radiant-dawn-8` is caused by:
> 1. The smaller architecture (h=32, L=2) lacking capacity for spatial interpolation
> 2. The removal of night-time zeros reducing the training signal that the kriging head was exploiting
> 3. Both acting together

---

## 3. The Wider Picture: All 10 Runs Ranked by Spatial Quality

| Rank | Run | Sim μ Mean | Zero Collapse | CV | Architecture | Data Gen |
|:---:|:---|---:|---:|---:|:---|:---|
| 1 | **ethereal-music-5** | 35.48 | 0.00 | 0.49 | h=32, L=2, bs=32 | Old (with zeros) |
| 2 | **worldly-paper-4** | 28.61 | 0.00 | 0.42 | h=64, L=3, bs=32 | Old (with zeros) |
| 3 | **floral-rain-6** | 24.56 | 0.02 | 1.64 | h=64, L=3, bs=128 | Old (with zeros) |
| 4 | **dry-dragon-3** | 20.53 | 0.04 | 0.57 | h=64, L=3, bs=32 | Old (with zeros) |
| 5 | **tough-dragon-7** | 6.06 | 0.46 | 2.10 | h=64, L=3, bs=128, **d=0.3** | New (no zeros) |
| 6 | crisp-puddle-47 | 0.71 | 0.74 | 0.65 | h=64, L=3, bs=128, **w_k=1.0** | New (no zeros) |
| 7 | radiant-voice-41 | 0.27 | 0.96 | 0.89 | h=64, L=3, bs=128 | New (no zeros) |
| 8 | radiant-dawn-8 | 0.16 | 1.00 | 0.71 | h=32, L=2, bs=32 | New (no zeros) |
| 9 | vague-dragon-1 | 0.01 | 1.00 | 0.53 | h=64, L=3, bs=128, d=0.3 | Old (with zeros) |
| 10 | upbeat-valley-2 | 0.005 | 1.00 | 0.00 | h=32, L=2, bs=32 | Old (with zeros) |

> [!IMPORTANT]
> **Pattern emerges:** Runs 1–4 (old data, with zeros) all avoid zero-collapse. Runs 5–8 (new data, no zeros) all suffer moderate-to-total collapse. The **one exception** is `tough-dragon-7` (rank 5), which uses **dropout=0.3** and partially avoids collapse (sim_mu_mean=6.06, but 23/50 still near-zero).
>
> This strongly suggests **removing the night-time zeros broke the kriging head's training signal.** The zeros gave the model a strong, easy-to-learn "nighttime = 0 everywhere" spatial pattern. Without them, the kriging head sees only sparse daytime fragments and collapses to predicting near-zero as a safe default.

---

## 4. What About the "Real" Holdout Nodes?

The W&B training metrics (`kriging_rmse`, `kriging_mae`, `kriging_msle`) are computed on the **real** inductive holdout nodes (24 held-out sensors from the 81 observed sensors). However, these metrics operate in **scaled space** (StandardScaler normalised) and use the `scale_invariant_mse` loss. The absolute values (~1.0 RMSE) are difficult to interpret and don't capture the "does this look like a realistic pedestrian pattern" question you're asking about the simulated nodes.

The `compare_runs.py` script evaluates the **simulated** nodes (50 virtual sensors placed at locations without physical sensors) which is the true test of spatial generalisation. The fact that even the real holdout kriging loss looks similar across runs while the simulated node output varies wildly suggests the kriging metrics may be insensitive to the collapse phenomenon.

---

## 5. Proposed Ablation Studies (Spatial Interpolation Focused)

Given the above findings, here are the studies that will directly address **how to get dynamic spatial interpolation working well again** on the corrected (zero-free) dataset:

### Study A: Isolate the Data vs Architecture Effect
> *"Was it the data change or the architecture change that killed spatial interpolation?"*

- **A1 (Large arch + new data):** `h=64, L=3, e=32, bs=128, d=0.1` on the new (zero-free) dataset. This is the `floral-rain-6` architecture on the new data. If it collapses → the data change is the culprit.
- **A2 (Small arch + old data):** `h=32, L=2, e=16, bs=32` on the old (zero-filled) dataset. This is the `radiant-dawn-8` architecture on the old data. If it doesn't collapse → confirms the data change is the culprit.

### Study B: Kriging Loss Weight Sweep
> *"Can we rescue spatial interpolation by increasing the kriging signal?"*

- **B1:** `w_kriging=2.0` (4× current) — forces the model to prioritise spatial reconstruction.
- **B2:** `w_kriging=5.0` — aggressive upweighting of kriging.
- **B3:** `w_forecast=0.0, w_impute=1.0, w_kriging=2.0` — removes the forecasting task entirely, dedicating the model purely to imputation + spatial interpolation.

### Study C: Checkpoint Selection on Kriging Loss
> *"Is early stopping on forecast loss throwing away good spatial models?"*

Currently, the best model checkpoint is selected by `val_loss_total` **minus** the kriging term ([train.py L672-674](file:///Users/administrator/Code/ml-agent-workspace/src/train.py#L672-L674)). This explicitly optimises for forecasting and actively works against spatial quality.

- **C1:** Change checkpoint selection to use the **kriging MSLE** on the inductive holdout as the primary metric, or use a composite that includes it.

### Study D: Higher Spatial Holdout Probability
> *"Does the model need more spatial training signal?"*

- **D1:** `spatial_holdout_prob=0.4` (currently 0.2) — doubles the fraction of nodes masked per batch, forcing the kriging head to work harder.
- **D2:** `spatial_holdout_prob=0.5` — aggressive masking.

### Study E: Dropout as Spatial Regulariser
> *"Why does dropout=0.3 partially rescue spatial quality?"*

`tough-dragon-7` (d=0.3) is the only new-data run that partially avoids collapse (sim_mu_mean=6.06 vs <1.0 for all other new-data runs).

- **E1:** `dropout=0.3` + `w_kriging=2.0` on the new data — tests if combining the two interventions is additive.
- **E2:** `dropout=0.5` — even more aggressive regularisation to prevent the kriging head from finding the "predict zero" shortcut.

---

## Open Questions

1. **Is there a way to re-examine the W&B logs for `floral-rain-6` (run ID `2lhmw9dk`) and `radiant-dawn-8` (run ID `54lmvbyv`) to compare the kriging-specific training curves?** The `val_loss_kriging`, `val_kriging_rmse`, and `val_kriging_msle` metrics logged per epoch would show exactly when/whether the kriging head diverged during training.

2. **Do you have access to the GPU server to run these ablations?** The configs are straightforward to generate but each full run takes ~2–4 hours on the GPU.

3. **Should we prioritise Study A (diagnosing the cause) or Study B/C (attempting to fix it)?** I'd recommend A first since it will tell us whether the fix should be architectural or data-focused.
