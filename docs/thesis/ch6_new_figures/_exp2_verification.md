# Experiment 2 (Chapter 6): implementation facts checked against code and artefacts

Generated 2026-10-13. Read-only checks; no thesis files were edited.

**Where the code lives.** The working tree has only `src/spatial/__pycache__`. The source files are
committed on branch `feature/ch6-spatial-gat-rebuild` (HEAD `2fb322a`). For these checks they were
extracted with `git archive` to `/tmp/sp/`.

The branch source matches what produced `artifacts/spatial_full`:
- The `.pyc` files in the working tree (11 Sep 2026) contain the same docstrings and names: `masked_mse`,
  daily-seasonal MASE and `GATv2Conv`.
- The MASE scale implied by the artefacts matches the daily-naive recomputation to a relative error of
  4×10⁻⁸ (see §3b).
- Run manifests report `git_hash: "unknown"`, so there is no stronger provenance than this.

File references below are to the extracted branch: `src/spatial/...`, `scripts/aggregate_spatial_results.py`,
`configs/spatial/spatial_full.yaml`. Shared modules are in the working tree: `src/features.py`,
`src/graph_learner.py`, `src/quality/stats.py`, `src/dataset.py`.

---

## 1. Input channels (T2-P): D_in = 6, not 3

**Code:**
- `src/spatial/dataset.py:5-13,30-33`: D = 1 + int(mask_channel) + 2·len(time_features).
- `spatial_full.yaml` `inputs`: `mask_channel: true`, `time_features: [day, week]`, giving **D_in = 1 + 1 + 4 = 6**.
- Confirmed by instantiating the model (`input_dim(...) = 6`).

**The six channels, in order:**

| Channel | Content |
|---|---|
| 0 | Count, z-scored per node, with NaN set to 0 |
| 1 | Observed mask (1 = observed) |
| 2, 3 | sin and cos of time-of-day (period 24 h) |
| 4, 5 | sin and cos of time-of-week (period 168 h) |

How the channels are built:
- Time features use the fractional hour, so they resolve to 15 minutes (`dataset.py:36-54`). They are
  identical across nodes.
- Per-node z-score: mean and SD come only from grid steps touched by train windows (`loader.py:163-200`).
- Missing counts are zero-filled after scaling (`dataset.py:67-69`). Unobserved targets are zeroed and masked (`dataset.py:92-95`).

**Static node features:** there are 15 (`src/dataset.py:26-31`), standardised across nodes. Every
run has `use_static_features: false`, so the forecaster never sees them. The only exception is
`gat_learned`, whose graph learner uses them to build the topology.

## 2. Edge features (T2-P): the thesis vector (Eq. `exp2-edge-dim`) does not match the code

The thesis vector is [d_euc, log(d_euc+1), Δx, Δy, cos θ, sin θ, r_class, c_direct, is_bridge].
**None of Δx, Δy, θ, log-distance or is_bridge exist in the code.**

The actual 9 columns come from `src/spatial/graphs.py:34-44` and `src/features.py:59-76, 440-522`.
The cache is `data/external/edge_features_cache.json`.

| col | key | definition |
|---|---|---|
| 0 | `shortest_path_distance` | OSM **walking-network** Dijkstra distance (m), ÷ the global max shortest-path distance in the cache |
| 1 | `euclidean_distance` | haversine straight-line distance (m), ÷ the **same** global max shortest-path distance |
| 2 | `directness` | euclidean ÷ shortest_path, in (0, 1]; low = winding. Self-loop rows = 1 |
| 3–8 | `pedestrian, residential, tertiary, secondary, primary, service` | fraction of shortest-path length on each collapsed OSM `highway` class. The fractions sum to 1 |

How OSM `highway` tags are collapsed into the six classes (`features.py:63-71`):
- **pedestrian:** pedestrian, footway, path, steps, corridor.
- **residential:** residential, living_street.
- **tertiary:** tertiary, tertiary_link, unclassified.
- **secondary:** secondary, secondary_link.
- **primary:** primary, primary_link, trunk, trunk_link.
- **service:** service, track, bridleway, cycleway, plus **any unknown tag**.

Other properties of the edge features:
- Self-loop rows are all zero except directness = 1 (`graphs.py:160-162`).
- **c_direct** in the thesis corresponds to `directness` (circuity inverse = Euclidean / network).
- **r_class** is not one variable. It is six continuous road-class length fractions.
- **There is no is_bridge flag.**

**Topology construction is Euclidean, not on the walking network.**
- Gabriel, kNN-8 and FC are all built from the cached **haversine Euclidean** matrix
  (`graphs.py:57-80, 95-139`).
- The Gabriel test uses squared Euclidean distances (`graphs.py:98-107`).
- The walking network enters **only** through edge-feature columns 0, 2 and 3–8.

Two thesis passages need changing as a result:
- Line 195 ("planar walking corridor adjacency") and the Fig 6.2 caption ("strict planar walking corridor
  adjacency") should say Euclidean proximity.
- The attention-vs-distance analysis also uses Euclidean distance (`aggregate_spatial_results.py:169`).

**Edge-attribute availability by model:**
- `gabriel_plain` and `identity` have no edge attributes (edge_dim None).
- `gat_learned` has 10-dimensional edge attributes: the 9 physical columns plus the learned similarity
  (`gat.py:241-251`, `graph_learner.py:172-209`).

## 3. The six nodes with no test targets

### 3a. Zero test targets: confirmed

Source: `usable_targets.csv`, h=1.

| node | completeness | train targets | val targets | test targets | last observation |
|---|---|---|---|---|---|
| NclJohnDobsonStNorthumberlandRdNorth | 0.309 | 37,024 | 0 | 0 | 2024-10-03 18:45 |
| NclSideStCrownPosadaNorth | 0.333 | 39,886 | 0 | 0 | 2024-09-26 12:30 |
| NclSideStCrownPosadaSouth | 0.344 | 41,230 | 0 | 0 | 2024-09-26 03:00 |
| NclNinthAveSecondAve | 0.356 | 32,188 | 10,478 | 0 | 2025-11-21 11:45 |
| NclJesmondDeneRdBentonBank | 0.303 | 36,261 | 0 | 0 | 2024-10-07 11:00 |
| NclHeatonParkViewHeatonParkRd | 0.559 | 61,117 | 5,834 | 0 | 2025-08-04 15:15 |

What the counts show:
- 48 nodes have test targets and 50 have val targets.
- All six nodes stopped reporting before the test period began on 3 December 2025.

### 3b. The thesis explanation (line 150) is wrong

The thesis says these nodes lack "complete historical cycles to compute the in-sample weekly
seasonal-naive scale denominator". **This is refuted on two counts:**

1. Each of the six has 32k–61k observed train steps, so its MASE scale is finite.
   - Heaton Park View has a val MASE of 0.559, and Ninth Ave also has a val MASE.
   - Their test metrics are NaN only because `n_targets = 0` (`evaluate.py:69-72`).
2. **The MASE scale is DAILY seasonal-naive (m = 96), not weekly (m = 672).**
   - `evaluate.py:22-42` computes `err = |raw[t] − raw[t−96]|` over train-touched steps.
   - I verified this empirically: (val mae_raw ÷ val mase) for `lstm_identity_s42` equals the recomputed
     daily-naive in-sample MAE for every node, with max relative error 4.3×10⁻⁸.
   - The weekly-naive scale differs from the implied scale by up to 30%.

**Consequences for the thesis text:**
- The Fig 6.3 caption (line 356, "weekly seasonal-naive differencing (m = 672 …)") is wrong.
- The explanation of why snaive_weekly scores MASE = 0.865 must be reworded. Under the daily scale,
  MASE < 1 means the weekly-naive forecast beats the in-sample daily-naive forecast.

### 3c. Every node-level summary uses 48 nodes

`aggregate_nodes` and `model_comparison` drop NaN metrics before taking the node median
(`evaluate.py:96-110`; `aggregate_spatial_results.py:68-79`). DM tests also cover 48 nodes, because
there are no observed test targets for the other 6. So MAE, RMSE and MASE in every table are medians
over **48** nodes, not 54.

## 4. Splits: T = 119,712 confirmed. The 45 steps are purge gaps plus the window tail, not series-end data

**Grid.** The grid runs from 2023-01-08 00:00 to 2026-06-07 23:45 at 15 min, giving 119,712 steps
(1,247 days × 96). The source is `context_manifest.json`.

**Window counts** (`splits.py:20-72`, W = 12, max horizon = 4, stride 1, `purge_boundaries: true`):

| Quantity | Value |
|---|---|
| Possible window starts | 119,712 − 12 − 4 + 1 = 119,697 |
| Raw split | train ⌊0.7n⌋ = 83,787; val ⌊0.15n⌋ = 17,954; test the remaining 17,956 |
| Purge | the first ⌈(W + H_max − 1)⌉ = **15** windows of val and of test are dropped |
| Final | **train 83,787 / val 17,939 / test 17,941** (matches the manifest and `usable_targets.csv`) |

**Exact layout:**

| split | window-start timestamps | target timestamps (h=1 … h=4) |
|---|---|---|
| train | 2023-01-08 00:00 → 2025-05-29 18:30 | 2023-01-08 03:00 → 2025-05-29 22:15 |
| *purge* | 15 windows (2025-05-29 18:45 → 22:15) | — |
| val | 2025-05-29 22:30 → 2025-12-02 19:00 | 2025-05-30 01:30 → 2025-12-02 22:45 |
| *purge* | 15 windows (2025-12-02 19:15 → 22:45) | — |
| test | 2025-12-02 23:00 → 2026-06-07 20:00 | 2025-12-03 02:00 → 2026-06-07 23:45 |

**Where the 45 comes from:** 119,712 − (83,787 + 17,939 + 17,941) = 45, made up as follows:
- **15** purged windows at the train→val boundary.
- **15** purged windows at the val→test boundary.
- **15** trailing grid positions that cannot start a full W + H_max window.

**The data run to the end.** The last grid step, 2026-06-07 23:45, has observations, and test targets
reach it.

**Correcting the earlier audit claim.** The earlier claim that "the 45 dropped steps are 7 June 2026
12:45–23:45 at the series end" is **wrong**. It arose because the thesis split dates (line 103–105) were
computed as three **contiguous** blocks of 83,787 / 17,939 / 17,941 window starts. That gives:
- val 18:45 → 2 Dec 15:15;
- test 2 Dec 15:30 → 7 Jun 12:30.

Those dates ignore the purge. The thesis text "45 boundary transition steps between contiguous splits"
is approximately right: 30 of the 45 are boundary purges and 15 are the window tail. But the listed
start and end timestamps of val and test are wrong and should be replaced with the table above.

**Windowing guarantee.** Windows straddling a boundary are dropped by the purge. No grid step touched
by a val or test window (inputs or targets) was touched by an earlier split. This is asserted in
`splits.py:63-71`.

## 5. Diebold–Mariano tests

### Existing implementation

**What it computes:**
- **Script:** `scripts/aggregate_spatial_results.py:115-141` (`run_dm_pairs`), calling
  `src/spatial/stats.py:23-92` (`dm_by_node`, `dm_summary`) and `src/quality/stats.py:221-274`
  (`diebold_mariano`).
- **Pairs** (`spatial_full.yaml` `analysis.dm_pairs`): focal **gat_transformer|gabriel** (not the
  champion) vs lstm|identity, transformer|identity and gat_transformer|identity.
- **Loss: SQUARED error** (`dm_by_node` default `loss="squared"`; the pair loop passes no override).
- **Units:** `mean_loss_diff` is in **counts²**. The `dm_summary.csv` values are −0.9 to −41 counts².
- **Seeds:** one DM per seed (42, 43 and 44 each), with separate BH-FDR per seed. `dm_summary.csv` has
  27 rows (3 pairs × 3 horizons × 3 seeds).

**Variance estimate and its weakness:**
- Long-run variance uses lags 1…h−1 with **uniform (unweighted) autocovariances** plus the
  Harvey–Leybourne–Newbold correction, and a t(n−1) p-value.
- The docstring says "Bartlett", but no Bartlett weights are applied.
- **At h = 1 there is no autocorrelation correction at all.** With 15-minute data the loss
  differential is strongly autocorrelated, so the h = 1 tests are anti-conservative.

**Series handling:**
- Observed targets only.
- Paired on `target_dt` and sorted, so gaps in observation are concatenated.
- BH-FDR is applied across nodes (statsmodels `fdr_bh`, α = 0.05).

**Predictions are saved:** `runs/*/predictions.parquet` holds the test split, long format, for every run.

### New champion DM: `dm_champion_seedavg.csv`

**Method:**
- Focal model: **gat_lstm|gabriel**.
- Forecasts are averaged over seeds 42–44 per (node, horizon, target_dt) before computing the loss.
- Baselines: lstm, transformer and gat_transformer|identity are also seed-averaged over 3 seeds.
  persistence, snaive_weekly and hist_avg are deterministic (1 seed).
- **Absolute-error loss**, so the mean loss difference is in **counts**.
- I used the repository's `diebold_mariano`, giving the same lag h−1 and HLN correction.
- BH-FDR is applied across the 48 nodes for each comparison × horizon.
- For robustness I added a Newey–West Bartlett statistic with lag 96 (1 day), with normal p-values and
  BH correction (`*_nw96` columns).

**Results** (n = 48 nodes in every row):

| comparison | h | % nodes better | % BH-sig, champion better | % BH-sig, baseline better | median DM | mean Δ|e| (counts) | % BH-sig better, NW lag 96 |
|---|---|---|---|---|---|---|---|
| vs lstm\|identity | 1 | 95.8 | 91.7 | 4.2 | −8.93 | −0.189 | 91.7 |
| | 2 | 95.8 | 95.8 | 2.1 | −11.05 | −0.337 | 95.8 |
| | 4 | 97.9 | 97.9 | 0.0 | −11.26 | −0.536 | 97.9 |
| vs transformer\|identity | 1 | 91.7 | 89.6 | 6.3 | −10.08 | −0.259 | 87.5 |
| | 2 | 95.8 | 91.7 | 2.1 | −11.03 | −0.401 | 91.7 |
| | 4 | 100.0 | 95.8 | 0.0 | −10.91 | −0.640 | 95.8 |
| vs gat_transformer\|identity | 1 | 93.8 | 91.7 | 2.1 | −9.69 | −0.229 | 91.7 |
| | 2 | 97.9 | 93.8 | 2.1 | −10.60 | −0.374 | 95.8 |
| | 4 | 100.0 | 97.9 | 0.0 | −11.13 | −0.593 | 97.9 |
| vs gat_transformer\|gabriel | 1 | 64.6 | 47.9 | 20.8 | −1.96 | −0.043 | 45.8 |
| | 2 | 70.8 | 54.2 | 16.7 | −2.83 | −0.052 | 54.2 |
| | 4 | 68.8 | 54.2 | 14.6 | −2.51 | −0.064 | 54.2 |
| vs snaive_weekly | 1/2/4 | 100/100/100 | 100/100/97.9 | 0 | −33.2/−24.6/−16.9 | −2.92/−2.57/−2.12 | 100/100/95.8 |
| vs persistence | 1/2/4 | 100/100/100 | 100/100/100 | 0 | −32.2/−29.5/−29.1 | −2.09/−2.71/−3.26 | 100 |
| vs hist_avg | 1/2/4 | 100/100/100 | 100/100/100 | 0 | −33.7/−24.5/−17.7 | −4.05/−3.67/−3.17 | 100 |

**Caveats:**
- Seed-averaging makes each trained model a 3-member ensemble. Against the deterministic baselines,
  only the champion benefits from this.
- Per-node detail is in `/tmp/dm_champion_seedavg_pernode.csv` (scratch). Re-run `/tmp/sp_dm.py` from the
  repository root to regenerate it.

## 6. Parameter counts (T2-O): `param_counts.csv`

No run artefact logs a parameter count. The counts below come from instantiating each module from its
run `model_spec` on CPU, with N = 54, D_in = 6, W = 12 and 3 horizons. They are trainable parameters.

| config | params | breakdown |
|---|---|---|
| lstm\|identity | 51,907 | LSTM 51,712 + linear head 195 |
| transformer\|identity | 68,355 | encoder 68,160 + head 195 |
| gat_plain\|gabriel | 31,235 | flatten-MLP 8,832 + GAT 18,048 + MLP decoder 4,355 |
| **gat_lstm\|gabriel (champion)** | **74,115** | LSTM 51,712 + GAT 18,048 + decoder 4,355 |
| gat_transformer\|gabriel / knn8 / fc | 90,563 | encoder 68,160 + GAT 18,048 + decoder 4,355 |
| gat_transformer\|identity, gat_transformer\|gabriel_plain | 89,411 | GAT 16,896 (no edge_dim projection) |
| gat_learned\|identity | 91,203 | + learner 512, GAT 18,176 (edge_dim 10) |
| gat_h1, gat_h8 (gabriel) | 90,563 | the head count does not change parameters (hidden/heads per head, concatenated) |
| gat_l1 | 81,539 | GAT 9,024 |
| gat_l3 | 99,587 | GAT 27,072 |
| persistence, snaive_*, hist_avg | 0 | — |

**Two points matter for the thesis's parity claims:**
- The per-node temporal baselines use a single linear head. The GAT models add a 2-layer MLP decoder
  and a residual GATv2 stack. GAT-LSTM therefore has 43% more parameters than LSTM
  (74,115 vs 51,907).
- The parameter-matched "only message passing differs" control is the GAT with the identity graph.
  That control exists only for the Transformer backbone (gat_transformer|identity, 89,411 vs 90,563).
  **There is no gat_lstm|identity run.**

## 7. Tuning (T2-N)

**There is no evidence of the grid search described at thesis lines 226–232** (learning rate
{1e-4, 5e-4, 1e-3, 2e-3}, patience {5, 8, 12}, dropout {0, 0.1, 0.2}, batch {16, 32, 64}). Places searched:
- `configs/spatial/` contains only `spatial_full.yaml` and `spatial_smoke.yaml`.
- `artifacts/spatial_smoke` holds 6 smoke runs with the same lr, dropout and batch size, and patience 2.
- The implementation plan (`docs/spatial-depedencies/implementation_plan.md`) and the branch commit log
  mention no tuning stage.
- If a search was run elsewhere, the author must supply the artefacts. Otherwise the passage must be
  removed or reworded as "fixed a priori".

**Baselines were not tuned separately.** Every trainable model uses the identical fixed block:
- lr 1e-3, batch 32, 60 epochs, patience 8, StepLR(5, 0.7), gradient clip 1.0, dropout 0.1, hidden 64.

**Many runs hit the 60-epoch cap without early stopping**, which suggests they were under-trained.
- All three gat_lstm|gabriel seeds stopped at 60 epochs.
- So did gat_transformer knn8 ×3 and gabriel_plain ×3.
- Source: manifests, `fit_report.epochs_run`.

**Other mismatches between thesis lines 219–224 and the code:**

| Thesis claim | Code |
|---|---|
| "weight decay 10⁻⁴" | Plain `torch.optim.Adam(params, lr)` with **no weight decay** (`models/base.py:92`) |
| MAE training loss (Eq. `exp2-masked-loss`, line 128) | **Masked MSE on z-scored targets**, pooled over nodes and horizons (`dataset.py:109-122`, `base.py:113`; the `.pyc` confirms `masked_mse`) |
| Early stopping on "node-median validation MAE at h=1" | Early stopping on **pooled masked val MSE (z-scored, all horizons)**, threshold 1e-6 (`base.py:120-142`). Champion *selection* does use node-median val MAE at h=1 (`aggregate_spatial_results.py:82-103`) |
| — | Evaluation metrics are MAE on raw counts over observed targets, which is correct as described |

**Capacity ablations (gat_h1, h8, l1, l3) are GAT-Transformer, not GAT-LSTM.** Evidence:
- Every C_ablations manifest has `temporal_encoder: transformer`.
- The config comments say "Defined against gat_transformer; re-point at the champion after Stage A".
  This was never done.
- The edge-feature ablation (gabriel_plain) and the graph ablations (identity, knn8, fc) are also
  gat_transformer.
- **gat_lstm was run only on gabriel.**

## 8. GATv2

`GATv2Conv` is used everywhere (`models/gat.py:25,54-58`; the `.pyc` confirms it), with
`add_self_loops=False`. Self-loops are supplied explicitly by every static GraphSpec
(`graphs.py:83-86`). There is **no GAT-v1 (`GATConv`) ablation** anywhere.

## 9. Learned adjacency (`gat_learned`)

The thesis describes A = ReLU(tanh(E₁E₂ᵀ)). **The code is different** (`src/graph_learner.py:211-250`):

    E = Linear(15 static node features → 32)          # one embedding, E₁ = E₂
    A = ReLU( ReLU(E Eᵀ) − 0.5 )                      # fixed threshold 0.5, no tanh
    edge_index, edge_weight = dense_to_sparse(A)

Properties that follow from this code:
- **Symmetric:** E·Eᵀ.
- **Threshold, not top-k.**
- **Static after training:** it is a function only of the fixed static features. It is **not dynamic**
  over time or input; the class name `DynamicGraphLearner` is misleading.
- **Diagonal survives:** 54 self-loops are present in every seed.

**Geographic information is not absent, contrary to the thesis.** The thesis says "initialised without
geographic coordinates or physical distance priors". In fact:
- The static features include `dist_to_city_centre` and network centralities.
- Every learned edge carries the 9 physical edge features (both distances, directness, road fractions)
  plus the learned weight as edge attributes (`edge_attr_mode: physical_plus_learned`).

**Edge counts** come from manifest `resolved_graph`:

| seed | non-self edges | self-loops |
|---|---|---|
| 42 | 1,756 | 54 |
| 43 | 1,696 | 54 |
| 44 | 1,762 | 54 |

"≈1,756" is the **seed-42** value. The 3-seed mean is 1,738.

## 10. Attention exports

**The source is SEED 42 only.**
- Written by `scripts/aggregate_spatial_results.py:160-211`, which uses `records[0]`. Records are loaded
  from files sorted by name, so `_s42` comes first.
- Verified directly: `attention_distance_gat_transformer_gabriel.csv` matches the s42 `attention.parquet`
  (head-mean) to 2×10⁻⁸. It differs from s43 and s44 by up to 0.22 and 0.17.
- The same `records[0]` is used for `attention_entropy.csv`, the `attention_top_edges_*.csv` files and the
  `attention_graph_*.png` and `attention_distance_*.png` figures.
- `attention_stability.csv` is the only output that uses all 3 seeds.

**What each record holds:**
- Final GATv2 layer only.
- Mean over 4 heads.
- Mean over **256 randomly sampled test windows**, with RNG seeded by the run seed
  (`models/gat.py:130-178`; runner `predict_attention(splits["test"], …)`, `run_spatial_experiments.py:158-166`).
- It includes self-loop weights.

**Attention exists only for gat_transformer and gat_learned** (`run.attention.models`). The champion
gat_lstm has `attention_written: false`, so **no attention analysis of the champion is possible without
a re-run.**

## Other discrepancies found along the way

- **hist_avg** uses **hour-of-week** buckets (168 hourly bins; `models/temporal.py:120-135`). The
  thesis (line 245) says "15-minute time-of-day bin".
- **Seasonal-naive** baselines fall back to persistence when the seasonal reference is unobserved
  (`temporal.py:75-117`). The thesis does not mention this.
- **The "identical except message passing" claim does not hold for lstm vs gat_lstm.** The decoder
  differs: a linear head vs a 2-layer MLP (see §6).
