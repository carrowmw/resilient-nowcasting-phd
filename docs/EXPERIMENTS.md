# Comprehensive Guide to Experiments & Results

This document summarises the experimental methodology, configurations, key findings, and exact reproduction commands across all three research tracks in this codebase.

---

## 1. Overview of Experimental Tracks

| Track | Scope & Objective | Core Package & Configs | Scale |
| :--- | :--- | :--- | :--- |
| **Track 1: Sensor Quality & Observability** | Quantify how real-world IoT missingness, gap fragmentation, and interpolation impact single-sensor forecasting and establish minimum data-quality thresholds. | `src/quality/`<br>`configs/quality/`<br>`scripts/experiment1/` | 81 directional sensor streams × 6 model families × 5 interpolation regimes |
| **Track 2: Spatial Topology & GATv2** | Isolate whether graph attention message passing (`GATv2`) over road/proximity topologies outperforms capacity-matched node-isolated temporal encoders (`identity`). | `src/spatial/`<br>`configs/spatial/`<br>`scripts/run_spatial_experiments.py` | 41 active nodes × 4 topologies (`identity`, `gabriel`, `knn`, `fc`) + learned graph × 3 seeds |
| **Track 3: Resilient IoT (RIoT v4–v7 & Scaling)** | Multi-task spatiotemporal forecasting, self-healing imputation, synthetic outage resilience (`SO@1..96`), and zero-shot inductive transfer to 24 held-out stations. | `src/pipeline.py`, `src/harness/`<br>`configs/train/{v4,v5,v5_scaling,v6,v7}/`<br>`scripts/riotv5/` | **404 total GPU runs** (209 in v5, 69 in scaling, 78 in v6, 48 in v7) |

---

## 2. Track 1: Sensor Data Quality & Observability (`src/quality/`)

### 2.1 Experimental Design
Urban IoT pedestrian sensors suffer from severe, structured Missing-Not-At-Random (MNAR) dropouts (battery depletion, backhaul outages, street works). Track 1 evaluates 81 directional pedestrian streams on a 15-minute grid:
- **Quality Metrics (`src/quality/quality_metrics.py`)**: Completeness fraction, gap run-length distribution, longest consecutive run, usable window fraction, and Lomb–Scargle diurnal/weekly spectral power (`src/quality/periodogram.py`).
- **Models (`src/quality/forecasters.py`, `transformer_forecaster.py`)**: Persistence, Seasonal Naive (1d, 7d), SARIMAX (Kalman walk-forward), Univariate & Multivariate Bi-LSTM, and Transformer.
- **Interpolation Regimes (`src/quality/interpolation.py`)**: `none`, `night_zero`, `linear`, `spline`, and `knn` — with evaluation strictly masked to genuinely observed ground-truth targets so artificial interpolants never inflate accuracy.

### 2.2 Key Findings
1. **Fragmentation vs. Completeness**: Usable contiguous window fraction and Lomb–Scargle diurnal periodicity are stronger predictors of deep sequence model advantage over Seasonal Naive baselines than raw completeness alone.
2. **Segmented Breakpoint Threshold**: Piece-wise segmented regression identifies a critical completeness/continuity threshold below which deep recurrent/attention models fail to beat simple seasonal profiles, and above which Multivariate Bi-LSTM and Transformer models achieve statistically significant Diebold–Mariano gains (BH-FDR $q < 0.05$).
3. **Interpolation Distortion**: Naive `linear` and `spline` interpolation across multi-hour daytime outages distort learned diurnal dynamics; preserving explicit missingness (`none`) or scoring strictly on real observations is essential.

### 2.3 Reproduction Commands
```bash
# Run the fast smoke suite (3 sensors × 5 models × 2 interpolation variants)
docker compose run --rm cpu-runner python scripts/experiment1/run_quality_experiments.py \
  --config configs/quality/quality_smoke.yaml

# Run the full 81-sensor experiment suite
docker compose run --rm gpu-runner python scripts/experiment1/run_quality_experiments.py \
  --config configs/quality/quality.yaml

# Aggregate results and generate statistical tables & figures
docker compose run --rm cpu-runner python scripts/experiment1/aggregate_quality_results.py \
  --config configs/quality/quality.yaml
```

---

## 3. Track 2: Spatial Dependence & Graph Topology (`src/spatial/`)

### 3.1 Experimental Design
Prior GNN studies often confound spatial message passing with changes in model capacity, loss functions, or temporal windowing. Track 2 enforces a strict **`GraphForecaster` contract (`src/spatial/contract.py`)**:
- Every model predicts the exact same `[B, N, T, D]` chronological windows across 41 quality-vetted sensors under identical masked MSE loss and train-only per-node scalers.
- **Matched-Capacity Control (`identity` topology)**: Runs the exact same `SpatialGATv2` architecture with edges restricted to self-loops ($i \to i$). Any gain of `gabriel`, `knn`, `fc`, or `learned` topology over `identity` is purely attributable to cross-sensor spatial message passing.
- **Horizons**: $h = 1$ (15 min ahead) and $h = 4$ (60 min ahead).

### 3.2 Key Findings
1. **Planar & Sparse Topologies Beat Dense Graphs**: The planar **Gabriel graph** and **$k$-NN ($k=8$)** consistently outperform the Fully Connected (`fc`) graph, which suffers from attention dilution (high attention entropy across distant, uncorrelated sensors).
2. **Horizon Scaling**: Spatial message passing provides modest gains at $h = 1$ (where local autoregressive history dominates) but yields larger, statistically significant paired Diebold–Mariano improvements at $h = 4$ (60 min), as pedestrian flows propagate across the urban network.
3. **Physical Distance Decay**: Learned GATv2 attention weights exhibit clear exponential decay with road-network distance while selectively up-weighting corridors sharing high retail/transit POI similarity.

### 3.3 Reproduction Commands
```bash
# Run the smoke test (6 jobs on real data)
docker compose run --rm cpu-runner python scripts/run_spatial_experiments.py \
  --config configs/spatial/spatial_smoke.yaml

# Run the full multi-seed spatial suite (Stages A, B, C)
docker compose run --rm gpu-runner python scripts/run_spatial_experiments.py \
  --config configs/spatial/spatial_full.yaml

# Aggregate Diebold-Mariano tests, attention decay metrics, and figures
docker compose run --rm cpu-runner python scripts/aggregate_spatial_results.py \
  --config configs/spatial/spatial_full.yaml
```

---

## 4. Track 3: Resilient IoT (RIoT) Unified Pipeline (Stages v4 → v7)

Track 3 evaluates the full multi-task RIoT architecture across **81 sensor stations**: **57 active training stations** (transductive) and **24 permanently held-out stations** (`holdout_sha: 669860aa8b4cdd15`, inductive zero-shot evaluation). All metrics are scored in raw pedestrian counts on **genuinely observed targets only**.

### 4.1 Stage Progression & Headline Results

| Stage & Report | Runs | Key Architectural Innovation | Transductive RMSE | Held-Out (Inductive) RMSE | Outage Resilience (`SO@12` RMSE) | 24h Outage Mass Ratio (`SO@96`) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **RIoT v4 Replica (`v4r_full`)** | 51 | Purged 70/15/15 splits, train-only scalers, ZINB loss on all slots | 16.96 | 46.66 | 31.83 | 0.00 (collapsed) |
| **RIoT v5 Full (`v5_full`)**<br>[Report](experiments/riot_v5/riot_v5_results_report.md) · [3-Seed Report](experiments/riot_v5/riot_v5_seed_report.md) | 209 | **Unified missingness encoding**, observed-only targets, structured masks, lag 96/672, outage mixture | 13.56 ± 0.42 | 45.28 ± 1.04 | 15.96 ± 0.35 (**−50% vs v4**) | 1.01 (**solved collapse**) |
| **RIoT v5 Inductive Zero (`e3_3d`)** | 3 | Zero profile prior + no sensor ID embedding + unified encoding | 16.16 | **39.09** (s42) | 17.65 | 0.88 |
| **Sparsity Scaling Suite**<br>[Report](experiments/riot_v5_scaling/scaling_summary.md) | 69 | Sweep $N_{\text{train}} \in \{15, 25, 35, 45, 51, 57\}$ vs 24 frozen held-out stations | — | 46.93 ($N=15$) → **40.42** ($N=57$) | — | — |
| **RIoT v6 Static-Ridge (`v6_profile_static`)**<br>[3-Seed Report](experiments/riot_v6/riot_v6_seed_report.md) | 78 | Cosine LR warmup, `min_epochs=40`, NB likelihood, **static-feature ridge profile prior** (52 train nodes) | 12.30 ± 0.12 | 41.72 ± 1.29 (**−8.5% vs v6 base**) | 14.50 ± 0.20 | 0.91 |
| **RIoT v7 Static-Ridge (`v7_profile_static_e150`)**<br>[6-Seed Report](experiments/riot_v7/riot_v7_seed_report.md) | 48 | 6-seed confirmatory run on full **57 training stations** + static-ridge prior + 150 epochs (best seed `s47`: **20.08** all-node / **33.30** held-out) | **11.55 ± 0.15** | **41.31 ± 4.10** (s47: **33.30**) | **13.88 ± 0.21** | 0.90 |

### 4.2 Detailed Findings by Stage

#### A. RIoT v5 (209 Runs — Zero-Collapse Remediation & Ablation Suite)
- **Root Cause of Zero Collapse**: In v4, artificial training masks blanked dynamic features to `0.0` while leaving structural MNAR mask channels untouched (`0.0`), whereas real sensor outages had MNAR masks set to `1.0`. The encoder learned to treat `MNAR = 1.0` as "output zero".
- **Unified Missingness Encoding (`unify_missing_encoding: true`)**: Making synthetic training masks and real sensor gaps indistinguishable to the encoder reduced 3-hour Synthetic Outage (`SO@12`) RMSE by **50%** (31.8 → 15.8) and restored 24-hour outage predicted mass (`SO@96`) from **0.00 to 1.01** of true traffic.
- **Leave-One-Out (LOO) Ablations**: Removing unified missingness encoding from `v5_full` (`loo_no_unify`) caused `SO@12` RMSE to jump by +82% and `SO@96` mass to collapse back to 0.14, confirming it as the primary driver of sensor outage resilience.
- **Full Reports**: [`docs/experiments/riot_v5/riot_v5_results_report.md`](experiments/riot_v5/riot_v5_results_report.md) and [`docs/experiments/riot_v5/riot_v5_seed_report.md`](experiments/riot_v5/riot_v5_seed_report.md).

#### B. Spatial Sparsity Scaling Suite (69 Runs)
- Swept active training station counts $N_{\text{train}} \in \{15, 25, 35, 45, 51, 57\}$ over the $21.0\text{ km}^2$ Newcastle network against the frozen 24 held-out stations.
- Fitted the power-law scaling curve $\text{RMSE}(N) = A \cdot N^{-\alpha} + \text{RMSE}_\infty$, demonstrating that inductive GNN error drops sharply at $N_{\text{train}} = 57$ ($2.71\text{ stations/km}^2$, RMSE 40.42 vs. 42.63 for nearest geographic donor).
- **Full Report**: [`docs/experiments/riot_v5_scaling/scaling_summary.md`](experiments/riot_v5_scaling/scaling_summary.md).

#### C. RIoT v6 (78 Runs — Protocol Hardening & Static-Ridge Prior)
- Replaced early-stopping noise with a cosine learning rate schedule (3-epoch linear warmup, `min_epochs=40`, patience 20), cutting transductive seed variance in half (1.0% vs 2.0%).
- Introduced `profile_prior: inductive_static` (`v6_profile_static`), which predicts an unseen station's traffic volume scale from its 15 static POI/network centrality features via ridge regression. Across 3 seeds, `v6_profile_static` reduced held-out RMSE by **8.5%** (41.72 vs 45.59) and all-node RMSE by **7.1%** (24.89 vs 26.81) without degrading transductive accuracy (12.30).
- **Full Report**: [`docs/experiments/riot_v6/riot_v6_seed_report.md`](experiments/riot_v6/riot_v6_seed_report.md).

#### D. RIoT v7 (48 Runs — 6-Seed Confirmatory Stage on 57 Stations)
- Evaluated 8 configurations across 6 seeds (`42–47`) on the full 57-station training set:
  - **Tier 1 (`v7_base` vs `v7_profile_static`)**: Confirmed that the static-ridge prior wins **16 of 18 seed-matched pairs** across mixed masks, block masks, and 150-epoch budgets (sign test $p = 0.031$, mean held-out reduction −5.5%).
  - **Tier 2 (Inductive Validation Gate)**: Pre-check proved that small validation station subsets (8 stations) have low rank correlation ($\rho = 0.14$) with the remaining held-out stations, automatically skipping redundant training runs.
  - **Tier 5 (150-Epoch Budget)**: `v7_profile_static_e150` achieved the best overall performance in the project: **23.89 ± 2.18** mean all-node RMSE (**20.08** on best seed `s47`), **41.31 ± 4.10** held-out RMSE (**33.30** on `s47`), **11.55** transductive RMSE, and **13.88** `SO@12` outage RMSE.
- **Full Report**: [`docs/experiments/riot_v7/riot_v7_seed_report.md`](experiments/riot_v7/riot_v7_seed_report.md).

### 4.3 Reproduction Commands for Track 3

```bash
# 1. Single training run (e.g., RIoT v7 static-ridge champion)
docker compose run --rm gpu-runner python src/train.py \
  --config configs/train/v7/g5_epochs/v7_profile_static_e150__s47.yaml

# 2. Run the full v5 / v6 / v7 suites on a multi-GPU server
./scripts/riotv5/start_v5_suite.sh --gpus 0 1 2 3
./scripts/riotv5/start_v6_stage.sh --gpus 0 1 2 3
./scripts/riotv5/start_v7_stage.sh --gpus 0 1 2 3

# 3. Regenerate CSV summaries and figures from run manifests
docker compose run --rm cpu-runner python scripts/riotv5/seeds_v5_results.py
docker compose run --rm cpu-runner python scripts/riotv5/seeds_v6_results.py
docker compose run --rm cpu-runner python scripts/riotv5/seeds_v7_results.py

# 4. Export paginated payloads for the top models to the frontend / Cloudflare R2
bash scripts/generate_all_payloads.sh
```
