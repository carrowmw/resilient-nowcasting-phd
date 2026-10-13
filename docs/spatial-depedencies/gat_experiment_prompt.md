# Prompt: Rebuild Experiment 2 (GAT Spatial Forecasting) Inside a Quality-Track-Style Harness

> [!IMPORTANT]
> **Objective:** Generate a complete implementation plan for a new experiment branch `feature/ch6-spatial-gat-rebuild` that re-implements the legacy GAT research (RO2: *Multi-Sensor Spatio-Temporal Modelling via Graph Attention*) inside a new, self-contained test harness. The harness must mirror the architecture of the existing `src/quality/` package (the "chapter 5" / MDPI Sensors quality track), but operate on the **multi-node spatiotemporal graph forecasting** domain instead of per-sensor univariate/multivariate forecasting.

---

## 1. Background Context

You are working on a PhD thesis in urban mobility prediction. The codebase at `/Users/administrator/Code/ml-agent-workspace` contains:

### 1.1 The Three Thesis Experiments (from `docs/context/chapter4_methodology.tex`)

| Experiment | Chapter | Research Objective | Domain |
|---|---|---|---|
| **Exp 1** — Temporal baselines & missingness | Ch 5 | RO1 | Single-sensor temporal forecasting (LSTM, Transformer, ARIMA, naive) |
| **Exp 2** — Spatial graph attention | Ch 6 | RO2 | Multi-sensor spatio-temporal forecasting via GATv2 |
| **Exp 3** — Resilient reconstruction | Ch 7 | RO3 | Full RIoT pipeline (imputation + forecasting + kriging) |

**This prompt concerns Experiment 2 (RO2).** The goal is to isolate the marginal value of spatial graph neural networks (GATv2) over purely temporal baselines. The question is: *Does explicitly modelling inter-sensor correlation via graph attention improve temporal forecasting, and if so, by how much?*

### 1.2 Legacy GAT Research (Unusable — Context Only)

The directory `docs/context/GAT/` contains 7 test directories and 3 analysis directories from early notebook-based experimentation. **This code is entirely legacy and must NOT be reused.** It was developed as ad-hoc Jupyter notebooks with:

- Hardcoded scaler constants, inconsistent data hashes across tests
- Mixed architecture trials: MLP, GAT+MSE, GAT+ZINB/ZIP, LSTM+GAT, GRU+GAT, self-attention+GAT branches
- Varying temporal configurations (window_size=12 vs 128, different feature sets)
- Different graph structures tested (Gabriel graph, fully-connected graph)
- Oversampling experiments (Test-7) for extreme value robustness
- Attention explainability analysis (Analysis-2, Analysis-4, Analysis-5)
- No systematic evaluation harness — each notebook was a standalone experiment

**What to extract from the legacy work (concepts only, not code):**
- The architectural progression: MLP baseline → GAT → temporal+GAT branch models
- The importance of graph construction (Gabriel vs fully-connected vs learned)
- The role of edge attributes in attention
- ZINB vs MSE loss comparison
- Attention explainability analysis as a diagnostic tool

### 1.3 The Chapter 5 Harness (Reference Architecture — `src/quality/`)

The `src/quality/` package is the reference implementation for how experiment code should be structured. It was built for the MDPI Sensors paper (Exp 1 / RO1). Key architectural patterns to replicate:

#### Contract (`src/quality/contract.py`)
- `QualityCapabilities` — capability flags per forecaster (trainable, uses_history, multivariate)
- `SensorForecaster` — abstract base with `fit()` / `predict()` / `get_params()`
- `PerSensorSplit` — windowed data splits with offensive asserts
- `SensorContext` — sensor metadata + scaler + inverse transform
- `FitReport` — structured training outcome

#### Registry (`src/quality/registry.py`)
- Factory function `build_forecaster()` mapping config `type` strings to concrete classes

#### Evaluation (`src/quality/evaluate.py`)
- Standardised evaluation over splits, producing per-window prediction DataFrames

#### Loader (`src/quality/loader.py`)
- `SensorSeriesStore` — loads and serves individual sensor series

#### Config-Driven Experiment Runner (`scripts/run_quality_experiments.py`)
- YAML-driven experiment matrix: models × horizons × interpolation variants × sensors
- Resumable via `registry.json` keyed by config SHA
- Per-job output: `predictions.parquet`, `results.csv`, `manifest.json`
- Aggregator script: `scripts/aggregate_quality_results.py`

#### Statistics (`src/quality/stats.py`)
- Diebold–Mariano tests, Spearman correlations with BH-FDR, bootstrap CIs, segmented regression

### 1.4 The Existing Multi-Node Harness (`src/harness/`)

The `src/harness/` package serves the full RIoT pipeline (Exp 3). It defines:
- `ModelAdapter` / `RIoTAdapter` / `RecurrentBaselineAdapter` — adapters wrapping concrete models
- `Capabilities` — flags: uncertainty, zero_inflation, imputation, kriging, graph
- `Predictions` NamedTuple — mu, theta, pi, imputed, kriging_pred, edge_index, attention
- `Losses` NamedTuple — total, impute, forecast, kriging
- `MetricAccumulator` / `evaluate_batch` — batch-level metric aggregation
- `holdout.py` — inductive spatial holdout mask machinery

**The Exp 2 harness should NOT extend `src/harness/` directly.** It should be a parallel, self-contained package (like `src/quality/` is) that borrows the *philosophy* but defines its own contract appropriate for the simpler RO2 scope.

### 1.5 Current Branch State

You are currently on `feature/ch6-spatial-dependence`. The new experiment branch should be created from `main` (or rebased onto main).

---

## 2. Experiment 2 Scope (What Must Be Tested)

### 2.1 Research Question

> Does explicitly modelling inter-sensor spatial dependencies via Graph Attention Networks improve short-term urban traffic forecasting over purely temporal baselines?

### 2.2 Experimental Design

The experiment must provide a **controlled, progressive comparison** between:

1. **Temporal-only baselines** (no spatial information):
   - Persistence (last observed value)
   - Seasonal Naive (daily, weekly)
   - Per-node LSTM (univariate and multivariate) — reusing or adapting the quality track's LSTM
   - Per-node Transformer (univariate and multivariate)

2. **Spatial models** (graph-structured):
   - GAT-only (static graph, GATv2 message passing, MLP decoder)
   - GAT + Temporal encoder (GATv2 spatial + Transformer/LSTM temporal)
   - Learned-topology GAT (dynamic graph learner + GATv2)
   - Edge-enriched GAT (physical OSM edge features, `edge_dim=10`)

3. **Ablation studies** (isolating spatial contribution):
   - With vs without graph learner
   - Gabriel graph vs fully-connected vs learned topology
   - Edge features: learned-only (`edge_dim=1`) vs physical+learned (`edge_dim=10`)
   - Number of GAT attention heads
   - Number of GAT layers

### 2.3 Key Differences from the Quality Track (Ch 5)

| Dimension | Quality Track (Ch 5 / Exp 1) | GAT Track (Ch 6 / Exp 2) |
|---|---|---|
| Research unit | Single sensor series | Multi-node graph batch `[B, N, T, D]` |
| Spatial information | None | GATv2 message passing, edge features |
| Models | Per-sensor forecasters (fit once per sensor) | Graph-level models (trained on the full network) |
| Data format | Per-sensor windowed splits | Graph-structured temporal batches |
| Loss | MSE (scaled space) | MSE and optionally ZINB NLL (raw counts) |
| Evaluation | Per-sensor metrics → aggregated | Per-node metrics within graph batches → aggregated |
| Graph construction | N/A | Gabriel, FC, learned topology |
| No dashboard | Results not presented on frontend | Results not presented on frontend |

### 2.4 Data

- Source: same `data/processed/master_dataset_15T.parquet` and/or TimescaleDB connector
- Vehicle class: `person` (pedestrian — matches the MDPI paper)
- Temporal: `window_size`, `horizon`, `freq` should be config-driven (reference values from legacy: `window=12` at 15min = 3h lookback, `horizon=3` = 45min ahead)
- Static features: 15-dimensional node features from `src/features.py`
- Edge features: from `data/external/edge_features_cache.json` (9 physical + 1 learned)
- Graph: construct via `src/graph_learner.py` or pre-computed Gabriel/FC adjacency

### 2.5 Outputs (No Dashboard)

Results do **not** need to be visualised on the Next.js frontend dashboard. All outputs should be:
- Per-job: `predictions.parquet`, `results.csv`, `manifest.json`
- Aggregated: CSV/parquet tables and matplotlib/seaborn figures for the thesis chapter
- Statistical tests: Diebold–Mariano (paired, per-node), model comparison tables

---

## 3. Implementation Plan Requirements

Generate a detailed implementation plan covering:

### 3.1 Package Structure
Design a new `src/spatial/` (or `src/gat_track/`) package mirroring the `src/quality/` architecture:
- `contract.py` — Capabilities, data contracts (graph batch, splits), model ABC
- `registry.py` — factory for spatial forecasters
- `evaluate.py` — per-node, per-batch metric evaluation
- `loader.py` — graph-structured data loading from parquet/DB
- `dataset.py` — PyTorch Dataset yielding graph batches
- `models/` — concrete model implementations (GAT variants, temporal baselines adapted for graph context)
- `stats.py` — statistical comparison framework (DM tests, ablation analysis)
- `plotting.py` — thesis-quality figures

### 3.2 Config Structure
Design a YAML config under `configs/spatial/` (or `configs/gat/`) following the quality track pattern:
- Data source, temporal parameters, graph construction method
- Model specifications (all hyperparameters)
- Ablation matrix definition
- Run parameters (output_dir, device, seed)
- W&B integration (offline by default)

### 3.3 Experiment Runner
A script under `scripts/` mirroring `scripts/run_quality_experiments.py`:
- Config-SHA-based resumable registry
- Job matrix: models × graph_variants × ablation configs
- Per-job persistence (predictions, metrics, manifest)
- Aggregation script for cross-model tables

### 3.4 Model Implementations
For each model in the comparison:
- Architecture specification (layers, dims, activations)
- Training loop (optimizer, scheduler, early stopping, loss)
- Inference/prediction interface
- Capability declarations

### 3.5 Evaluation Framework
- Per-node RMSE, MAE, MAPE at each horizon
- Aggregate statistics (median, IQR across nodes)
- Diebold–Mariano tests (paired by node × timestep)
- Ablation tables isolating spatial contribution
- Attention weight analysis (which edges matter?)

### 3.6 Verification Plan
- Smoke test config (`configs/spatial/spatial_smoke.yaml`)
- Desk checks for loss functions and metric calculations
- Unit tests under `tests/test_spatial_*.py`
- Execution sequence (local code changes vs server-only execution)

---

## 4. Constraints & Non-Negotiables

1. **Offensive Programming** — assert everything, fail fast, no silent error handling, no `.dropna()` without explicit justification. Follow `NSTAR_BACKEND_ARCHITECTURE.md`.
2. **Zero Hardcoded Hyperparameters** — everything lives in YAML configs. Follow `NSTAR_MLOPS.md`.
3. **Container Execution** — all training/testing runs inside Docker containers. No `pip install` on host. Follow the Strict Container Isolation paradigm.
4. **Config SHA Resumability** — the experiment runner must be idempotent and resumable.
5. **Decoupled from `src/harness/`** — the new package must be self-contained. It may import shared utilities (e.g., `src/graph_learner.py`, `src/features.py`, `src/dataset.py`) but must define its own contract.
6. **Decoupled from Frontend** — no `ui_payload.json`, no dashboard integration. Results are thesis tables and figures only.
7. **EXPERIMENTS.md** — every completed task must be logged in the experiment ledger.
8. **Git Protocol** — work on a dedicated `exp/` or `feature/` branch. Auto-commit before multi-file changes.
9. **Reproducibility** — random seeds, deterministic data splits, recorded scaler constants.
10. **No data directory reads** — never read raw parquet into the agent context. Use `df.head()` or schema inspection only.

---

## 5. Reference Files

Read these files for architectural context before planning:

| File | Purpose |
|---|---|
| `src/quality/__init__.py` | Quality track package overview |
| `src/quality/contract.py` | Per-sensor contract (reference for spatial contract design) |
| `src/quality/registry.py` | Forecaster factory pattern |
| `src/quality/evaluate.py` | Evaluation framework |
| `src/quality/loader.py` | Data loading pattern |
| `src/quality/forecasters.py` | LSTM/Transformer forecaster implementations |
| `src/quality/stats.py` | Statistical testing framework |
| `scripts/run_quality_experiments.py` | Experiment runner (lines 1-80 for architecture) |
| `configs/quality/quality_full_v2.yaml` | Full experiment config (reference) |
| `src/harness/contract.py` | Multi-node harness contract (reference for graph-level capabilities) |
| `src/harness/adapters.py` | Model adapter pattern |
| `src/graph_learner.py` | Dynamic graph topology inference |
| `src/encoder.py` | GATv2 + Transformer encoder |
| `src/pipeline.py` | Full RIoT pipeline (reference only — do not extend) |
| `src/dataset.py` | Graph-structured dataset |
| `src/features.py` | Static feature computation (15-dim) |
| `NSTAR_BACKEND_ARCHITECTURE.md` | Backend architecture constraints |
| `NSTAR_MLOPS.md` | MLOps and experimentation standards |
| `docs/context/chapter4_methodology.tex` | Methodology chapter (lines 231-275 for experiment progression) |
| `docs/context/GAT/Test-1/README.md` through `Test-7/README.md` | Legacy experiment summaries (context only) |

---

## 6. Deliverable

A complete, file-by-file implementation plan (in the style of `docs/mdpi_sensors/implementation_plan.md`) covering:

1. **New files** to create (with file paths, class/function signatures, and docstrings)
2. **Existing files** to modify (with specific changes)
3. **Config files** to create
4. **Test files** to create
5. **Desk check scripts** to create
6. **Execution sequence** (local code changes → server execution)
7. **Verification plan** (automated tests + manual checks)

The plan should be structured as numbered phases that can be executed sequentially, with clear dependencies between phases.
