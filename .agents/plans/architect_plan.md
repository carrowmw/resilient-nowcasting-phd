# Architectural Design: RIoT Architecture v4 & Experiment 3 Audit Remediation

## 1. Executive Summary & Objective
This architectural specification details the comprehensive remediation of the Experiment 3 (RIoT) pipeline to transition from v3 to **v4**. It directly addresses the mathematical and procedural vulnerabilities identified in the thesis forensic audit (F-C01 through F-C12), implements strict container-safe chronological data partitioning, resolves the pandas MNAR aggregation defect, integrates sinusoidal positional temporal encoding, enforces topological masking consistency, and constructs a resilient multi-GPU batch execution harness capable of running the complete suite of thesis ablation studies across GPU 0 and GPU 1 on the remote server.

## 2. Alignment with North Star Guidelines
1. **NSTAR_BACKEND_ARCHITECTURE.md (Offensive Programming / Mathematical Rigor):**
   - **Tripwires:** All tensor operations must assert shapes, type stability, and absence of NaN/Inf before feeding decoders.
   - **ZINB Rigor:** Expected value point metrics must compute $\hat{y} = \hat{\mu} \cdot (1 - \hat{\pi})$. Reliability diagrams must calibrate against total zero probability $P(Y=0) = \pi + (1-\pi)(\frac{\theta}{\theta+\mu})^\theta$.
   - **Zero Leakage:** Training, validation, and test datasets must be chronologically segregated with a purge buffer of $\ge T + h = 13$ steps. Scalers must strictly fit on the training slice only.
2. **NSTAR_MLOPS.md (Reproducibility & Experiment Tracking):**
   - **Version 4 Namespacing:** Isolated W&B namespace (`project: resilient-iot-v4`, `collection: resilient-iot-model-v4`).
   - **Ledger Mandate:** Every executed ablation and run outcome must be documented in `EXPERIMENTS.md`.
3. **NSTAR_PHASES.md (Temporal Consistency):**
   - Respects Phase 3 multi-task dynamic reconstruction with 10-dimensional edge attributes and learned spatial topology.

## 3. High-Level Architectural Components

### Component 1: Data Pipeline & MNAR Integrity (`src/dataset.py`)
- **Direction Aggregation Bugfix:** Replace `'value': 'sum'` with `min_count=1` inside `groupby.agg`. If all directions at location $i$ and timestep $t$ are null, the aggregated value remains `NaN`.
- **Mask Channel Fidelity:** `total_mnar_mask` accurately assigns `1.0` to all true non-reports (36.3% of station-slots) and `0.0` only to valid observations.
- **Train-Only Scaler Isolation:** Implement explicit chronological train-cutoff logic. `StandardScaler` instances for traffic, weather, and static features fit strictly on the train period `[start_date, train_end_date]`.

### Component 2: Chronological Purged Split (`src/train.py`)
- **Eliminate `torch.utils.data.random_split`:** Replace with chronological contiguous slicing:
  - Train: First 70% of timeline
  - Val: Next 15% of timeline
  - Test: Final 15% of timeline
- **Purge Boundaries:** Enforce a buffer gap of $T + h = 12 + 1 = 13$ timesteps (3.25 hours) between Train/Val and Val/Test to prevent any sliding window from sharing observations across partitions.

### Component 3: Spatio-Temporal Positional Encoding (`src/encoder.py`)
- **Temporal Sequence Awareness:** Implement `SinusoidalPositionalEncoding` module adding harmonic positional embeddings $\mathbf{PE} \in \mathbb{R}^{T \times H}$ to GATv2 spatial node embeddings before ingestion by `nn.TransformerEncoder`.
- **Mathematical Form:** $PE(t, 2i) = \sin(t / 10000^{2i/H})$, $PE(t, 2i+1) = \cos(t / 10000^{2i/H})$.

### Component 4: Topology Masking Protocol (`src/pipeline.py`)
- **Bidirectional Isolation:** Outgoing edge filtering is extended from `simulated_nodes_mask` to include `spatial_holdout_mask` during training and evaluation.
- **Dynamic Mask Attenuation:** Null-state nodes (`structural_masks[:, :, -1, 0] == 1`) are prevented from transmitting corrupted zero-imputed features into neighbouring node attention.
- **Kriging Mask Blanking:** Mask channels for held-out kriging nodes are neutralized to prevent leaking target dropout status.

### Component 5: Point Prediction & Uncertainty Metrics (`src/harness/evaluate.py`, `src/export_ui_payload.py`)
- Point forecasts evaluate the exact ZINB mean: $\hat{y} = \hat{\mu} \cdot (1 - \hat{\pi})$.
- Conditional active-state mean $\hat{\mu}$ is logged as a secondary diagnostic.
- ZINB calibration diagram calculates true total zero probability.

### Component 6: Multi-GPU Automated Ablation Harness (`scripts/run_v4_server_suite.py`)
- Dual-GPU dispatcher allocating independent ablation runs across GPU 0 and GPU 1 concurrently, or queuing long sequential epochs with GPU-affinity.
- Suite covers:
  1. Full RIoT v4 Model (`train_riot_v4_full_inductive.yaml`)
  2. Ablation A: Without Reconstruction Loss (`w_impute = 0.0`)
  3. Ablation B: Without Kriging Loss (`w_kriging = 0.0`)
  4. Ablation C: Without Missingness Masks (Channel 3 & 4 ablated)
  5. Ablation D: Without Positional Encoding (isolating the exact contribution of PE)
  6. Ablation E: Inductive LSTM Baseline Comparator
- Automated payload export (`export_ui_payload.py`) post-training to generate LaTeX tables for Chapter 7.
