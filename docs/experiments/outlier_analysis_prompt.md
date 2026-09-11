# Outlier Event Analysis: Prompt for Claude Fable

## Context

You are analysing a **ResilientIoT** spatiotemporal graph neural network that predicts 15-minute pedestrian traffic counts across an 81-sensor urban IoT network in Newcastle upon Tyne. The system uses a unified architecture with three jointly trained heads: **imputation** (historical reconstruction), **ZINB forecasting** (uncertainty-aware point prediction), and **spatial kriging** (zero-shot estimation at unseen locations). There are also **50 simulated nodes** (zero-shot, no physical sensor) whose predictions are derived purely from GATv2 message passing.

### Architecture Summary

| Component | Purpose |
|---|---|
| `DynamicGraphLearner` | Learns adjacency from static features; enriches edges with 9 OSM physical features + 1 learned similarity → `edge_attr[E, 10]` |
| `MaskedSpatiotemporalEncoder` | GATv2 spatial attention + TransformerEncoder temporal modelling; receives MNAR structural masks [total, night] concatenated to dynamic features |
| `DecoupledDecoders` | Imputation MLP (full sequence), ZINB forecast head (μ, θ, π from last timestep), tripwire logging |
| `SpatialKrigingDecoder` | Reconstructs held-out nodes' series from latent embeddings shaped by message passing only |
| `SimulatedNodeInjector` | Injects 50 synthetic nodes with real static features (Places + OSMnx) but NaN dynamics; outgoing edges masked via `simulated_nodes_mask` |

### Loss Function

$$\mathcal{L} = w_\text{impute} \cdot \mathcal{L}_\text{impute} + w_\text{forecast} \cdot \mathcal{L}_\text{ZINB} + w_\text{kriging} \cdot \mathcal{L}_\text{kriging}$$

- **Traffic (ch 0):** TRUE MSLE on inverse-transformed raw counts (scale-invariant — low-count suburban nodes weighted equally with high-count city-centre nodes)
- **Weather (ch 1-2):** MSE on scaled features
- **ZINB NLL:** Zero-Inflated Negative Binomial on raw counts
- **Kriging:** Same channel split on spatially held-out nodes (20% holdout prob)

---

## Your Task

Perform a comprehensive analysis of the model's behaviour during **outlier events** — timestamps and spatial locations where observed traffic counts deviate significantly from normal patterns. These include:

1. **Temporal outliers**: Sudden spikes (e.g., football matches, concerts at St James' Park, Great North Run, festivals) or unexpected drops (storms, road closures, COVID-era remnants)
2. **Spatial outliers**: Individual sensors showing anomalous readings while neighbours remain normal (sensor malfunction vs. genuine localised event)
3. **Structural outliers**: Periods of extreme data missingness (MNAR patterns), night-time zero-inflation boundaries

---

## Data Sources to Review

### 1. Weights & Biases (W&B)
- **Project:** `resilient-iot-v3`
- **Sweep project:** `resilient-iot-sweeps-v3`
- **Key metrics to pull:** `val_loss`, `val_rmse`, `val_mae`, `val_zero_acc`, `val_loss_kriging`, `val_kriging_rmse`, `val_kriging_mae`, `val_kriging_msle`, `train_loss_total`, `train_loss_impute`, `train_loss_forecast`, `train_loss_kriging`, `impute_mse`
- **Per-epoch tripwire logs:** `PI_CLAMP`, `MU_CLAMP`, `THETA_CLAMP` events logged per (batch, node)
- **Run metadata:** `git_commit`, config YAML snapshot, `mask_rng: cuda`
- **Analysis runs:** Check for runs in the `resilient-iot-v2` project (legacy) as well as `v3`

### 2. Cloudflare R2 Inference Payloads
- **R2 Public URL:** `https://pub-b102fa91750348f78a6385248654688a.r2.dev`
- **Registry:** `runs/run_registry.json` — lists all deployed model inference runs
- **Per-run structure:**
  - `runs/{run_id}/spatial.json` — node coordinates, edge topology, run metadata, `metrics_summary`
  - `runs/{run_id}/{YYYY-MM-DD}/page.json` — 96-step (24h) timeseries: `ground_truth_series`, `imputed_series`, `mu_series`, `pi_series`, `theta_series`, `mask_series`, per-timestep attention weights
  - `runs/{run_id}/{YYYY-MM-DD}/nodes/node_{id}.json` — per-node deep analytics: ego-graph, `two_extremes` (MAE/RMSE under drop-NaN vs fill-NaN), PIT histograms, zero-inflation reliability curves
- **Simulated nodes** have `node_role: "simulated"` / `is_simulated: true` — they have predictions but no `ground_truth_series`

### 3. Inductive Holdout Nodes
- 20% of real nodes are permanently held out (`inductive_holdout_frac: 0.2`, `eval_mode: inductive`) — these have `node_role: "inductive"` and their targets are excluded from training loss. They are scored via the kriging decoder during validation and via `two_extremes` during export.

---

## Key Tests to Run

### Test 1: Outlier Detection & Cataloguing
**Goal:** Identify the top-N outlier events in the analysis window.

- For each real node, compute the residual series: `|ground_truth[t] - mu[t]|` across all timesteps
- Flag timesteps where the residual exceeds 3σ of that node's residual distribution
- Cluster flagged timesteps temporally (events spanning multiple consecutive 15-min windows) and spatially (multiple nearby nodes spiking simultaneously)
- Classify events: **network-wide** (≥30% of nodes affected), **cluster** (spatially contiguous subset), **isolated** (single node)
- Cross-reference with known Newcastle events (match dates against football fixtures, Great North Run, bank holidays, school terms)

### Test 2: ZINB Distribution Behaviour During Outliers
**Goal:** Assess whether the uncertainty head correctly widens during anomalous events.

- For each catalogued outlier event, extract the ZINB parameters: μ(t), θ(t), π(t)
- Compute the ZINB variance: `Var = μ + μ²/θ` — does variance increase during outlier timesteps?
- Compute the predicted zero-inflation π — does it correctly drop during spike events (the model should be confident these are *not* structural zeros)?
- Plot the empirical coverage: what fraction of outlier ground-truth values fall within the ZINB 50th/90th/99th percentile intervals?
- Compare coverage to non-outlier timesteps — is the model systematically overconfident during outliers?

### Test 3: Outlier Signal Propagation to Simulated Nodes
**Goal:** Determine whether outlier information at real sensors reaches simulated (zero-shot) nodes through GATv2 message passing.

> [!IMPORTANT]
> This is the critical question. Simulated nodes have `dynamic_features = NaN → 0.0` (no sensor readings), `structural_masks = 1.0` everywhere (fully missing), and **outgoing edges are masked** (`simulated_nodes_mask`). They can only *receive* messages from real nodes, never send.

- **Attention weight analysis:** For each simulated node, extract the GATv2 attention weights `α[sim → real_neighbour]` from the `page.json` attention data. During an outlier event at a nearby real node, do the attention weights shift? Does the simulated node "look" more at the outlier node?
- **Prediction response:** Extract `mu_series` for simulated nodes during and around a real-node outlier event. Does the simulated node's predicted traffic change? This would demonstrate information propagation through the learned graph topology.
- **Latency analysis:** If the outlier does propagate, is there a temporal lag? (e.g., outlier at node A at t=0 → simulated node B's prediction shifts at t=1, t=2?)
- **Spatial decay:** Plot the prediction change at simulated nodes as a function of graph distance (shortest path in the learned adjacency) and physical distance (euclidean/OSM) from the outlier source

### Test 4: Outlier Build-Up Dynamics
**Goal:** Trace how outlier events develop over time within the model's internal representations.

- **Pre-event ramp-up:** For each major outlier, look at the 2-6 hours leading up to the event. Do the following signals show a gradual build-up?
  - μ (predicted mean) — does it start rising before the ground truth spike?
  - θ (dispersion) — does uncertainty increase as a "warning sign"?
  - π (zero-inflation) — does it drop from night-time levels?
  - Imputation residual at the target node
  - Attention weight concentration (entropy of attention distribution)
- **Cascade analysis:** For network-wide events, trace the temporal ordering: which nodes "detect" the event first (via rising residuals), and does this propagate through the learned graph to neighbouring nodes?
- **Recovery dynamics:** After an outlier event subsides, how quickly do μ, θ, π return to baseline? Is there a "hysteresis" effect where the model remains uncertain for longer than the event duration?

### Test 5: Kriging Performance Stratified by Outlier Exposure
**Goal:** Assess whether the spatial kriging head generalises to outlier conditions.

- For inductive holdout nodes, compare kriging metrics (`kriging_rmse`, `kriging_mae`, `kriging_msle`) during:
  - Normal periods (no outliers at any neighbouring observed node)
  - Periods where at least one observed neighbour is experiencing an outlier
- Does the kriging error increase proportionally, or does the message-passing mechanism amplify errors?
- Compare with the imputation head's performance under the same conditions

### Test 6: Tripwire Event Correlation with Outliers
**Goal:** Determine whether ZINB distribution tripwires (PI_CLAMP, MU_CLAMP, THETA_CLAMP) correlate with real-world outlier events.

- Pull the tripwire logs from W&B runs
- For each PI_CLAMP event (π approaches 0 or 1), cross-reference with the ground truth at that node/timestep
- Hypothesis: MU_CLAMP events (μ → 0) should correlate with genuine structural zeros (night-time); PI_CLAMP events near 0 should correlate with unexpected high-traffic outliers where the model correctly assigns near-zero zero-inflation probability
- If tripwires cluster at specific nodes regardless of time → likely model capacity issue; if they cluster at specific times across many nodes → real event detection

### Test 7: Scale-Invariance Stress Test
**Goal:** Verify that the TRUE MSLE loss function genuinely equalises outlier treatment across low-count and high-count sensors.

- Select two sensors: one high-traffic city-centre sensor (mean count > 50/interval) and one low-traffic suburban sensor (mean count < 5/interval)
- For both, identify their respective outlier events
- Compare: does a 10× spike at the suburban sensor (5 → 50) produce a comparable loss contribution to a 2× spike at the city-centre sensor (50 → 100)? Under TRUE MSLE, `log1p(50)/log1p(5)` vs `log1p(100)/log1p(50)` should be weighted more fairly than raw MSE
- Does this equalisation hold in the kriging decoder's reconstruction of these outlier events?

### Test 8: Two-Extremes Divergence as Outlier Indicator
**Goal:** Use the existing `two_extremes` metric (drop-NaN vs fill-NaN error) as a proxy for outlier severity.

- For each node and day, compute: `gap = rmse_fill_nan - rmse_drop_nan`
- A large positive gap indicates the model predicts *something* during missing periods but the "fill with zero" assumption strongly penalises it — potential structural outlier
- Correlate `gap` with the number of flagged outlier timesteps on that day
- Is the gap larger for simulated nodes (which have no ground truth, so fill=drop is meaningless) vs. inductive holdout nodes?

---

## Output Format

Please produce:

1. **Executive Summary** (1 page): Key findings on outlier handling, simulated node information propagation, and event build-up dynamics
2. **Outlier Catalogue**: Table of top-20 events with timestamp range, affected nodes, classification, and suspected real-world cause
3. **Per-Test Results**: For each of the 8 tests above, provide:
   - Quantitative results with confidence intervals where appropriate
   - Visualisations (time-series plots, heatmaps, attention weight distributions)
   - Conclusions and implications for model architecture
4. **Recommendations**: Specific architectural or training modifications to improve outlier handling (e.g., outlier-aware augmentation, attention temperature scaling, explicit event detection head)

---

## Technical Notes

- The model uses `StandardScaler` normalisation on inputs; raw counts are recovered via inverse transform using `traffic_mean` and `traffic_std` buffers stored in the model checkpoint
- The ZINB parameters are constrained: `μ = softplus(logit) + ε`, `θ = softplus(logit) + ε`, `π = clamp(sigmoid(logit), ε, 1-ε)` where `ε = 1e-6`
- Simulated nodes have their outgoing edges dropped in `pipeline.py` (line 163-170) but can still *receive* messages from real nodes — this is the only pathway for information to reach them
- The training data spans `2023-04-02` to `2026-04-02` (3 years, `target_veh_class: person`, `target_dir: all`); inference/analysis window is `2026-05-01` to `2026-06-01`
- Night-time structural zeros are encoded via explicit MNAR masks (`structural_masks[:, :, :, 1]` = night missingness), not silent imputation
