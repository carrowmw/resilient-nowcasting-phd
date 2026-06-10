# Architect Plan: Spatial Kriging & Edge Enrichment

## 1. Objective and Scope
Implement the Spatial Estimation Architecture proposed in `docs/spatial_estimation_architecture.md` using the Decoupled Branching Strategy. The goal is to evaluate spatial kriging, a scale-invariant loss function (MSLE), and edge feature enrichment, ensuring strict separation of infrastructure (`feat/` branches) and hypothesis testing (`exp/` branches).

## 2. Global Constraints & North Star Adherence
- **Strict Container Isolation:** The Implementer must run all validation and execution strictly inside the Docker container (`docker compose exec cpu-runner <command>`). Do NOT run local environments. 
- **Offensive Programming:** Fail fast. Add explicit shape assertions (e.g., `assert tensor.shape == [B, N, T, D]`) and NaN tripwires.
- **The Experiment Ledger:** Upon completing the testing for EACH branch, the Implementer MUST append a row to `EXPERIMENTS.md` with the date, branch name, prompt goal, and testing outcome.
- **Desk Check Mandate:** Before writing loss logic (especially MSLE), trace extreme/edge-case values to guarantee theoretical correctness and scale-invariance.

## 3. Execution Plan (Branch-by-Branch)

The Implementer will execute the following phases in order. 

### Phase 1: `feat/osm-edge-enrichment`
1. **Branch Checkout:** Create and checkout `feat/osm-edge-enrichment` from `main`.
2. **Implementation:** 
   - Add `compute_pairwise_edge_features()` to `src/features.py`.
   - Write `scripts/cache_edge_features.py` to pre-compute Dijkstra shortest paths and OSMnx road type compositions.
3. **Dummy Data Test:** Run the script against a mock sub-graph of 5 sensors. Assert that physical distance and road-type fractions are properly outputted to a dummy JSON file.
4. **Conclusion:** Commit, update `EXPERIMENTS.md`, and prompt the user to merge into `main`.

### Phase 2: `feat/kriging-infrastructure`
1. **Branch Checkout:** Create and checkout `feat/kriging-infrastructure` from `main`.
2. **Implementation:**
   - Add `SpatialKrigingDecoder` in `src/decoders.py`.
   - Add `scale_invariant_mse(pred, target)` (MSLE) to `src/decoders.py` and metric helpers to `src/metrics.py`.
   - Add default inactive config parameters to `configs/train/train.yaml` (`w_kriging: 0.0`, `spatial_holdout_prob: 0.0`).
3. **Dummy Data Test (Desk Check):** Pass synthetic tensor data (zeros, large integers, NaNs) through `scale_invariant_mse`. Print exact outputs to guarantee no division-by-zero or shape mismatch. Run a 1-epoch dry-run on dummy tensors to ensure the pipeline functions seamlessly when kriging is set to `0.0`.
4. **Conclusion:** Commit, update `EXPERIMENTS.md`, and prompt the user to merge into `main`.

### Phase 3: `exp/scale-invariant-loss`
1. **Branch Checkout:** Create and checkout `exp/scale-invariant-loss` from `main`.
2. **Implementation:**
   - Update `compute_loss()` in `src/decoders.py` to replace standard MSE with MSLE on the *unscaled* traffic channel.
3. **Dummy Data Test:** Feed a batch of extreme outliers through the loss function. Prove that a predicted count of 5 vs target of 0 yields a similar relative penalty to 50 vs 0, preventing high-variance node domination.
4. **Conclusion:** Do NOT merge. Keep as an archive, update `EXPERIMENTS.md`.

### Phase 4: `exp/spatial-kriging-head`
1. **Branch Checkout:** Create and checkout `exp/spatial-kriging-head` from `main`.
2. **Implementation:**
   - Modify `src/dataset.py` to inject dynamic `spatial_holdout_mask` (~15% nodes masked to 0).
   - In `src/pipeline.py`, prevent held-out nodes from sending GAT messages.
   - Configure a fixed validation holdout in `src/train.py` and activate $w_k > 0$.
3. **Dummy Data Test:** Pass a batch with a known dummy spatial mask. Assert that the masked nodes' dynamic features are exactly zeroed out, but static features remain intact.
4. **Conclusion:** Do NOT merge. Keep as an archive, update `EXPERIMENTS.md`.

### Phase 5: `exp/enriched-gat-kriging`
1. **Branch Checkout:** Create and checkout `exp/enriched-gat-kriging` from `main`.
2. **Implementation:**
   - In `src/graph_learner.py`, load the cached JSON from Phase 1 and build the `[E, 10]` `edge_attr` tensor.
   - In `src/encoder.py`, update `GATv2Conv` to utilize `edge_dim=10`.
3. **Dummy Data Test:** Inject a synthetic `edge_attr` tensor with extreme values (e.g., directness = 0.0). Run a forward pass to assert that attention coefficients react properly to edge variations and shapes align.
4. **Conclusion:** Do NOT merge. Keep as an archive, update `EXPERIMENTS.md`.

## 4. User's Guide: W&B Experiment Comparison Strategy

To compare these branches empirically, the user should execute the following protocol:

1. **The Baseline (`main`):** 
   - Ensure the latest code is on `main`.
   - Create `configs/train_baseline.yaml`.
   - Run: `docker compose exec gpu-runner python src/train.py --config configs/train_baseline.yaml --tags "baseline"`.

2. **The Loss Experiment (`exp/scale-invariant-loss`):**
   - Checkout branch. Duplicate config to `configs/train_msle.yaml`.
   - Run: `docker compose exec gpu-runner python src/train.py --config configs/train_msle.yaml --tags "msle-only"`.
   - **W&B Contrast:** Plot `val_rmse` and `val_mae`. Observe if MSLE converges faster or achieves better relative error across low-count nodes.

3. **The Spatial Kriging Experiment (`exp/spatial-kriging-head`):**
   - Checkout branch. Create `configs/train_kriging.yaml` (set `w_kriging: 0.5`, `spatial_holdout_prob: 0.15`).
   - Run: `docker compose exec gpu-runner python src/train.py --config configs/train_kriging.yaml --tags "kriging-base"`.
   - **W&B Contrast:** Look at the new custom metric `val_kriging_loss`. Compare the overall validation loss to the baseline.

4. **The Enriched Edge Experiment (`exp/enriched-gat-kriging`):**
   - Checkout branch. Create `configs/train_kriging_edges.yaml`.
   - Run: `docker compose exec gpu-runner python src/train.py --config configs/train_kriging_edges.yaml --tags "kriging-enriched"`.
   - **W&B Contrast:** Group runs by the "Tags" in W&B. Plot `val_kriging_loss` for "kriging-base" vs "kriging-enriched". If the enriched graph provides a strong spatial signal, the kriging loss on held-out validation nodes will dramatically drop compared to the un-enriched model.
