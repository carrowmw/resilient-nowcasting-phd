# Generic Evaluation Harness & Cross-Model Standardization

This plan builds a single evaluation harness shared by **three consumers** — `train.py` (train/val loops), and the two payload exporters (`export_ui_payload.py`, `export_paginated_payload.py`) — so that every model produces a standardized, reproducible set of results, and so that spatially held-out nodes flow through to the frontend as a first-class node category.

## User Review Required
> [!IMPORTANT]
> **Dummy values must never become numbers.** The original idea (baselines emit `dispersion=1`, `P(zero)=0`) crashes inference today — `export_paginated_payload.py:478` asserts `pi > 0` strictly — and worse, dummy-derived metrics and uncertainty bands would render on the frontend as if real. This plan replaces "dummy values" with two honest alternatives per capability: a **genuine naive attempt** (real, deliberately bad numbers) or a **null** (metric reported as `N/A`). Dummy tensors exist only internally to satisfy shape contracts and are never scored or exported.

> [!IMPORTANT]
> **Three node roles reach the frontend.** `observed` (normal sensors), `holdout` (real sensors masked throughout training — genuine zero-shot predictions *with* ground truth available for display), and `simulated` (zero-baseline synthetic nodes — predictions with *no* ground truth). This requires a payload schema change and coordinated frontend work.

---

## 0. The Contracts

### 0.1 Contract In — the batch schema
The harness guarantees the batch dict contains exactly these **data** keys (matching `custom_collate_real`, `dataset.py:277-287`):
`static_features [N, S]`, `dynamic_features [B, N, T, D]`, `target_impute [B, N, T, D]`, `target_forecast [B, N]`, `node_mask [B, N, T, D]`, `location_ids`, and optionally `spatial_holdout_mask [N]` and `simulated_nodes_mask [N]`.

Loss weights (`w_impute`, `w_forecast`, `w_kriging`) are **removed from the batch** at the harness boundary — they are hyperparameters, not data. The RIoT wrapper (§0.3) re-injects them into the batch it hands to the unmodified `ResilientIoTPipeline.forward()`, absorbing the impedance mismatch so `pipeline.py` needs no signature change.

### 0.2 Contract Out — predictions, not internals
#### [NEW] `src/harness/contract.py`
```python
class Predictions(NamedTuple):
    mu: torch.Tensor                        # [B, N] point forecast (all models)
    theta: Optional[torch.Tensor]           # [B, N] dispersion — None if no uncertainty
    pi: Optional[torch.Tensor]              # [B, N] zero-inflation prob — None if unsupported
    imputed: Optional[torch.Tensor]         # [B, N, T, D] reconstruction — None if unsupported
    kriging_pred: Optional[torch.Tensor]    # [B, K, T, D] holdout reconstruction — None if unsupported
    edge_index: Optional[torch.Tensor]      # graph models only
    attention: Optional[torch.Tensor]       # graph models only
    tripwires: list                         # always present, may be empty

class Losses(NamedTuple):
    total: torch.Tensor                     # backprop target — always a device tensor
    impute: torch.Tensor                    # zero TENSOR (never a float) when unsupported
    forecast: torch.Tensor
    kriging: torch.Tensor
```
Absence is expressed as `None` in `Predictions`, and as a **device-placed zero tensor** in `Losses` (the train loop calls `.item()` on every component — `train.py:437-439`).

### 0.3 The ModelAdapter protocol
#### [NEW] `src/harness/adapters.py`
Every model participates through a thin wrapper — this is where model-specific wrangling lives, keeping the harness and exporters model-agnostic:
```python
class ModelAdapter(ABC):
    capabilities: Capabilities   # frozen dataclass of bools:
                                 # uncertainty, zero_inflation, imputation, kriging, graph
    def step(self, batch, weights, loss_node_mask=None) -> (Losses, Predictions): ...
    def predict(self, batch: dict) -> Predictions: ...   # inference-only (exporters)
```
(`step()` produces losses and predictions in a single forward — separate
`predict()`/`loss()` calls would double the compute for the RIoT pipeline.)
- **`RIoTAdapter`**: wraps `ResilientIoTPipeline` unchanged. `predict()` calls `forward(batch, return_analysis=True)` semantics; `loss()` calls the existing forward and repacks the 10-tuple. All capabilities `True`.
- **`RecurrentBaselineAdapter`** (LSTM/GRU): see §1.
- Loss functions legitimately differ per model (ZINB NLL vs MSE). Therefore `val_loss` is **per-model only** (used for early stopping / checkpointing within a run) and is explicitly excluded from cross-model comparison — cross-model comparison happens exclusively through the metrics manifest (§4).

### 0.4 Capability-gated metrics
#### [NEW] `src/harness/evaluate.py`
One function, used verbatim by the val loop and the exporters:
```python
def evaluate(adapter, batch, holdout_mask) -> dict
```
- Computes each metric **only if the adapter's capability supports it**; otherwise emits `None` (serialized as `null` in the manifest, rendered as "N/A" downstream, logged to W&B as absent — never as `0.0`).
  - `rmse`/`mae`: always (from `mu`).
  - `zero_acc`: gated on `zero_inflation` — a baseline with dummy `pi` must show `null`, not a fabricated 0%.
  - `impute_mse`: gated on `imputation`.
  - `kriging_msle/rmse/mae`: gated on `kriging` (a *naive* kriging head — §1 — counts as capable: real numbers, honestly bad).
- The existing ZINB tripwire assertions in the exporters (`mu > 0`, `0 < pi < 1`, attention NaN checks) move here and fire **only when the corresponding capability is claimed**. This is the change that stops dummy values from crashing inference.

---

## 1. Simple Baselines (LSTM / GRU)

#### [NEW] `src/baselines.py`
`LSTMForecaster` / `GRUForecaster`: per-node recurrent encoders over `dynamic_features`, ignoring all spatial structure.
- **Forecast head** → `mu [B, N]`. Trained with MSE against `target_forecast`.
- **Reconstruction head** (small linear head over per-timestep hidden states) → `imputed [B, N, T, D]`, trained with masked MSE. This makes the imputation and kriging attempts **genuine**: `kriging_pred` is `imputed` sliced at the holdout mask. Because the harness blanked those nodes' inputs, the reconstruction is honestly terrible — which is the ablation story we want, backed by real numbers instead of fabricated ones.
- `theta = None`, `pi = None`, `edge_index = None`, `attention = None`. **No dummy `theta=1`/`pi=0`** — those produced fabricated ZINB variance (`mu + mu²` uncertainty bands on the frontend) and tripwire crashes.
- Capabilities: `uncertainty=False, zero_inflation=False, imputation=True (naive), kriging=True (naive), graph=False`.
- `Losses`: `forecast` = MSE, `impute` = masked reconstruction MSE, `kriging` = zero tensor (the reconstruction loss already covers it), `total` = weighted sum — all device tensors.

#### [MODIFY] `src/train.py`
- `model_type: gnn | lstm | gru` selects the adapter; the train/val loops talk **only** to the adapter + `evaluate()`. The val-loop metric block (`train.py:446-501`) is replaced by the shared `evaluate()` call.

#### [MODIFY] `configs/train/train_lstm.yaml`
- `spatial_holdout_prob: 0.0` and `w_kriging: 0.0` (currently 0.2 / 0.5 — this would blank 20% of the LSTM's *inputs* per batch and corrupt the baseline). Transductive kriging pressure is a GNN-training concern; inductive evaluation is governed by `eval_mode` (§2), not this knob.

---

## 2. Inductive Spatial Holdout (Task 3)

IGNNK-style fixed holdout; the disjoint `allowed_locations` approach stays abandoned (it crashes `encoder.py:87` and breaks scaler propagation).

#### [MODIFY] `src/train.py` + [NEW] `src/harness/holdout.py`
- `eval_mode: temporal | inductive` config flag (explicit, not "a mechanism").
- **`temporal`** (default): current behavior, unchanged.
- **`inductive`**:
  - All nodes stay in every batch (`num_nodes`, scaler stats, and edge caches all remain valid).
  - A **fixed, seeded** holdout mask (~20% of nodes) is generated once per run and applied via `apply_spatial_holdout` to **train and val** inputs.
  - **No target leakage — the zero-shot guarantee.** In inductive mode, *every* training loss term (forecast, impute, kriging) is masked to exclude holdout nodes; their ground truth is used at **evaluation time only**. Without this, `compute_loss` fits the model against holdout targets each step and Task 3's claim is void. Implemented as a `loss_node_mask` the adapters must respect (`compute_loss` gains an optional node-mask argument; the baseline losses apply it directly).
  - The GNN's kriging **training** signal comes from per-batch re-rolled holdouts drawn *only from the visible ~80%* (the existing `train.py:412-422` re-roll, restricted). The fixed 20% is scored but never trained on. In `temporal` mode the re-roll behavior is unchanged.
  - **Holdout manifest**: the resolved held-out `location_ids` (not just the seed — node ordering could drift) are written to `<checkpoint_dir>/holdout_manifest.json` alongside the existing co-located `config.yaml`. This file is the bridge to inference (§3) and to cross-model comparability: comparing two models requires *identical* manifests, which the harness asserts when loading.

---

## 3. Frontend Pathway — Holdout Nodes in the Payload

#### [MODIFY] `src/export_ui_payload.py`, `src/export_paginated_payload.py`
- Load `holdout_manifest.json` from the checkpoint directory of the model being served. At inference, holdout nodes' dynamic features are blanked exactly as in training, so their displayed predictions are genuine zero-shot reconstructions.
- Replace `is_simulated: bool` in node metadata with `node_role: "observed" | "holdout" | "simulated"` (keep `is_simulated` temporarily for backward compatibility, derived from `node_role`).
  - `observed`: prediction + ground truth, normal display.
  - `holdout`: prediction + ground truth — the frontend can show predicted-vs-actual on nodes the model provably never trained on.
  - `simulated`: prediction, `ground_truth_series: null` (unchanged).
- Holdout nodes keep their graph edges (they are real sensors with real static features) — only `simulated` nodes get the outgoing-edge drop via `simulated_nodes_mask`. This mirrors training exactly.
- Payload fields backed by capabilities become **nullable**: `uncertainty`, `theta_data`, `pi_data`, and the whole graph/attention section are `null` for baseline-generated payloads. The per-timestep tripwire assertions are replaced by the capability-gated checks in `evaluate()`.
- **[COORDINATE] `frontend/`**: render `node_role: holdout` distinctly; handle `null` uncertainty/graph sections. This is a cross-repo change and should be sequenced after the payload schema is frozen.

---

## 4. Reproducibility — the Results Manifest

#### [NEW] emitted by the harness at the end of every val epoch and every export run
`<checkpoint_dir>/results.json`:
```json
{
  "model_type": "lstm",
  "eval_mode": "inductive",
  "seed": 42,
  "holdout_manifest_sha": "…",
  "config_sha": "…",
  "metrics": { "rmse": 12.3, "mae": 8.1, "zero_acc": null, "impute_mse": 0.041,
               "kriging_msle": 2.9, "kriging_rmse": 44.0, "kriging_mae": 31.2 }
}
```
- `null` means "model cannot produce this," never 0. Cross-model tables are a join over these files keyed on `holdout_manifest_sha` — identical hash ⇒ identical evaluation conditions ⇒ comparable rows. W&B logging mirrors this payload.

---

## Verification Plan

### Automated Tests
- `RIoTAdapter` and `RecurrentBaselineAdapter` both return `Predictions`/`Losses` with harness-consumable shapes and device tensors (no floats) — the contract test, shared via a fixture, not duplicated per model.
- `evaluate()` returns `null` (not 0.0) for capability-gated metrics on a baseline adapter, and full metrics for the RIoT adapter.
- Inductive loss masking: with `eval_mode: inductive`, assert zero gradient flows from holdout-node targets (finite-difference or `grad` inspection on a tiny mock batch) — this is the test that protects the zero-shot claim.
- Holdout manifest round-trip: train writes it, exporter loads it, node roles in the payload match.
- Capability-gated tripwires: baseline predictions with `pi=None` pass through `evaluate()` without assertion errors.

### Manual Verification
1. **Baseline transductive**: run `train_lstm.yaml` end-to-end; `results.json` shows real `rmse`/`mae`/`impute_mse`, `zero_acc: null`; no `.item()` or tripwire crashes.
2. **Inductive comparability**: run GNN and LSTM with `eval_mode: inductive`, same seed; assert identical `holdout_manifest_sha`; confirm LSTM `kriging_rmse` ≫ GNN `kriging_rmse` (the honest ablation number).
3. **GNN inductive**: trains without triggering the `encoder.py:87` num_nodes assertion; W&B shows kriging metrics on the fixed holdout only.
4. **Payload**: export from a GNN checkpoint with a holdout manifest; verify nodes carry all three `node_role` values, holdout nodes have both prediction and ground truth, and a baseline-generated payload has `null` uncertainty/graph sections without crashing the exporter.
