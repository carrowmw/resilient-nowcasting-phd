# Implementation Plan — Phase 2 (Layer 2: Kriging Integration)

**Implementer:** Claude (Senior ML Engineer)
**Date:** 2026-06-30
**Source spec:** `.agents/plans/architect_plan.md` §3.2 (Layer 2) + §3.4 (loss wiring)
**Target branch:** `main`
**Predecessor:** Phase 1 (Scale-Invariant Loss) — COMPLETE & merged (commit `b7c6399`).

---

## 0. Context & Scoping (what I found in the codebase)

Layer 2 is **mostly a wiring job**, not a from-scratch build. The dormant
infrastructure from `feat/kriging-infrastructure` already lives on `main`:

| Already present (dormant) | Location |
|---|---|
| `SpatialKrigingDecoder` (`[B,N,T,H] + mask[N] → [B,K,T,D]`) | `src/decoders.py:87` |
| `scale_invariant_mse` (MSLE) | `src/decoders.py:132` |
| `compute_msle`, `calculate_kriging_metrics` | `src/metrics.py:63,77` |
| Config keys `data.spatial_holdout_prob`, `loss.w_kriging` (both `0.0`) | `configs/train/train.yaml`, `train_test.yaml` |
| Shape/edge-case unit tests | `tests/test_kriging_infra.py` |
| MSLE/decoder desk check | `scripts/desk_check_kriging.py` |

**What is missing (the actual Phase 2 work) — activate the dormant path:**
1. Holdout-mask generation + application helpers (`dataset.py`).
2. `compute_kriging_loss` + 4-output `compute_loss` (`decoders.py`).
3. Kriging decoder instantiation + mask threading + loss wiring (`pipeline.py`).
4. Mask creation, holdout application, loss/metric logging in the loop (`train.py`).
5. Test + desk-check updates for the now-live path.

**Pre-existing regression I will also fix:** `tests/test_decoders.py::test_decoupled_decoders_and_loss`
currently **FAILS** — Phase 1 added required `traffic_mean`/`traffic_std` args to
`compute_loss` but never updated this test. Confirmed by running it:
`TypeError: compute_loss() missing 2 required positional arguments: 'traffic_mean'
and 'traffic_std'`. (Current suite: **1 failed, 14 passed**.) Phase 2 touches this
exact signature, so I will repair the test in the same change.

**Design guardrails (keep Layer 2 isolated & reversible):**
- **Default-off:** `w_kriging=0.0` and no `spatial_holdout_mask` in batch ⇒ numerically
  identical to today. All mock-data tests (which pass no mask) keep working unchanged.
- The kriging head only runs when a holdout mask is present in the batch.
- **Empty-holdout safety:** a random mask can select 0 nodes on small `N`; the loss path
  must return `0.0` (not NaN from `mse_loss` on an empty tensor).

---

## 1. File-by-File Changes

### 1.1 `src/dataset.py` — holdout mask helpers (NEW, module-level)

Add two pure functions (architect §3.2.1), placed near `custom_collate_real`:

```python
def make_spatial_holdout_mask(num_nodes, holdout_prob, device='cpu', seed=None) -> torch.Tensor
def apply_spatial_holdout(dynamic_features, holdout_mask) -> torch.Tensor
```

- `make_spatial_holdout_mask`: `np.random.RandomState(seed)` (deterministic when seed
  set, e.g. validation); returns `torch.bool` `[N]` on `device`.
  Assertions: `0.0 <= holdout_prob <= 1.0`, `num_nodes > 0`.
- `apply_spatial_holdout`: clone `[B,N,T,D]`, zero `masked[:, mask, :, :] = 0.0`.
  Assertions: 4-D input, 1-D bool mask, `mask.shape[0] == features.shape[1]`.

Module-level functions (not methods) so `train.py`/tests import them directly — matches
the architect's call sites and leaves `RealIoTDataset.__init__` / `__getitem__` /
`custom_collate_real` untouched. (location_ids + edge features are **Layer 3**, out of scope.)

### 1.2 `src/decoders.py` — kriging loss + 4-output `compute_loss`

**(a) NEW `compute_kriging_loss(kriging_pred, target_impute, holdout_mask, traffic_mean, traffic_std)`** (architect §3.2.2):
- `target_holdout = target_impute[:, holdout_mask, :, :]`; assert shape equals `kriging_pred`.
- **Empty-holdout guard:** `if kriging_pred.numel() == 0: return zeros((), device=...)`.
- Traffic ch0: inverse-transform pred & target to raw counts, clamp `min=0`,
  `scale_invariant_mse` → MSLE (identical treatment to the imputation head, so the two
  losses are directly comparable).
- Weather ch1-2: `F.mse_loss` on scaled features.
- Return `loss_traffic + loss_weather`; assert not NaN.

**(b) MODIFY `compute_loss` → return 4 values** (architect §3.4). New params after the
existing scaler args, all defaulted so only call sites we own must change:
```python
def compute_loss(..., traffic_mean, traffic_std,
                 kriging_pred: Optional[Tensor] = None,
                 spatial_holdout_mask: Optional[Tensor] = None,
                 w_impute=1.0, w_forecast=1.0, w_kriging=0.0
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:   # + loss_kriging
```
- `loss_kriging` computed only when `w_kriging > 0.0` (then assert `kriging_pred` and
  `spatial_holdout_mask` are not None → `compute_kriging_loss`), else
  `zeros((), device=imputed_features.device)`.
- `total = w_impute*loss_impute + w_forecast*loss_forecast + w_kriging*loss_kriging`.
- Keep all existing NaN/Inf tripwires; add one for `loss_kriging`.
- **Returns `(total_loss, loss_impute, loss_forecast, loss_kriging)`.**

### 1.3 `src/pipeline.py` — instantiate + thread + wire

- **`__init__`:** instantiate `self.kriging_decoder = SpatialKrigingDecoder(hidden_dim,
  time_series_dim, dropout)`; import `SpatialKrigingDecoder`, `compute_kriging_loss`.
  (Adds params but zero behaviour change when no mask / `w_kriging=0`.)
- **`forward` (normal mode):**
  - `holdout_mask = batch.get('spatial_holdout_mask')`.
  - After encoder+decoders: `kriging_pred = self.kriging_decoder(latent_embeddings,
    holdout_mask) if holdout_mask is not None else None`.
  - `compute_loss(..., kriging_pred=kriging_pred, spatial_holdout_mask=holdout_mask,
    w_kriging=batch.get('w_kriging', 0.0))`, unpack **4** values.
  - **Return 10-tuple** (Q2 = full metrics ⇒ surface `kriging_pred`):
    `(loss, loss_impute, loss_forecast, loss_kriging, imputed_features, mu, theta, pi,
    tripwires, kriging_pred)`. `kriging_pred` is `None` when no mask is present.
- **`forward` (analysis mode):** add `'kriging_pred'` to the dict (computed iff mask
  present, else `None`) — non-breaking for existing analysis consumers.
- **`__main__` smoke block:** update unpack to the new 9-tuple.

### 1.4 `src/train.py` — masks, application, logging

- Read `spatial_holdout_prob = data_cfg.get('spatial_holdout_prob', 0.0)` and
  `w_kriging = config.get('loss', {}).get('w_kriging', 0.0)`.
- **Validation mask (deterministic):** built once after model init iff `prob > 0.0`:
  `val_holdout_mask = make_spatial_holdout_mask(num_nodes, prob, device, seed=42)`; log count.
- **Train loop (per batch):** if `prob > 0.0`, build a *fresh random* mask (`seed=None`),
  set `batch['spatial_holdout_mask']`, and
  `batch['dynamic_features'] = apply_spatial_holdout(batch['dynamic_features'], mask)`.
  Always set `batch['w_kriging'] = w_kriging`.
- **Val loop (per batch):** if `val_holdout_mask is not None`, set it + apply holdout;
  set `batch['w_kriging'] = w_kriging`.
- **Unpack 10-tuple** at both call sites. Accumulate `epoch_loss_kriging` / `val_loss_kriging`.
- **W&B logging (Q2 = full metrics):** add `train_loss_kriging`, `val_loss_kriging`,
  `val_kriging_nodes` (`= val_holdout_mask.sum()`), **plus** `val_kriging_rmse`,
  `val_kriging_mae`, `val_kriging_msle` via `calculate_kriging_metrics(kriging_pred,
  target_impute[:, val_holdout_mask])` accumulated across val batches. All kriging logging
  gated behind `w_kriging > 0.0`.
- Mock-data path: never sets a mask ⇒ `loss_kriging == 0`, `kriging_pred = None`, fully
  backward-compatible.

### 1.5 `configs/` — ACTIVATE defaults (Q1 = activate)

Flip the committed defaults so kriging trains by default:
- `configs/train/train.yaml`: `data.spatial_holdout_prob: 0.15`, `loss.w_kriging: 0.5`.
- `configs/train/train_test.yaml`: same (`spatial_holdout_prob: 0.15`, `w_kriging: 0.5`)
  so the mandated verification command exercises the **live** kriging path directly — no
  separate temp config needed. (Update the now-stale "Inactive at 0.0 on main" comments.)

### 1.6 Tests

- **`tests/test_decoders.py`:** FIX Phase-1 regression — pass `traffic_mean=tensor(0.0)`,
  `traffic_std=tensor(1.0)`, unpack **4** values; assert `loss_kriging == 0` at `w_kriging=0`.
- **`tests/test_pipeline.py`:** update unpack to 10-tuple; add a test passing a
  `spatial_holdout_mask` + `w_kriging>0` asserting `loss_kriging > 0`, finite, and grads
  flow into `kriging_decoder`.
- **`tests/test_kriging_infra.py`** (Q3 = extend in place): add live-path tests alongside
  the existing dormant-infra tests, under a new section header:
  - `make_spatial_holdout_mask`: `[N]` bool, determinism under fixed seed, `prob=0→all
    False`, `prob=1→all True`.
  - `apply_spatial_holdout`: held-out nodes zeroed; others byte-identical.
  - `compute_kriging_loss`: shape match, empty-holdout → `0.0` (no NaN), MSLE-on-raw sanity.
  - `compute_loss` 4-output contract; `w_kriging=0` short-circuits the kriging path.

### 1.7 `scripts/desk_check_kriging.py` — extend (Part 3, Layer-2 wiring)

- Holdout masking: held-out nodes zeroed, complement untouched.
- Decoder shape `K == mask.sum()`.
- `compute_kriging_loss` desk numbers with extreme values (0, 1e6) + empty mask → exactly 0.
- End-to-end `compute_loss` returns 4 finite values with a synthetic kriging path; grads
  reach all heads when `w_kriging>0`, and kriging grad is `None` when `w_kriging=0`.

---

## 2. Desk Checks (run BEFORE training — prove correctness)

`python scripts/desk_check_kriging.py` must print & pass:
1. Inverse-transform identity (re-asserted in kriging context).
2. Holdout masking: `(masked[:, mask] == 0).all()` and `allclose(masked[:, ~mask], orig)`.
3. Decoder shape `== (B, mask.sum(), T, D)`.
4. **Empty holdout → exactly `0.0`, finite** (the critical NaN trap).
5. MSLE finite for raw counts up to 1e6.
6. `total_loss.backward()` populates `kriging_head` grads at `w_kriging>0`; `None` at 0.

## 3. Test Suite

`python -m pytest tests/ -q` — all green, including repaired `test_decoders.py` and the
new kriging-integration tests.

## 4. Integration Verification (database mode, as instructed)

The mandated command now exercises the **live** kriging path directly (defaults activated):
```bash
docker compose exec agent-workspace python src/train.py --use-database \
  --config configs/train/train_test.yaml   # spatial_holdout_prob=0.15, w_kriging=0.5
```
Expected: 1 epoch completes, no NaN, logs `val_loss_kriging` (finite, > 0),
`val_kriging_nodes > 0`, and `val_kriging_rmse/mae/msle`.

Regression sanity (kriging disabled) — confirm the default-off path is still clean:
```bash
docker compose exec agent-workspace python src/train.py --use-mock_data  # no mask → loss_kriging==0
```

---

## 5. Execution Order (after your approval)

0. **Git safety:** `git add . && git commit -m "Auto-commit before Phase 2"`.
1. `dataset.py` helpers.
2. `decoders.py` (`compute_kriging_loss` + 4-output `compute_loss`).
3. `pipeline.py` (instantiate + thread + 9-tuple) + fix `__main__`.
4. `train.py` (masks, apply, unpack, logging).
5. Fix `tests/test_decoders.py`, update `test_pipeline.py`, add kriging integration tests.
6. Extend `scripts/desk_check_kriging.py`; run it.
7. `pytest tests/` green.
8. Integration runs (§4); capture metrics.
9. Update `EXPERIMENTS.md` with the Layer-2 entry + desk-check/verification results.

---

## 6. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| `mse_loss` on empty holdout → NaN | Explicit `numel()==0 → 0.0` guard in `compute_kriging_loss`. |
| Return-arity change breaks callers | Audited 6 unpack sites (pipeline fwd + `__main__`, train ×2, tests ×2); all updated to the 10-tuple this phase. |
| Kriging loss dominates training | Separate logged term; `w_kriging` tunable; activated to `0.5` per Q1 — watched in verification, rollback to `0.0` if it overwhelms imputation/forecast. |
| Holdout zeroing vs temporal NaN mask | Safe order: holdout zeros nodes → pipeline `nan_to_num` handles residual NaNs; kriging target is the pre-mask `target_impute`. |
| Train/val mask leakage | Val mask deterministic (seed=42); train mask re-randomised per batch. |

---

## 7. Open Questions — RESOLVED

- **Q1 — `main` defaults:** ✅ **Activate.** Commit `w_kriging=0.5`,
  `spatial_holdout_prob=0.15` in `train.yaml` + `train_test.yaml`.
- **Q2 — Kriging val metrics:** ✅ **Full metrics.** Log loss + node count +
  RMSE/MAE/MSLE; surface `kriging_pred` from the forward pass (10-tuple).
- **Q3 — Test file:** ✅ **Extend `tests/test_kriging_infra.py`** in place.

---

**Awaiting your approval before writing any functional code.**
