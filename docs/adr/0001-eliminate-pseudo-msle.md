# ADR 0001: Eliminate Pseudo-MSLE — Compute Scale-Invariant Loss on Raw Counts

- **Status:** Accepted
- **Date:** 2026-06-30
- **Deciders:** ML engineering (unified architecture, Phases 1–3)
- **Supersedes:** the "pseudo-MSLE" approach explored in `exp/enriched-gat-kriging`
- **Related:** `docs/model/spatial_estimation_architecture.md` §5 (Scale-Invariant Loss),
  `NSTAR_BACKEND_ARCHITECTURE.md` §3 (Joint Training & Loss Logic)

---

## Context

Traffic counts span several orders of magnitude — quiet residential nodes record a handful of
vehicles per 15 minutes while city-centre nodes record hundreds. Plain MSE on these counts is
dominated by the high-magnitude nodes, so the model under-fits the low-count majority. The standard
scale-invariant remedy for count data is **Mean Squared Logarithmic Error (MSLE)**:

```
L_MSLE = mean( (log1p(pred) - log1p(target))^2 )
```

MSLE is only defined on **non-negative** values. The encoder, however, consumes
`StandardScaler`-normalized dynamic features (`z = (x - μ) / σ`), which are **signed floats**. This
creates a domain conflict: MSLE cannot be applied directly to the scaled tensors the model produces.

Two ways to resolve the conflict were on the table:

1. **Pseudo-MSLE on scaled features.** Clamp the negative scaled values to `0` and apply `log1p` to
   the result. This is cheap (no inverse transform) but **clamps away every value below the channel
   mean** — roughly 40–50% of the distribution — collapsing all sub-mean nodes to an identical `0`.
   The loss then ignores precisely the low-count nodes MSLE was meant to protect. It is also
   mathematically meaningless: `log1p` of a z-score has no count interpretation.

2. **TRUE MSLE on raw counts.** Inverse-transform the traffic channel back to raw counts before the
   loss, then apply MSLE in the count domain where it is well-defined.

## Decision

**Compute TRUE MSLE on raw counts by inverse-transforming the traffic channel.**

- The scaler statistics are computed once in `RealIoTDataset` and exposed as `traffic_mean` /
  `traffic_std`. They are registered on `ResilientIoTPipeline` as `nn.Module` **buffers**
  (`register_buffer`), so they are device-safe — they move to the GPU with the model and require no
  manual `.to(device)` plumbing.
- In `compute_loss` (and `compute_kriging_loss`) the **traffic channel (ch 0)** of both prediction and
  target is inverse-transformed:

  ```python
  traffic_raw = (traffic_scaled * traffic_std) + traffic_mean
  traffic_raw = torch.clamp(traffic_raw, min=0.0)   # enforce the count domain
  loss_traffic = scale_invariant_mse(pred_raw, target_raw)
  ```

- **Weather channels (1–2)** are continuous meteorological readings, not counts, so they keep
  `F.mse_loss` on the scaled values.
- The inverse transform is a **constant affine map**, so gradients flow through it without any special
  backward handling.
- This single path is shared by **both** the imputation head and the spatial-kriging head, keeping the
  two losses directly comparable.

## Consequences

**Positive**
- Scale-invariant: low-count and high-count nodes contribute on a comparable footing.
- Zero approximation error — the full distribution is preserved; nothing is clamped away in the scaled
  domain (the only clamp is the physically-correct `raw >= 0`).
- Device-safe by construction (buffers), eliminating a class of CPU/GPU mismatch bugs.
- Mathematically interpretable: the loss lives in the count domain.
- Guarded by `CRITICAL:` NaN/Inf tripwires on every loss term.

**Negative / cost**
- The pipeline must carry and correctly populate the scaler buffers; a wrong `mean`/`std` silently
  rescales the loss. Mitigated by the dataset being the single source of truth and by the
  `desk_check_scale_invariant.py` inverse-transform test.
- A small coupling between dataset preprocessing and loss computation (the loss must know the scaler
  stats).

## Alternatives considered

- **Pseudo-MSLE on clamped scaled features** — rejected: discards ~40–50% of the distribution and has
  no count-domain meaning (see Context).
- **Per-batch min-max rescaling before MSLE** — rejected: introduces batch-dependent, non-stationary
  loss scaling and breaks comparability across batches/epochs.
- **Plain MSE on raw counts** — rejected: not scale-invariant; reintroduces the high-magnitude
  dominance MSLE exists to remove.

## Verification

- `scripts/desk_check_scale_invariant.py` — inverse transform recovers the original scale (max error
  `< 1e-5`); raw vs pseudo-MSLE divergence demonstrated; CUDA-guarded device-consistency check.
- `scripts/desk_check_unified_loss.py` — end-to-end: all three loss terms finite, weighted-sum
  arithmetic exact, gradient flow to all heads, channel separation, and the NaN tripwire firing.
