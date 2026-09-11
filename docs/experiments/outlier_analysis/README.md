# Outlier Event Analysis — ResilientIoT Spatiotemporal GNN

Reproducible pipeline that investigates how the ResilientIoT unified GNN
(imputation + ZINB forecasting + spatial kriging) behaves during **outlier
events** in the 81-sensor Newcastle pedestrian-traffic network.

It pulls live inference payloads from **Cloudflare R2** and training curves from
**Weights & Biases**, runs the eight tests specified in
[`../outlier_analysis_prompt.md`](../outlier_analysis_prompt.md), and writes
figures, a machine-readable `results.json`, and a comprehensive markdown report.

## Quick start

```bash
cd docs/experiments/outlier_analysis
python run_analysis.py            # full 31-day analysis of the default R2 run
python run_analysis.py --quick    # first 3 days (fast sanity pass)
```

Outputs land in [`outputs/`](outputs/):

| File | Contents |
|------|----------|
| `outputs/report.md` | The comprehensive report (exec summary, catalogue, per-test results, recommendations) |
| `outputs/results.json` | All numeric results (JSON) for downstream use |
| `outputs/figures/*.png` | One or two figures per test |

Raw payloads are cached under `cache/` (gitignored). Re-runs read the cache;
pass `--refresh` to re-download.

## CLI

```
--run-id NAME       R2 run id to analyse (default: registry default_run_id)
--max-days N        Limit to the first N dates
--quick             Shortcut for --max-days 3
--max-workers N     Download concurrency (default 24)
--refresh           Ignore cache and re-download
--skip-wandb        Skip the W&B training-curve pull
--no-simulated      Don't download simulated-node files (use page predictions only)
```

## Package layout

```
oea/
  config.py     constants, thresholds, UK-2026 calendar
  data_io.py    cached concurrent R2 fetch + W&B pull
  loading.py    builds the per-node 31-day Dataset (gt, μ, θ, π, σ, masks, attention…)
  analysis.py   the eight tests (+ shared outlier detection / event catalogue)
  plotting.py   shared figure style
  report.py     assembles report.md + results.json
run_analysis.py orchestrator / CLI
```

## The eight tests

1. **Outlier detection & cataloguing** — per-node 3σ residual flags → temporal/spatial event clustering & classification.
2. **ZINB behaviour during outliers** — variance/π response and empirical 50/90/99% interval coverage vs baseline.
3. **Propagation to simulated (zero-shot) nodes** — attention re-weighting, prediction response, latency, spatial decay.
4. **Build-up dynamics** — event-aligned composites of μ, θ, π, imputation residual and attention entropy; recovery/hysteresis.
5. **Kriging stratified by outlier exposure** — held-out error under normal vs neighbour-outlier conditions, vs the imputation head.
6. **Tripwire correlation** — PI/MU/THETA clamp events vs traffic level, night-time and outlier coincidence; node vs time concentration.
7. **Scale-invariance stress test** — MSLE vs raw-SE loss contribution for a high- vs low-traffic sensor.
8. **Two-extremes divergence** — the `rmse_fill − rmse_drop` gap as an outlier-severity proxy.

## Method notes

- Outlier rule: `|gt − μ| > 3σ` per node (σ = residual std-dev). Residuals are
  heteroscedastic, so the plain std (not MAD) is used to avoid over-flagging the
  normal diurnal peak.
- ZINB coverage uses the NB parameterisation `n = θ`, `p = θ/(θ+μ)` with
  zero-inflation `F(k) = π + (1−π)·F_nb(k)`.
- Network-wide = ≥30% of real nodes flagged simultaneously; cluster = spatially
  contiguous (≤1 km); isolated = single node; multi-focal = many but spread and
  non-simultaneous.
