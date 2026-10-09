# Spatial sparsity scaling: GNN vs naive profile donors

Metric: `v5_test_metrics.obs_ind_rmse` on 24 frozen held-out stations (holdout identical to the v5 suite (seed 42)). 63 runs.

Network area (convex hull of all stations): 21.0 km².

| N_train | density /km² | GNN RMSE | naive RMSE | Δ (GNN − naive) | mean nearest-train km |
|---:|---:|---:|---:|---:|---:|
| 15 | 0.71 | 46.93 | 44.45 | +2.48 | 0.42 |
| 25 | 1.19 | 51.85 | 42.67 | +9.18 | 0.31 |
| 35 | 1.67 | 46.84 | 43.57 | +3.27 | 0.21 |
| 45 | 2.14 | 44.88 | 42.71 | +2.17 | 0.18 |
| 57 | 2.71 | 40.42 | 42.63 | -2.22 | 0.15 |

RMSE(N) = A · N^(−α) + RMSE∞  (bootstrap 90% intervals in brackets)

| arm | A | α | RMSE∞ | noise sd (source) |
|---|---:|---:|---:|---|
| gnn | 32.12 | 0.339 [0.29, 2.15] | 35.00 [35.0, 44.0] | 1.63 (pooled replicate sd over 41 runs at 5 N) |
| naive | 80465.61 | 4.000 [0.16, 4.00] (at bound — not identified) | 42.83 [35.0, 43.3] | 0.83 (pooled replicate sd over 20 runs at 4 N) |

- Point-fit crossover N* = 65 — **extrapolated beyond the sampled range**
- Definitive crossover N*₉₅ (P(GNN better) ≥ 0.95 from there on) = > 400 (not reached)
- Bootstrap N* 90% interval: [47, > 400 (not reached)]; never crosses in 32% of resamples, GNN always better in 0%
- As density: N* ≈ 3.09 training stations / km²

![crossover](figures/scaling_crossover.png)
