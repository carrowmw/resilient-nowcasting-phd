# Spatial Interpolation Ablation Studies

## Context

Analysis of 10 deployed runs revealed that **removing night-time zeros from the training data caused the kriging head to collapse** — simulated node predictions fell to near-zero for all new-data runs. Additionally, the **checkpoint selection explicitly excludes the kriging loss**, optimising for forecasting at the expense of spatial quality.

These ablations systematically test interventions to recover spatial interpolation quality on the corrected (zero-free) dataset.

## Experimental Design

All configs use: `eval_mode: inductive`, `inductive_holdout_frac: 0.2`, `model_type: gnn`, 3-year data range.

### Study A: Isolate Data vs Architecture Effect

| Config | What it tests | Key change vs baseline |
|:---|:---|:---|
| `ablation_a1_large_arch_new_data.yaml` | Is it the data or the architecture? | floral-rain-6 arch (h=64, L=3) on new zero-free data |

> **Note:** Study A2 (small arch on OLD zero-filled data) requires reverting the data preprocessing pipeline — this is a code change, not a config change. Flagged separately.

### Study B: Kriging Loss Weight Sweep

| Config | What it tests | Key change vs baseline |
|:---|:---|:---|
| `ablation_b1_kriging_w2.yaml` | Can 4× kriging weight rescue spatial quality? | `w_kriging: 0.5 → 2.0` |
| `ablation_b2_kriging_w5.yaml` | Upper bound of loss rebalancing | `w_kriging: 0.5 → 5.0` |
| `ablation_b3_pure_spatial.yaml` | Does forecasting compete with kriging? | `w_forecast: 0.0`, `w_kriging: 2.0` |

### Study C: Checkpoint Selection on Kriging Loss

> **Note:** This requires a code change to `src/train.py` L672–674, where `val_loss_total` currently strips the kriging term. Flagged separately.

### Study D: Higher Spatial Holdout

| Config | What it tests | Key change vs baseline |
|:---|:---|:---|
| `ablation_d1_holdout_04.yaml` | More training signal for kriging head | `spatial_holdout_prob: 0.2 → 0.4` |
| `ablation_d2_holdout_05.yaml` | Maximum spatial masking | `spatial_holdout_prob: 0.2 → 0.5` |

### Study E: Dropout as Spatial Regulariser

| Config | What it tests | Key change vs baseline |
|:---|:---|:---|
| `ablation_e1_dropout_kriging.yaml` | Dropout + kriging weight combined | `dropout: 0.3`, `w_kriging: 2.0` |
| `ablation_e2_high_dropout.yaml` | Can heavy dropout break zero-collapse? | `dropout: 0.5`, `w_kriging: 2.0` |

## How to interpret results

After training, run:
```bash
python scripts/compare_runs.py --all
```

**Key metrics to watch (in order of importance):**
1. **Zero Collapse Ratio** — fraction of simulated nodes with mean μ < 1.0 (target: < 0.05)
2. **Sim μ Mean** — average predicted pedestrian count for simulated nodes (target: > 10)
3. **Sim CV** — coefficient of variation across simulated nodes (target: > 0.4, indicating spatial diversity)
4. **Real-Sim Nearest Corr** — correlation between sim node predictions and their nearest real neighbour (target: > 0.2)
5. **Moran's I** — spatial autocorrelation (target: > 0.3, but beware trivially high values from zero-collapse)

## Suggested run priority

1. **A1** (diagnostic: data vs arch) — run first, informs all subsequent decisions
2. **E1** (d=0.3 + w_k=2.0) — combines two partial fixes, highest chance of recovery
3. **B1** (w_k=2.0) — clean single-variable test of kriging weight
4. **D1** (holdout=0.4) — orthogonal intervention, can run in parallel with B1
5. Remaining (B2, B3, D2, E2) — run based on results of above