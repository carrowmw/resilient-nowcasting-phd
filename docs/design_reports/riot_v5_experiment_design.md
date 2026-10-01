# RIoT pipeline v5 — experiment design

Sources: `docs/design_reports/zero_collapse_investigation.md` (recommendations R1–R7, Phases 1–7)
and `docs/thesis/chapter7_experiment3_reconstruction.tex` (Experiment 3 ablations, limitations,
AUTHOR-CHECKs). Generator: `scripts/riotv5/generate_v5_configs.py`. Runner:
`scripts/riotv5/run_v5_suite.py`, launched by `scripts/riotv5/start_v5_suite.sh`.

## 1. Pipeline v5 in one paragraph

Every config carries `pipeline_version`. `src/run_config.py::resolve_config` validates v5 configs
against `V5_SCHEMA` (every key explicit — no defaults) and maps configs without a version onto
`LEGACY_V4_EQUIVALENTS`, so v4 YAMLs and checkpoints still load and behave identically
(`tests/test_v5.py::test_legacy_v5_loss_equals_v4_tuple_loss`). v5 adds device-side batch
preparation (`src/harness/batch_prep.py`: structured artificial gaps, unified missing encoding,
holdout blanking, synthetic outages), covariates (`src/covariates.py`: calendar, seasonal profile,
gap run-length, lags), likelihoods (`src/likelihoods.py`: NB, ZTNB, outage mixture), the loss
stack (`src/losses.py`: observed-only routing, magnitude weighting, under-prediction pinball,
quantiles, Huber mix), MTL strategies (`src/harness/mtl.py`), a model factory with LSTM / naive
models (`src/model_factory.py`) and the v5 metric suite (`src/harness/evaluate_v5.py`:
peak / high-count RMSE, collapse rate, gap mass, SO@k, CRPS, inductive degeneracy checks).

Loss routing always uses the un-ablated `input_observed` / `target_observed` masks, so mask-channel
ablations change the model's inputs only — never which slots are scored (thesis C12).

## 2. Why pre-declared bases instead of gated phases

The report's phases are gated: each phase's winner seeds the next. One unattended 4-GPU launch
cannot gate, so every phase runs on a **pre-declared base** — the configuration the report
predicts will win. If a screen contradicts the prediction, the affected later groups get re-run
on the corrected base (cheap: the screens are one seed).

| Base | Definition |
|---|---|
| `V4R` | v4 `full_inductive` in v5 form (every v5 key at its legacy value). Reproduces v4 numerics, scored with v5 metrics. |
| `P2B` | V4R + hygiene (observed-only forecast / impute / kriging targets, obs-NLL + SO@12 selection) + NB + `complete_weather` |
| `P3B` | P2B + unified missing encoding + observed-only structured gaps + mixed strategy @ 0.30 |
| `P4B` | P3B + R3 stack: calendar, log1p inputs, seasonal profile (`inductive_neighbour`), run-length, sensor embedding (d 8, dropout 0.2) |
| `P5B` | P4B + anchored outage mixture (missingness-feature head, artificial-outage BCE weight 1.0) |
| `V5F` | P5B + magnitude α 0.5, q50 / q90 heads, Huber 0.5, missing-aware temporal attention, GAT drops missing sources, consistent night channel at held-out stations = **full v5** |

## 3. Groups

| Group | Runs | What it answers |
|---|---|---|
| `thesis_v4` | 17 × 3 seeds | Experiment 3 reproduced on V4R (full, no-impute, no-kriging, no-masks, no-PE, LSTM, bs32, dropout 0.3) plus the chapter's limitations: night-channel leak fix (Lim. 4), total-only / night-only channels (Lim. 5), identity-graph RIoT and a less-confounded LSTM (Lim. 7), masks × imputation 2×2 (Lim. 9), and three bs32-mechanism probes (update budget, lr / 4, peak weighting). |
| `p1_hygiene` | 6 × 3 | E1.1–E1.6 cumulative staircase from V4R to P2B. |
| `p2_masking` | 9 (P3B × 3) | Unified iid, observed-only iid control, block (empirical / geometric), transplant, tail, mixed at 0.15 / 0.30 / 0.50. |
| `p3_inputs` | 14 (stack × 3) | Each R3 input alone, four profile variants for held-out stations, seq 48 / 96, lags, key-padding attention, GAT missing-source dropping, the stack and stack @ seq 48. |
| `p4_likelihood` | 8 | ZTNB, mixture (latent / missingness / detached head), anchored = P5B, gap-length prior, missing-slot weight 0.25, mixture on all heads. |
| `p5_magnitude` | 8 | α 0.5 / 1.0, under-prediction pinball 0.1 / 0.5, quantiles, Huber, low-end tolerance, full + pinball. |
| `p6_mtl` | 5 | Uncertainty weighting, PCGrad, 10-epoch impute pre-training, detached imputation head, v4-style aux reconstruction. |
| `p7_v5_suite` | 19 | The thesis ablation suite rerun on V5F, plus isolations of each v5 component (no outage head, iid masks, no calendar / profile), a v5 LSTM and plain LSTM, and naive floors (seasonal profile, last observation). |
| `loo` | 16 × 1 | Leave-one-out from V5F: every component removed individually. |

Exact per-run descriptions are in `configs/train/v5/suite_manifest.yaml`; every generated YAML
has a header naming its base, overrides and purpose.

## 4. Budget and tiers

209 runs, 242.7 cost units (1 unit ≈ one v4 full run; cost scales with seq_len, batch / accumulation,
epochs, model type) ≈ **60.7 v4-run durations of wall time on 4 GPUs**.

| Tier | Runs | Cost | Content |
|---|---|---|---|
| 1 | 42 | 40.4 | Headline: seed 42 of every core result (thesis suite, hygiene staircase, bases, v5 suite, naive floors) |
| 2 | 61 | 84.3 | Single-seed screens and LOO |
| 3 | 106 | 117.9 | Seeds 43 / 44 of the core runs |

The runner schedules tier 1 first, then 2, then 3, longest job first within a tier, so a partial
run still yields the headline table.

## 5. Running it

On the GPU server, after pulling:

```bash
python scripts/riotv5/generate_v5_configs.py --check   # regenerates + validates all 209 YAMLs
./scripts/riotv5/start_v5_suite.sh --gpus 0 1 2 3 --dry-run   # inspect the schedule
./scripts/riotv5/start_v5_suite.sh                      # launch, default --gpus 0 1 2 3
./scripts/riotv5/show_v5_logs.sh                        # re-attach to the stream
./scripts/riotv5/stop_v5_suite.sh                       # graceful stop (children get SIGTERM)
```

Subsets: `--tiers 1`, `--groups thesis_v4 p7_v5_suite`, `--seeds 42`, `--jobs 'v5_*'`,
`--jobs-per-gpu 2` (if memory allows). Re-launching resumes: runs with a COMPLETED status in
`logs/v5_suite/status/` are skipped (`--force` re-runs). Starts are staggered 45 s apart to
spread the database load.

Outputs in `logs/v5_suite/`: `runs/<run>.log`, `status/<run>.json`, `v5_suite_runs.csv`
(one row per run), `v5_suite_summary.csv` (mean ± std over seeds, holdout-manifest sha consistency
flag) and `tables/<group>.tex`. `python scripts/riotv5/run_v5_suite.py --summarise-only` rebuilds the
summaries at any time. W&B project: `resilient-iot-v5`.

Local validation: `run_v5_suite.py --smoke --use_mock_data` runs every config for 2 epochs on
12 synthetic nodes with W&B disabled (all 103 seed-42 experiments pass); `pytest tests` → 154 passed.

## 6. Analysis notes

- Compare within a group against its base, not across groups. The P1 staircase links V4R to P2B,
  and P2B → P3B → P4B → P5B → V5F are each members of the preceding group, so the chain is connected.
- The headline metrics are high-count / peak RMSE and SO@k, per the stated priority (high values
  matter more than 0 vs 1). The collapse rate and gap mass are diagnostics, not objectives.
- Inductive degeneracy check (thesis): compare `ind_pred_pairwise_corr_mean` to
  `ind_true_pairwise_corr_mean`. If predictions at held-out stations correlate far more strongly
  than the truth does, the model is emitting a shared city-wide profile.
- `v4r_*` numbers are not comparable to the thesis tables in absolute terms: v5 scores observed
  targets only and fixes the scaler / weather leaks. Within-suite deltas are the result.
