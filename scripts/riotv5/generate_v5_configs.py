#!/usr/bin/env python3
"""
Generate the RIoT pipeline-v5 experiment suite.

Design: docs/design_reports/riot_v5_experiment_design.md
Sources: docs/design_reports/zero_collapse_investigation.md (Phases 1-7) and
         docs/thesis/chapter7_experiment3_reconstruction.tex (limitations / AUTHOR-CHECKs).

The report's phases are *gated* (each phase's winner seeds the next). A single
unattended 4-GPU launch cannot gate, so every phase runs on a PRE-DECLARED base:

  V4R   v4 replica in v5 format: every v5 key at its LEGACY_V4_EQUIVALENTS value.
        Reproduces v4 numerics exactly; scored with the v5 metric suite.
  P1    hygiene staircase on V4R                         (E1.1 - E1.6)
  P2B   P1 end state (observed-only targets, NB, complete weather) -> masking screens (E2.x)
  P3B   P2B + unified encoding + mixed gaps @ 0.30          -> input screens   (E3.x)
  P4B   P3B + the full R3 input stack ("P3 stack")          -> likelihood      (E4.x)
  P5B   P4B + anchored outage mixture (E4.4)                -> magnitude       (E5.x)
  V5F   P5B + magnitude/quantile/Huber + R7 attention items = full v5. The Phase 6
        (MTL), Phase 7 (thesis suite on v5) and leave-one-out groups centre on it.

Every screen is a one-factor change from its base, so the screens remain valid
whichever candidate later wins; a gated follow-up only needs a new base here.

Output:
  configs/train/v5/<group>/<name>__s<seed>.yaml   one file per (experiment, seed)
  configs/train/v5/suite_manifest.yaml            job list for scripts/riotv5/run_v5_suite.py

Usage:
  python scripts/riotv5/generate_v5_configs.py            # (re)write configs + manifest
  python scripts/riotv5/generate_v5_configs.py --check    # also resolve + validate every config
"""
import argparse
import copy
import shutil
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from run_config import LEGACY_V4_EQUIVALENTS, PIPELINE_VERSION, resolve_config  # noqa: E402

OUT_DIR = PROJECT_ROOT / "configs" / "train" / "v5"
WANDB_PROJECT = "resilient-iot-v5"
SEEDS_FULL = [42, 43, 44]
SEEDS_SCREEN = [42]

# ---------------------------------------------------------------------------
# Base configurations
# ---------------------------------------------------------------------------
# configs/train/v4/train_v4_full_inductive.yaml, verbatim, plus every v5 key at its
# v4-equivalent value. infra.yaml (num_workers, edge caches, data_source) is merged
# by train.load_config as before.
V4R = {
    "pipeline_version": PIPELINE_VERSION,
    "wandb": {
        "project": WANDB_PROJECT,
        "sweep_project": "resilient-iot-sweeps-v5",
        "collection": None,     # set per experiment
        "run_name": None,       # set per (experiment, seed)
    },
    "data": {
        "start_date": "2023-04-02",
        "end_date": "2026-04-02",
        "seq_len": 12,
        "target_veh_class": "person",
        "target_dir": "all",
        "mask_prob": 0.15,
        "spatial_holdout_prob": 0.2,
        "inductive_holdout_frac": 0.2,
        **copy.deepcopy(LEGACY_V4_EQUIVALENTS["data"]),
        # Pinned (v5 requires it): the v4 coupling gave inductive seed = random_seed = 42.
        # Pinning keeps the SAME 24 inductive stations for every seed, so seed variance is
        # optimisation variance only and every run shares one holdout manifest.
        "inductive_holdout_seed": 42,
    },
    "model": {
        "num_nodes": 81,
        "static_feat_dim": 15,
        "time_series_dim": 5,
        "embedding_dim": 32,
        "hidden_dim": 64,
        "sparsity_threshold": 0.5,
        "num_layers": 3,
        "num_heads": 4,
        "dropout": 0.1,
        "dim_feedforward": None,
        "zinb_eps": 1.0e-6,
        "use_positional_encoding": True,
        **copy.deepcopy(LEGACY_V4_EQUIVALENTS["model"]),
    },
    "training": {
        "epochs": 100,
        "batch_size": 128,
        "lr": 0.0003,
        "grad_clip_max_norm": 1.0,
        "early_stop_patience": 10,
        "early_stop_min_delta": 0.001,
        "random_seed": 42,
        **copy.deepcopy(LEGACY_V4_EQUIVALENTS["training"]),
        # R6 diagnostic, logged on every run (numerically inert: extra autograd.grad
        # calls with retain_graph, no RNG use).
        "grad_cosine_every": 100,
    },
    "loss": {
        "w_forecast": 1.0,
        "w_impute": 1.0,
        "w_kriging": 0.5,
        **copy.deepcopy(LEGACY_V4_EQUIVALENTS["loss"]),
    },
    "selection": copy.deepcopy(LEGACY_V4_EQUIVALENTS["selection"]),
    "evaluation": copy.deepcopy(LEGACY_V4_EQUIVALENTS["evaluation"]),
    "model_type": "gnn",
    "eval_mode": "inductive",
    "train_split": 0.70,
    "val_split": 0.15,
    "amp": True,
}


def derive(base: dict, overrides: dict) -> dict:
    """Copy `base` and apply dotted-key overrides ("loss.w_impute": 0.0). Every
    override key must already exist — a typo can never silently add a dead key."""
    cfg = copy.deepcopy(base)
    for dotted, value in overrides.items():
        node = cfg
        *path, leaf = dotted.split(".")
        for p in path:
            node = node[p]
        assert leaf in node, f"override {dotted!r} does not name an existing config key"
        node[leaf] = copy.deepcopy(value)
    return cfg


# R1 target hygiene + correct checkpoint selection (report E1.4).
HYGIENE = {
    "loss.target_mask": "observed",
    "loss.impute_positions": "masked_observed",
    "loss.kriging_observed_only": True,
    "selection.metric": "obs_loss_plus_so",
}
# End of Phase 1: + NB (no structural zeros among observed slots) + complete weather.
P2B = derive(V4R, {**HYGIENE, "loss.forecast_likelihood": "nb", "data.complete_weather": True})

# R2 structured gaps over observed traffic only, one encoding for "missing".
STRUCTURED = {
    "data.mask_weather": False,
    "data.artificial_mask_observed_only": True,
    "data.unify_missing_encoding": True,
}
P3B = derive(P2B, {**STRUCTURED, "data.mask_strategy": "mixed", "data.mask_prob": 0.30})

# R3 input stack (E3.1 - E3.5). Profile: inductive_neighbour is the only variant that
# gives held-out stations a non-trivial, leakage-free level (E3.3 screens all four).
R3_STACK = {
    "data.calendar_features": True,
    "data.log1p_inputs": True,
    "data.profile_prior": "inductive_neighbour",
    "data.gap_runlength_feature": True,
    "model.sensor_embedding_dim": 8,
    "model.sensor_embedding_dropout": 0.2,
}
P4B = derive(P3B, R3_STACK)

# R4 anchored outage mixture (E4.4): missingness-only outage head + supervised BCE on
# artificial outages.
OUTAGE = {
    "loss.forecast_likelihood": "outage_mixture",
    "model.outage_head_input": "missingness",
    "loss.w_artificial_outage": 1.0,
}
P5B = derive(P4B, OUTAGE)

# Full v5: + R5 (magnitude weighting, q50/q90 heads, Huber-mixed imputation) + R7
# (missing-aware temporal attention, GAT ignores missing sources) + consistent night
# channel at held-out stations (thesis limitation 4).
V5_EXTRAS = {
    "loss.magnitude_weight_alpha": 0.5,
    "model.quantiles": [0.5, 0.9],
    "loss.impute_huber_mix": 0.5,
    "model.temporal_key_padding": True,
    "model.gat_drop_missing_sources": True,
    "data.holdout_night_channel": "consistent",
}
V5F = derive(P5B, V5_EXTRAS)

# The thesis LSTM (configs/train/v4/train_v4_baseline_lstm.yaml) relative to V4R.
LSTM_V4 = {
    "model_type": "lstm",
    "data.spatial_holdout_prob": 0.0,
    "model.embedding_dim": 16,
    "model.num_layers": 2,
    "training.batch_size": 64,
    "loss.w_impute": 0.0,
    "loss.w_kriging": 0.0,
}
# Less-confounded LSTM (thesis limitation 7): same inputs RIoT sees (dynamic + mask
# channels + covariates + static) and a distributional (NB) head; still no graph.
LSTM_RICH = {"model.baseline_inputs": "dynamic_masks_covariates_static", "model.baseline_head": "nb"}

# Longer windows keep the effective batch at 128 via gradient accumulation (memory).
SEQ48 = {"data.seq_len": 48, "training.batch_size": 64, "training.grad_accum_steps": 2}
SEQ96 = {"data.seq_len": 96, "training.batch_size": 32, "training.grad_accum_steps": 4}

# ---------------------------------------------------------------------------
# Experiments: (group, name, base, overrides, seeds, tier, description)
#   tier 1 = headline results (thesis tables), tier 2 = mechanism / screens,
#   tier 3 = extra seeds. The runner schedules tier 1 first.
# ---------------------------------------------------------------------------
EXPERIMENTS = []


def E(group, name, base, overrides, seeds, tier, desc):
    EXPERIMENTS.append(dict(group=group, name=name, base=base, overrides=overrides,
                            seeds=seeds, tier=tier, desc=desc))


# ---- thesis_v4: Chapter 7 Experiment 3 re-run under v4 semantics, v5 metrics ----------
# Re-scores every Table 7.3 configuration on observed-only targets, with the trans/ind
# split, ZIR strata, mask-flip test, traffic-only kriging, per-node normalised error and
# the inductive pairwise-correlation check, and with three seeds (limitations 1-3, 6).
T = "thesis_v4"
E(T, "v4r_full", V4R, {}, SEEDS_FULL, 1, "v4 full inductive (reference wph1nhvj) — also report E1.0 control")
E(T, "v4r_no_impute", V4R, {"loss.w_impute": 0.0}, SEEDS_FULL, 1, "Ablation 1: no reconstruction loss")
E(T, "v4r_no_kriging", V4R, {"loss.w_kriging": 0.0}, SEEDS_FULL, 1, "Ablation 2: no kriging loss")
E(T, "v4r_no_masks", V4R, {"data.ablate_masks": True}, SEEDS_FULL, 1, "Ablation 3: no mask channels (loss mask un-ablated, C12)")
E(T, "v4r_no_pe", V4R, {"model.use_positional_encoding": False}, SEEDS_FULL, 1, "Ablation 4: no positional encoding")
E(T, "v4r_lstm", V4R, LSTM_V4, SEEDS_FULL, 1, "Non-spatial LSTM baseline (as in thesis)")
E(T, "v4r_bs32", V4R, {"training.batch_size": 32}, SEEDS_FULL, 1, "Tuning: batch size 32")
E(T, "v4r_dropout03", V4R, {"model.dropout": 0.3}, SEEDS_FULL, 1, "Tuning: dropout 0.3")
# Thesis limitations / AUTHOR-CHECKs that need new runs.
E(T, "v4r_night_zero", V4R, {"data.holdout_night_channel": "zero"}, SEEDS_FULL, 1,
  "Lim. 4: night channel blanked at held-out stations (fixes the v4 leak)")
E(T, "v4r_total_only", V4R, {"data.mask_channels": "total_only"}, SEEDS_FULL, 1,
  "Lim. 5: night channel ablated alone (keeps the total-missing channel)")
E(T, "v4r_night_only", V4R, {"data.mask_channels": "night_only"}, SEEDS_FULL, 2,
  "Lim. 5 complement: night channel only (total channel ablated)")
E(T, "v4r_identity_graph", V4R, {"model.graph_mode": "identity"}, SEEDS_FULL, 1,
  "Lim. 7: graph-ablated RIoT (identity adjacency, all else fixed)")
E(T, "v4r_lstm_rich", V4R, {**LSTM_V4, **LSTM_RICH}, SEEDS_FULL, 1,
  "Lim. 7: less-confounded LSTM (mask channels + static inputs, NB head)")
E(T, "v4r_no_masks_no_impute", V4R, {"data.ablate_masks": True, "loss.w_impute": 0.0}, SEEDS_FULL, 1,
  "Lim. 9: masks x imputation 2x2 factorial (completes full / no_masks / no_impute)")
E(T, "v4r_bs128_steps4x", V4R, {"training.epochs": 400, "training.early_stop_patience": 40}, SEEDS_FULL, 2,
  "bs32 mechanism: bs128 given the bs32 update budget (4x epochs / patience)")
E(T, "v4r_bs32_lr_div4", V4R, {"training.batch_size": 32, "training.lr": 0.000075}, SEEDS_FULL, 2,
  "bs32 mechanism: bs32 with lr/4 (bs128 lr/B noise scale, 4x steps)")
E(T, "v4r_bs128_peak_weighted", V4R, {"loss.magnitude_weight_alpha": 0.5}, SEEDS_FULL, 2,
  "bs32 mechanism: bs128 with explicit high-count up-weighting (rare-event account)")

# ---- p1_hygiene: R1 staircase from the v4 replica -----------------------------------
P = "p1_hygiene"
E(P, "e1_1_obs_forecast", V4R, {"loss.target_mask": "observed"}, SEEDS_FULL, 1,
  "E1.1 forecast loss on observed targets only")
E(P, "e1_2_obs_impute", V4R, {"loss.target_mask": "observed", "loss.impute_positions": "masked_observed"},
  SEEDS_FULL, 1, "E1.2 + imputation on masked & observed slots")
E(P, "e1_3_obs_kriging", V4R, {k: v for k, v in HYGIENE.items() if k != "selection.metric"}, SEEDS_FULL, 1,
  "E1.3 + kriging on observed slots (full hygiene)")
E(P, "e1_4_selection", V4R, HYGIENE, SEEDS_FULL, 1, "E1.4 + checkpoint on obs NLL + SO@12")
E(P, "e1_5_nb", V4R, {**HYGIENE, "loss.forecast_likelihood": "nb"}, SEEDS_FULL, 1, "E1.5 ZINB -> NB")
E(P, "e1_6_complete_weather", P2B, {}, SEEDS_FULL, 1, "E1.6 + weather filled at missing slots (removes leak) = P2 base")

# ---- p2_masking: R2 screens on P2B ----------------------------------------------------
P = "p2_masking"
E(P, "e2_1_unify_iid", P2B, {"data.unify_missing_encoding": True}, SEEDS_SCREEN, 2,
  "E2.1 unified missing encoding, v4 iid mask")
E(P, "e2_1b_iid_traffic", P2B, {**STRUCTURED, "data.mask_strategy": "iid"}, SEEDS_SCREEN, 2,
  "E2.1b iid over observed traffic only + unified (strategy control for E2.2-2.5)")
E(P, "e2_2_block", P2B, {**STRUCTURED, "data.mask_strategy": "block"}, SEEDS_SCREEN, 2,
  "E2.2 block gaps, empirical lengths")
E(P, "e2_2b_block_geometric", P2B, {**STRUCTURED, "data.mask_strategy": "block",
                                    "data.mask_block_len_dist": "geometric"}, SEEDS_SCREEN, 2,
  "E2.2b block gaps, geometric lengths (mean 8)")
E(P, "e2_3_transplant", P2B, {**STRUCTURED, "data.mask_strategy": "transplant"}, SEEDS_SCREEN, 2,
  "E2.3 transplanted real missingness patterns")
E(P, "e2_4_tail", P2B, {**STRUCTURED, "data.mask_strategy": "tail"}, SEEDS_SCREEN, 2,
  "E2.4 tail gaps (forecast from inside an outage)")
E(P, "e2_5_mixed_015", P2B, {**STRUCTURED, "data.mask_strategy": "mixed"}, SEEDS_SCREEN, 2,
  "E2.5/E2.6 mixed strategy, mask ratio 0.15")
E(P, "e2_7_mixed_030", P3B, {}, SEEDS_FULL, 1, "E2.7 mixed, mask ratio 0.30 = P3 base")
E(P, "e2_8_mixed_050", P2B, {**STRUCTURED, "data.mask_strategy": "mixed", "data.mask_prob": 0.50},
  SEEDS_SCREEN, 2, "E2.8 mixed, mask ratio 0.50")

# ---- p3_inputs: R3 / R7 one-factor screens on P3B, plus the full stack ------------------
P = "p3_inputs"
E(P, "e3_1_calendar", P3B, {"data.calendar_features": True}, SEEDS_SCREEN, 2, "E3.1 + calendar features")
E(P, "e3_2_log1p", P3B, {"data.log1p_inputs": True}, SEEDS_SCREEN, 2, "E3.2 + log1p traffic inputs")
for kind, tag in (("transductive", "a"), ("inductive_neighbour", "b"), ("inductive_static", "c"), ("inductive_zero", "d")):
    E(P, f"e3_3{tag}_profile_{kind}", P3B, {"data.profile_prior": kind}, SEEDS_SCREEN, 2,
      f"E3.3{tag} + seasonal-profile prior ({kind} at held-out stations)")
E(P, "e3_4_runlength", P3B, {"data.gap_runlength_feature": True}, SEEDS_SCREEN, 2, "E3.4 + gap run-length")
E(P, "e3_5_sensor_emb", P3B, {"model.sensor_embedding_dim": 8, "model.sensor_embedding_dropout": 0.2},
  SEEDS_SCREEN, 2, "E3.5 + sensor embedding (d=8, dropout 0.2)")
E(P, "e3_6_seq48", P3B, SEQ48, SEEDS_SCREEN, 2, "E3.6 seq_len 48 (bs64 x2 accumulation)")
E(P, "e3_7_seq96", P3B, SEQ96, SEEDS_SCREEN, 2, "E3.7 seq_len 96 (bs32 x4 accumulation)")
E(P, "e3_7b_lags", P3B, {"data.lag_features": [96, 672]}, SEEDS_SCREEN, 2,
  "E3.7b seq_len 12 + same-slot-yesterday / last-week lag channels")
E(P, "e3_9_key_padding", P3B, {"model.temporal_key_padding": True}, SEEDS_SCREEN, 2,
  "R7 missing-aware temporal attention")
E(P, "e3_10_gat_drop_missing", P3B, {"model.gat_drop_missing_sources": True}, SEEDS_SCREEN, 2,
  "R7 GAT ignores messages from missing sources")
E(P, "e3_stack", P4B, {}, SEEDS_FULL, 1, "R3 stack (E3.1-E3.5) = P4 base / E4.0 control")
E(P, "e3_stack_seq48", P4B, SEQ48, SEEDS_SCREEN, 2, "R3 stack with seq_len 48")

# ---- p4_likelihood: R4 on P4B --------------------------------------------------------
P = "p4_likelihood"
E(P, "e4_1_ztnb", P4B, {"loss.forecast_likelihood": "ztnb"}, SEEDS_FULL, 1, "E4.1 zero-truncated NB")
E(P, "e4_2_mixture_latent", P4B, {"loss.forecast_likelihood": "outage_mixture"}, SEEDS_FULL, 1,
  "E4.2 outage mixture, outage head reads the latent")
E(P, "e4_3_mixture_missingness", P4B, {"loss.forecast_likelihood": "outage_mixture",
                                       "model.outage_head_input": "missingness"}, SEEDS_FULL, 1,
  "E4.3 outage head reads missingness features only")
E(P, "e4_3b_mixture_detached", P4B, {"loss.forecast_likelihood": "outage_mixture",
                                     "model.outage_head_input": "latent_detached"}, SEEDS_SCREEN, 2,
  "E4.3b outage head reads a stop-gradient latent")
E(P, "e4_4_anchored", P5B, {}, SEEDS_FULL, 1, "E4.4 + supervised BCE on artificial outages = P5 base")
E(P, "e4_5_prior", P5B, {"loss.outage_prior_weight": 0.1}, SEEDS_FULL, 2, "E4.5 + gap-length outage prior 0.1")
E(P, "e4_6_wmiss025", P5B, {"loss.w_missing_lik": 0.25}, SEEDS_FULL, 2, "E4.6 missing-slot likelihood weight 0.25")
E(P, "e4_7_mixture_all_heads", P5B, {"loss.impute_likelihood": "outage_mixture"}, SEEDS_FULL, 1,
  "E4.7 outage mixture on imputation and kriging heads too")

# ---- p5_magnitude: R5 on P5B ---------------------------------------------------------
P = "p5_magnitude"
E(P, "e5_1_alpha05", P5B, {"loss.magnitude_weight_alpha": 0.5}, SEEDS_SCREEN, 2, "E5.1 magnitude weight alpha 0.5")
E(P, "e5_2_alpha10", P5B, {"loss.magnitude_weight_alpha": 1.0}, SEEDS_SCREEN, 2, "E5.2 magnitude weight alpha 1.0")
E(P, "e5_3a_underpred01", P5B, {"loss.w_underpred": 0.1}, SEEDS_SCREEN, 2, "E5.3a under-prediction pinball tau .75, w 0.1")
E(P, "e5_3b_underpred05", P5B, {"loss.w_underpred": 0.5}, SEEDS_SCREEN, 2, "E5.3b under-prediction pinball tau .75, w 0.5")
E(P, "e5_4_quantiles", P5B, {"model.quantiles": [0.5, 0.9]}, SEEDS_SCREEN, 2, "E5.4 q50 / q90 heads")
E(P, "e5_5_huber", P5B, {"loss.impute_huber_mix": 0.5}, SEEDS_SCREEN, 2, "E5.5 imputation 0.5 MSLE + 0.5 Huber")
E(P, "e5_6_low_tol", P5B, {"loss.low_end_tolerance": True}, SEEDS_SCREEN, 2, "E5.6 low-end tolerance band")
E(P, "e5_8_full_plus_underpred", V5F, {"loss.w_underpred": 0.1}, SEEDS_SCREEN, 2,
  "E5.8 full v5 + under-prediction pinball (does it add to quantiles?)")

# ---- p6_mtl: R6 on full v5 ------------------------------------------------------------
P = "p6_mtl"
E(P, "e6_1_uncertainty", V5F, {"loss.mtl_balancing": "uncertainty"}, SEEDS_FULL, 2, "E6.1 uncertainty weighting")
E(P, "e6_2_pcgrad", V5F, {"loss.mtl_balancing": "pcgrad"}, SEEDS_FULL, 2, "E6.2 PCGrad")
E(P, "e6_3_impute_pretrain", V5F, {"loss.impute_pretrain_epochs": 10}, SEEDS_FULL, 2,
  "E6.3 10 epochs imputation pre-training, then joint")
E(P, "e6_4_detached_impute", V5F, {"loss.detach_impute_latent": True}, SEEDS_FULL, 2,
  "E6.4 imputation head on a detached latent")
E(P, "e6_5_v4_impute_aux", V5F, {"loss.impute_positions": "masked_observed_plus_aux",
                                 "loss.impute_aux_weight": 0.1}, SEEDS_SCREEN, 2,
  "E6.5 + small auxiliary reconstruct-all-observed term")

# ---- p7_v5_suite: thesis ablation suite on full v5 (report Phase 7 + ch7 limitations) ---
P = "p7_v5_suite"
E(P, "v5_full", V5F, {}, SEEDS_FULL, 1, "Full v5 (E5.7)")
E(P, "v5_no_impute", V5F, {"loss.w_impute": 0.0}, SEEDS_FULL, 1,
  "No imputation loss (artificial gaps still drawn: they anchor the forecast outage head)")
E(P, "v5_no_kriging", V5F, {"loss.w_kriging": 0.0}, SEEDS_FULL, 1, "No kriging loss")
E(P, "v5_no_masks", V5F, {"data.mask_channels": "none"}, SEEDS_FULL, 1,
  "No mask channels (unified encoding: value still NaN->0; loss mask un-ablated)")
E(P, "v5_no_masks_no_impute", V5F, {"data.mask_channels": "none", "loss.w_impute": 0.0}, SEEDS_FULL, 1,
  "masks x imputation factorial cell")
E(P, "v5_total_only", V5F, {"data.mask_channels": "total_only"}, SEEDS_FULL, 1, "Night channel ablated alone")
E(P, "v5_night_only", V5F, {"data.mask_channels": "night_only"}, SEEDS_FULL, 2, "Night channel only")
E(P, "v5_no_pe", V5F, {"model.use_positional_encoding": False}, SEEDS_FULL, 1,
  "No positional encoding (calendar may make PE redundant)")
E(P, "v5_identity_graph", V5F, {"model.graph_mode": "identity"}, SEEDS_FULL, 1, "Graph-ablated v5")
E(P, "v5_bs32", V5F, {"training.batch_size": 32}, SEEDS_FULL, 1, "Batch size 32")
E(P, "v5_bs32_lr_div4", V5F, {"training.batch_size": 32, "training.lr": 0.000075}, SEEDS_FULL, 2,
  "Batch size 32, lr/4")
E(P, "v5_dropout03", V5F, {"model.dropout": 0.3}, SEEDS_FULL, 1, "Dropout 0.3")
E(P, "v5_no_outage_head", V5F, {"loss.forecast_likelihood": "nb", "model.outage_head_input": "latent",
                                "loss.w_artificial_outage": 0.0}, SEEDS_FULL, 1, "NB instead of outage mixture (isolates R4)")
E(P, "v5_iid_masks", V5F, {"data.mask_strategy": "iid", "data.mask_prob": 0.15}, SEEDS_FULL, 1,
  "iid masks over observed traffic (isolates R2 structure)")
E(P, "v5_no_calendar_profile", V5F, {"data.calendar_features": False, "data.profile_prior": "none"},
  SEEDS_FULL, 1, "No calendar or profile (isolates R3 level information)")
_LSTM_V5 = {k: v for k, v in LSTM_V4.items() if k != "loss.w_impute"}
E(P, "v5_lstm", V5F, {**_LSTM_V5, **LSTM_RICH, "model.quantiles": [], "model.outage_head_input": "latent",
                      "loss.forecast_likelihood": "nb", "loss.w_artificial_outage": 0.0,
                      "model.sensor_embedding_dim": 0, "model.sensor_embedding_dropout": 0.0,
                      "model.temporal_key_padding": False, "model.gat_drop_missing_sources": False},
  SEEDS_FULL, 1, "LSTM, v5 hygiene + inputs (masks, covariates, static), NB head, imputation head")
E(P, "v5_lstm_plain", V5F, {**LSTM_V4, "model.baseline_head": "nb", "model.quantiles": [],
                            "model.outage_head_input": "latent", "loss.forecast_likelihood": "nb",
                            "loss.w_artificial_outage": 0.0, "model.sensor_embedding_dim": 0,
                            "model.sensor_embedding_dropout": 0.0, "model.temporal_key_padding": False,
                            "model.gat_drop_missing_sources": False}, SEEDS_FULL, 2,
  "LSTM, dynamic channels only, NB head (input-access control for v5_lstm)")
# Non-learned floors (deterministic: one seed is exact).
_NAIVE = {"model_type": "naive", "loss.w_impute": 0.0, "loss.w_kriging": 0.0, "loss.w_artificial_outage": 0.0,
          "loss.forecast_likelihood": "nb", "model.outage_head_input": "latent", "model.quantiles": [],
          "model.sensor_embedding_dim": 0, "model.sensor_embedding_dropout": 0.0,
          "model.temporal_key_padding": False, "model.gat_drop_missing_sources": False,
          "data.spatial_holdout_prob": 0.0}
E(P, "naive_profile", V5F, {**_NAIVE, "model.naive_kind": "profile"}, SEEDS_SCREEN, 1,
  "Seasonal-profile median floor (E0.3)")
E(P, "naive_last_obs", V5F, {**_NAIVE, "model.naive_kind": "last_obs"}, SEEDS_SCREEN, 1,
  "Last-observed-value floor (E0.3)")

# ---- loo: leave-one-out around full v5 (report E3.8, attribution) ----------------------
P = "loo"
for name, ov, desc in (
    ("loo_no_calendar", {"data.calendar_features": False}, "- calendar features"),
    ("loo_no_log1p", {"data.log1p_inputs": False}, "- log1p inputs"),
    ("loo_no_profile", {"data.profile_prior": "none"}, "- seasonal profile"),
    ("loo_no_runlength", {"data.gap_runlength_feature": False}, "- gap run-length"),
    ("loo_no_sensor_emb", {"model.sensor_embedding_dim": 0, "model.sensor_embedding_dropout": 0.0}, "- sensor embedding"),
    ("loo_no_key_padding", {"model.temporal_key_padding": False}, "- missing-aware attention"),
    ("loo_no_gat_drop", {"model.gat_drop_missing_sources": False}, "- GAT missing-source dropping"),
    ("loo_no_complete_weather", {"data.complete_weather": False}, "- weather completion (weather leaks missingness)"),
    ("loo_night_leak", {"data.holdout_night_channel": "leak"}, "- consistent night channel (v4 leak)"),
    ("loo_mask_015", {"data.mask_prob": 0.15}, "mask ratio 0.30 -> 0.15"),
    ("loo_no_unify", {"data.unify_missing_encoding": False, "data.holdout_night_channel": "zero"},
     "- unified missing encoding (night channel zeroed at held-out stations)"),
    ("loo_unanchored", {"loss.w_artificial_outage": 0.0}, "- artificial-outage BCE anchor"),
    ("loo_no_magnitude", {"loss.magnitude_weight_alpha": 0.0}, "- magnitude weighting"),
    ("loo_no_quantiles", {"model.quantiles": []}, "- quantile heads"),
    ("loo_no_huber", {"loss.impute_huber_mix": 0.0}, "- Huber imputation mix"),
    ("loo_val_loss_selection", {"selection.metric": "val_loss_total"}, "- SO-aware checkpoint selection"),
):
    E(P, name, V5F, ov, SEEDS_SCREEN, 2, f"LOO: full v5 {desc}")


# ---------------------------------------------------------------------------
# Relative cost (1.0 = one v4 full run) — used to schedule long jobs first within a
# tier and to estimate the GPU budget. Approximate: calibrate against the first runs.
# ---------------------------------------------------------------------------
def relative_cost(cfg: dict) -> float:
    if cfg["model_type"] == "naive":
        return 0.05
    seq = cfg["data"]["seq_len"] / 12.0
    cost = seq ** 1.1  # GAT linear in T, attention quadratic but small
    cost *= cfg["training"]["epochs"] / 100.0
    if cfg["model_type"] in ("lstm", "gru"):
        cost *= 0.25
    if cfg["training"]["batch_size"] * cfg["training"]["grad_accum_steps"] < 128:
        cost *= 1.6  # more, smaller steps; patience in epochs means a longer run
    if cfg["loss"]["mtl_balancing"] == "pcgrad":
        cost *= 2.5
    if cfg["loss"]["mtl_balancing"] == "uncertainty" or cfg["data"]["profile_prior"] != "none":
        cost *= 1.05
    return round(cost, 2)


def build_jobs():
    names = [e["name"] for e in EXPERIMENTS]
    assert len(names) == len(set(names)), "duplicate experiment names"
    jobs = []
    for e in EXPERIMENTS:
        base_cfg = derive(e["base"], e["overrides"])
        for seed in e["seeds"]:
            cfg = derive(base_cfg, {"training.random_seed": seed})
            run_name = f"{e['name']}-s{seed}"
            cfg["wandb"]["collection"] = f"riot-v5-{e['group']}-{e['name']}"
            cfg["wandb"]["run_name"] = run_name
            # Extra seeds of a tier-1 experiment are scheduled after every tier-1 seed-42 run.
            tier = e["tier"] if seed == e["seeds"][0] else 3
            path = OUT_DIR / e["group"] / f"{e['name']}__s{seed}.yaml"
            jobs.append(dict(
                name=run_name, experiment=e["name"], group=e["group"], seed=seed, tier=tier,
                desc=e["desc"], config=str(path.relative_to(PROJECT_ROOT)), cost=relative_cost(cfg),
                cfg=cfg,
            ))
    return jobs


HEADER = """# ============================================================================
# RIoT pipeline v5 — {group} / {name} (seed {seed})
# {desc}
# GENERATED by scripts/riotv5/generate_v5_configs.py — edit the generator, not this file.
# ============================================================================
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="resolve_config every generated file")
    args = ap.parse_args()

    jobs = build_jobs()
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)
    for j in jobs:
        path = PROJECT_ROOT / j["config"]
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(HEADER.format(group=j["group"], name=j["experiment"], seed=j["seed"], desc=j["desc"]))
            yaml.safe_dump(j["cfg"], f, sort_keys=False, default_flow_style=None)
        if args.check:
            resolve_config(j["cfg"])

    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "wandb_project": WANDB_PROJECT,
        "jobs": [{k: v for k, v in j.items() if k != "cfg"} for j in jobs],
    }
    with open(OUT_DIR / "suite_manifest.yaml", "w") as f:
        f.write("# GENERATED by scripts/riotv5/generate_v5_configs.py\n")
        yaml.safe_dump(manifest, f, sort_keys=False)

    by_group = {}
    for j in jobs:
        g = by_group.setdefault(j["group"], [0, 0.0])
        g[0] += 1
        g[1] += j["cost"]
    print(f"Wrote {len(jobs)} configs ({len(EXPERIMENTS)} experiments) to {OUT_DIR.relative_to(PROJECT_ROOT)}")
    for g, (n, c) in by_group.items():
        print(f"  {g:<14} {n:>4} runs  {c:7.1f} cost units")
    total = sum(j["cost"] for j in jobs)
    print(f"  {'total':<14} {len(jobs):>4} runs  {total:7.1f} cost units "
          f"(~{total / 4:.1f} v4-full-run durations of wall time on 4 GPUs)")
    for t in (1, 2, 3):
        sel = [j for j in jobs if j["tier"] == t]
        print(f"  tier {t}: {len(sel):>4} runs  {sum(j['cost'] for j in sel):7.1f} cost units")
    if args.check:
        print("All configs resolve and validate.")


if __name__ == "__main__":
    main()
