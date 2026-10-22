"""
Pipeline versioning and config resolution (RIoT v5).

v5 introduces ~40 new config keys (zero-collapse remediation, see
docs/design_reports/zero_collapse_investigation.md and
docs/design_reports/riot_v5_experiment_design.md). Two kinds of config reach
run_training():

  - pipeline_version: 5  -> every key in V5_SCHEMA must be present. A missing
    key raises KeyError (fail-fast doctrine; no silent defaults).
  - no pipeline_version   -> a v4-era config (configs/train/v4/*, tune.py sweep
    bases). It is upgraded with LEGACY_V4_EQUIVALENTS: the values that make the
    v5 code path compute exactly what v4 computed. This is an explicit, logged
    migration table, not a fallback — the only place v4 semantics are encoded.

The resolved config is what gets saved next to the checkpoint, so every saved
run carries its full v5 key set.
"""
import copy
import logging
from typing import Dict

logger = logging.getLogger(__name__)

PIPELINE_VERSION = 5

# Values under which the v5 code path reproduces v4 behaviour. Keys mirror
# V5_SCHEMA exactly (asserted at import time below).
LEGACY_V4_EQUIVALENTS: Dict[str, Dict] = {
    "data": {
        "inductive_holdout_seed": None,          # None -> training.random_seed (v4 coupling)
        "mask_strategy": "iid",
        "mask_weather": True,                    # v4 iid mask covers every dynamic channel
        "mask_block_len_dist": "empirical",
        "mask_block_mean_len": 8,
        "mask_mix": {"iid": 0.25, "block": 0.25, "transplant": 0.30, "tail": 0.20},
        "artificial_mask_observed_only": False,
        "unify_missing_encoding": False,
        "mask_channels": "both",
        "holdout_night_channel": "leak",
        "calendar_features": False,
        "profile_prior": "none",
        "profile_knn_k": 5,
        "gap_runlength_feature": False,
        "runlength_cap": 672,
        "lag_features": [],
        "log1p_inputs": False,
        "complete_weather": False,               # v4 zero-filled weather at missing-traffic slots
        "ablate_masks": False,                   # v4 key; v5 prefers mask_channels=none
        "train_node_count": None,                # None -> every non-holdout station is in the network
        "train_node_seed": 0,                    # draw order of the nested training-station subsets
        "val_station_count": 0,                  # v6: training stations carved out as inductive validation nodes
        "val_station_seed": 0,                   # draw of the validation stations (independent of training seed)
        "weather_extreme_features": False,       # v6: wet / dry / dull-day / bright indicator covariates
        "val_station_strategy": "random",        # v7: "stratified" = one station per training-level stratum
        "val_station_min_coverage": 0.25,        # v7 stratified: min observed fraction in the validation window
        "weather_extreme_step": "target",        # v7: "origin" = forecast-column indicators from the last input step
    },
    "model": {
        "sensor_embedding_dim": 0,
        "sensor_embedding_dropout": 0.0,
        "temporal_key_padding": False,
        "gat_drop_missing_sources": False,
        "graph_mode": "learned",
        "outage_head_input": "latent",
        "quantiles": [],
        "baseline_inputs": "dynamic",
        "baseline_head": "mse",
        "naive_kind": "none",                    # profile | last_obs for model_type=naive
        "residual_to_profile": False,            # v6: forecast mean = (profile + 1) * exp(head output)
    },
    "loss": {
        "target_mask": "all",
        "impute_positions": "all",
        "impute_aux_weight": 0.0,
        "kriging_observed_only": False,
        "forecast_likelihood": "zinb",
        "impute_likelihood": "msle",
        "w_missing_lik": 1.0,
        "w_artificial_outage": 0.0,
        "outage_prior_weight": 0.0,
        "magnitude_weight_alpha": 0.0,
        "magnitude_weight_y0": 27.0,
        "underpred_tau": 0.75,
        "w_underpred": 0.0,
        "w_quantile": 1.0,
        "impute_huber_mix": 0.0,
        "impute_huber_delta": 20.0,
        "low_end_tolerance": False,
        "mtl_balancing": "none",
        "impute_pretrain_epochs": 0,
        "detach_impute_latent": False,
    },
    "training": {
        "grad_accum_steps": 1,
        "grad_cosine_every": 0,
        "lr_schedule": "none",                   # v6: none | cosine (per-epoch, linear warmup)
        "warmup_epochs": 0,
        "lr_min_frac": 0.0,                      # cosine floor as a fraction of training.lr
        "min_epochs": 0,                         # early stopping cannot trigger before this epoch
    },
    "selection": {
        "metric": "val_loss_total",
        "so_lambda": 1.0,
        "so_length": 12,
    },
    "evaluation": {
        "so_lengths": [1, 4, 12, 48, 96],
        "so_node_frac": 0.25,
        "eval_mask_seed": 1234,
        "diagnostics": True,
        "crps_max_count": 2048,
    },
}

# Allowed values for enumerated keys (validated for every resolved config).
_ENUMS = {
    ("data", "mask_strategy"): ("iid", "block", "transplant", "tail", "mixed"),
    ("data", "mask_block_len_dist"): ("empirical", "geometric"),
    ("data", "mask_channels"): ("both", "total_only", "night_only", "none"),
    ("data", "holdout_night_channel"): ("leak", "zero", "consistent"),
    ("data", "profile_prior"): (
        "none", "transductive", "inductive_neighbour", "inductive_static", "inductive_zero",
    ),
    ("model", "graph_mode"): ("learned", "identity"),
    ("model", "outage_head_input"): ("latent", "latent_detached", "missingness"),
    ("model", "baseline_inputs"): ("dynamic", "dynamic_masks_covariates_static"),
    ("model", "baseline_head"): ("mse", "nb"),
    ("model", "naive_kind"): ("none", "profile", "last_obs"),
    ("loss", "target_mask"): ("all", "observed"),
    ("loss", "impute_positions"): ("all", "masked_observed", "masked_observed_plus_aux"),
    ("loss", "forecast_likelihood"): ("zinb", "nb", "ztnb", "outage_mixture"),
    ("loss", "impute_likelihood"): ("msle", "outage_mixture"),
    ("loss", "mtl_balancing"): ("none", "uncertainty", "pcgrad"),
    ("selection", "metric"): ("val_loss_total", "obs_loss", "obs_loss_plus_so"),
    ("training", "lr_schedule"): ("none", "cosine"),
    ("data", "val_station_strategy"): ("random", "stratified"),
    ("data", "weather_extreme_step"): ("target", "origin"),
}

V5_SCHEMA = {section: tuple(keys) for section, keys in LEGACY_V4_EQUIVALENTS.items()}

# Keys added to the v5 schema after the 209-run suite was generated. Configs saved by
# that suite (and the scaling sweep) lack them; their legacy value is exactly the
# behaviour those runs had.
V5_LATE_KEYS = (
    ("data", "train_node_count"), ("data", "train_node_seed"),
    # v6 stage (docs/experiments/riot_v5/riot_v5_seed_report.md §14.2)
    ("data", "val_station_count"), ("data", "val_station_seed"), ("data", "weather_extreme_features"),
    ("model", "residual_to_profile"),
    ("training", "lr_schedule"), ("training", "warmup_epochs"), ("training", "lr_min_frac"),
    ("training", "min_epochs"),
    # v7 stage (docs/experiments/riot_v6/riot_v6_seed_report.md §9.2)
    ("data", "val_station_strategy"), ("data", "val_station_min_coverage"), ("data", "weather_extreme_step"),
)


def resolve_config(config: dict) -> dict:
    """Return a fully-specified copy of `config` (never mutates the input)."""
    cfg = copy.deepcopy(config)

    if "pipeline_version" in cfg:
        assert cfg["pipeline_version"] == PIPELINE_VERSION, (
            f"CRITICAL: config pipeline_version={cfg['pipeline_version']} but this code is "
            f"pipeline v{PIPELINE_VERSION}"
        )
        for section, key in V5_LATE_KEYS:
            if section in cfg and key not in cfg[section]:
                cfg[section][key] = copy.deepcopy(LEGACY_V4_EQUIVALENTS[section][key])
        missing = [
            f"{section}.{key}"
            for section, keys in V5_SCHEMA.items()
            for key in keys
            if section not in cfg or key not in cfg[section]
        ]
        if missing:
            raise KeyError(f"CRITICAL: v5 config is missing required keys: {missing}")
    else:
        logger.warning(
            "Config has no pipeline_version — treating as a v4 config and applying "
            "LEGACY_V4_EQUIVALENTS (run_config.py). Results reproduce v4 semantics."
        )
        cfg["pipeline_version"] = 4
        for section, defaults in LEGACY_V4_EQUIVALENTS.items():
            cfg.setdefault(section, {})
            for key, value in defaults.items():
                if key not in cfg[section]:
                    cfg[section][key] = copy.deepcopy(value)

    if cfg["data"]["inductive_holdout_seed"] is None:
        assert cfg["pipeline_version"] == 4, \
            "CRITICAL: v5 configs must pin data.inductive_holdout_seed explicitly"
        cfg["data"]["inductive_holdout_seed"] = cfg["training"]["random_seed"]

    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    """Cross-key consistency checks. Invalid combinations crash before data loads."""
    for (section, key), allowed in _ENUMS.items():
        value = cfg[section][key]
        assert value in allowed, f"CRITICAL: {section}.{key}={value!r} not in {allowed}"

    data, model, loss = cfg["data"], cfg["model"], cfg["loss"]

    if loss["forecast_likelihood"] in ("ztnb", "outage_mixture"):
        assert loss["target_mask"] == "observed", (
            f"CRITICAL: forecast_likelihood={loss['forecast_likelihood']} needs "
            "loss.target_mask=observed (it routes observed/missing targets itself)"
        )
    if loss["impute_likelihood"] == "outage_mixture":
        assert loss["impute_positions"] != "all", \
            "CRITICAL: impute_likelihood=outage_mixture needs masked impute_positions"
    if data["mask_strategy"] != "iid":
        assert not data["mask_weather"], \
            "CRITICAL: structured mask strategies mask the traffic channel only (mask_weather=false)"
        assert data["artificial_mask_observed_only"], \
            "CRITICAL: structured mask strategies require artificial_mask_observed_only=true"
    if data["mask_strategy"] == "mixed":
        mix = data["mask_mix"]
        assert set(mix) <= {"iid", "block", "transplant", "tail"} and abs(sum(mix.values()) - 1.0) < 1e-6, \
            f"CRITICAL: data.mask_mix must be a distribution over strategies, got {mix}"
    if loss["w_artificial_outage"] > 0.0:
        assert "outage_mixture" in (loss["forecast_likelihood"], loss["impute_likelihood"]), \
            "CRITICAL: w_artificial_outage only applies to outage_mixture likelihoods"
    if loss["outage_prior_weight"] > 0.0:
        assert loss["forecast_likelihood"] == "outage_mixture", \
            "CRITICAL: outage_prior_weight only applies to forecast_likelihood=outage_mixture"
    if data["holdout_night_channel"] == "consistent":
        assert data["unify_missing_encoding"], \
            "CRITICAL: holdout_night_channel=consistent is defined by the unified encoding"
    assert loss["impute_pretrain_epochs"] < cfg["training"]["epochs"], \
        "CRITICAL: impute_pretrain_epochs must leave epochs for forecasting"
    assert cfg["training"]["grad_accum_steps"] >= 1
    if loss["mtl_balancing"] == "pcgrad":
        assert cfg["training"]["grad_accum_steps"] == 1, \
            "CRITICAL: pcgrad is implemented for grad_accum_steps=1 only"
    for lag in data["lag_features"]:
        assert lag > data["seq_len"], \
            f"CRITICAL: lag feature {lag} must exceed seq_len={data['seq_len']} (else it is in-window)"
    if cfg["model_type"] == "naive":
        assert model["naive_kind"] in ("profile", "last_obs"), \
            f"CRITICAL: model.naive_kind must be profile | last_obs, got {model['naive_kind']!r}"
        assert data["profile_prior"] != "none", \
            "CRITICAL: naive baselines need data.profile_prior to define the inductive profile"
    if data["ablate_masks"]:
        # The dataset zeroes the structural masks; the unified encoding would re-raise
        # them at artificial gaps and mask_channels would ablate them a second way.
        assert not data["unify_missing_encoding"], \
            "CRITICAL: data.ablate_masks=true is incompatible with unify_missing_encoding"
        assert data["mask_channels"] == "both", \
            "CRITICAL: data.ablate_masks=true expects mask_channels=both (use mask_channels=none instead)"
    if cfg["model_type"] != "naive":
        assert model["naive_kind"] == "none", \
            "CRITICAL: model.naive_kind must be 'none' for learned models"
    if model["sensor_embedding_dim"] == 0:
        assert model["sensor_embedding_dropout"] == 0.0
    if data["train_node_count"] is not None:
        # The training-station pool is "every station outside the inductive holdout",
        # so subsampling it is only defined against a fixed inductive split.
        assert cfg["eval_mode"] == "inductive", \
            "CRITICAL: data.train_node_count requires eval_mode=inductive"
        assert isinstance(data["train_node_count"], int) and data["train_node_count"] >= 1, \
            f"CRITICAL: data.train_node_count must be a positive int, got {data['train_node_count']!r}"
    train = cfg["training"]
    assert 0 <= train["min_epochs"] <= train["epochs"], \
        f"CRITICAL: training.min_epochs={train['min_epochs']} outside [0, epochs={train['epochs']}]"
    assert 0.0 <= train["lr_min_frac"] <= 1.0, "CRITICAL: training.lr_min_frac must lie in [0, 1]"
    if train["lr_schedule"] == "none":
        assert train["warmup_epochs"] == 0 and train["lr_min_frac"] == 0.0, \
            "CRITICAL: warmup_epochs / lr_min_frac only apply to lr_schedule=cosine"
    else:
        assert 0 <= train["warmup_epochs"] < train["epochs"], \
            "CRITICAL: training.warmup_epochs must leave cosine epochs"
    if data["val_station_count"] > 0:
        # Validation stations are an inductive split of the training pool: they need the
        # fixed test holdout to be defined, and a learned model to select a checkpoint for.
        assert cfg["eval_mode"] == "inductive", "CRITICAL: data.val_station_count requires eval_mode=inductive"
        assert isinstance(data["val_station_count"], int), "CRITICAL: data.val_station_count must be an int"
        assert cfg["model_type"] != "naive", "CRITICAL: naive baselines have no checkpoint to select"
    if data["val_station_strategy"] != "random":
        assert data["val_station_count"] > 0, "CRITICAL: data.val_station_strategy needs data.val_station_count > 0"
    assert 0.0 <= data["val_station_min_coverage"] <= 1.0, "CRITICAL: data.val_station_min_coverage must lie in [0, 1]"
    if data["weather_extreme_step"] != "target":
        assert data["weather_extreme_features"], \
            "CRITICAL: data.weather_extreme_step only applies with weather_extreme_features=true"
    if data["weather_extreme_features"]:
        assert data["complete_weather"], \
            "CRITICAL: weather_extreme_features needs complete_weather=true (zero-filled weather leaks missingness)"
    if model["residual_to_profile"]:
        assert cfg["model_type"] == "gnn", "CRITICAL: model.residual_to_profile is implemented for the GNN head"
        assert data["profile_prior"] != "none", \
            "CRITICAL: model.residual_to_profile needs data.profile_prior (the anchor)"


assert all(
    set(keys) == set(LEGACY_V4_EQUIVALENTS[section]) for section, keys in V5_SCHEMA.items()
)
