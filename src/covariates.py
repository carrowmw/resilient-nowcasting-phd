"""
Device-side covariate builder (v5 R3).

Builds per-(b, n, step) covariate channels for the input window AND the
forecast target step from tables precomputed in RealIoTDataset (train-slice
fitted, F-C02 pattern). Output: cov [B, N, T + 1, C]. Columns [0, T) are
concatenated to the encoder input; column T feeds the forecast / outage heads
("what is known about the target step at forecast time").

Channel groups (each enabled by a config key):
  calendar_features      6  sin/cos hour, sin/cos day-of-week, weekend, bank holiday
  profile_prior          2  seasonal slot-of-week median (traffic-scaled) + own-profile flag
  gap_runlength_feature  2  log-scaled steps since last observation + node-blanked flag
  lag_features[L]        2  value at t - L (traffic-scaled) + observed flag, per lag
  weather_extreme_features 4  v6: wet / dry (humidity >= train p90 / <= p10), dull day
                              (daytime solar <= daytime p25), bright (solar >= daytime p90).
                              Thresholds from the training slice. Column T uses the weather
                              AT the target step: a perfect-weather-forecast assumption.

Blanked nodes (fixed inductive holdout and per-batch kriging re-rolls) never
see their own history: run-length = cap, lags blanked, and — for the inductive
profile modes — a substitute profile built WITHOUT the node's own data
(k-NN over static features among non-inductive nodes, or a leave-one-out ridge
regression from static features, or zero + flag). `transductive` gives every
node its own profile: an oracle upper bound for the inductive setting.
"""
import logging
from typing import Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

CALENDAR_DIM = 6
WEATHER_EXTREME_DIM = 4
RIDGE_LAMBDA = 1.0


def _weather_extremes(dataset) -> torch.Tensor:
    """[num_times, N, 4] float indicators from the (complete) humidity / solar channels.
    Thresholds are train-slice quantiles of the scaled channels (a monotone transform of
    the raw readings, so the indicators are the same as on raw values)."""
    weather = dataset.dynamic_data[:, :, 1:3].numpy()                         # [T, N, 2] scaled
    night = dataset.is_night.numpy()                                          # [T]
    t_end = dataset.train_t_end
    hum, sol = weather[..., 0], weather[..., 1]
    hum_hi, hum_lo = np.quantile(hum[:t_end], [0.9, 0.1])
    sol_day = sol[:t_end][~night[:t_end]]
    assert sol_day.size > 0, "CRITICAL: no daytime steps in the training slice for the solar thresholds"
    dull, bright = np.quantile(sol_day, [0.25, 0.9])
    day = ~night[:, None]
    ext = np.stack([hum >= hum_hi, hum <= hum_lo, day & (sol <= dull), sol >= bright], axis=-1)
    logger.info(
        "weather extremes: rates wet %.3f, dry %.3f, dull-day %.3f, bright %.3f",
        *[float(ext[:t_end, :, k].mean()) for k in range(WEATHER_EXTREME_DIM)],
    )
    return torch.from_numpy(ext.astype(np.float32))


def _substitute_profiles(
    profile_log1p: np.ndarray,      # [S, N]
    static: np.ndarray,             # [N, F] (already standardised)
    donor_mask: np.ndarray,         # [N] bool — nodes whose profile may be used
    mode: str,
    knn_k: int,
) -> np.ndarray:
    """Per-node profile estimate that never uses the node's own data."""
    num_slots, num_nodes = profile_log1p.shape
    sub = np.zeros_like(profile_log1p)
    if mode == "inductive_zero":
        return sub
    for n in range(num_nodes):
        donors = donor_mask.copy()
        donors[n] = False
        idx = np.where(donors)[0]
        assert len(idx) > 0, "CRITICAL: no donor nodes for the profile substitute"
        if mode == "inductive_neighbour":
            dist = np.linalg.norm(static[idx] - static[n][None, :], axis=1)
            nearest = idx[np.argsort(dist)[: min(knn_k, len(idx))]]
            sub[:, n] = profile_log1p[:, nearest].mean(axis=1)
        else:
            assert mode == "inductive_static", f"CRITICAL: unknown profile mode {mode!r}"
            x = np.concatenate([static[idx], np.ones((len(idx), 1))], axis=1)       # [D, F+1]
            y = profile_log1p[:, idx].T                                              # [D, S]
            reg = RIDGE_LAMBDA * np.eye(x.shape[1])
            reg[-1, -1] = 0.0  # no penalty on the intercept
            w = np.linalg.solve(x.T @ x + reg, x.T @ y)                              # [F+1, S]
            sub[:, n] = np.concatenate([static[n], [1.0]]) @ w
    return sub


class CovariateBuilder:
    def __init__(self, dataset, data_cfg: dict, inductive_mask: Optional[torch.Tensor], device: torch.device):
        self.device = device
        self.seq_len = data_cfg["seq_len"]
        self.calendar_on = data_cfg["calendar_features"]
        self.profile_mode = data_cfg["profile_prior"]
        self.runlength_on = data_cfg["gap_runlength_feature"]
        self.lags = list(data_cfg["lag_features"])
        self.weather_on = data_cfg["weather_extreme_features"]
        self.cap = int(data_cfg["runlength_cap"])
        self.num_nodes = dataset.num_nodes

        needs_grid = self.runlength_on or bool(self.lags) or self.profile_mode != "none"
        if needs_grid:
            assert dataset.regular_grid, \
                "CRITICAL: run-length / lag / slot-of-week covariates require a regular time grid"

        self.calendar = dataset.calendar.to(device)
        self.slot_of_week = dataset.slot_of_week.to(device)
        self.is_night = dataset.is_night.to(device)
        self.run_length = dataset.run_length.to(device)
        self.observed = dataset.observed.to(device)
        self.traffic_scaled = dataset.dynamic_data[:, :, 0].to(device)
        self.weather_extremes = _weather_extremes(dataset).to(device) if self.weather_on else None

        # Seasonal profile in the traffic channel's scaled domain (directly comparable
        # to the traffic input), plus the substitute table for blanked nodes.
        prof = dataset.profile_log1p.numpy().astype(np.float64)
        donor = (~inductive_mask.cpu().numpy()) if inductive_mask is not None else np.ones(self.num_nodes, bool)
        if self.profile_mode in ("none", "transductive"):
            sub = prof.copy()
        else:
            sub = _substitute_profiles(
                prof, dataset.static_features.numpy().astype(np.float64), donor,
                self.profile_mode, data_cfg["profile_knn_k"],
            )
        self.profile_log1p = torch.tensor(prof, dtype=torch.float32, device=device)
        self.profile_sub_log1p = torch.tensor(sub, dtype=torch.float32, device=device)
        self.profile_scaled = self._to_traffic_scaled(prof, dataset)
        self.profile_sub_scaled = (
            torch.zeros_like(self.profile_scaled) if self.profile_mode == "inductive_zero"
            else self._to_traffic_scaled(sub, dataset)
        )

        self.dim = (
            (CALENDAR_DIM if self.calendar_on else 0)
            + (2 if self.profile_mode != "none" else 0)
            + (2 if self.runlength_on else 0)
            + 2 * len(self.lags)
            + (WEATHER_EXTREME_DIM if self.weather_on else 0)
        )
        logger.info(
            f"CovariateBuilder: dim={self.dim} (calendar={self.calendar_on}, profile={self.profile_mode}, "
            f"runlength={self.runlength_on}, lags={self.lags}, weather_extremes={self.weather_on}, cap={self.cap})"
        )

    def _to_traffic_scaled(self, prof_log1p: np.ndarray, dataset) -> torch.Tensor:
        domain = prof_log1p if dataset.log1p_inputs else np.expm1(prof_log1p)
        return torch.tensor(
            (domain - dataset.traffic_mean) / dataset.traffic_std, dtype=torch.float32, device=self.device
        )

    def time_grid(self, time_idx: torch.Tensor) -> torch.Tensor:
        """[B, T + 1] absolute time indices of the window + target."""
        return time_idx.view(-1, 1) + torch.arange(self.seq_len + 1, device=self.device).view(1, -1)

    def night_flags(self, time_idx: torch.Tensor) -> torch.Tensor:
        return self.is_night[self.time_grid(time_idx)]                          # [B, T + 1]

    def profile_raw(self, time_idx: torch.Tensor, blank_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """[B, N, T + 1] profile in raw counts, substitute for blanked nodes (naive baseline)."""
        sow = self.slot_of_week[self.time_grid(time_idx)]                      # [B, T + 1]
        own = self.profile_log1p[sow].permute(0, 2, 1)                          # [B, N, T + 1]
        if blank_mask is not None:
            sub = self.profile_sub_log1p[sow].permute(0, 2, 1)
            own = torch.where(blank_mask.view(1, -1, 1), sub, own)
        return torch.expm1(own).clamp_min(0.0)

    def run_lengths(
        self,
        time_idx: torch.Tensor,
        input_missing: torch.Tensor,
        outage_start: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Causal steps-since-last-observation for each input step, [B, N, T] (raw, uncapped)."""
        t0 = time_idx.view(-1, 1)                                               # [B, 1]
        prev_idx = (t0 - 1).clamp_min(0)
        rl_prev = torch.where(t0 > 0, self.run_length[prev_idx.view(-1)], torch.zeros(1, dtype=torch.long, device=self.device))
        if outage_start is not None:
            # Synthetic outage that began before the window: steps since its start
            # plus whatever real run preceded it (the outage hides everything between).
            began_before = outage_start < t0
            pre = (outage_start - 1).clamp(0, self.run_length.shape[0] - 1)
            rl_before = self.run_length[pre, torch.arange(self.num_nodes, device=self.device)]  # [B, N]
            rl_before = torch.where(outage_start > 0, rl_before, torch.zeros_like(rl_before))
            rl_prev = torch.where(began_before, (t0 - outage_start) + rl_before, rl_prev)
        rls = []
        rl = rl_prev
        for t in range(self.seq_len):
            rl = torch.where(input_missing[:, :, t], rl + 1, torch.zeros_like(rl))
            rls.append(rl)
        return torch.stack(rls, dim=-1)

    def build(
        self,
        time_idx: torch.Tensor,
        input_missing: torch.Tensor,
        blank_mask: Optional[torch.Tensor],
        outage_start: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Args:
            time_idx:      [B] window start index.
            input_missing: [B, N, T] bool — effective missingness the model sees
                           (real gaps | artificial / synthetic gaps | blanked nodes).
            blank_mask:    [N] bool or None — nodes whose dynamic input is blanked.
            outage_start:  [B, N] long or None — absolute start of a synthetic
                           outage (evaluation passes); rows without one hold a
                           value > the target index.
        Returns:
            cov:            [B, N, T + 1, C] or None when C == 0
            rl_target_raw:  [B, N] long — run-length at the target step
                            (-1 for blanked nodes: no outage heuristic applies)
        """
        B, N, T = input_missing.shape
        grid = self.time_grid(time_idx)                                         # [B, T + 1]
        blank = blank_mask.view(1, -1, 1) if blank_mask is not None else None

        rl = self.run_lengths(time_idx, input_missing, outage_start)            # [B, N, T]
        rl_target = rl[:, :, -1] + 1
        if blank is not None:
            rl_target_raw = torch.where(blank_mask.view(1, -1), torch.full_like(rl_target, -1), rl_target)
        else:
            rl_target_raw = rl_target

        if self.dim == 0:
            return None, rl_target_raw

        chans = []
        if self.calendar_on:
            chans.append(self.calendar[grid].unsqueeze(1).expand(B, N, T + 1, CALENDAR_DIM))
        if self.profile_mode != "none":
            sow = self.slot_of_week[grid]
            own = self.profile_scaled[sow].permute(0, 2, 1)                     # [B, N, T + 1]
            flag = torch.ones_like(own)
            if blank is not None and self.profile_mode != "transductive":
                sub = self.profile_sub_scaled[sow].permute(0, 2, 1)
                own = torch.where(blank, sub, own)
                flag = torch.where(blank, torch.zeros_like(flag), flag)
            chans += [own.unsqueeze(-1), flag.unsqueeze(-1)]
        if self.runlength_on:
            full = torch.cat([rl, rl_target.unsqueeze(-1)], dim=-1).clamp_max(self.cap).float()
            blanked = torch.zeros_like(full)
            if blank is not None:
                full = torch.where(blank, torch.full_like(full, float(self.cap)), full)
                blanked = blank.expand_as(full).float()
            feat = torch.log1p(full) / float(np.log1p(self.cap))
            chans += [feat.unsqueeze(-1), blanked.unsqueeze(-1)]
        for lag in self.lags:
            src = grid - lag                                                    # [B, T + 1]
            valid_t = src >= 0
            src_c = src.clamp_min(0)
            val = self.traffic_scaled[src_c].permute(0, 2, 1)                   # [B, N, T + 1]
            obs = self.observed[src_c].permute(0, 2, 1) & valid_t.unsqueeze(1)
            if blank is not None:
                obs = obs & ~blank
            if outage_start is not None:
                obs = obs & ~(src.unsqueeze(1) >= outage_start.unsqueeze(-1))
            chans += [torch.where(obs, val, torch.zeros_like(val)).unsqueeze(-1), obs.float().unsqueeze(-1)]
        if self.weather_on:
            ext = self.weather_extremes[grid].permute(0, 2, 1, 3)                 # [B, N, T + 1, 4]
            if blank is not None:
                # Blanked nodes see no dynamic input of their own, weather included.
                ext = torch.where(blank.unsqueeze(-1), torch.zeros_like(ext), ext)
            chans.append(ext)

        cov = torch.cat(chans, dim=-1)
        assert cov.shape == (B, N, T + 1, self.dim), f"CRITICAL: covariate shape {tuple(cov.shape)}"
        assert not torch.isnan(cov).any(), "CRITICAL: NaN in covariates"
        return cov, rl_target_raw
