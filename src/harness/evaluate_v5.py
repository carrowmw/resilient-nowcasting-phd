"""
v5 evaluation (zero-collapse report §6 / docs/design_reports/riot_v5_experiment_design.md §4).

Everything is accumulated as on-device sums/counts (one host sync per summary,
not per batch) and reported in RAW COUNTS. Metrics a model cannot produce are
omitted from the summary (never fabricated as 0.0) — the same contract as
harness/evaluate.py.

Metric families
  forecast accuracy   obs_{all,trans,ind}_{rmse,mae}, peak_* (y >= train p90),
                      bias_d0..d9 (by train count decile), low_tol_acc (y <= 2, |e| <= 1),
                      nll_obs (count NLL on observed targets)
  real-gap behaviour  collapse_rate        P(point < 1 | target missing, own profile > 10)
                      real_gap_mass_ratio  sum point / sum own profile over missing targets
                      night_short_gap_zero_rate  P(point < 1 | missing, night, gap <= 4 steps)
  zero/missing heads  zir_{obs_zero,obs_pos,night_missing,day_missing}_{rate,mean},
                      zir_precision, pzero_missing_auc
  imputation/kriging  impute_raw_{rmse,mae} (artificial gaps on observed slots),
                      kriging_raw_{rmse,mae} (holdout nodes, observed slots)
  synthetic outages   so_{L}_{forecast_rmse,forecast_mae,mass_ratio,impute_rmse,collapse,outage_auc}
  diagnostics (test)  crps, pit_b0..b9, q90_coverage, quantile coverage,
                      peak_q90_{rmse,mae,underpred_rate} (predictive q90 as the peak forecast),
                      mask_flip_{dpzero,dpoint}, value_perturb_dpoint,
                      per-node nrmse / inductive correlation vs best transductive peer,
                      inductive pairwise-correlation degeneracy check
"""
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
import torch

from harness.contract import Capabilities, Predictions
from likelihoods import count_logpmf, predictive_pmf_grid

AUC_BINS = 200
PIT_BINS = 10


class _Acc:
    """Device-side sum / count accumulator keyed by metric name."""

    def __init__(self):
        self.s: Dict[str, torch.Tensor] = {}
        self.c: Dict[str, torch.Tensor] = {}

    def add(self, name: str, values: torch.Tensor, mask: torch.Tensor) -> None:
        m = mask.to(torch.float64)
        v = torch.where(mask, values.to(torch.float64), torch.zeros_like(m))
        self.s[name] = self.s.get(name, 0.0) + v.sum()
        self.c[name] = self.c.get(name, 0.0) + m.sum()

    def add_total(self, name: str, total: torch.Tensor) -> None:
        self.s[name] = self.s.get(name, 0.0) + total.to(torch.float64)
        self.c[name] = self.c.get(name, 0.0) + 1.0

    def means(self) -> Dict[str, Optional[float]]:
        out = {}
        for k in self.s:
            c = float(self.c[k])
            out[k] = float(self.s[k]) / c if c > 0 else None
        return out


class _AUC:
    """Histogram ROC-AUC for scores in [0, 1]."""

    def __init__(self, device):
        self.pos = torch.zeros(AUC_BINS, dtype=torch.float64, device=device)
        self.neg = torch.zeros(AUC_BINS, dtype=torch.float64, device=device)

    def add(self, score: torch.Tensor, label: torch.Tensor, valid: torch.Tensor) -> None:
        b = (score.float().clamp(0, 1) * (AUC_BINS - 1)).round().long()
        self.pos += torch.bincount(b[valid & label], minlength=AUC_BINS).double()
        self.neg += torch.bincount(b[valid & ~label], minlength=AUC_BINS).double()

    def value(self) -> Optional[float]:
        P, N = float(self.pos.sum()), float(self.neg.sum())
        if P == 0 or N == 0:
            return None
        neg_below = torch.cumsum(self.neg, 0) - self.neg
        return float((self.pos * (neg_below + 0.5 * self.neg)).sum()) / (P * N)


class V5Evaluator:
    def __init__(
        self,
        dataset,
        covariates,
        caps: Capabilities,
        config: dict,
        eval_holdout_mask: Optional[torch.Tensor],
        scale,
        device: torch.device,
        diagnostics: bool = False,
        store_series: bool = False,
        ignore_mask: Optional[torch.Tensor] = None,
    ):
        """eval_holdout_mask: nodes scored as "ind". ignore_mask (v6): nodes left out of
        every metric — validation stations at test, the test holdout at validation."""
        self.caps = caps
        self.covariates = covariates
        self.scale = scale
        self.device = device
        self.diagnostics = diagnostics
        self.store_series = store_series
        self.crps_max_count = config["evaluation"]["crps_max_count"]
        self.holdout = eval_holdout_mask
        self.ignore = ignore_mask
        if ignore_mask is not None and eval_holdout_mask is not None:
            assert not bool((ignore_mask & eval_holdout_mask).any()), "CRITICAL: ignored nodes overlap the ind set"
        self.seq_len = config["data"]["seq_len"]
        self.count_p90 = dataset.count_p90
        self.count_deciles = dataset.count_deciles.to(device)
        self.gap_length = dataset.gap_length.to(device)
        self.acc = _Acc()
        self.auc = defaultdict(lambda: _AUC(device))
        self.pit = torch.zeros(PIT_BINS, dtype=torch.float64, device=device)
        self.series = defaultdict(list)

    # ------------------------------------------------------------------ #
    def _valid(self, B: int, N: int) -> torch.Tensor:
        """[B, N] nodes that count in metrics (all but ignored ones)."""
        if self.ignore is None:
            return torch.ones(B, N, dtype=torch.bool, device=self.device)
        return (~self.ignore).view(1, N).expand(B, N)

    def _target_index(self, batch) -> torch.Tensor:
        return batch["time_idx"].view(-1, 1) + self.seq_len                     # [B, 1]

    def update(self, preds: Predictions, batch: dict) -> None:
        """Main (no synthetic outage) evaluation pass."""
        a = self.acc
        assert preds.point is not None, "CRITICAL: v5 evaluation needs Predictions.point"
        y = batch["target_forecast"].float()
        B, N = y.shape
        valid = self._valid(B, N)
        obs = batch["target_observed"] & valid
        point = preds.point.float()
        err = point - y
        ind = self.holdout.view(1, N).expand(B, N) if self.holdout is not None else torch.zeros_like(obs)

        for name, m in (("all", obs), ("trans", obs & ~ind), ("ind", obs & ind)):
            a.add(f"obs_{name}_se", err ** 2, m)
            a.add(f"obs_{name}_mae", err.abs(), m)

        peak = obs & (y >= self.count_p90)
        a.add("peak_se", err ** 2, peak)
        a.add("peak_mae", err.abs(), peak)
        a.add("peak_underpred_rate", (point < 0.5 * y).float(), peak)

        decile = torch.bucketize(y, self.count_deciles)                          # 0..9
        for d in range(10):
            a.add(f"bias_d{d}", err, obs & (decile == d))
        low = obs & (y <= 2.0)
        a.add("low_tol_acc", (err.abs() <= 1.0).float(), low)

        if preds.nll_params is not None:
            p = preds.nll_params
            ll = count_logpmf(p["likelihood"], y, p["mu"].float(), p["theta"].float(),
                              None if p["pi"] is None else p["pi"].float())
            a.add("nll_obs", -ll, obs)

        # Real-gap behaviour against the node's OWN seasonal profile.
        own_profile = self.covariates.profile_raw(batch["time_idx"], None)[..., -1]   # [B, N]
        miss = ~batch["target_observed"] & valid
        busy_gap = miss & (own_profile > 10.0)
        a.add("collapse_rate", (point < 1.0).float(), busy_gap)
        a.add("real_gap_point_sum", point, miss)
        a.add("real_gap_profile_sum", own_profile, miss)
        t_idx = self._target_index(batch)
        night = batch["night_flags"][:, -1].view(B, 1).expand(B, N)
        gap_len = self.gap_length[t_idx.expand(B, N), torch.arange(N, device=self.device).view(1, N)]
        a.add("night_short_gap_zero_rate", (point < 1.0).float(), miss & night & (gap_len <= 4))

        if preds.p_zero is not None:
            pz = preds.p_zero.float()
            flag = (pz > 0.5).float()
            strata = {
                "obs_zero": obs & (y == 0), "obs_pos": obs & (y > 0),
                "night_missing": miss & night, "day_missing": miss & ~night,
            }
            for s, m in strata.items():
                a.add(f"zir_{s}_rate", flag, m)
                a.add(f"zir_{s}_mean", pz, m)
            empty = miss | (y == 0)
            a.add("zir_precision", empty.float(), (pz > 0.5) & valid)
            self.auc["pzero_missing"].add(pz, miss, valid)

        if preds.outage_q is not None:
            a.add("outage_q_missing_mean", preds.outage_q.float(), miss)
            a.add("outage_q_observed_mean", preds.outage_q.float(), obs)

        if preds.quantiles is not None:
            qs = preds.quantiles.float()
            for i, tau in enumerate(preds.quantile_levels):
                a.add(f"q{int(round(tau * 100))}_coverage", (y <= qs[..., i]).float(), obs)

        # Raw-count imputation on artificial gaps over observed slots.
        if preds.imputed_raw is not None and batch["art"].any():
            truth = self.scale.to_raw(batch["target_impute"][..., 0].float())
            m = batch["art"] & batch["input_observed"] & valid.unsqueeze(-1)
            e = preds.imputed_raw.float() - truth
            a.add("impute_raw_se", e ** 2, m)
            a.add("impute_raw_mae", e.abs(), m)

        holdout = batch["spatial_holdout_mask"]
        if preds.kriging_pred is not None and holdout is not None and holdout.any():
            pred_raw = self.scale.to_raw(preds.kriging_pred[..., 0].float())
            truth = self.scale.to_raw(batch["target_impute"][:, holdout, :, 0].float())
            m = batch["input_observed"][:, holdout]
            if self.ignore is not None:
                # kriging_pred rows follow the holdout order; drop the ignored ones.
                m = m & ~self.ignore[holdout].view(1, -1, 1)
            e = pred_raw - truth
            a.add("kriging_raw_se", e ** 2, m)
            a.add("kriging_raw_mae", e.abs(), m)

        if self.diagnostics and preds.nll_params is not None:
            self._distributional(preds, y, obs)

        if self.store_series:
            self.series["point"].append(point.cpu())
            self.series["y"].append(y.cpu())
            self.series["obs"].append(obs.cpu())

    def _distributional(self, preds: Predictions, y: torch.Tensor, obs: torch.Tensor) -> None:
        p = preds.nll_params
        K = self.crps_max_count
        sel = obs & (y <= K)
        if not sel.any():
            return
        pmf = predictive_pmf_grid(
            p["likelihood"], p["mu"][sel].float(), p["theta"][sel].float(),
            None if p["pi"] is None else p["pi"][sel].float(), K,
        )                                                                       # [M, K + 1]
        cdf = torch.cumsum(pmf, dim=-1)
        ys = y[sel].long()
        k = torch.arange(K + 1, device=y.device).view(1, -1)
        crps = ((cdf - (k >= ys.view(-1, 1)).float()) ** 2).sum(dim=-1)
        self.acc.add_total("crps_sum", crps.sum())
        self.acc.add_total("crps_n", torch.tensor(float(crps.numel()), device=y.device))
        p_y = pmf.gather(1, ys.view(-1, 1)).squeeze(1)
        cdf_below = torch.where(ys > 0, cdf.gather(1, (ys - 1).clamp_min(0).view(-1, 1)).squeeze(1), torch.zeros_like(p_y))
        pit = cdf_below + 0.5 * p_y                                             # mid-PIT
        bins = (pit * PIT_BINS).long().clamp(0, PIT_BINS - 1)
        self.pit += torch.bincount(bins, minlength=PIT_BINS).double()
        q90 = (cdf < 0.9).sum(dim=-1)
        self.acc.add_total("q90_cov_sum", (ys <= q90).double().sum())
        self.acc.add_total("q90_cov_n", torch.tensor(float(ys.numel()), device=y.device))
        # v6: the predictive q90 as an operational peak forecast (high-count use case).
        peak = ys.float() >= self.count_p90
        e90 = q90.float() - ys.float()
        self.acc.add("peak_q90_se", e90 ** 2, peak)
        self.acc.add("peak_q90_mae", e90.abs(), peak)
        self.acc.add("peak_q90_underpred_rate", (q90.float() < 0.5 * ys.float()).float(), peak)

    def update_so(self, length: int, preds: Predictions, batch: dict) -> None:
        """Synthetic-outage pass of `length` steps (BatchPreparer.prepare(so_length=L))."""
        a = self.acc
        y = batch["target_forecast"].float()
        sel = batch["so_selected"] & self._valid(*y.shape)
        point = preds.point.float()
        err = point - y
        a.add(f"so_{length}_se", err ** 2, sel)
        a.add(f"so_{length}_forecast_mae", err.abs(), sel)
        a.add(f"so_{length}_point_sum", point, sel)
        a.add(f"so_{length}_y_sum", y, sel)
        a.add(f"so_{length}_collapse", (point < 1.0).float(), sel & (y >= 10.0))
        if preds.imputed_raw is not None:
            truth = self.scale.to_raw(batch["target_impute"][..., 0].float())
            m = batch["so_slots"] & batch["input_observed"]
            a.add(f"so_{length}_impute_se", (preds.imputed_raw.float() - truth) ** 2, m)
        if preds.outage_q is not None:
            valid = batch["target_observed"] & self._valid(*y.shape)
            if self.holdout is not None:
                valid = valid & ~self.holdout.view(1, -1)
            self.auc[f"so_{length}_outage"].add(preds.outage_q.float(), sel, valid)

    def update_perturbation(self, adapter, batch: dict, preds: Predictions) -> None:
        """Mask-vs-value sensitivity (report E0.5): flip ch0 of the final input step,
        or add N(0, 0.5) to the visible scaled traffic values, and measure how far the
        forecast (and P(zero)) moves. A model that reads the mask flag more than the
        values shows |mask_flip_dpoint| >> |value_perturb_dpoint|."""
        a = self.acc
        point = preds.point.float()
        flipped = dict(batch)
        masks = batch["structural_masks"].clone()
        masks[:, :, -1, 0] = 1.0 - masks[:, :, -1, 0]
        flipped["structural_masks"] = masks
        p_flip = adapter.predict(flipped)
        ones = self._valid(*point.shape)
        a.add("mask_flip_dpoint", (p_flip.point.float() - point).abs(), ones)
        if preds.p_zero is not None:
            a.add("mask_flip_dpzero", (p_flip.p_zero.float() - preds.p_zero.float()).abs(), ones)

        noisy = dict(batch)
        x = batch["dynamic_features"].clone()
        visible = ~batch["input_missing"]
        noise = 0.5 * torch.randn(x.shape[:3], device=x.device)
        x[..., 0] = torch.where(visible, x[..., 0] + noise, x[..., 0])
        noisy["dynamic_features"] = x
        p_noisy = adapter.predict(noisy)
        a.add("value_perturb_dpoint", (p_noisy.point.float() - point).abs(), ones)

    # ------------------------------------------------------------------ #
    def _series_metrics(self) -> Dict[str, Optional[float]]:
        point = torch.cat(self.series["point"]).numpy()
        y = torch.cat(self.series["y"]).numpy()
        obs = torch.cat(self.series["obs"]).numpy()
        N = y.shape[1]
        ind = self.holdout.cpu().numpy() if self.holdout is not None else np.zeros(N, bool)
        ign = self.ignore.cpu().numpy() if self.ignore is not None else np.zeros(N, bool)
        trans = ~ind & ~ign
        out: Dict[str, Optional[float]] = {}

        nrmse = np.full(N, np.nan)
        corr = np.full(N, np.nan)
        for n in range(N):
            m = obs[:, n]
            if m.sum() < 10 or y[m, n].mean() <= 0:
                continue
            nrmse[n] = np.sqrt(np.mean((point[m, n] - y[m, n]) ** 2)) / y[m, n].mean()
            if point[m, n].std() > 0 and y[m, n].std() > 0:
                corr[n] = np.corrcoef(point[m, n], y[m, n])[0, 1]

        def _nanmedian(v):
            return float(np.nanmedian(v)) if np.isfinite(v).any() else None

        out["node_nrmse_median_trans"] = _nanmedian(nrmse[trans])
        out["node_nrmse_median_ind"] = _nanmedian(nrmse[ind]) if ind.any() else None
        if ind.any():
            peer = np.full(N, np.nan)
            for n in np.where(ind)[0]:
                best = np.nan
                for j in np.where(trans)[0]:
                    m = obs[:, n] & obs[:, j]
                    if m.sum() < 10 or y[m, n].std() == 0 or y[m, j].std() == 0:
                        continue
                    c = np.corrcoef(y[m, n], y[m, j])[0, 1]
                    best = c if np.isnan(best) else max(best, c)
                peer[n] = best
            both = ind & np.isfinite(corr) & np.isfinite(peer)
            out["ind_corr_mean"] = float(np.nanmean(corr[ind])) if np.isfinite(corr[ind]).any() else None
            out["ind_peer_corr_mean"] = float(np.nanmean(peer[ind])) if np.isfinite(peer[ind]).any() else None
            out["ind_beats_peer_frac"] = float(np.mean(corr[both] >= peer[both])) if both.any() else None
            # Degeneracy check (thesis ch. 7 AUTHOR-CHECK; v3 produced r = 1.00): mean
            # pairwise correlation among the inductive stations' PREDICTED series vs the
            # same statistic for their true series. pred >> true = near-identical outputs.
            idx = np.where(ind)[0]
            pred_r, true_r = [], []
            for a_i in range(len(idx)):
                for b_i in range(a_i + 1, len(idx)):
                    i, j = idx[a_i], idx[b_i]
                    if point[:, i].std() > 0 and point[:, j].std() > 0:
                        pred_r.append(np.corrcoef(point[:, i], point[:, j])[0, 1])
                    m = obs[:, i] & obs[:, j]
                    if m.sum() >= 10 and y[m, i].std() > 0 and y[m, j].std() > 0:
                        true_r.append(np.corrcoef(y[m, i], y[m, j])[0, 1])
            out["ind_pred_pairwise_corr_mean"] = float(np.mean(pred_r)) if pred_r else None
            out["ind_true_pairwise_corr_mean"] = float(np.mean(true_r)) if true_r else None
        return out

    def summary(self) -> Dict[str, Optional[float]]:
        m = self.acc.means()
        out: Dict[str, Optional[float]] = {}
        for k, v in m.items():
            if v is None:
                out[k] = None
            elif k.endswith("_se"):
                out[k[:-3] + ("_forecast_rmse" if k.startswith("so_") and "impute" not in k else "_rmse")] = float(np.sqrt(v))
            elif not k.endswith(("_sum", "_n")):
                out[k] = v
        # Ratio metrics from sums (means x counts cancel the same counts).
        s, c = self.acc.s, self.acc.c
        if "real_gap_point_sum" in s and float(s["real_gap_profile_sum"]) > 0:
            out["real_gap_mass_ratio"] = float(s["real_gap_point_sum"]) / float(s["real_gap_profile_sum"])
        for k in list(s):
            if k.startswith("so_") and k.endswith("_point_sum"):
                L = k[3:-len("_point_sum")]
                ysum = float(s[f"so_{L}_y_sum"])
                out[f"so_{L}_mass_ratio"] = float(s[k]) / ysum if ysum > 0 else None
        if "crps_sum" in s:
            out["crps"] = float(s["crps_sum"]) / max(float(s["crps_n"]), 1.0)
            out["q90_coverage"] = float(s["q90_cov_sum"]) / max(float(s["q90_cov_n"]), 1.0)
            total = float(self.pit.sum())
            for b in range(PIT_BINS):
                out[f"pit_b{b}"] = float(self.pit[b]) / total if total > 0 else None
        for name, auc in self.auc.items():
            out[f"{name}_auc"] = auc.value()
        if self.store_series and self.series["y"]:
            out.update(self._series_metrics())
        return out
