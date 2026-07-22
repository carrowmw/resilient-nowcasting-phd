"""
Lomb–Scargle periodogram integration (reviewer requirement: make explicit
whether the periodogram is computed on the raw gapped series or on extracted
sequences — here it is computed on BOTH and the results are compared).

Variant (a) 'raw'  — observed (t, y) points of the gapped series. Lomb–Scargle
                     handles irregular sampling natively, so gaps need no
                     interpolation and the estimate reflects the true
                     underlying periodicity.
Variant (b) 'seq'  — the extracted consecutive sequences only (regularly
                     sampled within each run), i.e. exactly the data the
                     models see. Comparing (a) and (b) shows whether sequence
                     extraction preserves the dominant periodicity.

Also emits the derived indicator diurnal_capture_ratio =
avg_seq_len_hours / dominant_period_hours: the mechanism variable behind the
paper's ~20 h threshold claim (a run shorter than the dominant cycle cannot
contain one full cycle).
"""
import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.signal import lombscargle

logger = logging.getLogger(__name__)


def compute_lomb_scargle(
    series: pd.Series,
    min_period_hours: float = 2.0,
    max_period_hours: float = 24.0 * 14,
    n_freqs: int = 4000,
) -> Dict[str, np.ndarray]:
    """
    Normalized Lomb–Scargle periodogram of the observed points of a
    NaN-gapped series.

    Returns {'period_hours': [n_freqs], 'power': [n_freqs]}.
    """
    assert isinstance(series.index, pd.DatetimeIndex), \
        "CRITICAL: series must have a DatetimeIndex"
    assert 0 < min_period_hours < max_period_hours, \
        f"CRITICAL: bad period range [{min_period_hours}, {max_period_hours}]"

    observed = series.dropna()
    assert len(observed) >= 16, \
        f"CRITICAL: only {len(observed)} observed points — too few for a periodogram"

    t_hours = (observed.index - observed.index[0]) / pd.Timedelta(hours=1)
    t = t_hours.to_numpy(dtype=float)
    y = observed.to_numpy(dtype=float)
    y = y - y.mean()  # lombscargle assumes zero-mean input

    span_hours = float(t[-1] - t[0])
    max_p = min(max_period_hours, span_hours / 2.0)
    assert max_p > min_period_hours, \
        f"CRITICAL: series span {span_hours:.1f} h too short for period range"

    periods = np.geomspace(min_period_hours, max_p, n_freqs)
    ang_freqs = 2.0 * np.pi / periods
    power = lombscargle(t, y, ang_freqs, normalize=True)
    assert np.isfinite(power).all(), "CRITICAL: non-finite power in periodogram"
    return {"period_hours": periods, "power": power}


def dominant_periods(
    period_hours: np.ndarray,
    power: np.ndarray,
    top_k: int = 3,
    min_separation_factor: float = 1.2,
) -> List[Dict[str, float]]:
    """
    Top-k local maxima of the periodogram, suppressing near-duplicate periods
    (within min_separation_factor of an already-selected peak).
    """
    assert len(period_hours) == len(power), "CRITICAL: period/power length mismatch"
    order = np.argsort(power)[::-1]
    peaks: List[Dict[str, float]] = []
    for i in order:
        p = float(period_hours[i])
        if any(
            1.0 / min_separation_factor < p / pk["period_hours"] < min_separation_factor
            for pk in peaks
        ):
            continue
        peaks.append({"period_hours": p, "power": float(power[i])})
        if len(peaks) >= top_k:
            break
    return peaks


def compute_periodogram_features(
    series: pd.Series,
    spans,
    avg_seq_len_hours: float,
    top_k: int = 3,
) -> Dict[str, float]:
    """
    Periodogram-derived quality indicators for one sensor.

    Args:
        series: the NaN-gapped grid series (variant 'raw' input).
        spans: List[SequenceSpan] from find_consecutive_sequences (variant 'seq').
        avg_seq_len_hours: from compute_sequence_stats.
    """
    raw = compute_lomb_scargle(series)
    raw_peaks = dominant_periods(raw["period_hours"], raw["power"], top_k)

    # Variant (b): only points inside extracted runs. Timestamps keep their
    # true positions, so this is still an irregular-sampling problem across
    # runs — Lomb-Scargle remains the right tool.
    seq_mask = np.zeros(len(series), dtype=bool)
    for sp in spans:
        seq_mask[sp.start_idx: sp.end_idx + 1] = True
    seq_series = series.where(seq_mask)
    seq_peaks: List[Dict[str, float]] = []
    if seq_series.notna().sum() >= 16:
        seq = compute_lomb_scargle(seq_series)
        seq_peaks = dominant_periods(seq["period_hours"], seq["power"], top_k)

    dominant_raw = raw_peaks[0]["period_hours"] if raw_peaks else float("nan")
    features: Dict[str, float] = {
        "ls_dominant_period_raw_h": dominant_raw,
        "ls_dominant_power_raw": raw_peaks[0]["power"] if raw_peaks else float("nan"),
        "ls_dominant_period_seq_h": seq_peaks[0]["period_hours"] if seq_peaks else float("nan"),
        "ls_dominant_power_seq": seq_peaks[0]["power"] if seq_peaks else float("nan"),
    }
    for rank, pk in enumerate(raw_peaks[1:top_k], start=2):
        features[f"ls_period_raw_rank{rank}_h"] = pk["period_hours"]
    features["diurnal_capture_ratio"] = (
        avg_seq_len_hours / dominant_raw if np.isfinite(dominant_raw) and dominant_raw > 0
        else float("nan")
    )
    return features
