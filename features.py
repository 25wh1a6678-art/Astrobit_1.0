"""
Feature extraction and ML classifier for transit candidate vetting.
Replaces the single SDE threshold with a Random Forest trained on the train set.
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def phase_fold(t, f, period, t0, n_bins=128):
    """Fold light curve and return binned phase array of fixed length."""
    ph = ((t - t0) / period) % 1.0
    ph = np.where(ph > 0.5, ph - 1.0, ph)
    idx = np.floor((ph + 0.5) * n_bins).astype(int).clip(0, n_bins - 1)
    binned = np.full(n_bins, 1.0)
    for b in range(n_bins):
        pts = f[idx == b]
        if len(pts):
            binned[b] = np.median(pts)
    return binned


def extract_features(t, f, result):
    """Extract scalar features from a BLS result for ML classification."""
    period = result["period"]
    t0 = result["t0"]
    depth = result["depth_ppm"]
    duration = result["duration_hours"]
    sde_val = result["sde"]

    # odd/even transit depth consistency
    half_dur = (duration / 24.0) / 2.0
    transits = np.arange(
        np.floor((t.min() - t0) / period),
        np.ceil((t.max() - t0) / period)
    ).astype(int)
    odd_depths, even_depths = [], []
    for n in transits:
        tc = t0 + n * period
        in_transit = np.abs(t - tc) < half_dur
        if in_transit.sum() < 3:
            continue
        d = 1.0 - np.median(f[in_transit])
        (even_depths if n % 2 == 0 else odd_depths).append(d)

    odd_mean = np.mean(odd_depths) if odd_depths else 0.0
    even_mean = np.mean(even_depths) if even_depths else 0.0
    odd_even_ratio = abs(odd_mean - even_mean) / (abs(odd_mean + even_mean) + 1e-10)

    # secondary eclipse depth at phase 0.5
    ph = ((t - t0) / period) % 1.0
    sec_mask = np.abs(ph - 0.5) < (duration / 24.0 / period)
    secondary_depth = float(1.0 - np.median(f[sec_mask])) if sec_mask.sum() > 3 else 0.0

    return {
        "sde": sde_val,
        "depth_ppm": depth,
        "duration_hours": duration,
        "period": period,
        "scatter_ppm": float(np.std(f) * 1e6),
        "snr": depth / (np.std(f) * 1e6 + 1e-10),
        "n_transits": len(odd_depths) + len(even_depths),
        "odd_even_ratio": odd_even_ratio,
        "secondary_depth_ppm": secondary_depth * 1e6,
    }
