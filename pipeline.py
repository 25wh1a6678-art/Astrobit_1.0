import glob
import os
import warnings

import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")

TRAIN_DIR = "train"
DEV_DIR = "dev"
PRIVATE_DIR = "private"

DETREND_WINDOW_DAYS = 1.0
PERIOD_MIN = 3.0
PERIOD_MAX = 400.0
N_COARSE = 50000
N_PEAKS = 8
N_FINE = 600
DURATIONS = np.array([0.05, 0.1, 0.2, 0.4, 0.8])
SDE_THRESHOLD = 20.0


def sg_detrend(f, cadence, window_days=DETREND_WINDOW_DAYS, polyorder=2):
    """Savitzky-Golay detrend per segment."""
    k = max(polyorder + 2, int(window_days / cadence) | 1)
    if k % 2 == 0:
        k += 1
    return savgol_filter(f, window_length=k, polyorder=polyorder)


def sigma_clip(f, sigma=4, iters=3):
    """Iteratively mask outliers, returns boolean mask of good points."""
    mask = np.ones(len(f), bool)
    for _ in range(iters):
        med = np.median(f[mask])
        mad = np.median(np.abs(f[mask] - med))
        mask = np.abs(f - med) < sigma * (1.4826 * mad)
    return mask


def clean(df, window_days=DETREND_WINDOW_DAYS):
    m = (df.quality.values == 0) & np.isfinite(df.flux.values)
    t, f, q = df.time.values[m], df.flux.values[m].astype(float), df.quarter.values[m]
    if len(t) < 1000:
        return None, None
    for qq in np.unique(q):
        s = q == qq
        med = np.median(f[s])
        f[s] = f[s] / med if med > 0 else 1.0
    good = sigma_clip(f)
    t, f, q = t[good], f[good], q[good]
    cadence = np.median(np.diff(t))
    trend = np.ones_like(f)
    for qq in np.unique(q):
        s = q == qq
        if s.sum() < 10:
            continue
        trend[s] = sg_detrend(f[s], cadence, window_days)
    ok = np.isfinite(trend) & (trend > 0)
    return t[ok], f[ok] / trend[ok]


def sde(power, i):
    med = np.nanmedian(power)
    mad = np.nanmedian(np.abs(power - med))
    return float((power[i] - med) / (1.4826 * mad)) if mad > 0 else 0.0


def search(t, f):
    bls = BoxLeastSquares(t, f)
    baseline = t.max() - t.min()
    pmax = min(PERIOD_MAX, baseline / 2.0)
    coarse = np.exp(np.linspace(np.log(PERIOD_MIN), np.log(pmax), N_COARSE))
    res = bls.power(coarse, DURATIONS, objective="likelihood")
    power = np.asarray(res.power)
    order = np.argsort(power)[::-1]
    peaks, used = [], np.zeros(len(coarse), bool)
    for i in order:
        if used[i]:
            continue
        peaks.append(i)
        lo = np.searchsorted(coarse, coarse[i] * 0.9)
        hi = np.searchsorted(coarse, coarse[i] * 1.1)
        used[lo:hi] = True
        if len(peaks) >= N_PEAKS:
            break
    best = None
    for i in peaks:
        p0 = coarse[i]
        for alias in [p0 * 0.5, p0, p0 * 2.0, p0 * 3.0]:
            if alias < PERIOD_MIN or alias > pmax:
                continue
            fine = np.linspace(alias * 0.98, alias * 1.02, N_FINE)
            fine = fine[(fine > PERIOD_MIN) & (fine <= pmax)]
            if len(fine) < 10:
                continue
            r = bls.power(fine, DURATIONS, objective="likelihood")
            p = np.asarray(r.power)
            j = int(np.nanargmax(p))
            score = sde(power, i)
            if best is None or score > best["sde"]:
                best = {
                    "period": float(r.period[j]),
                    "depth_ppm": float(r.depth[j] * 1e6),
                    "duration_hours": float(r.duration[j] * 24),
                    "t0": float(r.transit_time[j]),
                    "sde": score,
                }
    return best or {"period": np.nan, "depth_ppm": np.nan,
                    "duration_hours": np.nan, "t0": np.nan, "sde": 0.0}


def confidence_from_sde(s, midpoint=SDE_THRESHOLD, steepness=0.4):
    return float(1.0 / (1.0 + np.exp(-steepness * (s - midpoint))))


def load_model():
    import pickle
    if os.path.exists("model.pkl"):
        with open("model.pkl", "rb") as fh:
            return pickle.load(fh)
    return None


def run_split(directory):
    from features import extract_features
    from vetting import vet_candidate
    model = load_model()
    out = []
    paths = sorted(glob.glob(f"{directory}/*.parquet"))
    for i, path in enumerate(paths, 1):
        sid = os.path.basename(path)[:-8]
        try:
            df = pd.read_parquet(path)
            t, f = clean(df)
            r = search(t, f) if t is not None else {"sde": 0.0}
        except Exception as e:
            print(f"  {sid} failed: {e}")
            r = {"sde": 0.0}
        s = r.get("sde", 0.0)
        # use raw SDE-based confidence for ranking (ML scores too compressed)
        conf = confidence_from_sde(s)
        hit = s > SDE_THRESHOLD
        # apply vetting: reduce confidence for flagged candidates
        if not np.isnan(r.get("period", np.nan)):
            try:
                q = df.quarter.values[(df.quality.values == 0) & np.isfinite(df.flux.values)]
                vet = vet_candidate(t, f, q[:len(t)], r)
                conf = conf * vet["vetting_score"]
            except Exception:
                pass
        hit = conf > 0.5
        out.append({
            "star_id": sid,
            "prediction": int(hit),
            "confidence": round(conf, 4),
            "period": round(r["period"], 5) if hit else None,
            "depth_ppm": round(r["depth_ppm"], 1) if hit else None,
            "duration_hours": round(r["duration_hours"], 3) if hit else None,
        })
        print(f"  [{i}/{len(paths)}] {sid}  SDE={s:.1f}  conf={conf:.3f}", flush=True)
    return pd.DataFrame(out)


if __name__ == "__main__":
    import sys
    directory = sys.argv[1] if len(sys.argv) > 1 else PRIVATE_DIR
    sub = run_split(directory)
    sub.to_csv("submission.csv", index=False)
    print(f"\n{len(sub)} rows, {sub.prediction.sum()} detections")
