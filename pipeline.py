import glob
import os
import warnings

import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares

warnings.filterwarnings("ignore")

TRAIN_DIR = "train"
DEV_DIR = "dev"
PRIVATE_DIR = "private"

DETREND_WINDOW_DAYS = 1.0
PERIOD_MIN = 3.0
PERIOD_MAX = 400.0
N_COARSE = 20000
N_PEAKS = 8
N_FINE = 600
DURATIONS = np.array([0.05, 0.1, 0.2, 0.4, 0.8])
SDE_THRESHOLD = 10.0


def clean(df, window_days=DETREND_WINDOW_DAYS):
    m = (df.quality.values == 0) & np.isfinite(df.flux.values)
    t, f, q = df.time.values[m], df.flux.values[m].astype(float), df.quarter.values[m]
    if len(t) < 1000:
        return None, None
    for qq in np.unique(q):
        s = q == qq
        med = np.median(f[s])
        f[s] = f[s] / med if med > 0 else 1.0
    cadence = np.median(np.diff(t))
    k = max(5, int(window_days / cadence) | 1)
    trend = pd.Series(f).rolling(k, center=True, min_periods=k // 3).median().values
    ok = np.isfinite(trend) & (trend > 0)
    return t[ok], f[ok] / trend[ok]


def sde(power, i):
    med = np.nanmedian(power)
    mad = np.nanmedian(np.abs(power - med))
    return float((power[i] - med) / (1.4826 * mad)) if mad > 0 else 0.0


def search(t, f):
    bls = BoxLeastSquares(t, f)
    pmax = min(PERIOD_MAX, (t.max() - t.min()) / 3.0)
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
        fine = np.linspace(p0 * 0.98, p0 * 1.02, N_FINE)
        fine = fine[fine > PERIOD_MIN]
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


def run_split(directory):
    out = []
    paths = sorted(glob.glob(f"{directory}/*.parquet"))
    for i, path in enumerate(paths, 1):
        sid = os.path.basename(path)[:-8]
        try:
            t, f = clean(pd.read_parquet(path))
            r = search(t, f) if t is not None else {"sde": 0.0}
        except Exception as e:
            print(f"  {sid} failed: {e}")
            r = {"sde": 0.0}
        s = r.get("sde", 0.0)
        hit = s > SDE_THRESHOLD
        out.append({
            "star_id": sid,
            "prediction": int(hit),
            "confidence": round(confidence_from_sde(s), 4),
            "period": round(r["period"], 5) if hit else None,
            "depth_ppm": round(r["depth_ppm"], 1) if hit else None,
            "duration_hours": round(r["duration_hours"], 3) if hit else None,
        })
        if i % 10 == 0:
            print(f"  [{i}/{len(paths)}]", flush=True)
    return pd.DataFrame(out)


if __name__ == "__main__":
    import sys
    directory = sys.argv[1] if len(sys.argv) > 1 else PRIVATE_DIR
    sub = run_split(directory)
    sub.to_csv("submission.csv", index=False)
    print(f"\n{len(sub)} rows, {sub.prediction.sum()} detections")
