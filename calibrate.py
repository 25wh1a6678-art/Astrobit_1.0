"""
Calibrate model confidence using Platt scaling on the dev set.
Usage: python calibrate.py
Saves calibrated model to model.pkl (updates in place).
"""
import glob
import os
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from features import extract_features
from pipeline import clean, search

warnings.filterwarnings("ignore")

DEV_DIR = "dev"
CACHE_FILE = "dev_bls_cache.csv"
N_WORKERS = 4


def process_star(path):
    warnings.filterwarnings("ignore")
    kepid = int(os.path.basename(path).replace("KIC_", "").replace(".parquet", ""))
    try:
        t, f = clean(pd.read_parquet(path))
        if t is None:
            return kepid, None, None, None
        r = search(t, f)
        feats = extract_features(t, f, r)
        return kepid, r, feats, (t, f)
    except Exception:
        return kepid, None, None, None


def build_dev_features():
    labels = pd.read_csv("dev_labels.csv")
    truth = pd.read_csv("dev_truth.csv")
    merged = labels.merge(truth[["kepid", "injected"]], on="kepid", how="left")
    merged["has_planet"] = ((merged["label"] == 1) | (merged["injected"] == 1)).fillna(0).astype(int)

    cache = pd.read_csv(CACHE_FILE) if os.path.exists(CACHE_FILE) else pd.DataFrame()
    cached_ids = set(cache.kepid.values) if not cache.empty else set()

    all_paths = sorted(glob.glob(f"{DEV_DIR}/*.parquet"))
    pending = [p for p in all_paths
               if int(os.path.basename(p).replace("KIC_", "").replace(".parquet", ""))
               not in cached_ids]

    print(f"  {len(cached_ids)} cached, {len(pending)} to process ({N_WORKERS} workers)")

    new_cache_rows = []
    done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(process_star, p): p for p in pending}
        for fut in as_completed(futures):
            kepid, r, feats, _ = fut.result()
            done += 1
            if r is not None:
                new_cache_rows.append({**r, "kepid": kepid})
                new_df = pd.DataFrame(new_cache_rows)
                combined = pd.concat([cache, new_df]).drop_duplicates("kepid")
                combined.to_csv(CACHE_FILE, index=False)
            print(f"  [{len(cached_ids)+done}/{len(all_paths)}] KIC_{kepid}"
                  f"  SDE={r['sde']:.1f}" if r else f"  [{done}] KIC_{kepid} FAILED",
                  flush=True)

    cache = pd.read_csv(CACHE_FILE) if os.path.exists(CACHE_FILE) else pd.DataFrame()
    rows = []
    for path in all_paths:
        kepid = int(os.path.basename(path).replace("KIC_", "").replace(".parquet", ""))
        row = merged[merged.kepid == kepid]
        if row.empty or cache.empty or kepid not in cache.kepid.values:
            continue
        label = int(row.has_planet.values[0])
        cr = cache[cache.kepid == kepid].iloc[0].to_dict()
        r = {k: cr[k] for k in ["period", "depth_ppm", "duration_hours", "t0", "sde"]}
        try:
            t, f = clean(pd.read_parquet(path))
            if t is None:
                continue
            feats = extract_features(t, f, r)
            feats["label"] = label
            rows.append(feats)
        except Exception:
            continue
    return pd.DataFrame(rows)


if __name__ == "__main__":
    with open("model.pkl", "rb") as fh:
        bundle = pickle.load(fh)

    clf = bundle["clf"]
    scaler = bundle["scaler"]
    feature_cols = bundle["features"]

    print("Building dev features for calibration...")
    dev_df = build_dev_features()
    dev_df.to_csv("dev_features.csv", index=False)

    X_dev = scaler.transform(dev_df[feature_cols].fillna(0).values)
    y_dev = dev_df["label"].values

    # Platt scaling: fit logistic regression on raw model scores
    raw_scores = clf.predict_proba(X_dev)[:, 1].reshape(-1, 1)
    platt = LogisticRegression()
    platt.fit(raw_scores, y_dev)

    bundle["platt"] = platt
    with open("model.pkl", "wb") as fh:
        pickle.dump(bundle, fh)

    # show calibration improvement
    cal_scores = platt.predict_proba(raw_scores)[:, 1]
    print(f"\nDev set: {y_dev.sum()} planets / {len(y_dev)} stars")
    print(f"Raw score mean (planet): {raw_scores[y_dev==1].mean():.3f}")
    print(f"Cal score mean (planet): {cal_scores[y_dev==1].mean():.3f}")
    print("Saved calibrated model to model.pkl")
