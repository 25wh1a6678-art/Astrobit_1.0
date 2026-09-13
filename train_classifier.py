"""
Train a Random Forest classifier on BLS features from the train set.
Usage: python train_classifier.py
Saves model to model.pkl. Caches BLS results to train_bls_cache.csv.
"""
import glob
import os
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

from features import extract_features
from pipeline import clean, search

warnings.filterwarnings("ignore")

TRAIN_DIR = "train"
CACHE_FILE = "train_bls_cache.csv"
N_WORKERS = 4
FEATURE_COLS = ["sde", "depth_ppm", "duration_hours", "period",
                "scatter_ppm", "snr", "n_transits", "odd_even_ratio",
                "secondary_depth_ppm"]


def process_star(path):
    """Run clean + search + extract_features for one star. Returns (kepid, r, feats)."""
    warnings.filterwarnings("ignore")
    kepid = int(os.path.basename(path).replace("KIC_", "").replace(".parquet", ""))
    try:
        t, f = clean(pd.read_parquet(path))
        if t is None:
            return kepid, None, None
        r = search(t, f)
        feats = extract_features(t, f, r)
        return kepid, r, feats
    except Exception as e:
        return kepid, None, None


def build_dataset(directory, labels_csv, truth_csv):
    labels = pd.read_csv(labels_csv)
    truth = pd.read_csv(truth_csv)
    merged = labels.merge(truth[["kepid", "injected"]], on="kepid", how="left")
    merged["has_planet"] = ((merged["label"] == 1) | (merged["injected"] == 1)).fillna(0).astype(int)

    # load existing cache
    cache = pd.read_csv(CACHE_FILE) if os.path.exists(CACHE_FILE) else pd.DataFrame()
    cached_ids = set(cache.kepid.values) if not cache.empty else set()

    all_paths = sorted(glob.glob(f"{directory}/*.parquet"))
    pending = [p for p in all_paths
               if int(os.path.basename(p).replace("KIC_", "").replace(".parquet", ""))
               not in cached_ids]

    print(f"  {len(cached_ids)} cached, {len(pending)} to process ({N_WORKERS} workers)")

    new_cache_rows = []
    done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(process_star, p): p for p in pending}
        for fut in as_completed(futures):
            kepid, r, feats = fut.result()
            done += 1
            if r is not None:
                new_cache_rows.append({**r, "kepid": kepid})
                # flush cache every star
                new_df = pd.DataFrame(new_cache_rows)
                combined = pd.concat([cache, new_df]).drop_duplicates("kepid")
                combined.to_csv(CACHE_FILE, index=False)
            print(f"  [{len(cached_ids)+done}/{len(all_paths)}] KIC_{kepid}"
                  f"  SDE={r['sde']:.1f}" if r else f"  [{done}] KIC_{kepid} FAILED",
                  flush=True)

    # reload full cache
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
            feats["kepid"] = kepid
            rows.append(feats)
        except Exception:
            continue
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Building train dataset...")
    df = build_dataset(TRAIN_DIR, "train_labels.csv", "train_truth.csv")
    df.to_csv("train_features.csv", index=False)
    print(f"  {len(df)} stars, {df.label.sum()} with planets")

    X = df[FEATURE_COLS].fillna(0).values
    y = df["label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = RandomForestClassifier(n_estimators=200, max_depth=8,
                                 class_weight="balanced", random_state=42)
    clf.fit(X_scaled, y)

    print("\nTrain set performance:")
    print(classification_report(y, clf.predict(X_scaled)))
    print(f"ROC-AUC: {roc_auc_score(y, clf.predict_proba(X_scaled)[:, 1]):.3f}")

    with open("model.pkl", "wb") as fh:
        pickle.dump({"clf": clf, "scaler": scaler, "features": FEATURE_COLS}, fh)
    print("\nSaved model.pkl")
