"""
Train a Random Forest classifier on BLS features from the train set.
Usage: python train_classifier.py
Saves model to model.pkl
"""
import glob
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

from features import extract_features
from pipeline import clean, search

warnings.filterwarnings("ignore")

TRAIN_DIR = "train"
FEATURE_COLS = ["sde", "depth_ppm", "duration_hours", "period",
                "scatter_ppm", "snr", "n_transits", "odd_even_ratio",
                "secondary_depth_ppm"]


def build_dataset(directory, labels_csv, truth_csv):
    labels = pd.read_csv(labels_csv)
    truth = pd.read_csv(truth_csv)
    merged = labels.merge(truth[["kepid", "injected"]], on="kepid", how="left")
    merged["has_planet"] = ((merged["label"] == 1) | (merged["injected"] == 1)).fillna(0).astype(int)

    # load cached BLS results if available
    cache_file = f"{directory}_bls_cache.csv"
    if os.path.exists(cache_file):
        print(f"  Loading cached BLS results from {cache_file}")
        cache = pd.read_csv(cache_file)
    else:
        cache = pd.DataFrame()

    rows = []
    paths = sorted(glob.glob(f"{directory}/*.parquet"))
    new_cache_rows = []

    for i, path in enumerate(paths, 1):
        kepid = int(os.path.basename(path).replace("KIC_", "").replace(".parquet", ""))
        row = merged[merged.kepid == kepid]
        if row.empty:
            continue
        label = int(row.has_planet.values[0])

        # use cache if available
        if not cache.empty and kepid in cache.kepid.values:
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
            except Exception as e:
                print(f"  skipping {kepid}: {e}")
            print(f"  [{i}/{len(paths)}] KIC_{kepid} (cached) SDE={r['sde']:.1f}", flush=True)
            continue

        try:
            t, f = clean(pd.read_parquet(path))
            if t is None:
                continue
            r = search(t, f)
            new_cache_rows.append({**r, "kepid": kepid})
            feats = extract_features(t, f, r)
            feats["label"] = label
            feats["kepid"] = kepid
            rows.append(feats)
        except Exception as e:
            print(f"  skipping {kepid}: {e}")
        print(f"  [{i}/{len(paths)}] KIC_{kepid} SDE={r.get('sde', 0):.1f}", flush=True)

    # save new cache entries
    if new_cache_rows:
        new_df = pd.DataFrame(new_cache_rows)
        combined = pd.concat([cache, new_df]).drop_duplicates("kepid")
        combined.to_csv(cache_file, index=False)
        print(f"  Saved BLS cache to {cache_file}")

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
