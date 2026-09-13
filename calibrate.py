"""
Calibrate model confidence using Platt scaling on the dev set.
Usage: python calibrate.py
Saves calibrated model to model.pkl (updates in place).
"""
import glob
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from features import extract_features
from pipeline import clean, search

warnings.filterwarnings("ignore")

DEV_DIR = "dev"


def build_dev_features():
    labels = pd.read_csv("dev_labels.csv")
    truth = pd.read_csv("dev_truth.csv")
    merged = labels.merge(truth[["kepid", "injected"]], on="kepid", how="left")
    merged["has_planet"] = ((merged["label"] == 1) | (merged["injected"] == 1)).fillna(0).astype(int)

    rows = []
    paths = sorted(glob.glob(f"{DEV_DIR}/*.parquet"))
    for i, path in enumerate(paths, 1):
        kepid = int(os.path.basename(path).replace("KIC_", "").replace(".parquet", ""))
        row = merged[merged.kepid == kepid]
        if row.empty:
            continue
        label = int(row.has_planet.values[0])
        try:
            t, f = clean(pd.read_parquet(path))
            if t is None:
                continue
            r = search(t, f)
            feats = extract_features(t, f, r)
            feats["label"] = label
            rows.append(feats)
        except Exception as e:
            print(f"  skipping {kepid}: {e}")
        if i % 10 == 0:
            print(f"  [{i}/{len(paths)}]", flush=True)
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
