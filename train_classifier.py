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

    rows = []
    paths = sorted(glob.glob(f"{directory}/*.parquet"))
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
            feats["kepid"] = kepid
            rows.append(feats)
        except Exception as e:
            print(f"  skipping {kepid}: {e}")
        if i % 20 == 0:
            print(f"  [{i}/{len(paths)}]", flush=True)
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
