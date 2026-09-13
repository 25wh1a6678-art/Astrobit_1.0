"""
Evaluate pipeline on dev set and report precision, recall, F1, PR-AUC.
Usage: python evaluate.py
Requires model.pkl (run train_classifier.py first).
"""
import glob
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (classification_report, average_precision_score,
                             roc_auc_score, precision_recall_curve)

from features import extract_features
from pipeline import clean, search, confidence_from_sde, SDE_THRESHOLD

warnings.filterwarnings("ignore")

DEV_DIR = "dev"


def evaluate():
    labels = pd.read_csv("dev_labels.csv")
    truth = pd.read_csv("dev_truth.csv")
    merged = labels.merge(truth[["kepid", "injected", "bin", "depth_ppm"]],
                          on="kepid", how="left")
    merged["has_planet"] = ((merged["label"] == 1) | (merged["injected"] == 1)).fillna(0).astype(int)

    model = None
    if os.path.exists("model.pkl"):
        with open("model.pkl", "rb") as fh:
            model = pickle.load(fh)
        print("Using trained RF model")
    else:
        print("No model.pkl found — using SDE threshold only")

    rows = []
    paths = sorted(glob.glob(f"{DEV_DIR}/*.parquet"))
    for i, path in enumerate(paths, 1):
        kepid = int(os.path.basename(path).replace("KIC_", "").replace(".parquet", ""))
        row = merged[merged.kepid == kepid]
        if row.empty:
            continue
        label = int(row.has_planet.values[0])
        depth = float(row.depth_ppm.values[0]) if not pd.isna(row.depth_ppm.values[0]) else 0.0
        bin_name = row.bin.values[0] if "bin" in row.columns else "unknown"

        try:
            t, f = clean(pd.read_parquet(path))
            r = search(t, f) if t is not None else {"sde": 0.0}
        except Exception:
            r = {"sde": 0.0}

        s = r.get("sde", 0.0)
        if model and not np.isnan(r.get("period", np.nan)):
            try:
                feats = extract_features(t, f, r)
                X = np.array([[feats[c] for c in model["features"]]])
                X = model["scaler"].transform(X)
                raw_conf = float(model["clf"].predict_proba(X)[0, 1])
                conf = float(model["platt"].predict_proba([[raw_conf]])[0, 1]) \
                    if "platt" in model else raw_conf
            except Exception:
                conf = confidence_from_sde(s)
        else:
            conf = confidence_from_sde(s)

        rows.append({
            "kepid": kepid, "label": label, "confidence": conf,
            "sde": s, "depth_ppm": depth, "bin": bin_name,
        })
        print(f"  [{i}/{len(paths)}] KIC_{kepid}  label={label}  conf={conf:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("dev_eval.csv", index=False)

    y_true = df.label.values
    y_conf = df.confidence.values
    y_pred = (y_conf > 0.5).astype(int)

    print("\n--- Overall ---")
    print(classification_report(y_true, y_pred, target_names=["no planet", "planet"]))
    print(f"PR-AUC:  {average_precision_score(y_true, y_conf):.3f}")
    print(f"ROC-AUC: {roc_auc_score(y_true, y_conf):.3f}")

    # breakdown by difficulty bin
    if "bin" in df.columns:
        print("\n--- By difficulty bin ---")
        for b in df.bin.dropna().unique():
            sub = df[df.bin == b]
            if sub.label.sum() == 0:
                continue
            ap = average_precision_score(sub.label, sub.confidence)
            rec = (sub[sub.label == 1].confidence > 0.5).mean()
            print(f"  {b:20s}  n={len(sub):3d}  recall={rec:.0%}  AP={ap:.3f}")


if __name__ == "__main__":
    evaluate()
