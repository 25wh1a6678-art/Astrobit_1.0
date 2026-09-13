"""
Validate submission.csv format before submitting.
Usage: python validate.py
"""
import sys
import pandas as pd

REQUIRED_COLS = ["star_id", "prediction", "confidence", "period", "depth_ppm", "duration_hours"]
EXPECTED_ROWS = 87


def validate(path="submission.csv"):
    try:
        s = pd.read_csv(path)
    except FileNotFoundError:
        print(f"ERROR: {path} not found. Run: python pipeline.py private")
        sys.exit(1)

    errors = []

    if list(s.columns) != REQUIRED_COLS:
        errors.append(f"columns must be exactly {REQUIRED_COLS}, got {list(s.columns)}")

    if len(s) != EXPECTED_ROWS:
        errors.append(f"expected {EXPECTED_ROWS} rows, got {len(s)}")

    if s.star_id.nunique() != EXPECTED_ROWS:
        errors.append("duplicate star_id found")

    if not s.star_id.str.match(r"^STAR_\d{4}$").all():
        bad = s[~s.star_id.str.match(r"^STAR_\d{4}$")].star_id.tolist()
        errors.append(f"bad star_id format: {bad[:5]}")

    if not s.prediction.isin([0, 1]).all():
        errors.append("prediction must be 0 or 1")

    if not s.confidence.between(0, 1).all():
        errors.append("confidence values out of [0, 1] range")

    pos = s[s.prediction == 1]
    for c in ("period", "depth_ppm", "duration_hours"):
        if pos[c].isna().any():
            errors.append(f"{c} missing for some detections")

    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    print(f"OK — {len(pos)} detections, {len(s) - len(pos)} non-detections")
    print(f"confidence range: [{s.confidence.min():.4f}, {s.confidence.max():.4f}]")
    print(f"unique confidence values: {s.confidence.nunique()}")


if __name__ == "__main__":
    validate()
