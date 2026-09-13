# Astrobit 1.0 — AI-Based Detection of Earth-Like Exoplanets in Kepler Data

## Pipeline Overview

```
raw SAP flux
  -> mask bad cadences + sigma-clip outliers
  -> per-quarter Savitzky-Golay detrend
  -> coarse-to-fine BLS period search (50k grid + alias checking)
  -> feature extraction (SDE, depth, SNR, odd/even, secondary eclipse)
  -> Random Forest classifier (trained on 269 labelled stars)
  -> Platt-scaled confidence calibration (dev set)
  -> candidate vetting (secondary eclipse, odd/even, recurrence, systematics)
  -> ranked submission
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Unzip data packs into the project root:
```bash
unzip train_pack.zip -d .
unzip dev_pack.zip -d .
# private_pack.zip released at hour 40
unzip private_pack.zip -d .
```

## Running

### Step 1 — Train the classifier (run once, ~8hrs on full train set)
```bash
.venv/bin/python3 train_classifier.py
```
Saves `model.pkl`. Caches BLS results to `train_bls_cache.csv` — safe to interrupt and resume.

### Step 2 — Calibrate confidence on dev set
```bash
.venv/bin/python3 calibrate.py
```
Updates `model.pkl` with Platt scaling.

### Step 3 — Generate submission
```bash
.venv/bin/python3 pipeline.py private
```
Writes `submission.csv`.

### Step 4 — Validate submission
```bash
.venv/bin/python3 validate.py
```

## Key Design Decisions

- **Savitzky-Golay detrending per quarter** — avoids edge artifacts at quarter boundaries, recovers ~90% of true transit depth vs ~33% with running median
- **Coarse-to-fine BLS** — 50k log-spaced coarse grid + 600-point fine refinement around each peak, ~100x cheaper than full-resolution search
- **Alias checking** — refines at 0.5x, 1x, 2x, 3x of each coarse peak to catch period aliases
- **Random Forest on BLS features** — replaces single SDE threshold with a multi-feature classifier trained on labelled data
- **Vetting** — secondary eclipse, odd/even depth consistency, per-quarter recurrence, known systematic periods
- **Platt scaling** — calibrates raw model scores to meaningful probabilities on the dev set

## Files

| File | Purpose |
|---|---|
| `pipeline.py` | Main pipeline: clean → search → classify → submit |
| `features.py` | Feature extraction from BLS results |
| `train_classifier.py` | Train Random Forest on train set |
| `calibrate.py` | Platt scaling calibration on dev set |
| `vetting.py` | Candidate vetting checks |
| `validate.py` | Submission format validation |
| `requirements.txt` | Pinned dependencies |
