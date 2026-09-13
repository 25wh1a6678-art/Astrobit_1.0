#!/usr/bin/env python3
"""
Entry point: runs the full Astrobit pipeline end-to-end.
Usage:
    python run.py train      # train classifier (step 1)
    python run.py calibrate  # calibrate on dev set (step 2)
    python run.py submit     # generate + validate submission (step 3)
    python run.py all        # run all steps in order
"""
import subprocess
import sys


def run(cmd):
    print(f"\n>>> {cmd}\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"FAILED: {cmd}")
        sys.exit(1)


STEPS = {
    "train":     ".venv/bin/python3 train_classifier.py",
    "calibrate": ".venv/bin/python3 calibrate.py",
    "submit":    ".venv/bin/python3 pipeline.py private && .venv/bin/python3 validate.py",
}

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "submit"
    if mode == "all":
        for step in STEPS.values():
            run(step)
    elif mode in STEPS:
        run(STEPS[mode])
    else:
        print(__doc__)
        sys.exit(1)
