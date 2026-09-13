"""
Candidate vetting checks to reject false positives.
Each check returns a flag (1=suspicious, 0=ok) and a score.
"""
import numpy as np

# Known Kepler systematic periods to flag (days)
KEPLER_SYSTEMATICS = [372.5, 186.25, 93.125, 371.0, 180.5]


def check_secondary_eclipse(t, f, period, t0, duration_hours):
    """Flag if secondary eclipse at phase 0.5 is comparable to primary depth."""
    ph = ((t - t0) / period) % 1.0
    dur_phase = (duration_hours / 24.0) / period

    primary = np.abs(ph) < dur_phase
    secondary = np.abs(ph - 0.5) < dur_phase

    if primary.sum() < 3 or secondary.sum() < 3:
        return 0, 0.0

    primary_depth = 1.0 - np.median(f[primary])
    secondary_depth = 1.0 - np.median(f[secondary])

    ratio = abs(secondary_depth) / (abs(primary_depth) + 1e-10)
    # ratio > 0.5 suggests eclipsing binary, not planet
    return int(ratio > 0.5), float(ratio)


def check_odd_even(t, f, period, t0, duration_hours):
    """Flag if odd/even transit depths are inconsistent (eclipsing binary sign)."""
    half_dur = (duration_hours / 24.0) / 2.0
    transits = np.arange(
        np.floor((t.min() - t0) / period),
        np.ceil((t.max() - t0) / period)
    ).astype(int)

    odd, even = [], []
    for n in transits:
        tc = t0 + n * period
        mask = np.abs(t - tc) < half_dur
        if mask.sum() < 3:
            continue
        d = 1.0 - np.median(f[mask])
        (even if n % 2 == 0 else odd).append(d)

    if not odd or not even:
        return 0, 0.0

    ratio = abs(np.mean(odd) - np.mean(even)) / (abs(np.mean(odd) + np.mean(even)) + 1e-10)
    return int(ratio > 0.3), float(ratio)


def check_per_quarter_recurrence(t, f, q, period, t0, duration_hours):
    """Flag if transit signal appears in fewer than half the expected quarters."""
    half_dur = (duration_hours / 24.0) / 2.0
    quarters_with_transit = set()
    quarters_expected = set()

    for qq in np.unique(q):
        s = q == qq
        t_q = t[s]
        # check if any transit falls in this quarter
        transits_in_q = np.arange(
            np.floor((t_q.min() - t0) / period),
            np.ceil((t_q.max() - t0) / period)
        ).astype(int)
        for n in transits_in_q:
            tc = t0 + n * period
            if t_q.min() <= tc <= t_q.max():
                quarters_expected.add(qq)
                mask = np.abs(t[s] - tc) < half_dur
                if mask.sum() >= 3:
                    d = 1.0 - np.median(f[s][mask])
                    if d > 0:
                        quarters_with_transit.add(qq)

    if not quarters_expected:
        return 0, 1.0

    recurrence = len(quarters_with_transit) / len(quarters_expected)
    return int(recurrence < 0.5), float(recurrence)


def check_systematic_period(period):
    """Flag if period is close to a known Kepler systematic."""
    for sp in KEPLER_SYSTEMATICS:
        for alias in [sp, sp / 2, sp * 2]:
            if abs(period - alias) / alias < 0.02:
                return 1, float(period)
    return 0, 0.0


def vet_candidate(t, f, q, result):
    """Run all vetting checks. Returns dict of flags and a combined vetting score."""
    period = result["period"]
    t0 = result["t0"]
    duration = result["duration_hours"]

    sec_flag, sec_ratio = check_secondary_eclipse(t, f, period, t0, duration)
    oe_flag, oe_ratio = check_odd_even(t, f, period, t0, duration)
    rec_flag, recurrence = check_per_quarter_recurrence(t, f, q, period, t0, duration)
    sys_flag, _ = check_systematic_period(period)

    # vetting score: 1.0 = clean, lower = more suspicious
    n_flags = sec_flag + oe_flag + rec_flag + sys_flag
    vetting_score = max(0.0, 1.0 - n_flags * 0.25)

    return {
        "secondary_flag": sec_flag,
        "odd_even_flag": oe_flag,
        "recurrence_flag": rec_flag,
        "systematic_flag": sys_flag,
        "vetting_score": vetting_score,
    }
