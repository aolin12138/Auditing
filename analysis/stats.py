"""Shared statistics for the final report's figures and tables.

Definitions are identical to analysis/extra_evidence.py (which produced the poster numbers), so
report and poster agree:
  ci95       mean and normal-approximation 95% CI (1.96 * SE) over finite values
  cohens_d   pooled-SD Cohen's d (a - b)
  d_with_ci  Cohen's d with a percentile-bootstrap 95% CI (2000 resamples, fixed RNG seed 0
             per call so every number is reproducible)
"""
import numpy as np


def finite(x):
    x = np.asarray(x, float)
    return x[np.isfinite(x)]


def ci95(x):
    x = finite(x)
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    return m, m - 1.96 * se, m + 1.96 * se


def cohens_d(a, b):
    a, b = finite(a), finite(b)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan


def d_with_ci(a, b, n_boot=2000, seed=0):
    """Returns (d, lo, hi, n_a, n_b)."""
    a, b = finite(a), finite(b)
    point = cohens_d(a, b)
    if not np.isfinite(point):
        return point, np.nan, np.nan, len(a), len(b)
    rng = np.random.default_rng(seed)
    boots = np.array([cohens_d(rng.choice(a, len(a)), rng.choice(b, len(b))) for _ in range(n_boot)])
    boots = boots[np.isfinite(boots)]
    if len(boots) == 0:
        return point, np.nan, np.nan, len(a), len(b)
    return point, np.percentile(boots, 2.5), np.percentile(boots, 97.5), len(a), len(b)
