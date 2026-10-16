"""
Independent re-derivation of every number quoted in poster/POSTER_CONTEXT.md §7.

DESIGN PRINCIPLE (verification-protocol: "vary the angle"):
This script does NOT import, call, or reuse any existing plot/analysis script in the
repo. It reads the raw parquet files and recomputes each claim from scratch. Agreement
between two independent code paths is evidence; disagreement is a finding.

Every claim reports: recomputed value, N (seeds actually present), 95% CI, and a
PASS / DRIFT / FAIL verdict against the value currently printed in the context pack.

Run:  .venv/Scripts/python.exe analysis/verify_poster_numbers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
pd.set_option("display.width", 200)

# Tolerance for a claim to count as confirmed, as a relative fraction.
REL_TOL = 0.05
RESULTS: list[dict] = []


# --------------------------------------------------------------------------------------
# statistics helpers (deliberately simple and explicit)
# --------------------------------------------------------------------------------------
def ci95(x: np.ndarray) -> tuple[float, float, float, int]:
    """Mean and 95% t-interval. Returns (mean, lo, hi, n)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    if n == 1:
        return float(x[0]), np.nan, np.nan, 1
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(n)
    # 1.96 is adequate at n>=30; use it uniformly and report n so the reader can judge.
    return float(m), float(m - 1.96 * se), float(m + 1.96 * se), n


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for independent samples, pooled SD. Positive => a > b."""
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    b = np.asarray(b, float)
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return np.nan
    na, nb = a.size, b.size
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return np.nan
    return float((a.mean() - b.mean()) / sp)


def cohens_d_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Bootstrap CI on Cohen's d. AGENTS hard rule: intervals, not point estimates."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    b = np.asarray(b, float)
    b = b[np.isfinite(b)]
    point = cohens_d(a, b)
    if not np.isfinite(point):
        return point, np.nan, np.nan
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = cohens_d(rng.choice(a, a.size, replace=True), rng.choice(b, b.size, replace=True))
    boots = boots[np.isfinite(boots)]
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def record(claim: str, claimed, actual, n, lo=np.nan, hi=np.nan, note: str = "") -> None:
    """Compare a recomputed value against what the context pack currently states."""
    if claimed is None or not np.isfinite(actual):
        verdict = "INFO"
    else:
        denom = abs(claimed) if abs(claimed) > 1e-9 else 1.0
        rel = abs(actual - claimed) / denom
        verdict = "PASS" if rel <= REL_TOL else ("DRIFT" if rel <= 0.25 else "FAIL")
    RESULTS.append(
        dict(claim=claim, claimed=claimed, actual=round(float(actual), 4) if np.isfinite(actual) else np.nan,
             n=n, lo=round(lo, 4) if np.isfinite(lo) else np.nan,
             hi=round(hi, 4) if np.isfinite(hi) else np.nan, verdict=verdict, note=note)
    )


def load(rel: str) -> pd.DataFrame:
    return pd.read_parquet(ROOT / rel)


def normalise_by_baseline(df: pd.DataFrame, value_col: str, sev_col: str,
                          group_cols: list[str], baseline_sev=0.0) -> pd.DataFrame:
    """
    Normalise `value_col` by the clean-baseline (severity == baseline_sev) value.

    Two normalisation conventions are computed, because the choice is a judgement call
    and a claim that survives only one of them is weaker than it looks:
      norm_pooled : value / mean(baseline over all seeds in the group)
      norm_paired : value / baseline value of the SAME seed
    """
    base = df[np.isclose(df[sev_col], baseline_sev)]
    pooled = base.groupby(group_cols, dropna=False)[value_col].mean().rename("base_pooled")
    paired = (base.groupby(group_cols + ["seed"], dropna=False)[value_col]
                  .mean().rename("base_paired"))
    out = df.merge(pooled, on=group_cols, how="left").merge(paired, on=group_cols + ["seed"], how="left")
    out["norm_pooled"] = out[value_col] / out["base_pooled"]
    out["norm_paired"] = out[value_col] / out["base_paired"]
    return out


# ======================================================================================
# CLAIM BLOCK 1 — Coverage gap vs class imbalance (the spatial-hole contrast)
# ======================================================================================
def verify_imbalance() -> None:
    print("\n" + "=" * 90)
    print("BLOCK 1 — coverage gap (spatial) vs class imbalance (random)")
    print("  source: defect_expansion_experiment/results_imbalance.parquet")
    df = load("defect_expansion_experiment/results_imbalance.parquet")
    print(f"  rows={len(df)}  frac={sorted(df.frac.unique())}  "
          f"tc={sorted(df.tc.unique())}  seeds={df.seed.nunique()}")

    # The spatial arm has no frac=0 row of its own in some grids; baseline is the
    # structure-agnostic frac=0 cell. Verify that assumption rather than assume it.
    base_rows = df[np.isclose(df.frac, 0.0)]
    print(f"  baseline (frac=0) rows={len(base_rows)} "
          f"structures={sorted(base_rows.structure.unique())} "
          f"mean spread={base_rows.mean_dist.mean():.4f}")

    nd = normalise_by_baseline(df, "mean_dist", "frac", group_cols=["tc"])

    claimed_spatial = {0.25: 1.07, 0.50: 1.19, 0.70: 1.23, 0.85: 1.34, 0.95: 1.37}
    claimed_random = {0.25: 1.03, 0.50: 1.07, 0.70: 1.13, 0.85: 1.19, 0.95: 1.24}

    print(f"\n  {'frac':>5} | {'random (n)':>22} | {'spatial (n)':>22} | CIs separate?")
    for frac in sorted(claimed_spatial):
        row = {}
        for struct in ("random", "spatial"):
            sub = nd[(np.isclose(nd.frac, frac)) & (nd.structure == struct)]
            m, lo, hi, n = ci95(sub["norm_pooled"].values)
            row[struct] = (m, lo, hi, n)
            claimed = claimed_spatial[frac] if struct == "spatial" else claimed_random[frac]
            record(f"imbalance norm-spread {struct} @frac={frac}", claimed, m, n, lo, hi)
        sep = (row["spatial"][1] > row["random"][2]) if np.isfinite(row["spatial"][1]) else False
        print(f"  {frac:>5} | {row['random'][0]:.3f} [{row['random'][1]:.3f},{row['random'][2]:.3f}] "
              f"(n={row['random'][3]:>3}) | "
              f"{row['spatial'][0]:.3f} [{row['spatial'][1]:.3f},{row['spatial'][2]:.3f}] "
              f"(n={row['spatial'][3]:>3}) | {'YES' if sep else 'no'}")
        record(f"imbalance CIs separate @frac={frac}", None, 1.0 if sep else 0.0, row["spatial"][3],
               note="CLAIM: separate from frac>=0.5" if frac >= 0.5 else "CLAIM: overlap")

    # Sensitivity: does the conclusion survive the other normalisation convention?
    print("\n  sensitivity — paired-seed normalisation instead of pooled:")
    for frac in (0.50, 0.95):
        vals = {}
        for struct in ("random", "spatial"):
            sub = nd[(np.isclose(nd.frac, frac)) & (nd.structure == struct)]
            vals[struct] = ci95(sub["norm_paired"].values)
        sep = vals["spatial"][1] > vals["random"][2] if np.isfinite(vals["spatial"][1]) else False
        print(f"    frac={frac}: random {vals['random'][0]:.3f}  spatial {vals['spatial'][0]:.3f}  "
              f"separate={'YES' if sep else 'no'}")

    # Minority recall + accuracy
    print("\n  minority-class recall and overall accuracy:")
    for frac in (0.25, 0.95):
        for struct in ("random", "spatial"):
            sub = df[(np.isclose(df.frac, frac)) & (df.structure == struct)]
            r, rlo, rhi, n = ci95(sub.min_recall.values)
            a, _, _, _ = ci95(sub.vacc.values)
            print(f"    frac={frac} {struct:<8} recall={r:.3f} [{rlo:.3f},{rhi:.3f}]  acc={a:.3f}  n={n}")
    for frac, struct, claimed in [(0.25, "spatial", 0.77), (0.95, "spatial", 0.14),
                                  (0.25, "random", 0.91), (0.95, "random", 0.61)]:
        sub = df[(np.isclose(df.frac, frac)) & (df.structure == struct)]
        m, lo, hi, n = ci95(sub.min_recall.values)
        record(f"min_recall {struct} @frac={frac}", claimed, m, n, lo, hi)
    sub = df[(np.isclose(df.frac, 0.95)) & (df.structure == "spatial")]
    m, lo, hi, n = ci95(sub.vacc.values)
    record("spatial test accuracy @frac=0.95", 0.71, m, n, lo, hi)


# ======================================================================================
# CLAIM BLOCK 2 — Shortcut feature, per-axis displacement
# ======================================================================================
def verify_shortcut() -> None:
    print("\n" + "=" * 90)
    print("BLOCK 2 — shortcut feature: per-axis displacement (spur_frac)")
    claimed = {
        ("tree", "iris"): {0.0: 0.108, 2.0: 0.262, 4.0: 0.321, 8.0: 0.500},
        ("tree", "wine"): {0.0: 0.002, 2.0: 0.082, 4.0: 0.197, 8.0: 1.000},
        ("svm", "iris"): {0.0: 0.015, 2.0: 0.127, 4.0: 0.354, 8.0: 0.230},
        ("svm", "wine"): {0.0: 0.024, 2.0: 0.226, 4.0: 0.455, 8.0: 0.365},
    }
    for tag, rel in [("tree", "defect_expansion_experiment/results_shortcut.parquet"),
                     ("svm", "defect_expansion_experiment/results_shortcut_svm.parquet")]:
        df = load(rel)
        print(f"\n  [{tag}] source: {rel}  rows={len(df)}  seeds={df.seed.nunique()}")
        print(f"  {'corr':>5} | {'dataset':<5} | {'spur_frac (95% CI)':>28} | {'vacc':>6} | "
              f"{'spread_m0':>9} | n")
        for ds in ("iris", "wine"):
            for corr in sorted(claimed[(tag, ds)]):
                # NB: df["corr"] not df.corr -- the latter is DataFrame.corr(), a method.
                sub = df[(df.dataset == ds) & (np.isclose(df["corr"], corr))]
                m, lo, hi, n = ci95(sub.spur_frac.values)
                a, _, _, _ = ci95(sub.vacc.values)
                s, _, _, _ = ci95(sub.spread_m0.values)
                print(f"  {corr:>5} | {ds:<5} | {m:.3f} [{lo:.3f},{hi:.3f}] "
                      f"{'':>6} | {a:.3f} | {s:>9.3f} | {n}")
                record(f"shortcut {tag}/{ds} spur_frac @corr={corr}",
                       claimed[(tag, ds)][corr], m, n, lo, hi)

        # Falsifiable check: is the rise actually monotone, as claimed?
        for ds in ("iris", "wine"):
            seq = [df[(df.dataset == ds) & (np.isclose(df["corr"], c))].spur_frac.mean()
                   for c in [0.0, 0.5, 1.0, 2.0, 4.0]]
            mono = all(seq[i] <= seq[i + 1] + 1e-9 for i in range(len(seq) - 1))
            record(f"shortcut {tag}/{ds} monotone over corr 0..4", None, 1.0 if mono else 0.0,
                   len(seq), note=f"seq={[round(v,3) for v in seq]} CLAIM: monotone rise")
            print(f"    monotone 0..4 on {ds}: {'YES' if mono else 'NO'}  "
                  f"{[round(v,3) for v in seq]}")

        # The claim that scalar spread is NULL on iris must also be falsifiable.
        for ds in ("iris", "wine"):
            base = df[(df.dataset == ds) & (np.isclose(df["corr"], 0.0))].spread_m0.mean()
            for corr in (2.0, 4.0, 8.0):
                sub = df[(df.dataset == ds) & (np.isclose(df["corr"], corr))]
                m, lo, hi, n = ci95((sub.spread_m0 / base).values)
                record(f"shortcut {tag}/{ds} scalar spread ratio @corr={corr}", None, m, n, lo, hi,
                       note="CLAIM: null on iris (within +-5%)")


# ======================================================================================
# CLAIM BLOCK 3 — The dimensional limit (fit vs measure)
# ======================================================================================
def verify_fit_vs_measure() -> None:
    print("\n" + "=" * 90)
    print("BLOCK 3 — dimensional limit: fit-time vs measure-time loss")
    df = load("defect_expansion_experiment/results_fitvsmeasure.parquet")
    print(f"  rows={len(df)}  m={sorted(df.m.unique())}  frac={sorted(df.frac.unique())}  "
          f"seeds={df.seed.nunique()}")

    claimed = {("spread_measure_full", 0): 1.96, ("spread_measure_full", 100): 0.11,
               ("spread_fit_full", 0): 1.96, ("spread_fit_full", 100): 0.23,
               ("spread_fit_real", 0): 1.96, ("spread_fit_real", 100): 1.96}
    frac = 0.5
    print(f"\n  Cohen's d (spatial vs random) at frac={frac}, with bootstrap 95% CI:")
    print(f"  {'pipeline':<20} | {'m':>4} | {'d [95% CI]':>26} | n_spatial/n_random")
    for col in ("spread_measure_full", "spread_fit_full", "spread_fit_real"):
        for m_dim in sorted(df.m.unique()):
            sub = df[(df.m == m_dim) & (np.isclose(df.frac, frac))]
            a = sub[sub.structure == "spatial"][col].values
            b = sub[sub.structure == "random"][col].values
            d, dlo, dhi = cohens_d_ci(a, b)
            flag = " *" if (m_dim in (0, 100)) else ""
            print(f"  {col:<20} | {m_dim:>4} | {d:+.2f} [{dlo:+.2f},{dhi:+.2f}]{'':>6} | "
                  f"{np.isfinite(a).sum()}/{np.isfinite(b).sum()}{flag}")
            if (col, m_dim) in claimed:
                record(f"fitvsmeasure {col} d @m={m_dim}", claimed[(col, m_dim)], d,
                       int(np.isfinite(a).sum()), dlo, dhi)

    # The headline is a CONTRAST, so test it directly: does fit_real hold while raw dies?
    sub0 = df[(df.m == 0) & (np.isclose(df.frac, frac))]
    sub100 = df[(df.m == 100) & (np.isclose(df.frac, frac))]
    d_real_100 = cohens_d(sub100[sub100.structure == "spatial"].spread_fit_real.values,
                          sub100[sub100.structure == "random"].spread_fit_real.values)
    d_raw_100 = cohens_d(sub100[sub100.structure == "spatial"].spread_measure_full.values,
                         sub100[sub100.structure == "random"].spread_measure_full.values)
    holds = (d_real_100 > 1.0) and (abs(d_raw_100) < 0.5)
    record("fitvsmeasure HEADLINE: pre-fit selection holds while raw dies", None,
           1.0 if holds else 0.0, len(sub100),
           note=f"d_fit_real@m100={d_real_100:+.2f}, d_raw@m100={d_raw_100:+.2f}")
    print(f"\n  HEADLINE CHECK: fit_real d@m=100 = {d_real_100:+.2f}, "
          f"raw d@m=100 = {d_raw_100:+.2f} -> {'HOLDS' if holds else 'DOES NOT HOLD'}")


# ======================================================================================
# CLAIM BLOCK 4 — Wine: pre-fit feature selection recovers the signal
# ======================================================================================
def verify_wine_h4() -> None:
    print("\n" + "=" * 90)
    print("BLOCK 4 — wine: pre-fit feature selection (+48% at k=2)")
    df = load("defect_expansion_experiment/results_wine_h4.parquet")
    print(f"  rows={len(df)}  k={sorted(df.k.unique())}  datasets={sorted(df.dataset.unique())}  "
          f"seeds={df.seed.nunique()}")
    wine = df[df.dataset == "wine"]
    nd = normalise_by_baseline(wine, "mean_dist", "frac", group_cols=["k"])
    claimed_spatial = {2: 1.48, 3: 1.11, 4: 1.06, 6: 1.01, 13: 0.99}
    claimed_d = {2: 2.04, 3: 1.47, 4: 1.51, 6: 0.38, 13: -0.12}
    frac = 0.5
    print(f"\n  {'k':>3} | {'spatial x base':>22} | {'random x base':>22} | {'d [95% CI]':>24} | acc")
    for k in sorted(claimed_spatial):
        sub = nd[(nd.k == k) & (np.isclose(nd.frac, frac))]
        sp = sub[sub.structure == "spatial"]
        rd = sub[sub.structure == "random"]
        ms, los, his, ns = ci95(sp["norm_pooled"].values)
        mr, lor, hir, nr = ci95(rd["norm_pooled"].values)
        d, dlo, dhi = cohens_d_ci(sp["norm_pooled"].values, rd["norm_pooled"].values)
        acc = wine[(wine.k == k) & (np.isclose(wine.frac, 0.0))].vacc.mean()
        print(f"  {k:>3} | {ms:.3f} [{los:.3f},{his:.3f}] (n={ns:>2}) | "
              f"{mr:.3f} [{lor:.3f},{hir:.3f}] (n={nr:>2}) | "
              f"{d:+.2f} [{dlo:+.2f},{dhi:+.2f}] | {acc:.3f}")
        record(f"wine k={k} spatial norm-spread", claimed_spatial[k], ms, ns, los, his)
        record(f"wine k={k} Cohen's d", claimed_d[k], d, ns, dlo, dhi)


# ======================================================================================
# CLAIM BLOCK 5 — Train-test leakage
# ======================================================================================
def verify_leakage() -> None:
    print("\n" + "=" * 90)
    print("BLOCK 5 — train-test leakage: accuracy fooled, geometry blind")
    df = load("defect_expansion_experiment/results_leakage.parquet")
    print(f"  rows={len(df)}  n_leak={sorted(df.n_leak.unique())}  seeds={df.seed.nunique()}")
    claimed = {("iris", 0): (0.943, 0.943), ("iris", 10): (0.957, 0.943), ("iris", 20): (0.966, 0.943)}
    print(f"\n  {'dataset':<5} | {'n_leak':>6} | {'measured acc':>22} | {'honest acc':>22} | n")
    for ds in ("iris", "wine"):
        for nl in sorted(df.n_leak.unique()):
            sub = df[(df.dataset == ds) & (df.n_leak == nl)]
            mv, mlo, mhi, n = ci95(sub.vacc.values)
            cv, clo, chi, _ = ci95(sub.vacc_clean.values)
            print(f"  {ds:<5} | {nl:>6} | {mv:.4f} [{mlo:.4f},{mhi:.4f}] | "
                  f"{cv:.4f} [{clo:.4f},{chi:.4f}] | {n}")
            if (ds, nl) in claimed:
                record(f"leakage {ds} measured acc @n_leak={nl}", claimed[(ds, nl)][0], mv, n, mlo, mhi)
                record(f"leakage {ds} honest acc @n_leak={nl}", claimed[(ds, nl)][1], cv, n, clo, chi)

    # The null must be falsifiable too: do leaked and clean margins actually overlap?
    print("\n  margin at leaked vs clean points (the H1 null):")
    for ds in ("iris", "wine"):
        for nl in (5, 20):
            sub = df[(df.dataset == ds) & (df.n_leak == nl)]
            ml, llo, lhi, n = ci95(sub.margin_leak.values)
            mc, clo, chi, _ = ci95(sub.margin_clean.values)
            overlap = not (llo > chi or lhi < clo)
            print(f"    {ds} n_leak={nl}: leak {ml:.3f} [{llo:.3f},{lhi:.3f}]  "
                  f"clean {mc:.3f} [{clo:.3f},{chi:.3f}]  overlap={'YES' if overlap else 'NO'}")
            record(f"leakage {ds} margin null @n_leak={nl}", None, 1.0 if overlap else 0.0, n,
                   note="CLAIM: CIs overlap (no locality signal)")


# ======================================================================================
# CLAIM BLOCK 6 — Negative controls
# ======================================================================================
def verify_healthy_control() -> None:
    print("\n" + "=" * 90)
    print("BLOCK 6 — negative controls (the credibility check)")
    df = load("defect_expansion_experiment/results_healthy_control.parquet")
    print(f"  rows={len(df)}  conditions={sorted(df.dataset.unique())}  seeds={df.seed.nunique()}")
    claimed = {"iris_contested": 2.23, "iris_separated": -0.05, "blobs_healthy": -0.26, "rand_null": 0.29}
    frac = 0.5
    print(f"\n  {'condition':<16} | {'clean acc':>9} | {'spatial':>7} | {'random':>7} | "
          f"{'d [95% CI]':>24} | recall delta")
    for cond in ["iris_contested", "iris_separated", "blobs_healthy", "rand_null"]:
        sub_all = df[df.dataset == cond]
        nd = normalise_by_baseline(sub_all, "mean_dist", "frac", group_cols=["dataset"])
        sub = nd[np.isclose(nd.frac, frac)]
        sp = sub[sub.structure == "spatial"]["norm_pooled"].values
        rd = sub[sub.structure == "random"]["norm_pooled"].values
        d, dlo, dhi = cohens_d_ci(sp, rd)
        acc = sub_all[np.isclose(sub_all.frac, 0.0)].vacc.mean()
        r_sp = sub_all[(np.isclose(sub_all.frac, frac)) & (sub_all.structure == "spatial")].min_recall.mean()
        r_base = sub_all[np.isclose(sub_all.frac, 0.0)].min_recall.mean()
        print(f"  {cond:<16} | {acc:>9.3f} | {np.nanmean(sp):>7.3f} | {np.nanmean(rd):>7.3f} | "
              f"{d:+.2f} [{dlo:+.2f},{dhi:+.2f}] | {r_sp - r_base:+.3f}")
        record(f"negative control {cond} Cohen's d", claimed[cond], d, len(sp), dlo, dhi)

    print("\n  TIGHTEST CONTROL: iris_contested vs iris_separated (same data, same dim,")
    print("  same manipulation, only the deleted class differs).")


# ======================================================================================
# CLAIM BLOCK 7 — Transfer across model families (coverage gap)
# ======================================================================================
def verify_model_family() -> None:
    print("\n" + "=" * 90)
    print("BLOCK 7 — transfer: coverage gap across model families")
    df = load("model_family_experiment/results.parquet")
    cg = df[df.defect == "coverage_gap"]
    print(f"  rows={len(cg)}  models={sorted(cg.model.unique())}  attacks={sorted(cg.attack.unique())}")
    print("  NOTE: this grid uses BEFORE-SPLIT injection -- see POSTER_CONTEXT.md section 3.")
    print(f"\n  {'model':<6} | {'bias':>5} | {'spread (95% CI)':>26} | {'acc':>6} | n_valid | n_NaN")
    for model in sorted(cg.model.unique()):
        for bias in sorted(cg.bias.dropna().unique()):
            sub = cg[(cg.model == model) & (np.isclose(cg.bias, bias))]
            m, lo, hi, n = ci95(sub.mean_dist.values)
            a, _, _, _ = ci95(sub.vacc.values)
            n_nan = int(sub.mean_dist.isna().sum())
            print(f"  {model:<6} | {bias:>5} | {m:.3f} [{lo:.3f},{hi:.3f}] (n={n:>2}) | "
                  f"{a:.3f} | {n:>7} | {n_nan:>5}")
    claimed_rf = {0.1: 0.657, 0.9: 0.793}
    for bias, c in claimed_rf.items():
        sub = cg[(cg.model == "rf") & (np.isclose(cg.bias, bias))]
        m, lo, hi, n = ci95(sub.mean_dist.values)
        record(f"model-family rf spread @bias={bias}", c, m, n, lo, hi,
               note="BEFORE-SPLIT protocol; low n per cell")


# ======================================================================================
# CLAIM BLOCK 8 — Outlier defect
# ======================================================================================
def verify_outlier() -> None:
    print("\n" + "=" * 90)
    print("BLOCK 8 — outlier: white-box only, within a distance window")
    df = load("outlier_experiment/results_extended.parquet")
    print(f"  rows={len(df)}  models={sorted(df.model.unique())}  attacks={sorted(df.attack.unique())}")
    print(f"\n  {'model':<9} | {'attack':<4} | {'dir':<8} | {'k':>4} | "
          f"{'norm spread (95% CI)':>26} | {'acc':>6} | n")
    for model in ("tree_d3", "tree_d10", "svm"):
        sub_m = df[df.model == model]
        if sub_m.empty:
            continue
        base = sub_m[(sub_m.direction == "none") | (sub_m.n_out == 0)].mean_dist.mean()
        if not np.isfinite(base):
            base = sub_m[np.isclose(sub_m.k, 0.0)].mean_dist.mean()
        for direction in ("toward", "outward"):
            for k in (4.0, 8.0, 12.0):
                sub = sub_m[(sub_m.direction == direction) & (np.isclose(sub_m.k, k))
                            & (sub_m.n_out == 10)]
                if sub.empty:
                    continue
                m, lo, hi, n = ci95((sub.mean_dist / base).values)
                a, _, _, _ = ci95(sub.vacc.values)
                print(f"  {model:<9} | {sub.attack.iloc[0]:<4} | {direction:<8} | {k:>4} | "
                      f"{m:.3f} [{lo:.3f},{hi:.3f}] {'':>4} | {a:.3f} | {n}")
                if model == "tree_d3" and direction == "toward" and k == 8.0:
                    record("outlier tree_d3 toward k=8 norm spread", 1.36, m, n, lo, hi)
                if model == "svm" and direction == "outward" and k == 8.0:
                    record("outlier svm outward k=8 norm spread (claimed null)", 1.00, m, n, lo, hi)


# ======================================================================================
def main() -> int:
    print("INDEPENDENT VERIFICATION OF POSTER NUMBERS")
    print(f"repo: {ROOT}")
    print("No existing analysis/plot script is imported. All values recomputed from parquet.")

    for fn in (verify_imbalance, verify_shortcut, verify_fit_vs_measure, verify_wine_h4,
               verify_leakage, verify_healthy_control, verify_model_family, verify_outlier):
        try:
            fn()
        except Exception as exc:  # keep going; a failed block is itself a finding
            print(f"\n!! BLOCK FAILED: {fn.__name__}: {type(exc).__name__}: {exc}")
            RESULTS.append(dict(claim=fn.__name__, claimed=None, actual=np.nan, n=0,
                                lo=np.nan, hi=np.nan, verdict="ERROR", note=str(exc)[:160]))

    res = pd.DataFrame(RESULTS)
    out = ROOT / "analysis" / "verification_report.csv"
    out.parent.mkdir(exist_ok=True)
    res.to_csv(out, index=False)

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    counts = res.verdict.value_counts().to_dict()
    for v in ("PASS", "DRIFT", "FAIL", "INFO", "ERROR"):
        if v in counts:
            print(f"  {v:<6} {counts[v]}")

    bad = res[res.verdict.isin(["DRIFT", "FAIL", "ERROR"])]
    if not bad.empty:
        print("\n  CLAIMS NEEDING ATTENTION:")
        for _, r in bad.iterrows():
            print(f"   [{r.verdict:<5}] {r.claim}")
            print(f"            claimed={r.claimed}  actual={r.actual}  n={r.n}  {r.note}")

    low_n = res[(res.n > 0) & (res.n < 10)]
    if not low_n.empty:
        print(f"\n  LOW SAMPLE SIZE (n < 10) on {len(low_n)} claims -- treat as directional only:")
        for _, r in low_n.iterrows():
            print(f"    n={r.n:<3} {r.claim}")

    print(f"\n  full table -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
