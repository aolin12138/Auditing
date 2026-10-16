"""
Two analyses produced during the poster session (2026-10-10) that are not in any FINDINGS_*.md.
Both feed the final report. Saved here so every number traces to a script.

  A. PROTOCOL COMPARISON: before-split vs training-only deletion, same seeds/model/attack.
     Shows the fix (a) made accuracy honest, (b) gave the attack its targets back,
     (c) made the spatial-vs-random spread signal CLEARER (wine: null -> separated).
  B. GENERALISATION SCORECARD: same defect (coverage gap vs random deletion at matched count),
     training-only protocol, varying only model / attack / dataset.
     Shows spread fails to generalise across attacks and models even on 4-D iris.

Run:  .venv/Scripts/python.exe analysis/extra_evidence.py
Out:  analysis/extra_evidence.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "defect_expansion_experiment"
RNG = np.random.default_rng(0)
ROWS = []


def cohens_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan


def d_with_ci(a, b, n_boot=2000):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    point = cohens_d(a, b)
    if not np.isfinite(point):
        return point, np.nan, np.nan, len(a), len(b)
    boots = np.array([cohens_d(RNG.choice(a, len(a)), RNG.choice(b, len(b))) for _ in range(n_boot)])
    boots = boots[np.isfinite(boots)]
    if len(boots) == 0:
        return point, np.nan, np.nan, len(a), len(b)
    return point, np.percentile(boots, 2.5), np.percentile(boots, 97.5), len(a), len(b)


def ci95(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    return m, m - 1.96 * se, m + 1.96 * se


# ------------------------------------------------------------------------------------------
# A. Protocol comparison  (results_variance.parquet: overfit tree + white-box DTA, 30 seeds)
# ------------------------------------------------------------------------------------------
print("A. PROTOCOL COMPARISON — tree + white-box attack, contested class, 30 seeds")
v = pd.read_parquet(P / "results_variance.parquet")
print(f"{'data':<5}{'frac':>5} {'protocol':<13}{'test acc':>8}{'recall':>8}{'n_adv':>7}   spatial-vs-random d [95% CI]")
for ds, tc in [("iris", 2), ("wine", 0)]:
    for frac in (0.6, 0.9):
        for protocol in ("before_split", "train_only"):
            s = v[(v.dataset == ds) & (v.protocol == protocol) & np.isclose(v.frac, frac) & (v.tc == tc)]
            sp, rd = s[s.structure == "spatial"], s[s.structure == "random"]
            acc, rec, nadv = ci95(sp.vacc)[0], ci95(sp.min_recall)[0], sp.nadv.mean()
            d, lo, hi, ns, nr = d_with_ci(sp.mean_dist.values, rd.mean_dist.values)
            sep = "separated" if lo > 0 else "overlap"
            print(f"{ds:<5}{frac:>5} {protocol:<13}{acc:>8.3f}{rec:>8.3f}{nadv:>7.1f}   {d:+.2f} [{lo:+.2f},{hi:+.2f}] {sep}")
            ROWS.append(dict(analysis="protocol", dataset=ds, setting=protocol, frac=frac, n_spatial=ns,
                             n_random=nr, test_acc_spatial=acc, recall_spatial=rec, n_adv_spatial=nadv,
                             d=d, d_lo=lo, d_hi=hi, separated=lo > 0))
base = v[(v.protocol == "train_only") & np.isclose(v.frac, 0)]
for ds in ("iris", "wine"):
    b = base[base.dataset == ds]
    print(f"  clean baseline {ds}: test acc {b.vacc.mean():.3f}, recall {b.min_recall.mean():.3f}, n_adv {b.nadv.mean():.1f}")

# ------------------------------------------------------------------------------------------
# B. Generalisation scorecard  (training-only protocol throughout)
# ------------------------------------------------------------------------------------------
print("\nB. GENERALISATION SCORECARD — coverage gap vs random deletion, training-only")


def score(label, df, fracs, source):
    base_spread = df[np.isclose(df.frac, 0)].mean_dist.mean()
    for frac in fracs:
        s = df[np.isclose(df.frac, frac)]
        a, b = s[s.structure == "spatial"].mean_dist.values, s[s.structure == "random"].mean_dist.values
        d, lo, hi, na, nb = d_with_ci(a, b)
        ok = np.isfinite(lo) and lo > 0
        ci = f"[{lo:+.2f},{hi:+.2f}]" if np.isfinite(lo) else "[  n/a  ]"
        print(f"{label:<32}{frac:>5}{na:>4}/{nb:<3}{np.nanmean(a) / base_spread:>7.2f}x{np.nanmean(b) / base_spread:>7.2f}x"
              f"   d {d:+.2f} {ci:<16} {'separated' if ok else 'no'}")
        ROWS.append(dict(analysis="scorecard", dataset=label.split("|")[0].strip(), setting=label.split("|")[1].strip(),
                         frac=frac, n_spatial=na, n_random=nb, spread_spatial=np.nanmean(a) / base_spread,
                         spread_random=np.nanmean(b) / base_spread, d=d, d_lo=lo, d_hi=hi, separated=ok, source=source))


print(f"{'setting':<32}{'frac':>5}{'n_s/n_r':>8}{'spatial':>8}{'random':>8}")
vs = pd.read_parquet(P / "results_variance_svm.parquet")
for ds, tc in [("iris", 2), ("wine", 0)]:
    for df, lab, src in [(v, "tree + white-box", "results_variance"), (vs, "SVM + black-box", "results_variance_svm")]:
        sub = df[(df.protocol == "train_only") & (df.dataset == ds) & ((df.tc == tc) | np.isclose(df.frac, 0))]
        score(f"{ds} | {lab}", sub, [0.6, 0.8, 0.9], src)
p1 = pd.read_parquet(P / "results_p1.parquet")
for model, lab in [("tree", "tree + black-box"), ("svm", "SVM + black-box (p1)")]:
    sub = p1[(p1.model == model) & ((p1.tc == 2) | (p1.structure == "baseline"))]
    score(f"iris | {lab}", sub, [0.5, 0.85, 0.95], "results_p1")
rf = pd.read_parquet(P / "results_p1_rf.parquet")
score("iris | forest + black-box", rf[(rf.tc == 2) | (rf.structure == "baseline")], [0.5, 0.85, 0.95], "results_p1_rf")
print("NOTE: forest cells have 2-3 valid runs (HopSkipJump hangs) -> not interpretable; their CIs are degenerate.")

out = ROOT / "analysis" / "extra_evidence.csv"
pd.DataFrame(ROWS).to_csv(out, index=False)
print(f"\n-> {out}")
