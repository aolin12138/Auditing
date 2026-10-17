"""Every number quoted in the running text of the Results section (report Section 4).

Writes analysis/results_numbers.csv: one row per quoted value (id, value, ci_lo, ci_hi, n, source).
Section 4 quotes these values; nothing in the prose is computed by hand. Figures and tables come from
analysis/figures.py; this file covers the numbers the text cites around them.

Run: .venv/Scripts/python.exe analysis/results_numbers.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats import ci95, d_with_ci  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEX, OUTL = ROOT / "defect_expansion_experiment", ROOT / "outlier_experiment"
TC = {"iris": 2, "wine": 0}
ROWS = []


def put(id_, val, lo=np.nan, hi=np.nan, n=np.nan, src=""):
    ROWS.append(dict(id=id_, value=val, ci_lo=lo, ci_hi=hi, n=n, source=src))


def mean_ci(id_, x, src):
    m, lo, hi = ci95(x); put(id_, m, lo, hi, int(np.isfinite(np.asarray(x, float)).sum()), src)


def dval(id_, a, b, src):
    d, lo, hi, na, nb = d_with_ci(a, b); put(id_, d, lo, hi, min(na, nb), src)


# ---- 4.1 coverage gap (results_variance, training-only, tree + DTA) --------------------------------
v = pd.read_parquet(DEX / "results_variance.parquet"); v = v[v.protocol == "train_only"]
for ds in ("iris", "wine"):
    s = v[(v.dataset == ds) & ((v.tc == TC[ds]) | np.isclose(v.frac, 0))]
    base = s[np.isclose(s.frac, 0)]
    mean_ci(f"cg.{ds}.clean.acc", base.vacc, "results_variance")
    mean_ci(f"cg.{ds}.clean.recall", base.min_recall, "results_variance")
    for frac in (0.4, 0.6, 0.9):
        for st in ("spatial", "random"):
            x = s[np.isclose(s.frac, frac) & (s.structure == st)]
            mean_ci(f"cg.{ds}.{st}.{int(frac*100)}.spread_ratio", x.mean_dist / base.mean_dist.mean(), "results_variance")
            mean_ci(f"cg.{ds}.{st}.{int(frac*100)}.recall", x.min_recall, "results_variance")
            mean_ci(f"cg.{ds}.{st}.{int(frac*100)}.acc", x.vacc, "results_variance")
        x = s[np.isclose(s.frac, frac)]
        dval(f"cg.{ds}.{int(frac*100)}.d_spread_sp_vs_rd", x[x.structure == "spatial"].mean_dist,
             x[x.structure == "random"].mean_dist, "results_variance")
# SVM + HopSkipJump (30-seed rerun)
vs = pd.read_parquet(DEX / "results_variance_svm.parquet"); vs = vs[vs.protocol == "train_only"]
for ds in ("iris", "wine"):
    for frac in (0.6, 0.8, 0.9):
        x = vs[(vs.dataset == ds) & (vs.tc == TC[ds]) & np.isclose(vs.frac, frac)]
        dval(f"cg.svm.{ds}.{int(frac*100)}.d_spread_sp_vs_rd", x[x.structure == "spatial"].mean_dist,
             x[x.structure == "random"].mean_dist, "results_variance_svm")
# a second, independent run of the same SVM + HopSkipJump configuration (run_attack_all.py --model svm):
# HopSkipJump is unseeded in ART 1.20.1, so the two runs differ
aas = pd.read_parquet(DEX / "results_attack_all_svm.parquet")
for ds in ("iris", "wine"):
    for frac in (0.6, 0.8, 0.9):
        x = aas[(aas.dataset == ds) & np.isclose(aas.frac, frac)]
        dval(f"cg.svm_run2.{ds}.{int(frac*100)}.d_spread_sp_vs_rd", x[x.structure == "spatial"].spread_all,
             x[x.structure == "random"].spread_all, "results_attack_all_svm")
# run-to-run variation of HopSkipJump (Discussion 5.5): wine, 80% deletion, mean spatial-minus-random
# spread on the 15 seeds of the original (archived) grid vs the same seeds in the 2026-10-15 rerun.
# Both files use the earlier correct-only probe, so the comparison isolates the attack's randomness.
snap = pd.read_parquet(ROOT / "results_archive" / "pre-rerun-2026-10-15" / "defect_expansion_experiment"
                       / "results_variance_svm.parquet")
snap = snap[snap.protocol == "train_only"]
rr = pd.read_parquet(ROOT / "rerun" / "2026-10-15" / "defect_expansion_experiment" / "results_variance_svm.parquet")
rr = rr[rr.protocol == "train_only"]
for lab, df in (("original", snap), ("rerun_same15", rr[rr.seed.isin(snap.seed.unique())])):
    x = df[(df.dataset == "wine") & (df.tc == 0) & np.isclose(df.frac, 0.8)]
    gap = (x[x.structure == "spatial"].set_index("seed").mean_dist - x[x.structure == "random"].set_index("seed").mean_dist)
    mean_ci(f"hsj_rerun.wine.80.gap.{lab}", gap, "results_variance_svm (archive vs rerun)")

# recall vs spread in the same runs (tree + DTA): hole vs random and hole vs defect-free, and the
# number of adversarial examples per fold (every test point attacked; prediction changed)
for ds in ("iris", "wine"):
    s = v[(v.dataset == ds) & ((v.tc == TC[ds]) | np.isclose(v.frac, 0))]
    base = s[np.isclose(s.frac, 0)]
    mean_ci(f"cg.{ds}.clean.nadv", base.nadv, "results_variance")
    for frac in (0.4, 0.6, 0.8, 0.9):
        x = s[np.isclose(s.frac, frac)]
        sp, rd = x[x.structure == "spatial"], x[x.structure == "random"]
        dval(f"cg.{ds}.{int(frac*100)}.d_recall_rd_vs_sp", rd.min_recall, sp.min_recall, "results_variance")
        dval(f"cg.{ds}.{int(frac*100)}.d_spread_sp_vs_clean", sp.mean_dist, base.mean_dist, "results_variance")
        dval(f"cg.{ds}.{int(frac*100)}.d_recall_clean_vs_sp", base.min_recall, sp.min_recall, "results_variance")
        dval(f"cg.{ds}.{int(frac*100)}.d_spread_sp_vs_rd.correct_only", sp.mean_dist_correct, rd.mean_dist_correct,
             "results_variance (*_correct)")
        mean_ci(f"cg.{ds}.spatial.{int(frac*100)}.nadv", sp.nadv, "results_variance")
        mean_ci(f"cg.{ds}.spatial.{int(frac*100)}.nadv_correct", sp.nadv_correct, "results_variance")
        # paired per seed: (hole - random) spread, in units of the defect-free spread
        g = (sp.set_index("seed").mean_dist - rd.set_index("seed").mean_dist) / base.mean_dist.mean()
        mean_ci(f"cg.{ds}.{int(frac*100)}.paired_gap_ratio", g, "results_variance")

# hole cut along petal length (unstandardised iris; Johnson's setting) - every point attacked
aa = pd.read_parquet(DEX / "results_attack_all.parquet")
pl_ = aa[aa.dataset == "iris_plen"]
pb = pl_[np.isclose(pl_.frac, 0)]
for frac in (0.4, 0.6, 0.8, 0.9):
    x = pl_[np.isclose(pl_.frac, frac)]
    sp, rd = x[x.structure == "spatial"], x[x.structure == "random"]
    dval(f"plen.{int(frac*100)}.d_spread_sp_vs_rd", sp.spread_all, rd.spread_all, "results_attack_all (iris_plen)")
    dval(f"plen.{int(frac*100)}.d_recall_rd_vs_sp", rd.min_recall, sp.min_recall, "results_attack_all (iris_plen)")
    mean_ci(f"plen.spatial.{int(frac*100)}.recall", sp.min_recall, "results_attack_all (iris_plen)")
    mean_ci(f"plen.random.{int(frac*100)}.recall", rd.min_recall, "results_attack_all (iris_plen)")
    mean_ci(f"plen.spatial.{int(frac*100)}.spread_ratio", sp.spread_all / pb.spread_all.mean(), "results_attack_all (iris_plen)")
    mean_ci(f"plen.random.{int(frac*100)}.spread_ratio", rd.spread_all / pb.spread_all.mean(), "results_attack_all (iris_plen)")

# Johnson Zhang's fixed-cohort check (his branch experiments/negative-controls, robustness_cohort.csv):
# iris, hole along petal length, raw features, tree + DTA; spread measured on the test points that are
# correctly classified in EVERY arm of a seed, so all arms measure the same points
import io, subprocess  # noqa: E402
# Joint cohort = test points correctly classified by EVERY (structure, severity) model of a fold. Files are
# wide (one row per seed). Ratios use the mean defect-free spread as base, as in his robustness_tree_moderate.py.
try:
    for fname, tag in (("robustness_joint_cohort_moderate", "moderate"), ("robustness_joint_cohort", "high")):
        src = f"johnson/negative-controls:{fname}.csv"
        jz = pd.read_csv(io.StringIO(subprocess.run(
            ["git", "show", f"johnson/negative-controls:defect_expansion_experiment/{fname}.csv"],
            cwd=ROOT, capture_output=True, text=True, check=True).stdout))
        base = jz["random_0.0"].mean()
        for f in sorted({c.split("_")[1] for c in jz.columns if c.startswith("spatial_")}, key=float):
            sp, rd = jz[f"spatial_{f}"] / base, jz[f"random_{f}"] / base
            pct = int(round(float(f) * 100))
            mean_ci(f"jz.{tag}.{pct}.paired_gap_ratio", sp - rd, src)
            mean_ci(f"jz.{tag}.{pct}.spatial_ratio", sp, src)
            mean_ci(f"jz.{tag}.{pct}.random_ratio", rd, src)
        put(f"jz.{tag}.n_cohort.min", jz.n_cohort.min(), src=src); put(f"jz.{tag}.n_cohort.max", jz.n_cohort.max(), src=src)
except (subprocess.CalledProcessError, FileNotFoundError) as e:
    print("  (Johnson's branch not fetched; skipping jz.* values)", e)

# Aidan's distance formula vs the corrected one, on a June 2026 rerun of his pipeline that recorded all
# three metric versions on the same runs (aiden_original/metric_validation/)
am = pd.read_parquet(ROOT / "aiden_original" / "metric_validation" / "validation_metric_fix_more.parquet")
am = am.dropna(subset=["old_buggy_inverse", "only_norm_fixed_inverse", "both_fixed_mean_distance"])
# Correlate WITHIN each experiment (dataset x bias direction). Pooling iris with wine gives r ~ 0.87 for
# the buggy formula only because wine points are both farther from the origin and more spread out.
src_am = "aiden_original/metric_validation/validation_metric_fix_more.parquet"
rb, rf = [], []
for (dsn, bt), g in am.groupby([am.dataset.str.split("_").str[0], "bias_type"]):
    rb.append(np.corrcoef(g.old_buggy_inverse, 1 / g.both_fixed_mean_distance)[0, 1])
    rf.append(np.corrcoef(g.only_norm_fixed_inverse, 1 / g.both_fixed_mean_distance)[0, 1])
    put(f"aidan.{dsn}.{bt}.r_buggy", rb[-1], n=len(g), src=src_am)
    put(f"aidan.{dsn}.{bt}.r_fixed", rf[-1], n=len(g), src=src_am)
put("aidan.r_buggy.min", min(rb), src=src_am); put("aidan.r_buggy.max", max(rb), src=src_am)
put("aidan.r_fixed.min", min(rf), src=src_am); put("aidan.r_fixed.max", max(rf), src=src_am)

# negative controls
hc = pd.read_parquet(DEX / "results_healthy_control.parquet")
cond = "condition" if "condition" in hc.columns else [c for c in hc.columns if hc[c].dtype == object and "struct" not in c][0]
for c in sorted(hc[cond].unique()):
    x = hc[hc[cond] == c]
    fr = sorted(x.frac.unique())[-1] if "frac" in x.columns else None
    x = x[np.isclose(x.frac, fr)] if fr is not None else x
    dval(f"ctrl.{c}.d_spread_sp_vs_rd", x[x.structure == "spatial"].mean_dist, x[x.structure == "random"].mean_dist,
         f"results_healthy_control (frac={fr})")

# ---- 4.2 outlier, label noise, leakage --------------------------------------------------------------
kp, kb = pd.read_parquet(OUTL / "results_kplateau.parquet"), pd.read_parquet(OUTL / "results_kplateau_baseline.parquet")
for m in ("tree_d3", "tree_d10"):
    base = kb[kb.model == m].mean_dist.mean()
    for k in sorted(kp.k.unique()):
        mean_ci(f"out.{m}.k{int(k)}.ratio", kp[(kp.model == m) & (kp.k == k)].mean_dist / base, "results_kplateau(+_baseline)")
    mean_ci(f"out.{m}.clean.acc", kb[kb.model == m].vacc, "results_kplateau_baseline")
    mean_ci(f"out.{m}.k8.acc", kp[(kp.model == m) & (kp.k == 8)].vacc, "results_kplateau")
ext = pd.read_parquet(OUTL / "results_extended.parquet")
if "mean_dist" in ext.columns:
    for (m, a), g in ext[ext.direction.isin(["toward", "none"])].groupby(["model", "attack"]):
        b = g[g.direction == "none"].mean_dist.mean()
        for k in sorted(g[g.direction == "toward"].k.unique()):
            x = g[(g.direction == "toward") & (g.k == k)]
            if len(x) >= 3:
                mean_ci(f"out.ext.{m}.{a}.k{k:g}.ratio", x.mean_dist / b, "results_extended")
for f, lab in (("results_label_noise", "tree"), ("results_label_noise_svm", "svm")):
    ln = pd.read_parquet(DEX / f"{f}.parquet")
    for ds in ("iris", "wine"):
        c = ln[(ln.dataset == ds) & (ln.noise == 0)]
        mean_ci(f"ln.{lab}.{ds}.clean.acc", c.vacc, f)
        for nz in (0.1, 0.2, 0.4):
            x = ln[(ln.dataset == ds) & np.isclose(ln.noise, nz)]
            mean_ci(f"ln.{lab}.{ds}.{int(nz*100)}.acc", x.vacc, f)
            mean_ci(f"ln.{lab}.{ds}.{int(nz*100)}.spread_ratio", x.mean_dist / c.mean_dist.mean(), f)
            mean_ci(f"ln.{lab}.{ds}.{int(nz*100)}.nadv", x.nadv, f)
            dval(f"ln.{lab}.{ds}.{int(nz*100)}.d_acc", x.vacc, c.vacc, f)
            dval(f"ln.{lab}.{ds}.{int(nz*100)}.d_spread", x.mean_dist, c.mean_dist, f)
        mean_ci(f"ln.{lab}.{ds}.clean.nadv", c.nadv, f)
        if "nadv_correct" in ln.columns:   # earlier probe: adversarial examples from correctly classified points
            mean_ci(f"ln.{lab}.{ds}.clean.nadv_correct", c.nadv_correct, f)
            mean_ci(f"ln.{lab}.{ds}.40.nadv_correct", ln[(ln.dataset == ds) & np.isclose(ln.noise, 0.4)].nadv_correct, f)
# ratio of accuracy-drop d to spread-rise d at every noise rate (tree), quoted as a range in 4.2
from stats import cohens_d  # noqa: E402
ln = pd.read_parquet(DEX / "results_label_noise.parquet")
# Every point attacked: spread d sits near 0, so a ratio of effect sizes is meaningless. Count instead the
# noise rates (10-90%, iris + wine = 18 cells) at which each measurement is separated from clean labels.
sep_spread, sep_acc, n_cells, max_spread_d = 0, 0, 0, -np.inf
for ds in ("iris", "wine"):
    c = ln[(ln.dataset == ds) & (ln.noise == 0)]
    for nz in sorted(ln.noise.unique())[1:]:
        x = ln[(ln.dataset == ds) & np.isclose(ln.noise, nz)]
        ds_, lo_s, hi_s, _, _ = d_with_ci(x.mean_dist, c.mean_dist)
        da, lo_a, hi_a, _, _ = d_with_ci(x.vacc, c.vacc)
        sep_spread += int(lo_s > 0 or hi_s < 0); sep_acc += int(lo_a > 0 or hi_a < 0); n_cells += 1
        max_spread_d = max(max_spread_d, ds_)
put("ln.tree.n_cells", n_cells, src="results_label_noise (10-90%, iris+wine)")
put("ln.tree.n_sep_spread", sep_spread, src="results_label_noise (10-90%, iris+wine)")
put("ln.tree.n_sep_acc", sep_acc, src="results_label_noise (10-90%, iris+wine)")
put("ln.tree.max_spread_d", max_spread_d, src="results_label_noise (10-90%, iris+wine)")

lk = pd.read_parquet(DEX / "results_leakage.parquet")
for ds in ("iris", "wine"):
    s0 = lk[(lk.dataset == ds) & (lk.n_leak == 0)].set_index("seed")
    mean_ci(f"lk.{ds}.0.acc", s0.vacc, "results_leakage")
    for n in (10, 20):
        s = lk[(lk.dataset == ds) & (lk.n_leak == n)].set_index("seed")
        mean_ci(f"lk.{ds}.{n}.acc", s.vacc, "results_leakage")
        mean_ci(f"lk.{ds}.{n}.acc_honest", s.vacc_clean, "results_leakage")
        mean_ci(f"lk.{ds}.{n}.margin_ratio_paired", s.margin_leak / s.margin_clean, "results_leakage")
        mean_ci(f"lk.{ds}.{n}.cloud_spread_ratio_paired", s.spread_m0 / s0.spread_m0, "results_leakage")
        mean_ci(f"lk.{ds}.{n}.nadv", s.nadv, "results_leakage")
    mean_ci(f"lk.{ds}.0.nadv", s0.nadv, "results_leakage")

# ---- 4.3 shortcut ----------------------------------------------------------------------------------
for f, lab in (("results_shortcut", "tree"), ("results_shortcut_svm", "svm")):
    sc = pd.read_parquet(DEX / f"{f}.parquet")
    for ds in ("iris", "wine"):
        c = sc[(sc.dataset == ds) & (sc["corr"] == 0)]
        mean_ci(f"sc.{lab}.{ds}.0.axis", c.spur_frac, f)
        mean_ci(f"sc.{lab}.{ds}.0.acc", c.vacc, f)
        for st in (1.0, 2.0, 4.0, 8.0):
            x = sc[(sc.dataset == ds) & (sc["corr"] == st)]
            mean_ci(f"sc.{lab}.{ds}.{st:g}.axis", x.spur_frac, f)
            mean_ci(f"sc.{lab}.{ds}.{st:g}.spread_ratio", x.spread_m0 / c.spread_m0.mean(), f)
            mean_ci(f"sc.{lab}.{ds}.{st:g}.acc", x.vacc, f)
            dval(f"sc.{lab}.{ds}.{st:g}.d_axis", x.spur_frac, c.spur_frac, f)
            dval(f"sc.{lab}.{ds}.{st:g}.d_acc", x.vacc, c.vacc, f)

# ---- 4.4 dimensionality ----------------------------------------------------------------------------
fvm = pd.read_parquet(DEX / "results_fitvsmeasure.parquet")
for frac in (0.5, 0.85):
    for col in ("spread_measure_full", "spread_fit_full"):
        for m in sorted(fvm.m.unique()):
            x = fvm[(fvm.m == m) & np.isclose(fvm.frac, frac)]
            dval(f"dim.{col}.{int(frac*100)}.m{m}", x[x.structure == "spatial"][col], x[x.structure == "random"][col],
                 "results_fitvsmeasure")
wh = pd.read_parquet(DEX / "results_wine_h4.parquet"); wh = wh[wh.dataset == "wine"]
for frac in (0.5, 0.7, 0.85):
    for k in sorted(wh.k.unique()):
        x = wh[(wh.k == k) & np.isclose(wh.frac, frac)]
        dval(f"wine_k.{int(frac*100)}.k{k}", x[x.structure == "spatial"].mean_dist, x[x.structure == "random"].mean_dist,
             "results_wine_h4")
for k in sorted(wh.k.unique()):
    mean_ci(f"wine_k.clean_acc.k{k}", wh[(wh.k == k) & np.isclose(wh.frac, 0)].vacc, "results_wine_h4")

out = pd.DataFrame(ROWS)
out.to_csv(ROOT / "analysis" / "results_numbers.csv", index=False, float_format="%.4f")
print(f"{len(out)} values -> analysis/results_numbers.csv")
