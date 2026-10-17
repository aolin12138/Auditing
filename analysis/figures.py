"""Sole producer of the final report's figures and tables.

One function per report item. Figures -> report/figures/<name>.pdf (vector, report.mplstyle,
10 pt minimum text). Tables -> report/tables/<name>.tex (booktabs). No number in the report body
is typed by hand: every table cell comes from a results file through this module.

Item            Function             Data (all training-only unless stated)
Table 1         table_protocol       defect_expansion_experiment/results_variance.parquet (both protocols)
Fig 2           fig_coverage_gap     defect_expansion_experiment/results_variance.parquet
Fig 3           fig_blind_spots      outlier_experiment/results_kplateau(+_baseline).parquet,
                                     defect_expansion_experiment/results_label_noise.parquet,
                                     defect_expansion_experiment/results_leakage.parquet
Fig 4           fig_shortcut         defect_expansion_experiment/results_shortcut(_svm).parquet
                                     (_svm = 30-seed rerun of 2026-10-15; 15-seed original archived)
Fig 5           fig_dimensionality   defect_expansion_experiment/results_fitvsmeasure.parquet,
                                     defect_expansion_experiment/results_wine_h4.parquet
Table 2         table_design         seed counts and levels read from every grid listed in the table
Table 3         table_panel          results_variance, results_label_noise, outlier_experiment/results_kplateau
                                     (+_baseline), results_shortcut, results_leakage (tree + white-box only)
Table A1        table_scorecard      results_variance(_svm), results_p1(_rf) [earlier pipeline], results_attack_all
Table A2        table_sensitivity    same files: every-point columns vs the *_correct columns

Probe (2026-10-16 switch): every test point is attacked from the model's own prediction and points
whose prediction changed are kept. The all-points values keep the old column names (mean_dist,
spread_m0, spur_frac, ...); the earlier correct-only probe is in the *_correct columns of the same
files (tree: identical to the pre-switch files, verified row by row) and is shown only in Table A2.

Usage:  .venv/Scripts/python.exe analysis/figures.py            # build everything
        .venv/Scripts/python.exe analysis/figures.py fig_coverage_gap table_protocol
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats import ci95, d_with_ci  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEX = ROOT / "defect_expansion_experiment"
FIG = ROOT / "report" / "figures"
TAB = ROOT / "report" / "tables"
plt.style.use(ROOT / "tools" / "report.mplstyle")

# Contested class per dataset (lowest clean recall) and the feature the spatial hole is cut along.
# Fixed in run_variance.py CFG; repeated here only for labelling.
CONTESTED = {"iris": dict(tc=2, cls="virginica", feat="petal width"),
             "wine": dict(tc=0, cls="class 0", feat="proline")}
C_SPATIAL, C_RANDOM = "#D55E00", "#0072B2"   # Okabe-Ito vermilion / blue


def _variance():
    return pd.read_parquet(DEX / "results_variance.parquet")


def _fmt_d(d, lo, hi):
    return f"${d:+.2f}$ [{lo:+.2f}, {hi:+.2f}]".replace("+-", "-")


# ------------------------------------------------------------------------------------------
# Table 1 - injection protocol: before-split vs training-only (Methods 3.4)
# ------------------------------------------------------------------------------------------
def table_protocol():
    v = _variance()
    lines = [r"\begin{tabular}{llcccccc}", r"\toprule",
             r" & & \multicolumn{3}{c}{Before-split deletion} & \multicolumn{3}{c}{Training-only deletion}\\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
             r"Dataset & Deleted & Acc. & $n_{\mathrm{adv}}$ & $d$ & Acc. & $n_{\mathrm{adv}}$ & $d$\\",
             r"\midrule"]
    for ds in ("iris", "wine"):
        tc = CONTESTED[ds]["tc"]
        for frac in (0.6, 0.9):
            cells = []
            for proto in ("before_split", "train_only"):
                s = v[(v.dataset == ds) & (v.protocol == proto) & np.isclose(v.frac, frac) & (v.tc == tc)]
                sp, rd = s[s.structure == "spatial"], s[s.structure == "random"]
                d, lo, hi, ns, nr = d_with_ci(sp.mean_dist, rd.mean_dist)
                assert ns == nr == 30, (ds, frac, proto, ns, nr)
                dtxt = f"{d:+.2f}".replace("-", "$-$") + ("$^{*}$" if lo > 0 else "")
                cells += [f"{ci95(sp.vacc)[0]:.2f}", f"{sp.nadv.mean():.1f}", dtxt]
            lines.append(f"{ds.capitalize() if frac == 0.6 else ''} & {int(frac * 100)}\\% & " + " & ".join(cells) + r"\\")
        if ds == "iris":
            lines.append(r"\addlinespace")
    base = v[(v.protocol == "train_only") & np.isclose(v.frac, 0)]
    b = {ds: base[base.dataset == ds] for ds in ("iris", "wine")}
    # Note goes below the tabular (a full-width row would stretch the last column).
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
              f"{{\\footnotesize No deletion: iris Acc.\\ {b['iris'].vacc.mean():.2f}, "
              f"$n_{{\\mathrm{{adv}}}}$ {b['iris'].nadv.mean():.1f}; wine Acc.\\ {b['wine'].vacc.mean():.2f}, "
              f"$n_{{\\mathrm{{adv}}}}$ {b['wine'].nadv.mean():.1f}. $^{{*}}$95\\% bootstrap CI of $d$ excludes 0.\\par}}"]
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / "protocol.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  wrote report/tables/protocol.tex")


# ------------------------------------------------------------------------------------------
# Fig 2 - coverage gap: spatial hole vs random deletion of the same count (Results 4.1)
# ------------------------------------------------------------------------------------------
def fig_coverage_gap():
    v = _variance()
    v = v[v.protocol == "train_only"]
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.5), constrained_layout=True)
    for k, ds in enumerate(("iris", "wine")):
        tc = CONTESTED[ds]["tc"]
        sub = v[(v.dataset == ds) & ((v.tc == tc) | np.isclose(v.frac, 0))]
        base = sub[np.isclose(sub.frac, 0)].mean_dist.mean()
        ax = axes[k]
        for struct, col, lab in (("spatial", C_SPATIAL, "Spatial hole"), ("random", C_RANDOM, "Random deletion")):
            xs, ms, los, his = [], [], [], []
            for frac in sorted(sub.frac.unique()):
                s = sub[np.isclose(sub.frac, frac) & ((sub.structure == struct) | np.isclose(sub.frac, 0))]
                m, lo, hi = ci95(s.mean_dist / base)
                xs.append(frac * 100); ms.append(m); los.append(lo); his.append(hi)
            ax.plot(xs, ms, "o-", color=col, label=lab, ms=3)
            ax.fill_between(xs, los, his, color=col, alpha=0.2, lw=0)
        ax.axhline(1, color="0.4", lw=0.8, ls=":")
        ax.set_title(f"({'ab'[k]}) Spread, {ds}")
        ax.set_xlabel("Class deleted (%)")
        ax.set_xticks([0, 20, 40, 60, 80])
        ax.set_ylabel("Spread / clean spread" if k == 0 else "")
    ax = axes[2]
    for ds, ls in (("iris", "-"), ("wine", "--")):
        tc = CONTESTED[ds]["tc"]
        sub = v[(v.dataset == ds) & ((v.tc == tc) | np.isclose(v.frac, 0))]
        for struct, col in (("spatial", C_SPATIAL), ("random", C_RANDOM)):
            xs, ms, los, his = [], [], [], []
            for frac in sorted(sub.frac.unique()):
                s = sub[np.isclose(sub.frac, frac) & ((sub.structure == struct) | np.isclose(sub.frac, 0))]
                m, lo, hi = ci95(s.min_recall)
                xs.append(frac * 100); ms.append(m); los.append(lo); his.append(hi)
            ax.plot(xs, ms, ls, marker="o", color=col, ms=3)
            ax.fill_between(xs, los, his, color=col, alpha=0.15, lw=0)
    ax.set_title("(c) Recall of deleted class")
    ax.set_xlabel("Class deleted (%)")
    ax.set_ylabel("Test recall")
    ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.plot([], [], "k-", label="iris"); ax.plot([], [], "k--", label="wine")
    handles = axes[0].get_legend_handles_labels()[0] + ax.get_legend_handles_labels()[0]
    labels = ["Spatial hole", "Random deletion", "iris (c)", "wine (c)"]
    fig.legend(handles, labels, loc="outside upper center", ncol=4, frameon=False, handlelength=1.8)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "coverage_gap.pdf")
    fig.savefig(ROOT / "analysis" / "_preview_coverage_gap.png", dpi=200)
    plt.close(fig)
    print("  wrote report/figures/coverage_gap.pdf")


# ------------------------------------------------------------------------------------------
# Fig 3 - where the geometry does not see the defect, or sees it late (Results 4.2)
# ------------------------------------------------------------------------------------------
def fig_blind_spots():
    OUTL = ROOT / "outlier_experiment"
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.6), constrained_layout=True)

    # (a) outlier: spread vs outlier distance k, normalised by the matched 50-seed no-outlier baseline
    ax = axes[0, 0]
    kp, kb = pd.read_parquet(OUTL / "results_kplateau.parquet"), pd.read_parquet(OUTL / "results_kplateau_baseline.parquet")
    for model, col, lab in (("tree_d3", "#009E73", "Tree, depth 3"), ("tree_d10", "#CC79A7", "Tree, depth 10")):
        base = kb[kb.model == model].mean_dist.mean()
        xs, ms, los, his = [], [], [], []
        for k in sorted(kp.k.unique()):
            m, lo, hi = ci95(kp[(kp.model == model) & (kp.k == k)].mean_dist / base)
            xs.append(k); ms.append(m); los.append(lo); his.append(hi)
        ax.plot(xs, ms, "o-", color=col, ms=3, label=lab)
        ax.fill_between(xs, los, his, color=col, alpha=0.2, lw=0)
    ax.axhline(1, color="0.4", lw=0.8, ls=":", label="No outlier")
    ax.set_xticks([6, 10, 15, 20, 24])
    ax.set_title("(a) Outlier: spread vs distance")
    ax.set_xlabel("Outlier distance $k$ (class s.d.)")
    ax.set_ylabel("Spread / no-outlier spread")
    ax.legend(loc="upper right", frameon=False, handlelength=1.2)

    # (b) label noise: effect size vs clean, spread rise vs accuracy drop (overfit tree + DTA)
    ax = axes[0, 1]
    ln = pd.read_parquet(DEX / "results_label_noise.parquet")
    for ds, ls in (("iris", "-"), ("wine", "--")):
        s = ln[ln.dataset == ds]; clean = s[s.noise == 0]
        for meas, col, sign in (("mean_dist", C_SPATIAL, 1), ("vacc", C_RANDOM, -1)):
            xs, ds_, los, his = [], [], [], []
            for nz in sorted(s.noise.unique())[1:]:
                a, b = s[s.noise == nz][meas], clean[meas]
                d, lo, hi, _, _ = d_with_ci(sign * a, sign * b)
                xs.append(nz * 100); ds_.append(d); los.append(lo); his.append(hi)
            ax.plot(xs, ds_, ls, marker="o", ms=3, color=col)
            ax.fill_between(xs, los, his, color=col, alpha=0.15, lw=0)
    # symmetric log: spread d sits near 0 (and can be negative) while the accuracy drop reaches ~40
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(-1.5, 3000)                     # headroom above the curves for the legend
    ax.set_yticks([-1, 0, 1, 10, 100]); ax.set_yticklabels(["$-1$", "0", "1", "10", "100"])
    ax.axhline(0, color="0.4", lw=0.8, ls=":")
    ax.set_title("(b) Label noise: effect size vs clean")
    ax.set_xlabel("Training labels flipped (%)")
    ax.set_ylabel("Cohen's $d$ (symmetric log)")
    ax.plot([], [], "-", color=C_SPATIAL, label="Spread change")
    ax.plot([], [], "-", color=C_RANDOM, label="Accuracy drop")
    ax.plot([], [], "k-", label="iris"); ax.plot([], [], "k--", label="wine")
    ax.legend(loc="upper left", frameon=False, handlelength=1.6, ncol=2, columnspacing=0.8)

    # (c, d) leakage: measured vs honest accuracy; adversarial margin of leaked vs clean points
    lk = pd.read_parquet(DEX / "results_leakage.parquet")
    for ds, ls in (("iris", "-"), ("wine", "--")):
        s = lk[lk.dataset == ds]
        doses = sorted(s.n_leak.unique())
        for meas, col, ax in (("vacc", C_SPATIAL, axes[1, 0]), ("vacc_clean", C_RANDOM, axes[1, 0])):
            st = [ci95(s[s.n_leak == n][meas]) for n in doses]
            ax.plot(doses, [t[0] for t in st], ls, marker="o", ms=3, color=col)
            ax.fill_between(doses, [t[1] for t in st], [t[2] for t in st], color=col, alpha=0.15, lw=0)
        ds_doses = [n for n in doses if n > 0]
        st = [ci95(s[s.n_leak == n].margin_leak / s[s.n_leak == n].margin_clean) for n in ds_doses]
        axes[1, 1].plot(ds_doses, [t[0] for t in st], ls, marker="o", ms=3, color=C_SPATIAL, label=ds)
        axes[1, 1].fill_between(ds_doses, [t[1] for t in st], [t[2] for t in st], color=C_SPATIAL, alpha=0.15, lw=0)
    ax = axes[1, 0]
    ax.set_title("(c) Leakage: measured vs honest accuracy")
    ax.set_xlabel("Training points copied into test fold")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.885, 0.99)
    ax.plot([], [], "-", color=C_SPATIAL, label="Measured (all test points)")
    ax.plot([], [], "-", color=C_RANDOM, label="Honest (non-leaked only)")
    ax.legend(loc="upper left", frameon=False, handlelength=1.2)
    ax = axes[1, 1]
    ax.axhline(1, color="0.4", lw=0.8, ls=":")
    ax.set_title("(d) Leakage: adversarial margin")
    ax.set_xlabel("Training points copied into test fold")
    ax.set_ylabel("Margin, leaked / non-leaked")
    ax.legend(loc="lower left", frameon=False, handlelength=1.6)
    for a in axes[1]:
        a.set_xticks([0, 5, 10, 15, 20])
    fig.savefig(FIG / "blind_spots.pdf")
    fig.savefig(ROOT / "analysis" / "_preview_blind_spots.png", dpi=200)
    plt.close(fig)
    print("  wrote report/figures/blind_spots.pdf")


# ------------------------------------------------------------------------------------------
# Fig 4 - shortcut feature: scalar spread vs the per-axis readout (Results 4.3)
# ------------------------------------------------------------------------------------------
def fig_shortcut():
    C_TREE, C_SVM = "#009E73", "#CC79A7"
    data = {("tree", C_TREE): pd.read_parquet(DEX / "results_shortcut.parquet"),
            ("svm", C_SVM): pd.read_parquet(DEX / "results_shortcut_svm.parquet")}
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.6), constrained_layout=True)
    for (model, col), d in data.items():
        for ds, ls in (("iris", "-"), ("wine", "--")):
            s = d[d.dataset == ds]
            base = s[s["corr"] == 0].spread_m0.mean()
            for ax, meas, scale in ((axes[0], "spread_m0", base), (axes[1], "spur_frac", 1.0), (axes[2], "vacc", 1.0)):
                xs, ms, los, his, few = [], [], [], [], []
                for c in sorted(s["corr"].unique()):
                    v = s[s["corr"] == c][meas] / scale
                    m, lo, hi = ci95(v)
                    xs.append(c); ms.append(m); los.append(lo); his.append(hi); few.append(v.notna().sum() < 20)
                ax.plot(xs, ms, ls, color=col)
                ax.fill_between(xs, los, his, color=col, alpha=0.15, lw=0)
                for x, m, f in zip(xs, ms, few):   # hollow marker = fewer than 20 of 30 runs produced a cloud
                    ax.plot(x, m, "o", ms=3.5, mec=col, mfc="white" if f else col)
    axes[0].axhline(1, color="0.4", lw=0.8, ls=":")
    axes[0].set_title("(a) Scalar spread")
    axes[0].set_ylabel("Spread / spread at strength 0")
    axes[1].set_title("(b) Per-axis displacement")
    axes[1].set_ylabel("Fraction along shortcut axis")
    axes[2].set_title("(c) Test accuracy")
    axes[2].set_ylabel("Test accuracy")
    for ax in axes:
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xticks([0, 0.5, 1, 2, 4, 8]); ax.set_xticklabels(["0", ".5", "1", "2", "4", "8"])
        ax.set_xlim(-0.15, 9)
        ax.set_xlabel("Shortcut strength")
    handles = [plt.Line2D([], [], color=C_TREE, marker="o", ms=3.5), plt.Line2D([], [], color=C_SVM, marker="o", ms=3.5),
               plt.Line2D([], [], color="k", ls="-"), plt.Line2D([], [], color="k", ls="--")]
    fig.legend(handles, ["Tree + white-box", "SVM + black-box", "iris", "wine"],
               loc="outside upper center", ncol=4, frameon=False, handlelength=1.8)
    fig.savefig(FIG / "shortcut.pdf")
    fig.savefig(ROOT / "analysis" / "_preview_shortcut.png", dpi=200)
    plt.close(fig)
    print("  wrote report/figures/shortcut.pdf")


# ------------------------------------------------------------------------------------------
# Fig 5 - dimensionality: where the signal is lost, and what selection before fitting buys (4.4)
# ------------------------------------------------------------------------------------------
def _d_series(df, col, xs_col, xs, frac):
    out = []
    for x in xs:
        s = df[(df[xs_col] == x) & np.isclose(df.frac, frac)]
        out.append(d_with_ci(s[s.structure == "spatial"][col], s[s.structure == "random"][col]))
    return np.array([o[:3] for o in out], dtype=float)


def fig_dimensionality():
    fvm = pd.read_parquet(DEX / "results_fitvsmeasure.parquet")
    w = pd.read_parquet(DEX / "results_wine_h4.parquet"); w = w[w.dataset == "wine"]
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.7), constrained_layout=True)

    # (a) padded iris: raw full-space spread vs oracle projection after fitting on all dims
    ax = axes[0]; ms_ = sorted(fvm.m.unique()); xpos = np.arange(len(ms_))   # evenly spaced levels
    for col, c, lab in (("spread_measure_full", C_SPATIAL, "Full space"),
                        ("spread_fit_full", C_RANDOM, "Real dims only")):
        for frac, ls in ((0.5, "-"), (0.85, "--")):
            r = _d_series(fvm, col, "m", ms_, frac)
            ax.plot(xpos, r[:, 0], ls, marker="o", ms=3, color=c, label=lab if frac == 0.5 else None)
            if frac == 0.5:
                ax.fill_between(xpos, r[:, 1], r[:, 2], color=c, alpha=0.15, lw=0)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.plot([], [], "k-", label="50% deleted"); ax.plot([], [], "k--", label="85% deleted")
    ax.set_xticks(xpos); ax.set_xticklabels([str(m) for m in ms_])
    ax.set_ylim(-0.8, 2.9)
    ax.set_title("(a) Iris + noise dimensions")
    ax.set_xlabel("Noise dimensions added"); ax.set_ylabel("Cohen's $d$, spatial vs random")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside upper left", ncol=4, frameon=False,
               handlelength=1.6, title="Panel (a):", title_fontsize=10, alignment="left")

    # (b) wine restricted to its top-k features BEFORE fitting: separation by deletion level
    ax = axes[1]; ks = sorted(w.k.unique())
    for frac, c in ((0.5, "#009E73"), (0.7, "#E69F00"), (0.85, "#CC79A7")):
        r = _d_series(w, "mean_dist", "k", ks, frac)
        ax.plot(ks, r[:, 0], "o-", ms=3, color=c, label=f"{int(frac * 100)}% deleted")
        ax.fill_between(ks, r[:, 1], r[:, 2], color=c, alpha=0.12, lw=0)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xticks(ks)
    ax.set_title("(b) Wine, top-$k$ features")
    ax.set_xlabel("Features kept ($k$)"); ax.set_ylabel("Cohen's $d$, spatial vs random")
    ax.legend(loc="upper right", frameon=False, handlelength=1.2)

    # (c) the price: clean test accuracy of the restricted model
    ax = axes[2]
    st = [ci95(w[(w.k == k) & np.isclose(w.frac, 0)].vacc) for k in ks]
    ax.plot(ks, [t[0] for t in st], "o-", ms=3, color="0.2")
    ax.fill_between(ks, [t[1] for t in st], [t[2] for t in st], color="0.2", alpha=0.15, lw=0)
    ax.set_xticks(ks)
    ax.set_title("(c) Wine, clean accuracy")
    ax.set_xlabel("Features kept ($k$)"); ax.set_ylabel("Test accuracy (no defect)")
    fig.savefig(FIG / "dimensionality.pdf")
    fig.savefig(ROOT / "analysis" / "_preview_dimensionality.png", dpi=200)
    plt.close(fig)
    print("  wrote report/figures/dimensionality.pdf")


# ------------------------------------------------------------------------------------------
# Table 3 - which measurement responds to which defect (Results 4.5). Numbers only: Cohen's d of
# (defect - clean) for each measurement at one stated dose, overfit tree + white-box attack.
# Bold = 95% bootstrap CI excludes 0. "--" = not recorded in that grid. Interpretation is left
# to the Discussion.
# ------------------------------------------------------------------------------------------
def _cell(a, b):
    d, lo, hi, na, nb = d_with_ci(a, b)
    if not np.isfinite(d):
        return "--", None
    txt = ("0.0" if abs(d) < 0.05 else f"{d:+.1f}").replace("-", "$-$")   # no "-0.0"
    return (f"\\textbf{{{txt}}}" if (lo > 0 or hi < 0) else txt), (d, lo, hi, na, nb)


def table_panel():
    v = _variance(); v = v[v.protocol == "train_only"]
    ln = pd.read_parquet(DEX / "results_label_noise.parquet")
    sc = pd.read_parquet(DEX / "results_shortcut.parquet")
    lk = pd.read_parquet(DEX / "results_leakage.parquet")
    kp = pd.read_parquet(ROOT / "outlier_experiment" / "results_kplateau.parquet")
    kb = pd.read_parquet(ROOT / "outlier_experiment" / "results_kplateau_baseline.parquet")
    rows, log = [], []

    def row(name, structure, dose, per_ds):
        cells = {k: [] for k in ("acc", "rec", "spread", "axis")}
        for ds in ("iris", "wine"):
            spec = per_ds.get(ds)
            for k in cells:
                if spec is None or spec.get(k) is None:
                    cells[k].append("--"); continue
                txt, st = _cell(*spec[k]); cells[k].append(txt)
                if st: log.append((name, ds, k) + st)
        joined = ["--" if cells[k] == ["--", "--"] else " / ".join(cells[k]) for k in cells]
        rows.append(f"{name} & {dose} & " + " & ".join(joined) + r"\\")   # structure: see Methods 3.3

    def var_spec(ds, struct, frac):
        tc = CONTESTED[ds]["tc"]
        s = v[(v.dataset == ds) & (v.tc == tc)]
        d, c = s[(s.structure == struct) & np.isclose(s.frac, frac)], s[np.isclose(s.frac, 0)]
        return dict(acc=(d.vacc, c.vacc), rec=(d.min_recall, c.min_recall), spread=(d.mean_dist, c.mean_dist))

    row("Coverage gap", "global, spatial", r"60\% of class", {ds: var_spec(ds, "spatial", 0.6) for ds in ("iris", "wine")})
    row("Class imbalance", "count only", r"60\% of class", {ds: var_spec(ds, "random", 0.6) for ds in ("iris", "wine")})
    row("Label noise", "global, unstructured", r"20\% of labels",
        {ds: dict(acc=(ln[(ln.dataset == ds) & np.isclose(ln.noise, 0.2)].vacc, ln[(ln.dataset == ds) & (ln.noise == 0)].vacc),
                  rec=(ln[(ln.dataset == ds) & np.isclose(ln.noise, 0.2)].worst_recall, ln[(ln.dataset == ds) & (ln.noise == 0)].worst_recall),
                  spread=(ln[(ln.dataset == ds) & np.isclose(ln.noise, 0.2)].mean_dist, ln[(ln.dataset == ds) & (ln.noise == 0)].mean_dist))
         for ds in ("iris", "wine")})
    k8, b = kp[(kp.model == "tree_d10") & (kp.k == 8)], kb[kb.model == "tree_d10"]
    row("Outlier", "local", r"10 points, $k=8$", {"iris": dict(acc=(k8.vacc, b.vacc), spread=(k8.mean_dist, b.mean_dist))})
    row("Shortcut feature", "global, directional", "strength 2",
        {ds: dict(acc=(sc[(sc.dataset == ds) & (sc["corr"] == 2)].vacc, sc[(sc.dataset == ds) & (sc["corr"] == 0)].vacc),
                  spread=(sc[(sc.dataset == ds) & (sc["corr"] == 2)].spread_m0, sc[(sc.dataset == ds) & (sc["corr"] == 0)].spread_m0),
                  axis=(sc[(sc.dataset == ds) & (sc["corr"] == 2)].spur_frac, sc[(sc.dataset == ds) & (sc["corr"] == 0)].spur_frac))
         for ds in ("iris", "wine")})
    row("Leakage", "local (evaluation)", "10 points/fold",
        {ds: dict(acc=(lk[(lk.dataset == ds) & (lk.n_leak == 10)].vacc, lk[(lk.dataset == ds) & (lk.n_leak == 0)].vacc),
                  spread=(lk[(lk.dataset == ds) & (lk.n_leak == 10)].spread_m0, lk[(lk.dataset == ds) & (lk.n_leak == 0)].spread_m0))
         for ds in ("iris", "wine")})

    tex = [r"\begin{tabular}{llcccc}", r"\toprule",
           r" & & \multicolumn{4}{c}{Cohen's $d$ vs defect-free, iris / wine} \\", r"\cmidrule(lr){3-6}",
           r"Defect & Dose & Accuracy & Recall & Spread & Per-axis \\", r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}"]
    (TAB / "panel.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    pd.DataFrame(log, columns=["defect", "dataset", "measure", "d", "lo", "hi", "n_defect", "n_clean"]).to_csv(
        ROOT / "analysis" / "panel_table_values.csv", index=False)
    print("  wrote report/tables/panel.tex (+ analysis/panel_table_values.csv with CIs and N)")


# ------------------------------------------------------------------------------------------
# Table 2 - experimental design (Methods 3.6). Descriptions are design facts; seed counts and
# level ranges are read from the result files so the table cannot drift from the data.
# ------------------------------------------------------------------------------------------
def table_design():
    OUTL = ROOT / "outlier_experiment"
    def seeds(df): return df.seed.nunique()
    def pct(vals): v = sorted(set(vals)); return f"{v[0]*100:.0f}--{v[-1]*100:.0f}\%"
    var, vsv = _variance(), pd.read_parquet(DEX / "results_variance_svm.parquet")
    ln, lns = pd.read_parquet(DEX / "results_label_noise.parquet"), pd.read_parquet(DEX / "results_label_noise_svm.parquet")
    kp = pd.read_parquet(OUTL / "results_kplateau.parquet")
    lk = pd.read_parquet(DEX / "results_leakage.parquet")
    sc, scs = pd.read_parquet(DEX / "results_shortcut.parquet"), pd.read_parquet(DEX / "results_shortcut_svm.parquet")
    fvm, wh = pd.read_parquet(DEX / "results_fitvsmeasure.parquet"), pd.read_parquet(DEX / "results_wine_h4.parquet")
    hc = pd.read_parquet(DEX / "results_healthy_control.parquet")
    for d_, n_ in ((vsv, var), (lns, ln), (scs, sc)):
        assert seeds(d_) == seeds(n_), "tree and SVM grids should share seeds"
    rows = [
        ("Coverage gap vs class imbalance", "1, 4", "iris, wine", "T, S", f"deletion {pct(var.frac)}", seeds(var)),
        ("Injection protocol", "1", "iris, wine", "T", "before-split vs training-only", seeds(var)),
        ("Label noise", "1, 4", "iris, wine", "T, S", f"labels flipped {pct(ln.noise)}", seeds(ln)),
        ("Outlier distance", "1", "iris (raw)", "T (depth 3, 10)", f"$k$ = {kp.k.min():.0f}--{kp.k.max():.0f}", seeds(kp)),
        ("Train--test leakage", "1, 4", "iris, wine", "T", f"{lk.n_leak.min()}--{lk.n_leak.max()} points per fold", seeds(lk)),
        ("Shortcut feature", "3", "iris, wine", "T, S", f"strength {sc['corr'].min():g}--{sc['corr'].max():g}", seeds(sc)),
        ("Dimensionality, fit vs measure", "2", "iris + noise", "T", f"{fvm.m.min()}--{fvm.m.max()} noise features", seeds(fvm)),
        ("Feature selection before fitting", "2", "wine", "T", f"top $k$ = {wh.k.min()}--{wh.k.max()} features", seeds(wh)),
        ("Negative controls", "1", "synthetic, iris", "T", "4 conditions", seeds(hc)),
    ]
    tex = [r"\begin{tabular}{p{0.28\linewidth}cp{0.13\linewidth}p{0.11\linewidth}p{0.21\linewidth}c}", r"\toprule",
           r"Experiment & RQ & Data & Model & Varied & Seeds \\", r"\midrule"]
    tex += [" & ".join(map(str, r)) + r" \\" for r in rows]
    tex += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
            r"{\footnotesize T: decision tree + white-box attack; S: SVM + HopSkipJump.\par}"]
    (TAB / "design.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print("  wrote report/tables/design.tex")


# ------------------------------------------------------------------------------------------
# Table A1 - generalisation scorecard (Appendix A): spatial hole vs random deletion of the same
# count, training-only, across model x attack x dataset, with the number of valid runs per arm.
# Same rule as analysis/extra_evidence.py, but on the current files (SVM grid = 30-seed rerun).
# ------------------------------------------------------------------------------------------
def table_scorecard():
    def block(label, df, fracs):
        base = df[np.isclose(df.frac, 0)].mean_dist.mean()
        out = []
        for frac in fracs:
            s = df[np.isclose(df.frac, frac)]
            a, b = s[s.structure == "spatial"].mean_dist, s[s.structure == "random"].mean_dist
            d, lo, hi, na, nb = d_with_ci(a, b)
            if min(na, nb) < 5:          # bootstrap CI of d is degenerate with so few runs
                dtxt = "too few runs"
            elif np.isfinite(lo):
                dtxt = (f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]" + ("$^{*}$" if lo > 0 else "")).replace("-", "$-$")
            else:
                dtxt = "--"
            ra = f"{np.nanmean(a) / base:.2f}" if a.notna().any() else "--"
            rb = f"{np.nanmean(b) / base:.2f}" if b.notna().any() else "--"
            out.append(f"{label} & {int(frac * 100)}\\% & {na} / {nb} & {ra} / {rb} & {dtxt}" + r" \\")
            label = ""
        return out
    v, vs = _variance(), pd.read_parquet(DEX / "results_variance_svm.parquet")
    p1, rf = pd.read_parquet(DEX / "results_p1.parquet"), pd.read_parquet(DEX / "results_p1_rf.parquet")
    aa = pd.read_parquet(DEX / "results_attack_all.parquet")
    plen = aa[aa.dataset == "iris_plen"].rename(columns={"spread_all": "mean_dist"})
    rows = []
    for ds in ("iris", "wine"):
        tc = CONTESTED[ds]["tc"]
        for df, lab in ((v, "tree + white-box"), (vs, "SVM + black-box")):
            sub = df[(df.protocol == "train_only") & (df.dataset == ds) & ((df.tc == tc) | np.isclose(df.frac, 0))]
            rows += block(f"{ds}, {lab}", sub, [0.6, 0.8, 0.9]) + [r"\addlinespace"]
    rows += block("iris, tree + white-box, petal length$^{\dagger}$", plen, [0.6, 0.8, 0.9]) + [r"\addlinespace"]
    for model, lab in (("tree", "tree + black-box$^{\ddagger}$"),):
        sub = p1[(p1.model == model) & ((p1.tc == 2) | (p1.structure == "baseline"))]
        rows += block(f"iris, {lab}", sub, [0.5, 0.85, 0.95]) + [r"\addlinespace"]
    rows += block("iris, forest + black-box$^{\ddagger}$", rf[(rf.tc == 2) | (rf.structure == "baseline")], [0.5, 0.85, 0.95])
    tex = [r"\begin{tabular}{p{0.25\linewidth}cccc}", r"\toprule",
           r"Setting & Deleted & Runs & Spread / clean & $d$ [95\% CI] \\",
           r" & & s / r & s / r & s vs r \\", r"\midrule"] + rows + [
           r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
           r"{\footnotesize s: spatial hole; r: random deletion of the same count. $^{*}$CI excludes 0. $d$ is reported only when both arms have at least 5 valid runs. $^{\dagger}$Unstandardised iris, hole cut along petal length (the other rows use "
           r"petal width). $^{\ddagger}$Earlier pipeline: only correctly classified test points attacked; black-box tree 10 seeds, forest 3 seeds. Other rows: every test point attacked, 30 seeds.\par}"]
    (TAB / "scorecard.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print("  wrote report/tables/scorecard.tex")


# ------------------------------------------------------------------------------------------
# Table A2 - probe sensitivity (Appendix A): every body comparison under the every-point probe and
# under the earlier correct-only probe, from the SAME runs (the *_correct columns). Tree + white-box.
# ------------------------------------------------------------------------------------------
def table_sensitivity():
    v = _variance(); v = v[v.protocol == "train_only"]
    ln = pd.read_parquet(DEX / "results_label_noise.parquet")
    sc = pd.read_parquet(DEX / "results_shortcut.parquet")
    lk = pd.read_parquet(DEX / "results_leakage.parquet")
    kp = pd.read_parquet(ROOT / "outlier_experiment" / "results_kplateau.parquet")
    kb = pd.read_parquet(ROOT / "outlier_experiment" / "results_kplateau_baseline.parquet")
    log = []

    def pair(name, ds, a_all, b_all, a_cor, b_cor):
        out = []
        for probe, a, b in (("every point", a_all, b_all), ("correct only", a_cor, b_cor)):
            txt, st = _cell(a, b); out.append(txt)
            if st: log.append((name, ds, probe) + st)
        return out

    rows = []
    def add(name, dose, per_ds):
        cells = []
        for ds in ("iris", "wine"):
            cells += pair(name, ds, *per_ds[ds]) if ds in per_ds else ["--", "--"]
        rows.append(f"{name} & {dose} & " + " & ".join(cells) + r"\\")

    def var_pair(ds, frac):
        s = v[(v.dataset == ds) & (v.tc == CONTESTED[ds]["tc"]) & np.isclose(v.frac, frac)]
        sp, rd = s[s.structure == "spatial"], s[s.structure == "random"]
        return sp.mean_dist, rd.mean_dist, sp.mean_dist_correct, rd.mean_dist_correct
    for frac in (0.4, 0.6, 0.9):
        add("Coverage gap", f"{int(frac * 100)}\\%", {ds: var_pair(ds, frac) for ds in ("iris", "wine")})
    def vs_clean(df, key, col, dose, clean):
        a, b = df[(df.dataset == key) & dose], df[(df.dataset == key) & clean]
        return a[col], b[col], a[col + "_correct"], b[col + "_correct"]
    add("Label noise, spread", r"20\%", {ds: vs_clean(ln, ds, "mean_dist", np.isclose(ln.noise, 0.2), ln.noise == 0) for ds in ("iris", "wine")})
    k8, b = kp[(kp.model == "tree_d10") & (kp.k == 8)], kb[kb.model == "tree_d10"]
    add("Outlier, spread", r"$k=8$", {"iris": (k8.mean_dist, b.mean_dist, k8.mean_dist_correct, b.mean_dist_correct)})
    add("Shortcut, per-axis", "strength 2", {ds: vs_clean(sc, ds, "spur_frac", sc["corr"] == 2, sc["corr"] == 0) for ds in ("iris", "wine")})
    add("Shortcut, per-axis", "strength 1", {ds: vs_clean(sc, ds, "spur_frac", sc["corr"] == 1, sc["corr"] == 0) for ds in ("iris", "wine")})
    add("Leakage, spread", "10 points", {ds: vs_clean(lk, ds, "spread_m0", lk.n_leak == 10, lk.n_leak == 0) for ds in ("iris", "wine")})
    tex = [r"\begin{tabular}{p{0.24\linewidth}ccccc}", r"\toprule",
           r" & & \multicolumn{2}{c}{Iris} & \multicolumn{2}{c}{Wine} \\",
           r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
           r"Comparison & Dose & Every point & Correct only & Every point & Correct only \\", r"\midrule"] + rows + [
           r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
           r"{\footnotesize Cohen's $d$; bold: 95\% bootstrap CI excludes 0. Coverage gap: spatial hole vs random deletion of the same count; other rows: defect vs defect-free. Both probes are computed from the same adversarial examples in each run; overfit tree, white-box attack, 30 seeds (outlier: 50).\par}"]
    (TAB / "sensitivity.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    pd.DataFrame(log, columns=["comparison", "dataset", "probe", "d", "lo", "hi", "n_a", "n_b"]).to_csv(
        ROOT / "analysis" / "sensitivity_table_values.csv", index=False)
    print("  wrote report/tables/sensitivity.tex (+ analysis/sensitivity_table_values.csv)")


ITEMS = {f.__name__: f for f in (table_protocol, table_design, fig_coverage_gap, fig_blind_spots,
                                    fig_shortcut, fig_dimensionality, table_panel, table_scorecard,
                                    table_sensitivity)}

if __name__ == "__main__":
    for name in (sys.argv[1:] or ITEMS):
        print(name)
        ITEMS[name]()
