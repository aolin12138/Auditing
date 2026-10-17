"""PLAN §9 — decisive fit-time vs measure-time disambiguation (padded iris).

Arm 1 showed raw full-space spread's spatial-vs-random signal decays as noise dims `m` grow.
Arm 3 (run_reduction.py) showed even the ORACLE (spread measured in the KNOWN real-4 dims) does not
recover the m=0 signal — it also decays. A mechanism diagnostic then found the overfit tree splits on
noise 0% of the time (iris is separable in petal dims), yet the real-dim adversarial clouds still
change with m. Hypothesis: the signal is degraded at FIT time (the mere presence of extra candidate
features perturbs the tree's split selection / tie-breaking → a slightly different tree → a different
adversarial cloud even in the real subspace), NOT at measurement time. If so, post-fit dimension
reduction is too late; the clean fix is feature selection BEFORE training.

This script isolates the two effects with three pipelines per cell, on the same padded data:
  measure_full  fit tree on ALL (4+m) dims, attack, measure spread in the FULL padded space.
                = the raw metric. Loses signal to BOTH measurement concentration AND fit perturbation.
  fit_full      fit tree on ALL (4+m) dims, attack, measure spread in the REAL-4 dims only (= oracle).
                Removes the MEASUREMENT problem; any residual decay is pure FIT-time damage.
  fit_real      DROP the noise dims BEFORE fitting: fit tree on the real-4 dims, attack, measure there.
                = feature selection before training. Independent of `m` by construction -> the recovery
                ceiling / proof the signal exists and is fully recoverable if you select before you fit.

disagree = fraction of test points where the fit_full tree and the fit_real tree predict differently
           = a direct measure of how much the noise features perturb the fitted model.

Verdict: if fit_real Cohen's d stays flat at the m=0 level while fit_full decays below it, the signal
is destroyed at FIT time and only pre-fit feature selection recovers it (post-fit reduction is capped).

-> results_fitvsmeasure.parquet  (one row per pipeline x m x structure x frac x seed, 5-fold mean)
Usage: python run_fitvsmeasure.py
"""
import os
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_v] = '1'
import warnings, time, importlib.util
from pathlib import Path
from itertools import product
import numpy as np, polars as pl
warnings.filterwarnings('ignore')
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import DecisionTreeAttack

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results_fitvsmeasure.parquet'
_spec = importlib.util.spec_from_file_location('imb_lib', HERE / 'run_imbalance.py')
_imb = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_imb)
inject_delete = _imb.inject_delete
_os = importlib.util.spec_from_file_location('ol', HERE.parent / 'outlier_experiment' / 'run_experiment.py')
_o = importlib.util.module_from_spec(_os); _os.loader.exec_module(_o)
cluster_stats = _o.cluster_stats
_ah = importlib.util.spec_from_file_location('ah', HERE / 'attack_all_helper.py')
_a = importlib.util.module_from_spec(_ah); _ah.loader.exec_module(_a)
attack_all, flips = _a.attack_all, _a.flips

M_PAD = [0, 5, 10, 20, 50, 100]
FRAC = [0.0, 0.5, 0.7, 0.85]
STRUCT = ['random', 'spatial']
TC, FEAT, REAL = 2, 2, 4
N_FOLDS = 5
SEEDS = [42, 58, 125, 7, 13, 21, 99, 123, 200, 777] + list(range(300, 320))
COLS = ['m', 'structure', 'frac', 'seed',
        'spread_measure_full', 'spread_fit_full', 'spread_fit_real', 'disagree',
        'spread_measure_full_correct', 'spread_fit_full_correct', 'spread_fit_real_correct']


def pad_noise(X, m, seed):
    if m <= 0:
        return X
    return np.hstack([X, np.random.default_rng(seed * 1_000 + 7).standard_normal((X.shape[0], m))])


def _fit_attack(Xt, yt, Xv, yv, yoh, cols):
    """Fit an overfit tree on Xt[:, cols], attack EVERY Xv point.
    Returns (adv_real, adv_real_correct, pred_v) in real-dim coordinates."""
    clf = DecisionTreeClassifier(max_depth=None, random_state=42).fit(Xt[:, cols], yt)
    art = SklearnClassifier(clf); art.fit(Xt[:, cols], yoh.transform(yt.reshape(-1, 1)))
    pv = np.argmax(art.predict(Xv[:, cols]), axis=1)
    c = pv == yv
    adv, _ = attack_all(art, Xv[:, cols])
    flip = flips(art, adv, pv)
    return adv[flip][:, :REAL], adv[c & flip][:, :REAL], pv


def run_cell(m, structure, frac, seed):
    X, y = load_iris(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    X = pad_noise(X, m, seed)
    yoh = OneHotEncoder(sparse_output=False).fit(y.reshape(-1, 1))
    all_cols = np.arange(X.shape[1]); real_cols = np.arange(REAL)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed)
    f_mf, f_ff, f_fr, f_dis = [], [], [], []
    f_mfc, f_ffc, f_frc = [], [], []
    for tr, te in skf.split(X, y):
        Xt, Xv, yt, yv = X[tr], X[te], y[tr], y[te]
        Xa, ya = inject_delete(Xt, yt, TC, FEAT, frac, structure, rng)
        # fit on full padded dims
        clf = DecisionTreeClassifier(max_depth=None, random_state=42).fit(Xa, ya)
        art = SklearnClassifier(clf); art.fit(Xa, yoh.transform(ya.reshape(-1, 1)))
        pv_full = np.argmax(art.predict(Xv), axis=1)
        c = pv_full == yv
        adv, _ = attack_all(art, Xv)                 # every test point, model's own prediction
        flip = flips(art, adv, pv_full)
        adv_full = adv[flip]
        adv_full_c = adv[c & flip]                   # correct-only subset = old pipeline
        f_mf.append(cluster_stats(adv_full)[2] if len(adv_full) else np.nan)            # measure_full
        f_mfc.append(cluster_stats(adv_full_c)[2] if len(adv_full_c) else np.nan)
        f_ff.append(cluster_stats(adv_full[:, :REAL])[2] if len(adv_full) else np.nan)    # fit_full (oracle)
        f_ffc.append(cluster_stats(adv_full_c[:, :REAL])[2] if len(adv_full_c) else np.nan)
        # fit on real dims only (feature selection before fit)
        adv_real, adv_real_c, pv_real = _fit_attack(Xa, ya, Xv, yv, yoh, real_cols)
        f_fr.append(cluster_stats(adv_real)[2] if len(adv_real) else np.nan)             # fit_real
        f_frc.append(cluster_stats(adv_real_c)[2] if len(adv_real_c) else np.nan)        # old pipeline
        f_dis.append(float((pv_full != pv_real).mean()))                                 # fit disagreement
    r = dict(m=int(m), structure=structure, frac=float(frac), seed=int(seed),
             spread_measure_full=float(np.nanmean(f_mf)), spread_fit_full=float(np.nanmean(f_ff)),
             spread_fit_real=float(np.nanmean(f_fr)), disagree=float(np.nanmean(f_dis)),
             spread_measure_full_correct=float(np.nanmean(f_mfc)),
             spread_fit_full_correct=float(np.nanmean(f_ffc)),
             spread_fit_real_correct=float(np.nanmean(f_frc)))
    return r


def cohend(a, b):
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan


def main():
    rows, t0 = [], time.time()
    grid = []
    for m, seed in product(M_PAD, SEEDS):
        grid.append((m, 'random', 0.0, seed))
        for st, fr in product(STRUCT, FRAC[1:]):
            grid.append((m, st, fr, seed))
    for i, (m, st, fr, sd) in enumerate(grid, 1):
        rows.append(run_cell(m, st, fr, sd))
        if i % 100 == 0 or i == len(grid):
            pl.DataFrame([{c: r[c] for c in COLS} for r in rows]).write_parquet(OUT)
            print(f'  [{i}/{len(grid)}] {(time.time()-t0)/60:.1f}m', flush=True)
    df = pl.DataFrame([{c: r[c] for c in COLS} for r in rows])
    df.write_parquet(OUT)

    print(f"\n{'='*80}\nFit-time vs measure-time: Cohen's d (spatial vs random) by pipeline, per m\n{'='*80}")
    for fr in [0.5, 0.7, 0.85]:
        print(f"\n--- frac={fr} ---   (disagree% = fit_full vs fit_real prediction mismatch)")
        print(f"{'m':>4} | {'measure_full':>12} | {'fit_full(oracle)':>16} | {'fit_real(preselect)':>19} | {'disagree%':>9}")
        for m in M_PAD:
            def col(pipe, st):
                return df.filter((pl.col('m') == m) & (pl.col('structure') == st) &
                                 (pl.col('frac') == fr))[pipe].to_numpy()
            base = {p: df.filter((pl.col('m') == m) & (pl.col('structure') == 'random') &
                                 (pl.col('frac') == 0.0))[p].to_numpy().mean()
                    for p in ['spread_measure_full', 'spread_fit_full', 'spread_fit_real']}
            dmf = cohend(col('spread_measure_full', 'spatial') / base['spread_measure_full'],
                         col('spread_measure_full', 'random') / base['spread_measure_full'])
            dff = cohend(col('spread_fit_full', 'spatial') / base['spread_fit_full'],
                         col('spread_fit_full', 'random') / base['spread_fit_full'])
            dfr = cohend(col('spread_fit_real', 'spatial') / base['spread_fit_real'],
                         col('spread_fit_real', 'random') / base['spread_fit_real'])
            dis = 100 * df.filter((pl.col('m') == m) & (pl.col('frac') == fr))['disagree'].to_numpy().mean()
            print(f"{m:>4} | {dmf:>+12.2f} | {dff:>+16.2f} | {dfr:>+19.2f} | {dis:>8.1f}%")
    print(f"\nDONE: {len(rows)} rows -> {OUT.name} in {(time.time()-t0)/60:.1f}m")


if __name__ == '__main__':
    main()
