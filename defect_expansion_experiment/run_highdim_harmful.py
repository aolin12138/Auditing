"""PLAN §9 frontier — can we detect a HARMFUL coverage gap in GENUINELY high-D data, without reducing?

The user's point: feature selection on a reducible dataset is trivial (you'd do it before training
anyway). The real question is a dataset that genuinely needs high dimension. And the earlier
all_informative test used a MILD single-axis gap (the model routes around it). This runner tests the
honest hard case:

  Dataset: make_classification with n_informative = n_features = 15 (every dim carries independent
           signal -> genuinely needs high-D; NOT reducible without accuracy loss).
  Defect:  a BOUNDARY-DIRECTED coverage gap -- delete the contested class's points that sit CLOSEST to
           its nearest neighbouring class (along the class-mean-difference direction v). This removes the
           hard-to-classify boundary region, so the decision boundary encroaches into the class => a
           genuinely HARMFUL gap (unlike the mild single-axis one). random arm = uniform deletion, matched count.

Three detectors, all on the FULL 15-D model (no crippling reduction of the model):
  spread_full     adversarial spread in full 15-D  (the raw metric; expect concentration-killed).
  recall_tc       minority-class recall of the contested class (the model-agnostic full-D detector).
  spread_vdir     adversarial spread PROJECTED onto the 1-D defect direction v -- a MEASUREMENT
                  projection onto where the defect acts (NOT a feature selection for the model). This is
                  the new idea: the defect may live in a low-D subspace even when PREDICTION needs high-D,
                  so projecting the measurement (not the model) onto the defect direction could reveal it.
  spread_topk     spread projected onto the top-4 F-score (PREDICTION) axes -- the contrast that failed
                  before (Finding 7); included to show defect-direction != prediction-features.

Reports, spatial vs random (30 seeds): absolute gap + paired CI, Cohen's d, recall gap, accuracy drop.
Verdict: if spread_vdir separates while spread_full/topk do not, a high-D defect IS detectable by
projecting the measurement onto its own direction (model kept full-D). If only recall separates, recall
remains the sole full-D detector. If nothing separates, the harmful high-D defect is genuinely invisible.

-> results_highdim_harmful.parquet
Usage: python run_highdim_harmful.py
"""
import os
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_v] = '1'
import warnings, time, importlib.util
from pathlib import Path
from itertools import product
import numpy as np, polars as pl
warnings.filterwarnings('ignore')
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import f_classif
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import DecisionTreeAttack

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results_highdim_harmful.parquet'
_os = importlib.util.spec_from_file_location('ol', HERE.parent / 'outlier_experiment' / 'run_experiment.py')
_o = importlib.util.module_from_spec(_os); _os.loader.exec_module(_o)
cluster_stats = _o.cluster_stats

D = 15
FRAC = [0.0, 0.5, 0.7]
STRUCT = ['random', 'spatial']
TC = 0
N_FOLDS = 5
SEEDS = [42, 58, 125, 7, 13, 21, 99, 123, 200, 777] + list(range(300, 320))


def make_data():
    # n_clusters_per_class=3 makes classes multi-modal => genuinely high intrinsic dim: accuracy climbs
    # steadily with #features (top-4 ~81% of full-15D acc), so it is NOT reducible by feature selection.
    X, y = make_classification(n_samples=600, n_features=D, n_informative=D, n_redundant=0,
                               n_classes=3, n_clusters_per_class=3, class_sep=1.0, flip_y=0.0,
                               random_state=0)
    return StandardScaler().fit_transform(X), y


def defect_dir(Xt, yt):
    """Unit vector from the contested class toward its NEAREST other-class centroid (the boundary dir)."""
    m0 = Xt[yt == TC].mean(0)
    others = [c for c in np.unique(yt) if c != TC]
    nearest = min(others, key=lambda c: np.linalg.norm(Xt[yt == c].mean(0) - m0))
    v = Xt[yt == nearest].mean(0) - m0
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def inject(Xt, yt, frac, structure, rng, v):
    if frac <= 0:
        return Xt, yt
    idx = np.where(yt == TC)[0]
    nrem = min(max(int(len(idx) * frac), 0), len(idx) - 3)
    if nrem <= 0:
        return Xt, yt
    if structure == 'spatial':
        proj = Xt[idx] @ v                       # boundary-facing = largest projection toward other class
        rem = idx[np.argsort(proj)[::-1][:nrem]]
    else:
        rem = rng.choice(idx, size=nrem, replace=False)
    keep = np.ones(len(yt), dtype=bool); keep[rem] = False
    return Xt[keep], yt[keep]


def run_cell(structure, frac, seed, X, y, topk):
    yoh = OneHotEncoder(sparse_output=False).fit(y.reshape(-1, 1))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed)
    folds = []
    for tr, te in skf.split(X, y):
        Xt, Xv, yt, yv = X[tr], X[te], y[tr], y[te]
        v = defect_dir(Xt, yt)                    # per-fold, train-only (clean)
        Xa, ya = inject(Xt, yt, frac, structure, rng, v)
        clf = DecisionTreeClassifier(max_depth=None, random_state=42).fit(Xa, ya)
        art = SklearnClassifier(clf); art.fit(Xa, yoh.transform(ya.reshape(-1, 1)))
        pv = np.argmax(art.predict(Xv), axis=1)
        vacc = float((pv == yv).mean())
        mmask = yv == TC
        recall = float((pv[mmask] == TC).mean()) if mmask.sum() else np.nan
        c = pv == yv
        if c.sum():
            adv = DecisionTreeAttack(classifier=art).generate(Xv[c])
            ap = np.argmax(art.predict(adv), axis=1); adv = adv[ap != yv[c]]
        else:
            adv = np.empty((0, D))
        if len(adv):
            sfull = cluster_stats(adv)[2]
            svdir = cluster_stats((adv @ v).reshape(-1, 1))[2]
            stopk = cluster_stats(adv[:, topk])[2]
        else:
            sfull = svdir = stopk = np.nan
        folds.append(dict(vacc=vacc, recall=recall, nadv=len(adv),
                          spread_full=sfull, spread_vdir=svdir, spread_topk=stopk))
    r = {k: float(np.nanmean([f[k] for f in folds])) for k in folds[0]}
    r.update(structure=structure, frac=float(frac), seed=int(seed))
    return r


def paired_gap(df, metric, fr):
    sp = df.filter((pl.col('structure') == 'spatial') & (pl.col('frac') == fr)).sort('seed')
    rr = df.filter((pl.col('structure') == 'random') & (pl.col('frac') == fr)).sort('seed')
    s = dict(zip(sp['seed'].to_list(), sp[metric].to_numpy()))
    r = dict(zip(rr['seed'].to_list(), rr[metric].to_numpy()))
    g = np.array([s[k] - r[k] for k in sorted(set(s) & set(r)) if s[k] == s[k] and r[k] == r[k]])
    if len(g) < 2:
        return np.nan, np.nan, np.nan
    base = df.filter((pl.col('structure') == 'random') & (pl.col('frac') == 0.0))[metric].to_numpy()
    base = np.nanmean(base)
    return g.mean(), 1.96 * g.std(ddof=1) / np.sqrt(len(g)), g.mean() / base if base else np.nan


def main():
    X, y = make_data()
    F, _ = f_classif(X, y)
    topk = np.argsort(np.nan_to_num(F, nan=-np.inf))[::-1][:4]
    cols = ['structure', 'frac', 'seed', 'vacc', 'recall', 'nadv', 'spread_full', 'spread_vdir', 'spread_topk']
    rows, t0 = [], time.time()
    grid = [('random', 0.0, s) for s in SEEDS] + [(st, fr, s) for st, fr, s in product(STRUCT, FRAC[1:], SEEDS)]
    for st, fr, s in grid:
        rows.append(run_cell(st, fr, s, X, y, topk))
    df = pl.DataFrame([{c: r.get(c, np.nan) for c in cols} for r in rows])
    df.write_parquet(OUT)

    v0 = df.filter((pl.col('structure') == 'random') & (pl.col('frac') == 0.0))['vacc'].mean()
    print(f"\n{'='*94}\nGenuinely high-D (15 informative dims), HARMFUL boundary-directed gap. clean vacc={v0:.3f}\n{'='*94}")
    for fr in FRAC[1:]:
        print(f"\n--- frac={fr} ---")
        # harm check: accuracy + recall drop (spatial vs clean)
        vsp = df.filter((pl.col('structure') == 'spatial') & (pl.col('frac') == fr))['vacc'].mean()
        rsp = df.filter((pl.col('structure') == 'spatial') & (pl.col('frac') == fr))['recall'].mean()
        rrn = df.filter((pl.col('structure') == 'random') & (pl.col('frac') == fr))['recall'].mean()
        r0 = df.filter((pl.col('structure') == 'random') & (pl.col('frac') == 0.0))['recall'].mean()
        print(f"  HARM: vacc {v0:.3f}->{vsp:.3f} | contested recall clean {r0:.3f} -> spatial {rsp:.3f} / random {rrn:.3f}"
              f"  (recall gap spatial-random = {rsp-rrn:+.3f})")
        print(f"  {'detector':>16} | {'paired abs gap (spatial-random)':>32} | {'norm gap':>8}")
        for metric, lab in [('spread_full', 'spread full-15D'), ('spread_vdir', 'spread defect-dir (1-D)'),
                            ('spread_topk', 'spread top-4 pred axes')]:
            g, ci, ng = paired_gap(df, metric, fr)
            sep = 'SEP' if (g - ci > 0) else ('SEP-neg' if (g + ci < 0) else 'overlap0')
            print(f"  {lab:>16} | {g:>+15.3f} ± {ci:.3f}  [{sep:>8}]      | {ng:>+7.3f}")
    print(f"\nDONE: {len(rows)} rows -> {OUT.name} in {(time.time()-t0)/60:.1f}m")


if __name__ == '__main__':
    main()
