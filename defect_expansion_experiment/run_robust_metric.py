"""§8 — dimension-robust spread metric evaluation (PLAN §8, the methodological gate).

Evaluates 4 candidate dispersion metrics on the SAME adversarial clouds that produced
FINDINGS_variance.md (persisted by run_variance.py --save-clouds), then tests with 95% CIs:
  (a) RECOVERY — on 13-D wine, do spatial vs random separate at frac >= 0.8 (train_only)?
  (b) SANITY   — on 4-D iris, does the metric still separate (must not break the working case)?
  (c) CONCENTRATION — std/mean of the metric's dispersion quantity comparable across
      iris<->wine (raw-spread reference measured in FINDINGS_variance: 0.54 iris vs 0.29 wine).

Candidates — what each measures, and why it could resist distance concentration:
  M1 knn_ratio    Per-point distance to its k-th nearest neighbour, divided by the cloud's
                  median of those distances. Measures RELATIVE local isolation. Taking ratios
                  cancels the global distance scale, which is what concentrates in high-D.
  M2 lof          Mean Local Outlier Factor (sklearn): each point's local reachability density
                  divided by the mean density of its neighbours. Measures local DENSITY ANOMALY.
                  A ratio of local densities — uniform distance inflation cancels by construction.
  M3 pca_spread   The existing OPTICS within-cluster mean pairwise spread, computed in the
                  cloud's top-3 PC subspace. Same metric, minus the noise axes: adversarial
                  displacement is carried by only ~2.8/13 axes on wine, and noise axes are what
                  drive concentration.
  M4 knn_local    Mean distance from each point to its k nearest neighbours. LOCAL dispersion —
                  excludes the bulk (far) pairwise distances that concentrate most in high-D.
  M0 raw_spread   The original OPTICS mean pairwise spread (reference: the fragile metric).

Protocol (matches plot_variance.py conventions): per-fold metric on the saved cloud ->
cell = mean over folds -> normalised by the per-dataset clean baseline (frac=0) -> CIs over
seeds (30 tree+DTA, 15 svm+HSJ).

Usage: python run_robust_metric.py
-> results_robust_metric.parquet (train_only cells), results_robust_summary.csv, verdict table on stdout.
"""
import os
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_v] = '1'
import warnings, importlib.util
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, polars as pl
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor

HERE = Path(__file__).resolve().parent
CLOUDS = HERE / 'clouds'
FRAC = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
SEP = [0.8, 0.9]
N_FOLDS = 5

_os = importlib.util.spec_from_file_location('ol', HERE.parent / 'outlier_experiment' / 'run_experiment.py')
_o = importlib.util.module_from_spec(_os); _os.loader.exec_module(_o)
cluster_stats = _o.cluster_stats

def _k(n):
    return min(5, max(2, n // 4))

def _knn_dists(points, k):
    """Distance from each point to its k-th nearest neighbour (self excluded)."""
    d = squareform(pdist(points))
    np.fill_diagonal(d, np.inf)
    return np.sort(d, axis=1)[:, k - 1]

def m0_raw(points):
    return cluster_stats(points)[2] if len(points) >= 4 else np.nan

def m1_knn_ratio(points):
    n = len(points)
    if n < 6:
        return np.nan
    dk = _knn_dists(points, _k(n))
    med = float(np.median(dk))
    return float(np.mean(dk / med)) if med > 0 else np.nan

def m2_lof(points):
    n = len(points)
    if n < 8:
        return np.nan
    k = min(5, n - 2)
    return float(np.mean(-LocalOutlierFactor(n_neighbors=k).fit(points).negative_outlier_factor_))

def m3_pca_spread(points, m=3):
    n = len(points)
    if n < 4:
        return np.nan
    mm = min(m, points.shape[1], n - 1)
    if mm < 2:
        return np.nan
    proj = PCA(n_components=mm).fit_transform(points)
    return cluster_stats(proj)[2]

def m4_knn_local(points):
    n = len(points)
    if n < 6:
        return np.nan
    return float(np.mean(_knn_dists(points, _k(n))))

def m2b_lof_robust(points):
    """LOF variant robust to near-duplicate adversarial points (exploding-LOF artifact).
    Deterministic jitter (1e-8 x cloud scale) breaks exact ties; median instead of mean so a
    few huge scores cannot dominate the cell value."""
    n = len(points)
    if n < 8:
        return np.nan
    k = min(5, n - 2)
    scale = max(float(np.std(points)), 1e-9)
    pts = points + np.random.default_rng(1234).normal(0, 1e-8 * scale, size=points.shape)
    lof = -LocalOutlierFactor(n_neighbors=k).fit(pts).negative_outlier_factor_
    return float(np.median(lof))

METRICS = {'m0_raw_spread': m0_raw, 'm1_knn_ratio': m1_knn_ratio,
           'm2_lof': m2_lof, 'm2b_lof_robust': m2b_lof_robust,
           'm3_pca_spread': m3_pca_spread, 'm4_knn_local': m4_knn_local}

def conc_quantity(metric, points):
    """The per-point (or per-pair) dispersion quantity used for the concentration check (c)."""
    n = len(points)
    if metric in ('m1_knn_ratio', 'm4_knn_local'):
        return _knn_dists(points, _k(n)) if n >= 6 else np.array([])
    if metric == 'm2_lof':
        return (-LocalOutlierFactor(n_neighbors=min(5, n - 2)).fit(points).negative_outlier_factor_
                if n >= 8 else np.array([]))
    if metric == 'm2b_lof_robust':
        if n < 8:
            return np.array([])
        pts = points + np.random.default_rng(1234).normal(0, 1e-8 * max(float(np.std(points)), 1e-9),
                                                         size=points.shape)
        return -LocalOutlierFactor(n_neighbors=min(5, n - 2)).fit(pts).negative_outlier_factor_
    if metric == 'm3_pca_spread':
        if n < 4:
            return np.array([])
        mm = min(3, points.shape[1], n - 1)
        return pdist(PCA(n_components=mm).fit_transform(points)) if mm >= 2 else np.array([])
    return pdist(points) if n >= 2 else np.array([])          # m0: all pairwise distances

def load_cell(ma, ds, structure, protocol, frac, seed):
    p = CLOUDS / ma / ds / structure / protocol / f'frac{frac:.1f}_seed{seed}.npz'
    if not p.exists():
        return None
    z = np.load(p)
    return [z[f'adv{i}'] for i in range(N_FOLDS)]

def cell_value(metric, folds):
    vals = [METRICS[metric](f) for f in folds if len(f)]
    vals = [v for v in vals if v == v]                        # drop NaN folds
    return float(np.nanmean(vals)) if vals else np.nan

def evaluate(ma, parquet, seeds, out_rows):
    ds_list = ['iris', 'wine']
    for ds in ds_list:
        for metric in METRICS:
            base = [cell_value(metric, load_cell(ma, ds, 'random', 'train_only', 0.0, s))
                    for s in seeds]
            norm = float(np.nanmean([b for b in base if b == b]))
            for s in seeds:
                b = base[seeds.index(s)]
                out_rows.append(dict(model_attack=ma, dataset=ds, structure='random',
                                     protocol='train_only', frac=0.0, seed=s, metric=metric,
                                     value=b, value_norm=b / norm if b == b else np.nan))
                for struct in ['spatial', 'random']:
                    for frac in FRAC[1:]:
                        folds = load_cell(ma, ds, struct, 'train_only', frac, s)
                        v = cell_value(metric, folds) if folds is not None else np.nan
                        out_rows.append(dict(model_attack=ma, dataset=ds, structure=struct,
                                             protocol='train_only', frac=frac, seed=s, metric=metric,
                                             value=v, value_norm=v / norm if v == v else np.nan))

def ci_series(df, ds, metric, structure, frac):
    v = df.filter((pl.col('dataset') == ds) & (pl.col('metric') == metric) &
                  (pl.col('structure') == structure) & (pl.col('frac') == frac))['value_norm'].drop_nans()
    if len(v) == 0:
        return None
    m, n = float(v.mean()), len(v)
    se = 1.96 * float(v.std()) / np.sqrt(n) if n > 1 else 0.0
    return m, se

def main():
    rows = []
    for ma, parquet in [('tree+dta', HERE / 'results_variance.parquet'),
                        ('svm+hsj', HERE / 'results_variance_svm.parquet')]:
        if not parquet.exists():
            print(f'skip {ma}: no parquet', flush=True); continue
        seeds = sorted(pl.read_parquet(parquet)['seed'].unique().to_list())
        # only seeds that actually have clouds (svm run still in flight → partial is fine)
        if not (CLOUDS / ma).exists():
            print(f'skip {ma}: no clouds yet', flush=True); continue
        seeds = [s for s in seeds if (CLOUDS / ma / 'iris' / 'random' / 'train_only' / f'frac0.0_seed{s}.npz').exists()]
        evaluate(ma, parquet, seeds, rows)
        print(f'{ma}: {len(seeds)} seeds with clouds', flush=True)
    df = pl.DataFrame(rows)
    df.write_parquet(HERE / 'results_robust_metric.parquet')

    # ---- verdict table + concentration check
    print('\nVerification: baseline mean_dist must reproduce raw-spread reference (sanity of pipeline)')
    sum_rows = []
    for ma in df['model_attack'].unique().to_list():
        d = df.filter(pl.col('model_attack') == ma)
        print(f'\n===== {ma} =====')
        print(f'{"metric":<16} | {"wine f0.8 spat vs rand":>28} | sep | {"wine f0.9 spat vs rand":>28} | sep | '
              f'{"iris f0.8 spat vs rand":>28} | sep | {"iris f0.9 spat vs rand":>28} | sep | {"conc wine/iris (base)":>20}')
        for metric in METRICS:
            line = []
            for ds in ['wine', 'iris']:
                for frac in SEP:
                    sp, rr = ci_series(d, ds, metric, 'spatial', frac), ci_series(d, ds, metric, 'random', frac)
                    if sp is None or rr is None:
                        cell, sep = '  n/a', ' - '
                    else:
                        cell = f'{sp[0]:.3f}±{sp[1]:.3f} vs {rr[0]:.3f}±{rr[1]:.3f}'
                        sep = 'YES' if (sp[0] + sp[1] < rr[0] - rr[1]) or (rr[0] + rr[1] < sp[0] - sp[1]) else 'no '
                    line.append(cell); line.append(sep)
            # concentration on baseline clouds (frac=0, pooled over seeds+folds)
            conc = {}
            for ds in ['iris', 'wine']:
                q = []
                for s in pl.read_parquet({'tree+dta': HERE / 'results_variance.parquet',
                                          'svm+hsj': HERE / 'results_variance_svm.parquet'}[ma]) \
                        .filter(pl.col('dataset') == ds)['seed'].unique().to_list():
                    folds = load_cell(ma, ds, 'random', 'train_only', 0.0, s)
                    if folds is None:
                        continue
                    for f in folds:
                        q.append(conc_quantity(metric, f))
                q = np.concatenate([a for a in q if len(a)])
                conc[ds] = float(q.std() / q.mean()) if len(q) and q.mean() > 0 else np.nan
            ratio = conc['wine'] / conc['iris'] if conc['iris'] and conc['iris'] > 0 else np.nan
            print(f'{metric:<16} | {line[0]:>28} | {line[1]:>3} | {line[2]:>28} | {line[3]:>3} | '
                  f'{line[4]:>28} | {line[5]:>3} | {line[6]:>28} | {line[7]:>3} | '
                  f'{ratio:.2f} (iris {conc["iris"]:.2f} / wine {conc["wine"]:.2f})')
            for ds in ['iris', 'wine']:
                for frac in SEP:
                    sp, rr = ci_series(d, ds, metric, 'spatial', frac), ci_series(d, ds, metric, 'random', frac)
                    sum_rows.append(dict(model_attack=ma, metric=metric, dataset=ds, frac=frac,
                                         spatial_m=sp[0] if sp else np.nan, spatial_se=sp[1] if sp else np.nan,
                                         random_m=rr[0] if rr else np.nan, random_se=rr[1] if rr else np.nan,
                                         conc_iris=conc['iris'], conc_wine=conc['wine'], conc_ratio=ratio))
    pl.DataFrame(sum_rows).write_csv(HERE / 'results_robust_summary.csv')
    print('\nwrote results_robust_metric.parquet + results_robust_summary.csv')

if __name__ == '__main__':
    main()
