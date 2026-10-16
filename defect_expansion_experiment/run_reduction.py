"""PLAN §9 Arm 3+4 — dimension reduction + metric normalization on the SAME padded-iris clouds.

Arm 1 (run_dimensionality.py) showed the raw spread spatial-vs-random signal decays to zero as the
number of pure-noise padding dims `m` grows (concentration 0.54 -> 0.07). The signal STILL LIVES in
the original 4 iris dims — it is only buried, not gone. This runner tests the headline question:

  H2 — can dimension reduction BEFORE the metric recover the m=0 signal at high m?

It evaluates reduction variants OFFLINE on the exact adversarial clouds persisted by
run_dimensionality.py --save-clouds (same-clouds guarantee, §8 pattern), so no model is refit.

Reduction / metric variants (each returns a scalar spread on the DEFECT cloud):
  raw          OPTICS mean pairwise spread in the FULL padded space (the fragile reference).
  raw_divmed   raw spread / the cloud's median pairwise distance (a relative-contrast normalization).
  raw_divsqrtd raw spread / sqrt(d) (removes the sqrt(d) growth of isotropic-noise distance).
  pca4         UNSUPERVISED PCA of the defect cloud -> top-4 PCs, spread there (§8 m3 under controlled m).
  axis_var     SUPERVISED axis SELECTION: keep the 4 axes with the largest var(defect)/var(clean)
               ratio (the defect scatters points along real dims; noise dims have ratio ~1). Spread there.
  axis_mean    SUPERVISED axis SELECTION: keep the 4 axes with the largest |mean(defect)-mean(clean)|
               / pooled-std (defect shifts the cloud along real dims; noise dims ~0). Spread there.
  lda1         SUPERVISED 1-D Fisher direction separating defect-cloud pts from clean-baseline pts;
               spread of the defect cloud projected onto it.
  oracle4      Spread in the KNOWN first-4 (real iris) dims — the UPPER BOUND on what any discovery
               method could recover. Proves the signal exists and is recoverable; the question is
               whether a *blind* discovery method matches it.

"clean" reference = the frac=0 baseline cloud for the same (m, seed), matched per fold (same split
seed + same padding => the test points' noise coords are identical across frac, so clean-vs-defect
per-fold comparison is apples-to-apples). Discovery is BLIND to the true 4 dims except `oracle4`.

Protocol: per-fold reduced spread -> cell = mean over folds -> normalise by the per-(m,variant)
random-frac0 baseline -> Cohen's d (spatial vs random) + 95% CIs over 30 seeds. Success (criterion 2)
= a discovery variant whose padded-iris spatial-vs-random CIs RE-SEPARATE at m>=50 while raw does not.

-> results_reduction.parquet, results_reduction_summary.csv, verdict table on stdout.
Usage: python run_reduction.py
"""
import os
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_v] = '1'
import warnings, importlib.util, time
from functools import lru_cache
warnings.filterwarnings('ignore')
from pathlib import Path
from itertools import product
import numpy as np, polars as pl
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

HERE = Path(__file__).resolve().parent
CLOUDS = HERE / 'clouds_dim'
OUT = HERE / 'results_reduction.parquet'
M_PAD = [0, 5, 10, 20, 50, 100]
FRAC = [0.0, 0.25, 0.5, 0.7, 0.85, 0.95]
STRUCT = ['random', 'spatial']
SEEDS = [42, 58, 125, 7, 13, 21, 99, 123, 200, 777] + list(range(300, 320))
N_FOLDS = 5
K = 4                      # reduction target = true signal dimensionality (real iris dims)
REAL = 4                   # first REAL columns are the real iris features (oracle subspace)
SEP = [0.5, 0.7, 0.85]     # fracs to report separation at (0.95 self-hides even at m=0)

_os = importlib.util.spec_from_file_location('ol', HERE.parent / 'outlier_experiment' / 'run_experiment.py')
_o = importlib.util.module_from_spec(_os); _os.loader.exec_module(_o)
cluster_stats = _o.cluster_stats


def spread(points):
    return cluster_stats(points)[2] if len(points) >= 4 else np.nan


# ---- reduction variants: (defect_fold, clean_fold) -> scalar spread on the reduced defect cloud
def v_raw(defect, clean):
    return spread(defect)

def v_raw_divmed(defect, clean):
    if len(defect) < 4:
        return np.nan
    med = float(np.median(pdist(defect)))
    return spread(defect) / med if med > 0 else np.nan

def v_raw_divsqrtd(defect, clean):
    return spread(defect) / np.sqrt(defect.shape[1])

def v_pca4(defect, clean):
    n, d = defect.shape
    k = min(K, d, n - 1)
    if k < 2:
        return np.nan
    return spread(PCA(n_components=k).fit_transform(defect))

def _sel_var(defect, clean, k):
    vc = clean.var(0) + 1e-9
    vd = defect.var(0) + 1e-9
    return np.argsort(vd / vc)[::-1][:k]

def _sel_mean(defect, clean, k):
    pooled = np.sqrt((clean.var(0) + defect.var(0)) / 2.0) + 1e-9
    return np.argsort(np.abs(defect.mean(0) - clean.mean(0)) / pooled)[::-1][:k]

def v_axis_var(defect, clean):
    if len(clean) < 2 or len(defect) < 4:
        return np.nan
    idx = _sel_var(defect, clean, min(K, defect.shape[1]))
    return spread(defect[:, idx])

def v_axis_mean(defect, clean):
    if len(clean) < 2 or len(defect) < 4:
        return np.nan
    idx = _sel_mean(defect, clean, min(K, defect.shape[1]))
    return spread(defect[:, idx])

def v_lda1(defect, clean):
    if len(clean) < 2 or len(defect) < 4:
        return np.nan
    Xs = np.vstack([clean, defect])
    ys = np.array([0] * len(clean) + [1] * len(defect))
    try:
        z = LinearDiscriminantAnalysis(n_components=1).fit(Xs, ys).transform(defect)
    except Exception:
        return np.nan
    if z.ndim < 2 or z.shape[1] < 1:      # LDA degenerates to 0 usable discriminants (singular within-scatter)
        return np.nan
    return spread(z)

def v_oracle4(defect, clean):
    return spread(defect[:, :REAL])

VARIANTS = {'raw': v_raw, 'raw_divmed': v_raw_divmed, 'raw_divsqrtd': v_raw_divsqrtd,
            'pca4': v_pca4, 'axis_var': v_axis_var, 'axis_mean': v_axis_mean,
            'lda1': v_lda1, 'oracle4': v_oracle4}


@lru_cache(maxsize=None)
def load_folds(m, structure, frac, seed):
    p = CLOUDS / f'm{m}' / structure / f'frac{frac:.2f}_seed{seed}.npz'
    if not p.exists():
        return None
    z = np.load(p)
    return tuple(z[f'adv{i}'] for i in range(N_FOLDS))


def cell(variant, m, structure, frac, seed, clean_cache):
    folds = load_folds(m, structure, frac, seed)
    clean = clean_cache[(m, seed)]
    if folds is None or clean is None:
        return np.nan
    vals = []
    for i in range(N_FOLDS):
        try:
            v = VARIANTS[variant](folds[i], clean[i])
        except Exception:
            v = np.nan
        if v == v:
            vals.append(v)
    return float(np.mean(vals)) if vals else np.nan


def cohend(a, b):
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, (np.nan, np.nan), (np.nan, np.nan)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    d = (a.mean() - b.mean()) / sp if sp > 0 else np.nan
    ca = (a.mean(), 1.96 * a.std(ddof=1) / np.sqrt(len(a)))
    cb = (b.mean(), 1.96 * b.std(ddof=1) / np.sqrt(len(b)))
    return d, ca, cb


def main():
    t0 = time.time()
    clean_cache = {(m, s): load_folds(m, 'random', 0.0, s) for m, s in product(M_PAD, SEEDS)}
    rows = []
    for vi, variant in enumerate(VARIANTS, 1):
        print(f'  [{vi}/{len(VARIANTS)}] {variant} ({(time.time()-t0)/60:.1f}m)', flush=True)
        for m in M_PAD:
            base = np.array([cell(variant, m, 'random', 0.0, s, clean_cache) for s in SEEDS])
            norm = float(np.nanmean(base))
            for structure, frac in [('random', 0.0)] + list(product(STRUCT, FRAC[1:])):
                for s in SEEDS:
                    v = cell(variant, m, structure, frac, s, clean_cache)
                    rows.append(dict(variant=variant, m=m, structure=structure, frac=frac, seed=s,
                                     value=v, value_norm=v / norm if (v == v and norm > 0) else np.nan))
        pl.DataFrame(rows).write_parquet(OUT)      # checkpoint per variant (crash-safe)
    df = pl.DataFrame(rows)
    df.write_parquet(OUT)

    # ---- verdict: for each variant, Cohen's d spatial vs random (normalised) vs m, per frac
    print(f"\n{'='*96}\nReduction recovery — Cohen's d (spatial vs random, normalised spread), by m (K={K})\n{'='*96}")
    sum_rows = []
    for variant in VARIANTS:
        print(f"\n--- {variant} ---")
        print(f"{'m':>4} |" + "".join(f"  f{fr}  d(sep)   |" for fr in SEP))
        for m in M_PAD:
            line = f"{m:>4} |"
            for fr in SEP:
                sp = df.filter((pl.col('variant') == variant) & (pl.col('m') == m) &
                               (pl.col('structure') == 'spatial') & (pl.col('frac') == fr))['value_norm'].to_numpy()
                rr = df.filter((pl.col('variant') == variant) & (pl.col('m') == m) &
                               (pl.col('structure') == 'random') & (pl.col('frac') == fr))['value_norm'].to_numpy()
                d, ca, cb = cohend(sp, rr)
                sep = 'Y' if (ca[0] - ca[1] > cb[0] + cb[1]) else 'n'
                line += f" {ca[0]:.2f}/{cb[0]:.2f} {d:+.2f}({sep}) |"
                sum_rows.append(dict(variant=variant, m=m, frac=fr, spatial_m=ca[0], spatial_se=ca[1],
                                     random_m=cb[0], random_se=cb[1], cohens_d=d, separated=(sep == 'Y')))
            print(line)
    pl.DataFrame(sum_rows).write_csv(HERE / 'results_reduction_summary.csv')
    print('\nwrote results_reduction.parquet + results_reduction_summary.csv')


if __name__ == '__main__':
    main()
