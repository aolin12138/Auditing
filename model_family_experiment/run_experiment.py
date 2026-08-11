"""RF + XGBoost on ALL three defects (coverage_gap, label_noise, outlier) via the
black-box HopSkipJump attack. Ensembles are tree-based, so HSJ can HANG on pathological
folds (same facet pathology as single trees) -> bulletproof driver/worker: each row runs
in a fresh subprocess with a hard timeout; hung rows are recorded NaN and never retried.
Resumable. Reuses inject_outliers + cluster_stats from ../outlier_experiment.

Usage:
  python run_experiment.py --smoke        # 1 cell per (model,defect), timing + correctness
  python run_experiment.py --defect coverage_gap [--model rf]   # run one defect (optionally one model)
  python run_experiment.py --worker       # internal: do one undone row
  python run_experiment.py                # run everything
"""
import os
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_v] = '1'
import sys, time, subprocess, warnings
from pathlib import Path
from itertools import product
import numpy as np, polars as pl
warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results.parquet'
PROGRESS = HERE / 'progress.txt'
import importlib.util                                                # load ../outlier_experiment/run_experiment.py
_spec = importlib.util.spec_from_file_location(                      # by explicit path (avoid name collision:
    'outlier_lib', HERE.parent / 'outlier_experiment' / 'run_experiment.py')  # both files are run_experiment.py)
_ol = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_ol)
inject_outliers, cluster_stats = _ol.inject_outliers, _ol.cluster_stats

# ---- model config ----
RF_TREES, XGB_TREES, XGB_DEPTH = 60, 100, 3
N_FOLDS = 5
# RF never hangs but is slow (~42s/cell) -> generous timeout. XGB hangs pervasively on the
# flat-facet pathology (even the clean baseline) but is fast when it converges -> short timeout
# so hung cells die quickly (recorded NaN) instead of wasting 2 min each.
TIMEOUT_BY_MODEL = {'rf': 90, 'xgb': 25, 'tree': 25}  # tree+HSJ hangs like xgb -> short timeout

# ---- defect grids (pilot defaults; expand here to go full) ----
CG_BIAS = [0.1, 0.3, 0.5, 0.7, 0.9]
CG_SEEDS = [42, 58, 125]
CG_TC = [0, 1, 2]
CG_FEAT = [2]                                   # petal length (most informative); full = [0,1,2,3]
LN_NOISE = [0.1, 0.2, 0.3, 0.4, 0.5]
LN_SEEDS = [42, 58, 125]
LN_NOISE_SEEDS = [0, 1, 2]                       # full = range(12)
OL_K = [4, 8]
OL_N = [5, 10]
OL_N_CEILING = [20, 30, 40]                                 # push RF to the class-size ceiling
OL_DIR = ['toward', 'outward', 'random']              # random added (meeting ask)
OL_SEEDS = [42, 58, 125, 7, 13, 21, 99, 123, 200, 777]  # 10 seeds to firm up the white-box/black-box flip
OL_TC = 2
MODELS = ['rf', 'tree']                               # xgb deferred (hangs); 'tree' = single overfit tree + HSJ (black-box on tree)

COLS = ['defect', 'model', 'attack', 'seed', 'noise_seed', 'bias', 'noise', 'tc', 'feat',
        'k', 'n_out', 'direction', 'tacc', 'vacc', 'asucc', 'nadv', 'density', 'nclust',
        'mean_dist', 'clust_size', 'aiden_density', 'spread_c0', 'spread_c1', 'spread_c2']
KEYCOLS = ['defect', 'model', 'seed', 'noise_seed', 'bias', 'noise', 'tc', 'feat', 'k', 'n_out', 'direction']


def blank(**kw):
    r = {'defect': '', 'model': '', 'attack': 'hsj', 'seed': -1, 'noise_seed': -1, 'bias': -1.0,
         'noise': -1.0, 'tc': -1, 'feat': -1, 'k': -1.0, 'n_out': -1, 'direction': 'none'}
    r.update(kw); return r


def build_grid():
    jobs = []
    for m in MODELS:
        for b, s, tc, f in product(CG_BIAS, CG_SEEDS, CG_TC, CG_FEAT):
            jobs.append(blank(defect='coverage_gap', model=m, seed=s, bias=round(b, 2), tc=tc, feat=f))
        for nz, s, ns in product(LN_NOISE, LN_SEEDS, LN_NOISE_SEEDS):
            jobs.append(blank(defect='label_noise', model=m, seed=s, noise_seed=ns, noise=round(nz, 2)))
        for s in OL_SEEDS:
            jobs.append(blank(defect='outlier', model=m, seed=s, tc=OL_TC, k=0.0, n_out=0))   # baseline
            for d, k, n in product(OL_DIR, OL_K, OL_N):
                jobs.append(blank(defect='outlier', model=m, seed=s, tc=OL_TC, k=float(k), n_out=n, direction=d))
            if '--ceiling' in sys.argv and m == 'rf':   # RF ceiling: push toward/k=8 to n=20,30,40
                for n in OL_N_CEILING:
                    jobs.append(blank(defect='outlier', model=m, seed=s, tc=OL_TC, k=8.0, n_out=n, direction='toward'))
    return jobs


def key(r):
    return tuple(r[c] if c not in ('bias', 'k', 'noise') else round(float(r[c]), 2) for c in KEYCOLS)


def _argval(flag):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else None


def apply_filter(grid):
    d, m = _argval('--defect'), _argval('--model')
    if d:
        grid = [j for j in grid if j['defect'] == d]
    if m:
        grid = [j for j in grid if j['model'] == m]
    return grid


def filter_args():
    args = []
    for flag in ('--defect', '--model'):
        v = _argval(flag)
        if v:
            args += [flag, v]
    if '--ceiling' in sys.argv:       # boolean flag must reach the worker or its grid omits the ceiling cells
        args.append('--ceiling')
    return args


def done_keys():
    if not OUT.exists():
        return set()
    d = pl.read_parquet(OUT)
    return {key(row) for row in d.select(KEYCOLS).to_dicts()}


def append_row(r):
    for c in COLS:
        r.setdefault(c, np.nan)
    new = pl.DataFrame([{c: r[c] for c in COLS}])
    merged = pl.concat([pl.read_parquet(OUT), new]) if OUT.exists() else new
    merged.write_parquet(OUT)


# ─────────────────────────── model + attack ───────────────────────────
def make_and_wrap(model, Xt, yt):
    from sklearn.ensemble import RandomForestClassifier
    from art.estimators.classification import SklearnClassifier, XGBoostClassifier
    if model == 'rf':
        m = RandomForestClassifier(n_estimators=RF_TREES, random_state=42).fit(Xt, yt)
        return SklearnClassifier(m), m
    if model == 'tree':
        from sklearn.tree import DecisionTreeClassifier
        m = DecisionTreeClassifier(max_depth=None, random_state=42).fit(Xt, yt)
        return SklearnClassifier(m), m
    from xgboost import XGBClassifier
    m = XGBClassifier(n_estimators=XGB_TREES, max_depth=XGB_DEPTH, random_state=42,
                      eval_metric='mlogloss').fit(Xt, yt)
    return XGBoostClassifier(model=m, nb_features=Xt.shape[1], nb_classes=3), m


def attack_and_measure(art, Xv, yv):
    from art.attacks.evasion import HopSkipJump
    p = np.argmax(art.predict(Xv), axis=1)
    c = p == yv
    vacc = float((p == yv).mean())
    if c.sum() == 0:
        return vacc, 0.0, np.empty((0, Xv.shape[1])), np.empty(0, dtype=yv.dtype)
    hs = HopSkipJump(classifier=art, norm=2, max_iter=10, max_eval=200, init_eval=50, verbose=False)
    adv = hs.generate(Xv[c])
    ap = np.argmax(art.predict(adv), axis=1)
    fl = ap != yv[c]
    return vacc, float(c.mean()), adv[fl], yv[c][fl]


# ─────────────────────────── defect injection ───────────────────────────
def inject_cg(X, y, tc, feat, bias):
    mask = y == tc; order = np.argsort(X[mask][:, feat])
    Xs, ys = X[mask][order], y[mask][order]
    nk = max(int(len(Xs) * (1 - bias)), 3)
    return np.vstack([X[~mask], Xs[-nk:]]), np.hstack([y[~mask], ys[-nk:]])


def flip_labels(y_train, noise, rng):
    y = y_train.copy(); n = int(len(y) * noise); classes = np.unique(y)
    for idx in rng.choice(len(y), size=n, replace=False):
        y[idx] = rng.choice([c for c in classes if c != y[idx]])
    return y


# ─────────────────────────── WORKER (one row) ───────────────────────────
def worker():
    from sklearn.datasets import load_iris
    from sklearn.model_selection import StratifiedKFold
    dk = done_keys()
    job = next((j for j in apply_filter(build_grid()) if key(j) not in dk), None)
    if job is None:
        print('worker: nothing to do', flush=True); return
    X, y = load_iris(return_X_y=True)
    defect, model, seed = job['defect'], job['model'], job['seed']

    if defect == 'coverage_gap':                       # injected before split (matches existing CG grids)
        np.random.seed(seed)
        Xg, yg = inject_cg(X, y, job['tc'], job['feat'], job['bias'])
    else:
        Xg, yg = X, y
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    rng = np.random.default_rng((job['noise_seed'] if job['noise_seed'] >= 0 else 0) * 1000 + seed)
    classes = [0, 1, 2]
    folds = []
    for tr, te in skf.split(Xg, yg):
        Xt, Xv, yt, yv = Xg[tr], Xg[te], yg[tr], yg[te]
        if defect == 'label_noise':
            yt = flip_labels(yt, job['noise'], rng)
        elif defect == 'outlier':
            Xt, yt = inject_outliers(Xt, yt, job['tc'], job['k'], job['n_out'],
                                     direction=job['direction'], rng=rng)
        art, m = make_and_wrap(model, Xt, yt)
        tacc = float((np.argmax(art.predict(Xt), axis=1) == yt).mean())
        vacc, succ, adv, labels = attack_and_measure(art, Xv, yv)
        dens, nc, md, cs, aiden = cluster_stats(adv)
        rec = {'tacc': tacc, 'vacc': vacc, 'asucc': succ, 'nadv': len(adv), 'density': dens,
               'nclust': nc, 'mean_dist': md, 'clust_size': cs, 'aiden_density': aiden}
        for cl in classes:
            sub = adv[labels == cl] if len(adv) else adv
            rec[f'spread_c{cl}'] = cluster_stats(sub)[2]
        folds.append(rec)
    r = {kk: float(np.nanmean([f[kk] for f in folds])) for kk in folds[0]}
    r.update(job)
    append_row(r)
    print(f"worker done: {defect} {model} " + " ".join(f"{k}={job[k]}" for k in KEYCOLS[2:] if job[k] not in (-1, -1.0, 'none')), flush=True)


# ─────────────────────────── DRIVER ───────────────────────────
def driver(grid):
    total = len(grid)
    t0 = time.time()
    print(f'Driver: {total} rows, subprocess-per-row, timeouts={TIMEOUT_BY_MODEL}', flush=True)
    while True:
        dk = done_keys()
        remaining = [j for j in grid if key(j) not in dk]
        have = total - len(remaining)
        if not remaining:
            break
        el = time.time() - t0; pct = have / total if total else 1
        print(f'[{have}/{total}] {pct:.0%} elapsed={el/60:.1f}m', flush=True)
        PROGRESS.write_text(f'[{have}/{total}] elapsed={el/60:.1f}m\n')
        job = remaining[0]
        to = TIMEOUT_BY_MODEL.get(job['model'], 90)
        p = subprocess.Popen([sys.executable, __file__, '--worker'] + filter_args(), cwd=str(HERE),
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            out, err = p.communicate(timeout=to)
            if p.returncode != 0:
                print('  worker error:', (err or '')[-300:], flush=True)
        except subprocess.TimeoutExpired:
            p.kill(); p.communicate()
            append_row(dict(job))          # NaN metrics placeholder -> never retried
            print(f'  >>> HUNG, skipped: {job["defect"]} {job["model"]} {key(job)}', flush=True)
    n_skip = int(pl.read_parquet(OUT)['mean_dist'].is_nan().sum()) if OUT.exists() else 0
    print(f'DONE: {total} rows in {(time.time()-t0)/60:.1f}m ({n_skip} NaN/skipped)', flush=True)


def main():
    a = sys.argv
    if '--worker' in a:
        worker(); return
    if '--smoke' in a:
        # one representative cell per (model, defect); run inline (no subprocess) with timing
        from sklearn.datasets import load_iris
        from sklearn.model_selection import StratifiedKFold
        globals()['load_iris'] = load_iris
        smoke = []
        for m in MODELS:
            smoke += [blank(defect='coverage_gap', model=m, seed=42, bias=0.5, tc=0, feat=2),
                      blank(defect='label_noise', model=m, seed=42, noise_seed=0, noise=0.3),
                      blank(defect='outlier', model=m, seed=42, tc=2, k=8.0, n_out=10, direction='toward')]
        for job in smoke:
            t0 = time.time()
            _run_one_inline(job)
            print(f'  smoke {job["defect"]:>13} {job["model"]:>3}: {time.time()-t0:.1f}s', flush=True)
        return
    driver(apply_filter(build_grid()))


def _run_one_inline(job):
    """Smoke helper: run a single job inline (no subprocess), append result."""
    from sklearn.datasets import load_iris
    from sklearn.model_selection import StratifiedKFold
    X, y = load_iris(return_X_y=True)
    defect, model, seed = job['defect'], job['model'], job['seed']
    if defect == 'coverage_gap':
        np.random.seed(seed); Xg, yg = inject_cg(X, y, job['tc'], job['feat'], job['bias'])
    else:
        Xg, yg = X, y
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed)
    folds = []
    for tr, te in skf.split(Xg, yg):
        Xt, Xv, yt, yv = Xg[tr], Xg[te], yg[tr], yg[te]
        if defect == 'label_noise':
            yt = flip_labels(yt, job['noise'], rng)
        elif defect == 'outlier':
            Xt, yt = inject_outliers(Xt, yt, job['tc'], job['k'], job['n_out'], direction=job['direction'], rng=rng)
        art, m = make_and_wrap(model, Xt, yt)
        vacc, succ, adv, labels = attack_and_measure(art, Xv, yv)
        folds.append({'vacc': vacc, 'nadv': len(adv), 'mean_dist': cluster_stats(adv)[2]})
    md = float(np.nanmean([f['mean_dist'] for f in folds]))
    vacc = float(np.nanmean([f['vacc'] for f in folds]))
    print(f'      -> vacc={vacc:.3f} spread={md:.3f} nadv~{int(np.nanmean([f["nadv"] for f in folds]))}', flush=True)


if __name__ == '__main__':
    main()
