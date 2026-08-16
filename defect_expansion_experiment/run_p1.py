"""Defect-expansion Phase 1 — CLASS IMBALANCE across MODELS via black-box HopSkipJump.

Phase 0 (run_imbalance.py, tree + white-box DTA, 30 seeds) established, at matched COUNT
removal from a class:
  H1  spatial deletion (coverage gap) > random deletion (imbalance) spread, CIs separate frac>=0.5
  H2  minority-class recall is the clean discriminator (spatial craters it; random holds)

Phase 1 asks whether that survives a change of MODEL and ATTACK, and whether it is
class-ASYMMETRIC:
  H1-survive : does spatial > random spread survive black-box HSJ (svm, tree) + ensembling (rf)?
  H-asym     : tc=0 (setosa, separable) vs tc=2 (virginica, contested) — a separable class needs
               no boundary to move, so predicted weaker geometry signal.
  H2-recall  : minority-class recall stays the clean discriminator across models.

Both deletion arms remove from the TRAIN FOLD only (clean test) at MATCHED count. Trees / RF
+ HSJ can HANG on fragmented boundaries -> bulletproof subprocess-per-row driver with a hard
per-model timeout; hung rows recorded NaN and never retried. Resumable.
Reuses cluster_stats (../outlier_experiment) and inject_delete (./run_imbalance.py).

Usage:
  python run_p1.py --smoke                 # 1 cell per model, inline, timing + schema check
  python run_p1.py --model svm             # run one model (svm|tree|rf)
  python run_p1.py --worker [--model M]    # internal: do one undone row
  python run_p1.py                         # run everything (svm, tree, rf)
"""
import os
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_v] = '1'
import sys, time, subprocess, warnings, importlib.util
from pathlib import Path
from itertools import product
import numpy as np, polars as pl
warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results_p1.parquet'
PROGRESS = HERE / 'progress_p1.txt'

# reuse: cluster_stats (OPTICS spread) + inject_delete (random vs spatial deletion, matched count)
def _load(mod_name, path):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
_ol = _load('outlier_lib', HERE.parent / 'outlier_experiment' / 'run_experiment.py')
_imb = _load('imb_lib', HERE / 'run_imbalance.py')
cluster_stats = _ol.cluster_stats
inject_delete = _imb.inject_delete

# ---- experiment configuration ----
N_FOLDS = 5
FEAT = 2                                    # petal length (spatial arm sorts by this)
FRAC_POS = [0.25, 0.5, 0.7, 0.85, 0.95]     # severities (matched count both arms) — fast models
RF_FRAC_POS = [0.5, 0.85, 0.95]             # rf is ~73s/cell -> high-severity subset where the gap appears
STRUCT = ['random', 'spatial']              # imbalance vs coverage-gap control (key factor)
TC = [0, 2]                                 # setosa(separable) vs virginica(contested) — asymmetry
SEEDS = [42, 58, 125, 7, 13, 21, 99, 123, 200, 777]      # 10 seeds
SEEDS_FAST = SEEDS                          # tree/svm — fast, tighten CIs
SEEDS_RF = SEEDS[:3]                        # rf — slow (~40s/cell, may hang), 3 seeds
MODELS = ['svm', 'tree', 'rf']              # HSJ on each; tree = single overfit tree (black-box control)
TIMEOUT_BY_MODEL = {'svm': 40, 'tree': 25, 'rf': 100}
RF_TREES = 60

COLS = ['defect', 'model', 'attack', 'structure', 'tc', 'feat', 'frac', 'seed',
        'tacc', 'vacc', 'recall_c0', 'recall_c1', 'recall_c2', 'min_recall',
        'nadv', 'asucc', 'density', 'nclust', 'mean_dist', 'clust_size',
        'spread_c0', 'spread_c1', 'spread_c2']
KEYCOLS = ['model', 'structure', 'tc', 'frac', 'seed']


def blank(**kw):
    r = {'defect': 'imbalance', 'model': '', 'attack': 'hsj', 'structure': 'baseline',
         'tc': -1, 'feat': FEAT, 'frac': 0.0, 'seed': -1}
    r.update(kw); return r


def build_grid():
    jobs = []
    for m in MODELS:
        seeds = SEEDS_RF if m == 'rf' else SEEDS_FAST
        fracs = RF_FRAC_POS if m == 'rf' else FRAC_POS
        for seed in seeds:
            jobs.append(blank(model=m, seed=seed))                                  # clean baseline (frac=0)
            for struct, frac, tc in product(STRUCT, fracs, TC):
                jobs.append(blank(model=m, structure=struct, tc=tc, frac=round(frac, 2), seed=seed))
    return jobs


def key(r):
    return (r['model'], r['structure'], int(r['tc']), round(float(r['frac']), 2), int(r['seed']))


def _argval(flag):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else None


def apply_filter(grid):
    m = _argval('--model')
    return [j for j in grid if j['model'] == m] if m else grid


def filter_args():
    args = []
    m = _argval('--model')
    if m:
        args += ['--model', m]
    ov = _argval('--out')          # MUST reach the worker or it writes/reads a different file (see Attempt 11 bug)
    if ov:
        args += ['--out', ov]
    return args


def done_keys():
    if not OUT.exists():
        return set()
    return {key(row) for row in pl.read_parquet(OUT).select(KEYCOLS).to_dicts()}


def append_row(r):
    for c in COLS:
        r.setdefault(c, np.nan)
    new = pl.DataFrame([{c: r[c] for c in COLS}])
    merged = pl.concat([pl.read_parquet(OUT), new]) if OUT.exists() else new
    merged.write_parquet(OUT)


# ─────────────────────────── model + attack ───────────────────────────
def make_and_wrap(model, Xt, yt):
    from art.estimators.classification import SklearnClassifier
    if model == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier(n_estimators=RF_TREES, random_state=42).fit(Xt, yt)
    elif model == 'svm':
        from sklearn.svm import SVC
        m = SVC(kernel='rbf', probability=True, random_state=42).fit(Xt, yt)
    else:  # tree — single overfit tree (black-box control vs the Phase-0 white-box DTA)
        from sklearn.tree import DecisionTreeClassifier
        m = DecisionTreeClassifier(max_depth=None, random_state=42).fit(Xt, yt)
    return SklearnClassifier(m), m


def attack_and_measure(art, Xv, yv, classes=(0, 1, 2)):
    """Returns vacc, per-class recall dict, asucc, adv points, adv labels (all HSJ, norm=2)."""
    from art.attacks.evasion import HopSkipJump
    p = np.argmax(art.predict(Xv), axis=1)
    vacc = float((p == yv).mean())
    recall = {}
    for cl in classes:
        mask = yv == cl
        recall[cl] = float((p[mask] == cl).mean()) if mask.sum() else np.nan
    c = p == yv
    if c.sum() == 0:
        return vacc, recall, 0.0, np.empty((0, Xv.shape[1])), np.empty(0, dtype=yv.dtype)
    hs = HopSkipJump(classifier=art, norm=2, max_iter=10, max_eval=200, init_eval=50, verbose=False)
    adv = hs.generate(Xv[c])
    ap = np.argmax(art.predict(adv), axis=1)
    fl = ap != yv[c]
    return vacc, recall, float(c.mean()), adv[fl], yv[c][fl]


# ─────────────────────────── one cell (5-fold mean) ───────────────────────────
def run_cell(job):
    from sklearn.datasets import load_iris
    from sklearn.model_selection import StratifiedKFold
    X, y = load_iris(return_X_y=True)
    model, structure, tc, frac, seed = job['model'], job['structure'], job['tc'], job['frac'], job['seed']
    classes = [0, 1, 2]
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed)
    folds = []
    for tr, te in skf.split(X, y):
        Xt, Xv, yt, yv = X[tr], X[te], y[tr], y[te]
        if frac > 0 and tc >= 0:                                   # baseline (frac=0) => no deletion
            Xt, yt = inject_delete(Xt, yt, tc, FEAT, frac, structure, rng)
        art, m = make_and_wrap(model, Xt, yt)
        tacc = float((np.argmax(art.predict(Xt), axis=1) == yt).mean())
        vacc, recall, asucc, adv, labels = attack_and_measure(art, Xv, yv, classes)
        dens, nc, md, cs, _aiden = cluster_stats(adv)
        rec = {'tacc': tacc, 'vacc': vacc, 'asucc': asucc, 'nadv': len(adv),
               'density': dens, 'nclust': nc, 'mean_dist': md, 'clust_size': cs}
        for cl in classes:
            rec[f'recall_c{cl}'] = recall[cl]
            sub = adv[labels == cl] if len(adv) else adv
            rec[f'spread_c{cl}'] = cluster_stats(sub)[2]
        rec['min_recall'] = recall[tc] if tc >= 0 else np.nan     # depleted-class recall
        folds.append(rec)
    r = {kk: float(np.nanmean([f[kk] for f in folds])) for kk in folds[0]}
    r.update(job)
    return r


# ─────────────────────────── worker / driver ───────────────────────────
def worker():
    global OUT
    ov = _argval('--out')
    if ov:
        OUT = HERE / ov
    dk = done_keys()
    job = next((j for j in apply_filter(build_grid()) if key(j) not in dk), None)
    if job is None:
        print('worker: nothing to do', flush=True); return
    r = run_cell(job)
    append_row(r)
    print(f"worker done: {job['model']} {job['structure']} tc={job['tc']} frac={job['frac']} "
          f"seed={job['seed']} -> vacc={r['vacc']:.3f} spread={r['mean_dist']:.3f} "
          f"min_recall={r['min_recall'] if not np.isnan(r['min_recall']) else float('nan'):.3f}", flush=True)


def driver(grid):
    total = len(grid); t0 = time.time()
    print(f'Driver: {total} rows, subprocess-per-row, timeouts={TIMEOUT_BY_MODEL}', flush=True)
    while True:
        dk = done_keys()
        remaining = [j for j in grid if key(j) not in dk]
        have = total - len(remaining)
        if not remaining:
            break
        el = time.time() - t0
        print(f'[{have}/{total}] {have/total:.0%} elapsed={el/60:.1f}m', flush=True)
        PROGRESS.write_text(f'[{have}/{total}] elapsed={el/60:.1f}m\n')
        job = remaining[0]
        to = TIMEOUT_BY_MODEL.get(job['model'], 60)
        p = subprocess.Popen([sys.executable, __file__, '--worker'] + filter_args(), cwd=str(HERE),
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            out, err = p.communicate(timeout=to)
            if p.returncode != 0:
                print('  worker error:', (err or '')[-300:], flush=True)
        except subprocess.TimeoutExpired:
            p.kill(); p.communicate()
            append_row(dict(job))                       # NaN metrics -> never retried
            print(f'  >>> HUNG, skipped: {key(job)}', flush=True)
    n_skip = int(pl.read_parquet(OUT)['mean_dist'].is_nan().sum()) if OUT.exists() else 0
    print(f'DONE: {total} rows in {(time.time()-t0)/60:.1f}m ({n_skip} NaN/skipped)', flush=True)


def main():
    global OUT
    ov = _argval('--out')
    if ov:
        OUT = HERE / ov
    a = sys.argv
    if '--worker' in a:
        worker(); return
    if '--smoke' in a:
        last = None
        for m in MODELS:
            job = blank(model=m, structure='spatial', tc=2, frac=0.85, seed=42)
            t0 = time.time(); last = run_cell(job)
            print(f'  smoke {m:>4}: {time.time()-t0:5.1f}s  vacc={last["vacc"]:.3f} '
                  f'spread={last["mean_dist"]:.3f} min_recall={last["min_recall"]:.3f} nadv~{int(last["nadv"])}', flush=True)
        missing = [c for c in COLS if c not in last]
        print('  schema check: missing cols =', missing or 'NONE (all present)', flush=True)
        return
    driver(apply_filter(build_grid()))


if __name__ == '__main__':
    main()
