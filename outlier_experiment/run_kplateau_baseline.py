"""Matched no-outlier baseline for the k-plateau sweep (final report Fig 3a).

plot_kplateau.py normalised by hand-typed baselines (0.448 / 0.440) taken from the n_out=0 rows of
results_extended.parquet, which used 10 seeds; the sweep itself uses seeds 0-49. This runs the
identical cell (run_kplateau.run_cell) with N_OUT = 0, so the baseline shares the sweep's seeds,
folds and models. inject_outliers returns the training data unchanged when n_out == 0.

-> results_kplateau_baseline.parquet  (model x seed, k = 0)
Usage: python run_kplateau_baseline.py
"""
import importlib.util, time
from pathlib import Path
import numpy as np, polars as pl

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('kp', HERE / 'run_kplateau.py')
kp = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(kp)
kp.N_OUT = 0                                   # run_cell reads the module global
OUT = HERE / 'results_kplateau_baseline.parquet'

if __name__ == '__main__':
    t0 = time.time()
    rows = [kp.run_cell(m, 0, s) for m in kp.MODELS for s in kp.SEEDS]
    pl.DataFrame([{c: r.get(c, np.nan) for c in kp.COLS} for r in rows]).write_parquet(OUT)
    print(f'DONE {len(rows)} rows -> {OUT.name} in {(time.time() - t0) / 60:.1f}m', flush=True)
