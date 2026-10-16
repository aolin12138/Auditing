"""PLAN §9 figures — dimensionality decay (Arm 1) + reduction recovery (Arm 3).

Fig 1 (dimensionality_decay): raw spread's spatial-vs-random signal decays as noise dims `m` grow.
  (a) normalised spread vs m (spatial vs random) at a fixed frac — the two arms converge.
  (b) Cohen's d (spatial vs random) vs m for several fracs — decay to ~0 (H1).
  (c) distance concentration std/mean vs m — the measured mechanism (0.54 iris -> 0.07 at m=100).

Fig 2 (reduction_recovery): does dimension reduction recover the buried m=0 signal? (H2)
  Cohen's d vs m per reduction variant, at two fracs. oracle4 = upper bound (knows the 4 real dims);
  raw = decaying reference; discovery variants (pca4/axis_var/axis_mean/lda1) tested against them.

Reads results_dimensionality.parquet (Arm 1) + results_reduction.parquet (Arm 3, optional).
Usage: python plot_dimensionality.py
"""
import os
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_v] = '1'
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, polars as pl
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
import plotstyle as ps

HERE = Path(__file__).resolve().parent
PLOTS = HERE / 'plots' / 'dimensionality'
PLOTS.mkdir(parents=True, exist_ok=True)
CLOUDS = HERE / 'clouds_dim'
M_PAD = [0, 5, 10, 20, 50, 100]
SEEDS = [42, 58, 125, 7, 13, 21, 99, 123, 200, 777] + list(range(300, 320))
ps.apply()


def cohend_ci(a, b):
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, (np.nan, np.nan), (np.nan, np.nan)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    d = (a.mean() - b.mean()) / sp if sp > 0 else np.nan
    return d, (a.mean(), 1.96 * a.std(ddof=1) / np.sqrt(len(a))), (b.mean(), 1.96 * b.std(ddof=1) / np.sqrt(len(b)))


def arm(df, m, structure, frac, col='mean_dist'):
    return df.filter((pl.col('m') == m) & (pl.col('structure') == structure) &
                     (pl.col('frac') == frac))[col].to_numpy()


def concentration_by_m():
    out = {}
    for m in M_PAD:
        qs = []
        for s in SEEDS:
            p = CLOUDS / f'm{m}' / 'random' / f'frac0.00_seed{s}.npz'
            if not p.exists():
                continue
            z = np.load(p)
            for i in range(5):
                a = z[f'adv{i}']
                if len(a) >= 2:
                    qs.append(pdist(a))
        if qs:
            q = np.concatenate(qs)
            out[m] = q.std() / q.mean()
    return out


def fig_decay(df):
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))
    # (a) normalised spread vs m at frac=0.7 (per-m random-baseline normalisation)
    FR = 0.7
    for st in ['spatial', 'random']:
        ys, es = [], []
        for m in M_PAD:
            base = arm(df, m, 'random', 0.0).mean()
            v = arm(df, m, st, FR) / base
            ys.append(v.mean()); es.append(1.96 * v.std(ddof=1) / np.sqrt(len(v)))
        ax[0].errorbar(M_PAD, ys, yerr=es, color=ps.DEFECT[st], ls=ps.DEFECT_LS[st],
                       marker=ps.DEFECT_MK[st], capsize=3, label=ps.DEFECT_LAB[st])
    ax[0].axhline(1.0, color='0.6', lw=1, ls=':')
    ax[0].set_xlabel('padding noise dims  $m$'); ax[0].set_ylabel(f'normalised spread (frac={FR})')
    ax[0].set_title('(a) raw spread arms converge as $m$ grows'); ax[0].legend(frameon=False)
    ps.panel_label(ax[0], 'a')

    # (b) Cohen's d vs m for several fracs
    for fr, c in zip([0.5, 0.7, 0.85], ps.CATEGORICAL):
        ds = []
        for m in M_PAD:
            base = arm(df, m, 'random', 0.0).mean()
            d, _, _ = cohend_ci(arm(df, m, 'spatial', fr) / base, arm(df, m, 'random', fr) / base)
            ds.append(d)
        ax[1].plot(M_PAD, ds, marker='o', color=c, label=f'frac={fr}')
    ax[1].axhline(0, color='0.6', lw=1, ls=':')
    ax[1].axhspan(-0.2, 0.2, color='0.85', alpha=0.5, lw=0)
    ax[1].set_xlabel('padding noise dims  $m$'); ax[1].set_ylabel("Cohen's d  (spatial vs random)")
    ax[1].set_title("(b) signal decays with dimensionality (H1)"); ax[1].legend(frameon=False)
    ps.panel_label(ax[1], 'b')

    # (c) concentration std/mean vs m
    conc = concentration_by_m()
    ax[2].plot(list(conc.keys()), list(conc.values()), marker='D', color=ps.OKABE_ITO['green'])
    ax[2].axhline(0.29, color=ps.OKABE_ITO['vermillion'], lw=1.2, ls='--', label='wine ref (0.29)')
    ax[2].set_xlabel('padding noise dims  $m$'); ax[2].set_ylabel('distance concentration  std/mean')
    ax[2].set_title('(c) mechanism: concentration collapses'); ax[2].legend(frameon=False)
    ps.panel_label(ax[2], 'c')

    fig.tight_layout()
    outs = ps.save(fig, PLOTS / 'dimensionality_decay')
    plt.close(fig)
    print('wrote', *[o.name for o in outs])


def fig_recovery(rd):
    variants = ['oracle4', 'axis_var', 'axis_mean', 'pca4', 'raw']   # lda1 excluded: frac0 baseline degenerate (defect==clean) -> unnormalizable
    fracs = [0.5, 0.7]
    fig, ax = plt.subplots(1, len(fracs), figsize=(11, 4.2), sharey=True)
    for j, fr in enumerate(fracs):
        for vi, variant in enumerate(variants):
            ds = []
            for m in M_PAD:
                sp = rd.filter((pl.col('variant') == variant) & (pl.col('m') == m) &
                               (pl.col('structure') == 'spatial') & (pl.col('frac') == fr))['value_norm'].to_numpy()
                rr = rd.filter((pl.col('variant') == variant) & (pl.col('m') == m) &
                               (pl.col('structure') == 'random') & (pl.col('frac') == fr))['value_norm'].to_numpy()
                d, _, _ = cohend_ci(sp, rr)
                ds.append(d)
            lw = 2.8 if variant in ('oracle4', 'raw') else 2.0
            lss = '-' if variant == 'oracle4' else ('--' if variant == 'raw' else '-')
            ax[j].plot(M_PAD, ds, marker='o', lw=lw, ls=lss, color=ps.CATEGORICAL[vi % len(ps.CATEGORICAL)],
                       label=variant)
        ax[j].axhline(0, color='0.6', lw=1, ls=':')
        ax[j].axhspan(-0.2, 0.2, color='0.85', alpha=0.5, lw=0)
        ax[j].set_xlabel('padding noise dims  $m$'); ax[j].set_title(f'frac={fr}')
        ps.panel_label(ax[j], 'ab'[j])
    ax[0].set_ylabel("Cohen's d  (spatial vs random)")
    ax[-1].legend(frameon=False, title='reduction', ncol=1)
    fig.suptitle('Reduction recovery: does projecting out noise dims rescue the buried signal? (H2)', y=1.02)
    fig.tight_layout()
    outs = ps.save(fig, PLOTS / 'reduction_recovery')
    plt.close(fig)
    print('wrote', *[o.name for o in outs])


def fig_fitvsmeasure(fm):
    PIPES = [('spread_measure_full', 'raw (fit+measure full)', ps.OKABE_ITO['vermillion'], '--', 'o'),
             ('spread_fit_full', 'oracle: reduce AFTER fit', ps.OKABE_ITO['orange'], '-', 's'),
             ('spread_fit_real', 'select real dims BEFORE fit', ps.OKABE_ITO['blue'], '-', 'D')]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for j, fr in enumerate([0.5, 0.7, 0.85]):
        for pipe, lab, col, lss, mk in PIPES:
            ds = []
            for m in M_PAD:
                base = fm.filter((pl.col('m') == m) & (pl.col('structure') == 'random') &
                                 (pl.col('frac') == 0.0))[pipe].to_numpy().mean()
                sp = fm.filter((pl.col('m') == m) & (pl.col('structure') == 'spatial') & (pl.col('frac') == fr))[pipe].to_numpy() / base
                rr = fm.filter((pl.col('m') == m) & (pl.col('structure') == 'random') & (pl.col('frac') == fr))[pipe].to_numpy() / base
                d, _, _ = cohend_ci(sp, rr)
                ds.append(d)
            ax[j].plot(M_PAD, ds, marker=mk, ls=lss, color=col, label=lab)
        ax[j].axhline(0, color='0.6', lw=1, ls=':'); ax[j].axhspan(-0.2, 0.2, color='0.85', alpha=0.5, lw=0)
        ax[j].set_xlabel('padding noise dims  $m$'); ax[j].set_title(f'frac={fr}')
        ps.panel_label(ax[j], 'abc'[j])
    ax[0].set_ylabel("Cohen's d  (spatial vs random)")
    ax[0].legend(frameon=False, fontsize=8)
    fig.suptitle('Signal is destroyed at FIT time, not measurement time: only pre-fit feature selection recovers it',
                 y=1.02)
    fig.tight_layout()
    outs = ps.save(fig, PLOTS / 'fit_vs_measure')
    plt.close(fig)
    print('wrote', *[o.name for o in outs])


def fig_wine_h4(wh):
    KS_WINE = [2, 3, 4, 6, 13]; KS_IRIS = [2, 3, 4]
    def dseries(ds, ks, fr):
        out = []
        for k in ks:
            base = wh.filter((pl.col('dataset') == ds) & (pl.col('k') == k) & (pl.col('structure') == 'random') & (pl.col('frac') == 0.0))['mean_dist'].to_numpy().mean()
            sp = wh.filter((pl.col('dataset') == ds) & (pl.col('k') == k) & (pl.col('structure') == 'spatial') & (pl.col('frac') == fr))['mean_dist'].to_numpy() / base
            rr = wh.filter((pl.col('dataset') == ds) & (pl.col('k') == k) & (pl.col('structure') == 'random') & (pl.col('frac') == fr))['mean_dist'].to_numpy() / base
            d, _, _ = cohend_ci(sp, rr)
            out.append((d, sp.mean(), 1.96 * sp.std(ddof=1) / np.sqrt(len(sp)), rr.mean(), 1.96 * rr.std(ddof=1) / np.sqrt(len(rr))))
        return out
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    # (a) wine normalised spread vs k @ frac0.5 — the gap opens at low k, closes at k=13
    s = dseries('wine', KS_WINE, 0.5)
    ax[0].errorbar(KS_WINE, [x[1] for x in s], yerr=[x[2] for x in s], color=ps.DEFECT['spatial'],
                   ls=ps.DEFECT_LS['spatial'], marker=ps.DEFECT_MK['spatial'], capsize=3, label=ps.DEFECT_LAB['spatial'])
    ax[0].errorbar(KS_WINE, [x[3] for x in s], yerr=[x[4] for x in s], color=ps.DEFECT['random'],
                   ls=ps.DEFECT_LS['random'], marker=ps.DEFECT_MK['random'], capsize=3, label=ps.DEFECT_LAB['random'])
    ax[0].axhline(1.0, color='0.6', lw=1, ls=':')
    ax[0].set_xlabel('wine: # top-F features kept  $k$'); ax[0].set_ylabel('normalised spread (frac=0.5)')
    ax[0].set_title('(a) wine signal recovered by pre-fit selection'); ax[0].legend(frameon=False)
    ax[0].invert_xaxis()   # low-D (recovered) on the right-to-left = toward full 13-D
    ps.panel_label(ax[0], 'a')
    # (b) Cohen's d vs k: wine (decays to 0 at 13) vs iris control (holds)
    for ds, ks, c, mk in [('wine', KS_WINE, ps.OKABE_ITO['vermillion'], 'o'), ('iris', KS_IRIS, ps.OKABE_ITO['blue'], 's')]:
        s = dseries(ds, ks, 0.5)
        ax[1].plot(ks, [x[0] for x in s], marker=mk, color=c, label=f'{ds}')
    ax[1].axhline(0, color='0.6', lw=1, ls=':'); ax[1].axhspan(-0.2, 0.2, color='0.85', alpha=0.5, lw=0)
    ax[1].set_xlabel('# top-F features kept  $k$'); ax[1].set_ylabel("Cohen's d (spatial vs random)")
    ax[1].set_title('(b) wine signal is dimension-limited, not dead'); ax[1].legend(frameon=False, title='dataset')
    ps.panel_label(ax[1], 'b')
    fig.suptitle('H4: the wine defect DOES distort geometry — buried by 13-D, recovered in a low-D discriminative subspace', y=1.02)
    fig.tight_layout()
    outs = ps.save(fig, PLOTS / 'wine_h4_disambiguation')
    plt.close(fig)
    print('wrote', *[o.name for o in outs])


def fig_healthy_control(hc):
    CONDS = [('iris_contested', 'iris\ncontested\n(real defect)'), ('iris_separated', 'iris\nseparated\nclass'),
             ('blobs_healthy', 'well-sep\nblobs'), ('rand_null', 'random\nlabels')]
    fr = 0.5
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    xs = np.arange(len(CONDS)); w = 0.36
    for off, st in [(-w / 2, 'spatial'), (w / 2, 'random')]:
        ms, es = [], []
        for name, _ in CONDS:
            base = hc.filter((pl.col('dataset') == name) & (pl.col('structure') == 'random') & (pl.col('frac') == 0.0))['mean_dist'].to_numpy().mean()
            v = hc.filter((pl.col('dataset') == name) & (pl.col('structure') == st) & (pl.col('frac') == fr))['mean_dist'].to_numpy() / base
            ms.append(v.mean()); es.append(1.96 * v.std(ddof=1) / np.sqrt(len(v)))
        ax.bar(xs + off, ms, w, yerr=es, capsize=3, color=ps.DEFECT[st], label=ps.DEFECT_LAB[st],
               edgecolor='white', linewidth=0.6)
    for i, (name, _) in enumerate(CONDS):
        base = hc.filter((pl.col('dataset') == name) & (pl.col('structure') == 'random') & (pl.col('frac') == 0.0))['mean_dist'].to_numpy().mean()
        sp = hc.filter((pl.col('dataset') == name) & (pl.col('structure') == 'spatial') & (pl.col('frac') == fr))['mean_dist'].to_numpy() / base
        rr = hc.filter((pl.col('dataset') == name) & (pl.col('structure') == 'random') & (pl.col('frac') == fr))['mean_dist'].to_numpy() / base
        d, _, _ = cohend_ci(sp, rr)
        ax.text(i, max(sp.mean(), rr.mean()) + 0.06, f"d={d:+.2f}", ha='center', fontsize=9,
                fontweight='bold' if abs(d) > 0.8 else 'normal')
    ax.axhline(1.0, color='0.6', lw=1, ls=':')
    ax.set_xticks(xs); ax.set_xticklabels([c[1] for c in CONDS])
    ax.set_ylabel('normalised spread (frac=0.5)')
    ax.set_title('Negative control: the spatial>random gap appears ONLY for the real defect (3-D)')
    ax.legend(frameon=False, loc='upper right')
    fig.tight_layout()
    outs = ps.save(fig, PLOTS / 'healthy_control')
    plt.close(fig)
    print('wrote', *[o.name for o in outs])


def fig_highdim_need(hd):
    KS = [2, 3, 4, 6, 8, 10]
    DS = [('redundant', ps.OKABE_ITO['blue'], 's', 'redundant (intrinsic ~3-D)'),
          ('all_informative', ps.OKABE_ITO['vermillion'], 'o', 'all-informative (true 10-D)')]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for name, c, mk, lab in DS:
        accs = [hd.filter((pl.col('dataset') == name) & (pl.col('k') == k) & (pl.col('structure') == 'random') & (pl.col('frac') == 0.0))['vacc'].mean() for k in KS]
        ax[0].plot(KS, accs, marker=mk, color=c, label=lab)
        ds, recs = [], []
        for k in KS:
            base = hd.filter((pl.col('dataset') == name) & (pl.col('k') == k) & (pl.col('structure') == 'random') & (pl.col('frac') == 0.0))['mean_dist'].to_numpy().mean()
            sp = hd.filter((pl.col('dataset') == name) & (pl.col('k') == k) & (pl.col('structure') == 'spatial') & (pl.col('frac') == 0.5))['mean_dist'].to_numpy() / base
            rr = hd.filter((pl.col('dataset') == name) & (pl.col('k') == k) & (pl.col('structure') == 'random') & (pl.col('frac') == 0.5))['mean_dist'].to_numpy() / base
            d, _, _ = cohend_ci(sp, rr)
            ds.append(d)
            rsp = hd.filter((pl.col('dataset') == name) & (pl.col('k') == k) & (pl.col('structure') == 'spatial') & (pl.col('frac') == 0.5))['min_recall'].to_numpy()
            rrr = hd.filter((pl.col('dataset') == name) & (pl.col('k') == k) & (pl.col('structure') == 'random') & (pl.col('frac') == 0.5))['min_recall'].to_numpy()
            recs.append(np.nanmean(rsp) - np.nanmean(rrr))
        ax[1].plot(KS, ds, marker=mk, color=c, label=lab)
        ax[2].plot(KS, recs, marker=mk, color=c, label=lab)
    ax[0].set_ylabel('clean test accuracy'); ax[0].set_title('(a) accuracy vs #features kept')
    ax[0].legend(frameon=False, fontsize=8); ps.panel_label(ax[0], 'a')
    ax[1].axhline(0, color='0.6', lw=1, ls=':'); ax[1].axhspan(-0.2, 0.2, color='0.85', alpha=0.5, lw=0)
    ax[1].set_ylabel("spread Cohen's d (spatial vs random)"); ax[1].set_title('(b) geometry probe signal')
    ps.panel_label(ax[1], 'b')
    ax[2].axhline(0, color='0.6', lw=1, ls=':')
    ax[2].set_ylabel('minority recall gap (spatial − random)'); ax[2].set_title('(c) recall detector')
    ps.panel_label(ax[2], 'c')
    for a in ax:
        a.set_xlabel('# top-F features kept  $k$')
    fig.suptitle('When the problem TRULY needs high-D (orange): reducing costs accuracy AND the geometry probe finds no signal', y=1.02)
    fig.tight_layout()
    outs = ps.save(fig, PLOTS / 'highdim_need')
    plt.close(fig)
    print('wrote', *[o.name for o in outs])


def main():
    df = pl.read_parquet(HERE / 'results_dimensionality.parquet')
    fig_decay(df)
    rp = HERE / 'results_reduction.parquet'
    if rp.exists():
        fig_recovery(pl.read_parquet(rp))
    else:
        print('(results_reduction.parquet not ready — skipping recovery figure)')
    fmp = HERE / 'results_fitvsmeasure.parquet'
    if fmp.exists():
        fig_fitvsmeasure(pl.read_parquet(fmp))
    else:
        print('(results_fitvsmeasure.parquet not ready — skipping fit-vs-measure figure)')
    whp = HERE / 'results_wine_h4.parquet'
    if whp.exists():
        fig_wine_h4(pl.read_parquet(whp))
    else:
        print('(results_wine_h4.parquet not ready — skipping wine-h4 figure)')
    hcp = HERE / 'results_healthy_control.parquet'
    if hcp.exists():
        fig_healthy_control(pl.read_parquet(hcp))
    else:
        print('(results_healthy_control.parquet not ready — skipping healthy-control figure)')
    hdp = HERE / 'results_highdim_need.parquet'
    if hdp.exists():
        fig_highdim_need(pl.read_parquet(hdp))
    else:
        print('(results_highdim_need.parquet not ready — skipping highdim-need figure)')


if __name__ == '__main__':
    main()
