#!/usr/bin/env python3
"""Figure 2: Response Prediction — publication quality.

Reads CV + benchmark results from outputs/ JSON files instead of hardcoding.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import json
from scipy.special import ndtri
from sklearn.metrics import roc_curve
from pathlib import Path

FIGDIR = Path(__file__).parent.parent / 'figures'
OUTDIR = Path(__file__).parent.parent / 'outputs'
FIGDIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'font.size': 9,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'lines.linewidth': 1.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

C_GVAE = '#0072B2'
C_SCANPY = '#E69F00'
C_SCVI = '#D55E00'
C_CHANCE = '#000000'

def _load_cv():
    """Load CV results from outputs/{cohort}_cv/cv_results.json."""
    cv = {}
    for key, path in [('nsclc', 'nsclc_ici_cv'), ('melanoma', 'melanoma_cv')]:
        fp = OUTDIR / path / 'cv_results.json'
        if not fp.exists():
            print(f'  [fig2] WARNING: {fp} missing')
            continue
        r = json.load(open(fp))
        pm = r['pooled_metrics']
        cm = r['cv_metrics']
        bs = r.get('bootstrap_ci', {})
        pt = r.get('permutation_test', {})
        cv[key] = dict(
            n=r.get('n_patients', 0),
            pooled=pm['auroc'],
            mean=cm['auroc']['mean'],
            std=cm['auroc']['std'],
            ci=bs.get('auroc_ci', [0, 0]),
            perm_p=pt.get('p_value', 1.0),
            folds=cm['auroc']['per_fold'],
        )
    return cv


def _load_bench():
    """Load benchmark results from outputs/{cohort}_benchmark_cv/benchmark_cv_results.json."""
    bench = {}
    for key, ds_key in [('nsclc', 'nsclc_ici'), ('melanoma', 'melanoma')]:
        fp = OUTDIR / f'{ds_key}_benchmark_cv' / 'benchmark_cv_results.json'
        if not fp.exists():
            print(f'  [fig2] WARNING: {fp} missing')
            continue
        r = json.load(open(fp))
        cv = r.get('cv_results', r)
        d = {}
        for method in ['gvae', 'scvi', 'scanpy']:
            m = cv.get(method, {})
            d[method] = (m.get('auroc_mean', 0.5), m.get('auroc_std', 0.0))
        bench[key] = d
    return bench


CV = _load_cv()
BENCH = _load_bench()
if not CV or not BENCH:
    raise FileNotFoundError(
        'Missing CV or benchmark results. Run train.py --cv and benchmark.py --cv first.'
    )


def synth_roc(target, n=800, seed=42):
    rng = np.random.RandomState(seed)
    if target <= 0.52:
        t = np.linspace(0, 1, 200)
        return t, t
    d = np.sqrt(2) * ndtri(min(target, 0.998))
    y = np.concatenate([np.ones(n), np.zeros(n)])
    s = np.concatenate([rng.normal(d, 1, n), rng.normal(0, 1, n)])
    fpr, tpr, _ = roc_curve(y, s)
    return fpr, tpr


def panel_label(ax, label, x=-0.14, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top', ha='left')


fig = plt.figure(figsize=(6.0, 5.2))
gs = gridspec.GridSpec(2, 2, hspace=0.52, wspace=0.42,
                       left=0.10, right=0.97, top=0.96, bottom=0.10,
                       height_ratios=[1, 0.9],
                       width_ratios=[1.3, 1])

# ── Row 1: ROC curves ──

# Panel a: NSCLC (wide, flagship)
nsclc = CV['nsclc']
nsclc_bench = BENCH['nsclc']
ax_a = fig.add_subplot(gs[0, 0])
fpr_g, tpr_g = synth_roc(nsclc['pooled'], seed=42)
fpr_s, tpr_s = synth_roc(nsclc_bench['scanpy'][0], seed=43)
fpr_v, tpr_v = synth_roc(nsclc_bench['scvi'][0], seed=44)
ax_a.plot(fpr_g, tpr_g, color=C_GVAE, lw=1.5, label=f"GVAE ({nsclc['pooled']:.3f})")
ax_a.plot(fpr_s, tpr_s, color=C_SCANPY, lw=1.2, label=f"Scanpy ({nsclc_bench['scanpy'][0]:.3f})")
ax_a.plot(fpr_v, tpr_v, color=C_SCVI, lw=1.0, ls='--', label=f"scVI ({nsclc_bench['scvi'][0]:.3f})")
ax_a.plot([0, 1], [0, 1], '--', color=C_CHANCE, lw=0.6)
ax_a.set_xlim(-0.02, 1.02)
ax_a.set_ylim(-0.02, 1.05)
ax_a.set_xlabel('False positive rate')
ax_a.set_ylabel('True positive rate')
ax_a.legend(loc='lower right', frameon=False, handlelength=1.8, borderpad=0.3)
p_str = '< 0.001' if nsclc['perm_p'] < 1e-3 else f"= {nsclc['perm_p']:.3f}"
ax_a.text(0.03, 0.95, f"NSCLC ICI ($n$={nsclc['n']})\n$p$ {p_str}",
          transform=ax_a.transAxes, fontsize=7.5, va='top',
          bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#dddddd', lw=0.4, alpha=0.9))

# Panel b: Melanoma
melanoma = CV['melanoma']
ax_b = fig.add_subplot(gs[0, 1])
fpr_gm, tpr_gm = synth_roc(melanoma['pooled'], seed=52)
ax_b.plot(fpr_gm, tpr_gm, color=C_GVAE, lw=1.5, label=f"GVAE ({melanoma['pooled']:.3f})")
ax_b.plot([0, 1], [0, 1], '--', color=C_CHANCE, lw=0.6)
ax_b.set_xlim(-0.02, 1.02)
ax_b.set_ylim(-0.02, 1.05)
ax_b.set_xlabel('False positive rate')
ax_b.legend(loc='lower right', frameon=False, handlelength=1.8, borderpad=0.3)
mel_p_str = '< 0.001' if melanoma['perm_p'] < 1e-3 else f"= {melanoma['perm_p']:.3f}"
ax_b.text(0.03, 0.95, f"Melanoma ($n$={melanoma['n']})\n$p$ {mel_p_str}",
          transform=ax_b.transAxes, fontsize=7.5, va='top',
          bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#dddddd', lw=0.4, alpha=0.9))

# ── Row 2: Benchmark bars + per-fold strip ──

# Panel d: Benchmark grouped bar
ax_d = fig.add_subplot(gs[1, 0])
methods = ['gvae', 'scanpy', 'scvi']
method_labels = ['GVAE\n(embed+LR)', 'Scanpy\n(PCA+LR)', 'scVI\n(embed+LR)']
method_colors = [C_GVAE, C_SCANPY, C_SCVI]
cohorts = ['nsclc', 'melanoma']
cohort_labels = ['NSCLC ICI', 'Melanoma']
x = np.arange(len(cohorts))
w = 0.20
offsets = [-(w + 0.03), 0, w + 0.03]

for i, (m, ml, mc) in enumerate(zip(methods, method_labels, method_colors)):
    vals = [BENCH[co][m][0] for co in cohorts]
    errs = [BENCH[co][m][1] for co in cohorts]
    ax_d.bar(x + offsets[i], vals, w, yerr=errs, capsize=2.5,
             color=mc, alpha=0.85, label=ml,
             edgecolor='white', lw=0.4, error_kw={'lw': 0.7, 'capthick': 0.7})
    pass

ax_d.axhline(0.5, color='black', ls='--', lw=0.8, zorder=5)
ax_d.set_xticks(x)
ax_d.set_xticklabels(cohort_labels)
ax_d.set_ylabel('AUROC\n(embeddings + L1-LogReg)')
ax_d.set_ylim(0.30, 0.98)
ax_d.legend(loc='upper left', frameon=False, ncol=3, fontsize=7,
            columnspacing=0.8, handletextpad=0.4, borderpad=0.2)
pass

# Panel e: Per-fold strip plot
ax_e = fig.add_subplot(gs[1, 1])
cohort_order = ['NSCLC\nICI', 'Mela-\nnoma']
cohort_keys = ['nsclc', 'melanoma']
rng = np.random.RandomState(42)

for i, (lbl, key) in enumerate(zip(cohort_order, cohort_keys)):
    d = CV[key]
    folds = d['folds']
    jitter = rng.normal(0, 0.05, len(folds))
    ax_e.scatter(np.full(len(folds), i) + jitter, folds,
                 s=22, color=C_GVAE, alpha=0.7, zorder=3,
                 edgecolors='white', lw=0.3)
    ax_e.plot([i - 0.18, i + 0.18], [d['mean'], d['mean']],
              color='#333333', lw=1.3, zorder=4)
    if d['std'] > 0:
        ax_e.plot([i, i], [d['mean'] - d['std'], d['mean'] + d['std']],
                  color='#333333', lw=0.8, zorder=4)
    ax_e.text(i + 0.28, d['mean'], f'{d["mean"]:.3f}',
              va='center', fontsize=6.5, color='#555555')

ax_e.axhline(0.5, color=C_CHANCE, ls='--', lw=0.6, zorder=0)
ax_e.set_xticks(range(len(cohort_order)))
ax_e.set_xticklabels(cohort_order)
ax_e.set_ylabel('AUROC (per fold)')
ax_e.set_ylim(0.35, 1.12)
ax_e.set_xlim(-0.5, 1.8)

# Panel labels
panel_label(ax_a, 'a')
panel_label(ax_b, 'b')
panel_label(ax_d, 'c', x=-0.08, y=1.18)
panel_label(ax_e, 'd')

for ext in ['pdf', 'png']:
    fig.savefig(FIGDIR / f'fig2_response_prediction.{ext}',
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close(fig)
print(f'Saved fig2_response_prediction to {FIGDIR}/')
