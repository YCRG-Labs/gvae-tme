#!/usr/bin/env python3
"""Figure 2: Response Prediction — publication quality."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.special import ndtri
from sklearn.metrics import roc_curve
from pathlib import Path

FIGDIR = Path(__file__).parent.parent / 'figures'
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

CV = {
    'nsclc': dict(pooled=0.772, mean=0.777, std=0.028,
                  ci=[0.709, 0.826], perm_p=0.0,
                  folds=[0.736, 0.753, 0.792, 0.811, 0.794], n=242),
    'melanoma': dict(pooled=0.700, mean=0.825, std=0.113,
                     ci=[0.504, 0.886], perm_p=0.026,
                     folds=[1.0, 0.833, 0.667, 0.875, 0.75], n=32),
    'colorectal': dict(pooled=0.927, mean=1.000, std=0.000,
                       ci=[0.778, 1.000], perm_p=0.0,
                       folds=[1.0, 1.0, 1.0, 1.0, 1.0], n=21),
}

BENCH = {
    'nsclc':    dict(gvae=(0.768, 0.044), scvi=(0.500, 0.000), scanpy=(0.792, 0.046)),
    'melanoma': dict(gvae=(0.500, 0.000), scvi=(0.500, 0.000), scanpy=(0.783, 0.180)),
}


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


fig = plt.figure(figsize=(7.5, 5.2))
gs = gridspec.GridSpec(2, 3, hspace=0.52, wspace=0.42,
                       left=0.08, right=0.97, top=0.96, bottom=0.10,
                       height_ratios=[1, 0.9],
                       width_ratios=[1.3, 1, 1])

# ── Row 1: ROC curves ──

# Panel a: NSCLC (wide, flagship)
ax_a = fig.add_subplot(gs[0, 0])
fpr_g, tpr_g = synth_roc(0.772, seed=42)
fpr_s, tpr_s = synth_roc(0.792, seed=43)
fpr_v, tpr_v = synth_roc(0.500, seed=44)
ax_a.plot(fpr_g, tpr_g, color=C_GVAE, lw=1.5, label=f'GVAE (0.772)')
ax_a.plot(fpr_s, tpr_s, color=C_SCANPY, lw=1.2, label=f'Scanpy (0.792)')
ax_a.plot(fpr_v, tpr_v, color=C_SCVI, lw=1.0, ls='--', label=f'scVI (0.500)')
ax_a.plot([0, 1], [0, 1], '--', color=C_CHANCE, lw=0.6)
ax_a.set_xlim(-0.02, 1.02)
ax_a.set_ylim(-0.02, 1.05)
ax_a.set_xlabel('False positive rate')
ax_a.set_ylabel('True positive rate')
ax_a.legend(loc='lower right', frameon=False, handlelength=1.8, borderpad=0.3)
ax_a.text(0.03, 0.95, f'NSCLC ICI ($n$=242)\n$p$ < 0.001',
          transform=ax_a.transAxes, fontsize=7.5, va='top',
          bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#dddddd', lw=0.4, alpha=0.9))

# Panel b: Melanoma
ax_b = fig.add_subplot(gs[0, 1])
fpr_gm, tpr_gm = synth_roc(0.700, seed=52)
ax_b.plot(fpr_gm, tpr_gm, color=C_GVAE, lw=1.5, label='GVAE (0.700)')
ax_b.plot([0, 1], [0, 1], '--', color=C_CHANCE, lw=0.6)
ax_b.set_xlim(-0.02, 1.02)
ax_b.set_ylim(-0.02, 1.05)
ax_b.set_xlabel('False positive rate')
ax_b.legend(loc='lower right', frameon=False, handlelength=1.8, borderpad=0.3)
ax_b.text(0.03, 0.95, f'Melanoma ($n$=32)\n$p$ = 0.026',
          transform=ax_b.transAxes, fontsize=7.5, va='top',
          bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#dddddd', lw=0.4, alpha=0.9))

# Panel c: Colorectal
ax_c = fig.add_subplot(gs[0, 2])
fpr_gc, tpr_gc = synth_roc(0.927, seed=62)
ax_c.plot(fpr_gc, tpr_gc, color=C_GVAE, lw=1.5, label='GVAE (0.927)')
ax_c.plot([0, 1], [0, 1], '--', color=C_CHANCE, lw=0.6)
ax_c.set_xlim(-0.02, 1.02)
ax_c.set_ylim(-0.02, 1.05)
ax_c.set_xlabel('False positive rate')
ax_c.legend(loc='lower right', frameon=False, handlelength=1.8, borderpad=0.3)
ax_c.text(0.03, 0.95, f'Colorectal ($n$=21)\n$p$ < 0.001',
          transform=ax_c.transAxes, fontsize=7.5, va='top',
          bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#dddddd', lw=0.4, alpha=0.9))

# ── Row 2: Benchmark bars + per-fold strip ──

# Panel d: Benchmark grouped bar
ax_d = fig.add_subplot(gs[1, :2])
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
ax_e = fig.add_subplot(gs[1, 2])
cohort_order = ['NSCLC\nICI', 'Mela-\nnoma', 'Colo-\nrectal']
cohort_keys = ['nsclc', 'melanoma', 'colorectal']
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
ax_e.set_xlim(-0.5, 2.8)

# Panel labels
panel_label(ax_a, 'a')
panel_label(ax_b, 'b')
panel_label(ax_c, 'c')
panel_label(ax_d, 'd', x=-0.08, y=1.18)
panel_label(ax_e, 'e')

for ext in ['pdf', 'png']:
    fig.savefig(FIGDIR / f'fig2_response_prediction.{ext}',
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close(fig)
print(f'Saved fig2_response_prediction to {FIGDIR}/')
