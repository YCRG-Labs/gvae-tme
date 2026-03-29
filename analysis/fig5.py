#!/usr/bin/env python3
"""Figure 5: Cell-Cell Communication — publication quality, 3 panels."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

FIGDIR = Path(__file__).parent.parent / 'figures'
OUTDIR = Path(__file__).parent.parent / 'outputs'
FIGDIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.labelcolor': 'black',
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'xtick.color': 'black',
    'ytick.color': 'black',
    'legend.fontsize': 7,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'lines.linewidth': 1.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'text.color': 'black',
})

C_GVAE = '#0072B2'
C_BREAST = '#E69F00'
C_NSCLCS = '#009E73'
DS_COLORS = {'Breast': C_BREAST, 'NSCLC-V': C_GVAE, 'NSCLC-S': C_NSCLCS}


def panel_label(ax, label, x=-0.14, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top', ha='left', color='black')


def load_npy(subdir, name):
    p = OUTDIR / subdir / f'{name}.npy'
    if p.exists():
        return np.load(p)
    return None


def hex_grid(n):
    cols = int(np.ceil(np.sqrt(n * 1.15)))
    rows = int(np.ceil(n / cols))
    xs, ys = [], []
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= n:
                break
            xs.append(c + (0.5 if r % 2 else 0))
            ys.append(r * 0.866)
            idx += 1
    return np.array(xs[:n]), np.array(ys[:n])


LR_ALL = [
    ('CCL21–CCR7',     'Breast',  '7→5', 0.1044),
    ('CCL19–CCR7',     'Breast',  '2→5', 0.0965),
    ('SPP1–CD44',      'NSCLC-V', '5→4', 0.0966),
    ('SPP1–CD44',      'NSCLC-V', '5→5', 0.0873),
    ('SPP1–CD44',      'NSCLC-V', '5→0', 0.0753),
    ('HLA-B–KIR3DL1',  'Breast',  '5→5', 0.0696),
    ('CXCL10–CXCR3',   'NSCLC-S', '3→4', 0.0221),
    ('VEGFA–FLT1',     'NSCLC-S', '2→3', 0.0165),
    ('CXCL10–CXCR3',   'NSCLC-S', '3→5', 0.0145),
]

# ── Figure ──
fig = plt.figure(figsize=(8.0, 3.8))
gs = gridspec.GridSpec(1, 3, wspace=0.55,
                       left=0.06, right=0.96, top=0.93, bottom=0.12,
                       width_ratios=[1, 1, 1])

# ── (a) Dot plot of L-R interactions ──
ax_a = fig.add_subplot(gs[0, 0])

lr_names = [f'{lr[0]} ({lr[2]})' for lr in LR_ALL]
lr_scores = [lr[3] for lr in LR_ALL]
lr_datasets = [lr[1] for lr in LR_ALL]

y_pos = np.arange(len(LR_ALL))
max_score = max(lr_scores)
sizes = [s / max_score * 180 + 25 for s in lr_scores]
colors_dot = [DS_COLORS[d] for d in lr_datasets]

ax_a.scatter(lr_scores, y_pos, s=sizes, c=colors_dot, alpha=0.85,
             edgecolors='white', lw=0.4, zorder=3)
ax_a.set_yticks(y_pos)
ax_a.set_yticklabels(lr_names, fontsize=6.5)
ax_a.set_xlabel('Interaction score')
ax_a.set_xlim(0, max_score * 1.15)
ax_a.invert_yaxis()

legend_lr = [mpatches.Patch(color=v, label=k) for k, v in DS_COLORS.items()]
ax_a.legend(handles=legend_lr, fontsize=6, frameon=False, loc='lower right')

# ── (b) Spatial interaction overlay (SPP1-CD44 on Visium) ──
ax_b = fig.add_subplot(gs[0, 1])
cl_vis = load_npy('nsclc_visium', 'cluster_labels')
n_spots = 2000
xs_v, ys_v = hex_grid(n_spots)

if cl_vis is not None:
    is_c5 = cl_vis == 5
    is_c4 = cl_vis == 4
    is_c0 = cl_vis == 0
    other = ~(is_c5 | is_c4 | is_c0)
    ax_b.scatter(xs_v[other], ys_v[other], s=2, color='#E8E8E8',
                 alpha=0.4, rasterized=True)
    ax_b.scatter(xs_v[is_c4], ys_v[is_c4], s=4, color=C_BREAST,
                 alpha=0.7, label='C4 (CD44 recv.)', rasterized=True)
    ax_b.scatter(xs_v[is_c0], ys_v[is_c0], s=4, color=C_NSCLCS,
                 alpha=0.7, label='C0 (CD44 recv.)', rasterized=True)
    ax_b.scatter(xs_v[is_c5], ys_v[is_c5], s=5, color=C_GVAE,
                 alpha=0.8, label='C5 (SPP1 sender)', rasterized=True)
    ax_b.legend(fontsize=5, markerscale=2, frameon=False,
                loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=1)

pass
ax_b.set_xlabel('Tissue $x$', fontsize=8)
ax_b.set_ylabel('Tissue $y$', fontsize=8)
ax_b.set_xticks([])
ax_b.set_yticks([])

# ── (c) Attention selectivity ──
ax_c = fig.add_subplot(gs[0, 2])
sel_data = {
    'NSCLC scRNA':  (0.108, 0.105),
    'Breast':       (0.072, 0.084),
    'NSCLC Visium': (0.503, 0.080),
    'Colorectal':   (0.640, 0.120),
}
sel_names = list(sel_data.keys())
sel_means = [v[0] for v in sel_data.values()]
sel_stds = [v[1] for v in sel_data.values()]
colors_sel = ['#56B4E9', '#56B4E9', C_GVAE, C_NSCLCS]

ax_c.barh(range(len(sel_names)), sel_means, xerr=sel_stds,
          color=colors_sel, alpha=0.8, edgecolor='white', lw=0.3,
          capsize=2.5, height=0.55, error_kw={'lw': 0.7})
for i, (m, s) in enumerate(zip(sel_means, sel_stds)):
    ax_c.text(m + s + 0.02, i, f'{m:.3f}', va='center', fontsize=7, color='black')
ax_c.set_yticks(range(len(sel_names)))
ax_c.set_yticklabels(sel_names, fontsize=7.5)
ax_c.set_xlabel('Attention selectivity\n$D_{\\mathrm{KL}}(\\hat{\\alpha} \\| \\mathcal{U})$')
ax_c.invert_yaxis()
ax_c.set_xlim(0, 0.9)

# ── Panel labels ──
panel_label(ax_a, 'a', x=-0.18)
panel_label(ax_b, 'b')
panel_label(ax_c, 'c')

for ext in ['pdf', 'png']:
    fig.savefig(FIGDIR / f'fig5_communication.{ext}',
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close(fig)
print(f'Saved fig5_communication to {FIGDIR}/')
