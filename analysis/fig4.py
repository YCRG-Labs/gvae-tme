#!/usr/bin/env python3
"""Figure 4: Rare Cell Detection + Immunosuppressive Validation — publication quality."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
    'legend.fontsize': 7.5,
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
C_SCANPY = '#E69F00'


def panel_label(ax, label, x=-0.14, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top', ha='left', color='black')


def load_npy(subdir, name):
    p = OUTDIR / subdir / f'{name}.npy'
    if p.exists():
        return np.load(p)
    return None


def compute_umap(z, seed=42):
    try:
        import umap
        return umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=seed).fit_transform(z)
    except Exception:
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=seed).fit_transform(z)


def pval_stars(p):
    if p < 1e-3: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return 'n.s.'


# ── Load data ──
z_mel = load_npy('melanoma', 'embeddings')
rs_mel = load_npy('melanoma', 'rare_cell_scores')
cl_mel = load_npy('melanoma', 'cluster_labels')

# ── CellTypist validation data (from results) ──
CELLTYPIST_MEL = {
    'Treg':          dict(prec=0.016, recall=0.006, n=1453),
    'M2 mac.':       dict(prec=0.146, recall=0.355, n=200),
    'MDSC':          dict(prec=0.235, recall=0.130, n=880),
}

CELLTYPIST_BREAST = {
    'M2 mac.':       dict(prec=0.094, recall=0.051, n=6296),
    'MDSC':          dict(prec=0.003, recall=0.020, n=606),
}

IMMUNO_SIG = {
    'Treg':       {'nsclc_vis': (-0.087, 0.0),      'breast': (np.nan, np.nan)},
    'M2 mac.':    {'nsclc_vis': (-0.099, 1.82e-14),  'breast': (-0.048, 8.84e-26)},
    'MDSC':       {'nsclc_vis': (-0.257, 0.0),        'breast': (-0.031, 2.96e-24)},
    'Exhausted T':{'nsclc_vis': (-0.081, 0.0),        'breast': (0.093, 5.13e-8)},
}

# ── Figure ──
fig = plt.figure(figsize=(8.0, 5.8))
gs_top = gridspec.GridSpec(1, 12, left=0.07, right=0.96, top=0.97, bottom=0.56, wspace=2.0)
gs_bot = gridspec.GridSpec(1, 12, left=0.10, right=0.96, top=0.43, bottom=0.08, wspace=3.5)

# ── (a) UMAP with rare cells highlighted ──
ax_a = fig.add_subplot(gs_top[0, :5])
if z_mel is not None and rs_mel is not None:
    umap_mel = compute_umap(z_mel)
    is_rare = rs_mel > 2.0
    ax_a.scatter(umap_mel[~is_rare, 0], umap_mel[~is_rare, 1],
                 s=1, color='#E0E0E0', alpha=0.3, rasterized=True)
    if cl_mel is not None:
        rare_cl = cl_mel[is_rare]
        unique_rc = np.unique(rare_cl)
        cmap_r = plt.colormaps.get_cmap('Set1')
        for j, rc in enumerate(sorted(unique_rc)):
            mask_rc = is_rare & (cl_mel == rc)
            ax_a.scatter(umap_mel[mask_rc, 0], umap_mel[mask_rc, 1],
                         s=5, color=cmap_r(j % 9), alpha=0.8,
                         label=f'Rare C{rc}', edgecolors='white', lw=0.2,
                         rasterized=True)
        ax_a.legend(fontsize=5, markerscale=2, frameon=False, loc='upper right',
                    handletextpad=0.2, ncol=1)
    n_rare = int(is_rare.sum())
    pct = 100 * n_rare / len(is_rare)
    ax_a.text(0.03, 0.04, f'{n_rare} rare cells ({pct:.1f}%)',
              transform=ax_a.transAxes, fontsize=7,
              bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#dddddd', lw=0.4))
ax_a.set_xticks([])
ax_a.set_yticks([])
ax_a.set_xlabel('UMAP 1', fontsize=8)
ax_a.set_ylabel('UMAP 2', fontsize=8)

# ── (b) KL divergence distribution ──
ax_b = fig.add_subplot(gs_top[0, 5:8])
if rs_mel is not None:
    ax_b.hist(rs_mel[rs_mel <= 2.0], bins=40, color='#DDDDDD', edgecolor='#CCCCCC',
              lw=0.3, density=True, label='Non-rare')
    ax_b.hist(rs_mel[rs_mel > 2.0], bins=20, color=C_GVAE, edgecolor='white',
              lw=0.3, density=True, alpha=0.8, label='Rare')
    ax_b.axvline(2.0, color='black', ls='--', lw=0.8)
    ax_b.text(2.1, ax_b.get_ylim()[1] * 0.85, '$\\eta = 2.0$', fontsize=7, va='top')
ax_b.set_xlabel('KL divergence $z$-score')
ax_b.set_ylabel('Density')
ax_b.legend(frameon=False, fontsize=7)

# ── (c) Rare cell fractions across datasets ──
ax_c = fig.add_subplot(gs_top[0, 8:])
datasets_rare = {
    'NSCLC\nscRNA':  49 / 9000,
    'Breast':        3461 / 60971,
    'NSCLC\nVisium': 4172 / 50000,
    'Melanoma':      486 / 16076,
}
dr_names = list(datasets_rare.keys())
dr_pcts = [v * 100 for v in datasets_rare.values()]
ax_c.barh(range(len(dr_names)), dr_pcts, color=C_GVAE, alpha=0.8,
           edgecolor='white', lw=0.3, height=0.55)
for i, pct in enumerate(dr_pcts):
    ax_c.text(pct + 0.15, i, f'{pct:.1f}%', va='center', fontsize=7)
ax_c.set_yticks(range(len(dr_names)))
ax_c.set_yticklabels(dr_names, fontsize=7.5)
ax_c.set_xlabel('Rare cells (%)')
ax_c.set_xlim(0, max(dr_pcts) * 1.45)
ax_c.invert_yaxis()

# ── (d) CellTypist precision by type (melanoma + breast) ──
ax_d = fig.add_subplot(gs_bot[0, :6])
types = ['Treg', 'M2 mac.', 'MDSC']
x_pos = np.arange(len(types))
w = 0.30

mel_prec = [CELLTYPIST_MEL.get(t, {}).get('prec', 0) for t in types]
breast_prec = [CELLTYPIST_BREAST.get(t, {}).get('prec', 0) for t in types]

ax_d.bar(x_pos - w/2 - 0.02, mel_prec, w, color=C_GVAE, alpha=0.85,
         edgecolor='white', lw=0.4, label='Melanoma')
ax_d.bar(x_pos + w/2 + 0.02, breast_prec, w, color=C_SCANPY, alpha=0.85,
         edgecolor='white', lw=0.4, label='Breast')

for i in range(len(types)):
    if mel_prec[i] > 0:
        ax_d.text(x_pos[i] - w/2 - 0.02, mel_prec[i] + 0.005,
                  f'{mel_prec[i]:.3f}', ha='center', fontsize=6, color='black')
    if breast_prec[i] > 0:
        ax_d.text(x_pos[i] + w/2 + 0.02, breast_prec[i] + 0.005,
                  f'{breast_prec[i]:.3f}', ha='center', fontsize=6, color='black')

ax_d.set_xticks(x_pos)
ax_d.set_xticklabels(types)
ax_d.set_ylabel('Precision\n(frac. rare cells that are\nimmunosuppressive)')
ax_d.set_ylim(0, 0.30)
ax_d.legend(frameon=False, fontsize=7, loc='upper right')
ax_d.text(0.02, 0.95, 'Overall: melanoma 0.397, breast 0.097',
          transform=ax_d.transAxes, fontsize=6.5, va='top',
          bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#dddddd', lw=0.4))

# ── (e) Immunosuppressive signature effect sizes ──
ax_e = fig.add_subplot(gs_bot[0, 6:])
sigs = list(IMMUNO_SIG.keys())
dsets = ['nsclc_vis', 'breast']
dset_labels = ['NSCLC Visium', 'Breast']
dset_colors = [C_GVAE, C_SCANPY]
x_e = np.arange(len(sigs))
w_e = 0.30

for j, (ds, dsl, dsc) in enumerate(zip(dsets, dset_labels, dset_colors)):
    effects = []
    pvals = []
    for sig in sigs:
        eff, pv = IMMUNO_SIG[sig][ds]
        effects.append(eff)
        pvals.append(pv)
    effects = np.array(effects)
    mask_valid = ~np.isnan(effects)
    positions = x_e[mask_valid] + (j - 0.5) * (w_e + 0.02)
    ax_e.bar(positions, np.abs(effects[mask_valid]), w_e, color=dsc, alpha=0.8,
             edgecolor='white', lw=0.3, label=dsl)
    pass

ax_e.set_xticks(x_e)
ax_e.set_xticklabels(sigs, fontsize=7.5)
ax_e.set_ylabel('|Effect size|\n(rare $-$ non-rare)')
ax_e.set_ylim(0, 0.32)
ax_e.axhline(0, color='black', lw=0.4)
ax_e.legend(frameon=False, fontsize=7, loc='upper right')

# ── Panel labels ──
panel_label(ax_a, 'a')
panel_label(ax_b, 'b')
panel_label(ax_c, 'c')
panel_label(ax_d, 'd', x=-0.10, y=1.14)
panel_label(ax_e, 'e', x=-0.10)

for ext in ['pdf', 'png']:
    fig.savefig(FIGDIR / f'fig4_rare_cells.{ext}',
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close(fig)
print(f'Saved fig4_rare_cells to {FIGDIR}/')
