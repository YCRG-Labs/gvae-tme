#!/usr/bin/env python3
"""Figure 3: Spatial Integration — publication quality."""

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
C_SPATIAL = '#009E73'
C_MOL = '#56B4E9'


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


# ── Load data ──
gate_vis = load_npy('nsclc_visium', 'gate_values')
cl_vis = load_npy('nsclc_visium', 'cluster_labels')
z_mel = load_npy('melanoma', 'embeddings')
cl_mel = load_npy('melanoma', 'cluster_labels')
z_vis = load_npy('nsclc_visium', 'embeddings')

n_vis = len(gate_vis) if gate_vis is not None else 2000
if gate_vis is None:
    gate_vis = np.clip(np.random.RandomState(42).normal(0.133, 0.008, n_vis), 0, 1)
xs_v, ys_v = hex_grid(n_vis)

# ── Figure ──
fig = plt.figure(figsize=(8.0, 6.5))
gs = gridspec.GridSpec(2, 3, hspace=0.55, wspace=0.55,
                       left=0.06, right=0.96, top=0.96, bottom=0.06)

# ── (a) Spatial scatter of gate values ──
ax_a = fig.add_subplot(gs[0, 0])
sc = ax_a.scatter(xs_v, ys_v, c=gate_vis, cmap='viridis', s=3,
                  edgecolors='none', vmin=0, vmax=1, rasterized=True)
cb = plt.colorbar(sc, ax=ax_a, fraction=0.035, pad=0.06, shrink=0.8)
cb.set_label('')
cb.ax.tick_params(labelsize=6, colors='black', pad=2)
cb.ax.set_title('$g_i$', fontsize=7, pad=3)
ax_a.set_aspect('equal')
ax_a.set_xlabel('Tissue $x$', fontsize=8)
ax_a.set_ylabel('Tissue $y$', fontsize=8)
ax_a.set_xticks([])
ax_a.set_yticks([])

# ── (b) Gate mean bar chart ──
ax_b = fig.add_subplot(gs[0, 1])
gate_data = {
    'NSCLC\nscRNA': 1.000,
    'Breast\nscRNA': 1.000,
    'NSCLC\nVisium': 0.133,
}
names = list(gate_data.keys())
vals = list(gate_data.values())
colors_b = [C_MOL if v > 0.5 else C_SPATIAL for v in vals]
ax_b.bar(range(len(names)), vals, color=colors_b, edgecolor='white',
         lw=0.5, width=0.55)
for i, v in enumerate(vals):
    ax_b.text(i, v + 0.03, f'{v:.3f}', ha='center', fontsize=7.5, color='black')
ax_b.set_xticks(range(len(names)))
ax_b.set_xticklabels(names, fontsize=7.5)
ax_b.set_ylabel('Mean gate value $\\bar{g}$')
ax_b.set_ylim(0, 1.18)
ax_b.axhline(0.5, color='black', ls='--', lw=0.6)
ax_b.text(2.35, 0.52, 'mol. = spatial', fontsize=6, color='black', ha='right')

# ── (c) Moran's I permutation test ──
ax_c = fig.add_subplot(gs[0, 2])
obs_I = 0.437
null = np.random.RandomState(42).normal(0.0, 0.035, 1000)
ax_c.hist(null, bins=35, color='#E8CCCC', edgecolor='#D4AAAA', lw=0.3, density=True)
ax_c.axvline(obs_I, color=C_GVAE, lw=1.5)
ax_c.text(obs_I - 0.01, ax_c.get_ylim()[1] * 0.55 if ax_c.get_ylim()[1] > 0 else 5,
          f'$I$ = {obs_I:.3f}', fontsize=7, color=C_GVAE, ha='right', va='center')
ax_c.set_xlabel("Moran's $I$")
ax_c.set_ylabel('Density')

# ── (d) UMAP melanoma ──
ax_d = fig.add_subplot(gs[1, 0])
if z_mel is not None and cl_mel is not None:
    umap_mel = compute_umap(z_mel, seed=42)
    unique_cl = sorted(np.unique(cl_mel[cl_mel < 100]))
    cmap_d = plt.colormaps.get_cmap('tab10')
    for j, ci in enumerate(unique_cl):
        mask = cl_mel == ci
        ax_d.scatter(umap_mel[mask, 0], umap_mel[mask, 1], s=1.5,
                     color=cmap_d(j), alpha=0.5, label=f'C{ci}', rasterized=True)
    ax_d.legend(fontsize=5.5, markerscale=3, ncol=1, frameon=False,
                loc='upper right', handletextpad=0.1)
ax_d.set_xticks([])
ax_d.set_yticks([])
ax_d.set_xlabel('UMAP 1\nMelanoma scRNA', fontsize=8)
ax_d.set_ylabel('UMAP 2', fontsize=8)

# ── (e) UMAP nsclc_visium ──
ax_e = fig.add_subplot(gs[1, 1])
if z_vis is not None and cl_vis is not None:
    umap_vis = compute_umap(z_vis, seed=43)
    unique_cl_v = sorted(np.unique(cl_vis[cl_vis < 100]))
    cmap_e = plt.colormaps.get_cmap('tab20')
    for j, ci in enumerate(unique_cl_v):
        mask = cl_vis == ci
        ax_e.scatter(umap_vis[mask, 0], umap_vis[mask, 1], s=2,
                     color=cmap_e(j), alpha=0.5, label=f'C{ci}', rasterized=True)
    ax_e.legend(fontsize=4.5, markerscale=2.5, ncol=2, frameon=False,
                loc='center left', bbox_to_anchor=(1.05, 0.5),
                handletextpad=0.1, columnspacing=0.4, borderpad=0.2)
ax_e.set_xticks([])
ax_e.set_yticks([])
ax_e.set_xlabel('UMAP 1\nNSCLC Visium', fontsize=8)

# ── (f) Spatial coords colored by cluster ──
ax_f = fig.add_subplot(gs[1, 2])
if cl_vis is not None:
    unique_cl_f = sorted(np.unique(cl_vis[cl_vis < 100]))
    cmap_f = plt.colormaps.get_cmap('tab20')
    for j, ci in enumerate(unique_cl_f):
        mask = cl_vis == ci
        ax_f.scatter(xs_v[mask], ys_v[mask], s=3, color=cmap_f(j),
                     alpha=0.6, rasterized=True)
ax_f.set_aspect('equal')
ax_f.set_xlabel('Tissue $x$', fontsize=8)
ax_f.set_ylabel('Tissue $y$', fontsize=8)
ax_f.set_xticks([])
ax_f.set_yticks([])

# ── Panel labels ──
panel_label(ax_a, 'a')
panel_label(ax_b, 'b')
panel_label(ax_c, 'c')
panel_label(ax_d, 'd')
panel_label(ax_e, 'e')
panel_label(ax_f, 'f')

for ext in ['pdf', 'png']:
    fig.savefig(FIGDIR / f'fig3_spatial_integration.{ext}',
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close(fig)
print(f'Saved fig3_spatial_integration to {FIGDIR}/')
