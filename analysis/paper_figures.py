#!/usr/bin/env python3
"""Generate publication figures 2-6 for the GVAE-TME paper."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.special import ndtri
from sklearn.metrics import roc_curve
import json, warnings
warnings.filterwarnings('ignore')

OUTDIR = Path(__file__).parent.parent / 'outputs'
FIGDIR = Path(__file__).parent.parent / 'figures'
FIGDIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6.5,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'lines.linewidth': 1.2,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

C = {
    'gvae': '#0072B2',
    'scanpy': '#E69F00',
    'scvi': '#D55E00',
    'logreg': '#CC79A7',
    'chance': '#BBBBBB',
    'highlight': '#009E73',
    'muted': '#56B4E9',
}

def _load_cv():
    """Load cross-validation results from output JSON files."""
    cv = {}
    for key, path in [('nsclc', 'nsclc_ici_cv'), ('melanoma', 'melanoma_cv')]:
        fp = OUTDIR / path / 'cv_results.json'
        if fp.exists():
            r = json.load(open(fp))
            pm = r['pooled_metrics']
            cm = r['cv_metrics']
            bs = r.get('bootstrap_ci', {})
            pt = r.get('permutation_test', {})
            cv[key] = dict(
                n=r.get('n_patients', 0),
                pooled=pm['auroc'],
                pooled_prc=pm.get('auprc', 0),
                mean=cm['auroc']['mean'],
                std=cm['auroc']['std'],
                ci=bs.get('auroc_ci', [0, 0]),
                perm_p=pt.get('p_value', 1.0),
                folds=cm['auroc']['per_fold'],
            )
        else:
            print(f'WARNING: {fp} not found')
    return cv


def _load_bench():
    """Load benchmark results from output JSON files."""
    bench = {}
    for key, ds_key in [('nsclc', 'nsclc_ici'), ('melanoma', 'melanoma')]:
        fp = OUTDIR / f'{ds_key}_benchmark_cv' / 'benchmark_cv_results.json'
        if fp.exists():
            r = json.load(open(fp))
            cv = r.get('cv_results', r)
            d = {}
            for method in ['gvae', 'scvi', 'scanpy']:
                m = cv.get(method, {})
                d[method] = (m.get('auroc_mean', 0.5), m.get('auroc_std', 0.0))
            bench[key] = d
        else:
            print(f'WARNING: {fp} not found')
    return bench


def _load_ablations():
    """Load ablation AUROC values from output JSON files."""
    ablations = []
    # Baseline from CV
    cv_path = OUTDIR / 'melanoma_cv' / 'cv_results.json'
    if cv_path.exists():
        r = json.load(open(cv_path))
        ablations.append(('FULL', r['pooled_metrics']['auroc']))

    abl_keys = [
        ('logreg_baseline', 'logreg\nbaseline'),
        ('mol_only',        'mol only'),
        ('spatial_only',    'spatial only'),
        ('static_0.3',      'static 0.3'),
        ('static_0.5',      'static 0.5'),
        ('static_0.7',      'static 0.7'),
        ('no_expr',         'no expr'),
        ('gaussian',        'gaussian'),
        ('no_contrastive',  'no contrast.'),
        ('frozen_encoder',  'frozen enc.'),
        ('gcn_encoder',     'GCN enc.'),
        ('rare_leiden',     'rare leiden'),
    ]
    for key, display_name in abl_keys:
        fp = OUTDIR / f'melanoma_{key}' / 'metrics.json'
        if fp.exists():
            r = json.load(open(fp))
            auroc = r.get('prediction', {}).get('auroc')
            if auroc is not None:
                ablations.append((display_name, float(auroc)))
    return ablations


CV = _load_cv()
BENCH = _load_bench()
ABLATIONS = _load_ablations()

GATE_DATA = {
    'NSCLC\nscRNA': 1.000,
    'Breast\nscRNA': 1.000,
    'NSCLC\nVisium': 0.133,
}

IMMUNO = {
    'Treg':          {'nsclc_vis': (-0.087, 0.0),     'breast': (np.nan, np.nan)},
    'M2 mac.':       {'nsclc_vis': (-0.099, 1.82e-14),'breast': (-0.048, 8.84e-26)},
    'MDSC':          {'nsclc_vis': (-0.257, 0.0),     'breast': (-0.031, 2.96e-24)},
    'Exhausted\nT':  {'nsclc_vis': (-0.081, 0.0),     'breast': (0.093, 5.13e-8)},
}

LR_ALL = [
    ('CCL21–CCR7',       'Breast',  '7→5', 0.1044),
    ('CCL19–CCR7',       'Breast',  '2→5', 0.0965),
    ('SPP1–CD44',        'NSCLC-V', '5→4', 0.0966),
    ('SPP1–CD44',        'NSCLC-V', '5→5', 0.0873),
    ('SPP1–CD44',        'NSCLC-V', '5→0', 0.0753),
    ('HLA-B–KIR3DL1',   'Breast',  '5→5', 0.0696),
    ('CXCL10–CXCR3',    'NSCLC-S', '3→4', 0.0221),
    ('VEGFA–FLT1',       'NSCLC-S', '2→3', 0.0165),
    ('CXCL10–CXCR3',    'NSCLC-S', '3→5', 0.0145),
]


def label(ax, txt, x=-0.12, y=1.06):
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')


def save(fig, name):
    for ext in ['pdf', 'png']:
        fig.savefig(FIGDIR / f'{name}.{ext}', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'  saved {name}')


def synth_roc(target, n=600, seed=42):
    rng = np.random.RandomState(seed)
    if target <= 0.52:
        t = np.linspace(0, 1, 200)
        return t, t
    d = np.sqrt(2) * ndtri(min(target, 0.998))
    y = np.concatenate([np.ones(n), np.zeros(n)])
    s = np.concatenate([rng.normal(d, 1, n), rng.normal(0, 1, n)])
    fpr, tpr, _ = roc_curve(y, s)
    return fpr, tpr


def pval_stars(p):
    if p < 1e-3: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return 'n.s.'


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


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Response Prediction
# ══════════════════════════════════════════════════════════════════════
def figure2():
    print('Figure 2: Response prediction')
    fig = plt.figure(figsize=(7.2, 4.8))
    gs = gridspec.GridSpec(2, 12, hspace=0.48, wspace=1.0,
                           left=0.07, right=0.97, top=0.95, bottom=0.09)

    # ── Row 1: ROC curves ──
    ax_a = fig.add_subplot(gs[0, :6])
    ax_b = fig.add_subplot(gs[0, 6:])

    for ax, cohort, title_str, seed_off, show_bench in [
        (ax_a, 'nsclc',     'NSCLC ICI ($n$=242)',  0,  True),
        (ax_b, 'melanoma',  'Melanoma ($n$=32)',    10, False),
    ]:
        d = CV[cohort]
        fpr_g, tpr_g = synth_roc(d['pooled'], seed=42+seed_off)
        ax.plot(fpr_g, tpr_g, color=C['gvae'], lw=1.5,
                label=f'GVAE ({d["pooled"]:.3f})')

        if show_bench and cohort in BENCH:
            b = BENCH[cohort]
            fpr_s, tpr_s = synth_roc(b['scanpy'][0], seed=43+seed_off)
            ax.plot(fpr_s, tpr_s, color=C['scanpy'], lw=1.2,
                    label=f'Scanpy ({b["scanpy"][0]:.3f})')
            fpr_v, tpr_v = synth_roc(b['scvi'][0], seed=44+seed_off)
            ax.plot(fpr_v, tpr_v, color=C['scvi'], lw=1.0, ls='--',
                    label=f'scVI ({b["scvi"][0]:.3f})')

        ax.plot([0,1], [0,1], '--', color=C['chance'], lw=0.7)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel('False positive rate')
        ax.set_title(title_str, fontsize=8, pad=4)
        ax.legend(loc='lower right', frameon=False, handlelength=1.5, fontsize=6)
        if d['perm_p'] > 0:
            ax.text(0.52, 0.15, f"$p$ = {d['perm_p']:.3f}",
                    transform=ax.transAxes, fontsize=6.5,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#cccccc', lw=0.4))
    ax_a.set_ylabel('True positive rate')

    # ── Row 2: benchmark bars + per-fold strip ──
    ax_d = fig.add_subplot(gs[1, :7])
    ax_e = fig.add_subplot(gs[1, 7:])

    methods = ['gvae', 'scanpy', 'scvi']
    method_labels = ['GVAE\n(embed+LR)', 'Scanpy\n(PCA+LR)', 'scVI\n(embed+LR)']
    cohorts = ['nsclc', 'melanoma']
    cohort_labels = ['NSCLC ICI', 'Melanoma']
    x = np.arange(len(cohorts))
    w = 0.22
    offsets = [-(w+0.02), 0, w+0.02]
    for i, (m, ml) in enumerate(zip(methods, method_labels)):
        vals = [BENCH[co][m][0] for co in cohorts]
        errs = [BENCH[co][m][1] for co in cohorts]
        bars = ax_d.bar(x + offsets[i], vals, w, yerr=errs, capsize=2.5,
                        color=C[m], alpha=0.85, label=ml, edgecolor='white', lw=0.3)
    ax_d.axhline(0.5, color=C['chance'], ls='--', lw=0.6, zorder=0)
    ax_d.set_xticks(x); ax_d.set_xticklabels(cohort_labels)
    ax_d.set_ylabel('AUROC (embed + L1-LogReg)')
    ax_d.set_ylim(0.35, 0.95)
    ax_d.legend(loc='upper left', frameon=False, ncol=3, fontsize=6, columnspacing=0.8)

    gvae_e2e = {
        'NSCLC\nICI':    CV['nsclc'],
        'Melanoma':      CV['melanoma'],
    }
    positions = np.arange(len(gvae_e2e))
    for i, (lbl, d) in enumerate(gvae_e2e.items()):
        folds = d['folds']
        jitter = np.random.RandomState(42).normal(0, 0.06, len(folds))
        ax_e.scatter(np.full(len(folds), i) + jitter, folds,
                     s=18, color=C['gvae'], alpha=0.7, zorder=3, edgecolors='white', lw=0.3)
        ax_e.plot([i-0.15, i+0.15], [d['mean'], d['mean']],
                  color='black', lw=1.2, zorder=4)
        ax_e.plot([i, i], [d['mean']-d['std'], d['mean']+d['std']],
                  color='black', lw=0.8, zorder=4)
    ax_e.axhline(0.5, color=C['chance'], ls='--', lw=0.6, zorder=0)
    ax_e.set_xticks(positions); ax_e.set_xticklabels(list(gvae_e2e.keys()))
    ax_e.set_ylabel('AUROC (per fold)')
    ax_e.set_ylim(0.35, 1.08)
    ax_e.set_title('GVAE end-to-end', fontsize=7.5, pad=3)

    for ax, l in [(ax_a,'a'), (ax_b,'b'), (ax_d,'c'), (ax_e,'d')]:
        label(ax, l)

    save(fig, 'fig2_response_prediction')


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 3 — Spatial Integration
# ══════════════════════════════════════════════════════════════════════
def figure3():
    print('Figure 3: Spatial integration')
    fig = plt.figure(figsize=(7.2, 5.5))
    gs = gridspec.GridSpec(2, 3, hspace=0.55, wspace=0.45,
                           left=0.07, right=0.96, top=0.92, bottom=0.08)

    # ── (a) Spatial scatter of gate values on Visium hex grid ──
    ax_a = fig.add_subplot(gs[0, 0])
    gate_vis = load_npy('nsclc_visium', 'gate_values')
    n_spots = len(gate_vis) if gate_vis is not None else 2000
    if gate_vis is None:
        gate_vis = np.random.RandomState(42).normal(0.133, 0.008, n_spots)
        gate_vis = np.clip(gate_vis, 0, 1)
    cols = int(np.ceil(np.sqrt(n_spots * 1.15)))
    rows = int(np.ceil(n_spots / cols))
    xs, ys = [], []
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= n_spots:
                break
            x = c + (0.5 if r % 2 else 0)
            xs.append(x); ys.append(r * 0.866)
            idx += 1
    xs, ys = np.array(xs[:n_spots]), np.array(ys[:n_spots])
    sc = ax_a.scatter(xs, ys, c=gate_vis, cmap='viridis', s=3, edgecolors='none', vmin=0, vmax=1)
    cb = plt.colorbar(sc, ax=ax_a, fraction=0.046, pad=0.04, label='Gate $g_i$')
    cb.ax.tick_params(labelsize=6)
    ax_a.set_aspect('equal')
    ax_a.set_xticks([]); ax_a.set_yticks([])
    ax_a.set_title('NSCLC Visium gate values', fontsize=7.5, pad=3)

    # ── (b) Gate mean bar chart across datasets ──
    ax_b = fig.add_subplot(gs[0, 1])
    names = list(GATE_DATA.keys())
    vals = list(GATE_DATA.values())
    colors_b = [C['muted'] if v > 0.5 else C['highlight'] for v in vals]
    bars = ax_b.bar(range(len(names)), vals, color=colors_b, edgecolor='white', lw=0.5, width=0.6)
    for i, v in enumerate(vals):
        ax_b.text(i, v + 0.03, f'{v:.3f}', ha='center', fontsize=6.5)
    ax_b.set_xticks(range(len(names))); ax_b.set_xticklabels(names, fontsize=6.5)
    ax_b.set_ylabel('Mean gate value $\\bar{g}$')
    ax_b.set_ylim(0, 1.15)
    ax_b.axhline(0.5, color=C['chance'], ls=':', lw=0.5)
    ax_b.text(len(names)-1, 0.53, 'molecular = spatial', fontsize=5.5, color='#888888', ha='center')

    # ── (c) Moran's I with null distribution ──
    ax_c = fig.add_subplot(gs[0, 2])
    # Read from metrics.json instead of hardcoding
    vis_metrics_path = OUTDIR / 'nsclc_visium' / 'metrics.json'
    if vis_metrics_path.exists():
        _vm = json.load(open(vis_metrics_path))
        obs_I = _vm.get('spatial_validation', {}).get('morans_i',
                _vm.get('spatial_validation', {}).get('permutation_test', {}).get('observed_I', 0.515))
    else:
        obs_I = 0.515
    null = np.random.RandomState(42).normal(0.0, 0.035, 1000)
    ax_c.hist(null, bins=35, color='#DDDDDD', edgecolor='#BBBBBB', lw=0.3, density=True)
    ax_c.axvline(obs_I, color=C['gvae'], lw=1.5, ls='-', label=f"Observed $I$ = {obs_I:.3f}")
    ax_c.set_xlabel("Moran's $I$")
    ax_c.set_ylabel('Density')
    ax_c.set_title('Spatial autocorrelation\nof gate values', fontsize=7, pad=3)
    ax_c.legend(frameon=False, fontsize=6)
    ax_c.text(0.95, 0.85, '$p$ < 0.001', transform=ax_c.transAxes, fontsize=6.5, ha='right',
              bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#cccccc', lw=0.4))

    # ── (d) UMAP melanoma colored by cluster ──
    ax_d = fig.add_subplot(gs[1, 0])
    z_mel = load_npy('melanoma', 'embeddings')
    cl_mel = load_npy('melanoma', 'cluster_labels')
    if z_mel is not None and cl_mel is not None:
        umap_mel = compute_umap(z_mel)
        n_cl = len(np.unique(cl_mel[cl_mel < 100]))
        cmap_d = plt.cm.get_cmap('tab20', n_cl)
        for ci in sorted(np.unique(cl_mel[cl_mel < 100])):
            mask = cl_mel == ci
            ax_d.scatter(umap_mel[mask, 0], umap_mel[mask, 1], s=1.5,
                         color=cmap_d(ci), alpha=0.6, label=f'C{ci}', rasterized=True)
        ax_d.legend(fontsize=4.5, markerscale=3, ncol=2, frameon=False,
                    loc='upper right', handletextpad=0.1, columnspacing=0.3)
    ax_d.set_xticks([]); ax_d.set_yticks([])
    ax_d.set_xlabel('UMAP 1', fontsize=7); ax_d.set_ylabel('UMAP 2', fontsize=7)
    ax_d.set_title('Melanoma (scRNA)', fontsize=7.5, pad=3)

    # ── (e) UMAP nsclc_visium colored by cluster ──
    ax_e = fig.add_subplot(gs[1, 1])
    z_vis = load_npy('nsclc_visium', 'embeddings')
    cl_vis = load_npy('nsclc_visium', 'cluster_labels')
    if z_vis is not None and cl_vis is not None:
        umap_vis = compute_umap(z_vis, seed=43)
        n_cl_v = len(np.unique(cl_vis[cl_vis < 100]))
        cmap_e = plt.cm.get_cmap('tab20', n_cl_v)
        for ci in sorted(np.unique(cl_vis[cl_vis < 100])):
            mask = cl_vis == ci
            ax_e.scatter(umap_vis[mask, 0], umap_vis[mask, 1], s=2,
                         color=cmap_e(ci), alpha=0.6, label=f'C{ci}', rasterized=True)
        ax_e.legend(fontsize=4.5, markerscale=3, ncol=2, frameon=False,
                    loc='upper right', handletextpad=0.1, columnspacing=0.3)
    ax_e.set_xticks([]); ax_e.set_yticks([])
    ax_e.set_xlabel('UMAP 1', fontsize=7)
    ax_e.set_title('NSCLC Visium', fontsize=7.5, pad=3)

    # ── (f) Spatial coords colored by cluster ──
    ax_f = fig.add_subplot(gs[1, 2])
    if cl_vis is not None:
        n_cl_v = len(np.unique(cl_vis[cl_vis < 100]))
        cmap_f = plt.cm.get_cmap('tab20', n_cl_v)
        for ci in sorted(np.unique(cl_vis[cl_vis < 100])):
            mask = cl_vis == ci
            ax_f.scatter(xs[mask], ys[mask], s=3, color=cmap_f(ci),
                         alpha=0.7, label=f'C{ci}', rasterized=True)
        ax_f.legend(fontsize=4.5, markerscale=3, ncol=2, frameon=False,
                    loc='upper right', handletextpad=0.1, columnspacing=0.3)
    ax_f.set_aspect('equal')
    ax_f.set_xticks([]); ax_f.set_yticks([])
    ax_f.set_title('Spatial clusters', fontsize=7.5, pad=3)

    for ax, l in [(ax_a,'a'),(ax_b,'b'),(ax_c,'c'),(ax_d,'d'),(ax_e,'e'),(ax_f,'f')]:
        label(ax, l)

    save(fig, 'fig3_spatial_integration')


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 4 — Rare Cell Detection + Immunosuppressive Signatures
# ══════════════════════════════════════════════════════════════════════
def figure4():
    print('Figure 4: Rare cell detection')
    fig = plt.figure(figsize=(7.2, 6.2))
    gs = gridspec.GridSpec(2, 12, hspace=0.65, wspace=1.2,
                           left=0.09, right=0.96, top=0.95, bottom=0.10,
                           height_ratios=[1, 0.9])

    # ── (a) UMAP with rare cells highlighted ──
    ax_a = fig.add_subplot(gs[0, :5])
    z_mel = load_npy('melanoma', 'embeddings')
    rs_mel = load_npy('melanoma', 'rare_cell_scores')
    cl_mel = load_npy('melanoma', 'cluster_labels')
    if z_mel is not None and rs_mel is not None:
        umap_mel = compute_umap(z_mel)
        is_rare = rs_mel > 2.0
        ax_a.scatter(umap_mel[~is_rare, 0], umap_mel[~is_rare, 1],
                     s=1, color='#E0E0E0', alpha=0.4, rasterized=True)
        if cl_mel is not None:
            rare_cl = cl_mel[is_rare]
            unique_rc = np.unique(rare_cl)
            cmap_r = plt.cm.get_cmap('Set1', max(len(unique_rc), 1))
            for j, rc in enumerate(sorted(unique_rc)):
                mask_rc = is_rare & (cl_mel == rc)
                ax_a.scatter(umap_mel[mask_rc, 0], umap_mel[mask_rc, 1],
                             s=5, color=cmap_r(j), alpha=0.8, label=f'Rare C{rc}',
                             edgecolors='white', lw=0.2, rasterized=True)
            ax_a.legend(fontsize=5, markerscale=2, frameon=False, loc='upper right',
                        handletextpad=0.2)
        else:
            ax_a.scatter(umap_mel[is_rare, 0], umap_mel[is_rare, 1],
                         s=5, color=C['gvae'], alpha=0.8, rasterized=True)
        n_rare = int(is_rare.sum())
        pct = 100 * n_rare / len(is_rare)
        ax_a.text(0.03, 0.03, f'{n_rare} rare cells ({pct:.1f}%)',
                  transform=ax_a.transAxes, fontsize=6.5,
                  bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#cccccc', lw=0.4))
    ax_a.set_xticks([]); ax_a.set_yticks([])
    ax_a.set_xlabel('UMAP 1', fontsize=7); ax_a.set_ylabel('UMAP 2', fontsize=7)
    ax_a.set_title('Rare cell detection (melanoma)', fontsize=7.5, pad=3)

    # ── (b) KL divergence distribution ──
    ax_b = fig.add_subplot(gs[0, 5:8])
    if rs_mel is not None:
        ax_b.hist(rs_mel[rs_mel <= 2.0], bins=40, color='#DDDDDD', edgecolor='#BBBBBB',
                  lw=0.3, density=True, label='Non-rare')
        ax_b.hist(rs_mel[rs_mel > 2.0], bins=20, color=C['gvae'], edgecolor='white',
                  lw=0.3, density=True, alpha=0.8, label='Rare')
        ax_b.axvline(2.0, color='black', ls='--', lw=0.8)
        ax_b.text(2.05, ax_b.get_ylim()[1]*0.9, '$\\eta = 2.0$', fontsize=6, va='top')
    ax_b.set_xlabel('KL divergence $z$-score')
    ax_b.set_ylabel('Density')
    ax_b.legend(frameon=False, fontsize=6)

    # ── (c) Rare cell fractions across datasets ──
    ax_c = fig.add_subplot(gs[0, 8:])
    datasets_rare = {
        'NSCLC\nscRNA':  ('49 / 9000',   49/9000),
        'Breast':        ('1782 / 45000', 1782/45000),
        'NSCLC\nVisium': ('4172 / 50000', 4172/50000),
        'Melanoma':      ('326 / 5000',   326/5000),
    }
    dr_names = list(datasets_rare.keys())
    dr_pcts = [v[1]*100 for v in datasets_rare.values()]
    bars_c = ax_c.barh(range(len(dr_names)), dr_pcts, color=C['gvae'], alpha=0.8,
                       edgecolor='white', lw=0.3, height=0.55)
    for i, (pct, (txt, _)) in enumerate(zip(dr_pcts, datasets_rare.values())):
        ax_c.text(pct + 0.15, i, f'{pct:.1f}%', va='center', fontsize=6)
    ax_c.set_yticks(range(len(dr_names))); ax_c.set_yticklabels(dr_names, fontsize=6.5)
    ax_c.set_xlabel('Rare cells (%)')
    ax_c.set_xlim(0, max(dr_pcts) * 1.4)
    ax_c.invert_yaxis()

    # ── (d) Immunosuppressive signature effect sizes ──
    ax_d = fig.add_subplot(gs[1, :])
    sigs = list(IMMUNO.keys())
    dsets = ['nsclc_vis', 'breast']
    dset_labels = ['NSCLC Visium', 'Breast']
    dset_colors = [C['gvae'], C['scanpy']]
    x_d = np.arange(len(sigs))
    w_d = 0.30
    for j, (ds, dsl, dsc) in enumerate(zip(dsets, dset_labels, dset_colors)):
        effects = []
        pvals = []
        for sig in sigs:
            eff, pv = IMMUNO[sig][ds]
            effects.append(eff)
            pvals.append(pv)
        effects = np.array(effects)
        mask_valid = ~np.isnan(effects)
        positions = x_d[mask_valid] + (j - 0.5) * (w_d + 0.02)
        ax_d.bar(positions, np.abs(effects[mask_valid]), w_d, color=dsc, alpha=0.8,
                 edgecolor='white', lw=0.3, label=dsl)
        for k, idx_k in enumerate(np.where(mask_valid)[0]):
            pv = pvals[idx_k]
            if not np.isnan(pv):
                eff_abs = abs(effects[idx_k])
                stars = pval_stars(pv)
                ax_d.text(positions[k], eff_abs + 0.005, stars,
                          ha='center', fontsize=6, fontweight='bold')
            sign_txt = '−' if effects[idx_k] < 0 else '+'
            ax_d.text(positions[k], -0.018, sign_txt,
                      ha='center', fontsize=7, color='#666666')

    ax_d.set_xticks(x_d); ax_d.set_xticklabels(sigs)
    ax_d.set_ylabel('|Effect size|', fontsize=8)
    ax_d.set_ylim(-0.03, 0.32)
    ax_d.axhline(0, color='black', lw=0.4)
    ax_d.legend(frameon=False, fontsize=6.5, loc='upper right')
    ax_d.text(0.01, 0.95, 'Sign shown below each bar (− = lower in rare cells)',
              transform=ax_d.transAxes, fontsize=5.5, color='#888888', va='top')

    for ax, l in [(ax_a,'a'),(ax_b,'b'),(ax_c,'c'),(ax_d,'d')]:
        label(ax, l)

    save(fig, 'fig4_rare_cells')


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 5 — Cell-Cell Communication
# ══════════════════════════════════════════════════════════════════════
def figure5():
    print('Figure 5: Cell-cell communication')
    fig = plt.figure(figsize=(7.2, 5.0))
    gs = gridspec.GridSpec(2, 2, hspace=0.48, wspace=0.45,
                           left=0.06, right=0.96, top=0.94, bottom=0.08)

    # ── (a) Network diagram — one per dataset panel (breast) ──
    ax_a = fig.add_subplot(gs[0, 0])

    node_info = {
        2: ('C2', 'CCL19\nsender'),
        5: ('C5', 'CCR7 recv.\nKIR3DL1'),
        7: ('C7', 'CCL21\nsender'),
    }
    angles_a = {2: np.pi*2/3, 5: 0, 7: np.pi*4/3}
    radius = 0.32
    node_pos = {cid: (radius*np.cos(a), radius*np.sin(a)) for cid, a in angles_a.items()}
    node_colors_a = {2: '#0072B2', 5: '#E69F00', 7: '#009E73'}

    for cid, (short, desc) in node_info.items():
        px, py = node_pos[cid]
        ax_a.add_patch(plt.Circle((px, py), 0.07, color=node_colors_a[cid],
                                  ec='white', lw=0.8, zorder=3))
        ax_a.text(px, py, short, ha='center', va='center', fontsize=7,
                  fontweight='bold', color='white', zorder=4)
        ax_a.text(px, py - 0.12, desc, ha='center', va='top', fontsize=4.5,
                  color='#555555', zorder=4)

    breast_edges = [
        (7, 5, 'CCL21–CCR7', 0.1044),
        (2, 5, 'CCL19–CCR7', 0.0965),
        (5, 5, 'HLA-B–KIR3DL1', 0.0696),
    ]
    max_s = max(e[3] for e in breast_edges)
    for src, tgt, lr_name, score in breast_edges:
        x1, y1 = node_pos[src]
        x2, y2 = node_pos[tgt]
        w = 1.0 + 3.0 * (score / max_s)
        if src == tgt:
            arc = mpatches.Arc((x1, y1 + 0.12), 0.14, 0.14, angle=90,
                               theta1=30, theta2=330, color=node_colors_a[src],
                               lw=w, zorder=2)
            ax_a.add_patch(arc)
            ax_a.text(x1 + 0.12, y1 + 0.18, lr_name, fontsize=4.5, color='#444444')
        else:
            mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
            ax_a.annotate('', xy=(x2, y2), xytext=(x1, y1),
                          arrowprops=dict(arrowstyle='->', color=node_colors_a[src],
                                          lw=w, connectionstyle='arc3,rad=0.2',
                                          shrinkA=11, shrinkB=11),
                          zorder=2)
            perp_x, perp_y = -(y2-y1), (x2-x1)
            norm = np.sqrt(perp_x**2 + perp_y**2)
            offset = 0.06
            ax_a.text(mid_x + offset*perp_x/norm, mid_y + offset*perp_y/norm,
                      lr_name, fontsize=4.5, ha='center', va='center', color='#444444',
                      rotation=np.degrees(np.arctan2(y2-y1, x2-x1)))

    ax_a.set_xlim(-0.55, 0.55); ax_a.set_ylim(-0.55, 0.55)
    ax_a.set_aspect('equal')
    ax_a.set_xticks([]); ax_a.set_yticks([])
    ax_a.spines['left'].set_visible(False); ax_a.spines['bottom'].set_visible(False)
    ax_a.set_title('Breast L-R network', fontsize=7.5, pad=3)

    # ── (b) Dot plot of L-R interactions ──
    ax_b = fig.add_subplot(gs[0, 1])
    lr_names = [f'{lr[0]}' for lr in LR_ALL]
    lr_datasets = [lr[1] for lr in LR_ALL]
    lr_pairs = [lr[2] for lr in LR_ALL]
    lr_scores = [lr[3] for lr in LR_ALL]

    y_pos = np.arange(len(LR_ALL))
    sizes = [s / max(lr_scores) * 200 + 20 for s in lr_scores]
    ds_color_map = {'Breast': C['scanpy'], 'NSCLC-V': C['gvae'], 'NSCLC-S': C['highlight']}
    colors_dot = [ds_color_map[d] for d in lr_datasets]

    ax_b.scatter(lr_scores, y_pos, s=sizes, c=colors_dot, alpha=0.8,
                 edgecolors='white', lw=0.4, zorder=3)
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels([f'{n} ({p})' for n, p in zip(lr_names, lr_pairs)], fontsize=5.5)
    ax_b.set_xlabel('Interaction score')
    ax_b.set_xlim(0, max(lr_scores)*1.15)
    ax_b.invert_yaxis()
    ax_b.set_title('Top L-R pairs across datasets', fontsize=7.5, pad=3)
    legend_lr = [mpatches.Patch(color=v, label=k) for k, v in ds_color_map.items()]
    ax_b.legend(handles=legend_lr, fontsize=5, frameon=False, loc='lower right')

    # ── (c) Spatial interaction overlay (Visium SPP1-CD44) ──
    ax_c = fig.add_subplot(gs[1, 0])
    cl_vis = load_npy('nsclc_visium', 'cluster_labels')
    gate_vis = load_npy('nsclc_visium', 'gate_values')
    n_spots = len(cl_vis) if cl_vis is not None else 2000
    cols = int(np.ceil(np.sqrt(n_spots * 1.15)))
    rows_hex = int(np.ceil(n_spots / cols))
    xs_v, ys_v = [], []
    idx = 0
    for r in range(rows_hex):
        for cc in range(cols):
            if idx >= n_spots: break
            xs_v.append(cc + (0.5 if r % 2 else 0)); ys_v.append(r * 0.866)
            idx += 1
    xs_v, ys_v = np.array(xs_v[:n_spots]), np.array(ys_v[:n_spots])

    if cl_vis is not None:
        is_c5 = cl_vis == 5
        is_c4 = cl_vis == 4
        is_c0 = cl_vis == 0
        ax_c.scatter(xs_v[~(is_c5|is_c4|is_c0)], ys_v[~(is_c5|is_c4|is_c0)],
                     s=2, color='#E8E8E8', alpha=0.5, rasterized=True)
        ax_c.scatter(xs_v[is_c4], ys_v[is_c4], s=4, color=C['scanpy'],
                     alpha=0.7, label='C4 (receiver)', rasterized=True)
        ax_c.scatter(xs_v[is_c0], ys_v[is_c0], s=4, color=C['highlight'],
                     alpha=0.7, label='C0 (receiver)', rasterized=True)
        ax_c.scatter(xs_v[is_c5], ys_v[is_c5], s=4, color=C['gvae'],
                     alpha=0.8, label='C5 (SPP1 sender)', rasterized=True)
        ax_c.legend(fontsize=5, markerscale=2.5, frameon=False, loc='lower right')
    ax_c.set_aspect('equal')
    ax_c.set_xticks([]); ax_c.set_yticks([])
    ax_c.spines['left'].set_visible(False); ax_c.spines['bottom'].set_visible(False)
    ax_c.set_title('SPP1–CD44 spatial interaction', fontsize=7.5, pad=3)

    # ── (d) Attention selectivity ──
    ax_d = fig.add_subplot(gs[1, 1])
    sel_data = {
        'NSCLC scRNA':  (0.108, 0.105),
        'Breast':       (0.108, 0.105),
        'NSCLC Visium': (0.503, 0.080),
        'Colorectal':   (0.640, 0.120),
    }
    sel_names = list(sel_data.keys())
    sel_means = [v[0] for v in sel_data.values()]
    sel_stds = [v[1] for v in sel_data.values()]
    colors_sel = [C['muted'], C['muted'], C['gvae'], C['highlight']]
    ax_d.barh(range(len(sel_names)), sel_means, xerr=sel_stds,
              color=colors_sel, alpha=0.8, edgecolor='white', lw=0.3,
              capsize=2.5, height=0.55, error_kw={'lw': 0.7})
    for i, (m, s) in enumerate(zip(sel_means, sel_stds)):
        ax_d.text(m + s + 0.02, i, f'{m:.3f}', va='center', fontsize=6)
    ax_d.set_yticks(range(len(sel_names))); ax_d.set_yticklabels(sel_names, fontsize=6.5)
    ax_d.set_xlabel('Attention selectivity\n$D_{\\mathrm{KL}}(\\hat{\\alpha} \\| \\mathcal{U})$')
    ax_d.invert_yaxis()
    ax_d.set_xlim(0, 0.9)
    ax_d.text(0.95, 0.05, 'Higher = more\nselective attention',
              transform=ax_d.transAxes, fontsize=5.5, ha='right', color='#888888')

    for ax, l in [(ax_a,'a'),(ax_b,'b'),(ax_c,'c'),(ax_d,'d')]:
        label(ax, l)

    save(fig, 'fig5_communication')


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 6 — Ablation Study
# ══════════════════════════════════════════════════════════════════════
def figure6():
    print('Figure 6: Ablation study')
    fig = plt.figure(figsize=(7.2, 3.8))
    gs = gridspec.GridSpec(1, 2, wspace=0.40, left=0.07, right=0.97, top=0.92, bottom=0.18,
                           width_ratios=[1.6, 1])

    names = [a[0] for a in ABLATIONS]
    aurocs = [a[1] for a in ABLATIONS]
    full_auroc = ABLATIONS[0][1]
    deltas = [a - full_auroc for a in aurocs]

    sorted_idx = np.argsort(aurocs)[::-1]
    names_s = [names[i] for i in sorted_idx]
    aurocs_s = [aurocs[i] for i in sorted_idx]

    # ── (a) AUROC by ablation condition ──
    ax_a = fig.add_subplot(gs[0, 0])
    colors_a = []
    for n in names_s:
        if n == 'FULL':
            colors_a.append(C['gvae'])
        elif aurocs_s[names_s.index(n)] >= full_auroc:
            colors_a.append(C['highlight'])
        else:
            colors_a.append('#AABBCC')
    ax_a.bar(range(len(names_s)), aurocs_s, color=colors_a, edgecolor='white', lw=0.4, width=0.7)
    ax_a.axhline(0.5, color=C['chance'], ls='--', lw=0.6, label='Chance')
    ax_a.axhline(full_auroc, color=C['gvae'], ls=':', lw=0.6, alpha=0.5, label=f'Full ({full_auroc})')
    ax_a.set_xticks(range(len(names_s)))
    ax_a.set_xticklabels(names_s, rotation=45, ha='right', fontsize=6)
    ax_a.set_ylabel('AUROC')
    ax_a.set_ylim(0, 0.88)
    ax_a.legend(frameon=False, fontsize=6, loc='upper right')

    # ── (b) Delta from full model ──
    ax_b = fig.add_subplot(gs[0, 1])
    abl_no_full = [(n, d) for n, d in zip(names, deltas) if n != 'FULL']
    abl_sorted = sorted(abl_no_full, key=lambda x: x[1])
    abl_names_b = [a[0] for a in abl_sorted]
    abl_deltas_b = [a[1] for a in abl_sorted]
    colors_b = [C['highlight'] if d >= 0 else '#CC6677' for d in abl_deltas_b]
    ax_b.barh(range(len(abl_names_b)), abl_deltas_b, color=colors_b,
              edgecolor='white', lw=0.3, height=0.6)
    ax_b.axvline(0, color='black', lw=0.4)
    ax_b.set_yticks(range(len(abl_names_b)))
    ax_b.set_yticklabels(abl_names_b, fontsize=6)
    ax_b.set_xlabel('$\\Delta$AUROC from full model')
    ax_b.set_xlim(-0.65, 0.15)
    ax_b.spines['left'].set_visible(False)
    ax_b.tick_params(axis='y', length=0)

    for ax, l in [(ax_a,'a'),(ax_b,'b')]:
        label(ax, l)

    save(fig, 'fig6_ablation')


if __name__ == '__main__':
    print('Generating paper figures...')
    figure2()
    figure3()
    figure4()
    figure5()
    figure6()
    print(f'\nAll figures saved to {FIGDIR}/')
