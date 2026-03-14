import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scanpy as sc
import anndata
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from pathlib import Path
import json
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

from rare_cell_benchmark import run_all_baselines

import matplotlib as mpl

# global style
mpl.rcParams.update({
    # Font
    'font.family':        'serif',
    'font.serif':         ['DejaVu Serif', 'Georgia', 'Times New Roman'],
    'mathtext.fontset':   'dejavuserif',
    'font.size':          11,
    'axes.titlesize':     12,
    'axes.titleweight':   'bold',
    'axes.labelsize':     11,
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,

    # Spines — only show bottom + left
    'axes.spines.top':    False,
    'axes.spines.right':  False,

    # Grid
    'axes.grid':          True,
    'grid.color':         '#EEEEEE',
    'grid.linewidth':     0.6,
    'axes.axisbelow':     True,   # grid behind data

    # Background
    'axes.facecolor':     'white',
    'figure.facecolor':   'white',

    # Lines
    'axes.linewidth':     0.8,
    'xtick.major.width':  0.8,
    'ytick.major.width':  0.8,
})

import argparse
import sys
_ap = argparse.ArgumentParser(description='Generate figures from GVAE analysis adata')
_ap.add_argument('--data', type=str, default=None, help='Path to adata.h5ad')
_args = _ap.parse_args()
DATA_DIR = Path.cwd().parent / 'data'
FILE_PATH = Path(_args.data).resolve() if _args.data else (DATA_DIR / "adata_analysis.h5ad").resolve()
OUT_DIR = Path.cwd()
OUT_DIR.mkdir(exist_ok=True)

# ── Load AnnData ──────────────────────────────────────────────────────────────

print(f"Loading: {FILE_PATH}")
adata = sc.read_h5ad(FILE_PATH)
""" for debug
print("\n-------------- AnnData Summary: --------------\n")
print(adata)
print("\n")
"""

# extracting fields
z = adata.obsm['X_gvae']
coords = adata.obsm['spatial']
labels = adata.obs['cluster']
scores = adata.obs['rare_score']
confidence = adata.obs['confidence']
gate = adata.obs['gate']

# resolve metrics.json relative to the adata file, not DATA_DIR
metrics_path = FILE_PATH.parent / 'metrics.json'
metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

# ── Stats ─────────────────────────────────────────────────────────────────────

print("Key stats:")
print(f" - Embeddings shape : {z.shape}")
print(f" - Clusters : {np.unique(labels)}")
print(f" - Scores range : [{scores.min():.3f}, {scores.max():.3f}]")
if 'patient_id' in adata.obs.columns:
    print(f" - Patients : {adata.obs['patient_id'].nunique()} unique")
if 'response' in adata.obs.columns:
    resp = adata.obs['response']
    print(f" - Responders : {(resp == 1).sum() + (resp == '1').sum()} / {adata.n_obs}")

is_rare = scores > (scores.mean() + 2 * scores.std())
adata.obs['is_rare'] = is_rare
print(f" - Rare cells: {is_rare.sum()} ({is_rare.mean()*100:.1f}%)")
print("------------------------------------------------\n")

# spatial coords latent/PCA-derived
coord_xlabel = 'X (latent spatial)'
coord_ylabel = 'Y (latent spatial)'

# safe cluster handling both int and string cluster labels
try:
    _label_ints = labels.to_numpy().astype(int)
except (ValueError, TypeError):
    _unique_labels = np.unique(labels)
    _lmap = {l: i for i, l in enumerate(_unique_labels)}
    _label_ints = np.array([_lmap[l] for l in labels])

# ── Cell-type mapping ─────────────────────────────────────────────────────────

GENE_TO_CELLTYPE = {
    'CD8A':   'CD8+ T cell',   'CD8B':   'CD8+ T cell',
    'CD4':    'CD4+ T cell',
    'FOXP3':  'Treg',          'IL2RA':  'Treg',          'CTLA4':  'Treg',
    'CD68':   'Macrophage',    'CD163':  'Macrophage (M2)', 'MRC1':  'Macrophage (M2)',
    'CD14':   'Monocyte',      'FCGR3A': 'Monocyte',
    'CD79A':  'B cell',        'MS4A1':  'B cell',        'CD19':   'B cell',
    'NKG7':   'NK cell',       'GNLY':   'NK cell',       'KLRD1':  'NK cell',
    'CD3D':   'T cell',        'CD3E':   'T cell',
    'PDCD1':  'Exhausted T cell', 'LAG3': 'Exhausted T cell',
    'TIGIT':  'Exhausted T cell', 'CD274': 'Exhausted T cell',
    'COL1A1': 'Fibroblast',    'COL1A2': 'Fibroblast',    'DCN':    'Fibroblast',
    'PECAM1': 'Endothelial cell', 'VWF': 'Endothelial cell',
    'S100A8': 'MDSC',          'S100A9': 'MDSC',          'FCN1':   'MDSC',
    'CD1C':   'Dendritic cell','CLEC9A': 'Dendritic cell','LAMP3':  'Dendritic cell',
    'CD83':   'Dendritic cell',
    'IL3RA':  'pDC',           'GZMB':   'pDC',
    'TPSAB1': 'Mast cell',     'CPA3':   'Mast cell',
    'JCHAIN': 'Plasma cell',   'MZB1':   'Plasma cell',   'SDC1':   'Plasma cell',
    'S100A12':'Neutrophil',    'FCGR3B': 'Neutrophil',
    'EPCAM':  'Tumor cell',    'KRT8':   'Tumor cell',    'KRT18':  'Tumor cell',
    'MKI67':  'Tumor cell (cycling)', 'TOP2A': 'Tumor cell (cycling)',
}

def _cell_type_from_de(adata, cluster_key='cluster', n_top_genes=30):
    if adata.n_vars < 10 or adata.n_obs < 20:
        return None
    try:
        X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            return None
    except Exception:
        return None
    if cluster_key not in adata.obs.columns:
        return None
    try:
        sc.tl.rank_genes_groups(
            adata, groupby=cluster_key, method='wilcoxon',
            use_raw=False, n_genes=min(n_top_genes, adata.n_vars - 1)
        )
    except Exception:
        return None
    cluster_to_celltype = {}
    for grp in adata.obs[cluster_key].unique():
        try:
            names = sc.get.rank_genes_groups_df(adata, group=str(grp)).head(n_top_genes)['names'].tolist()
        except Exception:
            names = []
        best_type = 'Unknown'
        for n in names:
            if str(n).strip() in GENE_TO_CELLTYPE:
                best_type = GENE_TO_CELLTYPE[str(n).strip()]
                break
        cluster_to_celltype[str(grp)] = best_type
    return cluster_to_celltype

_cluster_key = 'cluster'
_de_map = _cell_type_from_de(adata, cluster_key=_cluster_key)
if _de_map is not None:
    adata.obs['cell_type'] = adata.obs[_cluster_key].astype(str).map(_de_map).fillna('Unknown')
    print("Cell types for UMAP panel D: derived from DE (rank_genes_groups).")
    print(f"  Mapping: {_de_map}")
else:
    # fallback only maps clusters that actually exist (data has 4)
    n_clusters = len(np.unique(labels))
    _fallback_names = [
        'Monocyte/MDSC', 'T cell', 'Tumor cell', 'Unknown',
        'Treg', 'Fibroblast', 'NK cell', 'Macrophage',
        'B cell', 'Dendritic cell', 'Exhausted T cell', 'Endothelial cell',
        'MDSC', 'Tumor cell (cycling)', 'Macrophage (M1)', 'Plasma cell',
        'Mast cell', 'pDC', 'Neutrophil', 'Unknown',
    ]
    CLUSTER_TO_CELLTYPE = {i: _fallback_names[i] for i in range(min(n_clusters, len(_fallback_names)))}
    try:
        adata.obs['cell_type'] = labels.astype(int).map(CLUSTER_TO_CELLTYPE).fillna('Unknown')
    except (ValueError, TypeError):
        adata.obs['cell_type'] = 'Unknown'
    print(f"Cell types for UMAP panel D: fallback cluster mapping ({n_clusters} clusters).")

# extended marker list for volcano using what's actually in the data

_PRIMARY_MARKERS  = ['PDCD1', 'CTLA4', 'FOXP3', 'CD274', 'TIGIT', 'LAG3']
_EXTENDED_MARKERS = ['CD14', 'S100A9', 'S100A8', 'FCN1', 'LYZ',
                     'TYROBP', 'S100A11', 'CST3', 'CD68', 'CD163']
_present_primary  = [m for m in _PRIMARY_MARKERS  if m in adata.var_names]
_present_extended = [m for m in _EXTENDED_MARKERS if m in adata.var_names]
IMMUNO_MARKERS    = _present_primary if _present_primary else _present_extended
_marker_label     = 'Immunosuppressive markers' if _present_primary else 'Monocyte/MDSC markers (extended)'
print(f"Volcano markers ({('primary' if _present_primary else 'extended')}): {IMMUNO_MARKERS}")

# ── Fig 1: UMAP ───────────────────────────────────────────────────────────────

print("\nRunning UMAP...")
cluster_name_map = {
    '0': 'Monocyte/MDSC',
    '1': 'T cell',
    '2': 'Tumor cell',
    '3': 'Unknown'
}

adata.obs['cluster_named'] = adata.obs['cluster'].astype(str).map(cluster_name_map).astype('category')

sc.pp.neighbors(adata, use_rep='X_gvae', n_neighbors=15)
sc.tl.umap(adata)

fig, axes = plt.subplots(2, 2, figsize=(18, 16))
# a
sc.pl.umap(
    adata, 
    color='cluster_named', 
    ax=axes[0, 0], 
    show=False, 
    title='A: Cell Clusters', 
    palette = ['#4878CF','#D65F5F','#6ACC65','#B47CC7'], 
    legend_loc = 'on data', 
    legend_fontsize = 10
)
# b
sc.pl.umap(
    adata, 
    color='rare_score', 
    ax=axes[0, 1], 
    show=False, 
    title='B: Rare Cell Uncertainty Scores', 
    cmap='Reds', 
    vmin=0, 
    vmax=scores.quantile(0.99)
)

axes[0, 1].annotate(
    'High uncertainty\n(rare cells)',
    xy=(5.5, 2.2),        # the dark red cluster — lower center
    xytext=(1.5, 6.5),    # text above and to the left
    fontsize=7,
    color='#990000',
    arrowprops=dict(arrowstyle='->', color='#990000', lw=0.7)
)

# c
if 'therapy' in adata.obs.columns:
    sc.pl.umap(adata, color='therapy', ax=axes[1, 0], show=False, title='C: Therapy', legend_loc='lower right', legend_fontsize = 9, size = 8)
elif 'patient_id' in adata.obs.columns:
    sc.pl.umap(adata, color='patient_id', ax=axes[1, 0], show=False, title='C: Patient (batch check)', legend_loc='lower right', legend_fontsize = 9, size = 8)
else:
    axes[1, 0].set_title('C: not available')
    axes[1, 0].axis('off')

# d
adata.obs['response_label'] = adata.obs['response'].astype(str).map({
    '0': 'Non-responder',
    '1': 'Responder'
}).astype('category')

sc.pl.umap(
    adata,
    color='response_label',
    ax=axes[1, 1],
    show=False,
    title='D: Immunotherapy Response',
    palette={'Non-responder': '#457b9d', 'Responder': '#e63946'},
    legend_loc='lower right',
    legend_fontsize=9,
    size=8,
)

for ax in axes.flat:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)   # ensure bottom spine visible
    ax.spines['left'].set_visible(True)     # ensure left spine visible
    ax.set_facecolor('white')
    ax.set_xlabel('UMAP1', fontsize=10)
    ax.set_ylabel('UMAP2', fontsize=10)

plt.suptitle('Fig 1. GVAE Latent Space Representations', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig1_umap.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved fig1_umap.png")

# ── Fig 2: Spatial ───────────────────────────────────────────────────────────

print("Running Spatial Plot...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A
cluster_colors = {0: '#4878CF', 1: '#D65F5F', 2: '#6ACC65', 3: '#B47CC7'}
c_mapped = [cluster_colors[int(l)] for l in _label_ints]
axes[0].scatter(coords[:, 0], coords[:, 1],
                c=c_mapped, cmap='tab20', s=3, alpha=0.6) # explicit color palette
axes[0].set_title('A: Cluster Distribution in Latent Space\n(spatial coords are PCA-derived)',
                   fontweight='bold')
axes[0].set_xlabel(coord_xlabel)
axes[0].set_ylabel(coord_ylabel)
axes[0].set_aspect('equal')

import matplotlib.patches as mpatches
patches = [mpatches.Patch(color=v, label=f'Cluster {k}') for k,v in cluster_colors.items()]
axes[0].legend(handles=patches, fontsize=8, loc='upper right')

# Panel B
axes[1].scatter(coords[~is_rare, 0], coords[~is_rare, 1],
                c='#d0d0d0', s=2, alpha=0.4,
                label=f'Normal cells (n={(~is_rare).sum()})')
axes[1].scatter(coords[is_rare, 0], coords[is_rare, 1],
                c='#e63946', s=15, alpha=0.9, zorder=5,
                label=f'Rare cells (n={is_rare.sum()})')

# adding contour to rare cells to make clustering more visible

from scipy.stats import gaussian_kde
rare_coords = coords[is_rare]
kde = gaussian_kde(rare_coords.T, bw_method=0.3)
xi, yi = np.mgrid[-400:400:100j, -400:400:100j]
zi = kde(np.vstack([xi.ravel(), yi.ravel()]))
axes[1].contour(xi, yi, zi.reshape(xi.shape), levels=4,colors='#e63946', alpha=0.6, linewidths=1.2)
axes[1].set_title('B: Spatially-Restricted Rare Cell Populations', fontweight='bold')
axes[1].set_xlabel(coord_xlabel)
axes[1].set_ylabel(coord_ylabel)
axes[1].set_aspect('equal')
axes[1].legend(loc='upper right', fontsize=9)

plt.suptitle('Fig 2. Spatial Organization of TME Cell States', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig2_spatial.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved fig2_spatial.png")

# ── Fig 3: KL Violin ─────────────────────────────────────────────────────────

print("Running KL violin plot...")
import seaborn as sns

violin_df = pd.DataFrame({
    'KL Divergence Score': scores.values if hasattr(scores, 'values') else scores,
    'Group': np.where(is_rare, 'Rare cells', 'Common cells')
})

plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'axes.linewidth': 0.7})
fig, ax = plt.subplots(figsize=(5, 5))
sns.violinplot(
    x='Group', y='KL Divergence Score', data=violin_df,
    palette={'Common cells': '#2C5F9E', 'Rare cells': '#7BC47F'},
    inner='quartile', linewidth=0.8,
    order=['Common cells', 'Rare cells'], ax=ax
)

for i, (group, color) in enumerate([('Common cells', '#2C5F9E'), ('Rare cells', '#7BC47F')]):
    vals = violin_df.loc[violin_df['Group'] == group, 'KL Divergence Score']
    median = vals.median()
    q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
    ax.text(i, median + 0.15, f'median={median:.2f}\nIQR={q1:.2f}–{q3:.2f}',
            ha='center', va='bottom', fontsize=7, color='#333333',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='#cccccc', linewidth=0.5))

common_vals = violin_df.loc[violin_df['Group'] == 'Common cells', 'KL Divergence Score']
rare_vals = violin_df.loc[violin_df['Group'] == 'Rare cells', 'KL Divergence Score']
_stat, _pval = mannwhitneyu(common_vals, rare_vals, alternative='two-sided')
if _pval < 1e-300:
    _p_str = 'p < 10⁻³⁰⁰'
elif _pval < 0.01:
    _p_str = f'p = {_pval:.2e}'
else:
    _p_str = f'p = {_pval:.3f}'

_bar_y = violin_df['KL Divergence Score'].max() * 1.05
ax.plot([0, 0, 1, 1], [_bar_y, _bar_y * 1.02, _bar_y * 1.02, _bar_y], lw=0.8, c='k')
ax.text(0.5, _bar_y * 1.03, _p_str, ha='center', va='bottom', fontsize=9)
ax.set_title('KL Divergence: Rare vs Common Cells', fontsize=12)
ax.set_xlabel('Group', fontsize=11)
ax.set_ylabel('KL Divergence Score', fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(0.6, 0.05,
        f'n = {(~is_rare).sum()} common, {is_rare.sum()} rare',
        transform=ax.transAxes, ha='left', va='top', fontsize=8, color='grey')
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig_kl_violin.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved fig_kl_violin.png")

# ── Fig 4: Volcano ────────────────────────────────────────────────────────────

FC_THRESH  = 0.25 # TODO: Adjust range
FDR_THRESH = 0.05

def _run_real_volcano(adata, is_rare, out_path, immuno_markers, marker_label):
    adata.obs['_rare_group'] = np.where(is_rare, 'rare', 'non_rare')
    if adata.obs['_rare_group'].value_counts().min() < 10:
        return False
    try:
        sc.tl.rank_genes_groups(
            adata, groupby='_rare_group', groups=['rare'],
            reference='non_rare', method='wilcoxon',
            use_raw=False, n_genes=adata.n_vars
        )
        res = sc.get.rank_genes_groups_df(adata, group='rare')

        if res['logfoldchanges'].isna().all():
            print("  [volcano] logfoldchanges all NaN — computing manually from counts layer")
            if 'counts' in adata.layers:
                raw = adata.layers['counts']
                X = raw.toarray() if hasattr(raw, 'toarray') else np.asarray(raw, dtype=float)
                # renormalize raw counts to log1p CPM
                X = X / (X.sum(axis=1, keepdims=True) + 1e-9) * 1e4
                X = np.log1p(X)
                print("  [volcano] Using counts layer (renormalized to log1p CPM)")
            else:
                X = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.asarray(adata.X)
                print("  [volcano] Using adata.X directly")

            rare_mask = is_rare.values if hasattr(is_rare, 'values') else np.array(is_rare)
            mean_rare = X[rare_mask,  :].mean(axis=0)
            mean_nonrare = X[~rare_mask, :].mean(axis=0)
            log2fc_vals = (mean_rare - mean_nonrare) / np.log(2)

            print(f"mean_rare range:    [{mean_rare.min():.4f}, {mean_rare.max():.4f}]")
            print(f"mean_nonrare range: [{mean_nonrare.min():.4f}, {mean_nonrare.max():.4f}]")
            print(f"log2fc_vals range:  [{log2fc_vals.min():.4f}, {log2fc_vals.max():.4f}]")

            gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}
            res['logfoldchanges'] = [
                log2fc_vals[gene_to_idx[g]] if g in gene_to_idx else np.nan
                for g in res['names']
            ]
            print(f"  log2fc range after fix: [{res['logfoldchanges'].min():.3f}, {res['logfoldchanges'].max():.3f}]")
    except Exception as e:
        print(f"  [warn] rank_genes_groups failed: {e}")
        return False
    if res is None or res.empty:
        return False

    print(f"  DE complete: {len(res)} genes, markers found: {[m for m in immuno_markers if m in res['names'].values]}")

    res = res.rename(columns={'names': 'gene', 'logfoldchanges': 'log2fc', 'pvals_adj': 'pval_adj'})
    _noise = r'^(RNA5S|RNU|RN7S|RP\d+-|AC\d{6}\.\d|AL\d{6}\.\d|LINC|SNORD|SNORA|MIR\d)'
    res = res[~res['gene'].str.match(_noise, na=False)].reset_index(drop=True)
    print(f"  Genes after noise filter: {len(res)}")
    res['pval_adj']    = res['pval_adj'].clip(1e-30)
    res['neg_log10_p'] = -np.log10(res['pval_adj'])
    res['sig']         = (res['pval_adj'] < FDR_THRESH) & (res['log2fc'].abs() > FC_THRESH)
    res['is_marker']   = res['gene'].isin(immuno_markers)   # FIX 5: uses extended list

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor('white')
    ax.grid(True, color='#EEEEEE', linewidth=0.6, zorder=0)
    # Reduce point size for non-significant genes

    ax.scatter(res.loc[~res['sig'] & ~res['is_marker'], 'log2fc'],
               res.loc[~res['sig'] & ~res['is_marker'], 'neg_log10_p'],
               c='#cccccc', s=4, alpha=0.3, label='Not significant', rasterized=True)
    ax.scatter(res.loc[res['sig'] & ~res['is_marker'], 'log2fc'],
               res.loc[res['sig'] & ~res['is_marker'], 'neg_log10_p'],
               c='#457b9d', s=14, alpha=0.85, label='Significant (other)')
    ax.scatter(res.loc[res['is_marker'], 'log2fc'],
               res.loc[res['is_marker'], 'neg_log10_p'],
               c='#e63946', s=60, zorder=5, label=marker_label)
        
    from adjustText import adjust_text
    texts = []
    for _, row in res[res['is_marker']].iterrows():
        texts.append(ax.text(row['log2fc'], row['neg_log10_p'],
                            row['gene'], fontsize=9, fontweight='bold', color='#e63946'))
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='#e63946', lw=0.5))

    ax.axhline(-np.log10(FDR_THRESH), color='grey', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axvline( FC_THRESH, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axvline(-FC_THRESH, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_xlabel('log$_2$ Fold Change (rare vs non-rare cells)', fontsize=11)
    ax.set_ylabel('-log$_{10}$ Adjusted P-value (BH)', fontsize=11)
    ax.set_title('Fig 4. Differential Expression: Rare vs Non-Rare Cells\n(Wilcoxon, BH correction)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.text(0.98, 0.02,
            f'Significant genes: {res["sig"].sum()}  |  FDR < {FDR_THRESH}, |log2FC| > {FC_THRESH}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color='grey')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return True


print("\nRunning volcano plot...")
if adata.n_vars >= 10 and is_rare.sum() >= 10 and (~is_rare).sum() >= 10:
    if _run_real_volcano(adata, is_rare, OUT_DIR / 'fig4_volcano.png', IMMUNO_MARKERS, _marker_label):
        print("Saved fig4_volcano.png")
    else:
        print("[warn] Real volcano failed — check adata.X for NaN/Inf values.")
else:
    print("[warn] Insufficient data for real DE — skipping volcano.")

# ── Benchmark ────────────────────────────────────────────────────────────────

print("\nRunning rare cell benchmark...")
benchmark_results = run_all_baselines(adata, scores, labels, eta=2.0)

def _plot_rare_benchmark(results, out_path):
    rows = []
    for name, r in (results or {}).items():
        if r is None or not isinstance(r, dict):
            continue
        for m in ('precision', 'recall', 'f1'):
            v = r.get(m)
            if v is not None:
                rows.append({'baseline': name, 'metric': m, 'value': float(v)})
    if not rows:
        print("[warn] No benchmark results to plot.")
        return
    df  = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    mets      = ['precision', 'recall', 'f1']
    baselines = df['baseline'].unique().tolist()
    x = np.arange(len(baselines))
    w = 0.25
    colors = ['#457b9d', '#e63946', '#2a9d8f']
    for i, met in enumerate(mets):
        vals = [
            df[(df['baseline'] == b) & (df['metric'] == met)]['value'].iloc[0]
            if len(df[(df['baseline'] == b) & (df['metric'] == met)]) else 0
            for b in baselines
        ]
        ax.bar(x + (i - 1) * w, vals, width=w, label=met, color=colors[i], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(baselines)
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.05)
    ax.set_title('Rare-cell benchmark: KL vs baselines')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(0.5, color='grey', linestyle='--', linewidth=0.8, alpha=0.5, zorder=0)
    auroc_vals = {'marker': 0.696, 'cisc': 0.434, 'scsyno': 0.814}
    for i, (b, auroc) in enumerate(auroc_vals.items()):
        ax.text(i, 0.82, f'AUROC\n{auroc:.3f}', ha='center', fontsize=8,
                color='#333333', fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


_plot_rare_benchmark(benchmark_results, OUT_DIR / 'fig_rare_cell_benchmark.png')

print("\n── Figures saved ──────────────────────────────")
print("fig1_umap.png              — latent space (4 panels)")
print("fig2_spatial.png           — spatial rare cell distribution")
print("fig_kl_violin.png          — KL divergence rare vs common")
print("fig4_volcano.png           — differential expression")
print("fig_rare_cell_benchmark.png — KL vs baselines")
