import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scanpy as sc
import anndata
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
# use in volcano? Determine if there's a significant diff
# between two independent, non-normally distributed groups
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

# Paths: use --data path/to/adata.h5ad (e.g. ../outputs/melanoma/adata_analysis.h5ad) or default
import argparse
import sys
_ap = argparse.ArgumentParser(description='Generate figures from GVAE analysis adata')
_ap.add_argument('--data', type=str, default=None, help='Path to adata.h5ad (with X_gvae, cluster, rare_score, .X for volcano/markers)')
_args = _ap.parse_args()
DATA_DIR = Path.cwd().parent / 'data'
FILE_PATH = Path(_args.data).resolve() if _args.data else (DATA_DIR / "adata.h5ad").resolve()
OUT_DIR = Path.cwd()
OUT_DIR.mkdir(exist_ok=True)

# Loading real AnnData ------------------------------------------------------------

if not FILE_PATH.exists():
    print(f"Error: data file not found: {FILE_PATH}")
    print("Run the pipeline first to create it, e.g.:")
    print("  (from repo root) python train.py --data melanoma --config local --max-cells 2000")
    print("Then: python plots.py --data ../outputs/melanoma/adata_analysis.h5ad")
    sys.exit(1)
print(f"Loading: {FILE_PATH}")
adata = sc.read_h5ad(FILE_PATH)
print("\n-------------- AnnData Summary: --------------\n")
print(adata)
print("\n")

"""
z = np.load(DATA_DIR / 'embeddings.npy')
labels = np.load(DATA_DIR / 'cluster_labels.npy')
scores = np.load(DATA_DIR / 'rare_cell_scores.npy')
confidence = np.load(DATA_DIR / 'confidence.npy')
gate = np.load(DATA_DIR / 'gate_values.npy')
metrics = json.loads((DATA_DIR / 'metrics.json').read_text())
"""

z = adata.obsm['X_gvae']
coords = adata.obsm['spatial']
labels = adata.obs['cluster']
scores = adata.obs['rare_score']
confidence = adata.obs['confidence']
gate = adata.obs['gate']

metrics_path = DATA_DIR / 'metrics.json'
metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

# stats
print(f"Key stats:")
print(f"  Embeddings shape : {z.shape}")
print(f"  Clusters         : {np.unique(labels)}")
print(f"  Scores range     : [{scores.min():.3f}, {scores.max():.3f}]")
if 'patient_id' in adata.obs.columns:
    print(f"  Patients         : {adata.obs['patient_id'].nunique()} unique")
if 'response' in adata.obs.columns:
    resp = adata.obs['response']
    print(f"  Responders       : {(resp == 1).sum() + (resp == '1').sum()} / {adata.n_obs}")

# finding if cell-type is rare
is_rare = scores > (scores.mean() + 2 * scores.std())
adata.obs['is_rare'] = is_rare
print(f"Rare cells: {is_rare.sum()} ({is_rare.mean()*100:.1f}%)")
print("------------------------------------------------\n")

# Cell-type mapping: DE-derived when gene expression is available, else fallback 
# Canonical markers (gene -> cell type) for TME; one representative gene per type.
GENE_TO_CELLTYPE = {
    'CD8A': 'CD8+ T cell', 'CD8B': 'CD8+ T cell',
    'CD4': 'CD4+ T cell',
    'FOXP3': 'Treg', 'IL2RA': 'Treg', 'CTLA4': 'Treg',
    'CD68': 'Macrophage', 'CD163': 'Macrophage (M2)', 'MRC1': 'Macrophage (M2)',
    'CD14': 'Monocyte', 'FCGR3A': 'Monocyte',
    'CD79A': 'B cell', 'MS4A1': 'B cell',
    'CD19': 'B cell',
    'NKG7': 'NK cell', 'GNLY': 'NK cell', 'KLRD1': 'NK cell',
    'CD3D': 'T cell', 'CD3E': 'T cell',
    'PDCD1': 'Exhausted T cell', 'LAG3': 'Exhausted T cell', 'TIGIT': 'Exhausted T cell',
    'CD274': 'Exhausted T cell',
    'COL1A1': 'Fibroblast', 'COL1A2': 'Fibroblast', 'DCN': 'Fibroblast',
    'PECAM1': 'Endothelial cell', 'VWF': 'Endothelial cell',
    'S100A8': 'MDSC', 'S100A9': 'MDSC', 'FCN1': 'MDSC',
    'CD1C': 'Dendritic cell', 'CLEC9A': 'Dendritic cell', 'LAMP3': 'Dendritic cell',
    'CD83': 'Dendritic cell',
    'IL3RA': 'pDC', 'GZMB': 'pDC',
    'TPSAB1': 'Mast cell', 'CPA3': 'Mast cell',
    'JCHAIN': 'Plasma cell', 'MZB1': 'Plasma cell', 'SDC1': 'Plasma cell',
    'S100A12': 'Neutrophil', 'FCGR3B': 'Neutrophil',
    'EPCAM': 'Tumor cell', 'KRT8': 'Tumor cell', 'KRT18': 'Tumor cell',
    'MKI67': 'Tumor cell (cycling)', 'TOP2A': 'Tumor cell (cycling)',
}


def _cell_type_from_de(adata, cluster_key='cluster', n_top_genes=30):
    """Assign cell_type per cluster from rank_genes_groups; uses top markers + GENE_TO_CELLTYPE."""
    if adata.n_vars < 10 or adata.n_obs < 20:
        return None
    try:
        X = adata.X
        if hasattr(X, 'toarray'):
            X = X.toarray()
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            return None
    except Exception:
        return None
    if cluster_key not in adata.obs.columns:
        return None
    var_names = list(adata.var_names)
    # Build cluster -> list of top marker gene names
    try:
        sc.tl.rank_genes_groups(adata, groupby=cluster_key, method='wilcoxon', use_raw=False, n_genes=min(n_top_genes, adata.n_vars - 1))
    except Exception:
        return None
    cluster_to_celltype = {}
    groups = adata.obs[cluster_key].unique()
    for grp in groups:
        try:
            names = sc.get.rank_genes_groups_df(adata, group=str(grp)).head(n_top_genes)['names'].tolist()
        except Exception:
            names = []
        best_type = 'Unknown'
        for n in names:
            gene = str(n).strip()
            if gene in GENE_TO_CELLTYPE:
                best_type = GENE_TO_CELLTYPE[gene]
                break
        cluster_to_celltype[str(grp)] = best_type
    return cluster_to_celltype


_cluster_key = 'cluster'
_de_map = _cell_type_from_de(adata, cluster_key=_cluster_key)
if _de_map is not None:
    adata.obs['cell_type'] = adata.obs[_cluster_key].astype(str).map(_de_map).fillna('Unknown')
    print("Cell types for UMAP panel D: derived from differential expression (rank_genes_groups).")
else:
    CLUSTER_TO_CELLTYPE = {
        0: 'CD8+ T cell', 1: 'Macrophage (M2)', 2: 'Tumor cell', 3: 'CD4+ T cell',
        4: 'Treg', 5: 'Fibroblast', 6: 'NK cell', 7: 'Monocyte', 8: 'B cell',
        9: 'Dendritic cell', 10: 'Exhausted T cell', 11: 'Endothelial cell', 12: 'MDSC',
        13: 'Tumor cell (cycling)', 14: 'Macrophage (M1)', 15: 'Plasma cell',
        16: 'Mast cell', 17: 'pDC', 18: 'Neutrophil', 19: 'Unknown',
    }
    adata.obs['cell_type'] = adata.obs[_cluster_key].astype(int).map(CLUSTER_TO_CELLTYPE).fillna('Unknown')
    print("Cell types for UMAP panel D: no gene expression in adata — using fallback cluster mapping.")

#1. UMAP ------------------------------------------------------------------------------------

print("Running UMAP:")

sc.pp.neighbors(adata, use_rep='X_gvae', n_neighbors=15)
sc.tl.umap(adata)

fig, axes = plt.subplots(2, 2, figsize=(18, 16))

sc.pl.umap(adata, color='cluster', ax=axes[0, 0], show=False, title='A: Cell Clusters')
sc.pl.umap(adata, color='rare_score', ax=axes[0, 1], show=False, title='B: Rare Cell Uncertainty Scores', cmap='Reds')
if 'patient_id' in adata.obs.columns:
    sc.pl.umap(adata, color='patient_id', ax=axes[1, 0], show=False, title='C: Patient (batch check)')
else:
    axes[1, 0].set_title('C: Patient (batch check) — not available')
    axes[1, 0].axis('off')
sc.pl.umap(adata, color='cell_type', ax=axes[1, 1], show=False, frameon=False, title='D: Cell Type Annotation\n(DE-derived when gene matrix available)')

plt.suptitle('Fig 1. GVAE Latent Space Representations', fontsize=14, fontweight='bold', y=1.01)

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig1_umap.png', dpi=300, bbox_inches = 'tight')
plt.close()
print("Saved fig1_umap.png")

#2. Spatial Plot ----------------------------------------------------------------------------
print("Running Spatial Plot:")
#accessing spatial coordinatesf
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A, coloring all cells by cluster w/ 20 distinct colors
scatter = axes[0].scatter(
    coords[:, 0], coords[:, 1],
    c=labels.to_numpy().astype(int), cmap='tab20',
    s=3, alpha=0.6
)

axes[0].set_title('A: Spatial Cluster Distribution',fontweight='bold')
axes[0].set_xlabel('X coordinate (µm)')
axes[0].set_ylabel('Y coordinate (µm)')
axes[0].set_aspect('equal')

# Panel B, highlighting rare cells
axes[1].scatter(coords[~is_rare, 0], coords[~is_rare, 1], c='#d0d0d0', s=2, alpha=0.4, label=f'Normal cells (n={(~is_rare).sum()})') # normal
axes[1].scatter(coords[is_rare, 0], coords[is_rare, 1], c='#e63946', s=15, alpha=0.9, label=f'Rare cells (n={is_rare.sum()})', zorder=5) # rare
axes[1].set_title('B: Spatially-Restricted Rare Cell Populations', fontweight='bold')
axes[1].set_xlabel('X coordinate (µm)')
axes[1].set_ylabel('Y coordinate (µm)')
axes[1].set_aspect('equal')
axes[1].legend(loc='upper right', fontsize=9)

plt.suptitle('Fig 2. Spatial Organization of TME Cell States', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig2_spatial.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved fig2_spatial.png")

#3. KL Violin Plot (rare vs common) -------------------------------------------
print("Running KL violin plot:")

import seaborn as sns

_violin_df = pd.DataFrame({
    'KL Divergence Score': scores.values if hasattr(scores, 'values') else scores,
    'Group': np.where(is_rare, 'Rare cells', 'Common cells')
})

plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.linewidth': 0.7,
})

fig, ax = plt.subplots(figsize=(5, 5))
palette = {'Common cells': '#2C5F9E', 'Rare cells': '#7BC47F'}
sns.violinplot(
    x='Group', y='KL Divergence Score', data=_violin_df,
    palette=palette, inner='quartile', linewidth=0.8,
    order=['Common cells', 'Rare cells'], ax=ax
)

# Stats annotation
from scipy.stats import mannwhitneyu as _mwu
_common_vals = _violin_df.loc[_violin_df['Group'] == 'Common cells', 'KL Divergence Score']
_rare_vals = _violin_df.loc[_violin_df['Group'] == 'Rare cells', 'KL Divergence Score']
_stat, _pval = _mwu(_common_vals, _rare_vals, alternative='two-sided')
_p_str = f'p = {_pval:.2e}' if _pval < 0.01 else f'p = {_pval:.3f}'

_ymax = _violin_df['KL Divergence Score'].max()
_bar_y = _ymax * 1.05
ax.plot([0, 0, 1, 1], [_bar_y, _bar_y * 1.02, _bar_y * 1.02, _bar_y], lw=0.8, c='k')
ax.text(0.5, _bar_y * 1.03, _p_str, ha='center', va='bottom', fontsize=9)

ax.set_title('KL Divergence: Rare vs Common Cells', fontsize=12)
ax.set_xlabel('Group', fontsize=11)
ax.set_ylabel('KL Divergence Score', fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

n_common = (~is_rare).sum()
n_rare = is_rare.sum()
ax.text(0.02, 0.98, f'n = {n_common} common, {n_rare} rare',
        transform=ax.transAxes, ha='left', va='top', fontsize=8, color='grey')

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig_kl_violin.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved fig_kl_violin.png")

#4. Volcano Plot (real DE when gene matrix available) --------------------------

FC_THRESH = 1.5
FDR_THRESH = 0.05
IMMUNO_MARKERS = ['PDCD1', 'CTLA4', 'FOXP3', 'CD274', 'TIGIT', 'LAG3']

def _run_real_volcano(adata, is_rare, out_path):
    """Build volcano from rank_genes_groups(rare vs non-rare)."""
    adata.obs['_rare_group'] = np.where(is_rare, 'rare', 'non_rare')
    if adata.obs['_rare_group'].value_counts().min() < 10:
        return False
    try:
        sc.tl.rank_genes_groups(adata, groupby='_rare_group', groups=['rare'], reference='non_rare', method='wilcoxon', use_raw=False, n_genes=adata.n_vars)
    except Exception:
        return False
    try:
        res = sc.get.rank_genes_groups_df(adata, group='rare')
    except Exception:
        return False
    if res is None or res.empty:
        return False
    res = res.rename(columns={'names': 'gene', 'logfoldchanges': 'log2fc', 'pvals_adj': 'pval_adj'})
    res['pval_adj'] = res['pval_adj'].clip(1e-30)
    res['neg_log10_p'] = -np.log10(res['pval_adj'])
    res['sig'] = (res['pval_adj'] < FDR_THRESH) & (res['log2fc'].abs() > FC_THRESH)
    res['is_marker'] = res['gene'].isin(IMMUNO_MARKERS)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(res.loc[~res['sig'] & ~res['is_marker'], 'log2fc'], res.loc[~res['sig'] & ~res['is_marker'], 'neg_log10_p'], c='#cccccc', s=8, alpha=0.5, label='Not significant', rasterized=True)
    ax.scatter(res.loc[res['sig'] & ~res['is_marker'], 'log2fc'], res.loc[res['sig'] & ~res['is_marker'], 'neg_log10_p'], c='#457b9d', s=18, alpha=0.8, label='Significant (other)')
    ax.scatter(res.loc[res['is_marker'], 'log2fc'], res.loc[res['is_marker'], 'neg_log10_p'], c='#e63946', s=60, zorder=5, label='Immunosuppressive markers')
    for _, row in res[res['is_marker']].iterrows():
        ax.annotate(row['gene'], xy=(row['log2fc'], row['neg_log10_p']), xytext=(8, 4), textcoords='offset points', fontsize=9, fontweight='bold', color='#e63946')
    ax.axhline(-np.log10(FDR_THRESH), color='grey', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axvline(FC_THRESH, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axvline(-FC_THRESH, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_xlabel('log$_2$ Fold Change (rare vs non-rare cells)', fontsize=11)
    ax.set_ylabel('-log$_{10}$ Adjusted P-value (BH)', fontsize=11)
    ax.set_title('Fig 3. Differential Expression: Rare vs Non-Rare Cells\n(real DE from data)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    n_sig = res['sig'].sum()
    ax.text(0.98, 0.02, f'Significant genes: {n_sig}  |  FDR < {FDR_THRESH}, |log2FC| > {FC_THRESH}', transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color='grey')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return True

if adata.n_vars >= 10 and is_rare.sum() >= 10 and (~is_rare).sum() >= 10:
    print("Running volcano plot (real DE: rare vs non-rare)...")
    if _run_real_volcano(adata, is_rare, OUT_DIR / 'fig3_volcano.png'):
        print("Saved fig3_volcano.png")
    else:
        print("[warn] Real volcano failed, skipping.")
else:
    print("Running volcano plot [SYNTHETIC] (insufficient genes or rare/non-rare cells for DE)")
    np.random.seed(42)
    N_GENES = min(2000, adata.n_vars) if adata.n_vars else 2000
    log2fc_bg = np.random.normal(0, 0.6, N_GENES)
    base_pval = np.random.beta(0.4, 1.0, N_GENES)
    fc_weight = (np.abs(log2fc_bg) / (np.abs(log2fc_bg).max() + 1e-9)) ** 2
    pval_bg = (base_pval * (1 - 0.85 * fc_weight)).clip(1e-5, 1.0)
    gene_names = [f'Gene_{i}' for i in range(N_GENES)]
    df = pd.DataFrame({'gene': gene_names, 'log2fc': log2fc_bg, 'pval': pval_bg})
    _, pval_adj, _, _ = multipletests(df['pval'].values, method='fdr_bh')
    df['pval_adj'] = pval_adj
    df['neg_log10_p'] = -np.log10(df['pval_adj'].clip(1e-30))
    df['sig'] = (df['pval_adj'] < FDR_THRESH) & (df['log2fc'].abs() > FC_THRESH)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(df.loc[~df['sig'], 'log2fc'], df.loc[~df['sig'], 'neg_log10_p'], c='#cccccc', s=8, alpha=0.5, label='Not significant', rasterized=True)
    ax.scatter(df.loc[df['sig'], 'log2fc'], df.loc[df['sig'], 'neg_log10_p'], c='#457b9d', s=18, alpha=0.8, label='Significant')
    ax.axhline(-np.log10(FDR_THRESH), color='grey', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axvline(FC_THRESH, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axvline(-FC_THRESH, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_xlabel('log$_2$ Fold Change (rare vs non-rare cells)', fontsize=11)
    ax.set_ylabel('-log$_{10}$ Adjusted P-value (BH)', fontsize=11)
    ax.set_title('Fig 3. Differential Expression [SYNTHETIC]', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'fig3_volcano_[SYNTHETIC].png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved fig3_volcano_[SYNTHETIC].png")

# -------------------------------------------------------------------------------------------

print("\nAll figures saved to analysis/")
print("fig1_umap.png — latent space structure")
print("fig2_spatial.png — spatial rare cell distribution")
print("fig3_volcano.png — differential expression of rare cluster")

benchmark_results = run_all_baselines(adata, scores, labels, eta=2.0)
print(benchmark_results)

# Rare-cell benchmark visualization
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
        return
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    metrics = ['precision', 'recall', 'f1']
    baselines = df['baseline'].unique().tolist()
    x = np.arange(len(baselines))
    w = 0.25
    for i, met in enumerate(metrics):
        vals = [df[(df['baseline'] == b) & (df['metric'] == met)]['value'].iloc[0] if len(df[(df['baseline'] == b) & (df['metric'] == met)]) else 0 for b in baselines]
        off = (i - 1) * w
        ax.bar(x + off, vals, width=w, label=met)
    ax.set_xticks(x)
    ax.set_xticklabels(baselines)
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.05)
    ax.set_title('Rare-cell benchmark: KL vs baselines')
    ax.legend()
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

_plot_rare_benchmark(benchmark_results, OUT_DIR / 'fig_rare_cell_benchmark.png')
print("Saved fig_rare_cell_benchmark.png")