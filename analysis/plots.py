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

# Paths
DATA_DIR = Path.cwd().parent / 'data'
FILE_PATH = DATA_DIR / "adata.h5ad"
OUT_DIR = Path.cwd()
OUT_DIR.mkdir(exist_ok=True)

# Loading real AnnData ------------------------------------------------------------

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
print(f"  Patients         : {adata.obs['patient_id'].nunique()} unique")
print(f"  Responders       : {(adata.obs['response'] == '1').sum()} / {adata.n_obs}")

# finding if cell-type is rare
is_rare = scores > (scores.mean() + 2 * scores.std())
print(f"Rare cells: {is_rare.sum()} ({is_rare.mean()*100:.1f}%)")
print("------------------------------------------------\n")

# Cell-type mapping  ------------------------------------------------------------------------
# TODO: Replace with mapping derived from differential expression on real data.
CLUSTER_TO_CELLTYPE = {
    0:  'CD8+ T cell',
    1:  'Macrophage (M2)',
    2:  'Tumor cell',
    3:  'CD4+ T cell',
    4:  'Regulatory T cell (Treg)',
    5:  'Fibroblast',
    6:  'NK cell',
    7:  'Monocyte',
    8:  'B cell',
    9:  'Dendritic cell',
    10: 'Exhausted T cell',
    11: 'Endothelial cell',
    12: 'MDSC',
    13: 'Tumor cell (cycling)',
    14: 'Macrophage (M1)',
    15: 'Plasma cell',
    16: 'Mast cell',
    17: 'pDC',
    18: 'Neutrophil',
    19: 'Unknown',
}

adata.obs['cell_type'] = adata.obs['cluster'].astype(int).map(CLUSTER_TO_CELLTYPE).fillna('Unknown')

#1. UMAP ------------------------------------------------------------------------------------

print("Running UMAP:")

sc.pp.neighbors(adata, use_rep='X_gvae', n_neighbors=15)
sc.tl.umap(adata)

fig, axes = plt.subplots(2, 2, figsize=(18, 16))

sc.pl.umap(adata, color='cluster', ax=axes[0, 0], show=False, title='A: Cell Clusters')
sc.pl.umap(adata, color='rare_score', ax=axes[0, 1], show=False, title='B: Rare Cell Uncertainty Scores', cmap='Reds')
sc.pl.umap(adata, color='patient_id', ax=axes[1, 0], show=False, title='C: Patient (batch check)')
sc.pl.umap(adata, color='cell_type', ax=axes[1,1], show=False, frameon=False, title='D: Cell Type Annotation\n(TODO: derive from diff. expression)')

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

#3. Volcano Plot [SYNTHETIC]-----------------------------------------------------------------

# synthetic since immunosuppresive marker genes are hardcoded instead of derived

print("Running volcano plot [SYNTHETIC]: ")
np.random.seed(42)
N_GENES = 2000

# immunosuppresive markers that are significant hits
# generated p-values based off of biological hierarchy

MARKER_GENES = {
    'PDCD1':  {'log2fc':  3.8, 'pval': 1e-7},
    'CTLA4':  {'log2fc':  3.2, 'pval': 1e-6},
    'FOXP3':  {'log2fc':  4.1, 'pval': 1e-8},
    'CD274':  {'log2fc':  3.5, 'pval': 5e-7},
    'TIGIT':  {'log2fc':  2.9, 'pval': 5e-6},
    'LAG3':   {'log2fc':  3.1, 'pval': 2e-6},
}

# generates genes based off of statistical distributions
log2fc_bg = np.random.normal(0, 0.6, N_GENES)
base_pval = np.random.beta(0.4, 1.0, N_GENES)
fc_weight = (np.abs(log2fc_bg) / np.abs(log2fc_bg).max()) ** 2
pval_bg   = (base_pval * (1 - 0.85 * fc_weight)).clip(1e-5, 1.0)

# non-marker significant genes
n_other_sig = 18
sig_idx = np.random.choice(N_GENES, n_other_sig, replace=False)
log2fc_bg[sig_idx] = np.random.choice([-1, 1], n_other_sig) * np.random.uniform(1.8, 2.8, n_other_sig)
pval_bg[sig_idx] = np.random.uniform(1e-8, 5e-4, n_other_sig)

gene_names = [f'Gene_{i}' for i in range(N_GENES)]
df = pd.DataFrame({'gene': gene_names, 'log2fc': log2fc_bg, 'pval': pval_bg})

# Append the marker genes
marker_rows = pd.DataFrame([
    {'gene': g, 'log2fc': v['log2fc'], 'pval': v['pval']}
    for g, v in MARKER_GENES.items()
])
df = pd.concat([df, marker_rows], ignore_index=True)

# benjamin hochberg procedure ranking all p-values
_, pval_adj, _, _ = multipletests(df['pval'].values, method='fdr_bh')
df['pval_adj'] = pval_adj
df['neg_log10_p'] = -np.log10(df['pval_adj'].clip(1e-30))

# signficance
FC_THRESH  = 1.5 # |log2FC| > 1.5
FDR_THRESH = 0.05  # adjusted p < 0.05

df['sig'] = (df['pval_adj'] < FDR_THRESH) & (df['log2fc'].abs() > FC_THRESH)
is_marker = df['gene'].isin(MARKER_GENES.keys())

# grey = NS, steelblue = sig non-marker, red = marker hits
colors = np.where(is_marker, '#e63946', np.where(df['sig'], '#457b9d', '#cccccc'))
fig, ax = plt.subplots(figsize=(10, 7))

ax.scatter(df.loc[~df['sig'] & ~is_marker, 'log2fc'], df.loc[~df['sig'] & ~is_marker, 'neg_log10_p'], c='#cccccc', s=8, alpha=0.5, label='Not significant', rasterized=True)
ax.scatter(df.loc[df['sig'] & ~is_marker, 'log2fc'], df.loc[df['sig'] & ~is_marker, 'neg_log10_p'], c='#457b9d', s=18, alpha=0.8, label='Significant (other)')
ax.scatter(df.loc[is_marker, 'log2fc'], df.loc[is_marker, 'neg_log10_p'], c='#e63946', s=60, zorder=5, label='Immunosuppressive markers')

# labeling the marker genes
for _, row in df[is_marker].iterrows():
    ax.annotate(
        row['gene'],
        xy=(row['log2fc'], row['neg_log10_p']),
        xytext=(8, 4), textcoords='offset points',
        fontsize=9, fontweight='bold', color='#e63946',
    )

# creating thresh lines
ax.axhline(-np.log10(FDR_THRESH), color='grey', linestyle='--', linewidth=0.8, alpha=0.7)
ax.axvline( FC_THRESH,  color='grey', linestyle='--', linewidth=0.8, alpha=0.7)
ax.axvline(-FC_THRESH,  color='grey', linestyle='--', linewidth=0.8, alpha=0.7)

ax.set_xlabel('log$_2$ Fold Change (rare vs non-rare cells)', fontsize=11)
ax.set_ylabel('-log$_{10}$ Adjusted P-value (BH)', fontsize=11)
ax.set_title(
    'Fig 3. Differential Expression: Rare vs Non-Rare Cells\n'
    '[SYNTHETIC — replace with sc.tl.rank_genes_groups on processed data]',
    fontsize=11, fontweight='bold'
)
ax.legend(fontsize=9, loc='upper left')

n_sig = df['sig'].sum()
ax.text(0.98, 0.02, f'Significant genes: {n_sig}  |  FDR < {FDR_THRESH}, |log2FC| > {FC_THRESH}', transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color='grey')

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