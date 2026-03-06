"""
NOTE: The embeddings and cluster labels are from Jacob's model (real data),
      but patient metadata and spatial coordinates are currently synthetic.
      Replace the synthetic sections with real data when available.
"""
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

# Paths
# Use artifacts produced by train.py directly.
DATA_DIR = Path('outputs')
OUT_DIR = Path('analysis')
OUT_DIR.mkdir(exist_ok=True)

# Loading real data from Jacob's model --------------------------------------------

z = np.load(DATA_DIR / 'embeddings.npy')
labels = np.load(DATA_DIR / 'cluster_labels.npy')
scores = np.load(DATA_DIR / 'rare_cell_scores.npy')
confidence = np.load(DATA_DIR / 'confidence.npy')
gate = np.load(DATA_DIR / 'gate_values.npy')
metrics = json.loads((DATA_DIR / 'metrics.json').read_text())

# Debug
print(f"Loaded real data:")
print(f"- Embeddings shape: {z.shape}")
print(f"- Number of clusters: {len(np.unique(labels))}")
print(f"- Scores range: [{scores.min():.3f}, {scores.max():.3f}]")

# finding if cell-type is rare
is_rare = scores > (scores.mean() + 2 * scores.std())
print(f"  - Rare cells: {is_rare.sum()} ({is_rare.mean()*100:.1f}%)")

# Synthetic patient data (to be replaced with real data) --------------------------

# Random seed for reproducability
np.random.seed(42)

# TODO: Replace with real patient data
n_cells = z.shape[0]
n_patients = 10 #should also come with real data

# SYNTHETIC: generating random patient assignments
patient_ids = np.random.choice(n_patients, n_cells)

# SYNTHETIC: generating random treatment responses
patient_response = np.random.choice([0, 1], n_patients)
response = patient_response[patient_ids]

# SYNTHETIC: generating random spatial coords
coords = np.random.randn(n_cells, 2) * 100

# Building AnnData Obj. (Combining the real and synthetic data) -------------------

adata = anndata.AnnData(X=z)
adata.obsm['X_gvae'] = z
adata.obsm['spatial'] = coords #TODO: Replace with actual spatial coordinates
adata.obs['cluster'] = labels.astype(str)
adata.obs['rare_score'] = scores
adata.obs['confidence'] = confidence
adata.obs['gate'] = gate
adata.obs['patient_id'] = [f'P{i:03d}' for i in patient_ids] # TODO: Replace with real patient IDs
adata.obs['response'] = response.astype(str) # TODO: Replace with real response data

# Notes about which data is synthetic & which data isn't
adata.uns['data_notes'] = {
    'embeddings': 'real (from Jacob\'s model)',
    'cluster_labels': 'real (from Jacob\'s model)',
    'rare_scores': 'real (from Jacob\'s model)',
    'patient_ids': 'SYNTHETIC - REPLACE',
    'response': 'SYNTHETIC - REPLACE',
    'spatial_coords': 'SYNTHETIC - REPLACE'
}

# save to prevent rebuilding
adata.write_h5ad(OUT_DIR / 'adata.h5ad')
print("Saved adata.h5ad")
print(f"adata.uns['data_notes']: {adata.uns['data_notes']}")

#1. UMAP ------------------------------------------------------------------------------------

print("Running UMAP:")
sc.pp.neighbors(adata, use_rep='X_gvae', n_neighbors=15)
sc.tl.umap(adata)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sc.pl.umap(adata, color='cluster', ax=axes[0], show=False, title='Clusters')
sc.pl.umap(adata, color='rare_score', ax=axes[1], show=False, title='Rare Cell Scores', cmap='Reds')
sc.pl.umap(adata, color='patient_id', ax=axes[2], show=False, title='Patient (batch check)')

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig1_umap.png', dpi=150)
plt.close()
print("Saved fig1_umap.png")

#2. Spatial Plot ----------------------------------------------------------------------------

#accessing spatial coordinates
coords = adata.obsm['spatial']
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

#Panel A, coloring all cells by cluster w/ 20 distinct colors
scatter = axes[0].scatter(
    coords[:, 0], coords[:, 1],
    c=labels, cmap='tab20',
    s=3, alpha=0.6
)

#labels
axes[0].set_title('A: Spatial Cluster Distribution',fontweight='bold')
axes[0].set_xlabel('X coordinate (µm)')
axes[0].set_ylabel('Y coordinate (µm)')
axes[0].set_aspect('equal')

# Panel B

# First, highlighting all of the normal cells
axes[1].scatter(coords[~is_rare, 0], coords[~is_rare, 1], c='#d0d0d0', s=2, alpha=0.4, label=f'Normal cells (n={(~is_rare).sum()})')

# overlaying with the rare cells
axes[1].scatter(coords[is_rare, 0], coords[is_rare, 1], c='#e63946', s=15, alpha=0.9, label=f'Rare cells (n={is_rare.sum()})', zorder=5)

axes[1].set_title('B: Spatially-Restricted Rare Cell Populations', fontweight='bold')
axes[1].set_xlabel('X coordinate (µm)')
axes[1].set_ylabel('Y coordinate (µm)')

#prevents squashing and distorting of figure
axes[1].set_aspect('equal')
axes[1].legend(loc='upper right', fontsize=9)

plt.suptitle('Fig 2. Spatial Organization of TME Cell States', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig2_spatial.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved fig2_spatial.png")

#3. Volcano Plot ----------------------------------------------------------------------------

#4. ROC Curve -------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------

print("\nAll figures saved to analysis/")
print("fig1_umap.png — latent space structure")
print("fig2_spatial.png — spatial rare cell distribution")
#print("fig3_volcano.png — differential expression of rare cluster")
#print("fig4_roc.png — immunotherapy response prediction")
