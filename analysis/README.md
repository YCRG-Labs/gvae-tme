# Analysis Outputs

This folder stores post-training outputs from `analyze.py`.

## IMPORTANT NOTE:
Right now, the spatial coordinates used are just **fake/placeholder data**. Once real tissue coordinates are added, this figure will actually show real spatial patterns within the tissue. Currently just a template showing what the figure *will* look like.

## Data Source

`analyze.py` reads directly from `../data/`, files produced by `train.py`.

No model retraining happens in `analyze.py`.

## Rare Cell Detection Benchmarking
The `rare_cell_benchmark.py` script evaluates GVAE's KL-based rare cell detection against established biological baselines through:

1. Marker-based annotation
2. CISC-style baseline
3. scSynO-style baseline

However, marker-based annotation + scSyn0-style baseline requires real data with gene expression.

## Metrics:

- **Precision**: Of cells flagged as rare, how many are truly rare?
- **Recall**: Of truly rare cells, how many did we find?
- **F1 Score**: Harmonic mean of precision and recall
- **AUROC**: How well the method ranks rare vs. non-rare cells
- **Overlap**: Cells identified by both GVAE and baseline (high-confidence candidates)

## Expected Behavior (for now)

On synthetic data: Marker-based methods return None (as expected—no gene expression)
On real data (GSE243280, melanoma datasets): All baselines become active and provide meaningful comparisons

## Run Order

From `gvae-tme/analysis`:

```bash
python train.py
python analyze.py
```

## Files Written Here

- `fig1_umap.png` - Latent space visualization
- `fig2_spatial.png` - Spatial distribution

## Important Caveat

`analyze.py` currently uses synthetic placeholders for patient IDs, response labels, and spatial coordinates.
So `fig1_umap.png` is useful latent-space QC, but `fig2_spatial.png` is demo-only until real spatial metadata is used.
# Analysis Outputs

This folder stores post-training outputs from `analyze.py`.

## Data Source

`analyze.py` reads directly from `../data/adata.h5ad`, produced by `train.py`. All embeddings,
cluster labels, rare cell scores, patient IDs, response labels, and spatial coordinates are loaded
from this file. No model retraining happens in `analyze.py`.

## Data Notes

| Field | Status |
|---|---|
| Embeddings (`X_gvae`) | Real — from Jacob's GVAE model |
| Cluster labels | Real — from Jacob's GVAE model |
| Rare cell scores | Real — KL-based, from Jacob's GVAE model |
| Patient IDs | Real — from GSE243013 |
| Response labels | Real — MPR/non-MPR from GSE243013 |
| Spatial coordinates | Real — from GSE243013 |
| Cell type annotations | Putative — hardcoded mapping, replace with DE-derived labels after brev.dev |

## Figures

- `fig1_umap.png` — GVAE latent space (4 panels: clusters, rare scores, patient batch check, cell type annotation)
- `fig2_spatial.png` — Spatial distribution of clusters and rare cells
- `fig3_volcano_[SYNTHETIC].png` — Differential expression volcano plot. Background genes are simulated; immunosuppressive markers (PDCD1, CTLA4, FOXP3, CD274, TIGIT, LAG3) are hardcoded as significant hits. Replace with `sc.tl.rank_genes_groups` on `data/processed/nsclc_ici.h5ad` once available.
- `fig4_roc_test.png` — ROC curve testing KL rare cell burden vs immunotherapy response. **Uninformative at current sample size (n=10 patients)**   — revisit after full brev.dev preprocessing with all 243 patients.

## Rare Cell Detection Benchmarking

`rare_cell_benchmark.py` evaluates GVAE's KL-based rare cell detection against established baselines:

1. **Marker-based annotation** — flags cells expressing 2+ immunosuppressive markers (PDCD1, CTLA4, FOXP3, CD274, TIGIT, LAG3) above median. Requires real gene expression — returns None on current data since `adata.h5ad` contains embeddings only.
2. **CISC-style baseline** — flags cells in small clusters (<5% of total) as rare. Works on current data using cluster labels.
3. **scSynO baseline** — stub only, requires real gene expression data.

### Metrics

- **Precision**: Of cells flagged as rare, how many are truly rare?
- **Recall**: Of truly rare cells, how many did we find?
- **F1 Score**: Harmonic mean of precision and recall
- **AUROC**: How well the method ranks rare vs non-rare cells
- **Overlap**: Cells identified by both GVAE and baseline (high-confidence candidates)

### Current Benchmark Status

| Baseline | Works now? | Blocker |
|---|---|---|
| CISC-style | ✅ | — |
| Marker-based | ❌ | Needs gene expression matrix |
| scSynO | ❌ | Needs gene expression matrix |
| ROC vs response | ❌ | n=10 patients too small, needs all 243 |

All blocked baselines become active after running `brevdev_setup/preprocess_nsclc_ici_full.py` on full dataset.

## Run Order

From `gvae-tme/analysis`:

```bash
python train.py       # produces ../data/adata.h5ad
python plots.py     # produces all figures
python roc.py       # produces roc curve
```