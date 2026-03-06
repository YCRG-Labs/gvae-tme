# Analysis Outputs

This folder stores post-training outputs from `analyze.py`.

## IMPORTANT NOTE:
Right now, the spatial coordinates used are just **fake/placeholder data**. Once real tissue coordinates are added, this figure will actually show real spatial patterns within the tissue. Currently just a template showing what the figure *will* look like.

## Data Source

`analyze.py` reads directly from `../outputs/`, files produced by `train.py`:

- `embeddings.npy`
- `cluster_labels.npy`
- `rare_cell_scores.npy`
- `confidence.npy`
- `gate_values.npy`
- `metrics.json`

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
- `adata.h5ad` - Annotated object for downstream analysis

## Important Caveat

`analyze.py` currently uses synthetic placeholders for patient IDs, response labels, and spatial coordinates.
So `fig1_umap.png` is useful latent-space QC, but `fig2_spatial.png` is demo-only until real spatial metadata is used.
