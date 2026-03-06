# Analysis Outputs

This folder stores post-training outputs from `analyze.py`.

## IMPORTANT NOTE:
Right now, the spatial coordinates used are just **fake/placeholder data**. Once real tissue coordinates are added, this figure will actually show real spatial patterns within the tissue. Currently just a template showing what the figure *will* look like.

## Data Source

`analyze.py` reads directly from `../outputs/` (the files produced by `train.py`):

- `embeddings.npy`
- `cluster_labels.npy`
- `rare_cell_scores.npy`
- `confidence.npy`
- `gate_values.npy`
- `metrics.json`

No model retraining happens in `analyze.py`.

## Run Order

From `gvae-tme/`:

```bash
python train.py
python analyze.py
```

## Files Written Here

- `fig1_umap.png`
- `fig2_spatial.png`
- `adata.h5ad`

## Important Caveat

`analyze.py` currently uses synthetic placeholders for patient IDs, response labels, and spatial coordinates.
So `fig1_umap.png` is useful latent-space QC, but `fig2_spatial.png` is demo-only until real spatial metadata is used.
