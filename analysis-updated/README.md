# GVAE-TME Analysis

## Structure

```
.
├── data/                          # Model outputs from train.py
│   ├── adata_analysis.h5ad        # Full AnnData object with embeddings + metadata
│   ├── embeddings.npy             # GVAE latent embeddings (X_gvae)
│   ├── cluster_labels.npy         # Leiden cluster assignments
│   ├── rare_cell_scores.npy       # KL-based rare cell scores
│   ├── gate_values.npy            # Cell-adaptive spatial gate values
│   ├── pooling_attention.npy      # Attention weights from pooling predictor
│   ├── confidence.npy             # Per-cell confidence scores
│   ├── metrics.json               # Training + evaluation metrics
│   ├── clinical_association.json  # Patient-level response associations
│   ├── model.pt                   # Final model checkpoint
│   ├── model_phase1.pt            # Phase 1 (unsupervised) checkpoint
│   └── model_phase2.pt            # Phase 2 (supervised) checkpoint
│
├── src/                           # Source scripts
│   ├── plots.py                   # Exploratory figures (fig1-fig4)
│   ├── rare_cell_benchmark.py     # KL vs. baseline benchmarking
│   ├── roc.py                     # ROC curve for response prediction
│   ├── test.py                    # Scratch / testing
│   ├── fig1_umap.png              # Latent space (clusters, rare scores, therapy, response)
│   ├── fig2_spatial.png           # Spatial cluster + rare cell distribution
│   ├── fig_kl_violin.png          # KL divergence: rare vs. common cells
│   ├── fig4_volcano.png           # Differential expression: rare vs. non-rare
│   └── fig_rare_cell_benchmark.png# Benchmark: KL vs. marker, CISC, scSynO
│
└── run-results/                   # Final paper figures
    ├── plots.py                   # Paper figure generation script
    └── output/                    # Rendered panel outputs (PDF + PNG)
        ├── panel_A.png/pdf        # UMAP latent space
        ├── panel_B.png/pdf        # Rare cell spatial distribution
        ├── panel_C.png/pdf        # Response prediction ROC
        ├── panel_C_conf.png/pdf   # Confidence intervals
        ├── panel_D.png/pdf        # Spatial gate weight + Moran's I
        ├── panel_E.png/pdf        # Cross-validation stability (raincloud)
        └── panel_F.png/pdf        # Ligand-receptor interactions (heatmap)
```

## src/ Summary

Exploratory and validation scripts. Run these first to sanity-check outputs before generating paper figures.

| Script | What it does |
|---|---|
| `plots.py` | Generates fig1–fig4: UMAP latent space, spatial distribution, KL violin, volcano plot |
| `rare_cell_benchmark.py` | Benchmarks KL-based rare cell detection against marker, CISC, and scSynO baselines |
| `roc.py` | Plots ROC curve for rare cell burden vs. immunotherapy response |
| `test.py` | Scratch script for ad-hoc testing — not part of the pipeline |

| Figure | Description |
|---|---|
| `fig1_umap.png` | 4-panel UMAP: cell clusters, rare cell uncertainty scores, therapy type, immunotherapy response |
| `fig2_spatial.png` | Cluster distribution in latent spatial coords (Panel A) + spatially-restricted rare cells with KDE contours (Panel B) |
| `fig_kl_violin.png` | KL divergence distributions: rare vs. common cells (p < 10⁻³⁰⁰) |
| `fig4_volcano.png` | Differential expression rare vs. non-rare — top hits: TYROBP, CST3, LYZ, FCN1, S100A11 |
| `fig_rare_cell_benchmark.png` | Precision/recall/F1/AUROC for KL vs. baselines (marker AUROC=0.696, CISC=0.434, scSynO=0.814) |

## run-results/ Summary

Final publication-ready panels. All panels rendered as both PNG and PDF via `run-results/plots.py`.

| Panel | Description |
|---|---|
| `panel_A` | UMAP latent space colored by cell cluster |
| `panel_B` | Rare cell spatial distribution with KDE density contours |
| `panel_C` | Response prediction ROC curves across all 3 cohorts |
| `panel_C_conf` | AUROC confidence intervals (bootstrapped) |
| `panel_D` | Spatial gate weight + Moran's I per dataset (split dual-panel) |
| `panel_E` | Cross-validation stability — raincloud plot of per-fold AUROC (Melanoma + NSCLC ICI) |
| `panel_F` | Top ligand-receptor interactions by dataset — faceted heatmap (Breast, NSCLC Visium, NSCLC scRNA) |

## Run Order

From `gvae-tme/`:

```bash
python train.py                       # produces data/
cd src
python plots.py                       # produces src/fig*.png
python roc.py                         # produces ROC curve
python rare_cell_benchmark.py         # produces benchmark figure
cd ../run-results
python plots.py                       # produces run-results/output/panel_*.png/pdf
```

## Data Notes

| File | Status |
|---|---|
| `embeddings.npy` | Real — GVAE latent space |
| `cluster_labels.npy` | Real — Leiden clustering |
| `rare_cell_scores.npy` | Real — KL-based detection |
| `gate_values.npy` | Real — spatial/molecular balance per cell |
| `clinical_association.json` | Real — MPR/non-MPR from GSE243013 |
| Cell type annotations | Putative — hardcoded; replace with DE-derived labels |