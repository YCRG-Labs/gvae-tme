# GVAE-TME

Graph Variational Autoencoder for Tumor Microenvironment Analysis.

## Install

```bash
pip install -e .
```

## Quick Start

```bash
python train.py
```

Trains on 5K synthetic cells. Outputs saved to `outputs/`.

## Structure

- `src/model.py` - GVAEModel with cell-adaptive gate, dual decoder
- `src/trainer.py` - Two-phase training with reconstruction safeguards
- `src/analysis.py` - Rare cell detection, clustering, prediction
- `src/data_utils.py` - Graph construction from AnnData
- `train.py` - End-to-end pipeline
- `analyze.py` - 

## Key Features

- Cell-adaptive hybrid graph fusion
- ZINB expression decoder + adjacency decoder
- KL-based rare cell detection
- Attention-based patient pooling
- Two-phase training prevents latent collapse
