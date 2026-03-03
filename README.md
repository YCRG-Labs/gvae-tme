# GVAE-TME

Graph Variational Autoencoder for Tumor Microenvironment Analysis. Learns spatial and molecular cell representations via a dual-graph GAT encoder with cell-adaptive gating, ZINB expression decoder, and attention-based patient pooling for treatment response prediction.

## Quick Start

```bash
pip install -r requirements.txt

python train.py --config local   # CPU/MPS sanity check (500 cells, ~30s)
python train.py --config full    # Full run (5K cells, use GPU)
```

Outputs saved to `outputs/`.

## Structure

- `src/config.py` - LOCAL and FULL configs, device detection (cuda/mps/cpu)
- `src/model.py` - GVAEModel with cell-adaptive gate, dual decoder, response predictor
- `src/trainer.py` - Two-phase training with reconstruction safeguards
- `src/analysis.py` - Rare cell detection, clustering, prediction metrics
- `src/data_utils.py` - Graph construction from AnnData, patient-level splits
- `train.py` - End-to-end pipeline with config CLI

## Architecture

- **Encoder**: Two-layer GAT with cell-adaptive hybrid edge weights (molecular + spatial graphs fused per-cell via learned gate)
- **Expression decoder**: ZINB likelihood over raw counts
- **Adjacency decoder**: Dot-product with negative sampling
- **Response predictor**: Attention pooling over patient cells, binary classifier
- **Training**: Phase 1 (representation) then Phase 2 (clinical fine-tuning with reconstruction safeguards)

## Key Features

- Cell-adaptive hybrid graph fusion (learned gate blends molecular and spatial graphs per-cell)
- ZINB expression decoder + adjacency decoder
- KL-based rare cell detection
- Attention-based patient pooling for treatment response
- Two-phase training prevents latent collapse
- Patient-level train/val/test splits (no data leak in Phase 2)
- Contrastive loss with adaptive negative mining

## Requirements

- Python 3.9+
- PyTorch 2.0+
- PyTorch Geometric 2.3+
- scanpy, anndata, leidenalg
- See `requirements.txt` for full list
