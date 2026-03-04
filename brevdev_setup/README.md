# GVAE-TME: Brev GPU Setup

Run full-scale GVAE-TME training on a Brev A100 GPU instance.

## Quick Start

```bash
# 1. Set up environment
bash brevdev_setup/install.sh

# 2. Download datasets
bash brevdev_setup/download_data.sh

# 3. Preprocess (full 1.25M cells for NSCLC ICI)
bash brevdev_setup/preprocess_data.sh

# 4. Train on all datasets
bash brevdev_setup/train_all.sh

# 5. Run full paper experiments (CV, ablations, tuning)
bash brevdev_setup/run_experiments.sh
```

## Requirements

- Brev A100 instance (80GB VRAM, 128GB+ RAM recommended)
- CUDA 11.8+ with PyTorch 2.0+
- ~50GB disk space for datasets + outputs

## What Each Script Does

| Script | Purpose | Time Estimate |
|--------|---------|---------------|
| `install.sh` | Clone repo, install deps, verify GPU | ~5 min |
| `download_data.sh` | Download melanoma + NSCLC ICI from GEO | ~15 min |
| `preprocess_data.sh` | Process raw data into h5ad files | ~30 min |
| `train_all.sh` | Train GVAE on each dataset | ~2-4 hours |
| `run_experiments.sh` | CV + 12 ablations + Optuna tuning | ~12-24 hours |

## Monitoring

```bash
nvidia-smi -l 5        # GPU utilization every 5s
htop                    # CPU/RAM usage
tail -f outputs/*.log   # Training progress
```

## Troubleshooting

**CUDA OOM**: Reduce batch size
```bash
python3 train.py --config full --data nsclc_ici --batch-size 256
```

**Download fails**: NCBI FTP drops connections. Scripts use wget with auto-retry.

**regress_out slow**: Expected for 1M+ cells. Takes ~30-60 min on A100 node (CPU-bound).
