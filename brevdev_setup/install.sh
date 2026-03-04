#!/bin/bash
set -e

echo "=== GVAE-TME: Brev GPU Environment Setup ==="

if [ ! -d ".git" ]; then
    echo "Cloning repository..."
    git clone https://github.com/YCRG-Labs/gvae-tme.git
    cd gvae-tme
else
    echo "Already in repo directory"
fi

echo ""
echo "=== Installing core dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=== Installing optional dependencies ==="
pip install scikit-misc annoy gseapy celltypist scrublet h5py scvi-tools

echo ""
echo "=== Verifying GPU ==="
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

echo ""
nvidia-smi

echo ""
echo "=== Verifying imports ==="
python3 -c "
import torch_geometric
import scanpy
import anndata
import optuna
import scrublet
from skmisc.loess import loess
print('All imports OK')
"

echo ""
echo "=== Setup complete ==="
