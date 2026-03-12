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
echo "=== Installing PyG sparse extensions (required for mini-batch training) ==="
TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda.replace('.','') if torch.cuda.is_available() else 'cpu')")
pip install torch-sparse torch-scatter -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu${CUDA_VERSION}.html" 2>/dev/null || \
    pip install torch-sparse torch-scatter || \
    echo "[warn] Could not install torch-sparse/scatter — mini-batch training will fall back to full-batch"

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
try:
    import scvi
    print(f'  scvi-tools: {scvi.__version__}')
except ImportError:
    print('  [warn] scvi-tools not installed, scVI baseline will be skipped')
from src.baselines import ScVIBaseline, ScanpyBaseline, ImmunosuppressiveSignatures, CrossDatasetTransfer
from src.analysis import CrossDatasetAnalyzer, BiologicalValidation
import benchmark
try:
    from src.minibatch import MiniBatchTrainer, check_neighbor_loader
    print('  MiniBatchTrainer: OK (NeighborLoader available)')
except Exception as e:
    print(f'  [warn] MiniBatchTrainer unavailable: {e}')
    print('  Large datasets will use full-batch training (may OOM)')
print('All imports OK')
"

echo ""
echo "=== Setup complete ==="
