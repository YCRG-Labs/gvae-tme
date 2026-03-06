#!/bin/bash
set -e

echo "=== GVAE-TME: Train on All Datasets ==="

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p outputs

if [ -f "data/processed/melanoma.h5ad" ]; then
    echo ""
    echo "=== Training: Melanoma (full config) ==="
    python3 train.py --config full --data melanoma 2>&1 | tee outputs/melanoma_train.log
fi

if [ -f "data/processed/nsclc_ici.h5ad" ]; then
    echo ""
    echo "=== Training: NSCLC ICI (full config, mini-batch) ==="
    python3 train.py --config full --data nsclc_ici --batch-size 512 2>&1 | tee outputs/nsclc_ici_train.log
fi

if [ -f "data/processed/nsclc_scrna.h5ad" ]; then
    echo ""
    echo "=== Training: NSCLC scRNA (full config) ==="
    python3 train.py --config full --data nsclc_scrna 2>&1 | tee outputs/nsclc_scrna_train.log
fi

if [ -f "data/processed/nsclc_visium.h5ad" ]; then
    echo ""
    echo "=== Training: NSCLC Visium (full config) ==="
    python3 train.py --config full --data nsclc_visium 2>&1 | tee outputs/nsclc_visium_train.log
fi

if [ -f "data/processed/breast.h5ad" ]; then
    echo ""
    echo "=== Training: Breast (full config) ==="
    python3 train.py --config full --data breast 2>&1 | tee outputs/breast_train.log
fi

if [ -f "data/processed/colorectal.h5ad" ]; then
    echo ""
    echo "=== Training: Colorectal (full config, mini-batch) ==="
    python3 train.py --config full --data colorectal --batch-size 512 2>&1 | tee outputs/colorectal_train.log
fi

echo ""
echo "=== Training Complete ==="
echo "Results in outputs/"
ls -lh outputs/*.log 2>/dev/null
