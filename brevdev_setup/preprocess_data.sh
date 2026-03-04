#!/bin/bash
set -e

echo "=== GVAE-TME: Preprocess All Datasets ==="

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p data/processed

echo ""
echo "=== Processing Melanoma ==="
if [ -d "data/raw/melanoma" ]; then
    python3 data/download_datasets.py process-melanoma
else
    echo "  [skip] Raw data not found"
fi

echo ""
echo "=== Processing NSCLC ICI (full 1.25M cells) ==="
if [ -f "data/raw/nsclc_ici/GSE243013_NSCLC_immune_scRNA_counts.mtx.gz" ]; then
    python3 brevdev_setup/preprocess_nsclc_ici_full.py
else
    echo "  [skip] Raw data not found"
fi

echo ""
echo "=== Processing NSCLC Demo ==="
if [ -d "data/raw/nsclc" ]; then
    python3 data/download_datasets.py process-nsclc
else
    echo "  [skip] Raw data not found"
fi

# echo ""
# echo "=== Processing Breast ==="
# if [ -d "data/raw/breast" ]; then
#     python3 data/download_datasets.py process-breast
# fi

# echo ""
# echo "=== Processing Colorectal ==="
# if [ -d "data/raw/colorectal" ]; then
#     python3 data/download_datasets.py process-colorectal
# fi

echo ""
echo "=== Processed Dataset Summary ==="
for F in data/processed/*.h5ad; do
    if [ -f "$F" ]; then
        SIZE=$(du -h "$F" | cut -f1)
        echo "  $(basename "$F"): $SIZE"
    fi
done
