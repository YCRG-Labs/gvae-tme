#!/bin/bash
# Don't use set -e — continue if one dataset fails
echo "=== GVAE-TME: Preprocess All Datasets ==="

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p data/processed

FAILED=""

echo ""
echo "=== Processing Melanoma ==="
if [ -d "data/raw/melanoma" ]; then
    if python3 data/download_datasets.py process-melanoma; then
        echo "  [OK] Melanoma preprocessed"
    else
        echo "  [FAILED] Melanoma preprocessing"
        FAILED="$FAILED melanoma"
    fi
else
    echo "  [skip] Raw data not found (run download_data.sh first)"
fi

echo ""
echo "=== Processing NSCLC ICI (full 1.25M cells) ==="
echo "  NOTE: This loads the full count matrix (~32GB peak RAM). Takes 15-30 min."
if [ -f "data/raw/nsclc_ici/GSE243013_NSCLC_immune_scRNA_counts.mtx.gz" ]; then
    if python3 brevdev_setup/preprocess_nsclc_ici_full.py 2>&1 | tee data/processed/nsclc_ici_preprocess.log; then
        echo "  [OK] NSCLC ICI preprocessed"
    else
        echo "  [FAILED] NSCLC ICI preprocessing — check data/processed/nsclc_ici_preprocess.log"
        FAILED="$FAILED nsclc_ici"
    fi
else
    echo "  [skip] Raw data not found (run download_data.sh first)"
fi

echo ""
echo "=== Processing NSCLC Demo ==="
if [ -d "data/raw/nsclc" ]; then
    if python3 data/download_datasets.py process-nsclc; then
        echo "  [OK] NSCLC Demo preprocessed"
    else
        echo "  [FAILED] NSCLC Demo preprocessing"
        FAILED="$FAILED nsclc_demo"
    fi
else
    echo "  [skip] Raw data not found (requires manual 10x Genomics download)"
fi

echo ""
echo "=== Processing Breast Cancer ==="
if [ -d "data/raw/breast" ]; then
    if python3 data/download_datasets.py process-breast 2>&1 | tee data/processed/breast_preprocess.log; then
        echo "  [OK] Breast preprocessed"
    else
        echo "  [FAILED] Breast preprocessing — check data/processed/breast_preprocess.log"
        FAILED="$FAILED breast"
    fi
else
    echo "  [skip] Raw data not found (run download_data.sh first)"
fi

echo ""
echo "=== Processing Colorectal ==="
if [ -d "data/raw/colorectal" ]; then
    if python3 data/download_datasets.py process-colorectal 2>&1 | tee data/processed/colorectal_preprocess.log; then
        echo "  [OK] Colorectal preprocessed"
    else
        echo "  [FAILED] Colorectal preprocessing — check data/processed/colorectal_preprocess.log"
        FAILED="$FAILED colorectal"
    fi
else
    echo "  [skip] Raw data not found (run download_data.sh first)"
fi

echo ""
echo "=== Processed Dataset Summary ==="
for F in data/processed/*.h5ad; do
    if [ -f "$F" ]; then
        SIZE=$(du -h "$F" | cut -f1)
        echo "  $(basename "$F"): $SIZE"
    fi
done

if [ -n "$FAILED" ]; then
    echo ""
    echo "WARNING: Failed preprocessing:$FAILED"
    echo "Check logs above for details."
fi
