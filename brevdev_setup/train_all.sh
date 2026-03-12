#!/bin/bash
# Don't use set -e — continue training other datasets if one fails
echo "=== GVAE-TME: Train on All Datasets ==="

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p outputs

if [ -f "data/processed/melanoma.h5ad" ]; then
    echo ""
    echo "=== Training: Melanoma (full config) ==="
    python3 train.py --config full --data melanoma 2>&1 | tee outputs/melanoma_train.log
    if [ -f "outputs/melanoma/adata_analysis.h5ad" ]; then
        echo "=== Generating plots: Melanoma ==="
        cd analysis && python3 plots.py --data ../outputs/melanoma/adata_analysis.h5ad 2>&1 | tee ../outputs/melanoma_plots.log && cd ..
    fi
fi

if [ -f "data/processed/nsclc_ici.h5ad" ]; then
    echo ""
    echo "=== Training: NSCLC ICI (full config, mini-batch) ==="
    python3 train.py --config full --data nsclc_ici --batch-size 512 --max-cells 200000 2>&1 | tee outputs/nsclc_ici_train.log
    if [ -f "outputs/nsclc_ici/adata_analysis.h5ad" ]; then
        echo "=== Generating plots: NSCLC ICI ==="
        cd analysis && python3 plots.py --data ../outputs/nsclc_ici/adata_analysis.h5ad 2>&1 | tee ../outputs/nsclc_ici_plots.log && cd ..
    fi
fi

if [ -f "data/processed/breast.h5ad" ]; then
    echo ""
    echo "=== Training: Breast (full config, mini-batch) ==="
    python3 train.py --config full --data breast --batch-size 512 --max-cells 50000 2>&1 | tee outputs/breast_train.log
    if [ -f "outputs/breast/adata_analysis.h5ad" ]; then
        echo "=== Generating plots: Breast ==="
        cd analysis && python3 plots.py --data ../outputs/breast/adata_analysis.h5ad 2>&1 | tee ../outputs/breast_plots.log && cd ..
    fi
fi

for DATASET in nsclc_scrna nsclc_visium; do
    if [ -f "data/processed/${DATASET}.h5ad" ]; then
        echo ""
        echo "=== Training: ${DATASET} (full config) ==="
        python3 train.py --config full --data "$DATASET" --batch-size 512 2>&1 | tee "outputs/${DATASET}_train.log"
        if [ -f "outputs/${DATASET}/adata_analysis.h5ad" ]; then
            echo "=== Generating plots: ${DATASET} ==="
            cd analysis && python3 plots.py --data "../outputs/${DATASET}/adata_analysis.h5ad" 2>&1 | tee "../outputs/${DATASET}_plots.log" && cd ..
        fi
    fi
done

if [ -f "data/processed/colorectal.h5ad" ]; then
    echo ""
    echo "=== Training: Colorectal (full config, mini-batch) ==="
    python3 train.py --config full --data colorectal --batch-size 512 --max-cells 200000 2>&1 | tee outputs/colorectal_train.log
    if [ -f "outputs/colorectal/adata_analysis.h5ad" ]; then
        echo "=== Generating plots: Colorectal ==="
        cd analysis && python3 plots.py --data ../outputs/colorectal/adata_analysis.h5ad 2>&1 | tee ../outputs/colorectal_plots.log && cd ..
    fi
fi

echo ""
echo "=== Training Complete ==="
echo "Results in outputs/"
ls -lh outputs/*.log 2>/dev/null
