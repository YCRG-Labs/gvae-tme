#!/bin/bash
# Don't use set -e — continue experiments if one fails

echo "=== GVAE-TME: Full Paper Experiments ==="

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p outputs

echo ""
echo "=========================================="
echo "  1. Cross-Validation: Melanoma"
echo "=========================================="
if [ -f "data/processed/melanoma.h5ad" ]; then
    python3 train.py --config full --data melanoma --cv --n-folds 5 --inner-hp --n-permutations 1000 \
        2>&1 | tee outputs/melanoma_cv.log
    if [ -f "outputs/melanoma/adata_analysis.h5ad" ]; then
        echo "=== Generating plots: Melanoma ==="
        cd analysis && python3 plots.py --data ../outputs/melanoma/adata_analysis.h5ad 2>&1 | tee ../outputs/melanoma_plots.log && cd ..
    fi
fi

echo ""
echo "=========================================="
echo "  2. Cross-Validation: NSCLC ICI"
echo "=========================================="
if [ -f "data/processed/nsclc_ici.h5ad" ]; then
    python3 train.py --config full --data nsclc_ici --cv --n-folds 5 --batch-size 512 --n-permutations 1000 \
        2>&1 | tee outputs/nsclc_ici_cv.log
    if [ -f "outputs/nsclc_ici/adata_analysis.h5ad" ]; then
        echo "=== Generating plots: NSCLC ICI ==="
        cd analysis && python3 plots.py --data ../outputs/nsclc_ici/adata_analysis.h5ad 2>&1 | tee ../outputs/nsclc_ici_plots.log && cd ..
    fi
fi

echo ""
echo "=========================================="
echo "  2b. Cross-Validation: Breast"
echo "=========================================="
if [ -f "data/processed/breast.h5ad" ]; then
    python3 train.py --config full --data breast --cv --n-folds 5 --batch-size 512 --n-permutations 1000 \
        2>&1 | tee outputs/breast_cv.log
    if [ -f "outputs/breast/adata_analysis.h5ad" ]; then
        echo "=== Generating plots: Breast ==="
        cd analysis && python3 plots.py --data ../outputs/breast/adata_analysis.h5ad 2>&1 | tee ../outputs/breast_plots.log && cd ..
    fi
fi

echo ""
echo "=========================================="
echo "  3. Ablation Studies (Melanoma)"
echo "=========================================="
if [ -f "data/processed/melanoma.h5ad" ]; then
    ABLATIONS=(
        mol_only
        spatial_only
        static_0.3
        static_0.5
        static_0.7
        no_expr
        gaussian
        no_contrastive
        frozen_encoder
        gcn_encoder
        rare_leiden
        logreg_baseline
    )

    for ABL in "${ABLATIONS[@]}"; do
        echo ""
        echo "--- Ablation: $ABL ---"
        python3 train.py --config full --data melanoma --ablation "$ABL" \
            2>&1 | tee "outputs/melanoma_ablation_${ABL}.log"
    done
fi

echo ""
echo "=========================================="
echo "  4. Optuna Hyperparameter Tuning"
echo "=========================================="
if [ -f "data/processed/melanoma.h5ad" ]; then
    echo "--- Tuning: Melanoma (50 trials) ---"
    python3 tune.py --data melanoma --config full --n-trials 50 \
        2>&1 | tee outputs/melanoma_tune.log
fi

if [ -f "data/processed/nsclc_ici.h5ad" ]; then
    echo ""
    echo "--- Tuning: NSCLC ICI (50 trials) ---"
    python3 tune.py --data nsclc_ici --config full --n-trials 50 \
        2>&1 | tee outputs/nsclc_ici_tune.log
fi

echo ""
echo "=========================================="
echo "  5. Benchmark: Head-to-Head Comparison"
echo "=========================================="
if [ -f "data/processed/melanoma.h5ad" ]; then
    echo "--- Benchmark CV: Melanoma (GVAE vs scVI vs Scanpy) ---"
    python3 benchmark.py --config full --data melanoma --methods gvae,scvi,scanpy --cv \
        2>&1 | tee outputs/melanoma_benchmark.log
fi

if [ -f "data/processed/breast.h5ad" ]; then
    echo ""
    echo "--- Benchmark CV: Breast (GVAE vs scVI vs Scanpy) ---"
    python3 benchmark.py --config full --data breast --methods gvae,scvi,scanpy --batch-size 512 --cv \
        2>&1 | tee outputs/breast_benchmark.log
fi

if [ -f "data/processed/nsclc_ici.h5ad" ]; then
    echo ""
    echo "--- Benchmark CV: NSCLC ICI (GVAE vs scVI vs Scanpy) ---"
    python3 benchmark.py --config full --data nsclc_ici --methods gvae,scvi,scanpy --batch-size 512 --cv \
        2>&1 | tee outputs/nsclc_ici_benchmark.log
fi

echo ""
echo "=========================================="
echo "  6. Cross-Dataset Transfer"
echo "=========================================="
if [ -f "data/processed/melanoma.h5ad" ] && [ -f "data/processed/nsclc_ici.h5ad" ]; then
    echo "--- Transfer: Melanoma -> NSCLC ICI ---"
    python3 benchmark.py --config full --transfer-source melanoma --transfer-target nsclc_ici \
        2>&1 | tee outputs/melanoma_to_nsclc_transfer.log
fi

echo ""
echo "=========================================="
echo "  7. Synthetic Spike-In Validation"
echo "=========================================="
if [ -f "data/processed/melanoma.h5ad" ]; then
    echo "--- Spike-in: Melanoma ---"
    python3 benchmark.py --config full --data melanoma --spike-in \
        2>&1 | tee outputs/melanoma_spike_in.log
fi

echo ""
echo "=========================================="
echo "  Experiments Complete"
echo "=========================================="
echo ""
echo "=== Logs ==="
ls -lh outputs/*.log 2>/dev/null
echo ""
echo "=== Checkpoints ==="
find outputs -name "*.pt" -exec ls -lh {} \; 2>/dev/null
echo ""
echo "=== Results ==="
find outputs -name "*.json" -exec echo {} \; 2>/dev/null
echo ""
echo "Done."
