#!/bin/bash
set -e

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
fi

echo ""
echo "=========================================="
echo "  2. Cross-Validation: NSCLC ICI"
echo "=========================================="
if [ -f "data/processed/nsclc_ici.h5ad" ]; then
    python3 train.py --config full --data nsclc_ici --cv --n-folds 5 --batch-size 512 --n-permutations 1000 \
        2>&1 | tee outputs/nsclc_ici_cv.log
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
echo "  Experiments Complete"
echo "=========================================="
echo "All logs in outputs/"
ls -lh outputs/*.log 2>/dev/null
echo ""
echo "Check outputs/*/metrics.json for results"
