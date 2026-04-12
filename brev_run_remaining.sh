#!/bin/bash
cd ~/gvae-tme
source .venv/bin/activate

pip install gseapy

echo "=== ABLATIONS ==="
for abl in mol_only spatial_only static_0.3 static_0.5 static_0.7 no_expr gaussian no_contrastive frozen_encoder gcn_encoder rare_leiden logreg_baseline; do
    echo "=== $abl ==="
    python3 -u train.py --config full --data melanoma --ablation $abl --batch-size 4096 2>&1 | tee outputs/melanoma_${abl}.log
done

echo "=== BENCHMARKING ==="
python3 -u benchmark.py --config full --data melanoma --methods gvae,scvi,scanpy --cv 2>&1 | tee outputs/benchmark_melanoma.log
python3 -u benchmark.py --config full --data nsclc_ici --methods gvae,scvi,scanpy --cv --max-cells 100000 --batch-size 4096 2>&1 | tee outputs/benchmark_nsclc.log

echo "=== CROSS-DATASET TRANSFER ==="
python3 -u benchmark.py --config full --transfer-source melanoma --transfer-target nsclc_ici --max-cells 100000 2>&1 | tee outputs/transfer.log

echo "=== SPIKE-IN VALIDATION ==="
python3 -u benchmark.py --config full --data melanoma --spike-in --batch-size 4096 2>&1 | tee outputs/spikein.log

echo "=== OPTUNA TUNING ==="
python3 -u tune.py --data melanoma --config full --n-trials 50 2>&1 | tee outputs/melanoma_tune.log

echo "=== ALL DONE ==="
