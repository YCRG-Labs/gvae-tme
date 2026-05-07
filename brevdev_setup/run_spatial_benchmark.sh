#!/bin/bash
set -u

VISIUM="${VISIUM:-data/processed/nsclc_visium.h5ad}"
GVAE_RESULTS="${GVAE_RESULTS:-outputs/nsclc_visium/results.json}"
HISTOLOGY="${HISTOLOGY:-}"
OUT="${OUT:-outputs/spatial_benchmark}"
N_CLUSTERS="${N_CLUSTERS:-7}"
RAD_CUTOFF="${RAD_CUTOFF:-150}"

if [ ! -f "$VISIUM" ]; then
    echo "[error] Visium AnnData not found: $VISIUM"
    echo "        Run brevdev_setup/preprocess_data.sh first."
    exit 1
fi

if [ ! -f "${GVAE_RESULTS%/*}/cluster_labels.npy" ]; then
    echo "[warn] No GVAE cluster_labels.npy at ${GVAE_RESULTS%/*}/"
    echo "       Running benchmark.py for nsclc_visium first..."
    python3 benchmark.py --dataset nsclc_visium --method gvae \
        --output_dir outputs/nsclc_visium
fi

EXTRA_ARGS=""
if [ -n "$HISTOLOGY" ] && [ -f "$HISTOLOGY" ]; then
    EXTRA_ARGS="--histology $HISTOLOGY"
    echo "[info] Using histology image: $HISTOLOGY"
else
    echo "[info] No histology image; SpaGCN will run coords-only"
fi

echo ""
echo "=== Running spatial benchmark ==="
python3 analysis/spatial_benchmark.py \
    --visium "$VISIUM" \
    --gvae_results "$GVAE_RESULTS" \
    --output "$OUT" \
    --n_clusters "$N_CLUSTERS" \
    --rad_cutoff "$RAD_CUTOFF" \
    $EXTRA_ARGS \
    2>&1 | tee "$OUT.log"

echo ""
echo "=== Done. Results in $OUT/ ==="
echo "    JSON:     $OUT/spatial_benchmark.json"
echo "    Markdown: $OUT/spatial_benchmark.md"
echo "    Log:      $OUT.log"
