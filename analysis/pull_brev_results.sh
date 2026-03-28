#!/bin/bash
# Pull key result files from brev outputs and print summary

echo "============================================================"
echo "  MELANOMA METRICS"
echo "============================================================"
cat outputs/melanoma/metrics.json | python3 -m json.tool 2>/dev/null || echo "  NOT FOUND"

echo ""
echo "============================================================"
echo "  BREAST METRICS"
echo "============================================================"
cat outputs/breast/metrics.json | python3 -m json.tool 2>/dev/null || echo "  NOT FOUND"

echo ""
echo "============================================================"
echo "  TRANSFER RESULTS"
echo "============================================================"
cat outputs/melanoma_to_nsclc_ici/transfer_joint/results.json | python3 -m json.tool 2>/dev/null || echo "  NOT FOUND"
