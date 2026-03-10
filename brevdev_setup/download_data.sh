#!/bin/bash
# Don't use set -e — we want to continue if one dataset fails
echo "=== GVAE-TME: Download Datasets ==="

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p data/raw data/processed

FAILED=""

echo ""
echo "=== Melanoma (GSE120575) ~1GB ==="
if python3 data/download_datasets.py melanoma; then
    echo "  [OK] Melanoma download complete"
else
    echo "  [FAILED] Melanoma download"
    FAILED="$FAILED melanoma"
fi

echo ""
echo "=== NSCLC ICI (GSE243013) ~7GB ==="
NSCLC_DIR="data/raw/nsclc_ici"
mkdir -p "$NSCLC_DIR"

GEO_BASE="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE243nnn/GSE243013/suppl"

NSCLC_OK=true
for FILE in \
    "GSE243013_NSCLC_immune_scRNA_counts.mtx.gz" \
    "GSE243013_NSCLC_immune_scRNA_metadata.csv.gz" \
    "GSE243013_barcodes.csv.gz" \
    "GSE243013_genes.csv.gz"; do

    OUTPATH="$NSCLC_DIR/$FILE"
    if [ -f "$OUTPATH" ]; then
        echo "  [skip] $FILE already exists"
    else
        echo "  Downloading $FILE..."
        if ! wget -c -t 3 --retry-connrefused --waitretry=10 --timeout=120 \
            -O "$OUTPATH" "$GEO_BASE/$FILE"; then
            echo "  [FAILED] $FILE"
            NSCLC_OK=false
        fi
    fi
done

if $NSCLC_OK; then
    echo "  Decompressing metadata, barcodes, genes..."
    for GZ in "$NSCLC_DIR"/GSE243013_NSCLC_immune_scRNA_metadata.csv.gz \
               "$NSCLC_DIR"/GSE243013_barcodes.csv.gz \
               "$NSCLC_DIR"/GSE243013_genes.csv.gz; do
        OUT="${GZ%.gz}"
        if [ -f "$OUT" ]; then
            echo "    [skip] $(basename "$OUT") already decompressed"
        else
            gunzip -k "$GZ"
        fi
    done
    echo "  [OK] NSCLC ICI download complete"
else
    echo "  [FAILED] NSCLC ICI download — some files missing"
    FAILED="$FAILED nsclc_ici"
fi

echo ""
echo "=== Download Summary ==="
for D in data/raw/*/; do
    if [ -d "$D" ]; then
        SIZE=$(du -sh "$D" 2>/dev/null | cut -f1)
        COUNT=$(ls -1 "$D" 2>/dev/null | wc -l | tr -d ' ')
        echo "  $(basename "$D"): $COUNT files, $SIZE"
    fi
done

if [ -n "$FAILED" ]; then
    echo ""
    echo "WARNING: Failed downloads:$FAILED"
    echo "Check network / disk space and re-run."
fi
