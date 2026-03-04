#!/bin/bash
set -e

echo "=== GVAE-TME: Download Datasets ==="

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p data/raw data/processed

echo ""
echo "=== Melanoma (GSE120575) ~1GB ==="
python3 data/download_datasets.py melanoma

echo ""
echo "=== NSCLC ICI (GSE243013) ~7GB ==="
NSCLC_DIR="data/raw/nsclc_ici"
mkdir -p "$NSCLC_DIR"

GEO_BASE="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE243nnn/GSE243013/suppl"

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
        wget -c -t 0 --retry-connrefused --waitretry=10 --timeout=60 \
            -O "$OUTPATH" "$GEO_BASE/$FILE"
    fi
done

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

echo ""
echo "=== NSCLC Demo (10x Genomics) ==="
echo "  Manual download required (JS-rendered pages):"
echo "  scRNA: https://www.10xgenomics.com/datasets/nsclc-tumor-1-standard-5-0-0"
echo "  Visium HD: https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-human-lung-cancer-fixed-frozen"

# echo ""
# echo "=== Breast Cancer (GSE243280) ~32GB ==="
# python3 data/download_datasets.py breast

# echo ""
# echo "=== Colorectal (GSE280318) ~113GB ==="
# python3 data/download_datasets.py colorectal

echo ""
echo "=== Download Summary ==="
for D in data/raw/*/; do
    if [ -d "$D" ]; then
        SIZE=$(du -sh "$D" 2>/dev/null | cut -f1)
        COUNT=$(ls -1 "$D" 2>/dev/null | wc -l | tr -d ' ')
        echo "  $(basename "$D"): $COUNT files, $SIZE"
    fi
done
