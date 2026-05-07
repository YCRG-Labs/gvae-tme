#!/bin/bash
echo "=== Installing spatial-graph baselines ==="

PIP_FLAGS=""
if python3 -m pip install --help 2>&1 | grep -q break-system-packages; then
    PIP_FLAGS="--break-system-packages"
fi
PIP="python3 -m pip install $PIP_FLAGS"

echo ""
echo "--- GraphST ---"
$PIP GraphST==1.1.1 || $PIP GraphST || echo "[warn] GraphST install failed"

echo ""
echo "--- rpy2 + R/mclust (for GraphST clustering) ---"
$PIP rpy2 || echo "[warn] rpy2 install failed; GraphST will fall back to leiden"
if command -v R >/dev/null 2>&1; then
    Rscript -e 'if (!require("mclust")) install.packages("mclust", repos="https://cloud.r-project.org/")' \
        || echo "[warn] R/mclust install failed"
else
    echo "[warn] R not available; install with: apt-get install -y r-base"
fi

echo ""
echo "--- STAGATE_pyG ---"
$PIP STAGATE_pyG || $PIP "git+https://github.com/QIFEIDKN/STAGATE_pyG.git" \
    || echo "[warn] STAGATE_pyG install failed"

echo ""
echo "--- SpaGCN ---"
$PIP SpaGCN || $PIP "git+https://github.com/jianhuupenn/SpaGCN.git" \
    || echo "[warn] SpaGCN install failed"
$PIP opencv-python-headless || echo "[warn] opencv install failed; SpaGCN will run coords-only"

echo ""
echo "=== Verifying spatial baseline imports ==="
python3 - <<'PY'
import importlib
for mod in ['GraphST', 'STAGATE_pyG', 'SpaGCN']:
    try:
        m = importlib.import_module(mod)
        print(f"  {mod}: OK ({getattr(m, '__version__', 'unknown')})")
    except Exception as e:
        print(f"  {mod}: MISSING — {e}")
try:
    import cv2
    print(f"  cv2: {cv2.__version__}")
except Exception as e:
    print(f"  cv2: MISSING — {e}")
PY

echo ""
echo "=== Spatial baseline setup complete ==="
