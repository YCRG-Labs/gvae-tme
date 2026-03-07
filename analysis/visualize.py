import scanpy as sc
import pandas as pd
import anndata

from pathlib import Path

root = Path.cwd().parent
file_path = root / "data" / "adata.h5ad"

# actual file path

adata = sc.read_h5ad(file_path)
print(adata)
print("\n")
print(adata.obs.columns.tolist())
print(adata.obsm.keys())
print(adata.obs.head(3))
