import scanpy as sc
import json
from pathlib import Path

root_dir = Path.cwd().parent
source_dir = root_dir / 'data'
adata = sc.read_h5ad(source_dir / 'adata_analysis.h5ad')

print("=== STRUCTURE ===")
print(adata)
print("\n=== OBS COLUMNS ===")
print(list(adata.obs.columns))
print("\n=== OBSM KEYS ===")
print(list(adata.obsm.keys()))
print("\n=== VAR NAMES (first 20) ===")
print(list(adata.var_names[:20]))
print("\n=== RESPONSE DISTRIBUTION ===")
if 'response' in adata.obs.columns:
    print(adata.obs['response'].value_counts())
if 'patient_id' in adata.obs.columns:
    print(f"Patients: {adata.obs['patient_id'].nunique()}")
print("\n=== SPATIAL COORDS ===")
if 'spatial' in adata.obsm:
    print(f"Shape: {adata.obsm['spatial'].shape}")
    print(f"Range X: {adata.obsm['spatial'][:,0].min():.1f} to {adata.obsm['spatial'][:,0].max():.1f}")
    print(f"Range Y: {adata.obsm['spatial'][:,1].min():.1f} to {adata.obsm['spatial'][:,1].max():.1f}")
else:
    print("NO SPATIAL COORDS")
print("\n=== METRICS ===")
metrics = json.loads(open(source_dir / 'metrics.json').read())
print(json.dumps(metrics, indent=2)[:2000])
print("\n=== CLINICAL ===")
clinical = json.loads(open(source_dir / 'clinical_association.json').read())
print(json.dumps(clinical, indent=2))