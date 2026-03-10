import sys
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import anndata
import scanpy as sc
from scipy.io import mmread
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw" / "nsclc_ici"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def run_scrublet(adata):
    try:
        import scrublet as scr
        scrub = scr.Scrublet(adata.X)
        doublet_scores, predicted_doublets = scrub.scrub_doublets(verbose=False)
        adata.obs['doublet_score'] = doublet_scores
        adata.obs['predicted_doublet'] = predicted_doublets
        n_doublets = int(predicted_doublets.sum())
        adata = adata[~adata.obs['predicted_doublet']].copy()
        print(f"  Scrublet: removed {n_doublets} doublets ({adata.n_obs} cells remaining)")
        return adata
    except ImportError:
        print("  [warn] scrublet not installed, skipping")
        return adata


def main():
    print("=== Processing NSCLC ICI (GSE243013) — FULL 1.25M cells ===")

    mtx_path = RAW_DIR / "GSE243013_NSCLC_immune_scRNA_counts.mtx.gz"
    barcodes_path = RAW_DIR / "GSE243013_barcodes.csv"
    genes_path = RAW_DIR / "GSE243013_genes.csv"
    meta_path = RAW_DIR / "GSE243013_NSCLC_immune_scRNA_metadata.csv"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for p, name in [(mtx_path, "counts matrix"), (barcodes_path, "barcodes"),
                     (genes_path, "genes"), (meta_path, "metadata")]:
        if not p.exists():
            print(f"  [error] {name} not found at {p}")
            sys.exit(1)

    print("  Reading barcodes and genes...")
    barcodes = pd.read_csv(barcodes_path)
    genes = pd.read_csv(genes_path)
    barcode_list = barcodes.iloc[:, -1].astype(str).values
    gene_list = genes.iloc[:, -1].astype(str).values
    print(f"  {len(barcode_list)} barcodes, {len(gene_list)} genes")

    print("  Reading counts matrix (mmread, needs ~32GB RAM)...")
    X_raw = mmread(str(mtx_path))
    print(f"  Raw shape: {X_raw.shape}")

    if X_raw.shape[0] == len(barcode_list) and X_raw.shape[1] == len(gene_list):
        X = csr_matrix(X_raw)
    elif X_raw.shape[0] == len(gene_list) and X_raw.shape[1] == len(barcode_list):
        print("  Transposing to cells x genes...")
        X = csr_matrix(X_raw.T)
    else:
        print(f"  [error] Shape mismatch: matrix {X_raw.shape} vs barcodes {len(barcode_list)} x genes {len(gene_list)}")
        sys.exit(1)
    del X_raw
    gc.collect()
    print(f"  CSR matrix: {X.shape[0]} cells x {X.shape[1]} genes")

    print("  Building AnnData...")
    adata = anndata.AnnData(X=X, obs=pd.DataFrame(index=barcode_list), var=pd.DataFrame(index=gene_list))
    del X
    gc.collect()
    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    print("  Reading metadata...")
    meta = pd.read_csv(meta_path, low_memory=False)
    print(f"  Metadata: {len(meta)} rows")

    meta_indexed = meta.set_index('cellID')
    transfer_cols = ['sampleID', 'major_cell_type', 'sub_cell_type',
                     'gender', 'age', 'smoking_history', 'cancer_type',
                     'pre_treatment_staging', 'anti-PD1_therapy', 'chemotherapy',
                     'targeted_therapy', 'cycles', 'pathological_response',
                     'pathological_response_rate', 'radiological_response']
    for col in transfer_cols:
        if col in meta_indexed.columns:
            adata.obs[col] = meta_indexed.reindex(adata.obs_names)[col]
    del meta, meta_indexed
    gc.collect()

    adata.obs['patient_id'] = adata.obs['sampleID']

    resp_map = {'MPR': 1, 'pCR': 1, 'non-MPR': 0}
    adata.obs['response'] = adata.obs['pathological_response'].map(resp_map)
    n_before = adata.n_obs
    adata = adata[adata.obs['response'].notna()].copy()
    print(f"  Dropped {n_before - adata.n_obs} cells with unknown response")
    print(f"  {adata.n_obs} cells, {adata.obs['patient_id'].nunique()} patients")
    print(f"  Response: {dict(adata.obs['response'].value_counts())}")

    print("  QC filtering...")
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_cells(adata, max_genes=6000)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    adata = adata[adata.obs["pct_counts_mt"] < 20].copy()
    print(f"  After QC: {adata.n_obs} cells x {adata.n_vars} genes")

    adata = run_scrublet(adata)

    print("  Normalizing...")
    adata.layers['counts'] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=10000)
    sc.pp.log1p(adata)

    print("  Selecting 2000 HVGs (seurat_v3)...")
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat_v3', layer='counts')

    print(f"  Subsetting to HVGs before regress_out (avoids 268GB dense matrix)...")
    adata = adata[:, adata.var['highly_variable']].copy()
    gc.collect()
    print(f"  After HVG subset: {adata.n_obs} cells x {adata.n_vars} genes")

    print("  Regressing out confounders...")
    sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])

    print("  Running PCA...")
    n_comps = min(50, adata.n_obs - 1, adata.n_vars - 1)
    sc.pp.pca(adata, n_comps=n_comps)

    coords = np.random.randn(adata.n_obs, 2).astype(np.float32) * 100
    adata.obsm["spatial"] = coords

    out_path = PROCESSED_DIR / "nsclc_ici.h5ad"
    print(f"  Writing {out_path}...")
    adata.write(out_path)
    print(f"  Saved: {out_path}")
    print(f"  {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"  {adata.obs['patient_id'].nunique()} patients")
    print(f"  Response: {dict(adata.obs['response'].value_counts())}")


if __name__ == "__main__":
    main()
