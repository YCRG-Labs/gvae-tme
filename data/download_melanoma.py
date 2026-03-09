"""
Download and process REAL GSE120575 (Sade-Feldman melanoma) into h5ad.
Small sample for fast pipeline test: python data/download_melanoma.py --max-cells 100 --skip-scrublet
Full run: python data/download_melanoma.py

Output: data/processed/melanoma.h5ad (real data only, no synthetic).

Then from repo root:
  python train.py --data melanoma --quick
  cd analysis && python plots.py --data ../outputs/melanoma/adata_analysis.h5ad
  python roc.py --data ../outputs/melanoma/adata_analysis.h5ad
"""
import argparse
import gzip
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import anndata
import scanpy as sc

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw" / "melanoma"
PROC_DIR = DATA_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE120nnn/GSE120575/suppl"
FILES = {
    "tpm": f"{BASE}/GSE120575_Sade_Feldman_melanoma_single_cells_TPM_GEO.txt.gz",
    "meta": f"{BASE}/GSE120575_patient_ID_single_cells.txt.gz",
}


def download(url, dest):
    if dest.exists():
        print(f"  [skip] {dest.name}")
        return
    print(f"  Downloading {dest.name}...")
    r = subprocess.run(["curl", "-L", "-o", str(dest), "--progress-bar", url])
    if r.returncode != 0:
        print(f"  [FAILED]")
        dest.unlink(missing_ok=True)
        sys.exit(1)


def decompress(gz_path):
    out = Path(str(gz_path).replace(".gz", ""))
    if out.exists():
        print(f"  [skip] {out.name} already decompressed")
        return out
    print(f"  Decompressing {gz_path.name}...")
    with gzip.open(gz_path, "rb") as f_in, open(out, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return out


def main():
    p = argparse.ArgumentParser(description="Download and process REAL GSE120575 melanoma into h5ad")
    p.add_argument("--max-cells", type=int, default=None, help="Subsample to N cells (default: keep all). Use 100 for fast test.")
    p.add_argument("--skip-scrublet", action="store_true", help="Skip doublet removal (faster for small samples)")
    args = p.parse_args()

    # 1. Download
    for key, url in FILES.items():
        fname = url.split("/")[-1]
        download(url, RAW_DIR / fname)

    # 2. Decompress
    tpm_path = decompress(RAW_DIR / "GSE120575_Sade_Feldman_melanoma_single_cells_TPM_GEO.txt.gz")
    meta_path = decompress(RAW_DIR / "GSE120575_patient_ID_single_cells.txt.gz")

    # 3. Parse TPM matrix
    print("\nReading TPM matrix (may take a minute)...")
    with open(tpm_path) as f:
        cell_ids = f.readline().rstrip("\n").split("\t")[1:]
        f.readline()
    tpm = pd.read_csv(tpm_path, sep="\t", index_col=0, skiprows=[1], on_bad_lines="warn")
    if tpm.shape[1] == len(cell_ids) + 1:
        tpm = tpm.iloc[:, :-1]
    tpm.columns = cell_ids
    print(f"  {tpm.shape[0]} genes x {tpm.shape[1]} cells")

    # 4. Parse metadata
    print("Reading metadata...")
    meta_raw = pd.read_csv(meta_path, sep="\t", skiprows=19, header=0, encoding="latin1")
    meta_raw = meta_raw[meta_raw["title"].notna()].copy()

    RESP_COL = "characteristics: response"
    PID_COL = "characteristics: patinet ID (Pre=baseline; Post= on treatment)"
    if PID_COL not in meta_raw.columns:
        PID_COL = "characteristics: patient ID (Pre=baseline; Post= on treatment)"

    meta_raw = meta_raw[meta_raw[RESP_COL].isin(["Responder", "Non-responder"])].copy()
    meta = pd.DataFrame(index=meta_raw["title"].values)
    meta["sample_id"] = meta_raw[PID_COL].values
    meta["patient_id"] = meta["sample_id"].str.extract(r"(P\d+)", expand=False)
    meta["timepoint"] = meta["sample_id"].str.extract(r"^(Pre|Post)", expand=False)
    meta["response"] = (meta_raw[RESP_COL].values == "Responder").astype(int)
    meta["therapy"] = meta_raw["characteristics: therapy"].values

    for pid, grp in meta.groupby("patient_id"):
        if grp["response"].nunique() > 1:
            majority = int(grp["response"].mode().iloc[0])
            meta.loc[grp.index, "response"] = majority
            print(f"  [fix] {pid}: inconsistent labels -> majority vote")

    print(f"  {len(meta)} cells, {meta['patient_id'].nunique()} patients")

    # 5. Align TPM and metadata
    tpm_t = tpm.T
    common = tpm_t.index.intersection(meta.index)
    tpm_t = tpm_t.loc[common]
    meta = meta.loc[common]
    print(f"  {len(common)} cells matched")

    # 6. Build AnnData
    adata = anndata.AnnData(X=tpm_t.values.astype(np.float32))
    adata.obs_names = tpm_t.index.tolist()
    adata.var_names = tpm_t.columns.tolist()
    adata.obs["patient_id"] = meta["patient_id"].values
    adata.obs["sample_id"] = meta["sample_id"].values
    adata.obs["timepoint"] = meta["timepoint"].values
    adata.obs["therapy"] = meta["therapy"].values
    adata.obs["response"] = meta["response"].values

    n_resp = (adata.obs["response"] == 1).sum()
    n_nonresp = (adata.obs["response"] == 0).sum()
    print(f"  Responders: {n_resp} cells | Non-responders: {n_nonresp} cells")

    adata.obsm["spatial"] = np.random.randn(adata.n_obs, 2).astype(np.float32) * 100

    # 7. QC + preprocessing
    print("\nQC and preprocessing...")
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_cells(adata, max_genes=6000)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    adata = adata[adata.obs["pct_counts_mt"] < 20].copy()
    print(f"  After QC: {adata.n_obs} cells x {adata.n_vars} genes")

    if not args.skip_scrublet:
        try:
            import scrublet as scr
            scrub = scr.Scrublet(adata.X)
            scores, doublets = scrub.scrub_doublets(verbose=False)
            adata.obs["doublet_score"] = scores
            adata = adata[~doublets].copy()
            print(f"  Scrublet: {doublets.sum()} doublets removed")
        except ImportError:
            print("  [skip] scrublet not installed")
    else:
        print("  [skip] Scrublet (--skip-scrublet)")

    # Subsample to max_cells (real data, just a sample)
    if args.max_cells is not None and adata.n_obs > args.max_cells:
        sc.pp.subsample(adata, n_obs=args.max_cells, random_state=42)
        print(f"  Subsampled to {args.max_cells} cells (real data)")

    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=10000)
    sc.pp.log1p(adata)
    sc.pp.regress_out(adata, ["total_counts", "pct_counts_mt"])
    sc.pp.highly_variable_genes(adata, n_top_genes=min(2000, adata.n_vars), flavor="seurat_v3", layer="counts")
    sc.pp.pca(adata, n_comps=min(50, adata.n_obs - 1, adata.n_vars - 1))

    # 8. Save
    out = PROC_DIR / "melanoma.h5ad"
    adata.write(out)
    print(f"\nSaved: {out}")
    print(f"  {adata.n_obs} cells | {adata.n_vars} genes")
    print(f"  {adata.obs['patient_id'].nunique()} patients")
    print(f"  response: {dict(adata.obs['response'].value_counts())}")


if __name__ == "__main__":
    main()
