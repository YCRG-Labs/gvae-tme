import os
import sys
import gzip
import shutil
import subprocess
from pathlib import Path

DATA_DIR = Path(__file__).parent
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

GEO_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series"

DOWNLOADS = {
    "melanoma": {
        "source": "GEO",
        "accession": "GSE120575",
        "description": "Sade-Feldman et al. 2018 -- 48 samples, 32 patients, responder/non-responder labels",
        "files": {
            "tpm": f"{GEO_BASE}/GSE120nnn/GSE120575/suppl/GSE120575_Sade_Feldman_melanoma_single_cells_TPM_GEO.txt.gz",
            "metadata": f"{GEO_BASE}/GSE120nnn/GSE120575/suppl/GSE120575_patient_ID_single_cells.txt.gz",
        },
    },
    "breast": {
        "source": "GEO",
        "accession": "GSE243280",
        "description": "Janesick et al. 2023 -- Breast cancer scFFPE-seq + Visium + Xenium",
        "files": {
            "raw_tar": f"{GEO_BASE}/GSE243nnn/GSE243280/suppl/GSE243280_RAW.tar",
        },
        "note": "31.8 GB download. Contains H5, CSV, and spatial files for all samples.",
    },
    "colorectal": {
        "source": "GEO",
        "accession": "GSE280318",
        "description": "10x Genomics Visium HD colorectal cancer (Nature Genetics 2025)",
        "files": {
            "raw_tar": f"{GEO_BASE}/GSE280nnn/GSE280318/suppl/GSE280318_RAW.tar",
        },
        "manual_alt": (
            "Alternative: download processed Space Ranger outputs from\n"
            "  https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-human-crc\n"
            "  GitHub code: https://github.com/10XGenomics/HumanColonCancer_VisiumHD"
        ),
    },
    "nsclc": {
        "source": "manual",
        "description": "NSCLC scRNA-seq + Visium from 10x Genomics",
        "manual_instructions": (
            "Download from 10x Genomics (requires browser, JS-rendered page):\n"
            "\n"
            "  scRNA-seq (5' GEX, ~9K cells):\n"
            "    https://www.10xgenomics.com/datasets/nsclc-tumor-1-standard-5-0-0\n"
            "    Download: 'Feature / cell matrix (filtered)' (.h5)\n"
            "\n"
            "  Visium HD lung cancer:\n"
            "    https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-human-lung-cancer-fixed-frozen\n"
            "    Download: 'Feature / cell matrix (filtered)' and 'Spatial imaging data'\n"
            "\n"
            "  Place downloaded files in: data/raw/nsclc/"
        ),
    },
}


def run_curl(url, output_path, description=""):
    if output_path.exists():
        print(f"  [skip] {output_path.name} already exists")
        return True
    print(f"  Downloading {description or output_path.name}...")
    print(f"    {url}")
    result = subprocess.run(
        ["curl", "-L", "-o", str(output_path), "--progress-bar", url],
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"  [FAILED] curl returned {result.returncode}")
        if output_path.exists():
            output_path.unlink()
        return False
    size_mb = output_path.stat().st_size / 1e6
    print(f"  [ok] {size_mb:.1f} MB")
    return True


def decompress_gz(gz_path, out_path=None):
    if out_path is None:
        out_path = Path(str(gz_path).replace(".gz", ""))
    if out_path.exists():
        print(f"  [skip] {out_path.name} already decompressed")
        return out_path
    print(f"  Decompressing {gz_path.name}...")
    with gzip.open(gz_path, "rb") as f_in:
        with open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    return out_path


def download_melanoma():
    print("\n=== Melanoma (GSE120575) ===")
    print("  Sade-Feldman et al. 2018 -- 48 samples, 32 patients, response labels")
    out_dir = RAW_DIR / "melanoma"
    out_dir.mkdir(parents=True, exist_ok=True)

    info = DOWNLOADS["melanoma"]
    for key, url in info["files"].items():
        fname = url.split("/")[-1]
        gz_path = out_dir / fname
        if run_curl(url, gz_path, key):
            decompress_gz(gz_path)


def download_breast():
    print("\n=== Breast Cancer (GSE243280) ===")
    print("  Janesick et al. 2023 -- scFFPE-seq + Visium + Xenium")
    print("  WARNING: 31.8 GB download")
    out_dir = RAW_DIR / "breast"
    out_dir.mkdir(parents=True, exist_ok=True)

    info = DOWNLOADS["breast"]
    tar_url = info["files"]["raw_tar"]
    tar_path = out_dir / "GSE243280_RAW.tar"

    if tar_path.exists():
        print(f"  [skip] {tar_path.name} already exists")
    else:
        resp = input("  Download 31.8 GB? [y/N]: ").strip().lower()
        if resp != "y":
            print("  Skipped. Download manually from:")
            print(f"    https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE243280")
            return
        run_curl(tar_url, tar_path, "GSE243280_RAW.tar")

    if tar_path.exists() and not (out_dir / "extracted").exists():
        print("  Extracting tar...")
        subprocess.run(["tar", "xf", str(tar_path), "-C", str(out_dir)])
        (out_dir / "extracted").touch()


def download_colorectal():
    print("\n=== Colorectal Cancer (GSE280318 / Visium HD) ===")
    print("  10x Genomics -- Visium HD CRC (Nature Genetics 2025)")
    out_dir = RAW_DIR / "colorectal"
    out_dir.mkdir(parents=True, exist_ok=True)

    info = DOWNLOADS["colorectal"]
    print(f"\n  {info['manual_alt']}")
    print()

    tar_url = info["files"]["raw_tar"]
    tar_path = out_dir / "GSE280318_RAW.tar"

    if tar_path.exists():
        print(f"  [skip] {tar_path.name} already exists")
        return

    resp = input("  Download from GEO (may be large)? [y/N]: ").strip().lower()
    if resp != "y":
        print("  Skipped.")
        return
    run_curl(tar_url, tar_path, "GSE280318_RAW.tar")


def download_nsclc():
    print("\n=== NSCLC ===")
    out_dir = RAW_DIR / "nsclc"
    out_dir.mkdir(parents=True, exist_ok=True)
    info = DOWNLOADS["nsclc"]
    print(f"  {info['manual_instructions']}")


def process_melanoma():
    print("\n=== Processing melanoma (GSE120575) into h5ad ===")
    raw_dir = RAW_DIR / "melanoma"
    tpm_path = raw_dir / "GSE120575_Sade_Feldman_melanoma_single_cells_TPM_GEO.txt"
    meta_path = raw_dir / "GSE120575_patient_ID_single_cells.txt"

    if not tpm_path.exists() or not meta_path.exists():
        print("  [skip] Raw files not found. Run download first.")
        return

    try:
        import pandas as pd
        import anndata
        import numpy as np
        import scanpy as sc
    except ImportError as e:
        print(f"  [error] Missing dependency: {e}")
        return

    print("  Reading TPM matrix (this may take a minute)...")
    with open(tpm_path) as f:
        header_line = f.readline().rstrip("\n")
        patient_line = f.readline().rstrip("\n")
    cell_ids = header_line.split("\t")[1:]
    sample_ids = patient_line.split("\t")[1:]
    tpm = pd.read_csv(tpm_path, sep="\t", index_col=0, skiprows=[1],
                       on_bad_lines="warn")
    if tpm.shape[1] == len(cell_ids) + 1:
        tpm = tpm.iloc[:, :-1]
    tpm.columns = cell_ids
    print(f"    {tpm.shape[0]} genes x {tpm.shape[1]} cells")

    print("  Reading patient metadata (GEO template format)...")
    meta_raw = pd.read_csv(meta_path, sep="\t", skiprows=19, header=0, encoding="latin1")
    meta_raw = meta_raw[meta_raw["title"].notna()].copy()
    resp_col_name = "characteristics: response"
    meta_raw = meta_raw[meta_raw[resp_col_name].isin(["Responder", "Non-responder"])].copy()
    meta = pd.DataFrame(index=meta_raw["title"].values)
    pid_col_name = "characteristics: patinet ID (Pre=baseline; Post= on treatment)"
    meta["sample_id"] = meta_raw[pid_col_name].values
    meta["patient_id"] = meta["sample_id"].str.extract(r"(P\d+)", expand=False)
    meta["timepoint"] = meta["sample_id"].str.extract(r"^(Pre|Post)", expand=False)
    meta["response"] = (meta_raw[resp_col_name].values == "Responder").astype(int)
    meta["therapy"] = meta_raw["characteristics: therapy"].values

    # Fix GEO metadata inconsistency: some patients (P1, P4, P5, P28) have
    # conflicting response labels across sequencing batches (e.g. Post_P1 vs
    # Post_P1_2).  Resolve to a single label per patient via majority vote.
    for pid in meta["patient_id"].unique():
        pid_mask = meta["patient_id"] == pid
        resp_vals = meta.loc[pid_mask, "response"]
        if resp_vals.nunique() > 1:
            majority_label = int(resp_vals.mode().iloc[0])
            n_before = resp_vals.value_counts().to_dict()
            meta.loc[pid_mask, "response"] = majority_label
            print(f"    [fix] {pid}: inconsistent response labels {n_before} "
                  f"-> resolved to {'Responder' if majority_label else 'Non-responder'} (majority vote)")

    print(f"    {len(meta)} cells, {meta['patient_id'].nunique()} patients, "
          f"{meta['sample_id'].nunique()} samples")

    tpm_t = tpm.T
    common = tpm_t.index.intersection(meta.index)
    tpm_t = tpm_t.loc[common]
    meta = meta.loc[common]
    print(f"    {len(common)} cells matched between TPM and metadata")

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
    print(f"  Response: {n_resp} responder cells, {n_nonresp} non-responder cells")
    print(f"  Therapies: {dict(adata.obs['therapy'].value_counts())}")

    coords_placeholder = np.random.randn(adata.n_obs, 2).astype(np.float32) * 100
    adata.obsm["spatial"] = coords_placeholder
    print("  [note] No spatial coords in GSE120575. Using placeholder for scRNA-seq mode.")

    print("  Running QC and preprocessing...")
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=10000)
    sc.pp.log1p(adata)
    n_top = min(2000, adata.n_vars)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top)
    n_comps = min(50, adata.n_obs - 1, adata.n_vars - 1)
    sc.pp.pca(adata, n_comps=n_comps)

    out_path = PROCESSED_DIR / "melanoma.h5ad"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    adata.write(out_path)
    print(f"  Saved: {out_path} ({adata.n_obs} cells, {adata.n_vars} genes)")


def process_breast():
    print("\n=== Processing breast cancer (GSE243280) into h5ad ===")
    raw_dir = RAW_DIR / "breast"

    try:
        import scanpy as sc
        import anndata
        import numpy as np
        import warnings
        warnings.filterwarnings("ignore")
    except ImportError as e:
        print(f"  [error] Missing dependency: {e}")
        return

    h5_files = [
        ("5p_scrna", "GSM7782696_5p_count_filtered_feature_bc_matrix.h5"),
        ("3p_scrna", "GSM7782697_3p_count_filtered_feature_bc_matrix.h5"),
        ("visium_raw", "GSM7782698_count_raw_feature_bc_matrix.h5"),
        ("visium_filt", "GSM7782699_filtered_feature_bc_matrix.h5"),
    ]

    adatas = []
    for label, fname in h5_files:
        path = raw_dir / fname
        if not path.exists():
            print(f"  [skip] {fname} not found")
            continue
        print(f"  Reading {label}: {fname}...")
        ad = sc.read_10x_h5(str(path))
        ad.var_names_make_unique()
        ad.obs["sample"] = label
        print(f"    {ad.n_obs} cells x {ad.n_vars} genes")
        adatas.append(ad)

    if not adatas:
        print("  [skip] No H5 files found. Run download/extract first.")
        return

    print(f"  Concatenating {len(adatas)} samples...")
    adata = anndata.concat(adatas, join="outer", fill_value=0)
    adata.var_names_make_unique()
    print(f"  Combined: {adata.n_obs} cells x {adata.n_vars} genes")

    print("  QC filtering...")
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    adata = adata[adata.obs["pct_counts_mt"] < 20].copy()
    print(f"  After QC: {adata.n_obs} cells x {adata.n_vars} genes")

    sc.pp.normalize_total(adata, target_sum=10000)
    sc.pp.log1p(adata)
    n_top = min(2000, adata.n_vars)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top)
    n_comps = min(50, adata.n_obs - 1, adata.n_vars - 1)
    sc.pp.pca(adata, n_comps=n_comps)

    coords = np.random.randn(adata.n_obs, 2).astype(np.float32) * 100
    adata.obsm["spatial"] = coords

    out_path = PROCESSED_DIR / "breast.h5ad"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    adata.write(out_path)
    print(f"  Saved: {out_path} ({adata.n_obs} cells, {adata.n_vars} genes)")


def process_colorectal():
    print("\n=== Processing colorectal cancer (GSE280318) into h5ad ===")
    raw_dir = RAW_DIR / "colorectal"

    try:
        import scanpy as sc
        import anndata
        import numpy as np
        import re
        import warnings
        warnings.filterwarnings("ignore")
    except ImportError as e:
        print(f"  [error] Missing dependency: {e}")
        return

    h5_files = sorted([f for f in os.listdir(raw_dir) if f.endswith(".h5")])
    if not h5_files:
        print("  [skip] No H5 files found. Run download/extract first.")
        return
    print(f"  Found {len(h5_files)} H5 files")

    adatas = []
    for i, fname in enumerate(h5_files):
        path = raw_dir / fname
        ad = sc.read_10x_h5(str(path))
        ad.var_names_make_unique()
        m = re.search(r"_(P\d+)(CRC|NAT)_(BC\d+)_", fname)
        if m:
            ad.obs["patient"] = m.group(1)
            ad.obs["tissue"] = m.group(2)
        else:
            ad.obs["patient"] = "unknown"
            ad.obs["tissue"] = "unknown"
        adatas.append(ad)
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(h5_files)} files read")

    print(f"  Concatenating {len(adatas)} samples...")
    adata = anndata.concat(adatas, join="outer", fill_value=0)
    adata.var_names_make_unique()
    print(f"  Combined: {adata.n_obs} spots x {adata.n_vars} genes")

    print("  QC filtering...")
    sc.pp.filter_cells(adata, min_genes=50)
    sc.pp.filter_genes(adata, min_cells=3)
    print(f"  After QC: {adata.n_obs} spots x {adata.n_vars} genes")

    sc.pp.normalize_total(adata, target_sum=10000)
    sc.pp.log1p(adata)
    n_top = min(2000, adata.n_vars)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top)
    n_comps = min(50, adata.n_obs - 1, adata.n_vars - 1)
    sc.pp.pca(adata, n_comps=n_comps)

    coords = np.random.randn(adata.n_obs, 2).astype(np.float32) * 100
    adata.obsm["spatial"] = coords

    out_path = PROCESSED_DIR / "colorectal.h5ad"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    adata.write(out_path)
    print(f"  Saved: {out_path} ({adata.n_obs} spots, {adata.n_vars} genes)")


def process_nsclc():
    print("\n=== Processing NSCLC into h5ad ===")
    raw_dir = RAW_DIR / "nsclc"

    try:
        import scanpy as sc
        import anndata
        import numpy as np
        import h5py
        from scipy.sparse import lil_matrix
        import warnings
        warnings.filterwarnings("ignore")
    except ImportError as e:
        print(f"  [error] Missing dependency: {e}")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    mtx_dir = raw_dir / "filtered_feature_bc_matrix"
    if mtx_dir.exists():
        print("  Reading scRNA-seq (10x MTX)...")
        adata_sc = sc.read_10x_mtx(str(mtx_dir), var_names="gene_symbols", cache=False)
        sc.pp.filter_cells(adata_sc, min_genes=200)
        sc.pp.filter_genes(adata_sc, min_cells=3)
        adata_sc.var["mt"] = adata_sc.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(adata_sc, qc_vars=["mt"], inplace=True)
        adata_sc = adata_sc[adata_sc.obs["pct_counts_mt"] < 20].copy()
        sc.pp.normalize_total(adata_sc, target_sum=10000)
        sc.pp.log1p(adata_sc)
        n_top = min(2000, adata_sc.n_vars)
        sc.pp.highly_variable_genes(adata_sc, n_top_genes=n_top)
        n_comps = min(50, adata_sc.n_obs - 1, adata_sc.n_vars - 1)
        sc.pp.pca(adata_sc, n_comps=n_comps)
        coords = np.random.randn(adata_sc.n_obs, 2).astype(np.float32) * 100
        adata_sc.obsm["spatial"] = coords
        out_sc = PROCESSED_DIR / "nsclc_scrna.h5ad"
        adata_sc.write(out_sc)
        print(f"  Saved: {out_sc} ({adata_sc.n_obs} cells, {adata_sc.n_vars} genes)")

    h5_path = raw_dir / "Visium_HD_Human_Lung_Cancer_Fixed_Frozen_feature_slice.h5"
    if h5_path.exists():
        print("  Reading Visium HD feature slice (2um -> 16um binning)...")
        f = h5py.File(str(h5_path), "r")
        gene_names = [g.decode() for g in f["features"]["name"][:]]
        gene_ids = [g.decode() for g in f["features"]["id"][:]]
        n_genes_total = len(gene_names)

        bin_size = 8
        grid_rows, grid_cols = 3350, 3350
        binned_rows = (grid_rows + bin_size - 1) // bin_size
        binned_cols = (grid_cols + bin_size - 1) // bin_size

        umis_row = f["umis"]["total"]["row"][:]
        umis_col = f["umis"]["total"]["col"][:]
        umis_data = f["umis"]["total"]["data"][:]
        br = umis_row // bin_size
        bc = umis_col // bin_size
        bin_idx = br * binned_cols + bc
        total_umi = np.zeros(binned_rows * binned_cols, dtype=np.float64)
        np.add.at(total_umi, bin_idx, umis_data.astype(np.float64))
        valid_bins = np.where(total_umi > 10)[0]
        n_bins = len(valid_bins)
        bin_remap = {old: new for new, old in enumerate(valid_bins)}
        print(f"    {n_bins} bins with >10 UMI")

        slice_keys = sorted(f["feature_slices"].keys(), key=int)
        mat = lil_matrix((n_bins, n_genes_total), dtype=np.float32)
        for i, gk in enumerate(slice_keys):
            gi = int(gk)
            s = f["feature_slices"][gk]
            r = s["row"][:] // bin_size
            c = s["col"][:] // bin_size
            d = s["data"][:]
            bidx = r * binned_cols + c
            for b, v in zip(bidx, d):
                lin = int(b)
                if lin in bin_remap:
                    mat[bin_remap[lin], gi] += v
            if (i + 1) % 5000 == 0:
                print(f"      {i+1}/{len(slice_keys)} genes")

        coords = np.zeros((n_bins, 2), dtype=np.float32)
        for new_idx, old_idx in enumerate(valid_bins):
            coords[new_idx, 0] = (old_idx // binned_cols) * 16.0
            coords[new_idx, 1] = (old_idx % binned_cols) * 16.0

        adata_vis = anndata.AnnData(X=mat.tocsr())
        adata_vis.var_names = gene_names
        adata_vis.var["gene_ids"] = gene_ids
        adata_vis.var_names_make_unique()
        adata_vis.obsm["spatial"] = coords

        sc.pp.filter_cells(adata_vis, min_genes=50)
        sc.pp.filter_genes(adata_vis, min_cells=3)
        sc.pp.normalize_total(adata_vis, target_sum=10000)
        sc.pp.log1p(adata_vis)
        n_top = min(2000, adata_vis.n_vars)
        sc.pp.highly_variable_genes(adata_vis, n_top_genes=n_top)
        n_comps = min(50, adata_vis.n_obs - 1, adata_vis.n_vars - 1)
        sc.pp.pca(adata_vis, n_comps=n_comps)

        out_vis = PROCESSED_DIR / "nsclc_visium.h5ad"
        adata_vis.write(out_vis)
        print(f"  Saved: {out_vis} ({adata_vis.n_obs} bins, {adata_vis.n_vars} genes)")
        f.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download and process GVAE-TME datasets")
    parser.add_argument(
        "datasets",
        nargs="*",
        default=["all"],
        choices=["all", "melanoma", "breast", "colorectal", "nsclc",
                 "process", "process-melanoma", "process-breast",
                 "process-colorectal", "process-nsclc"],
        help="Which datasets to download or process",
    )
    args = parser.parse_args()

    targets = args.datasets
    if "all" in targets:
        targets = ["melanoma", "breast", "colorectal", "nsclc"]

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for t in targets:
        if t == "melanoma":
            download_melanoma()
        elif t == "breast":
            download_breast()
        elif t == "colorectal":
            download_colorectal()
        elif t == "nsclc":
            download_nsclc()
        elif t == "process":
            process_melanoma()
            process_breast()
            process_colorectal()
            process_nsclc()
        elif t == "process-melanoma":
            process_melanoma()
        elif t == "process-breast":
            process_breast()
        elif t == "process-colorectal":
            process_colorectal()
        elif t == "process-nsclc":
            process_nsclc()

    print("\n=== Summary ===")
    for name in ["melanoma", "breast", "colorectal", "nsclc"]:
        d = RAW_DIR / name
        if d.exists():
            n_files = len(list(d.iterdir()))
            print(f"  {name}: {n_files} files in {d}")
        else:
            print(f"  {name}: not downloaded")

    processed = PROCESSED_DIR
    if processed.exists():
        for f in sorted(processed.iterdir()):
            if f.suffix == ".h5ad":
                size_mb = f.stat().st_size / 1e6
                print(f"  processed/{f.name}: {size_mb:.0f} MB")


if __name__ == "__main__":
    main()
