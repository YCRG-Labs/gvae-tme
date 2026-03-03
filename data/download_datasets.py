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


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download GVAE-TME datasets")
    parser.add_argument(
        "datasets",
        nargs="*",
        default=["all"],
        choices=["all", "melanoma", "breast", "colorectal", "nsclc", "process"],
        help="Which datasets to download",
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

    print("\n=== Summary ===")
    for name in ["melanoma", "breast", "colorectal", "nsclc"]:
        d = RAW_DIR / name
        if d.exists():
            n_files = len(list(d.iterdir()))
            print(f"  {name}: {n_files} files in {d}")
        else:
            print(f"  {name}: not downloaded")

    print("\nNext steps:")
    print("  python data/download_datasets.py process   # convert melanoma to h5ad")
    print("  For breast/colorectal/nsclc, processing scripts coming after download.")


if __name__ == "__main__":
    main()
