import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

DATASETS = {
    "melanoma": {"min_cells": 1000, "min_genes": 500, "needs_response": True, "needs_spatial": False},
    "breast": {"min_cells": 500, "min_genes": 500, "needs_response": True, "needs_spatial": False},
    "colorectal": {"min_cells": 500, "min_genes": 500, "needs_response": True, "needs_spatial": True},
    "nsclc_scrna": {"min_cells": 30, "min_genes": 100, "needs_response": False, "needs_spatial": False},
    "nsclc_visium": {"min_cells": 30, "min_genes": 100, "needs_response": False, "needs_spatial": True},
    "nsclc_ici": {"min_cells": 5000, "min_genes": 500, "needs_response": True, "needs_spatial": False},
}


def validate(name, spec):
    path = PROCESSED_DIR / f"{name}.h5ad"
    if not path.exists():
        print(f"  [{name}] SKIP — {path} not found")
        return None

    import anndata
    import numpy as np
    from scipy.sparse import issparse

    adata = anndata.read_h5ad(path)
    errors = []

    if adata.n_obs < spec["min_cells"]:
        errors.append(f"too few cells: {adata.n_obs} < {spec['min_cells']}")
    if adata.n_vars < spec["min_genes"]:
        errors.append(f"too few genes: {adata.n_vars} < {spec['min_genes']}")

    if "X_pca" not in adata.obsm:
        errors.append("missing X_pca")
    else:
        pca = adata.obsm["X_pca"]
        if not np.all(np.isfinite(pca)):
            errors.append(f"X_pca has {np.sum(~np.isfinite(pca))} non-finite values")
        if pca.shape[1] < 2:
            errors.append(f"X_pca has only {pca.shape[1]} components")

    if "counts" not in adata.layers:
        errors.append("missing counts layer (needed for ZINB decoder)")
    else:
        raw = adata.layers["counts"]
        raw_arr = raw.toarray() if issparse(raw) else np.asarray(raw)
        if np.any(raw_arr < 0):
            errors.append("counts layer has negative values")
        lib_sizes = raw_arr.sum(axis=1)
        zero_libs = (lib_sizes == 0).sum()
        if zero_libs > 0:
            errors.append(f"counts layer has {zero_libs} cells with zero total counts")

    if "spatial" not in adata.obsm:
        errors.append("missing spatial coordinates")
    else:
        coords = adata.obsm["spatial"]
        if coords.shape != (adata.n_obs, 2):
            errors.append(f"spatial shape {coords.shape}, expected ({adata.n_obs}, 2)")
        if not np.all(np.isfinite(coords)):
            errors.append("spatial has non-finite values")

    if spec["needs_response"]:
        if "patient_id" not in adata.obs.columns:
            errors.append("missing patient_id column")
        else:
            n_patients = adata.obs["patient_id"].nunique()
            if n_patients < 2:
                errors.append(f"only {n_patients} patient(s) — need >=2 for splits")
            unknown = (adata.obs["patient_id"].astype(str) == "unknown").sum()
            if unknown > 0:
                errors.append(f"{unknown} cells have patient_id='unknown'")

        if "response" not in adata.obs.columns:
            errors.append("missing response column")
        else:
            resp = adata.obs["response"]
            na_count = resp.isna().sum()
            if na_count > 0:
                errors.append(f"response has {na_count} NaN values")
            unique = set(resp.dropna().unique())
            if not unique.issubset({0, 1, 0.0, 1.0}):
                errors.append(f"response has unexpected values: {unique}")
            if len(unique) < 2:
                errors.append(f"response has only one class: {unique}")

            if "patient_id" in adata.obs.columns:
                for pid in adata.obs["patient_id"].unique():
                    pid_resp = adata.obs.loc[adata.obs["patient_id"] == pid, "response"].dropna().unique()
                    if len(pid_resp) > 1:
                        errors.append(f"patient {pid} has mixed response labels: {pid_resp}")
                        break

    if spec.get("needs_spatial"):
        coords = adata.obsm.get("spatial")
        if coords is not None and np.allclose(coords, 0):
            errors.append("spatial coords are all zeros (placeholder)")
        if coords is not None:
            std = np.std(coords, axis=0)
            if np.all(std < 1e-3):
                errors.append(f"spatial coords have near-zero variance: std={std}")

    X = adata.X
    X_arr = X.toarray() if issparse(X) else np.asarray(X)
    if not np.all(np.isfinite(X_arr)):
        n_bad = np.sum(~np.isfinite(X_arr))
        errors.append(f"X has {n_bad} non-finite values")

    if "highly_variable" in adata.var.columns:
        n_hvg = adata.var["highly_variable"].sum()
        if n_hvg == 0:
            errors.append("highly_variable column exists but 0 genes are marked")
    else:
        errors.append("missing highly_variable column (HVG selection not run?)")

    if errors:
        print(f"  [{name}] FAIL — {adata.n_obs} cells, {adata.n_vars} genes")
        for e in errors:
            print(f"    ✗ {e}")
        return False
    else:
        extra = ""
        if "patient_id" in adata.obs.columns:
            n_p = adata.obs["patient_id"].nunique()
            extra += f", {n_p} patients"
        if "response" in adata.obs.columns:
            r1 = int((adata.obs["response"] == 1).sum())
            r0 = int((adata.obs["response"] == 0).sum())
            extra += f", response={r1}R/{r0}NR"
        print(f"  [{name}] OK — {adata.n_obs} cells, {adata.n_vars} genes, "
              f"{adata.obsm['X_pca'].shape[1]} PCs{extra}")
        return True


def main():
    print("=== Validating processed datasets ===\n")
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(DATASETS.keys())
    results = {}
    for name in targets:
        if name not in DATASETS:
            print(f"  [{name}] SKIP — unknown dataset")
            continue
        results[name] = validate(name, DATASETS[name])

    tested = {k: v for k, v in results.items() if v is not None}
    passed = sum(1 for v in tested.values() if v)
    failed = sum(1 for v in tested.values() if not v)
    skipped = sum(1 for v in results.values() if v is None)

    print(f"\n=== {passed} passed, {failed} failed, {skipped} skipped ===")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
