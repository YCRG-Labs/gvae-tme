import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
import anndata
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.baselines import (GraphSTBaseline, STAGATEBaseline, SpaGCNBaseline,
                           _evaluate_spatial_clustering)


def load_gvae_clustering(gvae_results_path, adata, ground_truth_key=None):
    gvae_results_path = Path(gvae_results_path)
    out_dir = gvae_results_path.parent

    labels_path = out_dir / 'cluster_labels.npy'
    z_path = out_dir / 'z.npy'

    if not labels_path.exists():
        raise FileNotFoundError(
            f"Expected GVAE cluster labels at {labels_path}. Re-run "
            f"benchmark.py --dataset nsclc_visium first.")

    labels = np.load(labels_path).astype(int)
    z = np.load(z_path) if z_path.exists() else None

    if len(labels) != adata.n_obs:
        raise ValueError(
            f"GVAE labels length {len(labels)} != adata.n_obs {adata.n_obs}. "
            f"Did you regenerate the AnnData since the GVAE run?")

    coords = adata.obsm.get('spatial')
    gt = adata.obs[ground_truth_key].values if ground_truth_key and \
        ground_truth_key in adata.obs.columns else None

    metrics = _evaluate_spatial_clustering(z, labels, coords, ground_truth=gt)
    print(f"  GVAE (loaded): {metrics['n_clusters']} clusters, "
          f"silhouette_spatial={metrics.get('silhouette_spatial')}, "
          f"morans_i={metrics.get('morans_i')}")
    return {'method': 'gvae', 'z': z, 'labels': labels, 'clustering': metrics}


def to_markdown_table(results):
    rows = []
    header = ("| Method | N domains | Silhouette (spatial) | "
              "Silhouette (embed) | Moran's I | ARI | NMI |")
    sep = "|---|---|---|---|---|---|---|"
    rows.append(header)
    rows.append(sep)
    for r in results:
        m = r.get('clustering', {})
        def fmt(v, p=3):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "—"
            if isinstance(v, (int, np.integer)):
                return str(int(v))
            return f"{v:.{p}f}"
        rows.append(
            f"| {r['method']} | {fmt(m.get('n_clusters'))} | "
            f"{fmt(m.get('silhouette_spatial'))} | "
            f"{fmt(m.get('silhouette_embedding'))} | "
            f"{fmt(m.get('morans_i'))} | "
            f"{fmt(m.get('ARI'))} | {fmt(m.get('NMI'))} |"
        )
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--visium', default='data/processed/nsclc_visium.h5ad')
    ap.add_argument('--gvae_results',
                    default='outputs/nsclc_visium/results.json')
    ap.add_argument('--histology', default=None)
    ap.add_argument('--output', default='outputs/spatial_benchmark')
    ap.add_argument('--ground_truth_key', default=None)
    ap.add_argument('--n_clusters', type=int, default=7)
    ap.add_argument('--rad_cutoff', type=float, default=150.0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--methods', nargs='+',
                    default=['gvae', 'graphst', 'stagate', 'spagcn'])
    args = ap.parse_args()

    visium_path = Path(args.visium)
    if not visium_path.exists():
        raise FileNotFoundError(f"AnnData not found: {visium_path}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {visium_path}")
    adata = sc.read_h5ad(visium_path)
    print(f"  {adata.n_obs} spots x {adata.n_vars} genes")
    print(f"  spatial coords: {adata.obsm.get('spatial', 'MISSING').shape if 'spatial' in adata.obsm else 'MISSING'}")

    results = []

    if 'gvae' in args.methods:
        print("\n=== GVAE (loading from prior run) ===")
        try:
            results.append(load_gvae_clustering(
                args.gvae_results, adata, ground_truth_key=args.ground_truth_key))
        except Exception as e:
            print(f"  [warn] could not load GVAE results: {e}")
            results.append({'method': 'gvae', 'note': str(e)})

    if 'graphst' in args.methods:
        print("\n=== GraphST ===")
        try:
            results.append(GraphSTBaseline.run(
                adata, n_clusters=args.n_clusters, device=args.device,
                ground_truth_key=args.ground_truth_key))
        except Exception as e:
            print(f"  [warn] GraphST failed: {e}")
            results.append({'method': 'graphst', 'note': str(e)})

    if 'stagate' in args.methods:
        print("\n=== STAGATE ===")
        try:
            results.append(STAGATEBaseline.run(
                adata, rad_cutoff=args.rad_cutoff,
                ground_truth_key=args.ground_truth_key))
        except Exception as e:
            print(f"  [warn] STAGATE failed: {e}")
            results.append({'method': 'stagate', 'note': str(e)})

    if 'spagcn' in args.methods:
        print("\n=== SpaGCN ===")
        try:
            results.append(SpaGCNBaseline.run(
                adata, histology_path=args.histology,
                n_clusters=args.n_clusters,
                ground_truth_key=args.ground_truth_key))
        except Exception as e:
            print(f"  [warn] SpaGCN failed: {e}")
            results.append({'method': 'spagcn', 'note': str(e)})

    serializable = []
    for r in results:
        rs = {k: v for k, v in r.items() if k not in ('z', 'labels')}
        if 'labels' in r and r['labels'] is not None:
            np.save(out_dir / f"labels_{r['method']}.npy", r['labels'])
            rs['labels_path'] = str(out_dir / f"labels_{r['method']}.npy")
        if 'z' in r and r['z'] is not None:
            np.save(out_dir / f"z_{r['method']}.npy", r['z'])
            rs['z_path'] = str(out_dir / f"z_{r['method']}.npy")
        serializable.append(rs)

    json_path = out_dir / 'spatial_benchmark.json'
    with open(json_path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nWrote {json_path}")

    md = to_markdown_table(results)
    md_path = out_dir / 'spatial_benchmark.md'
    md_path.write_text(md + "\n")
    print(f"Wrote {md_path}\n")
    print(md)


if __name__ == '__main__':
    main()
