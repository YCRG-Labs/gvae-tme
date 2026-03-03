import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import json
import anndata
import scanpy as sc
from src.model import GVAEModel
from src.trainer import Trainer
from src.analysis import RareCellDetector, ClusteringAnalyzer, PredictionAnalyzer
from src.data_utils import prepare_graph_data, create_synthetic_data
from src.config import CONFIGS

DATASETS = {
    'melanoma': {'path': 'data/processed/melanoma.h5ad', 'has_spatial': False},
    'breast': {'path': 'data/processed/breast.h5ad', 'has_spatial': False},
    'colorectal': {'path': 'data/processed/colorectal.h5ad', 'has_spatial': True, 'r_spatial': 24.0},
    'nsclc_scrna': {'path': 'data/processed/nsclc_scrna.h5ad', 'has_spatial': False},
    'nsclc_visium': {'path': 'data/processed/nsclc_visium.h5ad', 'has_spatial': True, 'r_spatial': 24.0},
}

def load_real_data(dataset_name, n_hvg=2000, max_cells=None):
    info = DATASETS[dataset_name]
    path = Path(__file__).parent / info['path']
    print(f"Loading {path}...")
    adata = anndata.read_h5ad(path)
    print(f"  Raw: {adata.n_obs} cells, {adata.n_vars} genes")
    if max_cells and adata.n_obs > max_cells:
        sc.pp.subsample(adata, n_obs=max_cells, random_state=42)
        print(f"  Subsampled to {adata.n_obs} cells")
    if 'highly_variable' not in adata.var.columns:
        sc.pp.highly_variable_genes(adata, n_top_genes=min(n_hvg, adata.n_vars))
    adata = adata[:, adata.var['highly_variable']].copy()
    if 'X_pca' not in adata.obsm:
        n_comps = min(50, adata.n_obs - 1, adata.n_vars - 1)
        sc.pp.pca(adata, n_comps=n_comps)
    print(f"  Final: {adata.n_obs} cells, {adata.n_vars} genes, {adata.obsm['X_pca'].shape[1]} PCs")
    return adata, info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', choices=['local', 'full'], default='local')
    parser.add_argument('--data', choices=list(DATASETS.keys()) + ['synthetic'], default='synthetic')
    parser.add_argument('--max-cells', type=int, default=None)
    parser.add_argument('--n-hvg', type=int, default=2000)
    args = parser.parse_args()
    config = CONFIGS[args.config].copy()
    print(f"Config: {args.config} | Device: {config['device']} | Data: {args.data}")

    output_dir = Path('outputs') / args.data
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.data == 'synthetic':
        print("Creating synthetic data...")
        adata = create_synthetic_data(
            config['n_cells'], config['n_genes'], config['n_patients'],
            n_cell_types=config.get('n_cell_types', 8),
        )
        has_spatial = True
        graph_kwargs = {}
    else:
        adata, info = load_real_data(args.data, n_hvg=args.n_hvg, max_cells=args.max_cells)
        has_spatial = info['has_spatial']
        graph_kwargs = {'r_spatial': info.get('r_spatial', 82.5)}
    print(f"  {adata.n_obs} cells, {adata.n_vars} genes")

    print("Building graph...")
    data = prepare_graph_data(adata, has_spatial=has_spatial, **graph_kwargs)
    print(f"  {data.edge_index.size(1)} edges, "
          f"{data.pos_pairs.size(0)} positive pairs, "
          f"{data.neg_pairs.size(0)} negative pairs")

    has_response = hasattr(data, 'y') and data.y is not None
    model = GVAEModel(
        n_features=data.x.size(1),
        n_genes=data.x_raw.size(1),
        hidden_dim=config['hidden_dim'],
        latent_dim=config['latent_dim'],
        n_heads=config['n_heads'],
        dropout=config['dropout'],
        n_neg_samples=config['n_neg_samples'],
        use_predictor=has_response,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters | Predictor: {has_response}")

    trainer = Trainer(model, config, device=config['device'], checkpoint_dir=output_dir)
    phase1_metrics = trainer.train(data)

    print("\n=== Downstream Analysis ===")
    model.eval()
    data_eval = data.to(config['device'])
    with torch.no_grad():
        outputs = model(data_eval)
        z = outputs['z'].cpu().numpy()
        mu = outputs['mu'].cpu().numpy()
        logvar = outputs['logvar'].cpu().numpy()
        gate_vals = outputs['gate_values'].cpu().numpy()

    detector = RareCellDetector(threshold=2.0)
    scores, is_rare = detector.detect(mu, logvar)
    rare_labels = None
    if is_rare.sum() > 10:
        rare_labels = detector.subcluster(z, is_rare, resolution=2.0)
    print(f"Rare cells: {is_rare.sum()} / {len(is_rare)}")

    clusterer = ClusteringAnalyzer()
    labels, confidence = clusterer.cluster(z, logvar=logvar, resolution=1.0)
    eval_metrics = clusterer.evaluate(z, labels)
    print(f"Clusters: {eval_metrics['n_clusters']}, Silhouette: {eval_metrics['silhouette']:.4f}")

    pred_metrics = {}
    if 'y_pred' in outputs and hasattr(data_eval, 'y'):
        y_all = data_eval.y.cpu().numpy()
        y_pred_all = outputs['y_pred'].detach().cpu().numpy()
        if hasattr(data_eval, 'test_patient_idx') and data_eval.test_patient_idx.numel() > 0:
            test_idx = data_eval.test_patient_idx.cpu().numpy()
            y_test = y_all[test_idx]
            y_pred_test = y_pred_all[test_idx]
            pred_metrics = PredictionAnalyzer.compute_metrics(y_test, y_pred_test)
            print(f"Test AUROC: {pred_metrics['auroc']:.3f}, Test AUPRC: {pred_metrics['auprc']:.3f} (n={len(test_idx)} patients)")
        else:
            pred_metrics = PredictionAnalyzer.compute_metrics(y_all, y_pred_all)
            print(f"AUROC: {pred_metrics['auroc']:.3f}, AUPRC: {pred_metrics['auprc']:.3f} (all patients, no split)")

    print(f"\nGate statistics: mean={gate_vals.mean():.3f}, "
          f"std={gate_vals.std():.3f}, "
          f"min={gate_vals.min():.3f}, max={gate_vals.max():.3f}")

    np.save(output_dir / 'embeddings.npy', z)
    np.save(output_dir / 'rare_cell_scores.npy', scores)
    np.save(output_dir / 'cluster_labels.npy', labels)
    np.save(output_dir / 'gate_values.npy', gate_vals)
    np.save(output_dir / 'confidence.npy', confidence)

    results = {
        'dataset': args.data,
        'config': {k: v for k, v in config.items() if isinstance(v, (int, float, str, bool))},
        'phase1_metrics': phase1_metrics,
        'clustering': eval_metrics,
        'prediction': pred_metrics,
        'rare_cells': {'n_rare': int(is_rare.sum()), 'threshold': 2.0},
        'gate': {
            'mean': float(gate_vals.mean()),
            'std': float(gate_vals.std()),
        },
    }
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(make_serializable(results), f, indent=2)
    torch.save(model.state_dict(), output_dir / 'model.pt')
    print(f"\nOutputs saved to {output_dir}/")

if __name__ == '__main__':
    main()
