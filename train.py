import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import json
import anndata
import scanpy as sc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
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

def build_model(config, data, use_predictor):
    return GVAEModel(
        n_features=data.x.size(1),
        n_genes=data.x_raw.size(1),
        hidden_dim=config['hidden_dim'],
        latent_dim=config['latent_dim'],
        n_heads=config['n_heads'],
        dropout=config['dropout'],
        n_neg_samples=config['n_neg_samples'],
        use_predictor=use_predictor,
    )

def assign_patient_splits(data, train_pids, val_pids, test_pids, unique_pids):
    pid_to_idx = {pid: i for i, pid in enumerate(unique_pids)}
    data.train_patient_idx = torch.tensor([pid_to_idx[p] for p in train_pids], dtype=torch.long)
    data.val_patient_idx = torch.tensor([pid_to_idx[p] for p in val_pids], dtype=torch.long)
    data.test_patient_idx = torch.tensor([pid_to_idx[p] for p in test_pids], dtype=torch.long)

def run_single(args, config):
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
    model = build_model(config, data, has_response)
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
    if is_rare.sum() > 10:
        detector.subcluster(z, is_rare, resolution=2.0)
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
            pred_metrics = PredictionAnalyzer.compute_metrics(y_all[test_idx], y_pred_all[test_idx])
            print(f"Test AUROC: {pred_metrics['auroc']:.3f}, AUPRC: {pred_metrics['auprc']:.3f} (n={len(test_idx)})")

    print(f"\nGate: mean={gate_vals.mean():.3f}, std={gate_vals.std():.3f}")

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
        'gate': {'mean': float(gate_vals.mean()), 'std': float(gate_vals.std())},
    }
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(make_serializable(results), f, indent=2)
    torch.save(model.state_dict(), output_dir / 'model.pt')
    print(f"\nOutputs saved to {output_dir}/")

def run_cv(args, config, n_outer=5, n_permutations=1000):
    output_dir = Path('outputs') / f'{args.data}_cv'
    output_dir.mkdir(parents=True, exist_ok=True)

    adata, info = load_real_data(args.data, n_hvg=args.n_hvg, max_cells=args.max_cells)
    has_spatial = info['has_spatial']
    graph_kwargs = {'r_spatial': info.get('r_spatial', 82.5)}
    print(f"  {adata.n_obs} cells, {adata.n_vars} genes")

    if 'response' not in adata.obs.columns or 'patient_id' not in adata.obs.columns:
        print("ERROR: --cv requires a dataset with response and patient_id columns")
        return

    patient_ids = adata.obs['patient_id'].values
    unique_pids = np.unique(patient_ids)
    patient_responses = np.array([
        adata.obs.loc[adata.obs['patient_id'] == pid, 'response'].iloc[0]
        for pid in unique_pids
    ])
    n_patients = len(unique_pids)
    n_resp = patient_responses.sum()
    print(f"  {n_patients} patients ({int(n_resp)} responders, {int(n_patients - n_resp)} non-responders)")

    print("Building graph...")
    data = prepare_graph_data(adata, has_spatial=has_spatial, **graph_kwargs)
    print(f"  {data.edge_index.size(1)} edges")

    skf = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=42)
    fold_results = []
    all_y_true = np.zeros(n_patients)
    all_y_pred = np.zeros(n_patients)
    pid_to_idx = {pid: i for i, pid in enumerate(unique_pids)}

    for fold, (train_val_idx, test_idx) in enumerate(skf.split(unique_pids, patient_responses)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{n_outer}")
        print(f"{'='*50}")

        test_pids = set(unique_pids[test_idx])
        train_val_pids = unique_pids[train_val_idx]
        train_val_resp = patient_responses[train_val_idx]

        n_val = max(1, len(train_val_pids) // 5)
        val_skf = StratifiedKFold(n_splits=max(2, len(train_val_pids) // n_val), shuffle=True, random_state=42 + fold)
        inner_train_idx, inner_val_idx = next(val_skf.split(train_val_pids, train_val_resp))
        train_pids = set(train_val_pids[inner_train_idx])
        val_pids = set(train_val_pids[inner_val_idx])

        print(f"  Train: {len(train_pids)} patients, Val: {len(val_pids)}, Test: {len(test_pids)}")

        cell_train = np.array([pid in train_pids for pid in patient_ids])
        cell_val = np.array([pid in val_pids for pid in patient_ids])
        cell_test = np.array([pid in test_pids for pid in patient_ids])
        data.train_mask = torch.tensor(cell_train, dtype=torch.bool)
        data.val_mask = torch.tensor(cell_val, dtype=torch.bool)
        data.test_mask = torch.tensor(cell_test, dtype=torch.bool)
        assign_patient_splits(data, sorted(train_pids), sorted(val_pids), sorted(test_pids), unique_pids)

        fold_dir = output_dir / f'fold_{fold}'
        fold_dir.mkdir(exist_ok=True)

        model = build_model(config, data, use_predictor=True)
        trainer = Trainer(model, config, device=config['device'], checkpoint_dir=fold_dir)
        trainer.train(data)

        model.eval()
        data_eval = data.to(config['device'])
        with torch.no_grad():
            outputs = model(data_eval)

        y_all = data_eval.y.cpu().numpy()
        y_pred_all = outputs['y_pred'].detach().cpu().numpy()

        test_patient_idx = data_eval.test_patient_idx.cpu().numpy()
        y_test = y_all[test_patient_idx]
        y_pred_test = y_pred_all[test_patient_idx]

        fold_metrics = PredictionAnalyzer.compute_metrics(y_test, y_pred_test)
        fold_results.append(fold_metrics)
        print(f"  Fold {fold + 1} AUROC: {fold_metrics['auroc']:.3f}, AUPRC: {fold_metrics['auprc']:.3f}")

        for pid_name in test_pids:
            idx = pid_to_idx[pid_name]
            patient_idx_in_y = list(unique_pids).index(pid_name)
            all_y_true[idx] = y_all[patient_idx_in_y]
            all_y_pred[idx] = y_pred_all[patient_idx_in_y]

        torch.save(model.state_dict(), fold_dir / 'model.pt')

    print(f"\n{'='*50}")
    print("Cross-Validation Results")
    print(f"{'='*50}")

    cv_metrics = {}
    for key in fold_results[0]:
        vals = [f[key] for f in fold_results]
        cv_metrics[key] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)), 'per_fold': vals}
        print(f"  {key}: {np.mean(vals):.3f} +/- {np.std(vals):.3f}  ({', '.join(f'{v:.3f}' for v in vals)})")

    pooled_metrics = PredictionAnalyzer.compute_metrics(all_y_true, all_y_pred)
    print(f"\n  Pooled AUROC: {pooled_metrics['auroc']:.3f}")
    print(f"  Pooled AUPRC: {pooled_metrics['auprc']:.3f}")

    print(f"\nPermutation test ({n_permutations} permutations)...")
    null_aurocs = np.zeros(n_permutations)
    for i in range(n_permutations):
        y_perm = np.random.permutation(all_y_true)
        try:
            null_aurocs[i] = roc_auc_score(y_perm, all_y_pred)
        except ValueError:
            null_aurocs[i] = 0.5
    p_value = float(np.mean(null_aurocs >= pooled_metrics['auroc']))
    print(f"  Observed AUROC: {pooled_metrics['auroc']:.3f}")
    print(f"  Null mean: {np.mean(null_aurocs):.3f} +/- {np.std(null_aurocs):.3f}")
    print(f"  p-value: {p_value:.4f}")

    results = {
        'dataset': args.data,
        'n_folds': n_outer,
        'n_patients': n_patients,
        'config': {k: v for k, v in config.items() if isinstance(v, (int, float, str, bool))},
        'cv_metrics': cv_metrics,
        'pooled_metrics': pooled_metrics,
        'permutation_test': {
            'p_value': p_value,
            'observed_auroc': pooled_metrics['auroc'],
            'null_mean': float(np.mean(null_aurocs)),
            'null_std': float(np.std(null_aurocs)),
            'n_permutations': n_permutations,
        },
        'fold_results': fold_results,
    }
    with open(output_dir / 'cv_results.json', 'w') as f:
        json.dump(make_serializable(results), f, indent=2)
    print(f"\nResults saved to {output_dir}/")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', choices=['local', 'full'], default='local')
    parser.add_argument('--data', choices=list(DATASETS.keys()) + ['synthetic'], default='synthetic')
    parser.add_argument('--max-cells', type=int, default=None)
    parser.add_argument('--n-hvg', type=int, default=2000)
    parser.add_argument('--cv', action='store_true')
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--n-permutations', type=int, default=1000)
    args = parser.parse_args()
    config = CONFIGS[args.config].copy()
    print(f"Config: {args.config} | Device: {config['device']} | Data: {args.data} | CV: {args.cv}")

    if args.cv:
        run_cv(args, config, n_outer=args.n_folds, n_permutations=args.n_permutations)
    else:
        run_single(args, config)

if __name__ == '__main__':
    main()
