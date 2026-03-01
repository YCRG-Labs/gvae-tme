import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import json
from src.model import GVAEModel
from src.trainer import Trainer
from src.analysis import RareCellDetector, ClusteringAnalyzer, PredictionAnalyzer
from src.data_utils import prepare_graph_data, create_synthetic_data

def main():
    config = {
        'n_cells': 5000,
        'n_genes': 2000,
        'n_patients': 10,
        'hidden_dim': 64,
        'latent_dim': 32,
        'n_heads': 4,
        'dropout': 0.2,
        'n_neg_samples': 5,
        'lambda1': 1.0,
        'lambda2': 0.5,
        'beta': 0.01,
        'beta_warmup_epochs': 50,
        'gamma': 0.1,
        'lr': 1e-3,
        'epochs_phase1': 100,
        'epochs_phase2': 50,
        'patience': 50,
        'max_grad_norm': 1.0,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    
    print("Creating synthetic data...")
    adata = create_synthetic_data(config['n_cells'], config['n_genes'], config['n_patients'])
    print(f"  {adata.n_obs} cells, {adata.n_vars} genes after QC")
    
    print("Building graph...")
    data = prepare_graph_data(adata)
    print(f"  {data.edge_index.size(1)} edges, "
          f"{data.pos_pairs.size(0)} positive pairs, "
          f"{data.neg_pairs.size(0)} negative pairs")
    
    model = GVAEModel(
        n_features=data.x.size(1),
        n_genes=data.x_raw.size(1),
        hidden_dim=config['hidden_dim'],
        latent_dim=config['latent_dim'],
        n_heads=config['n_heads'],
        dropout=config['dropout'],
        n_neg_samples=config['n_neg_samples'],
        use_predictor=True,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters")
    
    trainer = Trainer(model, config, device=config['device'])
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
        y_true = data_eval.y.cpu().numpy()
        y_pred = outputs['y_pred'].detach().cpu().numpy()
        pred_metrics = PredictionAnalyzer.compute_metrics(y_true, y_pred)
        print(f"AUROC: {pred_metrics['auroc']:.3f}, AUPRC: {pred_metrics['auprc']:.3f}")
    
    print(f"\nGate statistics: mean={gate_vals.mean():.3f}, "
          f"std={gate_vals.std():.3f}, "
          f"min={gate_vals.min():.3f}, max={gate_vals.max():.3f}")
    
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    np.save(output_dir / 'embeddings.npy', z)
    np.save(output_dir / 'rare_cell_scores.npy', scores)
    np.save(output_dir / 'cluster_labels.npy', labels)
    np.save(output_dir / 'gate_values.npy', gate_vals)
    np.save(output_dir / 'confidence.npy', confidence)
    
    results = {
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
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    torch.save(model.state_dict(), output_dir / 'model.pt')
    print(f"\nOutputs saved to {output_dir}/")

if __name__ == '__main__':
    main()
