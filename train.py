import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch
import numpy as np
import scanpy as sc
import anndata
import json

from src.model import GVAEModel
from src.trainer import Trainer
from src.analysis import RareCellDetector, ClusteringAnalyzer, PredictionAnalyzer
from src.data_utils import prepare_graph_data


def create_synthetic_data(n_cells=5000, n_genes=2000, n_patients=10):
    np.random.seed(42)
    torch.manual_seed(42)
    counts = np.random.poisson(2, size=(n_cells, n_genes))
    patient_ids = np.random.choice(n_patients, n_cells)
    patient_response = np.random.choice([0, 1], n_patients)
    cell_response = patient_response[patient_ids]
    coords = np.random.randn(n_cells, 2) * 100
    adata = anndata.AnnData(X=counts)
    adata.obs['patient_id'] = [f'P{i:03d}' for i in patient_ids]
    adata.obs['response'] = cell_response
    adata.obsm['spatial'] = coords
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=10000)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=min(1000, n_genes))
    sc.pp.pca(adata, n_comps=50)
    return adata


def main():
    config = {'n_cells': 5000, 'n_genes': 2000, 'n_patients': 10,
              'hidden_dim': 64, 'latent_dim': 32, 'n_heads': 4, 'dropout': 0.2,
              'lambda1': 1.0, 'lambda2': 0.5, 'beta': 0.01, 'gamma': 0.1,
              'lr': 1e-3, 'epochs_phase1': 100, 'epochs_phase2': 50,
              'patience': 20, 'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    
    adata = create_synthetic_data(config['n_cells'], config['n_genes'], config['n_patients'])
    data = prepare_graph_data(adata)
    
    model = GVAEModel(n_features=data.x.size(1), n_genes=data.x_raw.size(1),
                      hidden_dim=config['hidden_dim'], latent_dim=config['latent_dim'],
                      n_heads=config['n_heads'], dropout=config['dropout'], use_predictor=True)
    
    trainer = Trainer(model, config, device=config['device'])
    final_metrics = trainer.train(data)
    
    model.eval()
    with torch.no_grad():
        outputs = model(data)
        z = outputs['z'].cpu().numpy()
        mu = outputs['mu'].cpu().numpy()
        logvar = outputs['logvar'].cpu().numpy()
    
    detector = RareCellDetector()
    scores, is_rare = detector.detect(mu, logvar, eta=2.0)
    rare_labels = detector.subcluster(z, is_rare, resolution=2.0) if is_rare.sum() > 10 else None
    
    clusterer = ClusteringAnalyzer()
    labels = clusterer.cluster(z, logvar=logvar, resolution=1.0)
    eval_metrics = clusterer.evaluate(z, labels)
    
    pred_metrics = {}
    if 'predictions' in outputs:
        y_true = data.y.cpu().numpy()
        y_pred = outputs['predictions'].cpu().numpy()
        pred_metrics = PredictionAnalyzer.compute_metrics(y_true, y_pred)
    
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    np.save(output_dir / 'embeddings.npy', z)
    np.save(output_dir / 'rare_cell_scores.npy', scores)
    np.save(output_dir / 'cluster_labels.npy', labels)
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump({'config': config, 'final_metrics': final_metrics,
                   'clustering': eval_metrics, 'prediction': pred_metrics}, f, indent=2)
    torch.save(model.state_dict(), output_dir / 'model.pt')
    
    print(f"Training complete. Outputs saved to {output_dir}/")
    print(f"Detected {is_rare.sum()} rare cells, {eval_metrics['n_clusters']} clusters")
    if pred_metrics:
        print(f"AUROC: {pred_metrics['auroc']:.3f}, AUPRC: {pred_metrics['auprc']:.3f}")


if __name__ == '__main__':
    main()
