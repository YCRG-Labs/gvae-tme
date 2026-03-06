import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import pytest

from src.config import CONFIGS
from src.data_utils import create_synthetic_data, prepare_graph_data
from src.model import GVAEModel
from src.trainer import Trainer, GVAELoss
from src.ablations import ABLATION_REGISTRY, apply_ablation, LogisticRegressionBaseline
from src.analysis import (RareCellDetector, ClusteringAnalyzer, PredictionAnalyzer,
                          BatchMixingAnalyzer, BiologicalValidation, CrossDatasetAnalyzer,
                          ClinicalAssociationTest, LigandReceptorAnalyzer, AttentionAnalyzer)


@pytest.fixture(scope='module')
def config():
    cfg = CONFIGS['local'].copy()
    cfg['epochs_phase1'] = 5
    cfg['epochs_phase2'] = 3
    cfg['patience'] = 50
    cfg['beta_warmup_epochs'] = 2
    cfg['device'] = 'cpu'
    cfg['n_cells'] = 200
    cfg['n_genes'] = 50
    cfg['n_patients'] = 6
    cfg['n_cell_types'] = 3
    return cfg


@pytest.fixture(scope='module')
def synthetic_adata(config):
    return create_synthetic_data(
        config['n_cells'], config['n_genes'], config['n_patients'],
        n_cell_types=config['n_cell_types'])


@pytest.fixture(scope='module')
def graph_data(synthetic_adata):
    return prepare_graph_data(synthetic_adata, has_spatial=True)


@pytest.fixture(scope='module')
def trained_model(config, graph_data):
    model = GVAEModel(
        n_features=graph_data.x.size(1),
        n_genes=graph_data.x_raw.size(1),
        hidden_dim=config['hidden_dim'],
        latent_dim=config['latent_dim'],
        n_heads=config['n_heads'],
        dropout=config['dropout'],
        n_neg_samples=config['n_neg_samples'],
        use_predictor=True,
        encoder_type='gat',
        decoder_type='zinb',
        gate_mode='learned',
    )
    trainer = Trainer(model, config, device='cpu')
    trainer.train(graph_data)
    return model


def test_synthetic_data_shape(synthetic_adata, config):
    assert synthetic_adata.n_obs == config['n_cells']
    assert 'X_pca' in synthetic_adata.obsm
    assert 'spatial' in synthetic_adata.obsm
    assert 'patient_id' in synthetic_adata.obs.columns
    assert 'response' in synthetic_adata.obs.columns


def test_graph_data_structure(graph_data):
    assert hasattr(graph_data, 'x')
    assert hasattr(graph_data, 'x_raw')
    assert hasattr(graph_data, 'edge_index')
    assert hasattr(graph_data, 'mol_weight')
    assert hasattr(graph_data, 'spatial_weight')
    assert hasattr(graph_data, 'pos_pairs')
    assert hasattr(graph_data, 'neg_pairs')
    assert hasattr(graph_data, 'coords')
    assert graph_data.edge_index.size(0) == 2
    assert graph_data.edge_index.size(1) > 0


def test_model_forward(config, graph_data):
    model = GVAEModel(
        n_features=graph_data.x.size(1),
        n_genes=graph_data.x_raw.size(1),
        hidden_dim=config['hidden_dim'],
        latent_dim=config['latent_dim'],
        n_heads=config['n_heads'],
        dropout=config['dropout'],
        n_neg_samples=config['n_neg_samples'],
        use_predictor=False,
    )
    model.eval()
    with torch.no_grad():
        outputs = model(graph_data)
    assert 'z' in outputs
    assert 'mu' in outputs
    assert 'logvar' in outputs
    assert 'pos_scores' in outputs
    assert 'neg_scores' in outputs
    assert 'gate_values' in outputs
    assert outputs['z'].shape == (graph_data.x.size(0), config['latent_dim'])


def test_model_with_predictor(config, graph_data):
    model = GVAEModel(
        n_features=graph_data.x.size(1),
        n_genes=graph_data.x_raw.size(1),
        hidden_dim=config['hidden_dim'],
        latent_dim=config['latent_dim'],
        n_heads=config['n_heads'],
        dropout=config['dropout'],
        n_neg_samples=config['n_neg_samples'],
        use_predictor=True,
    )
    model.eval()
    with torch.no_grad():
        outputs = model(graph_data)
    assert 'y_pred' in outputs


def test_trainer_phase1(config, graph_data):
    model = GVAEModel(
        n_features=graph_data.x.size(1),
        n_genes=graph_data.x_raw.size(1),
        hidden_dim=config['hidden_dim'],
        latent_dim=config['latent_dim'],
        n_heads=config['n_heads'],
        dropout=config['dropout'],
        n_neg_samples=config['n_neg_samples'],
        use_predictor=False,
    )
    trainer = Trainer(model, config, device='cpu')
    metrics = trainer.train(graph_data)
    assert 'loss_adj' in metrics
    assert 'loss_expr' in metrics
    assert metrics['loss_adj'] < 10.0


def test_gcn_encoder(config, graph_data):
    model = GVAEModel(
        n_features=graph_data.x.size(1),
        n_genes=graph_data.x_raw.size(1),
        hidden_dim=config['hidden_dim'],
        latent_dim=config['latent_dim'],
        n_heads=config['n_heads'],
        dropout=config['dropout'],
        n_neg_samples=config['n_neg_samples'],
        encoder_type='gcn',
    )
    model.eval()
    with torch.no_grad():
        outputs = model(graph_data)
    assert outputs['z'].shape[1] == config['latent_dim']


def test_gaussian_decoder(config, graph_data):
    model = GVAEModel(
        n_features=graph_data.x.size(1),
        n_genes=graph_data.x_raw.size(1),
        hidden_dim=config['hidden_dim'],
        latent_dim=config['latent_dim'],
        n_heads=config['n_heads'],
        dropout=config['dropout'],
        n_neg_samples=config['n_neg_samples'],
        decoder_type='gaussian',
    )
    model.eval()
    with torch.no_grad():
        outputs = model(graph_data)
    assert 'expr_mu' in outputs
    assert 'expr_logvar' in outputs


def test_gate_modes(config, graph_data):
    for mode in ['learned', 'mol_only', 'spatial_only', 'static_0.5']:
        model = GVAEModel(
            n_features=graph_data.x.size(1),
            n_genes=graph_data.x_raw.size(1),
            hidden_dim=config['hidden_dim'],
            latent_dim=config['latent_dim'],
            n_heads=config['n_heads'],
            dropout=config['dropout'],
            n_neg_samples=config['n_neg_samples'],
            gate_mode=mode,
        )
        model.eval()
        with torch.no_grad():
            outputs = model(graph_data)
        assert outputs['z'].shape[0] == graph_data.x.size(0)


def test_ablation_registry():
    assert len(ABLATION_REGISTRY) >= 10
    for name, overrides in ABLATION_REGISTRY.items():
        cfg = CONFIGS['local'].copy()
        result = apply_ablation(cfg, name)
        assert isinstance(result, dict)


def test_loss_functions():
    loss_fn = GVAELoss()

    pos = torch.sigmoid(torch.randn(100))
    neg = torch.sigmoid(torch.randn(100))
    l_adj = loss_fn.adjacency_negsampling(pos, neg)
    assert l_adj.item() > 0

    x = torch.randn(50, 20).abs()
    rho = torch.randn(50, 20).abs() + 0.1
    theta = torch.randn(50, 20).abs() + 0.1
    pi = torch.sigmoid(torch.randn(50, 20))
    l_zinb = loss_fn.zinb(x, rho, theta, pi)
    assert torch.isfinite(l_zinb)

    mu = torch.randn(50, 20)
    logvar = torch.randn(50, 20)
    l_gauss = loss_fn.gaussian(x, mu, logvar)
    assert torch.isfinite(l_gauss)

    mu_kl = torch.randn(50, 16)
    logvar_kl = torch.randn(50, 16)
    l_kl = loss_fn.kl_divergence(mu_kl, logvar_kl)
    assert l_kl.item() >= 0


def test_rare_cell_detection(trained_model, graph_data):
    trained_model.eval()
    with torch.no_grad():
        outputs = trained_model(graph_data)
    mu = outputs['mu'].cpu().numpy()
    logvar = outputs['logvar'].cpu().numpy()
    detector = RareCellDetector(threshold=2.0)
    scores, is_rare = detector.detect(mu, logvar)
    assert len(scores) == mu.shape[0]
    assert len(is_rare) == mu.shape[0]


def test_clustering(trained_model, graph_data):
    trained_model.eval()
    with torch.no_grad():
        outputs = trained_model(graph_data)
    z = outputs['z'].cpu().numpy()
    logvar = outputs['logvar'].cpu().numpy()
    clusterer = ClusteringAnalyzer()
    best_res, best_sil = clusterer.select_resolution(z, logvar=logvar)
    labels, confidence = clusterer.cluster(z, logvar=logvar, resolution=best_res)
    metrics = clusterer.evaluate(z, labels)
    assert 'silhouette' in metrics
    assert 'n_clusters' in metrics
    assert metrics['n_clusters'] >= 1


def test_batch_mixing(trained_model, graph_data, synthetic_adata):
    trained_model.eval()
    with torch.no_grad():
        outputs = trained_model(graph_data)
    z = outputs['z'].cpu().numpy()
    batch_labels = synthetic_adata.obs['patient_id'].values
    k = min(20, len(z) - 1)
    kbet = BatchMixingAnalyzer.kbet(z, batch_labels, k=k)
    assert 'rejection_rate' in kbet
    entropy = BatchMixingAnalyzer.batch_entropy(z, batch_labels, k=k)
    assert 'batch_entropy' in entropy


def test_contrastive_loss_temperature():
    loss_fn = GVAELoss()
    z = torch.randn(50, 16)
    pos_pairs = torch.tensor([[0, 1], [2, 3], [4, 5]])
    neg_pairs = torch.tensor([[0, 10], [0, 11], [2, 12], [2, 13], [4, 14], [4, 15]])
    l1 = loss_fn.contrastive(z, pos_pairs, neg_pairs, temperature=0.1)
    l2 = loss_fn.contrastive(z, pos_pairs, neg_pairs, temperature=1.0)
    assert torch.isfinite(l1)
    assert torch.isfinite(l2)


def test_spatial_validation(trained_model, graph_data):
    trained_model.eval()
    with torch.no_grad():
        outputs = trained_model(graph_data)
    gate_vals = outputs['gate_values'].cpu().numpy()
    coords = graph_data.coords.cpu().numpy()
    if coords.std() > 1.0:
        result = BiologicalValidation.morans_i(gate_vals, coords)
        assert 'morans_i' in result


def test_minibatch_trainer_import():
    from src.minibatch import MiniBatchTrainer, check_neighbor_loader
    assert MiniBatchTrainer is not None


def test_config_keys():
    for name, cfg in CONFIGS.items():
        assert 'temperature' in cfg
        assert 'rare_threshold' in cfg
        assert 'k_mol' in cfg
        assert 'batch_size' in cfg
        assert 'num_neighbors' in cfg
        assert 'encoder_type' in cfg
        assert 'decoder_type' in cfg
        assert 'gate_mode' in cfg
