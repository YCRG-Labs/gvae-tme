import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

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
from src.analysis import (RareCellDetector, ClusteringAnalyzer, PredictionAnalyzer,
                          BatchMixingAnalyzer, ClinicalAssociationTest,
                          BiologicalValidation, CrossDatasetAnalyzer,
                          LigandReceptorAnalyzer, AttentionAnalyzer,
                          Cell2LocationWrapper, CellTypeAnnotator)
from src.data_utils import prepare_graph_data, create_synthetic_data, SplatterSimulator
from src.config import CONFIGS
from src.ablations import ABLATION_REGISTRY, apply_ablation, LogisticRegressionBaseline
from src.baselines import ImmunosuppressiveSignatures

DATASETS = {
    'melanoma': {'path': 'data/processed/melanoma.h5ad', 'has_spatial': False},
    'breast': {'path': 'data/processed/breast.h5ad', 'has_spatial': False},
    'colorectal': {'path': 'data/processed/colorectal.h5ad', 'has_spatial': True, 'r_spatial': 24.0},
    'nsclc_scrna': {'path': 'data/processed/nsclc_scrna.h5ad', 'has_spatial': False, 'epochs_phase1': 500, 'epochs_phase2': 300},
    'nsclc_visium': {'path': 'data/processed/nsclc_visium.h5ad', 'has_spatial': True, 'r_spatial': 24.0, 'epochs_phase1': 500, 'epochs_phase2': 300},
    'nsclc_ici': {'path': 'data/processed/nsclc_ici.h5ad', 'has_spatial': False, 'epochs_phase1': 500, 'epochs_phase2': 300},
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
        hvg_kwargs = {'n_top_genes': min(n_hvg, adata.n_vars)}
        if 'counts' in adata.layers:
            hvg_kwargs['flavor'] = 'seurat_v3'
            hvg_kwargs['layer'] = 'counts'
        sc.pp.highly_variable_genes(adata, **hvg_kwargs)
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
        encoder_type=config.get('encoder_type', 'gat'),
        decoder_type=config.get('decoder_type', 'zinb'),
        gate_mode=config.get('gate_mode', 'learned'),
        spatial_bias=config.get('spatial_bias', 0.0),
    )

def make_trainer(model, config, device, output_dir, freeze_encoder=False, data=None):

    batch_size = config.get('batch_size')
    if batch_size is not None:
        try:
            from src.minibatch import MiniBatchTrainer, check_neighbor_loader
            if data is not None:
                check_neighbor_loader(data, config.get('num_neighbors', [15, 10]))
            return MiniBatchTrainer(model, config, device=device,
                                    checkpoint_dir=output_dir,
                                    freeze_encoder=freeze_encoder)
        except (ImportError, Exception) as e:
            print(f"  [warn] Mini-batch unavailable: {e}")
            print(f"  Install pyg-lib or torch-sparse for NeighborLoader support.")
            print(f"  Falling back to full-batch training.")
            config['batch_size'] = None
    return Trainer(model, config, device=device,
                   checkpoint_dir=output_dir,
                   freeze_encoder=freeze_encoder)

def assign_patient_splits(data, train_pids, val_pids, test_pids, unique_pids):
    pid_to_idx = {pid: i for i, pid in enumerate(unique_pids)}
    data.train_patient_idx = torch.tensor([pid_to_idx[p] for p in train_pids], dtype=torch.long)
    data.val_patient_idx = torch.tensor([pid_to_idx[p] for p in val_pids], dtype=torch.long)
    data.test_patient_idx = torch.tensor([pid_to_idx[p] for p in test_pids], dtype=torch.long)

def run_downstream(model, data, config, adata, output_dir, ablation=None):
    model.eval()
    data_eval = data.to(config['device'])
    with torch.no_grad():
        outputs = model(data_eval)
        z = outputs['z'].cpu().numpy()
        mu = outputs['mu'].cpu().numpy()
        logvar = outputs['logvar'].cpu().numpy()
        gate_vals = outputs['gate_values'].cpu().numpy()

    rare_method = config.get('_rare_method', 'kl')
    detector = RareCellDetector(threshold=config.get('rare_threshold', 2.0))
    if rare_method == 'leiden':
        print("  [ablation] Using Leiden-only rare cell detection")
        scores = np.zeros(len(z))
        is_rare = np.ones(len(z), dtype=bool)  # all cells go through Leiden
        rare_labels = detector.subcluster(z, is_rare, resolution=2.0)
    else:
        scores, is_rare = detector.detect(mu, logvar)
        rare_labels = np.full(len(z), -1, dtype=int)
        if is_rare.sum() > 10:
            rare_labels = detector.subcluster(z, is_rare, resolution=2.0)
    print(f"Rare cells: {is_rare.sum()} / {len(is_rare)}")

    clusterer = ClusteringAnalyzer()
    best_res, best_sil = clusterer.select_resolution(z, logvar=logvar)
    labels, confidence = clusterer.cluster(z, logvar=logvar, resolution=best_res)
    eval_metrics = clusterer.evaluate(z, labels)
    eval_metrics['resolution'] = best_res
    print(f"Clusters: {eval_metrics['n_clusters']}, Silhouette: {eval_metrics['silhouette']:.4f}, Resolution: {best_res}")

    ari_result = BiologicalValidation.ari_stability(z, resolution=best_res)
    eval_metrics['ari_stability'] = ari_result
    print(f"ARI stability: {ari_result.get('mean_ari', 0):.3f} +/- {ari_result.get('std_ari', 0):.3f}")

    spatial_metrics = {}
    coords = data_eval.coords.cpu().numpy()
    if coords.std() > 1.0:
        spatial_metrics = BiologicalValidation.morans_i(gate_vals, coords)
        print(f"Moran's I (gate): {spatial_metrics.get('morans_i', 0):.3f}, "
              f"p={spatial_metrics.get('p_value', 1):.4f}")
        perm_result = BiologicalValidation.spatial_permutation_test(
            gate_vals, coords, n_permutations=1000)
        spatial_metrics['permutation_test'] = perm_result
        print(f"Spatial permutation test: p={perm_result['p_value']:.4f}")

    marker_results = {}
    gsea_results = {}
    lr_results = {}
    try:
        markers = CrossDatasetAnalyzer.marker_genes(z, labels, adata)
        marker_results = markers
        gsea_results = BiologicalValidation.gsea_enrichment(markers)
        if gsea_results and not gsea_results.get('note'):
            n_enriched = sum(1 for v in gsea_results.values()
                             if isinstance(v, list) and len(v) > 0)
            print(f"GSEA: {n_enriched}/{len(gsea_results)} clusters with enriched pathways")
        lr_results = LigandReceptorAnalyzer.score_interactions(adata, labels)
        if lr_results.get('n_interactions', 0) > 0:
            print(f"L-R interactions: {lr_results['n_interactions']} scored, "
                  f"{lr_results['n_valid_pairs']} valid pairs")
    except Exception as e:
        print(f"  [warn] Marker/GSEA/LR analysis: {e}")

    rare_marker_results = {}
    rare_gsea_results = {}
    signature_results = {}
    try:
        unique_rare = np.unique(rare_labels[rare_labels >= 100])
        if is_rare.sum() > 10 and len(unique_rare) >= 2:
            rare_marker_results = CrossDatasetAnalyzer.marker_genes_rare_subclusters(
                z, rare_labels, adata)
            if rare_marker_results:
                rare_gsea_results = BiologicalValidation.gsea_rare_subclusters(
                    rare_marker_results)
                if rare_gsea_results and not rare_gsea_results.get('note'):
                    n_enriched = sum(1 for v in rare_gsea_results.values()
                                     if isinstance(v, dict) and 'error' not in str(v))
                    print(f"Rare GSEA: {n_enriched}/{len(rare_gsea_results)} subclusters enriched")
        signature_results = ImmunosuppressiveSignatures.compare_rare_vs_nonrare(adata, is_rare)
        if isinstance(signature_results, dict) and 'note' not in signature_results:
            for sig_name, sig_data in signature_results.items():
                if isinstance(sig_data, dict) and 'rare_mean' in sig_data:
                    print(f"  {sig_name}: rare={sig_data['rare_mean']:.3f}, "
                          f"nonrare={sig_data['nonrare_mean']:.3f}, "
                          f"p={sig_data['p_value']:.4f}")
    except Exception as e:
        print(f"  [warn] Rare subcluster validation: {e}")

    batch_metrics = {}
    if 'patient_id' in adata.obs.columns:
        batch_labels = adata.obs['patient_id'].values
        kbet_result = BatchMixingAnalyzer.kbet(z, batch_labels, k=min(50, len(z) - 1))
        entropy_result = BatchMixingAnalyzer.batch_entropy(z, batch_labels, k=min(50, len(z) - 1))
        batch_metrics = {**kbet_result, **entropy_result}
        print(f"kBET rejection: {kbet_result['rejection_rate']:.3f}, "
              f"Batch entropy: {entropy_result['batch_entropy']:.3f}")

    pred_metrics = {}
    if 'y_pred' in outputs and hasattr(data_eval, 'y'):
        y_all = data_eval.y.cpu().numpy()
        y_pred_all = outputs['y_pred'].detach().cpu().numpy()
        if hasattr(data_eval, 'test_patient_idx') and data_eval.test_patient_idx.numel() > 0:
            test_idx = data_eval.test_patient_idx.cpu().numpy()
            pred_metrics = PredictionAnalyzer.compute_metrics(y_all[test_idx], y_pred_all[test_idx])
            print(f"Test AUROC: {pred_metrics['auroc']:.3f}, AUPRC: {pred_metrics['auprc']:.3f} (n={len(test_idx)})")

    logreg_results = {}
    if config.get('_prediction_method') == 'logreg' and hasattr(data_eval, 'y'):
        y = data_eval.y.cpu().numpy()
        logreg_results = LogisticRegressionBaseline.run(
            z, labels, data_eval.patient_masks, y)

    clinical_results = {}
    if ('response' in adata.obs.columns and 'patient_id' in adata.obs.columns
            and hasattr(data_eval, 'y')):
        fractions = ClinicalAssociationTest.compute_rare_fractions(
            rare_labels, data_eval.patient_masks)
        if not fractions.empty:
            therapy = adata.obs.groupby('patient_id')['therapy'].first().values \
                if 'therapy' in adata.obs.columns else None
            response = data_eval.y.cpu().numpy()
            clinical_results = ClinicalAssociationTest.test_association(
                fractions, response, therapy=therapy)
            ClinicalAssociationTest.summary(clinical_results)

    attention_metrics = {}
    attn_weights = outputs.get('attention_weights')
    if attn_weights is not None and config.get('encoder_type', 'gat') == 'gat':
        try:
            edge_index = data_eval.edge_index.cpu()
            sel = AttentionAnalyzer.selectivity(attn_weights.cpu(), edge_index, len(z))
            attention_metrics['mean_selectivity'] = float(np.mean(sel))
            attention_metrics['std_selectivity'] = float(np.std(sel))
            if 'cell_type' in adata.obs.columns:
                cell_types = adata.obs['cell_type'].values
                interactions = AttentionAnalyzer.interaction_network(
                    attn_weights.cpu(), edge_index, cell_types, len(z))
                attention_metrics['n_interaction_pairs'] = len(interactions)
                top_interactions = sorted(interactions.items(), key=lambda x: x[1], reverse=True)[:20]
                attention_metrics['top_interactions'] = [
                    {'source': k[0], 'target': k[1], 'count': v}
                    for k, v in top_interactions
                ]
            print(f"Attention selectivity: {attention_metrics['mean_selectivity']:.3f} +/- "
                  f"{attention_metrics['std_selectivity']:.3f}")
            if 'cell_type' in adata.obs.columns:
                novel_result = AttentionAnalyzer.novel_interactions(
                    attn_weights.cpu(), edge_index, cell_types, adata)
                attention_metrics['novel_interactions'] = novel_result
                print(f"  Novel interactions: {novel_result['n_novel']}/{novel_result['n_high_attention']} "
                      f"high-attention edges have no known L-R pair "
                      f"({novel_result['fraction_novel']:.1%})")
        except Exception as e:
            print(f"  [warn] Attention analysis: {e}")

    pooling_map = {}
    if 'y_pred' in outputs and 'attentions' in outputs:
        try:
            pooling_map = PredictionAnalyzer.pooling_attention_map(
                outputs['attentions'], coords)
            np.save(output_dir / 'pooling_attention.npy', pooling_map['attention_per_cell'])
        except Exception:
            pass

    celltype_results = {}
    try:
        import celltypist
        if 'cell_type' not in adata.obs.columns:
            celltype_results = CellTypeAnnotator.annotate(adata)
            if celltype_results.get('n_types', 0) > 0:
                print(f"CellTypist: {celltype_results['n_types']} cell types annotated")
    except ImportError:
        pass

    if lr_results.get('n_interactions', 0) > 0:
        try:
            LigandReceptorAnalyzer.cellphonedb_format(adata, labels, output_dir / 'cellphonedb')
        except Exception:
            pass

    print(f"\nGate: mean={gate_vals.mean():.3f}, std={gate_vals.std():.3f}")

    np.save(output_dir / 'embeddings.npy', z)
    np.save(output_dir / 'rare_cell_scores.npy', scores)
    np.save(output_dir / 'cluster_labels.npy', labels)
    np.save(output_dir / 'gate_values.npy', gate_vals)
    np.save(output_dir / 'confidence.npy', confidence)

    # Save analysis-ready adata for plots.py
    adata.obsm['X_gvae'] = z
    adata.obs['cluster'] = labels
    adata.obs['rare_score'] = scores
    adata.obs['confidence'] = confidence
    adata.obs['gate'] = gate_vals
    adata_path = output_dir / 'adata_analysis.h5ad'
    adata.write(adata_path)
    print(f"Saved analysis adata: {adata_path}")

    if clinical_results:
        with open(output_dir / 'clinical_association.json', 'w') as f:
            json.dump(make_serializable(clinical_results), f, indent=2)

    return {
        'clustering': eval_metrics,
        'prediction': pred_metrics,
        'batch_mixing': batch_metrics,
        'rare_cells': {'n_rare': int(is_rare.sum()),
                       'threshold': config.get('rare_threshold', 2.0),
                       'method': rare_method,
                       'rare_markers': rare_marker_results,
                       'rare_gsea': rare_gsea_results,
                       'immunosuppressive_signatures': signature_results},
        'spatial_validation': spatial_metrics,
        'gate': {'mean': float(gate_vals.mean()), 'std': float(gate_vals.std())},
        'logreg_baseline': logreg_results,
        'clinical_association': clinical_results,
        'gsea': gsea_results,
        'ligand_receptor': lr_results,
        'attention': attention_metrics,
        'cell_types': celltype_results,
    }


def run_single(args, config):
    ablation = getattr(args, 'ablation', None)
    suffix = f'_{ablation}' if ablation else ''
    output_dir = Path('outputs') / f'{args.data}{suffix}'
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.data == 'synthetic':
        print("Creating synthetic data...")
        adata = SplatterSimulator.simulate(
            n_cells=config['n_cells'], n_genes=config['n_genes'],
            n_groups=config.get('n_cell_types', 8))
        if adata is None:
            print("  [info] Splatter unavailable, using built-in simulator")
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

    if adata.n_obs > 50_000 and config.get('batch_size') is None:
        config['batch_size'] = 512
        print(f"  [auto] Enabled mini-batch training (batch_size=512) for {adata.n_obs} cells")

    print("Building graph...")
    data = prepare_graph_data(adata, has_spatial=has_spatial, **graph_kwargs)
    print(f"  {data.edge_index.size(1)} edges, "
          f"{data.pos_pairs.size(0)} positive pairs, "
          f"{data.neg_pairs.size(0)} negative pairs")

    has_response = hasattr(data, 'y') and data.y is not None
    model = build_model(config, data, has_response)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters | Predictor: {has_response} "
          f"| Encoder: {config.get('encoder_type', 'gat')} "
          f"| Decoder: {config.get('decoder_type', 'zinb')} "
          f"| Gate: {config.get('gate_mode', 'learned')}")

    freeze = config.get('freeze_encoder', False)
    trainer = make_trainer(model, config, config['device'], output_dir, freeze_encoder=freeze, data=data)
    phase1_metrics = trainer.train(data)

    print("\n=== Downstream Analysis ===")
    downstream = run_downstream(model, data, config, adata, output_dir, ablation=ablation)

    results = {
        'dataset': args.data,
        'ablation': ablation,
        'config': {k: v for k, v in config.items() if isinstance(v, (int, float, str, bool, list))},
        'phase1_metrics': phase1_metrics,
        **downstream,
    }
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(make_serializable(results), f, indent=2)
    torch.save(model.state_dict(), output_dir / 'model.pt')
    print(f"\nOutputs saved to {output_dir}/")

INNER_HP_GRID = [
    {'latent_dim': 16, 'lambda1': 0.5},
    {'latent_dim': 32, 'lambda1': 1.0},
    {'latent_dim': 64, 'lambda1': 1.0},
    {'latent_dim': 32, 'lambda1': 0.5},
    {'latent_dim': 32, 'lambda1': 2.0},
]


def inner_hp_select(config, data, train_pids, val_pids, unique_pids, patient_ids,
                    device, hp_grid=None):
    if hp_grid is None:
        hp_grid = INNER_HP_GRID
    best_loss = float('inf')
    best_overrides = {}
    for overrides in hp_grid:
        trial_config = config.copy()
        trial_config.update(overrides)
        trial_config['hidden_dim'] = trial_config['n_heads'] * (
            trial_config['latent_dim'] // trial_config['n_heads'] or 1)
        trial_config['hidden_dim'] = max(trial_config['hidden_dim'],
                                         trial_config['n_heads'] * 4)
        trial_config['epochs_phase1'] = min(100, trial_config.get('epochs_phase1', 300))
        trial_config['patience'] = 15

        cell_train = np.array([pid in train_pids for pid in patient_ids])
        cell_val = np.array([pid in val_pids for pid in patient_ids])
        data.train_mask = torch.tensor(cell_train, dtype=torch.bool)
        data.val_mask = torch.tensor(cell_val, dtype=torch.bool)

        model = build_model(trial_config, data, use_predictor=False)
        trainer = Trainer(model, trial_config, device=device)
        try:
            trainer.train(data)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                torch.cuda.empty_cache()
                continue
            raise
        val = trainer.evaluate(data.to(device))
        val_loss = val['loss_adj'] + trial_config.get('lambda1', 1.0) * val['loss_expr']
        if val_loss < best_loss:
            best_loss = val_loss
            best_overrides = overrides
    return best_overrides, best_loss


def run_cv(args, config, n_outer=5, n_permutations=1000):
    ablation = getattr(args, 'ablation', None)
    suffix = f'_{ablation}' if ablation else ''
    output_dir = Path('outputs') / f'{args.data}_cv{suffix}'
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

    if adata.n_obs > 50_000 and config.get('batch_size') is None:
        config['batch_size'] = 512
        print(f"  [auto] Enabled mini-batch training (batch_size=512) for {adata.n_obs} cells")

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
        min_class_count = int(min(np.bincount(train_val_resp.astype(int))))
        n_splits_inner = max(2, min(len(train_val_pids) // max(n_val, 1), min_class_count))
        val_skf = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=42 + fold)
        inner_train_idx, inner_val_idx = next(val_skf.split(train_val_pids, train_val_resp))
        train_pids = set(train_val_pids[inner_train_idx])
        val_pids = set(train_val_pids[inner_val_idx])

        print(f"  Train: {len(train_pids)} patients, Val: {len(val_pids)}, Test: {len(test_pids)}")

        if args.inner_hp:
            print(f"  Inner HP selection ({len(INNER_HP_GRID)} configs)...")
            best_overrides, inner_loss = inner_hp_select(
                config, data, train_pids, val_pids, unique_pids,
                patient_ids, config['device'])
            if best_overrides:
                fold_config = config.copy()
                fold_config.update(best_overrides)
                fold_config['hidden_dim'] = fold_config['n_heads'] * (
                    fold_config['latent_dim'] // fold_config['n_heads'] or 1)
                fold_config['hidden_dim'] = max(fold_config['hidden_dim'],
                                                fold_config['n_heads'] * 4)
                print(f"  Selected: {best_overrides} (val_loss={inner_loss:.4f})")
            else:
                fold_config = config
        else:
            fold_config = config

        cell_train = np.array([pid in train_pids for pid in patient_ids])
        cell_val = np.array([pid in val_pids for pid in patient_ids])
        cell_test = np.array([pid in test_pids for pid in patient_ids])
        data.train_mask = torch.tensor(cell_train, dtype=torch.bool)
        data.val_mask = torch.tensor(cell_val, dtype=torch.bool)
        data.test_mask = torch.tensor(cell_test, dtype=torch.bool)
        assign_patient_splits(data, sorted(train_pids), sorted(val_pids), sorted(test_pids), unique_pids)

        fold_dir = output_dir / f'fold_{fold}'
        fold_dir.mkdir(exist_ok=True)

        model = build_model(fold_config, data, use_predictor=True)
        freeze = fold_config.get('freeze_encoder', False)
        trainer = make_trainer(model, fold_config, fold_config['device'], fold_dir, freeze_encoder=freeze, data=data)
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

        # Run downstream on the last fold to produce analysis adata
        if fold == n_outer - 1:
            print(f"\n=== Running downstream analysis (fold {fold + 1}) ===")
            run_downstream(model, data, fold_config, adata, output_dir, ablation=ablation)

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

    bootstrap = PredictionAnalyzer.bootstrap_ci(all_y_true, all_y_pred, n_bootstrap=1000)
    print(f"  AUROC 95% CI: [{bootstrap['auroc_ci'][0]:.3f}, {bootstrap['auroc_ci'][1]:.3f}]")
    print(f"  AUPRC 95% CI: [{bootstrap['auprc_ci'][0]:.3f}, {bootstrap['auprc_ci'][1]:.3f}]")

    print(f"\nPermutation test ({n_permutations} permutations)...")
    rng = np.random.RandomState(42)
    null_aurocs = np.zeros(n_permutations)
    for i in range(n_permutations):
        y_perm = rng.permutation(all_y_true)
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
        'ablation': ablation,
        'n_folds': n_outer,
        'n_patients': n_patients,
        'config': {k: v for k, v in config.items() if isinstance(v, (int, float, str, bool, list))},
        'cv_metrics': cv_metrics,
        'pooled_metrics': pooled_metrics,
        'bootstrap_ci': bootstrap,
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
    parser.add_argument('--ablation', choices=list(ABLATION_REGISTRY.keys()),
                        default=None, help='Ablation study to run')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Mini-batch size (enables NeighborLoader training)')
    parser.add_argument('--inner-hp', action='store_true',
                        help='Enable inner HP selection in nested CV')
    args = parser.parse_args()

    config = CONFIGS[args.config].copy()

    if args.ablation:
        config = apply_ablation(config, args.ablation)

    if args.data in DATASETS:
        dataset_overrides = {k: v for k, v in DATASETS[args.data].items()
                             if k in ('epochs_phase1', 'epochs_phase2')}
        config.update(dataset_overrides)

    if args.batch_size is not None:
        config['batch_size'] = args.batch_size

    print(f"Config: {args.config} | Device: {config['device']} | Data: {args.data} "
          f"| CV: {args.cv} | Ablation: {args.ablation}")

    if args.cv:
        run_cv(args, config, n_outer=args.n_folds, n_permutations=args.n_permutations)
    else:
        run_single(args, config)

if __name__ == '__main__':
    main()
