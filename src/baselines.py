import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import anndata
from scipy.stats import mannwhitneyu
from sklearn.neighbors import NearestNeighbors

from src.analysis import ClusteringAnalyzer, RareCellDetector, CrossDatasetAnalyzer
from src.ablations import LogisticRegressionBaseline


class ScVIBaseline:

    @staticmethod
    def train_and_embed(adata, n_latent=32, max_epochs=200, batch_size=256):
        try:
            import scvi
        except ImportError:
            return None, {'note': 'scvi-tools not installed. pip install scvi-tools'}

        ad = adata.copy()
        if 'counts' in ad.layers:
            scvi.model.SCVI.setup_anndata(ad, layer='counts')
        else:
            scvi.model.SCVI.setup_anndata(ad)

        model = scvi.model.SCVI(ad, n_latent=n_latent)
        model.train(max_epochs=max_epochs, batch_size=min(batch_size, ad.n_obs),
                    early_stopping=True, enable_progress_bar=False)
        z = model.get_latent_representation()
        return model, {'z': z, 'n_latent': n_latent, 'n_epochs': max_epochs}

    @staticmethod
    def estimate_cell_uncertainty(model, n_samples=20):
        samples = []
        for _ in range(n_samples):
            z_sample = model.get_latent_representation(give_mean=False)
            samples.append(z_sample)
        samples = np.stack(samples, axis=0)
        variance = np.mean(np.var(samples, axis=0), axis=1)
        return variance

    @staticmethod
    def run_downstream(z, adata, patient_masks=None, y=None, model=None):
        results = {'method': 'scvi'}

        clusterer = ClusteringAnalyzer()
        best_res, best_sil = clusterer.select_resolution(z)
        labels, confidence = clusterer.cluster(z, resolution=best_res)
        eval_metrics = clusterer.evaluate(z, labels)
        eval_metrics['resolution'] = best_res
        results['clustering'] = eval_metrics
        print(f"  scVI: {eval_metrics['n_clusters']} clusters, "
              f"silhouette={eval_metrics['silhouette']:.4f}")

        n_rare = 0
        if model is not None:
            try:
                variance = ScVIBaseline.estimate_cell_uncertainty(model)
                var_z = (variance - np.mean(variance)) / (np.std(variance) + 1e-8)
                is_rare = var_z > 2.0
                n_rare = int(is_rare.sum())
            except Exception:
                n_rare = 0
        results['rare_cells'] = {'n_rare': n_rare, 'method': 'scvi_variance'}

        pred_results = {}
        if patient_masks is not None and y is not None:
            pred_results = LogisticRegressionBaseline.run(z, labels, patient_masks, y)
            pred_results['method'] = 'scvi_logreg'
        results['prediction'] = pred_results

        return results


class ScanpyBaseline:

    @staticmethod
    def run(adata, patient_masks=None, y=None, n_pcs=50, resolution=1.0):
        results = {'method': 'scanpy_standard'}

        ad = adata.copy()
        if 'X_pca' not in ad.obsm:
            n_comps = min(n_pcs, ad.n_obs - 1, ad.n_vars - 1)
            sc.pp.pca(ad, n_comps=n_comps)

        z = ad.obsm['X_pca']
        sc.pp.neighbors(ad, use_rep='X_pca', n_neighbors=15)
        sc.tl.leiden(ad, resolution=resolution)
        labels = ad.obs['leiden'].astype(int).values

        clusterer = ClusteringAnalyzer()
        eval_metrics = clusterer.evaluate(z, labels)
        eval_metrics['resolution'] = resolution
        results['clustering'] = eval_metrics
        print(f"  Scanpy: {eval_metrics['n_clusters']} clusters, "
              f"silhouette={eval_metrics['silhouette']:.4f}")

        total_cells = len(labels)
        unique_labels, counts = np.unique(labels, return_counts=True)
        rare_clusters = unique_labels[counts / total_cells < 0.01]
        n_rare = int(np.isin(labels, rare_clusters).sum())
        results['rare_cells'] = {'n_rare': n_rare, 'method': 'frequency_threshold'}

        pred_results = {}
        if patient_masks is not None and y is not None:
            pred_results = LogisticRegressionBaseline.run(z, labels, patient_masks, y)
            pred_results['method'] = 'scanpy_logreg'
        results['prediction'] = pred_results

        return results


class ImmunosuppressiveSignatures:

    TREG = ['FOXP3', 'IL2RA', 'CTLA4', 'IKZF2', 'TNFRSF18']
    M2_MACROPHAGE = ['CD163', 'MRC1', 'MSR1', 'MARCO', 'CD209']
    MDSC = ['S100A8', 'S100A9', 'S100A12', 'FCN1', 'VCAN']
    EXHAUSTED_T = ['PDCD1', 'LAG3', 'HAVCR2', 'TIGIT', 'TOX']

    SIGNATURES = {
        'Treg': TREG,
        'M2_macrophage': M2_MACROPHAGE,
        'MDSC': MDSC,
        'Exhausted_T': EXHAUSTED_T,
    }

    @staticmethod
    def score_cells(adata, gene_list):
        available = [g for g in gene_list if g in adata.var_names]
        if len(available) == 0:
            return np.zeros(adata.n_obs), 0
        X_sub = adata[:, available].X
        if hasattr(X_sub, 'toarray'):
            X_sub = X_sub.toarray()
        scores = np.asarray(X_sub).mean(axis=1).flatten()
        return scores, len(available)

    @staticmethod
    def compare_rare_vs_nonrare(adata, is_rare, signatures=None):
        if signatures is None:
            signatures = ImmunosuppressiveSignatures.SIGNATURES

        if is_rare.sum() < 5 or (~is_rare).sum() < 5:
            return {'note': 'Too few rare or non-rare cells for comparison'}

        results = {}
        for sig_name, gene_list in signatures.items():
            scores, n_available = ImmunosuppressiveSignatures.score_cells(adata, gene_list)
            if n_available == 0:
                results[sig_name] = {
                    'note': 'No signature genes found in HVG set',
                    'n_available': 0,
                    'n_total': len(gene_list),
                }
                continue

            rare_scores = scores[is_rare]
            nonrare_scores = scores[~is_rare]
            try:
                stat, p_value = mannwhitneyu(rare_scores, nonrare_scores, alternative='two-sided')
            except ValueError:
                stat, p_value = 0.0, 1.0

            rare_mean = float(np.mean(rare_scores))
            nonrare_mean = float(np.mean(nonrare_scores))
            pooled_std = float(np.std(scores)) + 1e-8
            effect_size = (rare_mean - nonrare_mean) / pooled_std

            results[sig_name] = {
                'rare_mean': rare_mean,
                'nonrare_mean': nonrare_mean,
                'effect_size': float(effect_size),
                'p_value': float(p_value),
                'u_statistic': float(stat),
                'n_available': n_available,
                'n_total': len(gene_list),
            }
        return results


class CrossDatasetTransfer:

    @staticmethod
    def transfer_embeddings(model, source_adata, target_adata, target_info, config):
        import torch
        from src.data_utils import prepare_graph_data

        source_genes = set(source_adata.var_names)
        target_genes = set(target_adata.var_names)
        shared_genes = sorted(source_genes & target_genes)
        print(f"  Transfer: {len(shared_genes)} shared genes "
              f"(source={len(source_genes)}, target={len(target_genes)})")

        if len(shared_genes) < 500:
            return None, {'note': f'Only {len(shared_genes)} shared genes, need >= 500'}

        target_ad = target_adata[:, shared_genes].copy()
        source_ad = source_adata[:, shared_genes].copy()

        n_comps = min(50, target_ad.n_obs - 1, target_ad.n_vars - 1)
        sc.pp.pca(target_ad, n_comps=n_comps)

        source_input_dim = config.get('input_dim', 50)
        target_pca = target_ad.obsm['X_pca']
        if target_pca.shape[1] < source_input_dim:
            pad = np.zeros((target_pca.shape[0], source_input_dim - target_pca.shape[1]))
            target_ad.obsm['X_pca'] = np.hstack([target_pca, pad])
        elif target_pca.shape[1] > source_input_dim:
            target_ad.obsm['X_pca'] = target_pca[:, :source_input_dim]

        has_spatial = target_info.get('has_spatial', False)
        r_spatial = target_info.get('r_spatial', 82.5)
        data = prepare_graph_data(
            target_ad, spatial_key='spatial' if has_spatial else None,
            k_mol=config.get('k_mol', 15),
            r_spatial=r_spatial,
            has_spatial=has_spatial)

        device = config.get('device', 'cpu')
        model.eval()
        data_eval = data.to(device)
        with torch.no_grad():
            outputs = model(data_eval)
            z = outputs['z'].cpu().numpy()
            mu = outputs['mu'].cpu().numpy()
            logvar = outputs['logvar'].cpu().numpy()

        return {
            'z': z, 'mu': mu, 'logvar': logvar,
            'n_shared_genes': len(shared_genes),
            'target_cells': target_ad.n_obs,
        }, target_ad

    @staticmethod
    def evaluate_transfer(z, mu, logvar, target_adata, source_rare_markers=None):
        results = {'method': 'cross_dataset_transfer'}

        clusterer = ClusteringAnalyzer()
        best_res, _ = clusterer.select_resolution(z)
        labels, confidence = clusterer.cluster(z, logvar=logvar, resolution=best_res)
        eval_metrics = clusterer.evaluate(z, labels)
        results['clustering'] = eval_metrics

        detector = RareCellDetector(threshold=2.0)
        scores, is_rare = detector.detect(mu, logvar)
        rare_labels = np.full(len(z), -1, dtype=int)
        if is_rare.sum() > 10:
            rare_labels = detector.subcluster(z, is_rare, resolution=2.0)
        results['rare_cells'] = {'n_rare': int(is_rare.sum())}

        if source_rare_markers and is_rare.sum() > 10:
            target_markers = CrossDatasetAnalyzer.marker_genes(z, rare_labels, target_adata)
            target_rare_markers = {k: v for k, v in target_markers.items()
                                   if k.isdigit() and int(k) >= 100}
            if target_rare_markers:
                concordance = CrossDatasetAnalyzer.jaccard_concordance(
                    source_rare_markers, target_rare_markers)
                results['marker_concordance'] = concordance
                print(f"  Transfer rare marker Jaccard: {concordance['mean_jaccard']:.3f}")

        return results
