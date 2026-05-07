import warnings
import os

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
    def score_cells(adata, gene_list, use_raw=False):
        """Score cells by the mean expression of a curated marker list.

        When use_raw=True and adata.raw is present, score against the pre-HVG
        full gene set so curated markers that didn't survive HVG selection
        still contribute. Falls back to adata.var_names if raw is missing.
        """
        source = adata.raw if use_raw and adata.raw is not None else adata
        var_names = source.var_names
        available = [g for g in gene_list if g in var_names]
        if len(available) == 0:
            return np.zeros(adata.n_obs), 0
        if use_raw and adata.raw is not None:
            # adata.raw only supports gene subsetting through .to_adata() or
            # an index selection on raw directly; raw[:, available] returns a
            # Raw view whose .X is the matrix we want.
            X_sub = adata.raw[:, available].X
        else:
            X_sub = adata[:, available].X
        if hasattr(X_sub, 'toarray'):
            X_sub = X_sub.toarray()
        scores = np.asarray(X_sub).mean(axis=1).flatten()
        return scores, len(available)

    @staticmethod
    def compare_rare_vs_nonrare(adata, is_rare, signatures=None, use_raw=True):
        if signatures is None:
            signatures = ImmunosuppressiveSignatures.SIGNATURES

        if is_rare.sum() < 5 or (~is_rare).sum() < 5:
            return {'note': 'Too few rare or non-rare cells for comparison'}

        results = {}
        for sig_name, gene_list in signatures.items():
            scores, n_available = ImmunosuppressiveSignatures.score_cells(
                adata, gene_list, use_raw=use_raw)
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
            return None, {
                'note': (
                    f'Only {len(shared_genes)} shared genes after per-dataset HVG selection. '
                    f'benchmark.py:run_transfer uses naive intersection on pre-filtered HVGs. '
                    f'Use train.py --mode transfer_joint --gene-set union instead, which '
                    f'selects HVGs jointly before intersecting.'
                )
            }

        target_ad = target_adata[:, shared_genes].copy()
        source_ad = source_adata[:, shared_genes].copy()

        source_input_dim = config.get('input_dim', 50)
        n_comps = min(source_input_dim, target_ad.n_obs - 1, target_ad.n_vars - 1)

        from sklearn.decomposition import PCA as SkPCA
        X_source = source_ad.X.toarray() if hasattr(source_ad.X, 'toarray') else np.asarray(source_ad.X)
        pca_fit = SkPCA(n_components=n_comps).fit(X_source)
        X_target = target_ad.X.toarray() if hasattr(target_ad.X, 'toarray') else np.asarray(target_ad.X)
        target_pca = pca_fit.transform(X_target)

        if target_pca.shape[1] < source_input_dim:
            pad = np.zeros((target_pca.shape[0], source_input_dim - target_pca.shape[1]))
            target_ad.obsm['X_pca'] = np.hstack([target_pca, pad])
        elif target_pca.shape[1] > source_input_dim:
            target_ad.obsm['X_pca'] = target_pca[:, :source_input_dim]
        else:
            target_ad.obsm['X_pca'] = target_pca

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
            unique_rare = np.unique(rare_labels[rare_labels >= 100])
            if len(unique_rare) >= 2:
                target_rare_markers = CrossDatasetAnalyzer.marker_genes_rare_subclusters(
                    z, rare_labels, target_adata)
            else:
                target_rare_markers = {}
            target_rare_markers = {k: v for k, v in target_rare_markers.items()
                                   if k.isdigit() and int(k) >= 100}
            if target_rare_markers:
                concordance = CrossDatasetAnalyzer.jaccard_concordance(
                    source_rare_markers, target_rare_markers)
                results['marker_concordance'] = concordance
                print(f"  Transfer rare marker Jaccard: {concordance['mean_jaccard']:.3f}")

        return results


def _evaluate_spatial_clustering(z, labels, coords, ground_truth=None, k_moran=15):
    from sklearn.metrics import silhouette_score
    n_clusters = int(len(np.unique(labels)))
    out = {'n_clusters': n_clusters}

    if n_clusters > 1 and coords is not None:
        try:
            out['silhouette_spatial'] = float(silhouette_score(coords, labels))
        except Exception as e:
            out['silhouette_spatial'] = None
            out['silhouette_error'] = str(e)
    else:
        out['silhouette_spatial'] = None

    if z is not None and n_clusters > 1:
        try:
            out['silhouette_embedding'] = float(silhouette_score(z, labels))
        except Exception:
            out['silhouette_embedding'] = None

    if coords is not None and n_clusters > 1:
        from src.analysis import BiologicalValidation
        m = BiologicalValidation.morans_i(labels.astype(float), coords, k=k_moran)
        out['morans_i'] = m['morans_i']
        out['morans_i_p'] = m.get('p_value')

    if ground_truth is not None:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        gt = np.asarray(ground_truth)
        mask = ~pd.isna(gt) if hasattr(pd, 'isna') else np.ones(len(gt), bool)
        if mask.sum() > 1:
            out['ARI'] = float(adjusted_rand_score(gt[mask], labels[mask]))
            out['NMI'] = float(normalized_mutual_info_score(gt[mask], labels[mask]))

    return out


class GraphSTBaseline:

    @staticmethod
    def run(adata, n_clusters=7, device='cuda', radius=50, n_top_genes=3000,
            ground_truth_key=None):
        try:
            from GraphST import GraphST
            from GraphST.utils import clustering as graphst_clustering
        except ImportError as e:
            return {'method': 'graphst',
                    'note': f'GraphST not installed ({e}). pip install GraphST'}

        if 'spatial' not in adata.obsm:
            return {'method': 'graphst', 'note': 'adata.obsm["spatial"] missing'}

        ad = adata.copy()
        if ad.n_vars > n_top_genes and 'highly_variable' not in ad.var:
            sc.pp.highly_variable_genes(ad, n_top_genes=n_top_genes, flavor='seurat_v3')
            ad = ad[:, ad.var['highly_variable']].copy()

        print(f"  GraphST: training on {ad.n_obs} spots x {ad.n_vars} genes "
              f"(device={device})")
        model = GraphST.GraphST(ad, device=device, random_seed=42)
        ad = model.train()

        try:
            graphst_clustering(ad, n_clusters=n_clusters, method='mclust', radius=radius,
                               refinement=True)
            labels = ad.obs['domain'].astype(int).values
        except Exception as e:
            print(f"  GraphST: mclust failed ({e}); falling back to leiden")
            sc.pp.neighbors(ad, use_rep='emb')
            sc.tl.leiden(ad, resolution=0.5)
            labels = ad.obs['leiden'].astype(int).values

        z = ad.obsm.get('emb')
        coords = ad.obsm.get('spatial')
        gt = adata.obs[ground_truth_key].values if (ground_truth_key and ground_truth_key in adata.obs.columns) else None

        metrics = _evaluate_spatial_clustering(z, labels, coords, ground_truth=gt)
        print(f"  GraphST: {metrics['n_clusters']} clusters, "
              f"silhouette_spatial={metrics.get('silhouette_spatial')}, "
              f"morans_i={metrics.get('morans_i')}")

        return {'method': 'graphst', 'z': z, 'labels': labels, 'clustering': metrics}


class STAGATEBaseline:

    @staticmethod
    def run(adata, rad_cutoff=150, leiden_resolution=0.5, n_top_genes=3000,
            ground_truth_key=None):
        try:
            import STAGATE_pyG as STAGATE
        except ImportError as e:
            return {'method': 'stagate',
                    'note': f'STAGATE_pyG not installed ({e}). pip install STAGATE_pyG'}

        if 'spatial' not in adata.obsm:
            return {'method': 'stagate', 'note': 'adata.obsm["spatial"] missing'}

        ad = adata.copy()
        if ad.n_vars > n_top_genes and 'highly_variable' not in ad.var:
            sc.pp.highly_variable_genes(ad, n_top_genes=n_top_genes, flavor='seurat_v3')
            ad = ad[:, ad.var['highly_variable']].copy()

        print(f"  STAGATE: building spatial graph (rad_cutoff={rad_cutoff})")
        STAGATE.Cal_Spatial_Net(ad, rad_cutoff=rad_cutoff)
        STAGATE.Stats_Spatial_Net(ad)

        print(f"  STAGATE: training on {ad.n_obs} spots x {ad.n_vars} genes")
        ad = STAGATE.train_STAGATE(ad)

        z = ad.obsm['STAGATE']
        sc.pp.neighbors(ad, use_rep='STAGATE')
        sc.tl.leiden(ad, resolution=leiden_resolution)
        labels = ad.obs['leiden'].astype(int).values

        coords = ad.obsm.get('spatial')
        gt = adata.obs[ground_truth_key].values if (ground_truth_key and ground_truth_key in adata.obs.columns) else None

        metrics = _evaluate_spatial_clustering(z, labels, coords, ground_truth=gt)
        print(f"  STAGATE: {metrics['n_clusters']} clusters, "
              f"silhouette_spatial={metrics.get('silhouette_spatial')}, "
              f"morans_i={metrics.get('morans_i')}")

        return {'method': 'stagate', 'z': z, 'labels': labels, 'clustering': metrics}


class SpaGCNBaseline:

    @staticmethod
    def run(adata, histology_path=None, n_clusters=7, p=0.5, n_top_genes=3000,
            ground_truth_key=None, seed=42):
        try:
            import SpaGCN as spg
            import torch
        except ImportError as e:
            return {'method': 'spagcn',
                    'note': f'SpaGCN not installed ({e}). pip install SpaGCN'}

        if 'spatial' not in adata.obsm:
            return {'method': 'spagcn', 'note': 'adata.obsm["spatial"] missing'}

        ad = adata.copy()
        if ad.n_vars > n_top_genes and 'highly_variable' not in ad.var:
            sc.pp.highly_variable_genes(ad, n_top_genes=n_top_genes, flavor='seurat_v3')
            ad = ad[:, ad.var['highly_variable']].copy()

        coords = ad.obsm['spatial']
        x = coords[:, 0].astype(int)
        y = coords[:, 1].astype(int)

        img = None
        use_histology = False
        if histology_path is not None and os.path.exists(histology_path):
            try:
                import cv2
                img = cv2.imread(histology_path)
                if img is not None:
                    use_histology = True
                    print(f"  SpaGCN: loaded histology {histology_path} shape={img.shape}")
            except Exception as e:
                print(f"  SpaGCN: histology load failed ({e}); using coords-only adjacency")

        if use_histology:
            x_pix = ad.obs.get('pxl_col_in_fullres', x).astype(int).values \
                if 'pxl_col_in_fullres' in ad.obs.columns else x
            y_pix = ad.obs.get('pxl_row_in_fullres', y).astype(int).values \
                if 'pxl_row_in_fullres' in ad.obs.columns else y
            adj = spg.calculate_adj_matrix(x=x, y=y, x_pixel=x_pix, y_pixel=y_pix,
                                           image=img, beta=49, alpha=1, histology=True)
        else:
            adj = spg.calculate_adj_matrix(x=x, y=y, histology=False)

        l_search = spg.search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100)

        r_seed = seed
        try:
            r_search = spg.search_res(ad, adj, l_search, n_clusters, start=0.7,
                                      step=0.1, tol=5e-3, lr=0.05, max_epochs=20,
                                      r_seed=r_seed, t_seed=r_seed, n_seed=r_seed)
        except Exception as e:
            print(f"  SpaGCN: search_res failed ({e}); using r=1.0")
            r_search = 1.0

        clf = spg.SpaGCN()
        clf.set_l(l_search)
        torch.manual_seed(r_seed)
        np.random.seed(r_seed)
        clf.train(ad, adj, init_spa=True, init='louvain', res=r_search, tol=5e-3,
                  lr=0.05, max_epochs=200)
        y_pred, prob = clf.predict()
        labels = np.asarray(y_pred).astype(int)

        gt = adata.obs[ground_truth_key].values if (ground_truth_key and ground_truth_key in adata.obs.columns) else None
        metrics = _evaluate_spatial_clustering(z=None, labels=labels, coords=coords,
                                                ground_truth=gt)
        metrics['used_histology'] = bool(use_histology)
        metrics['l'] = float(l_search)
        metrics['res'] = float(r_search)
        print(f"  SpaGCN: {metrics['n_clusters']} clusters "
              f"(histology={use_histology}), "
              f"silhouette_spatial={metrics.get('silhouette_spatial')}, "
              f"morans_i={metrics.get('morans_i')}")

        return {'method': 'spagcn', 'z': None, 'labels': labels, 'clustering': metrics}
