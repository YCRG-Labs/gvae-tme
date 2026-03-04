import importlib.util
import warnings
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.stats import chi2
from sklearn.metrics import (
    silhouette_score,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    adjusted_rand_score,
)
from sklearn.neighbors import NearestNeighbors

_HAS_LEIDEN = importlib.util.find_spec("leidenalg") is not None

class RareCellDetector:
    def __init__(self, threshold=2.0):
        self.threshold = threshold

    @staticmethod
    def compute_kl(mu, logvar):
        sigma2 = np.exp(logvar)
        kl = 0.5 * np.sum(mu ** 2 + sigma2 - logvar - 1.0, axis=1)
        return kl

    def detect(self, mu, logvar, eta=None):
        if eta is None:
            eta = self.threshold
        kl = self.compute_kl(mu, logvar)
        kl_mean = np.mean(kl)
        kl_std = np.std(kl) + 1e-8
        scores = (kl - kl_mean) / kl_std
        is_rare = scores > eta
        return scores, is_rare

    @staticmethod
    def subcluster(z, is_rare, resolution=2.0):
        z_rare = z[is_rare]
        if z_rare.shape[0] < 5:
            labels = np.full(len(z), -1, dtype=int)
            labels[is_rare] = 100
            return labels
        if not _HAS_LEIDEN:
            warnings.warn(
                "leidenalg is not installed; returning a single rare-cell cluster "
                "instead of performing Leiden subclustering.",
                RuntimeWarning,
            )
            labels = np.full(len(z), -1, dtype=int)
            labels[is_rare] = 100
            return labels
        adata = anndata.AnnData(X=z_rare)
        adata.obsm['X_latent'] = z_rare
        n_neighbors = min(15, z_rare.shape[0] - 1)
        sc.pp.neighbors(adata, use_rep='X_latent', n_neighbors=n_neighbors)
        sc.tl.leiden(adata, resolution=resolution)
        labels = np.full(len(z), -1, dtype=int)
        labels[is_rare] = adata.obs['leiden'].astype(int).values + 100
        return labels

class ClusteringAnalyzer:
    def __init__(self, resolutions=(0.5, 1.0, 1.5, 2.0)):
        self.resolutions = resolutions

    @staticmethod
    def compute_confidence(logvar):
        sigma2_mean = np.mean(np.exp(logvar), axis=1)
        return np.exp(-sigma2_mean)

    def cluster(self, z, adjacency=None, logvar=None, resolution=1.0):
        if _HAS_LEIDEN:
            adata = anndata.AnnData(X=z)
            adata.obsm['X_latent'] = z
            if adjacency is not None:
                import scipy.sparse as sp
                if not sp.issparse(adjacency):
                    adjacency = sp.csr_matrix(adjacency)
                adata.obsp['connectivities'] = adjacency
                adata.uns['neighbors'] = {
                    'connectivities_key': 'connectivities',
                    'params': {'method': 'custom'},
                }
            else:
                sc.pp.neighbors(adata, use_rep='X_latent', n_neighbors=15)
            sc.tl.leiden(adata, resolution=resolution)
            hard_labels = adata.obs['leiden'].astype(int).values
        else:
            from sklearn.cluster import KMeans

            if len(z) < 2:
                hard_labels = np.zeros(len(z), dtype=int)
            else:
                n_clusters = max(2, int(resolution * 5))
                n_clusters = min(n_clusters, len(z))
                if n_clusters <= 1:
                    hard_labels = np.zeros(len(z), dtype=int)
                else:
                    warnings.warn(
                        "leidenalg is not installed; using KMeans clustering as a "
                        "fallback instead of Leiden.",
                        RuntimeWarning,
                    )
                    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
                    hard_labels = kmeans.fit_predict(z)
        if logvar is None:
            return hard_labels, np.ones(len(z))
        confidence = self.compute_confidence(logvar)
        c_thresh = np.percentile(confidence, 20)
        unique_labels = np.unique(hard_labels)
        n_clusters = len(unique_labels)
        label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        mapped_labels = np.array([label_to_idx[l] for l in hard_labels])
        centroids = np.zeros((n_clusters, z.shape[1]))
        for m in range(n_clusters):
            mask = (mapped_labels == m) & (confidence >= c_thresh)
            if mask.sum() > 0:
                centroids[m] = z[mask].mean(axis=0)
            else:
                centroids[m] = z[mapped_labels == m].mean(axis=0)
        soft_assignments = np.zeros((len(z), n_clusters))
        uncertain_mask = confidence < c_thresh
        for i in np.where(uncertain_mask)[0]:
            mu_i = z[i]
            sigma2_i = np.mean(np.exp(logvar[i]))
            dists = np.sum((mu_i - centroids) ** 2, axis=1)
            logits = -dists / (2.0 * sigma2_i + 1e-8)
            logits -= logits.max()
            exp_logits = np.exp(logits)
            soft_assignments[i] = exp_logits / (exp_logits.sum() + 1e-8)
        for i in np.where(~uncertain_mask)[0]:
            soft_assignments[i, mapped_labels[i]] = 1.0
        final_mapped = mapped_labels.copy()
        final_mapped[uncertain_mask] = np.argmax(soft_assignments[uncertain_mask], axis=1)
        idx_to_label = {i: l for l, i in label_to_idx.items()}
        final_labels = np.array([idx_to_label[m] for m in final_mapped])
        return final_labels, confidence

    def select_resolution(self, z, adjacency=None, logvar=None):
        best_res = self.resolutions[0]
        best_sil = -1.0
        for res in self.resolutions:
            labels, _ = self.cluster(z, adjacency, logvar, resolution=res)
            n_unique = len(np.unique(labels))
            if n_unique < 2 or n_unique >= len(z):
                continue
            sil = silhouette_score(z, labels, sample_size=min(5000, len(z)))
            if sil > best_sil:
                best_sil = sil
                best_res = res
        return best_res, best_sil

    @staticmethod
    def evaluate(z, labels):
        n_unique = len(np.unique(labels))
        if n_unique > 1 and n_unique < len(z):
            sil = silhouette_score(z, labels, sample_size=min(5000, len(z)))
        else:
            sil = 0.0
        return {'n_clusters': n_unique, 'silhouette': sil}

class AttentionAnalyzer:
    @staticmethod
    def selectivity(attention_weights, edge_index, n_nodes):
        neighbor_attns = {}
        for idx in range(edge_index.size(1)):
            tgt = edge_index[1, idx].item()
            w = attention_weights[idx]
            w = w.mean().item() if w.dim() > 0 else w.item()
            if tgt not in neighbor_attns:
                neighbor_attns[tgt] = []
            neighbor_attns[tgt].append(w)
        selectivity = np.zeros(n_nodes)
        for node, attns in neighbor_attns.items():
            attns = np.array(attns)
            attns = attns / (attns.sum() + 1e-10)
            uniform = 1.0 / len(attns)
            kl = np.sum(attns * np.log(attns / (uniform + 1e-10) + 1e-10))
            selectivity[node] = kl
        return selectivity

    @staticmethod
    def interaction_network(attention_weights, edge_index, cell_types, n_nodes, percentile=75):
        sel = AttentionAnalyzer.selectivity(attention_weights, edge_index, n_nodes)
        sel_thresh = np.percentile(sel, percentile)
        interactions = {}
        for idx in range(edge_index.size(1)):
            src = edge_index[0, idx].item()
            tgt = edge_index[1, idx].item()
            if sel[tgt] < sel_thresh:
                continue
            src_type = cell_types[src]
            tgt_type = cell_types[tgt]
            key = (src_type, tgt_type)
            if key not in interactions:
                interactions[key] = 0
            interactions[key] += 1
        return interactions

class PredictionAnalyzer:
    @staticmethod
    def compute_metrics(y_true, y_pred):
        y_bin = (y_pred > 0.5).astype(int)
        try:
            auroc = roc_auc_score(y_true, y_pred)
            auprc = average_precision_score(y_true, y_pred)
        except ValueError:
            auroc = 0.5
            auprc = 0.0
        tn = int(((y_bin == 0) & (y_true == 0)).sum())
        fp = int(((y_bin == 1) & (y_true == 0)).sum())
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        return {
            'auroc': auroc,
            'auprc': auprc,
            'accuracy': accuracy_score(y_true, y_bin),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_bin),
            'precision': precision_score(y_true, y_bin, zero_division=0),
            'recall': recall_score(y_true, y_bin, zero_division=0),
            'specificity': specificity,
            'f1': f1_score(y_true, y_bin, zero_division=0),
        }

    @staticmethod
    def permutation_test(model, data, n_permutations=1000, device='cpu'):
        model.eval()
        data = data.to(device)
        with torch.no_grad():
            outputs = model(data)
            y_true = data.y.cpu().numpy()
            y_pred = outputs['y_pred'].cpu().numpy()
        try:
            actual_auroc = roc_auc_score(y_true, y_pred)
        except ValueError:
            actual_auroc = 0.5
        null_aurocs = np.zeros(n_permutations)
        for i in range(n_permutations):
            y_perm = np.random.permutation(y_true)
            try:
                null_aurocs[i] = roc_auc_score(y_perm, y_pred)
            except ValueError:
                null_aurocs[i] = 0.5
        p_value = float(np.mean(null_aurocs >= actual_auroc))
        return {
            'actual_auroc': actual_auroc,
            'p_value': p_value,
            'null_mean': float(np.mean(null_aurocs)),
            'null_std': float(np.std(null_aurocs)),
        }

    @staticmethod
    def bootstrap_ci(y_true, y_pred, n_bootstrap=1000, alpha=0.05, seed=42):
        rng = np.random.RandomState(seed)
        n = len(y_true)
        aurocs = np.zeros(n_bootstrap)
        auprcs = np.zeros(n_bootstrap)
        for i in range(n_bootstrap):
            idx = rng.choice(n, n, replace=True)
            y_t = y_true[idx]
            y_p = y_pred[idx]
            if len(np.unique(y_t)) < 2:
                aurocs[i] = 0.5
                auprcs[i] = 0.0
                continue
            try:
                aurocs[i] = roc_auc_score(y_t, y_p)
                auprcs[i] = average_precision_score(y_t, y_p)
            except ValueError:
                aurocs[i] = 0.5
                auprcs[i] = 0.0
        lo = alpha / 2
        hi = 1.0 - alpha / 2
        return {
            'auroc_ci': (float(np.percentile(aurocs, lo * 100)),
                         float(np.percentile(aurocs, hi * 100))),
            'auprc_ci': (float(np.percentile(auprcs, lo * 100)),
                         float(np.percentile(auprcs, hi * 100))),
            'auroc_mean': float(np.mean(aurocs)),
            'auprc_mean': float(np.mean(auprcs)),
        }

    @staticmethod
    def pooling_attention_map(attentions, coords):
        all_attns = torch.stack([a.cpu() for a in attentions]).numpy()
        mean_attn = all_attns.mean(axis=0)
        return {
            'attention_per_cell': mean_attn,
            'coords': coords,
            'high_attention_mask': mean_attn > np.percentile(mean_attn, 90),
        }



class BatchMixingAnalyzer:

    @staticmethod
    def kbet(z, batch_labels, k=50, alpha=0.05, subsample=5000):
        n = len(z)
        k = min(k, n - 1)
        batch_labels = np.asarray(batch_labels)
        unique_batches = np.unique(batch_labels)
        n_batches = len(unique_batches)
        if n_batches < 2:
            return {'rejection_rate': 0.0, 'mean_p_value': 1.0, 'n_batches': n_batches}

        batch_map = {b: i for i, b in enumerate(unique_batches)}
        batch_idx = np.array([batch_map[b] for b in batch_labels])
        global_freq = np.bincount(batch_idx, minlength=n_batches) / n

        if n > subsample:
            rng = np.random.RandomState(42)
            idx = rng.choice(n, subsample, replace=False)
        else:
            idx = np.arange(n)

        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto')
        nbrs.fit(z)
        _, neighbors = nbrs.kneighbors(z[idx])
        neighbors = neighbors[:, 1:]  # exclude self

        p_values = np.zeros(len(idx))
        for i, row in enumerate(neighbors):
            observed = np.bincount(batch_idx[row], minlength=n_batches)
            expected = global_freq * k
            mask = expected > 0
            chi2_stat = np.sum((observed[mask] - expected[mask]) ** 2 / expected[mask])
            df = mask.sum() - 1
            if df > 0:
                p_values[i] = 1.0 - chi2.cdf(chi2_stat, df)
            else:
                p_values[i] = 1.0

        rejection_rate = float(np.mean(p_values < alpha))
        return {
            'rejection_rate': rejection_rate,
            'mean_p_value': float(np.mean(p_values)),
            'n_batches': n_batches,
        }

    @staticmethod
    def batch_entropy(z, batch_labels, k=50, subsample=5000):
        n = len(z)
        k = min(k, n - 1)
        batch_labels = np.asarray(batch_labels)
        unique_batches = np.unique(batch_labels)
        n_batches = len(unique_batches)
        if n_batches < 2:
            return {'batch_entropy': 1.0, 'n_batches': n_batches}

        batch_map = {b: i for i, b in enumerate(unique_batches)}
        batch_idx = np.array([batch_map[b] for b in batch_labels])

        if n > subsample:
            rng = np.random.RandomState(42)
            idx = rng.choice(n, subsample, replace=False)
        else:
            idx = np.arange(n)

        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto')
        nbrs.fit(z)
        _, neighbors = nbrs.kneighbors(z[idx])
        neighbors = neighbors[:, 1:]

        max_entropy = np.log(n_batches)
        entropies = np.zeros(len(idx))
        for i, row in enumerate(neighbors):
            counts = np.bincount(batch_idx[row], minlength=n_batches)
            p = counts / counts.sum()
            p = p[p > 0]
            entropies[i] = -np.sum(p * np.log(p)) / max_entropy

        return {'batch_entropy': float(np.mean(entropies)), 'n_batches': n_batches}



class CrossDatasetAnalyzer:

    @staticmethod
    def marker_genes(z, labels, adata, n_markers=50):
        ad = adata.copy()
        ad.obs['gvae_cluster'] = pd.Categorical(labels.astype(str))
        ad.obsm['X_latent'] = z
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return {str(labels[0]): list(adata.var_names[:n_markers])}
        sc.tl.rank_genes_groups(ad, groupby='gvae_cluster', method='wilcoxon',
                                n_genes=n_markers, use_raw=False)
        markers = {}
        for cl in unique_labels:
            names = ad.uns['rank_genes_groups']['names'][str(cl)]
            pvals = ad.uns['rank_genes_groups']['pvals_adj'][str(cl)]
            logfcs = ad.uns['rank_genes_groups']['logfoldchanges'][str(cl)]
            filtered = []
            for g, p, lfc in zip(names, pvals, logfcs):
                if p < 0.01 and abs(lfc) > 1.0:
                    filtered.append(g)
                if len(filtered) >= n_markers:
                    break
            if len(filtered) < 5:
                filtered = list(names[:n_markers])
            markers[str(cl)] = filtered
        return markers

    @staticmethod
    def marker_genes_rare_subclusters(z, rare_labels, adata, n_markers=50):
        rare_mask = rare_labels >= 100
        if rare_mask.sum() < 10:
            return {}
        unique_rare = np.unique(rare_labels[rare_mask])
        if len(unique_rare) < 2:
            return {}

        ad = adata[rare_mask].copy()
        ad.obs['rare_subcluster'] = pd.Categorical(rare_labels[rare_mask].astype(str))
        ad.obsm['X_latent'] = z[rare_mask]

        sc.tl.rank_genes_groups(ad, groupby='rare_subcluster', method='wilcoxon',
                                n_genes=n_markers, use_raw=False)
        markers = {}
        for cl in unique_rare:
            names = ad.uns['rank_genes_groups']['names'][str(cl)]
            pvals = ad.uns['rank_genes_groups']['pvals_adj'][str(cl)]
            logfcs = ad.uns['rank_genes_groups']['logfoldchanges'][str(cl)]
            filtered = []
            for g, p, lfc in zip(names, pvals, logfcs):
                if p < 0.01 and abs(lfc) > 1.0:
                    filtered.append(g)
                if len(filtered) >= n_markers:
                    break
            if len(filtered) < 5:
                filtered = list(names[:n_markers])
            markers[str(cl)] = filtered
        return markers

    @staticmethod
    def jaccard_concordance(markers_a, markers_b):
        pairs = {}
        for cl_a, genes_a in markers_a.items():
            set_a = set(genes_a)
            best_j, best_cl = 0.0, None
            for cl_b, genes_b in markers_b.items():
                set_b = set(genes_b)
                inter = len(set_a & set_b)
                union = len(set_a | set_b)
                j = inter / union if union > 0 else 0.0
                if j > best_j:
                    best_j = j
                    best_cl = cl_b
            pairs[cl_a] = {'matched_cluster': best_cl, 'jaccard': best_j}

        mean_jaccard = float(np.mean([v['jaccard'] for v in pairs.values()])) if pairs else 0.0
        return {'per_cluster': pairs, 'mean_jaccard': mean_jaccard}


class BiologicalValidation:

    @staticmethod
    def morans_i(values, coords, k=15):
        n = len(values)
        if n < k + 1:
            return {'morans_i': 0.0, 'expected_i': -1.0 / max(n - 1, 1), 'p_value': 1.0}

        nbrs = NearestNeighbors(n_neighbors=min(k + 1, n), algorithm='auto')
        nbrs.fit(coords)
        _, indices = nbrs.kneighbors(coords)
        indices = indices[:, 1:]

        z = values - values.mean()
        z_var = np.sum(z ** 2)
        if z_var < 1e-10:
            return {'morans_i': 0.0, 'expected_i': -1.0 / (n - 1), 'p_value': 1.0}

        W = 0.0
        lag_sum = 0.0
        for i in range(n):
            for j_idx in indices[i]:
                W += 1.0
                lag_sum += z[i] * z[j_idx]

        I = (n / W) * (lag_sum / z_var)
        expected_I = -1.0 / (n - 1)

        import scipy.sparse as sp_sparse
        rows = np.repeat(np.arange(n), indices.shape[1])
        cols = indices.ravel().astype(int)
        adj_sp = sp_sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        n_mutual_directed = int(adj_sp.multiply(adj_sp.T).nnz)
        in_degree = np.array(adj_sp.sum(axis=0)).flatten()
        out_degree = np.array(adj_sp.sum(axis=1)).flatten()
        S2 = np.sum((out_degree + in_degree) ** 2)
        S0 = W
        S1 = 0.5 * (n_mutual_directed + S0)
        n2 = n * n
        EI2 = (n2 * S1 - n * S2 + 3 * S0 * S0) / (S0 * S0 * (n2 - 1))
        var_I = EI2 - expected_I ** 2
        if var_I <= 0:
            return {'morans_i': float(I), 'expected_i': float(expected_I), 'p_value': 1.0}

        from scipy.stats import norm
        z_score = (I - expected_I) / np.sqrt(var_I)
        p_value = 2.0 * (1.0 - norm.cdf(abs(z_score)))
        return {
            'morans_i': float(I),
            'expected_i': float(expected_I),
            'z_score': float(z_score),
            'p_value': float(p_value),
        }

    @staticmethod
    def ari_stability(z, n_runs=20, resolution=1.0):
        if not _HAS_LEIDEN:
            return {'mean_ari': 0.0, 'std_ari': 0.0, 'note': 'leidenalg not installed'}

        all_labels = []
        for run in range(n_runs):
            ad = anndata.AnnData(X=z)
            ad.obsm['X_latent'] = z
            n_neighbors = min(15, z.shape[0] - 1)
            sc.pp.neighbors(ad, use_rep='X_latent', n_neighbors=n_neighbors,
                            random_state=run)
            sc.tl.leiden(ad, resolution=resolution, random_state=run)
            all_labels.append(ad.obs['leiden'].astype(int).values)

        aris = []
        for i in range(n_runs):
            for j in range(i + 1, n_runs):
                aris.append(adjusted_rand_score(all_labels[i], all_labels[j]))
        return {
            'mean_ari': float(np.mean(aris)),
            'std_ari': float(np.std(aris)),
            'n_runs': n_runs,
        }

    @staticmethod
    def gsea_enrichment(marker_genes_dict, gene_sets='MSigDB_Hallmark_2020', organism='Human'):
        try:
            import gseapy as gp
        except ImportError:
            return {'note': 'gseapy not installed. Install via: pip install gseapy'}

        results = {}
        for cluster_id, genes in marker_genes_dict.items():
            if not genes:
                continue
            try:
                enr = gp.enrichr(
                    gene_list=genes,
                    gene_sets=gene_sets,
                    organism=organism,
                    outdir=None,
                    no_plot=True,
                )
                top = enr.results.head(10)[['Term', 'Adjusted P-value', 'Combined Score']]
                results[cluster_id] = top.to_dict('records')
            except Exception as e:
                results[cluster_id] = {'error': str(e)}
        return results

    @staticmethod
    def gsea_rare_subclusters(rare_marker_genes, organism='Human'):
        if not rare_marker_genes:
            return {'note': 'No rare subcluster markers provided'}
        try:
            import gseapy as gp
        except ImportError:
            return {'note': 'gseapy not installed. Install via: pip install gseapy'}

        gene_set_libs = ['MSigDB_Hallmark_2020', 'GO_Biological_Process_2023',
                         'KEGG_2021_Human']
        results = {}
        for cluster_id, genes in rare_marker_genes.items():
            if not genes:
                continue
            cluster_results = {}
            for lib in gene_set_libs:
                try:
                    enr = gp.enrichr(
                        gene_list=genes,
                        gene_sets=lib,
                        organism=organism,
                        outdir=None,
                        no_plot=True,
                    )
                    top = enr.results.head(10)[['Term', 'Adjusted P-value', 'Combined Score']]
                    cluster_results[lib] = top.to_dict('records')
                except Exception as e:
                    cluster_results[lib] = {'error': str(e)}
            results[cluster_id] = cluster_results
        return results

    @staticmethod
    def spatial_permutation_test(gate_values, coords, n_permutations=1000, k=15, seed=42):
        observed = BiologicalValidation.morans_i(gate_values, coords, k=k)
        rng = np.random.RandomState(seed)
        null_Is = np.zeros(n_permutations)
        for i in range(n_permutations):
            perm_vals = rng.permutation(gate_values)
            null_result = BiologicalValidation.morans_i(perm_vals, coords, k=k)
            null_Is[i] = null_result['morans_i']
        p_value = float(np.mean(null_Is >= observed['morans_i']))
        return {
            'observed_I': observed['morans_i'],
            'null_mean': float(np.mean(null_Is)),
            'null_std': float(np.std(null_Is)),
            'p_value': p_value,
            'n_permutations': n_permutations,
        }



class ClinicalAssociationTest:

    @staticmethod
    def compute_rare_fractions(rare_labels, patient_masks):
        rare_clusters = sorted(set(rare_labels[rare_labels >= 100]))
        if len(rare_clusters) == 0:
            return pd.DataFrame()

        rows = []
        for mask in patient_masks:
            mask_np = mask.cpu().numpy() if hasattr(mask, 'cpu') else np.asarray(mask)
            n_cells = mask_np.sum()
            patient_labels = rare_labels[mask_np]
            fracs = {}
            for rc in rare_clusters:
                fracs[f'rare_{rc}'] = (patient_labels == rc).sum() / max(n_cells, 1)
            rows.append(fracs)
        return pd.DataFrame(rows)

    @staticmethod
    def test_association(fractions_df, response, therapy=None, alpha=0.05):
        if fractions_df.empty:
            return {'features': [], 'note': 'No rare subpopulations detected'}

        try:
            import statsmodels.api as sm
        except ImportError:
            warnings.warn("statsmodels not installed; skipping clinical association test",
                          RuntimeWarning)
            return {'features': [], 'note': 'statsmodels not installed'}

        y = np.asarray(response, dtype=float)
        X = fractions_df.values.copy()
        feature_names = list(fractions_df.columns)

        if therapy is not None:
            therapy = np.asarray(therapy)
            unique_therapies = np.unique(therapy)
            if len(unique_therapies) > 1:
                for t in unique_therapies[1:]:  # drop first as reference
                    col = (therapy == t).astype(float)
                    X = np.column_stack([X, col])
                    feature_names.append(f'therapy_{t}')

        X = sm.add_constant(X, has_constant='add')

        try:
            model = sm.Logit(y, X)
            result = model.fit(disp=0, maxiter=500)
        except Exception as e:
            return {'features': [], 'note': f'Logit fit failed: {e}'}

        features = []
        pvals = []
        for i, name in enumerate(feature_names):
            idx = i + 1  # skip constant
            features.append({
                'name': name,
                'coef': float(result.params[idx]),
                'se': float(result.bse[idx]),
                'z': float(result.tvalues[idx]),
                'p_value': float(result.pvalues[idx]),
            })
            pvals.append(result.pvalues[idx])

        if len(pvals) > 0:
            from statsmodels.stats.multitest import multipletests
            reject, qvals, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')
            for feat, q, rej in zip(features, qvals, reject):
                feat['q_value'] = float(q)
                feat['significant'] = bool(rej)

        significant = [f for f in features if f.get('significant', False)]
        return {
            'features': features,
            'n_significant': len(significant),
            'n_tested': len(features),
        }

    @staticmethod
    def summary(results):
        if not results.get('features'):
            print(f"  Clinical association: {results.get('note', 'no results')}")
            return
        print(f"  Clinical association: {results['n_significant']}/{results['n_tested']} "
              f"features significant (FDR < 0.05)")
        for f in results['features']:
            sig_marker = '*' if f.get('significant', False) else ' '
            print(f"    {sig_marker} {f['name']}: coef={f['coef']:.3f}, "
                  f"p={f['p_value']:.4f}, q={f.get('q_value', 'N/A')}")


class Cell2LocationWrapper:

    @staticmethod
    def deconvolve(adata_vis, adata_sc, max_epochs=30000, batch_size=2500):
        try:
            import cell2location
            from cell2location.models import RegressionModel
        except ImportError:
            return {'note': 'cell2location not installed. pip install cell2location'}

        RegressionModel.setup_anndata(adata_sc, labels_key='cell_type')
        ref_model = RegressionModel(adata_sc)
        ref_model.train(max_epochs=max_epochs, batch_size=batch_size, use_gpu=True)
        adata_sc = ref_model.export_posterior(adata_sc)
        inf_aver = adata_sc.varm['means_per_cluster_mu_fg'].copy()

        cell2location.models.Cell2location.setup_anndata(adata_vis)
        mod = cell2location.models.Cell2location(
            adata_vis, cell_state_df=inf_aver,
            N_cells_per_location=30, detection_alpha=20)
        mod.train(max_epochs=max_epochs, batch_size=batch_size, use_gpu=True)
        adata_vis = mod.export_posterior(adata_vis)

        cell_abundances = adata_vis.obsm['q05_cell_abundance_w_sf'].copy()
        return {
            'cell_abundances': cell_abundances,
            'cell_types': list(inf_aver.columns),
            'n_spots': adata_vis.n_obs,
        }


class CellTypeAnnotator:

    @staticmethod
    def annotate(adata, model_name='Immune_All_Low.pkl'):
        try:
            import celltypist
        except ImportError:
            return {'note': 'celltypist not installed. pip install celltypist'}

        celltypist.models.download_models(force_update=False, model=model_name)
        model = celltypist.models.Model.load(model=model_name)
        predictions = celltypist.annotate(adata, model=model, majority_voting=True)
        adata.obs['cell_type'] = predictions.predicted_labels['majority_voting']

        type_counts = adata.obs['cell_type'].value_counts().to_dict()
        return {
            'cell_types': type_counts,
            'n_types': len(type_counts),
            'model': model_name,
        }

    @staticmethod
    def annotate_with_reference(adata, ref_adata, ref_label_key='cell_type'):
        try:
            import celltypist
        except ImportError:
            return {'note': 'celltypist not installed'}

        new_model = celltypist.train(ref_adata, labels=ref_label_key, use_SGD=True)
        predictions = celltypist.annotate(adata, model=new_model, majority_voting=True)
        adata.obs['cell_type'] = predictions.predicted_labels['majority_voting']
        return {
            'cell_types': adata.obs['cell_type'].value_counts().to_dict(),
            'n_types': adata.obs['cell_type'].nunique(),
        }


class LigandReceptorAnalyzer:

    IMMUNE_LR_PAIRS = [
        ('CCL5', 'CCR5'), ('CXCL9', 'CXCR3'), ('CXCL10', 'CXCR3'),
        ('CXCL13', 'CXCR5'), ('CCL19', 'CCR7'), ('CCL21', 'CCR7'),
        ('CD274', 'PDCD1'), ('CD80', 'CD28'), ('CD80', 'CTLA4'),
        ('CD86', 'CD28'), ('CD86', 'CTLA4'), ('LGALS9', 'HAVCR2'),
        ('TNFSF9', 'TNFRSF9'), ('FAS', 'FASLG'), ('IFNG', 'IFNGR1'),
        ('TNF', 'TNFRSF1A'), ('IL2', 'IL2RA'), ('IL10', 'IL10RA'),
        ('TGFB1', 'TGFBR1'), ('HLA-A', 'KIR2DL1'), ('HLA-B', 'KIR3DL1'),
        ('VEGFA', 'FLT1'), ('VEGFA', 'KDR'), ('SPP1', 'CD44'),
    ]

    @staticmethod
    def score_interactions(adata, labels, lr_pairs=None):
        if lr_pairs is None:
            lr_pairs = LigandReceptorAnalyzer.IMMUNE_LR_PAIRS

        gene_set = set(adata.var_names)
        valid_pairs = [(l, r) for l, r in lr_pairs if l in gene_set and r in gene_set]
        if not valid_pairs:
            return {'note': 'No L-R pairs found in gene set', 'n_valid': 0}

        unique_clusters = sorted(np.unique(labels))
        cluster_means = {}
        for cl in unique_clusters:
            mask = labels == cl
            if hasattr(adata.X, 'toarray'):
                cluster_means[cl] = np.asarray(adata[mask].X.toarray()).mean(axis=0)
            else:
                cluster_means[cl] = np.asarray(adata[mask].X).mean(axis=0)

        var_to_idx = {g: i for i, g in enumerate(adata.var_names)}
        interactions = []
        for ligand, receptor in valid_pairs:
            l_idx = var_to_idx[ligand]
            r_idx = var_to_idx[receptor]
            for cl_src in unique_clusters:
                for cl_tgt in unique_clusters:
                    l_expr = cluster_means[cl_src][l_idx]
                    r_expr = cluster_means[cl_tgt][r_idx]
                    score = float(l_expr * r_expr)
                    if score > 0:
                        interactions.append({
                            'ligand': ligand,
                            'receptor': receptor,
                            'source': int(cl_src),
                            'target': int(cl_tgt),
                            'score': score,
                        })

        interactions.sort(key=lambda x: x['score'], reverse=True)
        return {
            'interactions': interactions[:100],
            'n_valid_pairs': len(valid_pairs),
            'n_clusters': len(unique_clusters),
            'n_interactions': len(interactions),
        }

    @staticmethod
    def cellphonedb_format(adata, labels, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        meta = pd.DataFrame({
            'Cell': adata.obs_names,
            'cell_type': [str(l) for l in labels],
        })
        meta.to_csv(output_dir / 'meta.txt', sep='\t', index=False)

        if hasattr(adata.X, 'toarray'):
            counts = pd.DataFrame(
                adata.X.toarray().T,
                index=adata.var_names,
                columns=adata.obs_names)
        else:
            counts = pd.DataFrame(
                adata.X.T,
                index=adata.var_names,
                columns=adata.obs_names)
        counts.to_csv(output_dir / 'counts.txt', sep='\t')

        return {
            'meta_path': str(output_dir / 'meta.txt'),
            'counts_path': str(output_dir / 'counts.txt'),
        }
