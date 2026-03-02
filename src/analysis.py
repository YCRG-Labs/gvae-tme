import importlib.util
import warnings

import anndata
import numpy as np
import scanpy as sc
import torch
from sklearn.metrics import (
    silhouette_score,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

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
        n_clusters = len(np.unique(hard_labels))
        centroids = np.zeros((n_clusters, z.shape[1]))
        for m in range(n_clusters):
            mask = (hard_labels == m) & (confidence >= c_thresh)
            if mask.sum() > 0:
                centroids[m] = z[mask].mean(axis=0)
            else:
                centroids[m] = z[hard_labels == m].mean(axis=0)
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
            soft_assignments[i, hard_labels[i]] = 1.0
        final_labels = hard_labels.copy()
        final_labels[uncertain_mask] = np.argmax(soft_assignments[uncertain_mask], axis=1)
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
            w = attention_weights[idx].item()
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
            # Degenerate case: only one class present or invalid inputs.
            auroc = 0.5
            auprc = 0.0
        return {
            'auroc': auroc,
            'auprc': auprc,
            'accuracy': accuracy_score(y_true, y_bin),
            'precision': precision_score(y_true, y_bin, zero_division=0),
            'recall': recall_score(y_true, y_bin, zero_division=0),
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
            # If only one class is present, permutation test is not informative;
            # fall back to a neutral AUROC.
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
    def pooling_attention_map(attentions, coords):
        all_attns = torch.stack([a.cpu() for a in attentions]).numpy()
        mean_attn = all_attns.mean(axis=0)
        return {
            'attention_per_cell': mean_attn,
            'coords': coords,
            'high_attention_mask': mean_attn > np.percentile(mean_attn, 90),
        }
