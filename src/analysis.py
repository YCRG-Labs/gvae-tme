import importlib.util
import warnings

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
    precision_score,
    recall_score,
    f1_score,
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


# ── Feature 3: Batch Mixing Metrics ────────────────────────────────────────

class BatchMixingAnalyzer:
    """kBET and batch entropy for evaluating latent space batch mixing."""

    @staticmethod
    def kbet(z, batch_labels, k=50, alpha=0.05, subsample=5000):
        """k-nearest-neighbor Batch Effect Test.

        For each cell, tests whether the batch composition of its k-NN
        matches the global batch proportions via chi-squared test.

        Returns:
            dict with 'rejection_rate' (lower = better mixing) and 'mean_p_value'.
        """
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

        # Subsample for speed
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
            # Chi-squared statistic, only for bins with expected > 0
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
        """Normalized Shannon entropy of batch labels in k-NN neighborhoods.

        Returns mean entropy in [0, 1] where 1 = perfectly mixed.
        """
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


# ── Feature 3: Cross-Dataset Concordance ────────────────────────────────────

class CrossDatasetAnalyzer:
    """Marker gene overlap (Jaccard) for cross-dataset concordance."""

    @staticmethod
    def marker_genes(z, labels, adata, n_markers=50):
        """Identify marker genes per cluster using Wilcoxon rank-sum test.

        Returns dict mapping cluster_id → list of marker gene names.
        """
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
            genes = ad.uns['rank_genes_groups']['names'][str(cl)][:n_markers]
            markers[str(cl)] = list(genes)
        return markers

    @staticmethod
    def jaccard_concordance(markers_a, markers_b):
        """Compute Jaccard similarity between two sets of cluster marker genes.

        Matches clusters greedily by highest Jaccard overlap.
        Returns per-pair scores and mean Jaccard.
        """
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


# ── Feature 5: Clinical Association Testing ─────────────────────────────────

class ClinicalAssociationTest:
    """Test clinical relevance of GVAE-discovered rare cell states (Section 3.6).

    Associates abundance of each rare subpopulation with immunotherapy response
    via logistic regression, adjusting for therapy type.
    """

    @staticmethod
    def compute_rare_fractions(rare_labels, patient_masks):
        """Per-patient fraction of each rare subpopulation.

        Args:
            rare_labels: array of cluster labels (-1=normal, 100+=rare)
            patient_masks: list of boolean tensors, one per patient

        Returns:
            DataFrame of shape (n_patients, n_rare_clusters)
        """
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
        """Logistic regression: response ~ rare_fractions [+ therapy].

        Returns per-feature coefficients, p-values, and BH-corrected q-values.
        """
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

        # Add therapy as dummy variables if provided
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

        # Extract results (skip constant at index 0)
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

        # Benjamini-Hochberg FDR correction
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
        """Print summary of clinical association results."""
        if not results.get('features'):
            print(f"  Clinical association: {results.get('note', 'no results')}")
            return
        print(f"  Clinical association: {results['n_significant']}/{results['n_tested']} "
              f"features significant (FDR < 0.05)")
        for f in results['features']:
            sig_marker = '*' if f.get('significant', False) else ' '
            print(f"    {sig_marker} {f['name']}: coef={f['coef']:.3f}, "
                  f"p={f['p_value']:.4f}, q={f.get('q_value', 'N/A')}")
