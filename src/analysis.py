import numpy as np
import scanpy as sc
import anndata
from sklearn.metrics import silhouette_score


class RareCellDetector:
    def __init__(self, threshold=2.0):
        self.threshold = threshold
        
    def compute_kl(self, mu, logvar):
        sigma2 = np.exp(logvar)
        kl = 0.5 * np.sum(mu**2 + sigma2 - logvar - 1, axis=1)
        return kl
    
    def detect(self, mu, logvar, eta=2.0):
        kl = self.compute_kl(mu, logvar)
        kl_mean = np.mean(kl)
        kl_std = np.std(kl)
        scores = (kl - kl_mean) / kl_std
        is_rare = scores > eta
        return scores, is_rare
    
    def subcluster(self, z, is_rare, resolution=2.0):
        z_rare = z[is_rare]
        adata = anndata.AnnData(X=z_rare)
        adata.obsm['X_latent'] = z_rare
        sc.pp.neighbors(adata, use_rep='X_latent', n_neighbors=15)
        sc.tl.leiden(adata, resolution=resolution)
        labels = np.full(len(z), -1, dtype=int)
        labels[is_rare] = adata.obs['leiden'].astype(int).values + 100
        return labels


class ClusteringAnalyzer:
    def __init__(self, resolutions=[0.5, 1.0, 1.5, 2.0]):
        self.resolutions = resolutions
        
    def compute_confidence(self, logvar):
        sigma2_mean = np.mean(np.exp(logvar), axis=1)
        return np.exp(-sigma2_mean)
    
    def cluster(self, z, adjacency=None, logvar=None, resolution=1.0):
        adata = anndata.AnnData(X=z)
        adata.obsm['X_latent'] = z
        if adjacency is not None:
            adata.obsp['connectivities'] = adjacency
        else:
            sc.pp.neighbors(adata, use_rep='X_latent', n_neighbors=15)
        sc.tl.leiden(adata, resolution=resolution)
        labels = adata.obs['leiden'].astype(int).values
        if logvar is not None:
            confidence = self.compute_confidence(logvar)
            c_thresh = np.percentile(confidence, 20)
        return labels
    
    def evaluate(self, z, labels):
        if len(np.unique(labels)) > 1:
            sil = silhouette_score(z, labels)
        else:
            sil = 0.0
        return {'n_clusters': len(np.unique(labels)), 'silhouette': sil}


class PredictionAnalyzer:
    @staticmethod
    def compute_metrics(y_true, y_pred):
        from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
        y_pred_binary = (y_pred > 0.5).astype(int)
        return {'auroc': roc_auc_score(y_true, y_pred),
                'auprc': average_precision_score(y_true, y_pred),
                'accuracy': accuracy_score(y_true, y_pred_binary),
                'precision': precision_score(y_true, y_pred_binary, zero_division=0),
                'recall': recall_score(y_true, y_pred_binary, zero_division=0),
                'f1': f1_score(y_true, y_pred_binary, zero_division=0)}
    
    @staticmethod
    def permutation_test(model, data, n_permutations=1000):
        model.eval()
        with torch.no_grad():
            outputs = model(data)
            y_true = data.y.cpu().numpy()
            y_pred = outputs['predictions'].cpu().numpy()
            actual_auroc = roc_auc_score(y_true, y_pred)
        null_aurocs = []
        for _ in range(n_permutations):
            y_perm = np.random.permutation(y_true)
            null_aurocs.append(roc_auc_score(y_perm, y_pred))
        return np.mean(np.array(null_aurocs) >= actual_auroc)
    
    @staticmethod
    def attention_maps(attentions, cell_coords):
        all_attns = torch.cat([a.cpu() for a in attentions]).numpy()
        return {'mean_attention': np.mean(all_attns),
                'max_attention': np.max(all_attns),
                'high_attention_cells': np.sum(all_attns > np.percentile(all_attns, 90)),
                'attention_entropy': -np.sum(all_attns * np.log(all_attns + 1e-10))}
