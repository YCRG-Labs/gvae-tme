"""
Evaluates rare cell detection component of GVAE in isolation against known biological methods (testing vs. baseline)
Compares KL-based detection against marker-based annotation.
However, scSynO and CISC require real data, so the infrastructure will be built 
(and will produce meaningful results when the data actually gets produced).
"""
import numpy as np
from pathlib import Path
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score)

IMMUNO_MARKERS = ['PDCD1', 'CTLA4', 'FOXP3', 'CD274', 'TIGIT', 'LAG3']

def marker_based_annotation(adata, markers = IMMUNO_MARKERS):
    """
    Ground truth baseline using known immunosuppressive markers.
    Cells are immunosuppressive if they expresses 2+ markers above median.
    
    Markers will be actual gene expression values on real data, & returns None on synthetic data
    because genes don't exist on synthetic data.
    """
    available = [m for m in markers if m in adata.var_names]
    missing = [m for m in markers if m not in adata.var_names]

    if not available:
        print(f"None of {markers} found in dataset, as expected on synthetic data")
        print(f"On real data (GSE243280, melanoma etc.) these will exist")
        return None, None
    
    print(f"Missing marker baselines: {missing}") if missing else print(f"Using marker baselines: {available}")

    # extract marker expression
    marker_idx = [list(adata.var_names).index(m) for m in available]
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.asarray(adata.X)
    marker_expr = X[:, marker_idx]

    # find median threshhold, check markers that are 2+ above median
    thresholds = np.percentile(marker_expr, 50, axis=0)
    marker_labels = (marker_expr > thresholds).sum(axis=1) >= 2

    # only real flaw with this approach is that median is dataset-dependent

    print(f"[marker_baseline] {marker_labels.sum()} cells positive " f"({marker_labels.mean()*100:.1f}%)")
    return marker_labels, available

def evaluate_kl_vs_baseline(kl_scores, baseline_labels, eta=2.0):
    """
    Compare KL-based rare cell detection against a ground truth baseline by reporting precision, recall, F1, & AUROC.
    """
    if baseline_labels is None:
        print("No baseline labels — skipping comparison.")
        return None

    kl_binary = kl_scores > (kl_scores.mean() + eta * kl_scores.std())

    results = {
        'n_kl_rare': int(kl_binary.sum()),
        'n_baseline_rare': int(baseline_labels.sum()),
        'overlap': int((kl_binary & baseline_labels).sum()),
        'precision': float(precision_score(baseline_labels, kl_binary, zero_division=0)),
        'recall': float(recall_score(baseline_labels, kl_binary, zero_division=0)),
        'f1': float(f1_score(baseline_labels, kl_binary, zero_division=0)),
        'auroc': float(roc_auc_score(baseline_labels, kl_scores)) if baseline_labels.sum() > 0 else None
    }

    print("\n──────────── KL vs Marker Baseline ─────────────")
    print(f"KL rare cells: {results['n_kl_rare']}")
    print(f"Marker rare cells: {results['n_baseline_rare']}")
    print(f"Overlap: {results['overlap']}")
    print(f"Precision: {results['precision']:.3f}")
    print(f"Recall: {results['recall']:.3f}")
    print(f"F1: {results['f1']:.3f}")
    print(f"AUROC: {results['auroc']:.3f}") if results['auroc'] else None
    return results

def scsyno_baseline(adata, kl_scores, eta=2.0):
    """
    TODO: Implement when real data is available
    """
    print("scSynO Markers not found — requires real data")
    return None

def cisc_baseline(adata, kl_scores, eta=2.0):
    """
    Clustering-Independent Single-Cell analysis (CISC-style) baseline:
    identifies rare cells as those in small clusters in *expression* space (PCA + Leiden),
    independent of GVAE latent space. Small clusters (<5% of cells) are deemed rare.
    """
    import scanpy as sc

    # Cluster in expression space only (no GVAE latent)
    ad = adata.copy()
    if 'X_pca' not in ad.obsm or ad.obsm['X_pca'].shape[1] < 2:
        n_comps = min(50, ad.n_obs - 1, ad.n_vars - 1)
        if n_comps < 2:
            print("CISC: insufficient cells/genes for PCA — skipping.")
            return None
        sc.pp.pca(ad, n_comps=n_comps)
    sc.pp.neighbors(ad, use_rep='X_pca', n_neighbors=15)
    sc.tl.leiden(ad, resolution=1.0)
    labels = ad.obs['leiden'].astype(int).values

    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    rare_cluster_ids = unique[counts / total < 0.05]
    cisc_labels = np.isin(labels, rare_cluster_ids)

    kl_binary = kl_scores > (kl_scores.mean() + eta * kl_scores.std())

    print(f"\n────────── CISC-style baseline ───────────")
    print(f"Rare clusters (<5%): {len(rare_cluster_ids)} of {len(unique)}")
    print(f"CISC rare cells: {cisc_labels.sum()} ({cisc_labels.mean()*100:.1f}%)")
    print(f"KL rare cells: {kl_binary.sum()}")
    # num. of cells both methods agree are rare (high-confidence candidates)
    print(f"Overlap: {(cisc_labels & kl_binary).sum()}")

    metrics = {
        'cisc_labels': cisc_labels,
        'n_rare_clusters': int(len(rare_cluster_ids)),
        'precision': float(precision_score(cisc_labels, kl_binary, zero_division=0)),
        'recall': float(recall_score(cisc_labels, kl_binary, zero_division=0)),
        'f1': float(f1_score(cisc_labels, kl_binary, zero_division=0)),
        'auroc': float(roc_auc_score(cisc_labels, kl_scores)) if cisc_labels.sum() > 0 else None
    }

    print(f"Precision: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f} | F1: {metrics['f1']:.3f}")
    return metrics

def run_all_baselines(adata, kl_scores, labels=None, eta=2.0):
    """
    labels: optional GVAE cluster labels (used only for display/consistency).
    CISC clusters independently on expression (PCA+Leiden), not on GVAE latent.
    """
    print("\n════════ Rare Cell Detection Benchmark ════════\n")
    print(f"KL threshold calculations: mean + {eta}*std")

    results = {}

    # 1. marker-based
    marker_labels, used_markers = marker_based_annotation(adata)
    results['marker'] = evaluate_kl_vs_baseline(kl_scores, marker_labels, eta)

    # 2. CISC: cluster in expression space (PCA+Leiden), not GVAE latent
    results['cisc'] = cisc_baseline(adata, kl_scores, eta=eta)

    # 3. scSynO-based - will come
    return results