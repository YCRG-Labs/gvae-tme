"""
Evaluates rare cell detection component of GVAE in isolation against known biological methods (testing vs. baseline)
Compares KL-based detection against marker-based annotation.
However, scSynO and CISC require real data, so the infrastructure will be built 
(and will produce meaningful results when the data actually gets produced).
"""
import numpy as np
from pathlib import Path
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score)

IMMUNO_MARKERS_PRIMARY  = ['PDCD1', 'CTLA4', 'FOXP3', 'CD274', 'TIGIT', 'LAG3']
IMMUNO_MARKERS_EXTENDED = ['CD14', 'S100A9', 'FCN1', 'LYZ', 'TYROBP', 'S100A11', 'CST3', 'FCER1G', 'IFI30', 'C1QA']

def marker_based_annotation(adata, markers=None):
    # Try primary first, auto-fall-back to extended
    if markers is None:
        primary_found  = [m for m in IMMUNO_MARKERS_PRIMARY  if m in adata.var_names]
        extended_found = [m for m in IMMUNO_MARKERS_EXTENDED if m in adata.var_names]
        if primary_found:
            markers = primary_found
            print(f"[marker_baseline] Using primary immunosuppressive markers: {markers}")
        elif extended_found:
            markers = extended_found
            print(f"[marker_baseline] Primary markers not in HVGs. Using MDSC/monocyte markers: {markers}")
        else:
            print(f"[marker_baseline] No markers found in {adata.n_vars} HVGs — skipping.")
            return None, None
    
    available = [m for m in markers if m in adata.var_names]
    missing   = [m for m in markers if m not in adata.var_names]
    if missing:
        print(f"[marker_baseline] Missing from HVGs: {missing}")
    if not available:
        print(f"[marker_baseline] None of {markers} found in the {adata.n_vars} HVGs.")
        print(f"  Tip: rerun training with n_hvg=5000 to capture low-prevalence exhaustion markers.")
        return None, None

    # extract expression
    marker_idx   = [list(adata.var_names).index(m) for m in available]
    if 'counts' in adata.layers:
        raw  = adata.layers['counts']
        X    = raw.toarray() if hasattr(raw, 'toarray') else np.asarray(raw)
        print("[marker_baseline] Using raw counts layer for thresholding")
    else:
        X = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.asarray(adata.X)
    marker_expr = X[:, marker_idx]

    # threshold: above median on 2+ markers
    thresholds    = np.percentile(marker_expr, 50, axis=0)
    marker_labels = (marker_expr > thresholds).sum(axis=1) >= 2

    print(f"[marker_baseline] {marker_labels.sum()} cells positive ({marker_labels.mean()*100:.1f}%)")
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

def cisc_baseline(adata, kl_scores, eta=2.0):
    """
    Clustering-Independent Single-Cell analysis (CISC-style) baseline:
    identifies rare cells as those in small clusters in *expression* space (PCA + Leiden),
    independent of GVAE latent space. Small clusters (<5% of cells) are deemed rare.
    """
    import scanpy as sc
    ad = adata.copy() 
    if 'X_pca' not in ad.obsm or ad.obsm['X_pca'].shape[1] < 2:
        X_check = ad.X[:100].toarray() if hasattr(ad.X, 'toarray') else np.asarray(ad.X[:100])
        if X_check.max() > 50:
            print("[CISC] Raw counts detected — normalizing before PCA")
            sc.pp.normalize_total(ad, target_sum=1e4)
            sc.pp.log1p(ad)
        n_comps = min(50, ad.n_obs - 1, ad.n_vars - 1)
        if n_comps < 2:
            print("CISC: insufficient cells/genes for PCA — skipping.")
            return None
        sc.pp.pca(ad, n_comps=n_comps)

    # Cluster in expression space only (no GVAE latent)
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

def scsyno_baseline(adata, kl_scores, eta=2.0):
    """
    scSynO-style baseline: scores cells by combined z-score across
    immunosuppressive marker expression. Falls back to extended
    monocyte/MDSC markers if primary markers aren't in HVGs.
    """
    from scipy.stats import zscore as _zscore

    PRIMARY  = ['PDCD1', 'CTLA4', 'FOXP3', 'CD274', 'TIGIT', 'LAG3']
    EXTENDED = ['CD14', 'S100A9', 'S100A8', 'FCN1', 'LYZ', 'TYROBP', 'S100A11']

    available = [m for m in PRIMARY  if m in adata.var_names]
    if not available:
        available = [m for m in EXTENDED if m in adata.var_names]
    if not available:
        print("[scSynO] No markers found in HVGs — skipping.")
        return None

    marker_source = 'primary' if any(m in PRIMARY for m in available) else 'extended (MDSC)'
    print(f"\n────────── scSynO-style baseline ({marker_source}) ──────────")
    print(f"  Markers used: {available}")

    if 'counts' in adata.layers:
        raw = adata.layers['counts']
        X = raw.toarray() if hasattr(raw, 'toarray') else np.asarray(raw)
    else:
        X = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.asarray(adata.X)

    marker_idx  = [list(adata.var_names).index(m) for m in available]
    marker_expr = X[:, marker_idx]

    # z-score each marker, average across markers for a combined score
    z_scores = _zscore(marker_expr, axis=0)
    combined = z_scores.mean(axis=1)

    threshold     = combined.mean() + 2 * combined.std()
    scsyno_labels = combined > threshold
    kl_binary     = kl_scores > (kl_scores.mean() + eta * kl_scores.std())

    print(f"scSynO rare cells: {scsyno_labels.sum()} ({scsyno_labels.mean()*100:.1f}%)")
    print(f"KL rare cells:     {kl_binary.sum()}")
    print(f"Overlap:           {(scsyno_labels & kl_binary).sum()}")

    metrics = {
        'scsyno_labels':  scsyno_labels,
        'combined_score': combined,
        'markers_used':   available,
        'marker_source':  marker_source,
        'precision': float(precision_score(scsyno_labels, kl_binary, zero_division=0)),
        'recall':    float(recall_score(scsyno_labels,    kl_binary, zero_division=0)),
        'f1':        float(f1_score(scsyno_labels,        kl_binary, zero_division=0)),
        'auroc':     float(roc_auc_score(scsyno_labels, kl_scores))
                     if scsyno_labels.sum() > 0 else None,
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

    # 3. scSynO-based
    results['scsyno'] = scsyno_baseline(adata, kl_scores, eta=eta)

    return results