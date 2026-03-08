import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
# Prefer statsmodels.multitest if available; otherwise provide lightweight fallback for multipletests
try:
    from statsmodels.stats.multitest import multipletests  # type: ignore
except Exception:
    import numpy as _np

    def multipletests(pvals, alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False):
        """Lightweight fallback implementing Benjamini-Hochberg FDR correction (fdr_bh).
        Returns (reject, pvals_corrected, alphacSidak, alphacBonf) to mirror statsmodels API.
        Only 'fdr_bh' is supported by this fallback.
        """
        pvals = _np.asarray(pvals)
        m = pvals.size
        if m == 0:
            return _np.array([], dtype=bool), _np.array([]), alpha, alpha

        if method != 'fdr_bh':
            raise NotImplementedError(f"multipletests fallback only supports 'fdr_bh', got '{method}'")

        if is_sorted:
            sort_idx = _np.arange(m)
            p_sorted = pvals
        else:
            sort_idx = _np.argsort(pvals)
            p_sorted = pvals[sort_idx]

        ranks = _np.arange(1, m + 1, dtype=float)
        # BH adjusted p-values
        adj_sorted = p_sorted * m / ranks
        # enforce monotonicity of adjusted p-values (non-increasing when traversed from largest to smallest p)
        adj_sorted = _np.minimum.accumulate(adj_sorted[::-1])[::-1]
        adj_sorted = _np.minimum(adj_sorted, 1.0)

        # map back to original order
        p_adj = _np.empty_like(pvals, dtype=float)
        p_adj[sort_idx] = adj_sorted

        reject = p_adj <= alpha
        alphacSidak = 1.0 - (1.0 - alpha) ** (1.0 / float(m))
        alphacBonf = alpha / float(m)

        if returnsorted:
            return reject[sort_idx], p_adj[sort_idx], alphacSidak, alphacBonf

        return reject, p_adj, alphacSidak, alphacBonf

# Prefer scikit-learn if available; otherwise provide lightweight fallbacks for roc_curve and auc
try:
    from sklearn.metrics import roc_curve, auc  # type: ignore
except Exception:
    import numpy as _np

    def roc_curve(y_true, y_score):
        """Lightweight ROC implementation returning fpr, tpr, thresholds."""
        y_true = _np.asarray(y_true).astype(int)
        y_score = _np.asarray(y_score).astype(float)

        if y_true.size == 0:
            return _np.array([0.0, 1.0]), _np.array([0.0, 1.0]), _np.array([_np.inf, -_np.inf])

        # Sort scores and corresponding true labels in descending score order
        desc_score_indices = _np.argsort(-y_score, kind='mergesort')
        y_score_sorted = y_score[desc_score_indices]
        y_true_sorted = y_true[desc_score_indices]

        # Find indices where score changes (distinct thresholds)
        distinct_value_indices = _np.where(_np.diff(y_score_sorted) != 0)[0]
        threshold_idxs = _np.r_[distinct_value_indices, y_true_sorted.size - 1]
        thresholds = y_score_sorted[threshold_idxs]

        # True positives and false positives at each threshold index
        tps = _np.cumsum(y_true_sorted)[threshold_idxs]
        fps = (1 + threshold_idxs) - tps

        # Prepend (0,0) point to match sklearn output convention
        tps = _np.r_[0, tps]
        fps = _np.r_[0, fps]
        thresholds = _np.r_[thresholds[0] + 1, thresholds]  # a threshold above max score

        P = _np.sum(y_true == 1)
        N = y_true.size - P

        # Handle degenerate cases where only one class is present
        if P == 0 or N == 0:
            fpr = _np.array([0.0, 1.0])
            tpr = _np.array([0.0, 1.0])
            return fpr, tpr, thresholds

        tpr = tps / float(P)
        fpr = fps / float(N)
        return fpr, tpr, thresholds

    def auc(x, y):
        """Compute area under curve using the trapezoidal rule."""
        x = _np.asarray(x)
        y = _np.asarray(y)
        order = _np.argsort(x)
        x = x[order]
        y = y[order]
        return float(_np.trapz(y, x))

import matplotlib.pyplot as plt
import seaborn as sns
import os

np.random.seed(42)
os.makedirs('outputs', exist_ok=True)

def generate_synthetic_data(n_patients=100, n_clusters=15, cells_per_patient=1000):
    """Generate synthetic data"""
    data = []
    for patient_id in range(n_patients):
        is_responder = patient_id < 50  # First 50 are responders
        
        for _ in range(cells_per_patient):
            cluster = np.random.randint(0, n_clusters)
            # Rare cell score (higher for responders)
            kl_score = np.random.exponential(2.0 if is_responder else 1.0)
            
            data.append({
                'patient_id': patient_id,
                'cluster': f'Cluster_{cluster}',
                'responder': is_responder,
                'kl_rare_score': kl_score
            })
    
    return pd.DataFrame(data)

def cluster_enrichment_analysis(df):
    """Task 1: Fisher's exact test per cluster with BH correction"""
    print("\n=== Task 1: Cluster Enrichment Analysis ===")
    
    clusters = df['cluster'].unique()
    results = []
    
    for cluster in clusters:
        # Contingency table
        cluster_responder = len(df[(df['cluster'] == cluster) & (df['responder'] == True)])
        cluster_nonresponder = len(df[(df['cluster'] == cluster) & (df['responder'] == False)])
        other_responder = len(df[(df['cluster'] != cluster) & (df['responder'] == True)])
        other_nonresponder = len(df[(df['cluster'] != cluster) & (df['responder'] == False)])
        
        contingency = [[cluster_responder, cluster_nonresponder],
                      [other_responder, other_nonresponder]]
        
        odds_ratio, p_value = fisher_exact(contingency)
        
        results.append({
            'cluster': cluster,
            'odds_ratio': odds_ratio,
            'p_value': p_value,
            'enrichment': 'Responder' if odds_ratio > 1 else 'Non-responder'
        })
    
    results_df = pd.DataFrame(results)
    
    # BH correction
    _, results_df['p_adj'], _, _ = multipletests(results_df['p_value'], method='fdr_bh')
    results_df['significant'] = results_df['p_adj'] < 0.05
    
    # Plot (compute safe log2 odds to avoid -inf/inf/NaN)
    plt.figure(figsize=(12, 6))
    # replace problematic values: zeros -> small positive, infinities -> large finite
    or_vals = results_df['odds_ratio'].replace([0, np.inf, -np.inf], np.nan)
    finite = or_vals.dropna()
    if finite.empty:
        safe_or = np.ones_like(or_vals.fillna(1.0))
    else:
        min_f = float(finite.min())
        max_f = float(finite.max())
        safe_or = or_vals.fillna(min_f if min_f > 0 else 1e-6).replace(np.inf, max_f * 1e3)
    log2_or = np.log2(safe_or.astype(float))
    
    colors = ['red' if sig else 'gray' for sig in results_df['significant']]
    plt.bar(range(len(results_df)), log2_or, color=colors)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
    plt.xlabel('Cluster')
    plt.ylabel('Log2(Odds Ratio)')
    plt.title('Cluster Enrichment in Responders vs Non-Responders')
    plt.xticks(range(len(results_df)), results_df['cluster'], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('outputs/cluster_enrichment.png', dpi=300)
    print(f"✓ Saved: outputs/cluster_enrichment.png")
    
    print(f"\nSignificant clusters (p_adj < 0.05): {results_df['significant'].sum()}")
    print(results_df[results_df['significant']][['cluster', 'odds_ratio', 'p_adj']])

def rare_cell_burden_analysis(df):
    """Task 2: Rare cell burden with Mann-Whitney U test"""
    print("\n=== Task 2: Rare Cell Burden Analysis ===")
    
    # Aggregate per patient
    patient_scores = df.groupby(['patient_id', 'responder'])['kl_rare_score'].agg(['mean', 'max']).reset_index()
    
    responders_mean = patient_scores[patient_scores['responder'] == True]['mean']
    nonresponders_mean = patient_scores[patient_scores['responder'] == False]['mean']
    stat_mean, p_mean = mannwhitneyu(responders_mean, nonresponders_mean, alternative='two-sided')
    
    responders_max = patient_scores[patient_scores['responder'] == True]['max']
    nonresponders_max = patient_scores[patient_scores['responder'] == False]['max']
    stat_max, p_max = mannwhitneyu(responders_max, nonresponders_max, alternative='two-sided')
    
    print(f"\nMann-Whitney U test (mean aggregation): p = {p_mean:.4f}")
    print(f"Mann-Whitney U test (max aggregation): p = {p_max:.4f}")
    
    # Use more predictive aggregation
    best_metric = 'mean' if p_mean < p_max else 'max'
    print(f"\nMore predictive metric: {best_metric} (p = {min(p_mean, p_max):.4f})")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, (metric, p_val) in enumerate([('mean', p_mean), ('max', p_max)]):
        ax = axes[i]
        plot_data = patient_scores[['responder', metric]].copy()
        plot_data['Response'] = plot_data['responder'].map({True: 'Responder', False: 'Non-responder'})
        
        sns.boxplot(data=plot_data, x='Response', y=metric, ax=ax)
        ax.set_ylabel(f'KL Rare Score ({metric.capitalize()})')
        ax.set_title(f'Rare Cell Burden ({metric.capitalize()})\np = {p_val:.4f}')
    
    plt.tight_layout()
    plt.savefig('outputs/rare_cell_burden.png', dpi=300)
    print(f"✓ Saved: outputs/rare_cell_burden.png")
    
    return patient_scores, best_metric

def roc_analysis(patient_scores, metric):
    """Task 3: ROC curve for response prediction"""
    print("\n=== Task 3: ROC Curve Analysis ===")
    
    y_true = patient_scores['responder'].astype(int)
    y_score = patient_scores[metric]
    
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUROC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Immunotherapy Response Prediction\nUsing Rare Cell Burden ({metric.capitalize()})')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/roc_curve.png', dpi=300)
    print(f"✓ Saved: outputs/roc_curve.png")
    
    print(f"\n*** CLINICAL RELEVANCE: AUROC = {roc_auc:.3f} ***")

if __name__ == "__main__":
    print("Generating synthetic data...")
    df = generate_synthetic_data(n_patients=100, n_clusters=15, cells_per_patient=1000)
    print(f"Generated {len(df):,} cells from {df['patient_id'].nunique()} patients")
    
    cluster_enrichment_analysis(df)
    patient_scores, best_metric = rare_cell_burden_analysis(df)
    roc_analysis(patient_scores, best_metric)
    
    print("\n✓ Analysis complete. All outputs saved to ./outputs/")