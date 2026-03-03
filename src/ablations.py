"""Ablation study registry and baselines (Section 3.5 of paper).

Each ablation maps to config overrides that modify model architecture or
training behavior. Use via: python train.py --ablation <name>
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

# ── Registry ────────────────────────────────────────────────────────────────
# Keys match CLI --ablation choices. Values are dicts of config overrides.
ABLATION_REGISTRY = {
    # Ablation 1: Graph fusion
    'mol_only':     {'gate_mode': 'mol_only'},
    'spatial_only': {'gate_mode': 'spatial_only'},
    'static_0.3':   {'gate_mode': 'static_0.3'},
    'static_0.5':   {'gate_mode': 'static_0.5'},
    'static_0.7':   {'gate_mode': 'static_0.7'},

    # Ablation 2: Decoder
    'no_expr':   {'lambda1': 0.0},
    'gaussian':  {'decoder_type': 'gaussian'},

    # Ablation 3: Contrastive loss
    'no_contrastive': {'lambda2': 0.0},

    # Ablation 4: Rare cell detection (handled in analysis, not config)
    'rare_leiden': {'_rare_method': 'leiden'},

    # Ablation 5: Prediction pipeline
    'logreg_baseline': {'_prediction_method': 'logreg'},

    # Ablation 6: Frozen encoder
    'frozen_encoder': {'freeze_encoder': True},

    # Ablation 7: Encoder architecture
    'gcn_encoder': {'encoder_type': 'gcn'},
}


def apply_ablation(config, ablation_name):
    """Apply ablation overrides to a config dict. Returns modified copy."""
    if ablation_name is None:
        return config
    if ablation_name not in ABLATION_REGISTRY:
        raise ValueError(f"Unknown ablation '{ablation_name}'. "
                         f"Choose from: {list(ABLATION_REGISTRY.keys())}")
    config = config.copy()
    overrides = ABLATION_REGISTRY[ablation_name]
    config.update(overrides)
    print(f"  [ablation] '{ablation_name}' applied: {overrides}")
    return config


# ── Logistic Regression Baseline (Ablation 5) ──────────────────────────────

class LogisticRegressionBaseline:
    """Hand-crafted features + L1 logistic regression for prediction ablation.

    Features per patient:
    - Mean latent vector (D dims)
    - Shannon diversity of cluster assignments
    - Cluster proportion vector (K dims)
    """

    @staticmethod
    def extract_features(z, cluster_labels, patient_masks):
        """Build feature matrix (n_patients x n_features) from latent space."""
        n_patients = len(patient_masks)
        unique_clusters = np.unique(cluster_labels)
        n_clusters = len(unique_clusters)
        cluster_map = {c: i for i, c in enumerate(unique_clusters)}

        features = []
        for mask in patient_masks:
            mask_np = mask.cpu().numpy() if hasattr(mask, 'cpu') else np.asarray(mask)
            z_patient = z[mask_np]
            cl_patient = cluster_labels[mask_np]
            n_cells = len(z_patient)

            # Mean latent
            mean_z = z_patient.mean(axis=0)

            # Cluster proportions
            proportions = np.zeros(n_clusters)
            for c in cl_patient:
                proportions[cluster_map[c]] += 1
            if n_cells > 0:
                proportions /= n_cells

            # Shannon diversity
            p = proportions[proportions > 0]
            diversity = -np.sum(p * np.log(p + 1e-10))

            feat = np.concatenate([mean_z, proportions, [diversity]])
            features.append(feat)
        return np.array(features)

    @staticmethod
    def run(z, cluster_labels, patient_masks, y, n_folds=5, seed=42):
        """Run stratified CV with L1 logistic regression.

        Returns dict with AUROC, AUPRC, and per-fold results.
        """
        X = LogisticRegressionBaseline.extract_features(z, cluster_labels, patient_masks)
        y = np.asarray(y)
        n_patients = len(y)

        if n_patients < n_folds * 2:
            n_folds = max(2, n_patients // 4)

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        all_y_true = np.zeros(n_patients)
        all_y_pred = np.zeros(n_patients)
        fold_aurocs = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            clf = LogisticRegression(penalty='l1', solver='saga', max_iter=5000,
                                     random_state=seed, C=1.0)
            clf.fit(X[train_idx], y[train_idx])
            proba = clf.predict_proba(X[test_idx])
            # Handle case where only one class seen in training
            if proba.shape[1] == 2:
                preds = proba[:, 1]
            else:
                preds = np.full(len(test_idx), 0.5)
            all_y_true[test_idx] = y[test_idx]
            all_y_pred[test_idx] = preds
            try:
                fold_aurocs.append(roc_auc_score(y[test_idx], preds))
            except ValueError:
                fold_aurocs.append(0.5)

        try:
            pooled_auroc = roc_auc_score(all_y_true, all_y_pred)
        except ValueError:
            pooled_auroc = 0.5
        try:
            pooled_auprc = average_precision_score(all_y_true, all_y_pred)
        except ValueError:
            pooled_auprc = 0.0

        results = {
            'method': 'logreg_baseline',
            'n_features': X.shape[1],
            'n_patients': n_patients,
            'n_folds': n_folds,
            'fold_aurocs': [float(a) for a in fold_aurocs],
            'auroc_mean': float(np.mean(fold_aurocs)),
            'auroc_std': float(np.std(fold_aurocs)),
            'pooled_auroc': float(pooled_auroc),
            'pooled_auprc': float(pooled_auprc),
        }
        print(f"  LogReg Baseline: AUROC={results['auroc_mean']:.3f}+/-{results['auroc_std']:.3f} "
              f"(pooled={pooled_auroc:.3f})")
        return results
