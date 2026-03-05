import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

ABLATION_REGISTRY = {
    'mol_only':     {'gate_mode': 'mol_only'},
    'spatial_only': {'gate_mode': 'spatial_only'},
    'static_0.3':   {'gate_mode': 'static_0.3'},
    'static_0.5':   {'gate_mode': 'static_0.5'},
    'static_0.7':   {'gate_mode': 'static_0.7'},
    'no_expr':   {'lambda1': 0.0},
    'gaussian':  {'decoder_type': 'gaussian'},
    'no_contrastive': {'lambda2': 0.0},
    'rare_leiden': {'_rare_method': 'leiden'},
    'logreg_baseline': {'_prediction_method': 'logreg'},
    'frozen_encoder': {'freeze_encoder': True},
    'gcn_encoder': {'encoder_type': 'gcn'},
    'spatial_bias': {'gate_mode': 'learned', 'spatial_bias': -1.0},
}


def apply_ablation(config, ablation_name):
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


class LogisticRegressionBaseline:

    @staticmethod
    def extract_features(z, cluster_labels, patient_masks, all_clusters=None):
        n_patients = len(patient_masks)
        if all_clusters is None:
            all_clusters = np.unique(cluster_labels)
        unique_clusters = np.asarray(all_clusters)
        n_clusters = len(unique_clusters)
        cluster_map = {c: i for i, c in enumerate(unique_clusters)}

        features = []
        for mask in patient_masks:
            mask_np = mask.cpu().numpy() if hasattr(mask, 'cpu') else np.asarray(mask)
            z_patient = z[mask_np]
            cl_patient = cluster_labels[mask_np]
            n_cells = len(z_patient)

            mean_z = z_patient.mean(axis=0)

            proportions = np.zeros(n_clusters)
            for c in cl_patient:
                if c in cluster_map:
                    proportions[cluster_map[c]] += 1
            if n_cells > 0:
                proportions /= n_cells

            p = proportions[proportions > 0]
            diversity = -np.sum(p * np.log(p + 1e-10))

            feat = np.concatenate([mean_z, proportions, [diversity]])
            features.append(feat)
        return np.array(features)

    @staticmethod
    def run(z, cluster_labels, patient_masks, y, n_folds=5, seed=42):
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
