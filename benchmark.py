import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import json
import anndata
import scanpy as sc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from src.model import GVAEModel
from src.trainer import Trainer
from src.analysis import (RareCellDetector, ClusteringAnalyzer, PredictionAnalyzer,
                          CrossDatasetAnalyzer, BiologicalValidation)
from src.data_utils import prepare_graph_data
from src.config import CONFIGS
from src.ablations import LogisticRegressionBaseline
from src.baselines import (ScVIBaseline, ScanpyBaseline, ImmunosuppressiveSignatures,
                           CrossDatasetTransfer)

from train import (load_real_data, build_model, make_trainer, make_serializable,
                   run_downstream, assign_patient_splits, DATASETS)


def run_gvae(adata, info, config, output_dir):
    has_spatial = info['has_spatial']
    graph_kwargs = {'r_spatial': info.get('r_spatial', 82.5)}

    if adata.n_obs > 50_000 and config.get('batch_size') is None:
        config['batch_size'] = 512

    data = prepare_graph_data(adata, has_spatial=has_spatial, **graph_kwargs)
    has_response = hasattr(data, 'y') and data.y is not None
    model = build_model(config, data, has_response)
    freeze = config.get('freeze_encoder', False)
    trainer = make_trainer(model, config, config['device'], output_dir, freeze_encoder=freeze, data=data)
    trainer.train(data)

    downstream = run_downstream(model, data, config, adata, output_dir)
    results = {'method': 'gvae', **downstream}

    if has_response:
        model.eval()
        data_eval = data.to(config['device'])
        with torch.no_grad():
            outputs = model(data_eval)
        z = outputs['z'].cpu().numpy()
        labels = np.load(output_dir / 'cluster_labels.npy')
        y = data_eval.y.cpu().numpy()
        logreg = LogisticRegressionBaseline.run(z, labels, data.patient_masks, y)
        results['logreg_on_gvae'] = logreg

    return results, model, data


def run_scvi(adata, config, output_dir):
    print("\n--- scVI Baseline ---")
    model, info = ScVIBaseline.train_and_embed(
        adata, n_latent=config.get('latent_dim', 32),
        max_epochs=min(200, config.get('epochs_phase1', 300)))

    if model is None:
        print(f"  scVI skipped: {info.get('note', 'unknown error')}")
        return {'method': 'scvi', 'note': info.get('note')}

    z = info['z']
    patient_masks = None
    y = None
    if 'patient_id' in adata.obs.columns and 'response' in adata.obs.columns:
        unique_pids = adata.obs['patient_id'].unique()
        patient_masks = [
            torch.tensor(adata.obs['patient_id'].values == pid, dtype=torch.bool)
            for pid in unique_pids
        ]
        y = np.array([
            adata.obs.loc[adata.obs['patient_id'] == pid, 'response'].iloc[0]
            for pid in unique_pids
        ])

    results = ScVIBaseline.run_downstream(z, adata, patient_masks=patient_masks, y=y, model=model)
    np.save(output_dir / 'scvi_embeddings.npy', z)
    return results


def run_scanpy(adata, config, output_dir):
    print("\n--- Scanpy Standard Baseline ---")
    patient_masks = None
    y = None
    if 'patient_id' in adata.obs.columns and 'response' in adata.obs.columns:
        unique_pids = adata.obs['patient_id'].unique()
        patient_masks = [
            torch.tensor(adata.obs['patient_id'].values == pid, dtype=torch.bool)
            for pid in unique_pids
        ]
        y = np.array([
            adata.obs.loc[adata.obs['patient_id'] == pid, 'response'].iloc[0]
            for pid in unique_pids
        ])

    results = ScanpyBaseline.run(adata, patient_masks=patient_masks, y=y)
    return results


def run_benchmark(args, config):
    output_dir = Path('outputs') / f'{args.data}_benchmark'
    output_dir.mkdir(parents=True, exist_ok=True)

    adata, info = load_real_data(args.data, n_hvg=args.n_hvg, max_cells=args.max_cells)
    print(f"  {adata.n_obs} cells, {adata.n_vars} genes")

    methods = [m.strip() for m in args.methods.split(',')]
    all_results = {}

    for method in methods:
        method_dir = output_dir / method
        method_dir.mkdir(exist_ok=True)

        if method == 'gvae':
            print(f"\n--- GVAE ---")
            results, _, _ = run_gvae(adata, info, config.copy(), method_dir)
            all_results['gvae'] = results
        elif method == 'scvi':
            results = run_scvi(adata, config, method_dir)
            all_results['scvi'] = results
        elif method == 'scanpy':
            results = run_scanpy(adata, config, method_dir)
            all_results['scanpy'] = results
        else:
            print(f"  [warn] Unknown method: {method}")

    comparison = build_comparison_table(all_results)
    print_comparison_table(comparison, methods)

    output = {
        'dataset': args.data,
        'methods': all_results,
        'comparison': comparison,
    }
    with open(output_dir / 'comparison.json', 'w') as f:
        json.dump(make_serializable(output), f, indent=2)
    print(f"\nResults saved to {output_dir}/")


def run_benchmark_cv(args, config):
    output_dir = Path('outputs') / f'{args.data}_benchmark_cv'
    output_dir.mkdir(parents=True, exist_ok=True)

    adata, info = load_real_data(args.data, n_hvg=args.n_hvg, max_cells=args.max_cells)
    if 'response' not in adata.obs.columns or 'patient_id' not in adata.obs.columns:
        print("ERROR: --cv requires response and patient_id columns")
        return

    patient_ids = adata.obs['patient_id'].values
    unique_pids = np.unique(patient_ids)
    patient_responses = np.array([
        adata.obs.loc[adata.obs['patient_id'] == pid, 'response'].iloc[0]
        for pid in unique_pids
    ])
    n_patients = len(unique_pids)
    print(f"  {n_patients} patients, {adata.n_obs} cells")

    methods = [m.strip() for m in args.methods.split(',')]
    has_spatial = info['has_spatial']
    graph_kwargs = {'r_spatial': info.get('r_spatial', 82.5)}

    method_folds = {m: [] for m in methods}
    method_y_pred = {m: np.zeros(n_patients) for m in methods}
    all_y_true = np.zeros(n_patients)
    pid_to_idx = {pid: i for i, pid in enumerate(unique_pids)}

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)

    for fold, (train_val_idx, test_idx) in enumerate(skf.split(unique_pids, patient_responses)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{args.n_folds}")
        print(f"{'='*50}")

        test_pids = set(unique_pids[test_idx])
        train_val_pids = unique_pids[train_val_idx]
        train_val_resp = patient_responses[train_val_idx]

        n_val = max(1, len(train_val_pids) // 5)
        min_class_count = int(min(np.bincount(train_val_resp.astype(int))))
        n_splits_inner = max(2, min(len(train_val_pids) // max(n_val, 1), min_class_count))
        val_skf = StratifiedKFold(n_splits=n_splits_inner,
                                  shuffle=True, random_state=42 + fold)
        inner_train_idx, inner_val_idx = next(val_skf.split(train_val_pids, train_val_resp))
        train_pids = set(train_val_pids[inner_train_idx])
        val_pids = set(train_val_pids[inner_val_idx])

        for method in methods:
            fold_dir = output_dir / f'fold_{fold}' / method
            fold_dir.mkdir(parents=True, exist_ok=True)

            if method == 'gvae':
                fold_config = config.copy()
                train_val_mask = np.isin(patient_ids,
                    list(train_pids) + list(val_pids))
                ad_train_val = adata[train_val_mask].copy()
                n_comps = min(50, ad_train_val.n_obs - 1, ad_train_val.n_vars - 1)
                sc.pp.pca(ad_train_val, n_comps=n_comps)

                if ad_train_val.n_obs > 50_000 and fold_config.get('batch_size') is None:
                    fold_config['batch_size'] = 512

                data = prepare_graph_data(ad_train_val, has_spatial=has_spatial, **graph_kwargs)

                tv_patient_ids = ad_train_val.obs['patient_id'].values
                tv_unique_pids = np.unique(tv_patient_ids)
                tv_pid_to_idx = {pid: i for i, pid in enumerate(tv_unique_pids)}

                train_idx = [tv_pid_to_idx[p] for p in sorted(train_pids) if p in tv_pid_to_idx]
                val_idx = [tv_pid_to_idx[p] for p in sorted(val_pids) if p in tv_pid_to_idx]
                data.train_patient_idx = torch.tensor(train_idx, dtype=torch.long)
                data.val_patient_idx = torch.tensor(val_idx, dtype=torch.long)
                data.test_patient_idx = torch.tensor([], dtype=torch.long)

                cell_train = np.array([pid in train_pids for pid in tv_patient_ids])
                cell_val = np.array([pid in val_pids for pid in tv_patient_ids])
                data.train_mask = torch.tensor(cell_train, dtype=torch.bool)
                data.val_mask = torch.tensor(cell_val, dtype=torch.bool)
                data.test_mask = torch.tensor(np.zeros(len(tv_patient_ids), dtype=bool))

                model = build_model(fold_config, data, use_predictor=True)
                trainer = make_trainer(model, fold_config, fold_config['device'],
                                       fold_dir, data=data)
                trainer.train(data)

                model.eval()
                train_z = model(data.to(fold_config['device']))['z'].detach().cpu().numpy()
                train_labels_obj = ClusteringAnalyzer()
                best_res, _ = train_labels_obj.select_resolution(train_z)
                train_labels, _ = train_labels_obj.cluster(train_z, resolution=best_res)

                from sklearn.decomposition import PCA as SkPCA
                X_tv = ad_train_val.X
                if hasattr(X_tv, 'toarray'):
                    X_tv = X_tv.toarray()
                pca_fit = SkPCA(n_components=n_comps).fit(X_tv)

                test_cell_mask = np.isin(patient_ids, list(test_pids))
                X_test = adata[test_cell_mask].X
                if hasattr(X_test, 'toarray'):
                    X_test = X_test.toarray()
                test_pca = pca_fit.transform(X_test)

                ad_test = adata[test_cell_mask].copy()
                ad_test.obsm['X_pca'] = test_pca
                test_data = prepare_graph_data(ad_test, has_spatial=has_spatial, **graph_kwargs)

                with torch.no_grad():
                    test_outputs = model(test_data.to(fold_config['device']))
                test_z = test_outputs['z'].cpu().numpy()

                from sklearn.neighbors import KNeighborsClassifier
                knn_clf = KNeighborsClassifier(n_neighbors=15)
                knn_clf.fit(train_z, train_labels)
                test_labels = knn_clf.predict(test_z)

                test_patient_ids = patient_ids[test_cell_mask]
                test_pids_sorted = sorted(test_pids)

                train_feat_masks = [
                    torch.tensor(tv_patient_ids == pid, dtype=torch.bool)
                    for pid in tv_unique_pids if pid in train_pids
                ]
                train_resp = np.array([
                    patient_responses[list(unique_pids).index(pid)]
                    for pid in tv_unique_pids if pid in train_pids
                ])
                all_clusters = np.unique(train_labels)
                features_fit = LogisticRegressionBaseline.extract_features(
                    train_z, train_labels, train_feat_masks, all_clusters=all_clusters)

                test_feat_masks = [
                    torch.tensor(test_patient_ids == pid, dtype=torch.bool)
                    for pid in test_pids_sorted
                ]
                features_test = LogisticRegressionBaseline.extract_features(
                    test_z, test_labels, test_feat_masks, all_clusters=all_clusters)

                from sklearn.linear_model import LogisticRegression
                clf = LogisticRegression(penalty='l1', solver='saga', max_iter=5000,
                                          random_state=42, C=1.0)
                clf.fit(features_fit, train_resp)
                proba = clf.predict_proba(features_test)
                y_test = np.array([
                    patient_responses[list(unique_pids).index(pid)]
                    for pid in test_pids_sorted
                ])
                y_pred_test = proba[:, 1] if proba.shape[1] == 2 else np.full(len(y_test), 0.5)

            elif method == 'scvi':
                train_val_mask = np.isin(patient_ids,
                    list(train_pids) + list(val_pids))
                test_cell_mask = np.isin(patient_ids, list(test_pids))
                ad_train = adata[train_val_mask].copy()
                scvi_model, scvi_info = ScVIBaseline.train_and_embed(ad_train,
                    n_latent=config.get('latent_dim', 32))
                if scvi_model is None:
                    method_folds[method].append({'auroc': 0.5, 'auprc': 0.5})
                    continue

                z_train = scvi_info['z']
                clusterer = ClusteringAnalyzer()
                best_res, _ = clusterer.select_resolution(z_train)
                labels_train, _ = clusterer.cluster(z_train, resolution=best_res)

                train_val_pids_list = list(train_pids) + list(val_pids)
                train_val_patient_ids = patient_ids[train_val_mask]
                train_masks = [
                    torch.tensor(train_val_patient_ids == pid, dtype=torch.bool)
                    for pid in unique_pids if pid in train_pids or pid in val_pids
                ]
                train_only_mask = np.array([pid in train_pids for pid in unique_pids
                                            if pid in train_pids or pid in val_pids])
                test_only_mask = ~train_only_mask

                from sklearn.neighbors import KNeighborsClassifier
                knn_clf = KNeighborsClassifier(n_neighbors=15)
                knn_clf.fit(z_train, labels_train)

                try:
                    ad_test = adata[test_cell_mask].copy()
                    z_test = scvi_model.get_latent_representation(ad_test)
                except Exception:
                    n_comps = min(config.get('latent_dim', 32), ad_train.n_vars - 1)
                    from sklearn.decomposition import PCA as SkPCA
                    pca = SkPCA(n_components=n_comps).fit(ad_train.X.toarray()
                        if hasattr(ad_train.X, 'toarray') else ad_train.X)
                    X_test = adata[test_cell_mask].X
                    if hasattr(X_test, 'toarray'):
                        X_test = X_test.toarray()
                    z_test = pca.transform(X_test)

                labels_test = knn_clf.predict(z_test)

                test_patient_ids = patient_ids[test_cell_mask]
                test_pids_sorted = sorted(test_pids)
                test_masks = [
                    torch.tensor(train_val_patient_ids == pid, dtype=torch.bool)
                    for pid in unique_pids if pid in train_pids or pid in val_pids
                ]
                all_clusters = np.unique(labels_train)
                features_train = LogisticRegressionBaseline.extract_features(
                    z_train, labels_train, train_masks, all_clusters=all_clusters)
                z_all_test = z_test
                labels_all_test = labels_test
                test_feat_masks = [
                    torch.tensor(test_patient_ids == pid, dtype=torch.bool)
                    for pid in test_pids_sorted
                ]
                features_test = LogisticRegressionBaseline.extract_features(
                    z_all_test, labels_all_test, test_feat_masks, all_clusters=all_clusters)

                from sklearn.linear_model import LogisticRegression
                train_resp = np.array([
                    patient_responses[list(unique_pids).index(pid)]
                    for pid in unique_pids if pid in train_pids
                ])
                train_feat_masks = [
                    torch.tensor(train_val_patient_ids == pid, dtype=torch.bool)
                    for pid in unique_pids if pid in train_pids
                ]
                features_fit = LogisticRegressionBaseline.extract_features(
                    z_train, labels_train, train_feat_masks, all_clusters=all_clusters)
                clf = LogisticRegression(penalty='l1', solver='saga', max_iter=5000,
                                          random_state=42, C=1.0)
                clf.fit(features_fit, train_resp)
                proba = clf.predict_proba(features_test)
                y_test = np.array([
                    patient_responses[list(unique_pids).index(pid)]
                    for pid in test_pids_sorted
                ])
                y_pred_test = proba[:, 1] if proba.shape[1] == 2 else np.full(len(y_test), 0.5)

            elif method == 'scanpy':
                train_val_mask = np.isin(patient_ids,
                    list(train_pids) + list(val_pids))
                test_cell_mask = np.isin(patient_ids, list(test_pids))

                ad_train = adata[train_val_mask].copy()
                n_comps = min(50, ad_train.n_obs - 1, ad_train.n_vars - 1)
                sc.pp.pca(ad_train, n_comps=n_comps)
                sc.pp.neighbors(ad_train, use_rep='X_pca', n_neighbors=15)
                sc.tl.leiden(ad_train, resolution=1.0)
                z_train = ad_train.obsm['X_pca']
                labels_train = ad_train.obs['leiden'].astype(int).values

                from sklearn.neighbors import KNeighborsClassifier
                knn_clf = KNeighborsClassifier(n_neighbors=15)
                knn_clf.fit(z_train, labels_train)

                pca_loadings = ad_train.varm['PCs']
                X_train_raw = ad_train.X.toarray() if hasattr(ad_train.X, 'toarray') else np.asarray(ad_train.X)
                pca_mean = X_train_raw.mean(axis=0)
                X_test = adata[test_cell_mask].X
                if hasattr(X_test, 'toarray'):
                    X_test = X_test.toarray()
                else:
                    X_test = np.asarray(X_test)
                z_test = (X_test - pca_mean) @ pca_loadings
                labels_test = knn_clf.predict(z_test)

                train_val_patient_ids = patient_ids[train_val_mask]
                test_patient_ids = patient_ids[test_cell_mask]
                test_pids_sorted = sorted(test_pids)

                all_clusters = np.unique(labels_train)
                train_feat_masks = [
                    torch.tensor(train_val_patient_ids == pid, dtype=torch.bool)
                    for pid in unique_pids if pid in train_pids
                ]
                train_resp = np.array([
                    patient_responses[list(unique_pids).index(pid)]
                    for pid in unique_pids if pid in train_pids
                ])
                features_fit = LogisticRegressionBaseline.extract_features(
                    z_train, labels_train, train_feat_masks, all_clusters=all_clusters)

                test_feat_masks = [
                    torch.tensor(test_patient_ids == pid, dtype=torch.bool)
                    for pid in test_pids_sorted
                ]
                features_test = LogisticRegressionBaseline.extract_features(
                    z_test, labels_test, test_feat_masks, all_clusters=all_clusters)

                from sklearn.linear_model import LogisticRegression
                clf = LogisticRegression(penalty='l1', solver='saga', max_iter=5000,
                                          random_state=42, C=1.0)
                clf.fit(features_fit, train_resp)
                proba = clf.predict_proba(features_test)
                y_test = np.array([
                    patient_responses[list(unique_pids).index(pid)]
                    for pid in test_pids_sorted
                ])
                y_pred_test = proba[:, 1] if proba.shape[1] == 2 else np.full(len(y_test), 0.5)

            else:
                continue

            fold_metrics = PredictionAnalyzer.compute_metrics(y_test, y_pred_test)
            method_folds[method].append(fold_metrics)
            print(f"  {method} fold {fold+1}: AUROC={fold_metrics['auroc']:.3f}")

            test_pids_sorted = sorted(test_pids)
            for pidx, pid_name in enumerate(test_pids_sorted):
                idx = pid_to_idx[pid_name]
                all_y_true[idx] = patient_responses[list(unique_pids).index(pid_name)]
                if pidx < len(y_pred_test):
                    method_y_pred[method][idx] = y_pred_test[pidx]

    print(f"\n{'='*50}")
    print("Benchmark CV Results")
    print(f"{'='*50}")

    cv_results = {}
    for method in methods:
        folds = method_folds[method]
        if not folds:
            continue
        aurocs = [f['auroc'] for f in folds]
        auprcs = [f['auprc'] for f in folds]
        cv_results[method] = {
            'auroc_mean': float(np.mean(aurocs)),
            'auroc_std': float(np.std(aurocs)),
            'auprc_mean': float(np.mean(auprcs)),
            'auprc_std': float(np.std(auprcs)),
            'per_fold': folds,
        }
        print(f"  {method:12s}: AUROC={np.mean(aurocs):.3f}+/-{np.std(aurocs):.3f}  "
              f"AUPRC={np.mean(auprcs):.3f}+/-{np.std(auprcs):.3f}")

    output = {
        'dataset': args.data,
        'n_folds': args.n_folds,
        'n_patients': n_patients,
        'cv_results': cv_results,
    }
    with open(output_dir / 'benchmark_cv_results.json', 'w') as f:
        json.dump(make_serializable(output), f, indent=2)
    print(f"\nResults saved to {output_dir}/")


def run_transfer(args, config):
    output_dir = Path('outputs') / f'{args.transfer_source}_to_{args.transfer_target}_transfer'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Cross-Dataset Transfer: {args.transfer_source} -> {args.transfer_target} ===")

    source_adata, source_info = load_real_data(args.transfer_source, n_hvg=args.n_hvg,
                                                max_cells=args.max_cells)
    target_adata, target_info = load_real_data(args.transfer_target, n_hvg=args.n_hvg,
                                                max_cells=args.max_cells)

    print(f"  Source: {source_adata.n_obs} cells, {source_adata.n_vars} genes")
    print(f"  Target: {target_adata.n_obs} cells, {target_adata.n_vars} genes")

    source_config = config.copy()
    if source_adata.n_obs > 50_000 and source_config.get('batch_size') is None:
        source_config['batch_size'] = 512

    has_spatial = source_info['has_spatial']
    graph_kwargs = {'r_spatial': source_info.get('r_spatial', 82.5)}
    source_data = prepare_graph_data(source_adata, has_spatial=has_spatial, **graph_kwargs)
    has_response = hasattr(source_data, 'y') and source_data.y is not None

    source_dir = output_dir / 'source'
    source_dir.mkdir(exist_ok=True)
    model = build_model(source_config, source_data, has_response)
    trainer = make_trainer(model, source_config, source_config['device'], source_dir, data=source_data)
    trainer.train(source_data)

    source_downstream = run_downstream(model, source_data, source_config, source_adata, source_dir)
    source_rare_markers = source_downstream.get('rare_cells', {}).get('rare_markers', {})

    print("\n--- Transfer to target ---")
    source_config['input_dim'] = source_data.x.size(1)
    transfer_result, target_ad = CrossDatasetTransfer.transfer_embeddings(
        model, source_adata, target_adata, target_info, source_config)

    if transfer_result is None:
        print(f"  Transfer failed: {target_ad}")
        return

    eval_result = CrossDatasetTransfer.evaluate_transfer(
        transfer_result['z'], transfer_result['mu'], transfer_result['logvar'],
        target_ad, source_rare_markers=source_rare_markers)

    print(f"  Target clustering: {eval_result['clustering']['n_clusters']} clusters, "
          f"silhouette={eval_result['clustering']['silhouette']:.4f}")
    print(f"  Target rare cells: {eval_result['rare_cells']['n_rare']}")

    output = {
        'source': args.transfer_source,
        'target': args.transfer_target,
        'n_shared_genes': transfer_result['n_shared_genes'],
        'source_downstream': source_downstream,
        'transfer_evaluation': eval_result,
    }
    with open(output_dir / 'transfer_results.json', 'w') as f:
        json.dump(make_serializable(output), f, indent=2)
    print(f"\nResults saved to {output_dir}/")


def build_comparison_table(all_results):
    comparison = {}
    metrics_to_compare = ['auroc', 'auprc', 'silhouette', 'n_rare']

    for method, results in all_results.items():
        row = {}
        pred = results.get('prediction', {})
        row['auroc'] = pred.get('auroc', pred.get('pooled_auroc', None))
        row['auprc'] = pred.get('auprc', pred.get('pooled_auprc', None))
        clust = results.get('clustering', {})
        row['silhouette'] = clust.get('silhouette', None)
        row['n_clusters'] = clust.get('n_clusters', None)
        rare = results.get('rare_cells', {})
        row['n_rare'] = rare.get('n_rare', None)
        comparison[method] = row

    return comparison


def print_comparison_table(comparison, methods):
    print(f"\n{'='*70}")
    print("Method Comparison")
    print(f"{'='*70}")
    header = f"{'Method':12s} | {'AUROC':>7s} | {'AUPRC':>7s} | {'Silhouette':>10s} | {'Clusters':>8s} | {'Rare':>6s}"
    print(header)
    print('-' * len(header))
    for method in methods:
        if method not in comparison:
            continue
        row = comparison[method]
        auroc = f"{row['auroc']:.3f}" if row.get('auroc') is not None else '  N/A'
        auprc = f"{row['auprc']:.3f}" if row.get('auprc') is not None else '  N/A'
        sil = f"{row['silhouette']:.4f}" if row.get('silhouette') is not None else '    N/A'
        ncl = f"{row['n_clusters']:>8d}" if row.get('n_clusters') is not None else '     N/A'
        nrare = f"{row['n_rare']:>6d}" if row.get('n_rare') is not None else '   N/A'
        print(f"{method:12s} | {auroc:>7s} | {auprc:>7s} | {sil:>10s} | {ncl:>8s} | {nrare:>6s}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', choices=['local', 'full'], default='local')
    parser.add_argument('--data', choices=list(DATASETS.keys()), default='melanoma')
    parser.add_argument('--max-cells', type=int, default=None)
    parser.add_argument('--n-hvg', type=int, default=2000)
    parser.add_argument('--methods', type=str, default='gvae,scvi,scanpy')
    parser.add_argument('--cv', action='store_true')
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--transfer-source', type=str, default=None)
    parser.add_argument('--transfer-target', type=str, default=None)
    args = parser.parse_args()

    config = CONFIGS[args.config].copy()
    if args.data in DATASETS:
        dataset_overrides = {k: v for k, v in DATASETS[args.data].items()
                             if k in ('epochs_phase1', 'epochs_phase2')}
        config.update(dataset_overrides)
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size

    if args.transfer_source and args.transfer_target:
        run_transfer(args, config)
    elif args.cv:
        run_benchmark_cv(args, config)
    else:
        run_benchmark(args, config)


if __name__ == '__main__':
    main()
