import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model import ResponsePredictor
from src.ablations import _choose_cv_splitter


def build_patient_masks(adata):
    pids = adata.obs['patient_id'].values
    unique = np.array(sorted(np.unique(pids)))
    masks = [torch.tensor(pids == pid, dtype=torch.bool) for pid in unique]
    y = np.array([
        int(adata.obs.loc[adata.obs['patient_id'] == pid, 'response'].iloc[0])
        for pid in unique
    ])
    return masks, y, unique


def train_eval_one_fold(z, masks, y, train_idx, test_idx, latent_dim, lr,
                        epochs, dropout, weight_decay, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_masks = [masks[i] for i in train_idx]
    test_masks = [masks[i] for i in test_idx]
    y_train = torch.tensor(y[train_idx], dtype=torch.float32, device=device)
    y_test = y[test_idx]

    model = ResponsePredictor(latent_dim=latent_dim, dropout=dropout).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCELoss()
    z_dev = z.to(device)
    train_masks_dev = [m.to(device) for m in train_masks]
    test_masks_dev = [m.to(device) for m in test_masks]

    best_train_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        preds, _ = model(z_dev, train_masks_dev)
        preds = preds.clamp(1e-6, 1 - 1e-6)
        loss = bce(preds, y_train)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if loss.item() < best_train_loss:
            best_train_loss = loss.item()

    model.eval()
    with torch.no_grad():
        test_preds, _ = model(z_dev, test_masks_dev)
    test_preds_np = test_preds.cpu().numpy()
    if len(np.unique(y_test)) < 2:
        auroc = float('nan')
        auprc = float('nan')
    else:
        auroc = roc_auc_score(y_test, test_preds_np)
        auprc = average_precision_score(y_test, test_preds_np)
    return float(auroc), float(auprc), test_preds_np, y_test


def run_ablation(adata_path, n_pca, lr, epochs, dropout, weight_decay,
                 n_folds, seed, device, output):
    print(f"Loading {adata_path}")
    adata = sc.read_h5ad(adata_path)
    print(f"  {adata.n_obs} cells x {adata.n_vars} genes")

    if 'X_pca' not in adata.obsm or adata.obsm['X_pca'].shape[1] < n_pca:
        n_comps = min(n_pca, adata.n_obs - 1, adata.n_vars - 1)
        print(f"  Computing PCA ({n_comps} components)")
        sc.pp.pca(adata, n_comps=n_comps)

    z_pca = adata.obsm['X_pca'][:, :n_pca]
    z = torch.tensor(z_pca, dtype=torch.float32)
    print(f"  PCA features: {z.shape}")

    masks, y, unique_pids = build_patient_masks(adata)
    n_patients = len(unique_pids)
    print(f"  {n_patients} patients, response prevalence={y.mean():.3f}")

    splitter = _choose_cv_splitter(n_patients, n_folds, seed)

    fold_aurocs, fold_auprcs = [], []
    pooled_preds, pooled_y = [], []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(np.arange(n_patients), y)):
        auroc, auprc, preds, y_test = train_eval_one_fold(
            z, masks, y, train_idx, test_idx,
            latent_dim=n_pca, lr=lr, epochs=epochs, dropout=dropout,
            weight_decay=weight_decay, seed=seed + fold, device=device)
        valid = ~(np.isnan(auroc) or np.isnan(auprc))
        if valid:
            fold_aurocs.append(auroc)
            fold_auprcs.append(auprc)
            pooled_preds.append(preds)
            pooled_y.append(y_test)
            print(f"  fold {fold}: AUROC={auroc:.3f} AUPRC={auprc:.3f} "
                  f"(n_train={len(train_idx)} n_test={len(test_idx)})")
        else:
            print(f"  fold {fold}: skipped (single-class test fold)")

    if pooled_preds:
        pooled_preds_arr = np.concatenate(pooled_preds)
        pooled_y_arr = np.concatenate(pooled_y)
        pooled_auroc = float(roc_auc_score(pooled_y_arr, pooled_preds_arr))
        pooled_auprc = float(average_precision_score(pooled_y_arr, pooled_preds_arr))
    else:
        pooled_auroc = float('nan')
        pooled_auprc = float('nan')

    summary = {
        'method': 'scanpy_pca_attnpool_endtoend',
        'adata_path': str(adata_path),
        'n_patients': int(n_patients),
        'n_cells': int(adata.n_obs),
        'n_pca': int(n_pca),
        'n_folds_valid': len(fold_aurocs),
        'pooled_auroc': pooled_auroc,
        'pooled_auprc': pooled_auprc,
        'mean_auroc': float(np.mean(fold_aurocs)) if fold_aurocs else float('nan'),
        'std_auroc': float(np.std(fold_aurocs)) if fold_aurocs else float('nan'),
        'fold_aurocs': fold_aurocs,
        'config': {'lr': lr, 'epochs': epochs, 'dropout': dropout,
                   'weight_decay': weight_decay, 'n_folds': n_folds, 'seed': seed},
    }

    print()
    print("=" * 60)
    print(f"Scanpy-PCA + Attention-Pooling (end-to-end)")
    print("=" * 60)
    print(f"  pooled AUROC: {pooled_auroc:.3f}")
    print(f"  pooled AUPRC: {pooled_auprc:.3f}")
    if fold_aurocs:
        print(f"  mean AUROC:   {np.mean(fold_aurocs):.3f} +/- {np.std(fold_aurocs):.3f}")
        print(f"  per-fold:     {[round(a, 3) for a in fold_aurocs]}")
    print()

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"Wrote {out_path}")

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True,
                    help='Path to AnnData .h5ad with patient_id and response in obs')
    ap.add_argument('--n_pca', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--dropout', type=float, default=0.3)
    ap.add_argument('--weight_decay', type=float, default=1e-5)
    ap.add_argument('--n_folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--output', default=None)
    args = ap.parse_args()

    run_ablation(args.data, args.n_pca, args.lr, args.epochs, args.dropout,
                 args.weight_decay, args.n_folds, args.seed, args.device,
                 args.output)


if __name__ == '__main__':
    main()
