import sys
import argparse
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from sklearn.metrics import silhouette_score as sil_score, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("ERROR: optuna not installed. Run: pip install optuna>=3.0")
    sys.exit(1)

from src.model import GVAEModel
from src.trainer import Trainer, GVAELoss
from src.data_utils import prepare_graph_data, create_synthetic_data
from src.config import CONFIGS
from train import load_real_data, DATASETS, make_serializable


def build_trial_config(trial, base_config):
    config = base_config.copy()
    config['latent_dim'] = trial.suggest_categorical('latent_dim', [16, 32, 64])
    config['n_heads'] = trial.suggest_categorical('n_heads', [1, 2, 4, 8])
    config['lambda1'] = trial.suggest_categorical('lambda1', [0.1, 0.5, 1.0, 2.0])
    config['lambda2'] = trial.suggest_categorical('lambda2', [0.1, 0.5, 1.0])
    config['beta'] = trial.suggest_categorical('beta', [0.001, 0.01, 0.1])
    config['gamma'] = trial.suggest_categorical('gamma', [0.01, 0.05, 0.1, 0.5])
    config['n_neg_samples'] = trial.suggest_categorical('n_neg_samples', [5, 10, 20])
    config['temperature'] = trial.suggest_categorical('temperature', [0.05, 0.1, 0.2, 0.5])
    config['rare_threshold'] = trial.suggest_categorical('rare_threshold', [1.5, 2.0, 2.5, 3.0])
    config['k_mol'] = trial.suggest_categorical('k_mol', [10, 15, 20, 30])
    config['free_bits'] = trial.suggest_categorical('free_bits', [0.0, 0.25, 0.5, 1.0])
    config['hidden_dim'] = config['n_heads'] * (config['latent_dim'] // config['n_heads'] or 1)
    config['hidden_dim'] = max(config['hidden_dim'], config['n_heads'] * 4)

    config['epochs_phase1'] = 100
    config['epochs_phase2'] = 0
    config['patience'] = 20
    return config


def _auroc_objective(z_np, data, n_folds=5, seed=0):
    """Patient-level AUROC via 5-fold CV on frozen embeddings.

    Uses src/ablations.py:_choose_cv_splitter for small-cohort-aware CV
    (LOO for n<20, RepeatedStratifiedKFold for n<50, StratifiedKFold otherwise).
    Returns NaN if labels absent or only one class present.

    This is the objective that Optuna minimizes (as -AUROC in objective()).
    The full CV here matches the evaluation protocol used in train.py/cross-
    validation mode, so HPO-selected configs actually transfer to the final
    CV run instead of being single-split artifacts.
    """
    if not hasattr(data, 'y') or data.y is None:
        return float('nan')
    if not hasattr(data, 'patient_masks'):
        return float('nan')

    y = data.y.cpu().numpy() if hasattr(data.y, 'cpu') else np.asarray(data.y)
    if len(np.unique(y)) < 2:
        return float('nan')

    from sklearn.linear_model import LogisticRegression
    from src.ablations import _choose_cv_splitter

    masks = data.patient_masks
    n_patients = len(y)

    features = []
    for mask in masks:
        mask_np = mask.cpu().numpy() if hasattr(mask, 'cpu') else np.asarray(mask)
        z_pat = z_np[mask_np]
        if len(z_pat) == 0:
            features.append(np.zeros(z_np.shape[1]))
        else:
            features.append(z_pat.mean(axis=0))
    X = np.array(features)

    splitter, _ = _choose_cv_splitter(n_patients, n_folds, seed)
    try:
        pooled_true, pooled_pred = [], []
        for tr, te in splitter.split(X, y):
            clf = LogisticRegression(penalty='l1', solver='saga',
                                     max_iter=2000, random_state=seed, C=1.0)
            clf.fit(X[tr], y[tr])
            proba = clf.predict_proba(X[te])
            preds = proba[:, 1] if proba.shape[1] == 2 else np.full(len(te), 0.5)
            pooled_true.extend(y[te].tolist())
            pooled_pred.extend(preds.tolist())
        return float(roc_auc_score(pooled_true, pooled_pred))
    except (ValueError, Exception):
        return float('nan')


def objective(trial, base_config, data, device, adata=None, has_spatial=True,
              graph_kwargs=None, tune_for='auroc'):
    """Optuna objective.

    tune_for:
      'auroc'  — use patient-level AUROC on held-out fold when labels are
                 available. Falls back to val_loss if no labels.
      'recon'  — legacy objective: val_loss - 0.1*silhouette. Does NOT need labels.

    Returns a value Optuna minimizes. For AUROC mode, we return -AUROC so that
    "lower is better" matches the study direction.
    """
    config = build_trial_config(trial, base_config)
    config['device'] = device

    trial_data = data
    k_mol = config.get('k_mol', 15)
    if k_mol != 15 and adata is not None:
        trial_data = prepare_graph_data(
            adata, has_spatial=has_spatial, k_mol=k_mol, **(graph_kwargs or {}))

    has_labels = hasattr(trial_data, 'y') and trial_data.y is not None
    use_predictor = tune_for == 'auroc' and has_labels

    model = GVAEModel(
        n_features=trial_data.x.size(1),
        n_genes=trial_data.x_raw.size(1),
        hidden_dim=config['hidden_dim'],
        latent_dim=config['latent_dim'],
        n_heads=config['n_heads'],
        dropout=config['dropout'],
        n_neg_samples=config['n_neg_samples'],
        use_predictor=use_predictor,
    )
    trainer = Trainer(model, config, device=device)

    try:
        trainer.train(trial_data)
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            torch.cuda.empty_cache()
            return float('inf')
        raise

    model.eval()
    data_dev = trial_data.to(device)
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(0)
    with torch.no_grad():
        outputs = model(data_dev)
        loss_fn = GVAELoss()
        x_raw = data_dev.x_raw if hasattr(data_dev, 'x_raw') else data_dev.x
        L_adj = loss_fn.adjacency_negsampling(outputs['pos_scores'], outputs['neg_scores']).item()
        if 'expr_mu' in outputs:
            L_expr = loss_fn.gaussian(x_raw, outputs['expr_mu'], outputs['expr_logvar']).item()
        else:
            L_expr = loss_fn.zinb(x_raw, outputs['rho'], outputs['theta'], outputs['pi']).item()
        L_kl = loss_fn.kl_divergence(outputs['mu'], outputs['logvar']).item()
        L_contrast = 0.0
        if hasattr(data_dev, 'pos_pairs') and hasattr(data_dev, 'neg_pairs'):
            if data_dev.pos_pairs.size(0) > 0 and data_dev.neg_pairs.size(0) > 0:
                L_contrast = loss_fn.contrastive(
                    outputs['z'], data_dev.pos_pairs, data_dev.neg_pairs,
                    temperature=config.get('temperature', 0.1)).item()
    torch.random.set_rng_state(rng_state)
    val_loss = (L_adj +
                config['lambda1'] * L_expr +
                config['lambda2'] * L_contrast +
                config['beta'] * L_kl)

    z_np = outputs['z'].cpu().numpy()

    # Always log silhouette for diagnostics
    n_clusters = min(8, max(2, len(z_np) // 50))
    try:
        labels = KMeans(n_clusters=n_clusters, n_init=3, random_state=0).fit_predict(z_np)
        sil = sil_score(z_np, labels, sample_size=min(5000, len(z_np))) \
            if len(set(labels)) > 1 else 0.0
    except Exception:
        sil = 0.0

    trial.set_user_attr('val_loss', float(val_loss))
    trial.set_user_attr('silhouette', float(sil))
    trial.set_user_attr('L_adj', L_adj)
    trial.set_user_attr('L_expr', L_expr)
    trial.set_user_attr('L_contrast', L_contrast)
    trial.set_user_attr('L_kl', L_kl)

    if tune_for == 'auroc' and has_labels:
        auroc = _auroc_objective(z_np, data_dev)
        trial.set_user_attr('auroc', auroc)
        if np.isnan(auroc):
            # Fall back to val_loss if the AUROC computation couldn't be done
            return val_loss - 0.1 * sil
        return -auroc  # minimize negative AUROC

    return val_loss - 0.1 * sil


def retrain_top_k(top_configs, data, base_config, device, output_dir, k=3,
                  adata=None, has_spatial=True, graph_kwargs=None):
    results = []
    full_epochs = base_config.get('epochs_phase1', 300)

    for i, (trial_config, trial_val) in enumerate(top_configs[:k]):
        print(f"\n{'='*50}")
        print(f"Retraining config {i+1}/{k} (trial val_loss={trial_val:.4f})")
        print(f"{'='*50}")

        config = trial_config.copy()
        config['epochs_phase1'] = full_epochs
        config['patience'] = base_config.get('patience', 50)
        config['device'] = device

        trial_data = data
        k_mol = config.get('k_mol', 15)
        if k_mol != 15 and adata is not None:
            trial_data = prepare_graph_data(
                adata, has_spatial=has_spatial, k_mol=k_mol, **(graph_kwargs or {}))

        retrain_dir = output_dir / f'retrain_{i}'
        retrain_dir.mkdir(exist_ok=True)

        model = GVAEModel(
            n_features=trial_data.x.size(1),
            n_genes=trial_data.x_raw.size(1),
            hidden_dim=config['hidden_dim'],
            latent_dim=config['latent_dim'],
            n_heads=config['n_heads'],
            dropout=config['dropout'],
            n_neg_samples=config['n_neg_samples'],
            use_predictor=False,
        )
        trainer = Trainer(model, config, device=device, checkpoint_dir=retrain_dir)
        phase1 = trainer.train(trial_data)
        val = trainer.evaluate(trial_data.to(device))
        model.eval()
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(0)
        with torch.no_grad():
            data_dev = trial_data.to(device)
            outputs = model(data_dev)
            loss_fn = GVAELoss()
            L_kl = loss_fn.kl_divergence(outputs['mu'], outputs['logvar']).item()
            L_contrast = 0.0
            if hasattr(data_dev, 'pos_pairs') and hasattr(data_dev, 'neg_pairs'):
                if data_dev.pos_pairs.size(0) > 0 and data_dev.neg_pairs.size(0) > 0:
                    L_contrast = loss_fn.contrastive(
                        outputs['z'], data_dev.pos_pairs, data_dev.neg_pairs,
                        temperature=config.get('temperature', 0.1)).item()
        torch.random.set_rng_state(rng_state)
        val_loss = (val['loss_adj'] +
                    config['lambda1'] * val['loss_expr'] +
                    config['lambda2'] * L_contrast +
                    config['beta'] * L_kl)

        results.append({
            'rank': i,
            'config': {k_: v for k_, v in config.items()
                       if isinstance(v, (int, float, str, bool, list))},
            'trial_val_loss': trial_val,
            'retrain_val_loss': val_loss,
            'phase1_metrics': phase1,
        })
        torch.save(model.state_dict(), retrain_dir / 'model.pt')
        print(f"  Retrain val_loss: {val_loss:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Optuna hyperparameter tuning')
    parser.add_argument('--config', choices=['local', 'full', 'tuned_melanoma'], default='local')
    parser.add_argument('--data', choices=list(DATASETS.keys()) + ['synthetic'], default='synthetic')
    parser.add_argument('--max-cells', type=int, default=None)
    parser.add_argument('--n-hvg', type=int, default=2000)
    parser.add_argument('--n-trials', type=int, default=50)
    parser.add_argument('--n-retrain', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--objective', choices=['auroc', 'recon'], default='auroc',
                        help="'auroc' tunes for patient-level response AUROC (needs labels); "
                             "'recon' tunes for reconstruction+silhouette (legacy, label-free).")
    args = parser.parse_args()

    base_config = CONFIGS[args.config].copy()
    device = base_config['device']
    output_dir = Path('outputs') / f'{args.data}_tune'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Optuna HPO | Data: {args.data} | Trials: {args.n_trials} | "
          f"Objective: {args.objective} | Device: {device}")

    if args.data == 'synthetic':
        adata = create_synthetic_data(
            base_config['n_cells'], base_config['n_genes'],
            base_config['n_patients'],
            n_cell_types=base_config.get('n_cell_types', 8),
        )
        has_spatial = True
        graph_kwargs = {}
    else:
        adata, info = load_real_data(args.data, n_hvg=args.n_hvg, max_cells=args.max_cells)
        has_spatial = info['has_spatial']
        graph_kwargs = {'r_spatial': info.get('r_spatial', 82.5)}

    print(f"  {adata.n_obs} cells, {adata.n_vars} genes")
    print("Building graph...")
    data = prepare_graph_data(adata, has_spatial=has_spatial, **graph_kwargs)
    print(f"  {data.edge_index.size(1)} edges")

    sampler = optuna.samplers.TPESampler(
        seed=args.seed,
        n_startup_trials=min(10, args.n_trials),
    )
    study = optuna.create_study(direction='minimize', sampler=sampler)

    def _objective(trial):
        return objective(trial, base_config, data, device,
                         adata=adata, has_spatial=has_spatial, graph_kwargs=graph_kwargs,
                         tune_for=args.objective)

    print(f"\nRunning {args.n_trials} trials...")
    study.optimize(_objective, n_trials=args.n_trials, show_progress_bar=True)

    print(f"\n{'='*50}")
    print("Optimization Complete")
    print(f"{'='*50}")
    print(f"  Best val_loss: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('inf'))
    top_configs = []
    for t in sorted_trials[:args.n_retrain]:
        config = build_trial_config(t, base_config)
        top_configs.append((config, t.value))

    retrain_results = retrain_top_k(top_configs, data, base_config, device,
                                     output_dir, k=args.n_retrain,
                                     adata=adata, has_spatial=has_spatial,
                                     graph_kwargs=graph_kwargs)

    all_trials = []
    for t in study.trials:
        all_trials.append({
            'number': t.number,
            'value': t.value,
            'params': t.params,
            'state': str(t.state),
        })

    results = {
        'dataset': args.data,
        'n_trials': args.n_trials,
        'objective': args.objective,
        'objective_note': (
            'Minimized -AUROC (so best_value = -best AUROC).' if args.objective == 'auroc'
            else 'Minimized val_loss - 0.1*silhouette (legacy, NOT tuned for AUROC).'
        ),
        'best_value': study.best_value,
        'best_params': study.best_params,
        'all_trials': all_trials,
        'retrain_results': retrain_results,
    }
    with open(output_dir / 'study.json', 'w') as f:
        json.dump(make_serializable(results), f, indent=2)

    print(f"\nResults saved to {output_dir}/")
    print(f"Best config: {study.best_params}")


if __name__ == '__main__':
    main()
