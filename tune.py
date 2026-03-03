"""Bayesian hyperparameter optimization via Optuna (Appendix A of paper).

TPE sampler, 50 trials per dataset (10 random + 40 guided), 100 epochs each
with patience 20. Top-3 configs retrained at full budget.

Usage:
    python tune.py --data melanoma --config full --n-trials 50
    python tune.py --data synthetic --config local --n-trials 3  # smoke test
"""
import sys
import argparse
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("ERROR: optuna not installed. Run: pip install optuna>=3.0")
    sys.exit(1)

from src.model import GVAEModel
from src.trainer import Trainer
from src.data_utils import prepare_graph_data, create_synthetic_data
from src.config import CONFIGS
from train import load_real_data, DATASETS, make_serializable


def build_trial_config(trial, base_config):
    """Sample hyperparameters for a single Optuna trial."""
    config = base_config.copy()
    config['latent_dim'] = trial.suggest_categorical('latent_dim', [16, 32, 64])
    config['n_heads'] = trial.suggest_categorical('n_heads', [1, 2, 4, 8])
    config['lambda1'] = trial.suggest_categorical('lambda1', [0.1, 0.5, 1.0, 2.0])
    config['lambda2'] = trial.suggest_categorical('lambda2', [0.1, 0.5, 1.0])
    config['beta'] = trial.suggest_categorical('beta', [0.001, 0.01, 0.1])
    config['gamma'] = trial.suggest_categorical('gamma', [0.01, 0.05, 0.1, 0.5])
    config['n_neg_samples'] = trial.suggest_categorical('n_neg_samples', [5, 10, 20])
    # Hidden dim must be divisible by n_heads
    config['hidden_dim'] = config['n_heads'] * (config['latent_dim'] // config['n_heads'] or 1)
    config['hidden_dim'] = max(config['hidden_dim'], config['n_heads'] * 4)
    # Reduced budget for trials
    config['epochs_phase1'] = 100
    config['epochs_phase2'] = 0
    config['patience'] = 20
    return config


def objective(trial, base_config, data, device):
    """Optuna objective: validation reconstruction loss after Phase 1."""
    config = build_trial_config(trial, base_config)
    config['device'] = device

    model = GVAEModel(
        n_features=data.x.size(1),
        n_genes=data.x_raw.size(1),
        hidden_dim=config['hidden_dim'],
        latent_dim=config['latent_dim'],
        n_heads=config['n_heads'],
        dropout=config['dropout'],
        n_neg_samples=config['n_neg_samples'],
        use_predictor=False,
    )
    trainer = Trainer(model, config, device=device)

    try:
        trainer.train(data)
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            torch.cuda.empty_cache()
            return float('inf')
        raise

    val = trainer.evaluate(data.to(device))
    val_loss = (val['loss_adj'] +
                config['lambda1'] * val['loss_expr'])
    return val_loss


def retrain_top_k(top_configs, data, base_config, device, output_dir, k=3):
    """Retrain top-k configs at full epoch budget."""
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

        retrain_dir = output_dir / f'retrain_{i}'
        retrain_dir.mkdir(exist_ok=True)

        model = GVAEModel(
            n_features=data.x.size(1),
            n_genes=data.x_raw.size(1),
            hidden_dim=config['hidden_dim'],
            latent_dim=config['latent_dim'],
            n_heads=config['n_heads'],
            dropout=config['dropout'],
            n_neg_samples=config['n_neg_samples'],
            use_predictor=False,
        )
        trainer = Trainer(model, config, device=device, checkpoint_dir=retrain_dir)
        phase1 = trainer.train(data)
        val = trainer.evaluate(data.to(device))
        val_loss = val['loss_adj'] + config['lambda1'] * val['loss_expr']

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
    parser.add_argument('--config', choices=['local', 'full'], default='local')
    parser.add_argument('--data', choices=list(DATASETS.keys()) + ['synthetic'], default='synthetic')
    parser.add_argument('--max-cells', type=int, default=None)
    parser.add_argument('--n-hvg', type=int, default=2000)
    parser.add_argument('--n-trials', type=int, default=50)
    parser.add_argument('--n-retrain', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    base_config = CONFIGS[args.config].copy()
    device = base_config['device']
    output_dir = Path('outputs') / f'{args.data}_tune'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Optuna HPO | Data: {args.data} | Trials: {args.n_trials} | Device: {device}")

    # Load data
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

    # Create study: 10 random startup trials + guided
    sampler = optuna.samplers.TPESampler(
        seed=args.seed,
        n_startup_trials=min(10, args.n_trials),
    )
    study = optuna.create_study(direction='minimize', sampler=sampler)

    def _objective(trial):
        return objective(trial, base_config, data, device)

    print(f"\nRunning {args.n_trials} trials...")
    study.optimize(_objective, n_trials=args.n_trials, show_progress_bar=True)

    # Collect results
    print(f"\n{'='*50}")
    print("Optimization Complete")
    print(f"{'='*50}")
    print(f"  Best val_loss: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    # Extract top-k trial configs
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('inf'))
    top_configs = []
    for t in sorted_trials[:args.n_retrain]:
        config = build_trial_config(t, base_config)
        top_configs.append((config, t.value))

    # Retrain top configs at full budget
    retrain_results = retrain_top_k(top_configs, data, base_config, device,
                                     output_dir, k=args.n_retrain)

    # Save study results
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
