import torch

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

LOCAL = {
    'n_cells': 500,
    'n_genes': 200,
    'n_patients': 10,
    'n_cell_types': 5,
    'hidden_dim': 32,
    'latent_dim': 16,
    'n_heads': 2,
    'dropout': 0.2,
    'n_neg_samples': 3,
    'lambda1': 1.0,
    'lambda2': 0.5,
    'beta': 0.01,
    'beta_warmup_epochs': 10,
    'gamma': 0.1,
    'lr': 1e-3,
    'epochs_phase1': 30,
    'epochs_phase2': 15,
    'patience': 20,
    'max_grad_norm': 1.0,
    'device': get_device(),
}

FULL = {
    'n_cells': 5000,
    'n_genes': 2000,
    'n_patients': 10,
    'n_cell_types': 8,
    'hidden_dim': 64,
    'latent_dim': 32,
    'n_heads': 4,
    'dropout': 0.2,
    'n_neg_samples': 5,
    'lambda1': 1.0,
    'lambda2': 0.5,
    'beta': 0.01,
    'beta_warmup_epochs': 50,
    'gamma': 0.1,
    'lr': 1e-3,
    'epochs_phase1': 300,
    'epochs_phase2': 200,
    'patience': 50,
    'max_grad_norm': 1.0,
    'device': get_device(),
}

CONFIGS = {'local': LOCAL, 'full': FULL}
