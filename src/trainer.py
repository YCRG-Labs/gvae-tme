import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
from collections import defaultdict
import numpy as np
from pathlib import Path

class GVAELoss:
    @staticmethod
    def zinb(x, rho, theta, pi, eps=1e-8):
        theta = torch.clamp(theta, min=eps, max=1e6)
        rho = torch.clamp(rho, min=eps)
        pi = torch.clamp(pi, min=eps, max=1 - eps)
        theta_rho = theta + rho
        zero_nb = torch.pow(theta / theta_rho, theta)
        zero_case = torch.log(pi + (1.0 - pi) * zero_nb + eps)
        log_gamma_x_theta = torch.lgamma(x + theta)
        log_gamma_theta = torch.lgamma(theta)
        log_gamma_x1 = torch.lgamma(x + 1.0)
        nb_case = (torch.log(1.0 - pi + eps) + log_gamma_x_theta - log_gamma_theta - log_gamma_x1 +
                   theta * torch.log(theta / theta_rho + eps) + x * torch.log(rho / theta_rho + eps))
        mask = (x < 0.5).float()
        log_lik = mask * zero_case + (1.0 - mask) * nb_case
        return -log_lik.mean()

    @staticmethod
    def adjacency_negsampling(pos_scores, neg_scores):
        pos_loss = -torch.log(pos_scores + 1e-8).mean()
        neg_loss = -torch.log(1.0 - neg_scores + 1e-8).mean()
        return pos_loss + neg_loss

    @staticmethod
    def kl_divergence(mu, logvar):
        kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl.mean()

    @staticmethod
    def contrastive(z, pos_pairs, neg_pairs, temperature=0.1):
        if pos_pairs.size(0) == 0 or neg_pairs.size(0) == 0:
            return torch.tensor(0.0, device=z.device)
        z_norm = F.normalize(z, p=2, dim=1)
        anchor_to_pos = defaultdict(list)
        for idx in range(pos_pairs.size(0)):
            a = pos_pairs[idx, 0].item()
            p = pos_pairs[idx, 1].item()
            anchor_to_pos[a].append(p)
        anchor_to_neg = defaultdict(list)
        for idx in range(neg_pairs.size(0)):
            a = neg_pairs[idx, 0].item()
            n = neg_pairs[idx, 1].item()
            anchor_to_neg[a].append(n)
        valid_anchors = set(anchor_to_pos.keys()) & set(anchor_to_neg.keys())
        if not valid_anchors:
            return torch.tensor(0.0, device=z.device)
        loss = torch.tensor(0.0, device=z.device)
        n_terms = 0
        for a in valid_anchors:
            z_a = z_norm[a]
            pos_indices = torch.tensor(anchor_to_pos[a], device=z.device, dtype=torch.long)
            neg_indices = torch.tensor(anchor_to_neg[a], device=z.device, dtype=torch.long)
            pos_sims = (z_norm[pos_indices] * z_a.unsqueeze(0)).sum(dim=1) / temperature
            neg_sims = (z_norm[neg_indices] * z_a.unsqueeze(0)).sum(dim=1) / temperature
            neg_logsumexp = torch.logsumexp(neg_sims, dim=0)
            for ps in pos_sims:
                loss -= ps - torch.logaddexp(ps, neg_logsumexp)
                n_terms += 1
        return loss / max(n_terms, 1)

    @staticmethod
    def prediction(y_true, y_pred):
        return F.binary_cross_entropy(y_pred, y_true.float())

class Trainer:
    def __init__(self, model, config, device='cpu', checkpoint_dir=None):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.lambda1 = config.get('lambda1', 1.0)
        self.lambda2 = config.get('lambda2', 0.5)
        self.beta_target = config.get('beta', 0.01)
        self.beta = 0.0
        self.gamma = 0.0
        self.gamma_phase2 = config.get('gamma', 0.1)
        self.lr = config.get('lr', 1e-3)
        self.lr_phase2 = self.lr / 10.0
        self.epochs_phase1 = config.get('epochs_phase1', 300)
        self.epochs_phase2 = config.get('epochs_phase2', 200)
        self.patience = config.get('patience', 50)
        self.beta_warmup_epochs = config.get('beta_warmup_epochs', 50)
        self.tolerance = 1.1
        self.max_gamma_reductions = 3
        self.phase1_metrics = {}
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        self.loss_fn = GVAELoss()
        self.optimizer = None
        self.scheduler = None
        self.checkpoint_dir = None
        if checkpoint_dir is not None:
            self.checkpoint_dir = Path(checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, suffix: str) -> None:
        if self.checkpoint_dir is None:
            return
        checkpoint_path = self.checkpoint_dir / f"model_{suffix}.pt"
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "phase": suffix,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved model checkpoint to {checkpoint_path}")

    def setup_optimizer(self, phase=1):
        lr = self.lr if phase == 1 else self.lr_phase2
        epochs = self.epochs_phase1 if phase == 1 else self.epochs_phase2
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)

    def get_beta(self, epoch):
        if epoch >= self.beta_warmup_epochs:
            return self.beta_target
        return self.beta_target * (epoch / self.beta_warmup_epochs)

    def compute_loss(self, outputs, data, phase=1, epoch=1):
        L_adj = self.loss_fn.adjacency_negsampling(outputs['pos_scores'], outputs['neg_scores'])
        x_raw = data.x_raw if hasattr(data, 'x_raw') else data.x
        L_expr = self.loss_fn.zinb(x_raw, outputs['rho'], outputs['theta'], outputs['pi'])
        beta = self.get_beta(epoch) if phase == 1 else self.beta_target
        L_kl = self.loss_fn.kl_divergence(outputs['mu'], outputs['logvar'])
        L_contrast = torch.tensor(0.0, device=self.device)
        if hasattr(data, 'pos_pairs') and hasattr(data, 'neg_pairs'):
            if data.pos_pairs.size(0) > 0 and data.neg_pairs.size(0) > 0:
                L_contrast = self.loss_fn.contrastive(outputs['z'], data.pos_pairs, data.neg_pairs)
        L_pred = torch.tensor(0.0, device=self.device)
        if phase == 2 and 'y_pred' in outputs and hasattr(data, 'y'):
            L_pred = self.loss_fn.prediction(data.y, outputs['y_pred'])
        total = L_adj + self.lambda1 * L_expr + self.lambda2 * L_contrast + beta * L_kl + self.gamma * L_pred
        return {
            'total': total,
            'adj': L_adj.item(),
            'expr': L_expr.item(),
            'kl': L_kl.item(),
            'contrast': L_contrast.item(),
            'pred': L_pred.item(),
            'beta': beta,
        }

    @torch.no_grad()
    def evaluate(self, data):
        self.model.eval()
        outputs = self.model(data)
        L_adj = self.loss_fn.adjacency_negsampling(outputs['pos_scores'], outputs['neg_scores'])
        x_raw = data.x_raw if hasattr(data, 'x_raw') else data.x
        L_expr = self.loss_fn.zinb(x_raw, outputs['rho'], outputs['theta'], outputs['pi'])
        return {'loss_adj': L_adj.item(), 'loss_expr': L_expr.item()}

    def train_epoch(self, data, phase=1, epoch=1):
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(data)
        losses = self.compute_loss(outputs, data, phase, epoch)
        losses['total'].backward()
        clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        return losses

    def train_phase1(self, data):
        print("=== Phase 1: Representation Learning ===")
        self.gamma = 0.0
        self.setup_optimizer(phase=1)
        best_val_loss = float('inf')
        patience_counter = 0
        for epoch in range(1, self.epochs_phase1 + 1):
            losses = self.train_epoch(data, phase=1, epoch=epoch)
            if epoch % 10 == 0:
                val = self.evaluate(data)
                val_loss = val['loss_adj'] + val['loss_expr']
                print(f"Epoch {epoch:3d} | L_adj={losses['adj']:.4f} L_expr={losses['expr']:.4f} L_kl={losses['kl']:.4f} beta={losses['beta']:.5f} Val={val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self._best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                if patience_counter >= self.patience // 10:
                    print(f"Early stopping at epoch {epoch}")
                    break
        if hasattr(self, '_best_state'):
            self.model.load_state_dict(self._best_state)
        final_val = self.evaluate(data)
        self.phase1_metrics = {'loss_adj': final_val['loss_adj'], 'loss_expr': final_val['loss_expr']}
        print(f"Phase 1 complete: L_adj={final_val['loss_adj']:.4f}, L_expr={final_val['loss_expr']:.4f}")
        self.save_checkpoint("phase1")
        return self.phase1_metrics

    def train_phase2(self, data):
        print("\n=== Phase 2: Joint Clinical Fine-Tuning ===")
        self.gamma = self.gamma_phase2
        self.setup_optimizer(phase=2)
        best_val_loss = float('inf')
        patience_counter = 0
        gamma_reductions = 0
        for epoch in range(1, self.epochs_phase2 + 1):
            losses = self.train_epoch(data, phase=2, epoch=epoch)
            if epoch % 10 == 0:
                val = self.evaluate(data)
                val_loss = val['loss_adj'] + val['loss_expr']
                adj_degraded = val['loss_adj'] > self.tolerance * self.phase1_metrics['loss_adj']
                expr_degraded = val['loss_expr'] > self.tolerance * self.phase1_metrics['loss_expr']
                if adj_degraded or expr_degraded:
                    gamma_reductions += 1
                    self.gamma /= 2.0
                    print(f"  Reconstruction degraded (reduction {gamma_reductions}/{self.max_gamma_reductions})")
                    print(f"  gamma -> {self.gamma:.6f}")
                    if gamma_reductions >= self.max_gamma_reductions:
                        print(" Cannot maintain reconstruction quality. Stopping Phase 2.")
                        break
                print(f"Epoch {epoch:3d} | L_adj={losses['adj']:.4f} L_expr={losses['expr']:.4f} L_pred={losses['pred']:.4f} gamma={self.gamma:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= self.patience // 10:
                    print(f"Early stopping at epoch {epoch}")
                    break
        self.save_checkpoint("phase2")
        print("Phase 2 complete")

    def train(self, data):
        data = data.to(self.device)
        phase1_metrics = self.train_phase1(data)
        if self.model.use_predictor and hasattr(data, 'y'):
            self.train_phase2(data)
        return phase1_metrics
