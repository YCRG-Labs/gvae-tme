import torch
import torch.optim as optim
import numpy as np
from torch.nn.utils import clip_grad_norm_


class GVAELoss:
    """Loss functions for GVAE training."""
    
    @staticmethod
    def zinb(x, rho, theta, pi, eps=1e-8):
        """Zero-inflated negative binomial loss (Eq. 15)."""
        theta = torch.clamp(theta, min=eps, max=1e6)
        rho = torch.clamp(rho, min=eps)
        pi = torch.clamp(pi, min=eps, max=1-eps)
        
        theta_plus_rho = theta + rho
        zero_case = torch.log(pi + (1 - pi) * torch.pow(theta / theta_plus_rho, theta))
        
        log_gamma_x_theta = torch.lgamma(x + theta)
        log_gamma_theta = torch.lgamma(theta)
        log_gamma_x_1 = torch.lgamma(x + 1)
        
        nb_case = (torch.log(1 - pi) + log_gamma_x_theta - log_gamma_theta - log_gamma_x_1 +
                   theta * torch.log(theta / theta_plus_rho) + x * torch.log(rho / theta_plus_rho))
        
        mask = (x == 0).float()
        log_lik = mask * zero_case + (1 - mask) * nb_case
        return -log_lik.mean()
    
    @staticmethod
    def adjacency(A_true, A_pred, pos_weight=None):
        """Weighted binary cross-entropy for graph reconstruction (Eq. 17)."""
        if pos_weight is None:
            n_pos = (A_true > 0).sum().float()
            n_total = A_true.numel()
            pos_weight = (n_total - n_pos) / (n_pos + 1e-8)
        
        weight = torch.where(A_true > 0, pos_weight, 1.0)
        return torch.nn.functional.binary_cross_entropy(
            A_pred, A_true.float(), weight=weight, reduction='mean'
        )
    
    @staticmethod
    def kl_divergence(mu, logvar):
        """KL divergence for VAE regularization (Eq. 19)."""
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl.mean()
    
    @staticmethod
    def contrastive(z, pos_pairs, neg_pairs, temperature=0.1):
        """
        Vectorized contrastive loss (Eq. 18) - FIXED from O(N^2) to O(N).
        
        Args:
            z: Latent representations (N, D)
            pos_pairs: Positive pairs (P, 2) indices
            neg_pairs: Negative pairs (Q, 2) indices  
            temperature: Temperature parameter tau
        """
        z_norm = torch.nn.functional.normalize(z, p=2, dim=1)
        
        z_i = z_norm[pos_pairs[:, 0]]  
        z_j = z_norm[pos_pairs[:, 1]]  
        pos_sim = (z_i * z_j).sum(dim=1) / temperature  
        
        unique_anchors = torch.unique(pos_pairs[:, 0])
        
        anchor_neg_sims = []
        for anchor in unique_anchors:
            neg_mask = neg_pairs[:, 0] == anchor
            if neg_mask.sum() == 0:
                continue
            
            z_anchor = z_norm[anchor].unsqueeze(0)  
            z_negs = z_norm[neg_pairs[neg_mask, 1]]  
            neg_sim = (z_anchor * z_negs).sum(dim=1) / temperature  
            anchor_neg_sims.append(neg_sim)
        
        if len(anchor_neg_sims) == 0:
            return torch.tensor(0.0, device=z.device)
        
        loss = 0.0
        pos_idx = 0
        for anchor in unique_anchors:
            neg_mask = neg_pairs[:, 0] == anchor
            if neg_mask.sum() == 0:
                continue
                
            pos_mask = pos_pairs[:, 0] == anchor
            anchor_pos_sims = pos_sim[pos_mask]
            
            anchor_negs = anchor_neg_sims.pop(0)
            
            for pos_s in anchor_pos_sims:
                numerator = torch.exp(pos_s)
                denominator = numerator + torch.exp(anchor_negs).sum()
                loss -= torch.log(numerator / denominator)
        
        return loss / len(pos_pairs)
    
    @staticmethod
    def prediction(y_true, y_pred):
        """Binary cross-entropy for response prediction (Eq. 27)."""
        return torch.nn.functional.binary_cross_entropy(y_pred, y_true.float())


class Trainer:
    """Trainer with two-phase training and reconstruction monitoring."""
    
    def __init__(self, model, config, device='cpu'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        self.lambda1 = config.get('lambda1', 1.0)
        self.lambda2 = config.get('lambda2', 0.5)
        self.beta = config.get('beta', 0.01)
        self.gamma = 0.0 
        self.gamma_phase2 = config.get('gamma', 0.1)
        
        self.lr = config.get('lr', 1e-3)
        self.lr_phase2 = self.lr / 10
        self.epochs_phase1 = config.get('epochs_phase1', 300)
        self.epochs_phase2 = config.get('epochs_phase2', 200)
        self.patience = config.get('patience', 50)
        
        self.tolerance = 1.1 
        self.max_gamma_reductions = 3
        self.phase1_metrics = {}
        
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        
        self.loss_fn = GVAELoss()
        self.optimizer = None
        
    def setup_optimizer(self, phase=1):
        """Setup optimizer with phase-specific learning rate."""
        lr = self.lr if phase == 1 else self.lr_phase2
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    
    def create_adjacency_matrix(self, edge_index, n_nodes):
        """Create dense adjacency matrix from edge index."""
        adj = torch.zeros(n_nodes, n_nodes, device=self.device)
        adj[edge_index[0], edge_index[1]] = 1
        return adj
    
    def compute_loss(self, outputs, data, phase=1):
        """Compute total loss with all components (Eq. 20)."""
        
        adj_true = self.create_adjacency_matrix(
            data.edge_index, data.x.size(0)
        )
        L_adj = self.loss_fn.adjacency(adj_true, outputs['adj_recon'])
        
        x_raw = data.x_raw if hasattr(data, 'x_raw') else data.x
        L_expr = self.loss_fn.zinb(
            x_raw, outputs['rho'], outputs['theta'], outputs['pi']
        )
        
        L_kl = self.loss_fn.kl_divergence(outputs['mu'], outputs['logvar'])
        
        L_contrast = 0.0
        if hasattr(data, 'pos_pairs') and hasattr(data, 'neg_pairs'):
            L_contrast = self.loss_fn.contrastive(
                outputs['z'], data.pos_pairs, data.neg_pairs
            )
        
        L_pred = 0.0
        if phase == 2 and 'y_pred' in outputs and hasattr(data, 'y'):
            L_pred = self.loss_fn.prediction(data.y, outputs['y_pred'])
        
        total_loss = (
            L_adj + 
            self.lambda1 * L_expr + 
            self.lambda2 * L_contrast + 
            self.beta * L_kl +
            self.gamma * L_pred
        )
        
        return {
            'total': total_loss,
            'adj': L_adj.item(),
            'expr': L_expr.item(),
            'kl': L_kl.item(),
            'contrast': L_contrast.item() if isinstance(L_contrast, torch.Tensor) else L_contrast,
            'pred': L_pred.item() if isinstance(L_pred, torch.Tensor) else L_pred
        }
    
    @torch.no_grad()
    def evaluate(self, data):
        """Evaluate model on validation set."""
        self.model.eval()
        outputs = self.model(data)
        
        adj_true = self.create_adjacency_matrix(data.edge_index, data.x.size(0))
        L_adj = self.loss_fn.adjacency(adj_true, outputs['adj_recon'])
        
        x_raw = data.x_raw if hasattr(data, 'x_raw') else data.x
        L_expr = self.loss_fn.zinb(
            x_raw, outputs['rho'], outputs['theta'], outputs['pi']
        )
        
        return {
            'loss_adj': L_adj.item(),
            'loss_expr': L_expr.item()
        }
    
    def train_epoch(self, data, phase=1):
        """Train for one epoch with gradient clipping."""
        self.model.train()
        self.optimizer.zero_grad()
        
        outputs = self.model(data)
        losses = self.compute_loss(outputs, data, phase)
        
        losses['total'].backward()
        
        clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        
        return losses
    
    def train_phase1(self, data):
        """Phase 1: Representation learning (Section 3.3.5)."""
        print("=== Phase 1: Representation Learning ===")
        self.setup_optimizer(phase=1)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, self.epochs_phase1 + 1):
            losses = self.train_epoch(data, phase=1)
            
            if epoch % 10 == 0:
                val_metrics = self.evaluate(data)
                val_loss = val_metrics['loss_adj'] + val_metrics['loss_expr']
                
                print(f"Epoch {epoch:3d} | "
                      f"L_adj={losses['adj']:.4f} "
                      f"L_expr={losses['expr']:.4f} "
                      f"L_kl={losses['kl']:.4f} "
                      f"Val={val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.patience // 2:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        final_val = self.evaluate(data)
        self.phase1_metrics = {
            'loss_adj': final_val['loss_adj'],
            'loss_expr': final_val['loss_expr'],
            'epoch': epoch
        }
        
        print(f"Phase 1 complete: L_adj={final_val['loss_adj']:.4f}, "
              f"L_expr={final_val['loss_expr']:.4f}")
        
        return self.phase1_metrics
    
    def train_phase2(self, data):
        """
        Phase 2: Joint clinical fine-tuning with reconstruction monitoring.
        
        If reconstruction degrades beyond tolerance, gamma is reduced.
        If it cannot be maintained, fall back to frozen encoder.
        """
        print("\n=== Phase 2: Joint Clinical Fine-Tuning ===")
        self.gamma = self.gamma_phase2
        self.setup_optimizer(phase=2)
        
        best_val_loss = float('inf')
        patience_counter = 0
        gamma_reductions = 0
        
        for epoch in range(1, self.epochs_phase2 + 1):
            losses = self.train_epoch(data, phase=2)
            
            if epoch % 10 == 0:
                val_metrics = self.evaluate(data)
                val_loss = val_metrics['loss_adj'] + val_metrics['loss_expr']
                
                adj_degraded = val_metrics['loss_adj'] > self.tolerance * self.phase1_metrics['loss_adj']
                expr_degraded = val_metrics['loss_expr'] > self.tolerance * self.phase1_metrics['loss_expr']
                
                if adj_degraded or expr_degraded:
                    gamma_reductions += 1
                    self.gamma /= 2
                    print(f"WARNING: Reconstruction degraded (reduction {gamma_reductions}/{self.max_gamma_reductions})")
                    print(f"  L_adj: {val_metrics['loss_adj']:.4f} vs {self.phase1_metrics['loss_adj']:.4f}")
                    print(f"  L_expr: {val_metrics['loss_expr']:.4f} vs {self.phase1_metrics['loss_expr']:.4f}")
                    print(f"  Reducing gamma to {self.gamma:.4f}")
                    
                    if gamma_reductions >= self.max_gamma_reductions:
                        print("FATAL: Cannot maintain reconstruction quality. Stopping Phase 2.")
                        print("Recommendation: Use frozen encoder (Phase 1 only).")
                        break
                
                print(f"Epoch {epoch:3d} | "
                      f"L_adj={losses['adj']:.4f} "
                      f"L_expr={losses['expr']:.4f} "
                      f"L_pred={losses['pred']:.4f} "
                      f"gamma={self.gamma:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.patience // 2:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        print("Phase 2 complete")
    
    def train(self, data):
        """Full two-phase training pipeline."""
        data = data.to(self.device)
        
        phase1_metrics = self.train_phase1(data)
        
        if self.model.use_predictor and hasattr(data, 'y'):
            self.train_phase2(data)
        
        return phase1_metrics
