import torch
import torch.optim as optim
import numpy as np


class GVAELoss:
    @staticmethod
    def zinb(x, rho, theta, pi, eps=1e-8):
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
        if pos_weight is None:
            n_pos = (A_true > 0).sum().float()
            n_total = A_true.numel()
            pos_weight = (n_total - n_pos) / (n_pos + 1e-8)
        weight = torch.where(A_true > 0, pos_weight, 1.0)
        return torch.nn.functional.binary_cross_entropy(A_pred, A_true.float(), weight=weight, reduction='mean')
    
    @staticmethod
    def kl_divergence(mu, logvar):
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl.mean()
    
    @staticmethod
    def contrastive(z, pos_pairs, neg_pairs, temperature=0.1):
        z_norm = torch.nn.functional.normalize(z, p=2, dim=1)
        z_i = z_norm[pos_pairs[:, 0]]
        z_j = z_norm[pos_pairs[:, 1]]
        pos_sim = (z_i * z_j).sum(dim=1) / temperature
        loss = 0.0
        unique_anchors = torch.unique(pos_pairs[:, 0])
        for anchor in unique_anchors:
            pos_mask = pos_pairs[:, 0] == anchor
            anchor_pos_sim = pos_sim[pos_mask]
            neg_mask = neg_pairs[:, 0] == anchor
            if neg_mask.sum() == 0:
                continue
            z_anchor = z_norm[anchor].unsqueeze(0)
            z_negs = z_norm[neg_pairs[neg_mask, 1]]
            neg_sim = (z_anchor * z_negs).sum(dim=1) / temperature
            numerator = torch.exp(anchor_pos_sim[0])
            denominator = numerator + torch.exp(neg_sim).sum()
            loss -= torch.log(numerator / denominator)
        return loss / max(len(unique_anchors), 1)
    
    @staticmethod
    def prediction(y_true, y_pred):
        return torch.nn.functional.binary_cross_entropy(y_pred, y_true.float())


class Trainer:
    def __init__(self, model, config, device='cpu'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.lambda1 = config.get('lambda1', 1.0)
        self.lambda2 = config.get('lambda2', 0.5)
        self.beta = config.get('beta', 0.01)
        self.gamma = 0.0
        self.lr = config.get('lr', 1e-3)
        self.lr_phase2 = self.lr / 10
        self.epochs_phase1 = config.get('epochs_phase1', 300)
        self.epochs_phase2 = config.get('epochs_phase2', 200)
        self.patience = config.get('patience', 50)
        self.tolerance = 1.1
        self.max_gamma_reductions = 3
        self.loss_fn = GVAELoss()
        self.optimizer = None
        self.phase1_metrics = {}
        
    def setup_optimizer(self, phase=1):
        lr = self.lr if phase == 1 else self.lr_phase2
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    
    def create_adjacency_matrix(self, edge_index, n_nodes):
        adj = torch.zeros(n_nodes, n_nodes, device=self.device)
        adj[edge_index[0], edge_index[1]] = 1
        return adj
    
    def compute_loss(self, outputs, data, phase=1):
        adj_true = self.create_adjacency_matrix(data.edge_index, data.x.size(0))
        L_adj = self.loss_fn.adjacency(adj_true, outputs['adj_recon'])
        x_raw = data.x_raw if hasattr(data, 'x_raw') else data.x
        L_expr = self.loss_fn.zinb(x_raw, outputs['rho'], outputs['theta'], outputs['pi'])
        L_kl = self.loss_fn.kl_divergence(outputs['mu'], outputs['logvar'])
        L_contrast = 0
        if hasattr(data, 'pos_pairs') and hasattr(data, 'neg_pairs'):
            L_contrast = self.loss_fn.contrastive(outputs['z'], data.pos_pairs, data.neg_pairs)
        loss = L_adj + self.lambda1 * L_expr + self.beta * L_kl + self.lambda2 * L_contrast
        L_pred = torch.tensor(0.0)
        if phase == 2 and 'predictions' in outputs:
            L_pred = self.loss_fn.prediction(data.y, outputs['predictions'])
            loss += self.gamma * L_pred
        metrics = {'total': loss.item(), 'adj': L_adj.item(), 'expr': L_expr.item(),
                   'kl': L_kl.item(), 'contrast': L_contrast.item() if isinstance(L_contrast, float) else L_contrast.item(),
                   'pred': L_pred.item()}
        return loss, metrics
    
    def train_phase1(self, data, validate_fn=None):
        self.setup_optimizer(phase=1)
        self.model.train()
        best_val_loss = float('inf')
        patience_counter = 0
        history = []
        for epoch in range(self.epochs_phase1):
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss, metrics = self.compute_loss(outputs, data, phase=1)
            loss.backward()
            self.optimizer.step()
            history.append(metrics)
            if validate_fn and epoch % 10 == 0:
                val_loss = validate_fn(self.model)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= self.patience:
                    break
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(data)
            _, self.phase1_metrics = self.compute_loss(outputs, data, phase=1)
        return self.phase1_metrics
    
    def train_phase2(self, data):
        self.gamma = self.config.get('gamma', 0.1)
        self.setup_optimizer(phase=2)
        gamma_reductions = 0
        history = []
        for epoch in range(self.epochs_phase2):
            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss, metrics = self.compute_loss(outputs, data, phase=2)
            loss.backward()
            self.optimizer.step()
            history.append(metrics)
            if epoch % 10 == 0:
                adj_degraded = metrics['adj'] > self.tolerance * self.phase1_metrics['adj']
                expr_degraded = metrics['expr'] > self.tolerance * self.phase1_metrics['expr']
                if adj_degraded or expr_degraded:
                    gamma_reductions += 1
                    if gamma_reductions > self.max_gamma_reductions:
                        break
                    self.gamma /= 2
        return metrics
    
    def train(self, data, validate_fn=None):
        data = data.to(self.device)
        self.train_phase1(data, validate_fn)
        if self.model.use_predictor:
            final_metrics = self.train_phase2(data)
        else:
            final_metrics = self.phase1_metrics
        return final_metrics
