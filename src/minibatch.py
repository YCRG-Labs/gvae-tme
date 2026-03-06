import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import NeighborLoader
from pathlib import Path

from .trainer import Trainer, GVAELoss


def check_neighbor_loader(data, num_neighbors, batch_size=2):
    loader = NeighborLoader(data, num_neighbors=num_neighbors, batch_size=batch_size)
    _ = next(iter(loader))
    return True


class MiniBatchTrainer(Trainer):

    def __init__(self, model, config, device='cpu', checkpoint_dir=None, freeze_encoder=False):
        super().__init__(model, config, device, checkpoint_dir, freeze_encoder)
        self.batch_size = config.get('batch_size', 512)
        self.num_neighbors = config.get('num_neighbors', [15, 10])

    def _make_loader(self, data, shuffle=True):
        return NeighborLoader(
            data,
            num_neighbors=self.num_neighbors,
            batch_size=self.batch_size,
            shuffle=shuffle,
        )

    def _collect_embeddings(self, data):
        self.model.eval()
        orig = self.model.use_predictor
        self.model.use_predictor = False
        loader = self._make_loader(data, shuffle=False)
        z_parts = []
        idx_parts = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                outputs = self.model(batch)
                n_seed = batch.batch_size
                z_parts.append(outputs['z'][:n_seed].cpu())
                idx_parts.append(batch.n_id[:n_seed].cpu())
        self.model.use_predictor = orig
        z_cat = torch.cat(z_parts, dim=0)
        idx_cat = torch.cat(idx_parts, dim=0)
        z_all = torch.zeros(data.num_nodes, z_cat.shape[1])
        z_all[idx_cat] = z_cat
        return z_all

    def _ensure_pred_optimizer(self):
        if not hasattr(self, '_pred_optimizer') or self._pred_optimizer is None:
            import torch.optim as optim
            self._pred_optimizer = optim.Adam(
                self.model.predictor.parameters(), lr=self.lr_phase2)

    def _train_predictor_step(self, data):
        z_full = self._collect_embeddings(data)
        self.model.predictor.train()
        z_dev = z_full.to(self.device)
        masks_dev = [m.to(self.device) for m in data.patient_masks]
        y_pred, _ = self.model.predictor(z_dev, masks_dev)
        if hasattr(data, 'train_patient_idx'):
            y_train = data.y[data.train_patient_idx].to(self.device)
            pred_train = y_pred[data.train_patient_idx]
            if y_train.numel() == 0:
                return 0.0
            L_pred = self.loss_fn.prediction(y_train, pred_train)
        else:
            L_pred = self.loss_fn.prediction(data.y.to(self.device), y_pred)
        self._ensure_pred_optimizer()
        self._pred_optimizer.zero_grad()
        (self.gamma * L_pred).backward()
        clip_grad_norm_(self.model.predictor.parameters(), self.max_grad_norm)
        self._pred_optimizer.step()
        return L_pred.item()

    def train_epoch(self, data, phase=1, epoch=1):
        self.model.train()
        if phase == 2 and self.freeze_encoder:
            self.model.encoder.eval()
        orig_pred = self.model.use_predictor
        self.model.use_predictor = False
        loader = self._make_loader(data, shuffle=True)
        epoch_losses = {'total': 0, 'adj': 0, 'expr': 0, 'kl': 0, 'contrast': 0, 'pred': 0}
        n_batches = 0

        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(batch)
            losses = self._compute_batch_loss(outputs, batch, phase, epoch)
            losses['total'].backward()
            clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            for k in epoch_losses:
                epoch_losses[k] += losses[k].item() if k == 'total' else losses.get(k, 0)
            n_batches += 1

        self.model.use_predictor = orig_pred
        self.scheduler.step()

        for k in epoch_losses:
            if k == 'total':
                epoch_losses[k] = float(epoch_losses[k]) / max(n_batches, 1)
            else:
                epoch_losses[k] = epoch_losses[k] / max(n_batches, 1)
        epoch_losses['beta'] = self.get_beta(epoch) if phase == 1 else self.beta_target
        return epoch_losses

    def _compute_batch_loss(self, outputs, batch, phase, epoch):
        n_seed = batch.batch_size
        L_adj = self.loss_fn.adjacency_negsampling(outputs['pos_scores'], outputs['neg_scores'])
        x_raw = batch.x_raw[:n_seed] if hasattr(batch, 'x_raw') else batch.x[:n_seed]
        if 'expr_mu' in outputs:
            L_expr = self.loss_fn.gaussian(x_raw, outputs['expr_mu'][:n_seed], outputs['expr_logvar'][:n_seed])
        else:
            L_expr = self.loss_fn.zinb(x_raw, outputs['rho'][:n_seed], outputs['theta'], outputs['pi'][:n_seed])
        beta = self.get_beta(epoch) if phase == 1 else self.beta_target
        L_kl = self.loss_fn.kl_divergence(outputs['mu'][:n_seed], outputs['logvar'][:n_seed])

        total = L_adj + self.lambda1 * L_expr + beta * L_kl
        return {
            'total': total,
            'adj': L_adj.item(),
            'expr': L_expr.item(),
            'kl': L_kl.item(),
            'contrast': 0.0,
            'pred': 0.0,
        }

    @torch.no_grad()
    def evaluate(self, data):
        self.model.eval()
        orig = self.model.use_predictor
        self.model.use_predictor = False
        rng_state = torch.random.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state() if self.device != 'cpu' and torch.cuda.is_available() else None
        torch.manual_seed(0)
        if cuda_rng_state is not None:
            torch.cuda.manual_seed(0)
        loader = self._make_loader(data, shuffle=False)
        total_adj = 0.0
        total_expr = 0.0
        n_batches = 0
        for batch in loader:
            batch = batch.to(self.device)
            outputs = self.model(batch)
            n_seed = batch.batch_size
            L_adj = self.loss_fn.adjacency_negsampling(outputs['pos_scores'], outputs['neg_scores'])
            x_raw = batch.x_raw[:n_seed] if hasattr(batch, 'x_raw') else batch.x[:n_seed]
            if 'expr_mu' in outputs:
                L_expr = self.loss_fn.gaussian(x_raw, outputs['expr_mu'][:n_seed], outputs['expr_logvar'][:n_seed])
            else:
                L_expr = self.loss_fn.zinb(x_raw, outputs['rho'][:n_seed], outputs['theta'], outputs['pi'][:n_seed])
            total_adj += L_adj.item()
            total_expr += L_expr.item()
            n_batches += 1
        torch.random.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state)
        self.model.use_predictor = orig
        return {
            'loss_adj': total_adj / max(n_batches, 1),
            'loss_expr': total_expr / max(n_batches, 1),
        }

    @torch.no_grad()
    def evaluate_prediction(self, data):
        if not hasattr(data, 'patient_masks'):
            return 0.0
        z_full = self._collect_embeddings(data)
        z_dev = z_full.to(self.device)
        masks_dev = [m.to(self.device) for m in data.patient_masks]
        self.model.predictor.eval()
        y_pred, _ = self.model.predictor(z_dev, masks_dev)
        if hasattr(data, 'val_patient_idx') and data.val_patient_idx.numel() > 0:
            y_val = data.y[data.val_patient_idx].to(self.device)
            pred_val = y_pred[data.val_patient_idx]
            return self.loss_fn.prediction(y_val, pred_val).item()
        return 0.0

    def train(self, data):
        phase1_metrics = self.train_phase1(data)
        if self.model.use_predictor and hasattr(data, 'y'):
            self.train_phase2(data)
        return phase1_metrics

    def train_phase1(self, data):
        print("=== Phase 1: Representation Learning (mini-batch) ===")
        print(f"  batch_size={self.batch_size}, neighbors={self.num_neighbors}")
        self.gamma = 0.0
        self.setup_optimizer(phase=1)
        best_val_loss = float('inf')
        patience_counter = 0
        for epoch in range(1, self.epochs_phase1 + 1):
            losses = self.train_epoch(data, phase=1, epoch=epoch)
            if epoch % 10 == 0:
                val = self.evaluate(data)
                val_loss = val['loss_adj'] + val['loss_expr']
                print(f"Epoch {epoch:3d} | L_adj={losses['adj']:.4f} L_expr={losses['expr']:.4f} "
                      f"L_kl={losses['kl']:.4f} beta={losses['beta']:.5f} Val={val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self._best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                if patience_counter >= self.patience // 10:
                    print(f"Early stopping at epoch {epoch}")
                    break
            if epoch % self.checkpoint_every == 0:
                self.save_checkpoint(f"phase1_epoch{epoch}")
        if hasattr(self, '_best_state'):
            self.model.load_state_dict(self._best_state)
            del self._best_state
        final_val = self.evaluate(data)
        self.phase1_metrics = {'loss_adj': float(final_val['loss_adj']),
                               'loss_expr': float(final_val['loss_expr'])}
        print(f"Phase 1 complete: L_adj={final_val['loss_adj']:.4f}, "
              f"L_expr={final_val['loss_expr']:.4f}")
        self.save_checkpoint("phase1")
        return self.phase1_metrics

    def train_phase2(self, data):
        print("\n=== Phase 2: Joint Clinical Fine-Tuning (mini-batch) ===")
        if self.freeze_encoder:
            print("  [ablation] Encoder frozen — only training predictor head")
            self.model.encoder.requires_grad_(False)
            self.model.gate.requires_grad_(False)
        self.gamma = self.gamma_phase2
        self.setup_optimizer(phase=2)
        self._pred_optimizer = None
        best_val_loss = float('inf')
        patience_counter = 0
        gamma_reductions = 0
        has_predictor = self.model.use_predictor and hasattr(data, 'patient_masks')
        for epoch in range(1, self.epochs_phase2 + 1):
            losses = self.train_epoch(data, phase=2, epoch=epoch)
            pred_loss = 0.0
            if has_predictor:
                pred_loss = self._train_predictor_step(data)
                losses['pred'] = pred_loss
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
                        print("  Cannot maintain reconstruction quality. Stopping Phase 2.")
                        break
                val_pred = self.evaluate_prediction(data) if has_predictor else 0.0
                print(f"Epoch {epoch:3d} | L_adj={losses['adj']:.4f} L_expr={losses['expr']:.4f} "
                      f"L_pred={losses['pred']:.4f} val_pred={val_pred:.4f} gamma={self.gamma:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self._best_state_p2 = {k: v.clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                if patience_counter >= self.patience // 10:
                    print(f"Early stopping at epoch {epoch}")
                    break
            if epoch % self.checkpoint_every == 0:
                self.save_checkpoint(f"phase2_epoch{epoch}")
        if hasattr(self, '_best_state_p2'):
            self.model.load_state_dict(self._best_state_p2)
            del self._best_state_p2
        if self.freeze_encoder:
            self.model.encoder.requires_grad_(True)
            self.model.gate.requires_grad_(True)
        self.save_checkpoint("phase2")
        print("Phase 2 complete")
