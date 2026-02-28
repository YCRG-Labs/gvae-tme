import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class CellAdaptiveGate(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(in_dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        return torch.sigmoid(x @ self.w + self.b)


class GATEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, latent_dim=32, n_heads=4, dropout=0.2):
        super().__init__()
        self.latent_dim = latent_dim
        self.gat1 = GATConv(in_dim, hidden_dim // n_heads, heads=n_heads, concat=True, dropout=dropout, add_self_loops=False)
        self.gat_mu = GATConv(hidden_dim, latent_dim, heads=1, concat=False, dropout=dropout, add_self_loops=False)
        self.gat_logvar = GATConv(hidden_dim, latent_dim, heads=1, concat=False, dropout=dropout, add_self_loops=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        h = F.elu(self.gat1(x, edge_index))
        h = self.dropout(h)
        mu = self.gat_mu(h, edge_index)
        logvar = self.gat_logvar(h, edge_index)
        return mu, logvar


class ZINBDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims=[128, 256], n_genes=2000, dropout=0.2):
        super().__init__()
        self.n_genes = n_genes
        layers = []
        prev_dim = latent_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ELU(), nn.Dropout(dropout)])
            prev_dim = hidden_dim
        self.decoder = nn.Sequential(*layers)
        self.rho_layer = nn.Linear(prev_dim, n_genes)
        self.pi_layer = nn.Linear(prev_dim, n_genes)
        self.log_theta = nn.Parameter(torch.zeros(n_genes))
        
    def forward(self, z, library_size):
        h = self.decoder(z)
        rho_raw = self.rho_layer(h)
        pi_logits = self.pi_layer(h)
        rho = F.softmax(rho_raw, dim=1) * library_size.unsqueeze(1)
        pi = torch.sigmoid(pi_logits)
        theta = torch.exp(self.log_theta)
        return rho, theta, pi


class AdjacencyDecoder(nn.Module):
    def forward(self, z):
        return torch.sigmoid(z @ z.T)


class AttentionPooling(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.W = nn.Linear(latent_dim, latent_dim)
        self.w = nn.Parameter(torch.randn(latent_dim))
        
    def forward(self, z, mask):
        hidden = torch.tanh(self.W(z))
        scores = hidden @ self.w
        scores = scores.masked_fill(~mask, float('-inf'))
        attention = F.softmax(scores, dim=0)
        h_p = (attention.unsqueeze(1) * z).sum(dim=0)
        return h_p, attention


class ResponsePredictor(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.pooling = AttentionPooling(latent_dim)
        self.classifier = nn.Linear(latent_dim, 1)
        
    def forward(self, z, patient_masks):
        predictions = []
        attentions = []
        for mask in patient_masks:
            h_p, attn = self.pooling(z, mask)
            pred = torch.sigmoid(self.classifier(h_p))
            predictions.append(pred)
            attentions.append(attn)
        predictions = torch.stack(predictions).squeeze()
        return predictions, attentions


class GVAEModel(nn.Module):
    def __init__(self, n_features, n_genes, hidden_dim=64, latent_dim=32, n_heads=4, dropout=0.2, use_predictor=False):
        super().__init__()
        self.n_features = n_features
        self.n_genes = n_genes
        self.latent_dim = latent_dim
        self.use_predictor = use_predictor
        self.gate = CellAdaptiveGate(n_features)
        self.encoder = GATEncoder(n_features, hidden_dim, latent_dim, n_heads, dropout)
        self.adj_decoder = AdjacencyDecoder()
        self.expr_decoder = ZINBDecoder(latent_dim, n_genes=n_genes, dropout=dropout)
        if use_predictor:
            self.predictor = ResponsePredictor(latent_dim)
        else:
            self.predictor = None
            
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        mu, logvar = self.encoder(x, edge_index)
        z = self.reparameterize(mu, logvar)
        adj_recon = self.adj_decoder(z)
        rho, theta, pi = self.expr_decoder(z, data.library_size)
        outputs = {'mu': mu, 'logvar': logvar, 'z': z, 'adj_recon': adj_recon, 'rho': rho, 'theta': theta, 'pi': pi}
        if self.use_predictor and hasattr(data, 'patient_masks'):
            predictions, attentions = self.predictor(z, data.patient_masks)
            outputs['predictions'] = predictions
            outputs['attentions'] = attentions
        return outputs
    
    def compute_gate_values(self, x):
        return self.gate(x)
