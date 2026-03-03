import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

class CellAdaptiveGate(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(in_dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.sigmoid(x @ self.w + self.b)

class WeightedGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, negative_slope=0.2, dropout=0.0, bias=True):
        super().__init__(aggr='add', node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout_p = dropout
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att = nn.Parameter(torch.empty(1, heads, 2 * out_channels))
        nn.init.xavier_uniform_(self.att)
        total_out = heads * out_channels if concat else out_channels
        self.bias = nn.Parameter(torch.zeros(total_out)) if bias else None

    def forward(self, x, edge_index, edge_weight=None):
        H, C = self.heads, self.out_channels
        x_proj = self.W(x).view(-1, H, C)
        out = self.propagate(edge_index, x=x_proj, edge_weight=edge_weight)
        out = out.view(-1, H * C) if self.concat else out.mean(dim=1)
        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, x_i, x_j, edge_weight, index, ptr, size_i):
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout_p, training=self.training)
        msg = x_j * alpha.unsqueeze(-1)
        if edge_weight is not None:
            msg = msg * edge_weight.view(-1, 1, 1)
        return msg

class GATEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, latent_dim=32, n_heads=4, dropout=0.2):
        super().__init__()
        self.gat1 = WeightedGATConv(in_dim, hidden_dim // n_heads, heads=n_heads, concat=True, dropout=dropout)
        self.gat_mu = WeightedGATConv(hidden_dim, latent_dim, heads=1, concat=False, dropout=dropout)
        self.gat_logvar = WeightedGATConv(hidden_dim, latent_dim, heads=1, concat=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None):
        h = F.elu(self.gat1(x, edge_index, edge_weight))
        h = self.dropout(h)
        mu = self.gat_mu(h, edge_index, edge_weight)
        logvar = self.gat_logvar(h, edge_index, edge_weight)
        return mu, logvar

class ZINBDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims=(128, 256), n_genes=2000, dropout=0.2):
        super().__init__()
        layers = []
        prev = latent_dim
        for hd in hidden_dims:
            layers += [nn.Linear(prev, hd), nn.BatchNorm1d(hd), nn.ELU(), nn.Dropout(dropout)]
            prev = hd
        self.backbone = nn.Sequential(*layers)
        self.rho_head = nn.Linear(prev, n_genes)
        self.pi_head = nn.Linear(prev, n_genes)
        self.log_theta = nn.Parameter(torch.zeros(n_genes))

    def forward(self, z, library_size):
        h = self.backbone(z)
        rho = F.softmax(self.rho_head(h), dim=1) * library_size.unsqueeze(1)
        theta = torch.exp(self.log_theta)
        pi = torch.sigmoid(self.pi_head(h))
        return rho, theta, pi

class AdjacencyDecoder(nn.Module):
    def __init__(self, n_neg_samples=5):
        super().__init__()
        self.n_neg = n_neg_samples

    def forward(self, z, pos_edge_index, n_nodes):
        z_src = z[pos_edge_index[0]]
        z_dst = z[pos_edge_index[1]]
        pos_scores = torch.sigmoid((z_src * z_dst).sum(dim=1))
        neg_src = pos_edge_index[0].repeat_interleave(self.n_neg)
        neg_dst = torch.randint(0, n_nodes, (neg_src.size(0),), device=z.device)
        neg_scores = torch.sigmoid((z[neg_src] * z[neg_dst]).sum(dim=1))
        return pos_scores, neg_scores

class AttentionPooling(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.W = nn.Linear(latent_dim, latent_dim)
        self.w = nn.Parameter(torch.randn(latent_dim) * 0.01)

    def forward(self, z, mask):
        scores = torch.tanh(self.W(z)) @ self.w
        scores = scores.masked_fill(~mask, float('-inf'))
        attn = F.softmax(scores, dim=0)
        h_p = (attn.unsqueeze(1) * z).sum(dim=0)
        return h_p, attn

class ResponsePredictor(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.pooling = AttentionPooling(latent_dim)
        self.classifier = nn.Linear(latent_dim, 1)

    def forward(self, z, patient_masks):
        preds, attns = [], []
        for mask in patient_masks:
            h_p, attn = self.pooling(z, mask)
            pred = torch.sigmoid(self.classifier(h_p)).squeeze(-1)
            preds.append(pred)
            attns.append(attn)
        return torch.stack(preds), attns

class GVAEModel(nn.Module):
    def __init__(self, n_features, n_genes, hidden_dim=64, latent_dim=32, n_heads=4, dropout=0.2, n_neg_samples=5, use_predictor=False):
        super().__init__()
        self.gate = CellAdaptiveGate(n_features)
        self.encoder = GATEncoder(n_features, hidden_dim, latent_dim, n_heads, dropout)
        self.adj_decoder = AdjacencyDecoder(n_neg_samples)
        self.expr_decoder = ZINBDecoder(latent_dim, n_genes=n_genes, dropout=dropout)
        self.use_predictor = use_predictor
        if use_predictor:
            self.predictor = ResponsePredictor(latent_dim)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def compute_hybrid_weights(self, x, mol_weight, spatial_weight, edge_index):
        g = self.gate(x)
        g_tgt = g[edge_index[1]]
        hybrid = g_tgt * mol_weight + (1.0 - g_tgt) * spatial_weight
        return hybrid, g

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        has_spatial = data.spatial_weight.any()
        if has_spatial:
            hybrid_weight, gate_values = self.compute_hybrid_weights(x, data.mol_weight, data.spatial_weight, edge_index)
        else:
            hybrid_weight = data.mol_weight
            gate_values = torch.ones(x.size(0), device=x.device)
        mu, logvar = self.encoder(x, edge_index, edge_weight=hybrid_weight)
        z = self.reparameterize(mu, logvar)
        pos_scores, neg_scores = self.adj_decoder(z, edge_index, x.size(0))
        lib = data.library_size if hasattr(data, 'library_size') else x.sum(dim=1)
        rho, theta, pi = self.expr_decoder(z, lib)
        outputs = dict(z=z, mu=mu, logvar=logvar, pos_scores=pos_scores, neg_scores=neg_scores, rho=rho, theta=theta, pi=pi, gate_values=gate_values, hybrid_weight=hybrid_weight)
        if self.use_predictor and hasattr(data, 'patient_masks'):
            y_pred, attentions = self.predictor(z, data.patient_masks)
            outputs['y_pred'] = y_pred
            outputs['attentions'] = attentions
        return outputs
