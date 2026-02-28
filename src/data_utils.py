import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors


def prepare_graph_data(adata, spatial_key='spatial'):
    x = torch.tensor(adata.obsm['X_pca'], dtype=torch.float32)
    n_cells = x.size(0)
    nbrs = NearestNeighbors(n_neighbors=15).fit(adata.obsm['X_pca'])
    _, indices = nbrs.kneighbors(adata.obsm['X_pca'])
    edge_list = []
    for i in range(n_cells):
        for j in indices[i]:
            if i != j:
                edge_list.append([i, j])
    edge_index = torch.tensor(edge_list, dtype=torch.long).T
    coords = torch.tensor(adata.obsm[spatial_key], dtype=torch.float32)
    library_size = torch.tensor(adata.X.sum(axis=1).A1 if hasattr(adata.X, 'A1') else adata.X.sum(axis=1), dtype=torch.float32)
    x_raw = torch.tensor(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X, dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index, coords=coords, library_size=library_size, x_raw=x_raw)
    if 'response' in adata.obs:
        patient_ids = adata.obs['patient_id'].unique()
        patient_masks = []
        responses = []
        for pid in patient_ids:
            mask = torch.tensor(adata.obs['patient_id'].values == pid)
            patient_masks.append(mask)
            resp = adata.obs.loc[adata.obs['patient_id'] == pid, 'response'].iloc[0]
            responses.append(resp)
        data.patient_masks = patient_masks
        data.y = torch.tensor(responses, dtype=torch.float32)
    return data
