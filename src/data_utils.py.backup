import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
import numpy as np


def prepare_graph_data(adata, spatial_key='spatial', k_mol=10, r_spatial=82.5, 
                       r_far_factor=5.0, k_neg=10):
    x = torch.tensor(adata.obsm['X_pca'], dtype=torch.float32)
    n_cells = x.size(0)
    coords = torch.tensor(adata.obsm[spatial_key], dtype=torch.float32)
    
    nbrs_mol = NearestNeighbors(n_neighbors=k_mol+1).fit(adata.obsm['X_pca'])
    distances_mol, indices_mol = nbrs_mol.kneighbors(adata.obsm['X_pca'])
    
    mol_edge_list = []
    for i in range(n_cells):
        for idx, j in enumerate(indices_mol[i]):
            if i != j:
                mol_edge_list.append([i, j])
    
    edge_index_mol = torch.tensor(mol_edge_list, dtype=torch.long).T
    
    coords_np = coords.numpy()
    spatial_distances = np.sqrt(((coords_np[:, None, :] - coords_np[None, :, :]) ** 2).sum(axis=2))
    
    spatial_edge_list = []
    pos_pairs_list = []
    
    for i in range(n_cells):
        for j in range(n_cells):
            if i != j and spatial_distances[i, j] < r_spatial:
                spatial_edge_list.append([i, j])
                pos_pairs_list.append([i, j])
    
    edge_index_spatial = torch.tensor(spatial_edge_list, dtype=torch.long).T
    pos_pairs = torch.tensor(pos_pairs_list, dtype=torch.long)
    
    r_far = r_spatial * r_far_factor
    neg_pairs_list = []
    
    for i in range(n_cells):
        mol_neighbors = indices_mol[i][1:]
        far_neighbors = []
        for j in mol_neighbors:
            if spatial_distances[i, j] > r_far:
                far_neighbors.append(j)
            if len(far_neighbors) >= k_neg:
                break
        for j in far_neighbors:
            neg_pairs_list.append([i, j])
    
    neg_pairs = torch.tensor(neg_pairs_list, dtype=torch.long) if neg_pairs_list else torch.zeros((0, 2), dtype=torch.long)
    
    edge_index = edge_index_mol
    
    library_size = torch.tensor(
        adata.X.sum(axis=1).A1 if hasattr(adata.X, 'A1') else adata.X.sum(axis=1), 
        dtype=torch.float32
    )
    x_raw = torch.tensor(
        adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X, 
        dtype=torch.float32
    )
    
    data = Data(
        x=x, 
        edge_index=edge_index,
        edge_index_mol=edge_index_mol,
        edge_index_spatial=edge_index_spatial,
        coords=coords, 
        library_size=library_size, 
        x_raw=x_raw,
        pos_pairs=pos_pairs,
        neg_pairs=neg_pairs,
        spatial_distances=torch.tensor(spatial_distances, dtype=torch.float32)
    )
    
    if 'response' in adata.obs and 'patient_id' in adata.obs:
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


def create_synthetic_data(n_cells=5000, n_genes=2000, n_patients=10, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    counts = np.random.poisson(2, size=(n_cells, n_genes))
    patient_ids = np.random.choice(n_patients, n_cells)
    patient_response = np.random.choice([0, 1], n_patients)
    cell_response = patient_response[patient_ids]
    coords = np.random.randn(n_cells, 2) * 100
    
    import anndata
    import scanpy as sc
    
    adata = anndata.AnnData(X=counts)
    adata.obs['patient_id'] = [f'P{i:03d}' for i in patient_ids]
    adata.obs['response'] = cell_response
    adata.obsm['spatial'] = coords
    
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=10000)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=min(1000, n_genes))
    sc.pp.pca(adata, n_comps=50)
    
    return adata
