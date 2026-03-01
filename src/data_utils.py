import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GroupShuffleSplit
from scipy.spatial import cKDTree

def _build_molecular_graph(pca, k=15):
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto')
    nbrs.fit(pca)
    dists, idxs = nbrs.kneighbors(pca)
    sigma = float(np.median(dists[:, -1]))
    src, dst, wts = [], [], []
    for i in range(len(pca)):
        for rank in range(1, k + 1):
            j = int(idxs[i, rank])
            d = float(dists[i, rank])
            src.append(i)
            dst.append(j)
            wts.append(np.exp(-d ** 2 / (2 * sigma ** 2)))
    return np.array(src), np.array(dst), np.array(wts, dtype=np.float32)

def _build_spatial_graph(coords, r, delta=None):
    if delta is None:
        delta = r / 3.0
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=r, output_type='ndarray')
    if len(pairs) == 0:
        return (np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.float32))
    i, j = pairs[:, 0], pairs[:, 1]
    dists = np.linalg.norm(coords[i] - coords[j], axis=1)
    wts = np.exp(-dists ** 2 / (2 * delta ** 2)).astype(np.float32)
    src = np.concatenate([i, j])
    dst = np.concatenate([j, i])
    wts = np.concatenate([wts, wts])
    return src, dst, wts

def _build_union_edges(mol_src, mol_dst, mol_wts, spa_src, spa_dst, spa_wts, n_cells):
    edge_dict_mol = {}
    for s, d, w in zip(mol_src, mol_dst, mol_wts):
        edge_dict_mol[(int(s), int(d))] = float(w)
    edge_dict_spa = {}
    for s, d, w in zip(spa_src, spa_dst, spa_wts):
        edge_dict_spa[(int(s), int(d))] = float(w)
    all_edges = set(edge_dict_mol.keys()) | set(edge_dict_spa.keys())
    src_list, dst_list, mw_list, sw_list = [], [], [], []
    for (s, d) in all_edges:
        src_list.append(s)
        dst_list.append(d)
        mw_list.append(edge_dict_mol.get((s, d), 0.0))
        sw_list.append(edge_dict_spa.get((s, d), 0.0))
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    mol_weight = torch.tensor(mw_list, dtype=torch.float32)
    spatial_weight = torch.tensor(sw_list, dtype=torch.float32)
    return edge_index, mol_weight, spatial_weight

def _mine_contrastive_pairs(mol_idxs, coords, spatial_src, spatial_dst, r_far, k_neg=10):
    n_cells = len(coords)
    tree = cKDTree(coords)
    pos_set = set()
    for s, d in zip(spatial_src, spatial_dst):
        pos_set.add((int(s), int(d)))
    pos_pairs = torch.tensor(list(pos_set), dtype=torch.long) if pos_set else torch.zeros((0, 2), dtype=torch.long)
    neg_list = []
    for i in range(n_cells):
        count = 0
        for j in mol_idxs[i][1:]:
            j = int(j)
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist > r_far:
                neg_list.append([i, j])
                count += 1
            if count >= k_neg:
                break
    neg_pairs = torch.tensor(neg_list, dtype=torch.long) if neg_list else torch.zeros((0, 2), dtype=torch.long)
    return pos_pairs, neg_pairs

def patient_level_split(adata, train_frac=0.7, val_frac=0.15, seed=42):
    patient_ids = adata.obs['patient_id'].values
    unique_patients = np.unique(patient_ids)
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_patients)
    n = len(unique_patients)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_patients = set(unique_patients[:n_train])
    val_patients = set(unique_patients[n_train:n_train + n_val])
    test_patients = set(unique_patients[n_train + n_val:])
    train_mask = np.array([p in train_patients for p in patient_ids])
    val_mask = np.array([p in val_patients for p in patient_ids])
    test_mask = np.array([p in test_patients for p in patient_ids])
    return train_mask, val_mask, test_mask

def prepare_graph_data(adata, spatial_key='spatial', k_mol=15, r_spatial=82.5, r_far_factor=5.0, k_neg=10):
    pca = adata.obsm['X_pca']
    coords = adata.obsm[spatial_key]
    n_cells = pca.shape[0]
    x = torch.tensor(pca, dtype=torch.float32)
    coords_t = torch.tensor(coords, dtype=torch.float32)
    mol_src, mol_dst, mol_wts = _build_molecular_graph(pca, k=k_mol)
    spa_src, spa_dst, spa_wts = _build_spatial_graph(coords, r=r_spatial)
    edge_index, mol_weight, spatial_weight = _build_union_edges(mol_src, mol_dst, mol_wts, spa_src, spa_dst, spa_wts, n_cells)
    nbrs = NearestNeighbors(n_neighbors=k_mol + 1, algorithm='auto').fit(pca)
    _, mol_idxs = nbrs.kneighbors(pca)
    r_far = r_spatial * r_far_factor
    pos_pairs, neg_pairs = _mine_contrastive_pairs(mol_idxs, coords, spa_src, spa_dst, r_far, k_neg)
    raw_X = adata.X
    library_size = torch.tensor(np.asarray(raw_X.sum(axis=1)).flatten(), dtype=torch.float32)
    x_raw = torch.tensor(raw_X.toarray() if hasattr(raw_X, 'toarray') else np.asarray(raw_X), dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index, mol_weight=mol_weight, spatial_weight=spatial_weight, coords=coords_t, library_size=library_size, x_raw=x_raw, pos_pairs=pos_pairs, neg_pairs=neg_pairs)
    if 'response' in adata.obs.columns and 'patient_id' in adata.obs.columns:
        patient_ids = adata.obs['patient_id'].values
        unique_pids = np.unique(patient_ids)
        patient_masks = []
        responses = []
        for pid in unique_pids:
            mask = torch.tensor(patient_ids == pid, dtype=torch.bool)
            patient_masks.append(mask)
            resp = adata.obs.loc[adata.obs['patient_id'] == pid, 'response'].iloc[0]
            responses.append(float(resp))
        data.patient_masks = patient_masks
        data.y = torch.tensor(responses, dtype=torch.float32)
        train_mask, val_mask, test_mask = patient_level_split(adata)
        data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
        data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
        data.test_mask = torch.tensor(test_mask, dtype=torch.bool)
    return data

def create_synthetic_data(n_cells=5000, n_genes=2000, n_patients=10, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    counts = np.random.poisson(2, size=(n_cells, n_genes)).astype(np.float32)
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
