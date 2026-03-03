import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GroupShuffleSplit
from scipy.spatial import cKDTree


def _build_molecular_graph(pca, k=15):
    n = len(pca)
    # Use Annoy for large datasets (>100K cells) if available
    if n > 100_000:
        try:
            from annoy import AnnoyIndex
            d = pca.shape[1]
            index = AnnoyIndex(d, 'euclidean')
            for i in range(n):
                index.add_item(i, pca[i])
            index.build(50)  # 50 trees for recall >= 0.95
            idxs = np.zeros((n, k + 1), dtype=np.int64)
            dists = np.zeros((n, k + 1), dtype=np.float32)
            for i in range(n):
                nn_idx, nn_dist = index.get_nns_by_item(i, k + 1, include_distances=True)
                idxs[i] = nn_idx
                dists[i] = nn_dist
            print(f"  [annoy] Built k-NN graph with Annoy (50 trees, {n} cells)")
        except ImportError:
            print(f"  [warn] annoy not installed, falling back to sklearn for {n} cells")
            nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto')
            nbrs.fit(pca)
            dists, idxs = nbrs.kneighbors(pca)
    else:
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto')
        nbrs.fit(pca)
        dists, idxs = nbrs.kneighbors(pca)
    sigma = float(np.median(dists[:, -1]))
    src = np.repeat(np.arange(n), k)
    dst = idxs[:, 1:].ravel()
    d = dists[:, 1:].ravel()
    wts = np.exp(-d ** 2 / (2 * sigma ** 2)).astype(np.float32)
    return src, dst, wts


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
    mol_keys = mol_src.astype(np.int64) * n_cells + mol_dst.astype(np.int64)
    spa_keys = spa_src.astype(np.int64) * n_cells + spa_dst.astype(np.int64)
    all_keys, inverse = np.unique(np.concatenate([mol_keys, spa_keys]), return_inverse=True)
    n_mol = len(mol_keys)
    mol_lookup = np.zeros(len(all_keys), dtype=np.float32)
    spa_lookup = np.zeros(len(all_keys), dtype=np.float32)
    mol_lookup[inverse[:n_mol]] = mol_wts
    spa_lookup[inverse[n_mol:]] = spa_wts
    src_arr = (all_keys // n_cells).astype(np.int64)
    dst_arr = (all_keys % n_cells).astype(np.int64)
    edge_index = torch.tensor(np.stack([src_arr, dst_arr]), dtype=torch.long)
    mol_weight = torch.tensor(mol_lookup, dtype=torch.float32)
    spatial_weight = torch.tensor(spa_lookup, dtype=torch.float32)
    return edge_index, mol_weight, spatial_weight


def _mine_contrastive_pairs(mol_idxs, coords, spatial_src, spatial_dst, r_far, k_neg=10):
    n_cells = len(coords)
    if len(spatial_src) > 0:
        pos_edges = np.column_stack([spatial_src, spatial_dst])
        pos_edges_unique = np.unique(pos_edges, axis=0)
        pos_pairs = torch.tensor(pos_edges_unique, dtype=torch.long)
    else:
        pos_pairs = torch.zeros((0, 2), dtype=torch.long)
    neighbor_idxs = mol_idxs[:, 1:]
    neighbor_coords = coords[neighbor_idxs]
    dists = np.linalg.norm(neighbor_coords - coords[:, np.newaxis, :], axis=2)
    far_mask = dists > r_far
    cumsum = np.cumsum(far_mask, axis=1)
    far_mask = far_mask & (cumsum <= k_neg)
    cell_idx, neighbor_rank = np.where(far_mask)
    neg_dst = neighbor_idxs[cell_idx, neighbor_rank]
    if len(cell_idx) == 0:
        nn_dists = dists[:, 0]
        adaptive_r = np.percentile(nn_dists, 75)
        far_mask2 = dists > adaptive_r
        cumsum2 = np.cumsum(far_mask2, axis=1)
        far_mask2 = far_mask2 & (cumsum2 <= k_neg)
        cell_idx, neighbor_rank = np.where(far_mask2)
        neg_dst = neighbor_idxs[cell_idx, neighbor_rank]
    if len(cell_idx) > 0:
        neg_pairs = torch.tensor(np.column_stack([cell_idx, neg_dst]), dtype=torch.long)
    else:
        neg_pairs = torch.zeros((0, 2), dtype=torch.long)
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


def _get_patient_response(adata, pid):
    """Get response for a patient, asserting all cells agree."""
    vals = adata.obs.loc[adata.obs['patient_id'] == pid, 'response'].unique()
    if len(vals) != 1:
        raise ValueError(
            f"Patient {pid} has inconsistent response labels: {vals}. "
            "All cells from the same patient must have the same response."
        )
    return float(vals[0])


def prepare_graph_data(adata, spatial_key='spatial', k_mol=15, r_spatial=82.5, r_far_factor=5.0, k_neg=10, has_spatial=None):
    pca = adata.obsm['X_pca']
    n_cells = pca.shape[0]
    x = torch.tensor(pca, dtype=torch.float32)
    if has_spatial is None:
        has_spatial = spatial_key in adata.obsm and not np.allclose(adata.obsm[spatial_key], 0)
    mol_src, mol_dst, mol_wts = _build_molecular_graph(pca, k=k_mol)
    if has_spatial:
        coords = adata.obsm[spatial_key]
        coords_t = torch.tensor(coords, dtype=torch.float32)
        spa_src, spa_dst, spa_wts = _build_spatial_graph(coords, r=r_spatial)
        edge_index, mol_weight, spatial_weight = _build_union_edges(mol_src, mol_dst, mol_wts, spa_src, spa_dst, spa_wts, n_cells)
        nbrs = NearestNeighbors(n_neighbors=k_mol + 1, algorithm='auto').fit(pca)
        _, mol_idxs = nbrs.kneighbors(pca)
        r_far = r_spatial * r_far_factor
        pos_pairs, neg_pairs = _mine_contrastive_pairs(mol_idxs, coords, spa_src, spa_dst, r_far, k_neg)
    else:
        coords_t = torch.zeros((n_cells, 2), dtype=torch.float32)
        edge_index = torch.tensor(np.array([mol_src, mol_dst]), dtype=torch.long)
        mol_weight = torch.tensor(mol_wts, dtype=torch.float32)
        spatial_weight = torch.zeros_like(mol_weight)
        pos_pairs = torch.zeros((0, 2), dtype=torch.long)
        neg_pairs = torch.zeros((0, 2), dtype=torch.long)
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
            responses.append(_get_patient_response(adata, pid))
        data.patient_masks = patient_masks
        data.y = torch.tensor(responses, dtype=torch.float32)
        train_mask, val_mask, test_mask = patient_level_split(adata)
        data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
        data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
        data.test_mask = torch.tensor(test_mask, dtype=torch.bool)
        pid_to_idx = {pid: i for i, pid in enumerate(unique_pids)}
        train_patients = set()
        val_patients = set()
        test_patients = set()
        for i, pid in enumerate(patient_ids):
            if train_mask[i]:
                train_patients.add(pid)
            elif val_mask[i]:
                val_patients.add(pid)
            else:
                test_patients.add(pid)
        data.train_patient_idx = torch.tensor([pid_to_idx[p] for p in sorted(train_patients)], dtype=torch.long)
        data.val_patient_idx = torch.tensor([pid_to_idx[p] for p in sorted(val_patients)], dtype=torch.long)
        data.test_patient_idx = torch.tensor([pid_to_idx[p] for p in sorted(test_patients)], dtype=torch.long)
    return data


def create_synthetic_data(n_cells=5000, n_genes=2000, n_patients=10, n_cell_types=8, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    cell_types = np.random.choice(n_cell_types, n_cells)
    type_means = np.random.exponential(2.0, size=(n_cell_types, n_genes))
    for ct in range(n_cell_types):
        marker_start = ct * (n_genes // n_cell_types)
        marker_end = marker_start + max(n_genes // (n_cell_types * 2), 10)
        type_means[ct, marker_start:marker_end] *= 5.0
    counts = np.zeros((n_cells, n_genes), dtype=np.float32)
    for i in range(n_cells):
        counts[i] = np.random.poisson(type_means[cell_types[i]])
    patient_ids = np.random.choice(n_patients, n_cells)
    responder_types = set(range(n_cell_types // 2))
    patient_responder_frac = np.zeros(n_patients)
    for pid in range(n_patients):
        mask = patient_ids == pid
        if mask.sum() > 0:
            patient_responder_frac[pid] = np.mean(np.isin(cell_types[mask], list(responder_types)))
    median_frac = np.median(patient_responder_frac)
    patient_response = (patient_responder_frac > median_frac).astype(int)
    cell_response = patient_response[patient_ids]
    centers = np.random.randn(n_cell_types, 2) * 200
    coords = np.zeros((n_cells, 2))
    for i in range(n_cells):
        coords[i] = centers[cell_types[i]] + np.random.randn(2) * 30
    import anndata
    import scanpy as sc
    adata = anndata.AnnData(X=counts)
    adata.obs['patient_id'] = [f'P{i:03d}' for i in patient_ids]
    adata.obs['response'] = cell_response
    adata.obs['cell_type'] = [f'type_{ct}' for ct in cell_types]
    adata.obsm['spatial'] = coords
    sc.pp.filter_cells(adata, min_genes=10)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=10000)
    sc.pp.log1p(adata)
    n_top = min(1000, adata.n_vars)
    if n_top >= 2:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top)
    n_comps = min(50, adata.n_obs - 1, adata.n_vars - 1)
    sc.pp.pca(adata, n_comps=n_comps)
    return adata
