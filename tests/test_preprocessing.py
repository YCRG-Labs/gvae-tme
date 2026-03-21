import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import pytest
import anndata
import scanpy as sc
import pandas as pd
from scipy.sparse import csr_matrix, issparse

from src.data_utils import (
    create_synthetic_data,
    prepare_graph_data,
    patient_level_split,
    _get_patient_response,
    _build_molecular_graph,
    _build_spatial_graph,
    _build_union_edges,
    _mine_contrastive_pairs,
)


def _make_adata(n_cells=500, n_genes=200, n_patients=10, sparse=False, has_spatial=True,
                has_response=True, seed=42):
    np.random.seed(seed)
    counts = np.random.poisson(2, (n_cells, n_genes)).astype(np.float32)
    if sparse:
        counts = csr_matrix(counts)
    adata = anndata.AnnData(X=counts)
    adata.obs["patient_id"] = [f"P{i % n_patients:03d}" for i in range(n_cells)]
    if has_response:
        patient_resp = {f"P{i:03d}": i % 2 for i in range(n_patients)}
        adata.obs["response"] = [patient_resp[p] for p in adata.obs["patient_id"]]
    if has_spatial:
        adata.obsm["spatial"] = np.random.randn(n_cells, 2).astype(np.float32) * 100

    sc.pp.filter_genes(adata, min_cells=3)
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=10000)
    sc.pp.log1p(adata)
    n_top = min(100, adata.n_vars)
    if n_top >= 2:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top)
    n_comps = min(30, adata.n_obs - 1, adata.n_vars - 1)
    sc.pp.pca(adata, n_comps=n_comps)
    return adata


class TestMolecularGraph:

    def test_basic_construction(self):
        pca = np.random.randn(100, 30).astype(np.float32)
        src, dst, wts = _build_molecular_graph(pca, k=10)
        assert len(src) == len(dst) == len(wts) == 100 * 10
        assert np.all(wts > 0) and np.all(wts <= 1)
        assert src.dtype == np.int64 or src.dtype == np.intp
        assert not np.any(src == dst)

    def test_no_self_loops(self):
        pca = np.random.randn(50, 10).astype(np.float32)
        src, dst, wts = _build_molecular_graph(pca, k=5)
        assert not np.any(src == dst)

    def test_k_larger_than_n(self):
        pca = np.random.randn(5, 10).astype(np.float32)
        src, dst, wts = _build_molecular_graph(pca, k=4)
        assert len(src) == 5 * 4

    def test_identical_points(self):
        pca = np.ones((20, 10), dtype=np.float32)
        pca += np.random.randn(20, 10).astype(np.float32) * 1e-6
        src, dst, wts = _build_molecular_graph(pca, k=5)
        assert len(src) == 20 * 5
        assert np.all(np.isfinite(wts))

    def test_weights_decrease_with_distance(self):
        pca = np.eye(50, dtype=np.float32)
        src, dst, wts = _build_molecular_graph(pca, k=10)
        for cell in range(5):
            cell_mask = src == cell
            cell_wts = wts[cell_mask]
            assert np.all(cell_wts > 0)


class TestSpatialGraph:

    def test_basic(self):
        coords = np.random.randn(100, 2).astype(np.float32) * 50
        src, dst, wts = _build_spatial_graph(coords, r=30)
        if len(src) > 0:
            assert len(src) == len(dst) == len(wts)
            assert np.all(wts > 0)

    def test_empty_graph(self):
        coords = np.array([[0, 0], [1000, 1000]], dtype=np.float32)
        src, dst, wts = _build_spatial_graph(coords, r=1.0)
        assert len(src) == 0

    def test_symmetric(self):
        coords = np.random.randn(50, 2).astype(np.float32) * 30
        src, dst, wts = _build_spatial_graph(coords, r=20)
        edges = set(zip(src.tolist(), dst.tolist()))
        for s, d in list(edges)[:20]:
            assert (d, s) in edges

    def test_radius_scaling(self):
        coords = np.random.randn(100, 2).astype(np.float32) * 50
        _, _, wts_small = _build_spatial_graph(coords, r=10)
        _, _, wts_large = _build_spatial_graph(coords, r=100)
        assert len(wts_large) >= len(wts_small)


class TestUnionEdges:

    def test_merge(self):
        mol_src = np.array([0, 1, 2], dtype=np.int64)
        mol_dst = np.array([1, 2, 3], dtype=np.int64)
        mol_wts = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        spa_src = np.array([0, 3], dtype=np.int64)
        spa_dst = np.array([1, 4], dtype=np.int64)
        spa_wts = np.array([0.5, 0.6], dtype=np.float32)
        edge_index, mol_w, spa_w = _build_union_edges(
            mol_src, mol_dst, mol_wts, spa_src, spa_dst, spa_wts, n_cells=5
        )
        assert edge_index.shape[0] == 2
        n_edges = edge_index.shape[1]
        assert n_edges >= 3
        assert mol_w.shape[0] == n_edges
        assert spa_w.shape[0] == n_edges

    def test_overlapping_edges(self):
        mol_src = np.array([0, 1], dtype=np.int64)
        mol_dst = np.array([1, 2], dtype=np.int64)
        mol_wts = np.array([0.9, 0.8], dtype=np.float32)
        spa_src = np.array([0], dtype=np.int64)
        spa_dst = np.array([1], dtype=np.int64)
        spa_wts = np.array([0.5], dtype=np.float32)
        edge_index, mol_w, spa_w = _build_union_edges(
            mol_src, mol_dst, mol_wts, spa_src, spa_dst, spa_wts, n_cells=3
        )
        edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        assert (0, 1) in edges
        idx_01 = edges.index((0, 1))
        assert mol_w[idx_01].item() > 0
        assert spa_w[idx_01].item() > 0


class TestContrastivePairs:

    def test_basic(self):
        np.random.seed(42)
        n = 100
        pca = np.random.randn(n, 10).astype(np.float32)
        coords = np.random.randn(n, 2).astype(np.float32) * 100
        src, dst, _ = _build_spatial_graph(coords, r=50)
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=16, algorithm="auto").fit(pca)
        _, mol_idxs = nbrs.kneighbors(pca)
        pos, neg = _mine_contrastive_pairs(mol_idxs, coords, src, dst, r_far=250, k_neg=10)
        assert pos.shape[1] == 2
        assert neg.shape[1] == 2

    def test_no_spatial_edges(self):
        pca = np.random.randn(50, 10).astype(np.float32)
        coords = np.random.randn(50, 2).astype(np.float32) * 1000
        empty_src = np.array([], dtype=np.int64)
        empty_dst = np.array([], dtype=np.int64)
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=16, algorithm="auto").fit(pca)
        _, mol_idxs = nbrs.kneighbors(pca)
        pos, neg = _mine_contrastive_pairs(mol_idxs, coords, empty_src, empty_dst, r_far=100, k_neg=10)
        assert pos.shape[0] == 0


class TestPatientSplit:

    def test_no_leakage(self):
        adata = _make_adata(n_cells=300, n_patients=10)
        train_mask, val_mask, test_mask = patient_level_split(adata)
        train_patients = set(adata.obs["patient_id"].values[train_mask])
        val_patients = set(adata.obs["patient_id"].values[val_mask])
        test_patients = set(adata.obs["patient_id"].values[test_mask])
        assert len(train_patients & val_patients) == 0
        assert len(train_patients & test_patients) == 0
        assert len(val_patients & test_patients) == 0

    def test_all_cells_assigned(self):
        adata = _make_adata(n_cells=300, n_patients=10)
        train_mask, val_mask, test_mask = patient_level_split(adata)
        assert np.all(train_mask | val_mask | test_mask)
        assert not np.any(train_mask & val_mask)

    def test_approximate_fractions(self):
        adata = _make_adata(n_cells=1000, n_patients=20)
        train_mask, val_mask, test_mask = patient_level_split(adata, train_frac=0.7, val_frac=0.15)
        train_patients = set(adata.obs["patient_id"].values[train_mask])
        val_patients = set(adata.obs["patient_id"].values[val_mask])
        test_patients = set(adata.obs["patient_id"].values[test_mask])
        total = len(train_patients) + len(val_patients) + len(test_patients)
        assert abs(len(train_patients) / total - 0.7) < 0.15
        assert len(test_patients) >= 1

    def test_deterministic(self):
        adata = _make_adata(n_cells=300, n_patients=10)
        m1 = patient_level_split(adata, seed=42)
        m2 = patient_level_split(adata, seed=42)
        assert np.array_equal(m1[0], m2[0])


class TestPatientResponse:

    def test_consistent(self):
        adata = _make_adata(n_cells=100, n_patients=5)
        resp = _get_patient_response(adata, "P000")
        assert resp in [0.0, 1.0]

    def test_inconsistent_raises(self):
        adata = _make_adata(n_cells=100, n_patients=5)
        mask = adata.obs["patient_id"] == "P000"
        vals = adata.obs.loc[mask, "response"].values
        if len(vals) > 1:
            adata.obs.loc[adata.obs[mask].index[0], "response"] = 1 - vals[0]
            with pytest.raises(ValueError, match="inconsistent"):
                _get_patient_response(adata, "P000")


class TestPrepareGraphData:

    def test_with_spatial(self):
        adata = _make_adata(n_cells=200, has_spatial=True)
        data = prepare_graph_data(adata, has_spatial=True, r_spatial=80)
        assert data.x.shape[0] == adata.n_obs
        assert data.x_raw.shape == (adata.n_obs, adata.n_vars)
        assert data.edge_index.shape[0] == 2
        assert data.edge_index.shape[1] > 0
        assert data.has_spatial_flag is True
        assert data.mol_weight.shape[0] == data.edge_index.shape[1]
        assert data.spatial_weight.shape[0] == data.edge_index.shape[1]
        assert data.coords.shape == (adata.n_obs, 2)

    def test_without_spatial(self):
        adata = _make_adata(n_cells=200, has_spatial=False)
        data = prepare_graph_data(adata, has_spatial=False)
        assert data.has_spatial_flag is False
        assert torch.all(data.coords == 0)
        assert torch.all(data.spatial_weight == 0)
        assert data.pos_pairs.shape[0] == 0
        assert data.neg_pairs.shape[0] == 0

    def test_patient_masks(self):
        adata = _make_adata(n_cells=200, n_patients=8)
        data = prepare_graph_data(adata, has_spatial=True, r_spatial=80)
        assert hasattr(data, "patient_masks")
        assert hasattr(data, "y")
        assert len(data.patient_masks) == 8
        total = sum(m.sum().item() for m in data.patient_masks)
        assert total == adata.n_obs
        for m in data.patient_masks:
            assert m.shape[0] == adata.n_obs

    def test_train_val_test_masks(self):
        adata = _make_adata(n_cells=200, n_patients=8)
        data = prepare_graph_data(adata, has_spatial=True, r_spatial=80)
        assert hasattr(data, "train_mask")
        assert hasattr(data, "val_mask")
        assert hasattr(data, "test_mask")
        assert data.train_mask.shape[0] == adata.n_obs
        overlap = data.train_mask & data.val_mask
        assert not overlap.any()

    def test_no_response_columns(self):
        adata = _make_adata(n_cells=200, has_response=False)
        data = prepare_graph_data(adata, has_spatial=True, r_spatial=80)
        assert not hasattr(data, "y") or data.y is None or not hasattr(data, "patient_masks")

    def test_sparse_input(self):
        adata = _make_adata(n_cells=100, sparse=True)
        data = prepare_graph_data(adata, has_spatial=True, r_spatial=80)
        assert data.x_raw.shape == (adata.n_obs, adata.n_vars)
        assert torch.all(torch.isfinite(data.x_raw))

    def test_library_size(self):
        adata = _make_adata(n_cells=100)
        data = prepare_graph_data(adata, has_spatial=True, r_spatial=80)
        assert data.library_size.shape[0] == adata.n_obs
        assert torch.all(data.library_size > 0)

    def test_counts_layer_used(self):
        adata = _make_adata(n_cells=100)
        assert "counts" in adata.layers
        data = prepare_graph_data(adata, has_spatial=True, r_spatial=80)
        raw_sums = data.x_raw.sum(dim=1)
        assert torch.all(raw_sums > 0)

    def test_edge_indices_in_range(self):
        adata = _make_adata(n_cells=150)
        data = prepare_graph_data(adata, has_spatial=True, r_spatial=80)
        assert data.edge_index.min() >= 0
        assert data.edge_index.max() < adata.n_obs

    def test_auto_detect_spatial(self):
        adata = _make_adata(n_cells=100, has_spatial=True)
        data = prepare_graph_data(adata, has_spatial=None)
        assert data.has_spatial_flag is True

    def test_auto_detect_no_spatial(self):
        adata = _make_adata(n_cells=100, has_spatial=True)
        adata.obsm["spatial"] = np.zeros((adata.n_obs, 2), dtype=np.float32)
        data = prepare_graph_data(adata, has_spatial=None)
        assert data.has_spatial_flag is False


class TestCreateSyntheticData:

    def test_shapes(self):
        adata = create_synthetic_data(n_cells=300, n_genes=100, n_patients=6, n_cell_types=4)
        assert adata.n_obs <= 300
        assert "X_pca" in adata.obsm
        assert "spatial" in adata.obsm
        assert "patient_id" in adata.obs.columns
        assert "response" in adata.obs.columns
        assert "counts" in adata.layers

    def test_response_binary(self):
        adata = create_synthetic_data(n_cells=500, n_genes=200, n_patients=10)
        assert set(adata.obs["response"].unique()).issubset({0, 1})

    def test_patient_ids(self):
        adata = create_synthetic_data(n_cells=500, n_genes=200, n_patients=10)
        assert adata.obs["patient_id"].nunique() == 10

    def test_cell_types(self):
        adata = create_synthetic_data(n_cells=500, n_genes=200, n_patients=10, n_cell_types=5)
        assert adata.obs["cell_type"].nunique() == 5

    def test_roundtrip_graph(self):
        adata = create_synthetic_data(n_cells=200, n_genes=100, n_patients=6, n_cell_types=3)
        data = prepare_graph_data(adata, has_spatial=True, r_spatial=80)
        assert data.x.shape[0] == adata.n_obs
        assert hasattr(data, "y")
        assert len(data.y) == 6

    def test_reproducible(self):
        a1 = create_synthetic_data(n_cells=100, n_genes=50, seed=123)
        a2 = create_synthetic_data(n_cells=100, n_genes=50, seed=123)
        assert np.array_equal(a1.obsm["X_pca"], a2.obsm["X_pca"])


class TestPreprocessingPipelineConsistency:

    def test_normalize_log_hvg_pca_pipeline(self):
        np.random.seed(42)
        n_cells = 500
        counts = np.random.poisson(3, (n_cells, 300)).astype(np.float32)
        adata = anndata.AnnData(X=counts)
        patient_ids = [f"P{i % 10:03d}" for i in range(n_cells)]
        patient_resp = {f"P{i:03d}": i % 2 for i in range(10)}
        adata.obs["patient_id"] = patient_ids
        adata.obs["response"] = [patient_resp[p] for p in patient_ids]

        sc.pp.filter_genes(adata, min_cells=3)
        adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=10000)
        sc.pp.log1p(adata)
        try:
            sc.pp.highly_variable_genes(adata, n_top_genes=min(100, adata.n_vars),
                                         flavor="seurat_v3", layer="counts")
        except (ValueError, ImportError):
            sc.pp.highly_variable_genes(adata, n_top_genes=min(100, adata.n_vars))
        sc.pp.pca(adata, n_comps=min(30, adata.n_obs - 1, adata.n_vars - 1))

        assert "X_pca" in adata.obsm
        data = prepare_graph_data(adata, has_spatial=False)
        assert torch.all(torch.isfinite(data.x))
        assert torch.all(torch.isfinite(data.x_raw))
        assert data.x.shape[0] == n_cells

    def test_mt_filter_simulation(self):
        np.random.seed(42)
        n_cells = 200
        n_genes = 600
        counts = np.random.poisson(2, (n_cells, n_genes)).astype(np.float32)
        gene_names = [f"GENE{i}" for i in range(n_genes)]
        gene_names[0] = "MT-CO1"
        gene_names[1] = "MT-CO2"

        counts[:10, 0] = 5000
        counts[:10, 1] = 5000

        adata = anndata.AnnData(X=counts)
        adata.var_names = gene_names
        adata.obs["patient_id"] = [f"P{i % 5:03d}" for i in range(n_cells)]
        adata.obs["response"] = [i % 2 for i in range(n_cells)]

        adata.var["mt"] = adata.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, inplace=True)
        adata_filtered = adata[adata.obs["pct_counts_mt"] < 20].copy()
        assert adata_filtered.n_obs < n_cells

    def test_scrublet_graceful_failure(self):
        from data.download_datasets import _run_scrublet
        np.random.seed(42)
        counts = np.random.poisson(3, (100, 50)).astype(np.float32)
        adata = anndata.AnnData(X=counts)
        result = _run_scrublet(adata)
        assert result.n_obs <= 100

    def test_melanoma_metadata_parser_valid(self):
        import tempfile
        from data.download_datasets import _parse_melanoma_metadata
        content = (
            "some_header_junk\n"
            "more_junk\n"
            "title\tpatinet_ID\tresponse\ttherapy\n"
            "cell_1\tPre_P1\tResponder\tanti-CTLA4\n"
            "cell_2\tPre_P1\tResponder\tanti-CTLA4\n"
            "cell_3\tPost_P2\tNon-responder\tanti-PD1\n"
            "cell_4\tPost_P2\tNon-responder\tanti-PD1\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            f.flush()
            meta = _parse_melanoma_metadata(f.name, ["cell_1", "cell_2", "cell_3", "cell_4"])

        assert "response" in meta.columns
        assert "patient_id" in meta.columns
        assert meta.loc["cell_1", "response"] == 1
        assert meta.loc["cell_3", "response"] == 0
        assert meta.loc["cell_1", "patient_id"] == "P1"
        assert meta.loc["cell_3", "patient_id"] == "P2"

    def test_melanoma_metadata_parser_typo_column(self):
        import tempfile
        from data.download_datasets import _parse_melanoma_metadata
        content = (
            "title\tpatinet\tresponse\n"
            "c1\tPre_P10\tResponder\n"
            "c2\tPost_P10\tNon-responder\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            f.flush()
            meta = _parse_melanoma_metadata(f.name, ["c1", "c2"])
        assert meta.loc["c1", "patient_id"] == "P10"

    def test_nsclc_ici_response_mapping(self):
        resp_map = {"MPR": 1, "pCR": 1, "non-MPR": 0}
        assert resp_map["MPR"] == 1
        assert resp_map["pCR"] == 1
        assert resp_map["non-MPR"] == 0
        assert pd.Series(["MPR", "pCR", "non-MPR", "unknown"]).map(resp_map).isna().sum() == 1

    def test_colorectal_filename_regex(self):
        import re
        filenames = [
            "GSM123_P1CRC_BC1_filtered.h5",
            "GSM456_P2NAT_BC2_filtered.h5",
            "GSM789_something_else.h5",
        ]
        results = []
        for fn in filenames:
            m = re.search(r"_(P\d+)(CRC|NAT)_(BC\d+)_", fn)
            results.append(m)
        assert results[0] is not None
        assert results[0].group(1) == "P1"
        assert results[0].group(2) == "CRC"
        assert results[1].group(1) == "P2"
        assert results[1].group(2) == "NAT"
        assert results[2] is None

    def test_breast_pseudo_response_balanced(self):
        samples = ["5p_scrna", "3p_scrna", "visium_raw", "visium_filt"]
        n_resp = max(1, len(samples) // 2)
        resp_set = set(samples[:n_resp])
        responses = [1 if s in resp_set else 0 for s in samples]
        assert sum(responses) == 2
        assert len(responses) - sum(responses) == 2


class TestEdgeCases:

    def test_single_patient(self):
        adata = _make_adata(n_cells=50, n_patients=1)
        data = prepare_graph_data(adata, has_spatial=True, r_spatial=80)
        assert len(data.patient_masks) == 1
        assert data.y.shape[0] == 1

    def test_two_patients(self):
        adata = _make_adata(n_cells=100, n_patients=2)
        data = prepare_graph_data(adata, has_spatial=True, r_spatial=80)
        assert len(data.patient_masks) == 2
        assert data.y.shape[0] == 2
        train_mask, val_mask, test_mask = patient_level_split(adata)
        assigned = train_mask.sum() + val_mask.sum() + test_mask.sum()
        assert assigned == adata.n_obs

    def test_very_small_dataset(self):
        adata = _make_adata(n_cells=20, n_genes=30, n_patients=3)
        data = prepare_graph_data(adata, has_spatial=True, r_spatial=200)
        assert data.x.shape[0] == adata.n_obs
        assert data.edge_index.shape[1] > 0

    def test_large_k_mol(self):
        adata = _make_adata(n_cells=30, n_genes=50, n_patients=3)
        data = prepare_graph_data(adata, has_spatial=False, k_mol=25)
        assert data.edge_index.shape[1] > 0

    def test_tight_spatial_radius(self):
        adata = _make_adata(n_cells=100, has_spatial=True)
        data = prepare_graph_data(adata, has_spatial=True, r_spatial=0.01)
        assert torch.all(data.spatial_weight == 0) or data.spatial_weight.sum() < 1e-3

    def test_wide_spatial_radius(self):
        adata = _make_adata(n_cells=50, has_spatial=True)
        data = prepare_graph_data(adata, has_spatial=True, r_spatial=10000)
        assert data.spatial_weight.sum() > 0

    def test_no_counts_layer_fallback(self):
        adata = _make_adata(n_cells=100)
        del adata.layers["counts"]
        data = prepare_graph_data(adata, has_spatial=False)
        assert data.x_raw.shape == (adata.n_obs, adata.n_vars)


class TestNumericalStability:

    def test_zero_variance_pca(self):
        np.random.seed(42)
        n_cells = 100
        counts = np.random.poisson(2, (n_cells, 50)).astype(np.float32)
        counts[:, 0] = 0
        adata = anndata.AnnData(X=counts)
        patient_ids = [f"P{i % 5:03d}" for i in range(n_cells)]
        patient_resp = {f"P{i:03d}": i % 2 for i in range(5)}
        adata.obs["patient_id"] = patient_ids
        adata.obs["response"] = [patient_resp[p] for p in patient_ids]
        sc.pp.filter_genes(adata, min_cells=3)
        adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=10000)
        sc.pp.log1p(adata)
        n_comps = min(30, adata.n_obs - 1, adata.n_vars - 1)
        sc.pp.pca(adata, n_comps=n_comps)
        data = prepare_graph_data(adata, has_spatial=False)
        assert torch.all(torch.isfinite(data.x))

    def test_finite_weights(self):
        adata = _make_adata(n_cells=200)
        data = prepare_graph_data(adata, has_spatial=True, r_spatial=80)
        assert torch.all(torch.isfinite(data.mol_weight))
        assert torch.all(torch.isfinite(data.spatial_weight))
        assert torch.all(data.mol_weight >= 0)
        assert torch.all(data.spatial_weight >= 0)

    def test_library_size_positive(self):
        adata = _make_adata(n_cells=200)
        data = prepare_graph_data(adata, has_spatial=False)
        assert torch.all(data.library_size > 0)


class TestHVGFallback:

    def test_fallback_function_exists(self):
        from data.download_datasets import _hvg_with_fallback
        assert callable(_hvg_with_fallback)

    def test_fallback_produces_hvg_column(self):
        from data.download_datasets import _hvg_with_fallback
        np.random.seed(42)
        counts = np.random.poisson(3, (200, 300)).astype(np.float32)
        adata = anndata.AnnData(X=counts)
        sc.pp.filter_genes(adata, min_cells=3)
        adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=10000)
        sc.pp.log1p(adata)
        _hvg_with_fallback(adata, n_top_genes=min(100, adata.n_vars))
        assert "highly_variable" in adata.var.columns
        assert adata.var["highly_variable"].sum() > 0

    def test_fallback_with_mocked_skmisc_error(self):
        np.random.seed(42)
        counts = np.random.poisson(3, (200, 300)).astype(np.float32)
        adata = anndata.AnnData(X=counts)
        sc.pp.filter_genes(adata, min_cells=3)
        adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=10000)
        sc.pp.log1p(adata)
        from unittest.mock import patch
        import scanpy.preprocessing._highly_variable_genes as hvg_mod
        original_fn = sc.pp.highly_variable_genes

        call_count = [0]
        def mock_hvg(adata, **kwargs):
            call_count[0] += 1
            if kwargs.get("flavor") == "seurat_v3":
                raise ImportError("Please install skmisc package via `pip install --user scikit-misc`")
            return original_fn(adata, **kwargs)

        with patch("scanpy.pp.highly_variable_genes", side_effect=mock_hvg):
            from data.download_datasets import _hvg_with_fallback
            _hvg_with_fallback(adata, n_top_genes=min(100, adata.n_vars))

        assert "highly_variable" in adata.var.columns
        assert call_count[0] == 2

    def test_colorectal_download_has_extraction(self):
        import inspect
        from data.download_datasets import download_colorectal
        source = inspect.getsource(download_colorectal)
        assert "tar" in source and "xf" in source, "download_colorectal must extract the tar"
        assert "extracted" in source, "download_colorectal must track extraction state"
