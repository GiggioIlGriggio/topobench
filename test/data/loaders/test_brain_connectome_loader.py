"""Tests for BrainConnectomeDatasetLoader."""

import numpy as np
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data

from topobench.data.loaders import BrainConnectomeDatasetLoader


def _tiny_cache(tmp_path):
    d = tmp_path / "cache"
    d.mkdir()
    for s in ("0000000001", "0000000002", "0000000003"):
        np.savez(
            d / f"sub-{s}.npz",
            x=np.random.rand(20, 6).astype("float32"),
            edge_index=np.array([[0, 1], [1, 0]], dtype="int64"),
            y=np.random.randint(0, 7, size=20).astype("int64"),
        )
    return d


def test_loader_yields_pyg_dataset(tmp_path):
    cache = _tiny_cache(tmp_path)
    params = OmegaConf.create(
        {
            "data_dir": str(tmp_path),
            "data_name": "brain_pnc_community",
            "cache_dir": str(cache),
        }
    )
    dataset, data_dir = BrainConnectomeDatasetLoader(params).load()
    assert isinstance(dataset, torch.utils.data.Dataset)
    items = [d for d in dataset]  # PreProcessor iterates like this
    assert len(items) == 3
    assert isinstance(items[0], Data)
    assert items[0].x.shape[1] == 6 and items[0].y.dtype == torch.long
    assert items[0].edge_index.dtype == torch.long


def test_dataset_exposes_collated_inmemory_interface(tmp_path):
    """No-transform (graph models, e.g. GCN) path must find ``_data``/``slices``.

    ``PreProcessor`` has two branches: with a transform (the simplicial lift used
    by MPSN) it iterates the dataset and collates itself; with *no* transform it
    reads ``dataset._data`` / ``dataset.slices`` directly. Only the lift path was
    ever exercised, so a graph model hit the second branch and crashed with
    ``AttributeError: '...' object has no attribute '_data'``. The loaded dataset
    must therefore expose the collated InMemoryDataset interface as well.
    """
    cache = _tiny_cache(tmp_path)
    params = OmegaConf.create(
        {
            "data_dir": str(tmp_path),
            "data_name": "brain_pnc_community",
            "cache_dir": str(cache),
        }
    )
    dataset, _ = BrainConnectomeDatasetLoader(params).load()
    # Exactly the two attribute reads PreProcessor performs with no transform:
    data, slices = dataset._data, dataset.slices
    assert data is not None and slices is not None
    # 3 subjects x 20 nodes -> node-level tensors collate to 60 rows.
    assert data.x.shape == (60, 6)
    assert slices["x"][-1].item() == 60
    assert slices["y"][-1].item() == 60
    # Iteration (the lift path) must still yield the original per-subject graphs.
    items = [d for d in dataset]
    assert len(items) == 3 and items[0].x.shape[0] == 20
