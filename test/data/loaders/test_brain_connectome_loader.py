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
