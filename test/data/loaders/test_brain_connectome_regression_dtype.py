import numpy as np
import torch
from omegaconf import OmegaConf
from topobench.data.loaders import BrainConnectomeDatasetLoader


def _write_npz(d, y_val):
    x = np.eye(4, dtype=np.float32)
    ei = np.array([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=np.int64)
    np.savez(d / "sub-0000000001.npz", x=x, edge_index=ei, y=np.array([y_val], dtype=np.float32))


def test_regression_loader_yields_float_y(tmp_path):
    _write_npz(tmp_path, 15.0)
    params = OmegaConf.create(
        {"data_dir": str(tmp_path), "data_name": "t", "cache_dir": str(tmp_path), "y_dtype": "float"}
    )
    ds = BrainConnectomeDatasetLoader(params).load_dataset()
    assert ds[0].y.dtype == torch.float
    assert float(ds[0].y[0]) == 15.0


def test_default_loader_still_yields_long_y(tmp_path):
    _write_npz(tmp_path, 3.0)  # a class id, stored as float in npz
    params = OmegaConf.create({"data_dir": str(tmp_path), "data_name": "t", "cache_dir": str(tmp_path)})
    ds = BrainConnectomeDatasetLoader(params).load_dataset()
    assert ds[0].y.dtype == torch.long
