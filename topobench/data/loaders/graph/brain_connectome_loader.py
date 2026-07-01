"""Loader for the PNC brain-connectome community-detection dataset.

Reads a framework-agnostic per-subject npz cache (built by the harness env,
tbc/data/brain_cache.py) and wraps each subject as a PyG Data. Does NOT touch
the raw data drive from inside TopoBench.
"""

from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from torch_geometric.data import Data

from topobench.data.datasets import BrainConnectomeDataset
from topobench.data.loaders.base import AbstractLoader


class BrainConnectomeDatasetLoader(AbstractLoader):
    """Load per-subject brain connectome graphs from an npz cache.

    Parameters
    ----------
    parameters : DictConfig
        Must contain ``data_dir``, ``data_name`` and ``cache_dir`` (absolute
        path to the directory of ``sub-*.npz`` files).
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> BrainConnectomeDataset:
        """Build the in-memory dataset from the npz cache.

        Returns
        -------
        BrainConnectomeDataset
            One PyG ``Data`` per subject.

        Raises
        ------
        FileNotFoundError
            If no ``sub-*.npz`` files are found in ``cache_dir``.
        """
        cache_dir = Path(self.parameters["cache_dir"])
        data_list = []
        for npz_path in sorted(cache_dir.glob("sub-*.npz")):
            with np.load(npz_path) as z:
                data_list.append(
                    Data(
                        x=torch.tensor(z["x"], dtype=torch.float),
                        edge_index=torch.tensor(
                            z["edge_index"], dtype=torch.long
                        ),
                        y=torch.tensor(z["y"], dtype=torch.long),
                    )
                )
        if not data_list:
            raise FileNotFoundError(f"no sub-*.npz found in {cache_dir}")
        return BrainConnectomeDataset(data_list)
