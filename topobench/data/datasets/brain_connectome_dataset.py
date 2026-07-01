"""Minimal in-memory dataset holding one PyG Data per brain subject.

A ``torch.utils.data.Dataset`` is the lightest object ``PreProcessor.process``
accepts (``isinstance(dataset, torch.utils.data.Dataset)`` + ``[d for d in
dataset]``); when a transform is configured (the simplicial lift used by MPSN)
the generic PreProcessor iterates the dataset and does the collate itself.

With *no* transform (plain graph models, e.g. GCN) PreProcessor instead reads
``dataset._data`` / ``dataset.slices`` directly, so we also expose the collated
``InMemoryDataset`` interface up front. Iteration still yields the original
per-subject graphs, keeping the lift path unchanged.
"""

import torch
from torch_geometric.data import Data, InMemoryDataset


class BrainConnectomeDataset(torch.utils.data.Dataset):
    """Holds a list of PyG Data graphs (one per subject).

    Parameters
    ----------
    data_list : list[Data]
        List of PyG Data objects, one per brain subject.
    """

    def __init__(self, data_list: list[Data]) -> None:
        self.data_list = data_list
        # Collate once so the no-transform PreProcessor branch, which reads
        # ``dataset._data`` / ``dataset.slices``, has the InMemoryDataset interface
        # it expects. The lift branch ignores these and iterates instead.
        self._data, self.slices = InMemoryDataset.collate(data_list)

    def __len__(self) -> int:
        """Return the number of subjects in the dataset.

        Returns
        -------
        int
            Number of subjects.
        """
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Data:
        """Return the PyG Data object for subject at index ``idx``.

        Parameters
        ----------
        idx : int
            Index of the subject.

        Returns
        -------
        Data
            PyG Data object containing node features, edge index, and labels.
        """
        return self.data_list[idx]
