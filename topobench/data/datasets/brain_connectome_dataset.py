"""Minimal in-memory dataset holding one PyG Data per brain subject.

A torch.utils.data.Dataset is the lightest object PreProcessor.process accepts
(``isinstance(dataset, torch.utils.data.Dataset)`` + ``[d for d in dataset]``); the
generic PreProcessor does the collate + simplicial lifting, so no InMemoryDataset
download/process ceremony is needed here.
"""

import torch
from torch_geometric.data import Data


class BrainConnectomeDataset(torch.utils.data.Dataset):
    """Holds a list of PyG Data graphs (one per subject).

    Parameters
    ----------
    data_list : list[Data]
        List of PyG Data objects, one per brain subject.
    """

    def __init__(self, data_list: list[Data]) -> None:
        self.data_list = data_list

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
