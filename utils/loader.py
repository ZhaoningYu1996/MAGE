from collections.abc import Mapping
from typing import Any, List, Optional, Sequence, Union

import torch.utils.data
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter
from torch_geometric.data.on_disk_dataset import OnDiskDataset
from torch_geometric.typing import TensorFrame, torch_frame


def custom_collate(batch):
    tree_batch = []
    graph_batch = []
    for item in batch:
        tree_batch.append(item)
        graph_batch.append(item.data)
    tree_batch = Batch.from_data_list(tree_batch)
    graph_batch = Batch.from_data_list(graph_batch)

    return (tree_batch, graph_batch)

class DataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        batch_size: int = 1,
        shuffle: bool = False,
        **kwargs,
    ):

        if isinstance(dataset, OnDiskDataset):
            dataset = range(len(dataset))

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            **kwargs,
        )
