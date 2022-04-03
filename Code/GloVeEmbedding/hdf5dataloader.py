import contextlib
from dataclasses import dataclass, field

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class CoOccurrenceDataset(torch.utils.data.Dataset):
    token_ids: torch.Tensor
    co_occurr_counts: torch.Tensor

    def __getitem__(self, index):
        return [self.token_ids[index], self.co_occurr_counts[index]]

    def __len__(self):
        return self.token_ids.size()[0]


@dataclass
class HDF5DataLoader:
    filepath: str
    dataset_name: str
    batch_size: int
    device: str
    dataset: h5py.Dataset = field(init=False)

    def iter_batches(self):
        chunks = list(self.dataset.iter_chunks())
        np.random.shuffle(chunks)
        for chunk in chunks:
            chunked_dataset = self.dataset[chunk]
            dataloader = torch.utils.data.DataLoader(
                dataset=CoOccurrenceDataset(
                    token_ids=torch.from_numpy(chunked_dataset[:,:2]).long(),
                    co_occurr_counts=\
                        torch.from_numpy(chunked_dataset[:,2]).float()
                ),
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=True
            )
            for batch in dataloader:
                batch = [_.to(self.device) for _ in batch]
                yield batch

    @contextlib.contextmanager
    def open(self):
        with h5py.File(self.filepath, "r") as file:
            self.dataset = file[self.dataset_name]
            yield
