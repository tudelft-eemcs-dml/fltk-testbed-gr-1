import numpy as np
import pandas as pd
import torch
from torch.utils.data.sampler import RandomSampler
from fltk.datasets.distributed.dataset import DistDataset
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class BaseDistDataset(DistDataset):
    """
    Has all the boilerplate functions that DistDataset doesn't implement on its own for some reason...
    """

    def __init__(self, args, name, train_datset, test_dataset):
        super(BaseDistDataset, self).__init__(args)
        self.name = name
        self.train_dataset = train_datset
        self.test_dataset = test_dataset
        self.n_workers = 16
        self.init_train_dataset()
        self.init_test_dataset()

    def init_train_dataset(self):
        self.train_sampler = (
            DistributedSampler(
                self.train_dataset, rank=self.args.get_rank(), num_replicas=self.args.get_world_size(), shuffle=True
            )
            if self.args.get_distributed()
            else RandomSampler(self.train_dataset)
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            sampler=self.train_sampler,
            num_workers=self.n_workers,
            prefetch_factor=int(self.args.batch_size / self.n_workers),
            pin_memory=True,
        )

    def init_test_dataset(self):
        self.test_sampler = (
            DistributedSampler(
                self.test_dataset, rank=self.args.get_rank(), num_replicas=self.args.get_world_size(), shuffle=True
            )
            if self.args.get_distributed()
            else RandomSampler(self.test_dataset)
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.args.batch_size,
            sampler=self.test_sampler,
            num_workers=self.n_workers,
            prefetch_factor=int(self.args.batch_size / self.n_workers),
            pin_memory=True,
        )

    def load_train_dataset(self):
        self.args.get_logger().debug(f"Loading {self.name} train data")
        self.args.set_sampler(self.train_sampler)
        train_data = self.get_tuple_from_data_loader(self.train_loader)
        return train_data

    def load_test_dataset(self):
        self.args.get_logger().debug(f"Loading {self.name} test data")
        self.args.set_sampler(self.test_sampler)
        test_data = self.get_tuple_from_data_loader(self.test_loader)
        return test_data


class DataframeDataset(Dataset):
    """
    Loads rows from a dataframe.
    """

    def __init__(self, df: pd.DataFrame, lbls: np.array):
        super().__init__()
        self.data = torch.tensor(df.values).float()
        self.lbls = torch.tensor(lbls).long().squeeze()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.lbls[idx]
