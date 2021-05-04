import numpy as np
import pandas as pd
import torch
from fltk.datasets.distributed.dataset import DistDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
        self.init_train_dataset()
        self.init_test_dataset()

    def init_train_dataset(self):
        self.get_args().get_logger().debug(f"Loading {self.name} train data")
        self.train_sampler = (
            DistributedSampler(self.train_dataset, rank=self.args.get_rank(), num_replicas=self.args.get_world_size())
            if self.args.get_distributed()
            else None
        )
        self.train_loader = DataLoader(self.train_dataset, batch_size=16, sampler=self.train_sampler)

    def init_test_dataset(self):
        self.get_args().get_logger().debug(f"Loading {self.name} test data")
        self.test_sampler = (
            DistributedSampler(self.test_dataset, rank=self.args.get_rank(), num_replicas=self.args.get_world_size())
            if self.args.get_distributed()
            else None
        )
        self.test_loader = DataLoader(self.test_dataset, batch_size=16, sampler=self.test_sampler)

    def load_train_dataset(self):
        self.get_args().get_logger().debug(f"Loading {self.name} train data")
        self.args.set_sampler(self.train_sampler)
        train_data = self.get_tuple_from_data_loader(self.train_loader)
        return train_data

    def load_test_dataset(self):
        self.get_args().get_logger().debug(f"Loading {self.name} test data")
        self.args.set_sampler(self.test_sampler)
        test_data = self.get_tuple_from_data_loader(self.test_loader)
        return test_data


class DataframeDataset(Dataset):
    """
    Loads rows from a dataframe.
    """

    def __init__(self, df, lbls):
        super().__init__()
        self.df = df
        self.lbls = lbls

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return torch.tensor(self.df.loc[idx]).float(), torch.tensor(self.lbls[idx])
