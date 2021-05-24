import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from fltk.util.arguments import Arguments
import logging

from fltk.synthpriv.datasets.cifar100 import get_tuple_from_data_loader
from fltk.datasets.cifar100 import CIFAR100Dataset
from .base import BaseDistDataset, DataframeDataset


def load_cifar_train():
    # TODO:: Generate cifar100.txt
    args = Arguments(logging)
    cifar = CIFAR100Dataset(args)
    train_data = cifar.load_train_dataset()
    test_data = cifar.load_test_dataset()

    # train_df, test_df, train_labels, test_labels = train_test_split(feats, labels, test_size=1 / 3, random_state=42)
    return train_data

def load_cifar_test():
    # TODO:: Generate cifar100.txt
    args = Arguments(logging)
    cifar = CIFAR100Dataset(args)
    test_data = cifar.load_test_dataset()

    # train_df, test_df, train_labels, test_labels = train_test_split(feats, labels, test_size=1 / 3, random_state=42)
    return test_data


class DistCifarDataset(BaseDistDataset):
    def __init__(self, args):
        args.get_logger().debug(f"Loading Cifar100")
        super(DistCifarDataset, self).__init__(
            args=args,
            name=f"Cifar100",
            train_datset=load_cifar_train(),
            test_dataset=load_cifar_test(),
        )


if __name__ == "__main__":
    train_df, train_labels, test_df, test_labels = load_cifar()
    print(train_df)
    print(train_labels)
    print(test_df)
    print(test_labels)
