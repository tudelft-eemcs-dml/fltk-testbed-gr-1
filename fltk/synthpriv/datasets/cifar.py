import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from fltk.util.arguments import Arguments
import logging

from fltk.datasets.cifar100 import CIFAR100Dataset
from .base import BaseDistDataset, DataframeDataset

# def get_tuple_from_data_loader(data_loader):
#     """
#     Get a tuple representation of the data stored in a data loader.
#
#     :param data_loader: data loader to get data from
#     :type data_loader: torch.utils.data.DataLoader
#     :return: tuple
#     """
#     return (next(iter(data_loader))[0].numpy(), next(iter(data_loader))[1].numpy())

def load_cifar_train():
    # TODO:: Generate cifar100.txt
    args = Arguments(logging)
    cifar = CIFAR100Dataset(args)
    train_data = cifar.load_train_dataset()
    test_data = cifar.load_test_dataset()

    # train_df, test_df, train_labels, test_labels = train_test_split(feats, labels, test_size=1 / 3, random_state=42)

    print(type(train_data))
    print(train_data)
    print("Tuple size")
    print(len(train_data))
    print(train_data[0].shape)
    train_df = pd.DataFrame(train_data[0])
    train_labels = train_data[1]

    return (train_df, train_labels)

def load_cifar_test():
    # TODO:: Generate cifar100.txt
    args = Arguments(logging)
    cifar = CIFAR100Dataset(args)
    test_data = cifar.load_test_dataset()
    test_df = pd.DataFrame(test_data[0])
    test_labels = test_data[1]
    return (test_df, test_labels)

    # train_df, test_df, train_labels, test_labels = train_test_split(feats, labels, test_size=1 / 3, random_state=42)


class DistCifarDataset(BaseDistDataset):
    def __init__(self, args):
        args.get_logger().debug(f"Loading Cifar100")
        cifarr = CIFAR100Dataset(args)
        train_dataset = cifarr.get_train_dataset_loc()
        test_dataset = cifarr.get_test_dataset_loc()
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

        super(DistCifarDataset, self).__init__(
            args=args,
            name=f"Cifar100",
            train_datset=train_dataset,
            test_dataset=test_dataset,

        )


if __name__ == "__main__":
    train_df = load_cifar_train()
    test_df = load_cifar_test()
    print(train_df)
    # print(train_labels)
    print(test_df)
    # print(test_labels)
