import os

import pandas as pd
from sklearn.model_selection import train_test_split

from .base import BaseDistDataset, DataframeDataset


def load_texas(num):
    feats = pd.read_csv(f"data/texas/{num}/feats", index_col=False, header=None)
    labels = pd.read_csv(f"data/texas/{num}/labels", index_col=False, header=None) - 1
    train_df, test_df, train_labels, test_labels = train_test_split(feats, labels, test_size=1 / 3, random_state=42)
    return (
        train_df.reset_index(drop=True),
        train_labels.values.squeeze(),
        test_df.reset_index(drop=True),
        test_labels.values.squeeze(),
    )


class DistTexasDataset(BaseDistDataset):
    def __init__(self, args, num=100):
        args.get_logger().debug(f"Loading Texas{num}")
        train_df, train_labels, test_df, test_labels = load_texas(num)
        super(DistTexasDataset, self).__init__(
            args=args,
            name=f"Texas{num}",
            train_datset=DataframeDataset(train_df, train_labels),
            test_dataset=DataframeDataset(test_df, test_labels),
        )


if __name__ == "__main__":
    train_df, train_labels, test_df, test_labels = load_texas(100)
    print(train_df)
    print(train_labels)
    print(test_df)
    print(test_labels)
