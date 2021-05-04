import pandas as pd
from sklearn.model_selection import train_test_split

from .base import BaseDistDataset, DataframeDataset


def load_purchase():
    feats = pd.read_csv(f"data/purchase100.txt", index_col=False, header=None)
    last_col = feats.columns[-1]
    labels = feats[last_col].apply(lambda x: int(x.split(";")[-1]))
    feats[last_col] = feats[last_col].apply(lambda x: int(x.split(";")[0]))
    train_df, test_df, train_labels, test_labels = train_test_split(feats, labels, test_size=1 / 3, random_state=42)
    return (
        train_df.reset_index(drop=True),
        train_labels.values.squeeze(),
        test_df.reset_index(drop=True),
        test_labels.values.squeeze(),
    )


class DistPurchaseDataset(BaseDistDataset):
    def __init__(self, args):
        args.get_logger().debug(f"Loading Purchase100")
        train_df, train_labels, test_df, test_labels = load_purchase()
        super(DistPurchaseDataset, self).__init__(
            args=args,
            name=f"Purchase100",
            train_datset=DataframeDataset(train_df, train_labels),
            test_dataset=DataframeDataset(test_df, test_labels),
        )


if __name__ == "__main__":
    train_df, train_labels, test_df, test_labels = load_purchase()
    print(train_df)
    print(train_labels)
    print(test_df)
    print(test_labels)
