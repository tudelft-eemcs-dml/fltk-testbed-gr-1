import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .base import BaseDistDataset, DataframeDataset

columns = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

continuous = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]


def load_adult():
    train_df = pd.read_csv("data/adult/adult.data", sep=", ", index_col=False, header=None)
    train_df.columns = columns  # set column names
    scaler = StandardScaler().fit(train_df[continuous])
    train_df[continuous] = scaler.transform(train_df[continuous])  # scale continuous column to zero mean, unit variance
    train_labels = LabelEncoder().fit_transform(train_df.income)  # income level to 0 or 1
    train_df = train_df.drop(columns="income")
    train_df = pd.get_dummies(train_df)  # one-hot encode categorical variables

    test_df = pd.read_csv("data/adult/adult.test", sep=", ", index_col=False, header=None)
    test_df.columns = columns
    test_df.income = test_df.income.apply(lambda x: x.replace(".", ""))  # test data has periods at end for some reason
    test_df[continuous] = scaler.transform(test_df[continuous])
    test_labels = LabelEncoder().fit_transform(test_df.income)
    test_df = test_df.drop(columns="income")
    test_df = pd.get_dummies(test_df)
    test_df[list(set(train_df.columns) - set(test_df.columns))] = np.zeros(len(test_df))  # add in missing columns
    test_df = test_df[train_df.columns]  # reorder columns to match training data

    return train_df, train_labels, test_df, test_labels


class DistAdultDataset(BaseDistDataset):
    def __init__(self, args):
        train_df, train_labels, test_df, test_labels = load_adult()
        super(DistAdultDataset, self).__init__(
            args=args,
            name="Adult",
            train_datset=DataframeDataset(train_df, train_labels),
            test_dataset=DataframeDataset(test_df, test_labels),
        )


if __name__ == "__main__":
    train_df, train_labels, test_df, test_labels = load_adult()
    print(train_df)
    print(train_labels)
    print(test_df)
    print(test_labels)
