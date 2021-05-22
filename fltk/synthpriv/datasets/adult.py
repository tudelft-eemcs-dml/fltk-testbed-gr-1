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
    # "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]


class DistAdultDataset(BaseDistDataset):
    def __init__(self, args):
        self.train_df, self.train_labels, self.test_df, self.test_labels = self.load()
        super(DistAdultDataset, self).__init__(
            args=args,
            name="Adult",
            train_datset=DataframeDataset(self.train_df, self.train_labels),
            test_dataset=DataframeDataset(self.test_df, self.test_labels),
        )

    def preprocess(self, df):
        salary_map = {"<=50K": 1, "<=50K.": 1, ">50K": 0, ">50K.": 0}
        df["income"] = df["income"].map(salary_map).astype(int)
        df["sex"] = df["sex"].map({"Male": 1, "Female": 0}).astype(int)
        df["native-country"] = df["native-country"].replace("?", np.nan)
        df["workclass"] = df["workclass"].replace("?", np.nan)
        df["occupation"] = df["occupation"].replace("?", np.nan)
        df.dropna(how="any", inplace=True)

        df.loc[df["native-country"] != "United-States", "native-country"] = "Non-US"
        df.loc[df["native-country"] == "United-States", "native-country"] = "US"
        df["native-country"] = df["native-country"].map({"US": 1, "Non-US": 0}).astype(int)

        df["marital-status"] = df["marital-status"].replace(
            ["Divorced", "Married-spouse-absent", "Never-married", "Separated", "Widowed"], "Single"
        )
        df["marital-status"] = df["marital-status"].replace(["Married-AF-spouse", "Married-civ-spouse"], "Couple")
        df["marital-status"] = df["marital-status"].map({"Couple": 0, "Single": 1})
        rel_map = {"Unmarried": 0, "Wife": 1, "Husband": 2, "Not-in-family": 3, "Own-child": 4, "Other-relative": 5}
        df["relationship"] = df["relationship"].map(rel_map)

        df["race"] = df["race"].map(
            {"White": 0, "Amer-Indian-Eskimo": 1, "Asian-Pac-Islander": 2, "Black": 3, "Other": 4}
        )

        def f(x):
            if x["workclass"] == "Federal-gov" or x["workclass"] == "Local-gov" or x["workclass"] == "State-gov":
                return 0
            elif x["workclass"] == "Private":
                return 1
            elif x["workclass"] == "Self-emp-inc" or x["workclass"] == "Self-emp-not-inc":
                return 2
            else:
                return 3

        df["workclass"] = df.apply(f, axis=1)
        df.drop(labels=["education", "occupation"], axis=1, inplace=True)

        df.loc[(df["capital-gain"] > 0), "capital-gain"] = 1
        df.loc[(df["capital-gain"] == 0, "capital-gain")] = 0
        df.loc[(df["capital-loss"] > 0), "capital-loss"] = 1
        df.loc[(df["capital-loss"] == 0, "capital-loss")] = 0

        df.drop(["fnlwgt"], axis=1, inplace=True)
        return df.reset_index(drop=True)

    def load(self):
        train_df = pd.read_csv("data/adult/adult.data", sep=", ", index_col=False, header=None)
        train_df.columns = columns
        train_df = self.preprocess(train_df)
        train_labels = train_df.income
        train_df = train_df.drop(columns="income")

        test_df = pd.read_csv("data/adult/adult.test", sep=", ", index_col=False, header=None)
        test_df.columns = columns
        test_df = self.preprocess(test_df)
        test_labels = test_df.income
        test_df = test_df.drop(columns="income")

        return train_df, train_labels, test_df, test_labels
