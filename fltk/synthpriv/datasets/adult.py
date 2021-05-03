import pandas as pd
import torch
from fltk.datasets.distributed.dataset import DistDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import numpy as np

column_values = {
    "age": ["continuous"],
    "workclass": [
        "Private",
        "Self-emp-not-inc",
        "Self-emp-inc",
        "Federal-gov",
        "Local-gov",
        "State-gov",
        "Without-pay",
        "Never-worked",
    ],
    "fnlwgt": ["continuous"],
    "education": [
        "Bachelors",
        "Some-college",
        "11th",
        "HS-grad",
        "Prof-school",
        "Assoc-acdm",
        "Assoc-voc",
        "9th",
        "7th-8th",
        "12th",
        "Masters",
        "1st-4th",
        "10th",
        "Doctorate",
        "5th-6th",
        "Preschool",
    ],
    "education-num": ["continuous"],
    "marital-status": [
        "Married-civ-spouse",
        "Divorced",
        "Never-married",
        "Separated",
        "Widowed",
        "Married-spouse-absent",
        "Married-AF-spouse",
    ],
    "occupation": [
        "Tech-support",
        "Craft-repair",
        "Other-service",
        "Sales",
        "Exec-managerial",
        "Prof-specialty",
        "Handlers-cleaners",
        "Machine-op-inspct",
        "Adm-clerical",
        "Farming-fishing",
        "Transport-moving",
        "Priv-house-serv",
        "Protective-serv",
        "Armed-Forces",
    ],
    "relationship": ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"],
    "race": ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
    "sex": ["Female", "Male"],
    "capital-gain": ["continuous"],
    "capital-loss": ["continuous"],
    "hours-per-week": ["continuous"],
    "native-country": [
        "United-States",
        "Cambodia",
        "England",
        "Puerto-Rico",
        "Canada",
        "Germany",
        "Outlying-US(Guam-USVI-etc)",
        "India",
        "Japan",
        "Greece",
        "South",
        "China",
        "Cuba",
        "Iran",
        "Honduras",
        "Philippines",
        "Italy",
        "Poland",
        "Jamaica",
        "Vietnam",
        "Mexico",
        "Portugal",
        "Ireland",
        "France",
        "Dominican-Republic",
        "Laos",
        "Ecuador",
        "Taiwan",
        "Haiti",
        "Columbia",
        "Hungary",
        "Guatemala",
        "Nicaragua",
        "Scotland",
        "Thailand",
        "Yugoslavia",
        "El-Salvador",
        "Trinadad&Tobago",
        "Peru",
        "Hong",
        "Holand-Netherlands",
    ],
    "income": [">50K", "<=50K"],
}

continuous = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]


class AdultDataset(Dataset):
    def __init__(self, df, lbls):
        super().__init__()
        self.df = df
        self.lbls = lbls

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return torch.tensor(self.df.loc[idx]).float(), torch.tensor([self.lbls[idx]]).float()


class DistAdultDataset(DistDataset):
    def __init__(self, args):
        super(DistAdultDataset, self).__init__(args)
        self.init_train_dataset()
        self.init_test_dataset()

    def init_train_dataset(self):
        dist_loader_text = "distributed" if self.args.get_distributed() else ""
        self.get_args().get_logger().debug(f"Loading '{dist_loader_text}' Adult train data")

        train_df = pd.read_csv("data/adult/adult.data", delimiter=", ")
        train_df.columns = column_values.keys()
        self.scaler = StandardScaler().fit(train_df[continuous])
        train_df[continuous] = self.scaler.transform(train_df[continuous])
        labels = LabelEncoder().fit_transform(train_df.income)
        train_df = pd.get_dummies(train_df.drop(columns="income"))
        self.columns = train_df.columns

        self.train_dataset = AdultDataset(train_df, labels)

        self.train_sampler = (
            DistributedSampler(self.train_dataset, rank=self.args.get_rank(), num_replicas=self.args.get_world_size())
            if self.args.get_distributed()
            else None
        )

        self.train_loader = DataLoader(self.train_dataset, batch_size=16, sampler=self.train_sampler)

    def init_test_dataset(self):
        self.get_args().get_logger().debug("Loading Adult test data")

        test_df = pd.read_csv("data/adult/adult.test", delimiter=", ")
        test_df.columns = column_values.keys()
        test_df.income = test_df.income.apply(lambda x: x.replace(".", ""))
        test_df[continuous] = self.scaler.transform(test_df[continuous])
        labels = LabelEncoder().fit_transform(test_df.income)
        test_df = pd.get_dummies(test_df.drop(columns="income"))
        test_df[list(set(self.columns) - set(test_df.columns))] = np.zeros(len(test_df))

        self.test_dataset = AdultDataset(test_df[self.columns], labels)

        self.test_sampler = (
            DistributedSampler(self.test_dataset, rank=self.args.get_rank(), num_replicas=self.args.get_world_size())
            if self.args.get_distributed()
            else None
        )

        self.test_loader = DataLoader(self.test_dataset, batch_size=16, sampler=self.test_sampler)

    def load_train_dataset(self):
        self.get_args().get_logger().debug("Loading Adult train data")
        self.args.set_sampler(self.train_sampler)

        train_data = self.get_tuple_from_data_loader(self.train_loader)
        dist_loader_text = "distributed" if self.args.get_distributed() else ""
        self.get_args().get_logger().debug(f"Finished loading '{dist_loader_text}' Adult train data")

        return train_data

    def load_test_dataset(self):
        self.get_args().get_logger().debug("Loading Adult test data")
        self.args.set_sampler(self.test_sampler)

        test_data = self.get_tuple_from_data_loader(self.test_loader)

        self.get_args().get_logger().debug("Finished loading Adult test data")

        return test_data


if __name__ == "__main__":
    train_df = pd.read_csv("data/adult/adult.data", delimiter=", ")
    train_df.columns = column_values.keys()
    scaler = StandardScaler().fit(train_df[continuous])
    train_df[continuous] = scaler.transform(train_df[continuous])
    labels = LabelEncoder().fit_transform(train_df.income)
    train_df = pd.get_dummies(train_df.drop(columns="income"))
    print(train_df)

    test_df = pd.read_csv("data/adult/adult.test", delimiter=", ")
    test_df.columns = column_values.keys()
    test_df.income = test_df.income.apply(lambda x: x.replace(".", ""))
    test_df[continuous] = scaler.transform(test_df[continuous])
    labels = LabelEncoder().fit_transform(test_df.income)
    test_df = pd.get_dummies(test_df.drop(columns="income"))
    test_df[list(set(train_df.columns) - set(test_df.columns))] = np.zeros(len(test_df))
    print(test_df[train_df.columns])
