import collections
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, DistributedSampler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize

sys.path.append("fltk/synthpriv/synthetic")
from fltk.datasets.distributed.dataset import DistDataset
from fltk.synthpriv.datasets.base import DataframeDataset
from fltk.synthpriv.synthetic.models import dp_wgan, pate_gan, ron_gauss
from fltk.synthpriv.synthetic.models.IMLE import imle
from fltk.synthpriv.synthetic.models.Private_PGM import private_pgm

# good values for Adult dataset from borealis repo
# model : {epsilon : sigma}
DEFAULTS = {
    "imle": {
        2: 0.8,
        5: 0.7,
        8: 0.6,
    },
    "pate-gan": {
        2: 1e-4,
        5: 3e-4,
        8: 3e-4,
    },
    "dp-wgan": {
        2: 1.0,
        5: 0.9,
        8: 0.8,
    },
}


class SyntheticDataset(DistDataset):
    def __init__(self, real_dataset, args, device, id, model="imle", sigma=0.6, target_epsilon=8):
        super().__init__(args)

        self.real_dataset = real_dataset

        cache_path = f"data/IMLE_synthetic_{real_dataset.__class__.__name__}_eps={target_epsilon}_sig={sigma}_{id}.pkl"
        if not os.path.exists(cache_path):
            try:
                with torch.cuda.device(device):
                    self.fit(model, sigma, target_epsilon)
            except Exception as e:
                print(e)
                exit(1)
            joblib.dump((self.train_df, self.train_labels, self.test_df, self.test_labels), cache_path)
        else:
            self.train_df, self.train_labels, self.test_df, self.test_labels = joblib.load(cache_path)

        self.train_df = np.round(self.train_df)
        self.train_labels = np.round(self.train_labels)

        self.test_df = self.real_dataset.test_df
        self.test_labels = self.real_dataset.test_labels

        for col in self.real_dataset.train_df.columns:
            print(col)
            print(np.unique(self.real_dataset.train_df[col]))
            print(np.unique(self.train_df[col]))

        try:
            self.name = "Synthetic" + real_dataset.__class__.__name__
            self.train_dataset = DataframeDataset(self.train_df, self.train_labels)
            self.test_dataset = DataframeDataset(self.test_df, self.test_labels)
            self.n_workers = 16

            self.train_sampler = DistributedSampler(self.train_dataset, num_replicas=1, rank=0, shuffle=True)
            self.test_sampler = DistributedSampler(self.test_dataset, num_replicas=1, rank=0, shuffle=True)

            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.args.batch_size,
                sampler=self.train_sampler,
                num_workers=self.n_workers,
                prefetch_factor=int(self.args.batch_size / self.n_workers),
                pin_memory=True,
            )

            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.args.batch_size,
                sampler=self.test_sampler,
                num_workers=self.n_workers,
                prefetch_factor=int(self.args.batch_size / self.n_workers),
                pin_memory=True,
            )
        except Exception as e:
            print(e)
            exit(1)

    def fit(self, model_name, sigma, target_epsilon):
        print(f"Fitting synthetic {self.real_dataset.__class__.__name__}")

        cols = list(self.real_dataset.train_df.columns) + ["income"]
        train = pd.DataFrame(columns=cols)
        for batch, labels in self.real_dataset.train_loader:
            train = pd.concat(
                (train, pd.DataFrame(torch.cat((batch, labels[:, None]), axis=1).cpu().numpy(), columns=cols)), axis=0
            )
        train = train.reset_index(drop=True)

        test = pd.DataFrame(columns=cols)
        for batch, labels in self.real_dataset.test_loader:
            test = pd.concat(
                (test, pd.DataFrame(torch.cat((batch, labels[:, None]), axis=1).cpu().numpy(), columns=cols)), axis=0
            )
        test = test.reset_index(drop=True)

        target_variable = "income"

        data_columns = self.real_dataset.train_df.columns

        class_ratios = (
            train[target_variable].sort_values().groupby(train[target_variable]).size().values / train.shape[0]
        )

        X_train = np.nan_to_num(train.drop([target_variable], axis=1).values)
        y_train = np.nan_to_num(train[target_variable].values)
        X_test = np.nan_to_num(test.drop([target_variable], axis=1).values)
        y_test = np.nan_to_num(test[target_variable].values)

        input_dim = X_train.shape[1]
        z_dim = int(input_dim / 4 + 1) if input_dim % 4 == 0 else int(input_dim / 4)

        print("Training:", model_name.upper())
        if model_name == "pate-gan":
            Hyperparams = collections.namedtuple(
                "Hyperarams", "batch_size num_teacher_iters num_student_iters num_moments lap_scale class_ratios lr"
            )
            Hyperparams.__new__.__defaults__ = (None, None, None, None, None, None, None)

            model = pate_gan.PATE_GAN(input_dim, z_dim, 10, target_epsilon, 1e-5, "classification")
            model.train(
                X_train,
                y_train,
                Hyperparams(
                    batch_size=64,
                    num_teacher_iters=5,
                    num_student_iters=5,
                    num_moments=100,
                    lap_scale=sigma,
                    class_ratios=class_ratios,
                    lr=1e-4,
                ),
            )
        elif model_name == "dp-wgan":
            Hyperparams = collections.namedtuple(
                "Hyperarams",
                "batch_size micro_batch_size clamp_lower clamp_upper clip_coeff sigma class_ratios lr num_epochs",
            )
            Hyperparams.__new__.__defaults__ = (None, None, None, None, None, None, None, None, None)

            model = dp_wgan.DP_WGAN(input_dim, z_dim, target_epsilon, 1e-5, "classification")
            model.train(
                X_train,
                y_train,
                Hyperparams(
                    batch_size=64,
                    micro_batch_size=8,
                    clamp_lower=0.01,
                    clamp_upper=0.01,
                    clip_coeff=0.1,
                    sigma=sigma,
                    class_ratios=class_ratios,
                    lr=5e-5,
                    num_epochs=500,
                ),
                private=True,
            )
        elif model_name == "ron-gauss":
            model = ron_gauss.RONGauss(z_dim, target_epsilon, 1e-5, "classification")
        elif model_name == "imle":
            Hyperparams = collections.namedtuple(
                "Hyperarams",
                "lr batch_size micro_batch_size sigma num_epochs class_ratios clip_coeff decay_step decay_rate staleness num_samples_factor",
            )
            Hyperparams.__new__.__defaults__ = (None, None, None, None, None, None, None, None)

            model = imle.IMLE(input_dim, z_dim, target_epsilon, 1e-5, "classification")
            model.train(
                X_train,
                y_train,
                Hyperparams(
                    lr=1e-3,
                    batch_size=64,
                    micro_batch_size=8,
                    sigma=sigma,
                    num_epochs=500,
                    class_ratios=class_ratios,
                    clip_coeff=0.1,
                    decay_step=25,
                    decay_rate=1.0,
                    staleness=5,
                    num_samples_factor=10,
                ),
                private=True,
            )
        elif model_name == "private-pgm":
            combined = train.append(test)
            config = {}
            for col in combined.columns:
                col_count = len(combined[col].unique())
                config[col] = col_count
            model = private_pgm.Private_PGM(target_variable, target_epsilon, 1e-5)
            model.train(train, config)

        if model_name == "ron-gauss":
            X_syn, y_syn, dp_mean_dict = model.generate(X_train, y=y_train)
            for label in np.unique(y_test):
                idx = np.where(y_test == label)
                x_class = X_test[idx]
                x_norm = normalize(x_class)
                x_bar = x_norm - dp_mean_dict[label]
                x_bar = normalize(x_bar)
                X_test[idx] = x_bar
        elif model_name == "imle" or model == "dp-wgan" or model == "pate-gan":
            syn_data = model.generate(X_train.shape[0], class_ratios)
            X_syn, y_syn = syn_data[:, :-1], syn_data[:, -1]
        elif model_name == "private-pgm":
            syn_data = model.generate()
            X_syn, y_syn = syn_data[:, :-1], syn_data[:, -1]

        print("\nAUC scores of downstream classifiers on real data : ")
        for learner in [
            LogisticRegression(),
            RandomForestClassifier(),
            MLPClassifier(early_stopping=True),
            GaussianNB(),
            GradientBoostingClassifier(),
        ]:
            learner.fit(X_train, y_train)
            pred_probs = learner.predict_proba(X_test)
            auc_score = roc_auc_score(y_test, pred_probs[:, 1])
            print("-" * 40)
            print(f"{learner.__class__.__name__}: {auc_score}")

        print("\nAUC scores of downstream classifiers on fake data : ")
        for learner in [
            LogisticRegression(),
            RandomForestClassifier(),
            MLPClassifier(early_stopping=True),
            GaussianNB(),
            GradientBoostingClassifier(),
        ]:
            learner.fit(X_syn, y_syn)
            pred_probs = learner.predict_proba(X_test)
            auc_score = roc_auc_score(y_test, pred_probs[:, 1])
            print("-" * 40)
            print(f"{learner.__class__.__name__}: {auc_score}")
        print()

        self.train_df = pd.DataFrame(data=X_syn, columns=data_columns)
        self.train_labels = pd.DataFrame(data=y_syn, columns=[target_variable])


if __name__ == "__main__":
    import logging

    import yaml
    from fltk.synthpriv.config import SynthPrivConfig
    from fltk.synthpriv.datasets.adult import DistAdultDataset

    cfg = SynthPrivConfig()
    yaml_data = yaml.load("fltk/synthpriv/experiments/purchase.yaml", Loader=yaml.FullLoader)
    cfg.merge_yaml(yaml_data)
    cfg.init_logger(logging)

    real_data = DistAdultDataset(cfg)
    fake_data = SyntheticDataset(real_data, cfg)

    print(real_data.train_df)
    print(fake_data.train_df)
