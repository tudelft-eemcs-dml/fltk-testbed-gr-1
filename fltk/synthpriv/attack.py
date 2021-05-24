import argparse
import logging
import os

import joblib
import numpy as np
import torch
import torchvision
import yaml
from fltk.datasets.distributed.cifar100 import DistCIFAR100Dataset
from fltk.synthpriv.attacks import NasrAttack, UnsupervisedNasrAttack
from fltk.synthpriv.config import SynthPrivConfig
from fltk.synthpriv.datasets import *
from fltk.synthpriv.models import *
from tqdm import tqdm
import fltk

torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["purchase", "texas", "cifar", "adult"])
    parser.add_argument("model_checkpoint")
    parser.add_argument("--unsupervised", action="store_true")
    args = parser.parse_args()

    implemented_datasets = ["purchase", "texas", "cifar", "adult"]
    configs = {
        "purchase": {"data_location": "fltk/synthpriv/experiments/purchase.yaml"},
        "texas": {"data_location": "fltk/synthpriv/experiments/texas.yaml"},
        "cifar": {"data_location": "fltk/synthpriv/experiments/cifar100.yaml"},
        "adult": {"data_location": "fltk/synthpriv/experiments/adult.yaml"},
    }

    cfg = SynthPrivConfig()
    yaml_data = yaml.load(configs[args.dataset]["data_location"], Loader=yaml.FullLoader)
    cfg.merge_yaml(yaml_data)
    cfg.init_logger(logging)

    print("loading target models")
    if args.dataset == "purchase":
        target_model = PurchaseMLP()
        dataset = DistPurchaseDataset(cfg)
    elif args.dataset == "texas":
        target_model = TexasMLP()
        dataset = DistTexasDataset(cfg)
    elif args.dataset == "adult":
        target_model = AdultMLP()
        dataset = DistAdultDataset(cfg)
    elif args.dataset == "cifar":
        target_model = torchvision.models.densenet121()
        dataset = fltk.datasets.distributed.cifar100.DistCIFAR100Dataset(cfg)

    target_model.load_state_dict(torch.load(args.model_checkpoint))
    last_relu, last_linear = 0, 0
    for i, (name, mod) in enumerate(target_model.named_modules()):
        # print(i, name, mod.__class__.__name__)
        if mod.__class__.__name__ == "Linear":
            last_linear = i
        if mod.__class__.__name__ == "ReLU":
            last_relu = i

    print("loading data")
    if not args.unsupervised:
        member_train, nonmember_train, member_test, nonmember_test = [], [], [], []
        pbar = tqdm(zip(dataset.train_loader, dataset.test_loader), desc="Preparing data...")
        for (memfeat, memlabel), (nonmemfeat, nonmemlabel) in pbar:
            if len(member_train) * len(memfeat) < 5_000:
                member_train.append((memfeat, memlabel))
                nonmember_train.append((nonmemfeat, nonmemlabel))
            else:
                member_test.append((memfeat, memlabel))
                nonmember_test.append((nonmemfeat, nonmemlabel))
            if len(member_test) * len(memfeat) > 15_000:
                break
            pbar.update(len(memfeat))
        train_dataloader = member_train, nonmember_train
        test_dataloader = member_test, nonmember_test

        mftr, mltr = [], []
        for f, l in member_train:
            mftr.append(f)
            mltr.append(l)
        nmftr, nmltr = [], []
        for f, l in nonmember_train:
            nmftr.append(f)
            nmltr.append(l)
        mfte, mlte = [], []
        for f, l in member_test:
            mfte.append(f)
            mlte.append(l)
        nmfte, nmlte = [], []
        for f, l in nonmember_test:
            nmfte.append(f)
            nmlte.append(l)

        mftr, nmftr, mfte, nmfte = torch.cat(mftr), torch.cat(nmftr), torch.cat(mfte), torch.cat(nmfte)
        mltr, nmltr, mlte, nmlte = torch.cat(mltr), torch.cat(nmltr), torch.cat(mlte), torch.cat(nmlte)

        overlap = []
        for i in range(mftr.shape[1]):
            if mftr[:, i].sum() == 0 or nmftr[:, i].sum() == 0 or mfte[:, i].sum() == 0 or nmfte[:, i].sum() == 0:
                continue
            overlap.append(i)

        print()
        print("member train   ", mftr.shape)
        print("nonmember train", nmftr.shape)
        print("member test    ", mfte.shape)
        print("nonmember test ", nmfte.shape)
        print("\nmeans")
        ms = [
            mftr[:, overlap].mean(0),
            nmftr[:, overlap].mean(0),
            mfte[:, overlap].mean(0),
            nmfte[:, overlap].mean(0),
        ]
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                # print(torch.maximum(ms[i], ms[j]) == 0)
                diff = abs(ms[i] - ms[j]) / torch.maximum(ms[i], ms[j])
                print(
                    i,
                    j,
                    f"{diff.min().item() * 100:.2f}",
                    f"{diff.mean().item() * 100:.2f}",
                    f"{diff.max().item() * 100:.2f}",
                )
        ls = [
            torch.nn.functional.one_hot(mltr).float().mean(0),
            torch.nn.functional.one_hot(nmltr).float().mean(0),
            torch.nn.functional.one_hot(mlte).float().mean(0),
            torch.nn.functional.one_hot(nmlte).float().mean(0),
        ]
        print("\nlabels")
        for i in range(len(ls)):
            for j in range(i + 1, len(ls)):
                diff = abs(ls[i] - ls[j]) / torch.maximum(ls[i], ls[j])
                print(
                    i,
                    j,
                    f"{diff.min().item() * 100:.2f}",
                    f"{diff.mean().item() * 100:.2f}",
                    f"{diff.max().item() * 100:.2f}",
                )
        print()

        attacker = NasrAttack(
            "cuda",
            target_model,
            train_dataloader,
            test_dataloader,
            layers_to_exploit=[last_relu],
            gradients_to_exploit=[last_linear],
            exploit_loss=True,
            exploit_label=True,
        )
    else:
        train, member_test, nonmember_test = [], []
        pbar = tqdm(zip(dataset.train_loader, dataset.test_loader), desc="Preparing data...")
        for (memfeat, memlabel), (nonmemfeat, nonmemlabel) in pbar:
            if len(train) * len(memfeat) < 5_000:
                memidxs = np.random.permutation(len(memfeat))
                memfeat = memfeat[memidxs]
                memlabel = memlabel[memidxs]

                nonmemidxs = np.random.permutation(len(nonmemfeat))
                nonmemfeat = nonmemfeat[nonmemidxs]
                nonmemlabel = nonmemlabel[nonmemidxs]

                feat = torch.cat([memfeat, nonmemfeat])
                labels = torch.cat([memlabel, nonmemlabel])
                idxs = np.random.permutation(len(feat))
                feat = feat[idxs]
                labels = feat[idxs]

                train.append((feat, labels))
            else:
                member_test.append((memfeat, memlabel))
                nonmember_test.append((nonmemfeat, nonmemlabel))
            if len(member_test) * len(memfeat) > 15_000:
                break
            pbar.update(len(memfeat))

        attacker = UnsupervisedNasrAttack(
            "cuda",
            train,
            (member_test, nonmember_test),
            layers_to_exploit=[last_relu],
            gradients_to_exploit=[last_linear],
            exploit_loss=True,
            exploit_label=True,
        )

    print("initalizing attack")

    print("training attack model")
    attacker.train_attack()

    print("evaluating attack model")
    attacker.test_attack()
